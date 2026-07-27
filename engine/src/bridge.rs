//! Browser entry point (wasm32 only). Wraps the pure [`sim`] core in a
//! `wasm-bindgen` object the Next.js app drives: construct a scenario, feed real
//! elapsed time each animation frame, and read back a flat vehicle buffer ready
//! for instanced GPU rendering. Everything numeric lives in [`sim`] and is
//! tested natively; this layer is only marshalling.

use std::collections::HashMap;

use wasm_bindgen::prelude::*;

use crate::render::camera::Camera;
use crate::render::scene::{brake_intensity, class_color, class_dims, signal_color, signal_head_instances};
use crate::render::{geometry, Instance, StaticMesh};
use crate::sim::clock::SimClock;
use crate::sim::config::{SimConfig, VehicleClass};
use crate::sim::demand::{DemandGenerator, OdPair};
use crate::sim::map;
use crate::sim::net_world::NetWorld;
use crate::sim::network::{LinkId, Network, LANE_WIDTH};
use crate::sim::rng::{hash, Stream};

#[wasm_bindgen]
pub struct Simulation {
    world: NetWorld,
    clock: SimClock,
    seed: u64,
    demand: DemandGenerator,
    camera: Camera,
    /// `[x, y, heading, speed]` of each vehicle one tick ago, keyed by id, so the
    /// render interpolates pose between committed states (smooth at 60fps) and
    /// derives brake lights from the speed delta.
    prev: HashMap<u32, [f32; 4]>,
    /// The link the user has selected (clicked), highlighted in the density pass.
    selected: Option<usize>,
}

#[wasm_bindgen]
impl Simulation {
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u32) -> Simulation {
        Self::assemble(map::millbrae_sample(), seed)
    }

    /// Load a network scraped by `tools/osm-scraper` (its JSON schema) and drive
    /// origin–destination demand across it. Requires the `import` feature.
    #[cfg(feature = "import")]
    pub fn from_map_json(json: &str, seed: u32) -> Result<Simulation, JsValue> {
        let map = map::OsmMap::from_json(json).map_err(|e| JsValue::from_str(&e))?;
        Ok(Self::assemble(map.build(), seed))
    }

    fn assemble(network: Network, seed: u32) -> Simulation {
        let cfg = SimConfig { seed: seed as u64, ..SimConfig::default_config() };
        let camera = Camera::fit_bounds(network.bounds(), [900.0, 600.0], 24.0);
        let world = NetWorld::new(network, cfg);
        let demand = build_demand(&world, cfg.seed);
        let mut clock = SimClock::new(&cfg);
        clock.play();
        Simulation {
            world, clock, seed: cfg.seed, demand, camera,
            prev: HashMap::new(), selected: None,
        }
    }

    // --- camera control -------------------------------------------------------

    pub fn set_viewport(&mut self, width: f32, height: f32) {
        self.camera.viewport = [width as f64, height as f64];
    }

    /// Reset to a whole-network fit for the current viewport.
    pub fn fit(&mut self) {
        self.camera = Camera::fit_bounds(self.world.network.bounds(), self.camera.viewport, 24.0);
    }

    /// Pan by a drag delta in canvas pixels.
    pub fn pan_pixels(&mut self, dx: f32, dy: f32) {
        self.camera.pan_pixels(dx as f64, dy as f64);
    }

    /// Zoom by `factor` (<1 zooms in) about a canvas pixel (mouse wheel).
    pub fn zoom_at(&mut self, factor: f32, sx: f32, sy: f32) {
        self.camera.zoom_at(factor as f64, [sx as f64, sy as f64]);
    }

    pub fn set_meters_per_pixel(&mut self, mpp: f32) {
        self.camera.meters_per_pixel = (mpp as f64).clamp(0.02, 10_000.0);
    }

    pub fn meters_per_pixel(&self) -> f32 {
        self.camera.meters_per_pixel as f32
    }

    /// `[center_x, center_y, meters_per_pixel, viewport_w, viewport_h]` for the
    /// 2D fallback transform.
    pub fn camera_params(&self) -> Vec<f32> {
        [
            self.camera.center[0] as f32,
            self.camera.center[1] as f32,
            self.camera.meters_per_pixel as f32,
            self.camera.viewport[0] as f32,
            self.camera.viewport[1] as f32,
        ]
        .to_vec()
    }

    pub fn play(&mut self) {
        self.clock.play();
    }

    pub fn pause(&mut self) {
        self.clock.pause();
    }

    pub fn set_speed(&mut self, speed: f64) {
        self.clock.set_speed(speed);
    }

    pub fn single_step(&mut self) {
        let dt = self.clock.dt();
        self.clock.single_step();
        self.demand.step(&mut self.world, dt);
        self.world.step();
        self.prev = self.snapshot(); // discrete step: show the new state directly
    }

    pub fn advance(&mut self, real_elapsed_secs: f64) -> u32 {
        let ticks = self.clock.advance(real_elapsed_secs, 240);
        let dt = self.clock.dt();
        for _ in 0..ticks {
            self.prev = self.snapshot(); // state before this tick
            self.demand.step(&mut self.world, dt); // stream routed vehicles in
            self.world.step();
        }
        ticks
    }

    pub fn vehicle_count(&self) -> u32 {
        self.world.vehicles().len() as u32
    }

    /// Cumulative vehicles removed from the road after a collision.
    pub fn crashed(&self) -> u32 {
        self.world.crashed()
    }

    /// Measured flow (vehicles/hour) per link — the sim's own counts, to compare
    /// against real observed counts for calibration.
    pub fn link_flows(&self) -> Vec<f32> {
        self.world.link_flows().iter().map(|&f| f as f32).collect()
    }

    /// Select a link to highlight (negative clears the selection).
    pub fn set_selected_link(&mut self, index: i32) {
        self.selected = (index >= 0 && (index as usize) < self.world.network.links.len()).then_some(index as usize);
    }

    /// Live stats for a link: `[vehicle_count, mean_speed_mps, flow_vph,
    /// occupancy_ratio]`.
    pub fn link_stats(&self, index: u32) -> Vec<f32> {
        let i = index as usize;
        if i >= self.world.network.links.len() {
            return vec![0.0; 4];
        }
        let (count, mean, occ) = self.world.link_stats(LinkId(index));
        vec![count as f32, mean as f32, self.world.link_flows()[i] as f32, occ as f32]
    }

    /// `[x, y, heading, brake]` per vehicle: pose interpolated between the last
    /// two ticks by the clock's sub-tick `alpha`, and a brake-light intensity in
    /// `[0,1]` derived from deceleration.
    pub fn vehicle_instances(&self) -> Vec<f32> {
        let alpha = self.clock.alpha() as f32;
        let dt = self.clock.dt() as f32;
        let mut out = Vec::with_capacity(self.world.vehicles().len() * 4);
        for v in self.world.vehicles() {
            let c = self.world.vehicle_world_pose(v);
            let (cx, cy, ch, cs) = (c[0] as f32, c[1] as f32, c[2] as f32, v.speed as f32);
            // The sim samples the interior curve per tick, so a straight lerp
            // between consecutive poses already traces the turn.
            let (x, y, h, brake) = match self.prev.get(&v.id) {
                Some(&[px, py, ph, ps]) => (
                    px + (cx - px) * alpha,
                    py + (cy - py) * alpha,
                    ph + shortest_angle(ph, ch) * alpha,
                    brake_intensity((cs - ps) / dt),
                ),
                None => (cx, cy, ch, 0.0),
            };
            out.extend_from_slice(&[x, y, h, brake]);
        }
        out
    }

    /// Signal heads as `[x, y, r, g, b]`, one per signal group, placed at the
    /// stop line of the approach the group controls and coloured by its state.
    pub fn signal_heads(&self) -> Vec<f32> {
        let mut out = Vec::new();
        for (pos, _heading, state) in self.signal_head_slots() {
            let col = signal_color(state);
            out.extend_from_slice(&[pos[0], pos[1], col[0], col[1], col[2]]);
        }
        out
    }

    /// Each signal group's head as `(position, approach heading, state)`. The
    /// head sits at the stop line of one of the group's approach lanes, nudged
    /// laterally by group bit so several heads on one approach don't overlap.
    fn signal_head_slots(&self) -> Vec<([f32; 2], f32, crate::sim::signal::SignalState)> {
        let net = &self.world.network;
        let states = self.world.signal_states();
        let mut rep = vec![None; net.groups.len()];
        for mv in &net.movements {
            if let Some(g) = mv.signal_group {
                rep[g.idx()].get_or_insert(mv.from_lane);
            }
        }
        let mut out = Vec::new();
        for (gi, lane) in rep.into_iter().enumerate() {
            let Some(lane) = lane else { continue };
            let bit = net.groups[gi].bit as f64;
            let p = net.lane_point(lane, net.lane(lane).length);
            let dir = net.arrival_dir(net.lane(lane).link);
            let (px, py) = (p[0] + dir[1] * bit * 2.0, p[1] - dir[0] * bit * 2.0);
            out.push(([px as f32, py as f32], dir[1].atan2(dir[0]) as f32, states[gi]));
        }
        out
    }

    /// Paved junctions as `[x, y, radius]` for every node where roads meet.
    pub fn junctions(&self) -> Vec<f32> {
        let net = &self.world.network;
        let mut degree = vec![0u32; net.nodes.len()];
        let mut widest = vec![0.0f64; net.nodes.len()];
        for link in &net.links {
            let w = link.lane_count as f64 * LANE_WIDTH;
            for n in [link.from, link.to] {
                degree[n.idx()] += 1;
                widest[n.idx()] = widest[n.idx()].max(w);
            }
        }
        let mut out = Vec::new();
        for (i, node) in net.nodes.iter().enumerate() {
            if degree[i] >= 2 {
                out.extend_from_slice(&[node.position[0] as f32, node.position[1] as f32, (widest[i] * 0.55 + 0.5) as f32]);
            }
        }
        out
    }

    // --- WebGPU renderer feed -------------------------------------------------

    /// Roads + junctions as flat `StaticVertex` floats (drawn at all zooms).
    pub fn world_mesh_vertices(&self) -> Vec<f32> {
        bytemuck::cast_slice(&self.world_mesh().vertices).to_vec()
    }

    pub fn world_mesh_indices(&self) -> Vec<u32> {
        self.world_mesh().indices
    }

    /// Lane markings as flat `StaticVertex` floats (drawn only when zoomed in).
    pub fn marking_mesh_vertices(&self) -> Vec<f32> {
        bytemuck::cast_slice(&geometry::marking_mesh(&self.world.network).vertices).to_vec()
    }

    pub fn marking_mesh_indices(&self) -> Vec<u32> {
        geometry::marking_mesh(&self.world.network).indices
    }

    fn world_mesh(&self) -> StaticMesh {
        let net = &self.world.network;
        // At-grade + tunnels first (carriageways run to the nodes and overlap to
        // pave intersections), then bridges on top.
        let mut mesh = geometry::road_mesh(net);
        mesh.extend(&geometry::overpass_mesh(net));
        mesh
    }

    /// Column-major 4×4 view-projection for the current camera.
    pub fn view_proj(&self) -> Vec<f32> {
        self.camera.view_proj().to_vec()
    }

    pub fn alpha(&self) -> f32 {
        self.clock.alpha() as f32
    }

    /// Raw `Instance` bytes for the instanced draw: current + previous pose (the
    /// shader interpolates by `alpha`), class size/colour, and brake intensity.
    pub fn render_instances(&self) -> Vec<u8> {
        let dt = self.clock.dt() as f32;
        let instances: Vec<Instance> = self
            .world
            .vehicles()
            .iter()
            .map(|v| {
                let class = VehicleClass::from_length(v.driver.vehicle_length);
                let c = self.world.vehicle_world_pose(v);
                let (cx, cy, ch, cs) = (c[0] as f32, c[1] as f32, c[2] as f32, v.speed as f32);
                let [px, py, ph, ps] = self.prev.get(&v.id).copied().unwrap_or([cx, cy, ch, cs]);
                Instance {
                    pos: [cx, cy],
                    prev_pos: [px, py],
                    control: [(px + cx) * 0.5, (py + cy) * 0.5],
                    scale: class_dims(class),
                    color: class_color(class),
                    heading: ch,
                    prev_heading: ph,
                    brake: brake_intensity((cs - ps) / dt),
                }
            })
            .collect();
        bytemuck::cast_slice(&instances).to_vec()
    }

    pub fn render_instance_count(&self) -> u32 {
        self.world.vehicles().len() as u32
    }

    /// Signal heads as raw `Instance` bytes for the emissive-disc draw.
    pub fn signal_instances(&self) -> Vec<u8> {
        bytemuck::cast_slice(&self.signal_instance_vec()).to_vec()
    }

    pub fn signal_instance_count(&self) -> u32 {
        self.signal_instance_vec().len() as u32
    }

    /// Translucent congestion-overlay mesh for the far-LOD density view: per-link
    /// carriageway quads coloured by live occupancy, emitted only for links busy
    /// enough to matter. `StaticVertex` floats; pair with [`density_indices`].
    pub fn density_vertices(&self) -> Vec<f32> {
        bytemuck::cast_slice(&self.density_mesh().vertices).to_vec()
    }

    pub fn density_indices(&self) -> Vec<u32> {
        self.density_mesh().indices
    }

    fn density_mesh(&self) -> StaticMesh {
        let net = &self.world.network;
        let mut counts = vec![0u32; net.links.len()];
        for v in self.world.vehicles() {
            counts[net.lane(v.lane).link.idx()] += 1;
        }
        geometry::congestion_mesh(net, &counts, self.selected)
    }

    fn signal_instance_vec(&self) -> Vec<Instance> {
        self.signal_head_slots()
            .into_iter()
            .flat_map(|(pos, heading, state)| signal_head_instances(pos, heading, state))
            .collect()
    }

    fn snapshot(&self) -> HashMap<u32, [f32; 4]> {
        self.world
            .vehicles()
            .iter()
            .map(|v| {
                let p = self.world.vehicle_world_pose(v);
                (v.id, [p[0] as f32, p[1] as f32, p[2] as f32, v.speed as f32])
            })
            .collect()
    }

    /// Static carriageway quads `[cx0, cy0, cx1, cy1, width]` per link.
    pub fn road_strips(&self) -> Vec<f32> {
        flatten(self.world.network.road_strips())
    }

    /// Static lane-divider segments `[x0, y0, x1, y1]`.
    pub fn lane_dividers(&self) -> Vec<f32> {
        flatten(self.world.network.lane_dividers())
    }

    /// `[min_x, min_y, max_x, max_y]` world bounds for the camera fit.
    pub fn world_bounds(&self) -> Vec<f32> {
        self.world.network.bounds().iter().map(|&v| v as f32).collect()
    }

    pub fn seed(&self) -> f64 {
        self.seed as f64
    }
}

/// Sample origin–destination pairs whose routes cross at least one intersection,
/// so the scenario shows continuous traffic flowing through the network.
fn build_demand(world: &NetWorld, seed: u64) -> DemandGenerator {
    let net: &Network = &world.network;
    let link_count = net.links.len().max(1) as u64;
    let pick = |salt: u32, attempt: u64| LinkId((hash(seed, salt, attempt, Stream::RouteChoice) % link_count) as u32);

    let mut pairs = Vec::new();
    let mut attempt = 0u64;
    while pairs.len() < 32 && attempt < 2000 {
        let (o, d) = (pick(1, attempt), pick(2, attempt));
        attempt += 1;
        if o == d {
            continue;
        }
        if net.route_links(o, d).is_some_and(|r| r.len() >= 3) {
            pairs.push(OdPair { origin: o, dest: d, rate_per_sec: 0.2 });
        }
    }
    DemandGenerator::new(world, &pairs, seed)
}

/// Shortest signed angular difference `a → b`, so heading interpolation takes
/// the short way around instead of spinning across ±π.
fn shortest_angle(a: f32, b: f32) -> f32 {
    use std::f32::consts::PI;
    ((b - a + PI).rem_euclid(2.0 * PI)) - PI
}

fn flatten<const N: usize>(rows: Vec<[f64; N]>) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows.len() * N);
    for row in rows {
        out.extend(row.iter().map(|&v| v as f32));
    }
    out
}
