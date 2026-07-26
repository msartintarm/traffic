//! Browser entry point (wasm32 only). Wraps the pure [`sim`] core in a
//! `wasm-bindgen` object the Next.js app drives: construct a scenario, feed real
//! elapsed time each animation frame, and read back a flat vehicle buffer ready
//! for instanced GPU rendering. Everything numeric lives in [`sim`] and is
//! tested natively; this layer is only marshalling.

use std::collections::HashMap;

use wasm_bindgen::prelude::*;

use crate::render::camera::Camera;
use crate::render::scene::{brake_intensity, signal_color, VehicleClass};
use crate::render::{geometry, Instance};
use crate::sim::clock::SimClock;
use crate::sim::config::{DriverConfig, SimConfig};
use crate::sim::map;
use crate::sim::net_world::NetWorld;
use crate::sim::network::{LaneId, NodeControl, LANE_WIDTH};

#[wasm_bindgen]
pub struct Simulation {
    world: NetWorld,
    clock: SimClock,
    seed: u64,
    /// `[x, y, heading, speed]` of each vehicle one tick ago, keyed by id, so the
    /// render interpolates pose between committed states (smooth at 60fps) and
    /// derives brake lights from the speed delta.
    prev: HashMap<u32, [f32; 4]>,
}

#[wasm_bindgen]
impl Simulation {
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u32) -> Simulation {
        let cfg = SimConfig { seed: seed as u64, ..SimConfig::default_config() };
        let mut world = NetWorld::new(map::millbrae_sample(), cfg);
        let driver = DriverConfig::car();
        for i in 0..64u32 {
            world.spawn(i, LaneId(i % 4), (i as f64) * 12.0 % 180.0, 8.0, driver.sample(cfg.seed, i));
        }
        let mut clock = SimClock::new(&cfg);
        clock.play();
        Simulation { world, clock, seed: cfg.seed, prev: HashMap::new() }
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
        self.clock.single_step();
        self.world.step();
        self.prev = self.snapshot(); // discrete step: show the new state directly
    }

    pub fn advance(&mut self, real_elapsed_secs: f64) -> u32 {
        let ticks = self.clock.advance(real_elapsed_secs, 240);
        for _ in 0..ticks {
            self.prev = self.snapshot(); // state before this tick
            self.world.step();
        }
        ticks
    }

    pub fn vehicle_count(&self) -> u32 {
        self.world.vehicles().len() as u32
    }

    /// `[x, y, heading, brake]` per vehicle: pose interpolated between the last
    /// two ticks by the clock's sub-tick `alpha`, and a brake-light intensity in
    /// `[0,1]` derived from deceleration.
    pub fn vehicle_instances(&self) -> Vec<f32> {
        let alpha = self.clock.alpha() as f32;
        let dt = self.clock.dt() as f32;
        let mut out = Vec::with_capacity(self.world.vehicles().len() * 4);
        for v in self.world.vehicles() {
            let c = self.world.network.lane_point(v.lane, v.position);
            let (cx, cy, ch, cs) = (c[0] as f32, c[1] as f32, c[2] as f32, v.speed as f32);
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

    /// Signal heads as `[x, y, r, g, b]`, one per signal group at each signalized
    /// node, positioned around the junction and coloured by current state.
    pub fn signal_heads(&self) -> Vec<f32> {
        let net = &self.world.network;
        let states = net.signal_states(self.world.time());
        let mut out = Vec::new();
        for node in &net.nodes {
            let NodeControl::Signalized(program) = node.control else { continue };
            let mut k = 0usize;
            for (gi, g) in net.groups.iter().enumerate() {
                if g.program != program {
                    continue;
                }
                let ang = std::f32::consts::TAU * k as f32 / 4.0;
                let (px, py) = (node.position[0] as f32 + 6.0 * ang.cos(), node.position[1] as f32 + 6.0 * ang.sin());
                let col = signal_color(states[gi]);
                out.extend_from_slice(&[px, py, col[0], col[1], col[2]]);
                k += 1;
            }
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
                out.extend_from_slice(&[node.position[0] as f32, node.position[1] as f32, (widest[i] * 0.75 + 1.0) as f32]);
            }
        }
        out
    }

    // --- WebGPU renderer feed -------------------------------------------------

    /// The baked static world mesh vertices, flat `[x, y, r, g, b, light]` per
    /// vertex (roads + junctions + markings combined).
    pub fn world_mesh_vertices(&self) -> Vec<f32> {
        bytemuck::cast_slice(&self.world_mesh().vertices).to_vec()
    }

    pub fn world_mesh_indices(&self) -> Vec<u32> {
        self.world_mesh().indices
    }

    fn world_mesh(&self) -> crate::render::Mesh {
        let net = &self.world.network;
        let mut mesh = geometry::road_mesh(net);
        mesh.extend(&geometry::junction_mesh(net));
        mesh.extend(&geometry::marking_mesh(net));
        mesh
    }

    /// Column-major 4×4 view-projection fitting the whole network into a
    /// `width`×`height` viewport.
    pub fn view_proj(&self, width: f32, height: f32) -> Vec<f32> {
        Camera::fit_bounds(self.world.network.bounds(), [width as f64, height as f64], 24.0)
            .view_proj()
            .to_vec()
    }

    pub fn alpha(&self) -> f32 {
        self.clock.alpha() as f32
    }

    /// Raw `Instance` bytes for the instanced draw: current + previous pose (the
    /// shader interpolates by `alpha`), class size/colour, and brake intensity.
    pub fn render_instances(&self) -> Vec<u8> {
        let dt = self.clock.dt() as f32;
        let dims = VehicleClass::Car.dims();
        let color = VehicleClass::Car.body_color();
        let instances: Vec<Instance> = self
            .world
            .vehicles()
            .iter()
            .map(|v| {
                let c = self.world.network.lane_point(v.lane, v.position);
                let (cx, cy, ch, cs) = (c[0] as f32, c[1] as f32, c[2] as f32, v.speed as f32);
                let [px, py, ph, ps] = self.prev.get(&v.id).copied().unwrap_or([cx, cy, ch, cs]);
                Instance {
                    pos: [cx, cy],
                    prev_pos: [px, py],
                    scale: dims,
                    color,
                    heading: ch,
                    prev_heading: ph,
                    brake: brake_intensity((cs - ps) / dt),
                    _pad: 0.0,
                }
            })
            .collect();
        bytemuck::cast_slice(&instances).to_vec()
    }

    pub fn render_instance_count(&self) -> u32 {
        self.world.vehicles().len() as u32
    }

    fn snapshot(&self) -> HashMap<u32, [f32; 4]> {
        self.world
            .vehicles()
            .iter()
            .map(|v| {
                let p = self.world.network.lane_point(v.lane, v.position);
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
