//! Browser entry point (wasm32 only). Wraps the pure [`sim`] core in a
//! `wasm-bindgen` object the Next.js app drives: construct a scenario, feed real
//! elapsed time each animation frame, and read back a flat vehicle buffer ready
//! for instanced GPU rendering. Everything numeric lives in [`sim`] and is
//! tested natively; this layer is only marshalling.

use crate::sim::hash::IntMap;
/// Pose map keyed by (sparse) vehicle id, hashed cheaply for the per-frame rebuild.
type PoseMap = IntMap<[f32; 4]>;

use wasm_bindgen::prelude::*;

use crate::render::camera::Camera;
use crate::render::scene::{brake_intensity, class_color, class_dims, signal_color, signal_head_instances};
use crate::render::gpu::Renderer;
use crate::render::{geometry, Instance, StaticMesh};
use crate::sim::clock::SimClock;
use crate::sim::flowfield;
use crate::sim::flowfield_gpu::{GpuFlowField, PendingReadback};
use crate::sim::config::{SimConfig, VehicleClass};
use crate::sim::demand::{self, DemandGenerator, DemandSources};
use crate::sim::congestion::CongestionConfig;
use crate::sim::map;
use crate::sim::net_world::{AccelBackend, NetWorld};
use crate::sim::network::{LinkId, Network, LANE_WIDTH};

/// Soft wall-clock budget (ms) for one frame's catch-up stepping. Past this, the
/// frame stops advancing the sim and renders, dropping the backlog so a heavy step
/// degrades to slow-motion instead of locking the main thread.
const FRAME_BUDGET_MS: f64 = 8.0;
/// Hard ceiling on catch-up ticks per frame. The wall-clock budget is the real
/// limiter; this only bounds the loop if no monotonic clock is available.
const MAX_CATCHUP_TICKS: u32 = 240;
/// Window over which the achieved-speed meter averages, long enough to smooth the
/// sub-tick-per-frame quantization at low speeds.
const SPEED_WINDOW_SECS: f64 = 0.5;

/// Monotonic wall-clock milliseconds from the browser's high-resolution timer.
/// Returns 0.0 if unavailable, which disables the budget (falling back to the tick
/// ceiling) rather than erroring.
fn now_ms() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0)
}

/// Turn-signal blink phase: `true` during the lit half of a ~1.4 Hz cycle, off the
/// other half. Driven off the wall clock so blinkers keep flashing while the sim is
/// paused, and so timing is independent of frame rate.
fn blink_on() -> bool {
    (now_ms() * (1.0 / 700.0)).rem_euclid(1.0) < 0.5
}

#[wasm_bindgen]
pub struct Simulation {
    world: NetWorld,
    clock: SimClock,
    seed: u64,
    demand: DemandGenerator,
    demand_sources: DemandSources,
    /// Spawn-rate multiplier and entry-speed cap (m/s), carried across demand rebuilds
    /// so the UI's frequency / start-speed controls persist when sources are toggled.
    demand_rate: f64,
    entry_speed_cap: f64,
    camera: Camera,
    /// `[x, y, heading, speed]` of each vehicle one tick ago, keyed by id, so the
    /// render interpolates pose between committed states (smooth at 60fps) and
    /// derives brake lights from the speed delta.
    prev: PoseMap,
    /// The link the user has selected (clicked), highlighted in the density pass.
    selected: Option<usize>,
    /// Curbside signal-head placements `(group, pos, heading)` — static geometry
    /// computed once at assembly; each frame only pairs them with live colours.
    signal_heads: Vec<(usize, [f32; 2], f32)>,
    /// Optional GPU flow-field solver on the renderer's WebGPU device, plus the
    /// in-flight async readback and the cost snapshot it was dispatched with.
    gpu: Option<GpuFlowField>,
    /// An in-flight GPU readback paired with the router generation it was dispatched
    /// under. A demand toggle reinstalls the router (bumping the generation); the
    /// stale readback must still be *collected* (to unmap its staging buffer) but its
    /// distances are discarded, never fed into the new field.
    gpu_pending: Option<(PendingReadback, u64)>,
    /// Bumped whenever the router is reinstalled, to invalidate an in-flight readback.
    gpu_generation: u64,
    gpu_cost: Vec<u64>,
    gpu_last: f64,
    /// Achieved sim-time / wall-time over the last window — the speed the sim is
    /// *actually* running at (see [`Simulation::meter_speed`]).
    effective_speed: f64,
    /// True when the frame budget is dropping catch-up ticks (can't hold the selected
    /// speed). Surfaced instead of a noisy per-frame ratio.
    throttled: bool,
    /// `now_ms()` at the previous `advance`, for the inter-frame wall delta.
    last_advance_ms: f64,
    /// Rolling accumulators for the speed meter: sim- and wall-seconds since the last
    /// window flush, plus whether any backlog was dropped within it.
    speed_sim_accum: f64,
    speed_wall_accum: f64,
    speed_dropped: bool,
}

#[wasm_bindgen]
impl Simulation {
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u32) -> Simulation {
        Self::assemble(map::millbrae_sample(), seed)
    }

    /// Build a named built-in test scenario (a clean, isolated network) — the
    /// browser's scenario selector uses this to swap between hand-built junctions
    /// and the sample/real map for debugging intersection behaviour in isolation.
    pub fn scenario(name: &str, seed: u32) -> Simulation {
        let net = match name {
            "arterial" => map::arterial_intersection(),
            "corridor" => map::corridor_with_signal(),
            "gridlock" => map::gridlock(),
            _ => map::millbrae_sample(),
        };
        Self::assemble(net, seed)
    }

    /// Load a network scraped by `tools/osm-scraper` (its JSON schema) and drive
    /// origin–destination demand across it. Requires the `import` feature.
    #[cfg(feature = "import")]
    pub fn from_map_json(json: &str, seed: u32) -> Result<Simulation, JsValue> {
        let map = map::OsmMap::from_json(json).map_err(|e| JsValue::from_str(&e))?;
        Ok(Self::assemble(map.build(), seed))
    }

    fn assemble(network: Network, seed: u32) -> Simulation {
        // Browser default: the active-set scheduler on (a clean serial/GPU win, and it
        // auto-stands-down under actively-parallel threads). The pure engine default stays
        // off so native tests and A/B baselines are unaffected. Toggle live via the UI.
        let cfg = SimConfig { seed: seed as u64, sleep_scheduler: true, ..SimConfig::default_config() };
        let camera = Camera::fit_bounds(network.bounds(), [900.0, 600.0], 24.0);
        let mut world = NetWorld::new(network, cfg);
        let demand_sources = DemandSources::new(true, true); // freeway + surface by default
        let (demand_rate, entry_speed_cap) = (1.0, f64::INFINITY);
        let demand = build_demand(&world, cfg.seed, demand_sources, demand_rate, entry_speed_cap);
        world.install_router(&demand.destinations());
        let mut clock = SimClock::new(&cfg);
        clock.play();
        let signal_heads = geometry::signal_head_placements(&world.network);
        Simulation {
            world, clock, seed: cfg.seed, demand, demand_sources, demand_rate, entry_speed_cap, camera,
            prev: PoseMap::default(), selected: None, signal_heads,
            gpu: None, gpu_pending: None, gpu_generation: 0, gpu_cost: Vec::new(), gpu_last: 0.0,
            effective_speed: 0.0, throttled: false, last_advance_ms: 0.0,
            speed_sim_accum: 0.0, speed_wall_accum: 0.0, speed_dropped: false,
        }
    }

    /// Move route recomputes onto the browser's WebGPU device — the *same* device
    /// the `renderer` already owns (no second adapter, no headers). Call once after
    /// the renderer is created. Routing then rebuilds on the GPU; if a result isn't
    /// ready yet the previous field is used (never a stall). Idempotent.
    pub fn enable_gpu_routing(&mut self, renderer: &Renderer) {
        let adj = flowfield::adjacency(&self.world.network);
        let (offsets, targets) = flowfield::csr(&adj);
        let (device, queue) = renderer.device_queue();
        self.gpu = Some(GpuFlowField::from_device(device, queue, &offsets, &targets));
        self.gpu_pending = None;
        self.world.set_external_reroute(true);
        self.gpu_last = f64::NEG_INFINITY; // recompute on the next frame
    }

    /// Set which traffic streams spawn (freeway and/or surface). Affects *new* spawns
    /// only — existing vehicles keep driving to their destinations (the router is kept
    /// over the union of the new demand's and the in-flight cars' destinations).
    pub fn set_demand_sources(&mut self, highway: bool, surface: bool) {
        self.apply_demand_sources(DemandSources { highway, surface, ..self.demand_sources });
    }

    /// Drive the whole map by the simulated time of day: freeways at their real
    /// per-lane PeMS volumes (US-101 / I-280, by direction) and surface streets by the
    /// arterial diurnal shape, so both build and fade with the commute. Affects new
    /// spawns only; live and non-destructive like the other demand toggles.
    pub fn set_rush_hour(&mut self, enabled: bool) {
        self.apply_demand_sources(DemandSources { rush_hour: enabled, ..self.demand_sources });
    }

    /// Rebuild the demand generator for a new source mix and reinstall the router over
    /// its destinations plus those of cars already on the road, so in-flight trips
    /// aren't stranded. No-op if the mix is unchanged.
    fn apply_demand_sources(&mut self, sources: DemandSources) {
        if sources == self.demand_sources {
            return;
        }
        self.demand_sources = sources;
        self.demand = build_demand(&self.world, self.seed, sources, self.demand_rate, self.entry_speed_cap);
        let mut dests = self.demand.destinations();
        for v in self.world.vehicles() {
            if let Some(d) = v.dest {
                dests.push(d);
            }
        }
        dests.sort_by_key(|l| l.0);
        dests.dedup();
        self.world.install_router(&dests);
        // A GPU readback dispatched for the previous destination set is now stale.
        // Bump the generation so `drive_gpu_routing` discards its distances — but
        // leave the pending in place so it is still collected and its staging buffer
        // unmapped; dropping it here would leak a mapped buffer that the next
        // dispatch then writes into (a WebGPU error that crashes the frame loop).
        self.gpu_generation = self.gpu_generation.wrapping_add(1);
        self.gpu_last = f64::NEG_INFINITY;
    }

    /// Whether freeway traffic is currently spawning, for the UI to reflect.
    pub fn demand_highway(&self) -> bool {
        self.demand_sources.highway
    }

    /// Whether surface traffic is currently spawning.
    pub fn demand_surface(&self) -> bool {
        self.demand_sources.surface
    }

    /// Whether the freeway stream is running on the real rush-hour profile.
    pub fn demand_rush_hour(&self) -> bool {
        self.demand_sources.rush_hour
    }

    /// The simulated rush-hour time of day (hours, 0–24) the profile clock is at, for
    /// the UI to display the peak building and fading. 0 when the mode is off.
    pub fn rush_hour_time(&self) -> f64 {
        self.demand.rush_hour_day_secs() / 3600.0
    }

    /// Live per-lane freeway volumes (veh/h/lane) read off the real diurnal profile at
    /// the current simulated time of day, split by direction to show the commute
    /// asymmetry: `[US-101 NB, US-101 SB, I-280 NB, I-280 SB]`.
    pub fn rush_hour_flows(&self) -> Vec<f32> {
        let t = self.demand.rush_hour_day_secs();
        let f = |r: &str, nb: bool| crate::sim::rush_hour::per_lane(r, nb, t) as f32;
        vec![f("US 101", true), f("US 101", false), f("I 280", true), f("I 280", false)]
    }

    /// Trips waiting at gateways to enter — demand the entrances can't yet admit,
    /// held (not dropped) and released as lanes clear. Grows and dissipates with the
    /// peak, so it reads as real rush-hour pressure.
    pub fn demand_queued(&self) -> u32 {
        self.demand.queued()
    }

    /// Scale the spawn rate of every stream (1.0 = default). Live, non-destructive.
    pub fn set_demand_rate(&mut self, scale: f64) {
        self.demand_rate = scale.max(0.0);
        self.demand.set_rate_scale(self.demand_rate);
    }

    pub fn demand_rate(&self) -> f64 {
        self.demand_rate
    }

    /// Cap the speed vehicles enter the map at (m/s); still never above the origin
    /// road's limit. A very large value means "enter at the road's limit".
    pub fn set_entry_speed_cap(&mut self, mps: f64) {
        self.entry_speed_cap = mps.max(0.0);
        self.demand.set_entry_speed_cap(self.entry_speed_cap);
    }

    pub fn entry_speed_cap(&self) -> f64 {
        self.entry_speed_cap
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
        self.drive_gpu_routing();
        self.demand.step(&mut self.world, dt);
        self.world.step();
        self.prev = self.snapshot(); // discrete step: show the new state directly
    }

    pub fn advance(&mut self, real_elapsed_secs: f64) -> u32 {
        // Wall-clock delta since the previous frame (uncapped, unlike the JS-capped
        // `real_elapsed_secs`), clamped so a backgrounded tab's huge gap can't skew
        // the speed meter.
        let frame_ms = now_ms();
        let wall = if self.last_advance_ms > 0.0 {
            ((frame_ms - self.last_advance_ms) / 1000.0).clamp(0.0, 0.25)
        } else {
            0.0
        };
        self.last_advance_ms = frame_ms;

        self.drive_gpu_routing();
        let ticks = self.clock.advance(real_elapsed_secs, MAX_CATCHUP_TICKS);
        let dt = self.clock.dt();

        let mut ran = 0u32;
        if ticks > 0 {
            let loop_start = now_ms();
            while ran < ticks {
                // Once catch-up blows the frame budget, treat this tick as the last:
                // render the result and drop the remaining backlog so a heavy step
                // degrades to slow-motion instead of freezing the main thread. (Always
                // run at least one tick, hence `ran > 0`.)
                let over_budget = ran > 0 && now_ms() - loop_start > FRAME_BUDGET_MS;
                let last = over_budget || ran + 1 == ticks;
                // The render interpolates between `prev` and the post-step state, so
                // only the snapshot taken right before the *final* committed tick is
                // used; taking it once (not per catch-up tick) avoids rebuilding the
                // full pose map several times per frame.
                if last {
                    self.prev = self.snapshot();
                }
                self.demand.step(&mut self.world, dt); // stream routed vehicles in
                self.world.step();
                ran += 1;
                if last {
                    if over_budget {
                        self.clock.drop_backlog();
                    }
                    break;
                }
            }
        }

        self.meter_speed(ran, dt, wall, ran < ticks);
        ran
    }

    /// Roll the achieved-speed / throttled readout over a wall-clock window. Averaged
    /// over the window (not per frame) because at low speeds most frames legitimately
    /// run zero ticks — a per-frame ratio would read ~0 while perfectly keeping up.
    /// "Throttled" is defined by the budget dropping catch-up ticks (`ran < ticks`),
    /// the true can't-keep-up signal, rather than the noisy ratio.
    fn meter_speed(&mut self, ran: u32, dt: f64, wall: f64, dropped_backlog: bool) {
        if !self.clock.is_running() {
            self.effective_speed = 0.0;
            self.throttled = false;
            self.speed_sim_accum = 0.0;
            self.speed_wall_accum = 0.0;
            self.speed_dropped = false;
            return;
        }
        if wall <= 0.0 {
            return; // first frame / no clock — nothing to attribute
        }
        self.speed_sim_accum += ran as f64 * dt;
        self.speed_wall_accum += wall;
        self.speed_dropped |= dropped_backlog;
        if self.speed_wall_accum >= SPEED_WINDOW_SECS {
            self.effective_speed = self.speed_sim_accum / self.speed_wall_accum;
            self.throttled = self.speed_dropped;
            self.speed_sim_accum = 0.0;
            self.speed_wall_accum = 0.0;
            self.speed_dropped = false;
        }
    }

    /// Sim-seconds advanced per wall-second, actually achieved over the last window.
    pub fn effective_speed(&self) -> f64 {
        self.effective_speed
    }

    /// The speed multiplier the user selected (last `set_speed`).
    pub fn selected_speed(&self) -> f64 {
        self.clock.speed()
    }

    /// Whether the sim is failing to hold the selected speed — the frame budget is
    /// dropping catch-up ticks. False while it keeps up (including at 1×, where most
    /// frames legitimately run no tick at all).
    pub fn is_throttled(&self) -> bool {
        self.throttled
    }

    /// Choose the accel executor: `"serial"`, `"threads"`, or `"gpu"`. The sim falls
    /// back automatically when the request isn't available on this device (no CPU
    /// worker pool, or the GPU kernel not yet wired), so the UI can offer all three.
    pub fn set_accel_backend(&mut self, name: &str) {
        self.world.set_accel_backend(AccelBackend::from_name(name));
    }

    /// The backend actually running after availability fallback — for the UI to show
    /// what the device settled on (may differ from the requested one).
    pub fn accel_backend(&self) -> String {
        self.world.active_backend().name().to_string()
    }

    /// Tell the sim the CPU worker pool is up (the browser calls this once
    /// `initThreadPool` resolves), so the `Threads` backend stops falling back.
    pub fn set_threads_ready(&mut self, ready: bool) {
        self.world.set_threads_ready(ready);
    }

    /// Vehicle count at/above which the `Threads` backend parallelizes (below it,
    /// serial). Exposed so the UI can tune the crossover per device.
    pub fn set_par_threshold(&mut self, n: u32) {
        self.world.set_par_threshold(n as usize);
    }

    pub fn par_threshold(&self) -> u32 {
        self.world.par_threshold() as u32
    }

    /// Toggle the active-set scheduler: queued-behind-a-stopped-car vehicles skip the
    /// full per-tick gather, so step cost tracks the *deciding* fraction rather than the
    /// whole fleet. Safe live toggle (non-destructive; changes work distribution only).
    pub fn set_sleep_scheduler(&mut self, on: bool) {
        self.world.set_sleep_scheduler(on);
    }

    /// Vehicles the scheduler skipped last step — a diagnostic for the status line
    /// (0 when the scheduler is off).
    pub fn asleep_count(&self) -> u32 {
        self.world.asleep_count() as u32
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

    /// Enable or disable the congestion level-of-detail: while a link stays saturated,
    /// its queued cars use cheap leader-only car-following instead of the full model.
    /// Cars stay individual and keep their positions — only the per-car cost drops.
    /// Off by default (full detail everywhere).
    pub fn set_congestion_enabled(&mut self, enabled: bool) {
        let cfg = CongestionConfig { enabled, ..self.world.congestion_config() };
        self.world.set_congestion(cfg);
    }

    /// Occupancy ratio (0..1) at which a saturated link switches to the cheap queue
    /// model; the release threshold trails it so links don't thrash near the boundary.
    pub fn set_congestion_engage(&mut self, engage: f64) {
        let engage = engage.clamp(0.05, 1.0);
        let cfg = CongestionConfig {
            engage_occ: engage,
            release_occ: (engage - 0.3).max(0.05),
            ..self.world.congestion_config()
        };
        self.world.set_congestion(cfg);
    }

    /// How many links are currently running the cheap queue model — for the status line.
    pub fn congestion_active_links(&self) -> u32 {
        self.world.congestion_active_links()
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

    /// `[x, y, heading, brake, blinker]` per vehicle: pose interpolated between the
    /// last two ticks by the clock's sub-tick `alpha`, a brake-light intensity in
    /// `[0,1]` from deceleration, and a turn-signal side (`-1` left, `+1` right, `0`
    /// none) already gated by the blink phase.
    pub fn vehicle_instances(&self) -> Vec<f32> {
        let alpha = self.clock.alpha() as f32;
        let dt = self.clock.dt() as f32;
        let lit = blink_on();
        let mut out = Vec::with_capacity(self.world.vehicles().len() * 5);
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
            let blinker = if lit { self.world.vehicle_blinker(v) as f32 } else { 0.0 };
            out.extend_from_slice(&[x, y, h, brake, blinker]);
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

    /// Each signal group's head as `(position, approach heading, state)`, from the
    /// pure curbside placement in [`geometry::signal_head_placements`] paired with
    /// the group's live colour.
    fn signal_head_slots(&self) -> Vec<([f32; 2], f32, crate::sim::signal::SignalState)> {
        let states = self.world.signal_states();
        self.signal_heads.iter().map(|&(gi, pos, heading)| (pos, heading, states[gi])).collect()
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

    /// OSM road names for the engine's *own* links (post collapse/merge),
    /// index-aligned with link ids — so the browser labels the same links the
    /// engine selects and reports stats for, and click never deviates.
    pub fn link_names(&self) -> Vec<String> {
        self.world.network.link_names.clone()
    }

    /// Each engine link's centreline polyline, index-aligned with link ids, as a
    /// self-describing flat buffer: per link `[point_count, x0, y0, x1, y1, …]`.
    /// The browser hit-tests clicks against these (the engine's real geometry),
    /// not the raw import, so selection stays in sync.
    pub fn link_polylines(&self) -> Vec<f32> {
        let mut out = Vec::new();
        for poly in &self.world.network.polylines {
            out.push(poly.len() as f32);
            for p in poly {
                out.push(p[0] as f32);
                out.push(p[1] as f32);
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
        geometry::world_mesh(&self.world.network)
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
        let lit = blink_on();
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
                    blinker: if lit { self.world.vehicle_blinker(v) as f32 } else { 0.0 },
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

    fn snapshot(&self) -> PoseMap {
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

/// Origin–destination demand for the selected mode: the boundary mix, or (in
/// highway mode) traffic anchored on the US-101 / I-280 gateways. Vehicles are
/// then routed live by the world's flow field.
/// How often (sim seconds) to kick off a GPU flow-field rebuild — matches the CPU
/// path's reroute cadence.
const GPU_REROUTE_SECS: f64 = 3.0;

impl Simulation {
    /// Pump the async GPU routing recompute once per frame: collect a finished
    /// readback into the router, else (when idle and due) dispatch a fresh one.
    /// Non-blocking — the browser can't wait on the GPU, so a rebuild spans frames.
    fn drive_gpu_routing(&mut self) {
        if self.gpu.is_none() {
            return;
        }
        // Collect an in-flight readback, if the GPU has finished it. Always unmap
        // (via `try_take`); only feed the result if it's still for the current
        // router generation — a demand toggle since dispatch invalidates it.
        if let Some((pending, generation)) = self.gpu_pending.take() {
            match self.gpu.as_ref().unwrap().try_take(&pending) {
                Some(fields) if generation == self.gpu_generation => {
                    let dist: Vec<Vec<u64>> = fields
                        .into_iter()
                        .map(|f| f.into_iter().map(|d| if d == u32::MAX { flowfield::UNREACHABLE } else { d as u64 }).collect())
                        .collect();
                    self.world.feed_router_distances(&self.gpu_cost, &dist);
                    self.gpu_last = self.world.time();
                }
                Some(_) => self.gpu_last = f64::NEG_INFINITY, // stale: unmapped, dispatch fresh next frame
                None => self.gpu_pending = Some((pending, generation)), // still mapping
            }
            return; // one GPU action per frame
        }
        // Idle: dispatch a rebuild when the field is due for a refresh.
        if self.world.time() - self.gpu_last < GPU_REROUTE_SECS {
            return;
        }
        let dests: Vec<u32> = self.world.router_dest_links().iter().map(|d| d.0).collect();
        if dests.is_empty() {
            return;
        }
        self.gpu_cost = self.world.live_link_costs();
        let cost32: Vec<u32> = self.gpu_cost.iter().map(|&c| c.min(u32::MAX as u64) as u32).collect();
        let readback = self.gpu.as_mut().unwrap().begin_readback(&cost32, &dests);
        self.gpu_pending = Some((readback, self.gpu_generation));
    }
}

fn build_demand(world: &NetWorld, seed: u64, sources: DemandSources, rate: f64, entry_cap: f64) -> DemandGenerator {
    let pairs = demand::od_pairs(&world.network, seed, 48, sources);
    let mut gen = DemandGenerator::new(world, &pairs, seed);
    gen.set_rate_scale(rate);
    gen.set_entry_speed_cap(entry_cap);
    gen.set_rush_hour(&world.network, sources.rush_hour);
    gen
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
