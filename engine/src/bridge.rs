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
use crate::render::interp;
use crate::render::scene::{brake_intensity, class_color, class_dims, crash_instance, edge_ribbon, signal_color, signal_head_instances, train_color};
use crate::render::gpu::Renderer;
use crate::render::{geometry, Instance, StaticMesh, StaticVertex};
use crate::sim::clock::SimClock;
use crate::sim::flowfield;
use crate::sim::flowfield_gpu::{GpuFlowField, PendingFlag, PendingReadback};
use crate::sim::config::{SimConfig, VehicleClass};
use crate::sim::demand::{self, DemandGenerator, DemandSources};
use crate::sim::congestion::CongestionConfig;
use crate::sim::map;
use crate::sim::net_world::{AccelBackend, NetWorld};
use crate::sim::network::{LinkId, Network, RoadKind, LANE_WIDTH};

/// Soft wall-clock budget (ms) for one frame's catch-up stepping. Past this, the
/// frame stops advancing the sim and renders, dropping the backlog so a heavy step
/// degrades to slow-motion instead of locking the main thread.
const FRAME_BUDGET_MS: f64 = 8.0;
/// Catch-up budget (ms) when the frame budget is *off* and the camera is idle: a big batch of
/// steps amortizes the per-frame render/marshal cost (the reason "off" runs so much faster),
/// while still capping any single frame so it can't lock the worker for seconds.
const IDLE_BURST_MS: f64 = 100.0;
/// After a pan/zoom, keep advancing at the responsive `FRAME_BUDGET_MS` for this long even with
/// the frame budget off, so a heavy catch-up batch never starves the interaction.
const CAMERA_ACTIVE_MS: f64 = 250.0;
/// Hard ceiling on catch-up ticks per frame. The wall-clock budget is the real
/// limiter; this only bounds the loop if no monotonic clock is available.
const MAX_CATCHUP_TICKS: u32 = 240;
/// Window over which the achieved-speed meter averages, long enough to smooth the
/// sub-tick-per-frame quantization at low speeds.
const SPEED_WINDOW_SECS: f64 = 0.5;

/// Monotonic wall-clock milliseconds from the browser's high-resolution timer.
/// Resolves `performance` from whichever global is current — the `Window` on the main
/// thread or the `WorkerGlobalScope` inside the engine worker (where `window()` is
/// `None`) — so the frame budget and speed meter work in both. Returns 0.0 if truly
/// unavailable, which disables the budget (falling back to the tick ceiling).
fn now_ms() -> f64 {
    use wasm_bindgen::JsCast;
    let global = js_sys::global();
    if let Some(perf) = global.dyn_ref::<web_sys::WorkerGlobalScope>().and_then(|s| s.performance()) {
        return perf.now();
    }
    if let Some(perf) = global.dyn_ref::<web_sys::Window>().and_then(|w| w.performance()) {
        return perf.now();
    }
    0.0
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
    /// Real LODES commute flows (`tools/lodes`), once loaded via `set_commute_od`;
    /// carried across demand rebuilds so toggles keep the measured streams.
    commute: Option<demand::CommuteOd>,
    /// Spawn-rate multiplier and entry-speed cap (m/s), carried across demand rebuilds
    /// so the UI's frequency / start-speed controls persist when sources are toggled.
    demand_rate: f64,
    entry_speed_cap: f64,
    /// Day-seconds per sim second the simulated day plays at, carried across demand
    /// rebuilds like the other demand controls. 1.0 = real time.
    day_compression: f64,
    /// Ramp-metering master switch; actual activation follows the day-clock
    /// peak windows (see `apply_meter_schedule`).
    metering_enabled: bool,
    /// Named bus lines resolved from the map, carried across demand rebuilds.
    transit_lines: Vec<demand::TransitLine>,
    /// The compiled transit artifact as loaded, so the transit toggle can
    /// re-apply it after a disable; `transit_enabled` gates trains + real trips.
    transit_json: Option<String>,
    transit_enabled: bool,
    /// A wreck-clearance duration the user expressed in day-clock minutes, so a
    /// compression change re-derives the sim-seconds value it maps to.
    wreck_clear_day_minutes: Option<f64>,
    camera: Camera,
    /// `[x, y, heading, speed]` of each vehicle one tick ago, keyed by id, so the
    /// render interpolates pose between committed states (smooth at 60fps) and
    /// derives brake lights from the speed delta.
    prev: PoseMap,
    /// Each vehicle's lane one tick ago, keyed by id, so the render can detect a node
    /// crossing between `prev` and now and curve the interpolation through the corner
    /// (see [`control_point`]) instead of cutting straight across the intersection.
    prev_lane: IntMap<u32>,
    prev_crossing: IntMap<bool>,
    /// The link the user has selected (clicked), highlighted in the density pass.
    selected: Option<usize>,
    /// Curbside signal-head placements `(group, pos, heading, is_left)` — static
    /// geometry computed once at assembly; each frame only pairs them with live colours.
    signal_heads: Vec<(usize, [f32; 2], f32, bool)>,
    /// Optional GPU flow-field solver on the renderer's WebGPU device, plus the
    /// in-flight async readback and the cost snapshot it was dispatched with.
    gpu: Option<GpuFlowField>,
    /// An in-flight GPU readback paired with the router generation it was dispatched
    /// under. A demand toggle reinstalls the router (bumping the generation); the
    /// stale readback must still be *collected* (to unmap its staging buffer) but its
    /// distances are discarded, never fed into the new field.
    gpu_pending: Option<(PendingReadback, u64)>,
    /// An in-flight relax-chunk flag readback (the reroute is mid-convergence). Same
    /// generation tagging as `gpu_pending` so a reinstall abandons a stale solve.
    gpu_relax: Option<(PendingFlag, u64)>,
    /// Bumped whenever the router is reinstalled, to invalidate an in-flight readback.
    gpu_generation: u64,
    gpu_cost: Vec<u64>,
    gpu_last: f64,
    /// Congestion fingerprint at the last GPU solve; a solve is kicked off only when it moves.
    gpu_fingerprint: u64,
    /// Achieved sim-time / wall-time over the last window — the speed the sim is
    /// *actually* running at (see [`Simulation::meter_speed`]).
    effective_speed: f64,
    /// True when the frame budget is dropping catch-up ticks (can't hold the selected
    /// speed). Surfaced instead of a noisy per-frame ratio.
    throttled: bool,
    /// `now_ms()` at the previous `advance`, for the inter-frame wall delta.
    last_advance_ms: f64,
    /// `now_ms()` of the last camera move (pan/zoom/fit). While recent, `advance` renders at
    /// display rate even with the frame budget off, so interaction stays smooth mid-burst.
    last_camera_ms: f64,
    /// Rolling accumulators for the speed meter: sim- and wall-seconds since the last
    /// window flush, plus whether any backlog was dropped within it.
    speed_sim_accum: f64,
    speed_wall_accum: f64,
    speed_dropped: bool,
    /// Whether the per-frame wall-clock budget caps catch-up stepping. On (default): an
    /// overloaded frame drops backlog so the view stays smooth and the sim runs slower
    /// than the selected speed. Off: run the full catch-up (to the tick ceiling) so the
    /// sim holds the selected speed and the frame rate drops instead.
    frame_budget: bool,
    /// Whether to emit the crash-location overlay markers. Off by default.
    show_crashes: bool,
    /// The junction the user selected (stats panel + footprint highlight).
    selected_junction: Option<usize>,
    /// The static world surface as one fill mesh, built lazily the first time the ASCII
    /// view is requested and reused every frame after — the network geometry never changes
    /// once assembled, so this avoids rebuilding the whole city's triangles per frame (and
    /// costs no memory for sessions that never open the ASCII view).
    ascii_fill: Option<StaticMesh>,
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
    pub fn from_map_json(json: &str, seed: u32, split_junctions: bool) -> Result<Simulation, JsValue> {
        let map = map::OsmMap::from_json_opts(json, split_junctions).map_err(|e| JsValue::from_str(&e))?;
        let mut net = map.build();
        net.rail = crate::sim::rail::RailNetwork::from_map_json(json);
        net.attach_bus_stops(&map::bus_stops_from_json(json));
        let lines: Vec<demand::TransitLine> = map::bus_routes_from_json(json)
            .into_iter()
            .filter_map(|(name, pts)| net.resolve_route_chain(&pts).map(|route| demand::TransitLine::new(name, route)))
            .collect();
        let mut sim = Self::assemble(net, seed);
        sim.transit_lines = lines;
        sim.demand.set_transit_lines(sim.transit_lines.clone());
        Ok(sim)
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
        let demand = build_demand(&world, cfg.seed, demand_sources, demand_rate, entry_speed_cap, None);
        world.install_router(&demand.destinations());
        let mut clock = SimClock::new(&cfg);
        clock.play();
        let signal_heads = geometry::signal_head_placements(&world.network);
        Simulation {
            world, clock, seed: cfg.seed, demand, demand_sources, commute: None, demand_rate, entry_speed_cap,
            day_compression: demand::DEFAULT_DAY_COMPRESSION, wreck_clear_day_minutes: None,
            metering_enabled: true, transit_lines: Vec::new(), transit_json: None, transit_enabled: true, camera,
            prev: PoseMap::default(), prev_lane: IntMap::default(), prev_crossing: IntMap::default(), selected: None, signal_heads,
            gpu: None, gpu_pending: None, gpu_relax: None, gpu_generation: 0, gpu_cost: Vec::new(), gpu_last: 0.0, gpu_fingerprint: 0,
            effective_speed: 0.0, throttled: false, last_advance_ms: 0.0, last_camera_ms: 0.0,
            speed_sim_accum: 0.0, speed_wall_accum: 0.0, speed_dropped: false,
            frame_budget: true,
            show_crashes: false,
            selected_junction: None,
            ascii_fill: None,
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
        self.gpu_relax = None;
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

    /// Load real commute OD flows (`tools/lodes/fetch_lodes.py` output): measured
    /// home→work streams — AM toward work, PM the reverse — join the sampled
    /// categories, displacing their volume in proportion to the measured share
    /// (never their coverage). Rebuilds demand live (non-destructive, like the
    /// source toggles). Returns `false` on a parse failure, leaving demand unchanged.
    #[cfg(feature = "import")]
    pub fn set_commute_od(&mut self, json: &str) -> bool {
        match demand::CommuteOd::from_json(json) {
            Ok(od) => {
                self.commute = Some(od);
                self.rebuild_demand();
                true
            }
            Err(_) => false,
        }
    }

    /// Install a compiled transit artifact (`tools/gtfs` output): the rail
    /// timetable snaps onto the map's rail lines (crossing closures switch from
    /// the synthetic cadence to real trains), and bus trips attach to their
    /// named transit lines (real departures + timepoint holding replace the
    /// synthetic headway). Returns `[rail_kept, rail_dropped, bus_kept,
    /// bus_dropped]`; soft-fails per trip, never breaking the sim.
    #[cfg(feature = "import")]
    pub fn set_transit_json(&mut self, json: &str) -> Vec<u32> {
        self.transit_json = Some(json.to_string());
        self.transit_enabled = true;
        self.apply_transit_json(json)
    }

    /// Master transit switch: off clears the timetable (crossings fall back to
    /// the synthetic cadence) and reverts every line to headway service; on
    /// re-applies the stored artifact. Live and non-destructive.
    pub fn set_transit_enabled(&mut self, on: bool) {
        if on == self.transit_enabled {
            return;
        }
        self.transit_enabled = on;
        #[cfg(feature = "import")]
        if on {
            if let Some(json) = self.transit_json.clone() {
                self.apply_transit_json(&json);
            }
            return;
        }
        self.world.set_timetable(Default::default());
        for line in &mut self.transit_lines {
            line.set_trips(Vec::new());
        }
        self.demand.set_transit_lines(self.transit_lines.clone());
    }

    pub fn transit_enabled(&self) -> bool {
        self.transit_enabled
    }

    #[cfg(feature = "import")]
    fn apply_transit_json(&mut self, json: &str) -> Vec<u32> {
        use crate::sim::net_world::ScheduledStop;
        use crate::sim::rail;
        let Ok((rail_specs, bus_specs)) = rail::transit_from_json(json) else {
            return vec![0, 0, 0, 0];
        };
        let (tt, rail_dropped) = rail::build_timetable(&self.world.network.rail, &rail_specs);
        let rail_kept = tt.trips.len() as u32;
        self.world.set_timetable(tt);

        // Buses: resolve each trip's stops onto the road network, group by line
        // name, attach to the matching scraped line (or build a new line from
        // the stop trace when no OSM relation matched it). Trips of one line
        // share physical stops, so resolution is memoized by position — the
        // difference between a sub-second load and a frozen progress bar on a
        // city-sized artifact (nearest_surface_link scans every road segment).
        let (mut bus_kept, mut bus_dropped) = (0u32, 0u32);
        let mut by_line: std::collections::BTreeMap<String, (Vec<demand::BusTrip>, Vec<[f64; 2]>)> =
            std::collections::BTreeMap::new();
        let mut stop_cache: std::collections::HashMap<(i64, i64), Option<(u32, f64)>> = std::collections::HashMap::new();
        for spec in bus_specs {
            let mut stops = Vec::new();
            for st in &spec.stops {
                let key = ((st.pos[0] * 4.0).round() as i64, (st.pos[1] * 4.0).round() as i64);
                let net = &self.world.network;
                let hit = *stop_cache.entry(key).or_insert_with(|| {
                    net.nearest_surface_link(st.pos)
                        .filter(|&(_, _, d)| d <= 30.0)
                        .map(|(link, arc, _)| (link.0, arc))
                });
                if let Some((link, arc)) = hit {
                    stops.push(ScheduledStop {
                        link,
                        arc,
                        departure: st.departure.rem_euclid(86_400.0),
                        timepoint: st.timepoint,
                    });
                }
            }
            if stops.len() < 2 {
                bus_dropped += 1;
                continue;
            }
            let departure = spec.stops.first().map_or(0.0, |s| s.departure);
            let entry = by_line.entry(spec.line.clone()).or_default();
            if spec.stops.len() > entry.1.len() {
                entry.1 = spec.stops.iter().map(|s| s.pos).collect();
            }
            entry.0.push(demand::BusTrip { weekend: spec.weekend, departure, stops });
            bus_kept += 1;
        }
        let names_match = |a: &str, b: &str| {
            let (a, b) = (a.to_lowercase(), b.to_lowercase());
            a == b || a.contains(&b) || b.contains(&a)
        };
        for (name, (trips, rep_pts)) in by_line {
            let n = trips.len() as u32;
            if let Some(line) = self.transit_lines.iter_mut().find(|l| names_match(&l.name, &name)) {
                line.set_trips(trips);
            } else if let Some(route) = self.world.network.resolve_route_chain(&rep_pts) {
                let mut line = demand::TransitLine::new(name, route);
                line.set_trips(trips);
                self.transit_lines.push(line);
            } else {
                bus_dropped += n;
                bus_kept -= n;
            }
        }
        self.demand.set_transit_lines(self.transit_lines.clone());
        vec![rail_kept, rail_dropped as u32, bus_kept, bus_dropped]
    }

    /// Rebuild the demand generator for a new source mix and reinstall the router over
    /// its destinations plus those of cars already on the road, so in-flight trips
    /// aren't stranded. No-op if the mix is unchanged.
    fn apply_demand_sources(&mut self, sources: DemandSources) {
        if sources == self.demand_sources {
            return;
        }
        self.demand_sources = sources;
        self.rebuild_demand();
    }

    /// Swap in a freshly built demand generator (current sources/commute data) and
    /// reinstall the router, preserving live-vehicle ids and destinations.
    fn rebuild_demand(&mut self) {
        // Carry the id counter past every vehicle still on the road (and the old
        // generator's own counter), so the rebuilt generator never reissues a live id.
        // A restart at 0 would alias two cars onto one id in the render's per-id `prev`
        // map, flashing one of them across the screen.
        let next_id = self
            .world
            .vehicles()
            .iter()
            .map(|v| v.id)
            .max()
            .map_or(0, |m| m + 1)
            .max(self.demand.next_id());
        let clock = self
            .demand
            .rush_hour_active()
            .then(|| (self.demand.rush_hour_day_secs(), self.demand.day()));
        self.demand = build_demand(
            &self.world,
            self.seed,
            self.demand_sources,
            self.demand_rate,
            self.entry_speed_cap,
            self.commute.as_ref(),
        );
        self.demand.set_next_id(next_id);
        self.demand.set_day_compression(self.day_compression);
        self.demand.set_transit_lines(self.transit_lines.clone());
        if let Some((secs, day)) = clock {
            self.demand.resume_clock(secs, day);
        }
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

    /// Simulated wall-clock time of day (hours, 0–24) for the HUD clock — defined in
    /// every mode, unlike [`Self::rush_hour_time`], which reads 0 when rush hour is off.
    pub fn day_time_hours(&self) -> f64 {
        self.demand.day_secs(self.world.time()) / 3600.0
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

    /// Master ramp-metering switch (default on). Meters run only during the real
    /// D4 metering windows (AM/PM peaks) when the day clock is live; with rush
    /// hour off they follow the switch directly.
    pub fn set_ramp_metering(&mut self, enabled: bool) {
        self.metering_enabled = enabled;
        self.apply_meter_schedule();
    }

    pub fn ramp_metering(&self) -> bool {
        self.world.ramp_metering()
    }

    /// Apply the time-of-day metering schedule: within rush-hour mode the meters
    /// switch with the simulated clock (6–10 h and 15–19 h, the Caltrans D4
    /// pattern); otherwise the master switch alone decides.
    fn apply_meter_schedule(&mut self) {
        let on = self.metering_enabled
            && (!self.demand.rush_hour_active() || {
                let h = self.demand.rush_hour_day_secs() / 3600.0;
                (6.0..10.0).contains(&h) || (15.0..19.0).contains(&h)
            });
        if on != self.world.ramp_metering() {
            self.world.set_ramp_metering(on);
        }
    }

    /// How fast the simulated day plays: day-seconds per sim second (1 = real time,
    /// 60 = the default 24 h-in-24 min). Only the day clock scales — traffic dynamics
    /// run in real sim time at any setting. Live, non-destructive, carried across
    /// demand rebuilds; a day-clock-expressed wreck clearance is re-derived.
    pub fn set_day_compression(&mut self, x: f64) {
        self.demand.set_day_compression(x);
        self.day_compression = self.demand.day_compression();
        if let Some(mins) = self.wreck_clear_day_minutes {
            self.world.set_wreck_clear_secs(mins * 60.0 / self.day_compression);
        }
    }

    pub fn day_compression(&self) -> f64 {
        self.day_compression
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
        self.touch_camera();
    }

    /// Pan by a drag delta in canvas pixels.
    pub fn pan_pixels(&mut self, dx: f32, dy: f32) {
        self.camera.pan_pixels(dx as f64, dy as f64);
        self.touch_camera();
    }

    /// Zoom by `factor` (<1 zooms in) about a canvas pixel (mouse wheel).
    pub fn zoom_at(&mut self, factor: f32, sx: f32, sy: f32) {
        self.camera.zoom_at(factor as f64, [sx as f64, sy as f64]);
        self.touch_camera();
    }

    pub fn set_meters_per_pixel(&mut self, mpp: f32) {
        self.camera.meters_per_pixel = (mpp as f64).clamp(0.02, 10_000.0);
        self.touch_camera();
    }

    /// Note that the camera just moved, so [`advance`](Self::advance) keeps rendering at display
    /// rate for a short window afterwards — a heavy catch-up batch never starves a pan/zoom.
    fn touch_camera(&mut self) {
        self.last_camera_ms = now_ms();
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
        self.prev_lane = self.snapshot_lanes();
        self.prev_crossing = self.snapshot_crossing();
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
        self.apply_meter_schedule();
        // Day-scheduled infrastructure (rail-crossing timetables) reads the
        // simulated time of day.
        let day = self.demand.day_secs(self.world.time());
        self.world.set_day_secs(day);
        let ticks = self.clock.advance(real_elapsed_secs, MAX_CATCHUP_TICKS);
        let dt = self.clock.dt();

        // How long this frame may spend catching up before it renders. Frame budget on → the
        // tight `FRAME_BUDGET_MS` (smooth 60 fps, drop the rest). Off → a big `IDLE_BURST_MS`
        // batch that amortizes the per-frame render/marshal cost (why "off" runs so much
        // faster), *except* for a short window after a pan/zoom, where it drops back to the
        // tight budget so the interaction renders at display rate instead of waiting out the
        // whole batch. Either way a single frame is bounded, so it never locks the worker.
        let camera_active = frame_ms - self.last_camera_ms < CAMERA_ACTIVE_MS;
        let budget = if self.frame_budget || camera_active { FRAME_BUDGET_MS } else { IDLE_BURST_MS };

        let mut ran = 0u32;
        if ticks > 0 {
            let loop_start = now_ms();
            while ran < ticks {
                // Once catch-up blows the frame's budget, treat this tick as the last: render
                // the result and drop the remaining backlog so a heavy step degrades to
                // slow-motion instead of freezing the worker. (Always run at least one tick,
                // hence `ran > 0`.)
                let over_budget = ran > 0 && now_ms() - loop_start > budget;
                let last = over_budget || ran + 1 == ticks;
                // The render interpolates between `prev` and the post-step state, so
                // only the snapshot taken right before the *final* committed tick is
                // used; taking it once (not per catch-up tick) avoids rebuilding the
                // full pose map several times per frame.
                if last {
                    self.prev = self.snapshot();
                    self.prev_lane = self.snapshot_lanes();
                    self.prev_crossing = self.snapshot_crossing();
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

    /// Toggle the per-frame wall-clock budget. On (default): cap catch-up to keep the view
    /// smooth, letting the sim fall behind the selected speed under load. Off: run the full
    /// catch-up so the sim holds the selected speed and the frame rate drops instead.
    pub fn set_frame_budget(&mut self, enabled: bool) {
        self.frame_budget = enabled;
    }

    pub fn frame_budget(&self) -> bool {
        self.frame_budget
    }

    /// Choose the accel executor: `"serial"`, `"threads"`, or `"gpu"`. The sim falls
    /// back automatically when the request isn't available on this device (no CPU
    /// worker pool; the GPU evaluate runs native-only), so the UI can offer all three.
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


    /// Toggle solving several routing (flow-field) recompute destinations at once across the
    /// worker pool vs. one at a time. On by default; exposed so the parallel routing speed-up
    /// is observable. No effect on the single-threaded build or the browser-GPU routing path.
    pub fn set_parallel_routing(&mut self, on: bool) {
        self.world.set_parallel_routing(on);
    }

    pub fn parallel_routing(&self) -> bool {
        self.world.parallel_routing()
    }

    /// Toggle stop/yield control delay in routing costs — drivers plan around the
    /// ~9 s a four-way stop really costs, keeping through-traffic on the arterials
    /// instead of rat-running the stop-sign grid. On by default; takes effect on
    /// the next reroute cycle.
    pub fn set_control_aware_routing(&mut self, on: bool) {
        self.world.set_control_aware_routing(on);
    }

    /// Toggle the human-cadence lane-decision stagger: away from a junction (and
    /// while parked in queue) a driver re-weighs a discretionary lane change about
    /// once a second, not every tick. On by default; cuts the lane-change scan
    /// substantially at city scale with no change to mandatory turn positioning.
    pub fn set_lane_eval_stagger(&mut self, on: bool) {
        self.world.set_lane_eval_stagger(on);
    }

    /// Toggle arterial-first routing: through-plans are solved over the arterial
    /// network plus each destination's local access streets, and a car on outer
    /// local fabric first drives to a main road — the way drivers actually plan.
    /// Halves the routing solve and startup cost; rebuilds the router on toggle
    /// (a brief hitch on a city map).
    pub fn set_arterial_routing(&mut self, on: bool) {
        self.world.set_arterial_routing(on);
    }

    /// Toggle targeted route refresh: reroute cycles skip destination fields whose
    /// answers still price correctly, and solve the rest only far enough to cover
    /// the cars (and spawn gateways) that will actually read them — the rest of
    /// each field keeps its previous, still-valid values. On by default; uncheck
    /// to run every cycle exhaustively for comparison.
    pub fn set_targeted_routing(&mut self, on: bool) {
        self.world.set_targeted_routing(on);
    }

    /// Toggle the cache-friendly per-lane/per-corridor sort (a flat position-key array instead
    /// of reading a vehicle row per comparison). On by default; a display-neutral performance
    /// option — the simulation result is identical either way.
    pub fn set_cache_sort(&mut self, on: bool) {
        self.world.set_cache_sort(on);
    }

    pub fn cache_sort(&self) -> bool {
        self.world.cache_sort()
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

    /// Cumulative vehicles wrecked in a collision (wrecks may still be on the
    /// road awaiting clearance).
    pub fn crashed(&self) -> u32 {
        self.world.crashed()
    }

    /// The crashed tally split by nature: `[rear_end, junction]`.
    pub fn crash_counts(&self) -> Vec<u32> {
        self.world.crash_counts().to_vec()
    }

    /// Wreck persistence: seconds a crashed vehicle stays on the road blocking
    /// traffic (0 = removed instantly, the default). Overrides any prior
    /// day-clock-expressed duration.
    pub fn set_wreck_clear_secs(&mut self, secs: f64) {
        self.wreck_clear_day_minutes = None;
        self.world.set_wreck_clear_secs(secs);
    }

    /// Wreck clearance expressed in *day-clock* minutes: "a 20-minute incident" means
    /// 20 simulated-day minutes at any compression, so the sim-seconds duration is
    /// `mins·60 / compression` and re-derives when the compression changes. 0 clears.
    pub fn set_wreck_clear_day_minutes(&mut self, mins: f64) {
        if mins <= 0.0 {
            self.wreck_clear_day_minutes = None;
            self.world.set_wreck_clear_secs(0.0);
            return;
        }
        self.wreck_clear_day_minutes = Some(mins);
        self.world.set_wreck_clear_secs(mins * 60.0 / self.day_compression);
    }

    pub fn wreck_clear_secs(&self) -> f64 {
        self.world.wreck_clear_secs()
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

    /// The junction whose footprint contains the world point, else -1 — the
    /// click target for intersection selection.
    pub fn junction_hit(&self, wx: f64, wy: f64) -> i32 {
        let inside = |fp: &[[f64; 2]; 4]| {
            let mut sign = 0.0f64;
            for k in 0..4 {
                let (a, b) = (fp[k], fp[(k + 1) % 4]);
                let c = (b[0] - a[0]) * (wy - a[1]) - (b[1] - a[1]) * (wx - a[0]);
                if c.abs() < 1e-9 {
                    continue;
                }
                if sign == 0.0 {
                    sign = c.signum();
                } else if c.signum() != sign {
                    return false;
                }
            }
            true
        };
        self.world
            .network
            .junctions
            .iter()
            .position(|j| inside(&j.footprint))
            .map_or(-1, |i| i as i32)
    }

    /// Select a junction for the stats panel and footprint highlight (negative clears).
    pub fn set_selected_junction(&mut self, index: i32) {
        self.selected_junction =
            (index >= 0 && (index as usize) < self.world.network.junctions.len()).then_some(index as usize);
    }

    /// The crossing's display name: its two busiest distinct street names.
    pub fn junction_label(&self, index: u32) -> String {
        let Some(j) = self.world.network.junctions.get(index as usize) else { return String::new() };
        let mut names: Vec<&str> = Vec::new();
        for &l in j.approaches.iter().chain(&j.exits) {
            let n = self.world.network.link_names[l.idx()].as_str();
            if !n.is_empty() && !names.contains(&n) {
                names.push(n);
            }
        }
        match names[..] {
            [] => format!("junction {index}"),
            [a] => a.to_string(),
            [a, b, ..] => format!("{a} × {b}"),
        }
    }

    /// The control regime the junction runs: "signal", "all-way stop", "yield",
    /// or "uncontrolled".
    pub fn junction_control(&self, index: u32) -> String {
        let Some(j) = self.world.network.junctions.get(index as usize) else { return String::new() };
        use crate::sim::network::NodeControl;
        if j.program.is_some() {
            return "signal".into();
        }
        let controls = j.nodes.iter().map(|&nd| self.world.network.node(nd).control);
        if controls.clone().any(|c| matches!(c, NodeControl::Stop)) {
            "all-way stop".into()
        } else if controls.clone().any(|c| matches!(c, NodeControl::Yield)) {
            "yield".into()
        } else {
            "uncontrolled".into()
        }
    }

    /// Live junction stats: `[queued_on_approaches, crossing_inside,
    /// longest_current_wait_secs, throughput_vph]`.
    pub fn junction_stats(&self, index: u32) -> Vec<f32> {
        let Some(j) = self.world.network.junctions.get(index as usize) else { return vec![0.0; 4] };
        let net = &self.world.network;
        let approach: std::collections::HashSet<u32> = j.approaches.iter().map(|l| l.0).collect();
        let nodes: std::collections::HashSet<u32> = j.nodes.iter().map(|n| n.0).collect();
        let (mut queued, mut inside, mut max_wait) = (0u32, 0u32, 0u32);
        for v in self.world.vehicles() {
            if v.is_crossing() {
                if nodes.contains(&net.link(net.lane(v.lane).link).to.0) {
                    inside += 1;
                }
                continue;
            }
            let link = net.lane(v.lane).link;
            let internal = nodes.contains(&net.link(link).from.0) && nodes.contains(&net.link(link).to.0);
            if internal {
                inside += 1;
            } else if approach.contains(&link.0) && v.speed < 0.5 {
                queued += 1;
                max_wait = max_wait.max(v.wait_ticks());
            }
        }
        let flows = self.world.link_flows();
        let vph: f64 = j.exits.iter().map(|l| flows[l.idx()]).sum();
        vec![queued as f32, inside as f32, (max_wait as f64 * self.clock.dt()) as f32, vph as f32]
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

    /// `[x, y, heading, brake, blinker, length, width, class]` per vehicle:
    /// pose interpolated between the last two ticks by the clock's sub-tick
    /// `alpha`, a brake-light intensity in `[0,1]` from deceleration, a
    /// turn-signal side (`-1` left, `+1` right, `0` none) already gated by the
    /// blink phase, and the class dimensions/id (0 car, 1 truck, 2 bus) so the
    /// 2D fallback draws a bus as a bus, not a car-sized quad.
    pub fn vehicle_instances(&self) -> Vec<f32> {
        let alpha = self.clock.alpha() as f32;
        let dt = self.clock.dt() as f32;
        let lit = blink_on();
        let mut out = Vec::with_capacity(self.world.vehicles().len() * 8);
        for v in self.world.vehicles() {
            let c = self.world.vehicle_world_pose(v);
            let (cx, cy, ch, cs) = (c[0] as f32, c[1] as f32, c[2] as f32, v.speed as f32);
            // Quadratic Bézier through the turn control (a straight lerp when it's the
            // midpoint), so a car crossing a node curves through the corner even when
            // high-speed catch-up skipped the per-tick interior samples.
            let (x, y, h, brake) = match self.prev.get(&v.id) {
                Some(&[px, py, ph, ps]) => {
                    let [kx, ky] = self.control_point(v.id, v.lane.0, [px, py], [cx, cy]);
                    let (u, m, a2) = ((1.0 - alpha) * (1.0 - alpha), 2.0 * (1.0 - alpha) * alpha, alpha * alpha);
                    (
                        u * px + m * kx + a2 * cx,
                        u * py + m * ky + a2 * cy,
                        ph + shortest_angle(ph, ch) * alpha,
                        brake_intensity((cs - ps) / dt),
                    )
                }
                None => (cx, cy, ch, 0.0),
            };
            let blinker = if lit { self.world.vehicle_blinker(v) as f32 } else { 0.0 };
            let class = VehicleClass::from_length(v.driver.vehicle_length);
            let dims = class_dims(class);
            let class_id = match class {
                VehicleClass::Car => 0.0,
                VehicleClass::Truck => 1.0,
                VehicleClass::Bus => 2.0,
            };
            out.extend_from_slice(&[x, y, h, brake, blinker, dims[0], dims[1], class_id]);
        }
        out
    }

    /// Signal heads as `[x, y, r, g, b, heading, is_left]`, one per signal group,
    /// placed at the stop line of the approach the group controls and coloured by
    /// its state; `is_left` (1/0) flags a protected-left group so the 2D fallback
    /// can draw a turn arrow instead of a square.
    pub fn signal_heads(&self) -> Vec<f32> {
        let mut out = Vec::new();
        for (pos, heading, state, is_left) in self.signal_head_slots() {
            let col = signal_color(state);
            out.extend_from_slice(&[pos[0], pos[1], col[0], col[1], col[2], heading, is_left as u8 as f32]);
        }
        out
    }

    /// Each signal group's head as `(position, approach heading, state, is_left)`,
    /// from the pure curbside placement in [`geometry::signal_head_placements`]
    /// paired with the group's live colour.
    fn signal_head_slots(&self) -> Vec<([f32; 2], f32, crate::sim::signal::SignalState, bool)> {
        let states = self.world.signal_states();
        self.signal_heads.iter().map(|&(gi, pos, heading, is_left)| (pos, heading, states[gi], is_left)).collect()
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
        flatten_static_vertices(self.world_mesh().vertices)
    }

    pub fn world_mesh_indices(&self) -> Vec<u32> {
        self.world_mesh().indices
    }

    /// Lane markings as flat `StaticVertex` floats (drawn only when zoomed in).
    pub fn marking_mesh_vertices(&self) -> Vec<f32> {
        flatten_static_vertices(geometry::marking_mesh(&self.world.network).vertices)
    }

    pub fn marking_mesh_indices(&self) -> Vec<u32> {
        geometry::marking_mesh(&self.world.network).indices
    }

    /// Painter's-order draw ranges into the world/marking index buffers, four
    /// `u32`s per render band: `[world_index_start, world_index_count,
    /// marking_index_start, marking_index_count]`. The renderer draws each band's
    /// fill then its markings in order, so a higher grade layer's fill covers the
    /// road and lane lines it crosses over (see [`geometry::world_bands`]). The
    /// ranges line up with `world_mesh_*`/`marking_mesh_*`, which concatenate the
    /// same bands in the same order.
    pub fn render_band_ranges(&self) -> Vec<u32> {
        let mut out = Vec::new();
        let (mut wi, mut mi) = (0u32, 0u32);
        for band in geometry::world_bands(&self.world.network) {
            let (wc, mc) = (band.fill.indices.len() as u32, band.marking.indices.len() as u32);
            out.extend_from_slice(&[wi, wc, mi, mc]);
            wi += wc;
            mi += mc;
        }
        out
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
                    control: self.control_point(v.id, v.lane.0, [px, py], [cx, cy]),
                    scale: class_dims(class),
                    color: class_color(class),
                    heading: ch,
                    prev_heading: ph,
                    brake: brake_intensity((cs - ps) / dt),
                    blinker: if lit { self.world.vehicle_blinker(v) as f32 } else { 0.0 },
                }
            })
            .collect();
        let mut instances = instances;
        instances.extend(self.train_instance_vec());
        bytemuck::cast_slice(&instances).to_vec()
    }

    pub fn render_instance_count(&self) -> u32 {
        (self.world.vehicles().len() + self.train_instance_vec().len()) as u32
    }

    /// Active trains as carriage-chain instances riding the same instanced draw
    /// as the vehicles: each train is a run of carriage quads placed along its
    /// line's chainage. Positions are pure functions of the day clock, so both
    /// the previous and current poses come from direct evaluation — trains
    /// never touch the per-id `prev` pose map (and need no ids at all).
    fn train_instance_vec(&self) -> Vec<Instance> {
        let tt = self.world.timetable();
        if tt.is_empty() {
            return Vec::new();
        }
        let (day, rate, weekend) = self.world.day_view();
        let dt = self.clock.dt();
        let rail = &self.world.network.rail;
        let now = tt.active_trains(day, weekend);
        let prev: crate::sim::hash::IntMap<f64> = tt
            .active_trains(day - dt * rate, weekend)
            .into_iter()
            .map(|t| (t.trip as u32, t.route_pos))
            .collect();
        let mut out = Vec::new();
        for tr in now {
            let trip = &tt.trips[tr.trip];
            let car = tr.class.car_length();
            let color = train_color(tr.class);
            let prev_front = prev.get(&(tr.trip as u32)).copied().unwrap_or(tr.route_pos);
            for k in 0..tr.carriages {
                let off = k as f64 * car + car * 0.5;
                let c = trip.pose(rail, tr.route_pos - off);
                let p = trip.pose(rail, prev_front - off);
                out.push(Instance {
                    pos: [c[0] as f32, c[1] as f32],
                    prev_pos: [p[0] as f32, p[1] as f32],
                    control: [((c[0] + p[0]) * 0.5) as f32, ((c[1] + p[1]) * 0.5) as f32],
                    scale: [(car * 0.94) as f32, tr.class.width() as f32],
                    color,
                    heading: c[2] as f32,
                    prev_heading: p[2] as f32,
                    brake: 0.0,
                    blinker: 0.0,
                });
            }
        }
        out
    }

    /// Train carriages for the 2D fallback: `[x, y, heading, length, width]`
    /// per carriage, pose already interpolated by the clock's sub-tick alpha.
    pub fn train_poses(&self) -> Vec<f32> {
        let alpha = self.clock.alpha() as f32;
        let mut out = Vec::new();
        for i in self.train_instance_vec() {
            let x = i.prev_pos[0] + (i.pos[0] - i.prev_pos[0]) * alpha;
            let y = i.prev_pos[1] + (i.pos[1] - i.prev_pos[1]) * alpha;
            let h = i.prev_heading + shortest_angle(i.prev_heading, i.heading) * alpha;
            out.extend_from_slice(&[x, y, h, i.scale[0], i.scale[1]]);
        }
        out
    }

    /// Signal heads — plus the selected junction's footprint highlight — as raw
    /// `Instance` bytes for the emissive draw.
    pub fn signal_instances(&self) -> Vec<u8> {
        bytemuck::cast_slice(&self.signal_instance_vec()).to_vec()
    }

    pub fn signal_instance_count(&self) -> u32 {
        self.signal_instance_vec().len() as u32
    }

    /// Crash-site markers as raw `Instance` bytes (empty when the overlay is off). Drawn at
    /// every zoom, so each marker is sized in world metres to hold a constant on-screen size.
    pub fn crash_instances(&self) -> Vec<u8> {
        bytemuck::cast_slice(&self.crash_instance_vec()).to_vec()
    }

    pub fn crash_instance_count(&self) -> u32 {
        self.crash_instance_vec().len() as u32
    }

    /// Toggle the crash-location overlay. Independent of the sim; purely a display option.
    pub fn set_show_crashes(&mut self, on: bool) {
        self.show_crashes = on;
    }

    pub fn show_crashes(&self) -> bool {
        self.show_crashes
    }

    /// Forget the recorded crash sites — the overlay clears immediately.
    pub fn clear_crashes(&mut self) {
        self.world.clear_crash_sites();
    }

    fn crash_instance_vec(&self) -> Vec<Instance> {
        if !self.show_crashes {
            return Vec::new();
        }
        // Constant ~9 px marker regardless of zoom (world size = pixels · metres-per-pixel), so
        // collision hot-spots are visible fitted to the whole city and up close alike. Culled to
        // the viewport with a margin; the site list is already capped in the sim.
        const CRASH_MARKER_PX: f64 = 9.0;
        const CRASH_CULL_MARGIN_M: f64 = 60.0;
        let mpp = self.camera.meters_per_pixel;
        let size = (CRASH_MARKER_PX * mpp) as f32;
        let c = self.camera.center;
        let hx = self.camera.viewport[0] * mpp * 0.5 + CRASH_CULL_MARGIN_M;
        let hy = self.camera.viewport[1] * mpp * 0.5 + CRASH_CULL_MARGIN_M;
        self.world
            .crash_log()
            .iter()
            .filter(|r| (r.pos[0] as f64 - c[0]).abs() <= hx && (r.pos[1] as f64 - c[1]).abs() <= hy)
            .map(|r| crash_instance(r.pos, size))
            .collect()
    }

    /// Translucent per-link traffic overlay: carriageway quads tinted by live occupancy —
    /// a blue presence tint for a car or two, the percentage congestion heatmap once a
    /// link fills up — emitted for every link that has cars. `StaticVertex` floats; pair
    /// with [`density_indices`].
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
        geometry::occupancy_mesh(net, &counts, self.selected)
    }

    fn signal_instance_vec(&self) -> Vec<Instance> {
        // This buffer is rebuilt and re-uploaded to the GPU every frame, and a whole city has
        // tens of thousands of signal heads (Columbus: ~10k heads → 40k instances, ~2.4 MB).
        // Signals are ~5 m across, so past a couple of metres per pixel they're sub-pixel:
        // skip them entirely when zoomed out, and otherwise cull to the viewport, so the
        // per-frame cost (and the alloc/upload churn that stalls the render loop) scales with
        // *visible* signals, not the whole map. The marking mesh is zoom-gated the same way.
        const SIGNAL_MAX_MPP: f64 = 2.0;
        const SIGNAL_CULL_MARGIN_M: f64 = 40.0;
        let mpp = self.camera.meters_per_pixel;
        // The selected junction's footprint outline rides this stream at every
        // zoom, sized in world metres to hold a constant on-screen width.
        let mut out: Vec<Instance> = Vec::new();
        if let Some(ji) = self.selected_junction {
            let fp = self.world.network.junctions[ji].footprint;
            let hw = (1.5 * mpp).clamp(0.35, 3.0) as f32;
            for k in 0..4 {
                let (a, b) = (fp[k], fp[(k + 1) % 4]);
                out.push(edge_ribbon(
                    [a[0] as f32, a[1] as f32],
                    [b[0] as f32, b[1] as f32],
                    hw,
                    crate::render::geometry::HIGHLIGHT_COLOR,
                ));
            }
        }
        if mpp > SIGNAL_MAX_MPP {
            return out;
        }
        let c = self.camera.center;
        let hx = self.camera.viewport[0] * mpp * 0.5 + SIGNAL_CULL_MARGIN_M;
        let hy = self.camera.viewport[1] * mpp * 0.5 + SIGNAL_CULL_MARGIN_M;
        let states = self.world.signal_states();
        out.extend(
            self.signal_heads
                .iter()
                .filter(|&&(_, pos, _, _)| (pos[0] as f64 - c[0]).abs() <= hx && (pos[1] as f64 - c[1]).abs() <= hy)
                .flat_map(|&(gi, pos, heading, is_left)| signal_head_instances(pos, heading, states[gi], is_left)),
        );
        out
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

    fn snapshot_lanes(&self) -> IntMap<u32> {
        self.world.vehicles().iter().map(|v| (v.id, v.lane.0)).collect()
    }

    fn snapshot_crossing(&self) -> IntMap<bool> {
        self.world.vehicles().iter().map(|v| (v.id, v.is_crossing())).collect()
    }

    fn control_point(&self, v_id: u32, cur_lane: u32, prev: [f32; 2], cur: [f32; 2]) -> [f32; 2] {
        interp::control_point(
            &self.world.network,
            cur_lane,
            self.prev_lane.get(&v_id).copied(),
            self.prev_crossing.get(&v_id).copied().unwrap_or(false),
            prev,
            cur,
        )
    }

    /// Static carriageway quads `[cx0, cy0, cx1, cy1, width]` per link.
    pub fn road_strips(&self) -> Vec<f32> {
        flatten(self.world.network.road_strips())
    }

    /// Static lane-divider segments `[x0, y0, x1, y1]`.
    pub fn lane_dividers(&self) -> Vec<f32> {
        flatten(self.world.network.lane_dividers())
    }

    /// The current view rendered to a colourised ASCII grid (HTML for a `<pre>`) — the same
    /// [`ascii::Ascii`] rasteriser the native tests use, fed the exact geometry the GPU
    /// renderer draws. Roads (`#`) are shaded by class, vehicles (`@`) by speed (green
    /// free-flow → red stopped), and signal heads (`O`) by state (red/amber/green). `rows`
    /// sets the vertical resolution; the column count is derived from the camera aspect so
    /// cells read roughly square (monospace cells are ~2:1). Powers the browser's terminal view.
    pub fn ascii_view(&mut self, rows: u32) -> String {
        let rows = rows.max(1) as usize;
        let ([cx, cy], mpp, [vw, vh]) = (self.camera.center, self.camera.meters_per_pixel, self.camera.viewport);
        let (hx, hy) = (vw * mpp * 0.5, vh * mpp * 0.5);
        let cols = (2.0 * rows as f64 * (vw / vh).max(1e-6)).round().max(1.0) as usize;
        let mut a = crate::render::ascii::Ascii::new([cx - hx, cy - hy], [cx + hx, cy + hy], cols, rows);
        if self.ascii_fill.is_none() {
            self.ascii_fill = Some(geometry::world_fill_colored(&self.world.network, ascii_road_color, ASCII_JUNCTION_COLOR));
        }
        a.fill_mesh_colored(self.ascii_fill.as_ref().unwrap(), '#'); // roads, tinted by class
        for v in self.world.vehicles() {
            let p = self.world.vehicle_world_pose(v);
            // Buses get their own glyph so transit reads at a glance; the
            // colour still carries speed for everyone.
            let glyph = if VehicleClass::from_length(v.driver.vehicle_length) == VehicleClass::Bus { 'B' } else { '@' };
            a.plot_colored([p[0], p[1]], glyph, speed_color(v.speed));
        }
        for i in self.train_instance_vec() {
            a.plot_colored([i.pos[0] as f64, i.pos[1] as f64], 'T', i.color);
        }
        // Signal heads last, so a lit lens is never hidden behind a queued car.
        for (pos, _heading, state, _is_left) in self.signal_head_slots() {
            a.plot_colored([pos[0] as f64, pos[1] as f64], 'O', signal_color(state));
        }
        a.render_html()
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
    /// Pump the async GPU routing recompute once per frame. A whole-city reroute is a
    /// chunked state machine spread over several frames so no single frame encodes the
    /// worst-case pass count: `idle → start + run_chunk → poll_flag → {converged | capped
    /// → finish → collect → feed} else run_chunk`. Non-blocking — the browser can't wait
    /// on the GPU, so a solve spans frames.
    fn drive_gpu_routing(&mut self) {
        if self.gpu.is_none() {
            return;
        }
        // Phase A — collect the final distance readback. Always unmap (via `try_take`);
        // only feed the result if it's still for the current router generation — a demand
        // toggle since dispatch invalidates it.
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
        // Phase B — advance an in-flight convergence: poll the last chunk's flags, then
        // either finish (converged, or hit the safety cap) or encode the next chunk.
        if let Some((pending, generation)) = self.gpu_relax.take() {
            match self.gpu.as_ref().unwrap().poll_flag(&pending) {
                None => self.gpu_relax = Some((pending, generation)), // chunk still mapping
                Some(_) if generation != self.gpu_generation => self.gpu_last = f64::NEG_INFINITY, // stale solve: unmapped, dispatch fresh
                Some(done) => {
                    let gpu = self.gpu.as_mut().unwrap();
                    if done || gpu.passes_capped() {
                        let readback = gpu.finish();
                        self.gpu_pending = Some((readback, generation));
                    } else {
                        let flag = gpu.run_chunk();
                        self.gpu_relax = Some((flag, generation));
                    }
                }
            }
            return; // one GPU action per frame
        }
        // Phase C — idle: begin a fresh solve, but only when it will change anything. The 3 s
        // timer bounds how often we check; the congestion fingerprint then gates the actual
        // solve — under light or static traffic the installed free-flow fields stay optimal,
        // so a whole-map GPU solve every 3 s (which competes with rendering for the GPU) is
        // skipped and routing tracks congestion, not map size.
        if self.world.time() - self.gpu_last < GPU_REROUTE_SECS {
            return;
        }
        let fp = self.world.congestion_fingerprint();
        if fp == self.gpu_fingerprint {
            self.gpu_last = self.world.time(); // nothing moved — re-check after another interval
            return;
        }
        self.gpu_fingerprint = fp;
        let dests: Vec<u32> = self.world.router_dest_links().iter().map(|d| d.0).collect();
        if dests.is_empty() {
            return;
        }
        self.gpu_cost = self.world.live_link_costs();
        let cost32: Vec<u32> = self.gpu_cost.iter().map(|&c| c.min(u32::MAX as u64) as u32).collect();
        let gpu = self.gpu.as_mut().unwrap();
        gpu.start(&cost32, &dests);
        let flag = gpu.run_chunk();
        self.gpu_relax = Some((flag, self.gpu_generation));
    }
}

/// Reinterpret a `Vec<StaticVertex>` as its flat `Vec<f32>` in place — no copy. Marshaling
/// the world/marking meshes for the GPU is the peak wasm-memory moment when loading a city
/// map (the marking mesh alone is ~220 MB), and the old `cast_slice(..).to_vec()` doubled
/// that transiently — and since wasm memory never shrinks, the spike became permanent
/// high-water. `StaticVertex` is `#[repr(C)]` of exactly 8 `f32` with no padding, so its
/// buffer *is* a valid `[f32; 8·len]` with `8·cap` capacity and identical alignment.
fn flatten_static_vertices(mut v: Vec<StaticVertex>) -> Vec<f32> {
    const _: () = assert!(std::mem::size_of::<StaticVertex>() == 8 * std::mem::size_of::<f32>());
    let (ptr, len, cap) = (v.as_mut_ptr() as *mut f32, v.len() * 8, v.capacity() * 8);
    std::mem::forget(v);
    // SAFETY: layout-compatible per the doc comment; the original allocation size
    // (cap · size_of::<StaticVertex>()) equals 8·cap · size_of::<f32>() with the same 4-byte
    // alignment, so the reconstructed Vec owns exactly that buffer and frees it correctly.
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

fn build_demand(
    world: &NetWorld,
    seed: u64,
    sources: DemandSources,
    rate: f64,
    entry_cap: f64,
    commute: Option<&demand::CommuteOd>,
) -> DemandGenerator {
    let pairs = demand::od_pairs_with_commute(&world.network, seed, 48, sources, commute);
    let mut gen = DemandGenerator::new(world, &pairs, seed);
    gen.set_rate_scale(rate);
    gen.set_entry_speed_cap(entry_cap);
    gen.set_rush_hour(&world.network, sources.rush_hour);
    gen
}

/// Road-class palette for the ASCII view: the freeway system warm, surface streets cooler
/// and dimmer down the hierarchy, so the network structure reads at a glance.
fn ascii_road_color(kind: RoadKind) -> [f32; 3] {
    match kind {
        RoadKind::Freeway => [0.98, 0.74, 0.26],
        RoadKind::Ramp => [0.82, 0.56, 0.30],
        RoadKind::Arterial => [0.36, 0.76, 0.86],
        RoadKind::Collector => [0.44, 0.72, 0.48],
        RoadKind::Local => [0.42, 0.48, 0.60],
    }
}

/// Intersection-box fill colour in the ASCII view — a neutral light grey the coloured
/// approaches plug into.
const ASCII_JUNCTION_COLOR: [f32; 3] = [0.58, 0.60, 0.66];

/// Vehicle colour in the ASCII view: a red→yellow→green ramp by speed, so congestion
/// (slow/stopped) reads red against free-flowing green. ~13 m/s (≈29 mph) is full green.
fn speed_color(mps: f64) -> [f32; 3] {
    let t = (mps / 13.0).clamp(0.0, 1.0) as f32;
    [((1.0 - t) * 2.0).min(1.0), (t * 2.0).min(1.0), 0.18]
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
