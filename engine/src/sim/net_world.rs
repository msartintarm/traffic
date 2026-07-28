//! Vehicles driving a graph [`Network`]: IDM car-following within a lane, a red
//! signal (or unsatisfied movement) as a stationary virtual leader at the stop
//! line, and lane hand-off across a node when the movement is served.
//! Accelerations read committed pre-step state and apply in a second pass.

use std::collections::HashMap;
use std::hash::BuildHasherDefault;

use super::config::{DriverConfig, SimConfig};
use super::constraint::{self, LongContext, Obstacle, SpeedTarget};
use super::idm;
use super::mobil::{self, MobilParams};
use super::junction::{self, Junctions, SignalController};
use super::network::{LaneId, LinkId, MovementId, Network, NodeControl, NodeId, TurnType};
use super::router::FieldRouter;
use super::signal::SignalState;

#[derive(Clone, Debug, PartialEq)]
pub struct NetVehicle {
    pub id: u32,
    pub lane: LaneId,
    pub position: f64,
    pub speed: f64,
    pub driver: DriverConfig,
    pub route: Vec<LinkId>,
    pub route_idx: usize,
    /// Destination link for flow-field routing; when set (with a world router)
    /// it supersedes `route` and reroutes live around congestion.
    pub dest: Option<LinkId>,
    /// The stop-controlled node this vehicle has already halted at, so a stop
    /// sign is enforced once rather than forever.
    stopped_at: Option<NodeId>,
    /// Consecutive ticks spent essentially stopped — drives yield impatience.
    wait_ticks: u32,
    /// When set, the vehicle is inside a node traversing a movement's interior
    /// path (`lane`/`position` pinned at the stop line); `s` is its arc-length
    /// progress. Cleared when it lands on the destination lane.
    crossing: Option<Crossing>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Crossing {
    movement: MovementId,
    s: f64,
}

/// Outcome of advancing a vehicle one tick.
enum Fate {
    Alive,
    /// Crossed onto a new link (its id, for entry counting).
    Entered(LinkId),
    /// Left the network legitimately — reached its destination, finished its
    /// route, or ran off a genuine dead end.
    Exited,
    /// Removed at an interior node despite still having a routable next hop — a
    /// vehicle that *disappeared* at an intersection. Should never happen; counted
    /// so a regression in movement resolution is caught rather than silent.
    Leaked,
}

/// How many `(position, speed)` samples to retain — enough for the largest
/// plausible reaction delay at the fixed timestep.
const HISTORY_LEN: usize = 8;

type History = [(f64, f64); HISTORY_LEN];

/// Vehicle storage as columns: the rows plus the per-vehicle reaction-delay
/// history kept out of the row (it is only read for a leader, not while iterating
/// every row). Columns stay index-aligned; mutations go through here.
#[derive(Clone, Debug, Default)]
struct Fleet {
    rows: Vec<NetVehicle>,
    hist: Vec<History>,
    hist_len: Vec<u8>,
}

impl Fleet {
    fn len(&self) -> usize {
        self.rows.len()
    }

    fn push(&mut self, v: NetVehicle) {
        let mut h = [(0.0, 0.0); HISTORY_LEN];
        h[0] = (v.position, v.speed);
        self.hist.push(h);
        self.hist_len.push(1);
        self.rows.push(v);
    }

    fn clear(&mut self) {
        self.rows.clear();
        self.hist.clear();
        self.hist_len.clear();
    }

    /// Row `i`'s `(position, speed)` `ticks` steps ago (clamped to the oldest kept).
    fn delayed(&self, i: usize, ticks: usize) -> (f64, f64) {
        let n = self.hist_len[i] as usize;
        self.hist[i][n - 1 - ticks.min(n - 1)]
    }

    /// Whether row `i` has more than `ticks` samples on its *current* lane, so a
    /// `ticks`-delayed lookup is a real in-frame position rather than one clamped back
    /// to (or across) a recent segment crossing. History is reset on crossing, so this
    /// gates the reaction-delay model back on only once the car has settled.
    fn settled(&self, i: usize, ticks: usize) -> bool {
        self.hist_len[i] as usize > ticks
    }
}

/// Append the current `(position, speed)` to a history column entry, dropping the
/// oldest sample once full.
fn record_history(hist: &mut History, len: &mut u8, position: f64, speed: f64) {
    let n = *len as usize;
    if n < HISTORY_LEN {
        hist[n] = (position, speed);
        *len += 1;
    } else {
        hist.copy_within(1.., 0);
        hist[HISTORY_LEN - 1] = (position, speed);
    }
}

impl NetVehicle {
    /// Whether the vehicle is currently inside a node traversing a movement's
    /// interior path (its `lane`/`position` are pinned at the stop line).
    pub fn is_crossing(&self) -> bool {
        self.crossing.is_some()
    }
}

pub struct NetWorld {
    pub network: Network,
    cfg: SimConfig,
    fleet: Fleet,
    time: f64,
    tick: u64,
    exited: u32,
    leaked: u32,
    crashed: u32,
    /// Downstream lanes fed by more than one lane — the merge points; value is
    /// the list of feeding (from) lane ids.
    merges: HashMap<u32, Vec<u32>>,
    /// Actuated signal timing (see [`SignalController`]).
    signals: SignalController,
    /// Cumulative vehicles that have entered each link (spawned onto it or
    /// crossed onto it) — the raw counts calibration compares to real data.
    link_entries: Vec<u32>,
    /// Flow-field router for destination-based vehicles, rebuilt periodically
    /// against live costs so in-flight cars reroute around congestion.
    router: Option<FieldRouter>,
    /// When set, an external driver (the browser GPU flow-field) owns the routing
    /// recompute and feeds fresh fields in; the internal CPU recompute stands down.
    external_reroute: bool,
    /// Per-node index of movements and conflict points.
    junctions: Junctions,
    /// Executor requested for the per-vehicle accel passes (see [`AccelBackend`]).
    accel_backend: AccelBackend,
    /// Whether the CPU worker pool is up. Native: rayon's global pool auto-inits, so
    /// always true. Browser: false until JS finishes `initThreadPool` (SharedArrayBuffer
    /// needs cross-origin isolation), gating the `Threads` backend until then.
    threads_ready: bool,
    /// Vehicle count at/above which the `Threads` backend actually parallelizes (below
    /// it, serial — rayon overhead isn't worth it). Runtime-tunable for measurement.
    par_threshold: usize,
}

/// Sim seconds between flow-field rebuilds — often enough that routing tracks
/// congestion as it forms, rare enough that the recompute cost is negligible.
const REROUTE_INTERVAL_SECS: f64 = 3.0;

pub const STEP_PHASES: usize = 7;
pub const PHASE_NAMES: [&str; STEP_PHASES] =
    ["refresh_routes", "advance_signals", "lane_changes", "neighbors", "accel", "advance", "crashes"];
thread_local! {
    static PROF: std::cell::Cell<[f64; STEP_PHASES]> = const { std::cell::Cell::new([0.0; STEP_PHASES]) };
}
/// Read and reset the per-phase step timings (ms) accumulated since the last call.
/// Always zero on wasm (the profiler is a no-op there — no clock).
pub fn prof_take() -> [f64; STEP_PHASES] {
    PROF.with(|c| c.replace([0.0; STEP_PHASES]))
}

/// Per-phase step timer. Real on native (for the load tests); a zero-cost no-op on
/// wasm, where `Instant::now()` is unavailable.
#[cfg(not(target_arch = "wasm32"))]
struct Prof(std::time::Instant);
#[cfg(not(target_arch = "wasm32"))]
impl Prof {
    #[inline]
    fn new() -> Self {
        Self(std::time::Instant::now())
    }
    #[inline]
    fn lap(&mut self, phase: usize) {
        let ms = self.0.elapsed().as_secs_f64() * 1000.0;
        PROF.with(|c| {
            let mut a = c.get();
            a[phase] += ms;
            c.set(a);
        });
        self.0 = std::time::Instant::now();
    }
}
#[cfg(target_arch = "wasm32")]
struct Prof;
#[cfg(target_arch = "wasm32")]
impl Prof {
    #[inline]
    fn new() -> Self {
        Self
    }
    #[inline]
    fn lap(&mut self, _phase: usize) {}
}

/// Which executor runs the per-vehicle accel passes. Selected at runtime so a
/// device without a good GPU (or without cross-origin isolation for CPU threads)
/// can fall back; [`NetWorld::active_backend`] resolves the request against what's
/// actually available. Each backend is validated to match [`AccelBackend::Serial`]
/// bit-for-bit — the passes read only committed pre-step state, so order-preserving
/// parallelism is exact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccelBackend {
    /// Single-threaded — always available, and the correctness reference.
    Serial,
    /// Data-parallel across CPU cores (rayon). Native under `--features parallel`;
    /// the browser additionally needs a SharedArrayBuffer worker pool (a follow-up).
    Threads,
    /// GPU compute kernel for the evaluate pass (a follow-up); the gather stays on
    /// CPU. Until wired, [`NetWorld::active_backend`] falls it back to `Serial`.
    Gpu,
}

impl AccelBackend {
    pub fn from_name(s: &str) -> Self {
        match s {
            "threads" => Self::Threads,
            "gpu" => Self::Gpu,
            _ => Self::Serial,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Serial => "serial",
            Self::Threads => "threads",
            Self::Gpu => "gpu",
        }
    }
}

/// Whether the CPU worker pool is actually linked and usable. Native: the `parallel`
/// feature. Browser: this will additionally require the initialised SharedArrayBuffer
/// pool once that path lands.
fn threads_available() -> bool {
    cfg!(feature = "parallel")
}

/// Default for [`NetWorld::par_threshold`]: below this vehicle count rayon's per-task
/// overhead outweighs the win (measured crossover on the Millbrae step is ~4–5k, but the
/// default is set lower so threads engage sooner under climbing load), so [`map_collect`]
/// stays serial even on the `Threads` backend below it. Adjustable at runtime.
pub const DEFAULT_PAR_THRESHOLD: usize = 2000;

/// Map `0..n` through `f` on the given `backend`. On `Threads`, runs serially when
/// `n < threshold` (rayon overhead isn't worth it below the crossover). Order-preserving
/// on every backend, so the collected result is bit-for-bit the serial one — the
/// per-vehicle passes it drives read only committed pre-step state.
fn map_collect<T, F>(backend: AccelBackend, threshold: usize, n: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync + Send,
{
    let _ = threshold; // used by the parallel arm's guard below; referenced here so the
                       // serial-only (no `parallel` feature) build doesn't flag it unused.
    match backend {
        #[cfg(feature = "parallel")]
        AccelBackend::Threads if n >= threshold => {
            use rayon::prelude::*;
            (0..n).into_par_iter().map(f).collect()
        }
        _ => (0..n).map(f).collect(),
    }
}

/// A rolling vehicle's flattened accel-decision inputs — the output of the gather
/// pass ([`NetWorld::gather_context`]) and the sole input to the pure evaluate
/// kernel ([`VehicleContext::evaluate`]). `#[repr(C)]`/`Pod` so a slice of these
/// uploads to a GPU storage buffer verbatim; an absent optional constraint is
/// encoded as `+∞` (the constraints' own non-binding value), keeping the struct a
/// flat, branch-free bag of scalars that a WGSL kernel can consume directly.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VehicleContext {
    /// Driver params, capped to the current lane's speed limit.
    driver: DriverConfig,
    speed: f64,
    stop_line: f64,
    speed_target_speed: f64,
    speed_target_dist: f64,
    stop_sign: f64,
    yield_line: f64,
    curve_speed: f64,
    curve_dist: f64,
    leader_gap: f64,
    leader_speed: f64,
    merge_gap: f64,
    merge_speed: f64,
    agent_id: u32,
    _pad: u32,
}

impl VehicleContext {
    fn new(driver: DriverConfig, speed: f64, agent_id: u32) -> Self {
        Self {
            driver,
            speed,
            stop_line: f64::INFINITY,
            speed_target_speed: f64::INFINITY,
            speed_target_dist: f64::INFINITY,
            stop_sign: f64::INFINITY,
            yield_line: f64::INFINITY,
            curve_speed: f64::INFINITY,
            curve_dist: f64::INFINITY,
            leader_gap: f64::INFINITY,
            leader_speed: 0.0,
            merge_gap: f64::INFINITY,
            merge_speed: 0.0,
            agent_id,
            _pad: 0,
        }
    }

    fn set_leader(&mut self, o: Option<Obstacle>) {
        if let Some(o) = o {
            self.leader_gap = o.gap;
            self.leader_speed = o.speed;
        }
    }
    fn set_merge(&mut self, o: Option<Obstacle>) {
        if let Some(o) = o {
            self.merge_gap = o.gap;
            self.merge_speed = o.speed;
        }
    }
    fn set_curve(&mut self, t: Option<SpeedTarget>) {
        if let Some(t) = t {
            self.curve_speed = t.speed;
            self.curve_dist = t.distance;
        }
    }
    fn set_speed_target(&mut self, t: Option<SpeedTarget>) {
        if let Some(t) = t {
            self.speed_target_speed = t.speed;
            self.speed_target_dist = t.distance;
        }
    }
    fn set_stop_line(&mut self, d: Option<f64>) {
        if let Some(d) = d {
            self.stop_line = d;
        }
    }
    fn set_stop_sign(&mut self, d: Option<f64>) {
        if let Some(d) = d {
            self.stop_sign = d;
        }
    }
    fn set_yield_line(&mut self, d: Option<f64>) {
        if let Some(d) = d {
            self.yield_line = d;
        }
    }

    /// The deterministic binding acceleration — the constraint fold, no noise.
    /// Rebuilds the constraint context from the flat fields (`+∞` → `None`). Reads
    /// only `self` (no graph access), so this is exactly what the WGSL kernel
    /// (`accel.wgsl`) mirrors; `accel_noise` is added separately (its RNG is `u64`,
    /// which WGSL lacks, so it stays on the CPU).
    fn binding(&self) -> f64 {
        let opt = |x: f64| x.is_finite().then_some(x);
        let ctx = LongContext {
            driver: &self.driver,
            speed: self.speed,
            leader: opt(self.leader_gap).map(|gap| Obstacle { gap, speed: self.leader_speed }),
            stop_line: opt(self.stop_line),
            speed_target: opt(self.speed_target_dist)
                .map(|distance| SpeedTarget { speed: self.speed_target_speed, distance }),
            stop_sign: opt(self.stop_sign),
            yield_line: opt(self.yield_line),
            merge: opt(self.merge_gap).map(|gap| Obstacle { gap, speed: self.merge_speed }),
            curve: opt(self.curve_dist).map(|distance| SpeedTarget { speed: self.curve_speed, distance }),
        };
        constraint::binding_acceleration(&ctx, constraint::DEFAULT)
    }

    /// The full per-vehicle acceleration: the binding fold plus the reproducible
    /// per-tick noise.
    fn evaluate(&self, seed: u64, tick: u64) -> f64 {
        self.binding() + constraint::accel_noise(self.driver.accel_noise, seed, self.agent_id, tick)
    }

    /// Pack into the f32 layout the GPU kernel reads (`accel.wgsl`'s `Ctx`). An
    /// absent optional (`+∞`) becomes the `BIG` sentinel the shader tests against;
    /// every field is dropped to `f32` (WGSL has no `f64`). Exercised by the GPU
    /// equivalence test; the step wires it into the `Gpu` backend later.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    #[cfg_attr(not(test), allow(dead_code))]
    fn to_gpu(&self) -> VehicleContextGpu {
        let f = |x: f64| if x.is_finite() { x as f32 } else { GPU_BIG };
        VehicleContextGpu {
            desired_speed: self.driver.desired_speed as f32,
            accel_exponent: self.driver.accel_exponent as f32,
            min_gap: self.driver.min_gap as f32,
            time_headway: self.driver.time_headway as f32,
            max_accel: self.driver.max_accel as f32,
            comfort_decel: self.driver.comfort_decel as f32,
            speed: self.speed as f32,
            leader_gap: f(self.leader_gap),
            leader_speed: self.leader_speed as f32,
            stop_line: f(self.stop_line),
            speed_target_speed: f(self.speed_target_speed),
            speed_target_dist: f(self.speed_target_dist),
            stop_sign: f(self.stop_sign),
            yield_line: f(self.yield_line),
            curve_speed: f(self.curve_speed),
            curve_dist: f(self.curve_dist),
            merge_gap: f(self.merge_gap),
            merge_speed: self.merge_speed as f32,
        }
    }
}

/// Sentinel for an absent optional constraint in the GPU layout (mirrors `+∞` in
/// [`VehicleContext`]); `accel.wgsl` treats any field `>=` this as not binding.
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
#[cfg_attr(not(test), allow(dead_code))]
const GPU_BIG: f32 = 1e30;

/// The f32, `#[repr(C)]`/`Pod` layout a slice of which uploads to the GPU accel
/// kernel's storage buffer verbatim (field order matches `accel.wgsl`'s `Ctx`).
/// Only the scalars the binding fold reads — no `agent_id` (noise is CPU-side).
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VehicleContextGpu {
    pub desired_speed: f32,
    pub accel_exponent: f32,
    pub min_gap: f32,
    pub time_headway: f32,
    pub max_accel: f32,
    pub comfort_decel: f32,
    pub speed: f32,
    pub leader_gap: f32,
    pub leader_speed: f32,
    pub stop_line: f32,
    pub speed_target_speed: f32,
    pub speed_target_dist: f32,
    pub stop_sign: f32,
    pub yield_line: f32,
    pub curve_speed: f32,
    pub curve_dist: f32,
    pub merge_gap: f32,
    pub merge_speed: f32,
}

/// Per-vehicle input to the accel evaluate pass. A rolling vehicle carries its full
/// [`VehicleContext`]; an in-node crosser carries the acceleration its bespoke
/// [`NetWorld::crossing_accel`] already produced (a separate kernel, and noiseless).
enum AccelInput {
    Rolling(VehicleContext),
    Crossing(f64),
}

impl AccelInput {
    fn evaluate(&self, seed: u64, tick: u64) -> f64 {
        match self {
            AccelInput::Rolling(cx) => cx.evaluate(seed, tick),
            AccelInput::Crossing(a) => *a,
        }
    }
}

/// FxHash-style hasher for the dense integer ids that key the per-tick maps —
/// far cheaper than the default SipHash.
#[derive(Default)]
pub struct FxHasher(u64);
const FX_K: u64 = 0x51_7c_c1_b7_27_22_0a_95;
impl std::hash::Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(FX_K);
        }
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(FX_K);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}
/// Map/set keyed by a dense integer id, hashed with [`FxHasher`].
type IntMap<V> = HashMap<u32, V, BuildHasherDefault<FxHasher>>;


/// How far ahead a driver reads signals to ease off early for a red at the next
/// intersection (metres) — anticipatory braking across the current link.
const SIGNAL_LOOKAHEAD: f64 = 90.0;

impl NetWorld {
    pub fn new(network: Network, cfg: SimConfig) -> Self {
        let mut merges: HashMap<u32, Vec<u32>> = HashMap::new();
        for m in &network.movements {
            let entry = merges.entry(m.to_lane.0).or_default();
            if !entry.contains(&m.from_lane.0) {
                entry.push(m.from_lane.0);
            }
        }
        merges.retain(|_, froms| froms.len() > 1);

        let signals = SignalController::build(&network);
        let link_entries = vec![0u32; network.links.len()];
        let junctions = Junctions::build(&network);
        Self {
            network, cfg, fleet: Fleet::default(), time: 0.0, tick: 0, exited: 0, leaked: 0, crashed: 0,
            merges, signals, link_entries, router: None, external_reroute: false, junctions,
            accel_backend: AccelBackend::Serial,
            threads_ready: cfg!(not(target_arch = "wasm32")),
            par_threshold: DEFAULT_PAR_THRESHOLD,
        }
    }

    /// Mark the CPU worker pool ready (the browser calls this once `initThreadPool`
    /// resolves). Until then the `Threads` backend falls back to serial.
    pub fn set_threads_ready(&mut self, ready: bool) {
        self.threads_ready = ready;
    }

    /// Vehicle count at/above which the `Threads` backend parallelizes (below it,
    /// serial). Tune to find the crossover on a given device.
    pub fn set_par_threshold(&mut self, n: usize) {
        self.par_threshold = n;
    }

    pub fn par_threshold(&self) -> usize {
        self.par_threshold
    }

    /// Request an executor for the per-vehicle accel passes. Resolved against
    /// availability each step by [`active_backend`](Self::active_backend), so asking
    /// for `Threads`/`Gpu` where they aren't available cleanly falls back to serial.
    pub fn set_accel_backend(&mut self, backend: AccelBackend) {
        self.accel_backend = backend;
    }

    /// The backend actually used this step: the request, downgraded to `Serial` when
    /// it isn't available (no worker pool, or the GPU evaluate kernel isn't wired yet).
    pub fn active_backend(&self) -> AccelBackend {
        match self.accel_backend {
            AccelBackend::Threads if threads_available() && self.threads_ready => AccelBackend::Threads,
            // The GPU evaluate kernel is a follow-up; until then it runs serially
            // (the gather is CPU-side regardless).
            _ => AccelBackend::Serial,
        }
    }

    /// Install a flow-field router covering `dests`; vehicles spawned via
    /// [`NetWorld::spawn_to`] then route by the field and reroute live.
    pub fn install_router(&mut self, dests: &[LinkId]) {
        let costs = self.live_link_costs();
        self.router = Some(FieldRouter::new(&self.network, dests, &costs));
    }

    pub fn router_knows(&self, dest: LinkId) -> bool {
        self.router.as_ref().is_some_and(|r| r.knows(dest))
    }

    /// Hand the routing recompute to an external driver (the browser GPU
    /// flow-field). The internal CPU recompute then stands down; the driver calls
    /// [`feed_router_distances`](Self::feed_router_distances) with fresh fields.
    pub fn set_external_reroute(&mut self, external: bool) {
        self.external_reroute = external;
    }

    /// The router's destinations in slot order — the order an external solver must
    /// return `feed_router_distances`' `dist_per_slot` in.
    pub fn router_dest_links(&self) -> Vec<LinkId> {
        self.router.as_ref().map(|r| r.dests_in_slot_order().to_vec()).unwrap_or_default()
    }

    /// Feed externally-computed reverse distances (GPU) into the router's next-hop
    /// fields — the counterpart to [`set_external_reroute`](Self::set_external_reroute).
    pub fn feed_router_distances(&mut self, cost: &[u64], dist_per_slot: &[Vec<u64>]) {
        if let Some(r) = self.router.as_mut() {
            r.recompute_from_distances(cost, dist_per_slot);
        }
    }

    /// Remove all vehicles from the road, leaving the network and counters intact —
    /// used to reset traffic when the demand mode changes without rebuilding the map.
    pub fn clear_vehicles(&mut self) {
        self.fleet.clear();
    }

    /// Measured flow (vehicles/hour) on each link, from entries so far over
    /// elapsed sim time — the sim's own counts to calibrate against real data.
    pub fn link_flows(&self) -> Vec<f64> {
        let hours = (self.time / 3600.0).max(1e-9);
        self.link_entries.iter().map(|&e| e as f64 / hours).collect()
    }

    /// Live per-link stats: `(vehicle_count, mean_speed_mps, occupancy_ratio)`.
    pub fn link_stats(&self, link: LinkId) -> (u32, f64, f64) {
        let l = self.network.link(link);
        let lane = self.network.lane(l.lane_start);
        let (mut count, mut sum) = (0u32, 0.0);
        for v in &self.fleet.rows {
            if self.network.lane(v.lane).link == link {
                count += 1;
                sum += v.speed;
            }
        }
        let mean = if count > 0 { sum / count as f64 } else { 0.0 };
        let jam = (lane.length / 7.0 * l.lane_count as f64).max(1.0);
        (count, mean, (count as f64 / jam).min(1.0))
    }

    pub fn spawn(&mut self, id: u32, lane: LaneId, position: f64, speed: f64, driver: DriverConfig) {
        self.link_entries[self.network.lane(lane).link.idx()] += 1;
        self.fleet.push(NetVehicle {
            id, lane, position, speed, driver, route: Vec::new(), route_idx: 0, dest: None,
            stopped_at: None, wait_ticks: 0, crossing: None,
        });
    }

    /// Spawn at the start of a precomputed link route; the vehicle takes the
    /// route-consistent movement at each intersection and exits on the last link.
    /// Returns `false` (spawn refused) if the route is empty or the entrance is
    /// still occupied, so demand can't stack vehicles on top of each other.
    pub fn spawn_routed(&mut self, id: u32, route: Vec<LinkId>, speed: f64, driver: DriverConfig) -> bool {
        let Some(&first) = route.first() else { return false };
        let Some(lane) = self.network.lanes_of(first).next() else { return false };
        if !self.entrance_clear(lane, driver.min_gap) {
            return false;
        }
        self.link_entries[self.network.lane(lane).link.idx()] += 1;
        self.fleet.push(NetVehicle {
            id, lane, position: 0.0, speed, driver, route, route_idx: 0, dest: None,
            stopped_at: None, wait_ticks: 0, crossing: None,
        });
        true
    }

    /// Spawn at the start of `entry_link` bound for `dest`, routed live by the
    /// world's flow-field. Refused if the entrance is still occupied.
    pub fn spawn_to(&mut self, id: u32, entry_link: LinkId, dest: LinkId, speed: f64, driver: DriverConfig) -> bool {
        let Some(lane) = self.network.lanes_of(entry_link).next() else { return false };
        if !self.entrance_clear(lane, driver.min_gap) {
            return false;
        }
        self.link_entries[entry_link.idx()] += 1;
        self.fleet.push(NetVehicle {
            id, lane, position: 0.0, speed, driver, route: Vec::new(), route_idx: 0, dest: Some(dest),
            stopped_at: None, wait_ticks: 0, crossing: None,
        });
        true
    }

    /// Test-only: spawn a destination-routed vehicle in a specific lane at a
    /// specific position, so a scenario can place a car where it cannot reach a
    /// lane serving its route.
    #[cfg(test)]
    pub fn spawn_to_in_lane(&mut self, id: u32, lane: LaneId, position: f64, dest: LinkId, speed: f64, driver: DriverConfig) {
        self.link_entries[self.network.lane(lane).link.idx()] += 1;
        self.fleet.push(NetVehicle {
            id, lane, position, speed, driver, route: Vec::new(), route_idx: 0, dest: Some(dest),
            stopped_at: None, wait_ticks: 0, crossing: None,
        });
    }

    /// Whether a vehicle can be placed at the start of `lane` without overlapping
    /// one already there — measured against each occupant's *rear* (position minus
    /// its own length), so long vehicles are accounted for.
    pub fn entrance_clear(&self, lane: LaneId, min_gap: f64) -> bool {
        self.fleet.rows
            .iter()
            .filter(|v| v.lane == lane)
            .all(|v| v.position - v.driver.vehicle_length > min_gap)
    }

    pub fn time(&self) -> f64 {
        self.time
    }

    pub fn exited(&self) -> u32 {
        self.exited
    }

    /// Vehicles removed from the road after a collision.
    pub fn crashed(&self) -> u32 {
        self.crashed
    }

    /// Vehicles that disappeared at an interior intersection despite having a
    /// routable next hop. A correct engine never leaks — this stays zero.
    pub fn leaked(&self) -> u32 {
        self.leaked
    }

    /// Per-link travel time (ms) inflated by current occupancy — the live edge
    /// weights that make routing congestion-reactive. A jammed link costs several
    /// times its free-flow time, so routes computed with these steer around it.
    pub fn live_link_costs(&self) -> Vec<u64> {
        let mut count = vec![0u32; self.network.links.len()];
        for v in &self.fleet.rows {
            count[self.network.lane(v.lane).link.idx()] += 1;
        }
        (0..self.network.links.len() as u32)
            .map(|i| {
                let link = self.network.link(LinkId(i));
                let lane = self.network.lane(link.lane_start);
                let jam = (lane.length / 7.0 * link.lane_count as f64).max(1.0);
                let ratio = (count[i as usize] as f64 / jam).min(3.0);
                let base = self.network.link_travel_time_ms(LinkId(i)) as f64;
                (base * (1.0 + 2.0 * ratio)) as u64
            })
            .collect()
    }

    pub fn vehicles(&self) -> &[NetVehicle] {
        &self.fleet.rows
    }

    /// A vehicle's current world pose `[x, y, heading]` — its interior crossing
    /// path when inside a node, otherwise its lane position.
    pub fn vehicle_world_pose(&self, v: &NetVehicle) -> [f64; 3] {
        match v.crossing {
            Some(c) => self.network.interior_point(c.movement, c.s),
            None => self.network.lane_point(v.lane, v.position),
        }
    }

    pub fn vehicle(&self, id: u32) -> Option<&NetVehicle> {
        self.fleet.rows.iter().find(|v| v.id == id)
    }

    fn intended_movement(&self, veh: &NetVehicle) -> Option<MovementId> {
        let lane = self.network.lane(veh.lane);
        // The routed movement — onto the flow-field's (or explicit route's) next
        // link. `None` here means the car couldn't reach a lane that serves that
        // link, *not* that it should leave: a car that has arrived (no next hop) or
        // finished its route returns early below, before this fallback.
        let preferred = if let (Some(dest), Some(router)) = (veh.dest, self.router.as_ref()) {
            match router.next_hop(dest, lane.link) {
                None => return None, // reached the destination → leave the network
                Some(next_link) => self.movement_to(veh.lane, next_link),
            }
        } else if !veh.route.is_empty() {
            if veh.route_idx + 1 >= veh.route.len() {
                return None; // explicit route completed → leave
            }
            self.movement_to(veh.lane, veh.route[veh.route_idx + 1])
        } else {
            None
        };
        // Take the routed movement when this lane serves it; otherwise proceed on
        // whatever this lane offers (reroute from where it lands) — a car in the
        // wrong lane must never vanish mid-network. If even this lane serves
        // nothing, borrow a sibling lane's movement rather than dropping the car;
        // only a genuine dead end (no lane on the link moves on) truly exits.
        preferred
            .or_else(|| (lane.movement_count > 0).then_some(lane.movement_start))
            .or_else(|| self.any_movement_on(lane.link))
    }

    /// The first available movement from any lane of `link` — a last-resort so a
    /// vehicle at an interior node whose own lane serves nothing still proceeds
    /// instead of disappearing. `None` only for a true dead end.
    fn any_movement_on(&self, link: LinkId) -> Option<MovementId> {
        self.network
            .lanes_of(link)
            .map(|l| self.network.lane(l))
            .find(|l| l.movement_count > 0)
            .map(|l| l.movement_start)
    }

    /// The movement this vehicle would take if it were on `lane` — its flow-field
    /// next hop from that lane. Used to look ahead past the current link (only for
    /// destination-routed vehicles; explicit-route lookahead isn't supported).
    fn intended_movement_from(&self, veh: &NetVehicle, lane: LaneId) -> Option<MovementId> {
        if self.network.lane(lane).movement_count == 0 {
            return None;
        }
        let (dest, router) = (veh.dest?, self.router.as_ref()?);
        let next = router.next_hop(dest, self.network.lane(lane).link)?;
        self.movement_to(lane, next)
    }

    /// Distance to the nearest **red** signal ahead along the vehicle's path,
    /// looking past the immediate movement across up to a couple of links (within
    /// [`SIGNAL_LOOKAHEAD`]). Lets a car ease off for a red one intersection away
    /// instead of arriving at speed — anticipatory braking. `None` if the immediate
    /// movement is itself red (handled directly) or nothing red is close.
    fn red_ahead(&self, veh: &NetVehicle, immediate: Option<MovementId>, to_line: f64) -> Option<f64> {
        let mv0 = immediate?;
        if self.movement_state(mv0) == SignalState::Red {
            return None; // the immediate stop line already handles this
        }
        let mut dist = to_line + self.network.interior(mv0).len;
        let mut lane = self.network.movement(mv0).to_lane;
        for _ in 0..2 {
            dist += self.network.lane(lane).length;
            if dist > SIGNAL_LOOKAHEAD {
                return None;
            }
            let mv = self.intended_movement_from(veh, lane)?;
            if self.movement_state(mv) == SignalState::Red {
                return Some(dist);
            }
            dist += self.network.interior(mv).len;
            lane = self.network.movement(mv).to_lane;
        }
        None
    }

    /// The movement from `from_lane` onto `next_link`, if one exists.
    fn movement_to(&self, from_lane: LaneId, next_link: LinkId) -> Option<MovementId> {
        let start = self.network.lane(from_lane).movement_start;
        self.network
            .movements_of(from_lane)
            .iter()
            .position(|m| self.network.lane(m.to_lane).link == next_link)
            .map(|k| MovementId(start.0 + k as u32))
    }

    fn neighbors(&self) -> Neighbors {
        let mut by_lane: IntMap<Vec<usize>> = IntMap::default();
        let mut approaching: IntMap<Vec<usize>> = IntMap::default();
        let mut crossing_at: IntMap<Vec<usize>> = IntMap::default();
        for (i, v) in self.fleet.rows.iter().enumerate() {
            if let Some(c) = v.crossing {
                crossing_at.entry(self.network.movement(c.movement).node.0).or_default().push(i);
                continue; // inside a node — not part of any lane's car-following
            }
            by_lane.entry(v.lane.0).or_default().push(i);
            approaching.entry(self.downstream_node(v.lane).0).or_default().push(i);
        }
        let mut leader_of = vec![None; self.fleet.rows.len()];
        let mut lane_front: IntMap<usize> = IntMap::default();
        for members in by_lane.values_mut() {
            members.sort_by(|&a, &b| {
                self.fleet.rows[a].position.total_cmp(&self.fleet.rows[b].position)
            });
            for w in members.windows(2) {
                leader_of[w[0]] = Some(w[1]);
            }
            let front = *members.first().unwrap();
            lane_front.insert(self.fleet.rows[front].lane.0, front);
        }
        Neighbors { leader_of, lane_front, by_lane, approaching, crossing_at }
    }

    fn downstream_node(&self, lane: LaneId) -> NodeId {
        self.network.link(self.network.lane(lane).link).to
    }

    /// A strict priority order over links (higher wins): faster road first, then
    /// more lanes, then link id as a deterministic tie-break so opposing yields
    /// can never deadlock.
    fn priority_key(&self, link: LinkId) -> u64 {
        let l = self.network.link(link);
        let lane = self.network.lane(l.lane_start);
        ((lane.speed_limit * 1000.0) as u64) << 40 | (l.lane_count as u64) << 24 | (link.0 as u64 & 0xFF_FFFF)
    }

    /// MOBIL lane changes: evaluated on committed positions, applied before the
    /// longitudinal update. Discretionary (overtake a slow leader into a freer
    /// lane) and mandatory (move to a lane that serves the route's next link).
    fn lane_changes(&mut self) {
        let mut by_lane: IntMap<Vec<usize>> = IntMap::default();
        for (i, v) in self.fleet.rows.iter().enumerate() {
            if v.crossing.is_some() {
                continue; // no lane changes mid-intersection
            }
            by_lane.entry(v.lane.0).or_default().push(i);
        }
        for m in by_lane.values_mut() {
            m.sort_by(|&a, &b| self.fleet.rows[a].position.total_cmp(&self.fleet.rows[b].position));
        }
        let mut changes: Vec<(usize, LaneId)> = Vec::new();
        for i in 0..self.fleet.rows.len() {
            if let Some(t) = self.best_lane_change(i, &by_lane) {
                changes.push((i, t));
            }
        }
        for (i, target) in changes {
            // Preserve arc-length along the link across the change. Lanes normally
            // share a start offset (a no-op remap), but a turn-*pocket* lane begins
            // partway down the link, so a car can only move into it once it is
            // within the pocket's span.
            let veh = &self.fleet.rows[i];
            let arc = self.network.lane(veh.lane).start_offset + veh.position;
            let tgt = self.network.lane(target);
            let new_pos = arc - tgt.start_offset;
            if new_pos < 0.0 || new_pos > tgt.length {
                continue;
            }
            let len = veh.driver.vehicle_length;
            if self.lane_slot_clear(target, new_pos, len, i) {
                self.fleet.rows[i].lane = target;
                self.fleet.rows[i].position = new_pos;
            }
        }
    }

    fn best_lane_change(&self, i: usize, by_lane: &IntMap<Vec<usize>>) -> Option<LaneId> {
        let v = &self.fleet.rows[i];
        let lane = *self.network.lane(v.lane);
        let link = *self.network.link(lane.link);
        let idx = lane.index_in_link as i64;
        let cur_leader = self.nearest_ahead(v.lane, v.position, by_lane, i);
        let a_self_cur = idm_follow(v, lane.speed_limit, v.position, v.speed, cur_leader.map(|j| &self.fleet.rows[j]));

        let mut best: Option<(f64, LaneId)> = None;
        for delta in [-1i64, 1] {
            let ti = idx + delta;
            if ti < 0 || ti >= link.lane_count as i64 {
                continue;
            }
            let target = LaneId(link.lane_start.0 + ti as u32);
            let limit = self.network.lane(target).speed_limit;

            let a_self_new = idm_follow(
                v,
                limit,
                v.position,
                v.speed,
                self.nearest_ahead(target, v.position, by_lane, i).map(|j| &self.fleet.rows[j]),
            );

            let (a_nf_cur, a_nf_new) = match self.nearest_behind(target, v.position, by_lane, i) {
                Some(fj) => {
                    let f = &self.fleet.rows[fj];
                    let fl = self.nearest_ahead(target, f.position, by_lane, i).map(|j| &self.fleet.rows[j]);
                    (
                        idm_follow(f, limit, f.position, f.speed, fl),
                        idm_follow(f, limit, f.position, f.speed, Some(v)),
                    )
                }
                None => (0.0, 0.0),
            };

            let mandatory = self.mandatory_change(v, v.lane, target);
            let params = MobilParams::new(v.driver.politeness);
            if mobil::should_change(&params, a_self_cur, a_self_new, a_nf_cur, a_nf_new, mandatory) {
                let gain = (a_self_new - a_self_cur) + if mandatory { 100.0 } else { 0.0 };
                if best.is_none_or(|(g, _)| gain > g) {
                    best = Some((gain, target));
                }
            }
        }
        best.map(|(_, t)| t)
    }

    // `by_lane` lists are sorted ascending by position, so the neighbour just
    // ahead/behind is found by a binary partition instead of a full lane scan
    // (this ran ~7× per vehicle in `best_lane_change`).
    fn nearest_ahead(&self, lane: LaneId, pos: f64, by_lane: &IntMap<Vec<usize>>, exclude: usize) -> Option<usize> {
        let list = by_lane.get(&lane.0)?;
        let idx = list.partition_point(|&j| self.fleet.rows[j].position <= pos);
        list[idx..].iter().copied().find(|&j| j != exclude)
    }

    fn nearest_behind(&self, lane: LaneId, pos: f64, by_lane: &IntMap<Vec<usize>>, exclude: usize) -> Option<usize> {
        let list = by_lane.get(&lane.0)?;
        let idx = list.partition_point(|&j| self.fleet.rows[j].position < pos);
        list[..idx].iter().rev().copied().find(|&j| j != exclude)
    }

    /// Whether the current lane can't serve the route's next link but `target` can.
    fn mandatory_change(&self, veh: &NetVehicle, current: LaneId, target: LaneId) -> bool {
        matches!(
            (self.lane_serves_route(veh, current), self.lane_serves_route(veh, target)),
            (Some(false), Some(true))
        )
    }

    fn lane_serves_route(&self, veh: &NetVehicle, lane: LaneId) -> Option<bool> {
        let next = self.next_link_on_path(veh)?;
        Some(self.network.movements_of(lane).iter().any(|m| self.network.lane(m.to_lane).link == next))
    }

    /// The next link on this vehicle's path — from its explicit route, or (for
    /// destination-routed vehicles) the flow-field next hop. Without this,
    /// dest-routed cars never make the mandatory lane change into their turn lane
    /// once movements are channelised per lane.
    fn next_link_on_path(&self, veh: &NetVehicle) -> Option<LinkId> {
        if !veh.route.is_empty() {
            return (veh.route_idx + 1 < veh.route.len()).then(|| veh.route[veh.route_idx + 1]);
        }
        let (dest, router) = (veh.dest?, self.router.as_ref()?);
        router.next_hop(dest, self.network.lane(veh.lane).link)
    }

    fn lane_slot_clear(&self, target: LaneId, pos: f64, len: f64, exclude: usize) -> bool {
        self.fleet.rows
            .iter()
            .enumerate()
            .filter(|(j, o)| *j != exclude && o.lane == target)
            .all(|(_, o)| {
                if o.position > pos {
                    o.position - o.driver.vehicle_length - pos > 0.5
                } else {
                    pos - len - o.position > 0.5
                }
            })
    }

    /// Current colour of a movement under the actuated signal runtime
    /// (unsignalized movements are always green).
    fn movement_state(&self, mid: MovementId) -> SignalState {
        self.signals.movement_state(&self.network, mid)
    }

    /// Colour of every signal group, indexed by group id — for rendering.
    pub fn signal_states(&self) -> Vec<SignalState> {
        self.signals.states(&self.network)
    }

    /// Detect stop-line demand (vehicles within the detector zone) and advance
    /// the actuated signals.
    fn advance_signals(&mut self, dt: f64) {
        let mut demand: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for v in &self.fleet.rows {
            let lane = self.network.lane(v.lane);
            if lane.length - v.position < junction::DETECT {
                demand.insert(lane.link.0);
            }
        }
        self.signals.advance(&self.network, &demand, dt);
    }

    fn refresh_routes(&mut self) {
        if self.router.is_none() || self.external_reroute {
            return;
        }
        // Spread the flow-field rebuild across the reroute interval: refresh a slice
        // of destinations each tick so every field is current within the interval,
        // without the whole-graph recompute landing on one tick (a visible freeze).
        let interval_ticks = (REROUTE_INTERVAL_SECS / self.cfg.dt).max(1.0);
        let costs = self.live_link_costs();
        if let Some(r) = self.router.as_mut() {
            let per_tick = (r.destination_count() as f64 / interval_ticks).ceil().max(1.0) as usize;
            r.recompute_incremental(&costs, per_tick);
        }
    }

    pub fn step(&mut self) {
        let dt = self.cfg.dt;
        let mut prof = Prof::new();
        self.refresh_routes();
        prof.lap(0);
        self.advance_signals(dt);
        prof.lap(1);
        self.lane_changes();
        prof.lap(2);
        let nb = self.neighbors();
        prof.lap(3);

        let mut cross_by_mv: IntMap<Vec<usize>> = IntMap::default();
        for (i, v) in self.fleet.rows.iter().enumerate() {
            if let Some(c) = v.crossing {
                cross_by_mv.entry(c.movement.0).or_default().push(i);
            }
        }

        // Executor for the per-vehicle passes below (serial / CPU threads / GPU),
        // resolved against what's available on this device, plus the count at which the
        // Threads backend starts parallelizing.
        let backend = self.active_backend();
        let par_threshold = self.par_threshold;

        // Compute each vehicle's intended movement once (used by accel, the box
        // gate, and the transition).
        let intended_mv: Vec<Option<MovementId>> = map_collect(backend, par_threshold, self.fleet.rows.len(), |i| {
            let v = &self.fleet.rows[i];
            if v.crossing.is_some() { None } else { self.intended_movement(v) }
        });

        // Phase 4a — gather each vehicle's accel-decision inputs. Every graph,
        // neighbor, signal and router lookup lives here (CPU-only; a GPU kernel can't
        // chase these pointers). Rolling vehicles get a flat `VehicleContext`; in-node
        // crossers carry their bespoke `crossing_accel` through unchanged.
        let inputs: Vec<AccelInput> = map_collect(backend, par_threshold, self.fleet.rows.len(), |i| {
            if self.fleet.rows[i].crossing.is_some() {
                AccelInput::Crossing(self.crossing_accel(i, &nb, &cross_by_mv))
            } else {
                AccelInput::Rolling(self.gather_context(i, &nb, &intended_mv))
            }
        });
        // Phase 4b — evaluate: the pure constraint fold + reproducible noise over the
        // flat context. No graph access — this half transliterates to a WGSL kernel.
        let (seed, tick) = (self.cfg.seed, self.tick);
        let accels: Vec<f64> = map_collect(backend, par_threshold, inputs.len(), |i| inputs[i].evaluate(seed, tick));
        prof.lap(4);

        // Destination-lane occupancy (nearest-to-entrance rear), so a crosser
        // finishing its interior path never lands on top of a queued vehicle —
        // instead it holds at the far edge of the node (box-blocking, realistic).
        let mut front: IntMap<f64> = IntMap::default();
        for v in &self.fleet.rows {
            if v.crossing.is_none() && v.position < self.network.lane(v.lane).length {
                let e = front.entry(v.lane.0).or_insert(f64::MAX);
                *e = e.min(v.position - v.driver.vehicle_length);
            }
        }

        // A hard don't-enter-the-box gate: never begin crossing while a conflicting
        // movement still occupies the node (a straggler from the previous phase),
        // which the soft box-yield can't guarantee under momentum. Precomputed here
        // while committed state and `nb` are valid, since the advance pass empties
        // `self.fleet.rows`.
        let block_entry: Vec<bool> = map_collect(backend, par_threshold, self.fleet.rows.len(), |i| {
            let veh = &self.fleet.rows[i];
            intended_mv[i].is_some_and(|mid| {
                // Free-flow freeway movements never hard-block on a box conflict (the
                // merge is a zipper); only at-grade crossings gate on box occupancy.
                !self.network.is_interchange_movement(mid)
                    && self.box_conflict(mid, self.network.link(self.network.lane(veh.lane).link).to, &nb)
            })
        });

        let n = self.fleet.len();
        let mut rows = Vec::with_capacity(n);
        let mut hist = Vec::with_capacity(n);
        let mut hist_len = Vec::with_capacity(n);
        let mut exited = 0u32;
        let taken = std::mem::take(&mut self.fleet.rows);
        let taken_h = std::mem::take(&mut self.fleet.hist);
        let taken_hl = std::mem::take(&mut self.fleet.hist_len);
        for (((((mut veh, a), block), intended), mut h), mut hl) in taken
            .into_iter()
            .zip(accels)
            .zip(block_entry)
            .zip(intended_mv)
            .zip(taken_h)
            .zip(taken_hl)
        {
            let fate = self.advance_vehicle(&mut veh, a, dt, &mut front, block, intended);
            let keep = match fate {
                Fate::Alive => {
                    veh.wait_ticks = if veh.speed < 0.5 { veh.wait_ticks + 1 } else { 0 };
                    true
                }
                Fate::Entered(link) => {
                    self.link_entries[link.idx()] += 1;
                    veh.wait_ticks = 0;
                    // Crossed into a new lane: the retained position history is in the
                    // previous segment's frame. Discard it so the reaction-delay leader
                    // gap doesn't read a stale cross-frame position and phantom-brake the
                    // car to a dead stop the instant it traverses a segment boundary.
                    hl = 0;
                    true
                }
                Fate::Exited => {
                    exited += 1;
                    false
                }
                Fate::Leaked => {
                    self.leaked += 1;
                    false
                }
            };
            if keep {
                record_history(&mut h, &mut hl, veh.position, veh.speed);
                rows.push(veh);
                hist.push(h);
                hist_len.push(hl);
            }
        }
        self.fleet.rows = rows;
        self.fleet.hist = hist;
        self.fleet.hist_len = hist_len;
        self.exited += exited;
        prof.lap(5);
        self.remove_crashes();
        self.remove_conflict_crashes();
        self.remove_overlap_crashes();
        prof.lap(6);
        self.time += dt;
        self.tick += 1;
    }

    /// Advance one vehicle: integrate its longitudinal state, drive the
    /// lane→interior→lane transitions, and report whether it stayed on the road,
    /// entered a new link, or left the network.
    fn advance_vehicle(&self, veh: &mut NetVehicle, accel: f64, dt: f64, front: &mut IntMap<f64>, block_entry: bool, intended: Option<MovementId>) -> Fate {
        if let Some(mut c) = veh.crossing {
            let it = *self.network.interior(c.movement);
            veh.speed = (veh.speed + accel * dt).max(0.0);
            c.s += veh.speed * dt;
            if c.s < it.len {
                veh.crossing = Some(c);
                return Fate::Alive;
            }
            // Reached the far edge: land on the destination lane unless its
            // entrance is occupied, in which case wait inside the node.
            let to_lane = self.network.movement(c.movement).to_lane;
            let clear = self.receiving_lane_clear(c.movement, front, veh.driver.min_gap);
            if !clear {
                c.s = it.len;
                veh.speed = 0.0;
                veh.crossing = Some(c);
                return Fate::Alive;
            }
            veh.crossing = None;
            veh.lane = to_lane;
            veh.position = c.s - it.len;
            veh.stopped_at = None;
            let e = front.entry(to_lane.0).or_insert(f64::MAX);
            *e = e.min(veh.position - veh.driver.vehicle_length);
            let to_link = self.network.lane(to_lane).link;
            if veh.route_idx + 1 < veh.route.len() && veh.route[veh.route_idx + 1] == to_link {
                veh.route_idx += 1;
            }
            return Fate::Entered(to_link);
        }

        integrate(veh, accel, dt);
        let lane = *self.network.lane(veh.lane);
        let node = self.network.link(lane.link).to;
        if matches!(self.network.node(node).control, NodeControl::Stop)
            && veh.speed < 0.3
            && (lane.length - veh.position) < veh.driver.vehicle_length + veh.driver.min_gap + 1.0
        {
            veh.stopped_at = Some(node);
        }
        if veh.position < lane.length {
            return Fate::Alive;
        }
        // Reached the stop line. Enter the interior only when the movement is
        // served (green/yellow) *and* the receiving lane can accept the vehicle —
        // don't block the box, so a spillback holds at the line rather than
        // stalling inside the node where cross traffic will T-bone it.
        match intended {
            Some(mid)
                if self.movement_state(mid) != SignalState::Red
                    && self.receiving_lane_clear(mid, front, veh.driver.min_gap)
                    && !block_entry =>
            {
                let overflow = veh.position - lane.length;
                veh.position = lane.length;
                veh.crossing = Some(Crossing { movement: mid, s: overflow.min(self.network.interior(mid).len) });
                Fate::Alive
            }
            Some(_) => {
                veh.position = lane.length;
                veh.speed = 0.0;
                Fate::Alive
            }
            // No movement resolved. Legitimate when the car has arrived (no next
            // hop) or run off a genuine dead end; a leak if it still had somewhere
            // to go — which `intended_movement`'s fallback prevents.
            None if self.still_has_a_route(veh, lane.link) && self.network.links.iter().any(|l| l.from == node) => Fate::Leaked,
            None => Fate::Exited,
        }
    }

    /// Whether the vehicle still has an onward hop it hasn't taken — a routed next
    /// link or an unfinished explicit route. Used to tell a genuine exit from a leak.
    fn still_has_a_route(&self, veh: &NetVehicle, from: LinkId) -> bool {
        match (veh.dest, self.router.as_ref()) {
            (Some(dest), Some(router)) => router.next_hop(dest, from).is_some(),
            _ => veh.route_idx + 1 < veh.route.len(),
        }
    }

    /// Whether the lane a movement feeds into has room at its entrance to receive
    /// a vehicle (its nearest occupant's rear is at least `min_gap` from the start).
    fn receiving_lane_clear(&self, mid: MovementId, front: &IntMap<f64>, min_gap: f64) -> bool {
        let to_lane = self.network.movement(mid).to_lane;
        front.get(&to_lane.0).is_none_or(|&rear| rear >= min_gap)
    }

    /// IDM acceleration for a vehicle traversing a node interior: capped to the
    /// turn's comfortable speed and following either a crosser ahead on the same
    /// movement or the queue waiting on the destination lane.
    /// Gather a rolling vehicle's accel-decision context — the graph, neighbor,
    /// signal and router lookups a GPU evaluate kernel can't do. Returns a flat,
    /// owned [`VehicleContext`]; [`VehicleContext::evaluate`] turns it into an
    /// acceleration with no further graph access. Behaviour is identical to the old
    /// fused accel loop; this is purely the gather/evaluate split.
    fn gather_context(&self, i: usize, nb: &Neighbors, intended_mv: &[Option<MovementId>]) -> VehicleContext {
        let dt = self.cfg.dt;
        let veh = &self.fleet.rows[i];
        let lane = *self.network.lane(veh.lane);
        let driver = veh.driver.capped_to(lane.speed_limit);
        let intended = intended_mv[i];
        let node = self.downstream_node(veh.lane);
        let control = self.network.node(node).control;
        let to_line = (lane.length - veh.position).max(0.05);

        // Perceive the leader as of `reaction_time` ago, but cap the gap by the true
        // current gap so IDM can never under-brake (collision-free in-lane); the
        // delay only adds start-up lag.
        let leader = if let Some(li) = nb.leader_of[i] {
            let lead = &self.fleet.rows[li];
            let delay = (driver.reaction_time / dt).round() as usize;
            let current_gap = lead.position - veh.position - lead.driver.vehicle_length;
            // The reaction-delay gap is only meaningful when both cars have a full delay
            // window of history on the *current* lane. Right after either crosses a
            // segment boundary its delayed position is in a stale frame (history is reset
            // on crossing), so fall back to the true current gap and the leader's current
            // speed — otherwise the delayed lookup reads a cross-frame position and
            // phantom-brakes the car to a dead stop the moment it traverses a boundary.
            let (gap, speed) = if self.fleet.settled(i, delay) && self.fleet.settled(li, delay) {
                let (my_p, _) = self.fleet.delayed(i, delay);
                let (lead_p, lead_v) = self.fleet.delayed(li, delay);
                ((lead_p - my_p - lead.driver.vehicle_length).min(current_gap), lead_v)
            } else {
                (current_gap, lead.speed)
            };
            Some(Obstacle { gap, speed })
        } else if let Some(mid) = intended {
            let to_lane = self.network.movement(mid).to_lane;
            nb.lane_front.get(&to_lane.0).map(|&front| {
                let lead = &self.fleet.rows[front];
                Obstacle {
                    gap: (lane.length - veh.position) + lead.position - lead.driver.vehicle_length,
                    speed: lead.speed,
                }
            })
        } else {
            None
        };

        // Upcoming curve (lateral-accel limit) and turn speed; the LOD path below
        // needs these, so compute them before it.
        const A_LAT: f64 = 3.0;
        const CURVE_LOOKAHEAD: f64 = 45.0;
        let geom_curve = {
            let r = self.network.min_radius_ahead(veh.lane, veh.position, CURVE_LOOKAHEAD);
            r.is_finite().then(|| SpeedTarget { speed: (A_LAT * r).sqrt(), distance: CURVE_LOOKAHEAD })
        };
        let turn = intended
            .and_then(|mid| {
                if self.network.is_interchange_movement(mid) {
                    return None; // free-flow diverge/merge; the ramp curve slows it
                }
                match self.network.movement_turn(mid) {
                    TurnType::Left => Some(6.0),
                    TurnType::Right => Some(5.0),
                    TurnType::Through => None,
                }
            })
            .map(|speed| SpeedTarget { speed, distance: to_line.max(12.0) });
        let curve = match (geom_curve, turn) {
            (Some(a), Some(b)) => Some(if a.speed <= b.speed { a } else { b }),
            (a, b) => a.or(b),
        };

        let mut cx = VehicleContext::new(driver, veh.speed, veh.id);
        cx.set_leader(leader);
        cx.set_curve(curve);

        // LOD: beyond the range at which any node constraint can bind (and with no
        // slower zone downstream), only the leader and local curvature matter — skip
        // the node stack. Identical result, cheaper.
        const DECISION_HORIZON: f64 = 180.0;
        let downstream_slower = intended
            .is_some_and(|mid| self.network.lane(self.network.movement(mid).to_lane).speed_limit < lane.speed_limit);
        if to_line > DECISION_HORIZON && !downstream_slower {
            return cx;
        }

        // Don't-block-the-box: about to cross, but the downstream lane's entrance is
        // occupied — hold at the line rather than land on top of a stopped vehicle
        // (the main source of intersection crashes).
        let downstream_blocked = intended.is_some_and(|mid| {
            let to_lane = self.network.movement(mid).to_lane;
            nb.lane_front
                .get(&to_lane.0)
                .is_some_and(|&f| self.fleet.rows[f].position < driver.vehicle_length + driver.min_gap)
        });
        // Signal: red always stops; yellow stops only if the vehicle can brake
        // comfortably before the line (dilemma zone) — otherwise it proceeds and
        // clears on yellow.
        let signal_stop = intended.is_some_and(|mid| match self.movement_state(mid) {
            SignalState::Green => false,
            SignalState::Red => true,
            SignalState::Yellow => can_stop_before(veh.speed, driver.comfort_decel, to_line),
        });
        // Stop at the immediate line (red / blocked box), or ease toward a red one
        // intersection ahead so the slowdown starts an earlier link.
        let stop_line = (signal_stop || downstream_blocked)
            .then_some(to_line)
            .or_else(|| self.red_ahead(veh, intended, to_line));

        let speed_target = intended.and_then(|mid| {
            let to = self.network.lane(self.network.movement(mid).to_lane);
            let target = veh.driver.desired_speed.min(to.speed_limit);
            (target < driver.desired_speed).then_some(SpeedTarget { speed: target, distance: to_line })
        });

        let stop_sign = (matches!(control, NodeControl::Stop) && veh.stopped_at != Some(node))
            .then_some(to_line);

        // A freeway diverge/merge is free-flow — no crossing traffic to yield to (the
        // merge is a zipper, handled by `merge`, not a box crossing). So a freeway
        // through/merge/diverge movement never box-yields or box-blocks; that gating is
        // what was wrongly stopping cars mid-freeway at on-ramp merges.
        let free_flow = intended.is_some_and(|mid| self.network.is_interchange_movement(mid));
        // Never enter a box occupied by conflicting crossing traffic (at-grade nodes);
        // additionally, at unsignalized nodes defer to higher-priority approaching
        // traffic by right-of-way.
        let box_yield = !free_flow && intended.is_some_and(|mid| self.box_conflict(mid, node, nb));
        let prio_yield = !free_flow
            && matches!(control, NodeControl::Uncontrolled | NodeControl::Stop | NodeControl::Yield)
            && self.conflicting_priority_traffic(i, veh.lane, node, nb).is_some();
        let yield_line = (box_yield || prio_yield).then_some(to_line);

        let merge = self.merge_conflict(veh, lane.length, intended, nb);

        cx.set_stop_line(stop_line);
        cx.set_speed_target(speed_target);
        cx.set_stop_sign(stop_sign);
        cx.set_yield_line(yield_line);
        cx.set_merge(merge);
        cx
    }

    fn crossing_accel(&self, i: usize, nb: &Neighbors, cross_by_mv: &IntMap<Vec<usize>>) -> f64 {
        let veh = &self.fleet.rows[i];
        let c = veh.crossing.unwrap();
        let it = self.network.interior(c.movement);
        let to_lane = self.network.movement(c.movement).to_lane;
        // A freeway diverge/merge is free-flow — no at-grade turn throttle; the
        // ramp's own speed limit and curvature (curve-speed limiting on the ramp
        // link) are what slow a car down, not a hard 5 m/s crawl through the gore.
        let cap = if self.network.is_interchange_movement(c.movement) {
            f64::INFINITY
        } else {
            match self.network.movement_turn(c.movement) {
                TurnType::Left => 6.0,
                TurnType::Right => 5.0,
                TurnType::Through => f64::INFINITY,
            }
        };
        let mut d = veh.driver.capped_to(self.network.lane(to_lane).speed_limit);
        d.desired_speed = d.desired_speed.min(cap);

        let mut gap = f64::INFINITY;
        let mut lead_speed = veh.speed;
        for &j in cross_by_mv.get(&c.movement.0).into_iter().flatten() {
            if j == i {
                continue;
            }
            let o = self.fleet.rows[j].crossing.unwrap();
            if o.s > c.s {
                let g = o.s - c.s - self.fleet.rows[j].driver.vehicle_length;
                if g < gap {
                    gap = g;
                    lead_speed = self.fleet.rows[j].speed;
                }
            }
        }
        if let Some(&f) = nb.lane_front.get(&to_lane.0) {
            let l = &self.fleet.rows[f];
            let g = (it.len - c.s) + l.position - l.driver.vehicle_length;
            if g < gap {
                gap = g;
                lead_speed = l.speed;
            }
        }
        let mut accel = if gap.is_finite() {
            idm::acceleration(&d, veh.speed, veh.speed - lead_speed, gap.max(0.05))
        } else {
            idm::free_acceleration(&d, veh.speed)
        };

        // In-intersection avoidance: if a vehicle on a conflicting movement will
        // reach a shared conflict point first, brake to stop short of it. This is
        // the crash-avoidant behaviour; a collision only occurs when a driver is
        // already too close/fast to stop (then the conflict-point check crashes it).
        let node = self.network.movement(c.movement).node;
        for &ci in self.junctions.conflict_ids(node) {
            let cp = &self.network.conflicts[ci as usize];
            let (my_s, other_mv, other_s) = if cp.a == c.movement {
                (cp.sa, cp.b, cp.sb)
            } else if cp.b == c.movement {
                (cp.sb, cp.a, cp.sa)
            } else {
                continue;
            };
            let my_dist = my_s - c.s;
            if my_dist <= 0.0 {
                continue; // already through this point
            }
            for &j in cross_by_mv.get(&other_mv.0).into_iter().flatten() {
                let o = &self.fleet.rows[j];
                let their_dist = other_s - o.crossing.unwrap().s;
                if their_dist < -2.0 {
                    continue; // they have cleared the point
                }
                let they_go_first = their_dist < my_dist || (their_dist == my_dist && o.id < veh.id);
                if they_go_first {
                    let stop_gap = (my_dist - 1.0).max(0.05); // hold ~1 m short of the point
                    accel = accel.min(idm::acceleration(&d, veh.speed, veh.speed, stop_gap));
                }
            }
        }
        accel
    }

    /// Detect rear-end overlaps within a lane (a follower's front past the
    /// leader's rear) and take the crashed vehicles off the road, counting them.
    /// Reaction delay makes this physically possible; behaviour should minimize
    /// it over time. Crashes are resolved deterministically (scan by position).
    fn remove_crashes(&mut self) {
        const OVERLAP_TOL: f64 = 0.5;
        let mut by_lane: IntMap<Vec<usize>> = IntMap::default();
        for (i, v) in self.fleet.rows.iter().enumerate() {
            if v.crossing.is_some() {
                continue; // crossers are pinned at the line; handled by conflict crashes
            }
            by_lane.entry(v.lane.0).or_default().push(i);
        }
        let mut crashed = vec![false; self.fleet.rows.len()];
        for members in by_lane.values_mut() {
            members.sort_by(|&a, &b| self.fleet.rows[a].position.total_cmp(&self.fleet.rows[b].position));
            for w in members.windows(2) {
                let (rear, front) = (&self.fleet.rows[w[0]], &self.fleet.rows[w[1]]);
                let gap = front.position - rear.position - front.driver.vehicle_length;
                if gap < -OVERLAP_TOL {
                    crashed[w[0]] = true;
                    crashed[w[1]] = true;
                }
            }
        }
        self.take_crashed(crashed);
    }

    /// Detect collisions inside intersections: two vehicles traversing different
    /// movements that both occupy a precomputed conflict point this tick. This is
    /// what makes running a red/misjudging a gap actually crash. Cost is linear —
    /// only vehicles currently crossing are indexed, against static conflicts.
    fn remove_conflict_crashes(&mut self) {
        const CROSS_TOL: f64 = 3.0;
        let mut at: IntMap<Vec<(usize, f64)>> = IntMap::default();
        for (i, v) in self.fleet.rows.iter().enumerate() {
            if let Some(c) = v.crossing {
                at.entry(c.movement.0).or_default().push((i, c.s));
            }
        }
        if at.is_empty() {
            return;
        }
        let mut crashed = vec![false; self.fleet.rows.len()];
        for cp in &self.network.conflicts {
            let (Some(a), Some(b)) = (at.get(&cp.a.0), at.get(&cp.b.0)) else { continue };
            for &(i, sa) in a {
                if (sa - cp.sa).abs() >= CROSS_TOL {
                    continue;
                }
                for &(j, sb) in b {
                    if (sb - cp.sb).abs() < CROSS_TOL {
                        crashed[i] = true;
                        crashed[j] = true;
                    }
                }
            }
        }
        self.take_crashed(crashed);
    }

    /// General in-box collision detection: any two vehicles traversing the *same
    /// node* on *different* movements whose actual bodies overlap have collided —
    /// a real-time check on true positions, catching overlaps the precomputed
    /// conflict points miss (e.g. where merged geometry runs two paths together).
    /// Pruned to co-node crossers (collision-imminent only), so cost is the sum
    /// over nodes of k² with k = that node's handful of simultaneous crossers.
    fn remove_overlap_crashes(&mut self) {
        // (index, world position, heading, from-link, to-link)
        let mut by_node: IntMap<Vec<(usize, [f64; 2], f64, u32, u32)>> = IntMap::default();
        for (i, v) in self.fleet.rows.iter().enumerate() {
            if let Some(c) = v.crossing {
                let mv = self.network.movement(c.movement);
                let p = self.vehicle_world_pose(v);
                let (from_link, to_link) = (self.network.lane(mv.from_lane).link.0, self.network.lane(mv.to_lane).link.0);
                by_node.entry(mv.node.0).or_default().push((i, [p[0], p[1]], p[2], from_link, to_link));
            }
        }
        let mut crashed = vec![false; self.fleet.rows.len()];
        for group in by_node.values() {
            for a in 0..group.len() {
                for b in a + 1..group.len() {
                    let ((i, pi, hi, fi, ti), (j, pj, hj, fj, tj)) = (group[a], group[b]);
                    // Mirror the conflict builder: same approach fans out and same
                    // exit merges — both run parallel/zipper, not a crossing.
                    if fi == fj || ti == tj {
                        continue;
                    }
                    if crossing_overlap(pi, hi, pj, hj) {
                        crashed[i] = true;
                        crashed[j] = true;
                    }
                }
            }
        }
        self.take_crashed(crashed);
    }

    /// Remove the flagged vehicles (compacting every column) and count the crashes.
    fn take_crashed(&mut self, crashed: Vec<bool>) {
        if !crashed.iter().any(|&c| c) {
            return;
        }
        let mut keep = crashed.iter().map(|&c| !c);
        self.fleet.rows.retain(|_| keep.next().unwrap());
        let mut keep = crashed.iter().map(|&c| !c);
        self.fleet.hist.retain(|_| keep.next().unwrap());
        let mut keep = crashed.iter().map(|&c| !c);
        self.fleet.hist_len.retain(|_| keep.next().unwrap());
        self.crashed += crashed.iter().filter(|&&c| c).count() as u32;
    }

    /// Whether `mid`'s interior path conflicts with a vehicle already crossing
    /// `node` — you must not enter an occupied box, whatever the control.
    fn box_conflict(&self, mid: MovementId, node: NodeId, nb: &Neighbors) -> bool {
        nb.crossing_at.get(&node.0).into_iter().flatten().any(|&j| {
            let o = self.fleet.rows[j].crossing.unwrap().movement;
            self.network.movements_conflict(mid, o)
        })
    }

    /// `Some(())` when a higher-priority vehicle is approaching `node` from a
    /// different link and will arrive within the critical gap — the signal to
    /// give way. `None` means clear to proceed.
    fn conflicting_priority_traffic(&self, i: usize, lane: LaneId, node: NodeId, nb: &Neighbors) -> Option<()> {
        let me = &self.fleet.rows[i];
        // Impatience: the longer we've waited, the smaller the gap we'll accept.
        let critical = effective_critical_gap(me.driver.critical_gap, me.wait_ticks as f64 * self.cfg.dt);
        let my_link = self.network.lane(lane).link;
        let my_key = self.priority_key(my_link);
        let my_dir = self.network.arrival_dir(my_link);
        let my_turn = self.intended_movement(me).map_or(TurnType::Through, |m| self.network.movement_turn(m));
        for &j in nb.approaching.get(&node.0)? {
            if j == i {
                continue;
            }
            let o = &self.fleet.rows[j];
            let o_lane = *self.network.lane(o.lane);
            if o_lane.link == my_link || o.speed < 0.5 {
                continue;
            }
            if (o_lane.length - o.position) / o.speed.max(0.1) >= critical {
                continue;
            }
            let o_turn = self.intended_movement(o).map_or(TurnType::Through, |m| self.network.movement_turn(m));
            if should_yield_to(my_turn, my_dir, o_turn, self.network.arrival_dir(o_lane.link), my_key, self.priority_key(o_lane.link)) {
                return Some(());
            }
        }
        None
    }

    /// The nearest-to-the-merge conflicting vehicle on a converging lane, as an
    /// obstacle to follow. `None` when this vehicle is first to the merge (the
    /// other yields) or there is no merge.
    fn merge_conflict(&self, veh: &NetVehicle, lane_len: f64, intended: Option<MovementId>, nb: &Neighbors) -> Option<Obstacle> {
        let to_lane = self.network.movement(intended?).to_lane;
        let froms = self.merges.get(&to_lane.0)?;
        let my_dist = lane_len - veh.position;
        let mut best: Option<Obstacle> = None;
        for &from in froms {
            if from == veh.lane.0 {
                continue;
            }
            for &j in nb.by_lane.get(&from).into_iter().flatten() {
                let o = &self.fleet.rows[j];
                if o.speed < 0.5 {
                    continue;
                }
                let o_dist = self.network.lane(o.lane).length - o.position;
                if o_dist < my_dist {
                    let gap = my_dist - o_dist - o.driver.vehicle_length;
                    if best.is_none_or(|b| gap < b.gap) {
                        best = Some(Obstacle { gap, speed: o.speed });
                    }
                }
            }
        }
        best
    }

    pub fn run_ticks(&mut self, ticks: u32) {
        for _ in 0..ticks {
            self.step();
        }
    }
}

struct Neighbors {
    leader_of: Vec<Option<usize>>,
    lane_front: IntMap<usize>,
    by_lane: IntMap<Vec<usize>>,
    approaching: IntMap<Vec<usize>>,
    /// Vehicles currently inside each node (traversing an interior), by node id.
    crossing_at: IntMap<Vec<usize>>,
}

/// IDM acceleration for a vehicle placed at `pos`/`speed` on a lane with the
/// given speed limit, following `leader` (or free road if none). Used to score
/// hypothetical lane placements for MOBIL.
/// Turning-movement conflict rule: whether a vehicle making `my_turn` from
/// direction `my_dir` must yield to one making `o_turn` from `o_dir`.
/// - Same direction (parallel): no node conflict (car-following handles it).
/// - A right turn merges rather than crosses → doesn't yield here.
/// - Opposing: only a left turn yields, and only to oncoming through/right.
/// - Crossing: yield to the higher-priority approach.
fn should_yield_to(
    my_turn: TurnType,
    my_dir: [f64; 2],
    o_turn: TurnType,
    o_dir: [f64; 2],
    my_key: u64,
    o_key: u64,
) -> bool {
    let rel = (my_dir[0] * o_dir[0] + my_dir[1] * o_dir[1]).clamp(-1.0, 1.0).acos();
    if rel < 0.6 {
        return false; // ~same heading — parallel streams
    }
    if my_turn == TurnType::Right {
        return false; // right turn merges into the target road, not across
    }
    if rel > 2.5 {
        return my_turn == TurnType::Left && o_turn != TurnType::Left; // left yields to oncoming
    }
    o_key > my_key // crossing — defer to the major approach
}

/// Whether a vehicle can brake to a stop within `distance` at comfortable
/// deceleration — the dilemma-zone test for whether to stop on yellow.
fn can_stop_before(speed: f64, decel: f64, distance: f64) -> bool {
    speed * speed / (2.0 * decel.max(0.1)) <= distance
}

/// Whether two crossing vehicles' bodies genuinely collide. A collision is a
/// *crossing* — near-perpendicular headings within [`OVERLAP`] of each other. Two
/// vehicles running roughly parallel or anti-parallel (e.g. opposing protected
/// lefts diverging to opposite corners, or offset oncoming through lanes) are
/// passing, not colliding, even when their bodies momentarily come close.
fn crossing_overlap(pi: [f64; 2], hi: f64, pj: [f64; 2], hj: f64) -> bool {
    // Centre distance below which two ~2 m-wide bodies genuinely overlap;
    // conservative once vehicle length is accounted for.
    const OVERLAP: f64 = 2.4;
    if (hi - hj).cos().abs() > 0.7 {
        return false;
    }
    (pi[0] - pj[0]).hypot(pi[1] - pj[1]) < OVERLAP
}

/// The gap a driver will accept, shrinking from `base` as `waited` grows
/// (impatience), floored so nobody nudges into genuinely unsafe traffic.
fn effective_critical_gap(base: f64, waited: f64) -> f64 {
    (base - 0.15 * waited).max(1.5)
}

fn idm_follow(follower: &NetVehicle, lane_speed_limit: f64, pos: f64, speed: f64, leader: Option<&NetVehicle>) -> f64 {
    let d = follower.driver.capped_to(lane_speed_limit);
    match leader {
        Some(l) => idm::acceleration(&d, speed, speed - l.speed, (l.position - pos - l.driver.vehicle_length).max(0.05)),
        None => idm::free_acceleration(&d, speed),
    }
}

fn integrate(v: &mut NetVehicle, accel: f64, dt: f64) {
    if v.speed + accel * dt < 0.0 {
        v.position += -0.5 * v.speed * v.speed / accel;
        v.speed = 0.0;
    } else {
        v.position += v.speed * dt + 0.5 * accel * dt * dt;
        v.speed += accel * dt;
    }
}

#[cfg(test)]
mod tests {
    use super::super::boundary;
    use super::super::map::*;
    use super::super::network::{LaneId, LinkId};
    use super::*;

    fn cfg() -> SimConfig {
        SimConfig::default_config()
    }

    #[test]
    fn map_collect_matches_serial_on_every_backend() {
        // Every backend must collect element-for-element what the plain serial map does
        // — the per-vehicle step passes depend on this order preservation. A threshold of
        // 0 forces `Threads` onto rayon (under `--features parallel`); Gpu falls back but
        // must still agree.
        let n = 5000;
        let serial: Vec<usize> = (0..n).map(|i| i * i + 7).collect();
        for backend in [AccelBackend::Serial, AccelBackend::Threads, AccelBackend::Gpu] {
            assert_eq!(map_collect(backend, 0, n, |i| i * i + 7), serial, "backend {backend:?}");
        }
    }

    #[test]
    fn active_backend_falls_back_when_a_backend_is_unavailable() {
        let mut w = NetWorld::new(millbrae_sample(), cfg());
        assert_eq!(w.active_backend(), AccelBackend::Serial, "default is serial");
        // The GPU evaluate kernel isn't wired yet, so requesting it runs serially.
        w.set_accel_backend(AccelBackend::Gpu);
        assert_eq!(w.active_backend(), AccelBackend::Serial);
        // Threads resolves to itself only when the CPU pool is linked (the `parallel`
        // feature natively); otherwise it too falls back.
        w.set_accel_backend(AccelBackend::Threads);
        let expected = if cfg!(feature = "parallel") { AccelBackend::Threads } else { AccelBackend::Serial };
        assert_eq!(w.active_backend(), expected);
        assert_eq!(AccelBackend::from_name("threads"), AccelBackend::Threads);
        assert_eq!(AccelBackend::from_name("nonsense"), AccelBackend::Serial);
    }

    #[test]
    fn accel_wgsl_parses_and_validates() {
        // The GPU accel kernel must parse and type-check under plain `cargo test`, so a
        // WGSL typo fails CI here rather than silently in a browser (no adapter needed).
        let src = include_str!("accel.wgsl");
        let module = naga::front::wgsl::parse_str(src).expect("accel.wgsl should parse");
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .expect("accel.wgsl should type-check");
    }

    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    #[test]
    fn accel_gpu_matches_cpu_binding() {
        // The `accel.wgsl` fold must reproduce the CPU `binding` (noise excluded)
        // across a spread of constraint mixes, within f32 tolerance.
        use crate::sim::accel_gpu::binding_accels_gpu;
        let d = DriverConfig::car();
        let mut ctxs = Vec::new();
        ctxs.push(VehicleContext::new(d.capped_to(25.0), 12.0, 0)); // free road
        {
            let mut c = VehicleContext::new(d.capped_to(25.0), 20.0, 1);
            c.set_leader(Some(Obstacle { gap: 18.0, speed: 8.0 }));
            ctxs.push(c);
        }
        {
            let mut c = VehicleContext::new(d.capped_to(20.0), 15.0, 2);
            c.set_stop_line(Some(30.0));
            c.set_curve(Some(SpeedTarget { speed: 6.0, distance: 40.0 }));
            ctxs.push(c);
        }
        {
            let mut c = VehicleContext::new(d.capped_to(30.0), 22.0, 3);
            c.set_speed_target(Some(SpeedTarget { speed: 10.0, distance: 50.0 }));
            c.set_merge(Some(Obstacle { gap: 25.0, speed: 12.0 }));
            c.set_yield_line(Some(35.0));
            ctxs.push(c);
        }
        {
            let mut c = VehicleContext::new(d.capped_to(15.0), 8.0, 4);
            c.set_stop_sign(Some(12.0));
            ctxs.push(c);
        }

        let gpu_in: Vec<VehicleContextGpu> = ctxs.iter().map(VehicleContext::to_gpu).collect();
        let Some(gpu) = binding_accels_gpu(&gpu_in) else {
            eprintln!("no GPU adapter; skipping accel GPU/CPU equivalence test");
            return;
        };
        for (i, (c, &g)) in ctxs.iter().zip(&gpu).enumerate() {
            let cpu = c.binding();
            assert!((g as f64 - cpu).abs() < 0.02, "ctx {i}: gpu {g} vs cpu {cpu}");
        }
    }

    fn approach_lane(net: &Network) -> LaneId {
        net.lanes_of(LinkId(0)).next().unwrap()
    }

    fn signal_at(offset: f64) -> Network {
        let plan = SignalPlan { green_secs: 15.0, yellow_secs: 3.0, offset };
        OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::signalized(2, 150.0, 0.0, plan),
                NodeSpec::uncontrolled(3, 300.0, 0.0),
                NodeSpec::uncontrolled(4, 150.0, -120.0),
            ],
            links: vec![
                LinkSpec::oneway(1, 2, 1, 20.0),
                LinkSpec::oneway(2, 3, 1, 20.0),
                LinkSpec::oneway(4, 2, 1, 15.0),
            ],
        }
        .build()
    }

    #[test]
    fn actuated_signal_serves_both_competing_approaches() {
        // Two full queues on conflicting approaches: an actuated signal must
        // cycle to serve both (a stuck signal would strand one queue forever).
        let net = signal_at(0.0);
        let through = net.lanes_of(LinkId(0)).next().unwrap(); // 1->2
        let cross = net.lanes_of(LinkId(2)).next().unwrap(); // 4->2
        let mut world = NetWorld::new(net, cfg());
        let tlen = world.network.lane(through).length;
        let clen = world.network.lane(cross).length;
        for i in 0..4u32 {
            world.spawn(i, through, tlen - 8.0 - i as f64 * 7.0, 0.0, DriverConfig::car());
            world.spawn(100 + i, cross, clen - 8.0 - i as f64 * 7.0, 0.0, DriverConfig::car());
        }

        world.run_ticks(900);

        assert_eq!(world.exited(), 8, "both queues should clear once the signal cycles");
        assert_eq!(world.crashed(), 0);
    }

    #[test]
    fn link_flow_reflects_entries_over_time() {
        let mut w = NetWorld::new(straight_link(2000.0), cfg());
        for i in 0..10u32 {
            w.spawn(i, LaneId(0), 5.0 + i as f64 * 20.0, 5.0, DriverConfig::car());
        }
        for _ in 0..100 {
            w.step(); // 20 s
        }
        // 10 entries over 20 s ⇒ ~1800 veh/hour on the entry link.
        assert!((w.link_flows()[0] - 1800.0).abs() < 200.0, "flow {}", w.link_flows()[0]);
    }

    #[test]
    fn link_stats_report_count_speed_and_occupancy() {
        let mut w = NetWorld::new(straight_link(2000.0), cfg());
        for i in 0..8u32 {
            w.spawn(i, LaneId(0), 10.0 + i as f64 * 30.0, 12.0, DriverConfig::car());
        }
        w.step();
        let (count, mean, occ) = w.link_stats(LinkId(0));
        let manual: Vec<f64> = w.vehicles().iter().filter(|v| w.network.lane(v.lane).link == LinkId(0)).map(|v| v.speed).collect();
        assert_eq!(count as usize, manual.len());
        assert!((mean - manual.iter().sum::<f64>() / manual.len() as f64).abs() < 1e-9, "mean {mean}");
        assert!((0.0..=1.0).contains(&occ) && occ > 0.0, "occupancy {occ}");
    }

    #[test]
    fn link_stats_are_zeroed_for_an_empty_link() {
        let w = NetWorld::new(straight_link(500.0), cfg());
        assert_eq!(w.link_stats(LinkId(0)), (0, 0.0, 0.0));
    }

    #[test]
    fn congested_links_cost_more_than_empty_ones() {
        let net = OsmMap {
            nodes: vec![NodeSpec::uncontrolled(1, 0.0, 0.0), NodeSpec::uncontrolled(2, 300.0, 0.0)],
            links: vec![LinkSpec::oneway(1, 2, 1, 20.0)],
        }
        .build();
        let base = net.link_travel_time_ms(LinkId(0));
        let mut w = NetWorld::new(net, cfg());
        let empty = w.live_link_costs()[0];
        for i in 0..35u32 {
            w.spawn(i, LaneId(0), 10.0 + i as f64 * 7.0, 0.0, DriverConfig::car());
        }
        let jammed = w.live_link_costs()[0];
        assert_eq!(empty, base, "empty link is free-flow");
        assert!(jammed > empty * 2, "a jammed link should cost several times more: {jammed} vs {empty}");
    }

    #[test]
    fn impatience_shrinks_the_accepted_gap_to_a_floor() {
        assert_eq!(effective_critical_gap(4.0, 0.0), 4.0);
        assert!(effective_critical_gap(4.0, 10.0) < 4.0, "waiting lowers the bar");
        assert_eq!(effective_critical_gap(4.0, 1000.0), 1.5, "but never below the safety floor");
    }

    #[test]
    fn dilemma_zone_decision() {
        // Can stop comfortably before the line → stop; too fast/close → proceed.
        assert!(can_stop_before(10.0, 1.5, 100.0)); // plenty of room
        assert!(!can_stop_before(20.0, 1.5, 5.0)); // no chance — commit through
    }

    #[test]
    fn vehicle_proceeds_through_a_green_light() {
        let net = signal_at(0.0); // through movement green from t=0
        let lane = approach_lane(&net);
        let mut world = NetWorld::new(net, cfg());
        world.spawn(1, lane, 0.0, 15.0, DriverConfig::car());

        world.run_ticks(150); // 30 s: cross and reach the sink

        assert_eq!(world.exited(), 1, "vehicle should clear the intersection");
        assert!(world.vehicle(1).is_none());
    }

    #[test]
    fn follower_settles_behind_a_slower_leader_without_colliding() {
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::uncontrolled(2, 6000.0, 0.0),
            ],
            links: vec![LinkSpec::oneway(1, 2, 1, 30.0)],
        }
        .build();
        let lane = LaneId(0);
        let mut world = NetWorld::new(net, cfg());
        let slow = DriverConfig { desired_speed: 10.0, ..DriverConfig::car() };
        world.spawn(1, lane, 200.0, 10.0, slow);
        world.spawn(2, lane, 100.0, 10.0, DriverConfig::car());

        world.run_ticks(1500); // 300 s

        let leader = world.vehicle(1).unwrap();
        let follower = world.vehicle(2).unwrap();
        assert!((follower.speed - 10.0).abs() < 0.5, "follower speed={}", follower.speed);
        let gap = leader.position - follower.position - follower.driver.vehicle_length;
        assert!(gap > 0.0, "no collision, gap={gap}");
    }

    #[test]
    fn a_standing_queue_discharges_on_green() {
        let net = signal_at(18.0); // starts red, turns green at t=18
        let lane = approach_lane(&net);
        let length = net.lane(lane).length;
        let mut world = NetWorld::new(net, cfg());
        for i in 0..6u32 {
            world.spawn(i, lane, length - 8.0 - i as f64 * 7.0, 0.0, DriverConfig::car());
        }

        world.run_ticks(60); // settle into a stopped queue while red
        assert_eq!(world.exited(), 0, "nobody moves on red");

        world.run_ticks(500); // enough green cycles to serve the whole queue
        assert_eq!(world.exited(), 6, "the whole queue should discharge");
    }

    fn diamond() -> Network {
        OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(0, 0.0, 0.0),
                NodeSpec::uncontrolled(1, 100.0, 0.0),
                NodeSpec::uncontrolled(2, 200.0, -10.0),
                NodeSpec::uncontrolled(3, 200.0, 300.0),
                NodeSpec::uncontrolled(4, 300.0, 0.0),
                NodeSpec::uncontrolled(5, 400.0, 0.0),
            ],
            links: vec![
                LinkSpec::oneway(0, 1, 1, 20.0),
                LinkSpec::oneway(1, 2, 1, 20.0),
                LinkSpec::oneway(1, 3, 1, 20.0),
                LinkSpec::oneway(2, 4, 1, 20.0),
                LinkSpec::oneway(3, 4, 1, 20.0),
                LinkSpec::oneway(4, 5, 1, 20.0),
            ],
        }
        .build()
    }

    #[test]
    fn router_prefers_the_faster_path() {
        let net = diamond();
        let route = net.route_links(LinkId(0), LinkId(5)).expect("reachable");
        assert_eq!(route, vec![LinkId(0), LinkId(1), LinkId(3), LinkId(5)]);
    }

    #[test]
    fn router_reroutes_around_a_congested_link() {
        let net = diamond();
        let mut costs: Vec<u64> =
            (0..net.links.len() as u32).map(|i| net.link_travel_time_ms(LinkId(i))).collect();
        assert_eq!(net.route_links(LinkId(0), LinkId(5)).unwrap()[1], LinkId(1));
        costs[1] = 10_000_000; // link (1,2) is now jammed
        let rerouted = net.route_links_with_costs(LinkId(0), LinkId(5), &costs).unwrap();
        assert_eq!(rerouted, vec![LinkId(0), LinkId(2), LinkId(4), LinkId(5)]);
    }

    #[test]
    fn vehicles_follow_their_assigned_routes_to_exit() {
        let net = diamond();
        let mut world = NetWorld::new(net, cfg());
        let short = world.network.route_links(LinkId(0), LinkId(5)).unwrap();
        let long = vec![LinkId(0), LinkId(2), LinkId(4), LinkId(5)];
        assert!(world.spawn_routed(1, short, 12.0, DriverConfig::car()));
        world.run_ticks(50); // clear the shared entrance before the next spawn
        assert!(world.spawn_routed(2, long, 12.0, DriverConfig::car()));

        world.run_ticks(1000); // turns slow vehicles, so allow more time

        assert_eq!(world.exited(), 2, "both routed vehicles reach their destination");
    }

    fn symmetric_diamond() -> Network {
        OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(0, 0.0, 0.0),
                NodeSpec::uncontrolled(1, 100.0, 0.0),
                NodeSpec::uncontrolled(2, 200.0, -80.0),
                NodeSpec::uncontrolled(3, 200.0, 80.0),
                NodeSpec::uncontrolled(4, 300.0, 0.0),
                NodeSpec::uncontrolled(5, 400.0, 0.0),
            ],
            links: vec![
                LinkSpec::oneway(0, 1, 1, 20.0),
                LinkSpec::oneway(1, 2, 1, 20.0),
                LinkSpec::oneway(1, 3, 1, 20.0),
                LinkSpec::oneway(2, 4, 1, 20.0),
                LinkSpec::oneway(3, 4, 1, 20.0),
                LinkSpec::oneway(4, 5, 1, 20.0),
            ],
        }
        .build()
    }

    #[test]
    fn destination_routed_vehicle_drives_the_field_to_its_exit() {
        let mut world = NetWorld::new(diamond(), cfg());
        world.install_router(&[LinkId(5)]);
        assert!(world.spawn_to(1, LinkId(0), LinkId(5), 12.0, DriverConfig::car()));
        world.run_ticks(1000);
        assert_eq!(world.exited(), 1, "a vehicle with only a destination reaches it via the flow field");
    }

    #[test]
    fn through_traffic_enters_and_leaves_at_gateways() {
        let net = diamond();
        assert_eq!(boundary::entry_links(&net), vec![LinkId(0)]);
        assert_eq!(boundary::exit_links(&net), vec![LinkId(5)]);
        let mut world = NetWorld::new(net, cfg());
        world.install_router(&[LinkId(5)]);
        assert!(world.spawn_to(1, LinkId(0), LinkId(5), 12.0, DriverConfig::car()));
        world.run_ticks(1000);
        assert_eq!(world.exited(), 1, "external traffic crosses from the entry gateway to the exit gateway");
    }

    #[test]
    fn in_flight_traffic_spreads_onto_the_second_arm_when_the_first_congests() {
        let mut world = NetWorld::new(symmetric_diamond(), cfg());
        world.install_router(&[LinkId(5)]);
        let mut next = 0u32;
        for t in 0..3000 {
            if t % 3 == 0 && world.spawn_to(next, LinkId(0), LinkId(5), 12.0, DriverConfig::car()) {
                next += 1;
            }
            world.step();
        }
        let flows = world.link_flows();
        assert_eq!(world.crashed(), 0, "rerouting stays collision-free");
        assert!(flows[1] > 0.0, "the tie-preferred arm carries traffic");
        assert!(flows[2] > 0.0, "as the first arm congests, in-flight cars are steered onto the second");
        assert!(world.exited() > 20, "traffic keeps flowing through, got {}", world.exited());
    }

    #[test]
    fn brakes_in_advance_for_a_slower_road_ahead() {
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::uncontrolled(2, 200.0, 0.0),
                NodeSpec::uncontrolled(3, 400.0, 0.0),
            ],
            links: vec![
                LinkSpec::oneway(1, 2, 1, 25.0), // fast approach
                LinkSpec::oneway(2, 3, 1, 8.0),  // slow road ahead
            ],
        }
        .build();
        let fast = net.lanes_of(LinkId(0)).next().unwrap();
        let slow = net.lanes_of(LinkId(1)).next().unwrap();
        let mut world = NetWorld::new(net, cfg());
        world.spawn(1, fast, 0.0, 24.0, DriverConfig::car());

        let mut entry_speed = None;
        for _ in 0..300 {
            let before = world.vehicle(1).map(|v| (v.lane, v.speed));
            world.step();
            let after = world.vehicle(1).map(|v| v.lane);
            if let (Some((lane_before, s)), Some(lane_after)) = (before, after) {
                if lane_before == fast && lane_after == slow {
                    entry_speed = Some(s);
                    break;
                }
            }
        }

        let s = entry_speed.expect("vehicle should cross onto the slow road");
        assert!(s < 11.0, "should have slowed toward 8 m/s before crossing, entered at {s}");
    }

    #[test]
    fn a_vehicle_fully_stops_at_a_stop_sign_then_proceeds() {
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec { osm_id: 2, x: 150.0, y: 0.0, control: MapControl::Stop },
                NodeSpec::uncontrolled(3, 300.0, 0.0),
            ],
            links: vec![LinkSpec::oneway(1, 2, 1, 15.0), LinkSpec::oneway(2, 3, 1, 15.0)],
        }
        .build();
        let approach = net.lanes_of(LinkId(0)).next().unwrap();
        let mut world = NetWorld::new(net, cfg());
        world.spawn(1, approach, 0.0, 14.0, DriverConfig::car());

        let mut min_speed_on_approach = f64::MAX;
        for _ in 0..250 {
            world.step();
            if let Some(v) = world.vehicle(1) {
                if v.lane == approach {
                    min_speed_on_approach = min_speed_on_approach.min(v.speed);
                }
            }
        }
        assert!(min_speed_on_approach < 0.5, "must come to a full stop, min {min_speed_on_approach}");
        assert_eq!(world.exited(), 1, "and then continue through");
    }

    #[test]
    fn minor_road_yields_to_the_major_road_then_goes() {
        // Minor goes *straight across* (south→north) the major (west→east), so the
        // crossing conflict rule makes it defer to the higher-priority major. The
        // minor's approach is short so it reaches the line first yet still yields.
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -120.0, 0.0),
                NodeSpec { osm_id: 2, x: 0.0, y: 0.0, control: MapControl::Yield },
                NodeSpec::uncontrolled(3, 200.0, 0.0),
                NodeSpec::uncontrolled(4, 0.0, -40.0),
                NodeSpec::uncontrolled(5, 0.0, 200.0),
            ],
            links: vec![
                LinkSpec::oneway(1, 2, 1, 25.0), // major approach (long)
                LinkSpec::oneway(2, 3, 1, 25.0), // major exit
                LinkSpec::oneway(4, 2, 1, 10.0), // minor approach (short)
                LinkSpec::oneway(2, 5, 1, 10.0), // minor exit (straight across)
            ],
        }
        .build();
        let mut world = NetWorld::new(net, cfg());
        world.spawn_routed(10, vec![LinkId(0), LinkId(1)], 20.0, DriverConfig::car()); // major through
        world.spawn_routed(20, vec![LinkId(2), LinkId(3)], 9.0, DriverConfig::car()); // minor straight across

        // Record the tick each vehicle first enters the intersection interior.
        let (mut major_enter, mut minor_enter) = (None, None);
        for t in 0..400 {
            world.step();
            if major_enter.is_none() && world.vehicle(10).is_some_and(|v| v.is_crossing()) {
                major_enter = Some(t);
            }
            if minor_enter.is_none() && world.vehicle(20).is_some_and(|v| v.is_crossing()) {
                minor_enter = Some(t);
            }
        }
        let (mj, mn) = (major_enter.expect("major crosses"), minor_enter.expect("minor crosses"));
        assert!(mn > mj, "minor arrived first but must yield: minor@{mn} major@{mj}");
        assert_eq!(world.crashed(), 0, "yielding avoids a collision");
    }

    #[test]
    fn two_lanes_zipper_merge_without_colliding() {
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -100.0, 50.0),
                NodeSpec::uncontrolled(2, -100.0, -50.0),
                NodeSpec::uncontrolled(3, 0.0, 0.0), // merge point
                NodeSpec::uncontrolled(4, 150.0, 0.0),
            ],
            links: vec![
                LinkSpec::oneway(1, 3, 1, 15.0),
                LinkSpec::oneway(2, 3, 1, 15.0),
                LinkSpec::oneway(3, 4, 1, 15.0),
            ],
        }
        .build();
        let branch_a = net.lanes_of(LinkId(0)).next().unwrap();
        let branch_b = net.lanes_of(LinkId(1)).next().unwrap();
        let mut world = NetWorld::new(net, cfg());
        world.spawn(1, branch_a, 10.0, 10.0, DriverConfig::car()); // slightly ahead
        world.spawn(2, branch_b, 0.0, 10.0, DriverConfig::car());

        let mut min_gap = f64::MAX;
        for _ in 0..300 {
            world.step();
            let mut by_lane: std::collections::HashMap<u32, Vec<f64>> = std::collections::HashMap::new();
            for v in world.vehicles() {
                by_lane.entry(v.lane.0).or_default().push(v.position);
            }
            for positions in by_lane.values_mut() {
                positions.sort_by(f64::total_cmp);
                for w in positions.windows(2) {
                    min_gap = min_gap.min(w[1] - w[0]);
                }
            }
        }
        assert!(min_gap > 2.0, "vehicles overlapped at the merge: min gap {min_gap}");
        assert_eq!(world.exited(), 2, "both should clear the merge");
    }

    fn straight_link(length: f64) -> Network {
        OsmMap {
            nodes: vec![NodeSpec::uncontrolled(1, 0.0, 0.0), NodeSpec::uncontrolled(2, length, 0.0)],
            links: vec![LinkSpec::oneway(1, 2, 1, 40.0)],
        }
        .build()
    }

    /// A full uncontrolled four-way with every in/out leg, each `arm` metres.
    /// No signal or priority control, so conflicting movements are only kept
    /// apart by timing — the setup for the collision-model tests. Link ids:
    /// 0:W→C 1:E→C 2:S→C 3:N→C 4:C→W 5:C→E 6:C→S 7:C→N.
    fn uncontrolled_cross(arm: f64) -> Network {
        OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(0, 0.0, 0.0),  // C
                NodeSpec::uncontrolled(1, -arm, 0.0), // W
                NodeSpec::uncontrolled(2, arm, 0.0),  // E
                NodeSpec::uncontrolled(3, 0.0, -arm), // S
                NodeSpec::uncontrolled(4, 0.0, arm),  // N
            ],
            links: vec![
                LinkSpec::oneway(1, 0, 1, 15.0), // 0: W→C
                LinkSpec::oneway(2, 0, 1, 15.0), // 1: E→C
                LinkSpec::oneway(3, 0, 1, 15.0), // 2: S→C
                LinkSpec::oneway(4, 0, 1, 15.0), // 3: N→C
                LinkSpec::oneway(0, 1, 1, 15.0), // 4: C→W
                LinkSpec::oneway(0, 2, 1, 15.0), // 5: C→E
                LinkSpec::oneway(0, 3, 1, 15.0), // 6: C→S
                LinkSpec::oneway(0, 4, 1, 15.0), // 7: C→N
            ],
        }
        .build()
    }

    // Opposing left turns (W→N and E→S) share the intersection's diagonal, so
    // vehicles entering together meet head-on at the centre.
    const LEFT_W_TO_N: [LinkId; 2] = [LinkId(0), LinkId(7)];
    const LEFT_E_TO_S: [LinkId; 2] = [LinkId(1), LinkId(6)];

    #[test]
    fn conflicting_crossers_yield_to_avoid_a_collision() {
        // Two conflicting movements arriving together at an uncontrolled node: the
        // crash-avoidant behaviour makes one yield inside the box, so both clear
        // without colliding.
        let mut w = NetWorld::new(uncontrolled_cross(120.0), cfg());
        let d = DriverConfig { accel_noise: 0.0, ..DriverConfig::car() };
        assert!(w.spawn_routed(1, LEFT_W_TO_N.to_vec(), 12.0, d.clone()));
        assert!(w.spawn_routed(2, LEFT_E_TO_S.to_vec(), 12.0, d));
        for _ in 0..300 {
            w.step();
        }
        assert_eq!(w.crashed(), 0, "avoidance prevents the collision");
        assert_eq!(w.exited(), 2, "and both vehicles clear the intersection");
    }

    #[test]
    fn staggered_crossings_do_not_collide() {
        // The same conflict, but the second vehicle arrives after the first has
        // cleared the intersection — no collision.
        let mut w = NetWorld::new(uncontrolled_cross(100.0), cfg());
        let d = DriverConfig { accel_noise: 0.0, ..DriverConfig::car() };
        assert!(w.spawn_routed(1, LEFT_W_TO_N.to_vec(), 12.0, d.clone()));
        for _ in 0..40 {
            w.step();
        }
        assert!(w.spawn_routed(2, LEFT_E_TO_S.to_vec(), 12.0, d));
        for _ in 0..200 {
            w.step();
        }
        assert_eq!(w.crashed(), 0, "cleanly separated crossings never collide");
    }

    #[test]
    fn crossing_a_node_takes_several_ticks() {
        // A vehicle is inside the intersection (is_crossing) for more than one
        // tick — the interior is traversed over time, not teleported.
        let mut w = NetWorld::new(uncontrolled_cross(60.0), cfg());
        let d = DriverConfig { accel_noise: 0.0, ..DriverConfig::car() };
        w.spawn_routed(1, LEFT_W_TO_N.to_vec(), 12.0, d);
        let mut crossing_ticks = 0;
        let mut reached_exit = false;
        for _ in 0..200 {
            w.step();
            if w.vehicle(1).is_some_and(|v| v.is_crossing()) {
                crossing_ticks += 1;
            }
            if on_link(&w, 1, LinkId(7)) {
                reached_exit = true;
            }
        }
        assert!(crossing_ticks > 1, "spent multiple ticks inside the node: {crossing_ticks}");
        assert!(reached_exit, "and completed onto the departure link");
    }

    fn on_link(w: &NetWorld, id: u32, link: LinkId) -> bool {
        w.vehicle(id).is_some_and(|v| !v.is_crossing() && w.network.lane(v.lane).link == link)
    }

    #[test]
    fn arterial_intersection_flows_and_stays_safe_under_mixed_turning_demand() {
        // Millbrae-complexity: a two-way multi-lane signalized crossing carrying
        // through traffic on every approach plus left turns. It must keep flowing
        // (throughput scales with time) and stay collision-free.
        let mut w = NetWorld::new(super::super::map::arterial_intersection(), cfg());
        // link ids (see arterial_intersection): approaches W=0, E=3, S=4, N=7.
        let throughs = [
            [LinkId(0), LinkId(2)], // W→E
            [LinkId(3), LinkId(1)], // E→W
            [LinkId(4), LinkId(6)], // S→N
            [LinkId(7), LinkId(5)], // N→S
        ];
        let lefts = [
            [LinkId(0), LinkId(6)], // W→N
            [LinkId(3), LinkId(5)], // E→S
        ];
        let mut next = 0u32;
        let mut saw_crossing = false;
        for t in 0..2500u32 {
            if t % 22 == 0 {
                for r in &throughs {
                    if w.spawn_routed(next, r.to_vec(), 12.0, DriverConfig::car()) {
                        next += 1;
                    }
                }
            }
            if t % 55 == 0 {
                for r in &lefts {
                    if w.spawn_routed(next, r.to_vec(), 12.0, DriverConfig::car()) {
                        next += 1;
                    }
                }
            }
            w.step();
            saw_crossing |= w.vehicles().iter().any(|v| v.is_crossing());
        }
        assert!(next > 100, "the intersection should be busy: {next} spawned");
        assert!(w.exited() > 30, "traffic should keep clearing the intersection: {} exited", w.exited());
        assert!(saw_crossing, "vehicles actually traverse the intersection interior");
        // Crash-free over 130+ turning vehicles: signal phasing separates the
        // conflicting movements, and the crossing check distinguishes a genuine
        // T-bone from opposing lefts that merely pass close (anti-parallel).
        assert_eq!(w.crashed(), 0, "a signalized arterial stays crash-free, got {}", w.crashed());
    }

    #[test]
    fn real_map_render_signals_show_red_and_green() {
        // The renderer reads NetWorld::signal_states() (the actuated runtime, not
        // the pure program). Under real traffic those must include reds, or every
        // head would look green.
        use super::super::demand::{DemandGenerator, OdPair};
        use super::super::signal::SignalState;
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
        let Ok(text) = std::fs::read_to_string(path) else { return };
        let net = super::super::map::OsmMap::from_json(&text).expect("map json").build();
        if net.programs.is_empty() {
            return;
        }
        let mut w = NetWorld::new(net, cfg());
        // A small fixed set of long routes through the network (capped, so the
        // test doesn't route every O/D pair on a 400-link map).
        let n = w.network.links.len() as u32;
        let mut pairs = Vec::new();
        for o in (0..n).step_by(7) {
            for d in (0..n).step_by(11) {
                if o != d && w.network.route_links(LinkId(o), LinkId(d)).is_some_and(|r| r.len() >= 5) {
                    pairs.push(OdPair { origin: LinkId(o), dest: LinkId(d), rate_per_sec: 0.3 });
                    if pairs.len() >= 20 {
                        break;
                    }
                }
            }
            if pairs.len() >= 20 {
                break;
            }
        }
        let mut d = DemandGenerator::new(&w, &pairs, 1);
        let (mut reds, mut greens) = (0u64, 0u64);
        for _ in 0..800 {
            d.step(&mut w, cfg().dt);
            w.step();
            for s in w.signal_states() {
                match s {
                    SignalState::Red => reds += 1,
                    SignalState::Green => greens += 1,
                    _ => {}
                }
            }
        }
        let multiphase = w.network.programs.iter().filter(|p| p.phases.len() > 1).count();
        assert!(greens > 0, "some signals go green");
        assert!(reds > 0, "some signals go red (rendered heads aren't all green)");
        // Per-approach OSM signals are relocated onto their junctions and split
        // junctions are merged into one, so a smaller set of real intersections
        // cycle rather than sitting permanently green.
        assert!(multiphase >= 8, "most signalized intersections should cycle, got {multiphase}");
    }

    #[test]
    fn a_car_brakes_early_for_a_red_at_the_next_intersection() {
        // W →275m→ A →25m→ X(signalised 4-way) → E, with a cross street kept busy so
        // X's through stays red. A through car should already be braking while still
        // on the long W→A link — an earlier subsection — for the red one hop ahead.
        let plan = SignalPlan { green_secs: 10.0, yellow_secs: 3.0, offset: 0.0 };
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -300.0, 0.0), // W
                NodeSpec::uncontrolled(2, -25.0, 0.0),  // A
                NodeSpec::signalized(3, 0.0, 0.0, plan),// X
                NodeSpec::uncontrolled(4, 200.0, 0.0),  // E
                NodeSpec::uncontrolled(5, 0.0, -150.0), // S
                NodeSpec::uncontrolled(6, 0.0, 150.0),  // N
            ],
            links: vec![
                LinkSpec::oneway(1, 2, 1, 20.0), // 0: W→A
                LinkSpec::oneway(2, 3, 1, 20.0), // 1: A→X (short)
                LinkSpec::oneway(3, 4, 1, 20.0), // 2: X→E
                LinkSpec::oneway(5, 3, 1, 20.0), // 3: S→X (cross)
                LinkSpec::oneway(3, 6, 1, 20.0), // 4: X→N (cross)
            ],
        }
        .build();
        let mut w = NetWorld::new(net, cfg());
        w.install_router(&[LinkId(2), LinkId(4)]);
        let lane1 = w.network.lanes_of(LinkId(1)).next().unwrap();
        let through = w.movement_to(lane1, LinkId(2)).expect("X has a through movement"); // A→X→E

        let l0 = w.network.lane(w.network.lanes_of(LinkId(0)).next().unwrap()).length;
        assert!(w.spawn_to(1, LinkId(0), LinkId(2), 16.0, DriverConfig::car()));
        let mut braked_early = false;
        let mut next = 100u32;
        for t in 0..900 {
            if t % 12 == 0 {
                w.spawn_to(next, LinkId(3), LinkId(4), 12.0, DriverConfig::car()); // keep the cross busy
                next += 1;
            }
            w.step();
            if let Some(v) = w.vehicle(1) {
                let on_w_a = v.lane == w.network.lanes_of(LinkId(0)).next().unwrap() && v.position < l0 - 1.0;
                if on_w_a && w.movement_state(through) == SignalState::Red && v.speed < 10.0 {
                    braked_early = true;
                }
            }
        }
        assert!(braked_early, "the car should slow on the earlier W→A link for the red one intersection ahead");
        assert_eq!(w.crashed(), 0);
    }

    #[test]
    fn a_vehicle_holds_at_the_line_when_the_box_exit_is_blocked() {
        let map = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(0, 0.0, 0.0),
                NodeSpec::uncontrolled(1, 100.0, 0.0),
                NodeSpec::uncontrolled(2, 200.0, 0.0),
            ],
            links: vec![LinkSpec::oneway(0, 1, 1, 20.0), LinkSpec::oneway(1, 2, 1, 20.0)],
        };
        let net = map.build();
        let lane0 = net.lanes_of(LinkId(0)).next().unwrap();
        let lane1 = net.lanes_of(LinkId(1)).next().unwrap();
        let line = net.lane(lane0).length;

        let mut blocked = NetWorld::new(net.clone(), cfg());
        blocked.spawn(2, lane1, 1.0, 0.0, DriverConfig::car());
        blocked.spawn(1, lane0, line, 3.0, DriverConfig::car());
        blocked.step();
        assert!(!blocked.vehicle(1).unwrap().is_crossing(), "must not enter the box while the exit is blocked");
        assert_eq!(blocked.crashed(), 0);

        let mut clear = NetWorld::new(net, cfg());
        clear.spawn(1, lane0, line, 3.0, DriverConfig::car());
        clear.step();
        assert!(clear.vehicle(1).unwrap().is_crossing(), "with a clear exit it enters the box");
    }

    #[test]
    fn real_map_boundary_demand_routes_live_and_stays_safe() {
        use super::super::demand::{self, DemandGenerator};
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
        let Ok(text) = std::fs::read_to_string(path) else { return };
        let net = super::super::map::OsmMap::from_json(&text).expect("map json").build();
        assert!(!boundary::gateways(&net).is_empty(), "a bbox-clipped city map has edge gateways");

        for seed in 0..4u64 {
            let mut world = NetWorld::new(net.clone(), cfg());
            let pairs = demand::boundary_od_pairs(&world.network, seed, 16);
            assert!(!pairs.is_empty(), "boundary categories yield demand on the real map");
            let mut gen = DemandGenerator::new(&world, &pairs, seed);
            world.install_router(&gen.destinations());

            for _ in 0..1500 {
                gen.step(&mut world, cfg().dt);
                world.step();
            }
            assert_eq!(world.crashed(), 0, "flow-field-routed traffic stays collision-free (seed {seed})");
            assert!(world.exited() > 0, "vehicles complete boundary trips (seed {seed}), got {}", world.exited());
            assert_eq!(world.leaked(), 0, "no car disappears at an intersection (seed {seed}), leaked {}", world.leaked());
        }
    }

    #[test]
    fn real_map_highway_mode_originates_traffic_on_the_freeways() {
        use super::super::demand::{self, DemandGenerator, DemandSources};
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
        let Ok(text) = std::fs::read_to_string(path) else { return };
        let net = super::super::map::OsmMap::from_json(&text).expect("map json").build();
        // The expanded Millbrae map includes US-101 and I-280, so it has freeway
        // gateways for highway mode to anchor on.
        assert!(!boundary::highway_entry_links(&net).is_empty(), "the map has freeway gateways (101/280)");

        let seed = 2u64;
        let pairs = demand::od_pairs(&net, seed, 40, DemandSources::new(true, false));
        let hw = pairs.iter().filter(|p| boundary::is_highway_link(&net, p.origin)).count();
        assert!(hw * 2 > pairs.len(), "most trips originate on a freeway: {hw} of {}", pairs.len());

        // Every freeway trip: enters from outside (a highway gateway); ends at a highway
        // exit or a surface street, never on a mid-freeway segment; and a meaningful
        // share run the *same* freeway end-to-end (matched by OSM `ref`, US-101/I-280).
        use std::collections::HashSet;
        let entries: HashSet<u32> = boundary::highway_entry_links(&net).iter().map(|l| l.0).collect();
        let hw_exit: HashSet<u32> = boundary::highway_exit_links(&net).iter().map(|l| l.0).collect();
        let surface_int: HashSet<u32> = boundary::surface_interior_links(&net).iter().map(|l| l.0).collect();
        let mid_freeway: HashSet<u32> =
            boundary::interior_links(&net).iter().filter(|&&l| boundary::is_highway_link(&net, l)).map(|l| l.0).collect();
        let mut same_hw = 0;
        for p in &pairs {
            assert!(entries.contains(&p.origin.0), "trip {p:?} must enter at a freeway gateway (from outside)");
            assert!(!mid_freeway.contains(&p.dest.0), "no destination on a mid-freeway segment: {p:?}");
            assert!(hw_exit.contains(&p.dest.0) || surface_int.contains(&p.dest.0), "dest is a highway exit or surface street: {p:?}");
            let (ro, rd) = (net.link_ref(p.origin), net.link_ref(p.dest));
            if hw_exit.contains(&p.dest.0) && !ro.is_empty() && ro.split(';').any(|t| rd.split(';').any(|u| u == t)) {
                same_hw += 1;
            }
        }
        assert!(same_hw > 0, "some freeway trips run the same highway end-to-end (ref-matched): {same_hw}/{}", pairs.len());

        let mut world = NetWorld::new(net, cfg());
        let mut gen = DemandGenerator::new(&world, &pairs, seed);
        world.install_router(&gen.destinations());
        for _ in 0..1500 {
            gen.step(&mut world, cfg().dt);
            world.step();
        }
        assert_eq!(world.crashed(), 0, "highway-mode traffic stays collision-free");
        assert!(world.exited() > 0, "highway-mode trips complete, got {}", world.exited());
        assert_eq!(world.leaked(), 0, "no car disappears at an intersection, leaked {}", world.leaked());
    }

    #[test]
    fn real_map_freeway_interchanges_are_free_flowing() {
        // The scraped Millbrae map carries OSM road classes, so US-101 / I-280
        // ramps form free-flow interchange nodes (no stop box). Under highway
        // demand, cars crossing those interchange movements keep real speed instead
        // of crawling as if through an intersection.
        use super::super::demand::{self, DemandGenerator, DemandSources};
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
        let Ok(text) = std::fs::read_to_string(path) else { return };
        let net = super::super::map::OsmMap::from_json(&text).expect("map json").build();

        let interchange_nodes = (0..net.nodes.len() as u32).filter(|&n| net.is_interchange_node(NodeId(n))).count();
        let interchange_movs = (0..net.movements.len() as u32).filter(|&m| net.is_interchange_movement(MovementId(m))).count();
        assert!(interchange_nodes > 0, "the map has freeway interchange nodes from OSM road classes");
        assert!(interchange_movs > 0, "and free-flow interchange movements");
        // Interchange nodes shed the intersection stop box.
        assert!(
            (0..net.nodes.len() as u32).all(|n| !net.is_interchange_node(NodeId(n)) || net.render_setback[n as usize] <= 0.5),
            "interchange nodes carry no stop box"
        );

        let seed = 7u64;
        let pairs = demand::od_pairs(&net, seed, 48, DemandSources::new(true, false));
        let mut world = NetWorld::new(net, cfg());
        let mut gen = DemandGenerator::new(&world, &pairs, seed);
        world.install_router(&gen.destinations());
        // Count interchange crossings that clear the old 5 m/s turn cap — under the
        // previous model *no* interchange crossing could, since a ramp diverge was
        // throttled like a right turn. (Mean speed is confounded by realistic queues
        // at the ramp termini, so we assert the free-flow crossings now exist.)
        let (mut fast, mut total, mut peak) = (0u32, 0u32, 0.0f64);
        for _ in 0..2000 {
            gen.step(&mut world, cfg().dt);
            world.step();
            for v in world.vehicles() {
                if let Some(c) = v.crossing {
                    if world.network.is_interchange_movement(c.movement) {
                        total += 1;
                        peak = peak.max(v.speed);
                        if v.speed > 8.0 {
                            fast += 1;
                        }
                    }
                }
            }
        }
        assert!(total > 0, "cars actually traverse freeway interchanges");
        assert!(fast > 0, "interchange crossings now exceed the old 5 m/s turn crawl (peak {peak:.1} m/s)");
        assert!(peak > 12.0, "freeway diverges run at highway speed, peak {peak:.1} m/s");
        assert_eq!(world.crashed(), 0, "freeway interchanges stay collision-free");
        assert_eq!(world.leaked(), 0, "no car disappears at an interchange, leaked {}", world.leaked());
    }

    #[test]
    fn a_car_diverging_onto_a_ramp_keeps_highway_speed() {
        // A freeway (29 m/s) that continues straight and sheds a right-diverging
        // off-ramp. Because both sides are grade-separated, the diverge is a
        // free-flow interchange, not an at-grade turn: a car peeling onto the ramp
        // holds highway speed instead of crawling through a 5 m/s "intersection".
        let hw = |a, b, lanes, sp| LinkSpec { road_class: "motorway".into(), ..LinkSpec::oneway(a, b, lanes, sp) };
        let ramp = |a, b, lanes, sp| LinkSpec { road_class: "motorway_link".into(), ..LinkSpec::oneway(a, b, lanes, sp) };
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -400.0, 0.0),   // freeway in
                NodeSpec::uncontrolled(2, 0.0, 0.0),      // diverge point
                NodeSpec::uncontrolled(3, 400.0, 0.0),    // freeway continues
                NodeSpec::uncontrolled(4, 300.0, -260.0), // off-ramp (a real right turn angle)
            ],
            links: vec![hw(1, 2, 3, 29.0), hw(2, 3, 3, 29.0), ramp(2, 4, 1, 25.0)],
        }
        .build();
        // The diverge point carries no cross traffic and no stop box.
        assert!(net.is_interchange_node(NodeId(1)), "the diverge is a pure interchange node");
        assert!(net.render_setback[1] <= 0.5, "no intersection-like box at the diverge");
        // The off-ramp is a genuine right turn (would otherwise be capped to 5 m/s).
        let ramp_mv = (0..net.movements.len() as u32)
            .map(MovementId)
            .find(|&m| net.lane(net.movement(m).to_lane).link == LinkId(2))
            .expect("a freeway→ramp movement exists");
        assert_eq!(net.movement_turn(ramp_mv), TurnType::Right, "the ramp is a right diverge");
        assert!(net.is_interchange_movement(ramp_mv), "and it is a free-flow interchange");

        let mut w = NetWorld::new(net, cfg());
        w.install_router(&[LinkId(2)]); // destination: the off-ramp
        // Place the exiting car in the curb lane (the off-ramp is curb-side only),
        // as a driver bound for the exit would already have merged right.
        let curb = w.network.lanes_of(LinkId(0)).last().unwrap();
        w.spawn_to_in_lane(1, curb, 5.0, LinkId(2), 26.0, DriverConfig::car());
        let mut max_speed_crossing: f64 = 0.0;
        let mut reached_ramp = false;
        for _ in 0..400 {
            w.step();
            if let Some(v) = w.vehicle(1) {
                if v.is_crossing() {
                    max_speed_crossing = max_speed_crossing.max(v.speed);
                }
            }
            reached_ramp |= w.link_flows()[2] > 0.0;
        }
        assert!(reached_ramp, "the car takes the off-ramp");
        assert!(max_speed_crossing > 15.0, "it keeps highway speed through the diverge, got {max_speed_crossing:.1} m/s");
        assert_eq!(w.crashed(), 0);
    }

    #[test]
    fn high_load_real_map_stays_bounded_and_flowing() {
        // Browser-scale stress: the scraped map under saturating demand must stay
        // *stable* — vehicle count bounded (self-limiting spawns, no runaway),
        // throughput continues, and collisions stay a small fraction of completed
        // trips (the engine degrades gracefully, it doesn't melt down or pile up).
        use super::super::demand::{self, DemandGenerator};
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
        let Ok(text) = std::fs::read_to_string(path) else { return };
        let net = super::super::map::OsmMap::from_json(&text).expect("map json").build();

        let mut world = NetWorld::new(net, cfg());
        let pairs = demand::boundary_od_pairs(&world.network, 1, 64); // saturating
        let mut gen = DemandGenerator::new(&world, &pairs, 1);
        world.install_router(&gen.destinations());

        let mut peak = 0usize;
        for tick in 0..3000 {
            gen.step(&mut world, cfg().dt);
            world.step();
            peak = peak.max(world.vehicles().len());
            assert!(world.vehicles().len() < 8000, "runaway vehicle count at tick {tick}");
        }
        assert!(world.exited() > 200, "sustained throughput under saturation, exited {}", world.exited());
        assert!(peak > 100, "the map actually loads up under stress, peaked at {peak}");
        // Collisions, if any, are a small fraction of completed trips — no pileup.
        assert!(world.crashed() * 8 < world.exited(), "crashes stay a small fraction of trips: {} crashed vs {} exited", world.crashed(), world.exited());
        assert_eq!(world.leaked(), 0, "no car disappears at an intersection under saturation, leaked {}", world.leaked());
    }

    #[test]
    fn high_load_signalized_grid_saturates_gracefully() {
        // A self-contained (no map.json) browser-scale stress: a 4×4 signalised grid
        // driven into saturation. A dense grid gridlocks under heavy demand (a real
        // phenomenon) — the point is that it does so *gracefully*: vehicle count
        // stays bounded (spawns self-limit at blocked entrances, no runaway), some
        // traffic still clears, and there's no collision pile-up.
        use super::super::demand::{self, DemandGenerator};
        let (rows, cols, d) = (4usize, 4usize, 220.0);
        let plan = SignalPlan { green_secs: 12.0, yellow_secs: 3.0, offset: 0.0 };
        let id = |r: usize, c: usize| (r * cols + c) as i64;
        let mut nodes = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let (x, y) = (c as f64 * d, r as f64 * d);
                let interior = r > 0 && r < rows - 1 && c > 0 && c < cols - 1;
                nodes.push(if interior {
                    NodeSpec::signalized(id(r, c), x, y, plan)
                } else {
                    NodeSpec::uncontrolled(id(r, c), x, y)
                });
            }
        }
        let mut links = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                if c + 1 < cols {
                    links.extend(LinkSpec::twoway(id(r, c), id(r, c + 1), 1, 15.0));
                }
                if r + 1 < rows {
                    links.extend(LinkSpec::twoway(id(r, c), id(r + 1, c), 1, 15.0));
                }
            }
        }
        let net = OsmMap { nodes, links }.build();
        let mut world = NetWorld::new(net, cfg());
        let pairs = demand::boundary_od_pairs(&world.network, 3, 18);
        assert!(!pairs.is_empty());
        let mut gen = DemandGenerator::new(&world, &pairs, 3);
        world.install_router(&gen.destinations());
        let mut peak = 0usize;
        for _ in 0..2500 {
            gen.step(&mut world, cfg().dt);
            world.step();
            peak = peak.max(world.vehicles().len());
            assert!(world.vehicles().len() < 3000, "bounded under load");
        }
        assert!(peak > 20, "the grid loads up under demand, peaked at {peak}");
        assert!(world.exited() > 10, "some traffic still clears under saturation, exited {}", world.exited());
        assert!(world.crashed() * 4 < world.exited() + 20, "no collision pile-up: {} crashed vs {} exited", world.crashed(), world.exited());
    }

    #[test]
    fn a_signalized_crossing_stays_collision_free_under_demand() {
        // Two conflicting through streams under sustained demand through a
        // signalized four-way: conflict-derived phasing plus all-red clearance
        // must keep it crash-free while traffic flows.
        let plan = SignalPlan { green_secs: 12.0, yellow_secs: 3.0, offset: 0.0 };
        let net = OsmMap {
            nodes: vec![
                NodeSpec::signalized(0, 0.0, 0.0, plan),
                NodeSpec::uncontrolled(1, -250.0, 0.0),
                NodeSpec::uncontrolled(2, 250.0, 0.0),
                NodeSpec::uncontrolled(3, 0.0, -250.0),
                NodeSpec::uncontrolled(4, 0.0, 250.0),
            ],
            links: vec![
                LinkSpec::oneway(1, 0, 1, 15.0), // 0: W→C
                LinkSpec::oneway(3, 0, 1, 15.0), // 1: S→C
                LinkSpec::oneway(0, 2, 1, 15.0), // 2: C→E
                LinkSpec::oneway(0, 4, 1, 15.0), // 3: C→N
            ],
        }
        .build();
        let mut w = NetWorld::new(net, cfg());
        let mut next = 0u32;
        for t in 0..1500u32 {
            if t % 10 == 0 {
                let d = DriverConfig::car();
                if w.spawn_routed(next, vec![LinkId(0), LinkId(2)], 13.0, d.clone()) {
                    next += 1;
                }
                if w.spawn_routed(next, vec![LinkId(1), LinkId(3)], 13.0, d) {
                    next += 1;
                }
            }
            w.step();
        }
        assert!(w.exited() > 20, "traffic should flow through the signal: {} exited", w.exited());
        assert_eq!(w.crashed(), 0, "signal phasing + all-red keep it crash-free, got {}", w.crashed());
    }

    #[test]
    fn priority_yielding_keeps_an_uncontrolled_crossing_mostly_safe() {
        // A major E–W road (faster, priority) crosses a minor N–S road at an
        // uncontrolled node, both under sustained demand. Right-of-way yielding
        // should keep collisions rare, not constant — the fix for pervasive
        // intersection crashing on the real map.
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(0, 0.0, 0.0),
                NodeSpec::uncontrolled(1, -250.0, 0.0),
                NodeSpec::uncontrolled(2, 250.0, 0.0),
                NodeSpec::uncontrolled(3, 0.0, -250.0),
                NodeSpec::uncontrolled(4, 0.0, 250.0),
            ],
            links: vec![
                LinkSpec::oneway(1, 0, 1, 22.0), // 0: major W→C
                LinkSpec::oneway(0, 2, 1, 22.0), // 1: major C→E
                LinkSpec::oneway(3, 0, 1, 10.0), // 2: minor S→C
                LinkSpec::oneway(0, 4, 1, 10.0), // 3: minor C→N
            ],
        }
        .build();
        let mut w = NetWorld::new(net, cfg());
        let mut next = 0u32;
        for t in 0..1200u32 {
            if t % 12 == 0 {
                let d = DriverConfig::car();
                if w.spawn_routed(next, vec![LinkId(0), LinkId(1)], 20.0, d.clone()) {
                    next += 1;
                }
                if w.spawn_routed(next, vec![LinkId(2), LinkId(3)], 9.0, d) {
                    next += 1;
                }
            }
            w.step();
        }
        assert!(w.exited() > 20, "traffic should be flowing: {} exited", w.exited());
        assert!(w.crashed() <= 2, "priority yielding should make crashes rare, got {}", w.crashed());
    }

    #[test]
    fn reaction_delay_causes_start_up_lag() {
        // A follower behind a leader that accelerates away travels less over the
        // same window when it has a reaction delay (it's slow to notice the gap
        // opening) — realistic start-up lost time — and never crashes.
        let distance_travelled = |reaction: f64| {
            let mut w = NetWorld::new(straight_link(6000.0), cfg());
            let base = DriverConfig { accel_noise: 0.0, reaction_time: reaction, desired_speed: 25.0, ..DriverConfig::car() };
            w.spawn(1, LaneId(0), 20.0, 0.0, base); // leader just ahead, from rest
            w.spawn(2, LaneId(0), 0.0, 0.0, base); // follower from rest
            for _ in 0..200 {
                w.step();
            }
            (w.vehicle(2).map(|v| v.position).unwrap_or(0.0), w.vehicle(1).is_some() && w.vehicle(2).is_some())
        };
        let (delayed, ok_d) = distance_travelled(0.8);
        let (instant, ok_i) = distance_travelled(0.0);
        assert!(ok_d && ok_i, "neither should crash");
        assert!(delayed > 0.0 && instant > 0.0);
        assert!(delayed < instant, "reaction delay should lag start-up: {delayed} vs {instant}");
    }

    #[test]
    fn acceleration_noise_keeps_speed_just_below_desired() {
        let steady = |noise: f64| {
            let mut w = NetWorld::new(straight_link(20000.0), cfg());
            let d = DriverConfig { desired_speed: 20.0, accel_noise: noise, reaction_time: 0.0, ..DriverConfig::car() };
            w.spawn(1, LaneId(0), 0.0, 20.0, d);
            let (mut sum, mut n) = (0.0, 0);
            for t in 0..1000 {
                w.step();
                if t >= 800 {
                    if let Some(v) = w.vehicle(1) {
                        sum += v.speed;
                        n += 1;
                    }
                }
            }
            sum / n as f64
        };
        let without = steady(0.0);
        let with = steady(0.4);
        assert!((without - 20.0).abs() < 0.1, "noiseless reaches desired: {without}");
        assert!(with < without - 0.1 && with > 15.0, "noise slows a little: {with}");
    }

    #[test]
    fn overlapping_vehicles_crash_and_leave_the_road() {
        let mut w = NetWorld::new(straight_link(1000.0), cfg());
        w.spawn(1, LaneId(0), 100.0, 5.0, DriverConfig::car());
        w.spawn(2, LaneId(0), 99.0, 5.0, DriverConfig::car()); // deep overlap into the leader
        w.step();
        assert_eq!(w.crashed(), 2);
        assert_eq!(w.vehicles().len(), 0);
    }

    #[test]
    fn crossing_overlap_flags_t_bones_not_close_passes() {
        use std::f64::consts::{FRAC_PI_2, PI};
        let origin = [0.0, 0.0];
        let near = [0.5, 0.5]; // well within OVERLAP (2.4 m)
        // A near-perpendicular meeting is a genuine T-bone.
        assert!(crossing_overlap(origin, 0.0, near, FRAC_PI_2), "perpendicular crossing collides");
        // Two anti-parallel bodies (opposing protected lefts passing to opposite
        // corners) are passing, not colliding, even when they momentarily overlap.
        assert!(!crossing_overlap(origin, 0.0, near, PI), "anti-parallel pass does not collide");
        // Same-heading (parallel) bodies are following/side-by-side, not a crossing.
        assert!(!crossing_overlap(origin, 0.3, near, 0.3), "parallel run is not a crossing");
        // A true crossing that is still far apart does not collide.
        assert!(!crossing_overlap(origin, 0.0, [5.0, 5.0], FRAC_PI_2), "distant crossing does not collide");
    }

    #[test]
    fn conflict_matrix_rules() {
        let (east, west, north) = ([1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]);
        // A right turn merges rather than crosses → never yields at the node.
        assert!(!should_yield_to(TurnType::Right, east, TurnType::Through, north, 0, 999));
        // Opposing through movements don't conflict.
        assert!(!should_yield_to(TurnType::Through, east, TurnType::Through, west, 0, 999));
        // A left turn yields to the oncoming through.
        assert!(should_yield_to(TurnType::Left, east, TurnType::Through, west, 999, 0));
        // Crossing streams defer to the higher-priority approach.
        assert!(should_yield_to(TurnType::Through, east, TurnType::Through, north, 0, 999));
        assert!(!should_yield_to(TurnType::Through, east, TurnType::Through, north, 999, 0));
    }

    #[test]
    fn turning_vehicles_slow_through_the_intersection() {
        // Route forces a left turn (east approach → north exit) at node 2.
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -200.0, 0.0),
                NodeSpec::uncontrolled(2, 0.0, 0.0),
                NodeSpec::uncontrolled(3, 0.0, 200.0),
            ],
            links: vec![LinkSpec::oneway(1, 2, 1, 25.0), LinkSpec::oneway(2, 3, 1, 25.0)],
        }
        .build();
        let approach = net.lanes_of(LinkId(0)).next().unwrap();
        let app_len = net.lane(approach).length;
        let mut w = NetWorld::new(net, cfg());
        w.spawn_routed(1, vec![LinkId(0), LinkId(1)], 22.0, DriverConfig { accel_noise: 0.0, ..DriverConfig::car() });

        let mut speed_at_line = f64::MAX;
        for _ in 0..200 {
            w.step();
            if let Some(v) = w.vehicle(1) {
                if v.lane == approach && v.position > app_len - 10.0 {
                    speed_at_line = speed_at_line.min(v.speed);
                }
            }
        }
        assert!(speed_at_line < 12.0, "should slow for the left turn, speed {speed_at_line}");
    }

    #[test]
    fn vehicles_slow_for_a_curve() {
        // A long straight run-up into a ~20 m-radius bend (short arc segments).
        let net = OsmMap {
            nodes: vec![NodeSpec::uncontrolled(1, 0.0, 0.0), NodeSpec::uncontrolled(2, 220.0, 20.0)],
            links: vec![LinkSpec {
                from_osm: 1,
                to_osm: 2,
                lanes: 1,
                speed_limit: 30.0,
                geometry: vec![[200.0, 0.0], [202.68, 10.0], [210.0, 17.32]],
                layer: 0,
                name: String::new(),
                road_class: String::new(),
                highway_ref: String::new(),
            }],
        }
        .build();
        let mut w = NetWorld::new(net, cfg());
        w.spawn(1, LaneId(0), 0.0, 20.0, DriverConfig { accel_noise: 0.0, ..DriverConfig::car() });
        let mut min_speed_on_bend = f64::MAX;
        for _ in 0..300 {
            w.step();
            if let Some(v) = w.vehicle(1) {
                if v.position > 195.0 {
                    min_speed_on_bend = min_speed_on_bend.min(v.speed);
                }
            }
        }
        // Cruises the straight near the 30 m/s limit, then slows markedly for the bend.
        assert!(min_speed_on_bend < 16.0, "should slow into the bend, min {min_speed_on_bend}");
    }

    #[test]
    fn faster_vehicle_changes_lanes_to_overtake() {
        let net = OsmMap {
            nodes: vec![NodeSpec::uncontrolled(1, 0.0, 0.0), NodeSpec::uncontrolled(2, 6000.0, 0.0)],
            links: vec![LinkSpec::oneway(1, 2, 2, 30.0)],
        }
        .build();
        let lanes: Vec<LaneId> = net.lanes_of(LinkId(0)).collect();
        let mut w = NetWorld::new(net, cfg());
        let slow = DriverConfig { desired_speed: 8.0, accel_noise: 0.0, ..DriverConfig::car() };
        let fast = DriverConfig { desired_speed: 30.0, accel_noise: 0.0, ..DriverConfig::car() };
        w.spawn(1, lanes[0], 300.0, 8.0, slow); // slow leader
        w.spawn(2, lanes[0], 100.0, 20.0, fast); // fast follower, same lane

        let mut used_other_lane = false;
        for _ in 0..500 {
            w.step();
            if w.vehicle(2).is_some_and(|v| v.lane == lanes[1]) {
                used_other_lane = true;
            }
        }
        let f = w.vehicle(2).unwrap();
        let s = w.vehicle(1).unwrap();
        assert!(used_other_lane, "fast vehicle should use the adjacent lane");
        assert!(f.position > s.position, "and overtake the slow one: {} vs {}", f.position, s.position);
    }

    /// A busy scenario (mixed classes, signals, multi-lane, merges, curves-capable
    /// network) plus its demand, for property/invariant regressions.
    fn busy_scenario(seed: u64) -> (NetWorld, super::super::demand::DemandGenerator) {
        use super::super::demand::{DemandGenerator, OdPair};
        let net = super::super::map::millbrae_sample();
        let world = NetWorld::new(net, SimConfig { seed, ..cfg() });
        let n = world.network.links.len();
        let mut pairs = Vec::new();
        for o in 0..n {
            for d in 0..n {
                if o != d && world.network.route_links(LinkId(o as u32), LinkId(d as u32)).is_some_and(|r| r.len() >= 3) {
                    pairs.push(OdPair { origin: LinkId(o as u32), dest: LinkId(d as u32), rate_per_sec: 0.3 });
                }
            }
        }
        let demand = DemandGenerator::new(&world, &pairs, seed);
        (world, demand)
    }

    #[test]
    fn safety_invariants_hold_over_a_busy_run() {
        // Robust regression: across a long, busy, mixed-class run, at *every* tick
        // no vehicle reverses, none meaningfully exceeds the fastest speed limit,
        // and no two vehicles overlap on a lane. Catches a broad class of bugs
        // (e.g. the leader-length overlap) without brittle magic numbers.
        let (mut w, mut d) = busy_scenario(1);
        let max_limit = w.network.lanes.iter().map(|l| l.speed_limit).fold(0.0, f64::max);
        for _ in 0..1200 {
            d.step(&mut w, cfg().dt);
            w.step();
            let mut by_lane: HashMap<u32, Vec<(f64, f64)>> = HashMap::new();
            for v in w.vehicles() {
                assert!(v.speed >= -1e-6, "no reversing: {}", v.speed);
                assert!(v.speed <= max_limit + 1.5, "no gross speeding: {} > {}", v.speed, max_limit);
                if v.is_crossing() {
                    continue; // inside a node, not occupying the lane
                }
                by_lane.entry(v.lane.0).or_default().push((v.position, v.driver.vehicle_length));
            }
            for cars in by_lane.values_mut() {
                cars.sort_by(|a, b| a.0.total_cmp(&b.0));
                for w2 in cars.windows(2) {
                    let gap = w2[1].0 - w2[0].0 - w2[1].1;
                    assert!(gap > -0.6, "vehicles overlap on a lane: gap {gap}");
                }
            }
        }
        assert_eq!(w.crashed(), 0);
        assert!(d.spawned() > 100, "scenario should be busy");
    }

    #[test]
    fn runs_are_reproducible_from_the_seed() {
        // Same seed → identical aggregate outcome (a regression against accidental
        // nondeterminism creeping into the tick).
        let run = |seed: u64| {
            let (mut w, mut d) = busy_scenario(seed);
            for _ in 0..600 {
                d.step(&mut w, cfg().dt);
                w.step();
            }
            // A fingerprint of the whole live state — far more seed-sensitive than
            // the saturated vehicle count alone.
            let fp = w.vehicles().iter().fold(0u64, |h, v| {
                h.wrapping_mul(1000003).wrapping_add(v.id as u64).wrapping_add((v.position * 100.0) as u64)
            });
            (w.vehicles().len(), w.exited(), w.crashed(), fp)
        };
        assert_eq!(run(7), run(7), "identical seeds must reproduce");
        assert_ne!(run(7), run(8), "different seeds should differ");
    }

    #[test]
    fn congestion_slows_drivers_below_free_flow() {
        // Property regression for "driver slowness": under heavy demand the mean
        // speed settles well under the road's free-flow limit (queues, signals,
        // following all bite), rather than everyone cruising at the limit.
        let (mut w, mut d) = busy_scenario(3);
        for _ in 0..1500 {
            d.step(&mut w, cfg().dt);
            w.step();
        }
        let speeds: Vec<f64> = w.vehicles().iter().map(|v| v.speed).collect();
        let mean = speeds.iter().sum::<f64>() / speeds.len().max(1) as f64;
        let free_flow = w.network.lanes.iter().map(|l| l.speed_limit).fold(0.0, f64::max);
        assert!(mean < free_flow * 0.75, "congested mean speed {mean} should be well below free-flow {free_flow}");
    }

    #[test]
    fn mixed_class_traffic_does_not_crash_under_sustained_demand() {
        // Regression for the length-convention bug: a follower must reserve the
        // *leader's* length, and spawns/crossings must respect a long vehicle's
        // rear — otherwise cars overlap trucks/buses. Runs a busy grid with a
        // car/truck/bus mix and asserts nobody crashes.
        use super::super::demand::{DemandGenerator, OdPair};
        let net = super::super::map::millbrae_sample();
        let mut w = NetWorld::new(net, cfg());
        let n = w.network.links.len();
        let mut pairs = Vec::new();
        for o in 0..n {
            for d in 0..n {
                if o != d {
                    if let Some(r) = w.network.route_links(LinkId(o as u32), LinkId(d as u32)) {
                        if r.len() >= 3 {
                            pairs.push(OdPair { origin: LinkId(o as u32), dest: LinkId(d as u32), rate_per_sec: 0.3 });
                        }
                    }
                }
            }
        }
        let mut demand = DemandGenerator::new(&w, &pairs, 1);
        for _ in 0..1500 {
            demand.step(&mut w, 0.2);
            w.step();
        }
        assert!(demand.spawned() > 100, "scenario should be busy: {}", demand.spawned());
        assert_eq!(w.crashed(), 0, "mixed-class traffic should not crash");
    }

    #[test]
    fn lanes_are_independent() {
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::uncontrolled(2, 6000.0, 0.0),
            ],
            links: vec![LinkSpec::oneway(1, 2, 2, 30.0)],
        }
        .build();
        let lanes: Vec<LaneId> = net.lanes_of(LinkId(0)).collect();
        let mut world = NetWorld::new(net, cfg());
        // A slow vehicle in lane 0 must not slow a vehicle in lane 1. (No lane-0
        // follower here — that one would rightly change lanes; see the overtake
        // test — this isolates cross-lane car-following independence.)
        let slow = DriverConfig { desired_speed: 8.0, accel_noise: 0.0, ..DriverConfig::car() };
        world.spawn(1, lanes[0], 200.0, 8.0, slow);
        world.spawn(3, lanes[1], 100.0, 8.0, DriverConfig { accel_noise: 0.0, ..DriverConfig::car() });

        world.run_ticks(1000);

        let slow_v = world.vehicle(1).unwrap();
        let free = world.vehicle(3).unwrap();
        assert!(slow_v.speed < 9.0, "lane-0 vehicle stays slow, speed={}", slow_v.speed);
        assert!(free.speed > 20.0, "lane-1 vehicle is unaffected, speed={}", free.speed);
    }
}
