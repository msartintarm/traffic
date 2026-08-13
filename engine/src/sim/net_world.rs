//! Vehicles driving a graph [`Network`]: IDM car-following within a lane, a red
//! signal (or unsatisfied movement) as a stationary virtual leader at the stop
//! line, and lane hand-off across a node when the movement is served.
//! Accelerations read committed pre-step state and apply in a second pass.

use std::collections::HashMap;

use super::config::{DriverConfig, SimConfig, VehicleClass};
use super::rng::{self, Stream};
use super::constraint::{self, LongContext, Obstacle, SpeedTarget};
use super::congestion::{CongestionConfig, CongestionLod};
use super::hash::IntMap;
use super::idm;
use super::mobil::{self, MobilParams};
use super::junction::{self, Junctions, SignalController};
use super::network::{Lane, LaneId, LinkId, MovementId, Network, NodeControl, NodeId, RoadKind, TurnType};
use super::router::FieldRouter;
use super::signal::SignalState;

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
use super::accel_gpu::GpuAccel;

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
    /// When set, the vehicle has crossed its current lane's stop line and is
    /// traversing this movement's node interior. `position` keeps counting past
    /// `lane.length`, so the interior arc is `position - lane.length` — the road
    /// is one continuous corridor across the seam. Cleared when it lands on the
    /// destination lane (`position` rebased to the new lane's frame).
    crossing: Option<Crossing>,
    lane_change: Option<LaneChange>,
    /// Whether the active-set scheduler classified this car as sleeping on the
    /// last step — read by the next step's lane-change pass to stagger the
    /// (slow-timescale) queue-jump evaluation of parked cars.
    slept: bool,
    /// Ticks until this wreck is cleared from the road. `None` = not crashed. A
    /// wreck holds its pose at speed 0 and blocks traffic like any stopped car
    /// (leader chains, box occupancy) until the timer removes it.
    wreck: Option<u16>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Crossing {
    movement: MovementId,
    /// Lateral metres (right-positive) the car sat off its lane line when it hit
    /// the boundary — a lane-change blend still in flight. Carried through the
    /// crossing so the seam-landing blend starts from where the car actually is.
    lat_shift: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct LaneChange {
    from: LaneId,
    progress: f64,
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

    /// Drop row `i`'s retained history down to just `(position, speed)`, so its
    /// delayed leader-gap lookup falls back to the true current gap until it has
    /// re-accumulated a full window — used when a lane change moves the car to a new
    /// lane (and thus a new leader) whose old-frame positions would phantom-brake it.
    fn reset_history(&mut self, i: usize, position: f64, speed: f64) {
        self.hist[i][0] = (position, speed);
        self.hist_len[i] = 1;
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
    /// interior path (`position` counts on past its `lane`'s length; the overrun
    /// is the interior arc).
    pub fn is_crossing(&self) -> bool {
        self.crossing.is_some()
    }

    /// Whether this vehicle is a crashed wreck awaiting clearance.
    pub fn is_wrecked(&self) -> bool {
        self.wreck.is_some()
    }

    /// Consecutive ticks spent essentially stopped — the wait the yield
    /// impatience and the HUD's junction readout run on.
    pub fn wait_ticks(&self) -> u32 {
        self.wait_ticks
    }
}

/// Which detector took a vehicle off the road — the *nature* of a crash, so
/// artifact hunting and realism tuning can tell rear-end chains from junction
/// impacts instead of staring at one undifferentiated tally.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrashKind {
    /// A follower's front overran its corridor leader's rear (in-lane or across
    /// a seam, tunnelling included).
    RearEnd,
    /// Two bodies on distinct crossing paths genuinely intersected inside a node.
    Junction,
}

/// One crashed vehicle: where it wrecked, how, and how fast the pair was still
/// closing at impact (a severity proxy). Bounded by [`MAX_CRASH_SITES`].
#[derive(Clone, Copy, Debug)]
pub struct CrashRecord {
    pub pos: [f32; 2],
    pub kind: CrashKind,
    pub closing_speed: f32,
}

/// One metered on-ramp: the ramp link, the freeway mainline its merge feeds
/// (whose occupancy drives the ALINEA rate), and the live cycle state.
struct RampMeter {
    ramp: LinkId,
    mainline: LinkId,
    rate_vph: f64,
    cycle_t: f64,
    since_update: f64,
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
    /// Vehicles crashed per [`CrashKind`], indexed by the kind's discriminant —
    /// the artifact-vs-realism breakdown the flat `crashed` tally can't give.
    crashed_by: [u32; 2],
    /// One record per crashed vehicle, oldest first, capped at [`MAX_CRASH_SITES`]
    /// (the newest evict the oldest). Feeds the crash-location overlay and the
    /// cause breakdown; independent of the `crashed` tally.
    crash_log: Vec<CrashRecord>,
    /// Metered on-ramps (built on first enable) and the metering master switch.
    meters: Vec<RampMeter>,
    metering_on: bool,
    /// Simulated seconds-into-day (fed by the demand clock), the timebase for
    /// day-scheduled infrastructure: rail-crossing timetables.
    day_secs: f64,
    /// Which green-wave plan is loaded (false = AM progression, true = PM).
    pm_plan: bool,
    /// Rail preemption table: `(program, track-clearing phase, crossing node)`.
    rail_preempts: Vec<(u32, usize, NodeId)>,
    /// Bus stops per link `(stop index, link arc)` — from `Network::bus_stops`.
    stops_by_link: IntMap<Vec<(u32, f64)>>,
    /// Per movement: does it carry any registered conflict point? An
    /// "interchange" movement that genuinely crosses another (a frontage-road
    /// junction the road-class heuristic mislabels free-flow) must not ride the
    /// free-flow exemptions past the box gates.
    movement_conflicted: Vec<bool>,
    /// Live bus service state by vehicle id: `(dwell-until tick, link of the
    /// last-served stop, its arc)` — positional, so nearby stops (or re-matching
    /// the same one) can never re-trap a bus that has already served here.
    bus_dwell: HashMap<u32, (u64, u32, f64)>,
    /// Downstream lanes fed by more than one lane — the merge points; value is
    /// the list of feeding (from) lane ids.
    merges: HashMap<u32, Vec<u32>>,
    /// Per-link: is the whole link gentle enough that its curvature never limits speed
    /// (curve speed ≥ its limit)? Precomputed from fixed geometry; lets the active-set
    /// scheduler's free-car path skip the per-tick curve scan on straight links.
    link_straight: Vec<bool>,
    /// Per-movement curvature-limited interior speed (`√(a_lat·r)` over the
    /// interior Bézier, `INFINITY` when unconstrained) — precomputed, since the
    /// cap is read in the per-vehicle passes.
    turn_caps: Vec<f64>,
    /// Per-lane: does this lane feed a merge point? A car here can face a moving cross-merger
    /// (`merge_conflict`) that the synthesized sleeper contexts don't see, so the free-car
    /// path excludes it (e.g. freeway mainline lanes an on-ramp merges into).
    merge_feeder_lane: Vec<bool>,
    /// Per-lane through-continuation: the lane a straight-through car flows onto next and
    /// the interior length to reach it. Lets car-following see a leader across segment
    /// boundaries by walking this chain, instead of losing sight of it at every node (the
    /// cause of the phantom slam/stall when crossing segments). `None` at a dead end.
    through_next: Vec<Option<(LaneId, f64)>>,
    /// Corridor coalescing (Tier 1): a maximal chain of grade-separated lanes joined by
    /// *pure continuation* movements (a lane's only exit into a single-fed next lane) is one
    /// seamless corridor. `corridor_of[lane]` is its corridor id; `corridor_offset[lane]` is
    /// the arc-length (lane lengths + node interiors) from the corridor's start to this
    /// lane's start, so `corridor_offset[lane] + position` is a continuous coordinate along
    /// the whole freeway run. Following and advance use it to flow across the seam with no
    /// admission gate and no leader hand-off; the corridor breaks (and the gate returns) at
    /// merges, diverges, lane-drops, and at-grade nodes.
    corridor_of: Vec<u32>,
    corridor_offset: Vec<f64>,
    /// Per-movement: a continuation seam taken by its exit link's *through approach* —
    /// the freeway mainline at a merge, the ramp itself on a ramp-to-ramp chain. These
    /// cross ungated like corridor seams (a freeway never brake-checks at a segment
    /// boundary), even where a lane remap breaks the 1:1 corridor chain; every other
    /// feeder (the merging ramp) keeps the admission gate — it yields.
    seam_primary: Vec<bool>,
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
    /// Congestion fingerprint (which links carry enough traffic to shift routing costs) at
    /// the last route recompute. Routing recomputes only when this fingerprint moves — under
    /// light or static traffic the free-flow fields from `install_router` stay optimal, so the
    /// whole-map O(links) rebuild is skipped and routing cost tracks congestion, not map size.
    route_fingerprint: u64,
    /// Tick the current reroute cycle started, so a new one can't begin until the reroute
    /// interval elapses — churny traffic can't chain whole-map rebuilds. The per-field spread
    /// itself lives in the router (`advance_recompute`).
    route_cycle_tick: u64,
    /// Solve several destination fields per reroute across cores (one per thread) vs. one at a
    /// time. On by default; a UI toggle so the parallel speed-up is observable. No effect on
    /// the single-threaded build or when the browser GPU flow-field owns the recompute.
    parallel_routing: bool,
    /// Sort the per-lane / per-corridor vehicle groups against a precomputed flat position key
    /// (contiguous, cache-friendly) rather than reading a full vehicle row per comparison. On by
    /// default; a UI toggle so the win is measurable in the browser (bit-for-bit either way).
    cache_sort: bool,
    /// Per-node index of movements and conflict points.
    junctions: Junctions,
    /// Executor requested for the per-vehicle accel passes (see [`AccelBackend`]).
    accel_backend: AccelBackend,
    /// GPU binding-fold solver, installed by [`NetWorld::enable_gpu_accel`]. When
    /// present it backs the `Gpu` backend's evaluate pass; absent, `Gpu` falls back
    /// to `Serial`. Native only — the browser can't block on the same-step readback.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    gpu_accel: Option<GpuAccel>,
    /// Whether the CPU worker pool is up. Native: rayon's global pool auto-inits, so
    /// always true. Browser: false until JS finishes `initThreadPool` (SharedArrayBuffer
    /// needs cross-origin isolation), gating the `Threads` backend until then.
    threads_ready: bool,
    /// Vehicle count at/above which the `Threads` backend actually parallelizes (below
    /// it, serial — rayon overhead isn't worth it). Runtime-tunable for measurement.
    par_threshold: usize,
    /// Congestion level-of-detail: while a link stays saturated, its queued followers
    /// use cheap leader-only car-following instead of the full gather. Cars stay
    /// individual; this only cuts per-car work. Inert unless `congestion_cfg.enabled`.
    congestion: CongestionLod,
    congestion_cfg: CongestionConfig,
    /// Vehicles the active-set scheduler skipped last step (diagnostic; 0 when off).
    asleep_last: usize,
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
/// actually available. The CPU backends match [`AccelBackend::Serial`] bit-for-bit
/// (the passes read only committed pre-step state, so order-preserving parallelism is
/// exact); the `Gpu` backend matches only within f32 tolerance (it folds in f32).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccelBackend {
    /// Single-threaded — always available, and the correctness reference.
    Serial,
    /// Data-parallel across CPU cores (rayon). Native under `--features parallel`;
    /// the browser additionally needs a SharedArrayBuffer worker pool (a follow-up).
    Threads,
    /// GPU compute kernel (`accel.wgsl`) for the evaluate pass; the gather stays on
    /// CPU. Native + `gpu`, after [`NetWorld::enable_gpu_accel`] installs a solver;
    /// otherwise [`NetWorld::active_backend`] falls it back to `Serial`. The browser
    /// can't block on the same-step readback, so it too falls back there.
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
/// default is set well under it so threads engage sooner under climbing load — in the
/// browser choppy maps lock to a smooth speed once this trips), so [`map_collect`] stays
/// serial even on the `Threads` backend below it. Adjustable at runtime.
pub const DEFAULT_PAR_THRESHOLD: usize = 500;

/// Cap on retained crash-site positions (for the overlay). A long run can accumulate many
/// wrecks; keep the most recent so the overlay stays bounded in memory and upload size.
pub const MAX_CRASH_SITES: usize = 8192;

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

/// Sort a lane/corridor index group by a precomputed flat key (position or corridor position).
/// Each comparison then reads an 8-byte key from a contiguous array rather than chasing a
/// ~200-byte vehicle row — the pointer-chase into the fat fleet is what made these per-group
/// sorts cache-bound and superlinear on a big map. Positions within a group are distinct
/// (vehicles can't overlap), so this total order is the same result the row-chasing sort gave.
fn sort_group_by(members: &mut [usize], key: &[f64]) {
    members.sort_by(|&a, &b| key[a].total_cmp(&key[b]));
}

/// The `Gpu` backend's evaluate pass: the binding fold in `accel.wgsl` (f32) with the
/// u64-RNG noise term re-added on the CPU. Rolling vehicles go to the kernel; in-node
/// crossers carry their precomputed accel through unchanged. Native + `gpu` only.
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
fn gpu_evaluate(solver: &mut GpuAccel, inputs: &[AccelInput], seed: u64, tick: u64) -> Vec<f64> {
    let mut gpu_in = Vec::with_capacity(inputs.len());
    let mut rolling = Vec::with_capacity(inputs.len());
    for (i, inp) in inputs.iter().enumerate() {
        if let AccelInput::Rolling(cx) = inp {
            rolling.push(i);
            gpu_in.push(cx.to_gpu());
        }
    }
    let bindings = solver.binding_accels(&gpu_in);
    let mut out = vec![0.0f64; inputs.len()];
    for (i, inp) in inputs.iter().enumerate() {
        if let AccelInput::Crossing(a) = inp {
            out[i] = *a;
        }
    }
    for (j, &i) in rolling.iter().enumerate() {
        if let AccelInput::Rolling(cx) = &inputs[i] {
            out[i] = bindings[j] as f64 + constraint::accel_noise(cx.driver.accel_noise, seed, cx.agent_id, tick);
        }
    }
    out
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
    /// every field is dropped to `f32` (WGSL has no `f64`). The `Gpu` backend's
    /// evaluate pass ([`gpu_evaluate`]) uploads a slice of these each step.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
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

/// The active-set scheduler's per-vehicle decision for a tick: run the full gather, or
/// synthesize a cheap context because the car's motion is trivially determined.
#[derive(Clone, Copy)]
enum Sleep {
    /// Run the full gather (the car is actually deciding).
    Awake,
    /// Queued behind the given stopped-leader index — hold via [`NetWorld::queue_context`].
    Queued(usize),
    /// Isolated on open, straight road far from its node — cruise on a free-road context.
    Free,
}

/// Per-vehicle input to the accel evaluate pass. A rolling vehicle carries its full
/// [`VehicleContext`]; an in-node crosser carries the acceleration its bespoke
/// [`NetWorld::crossing_accel`] already produced, throttle noise included at gather.
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

/// How far ahead a driver reads signals to ease off early for a red at the next
/// intersection (metres) — anticipatory braking across the current link.
const SIGNAL_LOOKAHEAD: f64 = 90.0;

/// Below this speed (m/s) a vehicle counts as stopped for the sleep-scheduler's
/// state classification (matches the stop-line detection thresholds elsewhere).
const MOVE_EPS: f64 = 0.3;
/// Leader gap (m) beyond which the leader's IDM term is effectively non-binding, so
/// a car with nothing closer is cruising on the free road.
const LEADER_PERCEPTION: f64 = 120.0;
/// Distance (m) within which a stopped car is treated as queued at a stop line / behind
/// a stopped leader (so it can sleep as frozen rather than counting as deciding).
const STOP_QUEUE_GAP: f64 = 12.0;
/// Speed (m/s) below which the scheduler will sleep a queued car — deliberately far
/// stricter than [`MOVE_EPS`]: only a *genuinely parked* car behind a *genuinely
/// parked* leader gets the synthesized leader-only context, because for such a car the
/// leader term provably dominates every other constraint and the IDM fold is bitwise
/// identical to the full gather (so sleeping cannot perturb the trajectory). A car
/// still rolling — even slowly — takes the full gather, where `queue_context` would
/// diverge (no reaction delay, no curve/merge/yield terms).
const SLEEP_SPEED_EPS: f64 = 0.02;
/// Ticks between a sleeping queued car's lane-change (queue-jump) evaluations —
/// staggered by vehicle id. ~1 s at the 0.2 s step: parked-queue decisions are
/// slow-timescale, and skipping the MOBIL scan for sleepers is a third of the
/// lane-change pass at gridlock.
const SLEEPER_LC_PERIOD: u64 = 5;
/// Parallel crossover (vehicle count) for the *light* per-car passes — the
/// MOBIL lane-change scan and the in-lane integrate. Their per-car work is far
/// below the accel gather's, so rayon's dispatch+collect overhead only pays
/// much later than [`DEFAULT_PAR_THRESHOLD`]: measured on the loaded real map,
/// both parallel arms lose below ~8k cars (the integrate arm by 8×).
const LIGHT_PAR_THRESHOLD: usize = 8000;
/// Beyond this distance to the next node (m), and with no slower zone downstream, no node
/// constraint (signal/box/yield/stop) can yet bind — [`NetWorld::gather_context`] LOD-skips
/// the node stack, and the active-set scheduler's free-car path applies from here out.
const DECISION_HORIZON: f64 = 180.0;
/// Comfortable lateral acceleration (m/s²) bounding curve speed (`v = √(A_LAT·r)`), and how
/// far ahead a curve is read.
const A_LAT: f64 = 3.0;
const CURVE_LOOKAHEAD: f64 = 45.0;
/// Maximum physical deceleration (m/s², ~0.9 g of tyre grip). IDM's interaction term is
/// unbounded as the gap shrinks, so an unclamped car could compute hundreds of m/s² and
/// teleport from highway speed to a dead stop in one tick. `integrate`/`advance_crossing`
/// clamp every applied acceleration to this bound, so an over-committed driver physically
/// cannot always stop — which is exactly what lets reaction delay and misjudged gaps end
/// in a detectable crash instead of a superhuman save. It also caps the ease-down at a
/// segment boundary the car can't cross yet.
const MAX_BRAKE_DECEL: f64 = 9.0;

/// How close to a merge point a car begins cooperatively yielding to a cross-merger
/// (~1.5 s at freeway speed). Beyond it the two streams are still far enough apart that
/// their difference in distance-to-merge is not a real gap — treating it as one braked
/// mainline cars to a standstill a hundred-plus metres before an on-ramp.
const MERGE_APPROACH: f64 = 45.0;
/// Floor on the cross-boundary leader search: even stopped, a car looks this far ahead
/// (so it notices a queue starting to build just past the next node).
const LEADER_HORIZON_MIN: f64 = 30.0;

/// How far ahead a car needs to see to brake comfortably: its reaction (headway) distance
/// plus its stopping distance at comfortable deceleration. The cross-boundary leader walk
/// scans this far — so a fast car reaches across however many segments its speed spans,
/// and a slow one only looks just ahead. This is what lets a car detect, from its own
/// speed, which segments downstream it can still stop for.
fn leader_horizon(driver: &DriverConfig, speed: f64) -> f64 {
    let stop_dist = speed * speed / (2.0 * driver.comfort_decel.max(0.5));
    (speed * driver.time_headway + stop_dist).max(LEADER_HORIZON_MIN)
}
/// Ramp metering (ALINEA): green long enough for one car per cycle; rate adapts
/// toward holding the protected mainline at its critical occupancy (where flow
/// peaks), within the typical single-lane meter envelope.
const METER_GREEN_SECS: f64 = 2.5;
/// Gate-down time per train passage at a level crossing (day-clock seconds).
const RAIL_CLOSURE_SECS: f64 = 45.0;
/// Curbside service time a bus spends at each stop.
const BUS_DWELL_SECS: f64 = 25.0;
/// Speed at which a receiving-lane occupant counts as *departing* — a leader to
/// car-follow rather than a blockage the box gates must hold for. Walking pace:
/// a tail genuinely rolling off, not a stop-and-go twitch. This must sit *below*
/// the speeds a discharging queue crosses the line at (~2.5–3 m/s), or the gate
/// flaps on every launch and re-serializes the queue to one car per ~4.6 s —
/// measured: the flap held saturation flow to ~1080 veh/h/lane against the real
/// ~1900, invariant to driver acceleration (see
/// `queue_discharge_hits_real_saturation_flow`).
const DEPARTING_SPEED: f64 = 3.0;

/// Bumper margin demanded behind a departing tail: a following distance the
/// entering car can hold if the tail brakes mid-box (≈ half a headway at the
/// tail's speed), not the bare standstill gap.
fn departing_margin(driver: &DriverConfig, tail_speed: f64) -> f64 {
    driver.min_gap + 0.6 * tail_speed
}

impl NetWorld {
    /// Where the departing-tail exemption applies: signalized crossings and
    /// freeway seams/interchanges — the high-capacity contexts the absolute
    /// commit-room check was serializing to ⅓ of real saturation flow. At
    /// uncontrolled/stop/yield boxes the conservative room stays: those small
    /// grids ring-gridlock when cars follow each other into boxes that jam.
    fn departing_exemption(&self, mid: MovementId) -> bool {
        matches!(self.network.node(self.network.movement(mid).node).control, NodeControl::Signalized(_))
            || self.network.is_continuation_seam(mid)
            || self.network.is_interchange_movement(mid)
    }
}
const METER_MIN_VPH: f64 = 240.0;
const METER_MAX_VPH: f64 = 1500.0;
const ALINEA_PERIOD_SECS: f64 = 30.0;
const ALINEA_SETPOINT: f64 = 0.21;
const ALINEA_GAIN_VPH: f64 = 4000.0;

/// Share of vehicles allowed in HOV/express lanes: carpools plus toll-paying
/// SOVs on the US-101 express lanes (roughly the observed eligible fraction).
const HOV_ELIGIBLE_SHARE: f64 = 0.2;

/// Whether this vehicle may use HOV/express lanes — a stable per-vehicle draw
/// (occupancy is decided when the trip starts, not per lane change).
fn hov_eligible(seed: u64, id: u32) -> bool {
    rng::uniform01(seed, id, 2, Stream::DriverProfile) < HOV_ELIGIBLE_SHARE
}

const KEEP_RIGHT_BIAS: f64 = 0.3;
/// Lead *time* a car positions for the lane its route needs (leaving an exit-only lane,
/// reaching a turn lane) before a node — the distance is this times its speed, so a fast
/// car starts moving over sooner and a slow one later, rather than one distance for
/// everyone. Beyond it all lanes are fair game (a through car may still ride an exit lane
/// for capacity). ~8.5 s ≈ 250 m at freeway speed, where that distance was tuned.
const LANE_POSITION_LEAD: f64 = 8.5;
/// Floor on that distance so a near-stopped car still starts positioning.
const LANE_POSITION_MIN: f64 = 40.0;
const YELLOW_RUN_SALT: u64 = 0x59_4c_57;
const RED_RUN_SALT: u64 = 0x52_45_44;
const LANE_CHANGE_DURATION: f64 = 2.0;
/// Cap on how many stacked positioning windows a multi-lane fix opens
/// (`best_lane_change`): even a car four-plus lanes out starts at three windows,
/// so mid-block driving isn't dominated by far-off turn preparation.
const MAX_POSITION_WINDOWS: f64 = 3.0;
/// How many junctions deep lane preference follows the route's landing-lane
/// chain (`lanes_to_serving`): 3 covers a turn two short blocks ahead — the
/// closely-spaced-signals case — without scanning the whole route.
const LANE_ROUTE_DEPTH: usize = 3;
/// How far short of its first live conflict point a mid-box permissive-left
/// waiter stands (front bumper): enough that an oncoming through's swept
/// corridor clears the waiter's nose with a real margin at any crossing angle.
const PERMISSIVE_HOLD_MARGIN: f64 = 3.0;

/// A vehicle's decision state for the active-set scheduler. `Free` and `Frozen` are the
/// analytically-predictable states that may sleep; `Deciding` must run the full step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CarState {
    /// No leader within perception and no intersection/event within lookahead — cruising
    /// the free road toward desired speed.
    Free,
    /// Stopped behind a stopped leader or at a stop line — held until woken.
    Frozen,
    /// Anything else: car-following a live leader, approaching a node, mid-crossing, etc.
    Deciding,
}

impl NetWorld {
    /// Classify vehicle `i`'s decision state from committed state + neighbours. Reads
    /// only current positions/speeds (no router/signal mutation), so it is safe to call
    /// for measurement or to gate the active set.
    fn classify(&self, i: usize, nb: &Neighbors) -> CarState {
        let v = &self.fleet.rows[i];
        if v.crossing.is_some() {
            return CarState::Deciding;
        }
        let lane = self.network.lane(v.lane);
        let to_end = lane.length - v.position;
        let leader_gap = nb.leader_of[i].map(|l| self.corridor_gap(v, &self.fleet.rows[l]));
        let leader_stopped_close = nb.leader_of[i]
            .zip(leader_gap)
            .is_some_and(|(l, g)| g < STOP_QUEUE_GAP && self.fleet.rows[l].speed < MOVE_EPS);
        if v.speed < MOVE_EPS && (leader_stopped_close || to_end < STOP_QUEUE_GAP) {
            return CarState::Frozen;
        }
        let leader_close = leader_gap.is_some_and(|g| g < LEADER_PERCEPTION);
        if !leader_close && to_end >= SIGNAL_LOOKAHEAD {
            return CarState::Free;
        }
        CarState::Deciding
    }

    /// Census of the current fleet by [`CarState`] as `[free, frozen, deciding]` — the
    /// measurement behind the sleep-scheduler ROI (how large the sleepable fraction is).
    /// Builds neighbours internally; call sparingly (e.g. from the load bench window).
    pub fn state_census(&self) -> [usize; 3] {
        let nb = self.neighbors();
        let mut counts = [0usize; 3];
        for i in 0..self.fleet.rows.len() {
            counts[self.classify(i, &nb) as usize] += 1;
        }
        counts
    }

    pub fn new(network: Network, cfg: SimConfig) -> Self {
        let mut merges: HashMap<u32, Vec<u32>> = HashMap::new();
        for m in &network.movements {
            let entry = merges.entry(m.to_lane.0).or_default();
            if !entry.contains(&m.from_lane.0) {
                entry.push(m.from_lane.0);
            }
        }
        merges.retain(|_, froms| froms.len() > 1);

        let mut merge_feeder_lane = vec![false; network.lanes.len()];
        for froms in merges.values() {
            for &f in froms {
                merge_feeder_lane[f as usize] = true;
            }
        }

        // Per-lane through-continuation: the movement a car takes going straight (a
        // `Through` movement, else the lane's sole/first movement), giving the next lane
        // and the interior length to reach it. The leader walk follows this chain to see
        // across segment boundaries.
        let through_next: Vec<Option<(LaneId, f64)>> = (0..network.lanes.len())
            .map(|li| {
                let lane = LaneId(li as u32);
                let movs = network.movements_of(lane);
                if movs.is_empty() {
                    return None;
                }
                let start = network.lane(lane).movement_start.0;
                let k = (0..movs.len())
                    .find(|&k| network.movement_turn(MovementId(start + k as u32)) == TurnType::Through)
                    .unwrap_or(0);
                let mid = MovementId(start + k as u32);
                Some((movs[k].to_lane, network.interior(mid).len))
            })
            .collect();

        // Corridor coalescing: fold each maximal chain of grade-separated 1:1 through-lanes
        // into one seamless corridor (see the `corridor_of` field). A lane A continues into
        // lane B iff A→B is A's *only* movement, B is single-fed, and the movement is a
        // free-flow interchange (both links grade-separated) — so the chain breaks exactly at
        // merges (B multi-fed), diverges (A multi-out), lane-drops and at-grade nodes.
        let mut incoming = vec![0u32; network.lanes.len()];
        for m in &network.movements {
            incoming[m.to_lane.0 as usize] += 1;
        }
        let corridor_next: Vec<Option<(LaneId, f64)>> = (0..network.lanes.len())
            .map(|li| {
                let a = LaneId(li as u32);
                let movs = network.movements_of(a);
                if movs.len() != 1 {
                    return None;
                }
                let mid = network.lane(a).movement_start;
                let b = movs[0].to_lane;
                (incoming[b.0 as usize] == 1 && network.is_interchange_movement(mid))
                    .then(|| (b, network.interior(mid).len))
            })
            .collect();
        // Chains are disjoint (each lane has ≤1 successor and, being single-fed, ≤1
        // predecessor). Walk from each chain head assigning a corridor id and running offset.
        let mut has_pred = vec![false; network.lanes.len()];
        for n in corridor_next.iter().flatten() {
            has_pred[n.0.0 as usize] = true;
        }
        let mut corridor_of = vec![u32::MAX; network.lanes.len()];
        let mut corridor_offset = vec![0.0f64; network.lanes.len()];
        let mut next_corridor = 0u32;
        for head in 0..network.lanes.len() {
            if has_pred[head] {
                continue;
            }
            let cid = next_corridor;
            next_corridor += 1;
            let mut cur = Some(LaneId(head as u32));
            let mut offset = 0.0;
            while let Some(l) = cur {
                corridor_of[l.0 as usize] = cid;
                corridor_offset[l.0 as usize] = offset;
                cur = corridor_next[l.0 as usize].map(|(b, interior)| {
                    offset += network.lane(l).length + interior;
                    b
                });
            }
        }

        // Per-link straightness: does the tightest curve anywhere on the link still allow
        // its speed limit? If so, a car on it is never curve-limited, so the free-car path
        // can skip the curve scan. Conservative (whole-link min radius) and geometry-only.
        let link_straight = (0..network.links.len())
            .map(|i| {
                let lane_start = network.link(LinkId(i as u32)).lane_start;
                let r = network.min_radius_ahead(lane_start, 0.0, f64::INFINITY);
                (A_LAT * r).sqrt() >= network.lane(lane_start).speed_limit
            })
            .collect();

        // Curvature-limited interior speed per movement, the same lateral-comfort
        // law the link curve scan applies (`v = √(a_lat·r)`), from each interior
        // Bézier's tightest radius. A hooked right onto a narrow street crawls,
        // a sweeping channelized turn flows — instead of one flat cap per turn
        // direction. Floored so a degenerate sliver interior can't demand a
        // crawl, ceilinged at box speed (an intersection is never open road).
        let turn_caps = (0..network.movements.len() as u32)
            .map(|m| {
                let mid = MovementId(m);
                if network.is_interchange_movement(mid) {
                    return f64::INFINITY;
                }
                let r = network.interior_min_radius(mid);
                if r.is_finite() {
                    (A_LAT * r).sqrt().clamp(2.5, 10.0)
                } else {
                    f64::INFINITY
                }
            })
            .collect();

        // The through approach of each link: of the links feeding it, the one that
        // continues most directly — same road kind first (mainline over ramp), then
        // straightest, then widest, then lowest id for determinism.
        let mut primary: HashMap<u32, ((u8, f64, u32), u32)> = HashMap::new();
        for m in &network.movements {
            let fl = network.lane(m.from_lane).link;
            let tl = network.lane(m.to_lane).link;
            let a = network.arrival_dir(fl);
            let b = network.departure_dir(tl);
            let cand = (
                ((network.link(fl).kind == network.link(tl).kind) as u8, a[0] * b[0] + a[1] * b[1], network.link(fl).lane_count),
                fl.0,
            );
            let e = primary.entry(tl.0).or_insert(cand);
            let ord = cand.0.0.cmp(&e.0.0).then(cand.0.1.total_cmp(&e.0.1)).then(cand.0.2.cmp(&e.0.2)).then(e.1.cmp(&cand.1));
            if ord == std::cmp::Ordering::Greater {
                *e = cand;
            }
        }
        let seam_primary: Vec<bool> = (0..network.movements.len() as u32)
            .map(|m| {
                let mid = MovementId(m);
                let mv = network.movement(mid);
                let fl = network.lane(mv.from_lane).link;
                let tl = network.lane(mv.to_lane).link;
                network.is_continuation_seam(mid) && primary.get(&tl.0).is_some_and(|&(_, f)| f == fl.0)
            })
            .collect();

        let signals = SignalController::build(&network);
        let rail_preempts = Self::build_rail_preempts(&network);
        let mut movement_conflicted = vec![false; network.movements.len()];
        for c in &network.conflicts {
            movement_conflicted[c.a.idx()] = true;
            movement_conflicted[c.b.idx()] = true;
        }
        let mut stops_by_link: IntMap<Vec<(u32, f64)>> = IntMap::default();
        for (i, &(link, arc)) in network.bus_stops.iter().enumerate() {
            stops_by_link.entry(link.0).or_default().push((i as u32, arc));
        }
        let link_entries = vec![0u32; network.links.len()];
        let junctions = Junctions::build(&network);
        let congestion = CongestionLod::new(network.links.len());
        Self {
            network, cfg, fleet: Fleet::default(), time: 0.0, tick: 0, exited: 0, leaked: 0, crashed: 0, crashed_by: [0; 2], crash_log: Vec::new(),
            merges, link_straight, turn_caps, merge_feeder_lane, through_next, corridor_of, corridor_offset, seam_primary, signals, link_entries, router: None, external_reroute: false,
            meters: Vec::new(), metering_on: false, day_secs: 0.0, pm_plan: false, rail_preempts,
            stops_by_link, bus_dwell: HashMap::new(), movement_conflicted,
            route_fingerprint: 0, route_cycle_tick: 0, parallel_routing: true, cache_sort: true, junctions,
            accel_backend: AccelBackend::Serial,
            #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
            gpu_accel: None,
            threads_ready: cfg!(not(target_arch = "wasm32")),
            par_threshold: DEFAULT_PAR_THRESHOLD,
            congestion, congestion_cfg: CongestionConfig::disabled(),
            asleep_last: 0,
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

    /// Install the GPU binding-fold solver so the `Gpu` backend runs its evaluate pass
    /// on the GPU (native only; the browser can't block on the same-step readback).
    /// Returns whether an adapter was acquired — `false` leaves `Gpu` falling back to
    /// `Serial`. Idempotent enough to call once at setup.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn enable_gpu_accel(&mut self) -> bool {
        if self.gpu_accel.is_none() {
            self.gpu_accel = GpuAccel::new();
        }
        self.gpu_accel.is_some()
    }

    /// The backend actually used this step: the request, downgraded to `Serial` when
    /// it isn't available (no CPU worker pool, or no GPU solver installed).
    pub fn active_backend(&self) -> AccelBackend {
        match self.accel_backend {
            AccelBackend::Threads if threads_available() && self.threads_ready => AccelBackend::Threads,
            #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
            AccelBackend::Gpu if self.gpu_accel.is_some() => AccelBackend::Gpu,
            _ => AccelBackend::Serial,
        }
    }

    /// Run the evaluate pass over the gathered `inputs`. Serial/Threads fold the
    /// constraints and add per-tick noise in one order-preserving map; the `Gpu`
    /// backend runs the binding fold in `accel.wgsl` (f32) and re-adds the noise
    /// CPU-side, so it agrees with `Serial` only within f32 tolerance — a throughput
    /// option, not the bit-exact reference the CPU backends are.
    fn evaluate_accels(&mut self, backend: AccelBackend, par_threshold: usize, inputs: &[AccelInput], seed: u64, tick: u64) -> Vec<f64> {
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        if backend == AccelBackend::Gpu {
            if let Some(solver) = self.gpu_accel.as_mut() {
                return gpu_evaluate(solver, inputs, seed, tick);
            }
        }
        map_collect(backend, par_threshold, inputs.len(), |i| inputs[i].evaluate(seed, tick))
    }

    /// Gather one vehicle's accel-decision context — the graph, neighbor, signal and router
    /// lookups a GPU evaluate kernel can't chase. In-node crossers carry their bespoke
    /// `crossing_accel`; queued sleepers and congested-link followers take the leader-only
    /// `queue_context`; isolated free cars the bare `free_context`; everyone else the full
    /// gather. The single source of truth for both the fused CPU path and the GPU path's
    /// materialized inputs.
    fn gather_input(&self, i: usize, nb: &Neighbors, cross_by_mv: &IntMap<Vec<usize>>, sleep: Sleep, intended: Option<MovementId>) -> AccelInput {
        let veh = &self.fleet.rows[i];
        if veh.wreck.is_some() {
            AccelInput::Crossing(0.0) // a wreck decides nothing; integration pins it
        } else if veh.crossing.is_some() {
            // In-box driving carries the same throttle imperfection as the open road
            // — the crossing kernel must not be *more* precise than the driver is
            // everywhere else.
            let noise = constraint::accel_noise(veh.driver.accel_noise, self.cfg.seed, veh.id, self.tick);
            AccelInput::Crossing(self.crossing_accel(i, nb, cross_by_mv) + noise)
        } else {
            match sleep {
                Sleep::Queued(leader) => AccelInput::Rolling(self.queue_context(i, leader)),
                Sleep::Free => AccelInput::Rolling(self.free_context(i)),
                Sleep::Awake => match self.queue_follower(i, nb) {
                    Some(leader) => AccelInput::Rolling(self.queue_context(i, leader)),
                    None => AccelInput::Rolling(self.gather_context(i, nb, intended)),
                },
            }
        }
    }

    /// A vehicle's intended movement plus its sleep classification (the active-set scheduler's
    /// verdict): a crosser or an awake car carries its movement; a car queued behind a stopped
    /// leader or cruising free of any junction sleeps and drops the movement. Reads only this
    /// car's state and committed neighbours, so it fuses into the per-car decision pass.
    fn decide_intent(&self, i: usize, sleep_on: bool, nb: &Neighbors) -> (Option<MovementId>, Sleep) {
        let v = &self.fleet.rows[i];
        if v.crossing.is_some() || v.wreck.is_some() {
            return (None, Sleep::Awake);
        }
        if sleep_on {
            if let Some(l) = self.sleep_leader(i, nb) {
                return (None, Sleep::Queued(l));
            }
        }
        let intended = self.intended_movement(v);
        if sleep_on && self.is_free_sleeper(i, nb, intended) {
            return (None, Sleep::Free);
        }
        (intended, Sleep::Awake)
    }

    /// A movement that rides the free-flow interchange exemptions: interchange-
    /// classified *and* genuinely conflict-free (a freeway merge/diverge). A
    /// road-class-labelled "interchange" movement that crosses another still
    /// takes the full box discipline.
    fn free_flow_interchange(&self, m: MovementId) -> bool {
        self.network.is_interchange_movement(m) && !self.movement_conflicted[m.idx()]
    }

    /// The hard don't-enter-the-box gate for one car: never begin crossing while a conflicting
    /// movement still occupies the node. Free-flow interchange movements never hard-block (the
    /// merge is a zipper); only at-grade crossings gate on box occupancy.
    fn box_entry_blocked(&self, i: usize, intended: Option<MovementId>, nb: &Neighbors) -> bool {
        intended.is_some_and(|mid| {
            if self.free_flow_interchange(mid) {
                return false;
            }
            let veh = &self.fleet.rows[i];
            let node = self.network.link(self.network.lane(veh.lane).link).to;
            // A green permissive left that can stand inside the box short of its
            // first live conflict point is admitted as a *waiter* rather than
            // held at the line — its in-box hold takes the yield over.
            let hold = self.permissive_waiter_hold(i, mid, node, nb);
            // The cluster-exit gate binds the *admission* too, not just the approach
            // braking: a car that still reaches the line (creep, hot arrival) must
            // hold there rather than enter a junction it cannot clear.
            self.box_conflict_on_path_holding(veh, mid, node, nb, hold)
                || (hold.is_none() && self.is_permissive(mid) && self.permissive_must_yield(i, mid, node, nb))
                || self.junction_exit_blocked(veh, mid, node, nb)
                || self.rail_closed(node)
                // The gather's priority yield must also bind at the admission
                // line, or a slow roll past the paint enters across traffic the
                // driver was told to wait for. All-way stops are exempt: there
                // the FIFO turn-taking protocol is the arbiter, and this veto
                // deadlocked two creeping fronts against each other. A two-way
                // stop is not exempt — its major road never arms the FIFO, so
                // the minor street's creep must still respect priority.
                || ((matches!(
                    self.network.node(node).control,
                    NodeControl::Uncontrolled | NodeControl::Yield
                ) || (matches!(self.network.node(node).control, NodeControl::Stop)
                    && !self.network.all_way_stop(node)))
                    && self.conflicting_priority_traffic(i, veh.lane, node, nb).is_some())
        })
    }

    /// Install a flow-field router covering `dests`; vehicles spawned via
    /// [`NetWorld::spawn_to`] then route by the field and reroute live.
    pub fn install_router(&mut self, dests: &[LinkId]) {
        let costs = self.live_link_costs();
        self.router = Some(FieldRouter::new(&self.network, dests, &costs));
    }

    /// Configure the congestion level-of-detail. Turning it off returns every link to
    /// full per-car detail immediately.
    pub fn set_congestion(&mut self, cfg: CongestionConfig) {
        if !cfg.enabled {
            self.congestion.reset();
        }
        self.congestion_cfg = cfg;
    }

    pub fn congestion_config(&self) -> CongestionConfig {
        self.congestion_cfg
    }

    /// How many links are currently running the cheap queue model.
    pub fn congestion_active_links(&self) -> u32 {
        self.congestion.active_count()
    }

    /// Per-link occupancy ratio (rolling car count ÷ jam capacity), the signal the
    /// congestion LOD thresholds on.
    fn link_occupancy(&self) -> Vec<f64> {
        let n = self.network.links.len();
        let mut count = vec![0u32; n];
        for v in &self.fleet.rows {
            if v.crossing.is_none() {
                count[self.network.lane(v.lane).link.idx()] += 1;
            }
        }
        (0..n)
            .map(|i| {
                let l = self.network.link(LinkId(i as u32));
                let lane = self.network.lane(l.lane_start);
                let jam = (lane.length / 7.0 * l.lane_count as f64).max(1.0);
                (count[i] as f64 / jam).min(1.0)
            })
            .collect()
    }

    /// A cheap accel context for a queued follower: leader car-following (at the true
    /// current gap, no reaction-delay history) capped at the lane speed, and nothing
    /// else. The full gather's curve/merge/yield/signal-lookahead scans are skipped —
    /// they barely bind for a car crawling behind a leader, and the front-of-lane car
    /// (which has no leader, so it takes the full path) still handles the junction.
    fn queue_context(&self, i: usize, leader: usize) -> VehicleContext {
        let veh = &self.fleet.rows[i];
        let lane = self.network.lane(veh.lane);
        let lead = &self.fleet.rows[leader];
        let mut cx = VehicleContext::new(veh.driver.capped_to(lane.speed_limit), veh.speed, veh.id);
        cx.set_leader(Some(Obstacle {
            gap: self.corridor_gap(veh, lead),
            speed: lead.speed,
        }));
        cx
    }

    /// The same-lane leader if car `i` qualifies for the cheap queue model — it is on a
    /// congested (queue-mode) link and has a leader ahead. Returns `None` otherwise
    /// (including the front-of-lane car), so those take the full gather path.
    fn queue_follower(&self, i: usize, nb: &Neighbors) -> Option<usize> {
        if !self.congestion_cfg.enabled {
            return None;
        }
        let link = self.network.lane(self.fleet.rows[i].lane).link;
        if self.congestion.is_queue(link.idx()) {
            nb.leader_of[i]
        } else {
            None
        }
    }

    /// Active-set scheduler gate (queued sleepers): the same-lane leader if car `i` may
    /// sleep this tick — it is stopped behind a *close, stopped* leader, so its
    /// acceleration is pinned by that leader and the full node/signal/router gather can be
    /// skipped (synthesised as [`queue_context`](Self::queue_context)). Returns `None`
    /// (⇒ full gather) for moving cars and for the front-of-queue car (no leader), which
    /// must keep the full path so it reacts to a green / an opening gap. When the leader
    /// starts moving, this returns `None` next tick and the follower wakes — a one-tick
    /// discharge lag, within the reaction-time model. Cheap: one neighbour + gap check, no
    /// graph walk. The caller decides *whether* the scheduler runs (see `step`).
    fn sleep_leader(&self, i: usize, nb: &Neighbors) -> Option<usize> {
        let v = &self.fleet.rows[i];
        if v.crossing.is_some() || v.speed >= SLEEP_SPEED_EPS {
            return None;
        }
        let li = nb.leader_of[i]?;
        let lead = &self.fleet.rows[li];
        let gap = self.corridor_gap(v, lead);
        // gap ≥ 0 so an already-overlapping pair takes the full (hard-braking) gather;
        // gap < STOP_QUEUE_GAP so a nearby stopped leader is the binding constraint.
        ((0.0..STOP_QUEUE_GAP).contains(&gap) && lead.speed < SLEEP_SPEED_EPS).then_some(li)
    }

    /// Active-set scheduler gate (isolated free-flow cars): true when car `i` is cruising
    /// open road with nothing to decide — no leader within perception, far from its node
    /// (past [`DECISION_HORIZON`]), on a straight link, heading straight through onto a
    /// same-or-faster link. For exactly this case [`gather_context`](Self::gather_context)
    /// LOD-returns just the (non-binding) leader and (absent) curve, so a bare free-road
    /// context reproduces its fold — sleeping is behaviour-preserving. `intended` is the
    /// already-resolved next hop (so this adds no router lookup).
    fn is_free_sleeper(&self, i: usize, nb: &Neighbors, intended: Option<MovementId>) -> bool {
        let v = &self.fleet.rows[i];
        let lane = self.network.lane(v.lane);
        // "Far from the node" must also mean past this car's speed-scaled leader horizon,
        // so a fast car that could just be reaching a leader across the next boundary runs
        // the full gather (and its cross-boundary walk) rather than sleeping past it.
        let clear_horizon = leader_horizon(&v.driver, v.speed).max(DECISION_HORIZON);
        if lane.length - v.position <= clear_horizon
            || !self.link_straight[lane.link.idx()]
            || self.merge_feeder_lane[v.lane.0 as usize]
        {
            return false; // near a node, on a curve, or feeding a merge → run the full gather
        }
        if let Some(li) = nb.leader_of[i] {
            let gap = self.corridor_gap(v, &self.fleet.rows[li]);
            if gap <= LEADER_PERCEPTION {
                return false; // a leader is close enough to matter
            }
        }
        match intended {
            // A turn or a slower downstream link means gather anticipates a slowdown here.
            Some(mid) => {
                matches!(self.network.movement_turn(mid), TurnType::Through)
                    && self.network.lane(self.network.movement(mid).to_lane).speed_limit >= lane.speed_limit
            }
            None => true, // no onward hop resolved — nothing downstream to anticipate
        }
    }

    /// A free-road accel context: desired-speed pursuit with no leader/node constraints —
    /// what an isolated cruising car (see [`is_free_sleeper`](Self::is_free_sleeper)) needs.
    fn free_context(&self, i: usize) -> VehicleContext {
        let veh = &self.fleet.rows[i];
        let lane = self.network.lane(veh.lane);
        VehicleContext::new(veh.driver.capped_to(lane.speed_limit), veh.speed, veh.id)
    }

    /// Vehicles skipped by the active-set scheduler on the last step (queued sleepers).
    /// Zero when the scheduler is off. A diagnostic for the UI / load bench.
    pub fn asleep_count(&self) -> usize {
        self.asleep_last
    }

    /// Toggle the active-set scheduler (see [`SimConfig::sleep_scheduler`]).
    pub fn set_sleep_scheduler(&mut self, on: bool) {
        self.cfg.sleep_scheduler = on;
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

    /// Cumulative entry count per link since the world was built — the raw counter
    /// behind [`link_flows`](Self::link_flows), so a [`measure::Measurement`] window
    /// can difference two snapshots into a windowed flow.
    pub fn link_entry_counts(&self) -> &[u32] {
        &self.link_entries
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
            stopped_at: None, wait_ticks: 0, crossing: None, lane_change: None, wreck: None, slept: false,
        });
    }

    /// Spawn at the start of a precomputed link route; the vehicle takes the
    /// route-consistent movement at each intersection and exits on the last link.
    /// Returns `false` (spawn refused) if the route is empty or the entrance is
    /// still occupied, so demand can't stack vehicles on top of each other.
    pub fn spawn_routed(&mut self, id: u32, route: Vec<LinkId>, speed: f64, driver: DriverConfig) -> bool {
        let Some(&first) = route.first() else { return false };
        // Prefer a lane that already serves the route's next link (see
        // `entry_lane_toward` — gateway arrivals enter pre-positioned).
        let lane = route
            .get(1)
            .and_then(|&next| {
                let l = self.network.link(first);
                (0..l.lane_count)
                    .map(|k| LaneId(l.lane_start.0 + (id.wrapping_add(k)) % l.lane_count))
                    .find(|&lane| {
                        (!self.network.lane_is_hov(lane) || hov_eligible(self.cfg.seed, id))
                            && self.movement_to(lane, next).is_some()
                            && self.entrance_clear(lane, driver.min_gap)
                    })
            })
            .or_else(|| self.entry_lane(first, driver.min_gap, id));
        let Some(lane) = lane else { return false };
        let speed = self.safe_entry_speed(lane, speed, &driver);
        self.link_entries[first.idx()] += 1;
        self.fleet.push(NetVehicle {
            id, lane, position: 0.0, speed, driver, route, route_idx: 0, dest: None,
            stopped_at: None, wait_ticks: 0, crossing: None, lane_change: None, wreck: None, slept: false,
        });
        true
    }

    /// Spawn on a specific lane at `pos` with an explicit route — for scenarios
    /// that need a known starting lane rather than the pre-positioned default.
    pub fn spawn_routed_in_lane(
        &mut self,
        id: u32,
        route: Vec<LinkId>,
        lane: LaneId,
        pos: f64,
        speed: f64,
        driver: DriverConfig,
    ) -> bool {
        let Some(&first) = route.first() else { return false };
        if self.network.lane(lane).link != first {
            return false;
        }
        self.link_entries[first.idx()] += 1;
        self.fleet.push(NetVehicle {
            id, lane, position: pos, speed, driver, route, route_idx: 0, dest: None,
            stopped_at: None, wait_ticks: 0, crossing: None, lane_change: None, wreck: None, slept: false,
        });
        true
    }

    /// Spawn at the start of `entry_link` bound for `dest`, routed live by the
    /// world's flow-field. Refused if every entry lane is still occupied.
    pub fn spawn_to(&mut self, id: u32, entry_link: LinkId, dest: LinkId, speed: f64, driver: DriverConfig) -> bool {
        // Admit only with the following distance the entry speed needs (min gap + one
        // headway), not just a bumper length. A freeway car entering at 29 m/s a couple
        // of metres behind another would be far inside IDM's equilibrium gap and brake to
        // a standstill the instant it appears — the "cars enter the freeway at 0 speed"
        // report. Excess demand waits at the gateway (metered), as it already does.
        let admit_gap = driver.min_gap + speed * driver.time_headway;
        let Some(lane) = self.entry_lane_toward(entry_link, admit_gap, id, Some(dest)) else { return false };
        let speed = self.safe_entry_speed(lane, speed, &driver);
        self.link_entries[entry_link.idx()] += 1;
        self.fleet.push(NetVehicle {
            id, lane, position: 0.0, speed, driver, route: Vec::new(), route_idx: 0, dest: Some(dest),
            stopped_at: None, wait_ticks: 0, crossing: None, lane_change: None, wreck: None, slept: false,
        });
        true
    }

    /// The speed a driver actually enters `lane` with: their requested speed, capped
    /// so that comfortable braking can settle them behind the current tail-of-queue
    /// (`√(v_tail² + 2·b·gap)`) — a driver joining a road matches the traffic they
    /// can see. A free entrance (no tail in sight) keeps the requested speed.
    fn safe_entry_speed(&self, lane: LaneId, speed: f64, driver: &DriverConfig) -> f64 {
        let mut tail: Option<(f64, f64)> = None; // (rear position, speed) nearest the entrance
        for v in self.fleet.rows.iter().filter(|v| v.lane == lane && v.crossing.is_none()) {
            let rear = v.position - v.driver.vehicle_length;
            if tail.is_none_or(|(r, _)| rear < r) {
                tail = Some((rear, v.speed));
            }
        }
        let Some((rear, v_tail)) = tail else { return speed };
        let gap = (rear - driver.min_gap).max(0.0);
        speed.min((v_tail * v_tail + 2.0 * driver.comfort_decel * gap).sqrt())
    }

    /// Choose an entry lane on `link` whose start is clear, spreading successive
    /// spawns across the link's lanes (the try order is rotated by `id`) so a
    /// multi-lane gateway fills every lane instead of stacking all inflow on the
    /// median lane — a prerequisite for a wide road to accept its real per-lane
    /// rush-hour volume. `None` when every lane's entrance is still occupied, which
    /// refuses the spawn and becomes natural inflow backpressure.
    fn entry_lane(&self, link: LinkId, clearance: f64, id: u32) -> Option<LaneId> {
        self.entry_lane_toward(link, clearance, id, None)
    }

    /// [`entry_lane`] with route awareness: among clear lanes, prefer one whose
    /// movement leads toward `dest` by the router's field. Traffic arriving at a
    /// map gateway is mid-journey — real drivers are already positioned for
    /// their exit — so an exit-bound trip enters curbside and a through trip
    /// enters a continuing lane, instead of random lanes forcing a full-width
    /// weave inside the map (which broke the US-101 gateway down to ⅓ capacity).
    fn entry_lane_toward(&self, link: LinkId, clearance: f64, id: u32, dest: Option<LinkId>) -> Option<LaneId> {
        let l = self.network.link(link);
        let n = l.lane_count;
        let candidates = (0..n)
            .map(|k| LaneId(l.lane_start.0 + (id.wrapping_add(k)) % n))
            .filter(|&lane| {
                (!self.network.lane_is_hov(lane) || hov_eligible(self.cfg.seed, id))
                    && self.entrance_clear(lane, clearance)
            });
        if let (Some(d), Some(router)) = (dest, self.router.as_ref()) {
            let score = |lane: LaneId| -> u64 {
                self.network
                    .movements_of(lane)
                    .iter()
                    .filter_map(|m| router.distance(d, self.network.lane(m.to_lane).link))
                    .min()
                    .unwrap_or(u64::MAX)
            };
            let mut best: Option<(u64, LaneId)> = None;
            for lane in candidates {
                let s = score(lane);
                if best.is_none_or(|(bs, _)| s < bs) {
                    best = Some((s, lane));
                }
            }
            return best.map(|(_, lane)| lane);
        }
        candidates.into_iter().next()
    }

    /// Test-only: spawn a destination-routed vehicle in a specific lane at a
    /// specific position, so a scenario can place a car where it cannot reach a
    /// lane serving its route.
    #[cfg(test)]
    pub fn spawn_to_in_lane(&mut self, id: u32, lane: LaneId, position: f64, dest: LinkId, speed: f64, driver: DriverConfig) {
        self.link_entries[self.network.lane(lane).link.idx()] += 1;
        self.fleet.push(NetVehicle {
            id, lane, position, speed, driver, route: Vec::new(), route_idx: 0, dest: Some(dest),
            stopped_at: None, wait_ticks: 0, crossing: None, lane_change: None, wreck: None, slept: false,
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

    /// Vehicles wrecked in a collision (tallied at impact; wrecks may still be
    /// on the road awaiting clearance).
    pub fn crashed(&self) -> u32 {
        self.crashed
    }

    /// The `crashed` tally split by [`CrashKind`]: `[rear_end, junction]`.
    pub fn crash_counts(&self) -> [u32; 2] {
        self.crashed_by
    }

    /// Recent crashes with their nature (kind, closing speed, position) — for the
    /// overlay and for judging artifact vs plausible collision.
    pub fn crash_log(&self) -> &[CrashRecord] {
        &self.crash_log
    }

    /// Forget the recorded crash log (the overlay clears); the `crashed` tallies
    /// are untouched.
    pub fn clear_crash_sites(&mut self) {
        self.crash_log.clear();
    }

    /// Live toggle for wreck persistence: how long a crashed vehicle stays on the
    /// road as an obstruction (0 = removed instantly, the default). Applies to
    /// crashes from the next step on; existing wrecks keep their timers.
    pub fn set_wreck_clear_secs(&mut self, secs: f64) {
        self.cfg.wreck_clear_secs = secs.max(0.0);
    }

    pub fn wreck_clear_secs(&self) -> f64 {
        self.cfg.wreck_clear_secs
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

    /// A crossing vehicle's arc-length into the node interior. `position` counts
    /// continuously past the from-lane's end, so this is simply the overrun. Only
    /// meaningful while `crossing` is set.
    fn crossing_arc(&self, v: &NetVehicle) -> f64 {
        v.position - self.network.lane(v.lane).length
    }

    /// A vehicle's continuous arc-length along its corridor (see `corridor_of`): the lane's
    /// offset within the corridor plus its position. Two cars on different lanes of the same
    /// corridor compare directly in this coordinate, so following spans the seam.
    fn corridor_pos(&self, v: &NetVehicle) -> f64 {
        self.corridor_offset[v.lane.0 as usize] + v.position
    }

    /// Bumper-to-bumper gap from `follower` to `leader` when they share a corridor — measured
    /// in the continuous corridor coordinate, so it is correct even across a segment seam.
    fn corridor_gap(&self, follower: &NetVehicle, leader: &NetVehicle) -> f64 {
        self.corridor_pos(leader) - self.corridor_pos(follower) - leader.driver.vehicle_length
    }

    /// Whether a movement stays within one corridor (a grade-separated 1:1 continuation).
    /// Such a seam is not a junction: a car flows across it with no admission gate, its
    /// spacing already governed by continuous corridor following. Every other movement (a
    /// merge, diverge, lane-drop, or at-grade crossing) is a real junction and stays gated.
    fn is_intra_corridor(&self, mid: MovementId) -> bool {
        let m = self.network.movement(mid);
        self.corridor_of[m.from_lane.0 as usize] == self.corridor_of[m.to_lane.0 as usize]
    }

    /// A boundary crossed with no admission gate: a within-corridor seam, or a
    /// continuation seam taken by the exit link's through approach (see
    /// `seam_primary`). Only the landing-overrun guard applies at these.
    fn free_flow_seam(&self, mid: MovementId) -> bool {
        self.is_intra_corridor(mid) || self.seam_primary[mid.idx()]
    }


    /// A vehicle's current world pose `[x, y, heading]` — its interior crossing
    /// path when inside a node, otherwise its lane position.
    pub fn vehicle_world_pose(&self, v: &NetVehicle) -> [f64; 3] {
        if let Some(c) = v.crossing {
            let s = self.crossing_arc(v).clamp(0.0, self.network.interior(c.movement).len);
            return self.network.interior_point(c.movement, s);
        }
        let cur = self.network.lane_point(v.lane, v.position);
        let Some(lc) = v.lane_change else { return cur };
        let arc = self.network.lane(v.lane).start_offset + v.position;
        let from_lane = self.network.lane(lc.from);
        let from_pos = arc - from_lane.start_offset;
        if from_pos < 0.0 || from_pos > from_lane.length {
            return cur;
        }
        let from = self.network.lane_point(lc.from, from_pos);
        let t = lc.progress.clamp(0.0, 1.0);
        [from[0] + (cur[0] - from[0]) * t, from[1] + (cur[1] - from[1]) * t, cur[2]]
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
        // wrong lane must never vanish mid-network. Among the lane's movements pick
        // the one landing nearest the destination (`forward_movement`) rather than
        // an arbitrary first movement, so a wrong-lane car makes forward progress
        // instead of taking a backtracking turn and looping. If this lane serves
        // nothing, borrow a sibling lane's movement rather than dropping the car;
        // only a genuine dead end (no lane on the link moves on) truly exits.
        preferred
            .or_else(|| self.forward_movement(veh, veh.lane))
            .or_else(|| (lane.movement_count > 0).then_some(lane.movement_start))
            .or_else(|| self.any_movement_on(lane.link))
    }

    /// Of the movements this lane can actually take, the one landing on the link
    /// nearest the vehicle's destination by the router's field — the forward-most
    /// fallback when the lane can't serve the routed next link. Keeps a wrong-lane
    /// car progressing toward its goal instead of taking an arbitrary (possibly
    /// backtracking) movement and looping back. `None` for non-routed cars or when
    /// no movement reaches the destination.
    fn forward_movement(&self, veh: &NetVehicle, lane: LaneId) -> Option<MovementId> {
        let (dest, router) = (veh.dest?, self.router.as_ref()?);
        let l = self.network.lane(lane);
        (0..l.movement_count)
            .filter_map(|k| {
                let mid = MovementId(l.movement_start.0 + k);
                let to_link = self.network.lane(self.network.movement(mid).to_lane).link;
                router.distance(dest, to_link).map(|d| (d, mid))
            })
            .min_by_key(|&(d, _)| d)
            .map(|(_, mid)| mid)
    }

    /// Which way a vehicle is signalling a lane change: `-1` left, `+1` right,
    /// `0` none. A car whose current lane can't serve its route signals toward the
    /// nearest lane that can (the turn lane it needs to merge into) — higher lane
    /// index is further right, so the sign of the index gap is the physical turn
    /// side. Read only by the renderer to blink an indicator; never affects the sim.
    pub fn vehicle_blinker(&self, veh: &NetVehicle) -> i8 {
        if veh.crossing.is_some() || self.lane_serves_route(veh, veh.lane) != Some(false) {
            return 0; // mid-junction, or the lane already serves the route
        }
        let Some(next) = self.next_link_on_path(veh) else { return 0 };
        let link = self.network.link(self.network.lane(veh.lane).link);
        let cur = self.network.lane(veh.lane).index_in_link as i64;
        let goal = (0..link.lane_count as i64)
            .filter(|&k| {
                let l = LaneId(link.lane_start.0 + k as u32);
                self.network.movements_of(l).iter().any(|m| self.network.lane(m.to_lane).link == next)
            })
            .min_by_key(|&k| (k - cur).abs());
        goal.map_or(0, |g| (g - cur).signum() as i8)
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

    /// Don't-block-the-box, extended across a whole multi-node junction. When the
    /// intended movement crosses into a link that stays within the same junction
    /// cluster (a sibling-node hop), the car must be able to reach the far side of
    /// the junction before it enters — otherwise it halts on an internal link, sitting
    /// in the box in the way of cross traffic. Follows the route through the internal
    /// links; `true` when the exit past an internal node is occupied, so the caller
    /// holds the car at the junction's outer stop line instead of on the inside.
    fn junction_exit_blocked(&self, veh: &NetVehicle, mid: MovementId, node: NodeId, nb: &Neighbors) -> bool {
        let Some(jid) = self.network.node_junction(node) else { return false };
        let mut mv = mid;
        // Deep enough to cross the biggest split junction's sliver chains — a walk
        // that gives up early admits cars into a cluster whose far side is jammed.
        for _ in 0..8 {
            let to_lane = self.network.movement(mv).to_lane;
            let to_link = self.network.lane(to_lane).link;
            if self.network.node_junction(self.network.link(to_link).to) != Some(jid) {
                // This hop leaves the junction — and the car truly clears it only if
                // the exit street can receive its whole body, with spare-car slack for
                // the queue advance that happens during the multi-second traversal.
                // A *departing* occupant is a leader to follow, not a blockage (the
                // same exemption as `receiving_room` — without it every queued
                // crossing serialized to one car per ~7 s).
                return nb.lane_front.get(&to_lane.0).is_some_and(|&f| {
                    let o = &self.fleet.rows[f];
                    let rear = o.position - o.driver.vehicle_length;
                    // The exemption rides the chain: a car entering on a free-flow
                    // seam (`mid`) must not be throttled by a conservative margin
                    // at a deeper hop of the same traversal.
                    if o.speed >= DEPARTING_SPEED
                        && (self.departing_exemption(mv) || self.departing_exemption(mid))
                    {
                        return self.network.interior(mv).len + rear < departing_margin(&veh.driver, o.speed);
                    }
                    let unit = veh.driver.vehicle_length + veh.driver.min_gap;
                    rear < self.commit_room_needed(to_lane, unit)
                });
            }
            // The internal lane must have a free slot for this car — counting the
            // crossers already in flight toward it, who are still indexed on their
            // approach lane. Without them, several approaches admit into the same
            // one-car internal stub in the same window, the interior overfills, and
            // the junction's internal ring can gridlock permanently.
            if self.internal_lane_full(to_lane, veh, nb) {
                return true;
            }
            let Some(next) = self.intended_movement_from(veh, to_lane) else { return false };
            let onward = self.network.movement(next).to_lane;
            let blocked = nb.lane_front.get(&onward.0).is_some_and(|&f| {
                let o = &self.fleet.rows[f];
                o.position - o.driver.vehicle_length < veh.driver.vehicle_length + veh.driver.min_gap
            });
            if blocked {
                return true; // the far side of the internal node is occupied
            }
            mv = next;
        }
        false
    }

    /// Whether a junction-internal lane has no room for one more `veh`-sized car:
    /// its landed occupants plus crossers currently in flight toward it fill every
    /// `vehicle_length + min_gap` slot the lane's drivable span holds (always at
    /// least one — internal stubs are often shorter than a car).
    fn internal_lane_full(&self, lane: LaneId, veh: &NetVehicle, nb: &Neighbors) -> bool {
        let unit = veh.driver.vehicle_length + veh.driver.min_gap;
        let slots = (self.network.lane(lane).length / unit).floor().max(1.0) as usize;
        let landed = nb
            .by_lane
            .get(&lane.0)
            .map_or(0, |v| v.iter().filter(|&&i| self.fleet.rows[i].crossing.is_none()).count());
        let entry = self.network.link(self.network.lane(lane).link).from;
        let inbound = nb
            .crossing_at
            .get(&self.network.intersection_key(entry))
            .map_or(0, |v| {
                v.iter()
                    .filter(|&&i| {
                        self.fleet.rows[i]
                            .crossing
                            .is_some_and(|c| self.network.movement(c.movement).to_lane == lane)
                    })
                    .count()
            });
        landed + inbound >= slots
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
        let mut by_corridor: IntMap<Vec<usize>> = IntMap::default();
        let mut approaching: IntMap<Vec<usize>> = IntMap::default();
        let mut crossing_at: IntMap<Vec<usize>> = IntMap::default();
        let mut crossing_mvs: IntMap<Vec<MovementId>> = IntMap::default();
        let mut moving_crossing_mvs: IntMap<Vec<MovementId>> = IntMap::default();
        for (i, v) in self.fleet.rows.iter().enumerate() {
            // Every car (crossers included, at their continuous corridor position) joins the
            // leader chain for its corridor, so a follower keeps its leader across a seam.
            by_corridor.entry(self.corridor_of[v.lane.0 as usize]).or_default().push(i);
            if let Some(c) = v.crossing {
                let node = self.network.movement(c.movement).node;
                let key = self.network.intersection_key(node);
                crossing_at.entry(key).or_default().push(i);
                let mut path = vec![c.movement];
                let mut lane = self.network.movement(c.movement).to_lane;
                for _ in 0..4 {
                    if !self.junction_internal_lane(lane) {
                        break;
                    }
                    let Some(next) = self.movement_from_lane_for(v, lane) else { break };
                    path.push(next);
                    lane = self.network.movement(next).to_lane;
                }
                if v.speed >= 0.5 {
                    moving_crossing_mvs.entry(key).or_default().extend(path.iter().copied());
                }
                crossing_mvs.entry(key).or_default().extend(path);
                by_lane.entry(v.lane.0).or_default().push(i);
                continue;
            }
            by_lane.entry(v.lane.0).or_default().push(i);
            approaching.entry(self.network.intersection_key(self.downstream_node(v.lane))).or_default().push(i);
        }
        // Flat sort keys (contiguous, cache-friendly) precomputed once when the cache-sort
        // option is on; empty when off, so the helpers read each vehicle row instead.
        let (pos, cpos) = (self.position_keys(), self.corridor_keys());
        // The leader chain runs along the whole corridor (grade-separated 1:1 through-lanes
        // coalesced), so `leader_of` never loses the car ahead at a segment boundary.
        let mut leader_of = vec![None; self.fleet.rows.len()];
        for members in by_corridor.values_mut() {
            self.sort_corridor_members(members, &cpos);
            for w in members.windows(2) {
                leader_of[w[0]] = Some(w[1]);
            }
        }
        // Nearest-to-entrance car per physical lane — for the box gate and lateral checks,
        // which stay per-lane (a lane change targets a physical lane, not a corridor).
        let mut lane_front: IntMap<usize> = IntMap::default();
        for members in by_lane.values_mut() {
            self.sort_lane_members(members, &pos);
            let front = *members.first().unwrap();
            lane_front.insert(self.fleet.rows[front].lane.0, front);
        }
        Neighbors { leader_of, lane_front, by_lane, approaching, crossing_at, crossing_mvs, moving_crossing_mvs }
    }

    fn downstream_node(&self, lane: LaneId) -> NodeId {
        self.network.link(self.network.lane(lane).link).to
    }

    /// A strict priority order over links (higher wins): functional class first
    /// (the primary right-of-way determinant — see [`RoadKind::at_grade_rank`]),
    /// then speed, then lanes, then link id as a deterministic tie-break so
    /// opposing yields can never deadlock. `>> 24` drops the id, leaving the
    /// road-rank prefix the stop/merge rules compare.
    fn priority_key(&self, link: LinkId) -> u64 {
        let l = self.network.link(link);
        let lane = self.network.lane(l.lane_start);
        l.kind.at_grade_rank() << 56
            | ((lane.speed_limit * 1000.0) as u64) << 40
            | (l.lane_count as u64) << 24
            | (link.0 as u64 & 0xFF_FFFF)
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
        let pos = self.position_keys();
        for m in by_lane.values_mut() {
            self.sort_lane_members(m, &pos);
        }
        // Decide each car's best lane change. This is the expensive MOBIL scan and it reads only
        // committed state (positions + the sorted `by_lane`), so it parallelizes across cores
        // bit-for-bit: `map_collect` is order-preserving, so the flattened decisions match the
        // serial order exactly. The apply below stays serial — a slot-clear check reads state the
        // earlier applies mutate, so two cars can't be cleared into the same gap.
        // The MOBIL scan's per-car work is far lighter than the accel gather, so its
        // parallel crossover sits much later — measured on the loaded real map the
        // parallel arm *loses* below ~8k cars (dispatch + collect overhead beats the
        // work split). It gets its own floor rather than riding `par_threshold`.
        let threshold = self.par_threshold.max(LIGHT_PAR_THRESHOLD);
        let (backend, n) = (self.active_backend(), self.fleet.rows.len());
        let decided: Vec<Option<(usize, LaneId)>> = map_collect(backend, threshold, n, |i| {
            if self.fleet.rows[i].wreck.is_some() {
                return None; // wrecks don't change lanes
            }
            // A sleeping queued car re-evaluates its queue-jump on a slow cadence
            // (staggered by id so wakes spread over ticks): a parked car's lane
            // decision can't change tick-to-tick, and at gridlock a third of the
            // fleet is parked — this is the scheduler composing with the
            // lane-change pass, not just the accel gather.
            {
                let v = &self.fleet.rows[i];
                if v.slept
                    && v.speed < SLEEP_SPEED_EPS
                    && (self.tick.wrapping_add(v.id as u64)) % SLEEPER_LC_PERIOD != 0
                {
                    return None;
                }
            }
            // Cars on a congested (queue-mode) link skip lane-change evaluation — negligible
            // movement in a jam for a costly scan.
            if self.congestion_cfg.enabled {
                let link = self.network.lane(self.fleet.rows[i].lane).link;
                if self.congestion.is_queue(link.idx()) {
                    return None;
                }
            }
            self.best_lane_change(i, &by_lane).map(|t| (i, t))
        });
        for (i, target) in decided.into_iter().flatten() {
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
            let speed = self.fleet.rows[i].speed;
            if self.lane_slot_clear(target, new_pos, len, speed, i) {
                let from = self.fleet.rows[i].lane;
                self.fleet.rows[i].lane = target;
                self.fleet.rows[i].position = new_pos;
                self.fleet.rows[i].lane_change = Some(LaneChange { from, progress: 0.0 });
                // New lane → new leader; the retained history is in the old lane's
                // frame, so discard it (as a segment crossing does) to keep the
                // reaction-delay gap from reading a stale position and phantom-braking.
                self.fleet.reset_history(i, new_pos, speed);
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

        // Which way to the nearest lane that carries the route onward — but only once
        // within positioning distance of the node. A freeway exit lane is exit-only, so a
        // through car in it must move over (possibly across an intervening exit lane) before
        // the gore; far upstream it may still use that lane, so this stays quiet there.
        // The window scales with how many lanes remain to cross — one full window
        // apiece — so a car three lanes from its pocket starts working over early
        // instead of weaving everything into the last forty metres.
        let to_node = lane.length - v.position;
        let position_dist = (v.speed * LANE_POSITION_LEAD).max(LANE_POSITION_MIN);
        let need = (to_node < position_dist * MAX_POSITION_WINDOWS)
            .then(|| self.lanes_to_serving(v))
            .flatten()
            .filter(|d| to_node < position_dist * (d.abs() as f64).max(1.0));

        let mut best: Option<(f64, LaneId)> = None;
        for delta in [-1i64, 1] {
            let ti = idx + delta;
            if ti < 0 || ti >= link.lane_count as i64 {
                continue;
            }
            let target = LaneId(link.lane_start.0 + ti as u32);
            if self.network.lane_is_hov(target) && !hov_eligible(self.cfg.seed, v.id) {
                continue; // express/HOV lane: ineligible vehicles never target it
            }
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

            let mandatory = match need {
                Some(d) => d != 0 && d.signum() == delta.signum(),
                None => self.mandatory_change(v, v.lane, target),
            };
            // Never drift *voluntarily* into a lane that serves the route worse —
            // measured by the same landing-chain depth `lanes_to_serving`
            // positions by, so keep-right can't nudge a car out of the lane it
            // just pre-positioned into (they'd oscillate), nor a through car
            // into an exit-only lane it must scramble back out of at the gore.
            if !mandatory && to_node < position_dist * MAX_POSITION_WINDOWS {
                if let Some(next) = self.next_link_on_path(v) {
                    let hops = self.path_hops(v, next);
                    if self.lane_chain_depth(target, &hops) < self.lane_chain_depth(v.lane, &hops) {
                        continue;
                    }
                }
            }
            let params = {
                let mut p = MobilParams::new(v.driver.politeness);
                if mandatory {
                    // Forcing urgency mirrors gap-acceptance impatience
                    // (`effective_critical_gap`): with the junction closing in
                    // and lanes still to cross, the driver accepts imposing
                    // harder braking on the new follower rather than missing
                    // the turn.
                    let urgency = (1.0 - to_node / position_dist).clamp(0.0, 1.0);
                    p.safe_braking += 2.0 * urgency;
                }
                p
            };
            let bias = if mandatory {
                0.0
            } else if delta > 0 {
                KEEP_RIGHT_BIAS
            } else {
                -KEEP_RIGHT_BIAS
            };
            if mobil::should_change(&params, a_self_cur, a_self_new, a_nf_cur, a_nf_new, mandatory, bias) {
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

    fn lanes_to_serving(&self, veh: &NetVehicle) -> Option<i64> {
        let next = self.next_link_on_path(veh)?;
        let link = self.network.link(self.network.lane(veh.lane).link);
        let cur = self.network.lane(veh.lane).index_in_link as i64;
        // Lane-level route preference, `LANE_ROUTE_DEPTH` junctions deep: score
        // each lane by how far its *landing-lane chain* follows the upcoming
        // hops without a forced weave, and position for the deepest chain any
        // lane offers. A lane that merely reaches the next link but strands the
        // car in the wrong lane there owes a weave on a possibly short block —
        // the last-second turn-lane miss, pushed one block upstream per depth
        // level. Graceful per-depth fallback: when no lane reaches depth d, the
        // depth-(d−1) set serves (the later weave is then unavoidable). This is
        // lane-level routing evaluated on demand along the path; the standing
        // lane-graph route search remains open.
        let hops = self.path_hops(veh, next);
        let depth_of = |k: i64| {
            let l = LaneId(link.lane_start.0 + k as u32);
            self.lane_chain_depth(l, &hops)
        };
        let best = (0..link.lane_count as i64).map(depth_of).max().unwrap_or(0);
        if best == 0 {
            return None; // nothing reaches the next link from this carriageway
        }
        if depth_of(cur) == best {
            return Some(0);
        }
        (0..link.lane_count as i64)
            .filter(|&k| depth_of(k) == best)
            .min_by_key(|&k| (k - cur).abs())
            .map(|k| k - cur)
    }

    /// The next up-to-[`LANE_ROUTE_DEPTH`] links on this vehicle's path,
    /// starting with `next` — from its explicit route, or by chaining the
    /// flow-field's next hops.
    fn path_hops(&self, veh: &NetVehicle, next: LinkId) -> Vec<LinkId> {
        let mut hops = vec![next];
        if !veh.route.is_empty() {
            for d in 2..=LANE_ROUTE_DEPTH {
                match veh.route.get(veh.route_idx + d) {
                    Some(&l) => hops.push(l),
                    None => break,
                }
            }
            return hops;
        }
        if let (Some(dest), Some(router)) = (veh.dest, self.router.as_ref()) {
            while hops.len() < LANE_ROUTE_DEPTH {
                match router.next_hop(dest, *hops.last().unwrap()) {
                    Some(l) => hops.push(l),
                    None => break,
                }
            }
        }
        hops
    }

    /// How many of `hops` a car in `lane` can follow by pure movement landings
    /// (no lane change): the depth its committed chain serves.
    fn lane_chain_depth(&self, lane: LaneId, hops: &[LinkId]) -> usize {
        let Some(&hop) = hops.first() else { return 0 };
        self.network
            .movements_of(lane)
            .iter()
            .filter(|m| self.network.lane(m.to_lane).link == hop)
            .map(|m| 1 + self.lane_chain_depth(m.to_lane, &hops[1..]))
            .max()
            .unwrap_or(0)
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

    fn lane_slot_clear(&self, target: LaneId, pos: f64, len: f64, speed: f64, exclude: usize) -> bool {
        // Require more than a bumper's clearance at speed: the changing car needs room to
        // follow whoever is ahead in the slot, and the car behind needs room to follow it
        // — roughly a half-second headway each — so a change into fast traffic can't drop a
        // car a few centimetres off a leader that then brakes. Below this a mandatory
        // change simply waits for a real gap instead of forcing an unsafe merge.
        //
        // The slot is judged along the whole *corridor*, in continuous corridor coordinates:
        // a car merging near a segment seam must clear the follower one segment back (same
        // corridor, different lane), not just the cars physically on the target lane — else it
        // cuts in a few metres ahead of a fast car on the previous link and slams it.
        const HEADWAY: f64 = 0.5;
        let target_corridor = self.corridor_of[target.0 as usize];
        let my_cpos = self.corridor_offset[target.0 as usize] + pos;
        self.fleet.rows
            .iter()
            .enumerate()
            .filter(|(j, o)| *j != exclude && self.corridor_of[o.lane.0 as usize] == target_corridor)
            .all(|(_, o)| {
                let o_cpos = self.corridor_pos(o);
                if o_cpos > my_cpos {
                    // Room ahead: the changer must be able to brake to the leader's speed.
                    let closing = (speed * speed - o.speed * o.speed).max(0.0) / (2.0 * MAX_BRAKE_DECEL);
                    o_cpos - o.driver.vehicle_length - my_cpos > 0.5 + speed * HEADWAY + closing
                } else {
                    // Room behind: never drop in so close/slow that the follower must brake
                    // harder than physically possible to avoid the changer (the cut-in slam).
                    let closing = (o.speed * o.speed - speed * speed).max(0.0) / (2.0 * MAX_BRAKE_DECEL);
                    my_cpos - len - o_cpos > 0.5 + o.speed * HEADWAY + closing
                }
            })
    }

    /// Current colour of a movement under the actuated signal runtime
    /// (unsignalized movements are always green).
    fn movement_state(&self, mid: MovementId) -> SignalState {
        self.signals.movement_state(&self.network, mid)
    }

    fn signal_green_elapsed(&self, mid: MovementId) -> f64 {
        self.signals.green_elapsed(&self.network, mid)
    }

    /// Colour of every signal group, indexed by group id — for rendering.
    pub fn signal_states(&self) -> Vec<SignalState> {
        self.signals.states(&self.network)
    }

    /// Detect stop-line demand (vehicles within the detector zone) and advance
    /// the actuated signals.
    fn advance_signals(&mut self, dt: f64) {
        // Per-lane detection, like a real stop-line loop: a car calls only the
        // groups its *lane* feeds, so a through queue can't call the adjacent
        // bay's protected-left phase.
        let mut demand: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for v in &self.fleet.rows {
            let lane = self.network.lane(v.lane);
            if lane.length - v.position < junction::DETECT {
                demand.insert(v.lane.0);
            }
        }
        // Rail preemption: while a crossing is closed, adjacent signals force
        // the phase that flushes the from-crossing approach away from the tracks.
        let mut forced: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for &(pid, phase, crossing) in &self.rail_preempts {
            if self.rail_closed(crossing) {
                forced.insert(pid as usize, phase);
            }
        }
        self.signals.advance(&self.network, &demand, dt, &forced);
        self.advance_meters(dt);
        self.advance_bus_stops(dt);
    }

    /// Map each rail crossing to the adjacent signal phase that clears traffic
    /// *away from* the tracks: for every link departing a crossing into a nearby
    /// signalized node, the phase greening that approach's movements.
    fn build_rail_preempts(network: &Network) -> Vec<(u32, usize, NodeId)> {
        let mut out = Vec::new();
        for r in 0..network.nodes.len() {
            if !network.nodes[r].rail_crossing {
                continue;
            }
            for li in 0..network.links.len() {
                let l = network.link(LinkId(li as u32));
                if l.from != NodeId(r as u32) {
                    continue;
                }
                let lane = network.lane(l.lane_start);
                if lane.length > 120.0 {
                    continue; // not adjacent — the queue can't back onto the tracks
                }
                for k in 0..lane.movement_count {
                    let mid = MovementId(lane.movement_start.0 + k);
                    let Some(gid) = network.movement(mid).signal_group else { continue };
                    let group = network.groups[gid.idx()];
                    let program = &network.programs[group.program.idx()];
                    if let Some(phase) =
                        program.phases.iter().position(|ph| ph.green_mask & (1u64 << group.bit) != 0)
                    {
                        out.push((group.program.idx() as u32, phase, NodeId(r as u32)));
                    }
                }
            }
        }
        out.sort_unstable_by_key(|&(p, ph, n)| (p, ph, n.0));
        out.dedup();
        out
    }

    /// Turn ramp metering on or off (Caltrans D4 meters the peninsula freeway
    /// on-ramps at peak). Meters are detected on first enable: every ramp-class
    /// link whose exit movement merges onto a freeway mainline gets a one-car-
    /// per-green signal at its line, paced by ALINEA off the mainline occupancy.
    pub fn set_ramp_metering(&mut self, on: bool) {
        if on && self.meters.is_empty() {
            self.meters = self.build_ramp_meters();
        }
        self.metering_on = on;
    }

    pub fn ramp_metering(&self) -> bool {
        self.metering_on
    }

    /// Metered on-ramps discovered on the network (0 until first enabled).
    pub fn ramp_meter_count(&self) -> usize {
        self.meters.len()
    }

    fn build_ramp_meters(&self) -> Vec<RampMeter> {
        let mut out = Vec::new();
        for i in 0..self.network.links.len() as u32 {
            let link = LinkId(i);
            if self.network.link(link).kind != RoadKind::Ramp {
                continue;
            }
            // The ramp's merge: a movement from this link onto freeway mainline.
            let merge = self.network.lanes_of(link).find_map(|l| {
                self.network.movements_of(l).iter().find_map(|m| {
                    let to = self.network.lane(m.to_lane).link;
                    (self.network.link(to).kind == RoadKind::Freeway).then_some((m.node, to))
                })
            });
            if let Some((node, downstream)) = merge {
                // Occupancy detector: the mainline link *feeding* the merge — at
                // link granularity that's where merge congestion registers (the
                // downstream link drains at its own capacity and stays thin).
                let feeder = (0..self.network.links.len() as u32).map(LinkId).find(|&f| {
                    f != link && self.network.link(f).kind == RoadKind::Freeway && self.network.link(f).to == node
                });
                out.push(RampMeter {
                    ramp: link,
                    mainline: feeder.unwrap_or(downstream),
                    rate_vph: 900.0,
                    cycle_t: 0.0,
                    since_update: 0.0,
                });
            }
        }
        out
    }

    fn advance_meters(&mut self, dt: f64) {
        if !self.metering_on || self.meters.is_empty() {
            return;
        }
        // Mainline occupancy per metered mainline (count / jam capacity).
        let mut count: IntMap<u32> = IntMap::default();
        for v in &self.fleet.rows {
            *count.entry(self.network.lane(v.lane).link.0).or_default() += 1;
        }
        for k in 0..self.meters.len() {
            let m = &self.meters[k];
            let (mainline, mut rate, mut cycle_t, mut since) = (m.mainline, m.rate_vph, m.cycle_t, m.since_update);
            since += dt;
            if since >= ALINEA_PERIOD_SECS {
                since = 0.0;
                let l = self.network.link(mainline);
                let jam = (self.network.lane(l.lane_start).length / 7.0 * l.lane_count as f64).max(1.0);
                let occ = count.get(&mainline.0).copied().unwrap_or(0) as f64 / jam;
                rate = (rate + ALINEA_GAIN_VPH * (ALINEA_SETPOINT - occ)).clamp(METER_MIN_VPH, METER_MAX_VPH);
            }
            cycle_t += dt;
            if cycle_t >= 3600.0 / rate {
                cycle_t = 0.0;
            }
            let m = &mut self.meters[k];
            (m.rate_vph, m.cycle_t, m.since_update) = (rate, cycle_t, since);
        }
    }

    /// Feed the simulated time of day (seconds since midnight) for day-scheduled
    /// infrastructure — the rail-crossing timetable reads it, and the signal
    /// system switches its green-wave plan (AM progression up to noon, PM
    /// progression after, the way real corridor timing plans rotate).
    pub fn set_day_secs(&mut self, s: f64) {
        self.day_secs = s.rem_euclid(86_400.0);
        let pm = self.day_secs >= 12.0 * 3600.0;
        if pm != self.pm_plan && !self.network.am_offsets.is_empty() {
            self.pm_plan = pm;
            for p in 0..self.network.programs.len() {
                if self.network.programs[p].coordinated {
                    self.network.programs[p].offset =
                        if pm { self.network.pm_offsets[p] } else { self.network.am_offsets[p] };
                }
            }
        }
    }

    /// Whether a rail crossing at `node` is currently closed to road traffic.
    /// The timetable is a pure function of the day clock: Caltrain-like cadence
    /// (≈10 closures/h across both directions at the commute peaks, 4/h midday,
    /// 1/h overnight), each closure [`RAIL_CLOSURE_SECS`] of day time.
    fn rail_closed(&self, node: NodeId) -> bool {
        if !self.network.node(node).rail_crossing {
            return false;
        }
        let h = self.day_secs / 3600.0;
        let per_hour = if (6.0..9.0).contains(&h) || (16.0..19.0).contains(&h) {
            10.0
        } else if h < 5.0 {
            1.0
        } else {
            4.0
        };
        self.day_secs.rem_euclid(3600.0 / per_hour) < RAIL_CLOSURE_SECS
    }

    /// Distance to the line a bus must hold at for stop service: its next
    /// unserved stop on this link (approach braking), or its own position while
    /// the dwell timer runs. `None` for non-buses and between stops.
    fn bus_stop_line(&self, veh: &NetVehicle, lane: &Lane) -> Option<f64> {
        if veh.driver.vehicle_length < 11.0 || veh.crossing.is_some() {
            return None;
        }
        let (until, last_link, last_arc) = self.bus_dwell.get(&veh.id).copied().unwrap_or((0, u32::MAX, f64::MIN));
        if self.tick < until {
            return Some(0.05); // parked at the stop, serving
        }
        let arc = lane.start_offset + veh.position;
        self.stops_by_link
            .get(&lane.link.0)
            .into_iter()
            .flatten()
            .find(|&&(_, pos)| {
                pos > arc - 0.5 && !(last_link == lane.link.0 && pos <= last_arc + 4.0)
            })
            .map(|&(_, pos)| (pos - arc).max(0.05))
    }

    /// Advance bus service state: a bus arrived at an unserved stop starts its
    /// dwell; expired dwells mark the stop served so the bus departs.
    fn advance_bus_stops(&mut self, dt: f64) {
        if self.stops_by_link.is_empty() {
            return;
        }
        for v in &self.fleet.rows {
            if v.driver.vehicle_length < 11.0 || v.crossing.is_some() {
                continue;
            }
            let lane = self.network.lane(v.lane);
            let arc = lane.start_offset + v.position;
            let (until, last_link, last_arc) = self.bus_dwell.get(&v.id).copied().unwrap_or((0, u32::MAX, f64::MIN));
            if self.tick < until || v.speed > 0.5 {
                continue;
            }
            if let Some(&(_, pos)) = self
                .stops_by_link
                .get(&lane.link.0)
                .into_iter()
                .flatten()
                // IDM's brake-to-line settles the nose ~min_gap short of the
                // mark, so "arrived" tolerates that standoff; positional
                // last-served state means nothing at-or-before it re-triggers.
                .find(|&&(_, pos)| {
                    (pos - arc).abs() < 3.5 && !(last_link == lane.link.0 && pos <= last_arc + 4.0)
                })
            {
                self.bus_dwell.insert(v.id, (self.tick + (BUS_DWELL_SECS / dt) as u64, lane.link.0, pos));
            }
        }
        // Drop state for buses that have left the network.
        if self.tick % 1024 == 0 {
            let live: std::collections::HashSet<u32> = self.fleet.rows.iter().map(|v| v.id).collect();
            self.bus_dwell.retain(|id, _| live.contains(id));
        }
    }

    /// Whether `link`'s ramp meter currently shows red (holding the line).
    fn meter_red(&self, link: LinkId) -> bool {
        if !self.metering_on {
            return false;
        }
        self.meters
            .iter()
            .find(|m| m.ramp == link)
            .is_some_and(|m| m.cycle_t >= METER_GREEN_SECS)
    }

    /// The commanded rate (veh/h) of `link`'s meter, for tests and UI readouts.
    pub fn meter_rate(&self, link: LinkId) -> Option<f64> {
        self.meters.iter().find(|m| m.ramp == link).map(|m| m.rate_vph)
    }

    /// Order-independent hash of the links carrying enough traffic to shift routing costs —
    /// the only thing a flow-field recompute reacts to. O(cars): tally cars per occupied link
    /// (light traffic touches few links), keep those whose density crosses into a congestion
    /// band, and fold their `(link, band)` into the hash. Empty and stable under light or
    /// static traffic, so [`refresh_routes`] (and the browser GPU router) skip the rebuild.
    pub fn congestion_fingerprint(&self) -> u64 {
        let mut count: IntMap<u32> = IntMap::default();
        for v in &self.fleet.rows {
            *count.entry(self.network.lane(v.lane).link.0).or_default() += 1;
        }
        let mut h = 0u64;
        for (&link, &c) in &count {
            let l = self.network.link(LinkId(link));
            let jam = (self.network.lane(l.lane_start).length / 7.0 * l.lane_count as f64).max(1.0);
            // Quarter-jam bands: a tiny density change doesn't churn the fingerprint (and so
            // doesn't trigger a rebuild); a link crossing into a busier band does.
            let band = ((c as f64 / jam) * 4.0) as u64;
            if band > 0 {
                h = h.wrapping_add((link as u64).wrapping_mul(0x9e3779b97f4a7c15) ^ (band + 1).wrapping_mul(0x100000001b3));
            }
        }
        h
    }

    /// Toggle solving several routing fields per reroute across cores. Off falls back to one
    /// field at a time (still frame-spread, just serial) so the parallel win is observable.
    pub fn set_parallel_routing(&mut self, on: bool) {
        self.parallel_routing = on;
    }

    pub fn parallel_routing(&self) -> bool {
        self.parallel_routing
    }

    /// Toggle the cache-friendly per-group sort (flat position-key array vs. reading a vehicle
    /// row per comparison). Same total order either way; purely a performance option.
    pub fn set_cache_sort(&mut self, on: bool) {
        self.cache_sort = on;
    }

    pub fn cache_sort(&self) -> bool {
        self.cache_sort
    }

    /// The flat position key array for the cache-friendly sort, or empty when the option is off
    /// (the sort helpers then read each vehicle row instead).
    fn position_keys(&self) -> Vec<f64> {
        if self.cache_sort {
            self.fleet.rows.iter().map(|v| v.position).collect()
        } else {
            Vec::new()
        }
    }

    /// Sort a per-lane group by vehicle position. `keys` is the flat position array when the
    /// cache-sort option is on (contiguous, cache-friendly); empty falls back to chasing each
    /// vehicle row. The order is identical — positions within a lane are distinct.
    fn sort_lane_members(&self, members: &mut [usize], keys: &[f64]) {
        if keys.is_empty() {
            members.sort_by(|&a, &b| self.fleet.rows[a].position.total_cmp(&self.fleet.rows[b].position));
        } else {
            sort_group_by(members, keys);
        }
    }

    /// The flat corridor-position key array for the cache-friendly sort, or empty when off.
    fn corridor_keys(&self) -> Vec<f64> {
        if self.cache_sort {
            self.fleet.rows.iter().map(|v| self.corridor_pos(v)).collect()
        } else {
            Vec::new()
        }
    }

    /// Sort a per-corridor group by corridor position; `keys` empty falls back to reading rows.
    fn sort_corridor_members(&self, members: &mut [usize], keys: &[f64]) {
        if keys.is_empty() {
            members.sort_by(|&a, &b| self.corridor_pos(&self.fleet.rows[a]).total_cmp(&self.corridor_pos(&self.fleet.rows[b])));
        } else {
            sort_group_by(members, keys);
        }
    }

    /// How many destination fields to solve concurrently in a recompute: one per pool thread
    /// when parallel routing is on (and the threaded build is linked), else 1.
    fn route_field_width(&self) -> usize {
        if !self.parallel_routing {
            return 1;
        }
        #[cfg(feature = "parallel")]
        {
            rayon::current_num_threads().max(1)
        }
        #[cfg(not(feature = "parallel"))]
        {
            1
        }
    }

    fn refresh_routes(&mut self) {
        if self.router.is_none() || self.external_reroute {
            return;
        }
        // Recompute routing only when the cost landscape actually moves. The fields from
        // `install_router` are optimal for free-flow; while traffic stays light or static the
        // congestion fingerprint doesn't change, so we skip the O(links) rebuild entirely —
        // this is why a near-empty city map costs almost nothing.
        let fp = self.congestion_fingerprint();
        let interval_ticks = (REROUTE_INTERVAL_SECS / self.cfg.dt).max(1.0) as u64;
        // Start a new reroute cycle only when congestion has moved, none is in flight, and the
        // interval has elapsed since the last — so constantly-churning traffic can't chain
        // back-to-back whole-map rebuilds.
        let pending = self.router.as_ref().is_some_and(|r| r.recompute_pending());
        if fp != self.route_fingerprint && !pending && self.tick.saturating_sub(self.route_cycle_tick) >= interval_ticks {
            self.route_fingerprint = fp;
            self.route_cycle_tick = self.tick;
            let costs = self.live_link_costs();
            let width = self.route_field_width();
            if let Some(r) = self.router.as_mut() {
                r.begin_recompute(costs, width);
            }
        }
        // Advance any in-flight recompute by a bounded per-tick budget: each field's whole-map
        // Dijkstra is spread across frames, so no frame ever does the full ~O(links log links)
        // sweep (~40 ms in wasm on a city map). Sized so a 116k-link field settles over ~15
        // ticks at a few ms each; a small map's field finishes in a single call (no overhead).
        const ROUTE_SETTLE_BUDGET: usize = 8_000;
        if let Some(r) = self.router.as_mut() {
            r.advance_recompute(ROUTE_SETTLE_BUDGET);
        }
    }

    pub fn step(&mut self) {
        let dt = self.cfg.dt;
        let mut prof = Prof::new();
        self.refresh_routes();
        prof.lap(0);
        self.advance_signals(dt);
        prof.lap(1);
        if self.congestion_cfg.enabled {
            let occ = self.link_occupancy();
            let cfg = self.congestion_cfg;
            self.congestion.update_modes(&occ, &cfg);
        }
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

        // The active-set scheduler composes with every backend: classification is
        // a cheap neighbour/gap check, while the full gather it skips (signals,
        // node walks, conflict and pressure scans) is the heavy half of the
        // per-car pass — so sleeping queued cars pays whether the sweep is serial
        // or parallel, and keeping it backend-independent means a threads step
        // computes the same physics a serial step would. (An earlier gate stood the scheduler down under Threads at load;
        // measured on the loaded real map, composing them is faster than
        // either alone.)
        let n = self.fleet.rows.len();
        let sleep_on = self.cfg.sleep_scheduler;

        // Phase 4 — one fused per-car pass does what were four separate parallel sweeps of the
        // fleet: intended movement, sleep classification, the box-entry gate, and the accel
        // gather+evaluate. Collapsing them to a single fork-join is the win on a threaded step —
        // each extra dispatch has fixed overhead and re-sweeps the whole (bandwidth-bound) fleet.
        // The CPU path evaluates the accel inline, so the fat per-car `AccelInput` (~150 B) is
        // never materialized into a Vec; the `Gpu` backend keeps the inputs to ship to its kernel
        // (its evaluate half runs in `accel.wgsl`). Every per-car datum read is this car's own, so
        // the fusion is order-preserving and bit-for-bit identical to the separate passes.
        let (seed, tick) = (self.cfg.seed, self.tick);
        let (mut intended_mv, mut sleeping, mut block_entry) =
            (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        let accels: Vec<f64> = if matches!(backend, AccelBackend::Gpu) {
            let rows: Vec<(Option<MovementId>, Sleep, bool, AccelInput)> = map_collect(backend, par_threshold, n, |i| {
                let (intended, sleep) = self.decide_intent(i, sleep_on, &nb);
                let block = self.box_entry_blocked(i, intended, &nb);
                (intended, sleep, block, self.gather_input(i, &nb, &cross_by_mv, sleep, intended))
            });
            let mut inputs = Vec::with_capacity(n);
            for (m, s, b, inp) in rows {
                intended_mv.push(m);
                sleeping.push(s);
                block_entry.push(b);
                inputs.push(inp);
            }
            self.evaluate_accels(backend, par_threshold, &inputs, seed, tick)
        } else {
            let rows: Vec<(Option<MovementId>, Sleep, bool, f64)> = map_collect(backend, par_threshold, n, |i| {
                let (intended, sleep) = self.decide_intent(i, sleep_on, &nb);
                let block = self.box_entry_blocked(i, intended, &nb);
                let accel = self.gather_input(i, &nb, &cross_by_mv, sleep, intended).evaluate(seed, tick);
                (intended, sleep, block, accel)
            });
            let mut a = Vec::with_capacity(n);
            for (m, s, b, ac) in rows {
                intended_mv.push(m);
                sleeping.push(s);
                block_entry.push(b);
                a.push(ac);
            }
            a
        };
        self.asleep_last = sleeping.iter().filter(|s| !matches!(s, Sleep::Awake)).count();
        prof.lap(4);

        // Destination-lane occupancy (nearest-to-entrance rear, and that car's speed), so a
        // crosser finishing its interior path never lands on top of a queued vehicle — it
        // holds at the far edge of the node — and, when it does land, only sheds enough
        // speed to safely follow whatever is ahead rather than braking as if it were stopped.
        // A *departing* crosser still counts while its rear hangs over the lane it left:
        // on a junction-internal stub shorter than a car, that tail is exactly what the
        // next landing must not be admitted into (`position` is continuous past the lane
        // end, so its rear is directly comparable).
        let mut front: IntMap<f64> = IntMap::default();
        let mut front_speed: IntMap<f64> = IntMap::default();
        for v in &self.fleet.rows {
            let rear = v.position - v.driver.vehicle_length;
            if rear < self.network.lane(v.lane).length {
                let e = front.entry(v.lane.0).or_insert(f64::MAX);
                if rear < *e {
                    *e = rear;
                    front_speed.insert(v.lane.0, v.speed);
                }
            }
        }

        let mut taken = std::mem::take(&mut self.fleet.rows);
        for (v, s) in taken.iter_mut().zip(&sleeping) {
            v.slept = !matches!(s, Sleep::Awake);
        }
        let taken_h = std::mem::take(&mut self.fleet.hist);
        let taken_hl = std::mem::take(&mut self.fleet.hist_len);
        let n = taken.len();

        // Phase 5a — integrate every non-crossing car within its lane. This half reads no
        // shared occupancy, so it runs across cores (like the accel gather). It returns, per
        // car, whether the serial boundary resolution still has to run for it: crossers (they
        // touch `front` on landing) and cars that reached a lane end (they consult and advance
        // the shared occupancy). `front`/`front_speed` were built from committed positions
        // above, so integrating here doesn't disturb them.
        let deferred: Vec<bool> = {
            let this: &NetWorld = self;
            let integrate_one = |veh: &mut NetVehicle, a: f64, intended: Option<MovementId>| -> bool {
                if veh.wreck.is_some() {
                    veh.speed = 0.0; // a wreck holds its pose; no boundary to resolve
                    return false;
                }
                veh.crossing.is_some() || this.integrate_in_lane(veh, a, dt, intended)
            };
            #[cfg(feature = "parallel")]
            let out: Vec<bool> = if matches!(backend, AccelBackend::Threads) && n >= par_threshold.max(LIGHT_PAR_THRESHOLD) {
                use rayon::prelude::*;
                taken
                    .par_iter_mut()
                    .zip(accels.par_iter())
                    .zip(intended_mv.par_iter())
                    .map(|((veh, &a), &intended)| integrate_one(veh, a, intended))
                    .collect()
            } else {
                taken.iter_mut().zip(&accels).zip(&intended_mv).map(|((veh, &a), &intended)| integrate_one(veh, a, intended)).collect()
            };
            #[cfg(not(feature = "parallel"))]
            let out: Vec<bool> = taken.iter_mut().zip(&accels).zip(&intended_mv).map(|((veh, &a), &intended)| integrate_one(veh, a, intended)).collect();
            out
        };

        // Slot occupancy of junction-internal lanes: landed cars plus crossers in
        // flight toward each, in one combined count per lane. Admission into a
        // junction interior is then capacity-checked race-free in the serial pass —
        // several approaches can no longer commit crossers toward the same one-car
        // internal stub before the first lands, which stranded the extras mid-box
        // (a permanent `box_conflict` for everyone else) and let a big split
        // junction's internal ring gridlock for good.
        let mut interior_occ: IntMap<u32> = IntMap::default();
        for v in taken.iter() {
            let lane = match v.crossing {
                Some(c) => self.network.movement(c.movement).to_lane,
                None => v.lane,
            };
            if self.junction_internal_lane(lane) {
                *interior_occ.entry(lane.0).or_insert(0) += 1;
            }
        }

        // In-flight reservations per receiving lane (metres of entrance space each
        // unlanded crosser will consume). Admission subtracts them, so two approaches
        // can no longer bank on the *same* room in different ticks — the double-booking
        // that stranded the loser mid-box behind a tail that materialized while it
        // crossed. Free-flow seams are exempt (corridor following spaces those).
        let mut inbound: IntMap<f64> = IntMap::default();
        for v in taken.iter() {
            if let Some(c) = v.crossing {
                if !self.free_flow_seam(c.movement) {
                    let to_lane = self.network.movement(c.movement).to_lane;
                    *inbound.entry(to_lane.0).or_insert(0.0) += v.driver.vehicle_length + v.driver.min_gap;
                }
            }
        }

        // Phase 5b — resolve the deferred cars serially in index order, so the shared `front`
        // occupancy evolves exactly as a single-threaded pass would (independent cars never
        // touch it, so skipping them here is invisible to `front`). Each car's fate is recorded;
        // an undeferred car simply stayed in its lane (`Alive`). `entered_at` accumulates the
        // at-grade movements committed earlier in this same pass, closing the same-tick
        // double-entry race the pre-step box gate can't see.
        let mut entered_at: IntMap<Vec<MovementId>> = IntMap::default();
        let mut fates: Vec<Fate> = Vec::with_capacity(n);
        for i in 0..n {
            let fate = if !deferred[i] {
                Fate::Alive
            } else if taken[i].crossing.is_some() {
                self.advance_crossing(&mut taken[i], accels[i], dt, &mut front, &mut front_speed, &mut inbound)
            } else {
                self.resolve_boundary(&mut taken[i], dt, &mut front, &mut front_speed, block_entry[i], intended_mv[i], &mut interior_occ, &mut entered_at, &mut inbound)
            };
            fates.push(fate);
        }
        prof.lap(5);

        // Crash detection on the fully-advanced positions, before assembly drops any row
        // (indices still align with the pre-step fleet and `nb`).
        let crashes = self.detect_crashes(&taken, &fates, &nb);

        // Assembly — apply each fate's bookkeeping and keep/drop in index order, identical to
        // the former single-pass loop; freshly crashed cars become wrecks (or are cleared
        // instantly when `wreck_clear_secs` is zero) and existing wrecks count down.
        let wreck_ticks = (self.cfg.wreck_clear_secs / dt).ceil().clamp(0.0, u16::MAX as f64) as u16;
        let mut rows = Vec::with_capacity(n);
        let mut hist = Vec::with_capacity(n);
        let mut hist_len = Vec::with_capacity(n);
        let mut exited = 0u32;
        for ((((mut veh, fate), mut h), mut hl), hit) in
            taken.into_iter().zip(fates).zip(taken_h).zip(taken_hl).zip(crashes)
        {
            if let Some((kind, closing)) = hit {
                self.record_crash(&veh, kind, closing);
                if wreck_ticks == 0 {
                    continue;
                }
                veh.wreck = Some(wreck_ticks);
                veh.speed = 0.0;
            } else if let Some(t) = veh.wreck {
                if t <= 1 {
                    continue; // wreck cleared; already tallied at impact
                }
                veh.wreck = Some(t - 1);
            }
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
        prof.lap(6);
        self.time += dt;
        self.tick += 1;
    }

    /// Advance one vehicle: integrate its longitudinal state, drive the
    /// lane→interior→lane transitions, and report whether it stayed on the road,
    /// entered a new link, or left the network.
    /// Advance a vehicle already crossing a node interior. Touches the shared `front`/
    /// `front_speed` occupancy (via [`land_or_hold`]), so it runs serially.
    fn advance_crossing(&self, veh: &mut NetVehicle, accel: f64, dt: f64, front: &mut IntMap<f64>, front_speed: &mut IntMap<f64>, inbound: &mut IntMap<f64>) -> Fate {
        // Position keeps counting past the from-lane's end; the interior arc is
        // that overrun. The car flows through the node as one continuous move,
        // advanced exactly as before (semi-implicit: step by the post-accel speed)
        // so the in-node conflict-avoidance margins are unchanged. The same grip
        // bound as `integrate` applies: in-box avoidance cannot brake beyond physics.
        veh.speed = (veh.speed + accel.max(-MAX_BRAKE_DECEL) * dt).max(0.0);
        veh.position += veh.speed * dt;
        self.land_or_hold(veh, front, front_speed, dt, inbound)
    }

    /// Integrate a non-crossing vehicle's longitudinal state (and lane-change progress and
    /// stop-line arming) within its current lane. Reads no shared occupancy — depends only on
    /// this vehicle, its precomputed `accel`, and the committed network — so it **parallelizes
    /// across the fleet**. Returns whether the car reached the lane end (`position >=
    /// lane.length`), i.e. whether [`resolve_boundary`] must run for it this tick.
    fn integrate_in_lane(&self, veh: &mut NetVehicle, accel: f64, dt: f64, intended: Option<MovementId>) -> bool {
        integrate(veh, accel, dt);
        if let Some(lc) = veh.lane_change.as_mut() {
            lc.progress += dt / LANE_CHANGE_DURATION;
            if lc.progress >= 1.0 {
                veh.lane_change = None;
            }
        }
        let lane = *self.network.lane(veh.lane);
        let node = self.network.link(lane.link).to;
        let must_stop_here = (matches!(self.network.node(node).control, NodeControl::Stop)
            && self.approach_must_stop(lane.link, node))
            || intended.is_some_and(|mid| self.is_rtor(mid));
        // The sign is served *at the line*: a stopped front car settles 2–3 m short,
        // inside this window, and each queued car behind arms afresh when it rolls
        // up in turn — so the turn-taking below always ranks the drivers actually
        // facing the intersection.
        if must_stop_here && veh.speed < Self::stop_roll_speed(&veh.driver) && (lane.length - veh.position) < 5.0 {
            veh.stopped_at = Some(node);
        }
        veh.position >= lane.length
    }

    /// Whether `lane` lies on a junction-internal link (both endpoints inside one
    /// junction cluster) — the crossing pavement of a big merged intersection.
    fn junction_internal_lane(&self, lane: LaneId) -> bool {
        let l = self.network.link(self.network.lane(lane).link);
        let j = self.network.node_junction(l.from);
        j.is_some() && j == self.network.node_junction(l.to)
    }

    /// Whether a junction-internal receiving lane still has a free `veh`-sized slot
    /// under the combined landed + in-flight count. Non-internal lanes always have
    /// room here (the ordinary entrance gate covers them).
    fn interior_has_room(&self, mid: MovementId, veh: &NetVehicle, occ: &IntMap<u32>) -> bool {
        let to_lane = self.network.movement(mid).to_lane;
        if !self.junction_internal_lane(to_lane) {
            return true;
        }
        let unit = veh.driver.vehicle_length + veh.driver.min_gap;
        let slots = (self.network.lane(to_lane).length / unit).floor().max(1.0) as u32;
        occ.get(&to_lane.0).copied().unwrap_or(0) < slots
    }

    /// Resolve a non-crossing vehicle that reached its lane end: enter the node interior,
    /// hold at the line, exit, or leak. Reads/writes the shared `front`/`front_speed`
    /// occupancy (admission gate + landing) and the junction-interior slot counts, so it
    /// runs serially after the parallel integrate.
    fn resolve_boundary(&self, veh: &mut NetVehicle, dt: f64, front: &mut IntMap<f64>, front_speed: &mut IntMap<f64>, block_entry: bool, intended: Option<MovementId>, interior_occ: &mut IntMap<u32>, entered_at: &mut IntMap<Vec<MovementId>>, inbound: &mut IntMap<f64>) -> Fate {
        let lane = *self.network.lane(veh.lane);
        let node = self.network.link(lane.link).to;
        // Reached the stop line. A sleeping car carries no intended movement (the
        // scheduler suppresses the router lookup for cars it expects to stay put),
        // but a queued sleeper can now follow its leader continuously across the
        // seam and actually arrive here — resolve its hop on the fly so it crosses
        // normally instead of being mistaken for a car that vanished at the node.
        let intended = intended.or_else(|| self.intended_movement(veh));
        // A free-flow seam (a within-corridor continuation, or any continuation seam
        // taken by the exit link's through approach) is not a junction — cross it with
        // no gate at all, so the freeway never stalls a car at a segment
        // boundary. At a real junction, enter only when the movement is served (green/yellow)
        // *and* the receiving lane can accept the vehicle — don't block the box, so a
        // spillback holds at the line rather than stalling inside the node.
        // A car already inside a multi-node junction (its link is cluster-internal)
        // was admitted at the cluster boundary: it commits through the interior
        // nodes' signals rather than stopping at a red on a few metres of box
        // pavement. Occupancy gates (receiving lane, interior slots) still apply.
        let internal_commit = self.junction_internal_lane(veh.lane);
        match intended {
            Some(mid)
                if {
                    // A car reaching the line so fast that even maximum braking would
                    // carry it well into the box (over ~2 m of encroachment) cannot
                    // physically hold there. It is committed — the dilemma-zone
                    // overrun — and the in-box conflict avoidance plus the body-
                    // overlap detector decide what happens, instead of an omniscient
                    // gate teleport-stopping it at the paint. A slow roll-past still
                    // pins (a bumper over the line, as real drivers do). Spillback
                    // gates stay hard: a full receiving lane was visible all the way
                    // in, so ordinary braking already held the car short.
                    let committed = veh.speed * veh.speed > 2.0 * MAX_BRAKE_DECEL * 2.0;
                    let signal_ok = (self.movement_state(mid) != SignalState::Red
                        || internal_commit
                        || (self.is_rtor(mid) && veh.stopped_at == Some(node))
                        || self.runs_red(veh, node))
                        && !self.meter_red(lane.link)
                        && !self.rail_closed(node);
                    // Same-tick entries are governed over the *whole* committed chain:
                    // two cars entering a cluster together on non-conflicting first
                    // hops whose paths cross deeper in must not both be admitted.
                    let same_tick_conflict = self.path_movements(veh, mid).iter().any(|&m| {
                        self.entered_conflicting(m, self.network.movement(m).node, entered_at)
                    });
                    let conflict_free = !block_entry && !same_tick_conflict;
                    // Don't-block-the-box, in full: commit into the interior only when
                    // the receiving lane has room for this *whole vehicle* past the box,
                    // net of the space every crosser already in flight toward it will
                    // consume. Landing itself (already committed) needs only the bumper
                    // gap. A free-flow seam skips the gates — except an *at-grade* seam
                    // with a live box conflict: an arterial's coalesced through-corridor
                    // still crosses a real intersection, and sailing in ungated while a
                    // permissive left swings across it was a guaranteed T-bone. Freeway
                    // seams are untouched (interchange movements never set `block_entry`).
                    // The seam branch honors the same-tick chain too: its
                    // `block_entry` is pre-step state, blind to a conflicting
                    // commit made earlier in this serial pass.
                    (self.free_flow_seam(mid) && !block_entry && !same_tick_conflict)
                        || (((signal_ok && conflict_free) || committed)
                            && self.receiving_room(mid, veh, front, front_speed, inbound)
                            && self.interior_has_room(mid, veh, interior_occ))
                } =>
            {
                // Enter the interior but keep `position` continuous — it already
                // counts past `lane.length`, and that overrun is the interior arc.
                // Land in this same tick if it already overran the whole interior (a
                // node shorter than one step), so it never dwells with a clamped pose.
                let lat_shift = veh.lane_change.as_ref().map_or(0.0, |lc| {
                    let own = self.network.lane_offset_at(&lane, 0.0);
                    let from = self.network.lane_offset_at(self.network.lane(lc.from), 0.0);
                    (1.0 - lc.progress.clamp(0.0, 1.0)) * (from - own)
                });
                // The car leaves its lane and is now in flight toward the movement's
                // receiver: move its interior slot accounting accordingly (landing
                // later is count-neutral — in-flight becomes landed on the same lane).
                if self.junction_internal_lane(veh.lane) {
                    if let Some(c) = interior_occ.get_mut(&veh.lane.0) {
                        *c = c.saturating_sub(1);
                    }
                }
                let to_lane = self.network.movement(mid).to_lane;
                if self.junction_internal_lane(to_lane) {
                    *interior_occ.entry(to_lane.0).or_insert(0) += 1;
                }
                // Register this at-grade commitment — the whole internal chain, so a
                // later car in this same serial pass can't accept a box any hop of
                // this path will cross — and reserve the entrance space this body
                // will consume on the receiving lane (free-flow seams have no cross
                // traffic and skip the bookkeeping).
                if !self.free_flow_seam(mid) {
                    if !self.network.is_interchange_movement(mid) {
                        for m in self.path_movements(veh, mid) {
                            let n = self.network.movement(m).node;
                            entered_at.entry(self.network.intersection_key(n)).or_default().push(m);
                        }
                    }
                    *inbound.entry(to_lane.0).or_insert(0.0) += veh.driver.vehicle_length + veh.driver.min_gap;
                }
                veh.crossing = Some(Crossing { movement: mid, lat_shift });
                veh.lane_change = None;
                self.land_or_hold(veh, front, front_speed, dt, inbound)
            }
            Some(_) => {
                veh.position = lane.length;
                veh.speed = (veh.speed - MAX_BRAKE_DECEL * dt).max(0.0);
                Fate::Alive
            }
            // No movement resolved. Legitimate when the car has arrived (no next
            // hop) or run off a genuine dead end; a leak if it still had somewhere
            // to go — which `intended_movement`'s fallback prevents.
            None if self.still_has_a_route(veh, lane.link) && self.network.links.iter().any(|l| l.from == node) => Fate::Leaked,
            None => Fate::Exited,
        }
    }

    /// A crossing vehicle whose `position` has already advanced this tick: land it on
    /// the destination lane once it clears the node interior (and the entrance is free),
    /// otherwise stay in the interior — or hold at its far edge when the exit is blocked.
    /// Called both while crossing and the instant a car enters at a boundary, so a node
    /// shorter than one step is entered and cleared in the same tick.
    fn land_or_hold(&self, veh: &mut NetVehicle, front: &mut IntMap<f64>, front_speed: &mut IntMap<f64>, dt: f64, inbound: &mut IntMap<f64>) -> Fate {
        let c = veh.crossing.unwrap();
        let it = *self.network.interior(c.movement);
        let lane_len = self.network.lane(veh.lane).length;
        let s = veh.position - lane_len;
        if s < it.len {
            return Fate::Alive;
        }
        // Reached the far edge: land on the destination lane. A real junction holds at the
        // far edge whenever the entrance is occupied (don't rear-end onto a queue), and —
        // like a free-flow seam — also when the discrete landing overrun would put the car
        // *inside* the occupant ahead (a fast crosser can overrun metres in one tick).
        let to_lane = self.network.movement(c.movement).to_lane;
        let land_pos = (s - it.len).min(self.network.lane(to_lane).length);
        let overrun = front.get(&to_lane.0).is_some_and(|&rear| land_pos > rear - veh.driver.min_gap);
        let blocked = overrun
            || (!self.free_flow_seam(c.movement) && !self.receiving_lane_clear(c.movement, front, veh.driver.min_gap));
        if blocked {
            veh.position = lane_len + it.len;
            veh.speed = (veh.speed - MAX_BRAKE_DECEL * dt).max(0.0);
            return Fate::Alive;
        }
        // Landed: the in-flight reservation converts into real occupancy (the `front`
        // update below), so release it for the approaches still deciding.
        if !self.free_flow_seam(c.movement) {
            if let Some(r) = inbound.get_mut(&to_lane.0) {
                *r = (*r - (veh.driver.vehicle_length + veh.driver.min_gap)).max(0.0);
            }
        }
        veh.crossing = None;
        veh.lane = to_lane;
        // Rebase the overrun into the new lane's frame (never skip a whole sub-tick segment).
        veh.position = land_pos;
        veh.stopped_at = None;
        veh.lane_change = self.seam_landing_blend(c.movement, c.lat_shift);
        // Land no faster than the car can safely follow whatever is already on the new
        // lane: braking at the physical max, it must not out-run the leader's own speed.
        // The safe landing speed is `sqrt(v_lead^2 + 2*b*gap)` — for a *stopped* leader this
        // is the bare stopping-distance clamp, but for a leader moving at road speed the
        // gap is fine and nothing changes, so a car no longer brakes as if emerging behind
        // a wall when it is merely joining flowing traffic (the abrupt seam standstill).
        if let Some(&rear) = front.get(&to_lane.0) {
            let gap = (rear - veh.position).max(0.0);
            let v_lead = front_speed.get(&to_lane.0).copied().unwrap_or(0.0);
            veh.speed = veh.speed.min((v_lead * v_lead + 2.0 * MAX_BRAKE_DECEL * gap).sqrt());
        }
        let rear = veh.position - veh.driver.vehicle_length;
        let e = front.entry(to_lane.0).or_insert(f64::MAX);
        if rear < *e {
            *e = rear;
            front_speed.insert(to_lane.0, veh.speed);
        }
        let to_link = self.network.lane(to_lane).link;
        if veh.route_idx + 1 < veh.route.len() && veh.route[veh.route_idx + 1] == to_link {
            veh.route_idx += 1;
        }
        Fate::Entered(to_link)
    }

    /// The pose blend a car lands with after a continuation seam whose movement put it
    /// on a different lateral line than it arrived on. `from` is the new link's lane
    /// nearest the geometric arrival point (shifted by any blend still in flight at the
    /// boundary) and `progress` the arrival's fraction of the way from that lane to the
    /// target, so the pose is continuous at the seam and eases over at the standard
    /// lane-change rate — a lane remap reads as a normal merge, never a sideways snap.
    fn seam_landing_blend(&self, mid: MovementId, lat_shift: f64) -> Option<LaneChange> {
        if !self.network.is_continuation_seam(mid) {
            return None;
        }
        let mv = self.network.movement(mid);
        let link = *self.network.link(self.network.lane(mv.to_lane).link);
        if link.lane_count <= 1 {
            return None;
        }
        let it = self.network.interior(mid);
        let arr = self.network.arrival_dir(self.network.lane(mv.from_lane).link);
        let p = [it.exit[0] + arr[1] * lat_shift, it.exit[1] - arr[0] * lat_shift];
        let d2 = |l: LaneId| {
            let q = self.network.lane_point(l, 0.0);
            (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2)
        };
        let from = (0..link.lane_count)
            .map(|i| LaneId(link.lane_start.0 + i))
            .filter(|&l| l != mv.to_lane)
            .min_by(|&a, &b| d2(a).total_cmp(&d2(b)))?;
        let a = self.network.lane_point(from, 0.0);
        let b = self.network.lane_point(mv.to_lane, 0.0);
        let ab = [b[0] - a[0], b[1] - a[1]];
        let len2 = (ab[0] * ab[0] + ab[1] * ab[1]).max(1e-9);
        let t = (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / len2).clamp(0.0, 1.0);
        (t < 0.97).then_some(LaneChange { from, progress: t })
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

    /// The entrance space a car needs on a receiving lane before committing into the
    /// box in front of it: its own body + gap, plus a second body's worth of slack —
    /// the queue on the far side can advance during the multi-second box traversal
    /// and trap the car inside, so a driver waits for clearly-more-than-one-car room
    /// (capped so a short exit lane can still ever be entered).
    fn commit_room_needed(&self, to_lane: LaneId, unit: f64) -> f64 {
        unit + unit.min((self.network.lane(to_lane).length - unit).max(0.0))
    }

    /// Whether the receiving lane can take this *whole vehicle* past the box, net of
    /// the space already promised to crossers in flight toward it. An empty lane with
    /// no reservations always can (even a stub shorter than the body — the car lands
    /// and straddles, and the occupancy map holds followers off it).
    ///
    /// A *departing* occupant (moving at ≥ [`DEPARTING_SPEED`]) is not a blockage:
    /// the crossing car already car-follows it through the cross-boundary leader,
    /// so only a bumper gap is demanded. Requiring absolute clearance from a tail
    /// that is accelerating away serialized every queued crossing to one car per
    /// ~7 s — the systemic capacity collapse behind the US-101 gateway jam and
    /// El Camino's missing volume. Full commit room still gates against slow or
    /// stopped tails — genuine spillback.
    fn receiving_room(
        &self,
        mid: MovementId,
        veh: &NetVehicle,
        front: &IntMap<f64>,
        front_speed: &IntMap<f64>,
        inbound: &IntMap<f64>,
    ) -> bool {
        let to_lane = self.network.movement(mid).to_lane;
        let unit = veh.driver.vehicle_length + veh.driver.min_gap;
        let reserved = inbound.get(&to_lane.0).copied().unwrap_or(0.0);
        match front.get(&to_lane.0) {
            Some(&rear) => {
                let tail_v = front_speed.get(&to_lane.0).copied().unwrap_or(0.0);
                if tail_v >= DEPARTING_SPEED && self.departing_exemption(mid) {
                    // Follow the departing tail *into* the box: the spacing that
                    // matters is along the continuous path (interior + landing
                    // overhang), at a real following distance — the tail can
                    // still brake mid-box, and a bare bumper gap plus reaction
                    // delay ends in a nudge. In-box spacing thereafter is the
                    // same-movement crossing follower's job.
                    self.network.interior(mid).len + rear - reserved
                        >= departing_margin(&veh.driver, tail_v)
                } else {
                    rear - reserved >= self.commit_room_needed(to_lane, unit)
                }
            }
            None => reserved <= 0.0 || self.network.lane(to_lane).length - reserved >= unit,
        }
    }

    /// IDM acceleration for a vehicle traversing a node interior: capped to the
    /// turn's comfortable speed and following either a crosser ahead on the same
    /// movement or the queue waiting on the destination lane.
    /// Gather a rolling vehicle's accel-decision context — the graph, neighbor,
    /// signal and router lookups a GPU evaluate kernel can't do. Returns a flat,
    /// owned [`VehicleContext`]; [`VehicleContext::evaluate`] turns it into an
    /// acceleration with no further graph access. Behaviour is identical to the old
    /// fused accel loop; this is purely the gather/evaluate split.
    fn gather_context(&self, i: usize, nb: &Neighbors, intended: Option<MovementId>) -> VehicleContext {
        let dt = self.cfg.dt;
        let veh = &self.fleet.rows[i];
        let lane = *self.network.lane(veh.lane);
        let driver = veh.driver.capped_to(lane.speed_limit);
        let node = self.downstream_node(veh.lane);
        let control = self.network.node(node).control;
        let to_line = (lane.length - veh.position).max(0.05);

        // Human perception of the leader (Treiber's human-driver formulation): the
        // driver's last *observation* of the leader is `reaction_time` old, and they
        // extrapolate it forward at its observed speed; their own position is always
        // current. A stopped or steady leader is therefore perceived exactly, while
        // for `reaction_time` after a leader starts braking the driver still acts on
        // the old speed — under-braking that, with the applied deceleration clamped
        // to `MAX_BRAKE_DECEL`, can genuinely rear-end. That is the physical crash
        // mechanism.
        let leader = if let Some(li) = nb.leader_of[i] {
            let lead = &self.fleet.rows[li];
            let delay = (driver.reaction_time / dt).round() as usize;
            // The reaction-delay lookup only makes sense for a same-lane leader whose delayed
            // position is in this frame; across a corridor seam (different lane, or either car
            // mid-crossing) fall back to the true current gap so the delayed read never
            // phantom-brakes the car at a boundary.
            if lead.lane == veh.lane
                && veh.crossing.is_none()
                && lead.crossing.is_none()
                && self.fleet.settled(i, delay)
                && self.fleet.settled(li, delay)
            {
                let (lead_p, lead_v) = self.fleet.delayed(li, delay);
                let ahead = lead_p + lead_v * (delay as f64 * dt);
                let gap = ahead - veh.position - lead.driver.vehicle_length;
                Some(Obstacle { gap, speed: lead_v })
            } else {
                // The leader chain spans the corridor, so the true gap is the corridor-
                // coordinate gap — correct whether the leader shares this lane or is a
                // segment ahead.
                let ob = Obstacle { gap: self.corridor_gap(veh, lead), speed: lead.speed };
                // A corridor leader a *segment ahead* (a different lane in the same grade-
                // separated 1:1 chain): the gap is continuous, but at the instant of a seam
                // hand-off it can momentarily collapse below the physical stopping gap. Cap it
                // so following across the seam eases down at a physical rate instead of
                // commanding an impossible brake — the continuous corridor following (like the
                // crossing gate elsewhere) guarantees the car cannot actually land on it. A
                // same-lane leader here (this car mid-crossing, or either just spawned/landed)
                // is a genuine collision risk and keeps its true gap so IDM can brake for it.
                if lead.lane != veh.lane {
                    Some(self.cap_leader_brake(&driver, veh.speed, ob))
                } else {
                    Some(ob)
                }
            }
        } else {
            // No leader on this lane: follow the nearest car ahead across the coming
            // segment boundaries, so approaching a boundary the car keeps its gap and
            // never arrives on top of a leader that just crossed. A leader on a downstream
            // segment is likewise gate-protected — this car cannot cross onto it until the
            // entrance clears — so its brake is capped to physical too, killing the abrupt
            // full stop when a car drops into the blind spot just across the seam.
            self.cross_boundary_leader(veh, intended, nb).map(|ob| self.cap_leader_brake(&driver, veh.speed, ob))
        };

        // Upcoming curve (lateral-accel limit) and turn speed; the LOD path below
        // needs these, so compute them before it.
        let geom_curve = {
            let r = self.network.min_radius_ahead(veh.lane, veh.position, CURVE_LOOKAHEAD);
            r.is_finite().then(|| SpeedTarget { speed: (A_LAT * r).sqrt(), distance: CURVE_LOOKAHEAD })
        };
        let turn = intended
            .and_then(|mid| {
                // Interchange movements stay uncapped (the ramp curve slows them);
                // everything else brakes toward its interior's curvature speed.
                let cap = self.turn_speed_cap(mid);
                cap.is_finite().then_some(cap)
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
        let downstream_slower = intended
            .is_some_and(|mid| self.network.lane(self.network.movement(mid).to_lane).speed_limit < lane.speed_limit);
        if to_line > DECISION_HORIZON && !downstream_slower {
            // Mid-link bus stops bind far from any node, so the LOD fast path
            // still serves them.
            if let Some(d) = self.bus_stop_line(veh, &lane) {
                cx.set_stop_line(Some(d));
            }
            return cx;
        }

        // Don't-block-the-box: about to cross, but the downstream lane's entrance is
        // occupied — hold at the line rather than land on top of a stopped vehicle
        // (the main source of intersection crashes). This is a *box* concern, so it only
        // applies at grade: a free-flow interchange (freeway continuation/diverge/merge)
        // has no cross street to block, and continuous leader-following already keeps the
        // car off the one ahead. Applying it there braked the car abruptly to a dead stop
        // at every segment seam — the phantom freeway standstill.
        let downstream_blocked = intended.is_some_and(|mid| self.movement_downstream_blocked(mid, &driver, nb));
        // A multi-node junction acts as one signal at its entrances: once a vehicle
        // is on an internal link (both endpoints in the same junction) it has already
        // been admitted and commits through the interior nodes rather than stopping at
        // a sibling node's line inside the box — real drivers never stop mid-junction.
        let on_internal_link = {
            let from = self.network.link(lane.link).from;
            let j = self.network.node_junction(from);
            j.is_some() && j == self.network.node_junction(node)
        };
        // Signal: red stops (unless this driver is blind to this light — the rare
        // distraction that runs it outright); yellow stops only if the vehicle can
        // brake comfortably before the line (dilemma zone) — otherwise it proceeds
        // and clears on yellow.
        let signal_stop = !on_internal_link
            && intended.is_some_and(|mid| match self.movement_state(mid) {
                SignalState::Green => veh.speed < 2.0 && self.signal_green_elapsed(mid) < driver.reaction_time,
                SignalState::Red => !self.is_rtor(mid) && !self.runs_red(veh, node),
                SignalState::Yellow => {
                    let committed = veh.speed > 3.0 && to_line < veh.speed * 3.0;
                    let runs_it = committed
                        && rng::uniform01(self.cfg.seed, veh.id, YELLOW_RUN_SALT, Stream::GapAcceptance)
                            < yellow_run_prob(&veh.driver);
                    !runs_it && can_stop_before(veh.speed, driver.comfort_decel, to_line)
                }
            });
        // Also hold at the line when crossing would strand the car inside a
        // multi-node junction (a sibling node ahead is red or its exit is occupied).
        let junction_blocked =
            intended.is_some_and(|mid| self.junction_exit_blocked(veh, mid, node, nb));
        // Stop at the immediate line (red / blocked box / a red ramp meter), or
        // ease toward a red one intersection ahead so the slowdown starts an
        // earlier link.
        let stop_line = (signal_stop
            || downstream_blocked
            || junction_blocked
            || self.meter_red(lane.link)
            || self.rail_closed(node))
        .then_some(to_line)
        .or_else(|| self.red_ahead(veh, intended, to_line));
        // Bus service: a bus holds at (or brakes toward) its next unserved stop
        // — the curb-lane dwell that dips arterial speeds around real stops.
        let stop_line = match self.bus_stop_line(veh, &lane) {
            Some(d) => Some(stop_line.map_or(d, |s: f64| s.min(d))),
            None => stop_line,
        };

        let speed_target = intended.and_then(|mid| {
            let to = self.network.lane(self.network.movement(mid).to_lane);
            // The slower of the downstream road's limit and the turn's own crawl
            // speed: a driver planning a left slows toward turning speed *on the
            // approach*, not at the paint — arriving at the line above the
            // dilemma-zone threshold made every turn a potential committed T-bone.
            let target = veh.driver.desired_speed.min(to.speed_limit).min(self.turn_speed_cap(mid));
            (target < driver.desired_speed).then_some(SpeedTarget { speed: target, distance: to_line })
        });
        // Cover the brake on a live box: while a crosser conflicts with *any*
        // movement this lane serves, approach slow enough to still stop. Keyed to
        // the lane, not the routed movement, so a mid-approach reroute flicker
        // can't release the caution, re-accelerate the car, and deliver it to the
        // line dilemma-zone-committed into an occupied box.
        let busy_box = !on_internal_link
            && self
                .network
                .movements_of(veh.lane)
                .iter()
                .enumerate()
                .any(|(k, _)| {
                    let m = MovementId(lane.movement_start.0 + k as u32);
                    !self.network.is_interchange_movement(m)
                        && !self.free_flow_seam(m)
                        && self.box_conflict(m, node, nb)
                })
            // …or a priority vehicle is near an unsignalized node ahead: ease
            // below the dilemma-zone threshold while the yield may yet bind,
            // instead of arriving 0.5 m/s too fast to stop and plowing through.
            || (!on_internal_link
                && !intended.is_some_and(|m| self.free_flow_interchange(m))
                && matches!(control, NodeControl::Uncontrolled | NodeControl::Stop | NodeControl::Yield)
                && self.conflicting_priority_traffic_scaled(i, veh.lane, node, nb, 1.6).is_some());
        let speed_target = if busy_box {
            let cover = SpeedTarget { speed: 5.0, distance: to_line };
            Some(speed_target.map_or(cover, |t| if t.speed < cover.speed { t } else { cover }))
        } else {
            speed_target
        };

        let rtor = intended.is_some_and(|mid| self.is_rtor(mid));
        let stop_sign = (!on_internal_link
            && ((matches!(control, NodeControl::Stop) && self.approach_must_stop(lane.link, node)) || rtor)
            && veh.stopped_at != Some(node))
        .then_some(to_line);

        // A freeway diverge/merge is free-flow — no crossing traffic to yield to (the
        // merge is a zipper, handled by `merge`, not a box crossing). So a freeway
        // through/merge/diverge movement never box-yields or box-blocks; that gating is
        // what was wrongly stopping cars mid-freeway at on-ramp merges.
        let free_flow = intended.is_some_and(|mid| self.free_flow_interchange(mid));
        // Never enter a box occupied by conflicting crossing traffic (at-grade nodes);
        // additionally, at unsignalized nodes defer to higher-priority approaching
        // traffic by right-of-way.
        let waiter_hold = intended.and_then(|mid| self.permissive_waiter_hold(i, mid, node, nb));
        let box_yield = !free_flow
            && intended.is_some_and(|mid| self.box_conflict_on_path_holding(veh, mid, node, nb, waiter_hold));
        // At an all-way stop, an *armed* driver (their stop served, FIFO turn
        // theirs) does not gap-accept against approaching traffic — arrivals must
        // serve their own sign. With HCM-sized gaps, yielding to every mover
        // within ~7 s parked the armed car for good under a steady cross stream.
        // Only at a genuine all-way: a two-way stop's minor street has no such
        // protocol — the major road never arms, so a served stop must still
        // gap-accept against it.
        let armed_all_way = matches!(control, NodeControl::Stop)
            && self.network.all_way_stop(node)
            && veh.stopped_at == Some(node);
        let prio_yield = !free_flow
            && !armed_all_way
            && matches!(control, NodeControl::Uncontrolled | NodeControl::Stop | NodeControl::Yield)
            && self.conflicting_priority_traffic(i, veh.lane, node, nb).is_some();
        let permissive_yield = waiter_hold.is_none()
            && intended.is_some_and(|mid| self.is_permissive(mid) && self.permissive_must_yield(i, mid, node, nb));
        let fifo_yield = matches!(control, NodeControl::Stop)
            && veh.stopped_at == Some(node)
            && intended.is_some_and(|mid| self.earlier_stopped_conflict(i, mid, node, nb));
        // A car already committed on an interior link clears the junction rather than
        // pulling up and waiting for a gap inside it; only the hard box-occupancy gate
        // (crash safety) still applies to it.
        let soft_yield = (prio_yield || permissive_yield || fifo_yield) && !on_internal_link;
        let yield_line = (box_yield || soft_yield).then_some(to_line);

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
        let c_s = self.crossing_arc(veh);
        let to_lane = self.network.movement(c.movement).to_lane;
        // A within-corridor seam is one continuous lane: while traversing it, follow the
        // corridor leader (as the car did before and after the seam), so the gap is
        // maintained continuously and the car lands cleanly behind it rather than on top
        // of it. Conflict-point avoidance below still applies — an arterial's coalesced
        // through-corridor crosses a real intersection.
        let mut accel = if self.is_intra_corridor(c.movement) {
            let d = veh.driver.capped_to(self.network.lane(to_lane).speed_limit);
            match nb.leader_of[i] {
                Some(j) => {
                    let l = &self.fleet.rows[j];
                    idm::acceleration(&d, veh.speed, veh.speed - l.speed, self.corridor_gap(veh, l).max(0.05))
                }
                None => idm::free_acceleration(&d, veh.speed),
            }
        } else {
            let it = self.network.interior(c.movement);
            let mut d = veh.driver.capped_to(self.network.lane(to_lane).speed_limit);
            d.desired_speed = d.desired_speed.min(self.turn_speed_cap(c.movement));

            let mut gap = f64::INFINITY;
            let mut lead_speed = veh.speed;
            for &j in cross_by_mv.get(&c.movement.0).into_iter().flatten() {
                if j == i {
                    continue;
                }
                let o_s = self.crossing_arc(&self.fleet.rows[j]);
                if o_s > c_s {
                    let g = o_s - c_s - self.fleet.rows[j].driver.vehicle_length;
                    if g < gap {
                        gap = g;
                        lead_speed = self.fleet.rows[j].speed;
                    }
                }
            }
            if let Some(&f) = nb.lane_front.get(&to_lane.0) {
                let l = &self.fleet.rows[f];
                let g = (it.len - c_s) + l.position - l.driver.vehicle_length;
                if g < gap {
                    gap = g;
                    lead_speed = l.speed;
                }
            }
            if gap.is_finite() {
                idm::acceleration(&d, veh.speed, veh.speed - lead_speed, gap.max(0.05))
            } else {
                idm::free_acceleration(&d, veh.speed)
            }
        };
        let d = veh.driver.capped_to(self.network.lane(to_lane).speed_limit);

        // Next-hop gate: when the path continues onto another internal hop whose
        // box is conflicted, brake to hold at the end of this interior instead of
        // accelerating into an unstoppable "committed" overrun at the internal line.
        if self.junction_internal_lane(to_lane) {
            if let Some(next) = self.movement_from_lane_for(veh, to_lane) {
                let key = self.network.intersection_key(self.network.movement(next).node);
                let moving_conflict = nb
                    .moving_crossing_mvs
                    .get(&key)
                    .into_iter()
                    .flatten()
                    .any(|&o| self.network.movements_conflict(next, o));
                if moving_conflict {
                    let remaining = self.network.interior(c.movement).len - c_s;
                    accel = accel.min(idm::acceleration(&d, veh.speed, veh.speed, (remaining - 1.0).max(0.05)));
                }
            }
        }

        let node = self.network.movement(c.movement).node;
        // Mid-box waiter: a signalized permissive left stands
        // `PERMISSIVE_HOLD_MARGIN` short of its first live conflict point while
        // oncoming still owns the window, and sweeps on when the pressure lifts
        // — a gap, or the change interval stopping the opposing flow. Only
        // *before* the first point: once committed past it, the first-to-the-
        // point avoidance below arbitrates like any other crosser.
        if self.network.movement_turn(c.movement) == TurnType::Left
            && self.network.movement(c.movement).signal_group.is_some()
        {
            if let Some(first) = self.first_live_conflict_arc(c.movement, node) {
                let hold = first - PERMISSIVE_HOLD_MARGIN;
                if c_s < hold - 0.1 {
                    let d = veh.driver.capped_to(self.network.lane(to_lane).speed_limit);
                    let pressed = self.permissive_pressure(i, c.movement, node, nb, &|my_s| {
                        (my_s > c_s + 0.3).then(|| my_s - c_s)
                    });
                    if pressed {
                        accel = accel.min(idm::acceleration(&d, veh.speed, veh.speed, (hold - c_s).max(0.05)));
                    }
                }
            }
        }

        // In-intersection avoidance: if a vehicle on a conflicting movement will
        // reach a shared conflict point first, brake to stop short of it. This is
        // the crash-avoidant behaviour; a driver already too close/fast to stop
        // (within the grip bound) sweeps on, and the body-overlap detector decides.
        for &ci in self.junctions.conflict_ids(node) {
            let cp = &self.network.conflicts[ci as usize];
            let (my_s, other_mv, other_s) = if cp.a == c.movement {
                (cp.sa, cp.b, cp.sb)
            } else if cp.b == c.movement {
                (cp.sb, cp.a, cp.sa)
            } else {
                continue;
            };
            let my_dist = my_s - c_s;
            if my_dist <= 0.0 {
                continue; // already through this point
            }
            for &j in cross_by_mv.get(&other_mv.0).into_iter().flatten() {
                let o = &self.fleet.rows[j];
                let their_dist = other_s - self.crossing_arc(o);
                // The point is clear only once their whole body is past it — the
                // arc tracks the front, so the tail lingers a vehicle length.
                if their_dist < -(o.driver.vehicle_length + 1.0) {
                    continue;
                }
                // A stationary car genuinely short of the point — a mid-box
                // waiter standing its hold — doesn't claim it; the moment it
                // launches it is a mover again and contests normally.
                if o.speed < 0.5 && their_dist > 1.5 {
                    continue;
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

    /// Detect this tick's collisions on the fully-advanced positions, before assembly
    /// drops any row (`taken`/`fates` are index-aligned with the pre-step fleet and
    /// `nb`). Returns, per vehicle, the crash it was in and the pair's closing speed.
    ///
    /// Two ground-truth checks, no tolerance bands:
    /// - **Rear-end**: sweep the pre-step corridor leader chain — a follower whose
    ///   front passed its leader's rear collided, including a full pass-through in
    ///   one tick and overlaps across a segment seam (the chain spans the corridor).
    ///   Reaction delay plus the physical brake clamp is what makes these reachable.
    /// - **Junction**: exact oriented-body intersection between co-node crossers on
    ///   distinct paths — what makes running a red or misjudging a gap actually
    ///   crash. Cost is Σ k² over nodes with k = that node's simultaneous crossers;
    ///   nothing scans the network-wide conflict list.
    ///
    /// A live car hitting an existing wreck is a fresh crash for the live car only;
    /// wreck–wreck contact in a pileup is not re-counted.
    fn detect_crashes(&self, taken: &[NetVehicle], fates: &[Fate], nb: &Neighbors) -> Vec<Option<(CrashKind, f32)>> {
        const OVERLAP_TOL: f64 = 0.5;
        let mut hit: Vec<Option<(CrashKind, f32)>> = vec![None; taken.len()];
        let on_road = |i: usize| matches!(fates[i], Fate::Alive | Fate::Entered(_));

        for i in 0..taken.len() {
            let Some(li) = nb.leader_of[i] else { continue };
            if !on_road(i) || !on_road(li) || taken[i].wreck.is_some() {
                continue;
            }
            let (f, l) = (&taken[i], &taken[li]);
            // The chain was built per corridor; a leader that has since landed on a
            // different corridor left this coordinate frame (the landing gates kept
            // the entrance clear behind it).
            if self.corridor_of[f.lane.0 as usize] != self.corridor_of[l.lane.0 as usize] {
                continue;
            }
            // Two in-node crossers share the scalar coordinate only on the *same*
            // interior path; on movements fanning out from one lane the arcs diverge
            // in the world, so an arc "overlap" there is not a touch (their genuine
            // collisions are the body-overlap detector's concern).
            if let (Some(cf), Some(cl)) = (f.crossing, l.crossing) {
                if cf.movement != cl.movement {
                    continue;
                }
            }
            if self.corridor_gap(f, l) < -OVERLAP_TOL {
                let closing = (f.speed - l.speed).abs() as f32;
                if std::env::var_os("CRASH_DEBUG").is_some() {
                    eprintln!(
                        "RE tick={} gap={:.2} f=(id{} lane{} pos{:.1} v{:.1} cross{} fate{:?}) l=(id{} lane{} pos{:.1} v{:.1} cross{} wreck{})",
                        self.tick,
                        self.corridor_gap(f, l),
                        f.id, f.lane.0, f.position, f.speed, f.crossing.is_some(), matches!(fates[i], Fate::Entered(_)),
                        l.id, l.lane.0, l.position, l.speed, l.crossing.is_some(), l.wreck.is_some(),
                    );
                }
                hit[i] = Some((CrashKind::RearEnd, closing));
                if l.wreck.is_none() {
                    hit[li] = Some((CrashKind::RearEnd, closing));
                }
            }
        }

        let mut by_node: IntMap<Vec<usize>> = IntMap::default();
        for (i, v) in taken.iter().enumerate() {
            if let Some(c) = v.crossing {
                if on_road(i) {
                    let node = self.network.movement(c.movement).node;
                    by_node.entry(self.network.intersection_key(node)).or_default().push(i);
                }
            }
        }
        for group in by_node.values() {
            for a in 0..group.len() {
                for b in a + 1..group.len() {
                    let (i, j) = (group[a], group[b]);
                    let (vi, vj) = (&taken[i], &taken[j]);
                    if vi.wreck.is_some() && vj.wreck.is_some() {
                        continue;
                    }
                    let (idi, idj) = (vi.crossing.unwrap().movement, vj.crossing.unwrap().movement);
                    // Only conflict-graph pairs can collide: the interior Béziers are
                    // schematic, so non-conflicting paths (opposing lefts, shallow
                    // near-parallel passes) may legitimately graze in the compressed
                    // box the way real offset paths don't — that's geometry, not a
                    // collision. Same-approach fan-outs and same-exit zippers are
                    // excluded by the same rule (the builder never pairs them).
                    if !self.network.movements_conflict(idi, idj) {
                        continue;
                    }
                    let (pi, pj) = (self.vehicle_world_pose(vi), self.vehicle_world_pose(vj));
                    let width = |v: &NetVehicle| VehicleClass::from_length(v.driver.vehicle_length).width();
                    if body_overlap(pi, vi.driver.vehicle_length, width(vi), pj, vj.driver.vehicle_length, width(vj)) {
                        let closing = (vi.speed.max(vj.speed)) as f32;
                        if std::env::var_os("CRASH_DEBUG").is_some() {
                            let mv = |mid: MovementId| {
                                (self.network.movement_turn(mid), self.movement_state(mid))
                            };
                            eprintln!(
                                "JX tick={} node={} i=(id{} mid{} {:?} v{:.1} arc{:.1}/{:.0} len{:.0}) j=(id{} mid{} {:?} v{:.1} arc{:.1}/{:.0} len{:.0})",
                                self.tick,
                                self.network.movement(idi).node.0,
                                vi.id, idi.0, mv(idi), vi.speed, self.crossing_arc(vi), self.network.interior(idi).len, vi.driver.vehicle_length,
                                vj.id, idj.0, mv(idj), vj.speed, self.crossing_arc(vj), self.network.interior(idj).len, vj.driver.vehicle_length,
                            );
                        }
                        if vi.wreck.is_none() {
                            hit[i] = Some((CrashKind::Junction, closing));
                        }
                        if vj.wreck.is_none() {
                            hit[j] = Some((CrashKind::Junction, closing));
                        }
                    }
                }
            }
        }
        hit
    }

    /// Tally one crashed vehicle and append its bounded log record (kind, closing
    /// speed, world position) for the overlay and the cause breakdown.
    fn record_crash(&mut self, veh: &NetVehicle, kind: CrashKind, closing_speed: f32) {
        let p = self.vehicle_world_pose(veh);
        self.crash_log.push(CrashRecord { pos: [p[0] as f32, p[1] as f32], kind, closing_speed });
        if self.crash_log.len() > MAX_CRASH_SITES {
            let excess = self.crash_log.len() - MAX_CRASH_SITES;
            self.crash_log.drain(..excess);
        }
        self.crashed += 1;
        self.crashed_by[kind as usize] += 1;
    }

    /// Whether `mid`'s interior path conflicts with a vehicle already crossing
    /// `node` — you must not enter an occupied box, whatever the control.
    fn box_conflict(&self, mid: MovementId, node: NodeId, nb: &Neighbors) -> bool {
        nb.crossing_mvs
            .get(&self.network.intersection_key(node))
            .into_iter()
            .flatten()
            .any(|&o| self.network.movements_conflict(mid, o))
    }

    /// Graded box occupancy: a conflicting crosser blocks entry only while it
    /// still *owns the shared conflict point* during the entrant's own arrival
    /// window — the entrant's time to reach the point vs the crosser's time to
    /// clear it (tail past, plus a margin). The binary any-crosser-anywhere test
    /// double-counted traffic that had already swept past the entrant's path:
    /// a minor street facing a busy major never saw a usable instant even though
    /// each major car occupies the minor's point for barely a second. Future
    /// cluster hops of moving crossers stay binary (their timing is a guess).
    fn box_conflict_graded(
        &self,
        veh: &NetVehicle,
        mid: MovementId,
        dist_to_line: f64,
        node: NodeId,
        nb: &Neighbors,
        hold: Option<f64>,
    ) -> bool {
        let key = self.network.intersection_key(node);
        let conflict_ids = self.junctions.conflict_ids(node);
        for &j in nb.crossing_at.get(&key).into_iter().flatten() {
            let o = &self.fleet.rows[j];
            let cm = o.crossing.unwrap().movement;
            // Future hops of a moving crosser: binary (as before).
            if o.speed >= 0.5 {
                let mut lane = self.network.movement(cm).to_lane;
                for _ in 0..4 {
                    if !self.junction_internal_lane(lane) {
                        break;
                    }
                    let Some(next) = self.movement_from_lane_for(o, lane) else { break };
                    if self.network.movements_conflict(mid, next) {
                        return true;
                    }
                    lane = self.network.movement(next).to_lane;
                }
            }
            if !self.network.movements_conflict(mid, cm) {
                continue;
            }
            // Current movement: time the shared point(s).
            let mut timed_any = false;
            let o_arc = self.crossing_arc(o);
            for &ci in conflict_ids {
                let cp = &self.network.conflicts[ci as usize];
                let (my_s, o_s) = if cp.a == mid && cp.b == cm {
                    (cp.sa, cp.sb)
                } else if cp.b == mid && cp.a == cm {
                    (cp.sb, cp.sa)
                } else {
                    continue;
                };
                timed_any = true;
                // A point at or beyond this car's declared hold arc is never
                // contested: the mid-box waiter stops short of it, so its
                // arrival window there is empty by construction.
                if hold.is_some_and(|h| my_s >= h) {
                    continue;
                }
                let tail_clear = o_s + o.driver.vehicle_length + 1.0;
                if o_arc >= tail_clear {
                    continue; // their body is past this point
                }
                // A crosser standing still short of the point — a mid-box waiter
                // holding its yield — doesn't claim it; timed at the crawl floor
                // it would read as a ~12 s occupation and gate the whole
                // opposing flow at the line.
                if o.speed < 0.5 && o_s - o_arc > 1.5 {
                    continue;
                }
                let t_clear = (tail_clear - o_arc) / o.speed.max(0.8);
                let cap = self.turn_speed_cap(mid).min(veh.driver.desired_speed);
                let t_reach =
                    travel_time(dist_to_line + my_s, veh.speed.max(1.0), veh.driver.max_accel, cap);
                if t_reach < t_clear + 0.6 {
                    return true;
                }
            }
            if !timed_any {
                return true; // conflicting but no point registered here — stay safe
            }
        }
        false
    }

    /// [`box_conflict`] over the vehicle's whole committed path through the
    /// intersection: the immediate movement plus each subsequent hop while the
    /// path stays on junction-internal links (a multi-node cluster). Admission at
    /// the outer boundary commits the car through the entire cluster — interior
    /// signals are bypassed and the internal stubs are too short to stop on — so
    /// the outer gate must clear the whole path, not just the entry stub, or a
    /// fast through is waved in only to meet a crosser mid-cluster with no
    /// stopping distance.
    /// The whole-path box-conflict gate, waiter-aware: `hold` is the interior
    /// arc a mid-box waiter will stand short of, so shared points at or beyond
    /// it are not contested at admission (the in-box hold owns them); `None`
    /// gates every point on the committed path as before.
    fn box_conflict_on_path_holding(
        &self,
        veh: &NetVehicle,
        mid: MovementId,
        node: NodeId,
        nb: &Neighbors,
        hold: Option<f64>,
    ) -> bool {
        let dist_to_line = (self.network.lane(veh.lane).length - veh.position).max(0.0);
        if self.box_conflict_graded(veh, mid, dist_to_line, node, nb, hold) {
            return true;
        }
        let mut lane = self.network.movement(mid).to_lane;
        for _ in 0..4 {
            if !self.junction_internal_lane(lane) {
                return false;
            }
            // On an internal stub the car can reach any of the *link's* movements
            // (a mandatory lane change across a metres-long stub is routine), so
            // the gate binds on the union — a lane-accurate walk followed the
            // sibling exit while the vehicle swapped lanes into the conflict.
            let link = self.network.lane(lane).link;
            for l in self.network.lanes_of(link) {
                let ln = self.network.lane(l);
                for k in 0..ln.movement_count {
                    let m = MovementId(ln.movement_start.0 + k);
                    if self.box_conflict(m, self.network.movement(m).node, nb) {
                        return true;
                    }
                }
            }
            let Some(next) = self.movement_from_lane_for(veh, lane) else { return false };
            lane = self.network.movement(next).to_lane;
        }
        false
    }

    /// A vehicle's committed movement chain through an intersection: `mid` plus
    /// each following hop while the path stays on junction-internal links. This
    /// is the unit admission must reason about — entering a cluster commits the
    /// whole chain, not just the entry stub.
    fn path_movements(&self, veh: &NetVehicle, mid: MovementId) -> Vec<MovementId> {
        let mut out = vec![mid];
        let mut lane = self.network.movement(mid).to_lane;
        for _ in 0..4 {
            if !self.junction_internal_lane(lane) {
                break;
            }
            let Some(next) = self.movement_from_lane_for(veh, lane) else { break };
            out.push(next);
            lane = self.network.movement(next).to_lane;
        }
        out
    }

    /// Whether `o`'s committed path (its next movement plus following internal
    /// hops) reaches `target` — the approach-side counterpart of
    /// [`path_movements`] for gap-acceptance scans at multi-node clusters.
    fn approach_reaches(&self, o: &NetVehicle, target: MovementId) -> bool {
        let Some(first) = self.intended_movement(o) else { return false };
        if first == target {
            return true;
        }
        let mut lane = self.network.movement(first).to_lane;
        for _ in 0..4 {
            if !self.junction_internal_lane(lane) {
                return false;
            }
            let Some(next) = self.movement_from_lane_for(o, lane) else { return false };
            if next == target {
                return true;
            }
            lane = self.network.movement(next).to_lane;
        }
        false
    }

    /// The movement this vehicle would take from `lane`, for path lookahead:
    /// the flow-field hop for destination-routed cars, the route-sequence hop for
    /// explicitly-routed ones — with the same fallback chain [`intended_movement`]
    /// resolves through at the line, so lookahead and the actual hop agree even
    /// when the specific lane serves nothing and a sibling's movement is borrowed.
    fn movement_from_lane_for(&self, veh: &NetVehicle, lane: LaneId) -> Option<MovementId> {
        let routed = if veh.dest.is_some() {
            self.intended_movement_from(veh, lane)
        } else if !veh.route.is_empty() {
            let link = self.network.lane(lane).link;
            veh.route
                .iter()
                .position(|&l| l == link)
                .and_then(|idx| veh.route.get(idx + 1))
                .and_then(|&next| self.movement_to(lane, next))
        } else {
            None
        };
        routed
            .or_else(|| self.forward_movement(veh, lane))
            .or_else(|| {
                let l = self.network.lane(lane);
                (l.movement_count > 0).then_some(l.movement_start)
            })
            .or_else(|| self.any_movement_on(self.network.lane(lane).link))
    }

    /// `Some(())` when a higher-priority vehicle is approaching `node` from a
    /// different link and will arrive within the critical gap — the signal to
    /// give way. `None` means clear to proceed.
    fn conflicting_priority_traffic(&self, i: usize, lane: LaneId, node: NodeId, nb: &Neighbors) -> Option<()> {
        self.conflicting_priority_traffic_scaled(i, lane, node, nb, 1.0)
    }

    /// [`conflicting_priority_traffic`] with the acceptance window scaled — a
    /// `margin > 1` sees priority traffic *earlier*, the horizon at which a
    /// driver eases off and covers the brake rather than the point they must
    /// stand on it.
    fn conflicting_priority_traffic_scaled(
        &self,
        i: usize,
        lane: LaneId,
        node: NodeId,
        nb: &Neighbors,
        margin: f64,
    ) -> Option<()> {
        let me = &self.fleet.rows[i];
        let waited = me.wait_ticks as f64 * self.cfg.dt;
        // Commit hysteresis: a driver who has accepted a gap and is rolling into
        // the crossing no longer re-litigates the full window every tick — only
        // an *imminent* arrival aborts. Without this the acceptance must hold
        // through the entire creep-and-cross (re-tested 5×/s), which squared the
        // odds and starved minor streets that HCM says should trickle through.
        let rolling_commit =
            me.speed > 1.0 && (self.network.lane(lane).length - me.position) < 8.0;
        let hysteresis = if rolling_commit { 0.5 } else { 1.0 };
        // Physical floor: never accept less than the time to clear our own
        // crossing (interior arc plus body at the turn's speed profile).
        let (cap, arc) = self.intended_movement(me).map_or((me.driver.desired_speed, 8.0), |m| {
            (self.turn_speed_cap(m).min(me.driver.desired_speed), self.network.interior(m).len)
        });
        let t_clear = travel_time(arc + me.driver.vehicle_length, me.speed.max(1.0), me.driver.max_accel, cap);
        let my_link = self.network.lane(lane).link;
        let my_key = self.priority_key(my_link);
        let my_dir = self.network.arrival_dir(my_link);
        let my_mid = self.intended_movement(me);
        let my_turn = my_mid.map_or(TurnType::Through, |m| self.network.movement_turn(m));
        for &j in nb.approaching.get(&self.network.intersection_key(node))? {
            if j == i {
                continue;
            }
            let o = &self.fleet.rows[j];
            let o_lane = *self.network.lane(o.lane);
            if o_lane.link == my_link || o.speed < 0.5 {
                continue;
            }
            // HCM base for this conflict pair: minor when the conflicting
            // approach outranks ours; impatience shrinks it, physics floors it.
            let minor = my_key < self.priority_key(o_lane.link);
            // Hysteresis scales only the behavioural window; the physics floor
            // (own remaining clearance, recomputed from current speed) holds.
            let critical = (effective_critical_gap(hcm_critical_gap(my_turn, minor, &me.driver), waited)
                * hysteresis)
                .max(t_clear + 0.3)
                * margin;
            let o_mid = self.intended_movement(o);
            if (o_lane.length - o.position) / o.speed.max(0.1) >= critical {
                continue;
            }
            let o_turn = o_mid.map_or(TurnType::Through, |m| self.network.movement_turn(m));
            // A turn yields to a conflicting through outright — right-of-way
            // doesn't depend on approach angle when the paths genuinely cross
            // (a ramp running parallel to a frontage road before swinging over
            // it defeats the angle heuristic below).
            if my_turn != TurnType::Through && o_turn == TurnType::Through {
                if let (Some(a), Some(b)) = (self.intended_movement(me), o_mid) {
                    if self.network.movements_conflict(a, b) {
                        return Some(());
                    }
                }
            }
            if should_yield_to(my_turn, my_dir, o_turn, self.network.arrival_dir(o_lane.link), my_key, self.priority_key(o_lane.link)) {
                return Some(());
            }
        }
        None
    }

    fn is_rtor(&self, mid: MovementId) -> bool {
        self.network.movement_turn(mid) == TurnType::Right
            && !self.network.is_interchange_movement(mid)
            && self.movement_state(mid) == SignalState::Red
    }

    /// Whether this approach faces a stop line at a stop-controlled node.
    /// Drivers act on the posted signs: a node-level stop (surveyed on the
    /// junction node) lines every approach, while a per-approach sign (OSM
    /// stop/give_way surveyed on the way — the scraper's `sign` field) lines
    /// only its own street, so the cross traffic of a two-way stop rolls
    /// through on its right of way.
    fn approach_must_stop(&self, link: LinkId, _node: NodeId) -> bool {
        self.network.approach_stops(link)
    }

    /// The speed below which this driver treats a stop sign as served. Observed
    /// compliance is famously partial — across observational studies a minority
    /// of drivers come to a complete stop at clear stop signs; most roll through
    /// at a walking pace (the "California stop"). Cautious drivers still plant
    /// the wheel; the aggressive end rolls at up to ~1.3 m/s (3 mph). Yielding to
    /// actual traffic is unaffected — the priority gates hold regardless.
    fn stop_roll_speed(driver: &DriverConfig) -> f64 {
        let aggression = ((driver.desired_speed / 30.0 - 0.85) / 0.30).clamp(0.0, 1.0);
        0.3 + aggression
    }

    /// Whether this driver is blind to `node`'s signal — the rare distraction that
    /// runs a red outright. Decided once per vehicle–node pair (a stateless hash),
    /// so the stop-line constraint and the boundary admission agree, and scaled by
    /// the same speed appetite that drives yellow-running. The box-occupancy gates
    /// still apply: the driver misses the light, not the cars in front of it.
    fn runs_red(&self, veh: &NetVehicle, node: NodeId) -> bool {
        if self.cfg.red_run_prob <= 0.0 {
            return false;
        }
        let aggression = ((veh.driver.desired_speed / 30.0 - 0.85) / 0.30).clamp(0.0, 1.0);
        let p = self.cfg.red_run_prob * (0.5 + aggression);
        rng::uniform01(self.cfg.seed, veh.id, RED_RUN_SALT ^ node.0 as u64, Stream::GapAcceptance) < p
    }

    /// Whether a conflicting movement at `node`'s intersection was already committed
    /// by an earlier car in this tick's serial boundary pass — the same-tick
    /// complement of the pre-step `box_conflict` gate, so simultaneous entries are
    /// governed by driver gap acceptance rather than the tick length.
    fn entered_conflicting(&self, mid: MovementId, node: NodeId, entered_at: &IntMap<Vec<MovementId>>) -> bool {
        entered_at
            .get(&self.network.intersection_key(node))
            .into_iter()
            .flatten()
            .any(|&o| self.network.movements_conflict(mid, o))
    }

    /// In-box speed for a movement: at-grade turns throttle to a crawl; throughs
    /// and freeway diverge/merge ramps run at road speed (the ramp's own limit
    /// and curvature slow those, not a hard crawl through the gore).
    fn turn_speed_cap(&self, mid: MovementId) -> f64 {
        self.turn_caps[mid.idx()]
    }

    fn left_is_permissive(&self, mid: MovementId) -> bool {
        let node = self.network.movement(mid).node;
        self.junctions.conflict_ids(node).iter().any(|&ci| {
            let cp = &self.network.conflicts[ci as usize];
            let other = if cp.a == mid { cp.b } else if cp.b == mid { cp.a } else { return false };
            self.movement_state(other) != SignalState::Red
        })
    }

    fn is_permissive(&self, mid: MovementId) -> bool {
        if self.network.is_interchange_movement(mid) {
            return false;
        }
        match self.network.movement_turn(mid) {
            TurnType::Right => self.movement_state(mid) == SignalState::Red,
            TurnType::Left => {
                self.network.movement(mid).signal_group.is_some()
                    && self.movement_state(mid) == SignalState::Green
                    && self.left_is_permissive(mid)
            }
            TurnType::Through => false,
        }
    }

    /// Interior arc of `mid`'s first conflict point against a currently live
    /// (non-red) movement — where a mid-box waiter stands short of.
    fn first_live_conflict_arc(&self, mid: MovementId, node: NodeId) -> Option<f64> {
        let mut first = f64::INFINITY;
        for &ci in self.junctions.conflict_ids(node) {
            let cp = &self.network.conflicts[ci as usize];
            let (my_s, other) = if cp.a == mid {
                (cp.sa, cp.b)
            } else if cp.b == mid {
                (cp.sb, cp.a)
            } else {
                continue;
            };
            if self.movement_state(other) != SignalState::Red {
                first = first.min(my_s);
            }
        }
        first.is_finite().then_some(first)
    }

    /// The interior arc a green permissive left may advance to and *wait* at
    /// while oncoming still owns the window — the box-committed left every
    /// driver performs — or `None` when it must keep yielding at the line. The
    /// interior must be deep enough to stand `PERMISSIVE_HOLD_MARGIN` short of
    /// the first live conflict point, and only one waiter per movement is
    /// admitted (a queue in the box could not clear on the change interval).
    fn permissive_waiter_hold(&self, i: usize, mid: MovementId, node: NodeId, nb: &Neighbors) -> Option<f64> {
        if self.network.movement_turn(mid) != TurnType::Left || !self.is_permissive(mid) {
            return None;
        }
        let hold = self.first_live_conflict_arc(mid, node)? - PERMISSIVE_HOLD_MARGIN;
        if hold < 1.0 {
            return None; // too shallow to stand in
        }
        let key = self.network.intersection_key(node);
        let taken = nb
            .crossing_at
            .get(&key)
            .into_iter()
            .flatten()
            .any(|&j| j != i && self.fleet.rows[j].crossing.is_some_and(|c| c.movement == mid));
        (!taken).then_some(hold)
    }

    /// Whether the lane a movement feeds into is occupied right at its entrance, so a vehicle
    /// taking it couldn't land and must hold at the line. Shared by the stop-line gate and the
    /// all-way-stop FIFO (which must not keep yielding to a car that itself can't move).
    fn movement_downstream_blocked(&self, mid: MovementId, driver: &DriverConfig, nb: &Neighbors) -> bool {
        if self.network.is_interchange_movement(mid) {
            return false;
        }
        let to_lane = self.network.movement(mid).to_lane;
        nb.lane_front.get(&to_lane.0).is_some_and(|&f| {
            let o = &self.fleet.rows[f];
            // Occupant *rear* vs the room this whole vehicle needs to clear the box —
            // mirrors the serial admission gate (`receiving_room`), including its
            // departing-tail exemption: a moving occupant is a leader to follow
            // into the box, at a real following distance along the continuous path.
            let rear = o.position - o.driver.vehicle_length;
            if o.speed >= DEPARTING_SPEED && self.departing_exemption(mid) {
                return self.network.interior(mid).len + rear < departing_margin(driver, o.speed);
            }
            let unit = driver.vehicle_length + driver.min_gap;
            rear < self.commit_room_needed(to_lane, unit)
        })
    }

    fn earlier_stopped_conflict(&self, i: usize, mid: MovementId, node: NodeId, nb: &Neighbors) -> bool {
        let me = &self.fleet.rows[i];
        let key = self.network.intersection_key(node);
        nb.approaching.get(&key).into_iter().flatten().any(|&j| {
            if j == i {
                return false;
            }
            let o = &self.fleet.rows[j];
            // Turn-taking runs between the cars facing the intersection: the armed
            // front driver of each approach — the drivers who can actually exchange
            // the right-of-way and proceed when their turn comes. "The intersection"
            // is the whole cluster: at a multi-node all-way stop the approaches arm
            // at different member nodes, and matching on one node made them
            // mutually invisible — the heavier street streamed forever while the
            // cross street waited out the run.
            let armed_here = o.stopped_at.is_some_and(|n| self.network.intersection_key(n) == key);
            if o.speed > 0.5 || !armed_here || nb.lane_front.get(&o.lane.0) != Some(&j) {
                return false;
            }
            let stopped_earlier = o.wait_ticks > me.wait_ticks || (o.wait_ticks == me.wait_ticks && o.id < me.id);
            // Don't keep waiting on an earlier-stopped car that can't move anyway — its own exit
            // lane is blocked. Skipping it lets this car take its turn once its own path is clear,
            // which breaks the all-way-stop deadlock where everyone defers to a stuck leader. The
            // hard box-occupancy and approaching-priority gates still apply, so this stays safe.
            stopped_earlier
                && self.intended_movement(o).is_some_and(|o_mid| {
                    self.network.movements_conflict(mid, o_mid) && !self.movement_downstream_blocked(o_mid, &o.driver, nb)
                })
        })
    }

    fn permissive_must_yield(&self, i: usize, mid: MovementId, node: NodeId, nb: &Neighbors) -> bool {
        let me = &self.fleet.rows[i];
        let my_line = (self.network.lane(me.lane).length - me.position).max(0.0);
        self.permissive_pressure(i, mid, node, nb, &|my_s| Some(my_line + my_s))
    }

    /// Whether any of `mid`'s conflict points still demands waiting: a crosser
    /// inside the box owning a shared point during this driver's arrival window,
    /// or approaching conflict traffic inside the accepted gap. `dist_of(my_s)`
    /// is this driver's remaining distance to a conflict point at interior arc
    /// `my_s` (`None`: already past it) — measured from the stop line for the
    /// gap-acceptance yield, from the current interior arc for the mid-box
    /// waiter's hold.
    fn permissive_pressure(
        &self,
        i: usize,
        mid: MovementId,
        node: NodeId,
        nb: &Neighbors,
        dist_of: &dyn Fn(f64) -> Option<f64>,
    ) -> bool {
        let me = &self.fleet.rows[i];
        let key = self.network.intersection_key(node);
        self.junctions.conflict_ids(node).iter().any(|&ci| {
            let cp = &self.network.conflicts[ci as usize];
            let (my_s, other, other_s) = if cp.a == mid {
                (cp.sa, cp.b, cp.sb)
            } else if cp.b == mid {
                (cp.sb, cp.a, cp.sa)
            } else {
                return false;
            };
            let other_state = self.movement_state(other);
            if other_state == SignalState::Red {
                return false;
            }
            let Some(my_dist) = dist_of(my_s) else {
                return false; // already past this point
            };
            // A conflicting car inside the box blocks only while it still owns
            // the shared point during our own arrival window (graded — a crosser
            // whose tail has swept past frees the movement immediately). Future
            // cluster hops onto `other` stay binary.
            for &j in nb.crossing_at.get(&key).into_iter().flatten() {
                if j == i {
                    continue;
                }
                let o = &self.fleet.rows[j];
                let cm = o.crossing.unwrap().movement;
                if cm != other {
                    if o.speed >= 0.5 {
                        let mut olane = self.network.movement(cm).to_lane;
                        for _ in 0..4 {
                            if !self.junction_internal_lane(olane) {
                                break;
                            }
                            let Some(next) = self.movement_from_lane_for(o, olane) else { break };
                            if next == other {
                                return true;
                            }
                            olane = self.network.movement(next).to_lane;
                        }
                    }
                    continue;
                }
                let o_arc = self.crossing_arc(o);
                let tail_clear = other_s + o.driver.vehicle_length + 1.0;
                if o_arc >= tail_clear {
                    continue;
                }
                // A stationary waiter short of the point doesn't claim it (the
                // same rule the admission gate and in-box avoidance apply).
                if o.speed < 0.5 && other_s - o_arc > 1.5 {
                    continue;
                }
                let t_clear = (tail_clear - o_arc) / o.speed.max(0.8);
                let cap_me = self.turn_speed_cap(mid).min(me.driver.desired_speed);
                let t_reach = travel_time(my_dist, me.speed.max(1.0), me.driver.max_accel, cap_me);
                if t_reach < t_clear + 0.6 {
                    return true;
                }
            }
            // The window this driver needs: the physical time to clear the conflict
            // point (line → point plus the body, on the turn's speed profile) plus a
            // margin. The HCM gap for the movement can demand more than physics; it
            // can never shave below it — the artifact that let a slow left accept a
            // 3 s gap it needed 6 s to survive. A permissive left against opposing
            // green flow is the HCM major-left case; an RTOR faces cross traffic
            // like a minor right.
            let cap = self.turn_speed_cap(mid).min(me.driver.desired_speed);
            let t_clear = travel_time(my_dist + me.driver.vehicle_length, me.speed.max(1.0), me.driver.max_accel, cap);
            let minor = self.network.movement_turn(mid) == TurnType::Right;
            // Impatience shrinks the accepted gap as the wait grows, exactly as
            // the priority-yield path does — a sneaker eventually takes the 3 s
            // headway a fresh arrival wouldn't — floored by the physical
            // crossing time, which no amount of waiting can shave.
            let waited = me.wait_ticks as f64 * self.cfg.dt;
            let base_gap = hcm_critical_gap(self.network.movement_turn(mid), minor, &me.driver);
            let needed = (t_clear + 1.0).max(effective_critical_gap(base_gap, waited));
            let from_link = self.network.lane(self.network.movement(other).from_lane).link;
            nb.approaching.get(&key).into_iter().flatten().any(|&j| {
                let o = &self.fleet.rows[j];
                let o_lane = *self.network.lane(o.lane);
                // The conflicting movement's own from-link, or an approach whose
                // committed chain reaches it — at a multi-node cluster the real
                // stream arrives on an outer link and hops internal stubs onto
                // the conflicting movement, invisible to a from-link-only scan.
                if o_lane.link != from_link && !self.approach_reaches(o, other) {
                    return false;
                }
                let line_dist = o_lane.length - o.position;
                if o.speed >= 0.5 {
                    // Arrival measured at the conflict point, not the line — and a
                    // mover already stopping for its own yellow never arrives.
                    let stopping = other_state != SignalState::Green
                        && o.speed * o.speed / (2.0 * o.driver.comfort_decel) < line_dist;
                    !stopping && (line_dist + other_s) / o.speed.max(0.1) < needed
                } else {
                    // The queued front car under a green launches into this same
                    // window: startup lag plus kinematics to the conflict point.
                    other_state == SignalState::Green
                        && nb.lane_front.get(&o.lane.0) == Some(&j)
                        && 1.0 + travel_time(
                            line_dist + other_s,
                            0.0,
                            o.driver.max_accel,
                            o.driver.capped_to(o_lane.speed_limit).desired_speed,
                        ) < needed
                }
            })
        })
    }

    /// Floor an obstacle's gap so IDM cannot demand a harder-than-physical deceleration for
    /// it. Used only for leaders this car cannot actually land on — one across a segment seam
    /// (same corridor or gate-protected) or still crossing the node — where continuous
    /// corridor following (or the crossing gate) already prevents the collision. There the gap
    /// collapses momentarily in the boundary hand-off, and unbounded IDM would slam the car
    /// dead at freeway speed; capped, it eases down behind flowing traffic instead. Invert IDM
    /// `a·(free − (s*/s)²) ≥ −MAX_BRAKE_DECEL` for `s`.
    fn cap_leader_brake(&self, driver: &DriverConfig, speed: f64, ob: Obstacle) -> Obstacle {
        let s_star = driver.min_gap
            + speed * driver.time_headway
            + speed * (speed - ob.speed) / (2.0 * (driver.max_accel * driver.comfort_decel).sqrt());
        let free = 1.0 - (speed / driver.desired_speed.max(0.1)).powf(driver.accel_exponent);
        let headroom = (free + MAX_BRAKE_DECEL / driver.max_accel).max(1e-3);
        let g_min = s_star.max(0.0) / headroom.sqrt();
        Obstacle { gap: ob.gap.max(g_min), speed: ob.speed }
    }

    /// The nearest leader ahead of a car that has none on its own lane, found by walking
    /// the through-continuation chain across segment boundaries and accumulating the
    /// distance to it (interiors included). This gives continuous car-following across
    /// nodes — the car sees a leader one or two segments ahead and closes on it smoothly,
    /// rather than losing it at the boundary and braking to a standstill when it reappears.
    fn cross_boundary_leader(&self, veh: &NetVehicle, intended: Option<MovementId>, nb: &Neighbors) -> Option<Obstacle> {
        // Scan as far as this car could need to brake — its speed-scaled stopping distance
        // — spanning however many downstream segments that covers.
        let horizon = leader_horizon(&veh.driver, veh.speed);
        // Distance from this car's front to the end of its current lane (the first node).
        let mut dist = self.network.lane(veh.lane).length - veh.position;
        // First hop follows the car's actual next movement (it may be diverging off);
        // beyond that, the straight-through continuation of each lane.
        let mut hop = intended.map(|mid| (self.network.movement(mid).to_lane, self.network.interior(mid).len));
        let mut hops = 0;
        while let Some((to_lane, interior)) = hop {
            if dist > horizon || hops > 32 {
                break;
            }
            hops += 1;
            // Gap to a leader on `to_lane`, measured through the node interior in the
            // same continuous corridor coordinate the leader itself uses while crossing.
            // This is what makes the seam invisible: the instant a leader lands (its
            // `position` rebasing from `lane.length + interior` to ~0 on `to_lane`) the
            // gap this follower sees is unchanged, so nothing jumps or slams.
            if let Some(&front) = nb.lane_front.get(&to_lane.0) {
                let lead = &self.fleet.rows[front];
                let gap = dist + interior + lead.position - lead.driver.vehicle_length;
                return Some(Obstacle { gap: gap.max(veh.driver.min_gap), speed: lead.speed });
            }
            dist += interior + self.network.lane(to_lane).length; // cross the node, traverse the empty segment
            hop = self.through_next[to_lane.0 as usize];
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
        // Only zipper once actually approaching the merge. Far up the lane the streams
        // are still lanes apart; their distance-to-merge difference is not a bumper gap,
        // and treating it as one phantom-braked cars to a stop long before the merge.
        if my_dist > MERGE_APPROACH {
            return None;
        }
        let my_kind = self.network.link(self.network.lane(veh.lane).link).kind;
        let mut best: Option<Obstacle> = None;
        for &from in froms {
            if from == veh.lane.0 {
                continue;
            }
            let from_link = self.network.lane(LaneId(from)).link;
            let from_kind = self.network.link(from_link).kind;
            for &j in nb.by_lane.get(&from).into_iter().flatten() {
                let o = &self.fleet.rows[j];
                if o.speed < 0.5 {
                    continue;
                }
                let o_dist = self.network.lane(o.lane).length - o.position;
                // Right-of-way at the merge: an on-ramp yields to the mainline it joins,
                // never the reverse — mainline through-traffic must not brake for a
                // merger (the phantom highway stops). A higher-priority road (faster,
                // wider) likewise never cooperatively yields to a minor-approach merger:
                // the minor faces the give-way, the major holds speed — without this a
                // through car on the major brake-checked for a car creeping at a yield
                // triangle, collapsing its speed until the minor "legitimately" took the
                // gap. Only between genuinely equal roads (a lane drop, ramp-to-ramp)
                // does the car closer to the merge point go first.
                let yield_to_o = match (my_kind, from_kind) {
                    (RoadKind::Freeway, RoadKind::Ramp) => false,
                    (RoadKind::Ramp, RoadKind::Freeway) => true,
                    _ => {
                        // Rank = the id-free prefix of the priority key (speed, lanes).
                        let my_link = self.network.lane(veh.lane).link;
                        let (my_rank, o_rank) = (self.priority_key(my_link) >> 24, self.priority_key(from_link) >> 24);
                        if my_rank != o_rank {
                            o_rank > my_rank
                        } else {
                            o_dist < my_dist
                        }
                    }
                };
                if !yield_to_o {
                    continue;
                }
                // Follow only a car genuinely ahead in the merge (positive gap). A car
                // level with or behind us (gap ≤ 0) is not a leader — treating it as one
                // gives IDM a negative gap and stops us dead, which is how ramps got stuck
                // waiting for mainline traffic still far behind them. Once both are on the
                // merged lane, ordinary car-following settles who trails whom.
                let gap = my_dist - o_dist - o.driver.vehicle_length;
                if gap > 0.0 && best.is_none_or(|b| gap < b.gap) {
                    best = Some(Obstacle { gap, speed: o.speed });
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
    /// Movements each intersection's crossers occupy *or will still traverse*
    /// before leaving it: the current interior plus every following hop while the
    /// path stays on junction-internal links. Box gating tests against this, so a
    /// crosser mid-cluster reserves the conflicting movement it is about to swing
    /// onto — not only the stub it happens to be on this tick.
    crossing_mvs: IntMap<Vec<MovementId>>,
    /// The same reservation restricted to *moving* crossers (v ≥ 0.5). The in-box
    /// next-hop brake yields only to these: a stalled crosser is the conflict-point
    /// serializer's problem (which totally orders and cannot deadlock), and holding
    /// mid-box for a stationary one builds hold-for-each-other cycles that camp
    /// cars inside the junction.
    moving_crossing_mvs: IntMap<Vec<MovementId>>,
}

/// IDM acceleration for a vehicle placed at `pos`/`speed` on a lane with the
/// given speed limit, following `leader` (or free road if none). Used to score
/// hypothetical lane placements for MOBIL.
/// Turning-movement conflict rule: whether a vehicle making `my_turn` from
/// direction `my_dir` must yield to one making `o_turn` from `o_dir`.
/// - Same direction (parallel): no node conflict (car-following handles it).
/// - Opposing: only a left turn yields (a right turns away from oncoming).
/// - Crossing: a through or right turn yields to the higher-priority approach.
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
    if rel > 2.5 {
        return my_turn == TurnType::Left && o_turn != TurnType::Left; // opposing: left yields, right turns away
    }
    o_key > my_key // crossing — a through or right turn defers to the major approach
}

/// Whether a vehicle can brake to a stop within `distance` at comfortable
/// deceleration — the dilemma-zone test for whether to stop on yellow.
fn can_stop_before(speed: f64, decel: f64, distance: f64) -> bool {
    speed * speed / (2.0 * decel.max(0.1)) <= distance
}

fn yellow_run_prob(driver: &DriverConfig) -> f64 {
    let aggression = (driver.desired_speed / 30.0 - 1.0).max(0.0);
    (0.08 + aggression * 1.5).clamp(0.0, 0.35)
}

/// Whether two vehicle bodies genuinely intersect. Each body is the oriented
/// rectangle behind its pose (`[x, y, heading]` at the *front bumper*), `len` long
/// and `width` wide; the test is the exact separating-axis check on the two
/// rectangles. Ground truth replaces the old centre-distance + heading heuristics,
/// which missed strikes into a body's rear half and flagged close parallel passes.
fn body_overlap(pa: [f64; 3], la: f64, wa: f64, pb: [f64; 3], lb: f64, wb: f64) -> bool {
    let dot = |p: [f64; 2], q: [f64; 2]| p[0] * q[0] + p[1] * q[1];
    let ua = [pa[2].cos(), pa[2].sin()];
    let va = [-ua[1], ua[0]];
    let ub = [pb[2].cos(), pb[2].sin()];
    let vb = [-ub[1], ub[0]];
    let ca = [pa[0] - ua[0] * la * 0.5, pa[1] - ua[1] * la * 0.5];
    let cb = [pb[0] - ub[0] * lb * 0.5, pb[1] - ub[1] * lb * 0.5];
    let d = [cb[0] - ca[0], cb[1] - ca[1]];
    for axis in [ua, va, ub, vb] {
        let ra = la * 0.5 * dot(ua, axis).abs() + wa * 0.5 * dot(va, axis).abs();
        let rb = lb * 0.5 * dot(ub, axis).abs() + wb * 0.5 * dot(vb, axis).abs();
        if dot(d, axis).abs() > ra + rb {
            return false;
        }
    }
    true
}

/// The gap a driver will accept, shrinking from `base` as `waited` grows
/// (impatience), floored so nobody nudges into genuinely unsafe traffic. The
/// floor sits under the HCM bases the way observed impatient drivers do —
/// well below book values, never into physically blind acceptance.
fn effective_critical_gap(base: f64, waited: f64) -> f64 {
    (base - 0.15 * waited).max(2.8)
}

/// HCM 6th-ed base critical headways for unsignalized conflict, by movement
/// class: a left from the priority street crosses only the opposing stream
/// (4.1 s); minor-street movements face the full priority flow (right 6.2 s,
/// through 6.5 s, left 7.1 s). Heavy vehicles add ~1 s. Scaled by the driver's
/// sampled `critical_gap / 4.0`, so population heterogeneity (±20%) rides on
/// the book values.
fn hcm_critical_gap(turn: TurnType, minor: bool, driver: &DriverConfig) -> f64 {
    let base = match (minor, turn) {
        (false, _) => 4.1,
        (true, TurnType::Right) => 6.2,
        (true, TurnType::Through) => 6.5,
        (true, TurnType::Left) => 7.1,
    };
    let heavy = if driver.vehicle_length >= 8.0 { 1.0 } else { 0.0 };
    (base + heavy) * (driver.critical_gap / 4.0)
}

/// Seconds to cover `dist` from speed `v0`, accelerating at `a` toward `v_max` —
/// the kinematics behind clearance-time gap acceptance.
fn travel_time(dist: f64, v0: f64, a: f64, v_max: f64) -> f64 {
    if dist <= 0.0 {
        return 0.0;
    }
    let (a, v_max) = (a.max(0.1), v_max.max(0.5));
    let v0 = v0.min(v_max);
    let d_accel = (v_max * v_max - v0 * v0) / (2.0 * a);
    if d_accel >= dist {
        ((v0 * v0 + 2.0 * a * dist).sqrt() - v0) / a
    } else {
        (v_max - v0) / a + (dist - d_accel) / v_max
    }
}

fn idm_follow(follower: &NetVehicle, lane_speed_limit: f64, pos: f64, speed: f64, leader: Option<&NetVehicle>) -> f64 {
    let d = follower.driver.capped_to(lane_speed_limit);
    match leader {
        Some(l) => idm::acceleration(&d, speed, speed - l.speed, (l.position - pos - l.driver.vehicle_length).max(0.05)),
        None => idm::free_acceleration(&d, speed),
    }
}

fn integrate(v: &mut NetVehicle, accel: f64, dt: f64) {
    // Grip bound: IDM's interaction term is unbounded as the gap shrinks, but a real
    // car cannot shed speed faster than [`MAX_BRAKE_DECEL`]. Anything the fold demands
    // beyond it is a crash for the detector to find, not a teleport-stop.
    let accel = accel.max(-MAX_BRAKE_DECEL);
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
    use super::super::rush_hour::SurfaceClass;
    use super::*;


    /// A car crossing a freeway segment must not come to rest with open road ahead at a
    /// free-flow point. Runs the peninsula freeways at a quarter of capacity (they flow,
    /// so any standstill is phantom) and, the moment a freeway car has been stopped ~1 s
    /// on an uncontrolled node's approach with a clear leader gap, records the binding
    /// constraint. The regression this guards: `merge_conflict` gave ramps a negative gap
    /// to mainline traffic still far behind them, so they stalled at every interchange.

    #[test]
    fn freeway_traffic_does_not_stall_at_free_flow_points() {
        use super::super::demand::{self, DemandGenerator, DemandSources};
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/peninsula.json");
        let Ok(text) = std::fs::read_to_string(path) else { return };
        let net = super::super::map::OsmMap::from_json(&text).expect("map json").build();
        let pairs = demand::od_pairs(&net, 0, 600, DemandSources::new(true, false));
        let mut world = NetWorld::new(net, cfg());
        let mut gen = DemandGenerator::new(&world, &pairs, 0);
        gen.set_rate_scale(0.25);
        world.install_router(&gen.destinations());
        use std::collections::BTreeMap;
        let mut cause: BTreeMap<&str, u32> = BTreeMap::new();
        for _ in 0..2500 {
            gen.step(&mut world, cfg().dt);
            world.step();
            let newly_stuck: Vec<usize> = (0..world.fleet.rows.len())
                .filter(|&i| {
                    let v = &world.fleet.rows[i];
                    let node = world.network.link(world.network.lane(v.lane).link).to;
                    v.crossing.is_none() && v.wait_ticks == 25 && v.speed < 1.0
                        && world.network.lane(v.lane).speed_limit >= 22.0
                        && matches!(world.network.node(node).control, NodeControl::Uncontrolled)
                })
                .collect();
            if newly_stuck.is_empty() {
                continue;
            }
            let nb = world.neighbors();
            let intended: Vec<Option<MovementId>> =
                world.fleet.rows.iter().map(|v| if v.crossing.is_some() { None } else { world.intended_movement(v) }).collect();
            for i in newly_stuck {
                let cx = world.gather_context(i, &nb, intended[i]);
                if cx.leader_gap < 12.0 {
                    continue; // genuinely behind a close leader — a queue, not a phantom stall
                }
                let label = if cx.merge_gap.is_finite() { "merge" }
                    else if cx.yield_line.is_finite() { "yield_line" }
                    else if cx.stop_line.is_finite() { "stop_line" }
                    else if cx.stop_sign.is_finite() { "stop_sign" }
                    else if cx.curve_speed < 3.0 { "curve" }
                    else { "other" };
                *cause.entry(label).or_insert(0) += 1;
            }
        }
        assert_eq!(cause.get("merge").copied().unwrap_or(0), 0, "no freeway car stalls yielding at a merge: {cause:?}");
        assert_eq!(cause.get("yield_line").copied().unwrap_or(0), 0, "no freeway car stalls at a phantom yield: {cause:?}");
        let total: u32 = cause.values().sum();
        assert!(total <= 3, "freeway cars almost never stall on open road at a free-flow point: {cause:?}");
    }


    #[test]
    fn a_car_brakes_smoothly_for_a_slow_leader_several_segments_ahead() {
        // A motorway chopped into short 40 m segments, exactly as OSM splits a freeway
        // into a new way at every node. A slow car crawls *four* boundaries ahead (~165 m
        // — inside a 29 m/s car's ~184 m stopping-distance horizon, but far beyond a naive
        // one- or two-segment view). The fast car must see it across all four boundaries
        // and ease down, rather than cruise blindly and slam when the crawler finally
        // enters a single-segment view. Guards the speed-scaled cross-boundary leader walk.
        let hw = |a, b| LinkSpec { road_class: "motorway".into(), ..LinkSpec::oneway(a, b, 1, 29.0) };
        let net = OsmMap {
            nodes: (0..7).map(|k| NodeSpec::uncontrolled(k + 1, k as f64 * 40.0, 0.0)).collect(),
            links: (0..6).map(|k| hw(k + 1, k + 2)).collect(),
        }
        .build();
        let mut w = NetWorld::new(net, cfg());
        w.install_router(&[LinkId(5)]); // everyone bound for the last segment
        // A genuine slow crawler (desired speed 5 m/s) four boundaries ahead, on link 4.
        let crawler = DriverConfig { desired_speed: 5.0, ..DriverConfig::car() };
        let slow_lane = w.network.lanes_of(LinkId(4)).next().unwrap();
        w.spawn_to_in_lane(1, slow_lane, 5.0, LinkId(5), 5.0, crawler);
        let fast_lane = w.network.lanes_of(LinkId(0)).next().unwrap();
        w.spawn_to_in_lane(2, fast_lane, 5.0, LinkId(5), 29.0, DriverConfig::car());
        let mut prev = 29.0f64;
        let mut worst_drop = 0.0f64;
        let mut crossed_boundaries = 0u32;
        let mut last_link = 0u32;
        let mut saw_slowing = false;
        for _ in 0..250 {
            w.step();
            if let Some(v) = w.vehicle(2) {
                worst_drop = worst_drop.max(prev - v.speed);
                saw_slowing |= v.speed < 12.0; // it did have to slow for the crawler
                let link = w.network.lane(v.lane).link.0;
                if link != last_link {
                    crossed_boundaries += 1;
                    last_link = link;
                }
                prev = v.speed;
            }
        }
        assert!(saw_slowing, "the fast car actually catches the crawler and must slow");
        assert!(crossed_boundaries >= 3, "it followed the crawler across several segments, crossed {crossed_boundaries}");
        assert!(
            worst_drop < 3.0,
            "it eases down across the segment boundaries instead of slamming, worst one-tick drop {worst_drop:.1} m/s",
        );
        assert_eq!(w.crashed(), 0, "and never rear-ends the crawler across a boundary");
    }

    #[test]
    fn freeway_entrants_keep_their_entry_speed_under_heavy_inflow() {
        // A long freeway fed from a gateway as fast as it will take cars. Each admitted
        // car enters at freeway speed and must keep moving: the gateway only admits it
        // with a full following gap, so it never appears a couple of metres behind the
        // last entrant and brakes to a standstill (the "cars enter the freeway at 0" bug).
        let hw = |a, b, lanes, sp| LinkSpec { road_class: "motorway".into(), ..LinkSpec::oneway(a, b, lanes, sp) };
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),    // entry gateway
                NodeSpec::uncontrolled(2, 3000.0, 0.0), // exit gateway
            ],
            links: vec![hw(1, 2, 2, 29.0)],
        }
        .build();
        let mut w = NetWorld::new(net, cfg());
        w.install_router(&[LinkId(0)]); // everyone bound for the far exit
        let mut slowest_settled: f64 = f64::INFINITY;
        for t in 0..600 {
            w.spawn_to(t, LinkId(0), LinkId(0), 29.0, DriverConfig::car()); // admits only when a gap exists
            w.step();
            if t > 200 {
                // Past the first 20 m (spawn/startup), every car on the free-flowing entry
                // link should be cruising, not stalled behind a too-close leader.
                for v in w.vehicles() {
                    if w.network.lane(v.lane).link == LinkId(0) && v.position > 20.0 {
                        slowest_settled = slowest_settled.min(v.speed);
                    }
                }
            }
        }
        assert!(slowest_settled.is_finite(), "the gateway admits and carries a stream of cars");
        assert!(
            slowest_settled > 12.0,
            "freeway entrants keep highway speed rather than braking to a crawl at entry, slowest {slowest_settled:.1} m/s",
        );
        assert_eq!(w.crashed(), 0, "and the entering stream stays collision-free");
    }

    #[test]
    fn mainline_through_traffic_does_not_yield_to_an_on_ramp_merger() {
        // A two-lane freeway (29 m/s) with an on-ramp joining its curb lane. A slow car
        // sits on the ramp right at the merge; a mainline car approaches at speed in the
        // curb lane. Real right-of-way: the ramp yields to the mainline, so the mainline
        // car must keep highway speed — not brake for the merger. This is the regression
        // for "cars slow to 0 on the highway": the merge model used to make the mainline
        // yield to the ramp, stopping it dead before the merge.
        let hw = |a, b, lanes, sp| LinkSpec { road_class: "motorway".into(), ..LinkSpec::oneway(a, b, lanes, sp) };
        let ramp = |a, b, lanes, sp| LinkSpec { road_class: "motorway_link".into(), ..LinkSpec::oneway(a, b, lanes, sp) };
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -400.0, 0.0),   // mainline in
                NodeSpec::uncontrolled(2, 0.0, 0.0),      // merge point
                NodeSpec::uncontrolled(3, 400.0, 0.0),    // mainline out
                NodeSpec::uncontrolled(4, -180.0, -180.0), // on-ramp origin
            ],
            links: vec![hw(1, 2, 2, 29.0), hw(2, 3, 2, 29.0), ramp(4, 2, 1, 25.0)],
        }
        .build();
        // The ramp genuinely merges into the mainline's curb lane.
        let curb_in = net.lanes_of(LinkId(0)).last().unwrap();
        let ramp_lane = net.lanes_of(LinkId(2)).next().unwrap();
        let out_lane = net.movements_of(curb_in)[0].to_lane;
        assert!(
            net.movements_of(ramp_lane).iter().any(|m| m.to_lane == out_lane),
            "the ramp feeds the same downstream lane as the mainline curb lane — a merge",
        );

        let mut w = NetWorld::new(net, cfg());
        w.install_router(&[LinkId(1)]); // both bound for the mainline-out link
        // Slow ramp car right at the merge, and a fast mainline car a short way back in
        // the curb lane — close enough that the old symmetric zipper would fire.
        let ramp_len = w.network.lane(ramp_lane).length;
        w.spawn_to_in_lane(1, ramp_lane, ramp_len - 6.0, LinkId(1), 4.0, DriverConfig::car());
        let curb_len = w.network.lane(curb_in).length;
        w.spawn_to_in_lane(2, curb_in, curb_len - 12.0, LinkId(1), 27.0, DriverConfig::car());
        let mut min_mainline: f64 = f64::INFINITY;
        for _ in 0..60 {
            w.step();
            if let Some(v) = w.vehicle(2) {
                min_mainline = min_mainline.min(v.speed);
            }
        }
        assert!(
            min_mainline > 15.0,
            "mainline car holds highway speed past the on-ramp (min {min_mainline:.1} m/s), never yielding to the merger",
        );
        assert_eq!(w.crashed(), 0, "and the merge stays collision-free");
    }

    #[test]
    fn a_car_following_across_a_short_segment_boundary_is_never_slammed() {
        // I-280 over Hillcrest Blvd on Millbrae: a straight freeway with no turns or lane
        // changes, chopped by OSM into short segments (the overpass is ~27 m). Under load a
        // follower catches its leader right as the leader reaches a segment boundary and
        // begins crossing the node. In continuous road coordinates the two briefly overlap
        // (the leader is less than a car length past the boundary), so the cross-boundary
        // leader gap goes negative. Floored to a hair (0.05 m) that fed IDM an unphysical
        // ~-2000 m/s^2 brake that slammed the follower to a dead stop, then it re-accelerated
        // and the next car repeated it — the phantom stop-and-go that stalled I-280. The gap
        // must floor at the driver's min_gap so a boundary catch-up brakes at a *physical*
        // rate, letting the queue discharge instead of hammering itself to a standstill.
        let hw = |a, b| LinkSpec { road_class: "motorway".into(), ..LinkSpec::oneway(a, b, 1, 29.0) };
        let mut nodes: Vec<NodeSpec> = vec![NodeSpec::uncontrolled(1, 0.0, 0.0)];
        let mut x = 0.0;
        for k in 0..5 {
            x += if k == 2 { 27.0 } else { 120.0 }; // one short overpass segment among longer ones
            nodes.push(NodeSpec::uncontrolled(k + 2, x, 0.0));
        }
        let net = OsmMap { nodes, links: (0..5).map(|k| hw(k + 1, k + 2)).collect() }.build();
        let mut w = NetWorld::new(net, cfg());
        w.install_router(&[LinkId(4)]); // everyone bound for the last segment
        // Two cars queued nose-to-tail across a segment boundary: the leader has just
        // crossed onto the short overpass segment while the follower sits right at the
        // boundary a link back, the continuous-coordinate gap between them a hair above
        // zero — the state whose floored (0.05 m) gap fed IDM the ~-2000 m/s^2 slam.
        // (The bodies must not overlap: an overlap is now a detected rear-end crash.)
        let crawl = DriverConfig { desired_speed: 2.0, ..DriverConfig::car() };
        let lead_lane = w.network.lanes_of(LinkId(2)).next().unwrap();
        w.spawn_to_in_lane(1, lead_lane, 4.3, LinkId(4), 0.3, crawl);
        let chase_lane = w.network.lanes_of(LinkId(1)).next().unwrap();
        let chase_len = w.network.lane(chase_lane).length;
        w.spawn_to_in_lane(2, chase_lane, chase_len - 1.0, LinkId(4), 0.3, crawl);
        let mut worst_brake = 0.0f64;
        for _ in 0..80 {
            let nb = w.neighbors();
            let intended: Vec<Option<MovementId>> = w
                .fleet
                .rows
                .iter()
                .map(|v| if v.crossing.is_some() { None } else { w.intended_movement(v) })
                .collect();
            for i in 0..w.fleet.rows.len() {
                if w.fleet.rows[i].id == 2 && w.fleet.rows[i].crossing.is_none() {
                    worst_brake = worst_brake.min(w.gather_context(i, &nb, intended[i]).binding());
                }
            }
            w.step();
        }
        assert_eq!(w.crashed(), 0, "the queued pair never registers a rear-end across the seam");
        assert!(
            worst_brake > -12.0,
            "a follower queued behind a leader that just crossed a segment boundary brakes at a \
             physical rate, not an unphysical slam (worst commanded {worst_brake:.0} m/s^2)",
        );
    }

    #[test]
    fn a_dense_queue_discharges_across_a_seam_without_a_gap_discontinuity() {
        // The continuous-corridor guarantee under *normal* dynamics (not an artificial
        // overlapping spawn): a standing queue on a straight freeway chopped into short
        // segments discharges across the node seams. Under the old pin-and-teleport model
        // a follower lost its leader at each boundary and took an unphysical brake as the
        // leader reappeared downstream — the phantom stop-and-go. With continuous position
        // the gap stays smooth across the seam, so no car is ever commanded a super-physical
        // deceleration and none collide.
        let hw = |a, b| LinkSpec { road_class: "motorway".into(), ..LinkSpec::oneway(a, b, 1, 29.0) };
        let mut nodes = vec![NodeSpec::uncontrolled(1, 0.0, 0.0)];
        let mut x = 0.0;
        for k in 0..6 {
            x += if k % 2 == 0 { 30.0 } else { 12.0 }; // alternating short segments — frequent seams
            nodes.push(NodeSpec::uncontrolled(k + 2, x, 0.0));
        }
        let net = OsmMap { nodes, links: (0..6).map(|k| hw(k + 1, k + 2)).collect() }.build();
        let mut w = NetWorld::new(net, cfg());
        w.install_router(&[LinkId(5)]);
        // A standing queue, nose-to-tail (2.5 m bumper gaps), on the first segment.
        let lane0 = w.network.lanes_of(LinkId(0)).next().unwrap();
        for k in 0..4u32 {
            w.spawn_to_in_lane(k + 1, lane0, 4.0 + k as f64 * 7.0, LinkId(5), 0.3, DriverConfig::car());
        }
        let mut worst_brake = 0.0f64;
        for _ in 0..500 {
            let nb = w.neighbors();
            let intended: Vec<Option<MovementId>> = w
                .fleet
                .rows
                .iter()
                .map(|v| if v.crossing.is_some() { None } else { w.intended_movement(v) })
                .collect();
            for i in 0..w.fleet.rows.len() {
                if w.fleet.rows[i].crossing.is_none() {
                    worst_brake = worst_brake.min(w.gather_context(i, &nb, intended[i]).binding());
                }
            }
            w.step();
        }
        assert_eq!(w.crashed(), 0, "the discharging queue never collides across a seam");
        assert!(w.exited() > 0, "the queue discharges off the end of the corridor");
        assert!(
            worst_brake > -12.0,
            "no car is ever commanded a super-physical brake crossing a seam (worst {worst_brake:.0} m/s^2)",
        );
    }

    #[test]
    fn crossing_position_is_continuous_across_a_boundary() {
        // Continuous position means the world pose flows through a node — no teleport when
        // the car lands and its `position` rebases from `lane.length + interior` to the new
        // lane's frame. A rebase that dropped the interior length would show as a jump. The
        // car crawls (4 m/s) so it dwells in the node interior for several ticks — exercising
        // the whole crossing traversal, not just an enter-and-land in a single step.
        let hw = |a, b| LinkSpec { road_class: "motorway".into(), ..LinkSpec::oneway(a, b, 1, 4.0) };
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::uncontrolled(2, 60.0, 0.0),
                NodeSpec::uncontrolled(3, 120.0, 0.0),
            ],
            links: vec![hw(1, 2), hw(2, 3)],
        }
        .build();
        let mut w = NetWorld::new(net, cfg());
        w.install_router(&[LinkId(1)]);
        let lane0 = w.network.lanes_of(LinkId(0)).next().unwrap();
        w.spawn_to_in_lane(1, lane0, 40.0, LinkId(1), 4.0, DriverConfig::car());
        let mut prev = w.vehicle_world_pose(w.vehicle(1).unwrap());
        let mut worst_step = 0.0f64;
        let mut saw_crossing = false;
        let mut reached_second = false;
        for _ in 0..120 {
            w.step();
            let Some(v) = w.vehicle(1) else { break };
            saw_crossing |= v.is_crossing();
            reached_second |= !v.is_crossing() && self_link(&w, v) == LinkId(1);
            let p = w.vehicle_world_pose(v);
            let step = ((p[0] - prev[0]).powi(2) + (p[1] - prev[1]).powi(2)).sqrt();
            worst_step = worst_step.max(step);
            prev = p;
        }
        assert!(saw_crossing, "the car traverses the node interior");
        assert!(reached_second, "and lands on the downstream link");
        // ~15 m/s at dt=0.04 is ~0.6 m/tick; a landing teleport would dwarf that.
        assert!(worst_step < 1.2, "the world pose never jumps at the seam (worst step {worst_step:.2} m)");
    }

    fn self_link(w: &NetWorld, v: &NetVehicle) -> LinkId {
        w.network.lane(v.lane).link
    }

    #[test]
    fn a_car_is_not_slammed_to_a_stop_by_a_leader_across_a_freeway_seam() {
        // The abrupt-deceleration bug: a car cruising a freeway reaches a segment boundary
        // just as a leader drops into the blind spot a hair across the seam. The cross-
        // boundary gap collapses and unbounded IDM would demand a hundreds-of-g brake,
        // slamming the car dead on open freeway. The crossing gate guarantees it cannot
        // actually land on that leader, so the brake is capped to physical: it eases behind
        // the (moving) leader and never abruptly stops. Guards `cap_leader_brake`.
        let hw = |a, b| LinkSpec { road_class: "motorway".into(), ..LinkSpec::oneway(a, b, 1, 29.0) };
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::uncontrolled(2, 200.0, 0.0),
                NodeSpec::uncontrolled(3, 400.0, 0.0),
            ],
            links: vec![hw(1, 2), hw(2, 3)],
        }
        .build();
        let mut w = NetWorld::new(net, cfg());
        w.install_router(&[LinkId(1)]);
        // Leader a metre onto the second segment, cruising at road speed; follower a few
        // metres back on the first segment at the same speed — the leader has just dropped
        // into the seam's blind spot.
        let seg2 = w.network.lanes_of(LinkId(1)).next().unwrap();
        w.spawn_to_in_lane(1, seg2, 1.0, LinkId(1), 29.0, DriverConfig::car());
        let seg1 = w.network.lanes_of(LinkId(0)).next().unwrap();
        let len1 = w.network.lane(seg1).length;
        w.spawn_to_in_lane(2, seg1, len1 - 3.0, LinkId(1), 29.0, DriverConfig::car());
        let mut worst_brake = 0.0f64;
        let mut min_speed = f64::INFINITY;
        for _ in 0..80 {
            let nb = w.neighbors();
            let intended: Vec<Option<MovementId>> = w
                .fleet
                .rows
                .iter()
                .map(|v| if v.crossing.is_some() { None } else { w.intended_movement(v) })
                .collect();
            for i in 0..w.fleet.rows.len() {
                if w.fleet.rows[i].id == 2 && !w.fleet.rows[i].is_crossing() {
                    worst_brake = worst_brake.min(w.gather_context(i, &nb, intended[i]).binding());
                }
            }
            w.step();
            if let Some(v) = w.vehicle(2) {
                min_speed = min_speed.min(v.speed);
            }
        }
        assert_eq!(w.crashed(), 0, "the follower never rear-ends the leader across the seam");
        assert!(
            worst_brake > -9.5,
            "a leader across the seam never commands a super-physical brake (worst {worst_brake:.0} m/s^2)",
        );
        assert!(
            min_speed > 8.0,
            "following flowing traffic across the seam, the car never abruptly stops (min {min_speed:.1} m/s)",
        );
    }

    fn cfg() -> SimConfig {
        SimConfig::default_config()
    }

    /// Behavioral regression for the "cars stop in the middle of complex
    /// intersections" report (El Camino Real × Millbrae Avenue): drive boundary
    /// demand through the real junction fixture and require that no vehicle sits
    /// stationary inside the junction footprint for longer than a signal phase.
    /// Brief in-box waits are real (a left-turner yielding, spillback backpressure);
    /// camping there was the broken-setback artifact.
    #[cfg(feature = "import")]
    #[test]
    #[ignore] // diagnostic: why all-way-stop corridor cars stall; run with -- --ignored
    fn diag_all_way_stop_gridlock() {
        use super::super::demand::{self, DemandGenerator, DemandSources};
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
        let Ok(text) = std::fs::read_to_string(path) else { return };
        let net = super::super::map::OsmMap::from_json(&text).expect("map json").build();
        let mut world = NetWorld::new(net, cfg());
        let pairs = demand::od_pairs(&world.network, 0, 600, DemandSources::new(true, true));
        let mut gen = DemandGenerator::new(&world, &pairs, 0);
        world.install_router(&gen.destinations());
        let dt = cfg().dt;
        let stop_nodes: Vec<u32> = (0..world.network.nodes.len() as u32)
            .filter(|&i| matches!(world.network.node(NodeId(i)).control, NodeControl::Stop))
            .collect();
        let mut streak: std::collections::HashMap<u32, f64> = Default::default();
        let mut focus: Option<NodeId> = None;
        for tick in 0..1500 {
            gen.step(&mut world, dt);
            world.step();
            // Focus on the first stop node where a car has been parked 60 s.
            if focus.is_none() {
                let mut still: std::collections::HashMap<u32, f64> = Default::default();
                for v in world.fleet.rows.iter() {
                    if v.speed >= 0.3 || v.crossing.is_some() {
                        continue;
                    }
                    let node = world.downstream_node(v.lane);
                    if !stop_nodes.contains(&node.0) {
                        continue;
                    }
                    let t = streak.get(&v.id).copied().unwrap_or(0.0) + dt;
                    still.insert(v.id, t);
                    if t >= 60.0 && focus.is_none() {
                        eprintln!("FOCUS node {} (car id{} parked {t:.0}s) at tick {tick}", node.0, v.id);
                        focus = Some(node);
                    }
                }
                streak = still;
            }
            let Some(focus) = focus else { continue };
            if tick % 25 != 0 {
                continue;
            }
            let nb = world.neighbors();
            let mut lines: Vec<String> = Vec::new();
            for i in 0..world.fleet.rows.len() {
                let v = &world.fleet.rows[i];
                if v.crossing.is_some() {
                    if world.network.movement(v.crossing.unwrap().movement).node == focus {
                        lines.push(format!("  cross id{} mv{} arc {:.1} v{:.1}", v.id, v.crossing.unwrap().movement.0, world.crossing_arc(v), v.speed));
                    }
                    continue;
                }
                if world.downstream_node(v.lane) != focus {
                    continue;
                }
                let lane_len = world.network.lane(v.lane).length;
                if lane_len - v.position > 25.0 {
                    continue;
                }
                let intended = world.intended_movement(v);
                let fifo = intended.is_some_and(|m| world.earlier_stopped_conflict(i, m, focus, &nb));
                let cx = world.gather_context(i, &nb, intended);
                lines.push(format!(
                    "  id{} lane{} to_line {:.1} v{:.1} wait {} armed {} fifo {} yield {:.0} sign {:.0} stopline {:.0} lead {:.0} down {:?}",
                    v.id, v.lane.0, lane_len - v.position, v.speed, v.wait_ticks,
                    v.stopped_at == Some(focus), fifo, cx.yield_line, cx.stop_sign, cx.stop_line, cx.leader_gap,
                    intended.map(|m| world.movement_downstream_blocked(m, &v.driver, &nb)),
                ));
            }
            if !lines.is_empty() {
                eprintln!("t{tick}:");
                for l in lines {
                    eprintln!("{l}");
                }
            }
        }
    }

    #[test]
    #[ignore] // diagnostic: categorize stopped-in-box car-time; run with -- --ignored
    fn diag_where_cars_stop_inside_boxes() {
        use super::super::demand::{self, DemandGenerator};
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
        let Ok(text) = std::fs::read_to_string(path) else { return };
        let net = super::super::map::OsmMap::from_json(&text).expect("map json").build();
        let mut world = NetWorld::new(net, cfg());
        let pairs = demand::od_pairs(&world.network, 0, 600, super::super::demand::DemandSources::new(true, true));
        let mut gen = DemandGenerator::new(&world, &pairs, 0);
        world.install_router(&gen.destinations());
        if std::env::var_os("DIAG_RUSH").is_some() { gen.set_rush_hour(&world.network, true); }
        let dt = cfg().dt;
        // car-seconds stopped, per category
        let (mut far_edge, mut mid_interior, mut internal_lane) = (0.0f64, 0.0f64, 0.0f64);
        let mut worst: std::collections::HashMap<u32, (f64, &'static str)> = Default::default();
        let mut streak: std::collections::HashMap<u32, f64> = Default::default();
        for _ in 0..1500 {
            gen.step(&mut world, dt);
            world.step();
            let mut still: std::collections::HashMap<u32, f64> = Default::default();
            for v in world.vehicles() {
                if v.speed >= 0.3 {
                    continue;
                }
                let cat = if let Some(c) = v.crossing {
                    let arc = world.crossing_arc(v);
                    let len = world.network.interior(c.movement).len;
                    if arc >= len - 0.3 {
                        far_edge += dt;
                        "far_edge"
                    } else {
                        mid_interior += dt;
                        "mid_interior"
                    }
                } else if world.junction_internal_lane(v.lane) {
                    internal_lane += dt;
                    "internal_lane"
                } else {
                    continue;
                };
                let t = streak.get(&v.id).copied().unwrap_or(0.0) + dt;
                still.insert(v.id, t);
                let w = worst.entry(v.id).or_insert((0.0, cat));
                if t > w.0 {
                    *w = (t, cat);
                }
                if (t - 30.0).abs() < dt / 2.0 {
                    if let Some(c) = v.crossing {
                        let mv = world.network.movement(c.movement);
                        let to_lane = mv.to_lane;
                        let tail = world
                            .fleet
                            .rows
                            .iter()
                            .filter(|o| o.lane == to_lane && o.crossing.is_none())
                            .map(|o| (o.position - o.driver.vehicle_length, o.speed))
                            .fold(None::<(f64, f64)>, |m, r| Some(m.map_or(r, |m| if r.0 < m.0 { r } else { m })));
                        let co_crossers = world
                            .fleet
                            .rows
                            .iter()
                            .filter(|o| o.crossing.is_some_and(|oc| world.network.movement(oc.movement).node == mv.node))
                            .count();
                        eprintln!(
                            "CAMP id{} mv{} node{} arc {:.1}/{:.1} to_lane{} tail {:?} co-crossers {} internal_to {}",
                            v.id, c.movement.0, mv.node.0,
                            world.crossing_arc(v), world.network.interior(c.movement).len,
                            to_lane.0, tail, co_crossers,
                            world.junction_internal_lane(to_lane),
                        );
                    } else {
                        let nb = world.neighbors();
                        let gates = world.intended_movement(v).map(|mid| {
                            let node = world.downstream_node(v.lane);
                            let i = world.fleet.rows.iter().position(|o| o.id == v.id).unwrap();
                            let front: IntMap<f64> = Default::default();
                            (
                                mid.0,
                                world.movement_state(mid),
                                world.box_conflict(mid, node, &nb),
                                world.junction_exit_blocked(v, mid, node, &nb),
                                world.movement_downstream_blocked(mid, &v.driver, &nb),
                                world.box_entry_blocked(i, Some(mid), &nb),
                                front.len(),
                            )
                        });
                        eprintln!(
                            "CAMP id{} internal lane{} pos {:.1} len {:.1} gates {:?}",
                            v.id, v.lane.0, v.position, world.network.lane(v.lane).length, gates
                        );
                    }
                }
            }
            streak = still;
        }
        let mut tops: Vec<(u32, f64, &str)> = worst.iter().map(|(&id, &(t, c))| (id, t, c)).collect();
        tops.sort_by(|a, b| b.1.total_cmp(&a.1));
        eprintln!("in-box stopped car-seconds: far_edge {far_edge:.0}, mid_interior {mid_interior:.0}, internal_lane {internal_lane:.0}");
        eprintln!("cars with any in-box stop: {}", worst.len());
        for (id, t, c) in tops.iter().take(12) {
            eprintln!("  id{id}: {t:.1}s stopped ({c})");
        }
    }

    #[test]
    fn traffic_does_not_camp_inside_the_complex_junction() {
        use super::super::demand::{self, DemandGenerator};
        let net = super::super::map::millbrae_junction(0);
        let mut world = NetWorld::new(net, cfg());
        let pairs = demand::boundary_od_pairs(&world.network, 9, 8);
        assert!(!pairs.is_empty(), "the fixture's clipped arms are gateways");
        let mut gen = DemandGenerator::new(&world, &pairs, 9);
        world.install_router(&gen.destinations());
        gen.set_rate_scale(0.5);

        let dt = cfg().dt;
        let mut stopped_for: std::collections::HashMap<u32, f64> = Default::default();
        let mut worst = 0.0f64;
        for _ in 0..3000 {
            gen.step(&mut world, dt);
            world.step();
            // "Inside the box" = physically on junction pavement: mid-crossing in a
            // node interior, or on a cluster-internal link. A car at a boundary stop
            // line is at the edge by construction — waiting out a red there is fine.
            let mut still: std::collections::HashMap<u32, f64> = Default::default();
            for v in world.vehicles() {
                let in_box = v.is_crossing() || world.junction_internal_lane(v.lane);
                if v.speed < 0.3 && in_box {
                    let t = stopped_for.get(&v.id).copied().unwrap_or(0.0) + dt;
                    worst = worst.max(t);
                    still.insert(v.id, t);
                }
            }
            stopped_for = still;
        }
        assert!(world.exited() > 20, "traffic actually flows through the junction: {}", world.exited());
        // A car caught mid-cluster by a phase change legitimately waits out the
        // cross street's green from an internal stub (the box gates rightly hold
        // it against the conflicting stream), so the bound is one signal cycle —
        // not the old 30 s, which the tightened conflict gates now exceed. The
        // routine-camping defect this test was born from is separately guarded by
        // `stopped_traffic_queues_on_approaches_not_inside_junctions`.
        assert!(
            worst < 70.0,
            "a vehicle camped stationary inside the junction box for {worst:.0} s — \
             longer than any phase-trap can explain",
        );
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

    /// The whole `Threads` step must reproduce the serial one bit-for-bit. The passes it
    /// parallelizes — the accel gather/evaluate (fused on the CPU path), the in-lane integrate,
    /// and the MOBIL lane-change *decision* — are order-preserving pure maps over committed
    /// state; the front-dependent boundary landing stays serial in index order. So threading
    /// moves only *where* the work runs, never the outcome. This is the guard for the
    /// gather/evaluate fusion and the parallel-integrate / serial-resolve advance split. It has
    /// teeth only under `--features parallel` (otherwise `Threads` is itself serial).
    #[test]
    fn threads_backend_matches_serial_bit_for_bit() {
        use super::super::demand::{self, DemandGenerator, DemandSources};
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/peninsula.json");
        let Ok(text) = std::fs::read_to_string(path) else { return };
        let net = super::super::map::OsmMap::from_json(&text).expect("map json").build();
        let pairs = demand::od_pairs(&net, 0, 600, DemandSources::new(true, true));

        let run = |backend: AccelBackend, cache_sort: bool| {
            let mut world = NetWorld::new(net.clone(), cfg());
            world.set_accel_backend(backend);
            world.set_par_threshold(0); // engage rayon from the first car (under `parallel`)
            world.set_cache_sort(cache_sort);
            let mut gen = DemandGenerator::new(&world, &pairs, 0);
            world.install_router(&gen.destinations());
            for _ in 0..600 {
                gen.step(&mut world, cfg().dt);
                world.step();
            }
            // Exact f64 equality via the raw bits — a threading divergence is never a rounding
            // artifact here (the arithmetic is identical), so any drift must show.
            let fleet: Vec<(u32, u64, u64, bool, bool, usize)> = world
                .fleet
                .rows
                .iter()
                .map(|v| (v.lane.0, v.position.to_bits(), v.speed.to_bits(), v.crossing.is_some(), v.lane_change.is_some(), v.route_idx))
                .collect();
            (fleet, world.exited, world.leaked, world.crashed)
        };
        let serial = run(AccelBackend::Serial, true);
        assert!(!serial.0.is_empty(), "the scenario must actually build a fleet to compare");
        assert_eq!(serial, run(AccelBackend::Threads, true), "Threads must match Serial bit-for-bit");
        // The cache-friendly sort is a pure performance option: same total order, same result.
        assert_eq!(serial, run(AccelBackend::Serial, false), "cache-sort off must match cache-sort on");
        assert_eq!(serial, run(AccelBackend::Threads, false), "cache-sort off under threads too");
    }

    #[test]
    fn active_backend_falls_back_when_a_backend_is_unavailable() {
        let mut w = NetWorld::new(millbrae_sample(), cfg());
        assert_eq!(w.active_backend(), AccelBackend::Serial, "default is serial");
        // Requesting Gpu without a solver installed (enable_gpu_accel) runs serially.
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

    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    #[test]
    fn gpu_accel_backend_drives_a_safe_sim() {
        // The Gpu backend must drive a full sim end to end — rolling cars *and* in-node
        // crossers through the evaluate pass — as safely as Serial and with comparable
        // throughput. The fold is f32, so exact parity isn't expected; only collision-
        // free flow that tracks the serial reference within a band.
        let run = |gpu: bool| -> Option<u32> {
            let mut world = NetWorld::new(symmetric_diamond(), cfg());
            if gpu && !world.enable_gpu_accel() {
                return None; // no adapter in this environment
            }
            let want = if gpu { AccelBackend::Gpu } else { AccelBackend::Serial };
            world.set_accel_backend(want);
            assert_eq!(world.active_backend(), want, "requested backend is the active one");
            world.install_router(&[LinkId(5)]);
            let mut next = 0u32;
            for t in 0..3000 {
                if t % 3 == 0 && world.spawn_to(next, LinkId(0), LinkId(5), 12.0, DriverConfig::car()) {
                    next += 1;
                }
                world.step();
            }
            assert_eq!(world.crashed(), 0, "gpu={gpu}: rerouting stays collision-free");
            assert_eq!(world.leaked(), 0, "gpu={gpu}: no car vanishes at a node");
            Some(world.exited())
        };
        let serial = run(false).unwrap();
        let Some(gpu) = run(true) else {
            eprintln!("no GPU adapter; skipping GPU accel step test");
            return;
        };
        assert!(gpu > 20, "the GPU-backed sim keeps traffic flowing, got {gpu}");
        let ratio = gpu as f64 / serial as f64;
        assert!((0.6..1.7).contains(&ratio), "GPU throughput {gpu} tracks serial {serial} (ratio {ratio:.2})");
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
        assert_eq!(effective_critical_gap(4.0, 1000.0), 2.8, "but never below the safety floor");
        // HCM bases: minor movements demand more than the major left; trucks add ~1 s.
        let car = DriverConfig::car();
        assert_eq!(hcm_critical_gap(TurnType::Left, false, &car), 4.1);
        assert_eq!(hcm_critical_gap(TurnType::Right, true, &car), 6.2);
        assert_eq!(hcm_critical_gap(TurnType::Through, true, &car), 6.5);
        assert_eq!(hcm_critical_gap(TurnType::Left, true, &car), 7.1);
        let truck = VehicleClass::Truck.driver();
        assert_eq!(hcm_critical_gap(TurnType::Left, true, &truck), 8.1);
        // The sampled driver heterogeneity scales the book value.
        let hasty = DriverConfig { critical_gap: 3.2, ..car };
        assert!((hcm_critical_gap(TurnType::Through, true, &hasty) - 6.5 * 0.8).abs() < 1e-9);
    }

    #[test]
    fn dilemma_zone_decision() {
        // Can stop comfortably before the line → stop; too fast/close → proceed.
        assert!(can_stop_before(10.0, 1.5, 100.0)); // plenty of room
        assert!(!can_stop_before(20.0, 1.5, 5.0)); // no chance — commit through
    }

    #[test]
    fn a_bounded_minority_of_aggressive_drivers_run_a_stoppable_yellow() {
        let nominal = yellow_run_prob(&DriverConfig::car());
        let fast = yellow_run_prob(&DriverConfig { desired_speed: 34.5, ..DriverConfig::car() });
        assert!(fast > nominal, "aggressive drivers run yellows more often: {fast} vs {nominal}");
        assert!((0.0..=0.35).contains(&nominal) && (0.0..=0.35).contains(&fast), "the share stays a bounded minority");

        let n = 20_000u32;
        let runners = (0..n)
            .filter(|&id| rng::uniform01(7, id, YELLOW_RUN_SALT, Stream::GapAcceptance) < fast)
            .count();
        let frac = runners as f64 / n as f64;
        assert!((frac - fast).abs() < 0.02, "the runner fraction tracks the probability: {frac} vs {fast}");
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

    #[test]
    fn a_stopped_car_takes_a_startup_reaction_to_launch_on_green() {
        let launch = |reaction: f64| -> u32 {
            let net = signal_at(18.0);
            let lane = approach_lane(&net);
            let length = net.lane(lane).length;
            let mut world = NetWorld::new(net, cfg());
            let d = DriverConfig { reaction_time: reaction, accel_noise: 0.0, ..DriverConfig::car() };
            world.spawn(1, lane, length - 2.0, 0.0, d);
            for t in 0..600 {
                world.step();
                match world.vehicle(1) {
                    Some(v) if v.speed > 0.5 => return t,
                    _ => {}
                }
            }
            u32::MAX
        };
        let slow = launch(0.6);
        let fast = launch(0.05);
        assert!(slow < 600 && fast < 600, "both launch within the window: slow={slow} fast={fast}");
        assert!(slow > fast, "a longer reaction delays the launch on green: slow={slow} fast={fast}");
        assert!(slow - fast >= 2, "the start-up delay reflects the reaction-time difference: slow={slow} fast={fast}");
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
    fn a_vehicle_serves_a_clear_stop_sign_with_at_most_a_rolling_stop() {
        // Observed stop-sign compliance is famously partial: with no conflicting
        // traffic, most drivers roll through at a walking pace and only the
        // cautious end plants the wheel. The sign still binds — every driver
        // brakes to at most the ~3 mph roll — and then proceeds.
        let stop_line_net = || {
            OsmMap {
                nodes: vec![
                    NodeSpec::uncontrolled(1, 0.0, 0.0),
                    NodeSpec { osm_id: 2, x: 150.0, y: 0.0, control: MapControl::Stop, rail_crossing: false },
                    NodeSpec::uncontrolled(3, 300.0, 0.0),
                ],
                links: vec![LinkSpec::oneway(1, 2, 1, 15.0), LinkSpec::oneway(2, 3, 1, 15.0)],
            }
            .build()
        };
        // (driver, the roll speed the sign is served at)
        let cautious = DriverConfig { desired_speed: 25.0, ..DriverConfig::car() }; // aggression 0 → full stop
        let default = DriverConfig::car(); // mid aggression → ~0.8 m/s roll
        for (driver, max_roll, label) in [(cautious, 0.31, "cautious"), (default, 1.31, "default")] {
            let net = stop_line_net();
            let approach = net.lanes_of(LinkId(0)).next().unwrap();
            let mut world = NetWorld::new(net, cfg());
            world.spawn(1, approach, 0.0, 14.0, driver);
            let mut min_speed_on_approach = f64::MAX;
            for _ in 0..250 {
                world.step();
                if let Some(v) = world.vehicle(1) {
                    if v.lane == approach {
                        min_speed_on_approach = min_speed_on_approach.min(v.speed);
                    }
                }
            }
            assert!(
                min_speed_on_approach < max_roll,
                "{label}: serves the sign at no faster than a rolling stop, min {min_speed_on_approach}"
            );
            assert_eq!(world.exited(), 1, "{label}: and then continues through");
        }
    }

    #[test]
    fn all_way_stop_serves_the_first_to_stop_first() {
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -200.0, 0.0),
                NodeSpec { osm_id: 2, x: 0.0, y: 0.0, control: MapControl::Stop, rail_crossing: false },
                NodeSpec::uncontrolled(3, 200.0, 0.0),
                NodeSpec::uncontrolled(4, 0.0, -200.0),
                NodeSpec::uncontrolled(5, 0.0, 200.0),
            ],
            links: vec![
                LinkSpec::oneway(1, 2, 1, 12.0), // A approach (west→east through)
                LinkSpec::oneway(2, 3, 1, 12.0),
                LinkSpec::oneway(4, 2, 1, 12.0), // B approach (south→north through)
                LinkSpec::oneway(2, 5, 1, 12.0),
            ],
        }
        .build();
        let mut world = NetWorld::new(net, cfg());
        world.install_router(&[LinkId(1), LinkId(3)]);
        let d = DriverConfig { accel_noise: 0.0, ..DriverConfig::car() };
        let a_lane = world.network.lanes_of(LinkId(0)).next().unwrap();
        let a_len = world.network.lane(a_lane).length;
        world.spawn_to_in_lane(1, a_lane, a_len - 8.0, LinkId(1), 6.0, d.clone()); // A stops first
        let b_lane = world.network.lanes_of(LinkId(2)).next().unwrap();
        world.spawn_to_in_lane(2, b_lane, 5.0, LinkId(3), 6.0, d); // B arrives much later

        let (mut a_cross, mut b_cross) = (None, None);
        for t in 0..500 {
            world.step();
            if a_cross.is_none() && world.vehicle(1).is_some_and(|v| v.is_crossing()) {
                a_cross = Some(t);
            }
            if b_cross.is_none() && world.vehicle(2).is_some_and(|v| v.is_crossing()) {
                b_cross = Some(t);
            }
        }
        let (a, b) = (a_cross.expect("A crosses"), b_cross.expect("B crosses"));
        assert!(a < b, "the first vehicle to stop is served first: A@{a} B@{b}");
        assert_eq!(world.crashed(), 0, "arrival-order service stays collision-free");
    }

    #[test]
    fn all_way_stop_does_not_deadlock_when_the_first_car_is_blocked() {
        // A stops first, but a stalled car sits at the entrance of A's exit lane, so A can't
        // move. B, whose own exit is clear, must eventually take its turn rather than deferring
        // to the stuck A forever — the real-world "if your lane is clear, you go" behaviour.
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -200.0, 0.0),
                NodeSpec { osm_id: 2, x: 0.0, y: 0.0, control: MapControl::Stop, rail_crossing: false },
                NodeSpec::uncontrolled(3, 200.0, 0.0),
                NodeSpec::uncontrolled(4, 0.0, -200.0),
                NodeSpec::uncontrolled(5, 0.0, 200.0),
            ],
            links: vec![
                LinkSpec::oneway(1, 2, 1, 12.0), // A approach (west→east through), exits on link (2→3)
                LinkSpec::oneway(2, 3, 1, 12.0),
                LinkSpec::oneway(4, 2, 1, 12.0), // B approach (south→north through), exits on the clear (2→5)
                LinkSpec::oneway(2, 5, 1, 12.0),
            ],
        }
        .build();
        let mut world = NetWorld::new(net, cfg());
        world.install_router(&[LinkId(1), LinkId(3)]);
        let d = DriverConfig { accel_noise: 0.0, ..DriverConfig::car() };
        let a_lane = world.network.lanes_of(LinkId(0)).next().unwrap();
        let a_len = world.network.lane(a_lane).length;
        world.spawn_to_in_lane(1, a_lane, a_len - 8.0, LinkId(1), 6.0, d.clone()); // A stops first
        // A stalled crawler parked at the entrance of A's exit link (2→3) blocks A's landing.
        let block_lane = world.network.lanes_of(LinkId(1)).next().unwrap();
        let stalled = DriverConfig { accel_noise: 0.0, desired_speed: 0.15, ..DriverConfig::car() };
        world.spawn_to_in_lane(3, block_lane, 2.0, LinkId(1), 0.0, stalled);
        let b_lane = world.network.lanes_of(LinkId(2)).next().unwrap();
        world.spawn_to_in_lane(2, b_lane, 5.0, LinkId(3), 6.0, d); // B arrives later, exit clear

        let mut b_cross = None;
        for t in 0..500 {
            world.step();
            if b_cross.is_none() && world.vehicle(2).is_some_and(|v| v.is_crossing()) {
                b_cross = Some(t);
            }
        }
        // B is served promptly because it stops deferring to the blocked A. Without the fix B
        // waits until A finally clears (only once the obstruction crawls off, ~t400), so a well-
        // separated early bound is what gives this test teeth. The run is deterministic
        // (`accel_noise: 0.0`), so the crossing tick is stable.
        assert!(b_cross.is_some_and(|t| t < 250), "B takes its turn early despite the earlier-stopped A being blocked, got {b_cross:?}");
        assert_eq!(world.crashed(), 0, "breaking the deadlock stays collision-free");
        assert_eq!(world.leaked(), 0, "and no vehicle vanishes");
    }

    fn signalized_four_way() -> OsmMap {
        let plan = SignalPlan { green_secs: 15.0, yellow_secs: 3.0, offset: 0.0 };
        OsmMap {
            nodes: vec![
                NodeSpec::signalized(0, 0.0, 0.0, plan),
                NodeSpec::uncontrolled(1, -120.0, 0.0),
                NodeSpec::uncontrolled(2, 120.0, 0.0),
                NodeSpec::uncontrolled(3, 0.0, -120.0),
                NodeSpec::uncontrolled(4, 0.0, 120.0),
            ],
            links: vec![
                LinkSpec::oneway(1, 0, 1, 15.0),
                LinkSpec::oneway(0, 1, 1, 15.0),
                LinkSpec::oneway(2, 0, 1, 15.0),
                LinkSpec::oneway(0, 2, 1, 15.0),
                LinkSpec::oneway(3, 0, 1, 15.0),
                LinkSpec::oneway(0, 3, 1, 15.0),
                LinkSpec::oneway(4, 0, 1, 15.0),
                LinkSpec::oneway(0, 4, 1, 15.0),
            ],
        }
    }

    #[test]
    fn right_turn_on_red_clears_before_a_through_that_waits_for_green() {
        let node = NodeId(0);
        let find = |net: &Network, turn: TurnType, from: LinkId| -> Option<MovementId> {
            (0..net.movements.len() as u32).map(MovementId).find(|&m| {
                net.movement(m).node == node
                    && net.lane(net.movement(m).from_lane).link == from
                    && net.movement_turn(m) == turn
            })
        };

        let net = signalized_four_way().build();
        let probe = NetWorld::new(signalized_four_way().build(), cfg());
        let states = probe.signal_states();
        let red_from = (0..net.links.len() as u32)
            .map(LinkId)
            .filter(|&l| net.link(l).to == node)
            .find(|&l| {
                let red_through = find(&net, TurnType::Through, l)
                    .and_then(|m| net.movement(m).signal_group)
                    .is_some_and(|g| states[g.idx()] == SignalState::Red);
                red_through && find(&net, TurnType::Right, l).is_some()
            })
            .expect("an approach whose through is red at t=0");

        let run = |turn: TurnType| -> (u32, u32) {
            let mut w = NetWorld::new(signalized_four_way().build(), cfg());
            let mv = find(&w.network, turn, red_from).expect("approach has this movement");
            let dest = w.network.lane(w.network.movement(mv).to_lane).link;
            w.install_router(&[dest]);
            let lane = w.network.lanes_of(red_from).next().unwrap();
            let len = w.network.lane(lane).length;
            let driver = DriverConfig { accel_noise: 0.0, ..DriverConfig::car() };
            w.spawn_to_in_lane(1, lane, len - 8.0, dest, 6.0, driver);
            let mut exit = u32::MAX;
            for t in 0..400 {
                w.step();
                if exit == u32::MAX && w.vehicle(1).is_none() {
                    exit = t;
                }
            }
            (exit, w.crashed())
        };

        let (t_right, crashed_r) = run(TurnType::Right);
        let (t_through, crashed_t) = run(TurnType::Through);
        assert_eq!((crashed_r, crashed_t), (0, 0), "no collisions in either run");
        assert!(t_right < 400 && t_through < 400, "both eventually clear: right={t_right} through={t_through}");
        assert!(
            t_right < t_through,
            "right-turn-on-red ({t_right}) clears before the through that must wait for green ({t_through})",
        );
    }

    #[test]
    fn permissive_left_yields_to_oncoming_then_clears_without_colliding() {
        let mut w = NetWorld::new(signalized_four_way().build(), cfg());
        let d = DriverConfig { accel_noise: 0.0, ..DriverConfig::car() };
        assert!(w.spawn_routed(1, vec![LinkId(4), LinkId(1)], 8.0, d.clone()));
        let mut next = 100u32;
        for t in 0..800 {
            if t % 30 == 0 {
                w.spawn_routed(next, vec![LinkId(6), LinkId(5)], 8.0, d.clone());
                next += 1;
            }
            w.step();
        }
        assert_eq!(w.crashed(), 0, "a permissive left never collides with the oncoming through it yields to");
        assert!(w.vehicle(1).is_none(), "the left-turner still clears the intersection (no deadlock)");
    }

    #[test]
    fn permissive_left_advances_into_the_box_and_waits_at_its_conflict_point() {
        // Under a continuous sub-critical oncoming stream, the green permissive
        // left no longer camps at the stop line: it advances into the box, stands
        // short of the oncoming path, and completes when the pressure lifts —
        // without a collision and without gating the opposing flow.
        let mut w = NetWorld::new(signalized_four_way().build(), cfg());
        let d = DriverConfig { accel_noise: 0.0, ..DriverConfig::car() };
        assert!(w.spawn_routed(1, vec![LinkId(4), LinkId(1)], 8.0, d.clone()));
        let mut next = 100u32;
        let mut waited_in_box = 0u32;
        let mut oncoming_through = 0u32;
        for t in 0..900 {
            // A continuous sub-critical stream while the (actuated, unchallenged)
            // green holds — then it ends, and the waiter's gap arrives.
            if t % 12 == 0 && t < 500 {
                w.spawn_routed(next, vec![LinkId(6), LinkId(5)], 8.0, d.clone());
                next += 1;
            }
            w.step();
            if let Some(v) = w.vehicle(1) {
                if v.is_crossing() && v.speed < 0.5 {
                    waited_in_box += 1;
                }
                oncoming_through = w.exited();
            }
        }
        assert!(waited_in_box >= 10, "the left stood inside the box awaiting its window ({waited_in_box} ticks)");
        assert!(w.vehicle(1).is_none(), "the waiter completed its turn (no box deadlock)");
        assert!(oncoming_through >= 5, "the opposing flow kept moving past the waiter ({oncoming_through} cleared)");
        assert_eq!(w.crashed(), 0, "waiting mid-box stays collision-free");
    }

    #[test]
    fn minor_road_yields_to_the_major_road_then_goes() {
        // Minor goes *straight across* (south→north) the major (west→east), so the
        // crossing conflict rule makes it defer to the higher-priority major. The
        // minor's approach is short so it reaches the line first yet still yields.
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -120.0, 0.0),
                NodeSpec { osm_id: 2, x: 0.0, y: 0.0, control: MapControl::Yield, rail_crossing: false },
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
        // Noise-free drivers: the test asserts arrival-order logic, and sustained
        // throttle wander would shift the tuned arrival timing it depends on.
        let d = DriverConfig { accel_noise: 0.0, ..DriverConfig::car() };
        world.spawn_routed(10, vec![LinkId(0), LinkId(1)], 20.0, d.clone()); // major through
        world.spawn_routed(20, vec![LinkId(2), LinkId(3)], 9.0, d); // minor straight across

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
    fn minor_road_right_turn_yields_to_the_major_through() {
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -120.0, 0.0),
                NodeSpec { osm_id: 2, x: 0.0, y: 0.0, control: MapControl::Yield, rail_crossing: false },
                NodeSpec::uncontrolled(3, 200.0, 0.0),
                NodeSpec::uncontrolled(4, 0.0, -40.0),
            ],
            links: vec![
                LinkSpec::oneway(1, 2, 1, 25.0), // major approach (long)
                LinkSpec::oneway(2, 3, 1, 25.0), // major exit (east)
                LinkSpec::oneway(4, 2, 1, 10.0), // minor approach (south, short)
            ],
        }
        .build();
        let mut world = NetWorld::new(net, cfg());
        let d = DriverConfig { accel_noise: 0.0, ..DriverConfig::car() };
        world.spawn_routed(10, vec![LinkId(0), LinkId(1)], 20.0, d.clone()); // major through W→E
        world.spawn_routed(20, vec![LinkId(2), LinkId(1)], 9.0, d); // minor right turn S→E

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
        let (mj, mn) = (major_enter.expect("major crosses"), minor_enter.expect("minor right-turner crosses"));
        assert!(mn > mj, "the minor right-turner must yield to the major through: minor@{mn} major@{mj}");
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

    #[test]
    fn inflow_spreads_across_a_multilane_entry() {
        // A 3-lane gateway: three back-to-back spawns land on three distinct lanes,
        // not stacked on the median lane — so a wide road can physically accept its
        // full per-lane rush-hour inflow. A fourth (all entrances now occupied) is
        // refused, which is the correct backpressure.
        let net = OsmMap {
            nodes: vec![NodeSpec::uncontrolled(1, 0.0, 0.0), NodeSpec::uncontrolled(2, 300.0, 0.0)],
            links: vec![LinkSpec::oneway(1, 2, 3, 29.0)],
        }
        .build();
        let mut w = NetWorld::new(net, cfg());
        for id in 0..3 {
            assert!(w.spawn_to(id, LinkId(0), LinkId(0), 29.0, DriverConfig::car()), "spawn {id} accepted");
        }
        let lanes: std::collections::BTreeSet<u32> = w.vehicles().iter().map(|v| v.lane.0).collect();
        assert_eq!(lanes.len(), 3, "the three entrants occupy three distinct lanes, got {lanes:?}");
        assert!(!w.spawn_to(3, LinkId(0), LinkId(0), 29.0, DriverConfig::car()), "every lane occupied → spawn refused");
    }

    fn packed_queue(cong: CongestionConfig) -> (NetWorld, u32) {
        // A single link packed near jam density, feeding a gateway — the queued state
        // the congestion LOD is meant to engage on.
        let net = straight_link(400.0);
        let mut w = NetWorld::new(net, cfg());
        w.set_congestion(cong);
        let lane = w.network.lanes_of(LinkId(0)).next().unwrap();
        let n = 45u32;
        for i in 0..n {
            w.spawn(i, lane, i as f64 * 8.0, 0.0, DriverConfig::car());
        }
        (w, n)
    }

    #[test]
    fn congestion_lod_engages_but_never_crashes_or_loses_cars() {
        let cong = CongestionConfig { enabled: true, engage_occ: 0.3, release_occ: 0.1, dwell_ticks: 3 };
        let (mut w, n) = packed_queue(cong);
        let mut engaged = false;
        for _ in 0..3000 {
            w.step();
            engaged |= w.congestion_active_links() > 0;
            // Every car is either still on the road or cleanly exited — none lost, and
            // the cheap follower must never manufacture a collision.
            assert_eq!(w.crashed(), 0, "the queue model must not cause crashes");
            assert_eq!(w.vehicles().len() as u32 + w.exited(), n, "no vehicles created or lost");
        }
        assert!(engaged, "the packed link should engage the queue model");
        assert_eq!(w.exited(), n, "the whole queue discharges through the gateway");
    }

    #[test]
    fn congestion_lod_matches_full_detail_throughput() {
        // The cheap queue model should discharge a jam at essentially the same rate as
        // the full per-car model — behaviour stays equivalent, only cheaper.
        let off = CongestionConfig::disabled();
        let on = CongestionConfig { enabled: true, engage_occ: 0.3, release_occ: 0.1, dwell_ticks: 3 };
        let (mut full, _) = packed_queue(off);
        let (mut lod, _) = packed_queue(on);
        for _ in 0..1500 {
            full.step();
            lod.step();
        }
        assert!(!lod.congestion_config().enabled || lod.exited() > 0, "the queue should be discharging");
        let diff = (full.exited() as i64 - lod.exited() as i64).abs();
        assert!(diff <= 3, "throughput differs by {diff} (full {} vs lod {})", full.exited(), lod.exited());
    }

    #[test]
    fn congestion_disabled_by_default_keeps_full_detail() {
        let net = straight_link(300.0);
        let mut w = NetWorld::new(net, cfg());
        let lane = w.network.lanes_of(LinkId(0)).next().unwrap();
        for i in 0..10u32 {
            w.spawn(i, lane, i as f64 * 12.0, 8.0, DriverConfig::car());
        }
        for _ in 0..400 {
            w.step();
            assert_eq!(w.congestion_active_links(), 0, "no link uses the queue model while disabled");
        }
        assert_eq!(w.exited(), 10, "all vehicles drive out under full per-car");
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
    fn transit_lines_run_scheduled_buses_and_the_mix_has_no_random_ones() {
        use super::super::demand::{DemandGenerator, TransitLine};
        // A street with a stop: the named line spawns buses on its headway;
        // background demand contributes none.
        let mut net = OsmMap {
            nodes: vec![NodeSpec::uncontrolled(1, 0.0, 0.0), NodeSpec::uncontrolled(2, 500.0, 0.0)],
            links: vec![LinkSpec::oneway(1, 2, 2, 13.0)],
        }
        .build();
        net.attach_bus_stops(&[[240.0, 2.0]]);
        let mut w = NetWorld::new(net, cfg());
        let mut gen = DemandGenerator::new(&w, &[], 5);
        gen.set_transit_lines(vec![TransitLine::new("ECR".into(), vec![LinkId(0)])]);
        // The fallback day clock starts at 05:30 (headway 1800 s → 900 s at 06:00);
        // run 40 sim-minutes and count buses.
        let mut buses_seen = std::collections::HashSet::new();
        for _ in 0..12_000 {
            gen.step(&mut w, cfg().dt);
            w.step();
            for v in w.vehicles() {
                if v.driver.vehicle_length >= 11.0 {
                    buses_seen.insert(v.id);
                }
            }
        }
        assert!(
            (1..=4).contains(&buses_seen.len()),
            "the line runs on its headway (~2 departures in 40 min around 06:00): {}",
            buses_seen.len()
        );
        assert!(w.exited() >= 1, "buses complete the route");
    }

    #[test]
    fn buses_dwell_at_stops_and_cars_pass_through() {
        // One street with a mid-block bus stop: a bus brakes to it, serves ~25 s,
        // then continues; a car sails past without stopping.
        let mut net = OsmMap {
            nodes: vec![NodeSpec::uncontrolled(1, 0.0, 0.0), NodeSpec::uncontrolled(2, 400.0, 0.0)],
            links: vec![LinkSpec::oneway(1, 2, 2, 13.0)],
        }
        .build();
        net.attach_bus_stops(&[[180.0, 2.0]]);
        assert_eq!(net.bus_stops.len(), 1, "the stop lands on the street");
        let mut w = NetWorld::new(net, cfg());
        let quiet = |d: DriverConfig| DriverConfig { accel_noise: 0.0, ..d };
        assert!(w.spawn_routed(1, vec![LinkId(0)], 10.0, quiet(VehicleClass::Bus.driver())));
        assert!(w.spawn_routed(2, vec![LinkId(0)], 10.0, quiet(DriverConfig::car())));
        let (mut bus_stopped_ticks, mut car_stopped_ticks) = (0u32, 0u32);
        for _ in 0..1200 {
            w.step();
            for v in w.vehicles() {
                if v.speed < 0.3 {
                    if v.driver.vehicle_length >= 11.0 {
                        bus_stopped_ticks += 1;
                    } else {
                        car_stopped_ticks += 1;
                    }
                }
            }
        }
        let dwell_secs = bus_stopped_ticks as f64 * 0.2;
        assert!((20.0..40.0).contains(&dwell_secs), "the bus serves the stop ~25 s: {dwell_secs:.0}");
        assert!(car_stopped_ticks < 10, "cars don't stop for the bus stop: {car_stopped_ticks}");
        assert_eq!(w.exited(), 2, "both vehicles complete the street");
    }

    #[test]
    fn rail_closure_preempts_the_adjacent_signal_to_flush_the_tracks() {
        use super::super::map::SignalPlan;
        // Crossing R 80 m before a signalized cross street: when the gates come
        // down, the signal must switch to (and hold) the phase that greens the
        // from-crossing approach, flushing traffic off the tracks.
        let plan = SignalPlan { green_secs: 20.0, yellow_secs: 3.0, offset: 0.0 };
        let mut nodes = vec![
            NodeSpec::uncontrolled(1, -300.0, 0.0),
            NodeSpec::uncontrolled(2, -80.0, 0.0),
            NodeSpec::signalized(3, 0.0, 0.0, plan),
            NodeSpec::uncontrolled(4, 300.0, 0.0),
            NodeSpec::uncontrolled(5, 0.0, -250.0),
            NodeSpec::uncontrolled(6, 0.0, 250.0),
        ];
        nodes[1].rail_crossing = true;
        let mut links = vec![LinkSpec::oneway(1, 2, 1, 13.0), LinkSpec::oneway(2, 3, 1, 13.0), LinkSpec::oneway(3, 4, 1, 13.0)];
        links.extend(LinkSpec::twoway(5, 3, 1, 12.0));
        links.extend(LinkSpec::twoway(3, 6, 1, 12.0));
        let net = OsmMap { nodes, links }.build();
        let mut w = NetWorld::new(net, cfg());
        assert!(!w.rail_preempts.is_empty(), "the crossing maps to a preemptable phase");
        // The from-crossing approach's signalized movement.
        let lane = w.network.link(LinkId(1)).lane_start;
        let mid = MovementId(w.network.lane(lane).movement_start.0);
        // Load cross-street demand so the actuated controller would otherwise
        // cycle away, then close the gates mid-cross-phase.
        let mut id = 100u32;
        for t in 0..1200u32 {
            // Midday timetable: closures start each 900 s; park the clock inside one.
            w.set_day_secs(12.0 * 3600.0 + (t as f64 * 0.2) % 40.0);
            if t % 25 == 0 {
                if w.spawn_routed(id, vec![LinkId(3), LinkId(4)], 8.0, DriverConfig::car()) {
                    id += 1;
                }
            }
            w.step();
        }
        assert_eq!(
            w.movement_state(mid),
            SignalState::Green,
            "during a closure the track-clearing phase holds green"
        );
    }

    #[test]
    fn rail_crossing_closes_on_the_timetable_and_queues_traffic() {
        // A street across a level crossing: during a closure nobody enters the
        // node (the queue holds at the line); between closures traffic flows.
        let mut nodes = vec![
            NodeSpec::uncontrolled(1, -200.0, 0.0),
            NodeSpec::uncontrolled(2, 0.0, 0.0),
            NodeSpec::uncontrolled(3, 200.0, 0.0),
        ];
        nodes[1].rail_crossing = true;
        let net = OsmMap {
            nodes,
            links: vec![LinkSpec::oneway(1, 2, 1, 13.0), LinkSpec::oneway(2, 3, 1, 13.0)],
        }
        .build();
        assert!(net.node(NodeId(1)).rail_crossing, "the crossing survives the build");
        let mut w = NetWorld::new(net, cfg());
        let mut id = 0u32;
        let (mut closed_entries, mut open_flow) = (0u32, 0u32);
        // Midday cadence (4/h → 900 s period, 45 s closed); day time advances 1:1.
        for t in 0..6000u32 {
            let day = 12.0 * 3600.0 + t as f64 * 0.2;
            w.set_day_secs(day);
            if w.spawn_routed(id, vec![LinkId(0), LinkId(1)], 10.0, DriverConfig::car()) {
                id += 1;
            }
            let before: Vec<u32> =
                w.vehicles().iter().filter(|v| v.is_crossing()).map(|v| v.id).collect();
            w.step();
            let closed = day.rem_euclid(900.0) < 45.0;
            for v in w.vehicles().iter().filter(|v| v.is_crossing()) {
                if !before.contains(&v.id) {
                    if closed {
                        closed_entries += 1;
                    } else {
                        open_flow += 1;
                    }
                }
            }
        }
        assert_eq!(closed_entries, 0, "no vehicle enters a closed crossing");
        assert!(open_flow > 100, "traffic flows between trains: {open_flow}");
        assert_eq!(w.crashed(), 0);
    }

    #[test]
    fn hov_lane_admits_only_eligible_vehicles() {
        // A 3-lane freeway whose median lane is `hov:lanes`-designated: across a
        // congested run, no ineligible vehicle ever occupies it, and eligible
        // ones do use it (the incentive is the jammed GP lanes beside it).
        let net = OsmMap {
            nodes: vec![NodeSpec::uncontrolled(1, 0.0, 0.0), NodeSpec::uncontrolled(2, 900.0, 0.0)],
            links: vec![LinkSpec {
                road_class: "motorway".into(),
                hov_lanes: "designated|no|no".into(),
                ..LinkSpec::oneway(1, 2, 3, 29.0)
            }],
        }
        .build();
        let hov_lane = (0..net.lanes.len() as u32).map(LaneId).find(|&l| net.lane_is_hov(l));
        assert_eq!(hov_lane, Some(LinkId(0)).map(|l| net.link(l).lane_start), "the median lane carries the flag");
        let mut w = NetWorld::new(net, cfg());
        let mut id = 0u32;
        let (mut eligible_used, mut violations) = (0u32, 0u32);
        for _ in 0..1500 {
            if w.spawn_routed(id, vec![LinkId(0)], 20.0, DriverConfig::car().sample(cfg().seed, id)) {
                id += 1;
            }
            w.step();
            for v in w.vehicles() {
                if w.network.lane_is_hov(v.lane) {
                    if hov_eligible(cfg().seed, v.id) {
                        eligible_used += 1;
                    } else {
                        violations += 1;
                    }
                }
            }
        }
        assert_eq!(violations, 0, "ineligible vehicles never occupy the HOV lane");
        assert!(eligible_used > 50, "eligible vehicles do use the HOV lane: {eligible_used}");
    }

    #[test]
    fn ramp_meter_paces_the_ramp_and_adapts_to_mainline_occupancy() {
        // Freeway W→X→E with an on-ramp R joining at X. With metering on, ramp
        // discharge is paced near the commanded rate; when the mainline is
        // saturated the ALINEA update walks the rate down toward its floor.
        let hw = |a: i64, b: i64, lanes: u32, v: f64| {
            LinkSpec { road_class: "motorway".into(), ..LinkSpec::oneway(a, b, lanes, v) }
        };
        let ramp = |a: i64, b: i64| LinkSpec { road_class: "motorway_link".into(), ..LinkSpec::oneway(a, b, 1, 18.0) };
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -600.0, 0.0),
                NodeSpec::uncontrolled(2, 0.0, 0.0),
                NodeSpec::uncontrolled(3, 600.0, 0.0),
                NodeSpec::uncontrolled(4, -400.0, -120.0),
            ],
            // The protected mainline X→E is a one-lane 10 m/s choke, so merge
            // oversupply genuinely saturates it (the regime a meter exists for).
            links: vec![hw(1, 2, 2, 29.0), hw(2, 3, 1, 10.0), ramp(4, 2)], // 0: W→X, 1: X→E, 2: R→X
        }
        .build();
        let mut w = NetWorld::new(net, SimConfig { red_run_prob: 0.0, ..cfg() });
        w.set_ramp_metering(true);
        assert_eq!(w.ramp_meter_count(), 1, "the on-ramp is discovered");
        let ramp_link = LinkId(2);
        let initial_rate = w.meter_rate(ramp_link).unwrap();

        // Saturate the ramp (refill whenever the entrance clears) and load the
        // mainline heavily; run ten minutes.
        let mut id = 0u32;
        let entries_before = w.link_entry_counts()[1];
        for t in 0..3000u32 {
            if t % 3 == 0 {
                if w.spawn_routed(id, vec![LinkId(0), LinkId(1)], 24.0, DriverConfig::car()) {
                    id += 1;
                }
            }
            if w.spawn_routed(id, vec![LinkId(2), LinkId(1)], 12.0, DriverConfig::car()) {
                id += 1;
            }
            w.step();
        }
        let rate = w.meter_rate(ramp_link).unwrap();
        assert!(rate < initial_rate, "a saturated mainline walks the ALINEA rate down: {rate} < {initial_rate}");
        // Ramp throughput over the 600 s ≈ the average commanded rate, never a
        // free-flow flood (an unmetered saturated ramp would push well over
        // 1,200 veh/h through the merge).
        let merged = w.link_entry_counts()[1] - entries_before;
        let vph = merged as f64 / (600.0 / 3600.0);
        assert!(
            vph < 1900.0,
            "metering paces the combined merge below a flood: {vph:.0} veh/h"
        );
        assert!(w.crashed() == 0, "metered merge stays crash-free");
    }

    #[test]
    fn sober_platoon_bursts_through_permissive_lefts_stay_junction_crash_free() {
        // The latent conflict-crash gap the demand tuning has skirted (PLAN P1.1):
        // nose-to-tail platoons on the throughs while slow heavy vehicles turn left
        // through the permissive box, red-running disabled — so a junction-kind
        // crash here is a modeling artifact by definition, never "realism".
        let net = super::super::map::arterial_intersection();
        let mut total = [0u32; 2];
        for seed in 0..6u64 {
            let mut w = NetWorld::new(net.clone(), SimConfig { red_run_prob: 0.0, seed, ..cfg() });
            let throughs = [[LinkId(0), LinkId(2)], [LinkId(3), LinkId(1)]];
            let lefts = [[LinkId(0), LinkId(6)], [LinkId(3), LinkId(5)]]; // W→N, E→S
            let mut next = 0u32;
            for t in 0..3000u32 {
                // A 4-car burst fired nose-to-tail at each through approach — the
                // bunched arrivals an upstream signal releases.
                if t % 150 < 4 {
                    for r in &throughs {
                        let d = VehicleClass::Car.driver().sample(seed, next);
                        if w.spawn_routed(next, r.to_vec(), 12.0, d) {
                            next += 1;
                        }
                    }
                }
                // A steady left stream with the heavy classes mixed in — the slow
                // crossers whose clearance time the gap acceptance must cover.
                if t % 40 == 0 {
                    for (k, r) in lefts.iter().enumerate() {
                        let class = match (t / 40 + k as u32) % 3 {
                            0 => VehicleClass::Truck,
                            1 => VehicleClass::Bus,
                            _ => VehicleClass::Car,
                        };
                        let d = class.driver().sample(seed, next);
                        if w.spawn_routed(next, r.to_vec(), 10.0, d) {
                            next += 1;
                        }
                    }
                }
                w.step();
            }
            let counts = w.crash_counts();
            total[0] += counts[0];
            total[1] += counts[1];
            assert!(w.exited() > 50, "traffic keeps flowing (seed {seed}): {} exited", w.exited());
        }
        assert_eq!(total[1], 0, "junction crashes with no red-runners are artifacts: {total:?}");
    }

    #[test]
    fn sober_real_map_burst_stays_junction_crash_free() {
        // Regression for the P1.1 artifact chain at multi-node clusters (El Camino
        // × Millbrae Ave): permissive lefts accepting sub-clearance gaps, ungated
        // coalesced-corridor seams, same-tick chain races, and hot internal-line
        // overruns each produced sober (no red-running) junction T-bones under
        // burst demand. These seeds crashed before the fixes; they must stay clean.
        use super::super::demand::{self, DemandGenerator};
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
        let Ok(text) = std::fs::read_to_string(path) else { return };
        let net = super::super::map::OsmMap::from_json(&text).expect("map json").build();
        for seed in [2u64, 3] {
            let mut world = NetWorld::new(net.clone(), SimConfig { red_run_prob: 0.0, seed, ..cfg() });
            let pairs = demand::boundary_od_pairs(&world.network, seed, 32);
            let mut gen = DemandGenerator::new(&world, &pairs, seed);
            gen.set_rate_scale(2.0);
            world.install_router(&gen.destinations());
            for _ in 0..3000 {
                gen.step(&mut world, cfg().dt);
                world.step();
            }
            // The envelope after the 2026-08-11 gap/box hardening: the hot T-bone
            // classes are gone; what remains is a rare low-speed box-convergence
            // graze (a dilemma-committed turner meeting a just-entered through,
            // ≲5 m/s closing) — one pair per heavy-burst run at worst, tracked
            // for the graded-occupancy work in PLAN.md.
            assert!(
                world.crash_counts()[1] <= 2,
                "sober burst junction crashes stay within the graze envelope (seed {seed}): {:?}",
                world.crash_counts()
            );
            // Threshold set under the land-use-weighted map: pedestrian green
            // floors lengthen downtown cycles, so 2× burst throughput sits lower
            // than the pre-floor era.
            assert!(world.exited() > 140, "traffic still flows under the tightened gates (seed {seed}): {}", world.exited());
        }
    }

    #[test]
    #[ignore]
    fn diag_sequoia_stop_seizure() {
        // Reproduce the parked-forever car at Trousdale × Sequoia (tests/load.rs
        // diag_all_way_streak) and dump every gate holding it.
        use super::super::demand::{self, DemandGenerator, DemandSources};
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
        let Ok(text) = std::fs::read_to_string(path) else { return };
        let net = super::super::map::OsmMap::from_json(&text).expect("map json").build();
        let mut world = NetWorld::new(net, cfg());
        let pairs = demand::od_pairs(&world.network, 0, 600, DemandSources::new(true, true));
        let mut gen = DemandGenerator::new(&world, &pairs, 0);
        world.install_router(&gen.destinations());
        for _ in 0..4500 {
            gen.step(&mut world, cfg().dt);
            world.step();
        }
        let link = LinkId(1354);
        let nb = world.neighbors();
        let Some((i, v)) = world
            .fleet
            .rows
            .iter()
            .enumerate()
            .filter(|(_, v)| world.network.lane(v.lane).link == link && v.crossing.is_none())
            .max_by(|a, b| a.1.position.total_cmp(&b.1.position))
        else {
            println!("nobody on 1189");
            return;
        };
        let ln = world.network.lane(v.lane);
        let node = world.downstream_node(v.lane);
        let intended = world.intended_movement(v);
        println!(
            "front id{} pos {:.1}/{:.1} v {:.2} stopped_at {:?} wait {:.0}s intended {:?} node {} control {:?}",
            v.id, v.position, ln.length, v.speed, v.stopped_at.map(|n| n.0),
            v.wait_ticks as f64 * 0.2, intended.map(|m| m.0), node.0, world.network.node(node).control,
        );
        if let Some(mid) = intended {
            println!("  state {:?}", world.movement_state(mid));
            println!("  box_entry_blocked {}", world.box_entry_blocked(i, Some(mid), &nb));
            println!("  box_conflict_on_path {}", world.box_conflict_on_path_holding(v, mid, node, &nb, None));
            println!("  junction_exit_blocked {}", world.junction_exit_blocked(v, mid, node, &nb));
            println!("  downstream_blocked {}", world.movement_downstream_blocked(mid, &v.driver, &nb));
            println!("  earlier_stopped_conflict {}", world.earlier_stopped_conflict(i, mid, node, &nb));
            println!("  priority {:?}", world.conflicting_priority_traffic(i, v.lane, node, &nb));
            println!("  interior_len {:.1}", world.network.interior(mid).len);
            let to_lane = world.network.movement(mid).to_lane;
            println!(
                "  receiving link {} front {:?}",
                world.network.lane(to_lane).link.0,
                nb.lane_front.get(&to_lane.0).map(|&f| {
                    let o = &world.fleet.rows[f];
                    (o.id, o.position, o.speed)
                })
            );
            let cx = world.gather_context(i, &nb, intended);
            println!(
                "  cx: stop_line {:.1} stop_sign {:.1} yield {:.1} target ({:.1}@{:.1}) leader ({:.1}, v {:.1})",
                cx.stop_line, cx.stop_sign, cx.yield_line, cx.speed_target_speed, cx.speed_target_dist,
                cx.leader_gap, cx.leader_speed,
            );
            let key = world.network.intersection_key(node);
            for &j in nb.crossing_at.get(&key).into_iter().flatten() {
                let o = &world.fleet.rows[j];
                let c = o.crossing.unwrap();
                println!(
                    "  crosser id{} mid{} arc {:.1}/{:.1} v {:.2} conflicts_mine {}",
                    o.id, c.movement.0, world.crossing_arc(o), world.network.interior(c.movement).len,
                    o.speed, world.network.movements_conflict(mid, c.movement),
                );
            }
            println!("  node_junction(388) = {:?}", world.network.node_junction(node));
            for &j in nb.approaching.get(&key).into_iter().flatten() {
                let o = &world.fleet.rows[j];
                if o.id == v.id {
                    continue;
                }
                let ol = world.network.lane(o.lane);
                if ol.length - o.position > 40.0 {
                    continue;
                }
                let o_mid = world.intended_movement(o);
                println!(
                    "  near id{} '{}' pos {:.1}/{:.1} v {:.2} stopped_at {:?} defer {} ctrl {:?} conflicts_321 {:?}",
                    o.id,
                    world.network.link_names[ol.link.idx()],
                    o.position,
                    ol.length,
                    o.speed,
                    o.stopped_at.map(|n| n.0),
                    o_mid.is_some_and(|m| world.earlier_stopped_conflict(j, m, world.downstream_node(o.lane), &nb)),
                    world.network.node(world.downstream_node(o.lane)).control,
                    o_mid.map(|m| world.network.movements_conflict(m, mid)),
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn diag_discharge_stopper() {
        // The queue-discharge fixture: catch the front car the moment it stops
        // short of a green line and dump its whole constraint context.
        use super::super::map::SignalPlan;
        let plan = SignalPlan { green_secs: 60.0, yellow_secs: 4.0, offset: 30.0 };
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -600.0, 0.0),
                NodeSpec::signalized(2, 0.0, 0.0, plan),
                NodeSpec::uncontrolled(3, 400.0, 0.0),
                NodeSpec::uncontrolled(4, 0.0, -200.0),
                NodeSpec::uncontrolled(5, 0.0, 200.0),
            ],
            links: {
                let mut v = vec![LinkSpec::oneway(1, 2, 1, 15.0), LinkSpec::oneway(2, 3, 1, 15.0)];
                v.extend(LinkSpec::twoway(4, 2, 1, 10.0));
                v.extend(LinkSpec::twoway(2, 5, 1, 10.0));
                v
            },
        }
        .build();
        let mut w = NetWorld::new(net, SimConfig { red_run_prob: 0.0, ..cfg() });
        let mut id = 0u32;
        let quiet = DriverConfig { accel_noise: 0.0, ..DriverConfig::car() };
        for _ in 0..2200 {
            if w.spawn_routed(id, vec![LinkId(0), LinkId(1)], 10.0, quiet) {
                id += 1;
            }
            w.step();
        }
        let lane0 = w.network.link(LinkId(0)).lane_start;
        // Advance until the front car is parked just short of the green line.
        for _ in 0..3000 {
            let stopped_at_line = w
                .fleet
                .rows
                .iter()
                .filter(|v| v.lane == lane0 && v.crossing.is_none())
                .max_by(|a, b| a.position.total_cmp(&b.position))
                .is_some_and(|v| v.speed < 0.3 && v.position > 585.0);
            if stopped_at_line {
                break;
            }
            if w.spawn_routed(id, vec![LinkId(0), LinkId(1)], 10.0, quiet) {
                id += 1;
            }
            w.step();
        }
        let nb = w.neighbors();
        let Some((i, v)) = w
            .fleet
            .rows
            .iter()
            .enumerate()
            .filter(|(_, v)| v.lane == lane0 && v.crossing.is_none())
            .max_by(|a, b| a.1.position.total_cmp(&b.1.position))
        else {
            return;
        };
        let ln = w.network.lane(lane0);
        let intended = w.intended_movement(v);
        println!("front id{} pos {:.1}/{:.1} v {:.2} intended {:?}", v.id, v.position, ln.length, v.speed, intended.map(|m| m.0));
        if let Some(mid) = intended {
            println!("  state {:?} green_elapsed {:.1}", w.movement_state(mid), w.signal_green_elapsed(mid));
            println!("  downstream_blocked {}", w.movement_downstream_blocked(mid, &v.driver, &nb));
            println!("  box_entry_blocked {}", w.box_entry_blocked(i, Some(mid), &nb));
            let node = w.downstream_node(v.lane);
            println!("  box_conflict_on_path {}", w.box_conflict_on_path_holding(v, mid, node, &nb, None));
            println!("  junction_exit_blocked {}", w.junction_exit_blocked(v, mid, node, &nb));
            println!("  is_permissive {}", w.is_permissive(mid));
            println!(
                "  crossing_mvs at node: {:?}",
                nb.crossing_mvs.get(&w.network.intersection_key(node))
            );
            let cx = w.gather_context(i, &nb, intended);
            println!(
                "  cx: stop_line {:.1} target ({:.1}@{:.1}) stop_sign {:.1} yield {:.1} curve ({:.2}@{:.1}) leader ({:.1}, v {:.1}) merge ({:.1}, v {:.1})",
                cx.stop_line, cx.speed_target_speed, cx.speed_target_dist, cx.stop_sign, cx.yield_line,
                cx.curve_speed, cx.curve_dist, cx.leader_gap, cx.leader_speed, cx.merge_gap, cx.merge_speed,
            );
        }
    }

    #[test]
    #[ignore]
    fn diag_gateway_seam_holder() {
        // Reproduce the jammed US-101 gateway and dump every gate for the
        // front-most stuck car on a middle lane. Run with -- --ignored --nocapture.
        use super::super::demand::{self, DemandGenerator, DemandSources};
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
        let Ok(text) = std::fs::read_to_string(path) else { return };
        let net = super::super::map::OsmMap::from_json(&text).expect("map json").build();
        let mut world = NetWorld::new(net, cfg());
        let sources = DemandSources::with_rush_hour(true, false, true);
        let pairs = demand::od_pairs_with_commute(&world.network, 0xC0FFEE, 48, sources, None);
        let mut gen = DemandGenerator::new(&world, &pairs, 0xC0FFEE);
        gen.set_rush_hour(&world.network, true);
        gen.set_day_compression(12.0);
        gen.resume_clock(7.0 * 3600.0, 0);
        world.install_router(&gen.destinations());
        for _ in 0..4000 {
            gen.step(&mut world, cfg().dt);
            world.step();
        }
        let link = *world.network.link(LinkId(990));
        let lane = LaneId(link.lane_start.0 + 2);
        let nb = world.neighbors();
        let Some((i, v)) = world
            .fleet
            .rows
            .iter()
            .enumerate()
            .filter(|(_, v)| v.lane == lane && v.crossing.is_none())
            .max_by(|a, b| a.1.position.total_cmp(&b.1.position))
        else {
            println!("no car on 990 lane 2");
            return;
        };
        let ln = world.network.lane(lane);
        let node = world.downstream_node(lane);
        let intended = world.intended_movement(v);
        println!(
            "front car id{} pos {:.1}/{:.1} v {:.2} intended {:?}",
            v.id, v.position, ln.length, v.speed, intended.map(|m| m.0)
        );
        if let Some(mid) = intended {
            println!("  free_flow_seam={}", world.free_flow_seam(mid));
            println!("  is_intra_corridor={}", world.is_intra_corridor(mid));
            println!("  free_flow_interchange={}", world.free_flow_interchange(mid));
            println!("  box_entry_blocked={}", world.box_entry_blocked(i, Some(mid), &nb));
            println!("  box_conflict_on_path={}", world.box_conflict_on_path_holding(v, mid, node, &nb, None));
            println!("  junction_exit_blocked={}", world.junction_exit_blocked(v, mid, node, &nb));
            println!("  movement_downstream_blocked={}", world.movement_downstream_blocked(mid, &v.driver, &nb));
            println!("  meter_red={} rail_closed={}", world.meter_red(ln.link), world.rail_closed(node));
            println!("  movement_turn={:?} turn_cap={}", world.network.movement_turn(mid), world.turn_speed_cap(mid));
            println!("  receiving lane front: {:?}", nb.lane_front.get(&world.network.movement(mid).to_lane.0));
            if let Some(li) = nb.leader_of[i] {
                let l = &world.fleet.rows[li];
                println!(
                    "  leader id{} lane{} pos {:.1} v {:.2} gap {:.1}",
                    l.id, l.lane.0, l.position, l.speed, world.corridor_gap(v, l)
                );
            } else {
                println!("  no leader");
            }
            let cx = world.gather_context(i, &nb, intended);
            println!(
                "  cx: v {:.2} desired {:.1} stop_line {:.1} target ({:.1}@{:.1}) stop_sign {:.1} yield {:.1} curve ({:.2}@{:.1}) leader ({:.1} gap, v {:.1}) merge ({:.1} gap, v {:.1})",
                cx.speed, cx.driver.desired_speed, cx.stop_line, cx.speed_target_speed, cx.speed_target_dist,
                cx.stop_sign, cx.yield_line, cx.curve_speed, cx.curve_dist,
                cx.leader_gap, cx.leader_speed, cx.merge_gap, cx.merge_speed,
            );
        }
    }

    #[test]
    #[ignore]
    fn probe_real_map_burst_crashes() {
        // Diagnostic: sober drivers (no red-running), boundary demand at rising
        // burst intensity on the real map — junction-kind crashes here are the
        // P1.1 artifacts. Run with `-- --ignored --nocapture`.
        use super::super::demand::{self, DemandGenerator};
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
        let Ok(text) = std::fs::read_to_string(path) else { return };
        let net = super::super::map::OsmMap::from_json(&text).expect("map json").build();
        for scale in [1.0, 2.0, 4.0] {
            for seed in 0..4u64 {
                let mut world = NetWorld::new(net.clone(), SimConfig { red_run_prob: 0.0, seed, ..cfg() });
                let pairs = demand::boundary_od_pairs(&world.network, seed, 32);
                let mut gen = DemandGenerator::new(&world, &pairs, seed);
                gen.set_rate_scale(scale);
                world.install_router(&gen.destinations());
                for _ in 0..3000 {
                    gen.step(&mut world, cfg().dt);
                    world.step();
                }
                println!(
                    "scale {scale} seed {seed}: crashes {:?} (rear-end, junction), exited {}, on-road {}",
                    world.crash_counts(),
                    world.exited(),
                    world.vehicles().len()
                );
            }
        }
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
                    pairs.push(OdPair { origin: LinkId(o), dest: LinkId(d), rate_per_sec: 0.3, class: SurfaceClass::Through, anchored: false });
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
                // Clearly below the 16 m/s cruise: the P2.2 comfort-braking retune
                // (b = 2.0) legitimately starts the ease-off later than the old 1.5.
                if on_w_a && w.movement_state(through) == SignalState::Red && v.speed < 13.0 {
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
            // Crash mechanisms (reaction delay, bounded braking, red-running) are on by
            // design, so the envelope bound is *rarity*, not zero: a handful of plausible
            // low-speed collisions per run, never a systemic storm.
            assert!(
                world.crashed() <= 4,
                "flow-field-routed traffic stays near collision-free (seed {seed}): {} crashed ({:?})",
                world.crashed(),
                world.crash_counts()
            );
            assert!(world.exited() > 0, "vehicles complete boundary trips (seed {seed}), got {}", world.exited());
            assert_eq!(world.leaked(), 0, "no car disappears at an intersection (seed {seed}), leaked {}", world.leaked());
        }
    }

    #[test]
    fn map_build_is_deterministic_across_runs() {
        // A std `HashMap`'s iteration order is randomly seeded per instance, so any build step
        // that emits network structure from one (node clustering, signal phasing) makes the
        // whole sim non-reproducible run to run — and intermittently collide. Two builds of the
        // same map, each with freshly seeded hash maps, must produce identical node/link order
        // and identical signal green masks (the subtle path that keyed off node order).
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
        let Ok(text) = std::fs::read_to_string(path) else { return };
        let build = || super::super::map::OsmMap::from_json(&text).expect("map json").build();
        let (a, b) = (build(), build());
        let poses = |net: &Network| net.nodes.iter().map(|n| n.position).collect::<Vec<_>>();
        assert_eq!(poses(&a), poses(&b), "node order is deterministic across builds");
        let link_ends = |net: &Network| net.links.iter().map(|l| (l.from.0, l.to.0)).collect::<Vec<_>>();
        assert_eq!(link_ends(&a), link_ends(&b), "link topology is deterministic across builds");
        let masks =
            |net: &Network| net.programs.iter().flat_map(|p| p.phases.iter().map(|ph| ph.green_mask)).collect::<Vec<_>>();
        assert_eq!(masks(&a), masks(&b), "signal green masks are deterministic across builds");
    }

    #[test]
    fn stopped_traffic_queues_on_approaches_not_inside_junctions() {
        // The reported defect: at a multi-node junction a car crossed the first node
        // and stopped on the short link *inside* the box (between sibling nodes),
        // blocking cross traffic. `junction_exit_blocked` (don't-block-the-box across
        // the whole cluster) holds it at the outer stop line, so the bulk of stopped
        // traffic sits on approaches, not on the internal links inside a junction.
        use super::super::demand::{self, DemandGenerator};
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
        let Ok(text) = std::fs::read_to_string(path) else { return };
        let net = super::super::map::OsmMap::from_json(&text).expect("map json").build();
        let internal = |net: &Network, link: LinkId| {
            let l = net.link(link);
            let a = net.node_junction(l.from);
            a.is_some() && a == net.node_junction(l.to)
        };
        if !(0..net.links.len() as u32).any(|i| internal(&net, LinkId(i))) {
            return; // no multi-node junction with an internal link to measure
        }
        let mut world = NetWorld::new(net, cfg());
        let pairs = demand::boundary_od_pairs(&world.network, 7, 20);
        let mut gen = DemandGenerator::new(&world, &pairs, 7);
        world.install_router(&gen.destinations());
        let (mut idle_inside, mut idle_total) = (0u64, 0u64);
        for _ in 0..1200 {
            gen.step(&mut world, cfg().dt);
            world.step();
            for v in world.vehicles() {
                if v.speed >= 0.5 || v.crossing.is_some() {
                    continue;
                }
                idle_total += 1;
                if internal(&world.network, world.network.lane(v.lane).link) {
                    idle_inside += 1;
                }
            }
        }
        assert!(idle_total > 0, "the run produced stopped traffic to measure");
        let frac = idle_inside as f64 / idle_total as f64;
        assert!(frac < 0.35, "stopped traffic piles up inside junctions: {idle_inside}/{idle_total} = {frac:.3}");
    }

    #[test]
    fn sleep_scheduler_matches_the_all_cars_step() {
        // The active-set scheduler must drive a congesting network as safely as the
        // all-cars reference and carry comparable throughput — it changes which cars run
        // the full gather, not the physics of a car that is actually deciding. The
        // `corridor_with_signal` scenario backs its two-lane arterial into a standing
        // queue behind the signal (a *non-merge* approach, so the queue sleeps), while
        // the cross street keeps the intersection busy. Runs the same demand with the
        // scheduler off (reference) vs on and compares.
        let run = |sleep: bool| {
            let cfg = SimConfig { sleep_scheduler: sleep, ..cfg() };
            let mut world = NetWorld::new(corridor_with_signal(), cfg);
            // dests: link 1 (arterial through, 1→2→4), link 3 (cross, 3→2→5).
            world.install_router(&[LinkId(1), LinkId(3)]);
            let mut next = 0u32;
            let mut peak_asleep = 0usize;
            for t in 0..4000 {
                // Push both approaches; spawn_to refuses when the entrance is occupied,
                // which becomes natural inflow backpressure (a standing queue at the red).
                if world.spawn_to(next, LinkId(0), LinkId(1), 15.0, DriverConfig::car()) {
                    next += 1;
                }
                if t % 2 == 0 && world.spawn_to(next, LinkId(2), LinkId(3), 12.0, DriverConfig::car()) {
                    next += 1;
                }
                world.step();
                peak_asleep = peak_asleep.max(world.asleep_count());
            }
            assert_eq!(world.crashed(), 0, "sleep={sleep}: stays collision-free");
            assert_eq!(world.leaked(), 0, "sleep={sleep}: no car vanishes at a node");
            (world.exited(), peak_asleep)
        };
        let (ref_exit, ref_asleep) = run(false);
        let (sched_exit, sched_asleep) = run(true);
        assert_eq!(ref_asleep, 0, "the reference all-cars step never sleeps a car");
        assert!(sched_asleep > 0, "the scheduler actually sleeps queued cars, got {sched_asleep}");
        assert!(sched_exit > 0, "the scheduled run keeps traffic flowing, got {sched_exit}");
        let ratio = sched_exit as f64 / ref_exit as f64;
        assert!((0.85..1.18).contains(&ratio), "scheduler throughput {sched_exit} tracks reference {ref_exit} (ratio {ratio:.2})");
    }

    #[test]
    fn sleep_scheduler_stays_collision_free_on_the_real_map() {
        // The safety guarantee within the model's *guaranteed envelope*: the real city
        // under the same boundary demand the all-cars step keeps collision-free
        // (`real_map_boundary_demand_routes_live_and_stays_safe`), now with the scheduler
        // on. Must stay crash- and leak-free and keep traffic flowing.
        use super::super::demand::{self, DemandGenerator};
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
        let Ok(text) = std::fs::read_to_string(path) else { return };
        let net = super::super::map::OsmMap::from_json(&text).expect("map json").build();
        for seed in 0..4u64 {
            let mut world = NetWorld::new(net.clone(), SimConfig { sleep_scheduler: true, ..cfg() });
            let pairs = demand::boundary_od_pairs(&world.network, seed, 16);
            let mut gen = DemandGenerator::new(&world, &pairs, seed);
            world.install_router(&gen.destinations());
            let mut peak_asleep = 0usize;
            for _ in 0..1500 {
                gen.step(&mut world, cfg().dt);
                world.step();
                peak_asleep = peak_asleep.max(world.asleep_count());
            }
            // Same rarity envelope as the all-cars reference run above: the scheduler
            // must not add crashes beyond the model's own rare plausible collisions.
            assert!(
                world.crashed() <= 4,
                "scheduler stays near collision-free on the real map (seed {seed}): {} crashed ({:?})",
                world.crashed(),
                world.crash_counts()
            );
            assert_eq!(world.leaked(), 0, "scheduler leaks no car at a node (seed {seed})");
            assert!(world.exited() > 0, "scheduled traffic completes trips (seed {seed})");
            assert!(peak_asleep > 0, "the scheduler sleeps queued cars on the real map (seed {seed})");
        }
    }

    #[test]
    #[ignore] // needs the real map + a few thousand cars; run with `--features import -- --ignored`
    fn no_vehicle_pose_teleports_under_high_load() {
        // The render draws each car at its world pose and interpolates linearly toward it,
        // so a car can only appear "somewhere it shouldn't" (flash across the screen) if
        // its *sim* pose jumps there in a single tick. Encodes the invariant that a
        // vehicle's world position moves at most one tick of travel per tick — a generous
        // bound covering the longest node-interior crossing. Reproduces (or rules out) the
        // >2000-car flashing artifact in the sim layer. Checked with the active-set
        // scheduler both off and on, since it is on by default in the browser.
        use super::super::demand::{self, DemandGenerator, DemandSources};
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
        let Ok(text) = std::fs::read_to_string(path) else { return };
        let net = super::super::map::OsmMap::from_json(&text).expect("map json").build();
        let dt = cfg().dt;
        // A one-tick move can't exceed the fastest reachable speed × dt; add slack for the
        // longest node interior a crossing can complete in one tick. Anything past this is
        // a teleport, not motion.
        let max_speed = net.lanes.iter().map(|l| l.speed_limit).fold(0.0_f64, f64::max) * 1.3;
        let longest_interior = net.interiors.iter().map(|it| it.len).fold(0.0_f64, f64::max);
        let bound = max_speed * dt + longest_interior + 5.0;

        for sleep in [false, true] {
            let mut world = NetWorld::new(net.clone(), SimConfig { sleep_scheduler: sleep, ..cfg() });
            let pairs = demand::od_pairs(&world.network, 1, 600, DemandSources::new(true, true));
            let mut gen = DemandGenerator::new(&world, &pairs, 1);
            world.install_router(&gen.destinations());

            let mut prev: IntMap<[f64; 2]> = IntMap::default();
            let (mut worst, mut worst_id) = (0.0_f64, 0u32);
            for _ in 0..2500 {
                gen.step(&mut world, dt);
                world.step();
                let mut next: IntMap<[f64; 2]> = IntMap::default();
                for v in world.vehicles() {
                    let p = world.vehicle_world_pose(v);
                    if let Some(q) = prev.get(&v.id) {
                        let d = (p[0] - q[0]).hypot(p[1] - q[1]);
                        if d > worst {
                            (worst, worst_id) = (d, v.id);
                        }
                    }
                    next.insert(v.id, [p[0], p[1]]);
                }
                prev = next;
            }
            eprintln!(
                "sleep={sleep}: peak {} cars, worst one-tick pose jump {worst:.1} m (bound {bound:.1}, id {worst_id})",
                world.vehicles().len()
            );
            assert!(worst < bound, "sleep={sleep}: vehicle {worst_id} teleported {worst:.1} m in one tick (bound {bound:.1})");
        }
    }

    #[test]
    fn rebuilt_demand_keeps_vehicle_ids_unique() {
        // A demand-source / rush-hour toggle rebuilds the `DemandGenerator`. A fresh
        // generator restarts its id counter at 0, so its spawns reissue ids still held by
        // cars on the map — two live vehicles share an id, which the renderer's per-id
        // `prev` pose map can only key once, so one car interpolates from the other's
        // position and flashes across the screen. This is the >2000-car flashing artifact.
        // The rebuilt generator must continue the id sequence past every live vehicle.
        use super::super::demand::{self, DemandGenerator, DemandSources};
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
        let Ok(text) = std::fs::read_to_string(path) else { return };
        let net = super::super::map::OsmMap::from_json(&text).expect("map json").build();
        let dt = cfg().dt;

        // Run demand, rebuild the generator mid-flight (as a toggle does), run more, and
        // return (live vehicle count, distinct-id count). `carry` chooses the fix.
        let run = |carry: bool| -> (usize, usize) {
            let mut world = NetWorld::new(net.clone(), cfg());
            let pairs = demand::od_pairs(&world.network, 1, 300, DemandSources::new(true, true));
            let mut gen = DemandGenerator::new(&world, &pairs, 1);
            world.install_router(&gen.destinations());
            for _ in 0..600 {
                gen.step(&mut world, dt);
                world.step();
            }
            assert!(world.vehicles().len() > 50, "enough cars on the map to collide ids");
            let next_id = if carry {
                world.vehicles().iter().map(|v| v.id).max().map_or(0, |m| m + 1).max(gen.next_id())
            } else {
                0 // the bug: restart the counter
            };
            let mut gen2 = DemandGenerator::new(&world, &pairs, 1);
            gen2.set_next_id(next_id);
            for _ in 0..600 {
                gen2.step(&mut world, dt);
                world.step();
            }
            let mut ids: Vec<u32> = world.vehicles().iter().map(|v| v.id).collect();
            let live = ids.len();
            ids.sort_unstable();
            ids.dedup();
            (live, ids.len())
        };

        // Without carrying the counter, the rebuild reissues live ids → duplicates.
        let (live0, uniq0) = run(false);
        assert!(uniq0 < live0, "a naive rebuild must reissue live ids (live {live0}, unique {uniq0})");
        // Carrying it past every live vehicle keeps every id unique — no aliasing, no flash.
        let (live1, uniq1) = run(true);
        assert_eq!(uniq1, live1, "duplicate vehicle ids remained after a demand rebuild (live {live1}, unique {uniq1})");
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
    fn a_freeway_off_ramp_diverge_has_no_lane_merge() {
        // A 6-lane freeway splits into a 4-lane continuation and a 2-lane off-ramp. The
        // curb (exit) lanes map to the ramp *only* — past the gore they are physically
        // separated from the mainline — and each continuation lane is fed by exactly one
        // freeway lane, so there is no lane-drop merge choking the mainline. A through
        // driver caught in an exit lane reaches the freeway by changing lanes upstream,
        // not by three lanes funnelling into one at the node.
        let hw = |a, b, lanes| LinkSpec { road_class: "motorway".into(), ..LinkSpec::oneway(a, b, lanes, 29.0) };
        let ramp = |a, b, lanes| LinkSpec { road_class: "motorway_link".into(), ..LinkSpec::oneway(a, b, lanes, 11.0) };
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -400.0, 0.0),   // freeway in
                NodeSpec::uncontrolled(2, 0.0, 0.0),      // diverge
                NodeSpec::uncontrolled(3, 500.0, 0.0),    // continuation
                NodeSpec::uncontrolled(4, 300.0, -260.0), // off-ramp (a real right-turn angle)
            ],
            links: vec![hw(1, 2, 6), hw(2, 3, 4), ramp(2, 4, 2)],
        }
        .build();
        let (mainline, cont, ramp_l) = (LinkId(0), LinkId(1), LinkId(2));
        let feeders = |to: LinkId, idx: u32| -> Vec<u32> {
            (0..net.movements.len() as u32)
                .filter(|&m| {
                    let mv = net.movement(MovementId(m));
                    net.lane(mv.to_lane).link == to && net.lane(mv.to_lane).index_in_link == idx
                })
                .map(|m| net.lane(net.movement(MovementId(m)).from_lane).index_in_link)
                .collect()
        };
        // Every continuation lane is fed by exactly one freeway lane — no merge.
        for idx in 0..net.link(cont).lane_count {
            let f = feeders(cont, idx);
            assert_eq!(f.len(), 1, "continuation lane {idx} must have a single feeder (no merge), got {f:?}");
        }
        // The two curb lanes feed the ramp and do NOT continue on the mainline.
        for k in [4u32, 5] {
            let lane = LaneId(net.link(mainline).lane_start.0 + k);
            let dests: Vec<LinkId> = net.movements_of(lane).iter().map(|m| net.lane(m.to_lane).link).collect();
            assert!(dests.contains(&ramp_l), "exit lane {k} feeds the off-ramp");
            assert!(!dests.contains(&cont), "exit lane {k} is exit-only — no merge movement onto the mainline, got {dests:?}");
        }
        // And the inner four lanes carry the mainline (a through car in them just continues).
        for k in 0..4u32 {
            let lane = LaneId(net.link(mainline).lane_start.0 + k);
            let dests: Vec<LinkId> = net.movements_of(lane).iter().map(|m| net.lane(m.to_lane).link).collect();
            assert!(dests.contains(&cont), "through lane {k} continues on the mainline, got {dests:?}");
        }
    }

    #[test]
    fn cars_reposition_for_a_freeway_diverge_before_the_gore() {
        // The exit lanes are exit-only, so approaching the diverge a car repositions into
        // the lane its route needs — the *same* mechanism both ways: a through car starting
        // in an exit lane moves into a through lane and continues on the mainline (it is not
        // dragged onto the ramp), and an exit car starting in a through lane moves into an
        // exit lane and takes the ramp. Both happen upstream of the gore.
        let hw = |a, b, lanes| LinkSpec { road_class: "motorway".into(), ..LinkSpec::oneway(a, b, lanes, 29.0) };
        let ramp = |a, b, lanes| LinkSpec { road_class: "motorway_link".into(), ..LinkSpec::oneway(a, b, lanes, 11.0) };
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -400.0, 0.0), // approach start
                NodeSpec::uncontrolled(2, 0.0, 0.0),    // diverge (gore)
                NodeSpec::uncontrolled(3, 600.0, 0.0),  // continuation end
                NodeSpec::uncontrolled(4, 300.0, -300.0), // ramp end
            ],
            links: vec![hw(1, 2, 6), hw(2, 3, 4), ramp(2, 4, 2)],
        }
        .build();
        let mut w = NetWorld::new(net, cfg());
        w.install_router(&[LinkId(1), LinkId(2)]); // continuation exit, ramp exit
        let approach = LinkId(0);
        let curb = w.network.lanes_of(approach).last().unwrap(); // lane 5 (exit lane)
        let median = w.network.lanes_of(approach).next().unwrap(); // lane 0 (through lane)
        w.spawn_to_in_lane(1, curb, 20.0, LinkId(1), 26.0, DriverConfig::car()); // through car in an exit lane
        w.spawn_to_in_lane(2, median, 220.0, LinkId(2), 26.0, DriverConfig::car()); // exit car in a through lane, ahead so the two don't cross abreast
        let (mut through_continued, mut exit_took_ramp) = (false, false);
        for _ in 0..800 {
            w.step();
            if let Some(v) = w.vehicle(1) {
                through_continued |= w.network.lane(v.lane).link == LinkId(1);
            }
            if let Some(v) = w.vehicle(2) {
                exit_took_ramp |= w.network.lane(v.lane).link == LinkId(2);
            }
        }
        assert!(through_continued, "the through car left the exit lane and stayed on the mainline");
        assert!(exit_took_ramp, "the exit car reached an exit lane and took the ramp");
        assert_eq!(w.crashed(), 0, "and the repositioning is collision-free");
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
        // Speed as the car takes the diverge: while inside the node interior, and on the
        // first stretch of the off-ramp just after it lands. (At 26 m/s the sub-tick node
        // interior is entered and cleared within a single step, so the car may never be
        // sampled mid-crossing — the landing speed onto the ramp is the honest measure that
        // the free-flow diverge never throttled it to a 5 m/s turn crawl.)
        let mut max_speed_diverge: f64 = 0.0;
        let mut reached_ramp = false;
        for _ in 0..400 {
            w.step();
            if let Some(v) = w.vehicle(1) {
                let on_ramp_entry = !v.is_crossing()
                    && w.network.lane(v.lane).link == LinkId(2)
                    && v.position < 10.0;
                if v.is_crossing() || on_ramp_entry {
                    max_speed_diverge = max_speed_diverge.max(v.speed);
                }
            }
            reached_ramp |= w.link_flows()[2] > 0.0;
        }
        assert!(reached_ramp, "the car takes the off-ramp");
        assert!(max_speed_diverge > 15.0, "it keeps highway speed through the diverge, got {max_speed_diverge:.1} m/s");
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
    fn two_way_stop_halts_the_minor_street_and_not_the_major() {
        use crate::sim::network::LinkSign;
        // The same crossing as the uncontrolled test, but the minor approach
        // carries a per-approach stop sign (OSM's way-mapped survey). Serving a
        // stop line is exactly what arms `stopped_at`, so it is the clean
        // discriminator: at a two-way stop only minors ever arm; the old
        // all-way reading (node-level control) armed the major street too.
        let cross = |minor_sign: LinkSign, center: NodeSpec| {
            OsmMap {
                nodes: vec![
                    center,
                    NodeSpec::uncontrolled(1, -250.0, 0.0),
                    NodeSpec::uncontrolled(2, 250.0, 0.0),
                    NodeSpec::uncontrolled(3, 0.0, -250.0),
                    NodeSpec::uncontrolled(4, 0.0, 250.0),
                ],
                links: vec![
                    LinkSpec::oneway(1, 0, 1, 22.0), // 0: major W→C
                    LinkSpec::oneway(0, 2, 1, 22.0), // 1: major C→E
                    LinkSpec { sign: minor_sign, ..LinkSpec::oneway(3, 0, 1, 10.0) }, // 2: minor S→C
                    LinkSpec::oneway(0, 4, 1, 10.0), // 3: minor C→N
                ],
            }
            .build()
        };
        let majors_armed = |net: Network| {
            let mut w = NetWorld::new(net, cfg());
            let (mut next, mut armed) = (0u32, 0u32);
            for t in 0..600u32 {
                if t % 12 == 0 && w.spawn_routed(next, vec![LinkId(0), LinkId(1)], 20.0, DriverConfig::car()) {
                    next += 1;
                }
                w.step();
                armed += w.fleet.rows.iter().filter(|v| v.lane == LaneId(0) && v.stopped_at == Some(NodeId(0))).count() as u32;
            }
            armed
        };

        let net = cross(LinkSign::Stop, NodeSpec::uncontrolled(0, 0.0, 0.0));
        assert!(matches!(net.node(NodeId(0)).control, NodeControl::Stop));
        assert!(net.approach_stops(LinkId(2)) && !net.approach_stops(LinkId(0)));
        assert!(!net.all_way_stop(NodeId(0)));
        assert_eq!(majors_armed(net.clone()), 0, "the major street never serves a line at a two-way stop");
        assert!(
            majors_armed(cross(LinkSign::None, NodeSpec::stop(0, 0.0, 0.0))) > 0,
            "control: node-level stop (a surveyed all-way) still lines the major street"
        );

        // Both streams: minors serve their line and gap-accept; flow stays safe.
        let mut w = NetWorld::new(net, cfg());
        let mut next = 0u32;
        let mut minor_served_stop = false;
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
            minor_served_stop |= w
                .fleet
                .rows
                .iter()
                .any(|v| v.lane == w.network.link(LinkId(2)).lane_start && v.stopped_at == Some(NodeId(0)));
        }
        assert!(w.exited() > 20, "traffic should be flowing: {} exited", w.exited());
        assert!(w.crashed() <= 2, "two-way stop should stay safe, got {}", w.crashed());
        assert!(minor_served_stop, "minor-street drivers serve their sign");
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
    fn acceleration_noise_fluctuates_around_desired_without_downward_bias() {
        let steady = |noise: f64| {
            let mut w = NetWorld::new(straight_link(20000.0), cfg());
            let d = DriverConfig { desired_speed: 20.0, accel_noise: noise, reaction_time: 0.0, ..DriverConfig::car() };
            w.spawn(1, LaneId(0), 0.0, 20.0, d);
            let (mut sum, mut sumsq, mut n) = (0.0, 0.0, 0);
            for t in 0..1000 {
                w.step();
                if t >= 800 {
                    if let Some(v) = w.vehicle(1) {
                        sum += v.speed;
                        sumsq += v.speed * v.speed;
                        n += 1;
                    }
                }
            }
            let mean = sum / n as f64;
            (mean, (sumsq / n as f64 - mean * mean).max(0.0).sqrt())
        };
        let (mean0, std0) = steady(0.0);
        let (mean_n, std_n) = steady(0.4);
        assert!((mean0 - 20.0).abs() < 0.1 && std0 < 0.05, "noiseless holds desired: mean {mean0} std {std0}");
        assert!((mean_n - 20.0).abs() < 0.5, "zero-mean noise keeps the mean near desired, not biased below: {mean_n}");
        assert!(std_n > 0.1, "noise produces real speed fluctuation: std {std_n}");
    }

    #[test]
    fn overlapping_vehicles_crash_and_leave_the_road_by_default() {
        // Default config: wreck persistence is off, so crashed vehicles are
        // removed the tick they collide (but are still tallied and logged).
        let mut w = NetWorld::new(straight_link(1000.0), cfg());
        w.spawn(1, LaneId(0), 100.0, 5.0, DriverConfig::car());
        w.spawn(2, LaneId(0), 99.0, 5.0, DriverConfig::car()); // deep overlap into the leader
        w.step();
        assert_eq!(w.crashed(), 2);
        assert_eq!(w.crash_counts(), [2, 0], "both tallied as rear-end");
        assert_eq!(w.crash_log().len(), 2);
        assert_eq!(w.vehicles().len(), 0);
    }

    #[test]
    fn opting_into_wreck_persistence_leaves_the_wreck_until_cleared() {
        let secs = 30.0;
        let mut w = NetWorld::new(straight_link(1000.0), SimConfig { wreck_clear_secs: secs, ..cfg() });
        w.spawn(1, LaneId(0), 100.0, 5.0, DriverConfig::car());
        w.spawn(2, LaneId(0), 99.0, 5.0, DriverConfig::car());
        w.step();
        assert_eq!(w.crashed(), 2);
        assert_eq!(w.vehicles().len(), 2, "the wrecks stay on the road awaiting clearance");
        assert!(w.vehicles().iter().all(|v| v.is_wrecked() && v.speed == 0.0));
        let clear_ticks = (secs / cfg().dt).ceil() as u32;
        w.run_ticks(clear_ticks + 1);
        assert_eq!(w.vehicles().len(), 0, "cleared after wreck_clear_secs");
        assert_eq!(w.crashed(), 2, "clearance does not re-count");
    }

    #[test]
    fn a_follower_queues_behind_a_persistent_wreck_without_joining_it() {
        // With persistence on, a wreck is a real obstruction: traffic behind it
        // queues (no phantom pass-through) rather than piling in from a safe gap.
        let mut w = NetWorld::new(straight_link(1000.0), SimConfig { wreck_clear_secs: 30.0, ..cfg() });
        w.spawn(1, LaneId(0), 500.0, 5.0, DriverConfig::car());
        w.spawn(2, LaneId(0), 499.0, 5.0, DriverConfig::car()); // wrecks the pair
        w.spawn(3, LaneId(0), 400.0, 20.0, DriverConfig::car()); // fast, but far enough to stop
        w.run_ticks(200);
        assert_eq!(w.crashed(), 2, "the approaching car stops behind the wreck instead of joining it");
        let survivor = w.vehicle(3);
        assert!(
            survivor.is_none_or(|v| !v.is_wrecked()),
            "the follower is never flagged as crashed"
        );
    }

    #[test]
    fn red_running_is_rare_tunable_and_off_at_zero() {
        // The distraction draw is per vehicle–node: across many drivers it stays a
        // rare minority around the configured base rate, and a zero rate disables
        // it entirely (the sterile envelope for tests that need determinism).
        let mut w = NetWorld::new(straight_link(500.0), cfg());
        let node = w.network.link(LinkId(0)).to;
        let runners = |w: &NetWorld| {
            (0..20_000u32)
                .filter(|&id| {
                    let d = DriverConfig::car().sample(w.cfg.seed, id);
                    let probe = NetVehicle {
                        id, lane: LaneId(0), position: 0.0, speed: 0.0, driver: d,
                        route: Vec::new(), route_idx: 0, dest: None, stopped_at: None,
                        wait_ticks: 0, crossing: None, lane_change: None, wreck: None, slept: false,
                    };
                    w.runs_red(&probe, node)
                })
                .count()
        };
        let base = runners(&w);
        let expected = (20_000.0 * w.cfg.red_run_prob) as usize;
        assert!(base > 0, "some drivers do run reds at the default rate");
        assert!(base < expected * 3 + 20, "but they stay a rare minority: {base}");
        w.cfg.red_run_prob = 0.0;
        assert_eq!(runners(&w), 0, "a zero rate disables red-running");
    }

    #[test]
    fn body_overlap_is_exact_on_oriented_rectangles() {
        use std::f64::consts::{FRAC_PI_2, PI};
        let (len, wid) = (5.0, 2.0);
        // T-bone: B's front bumper at A's mid-body flank — genuinely intersecting.
        assert!(body_overlap([0.0, 0.0, 0.0], len, wid, [-2.5, -0.5, FRAC_PI_2], len, wid), "T-bone into the body collides");
        // Strike into the *rear half* of A (the case the old front-point distance missed):
        // B crossing at A's rear axle, 4 m behind A's front bumper.
        assert!(body_overlap([0.0, 0.0, 0.0], len, wid, [-4.0, -0.6, FRAC_PI_2], len, wid), "rear-half T-bone collides");
        // Anti-parallel bodies on offset lines (opposing lefts passing) do not touch.
        assert!(!body_overlap([0.0, 1.3, 0.0], len, wid, [-2.0, -1.3, PI], len, wid), "offset opposing pass is clean");
        // Side-by-side parallel bodies a lane apart do not touch.
        assert!(!body_overlap([0.0, 0.0, 0.0], len, wid, [0.5, 3.0, 0.0], len, wid), "parallel neighbours are clean");
        // A crossing pair still metres short of the meet point does not touch.
        assert!(!body_overlap([0.0, 0.0, 0.0], len, wid, [5.0, -5.0, FRAC_PI_2], len, wid), "distant crossing is clean");
        // The same pair with B's bumper protruding into A's flank does.
        assert!(body_overlap([0.0, 0.0, 0.0], len, wid, [-1.0, -0.5, FRAC_PI_2], len, wid), "converged crossing collides");
    }

    #[test]
    fn conflict_matrix_rules() {
        let (east, west, north) = ([1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]);
        // A minor-road right turn defers to higher-priority crossing traffic...
        assert!(should_yield_to(TurnType::Right, east, TurnType::Through, north, 0, 999));
        // ...but a major-road right turn does not yield to a minor crossing.
        assert!(!should_yield_to(TurnType::Right, east, TurnType::Through, north, 999, 0));
        // A right turn away from opposing traffic never yields to it.
        assert!(!should_yield_to(TurnType::Right, east, TurnType::Through, west, 0, 999));
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
                turn_lanes: String::new(),
                hov_lanes: String::new(),
                aadt: 0.0,
                res_weight: 0.0,
                attr_weight: 0.0,
                sign: crate::sim::network::LinkSign::None,
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

    #[cfg(feature = "import")]
    #[test]
    fn interior_speed_caps_follow_curvature() {
        // Turn speeds derive from each interior's tightest radius (v = √(a_lat·r)),
        // not one flat number per turn direction: a hooked turn crawls, a sweeping
        // one flows, and every cap stays inside the box-speed band.
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
        let Ok(text) = std::fs::read_to_string(path) else { return };
        let net = super::super::map::OsmMap::from_json(&text).expect("map json").build();
        let world = NetWorld::new(net, cfg());
        let mut by_radius: Vec<(f64, f64)> = (0..world.network.movements.len() as u32)
            .filter_map(|m| {
                let mid = MovementId(m);
                if world.network.is_interchange_movement(mid) {
                    return None;
                }
                let cap = world.turn_speed_cap(mid);
                let r = world.network.interior_min_radius(mid);
                (cap.is_finite() && world.network.movement_turn(mid) != TurnType::Through).then_some((r, cap))
            })
            .collect();
        assert!(by_radius.len() > 30, "enough capped turns to measure ({})", by_radius.len());
        for &(r, cap) in &by_radius {
            assert!((2.5..=10.0).contains(&cap), "cap {cap:.1} outside the box-speed band (r={r:.1})");
        }
        by_radius.sort_by(|a, b| a.0.total_cmp(&b.0));
        let tight = &by_radius[..by_radius.len() / 4];
        let sweep = &by_radius[by_radius.len() * 3 / 4..];
        let mean = |s: &[(f64, f64)]| s.iter().map(|&(_, c)| c).sum::<f64>() / s.len() as f64;
        assert!(
            mean(tight) + 1.0 < mean(sweep),
            "sharp turns are slower than sweeping ones: {:.1} vs {:.1}",
            mean(tight),
            mean(sweep)
        );
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
                    pairs.push(OdPair { origin: LinkId(o as u32), dest: LinkId(d as u32), rate_per_sec: 0.3, class: SurfaceClass::Through, anchored: false });
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
            let mut by_lane: HashMap<u32, Vec<(f64, f64, bool)>> = HashMap::new();
            for v in w.vehicles() {
                assert!(v.speed >= -1e-6, "no reversing: {}", v.speed);
                // Legal envelope: the speeding model runs aggressive drivers up to
                // limit × 1.2, and accel noise + reaction delay wander ~1 m/s past
                // the equilibrium; anything beyond that is a genuine runaway.
                assert!(v.speed <= max_limit * 1.2 + 1.5, "no gross speeding: {} > {}", v.speed, max_limit);
                if v.is_crossing() {
                    continue; // inside a node, not occupying the lane
                }
                by_lane.entry(v.lane.0).or_default().push((v.position, v.driver.vehicle_length, v.is_wrecked()));
            }
            for cars in by_lane.values_mut() {
                cars.sort_by(|a, b| a.0.total_cmp(&b.0));
                for w2 in cars.windows(2) {
                    let gap = w2[1].0 - w2[0].0 - w2[1].1;
                    // A wrecked pair legitimately overlaps — that collision was
                    // detected and tallied. The invariant hunts *silent* overlaps.
                    if w2[0].2 && w2[1].2 {
                        continue;
                    }
                    assert!(gap > -0.6, "vehicles overlap on a lane: gap {gap}");
                }
            }
        }
        assert!(
            w.crashed() * 40 <= w.exited(),
            "crashes stay rare over a busy run: {} crashed ({:?}) vs {} exited",
            w.crashed(),
            w.crash_counts(),
            w.exited()
        );
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
                            pairs.push(OdPair { origin: LinkId(o as u32), dest: LinkId(d as u32), rate_per_sec: 0.3, class: SurfaceClass::Through, anchored: false });
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

    #[test]
    fn keep_right_drifts_an_unobstructed_car_to_the_curb_lane() {
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::uncontrolled(2, 3000.0, 0.0),
            ],
            links: vec![LinkSpec::oneway(1, 2, 2, 25.0)],
        }
        .build();
        let mut world = NetWorld::new(net, cfg());
        let lanes: Vec<LaneId> = world.network.lanes_of(LinkId(0)).collect();
        world.spawn(1, lanes[0], 50.0, 20.0, DriverConfig { accel_noise: 0.0, ..DriverConfig::car() });
        assert_eq!(world.network.lane(world.vehicle(1).unwrap().lane).index_in_link, 0, "starts in the median lane");
        world.run_ticks(200);
        let end = world.vehicle(1).map(|v| world.network.lane(v.lane).index_in_link);
        assert_eq!(end, Some(1), "with no obstruction, keep-right moves the car to the curb lane");
    }

    #[test]
    fn a_surface_car_weaves_across_lanes_to_reach_its_turn_pocket() {
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::uncontrolled(2, 400.0, 0.0),
                NodeSpec::uncontrolled(3, 800.0, 0.0),
                NodeSpec::uncontrolled(4, 400.0, 300.0),
            ],
            links: vec![
                LinkSpec::oneway(1, 2, 3, 15.0), // 3-lane surface approach
                LinkSpec::oneway(2, 3, 2, 15.0), // straight exit
                LinkSpec::oneway(2, 4, 1, 15.0), // left-turn exit (channelised to lane 0)
            ],
        }
        .build();
        let mut world = NetWorld::new(net, cfg());
        world.install_router(&[LinkId(2)]);
        let right_lane = LaneId(world.network.link(LinkId(0)).lane_start.0 + 2);
        assert_eq!(world.network.lane(right_lane).index_in_link, 2, "spawns in the far right lane");
        world.spawn_to_in_lane(1, right_lane, 10.0, LinkId(2), 10.0, DriverConfig { accel_noise: 0.0, ..DriverConfig::car() });

        let mut min_idx = i64::MAX;
        for _ in 0..500 {
            world.step();
            if let Some(v) = world.vehicle(1) {
                if !v.is_crossing() && world.network.lane(v.lane).link == LinkId(0) {
                    min_idx = min_idx.min(world.network.lane(v.lane).index_in_link as i64);
                }
            }
        }
        assert_eq!(world.crashed(), 0, "weaving stays collision-free");
        assert_eq!(world.exited(), 1, "the car completes its left turn");
        assert_eq!(min_idx, 0, "it weaved across all lanes into the left-turn pocket");
    }

    #[test]
    fn queue_discharge_hits_real_saturation_flow() {
        // HCM ground truth at a signalized stop line: after start-up lost time
        // (~2 s), a standing queue discharges at a saturation headway of roughly
        // 1.9 s/veh (~1900 veh/h/lane). This is the supply-side number every
        // capacity comparison rests on — if discharge is far off, observed
        // counts can never be matched no matter what demand does.
        // Single-lane approach so the whole queue discharges through one stop
        // line — a multi-lane fixture splits the queue across channelized lanes
        // and measures lane-change noise instead of saturation.
        let plan = SignalPlan { green_secs: 25.0, yellow_secs: 3.0, offset: 0.0 };
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::signalized(2, 300.0, 0.0, plan),
                NodeSpec::uncontrolled(4, 600.0, 0.0),
                NodeSpec::uncontrolled(3, 300.0, -200.0),
                NodeSpec::uncontrolled(5, 300.0, 200.0),
            ],
            links: vec![
                LinkSpec::oneway(1, 2, 1, 15.0),
                LinkSpec::oneway(2, 4, 1, 15.0),
                LinkSpec::oneway(3, 2, 1, 15.0),
                LinkSpec::oneway(2, 5, 1, 15.0),
            ],
        }
        .build();
        let mut world = NetWorld::new(net, cfg());
        world.install_router(&[LinkId(1), LinkId(3)]);
        let d = || DriverConfig { accel_noise: 0.0, ..DriverConfig::car() };
        // A standing queue at the stop line, launched by the (undisputed, hence
        // resting-green) signal: the discharge wave itself is the saturation
        // process, no red-green cycling needed.
        let lane0 = world.network.link(LinkId(0)).lane_start;
        let lane_len = world.network.lane(lane0).length;
        let n_q = 12u32;
        for k in 0..n_q {
            let pos = lane_len - 2.0 - k as f64 * 7.0;
            assert!(
                world.spawn_routed_in_lane(k, vec![LinkId(0), LinkId(1)], lane0, pos, 0.0, d()),
                "queue car {k} placed"
            );
        }
        let mut cross: Vec<Option<(u32, f64)>> = vec![None; n_q as usize];
        let mut last_speed: Vec<f64> = vec![0.0; n_q as usize];
        for t in 0..1200u32 {
            world.step();
            for k in 0..n_q {
                let ku = k as usize;
                if cross[ku].is_some() {
                    continue;
                }
                match world.vehicle(k) {
                    Some(v) if v.is_crossing() || world.network.lane(v.lane).link != LinkId(0) => {
                        cross[ku] = Some((t, last_speed[ku]));
                    }
                    Some(v) => last_speed[ku] = v.speed,
                    None => cross[ku] = Some((t, last_speed[ku])),
                }
            }
        }
        let times: Vec<(f64, f64)> = cross.iter().flatten().map(|&(t, sp)| (t as f64 * 0.2, sp)).collect();
        assert!(times.len() >= 10, "the queue discharged ({} of {n_q})", times.len());
        for &(t, sp) in &times {
            eprintln!("  crossed t={t:.1} at {sp:.1} m/s");
        }
        // Saturation headways: consecutive stop-line crossings, skipping the
        // first two cars (start-up lost time by definition).
        let heads: Vec<f64> = times.windows(2).map(|w| w[1].0 - w[0].0).skip(2).collect();
        assert!(heads.len() >= 6, "enough saturation headways to measure ({})", heads.len());
        let sat = heads.iter().sum::<f64>() / heads.len() as f64;
        eprintln!("saturation headway {sat:.2}s ({:.0} veh/h/lane)", 3600.0 / sat);
        // Regression floor at the calibrated point (~3.05 s, ≈1180 veh/h/lane at
        // max_accel 2.0). The real-world target is ~1.9 s (~1900): the residual
        // sits in the time-gap stack (IDM headway + perception-reaction applying
        // in full during discharge) — the honest open item, not a tolerance to
        // widen. Tightening this band is the goal of any future launch-model work.
        assert!(
            sat <= 3.3,
            "saturation headway regressed to {sat:.2}s — stop-line discharge got slower"
        );
        assert!(sat >= 1.6, "faster than physical saturation ({sat:.2}s) — check the measurement");
    }

    #[test]
    fn a_multi_lane_fix_starts_one_window_per_lane_early() {
        // Two lanes from the turn pocket, the positioning window doubles: the car
        // must already be in its serving lane well before the last single window
        // — not weaving everything into the final forty metres.
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::uncontrolled(2, 600.0, 0.0),
                NodeSpec::uncontrolled(3, 1200.0, 0.0),
                NodeSpec::uncontrolled(4, 600.0, 300.0),
            ],
            links: vec![
                LinkSpec::oneway(1, 2, 3, 15.0), // 3-lane surface approach
                LinkSpec::oneway(2, 3, 2, 15.0), // straight exit
                LinkSpec::oneway(2, 4, 1, 15.0), // left-turn exit (channelised to lane 0)
            ],
        }
        .build();
        let mut world = NetWorld::new(net, cfg());
        world.install_router(&[LinkId(2)]);
        let right_lane = LaneId(world.network.link(LinkId(0)).lane_start.0 + 2);
        world.spawn_to_in_lane(1, right_lane, 10.0, LinkId(2), 15.0, DriverConfig { accel_noise: 0.0, ..DriverConfig::car() });
        let mut serving_at = None;
        for _ in 0..600 {
            world.step();
            if let Some(v) = world.vehicle(1) {
                if serving_at.is_none()
                    && !v.is_crossing()
                    && world.network.lane(v.lane).link == LinkId(0)
                    && world.network.lane(v.lane).index_in_link == 0
                {
                    serving_at = Some(world.network.lane(v.lane).length - v.position);
                }
            }
        }
        let to_node = serving_at.expect("the car reached its turn lane");
        assert_eq!(world.exited(), 1, "the car completes its left turn");
        assert!(
            to_node > 150.0,
            "two lanes out, positioning starts a window early: reached the pocket only {to_node:.0} m before the node"
        );
    }

    #[test]
    fn lane_choice_prepositions_for_the_turn_after_next() {
        // The route turns left off link B, whose turn lane is only fed by A's
        // lane 0. Judging lanes by "reaches B" alone, A's lane 1 looks fine and
        // the car lands on B owing a forced weave; judging one hop deeper, it
        // moves over while still on A.
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::uncontrolled(2, 400.0, 0.0),
                NodeSpec::uncontrolled(3, 440.0, 0.0),
                NodeSpec::uncontrolled(4, 440.0, 300.0),  // left exit off B
                NodeSpec::uncontrolled(5, 740.0, 0.0),    // straight exit off B
            ],
            links: vec![
                LinkSpec::oneway(1, 2, 2, 15.0), // A
                LinkSpec::oneway(2, 3, 2, 15.0), // B: a short block — too short to weave on
                LinkSpec::oneway(3, 4, 1, 15.0), // C: left, fed by B lane 0
                LinkSpec::oneway(3, 5, 1, 15.0), // D: straight, fed by B lane 1
            ],
        }
        .build();
        let mut world = NetWorld::new(net, cfg());
        let curb = LaneId(world.network.link(LinkId(0)).lane_start.0 + 1);
        world.spawn_routed_in_lane(1, vec![LinkId(0), LinkId(1), LinkId(2)], curb, 50.0, 12.0, DriverConfig { accel_noise: 0.0, ..DriverConfig::car() });
        let mut lane_at_a_end = None;
        for _ in 0..600 {
            world.step();
            if let Some(v) = world.vehicle(1) {
                if !v.is_crossing() && world.network.lane(v.lane).link == LinkId(0) {
                    lane_at_a_end = Some(world.network.lane(v.lane).index_in_link);
                }
            }
        }
        assert_eq!(world.exited(), 1, "the car completes the route");
        assert_eq!(lane_at_a_end, Some(0), "it pre-positioned on A for the turn off B");
    }

    #[test]
    fn lane_choice_prepositions_across_two_short_blocks() {
        // The turn is off link C, two *short* blocks ahead; only A offers room
        // to change. Depth-2 preference sees both A lanes reaching C and does
        // nothing; depth-3 sees that only A lane 0's landing chain feeds C's
        // turn lane and moves over while there is still road to do it.
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::uncontrolled(2, 400.0, 0.0),
                NodeSpec::uncontrolled(3, 440.0, 0.0),
                NodeSpec::uncontrolled(4, 480.0, 0.0),
                NodeSpec::uncontrolled(5, 480.0, 300.0),  // left exit off C
                NodeSpec::uncontrolled(6, 780.0, 0.0),    // straight exit off C
            ],
            links: vec![
                LinkSpec::oneway(1, 2, 2, 15.0), // A
                LinkSpec::oneway(2, 3, 2, 15.0), // B: short block
                LinkSpec::oneway(3, 4, 2, 15.0), // C: short block
                LinkSpec::oneway(4, 5, 1, 15.0), // D: left, fed by C lane 0
                LinkSpec::oneway(4, 6, 1, 15.0), // E: straight, fed by C lane 1
            ],
        }
        .build();
        let mut world = NetWorld::new(net, cfg());
        let curb = LaneId(world.network.link(LinkId(0)).lane_start.0 + 1);
        world.spawn_routed_in_lane(
            1,
            vec![LinkId(0), LinkId(1), LinkId(2), LinkId(3)],
            curb,
            50.0,
            12.0,
            DriverConfig { accel_noise: 0.0, ..DriverConfig::car() },
        );
        let mut lane_at_a_end = None;
        for _ in 0..700 {
            world.step();
            if let Some(v) = world.vehicle(1) {
                if !v.is_crossing() && world.network.lane(v.lane).link == LinkId(0) {
                    lane_at_a_end = Some(world.network.lane(v.lane).index_in_link);
                }
            }
        }
        assert_eq!(world.exited(), 1, "the car completes the route");
        assert_eq!(lane_at_a_end, Some(0), "it pre-positioned on A for the turn two blocks out");
    }

    #[test]
    fn a_lane_change_slides_the_pose_across_gradually() {
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::uncontrolled(2, 3000.0, 0.0),
            ],
            links: vec![LinkSpec::oneway(1, 2, 2, 25.0)],
        }
        .build();
        let mut world = NetWorld::new(net, cfg());
        let lanes: Vec<LaneId> = world.network.lanes_of(LinkId(0)).collect();
        world.spawn(1, lanes[0], 50.0, 20.0, DriverConfig { accel_noise: 0.0, ..DriverConfig::car() });
        let lane_gap = (world.network.lane_point(lanes[0], 100.0)[1] - world.network.lane_point(lanes[1], 100.0)[1]).abs();

        let (mut changed_at, mut settled) = (None, false);
        let mut prev_y = world.vehicle_world_pose(world.vehicle(1).unwrap())[1];
        let mut max_jump = 0.0f64;
        let settle_ticks = (LANE_CHANGE_DURATION / cfg().dt) as u32 + 2;
        for t in 0..80 {
            world.step();
            let v = world.vehicle(1).unwrap();
            let y = world.vehicle_world_pose(v)[1];
            max_jump = max_jump.max((y - prev_y).abs());
            prev_y = y;
            let centerline = world.network.lane_point(v.lane, v.position)[1];
            if changed_at.is_none() && world.network.lane(v.lane).index_in_link == 1 {
                changed_at = Some(t);
                assert!((y - centerline).abs() > 0.3, "pose does not teleport to the new lane: y={y} centerline={centerline}");
            }
            if !settled && changed_at.is_some_and(|c| t >= c + settle_ticks) {
                assert!((y - centerline).abs() < 0.05, "pose settles on the new lane after the slide: y={y} centerline={centerline}");
                settled = true;
            }
        }
        assert!(changed_at.is_some() && settled, "the car changed lanes and the transition completed");
        assert!(max_jump < lane_gap * 0.6, "the lateral slide is gradual, not a teleport: max_jump={max_jump} lane_gap={lane_gap}");
    }

    #[test]
    fn a_seam_lane_remap_slides_the_pose_across_gradually() {
        // A motorway lane-add boundary maps the curb lane onto a non-adjacent index
        // (lane 1 of 2 → lane 2 of 3). The car must cross the seam pointing down
        // the road and ease onto its new line at the lane-change rate — never snap
        // a lane width sideways or swing past 90° (the wrong-way lurch).
        let hw = |a, b, lanes| LinkSpec { road_class: "motorway".into(), ..LinkSpec::oneway(a, b, lanes, 25.0) };
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::uncontrolled(2, 700.0, 0.0),
                NodeSpec::uncontrolled(3, 1400.0, 0.0),
            ],
            links: vec![hw(1, 2, 2), hw(2, 3, 3)],
        }
        .build();
        let mut world = NetWorld::new(net, cfg());
        let curb = LaneId(world.network.link(LinkId(0)).lane_start.0 + 1);
        world.spawn(1, curb, 600.0, 25.0, DriverConfig { accel_noise: 0.0, ..DriverConfig::car() });
        let mut prev_y = world.vehicle_world_pose(world.vehicle(1).unwrap())[1];
        let (mut max_jump, mut crossed, mut settled) = (0.0f64, false, false);
        for _ in 0..200 {
            world.step();
            let Some(v) = world.vehicle(1) else { break };
            let p = world.vehicle_world_pose(v);
            assert!(p[2].abs() < 0.35, "heading stays down the road through the seam, got {:.1}°", p[2].to_degrees());
            max_jump = max_jump.max((p[1] - prev_y).abs());
            prev_y = p[1];
            if !v.is_crossing() && world.network.lane(v.lane).link == LinkId(1) {
                crossed = true;
                let centerline = world.network.lane_point(v.lane, v.position)[1];
                if v.lane_change.is_none() && (p[1] - centerline).abs() < 0.05 {
                    settled = true;
                }
            }
        }
        assert!(crossed, "the car crossed the seam onto the wide segment");
        assert!(settled, "the pose settled onto the remapped lane's own line");
        assert!(max_jump < 1.0, "lateral motion is a gradual ease, not a snap: max_jump={max_jump}");
    }

    #[test]
    fn the_mainline_flows_ungated_through_a_merge_but_the_ramp_yields() {
        // At an on-ramp merge, the freeway through movements cross the seam with no
        // admission gate (a freeway never brake-checks at a segment boundary) while
        // the ramp — not the exit link's through approach — still yields.
        let hw = |a, b, lanes| LinkSpec { road_class: "motorway".into(), ..LinkSpec::oneway(a, b, lanes, 29.0) };
        let ramp = |a, b, lanes| LinkSpec { road_class: "motorway_link".into(), ..LinkSpec::oneway(a, b, lanes, 25.0) };
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -400.0, 0.0),
                NodeSpec::uncontrolled(2, 0.0, 0.0),
                NodeSpec::uncontrolled(3, 400.0, 0.0),
                NodeSpec::uncontrolled(4, -200.0, -200.0),
            ],
            links: vec![hw(1, 2, 3), hw(2, 3, 3), ramp(4, 2, 1)],
        }
        .build();
        let world = NetWorld::new(net, cfg());
        let mut checked = 0;
        for m in 0..world.network.movements.len() as u32 {
            let mid = MovementId(m);
            let mv = world.network.movement(mid);
            if world.network.lane(mv.to_lane).link != LinkId(1) {
                continue;
            }
            checked += 1;
            if world.network.lane(mv.from_lane).link == LinkId(0) {
                assert!(world.free_flow_seam(mid), "a mainline continuation crosses ungated");
            } else {
                assert!(!world.free_flow_seam(mid), "the merging ramp keeps its admission gate");
            }
        }
        assert!(checked >= 4, "the merge fixture wires mainline and ramp movements, got {checked}");
    }
}
