//! Origin–destination travel demand: the "who goes where, when" that carries
//! most of the perceived realism at city scale. Boundary-aware categories
//! (through / inbound / outbound / internal) place origins and destinations at
//! the map's gateways and interior, and each spawn is a Bernoulli draw on the
//! stateless RNG. Vehicles are handed a *destination* and routed live by the
//! world's flow field (rerouting around jams), falling back to a precomputed
//! route when the world has no router for that destination.

use std::cell::RefCell;
use std::rc::Rc;

use super::boundary;
use super::config::VehicleClass;
use super::hash::IntMap;
use super::net_world::{NetWorld, ScheduledStop};
use super::network::{LinkId, Network, RoadKind};
use super::rng::{self, Stream};
use super::rush_hour::{self, SurfaceClass};

/// Memoized forward reachability over the link graph. Demand generation asks "can this
/// origin reach this destination?" thousands of times while sampling OD pairs; answering
/// each with a fresh `route_links` (a Dijkstra over `std::HashMap` that, for an *unreachable*
/// pair, scans the whole component) dominated city-map load — ~66 s of the Columbus hang.
/// One BFS per distinct origin (dense `Vec<bool>`, O(links)), cached; each check is then O(1).
/// Membership only, so the result is order-independent (no reproducibility concern).
struct Reachability<'a> {
    net: &'a Network,
    from: RefCell<IntMap<Rc<Vec<bool>>>>,
}

impl<'a> Reachability<'a> {
    fn new(net: &'a Network) -> Self {
        Self { net, from: RefCell::new(IntMap::default()) }
    }

    fn reachable(&self, from: LinkId, to: LinkId) -> bool {
        if from == to {
            return true;
        }
        if let Some(set) = self.from.borrow().get(&from.0) {
            return set[to.idx()];
        }
        let mut seen = vec![false; self.net.links.len()];
        let mut stack = vec![from.0];
        seen[from.idx()] = true;
        while let Some(l) = stack.pop() {
            for n in self.net.outgoing_links(LinkId(l)) {
                if !seen[n.idx()] {
                    seen[n.idx()] = true;
                    stack.push(n.0);
                }
            }
        }
        let hit = seen[to.idx()];
        self.from.borrow_mut().insert(from.0, Rc::new(seen));
        hit
    }
}

#[derive(Debug)]
pub struct OdPair {
    pub origin: LinkId,
    pub dest: LinkId,
    pub rate_per_sec: f64,
    /// Boundary geometry of the trip (through/inbound/outbound/internal) — picks
    /// which diurnal shape the stream breathes on under rush hour.
    pub class: SurfaceClass,
    /// Pinned to a measured location (a real LODES commute flow): origin churn
    /// must not move it, and its rate is absolute demand, not a per-origin
    /// capacity to split.
    pub anchored: bool,
}

/// Real commute origin–destination flows (Census LEHD LODES, via
/// `tools/lodes/fetch_lodes.py`) aggregated onto a metre grid in the map's
/// frame: `cells` are grid centres and `flows` are
/// `(home cell, work cell, commuters/day)` triples wholly inside the box.
pub struct CommuteOd {
    pub grid_m: f64,
    pub cells: Vec<[f64; 2]>,
    pub flows: Vec<(u32, u32, f64)>,
}

#[cfg(feature = "import")]
impl CommuteOd {
    /// Parse `fetch_lodes.py` output, dropping flows with out-of-range cell
    /// indices or non-positive volumes.
    pub fn from_json(s: &str) -> Result<CommuteOd, String> {
        use serde::Deserialize;
        #[derive(Deserialize)]
        struct Meta {
            grid_m: f64,
        }
        #[derive(Deserialize)]
        struct Doc {
            meta: Meta,
            cells: Vec<[f64; 2]>,
            flows: Vec<(u32, u32, f64)>,
        }
        let doc: Doc = serde_json::from_str(s).map_err(|e| e.to_string())?;
        let n = doc.cells.len() as u32;
        let flows = doc
            .flows
            .into_iter()
            .filter(|&(h, w, jobs)| h < n && w < n && jobs > 0.0)
            .collect();
        Ok(CommuteOd { grid_m: doc.meta.grid_m.max(1.0), cells: doc.cells, flows })
    }
}

/// Which streams of traffic to spawn — the independent toggles the UI exposes, so
/// they compose. Freeway traffic enters at highway gateways bound for the far end of
/// its highway (or another highway exit, or a surface street it leaves the freeway
/// for); surface traffic is the local/arterial boundary mix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DemandSources {
    pub highway: bool,
    pub surface: bool,
    /// Drive the whole map by the simulated time of day: freeway gateways at their
    /// real per-lane PeMS volumes and surface streets by the arterial diurnal shape,
    /// so the network builds and fades the way a real peak commute does.
    pub rush_hour: bool,
    /// How the topology-grounded OD generation is shaped (road-function weighting,
    /// distance decay, on-ramp/corridor shares). Rides in `DemandSources` so the
    /// public `od_pairs*` entry points carry it without a signature change.
    pub tuning: DemandTuning,
}

impl DemandSources {
    pub const fn new(highway: bool, surface: bool) -> Self {
        Self { highway, surface, rush_hour: false, tuning: DemandTuning::DEFAULT }
    }

    pub const fn with_rush_hour(highway: bool, surface: bool, rush_hour: bool) -> Self {
        Self { highway, surface, rush_hour, tuning: DemandTuning::DEFAULT }
    }
}

/// Tunable knobs for the topology-grounded OD generation — the UI-exposed demand
/// levers. Defaults reproduce the shipped calibration exactly; each field is a
/// single meaningful lever (the per-`RoadKind` class coefficients stay calibrated
/// constants, gated on/off by `road_function_weighting`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DemandTuning {
    /// L1: weight origins/destinations by road function (Local produces, Arterial is
    /// a through-corridor). When false the class terms are neutral 1.0 (the pre-lever
    /// uniform-origin / capacity-only-attraction behavior).
    pub road_function_weighting: bool,
    /// L3: distance-decay β for commute/through trips (lower = longer trips).
    pub gravity_beta: f64,
    /// L3: distance-decay β for internal/local errands (higher = they stay local).
    pub internal_beta: f64,
    /// L2: share of freeway demand that loads at interior on-ramps (vs edge gateways).
    pub on_ramp_share: f64,
    /// L4: share of a counted corridor's inflow that rides it end to end (the rest is
    /// mid-corridor egress/access at cross-streets).
    pub corridor_through_share: f64,
    /// L4: per-cross-street mid-corridor access share.
    pub corridor_access_share: f64,
}

impl DemandTuning {
    pub const DEFAULT: Self = Self {
        road_function_weighting: true,
        gravity_beta: GRAVITY_BETA,
        internal_beta: 1.8,
        on_ramp_share: 0.25,
        corridor_through_share: 0.6,
        corridor_access_share: 0.15,
    };
}

impl Default for DemandTuning {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// OD demand for the enabled sources — the union of the freeway and surface streams,
/// with `target` trips split across whichever are on. Falls back to the plain
/// boundary mix only if the enabled sources yield nothing (e.g. no gateways at all).
pub fn od_pairs(net: &Network, seed: u64, target: usize, sources: DemandSources) -> Vec<OdPair> {
    od_pairs_with_commute(net, seed, target, sources, None)
}

/// [`od_pairs`] with real commute data: when LODES flows are loaded, measured
/// home→work flows (each an AM-shaped stream plus its PM reverse) join the
/// sampled streams, and the sampled *surface* rates are scaled down by the
/// measured volume's share — the work trips the categories were approximating.
///
/// Measured volume displaces sampled **volume, keeping coverage**: every sampled
/// stream survives at a scaled rate, so the city stays evenly seeded whatever
/// the measured share — a small box whose residents mostly work outside it
/// (Millbrae: ~450 wholly-in-bbox trips/day) barely dents the sampled fabric,
/// while a metro box (Columbus: ~570k) dominates it. The scale is floored so
/// sampled non-work trips persist (LODES covers jobs only).
pub fn od_pairs_with_commute(
    net: &Network,
    seed: u64,
    target: usize,
    sources: DemandSources,
    commute: Option<&CommuteOd>,
) -> Vec<OdPair> {
    let n = (sources.highway as usize) + (sources.surface as usize);
    if n == 0 {
        return Vec::new();
    }
    let per = (target / n).max(1);
    let with_commute = sources.surface && commute.is_some_and(|od| !od.flows.is_empty());
    let mut pairs = Vec::new();
    if sources.highway {
        highway_od_pairs(net, seed, per, sources.tuning, &mut pairs);
    }
    let surface_start = pairs.len();
    if sources.surface {
        surface_od_pairs(net, seed, per, sources.tuning, &mut pairs);
    }
    if pairs.is_empty() && !with_commute {
        pairs = boundary_od_pairs(net, seed, target);
    }
    // Capacity-derived rates are split per origin; commute rates are absolute
    // measured demand, so they join only after calibration.
    calibrate_origin_inflow(&mut pairs);
    if with_commute {
        let sampled: f64 = pairs[surface_start..].iter().map(|p| p.rate_per_sec).sum();
        let n0 = pairs.len();
        commute_od_pairs(net, commute.unwrap(), seed, per / 2, sources.tuning, &mut pairs);
        let measured: f64 = pairs[n0..].iter().map(|p| p.rate_per_sec).sum();
        if measured > 0.0 && sampled > 0.0 {
            let keep = (1.0 - measured / sampled).max(0.3);
            for p in &mut pairs[surface_start..n0] {
                // Counted-corridor streams are measured volume themselves — the
                // LODES displacement applies only to the sampled approximations.
                if !p.anchored {
                    p.rate_per_sec *= keep;
                }
            }
        }
    }
    pairs
}

/// Nearest surface interior links per commute grid cell — where a cell's trips
/// enter and leave the road network. Each cell keeps its few closest candidates
/// (not just one): a trip and its reverse often need *different* links — the
/// opposite carriageway of the same street — since routing permits no U-turns.
/// Links are bucketed by grid cell so each lookup scans only the 3×3
/// neighbourhood; a cell with nothing within 1.5 grid spacings stays unanchored
/// (its flows are skipped).
struct CellAnchors {
    anchors: Vec<Vec<LinkId>>,
}

/// Candidate anchor links kept per cell. A few more than the minimum so commute
/// trips spread across a cell's streets (class-weighted in `routable_pair`) instead
/// of all piling onto the single nearest link.
const ANCHOR_CANDIDATES: usize = 8;

impl CellAnchors {
    fn new(net: &Network, od: &CommuteOd) -> Self {
        let cs = od.grid_m;
        let mut buckets: std::collections::HashMap<(i64, i64), Vec<(LinkId, [f64; 2])>> =
            std::collections::HashMap::new();
        for l in boundary::surface_interior_links(net) {
            let c = link_centroid(net, l);
            buckets.entry(((c[0] / cs).floor() as i64, (c[1] / cs).floor() as i64)).or_default().push((l, c));
        }
        let anchors = od
            .cells
            .iter()
            .map(|cell| {
                let (ci, cj) = ((cell[0] / cs).floor() as i64, (cell[1] / cs).floor() as i64);
                let mut near: Vec<(f64, u32)> = Vec::new();
                for di in -1..=1 {
                    for dj in -1..=1 {
                        for &(l, c) in buckets.get(&(ci + di, cj + dj)).into_iter().flatten() {
                            let d = (c[0] - cell[0]).hypot(c[1] - cell[1]);
                            if d <= 1.5 * cs {
                                near.push((d, l.0));
                            }
                        }
                    }
                }
                near.sort_by(|a, b| a.partial_cmp(b).unwrap());
                near.truncate(ANCHOR_CANDIDATES);
                near.into_iter().map(|(_, l)| LinkId(l)).collect()
            })
            .collect();
        Self { anchors }
    }

    fn candidates(&self, cell: u32) -> &[LinkId] {
        self.anchors.get(cell as usize).map_or(&[], Vec::as_slice)
    }

    /// A routable candidate pair `from → to`, chosen (seeded) by road function so a
    /// commute leg starts on the origin cell's local/collector streets (production)
    /// and ends on the destination cell's endpoint-suitable streets (attraction) —
    /// and spreads across the cell's candidates instead of always taking the single
    /// nearest link. The origin/dest roles hold for both legs: the AM leg's origin
    /// is home, the PM leg's origin is work; each leg's start is production-weighted
    /// and its end attraction-weighted. Falls back to the first routable pair if the
    /// class weights sum to zero.
    fn routable_pair(&self, net: &Network, reach: &Reachability, from: u32, to: u32, seed: u64, salt: u32, tune: DemandTuning) -> Option<(LinkId, LinkId)> {
        let mut cum: Vec<f64> = Vec::new();
        let mut pairs: Vec<(LinkId, LinkId)> = Vec::new();
        let mut total = 0.0;
        for &f in self.candidates(from) {
            for &t in self.candidates(to) {
                if f != t && reach.reachable(f, t) {
                    let w = class_production(net.link(f).kind, tune) * class_attraction(net.link(t).kind, tune);
                    total += w;
                    cum.push(total);
                    pairs.push((f, t));
                }
            }
        }
        if pairs.is_empty() {
            return None;
        }
        if total <= 0.0 {
            return Some(pairs[0]);
        }
        let r = rng::uniform01(seed, salt, from as u64, Stream::RouteChoice) * total;
        Some(pairs[cum.partition_point(|&c| c < r).min(pairs.len() - 1)])
    }
}

/// OD pairs from real commute flows: a jobs-weighted sample of home→work flows,
/// each becoming a morning-shaped stream (home link → work link, [`SurfaceClass::Inbound`]
/// — the AM arrival surge) and its evening reverse ([`SurfaceClass::Outbound`]).
/// Rates size the sampled set to carry the box's real total commuter volume:
/// Σ jobs trips per day each way, so the measured magnitude survives sampling.
pub fn commute_od_pairs(net: &Network, od: &CommuteOd, seed: u64, target: usize, t: DemandTuning, out: &mut Vec<OdPair>) {
    if od.flows.is_empty() || target == 0 {
        return;
    }
    let anchors = CellAnchors::new(net, od);
    let reach = Reachability::new(net);
    let mut cum = Vec::with_capacity(od.flows.len());
    let mut total_jobs = 0.0;
    for &(_, _, jobs) in &od.flows {
        total_jobs += jobs;
        cum.push(total_jobs);
    }
    let want = (target / 2).max(1);
    let mut ends = Vec::with_capacity(want);
    let mut attempt = 0u64;
    while ends.len() < want && attempt < want as u64 * 40 + 200 {
        let r = rng::uniform01(seed, 70, attempt, Stream::RouteChoice) * total_jobs;
        let (h, w, _) = od.flows[cum.partition_point(|&c| c < r).min(od.flows.len() - 1)];
        attempt += 1;
        // Anchor the trip and its reverse independently — the return leg may need
        // the opposite carriageway. Each leg is class-weighted (production origin,
        // attraction dest) and seeded off the attempt so trips spread across a
        // cell's streets.
        let am = anchors.routable_pair(net, &reach, h, w, seed, 71 ^ (attempt as u32).wrapping_mul(2), t);
        let pm = anchors.routable_pair(net, &reach, w, h, seed, 72 ^ (attempt as u32).wrapping_mul(2).wrapping_add(1), t);
        if let (Some(am), Some(pm)) = (am, pm) {
            ends.push((am, pm));
        }
    }
    if ends.is_empty() {
        return;
    }
    let rate = (total_jobs / 86_400.0 / ends.len() as f64).min(0.9);
    for ((home, work), (work_rev, home_rev)) in ends {
        out.push(OdPair { origin: home, dest: work, rate_per_sec: rate, class: SurfaceClass::Inbound, anchored: true });
        out.push(OdPair { origin: work_rev, dest: home_rev, rate_per_sec: rate, class: SurfaceClass::Outbound, anchored: true });
    }
}

/// Hold each origin's *aggregate* off-peak inflow to one `capacity_rate` by
/// splitting it across the pairs sharing that origin (each already carries the full
/// rate). Without this, a gateway that happens to seed many OD pairs would inject
/// `pairs × rate` — harmless while injection stacked everything on one lane and the
/// entrance throttled it, but a flood once inflow fills all lanes. Mirrors the
/// rush-hour path, which already divides a gateway's volume by its pair `share`.
///
/// Anchored pairs (a measured corridor's through stream) keep their absolute
/// rate; the sampled pairs share what remains of the origin's capacity.
fn calibrate_origin_inflow(pairs: &mut [OdPair]) {
    use std::collections::HashMap;
    let mut anchored_sum: HashMap<LinkId, f64> = HashMap::new();
    let mut unanchored: HashMap<LinkId, usize> = HashMap::new();
    for p in pairs.iter() {
        if p.anchored {
            *anchored_sum.entry(p.origin).or_insert(0.0) += p.rate_per_sec;
        } else {
            *unanchored.entry(p.origin).or_insert(0) += 1;
        }
    }
    for p in pairs.iter_mut() {
        if p.anchored {
            continue;
        }
        let cap = p.rate_per_sec;
        let remaining = (cap - anchored_sum.get(&p.origin).copied().unwrap_or(0.0)).max(cap * 0.15);
        p.rate_per_sec = remaining / unanchored[&p.origin] as f64;
    }
}

/// Hour of day (0–24) the rush-hour clock starts at — mid pre-peak build-up, so the
/// morning ramp is imminent when the mode is switched on.
const RUSH_START_HOUR: f64 = 5.5;
/// Default simulated day-seconds elapsed per second of sim time: the 24 h profile
/// plays over ~24 min of sim time, fast enough to watch the peak build and fade while
/// vehicles still have real time to form and clear the queues it creates. Runtime
/// range is [`MIN_DAY_COMPRESSION`]–[`MAX_DAY_COMPRESSION`]; 1.0 is real time, the
/// accuracy mode validation runs use. Only day-clock quantities (diurnal rates, the
/// weekend/day counter) scale with it — traffic dynamics, modulation epochs, churn,
/// and wreck clearance stay in real sim seconds.
pub const DEFAULT_DAY_COMPRESSION: f64 = 60.0;
pub const MIN_DAY_COMPRESSION: f64 = 1.0;
pub const MAX_DAY_COMPRESSION: f64 = 240.0;

/// One origin→destination stream and how fast it spawns. Off-peak it fires at a
/// fixed `base_rate`; under rush hour `rush` makes it follow the time of day.
struct OdStream {
    origin: LinkId,
    dest: LinkId,
    /// Generic (off-peak) spawn rate, veh/sec.
    base_rate: f64,
    /// Set when rush hour is on: how this stream tracks the simulated time of day.
    rush: Option<RushMode>,
    /// Boundary geometry of the trip — selects the diurnal shape under rush hour.
    class: SurfaceClass,
    /// Whether the origin is an ordinary street (not a freeway): surface streams
    /// platoon, churn, and carry the surface vehicle mix.
    surface: bool,
    /// Pinned to a measured location (a LODES commute flow) — excluded from
    /// origin churn so real geography stays put.
    anchored: bool,
}

/// How a stream's spawn rate varies through the day under rush hour.
#[derive(Clone, Copy)]
enum RushMode {
    /// Freeway gateway: its lane count × the route's real per-lane PeMS hourly volume,
    /// split across the gateway's pairs so the aggregate stays calibrated to real data.
    Freeway(RushRate),
    /// Surface street: the calibrated `base_rate` scaled by its class's diurnal
    /// multiplier (see [`rush_hour::surface_factor`]) so inbound, outbound, internal,
    /// and through streams each breathe on their own commute shape.
    Surface,
}

#[derive(Clone, Copy)]
struct RushRate {
    lanes: f64,
    profile: &'static [u16; 24],
    /// This stream's fraction (1 / pairs-sharing-origin) of its gateway's inflow.
    share: f64,
    /// Calibration to the gateway's *observed* AADT when counts are embedded in
    /// the map: the PeMS curves are regional per-lane averages, which overfeed a
    /// quiet corridor and starve a busy one (the audit measured I-280 at ~2× and
    /// US-101 at ~¼ of their local volumes). Scales the curve so its daily total
    /// matches the link's directional AADT share; 1.0 without counts.
    scale: f64,
}

/// One real (GTFS-compiled) bus departure on a line: when it leaves and the
/// per-stop schedule the world holds it to at timepoints.
#[derive(Clone, Debug)]
pub struct BusTrip {
    pub weekend: bool,
    /// First-stop departure, day-seconds (normalized into `[0, 86400)`).
    pub departure: f64,
    pub stops: Vec<ScheduledStop>,
}

/// A named bus line: a resolved link chain, served on real GTFS trips when the
/// transit artifact provides them, else on a synthetic day-clock headway.
#[derive(Clone, Debug)]
pub struct TransitLine {
    pub name: String,
    pub route: Vec<LinkId>,
    /// Real trips sorted by departure; empty → synthetic headway service.
    pub trips: Vec<BusTrip>,
    /// Day-seconds of the next synthetic departure; `due` holds an undelivered
    /// departure (entrance blocked) for retry.
    next_departure: f64,
    due: bool,
    /// Cursor into `trips` for the current service day; `due_trip` retries a
    /// fired-but-blocked real departure. `last_scan` detects the midnight wrap
    /// (negative-infinity sentinel = fast-forward past history on first scan,
    /// so a rebuild mid-day never re-fires the morning's departures).
    next_trip: usize,
    due_trip: Option<usize>,
    last_scan: f64,
}

impl TransitLine {
    pub fn new(name: String, route: Vec<LinkId>) -> Self {
        Self {
            name,
            route,
            trips: Vec::new(),
            next_departure: 0.0,
            due: false,
            next_trip: 0,
            due_trip: None,
            last_scan: f64::NEG_INFINITY,
        }
    }

    /// Install real trips (replacing the synthetic headway): departures are
    /// normalized into one day and sorted so the firing cursor can sweep them.
    pub fn set_trips(&mut self, mut trips: Vec<BusTrip>) {
        for t in &mut trips {
            t.departure = t.departure.rem_euclid(86_400.0);
        }
        trips.sort_by(|a, b| a.departure.total_cmp(&b.departure));
        self.trips = trips;
        self.next_trip = 0;
        self.due_trip = None;
        self.last_scan = f64::NEG_INFINITY;
    }
}

/// Service headway (day-seconds) by hour: ~15 min through the service day
/// (SamTrans ECR-class), 30 min early/late.
fn transit_headway(day_secs: f64) -> f64 {
    let h = day_secs.rem_euclid(86_400.0) / 3600.0;
    if (6.0..20.0).contains(&h) {
        900.0
    } else {
        1800.0
    }
}

pub struct DemandGenerator {
    /// OD streams with at least one valid route.
    pairs: Vec<OdStream>,
    /// Scheduled bus lines (fixed routes, day-clock headways).
    transit: Vec<TransitLine>,
    seed: u64,
    tick: u64,
    next_id: u32,
    spawned: u32,
    /// Trips discarded because their gateway's backlog hit [`MAX_QUEUE`] — real
    /// demand the sim failed to carry; nonzero means gateways are underfeeding.
    dropped: u32,
    /// Global multiplier on every stream's spawn rate (the UI frequency control).
    rate_scale: f64,
    /// Cap (m/s) on the speed a vehicle enters the map at, applied on top of the
    /// origin road's limit and the driver's desired speed (the UI start-speed control).
    entry_speed_cap: f64,
    /// Simulated seconds-into-day while the rush-hour profile is driving the freeway
    /// streams; `None` off-peak. Advances by `day_compression` each sim second.
    rush_clock: Option<f64>,
    /// Freeze the day clock (warmup pre-population): demand keeps spawning at the
    /// frozen hour's rates and schedule-driven transit holds position, while sim
    /// time (trips, physics) runs normally underneath.
    day_frozen: bool,
    /// Day-seconds per sim second the rush clock advances at (1.0 = real time).
    day_compression: f64,
    /// Days the rush clock has wrapped since it was switched on: `day % 7 ≥ 5` is a
    /// weekend, and each day draws its own demand-level multiplier.
    day: u64,
    /// Sim seconds elapsed — the timebase for rate modulation and OD churn.
    sim_secs: f64,
    /// Last churn window acted on, so each window swaps at most once.
    churn_epoch: u64,
    /// Per-gateway (origin link) backlog of fired-but-not-yet-admitted trips
    /// `(stream, id)`. When demand outruns what the entrance can accept, trips wait
    /// here instead of being dropped, and are released FIFO as lanes clear — a real
    /// metered gateway queue that conserves demand and grows/dissipates with the peak.
    queues: std::collections::BTreeMap<u32, std::collections::VecDeque<(usize, u32)>>,
    /// Online flow calibration toward observed AADT counts (see
    /// [`FlowCalibrator`]); `None` (the default) leaves demand untouched.
    calibrator: Option<FlowCalibrator>,
}

/// Max backlog held per gateway before further trips spill (are dropped) — the
/// off-map storage a gateway queue can represent before it backs out of the region.
const MAX_QUEUE: usize = 400;

/// How often the flow calibrator corrects, in *day* seconds (compression-aware):
/// long enough for a stable windowed count, short enough for several corrections
/// across a peak.
const CAL_WINDOW_DAY_SECS: f64 = 900.0;
/// Correction gain per window (`ratio.powf(GAIN)`), and the bounds a corridor's
/// cumulative correction may reach — a calibrator trims systematic bias, it must
/// never manufacture or erase a corridor's demand outright.
const CAL_GAIN: f64 = 0.5;
const CAL_MIN: f64 = 0.5;
const CAL_MAX: f64 = 2.0;

/// Online flow calibration (SUMO's calibrators, at corridor grain): observed
/// links (`Network::link_aadt`) are grouped by route ref / road name, each group
/// carrying the same hourly expectation model the scorecard judges GEH against;
/// every window the group's simulated flow is compared to its target and the
/// spawn rates of the demand streams *originating* on that corridor are nudged
/// by a bounded, damped multiplier. Per-link detector grain (and mid-map
/// injection, SUMO's other half) stays out of scope: gateway-rate nudging is
/// the lever this demand model actually has.
struct FlowCalibrator {
    /// Per group: observed links with their (time-shaped) hourly target factory.
    groups: Vec<CalGroup>,
    /// stream index → group correcting it.
    stream_group: Vec<Option<usize>>,
    /// Cumulative correction per group (1.0 = untouched).
    factor: Vec<f64>,
    /// `link_entry_counts` snapshot at the last correction.
    last_entries: Vec<u32>,
    /// Sim seconds at the last correction.
    last_secs: f64,
    /// K factor of the arterial peak-hour expectation (the directional split
    /// cancels out of the two-way mean target).
    k_factor: f64,
}

struct CalGroup {
    /// Observed member links: `(link, aadt, is_highway)`.
    links: Vec<(u32, f64, bool)>,
}

impl FlowCalibrator {
    /// Directional hourly target for one observed link at day-time `t` — the
    /// scorecard's expectation model: highways take their peak share from the
    /// PeMS diurnal shape (time-flat); arterials scale K·D by the diurnal
    /// factor's ratio to the daily peak. Direction unknowable per link, so the
    /// mean of the two directional splits.
    fn target_vph(&self, aadt: f64, highway: bool, day_secs: f64) -> f64 {
        if highway {
            let shape = &super::rush_hour::FALLBACK;
            let k_dir = shape.iter().map(|&v| v as f64).fold(0.0, f64::max)
                / shape.iter().map(|&v| v as f64).sum::<f64>();
            return aadt * 0.5 * k_dir;
        }
        let day_peak = (0..96).map(|k| super::rush_hour::arterial_factor(k as f64 * 900.0)).fold(0.0f64, f64::max);
        let tod = (super::rush_hour::arterial_factor(day_secs) / day_peak.max(1e-9)).min(1.0);
        aadt * self.k_factor * 0.5 * tod
    }
}

/// Window length for the doubly-stochastic rate modulation: long enough that a
/// level reads as a "spell" of heavier/lighter traffic, short enough to see it turn
/// over while watching.
const MODULATION_EPOCH_SECS: f64 = 90.0;
/// Lognormal σ of a modulation level — the overdispersion strength (~±45% swings).
const MODULATION_SIGMA: f64 = 0.45;
/// Day-to-day demand-level σ: weekday totals vary with ~10% CV around the mean.
const DAILY_SIGMA: f64 = 0.10;

/// Share of surface firings that launch a platoon rather than a lone vehicle — the
/// bunched fraction of Cowan's M3 arrival model, the signature of arrivals released
/// by an upstream signal. 0.30 is as high as the zero-crash artifact gates allow
/// (observed urban range ~0.25–0.6): re-tested 2026-08-13 after graded yielding,
/// mid-box waiters, and the discharge recalibration — 0.35 still puts two
/// crashes into `mixed_class_traffic_does_not_crash_under_sustained_demand`,
/// so denser bunching stays coupled to whatever artifact that gate is holding.
const PLATOON_PROB: f64 = 0.30;
/// Mean vehicles per firing given the extras distribution in [`platoon_extras`]
/// (1 + 0.25 × 4/3); the firing rate is divided by this so volume is conserved.
const PLATOON_MEAN_SIZE: f64 = 1.0 + PLATOON_PROB * (4.0 / 3.0);

/// How often the origin-churn swap runs (sim seconds).
const CHURN_PERIOD_SECS: f64 = 45.0;

/// Followers behind a platoon leader: none for a lone vehicle, else 1–2 weighted
/// 2:1 — mean 4/3 extras within the platoon branch.
fn platoon_extras(seed: u64, id: u32, tick: u64) -> usize {
    let u = rng::uniform01(seed, id, tick, Stream::Demand);
    if u >= PLATOON_PROB {
        return 0;
    }
    if u / PLATOON_PROB < 2.0 / 3.0 { 1 } else { 2 }
}

/// Mean-one lognormal draw `exp(σz − σ²/2)` keyed by `(seed, agent, tick)`, with
/// `z` an Irwin–Hall(4) approximate standard normal — smooth multiplicative noise
/// on the deterministic counter RNG.
fn lognormal(seed: u64, agent: u32, tick: u64, sigma: f64) -> f64 {
    let sum: f64 = (0..4u32).map(|k| rng::uniform01(seed, agent ^ (k << 24), tick, Stream::Demand)).sum();
    let z = (sum - 2.0) * (3.0f64).sqrt();
    (sigma * z - sigma * sigma * 0.5).exp()
}

/// The whole map's demand-level multiplier for simulated day `day` — some days are
/// simply busier than others (mean 1, ~10% CV), like real count data.
fn daily_factor(seed: u64, day: u64) -> f64 {
    lognormal(seed, 0xDA11_0000, day, DAILY_SIGMA)
}

impl DemandGenerator {
    pub fn new(world: &NetWorld, pairs: &[OdPair], seed: u64) -> Self {
        let reach = Reachability::new(&world.network);
        let pairs = pairs
            .iter()
            .filter(|p| reach.reachable(p.origin, p.dest))
            .map(|p| OdStream {
                origin: p.origin,
                dest: p.dest,
                base_rate: p.rate_per_sec,
                rush: None,
                class: p.class,
                surface: !boundary::is_highway_link(&world.network, p.origin),
                anchored: p.anchored,
            })
            .collect();
        Self {
            pairs, transit: Vec::new(), seed, tick: 0, next_id: 0, spawned: 0, dropped: 0,
            rate_scale: 1.0, entry_speed_cap: f64::INFINITY, rush_clock: None, day_frozen: false,
            day_compression: DEFAULT_DAY_COMPRESSION,
            day: 0, sim_secs: 0.0, churn_epoch: 0,
            queues: std::collections::BTreeMap::new(),
            calibrator: None,
        }
    }

    /// Install the map's named bus lines; each runs its fixed route on the
    /// day-clock headway (buses come only from here — the background traffic
    /// mix carries no random buses).
    pub fn set_transit_lines(&mut self, lines: Vec<TransitLine>) {
        self.transit = lines;
    }

    pub fn transit_lines(&self) -> &[TransitLine] {
        &self.transit
    }

    /// Fire due transit departures: each line spawns a bus on its route when the
    /// day clock passes the next departure; a blocked entrance retries until
    /// admitted (schedules slip, they don't skip). Lines with real GTFS trips
    /// fire those exact departures (on the right service day) and hand the
    /// spawned bus its per-stop schedule; lines without run the synthetic
    /// headway.
    fn step_transit(&mut self, world: &mut NetWorld, day_secs: f64) {
        let n_lines = self.transit.len();
        let weekend = self.weekend();
        for k in 0..n_lines {
            let real = !self.transit[k].trips.is_empty();
            if real {
                let line = &mut self.transit[k];
                if line.last_scan == f64::NEG_INFINITY {
                    // First scan (fresh build or mid-day rebuild): skip history.
                    line.next_trip = line.trips.partition_point(|t| t.departure <= day_secs);
                } else if day_secs + 1800.0 < line.last_scan {
                    line.next_trip = 0; // midnight wrap: sweep from the top
                }
                line.last_scan = day_secs;
                if line.due_trip.is_none() {
                    while line.next_trip < line.trips.len() {
                        let t = &line.trips[line.next_trip];
                        if t.departure > day_secs {
                            break;
                        }
                        line.next_trip += 1;
                        // Only fresh departures on the right service day; a
                        // stale one (a clock jump) is skipped, not bunch-fired.
                        if t.weekend == weekend && day_secs - t.departure < 600.0 {
                            line.due_trip = Some(line.next_trip - 1);
                            break;
                        }
                    }
                }
            } else {
                let line = &mut self.transit[k];
                if line.next_departure == 0.0 && !line.due {
                    // First call: phase departures across lines so they don't bunch.
                    line.next_departure =
                        day_secs + transit_headway(day_secs) * (k as f64 + 1.0) / (n_lines as f64 + 1.0);
                }
                // Midnight wrap: pull a departure scheduled "yesterday" back into
                // range rather than waiting a whole day for the clock to catch it.
                if line.next_departure > day_secs + 43_200.0 {
                    line.next_departure -= 86_400.0;
                }
                if !line.due && day_secs >= line.next_departure {
                    line.due = true;
                    line.next_departure += transit_headway(day_secs);
                }
            }
            if if real { self.transit[k].due_trip.is_none() } else { !self.transit[k].due } {
                continue;
            }
            let route = self.transit[k].route.clone();
            let id = self.next_id;
            let driver = super::config::VehicleClass::Bus.driver().sample(self.seed, id);
            let speed = 8.0f64.min(self.entry_speed_cap);
            if world.spawn_routed(id, route, speed, driver) {
                self.next_id += 1;
                self.spawned += 1;
                let line = &mut self.transit[k];
                if let Some(ti) = line.due_trip.take() {
                    world.set_bus_schedule(id, line.trips[ti].stops.clone());
                }
                line.due = false;
            }
        }
    }

    /// Trips lost to full gateway backlogs — demand the network failed to admit.
    pub fn dropped(&self) -> u32 {
        self.dropped
    }

    /// Set how fast the simulated day plays (day-seconds per sim second, clamped to
    /// [`MIN_DAY_COMPRESSION`]–[`MAX_DAY_COMPRESSION`]). Only the day clock scales;
    /// traffic dynamics stay in real sim time.
    pub fn set_day_compression(&mut self, x: f64) {
        self.day_compression = x.clamp(MIN_DAY_COMPRESSION, MAX_DAY_COMPRESSION);
    }

    pub fn day_compression(&self) -> f64 {
        self.day_compression
    }

    /// Continue a prior generator's day clock (a demand rebuild mid-day): restores the
    /// seconds-into-day and day counter so the commute doesn't jump back to the start
    /// hour. Only meaningful when rush hour is (or is being switched) on.
    pub fn resume_clock(&mut self, day_secs: f64, day: u64) {
        if self.rush_clock.is_some() {
            self.rush_clock = Some(day_secs.rem_euclid(86_400.0));
            self.day = day;
        }
    }

    /// The wrapped-day counter, paired with [`rush_hour_day_secs`](Self::rush_hour_day_secs)
    /// for carrying the clock across a demand rebuild.
    pub fn day(&self) -> u64 {
        self.day
    }

    /// Trips currently waiting at gateways to enter (demand the entrances can't yet
    /// admit) — the metered backlog, for the UI to surface rush-hour pressure.
    pub fn queued(&self) -> u32 {
        self.queues.values().map(|q| q.len()).sum::<usize>() as u32
    }

    pub fn spawned(&self) -> u32 {
        self.spawned
    }

    /// The next vehicle id this generator will assign. Read it before rebuilding the
    /// generator (a demand-source / rush-hour toggle) so the replacement can continue the
    /// id sequence — a fresh generator restarts at 0 and would reissue ids still held by
    /// cars on the map, aliasing them in the renderer's per-id interpolation map.
    pub fn next_id(&self) -> u32 {
        self.next_id
    }

    /// Continue issuing ids from at least `id` (see [`next_id`](Self::next_id)), so a
    /// rebuilt generator never collides with a live vehicle. Only ever advances the
    /// counter.
    pub fn set_next_id(&mut self, id: u32) {
        self.next_id = self.next_id.max(id);
    }

    /// Switch the freeway streams onto the real diurnal PeMS profile (see
    /// [`rush_hour`]) or back to the generic flat rate. When on, the whole map tracks
    /// the simulated time of day: each freeway gateway feeds in its lane count × the
    /// route's per-lane PeMS volume, and each surface street scales by the arterial
    /// diurnal shape — so both breathe with the commute as the day advances.
    pub fn set_rush_hour(&mut self, net: &Network, enabled: bool) {
        if !enabled {
            for s in &mut self.pairs {
                s.rush = None;
            }
            self.rush_clock = None;
            return;
        }
        for i in 0..self.pairs.len() {
            let o = self.pairs[i].origin;
            self.pairs[i].rush = Some(if !self.pairs[i].surface {
                let n = self.pairs.iter().filter(|s| s.origin == o).count();
                // The gateway's travel direction picks the matching directional curve,
                // so an inbound and an outbound freeway gate peak at different times.
                let northbound = net.node(net.link(o).to).position[1] > net.node(net.link(o).from).position[1];
                let lanes = net.link(o).lane_count as f64;
                let profile = rush_hour::profile_for(net.link_ref(o), northbound);
                let aadt = net.link_aadt(o);
                let scale = if aadt > 0.0 {
                    let curve_daily: f64 = profile.iter().map(|&v| v as f64).sum::<f64>() * lanes;
                    (aadt * 0.5 / curve_daily.max(1.0)).clamp(0.1, 4.0)
                } else {
                    1.0
                };
                RushMode::Freeway(RushRate { lanes, profile, share: 1.0 / n as f64, scale })
            } else {
                RushMode::Surface
            });
        }
        self.rush_clock.get_or_insert(RUSH_START_HOUR * 3600.0);
    }

    /// Whether the rush-hour profile is currently driving the freeway streams.
    pub fn rush_hour_active(&self) -> bool {
        self.rush_clock.is_some()
    }

    /// The simulated time of day (seconds since midnight) the rush-hour clock is at,
    /// for the UI readout; 0 when the mode is off.
    pub fn rush_hour_day_secs(&self) -> f64 {
        self.rush_clock.unwrap_or(0.0)
    }

    /// The simulated time of day (seconds since midnight) for a wall-clock readout,
    /// defined in every mode: the rush-hour profile clock when that drives demand,
    /// otherwise the sim clock anchored at the same pre-peak start hour.
    pub fn day_secs(&self, sim_time: f64) -> f64 {
        self.rush_clock.unwrap_or(RUSH_START_HOUR * 3600.0 + sim_time) % 86_400.0
    }

    /// Freeze/unfreeze the day clock (see the field docs).
    pub fn set_day_frozen(&mut self, on: bool) {
        self.day_frozen = on;
    }

    /// Scale every stream's spawn rate (1.0 = as configured; 0.0 = no spawning).
    pub fn set_rate_scale(&mut self, scale: f64) {
        self.rate_scale = scale.max(0.0);
    }

    /// Cap the entry speed (m/s); vehicles still never exceed the origin road's limit
    /// or the driver's desired speed. `f64::INFINITY` = enter at the road's limit.
    pub fn set_entry_speed_cap(&mut self, cap: f64) {
        self.entry_speed_cap = cap.max(0.0);
    }

    /// Switch on online flow calibration: observed links (`link_aadt` > 0) are
    /// grouped by route ref / road name, and each window the spawn rates of the
    /// streams originating on a group's corridor are nudged toward its hourly
    /// target (bounded, damped — see [`FlowCalibrator`]). `k_factor` is the
    /// arterial peak-hour share of AADT the targets are judged by.
    pub fn enable_flow_calibration(&mut self, world: &NetWorld, k_factor: f64) {
        use std::collections::BTreeMap;
        let net = &world.network;
        let key_of = |l: usize| -> Option<String> {
            let r = net.link_refs.get(l).map(String::as_str).unwrap_or("");
            if !r.is_empty() {
                return Some(r.to_string());
            }
            let n = net.link_names.get(l).map(String::as_str).unwrap_or("");
            (!n.is_empty()).then(|| n.to_string())
        };
        let mut index: BTreeMap<String, usize> = BTreeMap::new();
        let mut groups: Vec<CalGroup> = Vec::new();
        for l in 0..net.links.len() {
            let aadt = net.link_aadt(LinkId(l as u32));
            if aadt <= 0.0 {
                continue;
            }
            let Some(key) = key_of(l) else { continue };
            let gi = *index.entry(key).or_insert_with(|| {
                groups.push(CalGroup { links: Vec::new() });
                groups.len() - 1
            });
            groups[gi].links.push((l as u32, aadt, boundary::is_highway_link(net, LinkId(l as u32))));
        }
        let stream_group = self
            .pairs
            .iter()
            .map(|s| key_of(s.origin.idx()).and_then(|k| index.get(&k).copied()))
            .collect();
        self.calibrator = Some(FlowCalibrator {
            factor: vec![1.0; groups.len()],
            groups,
            stream_group,
            last_entries: world.link_entry_counts().to_vec(),
            last_secs: self.sim_secs,
            k_factor,
        });
    }

    /// The calibration multiplier for stream `i` (1.0 when calibration is off or
    /// the stream's corridor carries no observed counts).
    fn cal_factor(&self, i: usize) -> f64 {
        self.calibrator
            .as_ref()
            .and_then(|c| c.stream_group[i].map(|g| c.factor[g]))
            .unwrap_or(1.0)
    }

    /// Periodic calibration step: once per window (day-time), difference the
    /// link-entry counters into windowed flows and nudge each group's factor
    /// toward its target.
    fn update_calibration(&mut self, world: &NetWorld) {
        let compression = if self.rush_clock.is_some() { self.day_compression } else { 1.0 };
        let day_now = self.day_secs(self.sim_secs);
        let sim_secs = self.sim_secs;
        let Some(cal) = self.calibrator.as_mut() else { return };
        let elapsed_day = (sim_secs - cal.last_secs) * compression;
        if elapsed_day < CAL_WINDOW_DAY_SECS {
            return;
        }
        let entries = world.link_entry_counts();
        let hours = elapsed_day / 3600.0;
        let ratios: Vec<f64> = cal
            .groups
            .iter()
            .map(|g| {
                let (mut sim, mut tgt) = (0.0f64, 0.0f64);
                for &(l, aadt, hw) in &g.links {
                    sim += entries[l as usize].saturating_sub(cal.last_entries[l as usize]) as f64 / hours;
                    tgt += cal.target_vph(aadt, hw, day_now);
                }
                if tgt <= 1.0 {
                    return 1.0;
                }
                // A silent corridor reads as a bounded shortfall, not a
                // divide-by-zero demand explosion.
                (tgt / sim.max(tgt * 0.1)).powf(CAL_GAIN)
            })
            .collect();
        for (f, r) in cal.factor.iter_mut().zip(&ratios) {
            *f = (*f * r).clamp(CAL_MIN, CAL_MAX);
        }
        cal.last_entries = entries.to_vec();
        cal.last_secs = sim_secs;
    }

    /// The distinct destinations demanded — the destination set to build a
    /// [`NetWorld`] flow-field router over.
    pub fn destinations(&self) -> Vec<LinkId> {
        let mut dests: Vec<LinkId> = self.pairs.iter().map(|s| s.dest).collect();
        dests.sort_by_key(|l| l.0);
        dests.dedup();
        dests
    }

    pub fn step(&mut self, world: &mut NetWorld, dt: f64) {
        self.update_calibration(world);
        let costs = world.live_link_costs();
        let day = self.rush_clock;
        let weekend = self.weekend();
        let daily = if day.is_some() { daily_factor(self.seed, self.day) } else { 1.0 };

        // 1. Release each gateway's backlog first, oldest trip first, until its
        //    entrance is occupied again — so waiting demand takes priority over fresh.
        let origins: Vec<u32> = self.queues.keys().copied().collect();
        for o in origins {
            while let Some(&(stream, id)) = self.queues[&o].front() {
                if self.launch(world, &costs, stream, id) {
                    self.queues.get_mut(&o).unwrap().pop_front();
                    self.spawned += 1;
                } else {
                    break;
                }
            }
        }

        // 2. Fire fresh trips from each stream's Poisson rate; a trip the entrance
        //    can't admit joins its gateway's queue rather than vanishing. Surface
        //    firings occasionally launch a *platoon* (extras join the queue, so they
        //    enter nose-to-tail as the entrance clears), with the firing rate divided
        //    by the mean platoon size so bunching reshapes arrivals, never volume.
        for i in 0..self.pairs.len() {
            let s = &self.pairs[i];
            let rate = match (day, s.rush) {
                (Some(t), Some(RushMode::Freeway(r))) => {
                    r.lanes * r.scale * rush_hour::freeway_flow(r.profile, weekend, t) / 3600.0 * r.share
                }
                (Some(t), Some(RushMode::Surface)) => s.base_rate * rush_hour::surface_factor(s.class, weekend, t),
                _ => s.base_rate,
            };
            let mut rate = rate * daily * self.modulation(i) * self.cal_factor(i);
            if s.surface {
                rate /= PLATOON_MEAN_SIZE;
            }
            if rng::uniform01(self.seed, i as u32, self.tick, Stream::RouteChoice) >= (rate * self.rate_scale * dt).min(1.0) {
                continue;
            }
            let (origin, surface) = (s.origin, s.surface);
            let id = self.next_id;
            self.next_id += 1;
            if self.launch(world, &costs, i, id) {
                self.spawned += 1;
            } else {
                let q = self.queues.entry(origin.0).or_default();
                if q.len() < MAX_QUEUE {
                    q.push_back((i, id));
                } else {
                    self.dropped += 1;
                }
            }
            if surface {
                for _ in 0..platoon_extras(self.seed, id, self.tick) {
                    let follower = self.next_id;
                    self.next_id += 1;
                    let q = self.queues.entry(origin.0).or_default();
                    if q.len() < MAX_QUEUE {
                        q.push_back((i, follower));
                    } else {
                        self.dropped += 1;
                    }
                }
            }
        }

        let day_now = self.day_secs(world.time());
        // Per-tick day clock for day-scheduled infrastructure (crossings,
        // trains, bus holding) — finer than the bridge's per-frame feed, and
        // the only feed in headless runs.
        let day_rate = if self.day_frozen {
            0.0 // trains hold position; nothing day-scheduled moves during warmup
        } else if self.rush_clock.is_some() {
            self.day_compression
        } else {
            1.0
        };
        world.set_day_clock(day_now, day_rate, weekend);
        self.step_transit(world, day_now);

        if let Some(t) = &mut self.rush_clock {
            if !self.day_frozen {
                let next = (*t + dt * self.day_compression).rem_euclid(86_400.0);
                if next < *t {
                    self.day += 1;
                }
                *t = next;
            }
        }
        self.sim_secs += dt;
        let churn = (self.sim_secs / CHURN_PERIOD_SECS) as u64;
        if churn != self.churn_epoch {
            self.churn_epoch = churn;
            self.churn_origins(&world.network, churn);
        }
        self.tick += 1;
    }

    /// Whether the simulated day is a weekend (days 5 and 6 of each week; the rush
    /// clock starts on a Monday). Never a weekend while the clock is off.
    fn weekend(&self) -> bool {
        self.rush_clock.is_some() && self.day % 7 >= 5
    }

    /// Slowly-wandering mean-one multiplier on stream `i`'s rate — a doubly-stochastic
    /// (Cox) rate: each [`MODULATION_EPOCH_SECS`] window draws a lognormal level and
    /// the rate glides between them. Real counts are overdispersed (variance > mean);
    /// a plain Poisson drip reads as unnaturally even. Deterministic in the seed.
    fn modulation(&self, i: usize) -> f64 {
        let e = self.sim_secs / MODULATION_EPOCH_SECS;
        let (k, frac) = (e as u64, e.fract());
        let g = |k: u64| lognormal(self.seed, i as u32, k, MODULATION_SIGMA);
        g(k) * (1.0 - frac) + g(k + 1) * frac
    }

    /// Swap `(origin, base_rate)` between two same-class surface streams so spawn
    /// locations wander over the run instead of the same few corners feeding the map
    /// forever. Swapping (rather than resampling) preserves every origin's calibrated
    /// aggregate and reuses the existing destination set, so no new flow field is
    /// ever needed. Streams with a queued backlog are left alone (their trips are
    /// anchored to the origin they were fired from).
    fn churn_origins(&mut self, net: &Network, epoch: u64) {
        if self.pairs.is_empty() {
            return;
        }
        let reach = Reachability::new(net);
        let idle = |gen: &Self, s: &OdStream| {
            s.surface && !s.anchored && gen.queues.get(&s.origin.0).map_or(true, |q| q.is_empty())
        };
        for attempt in 0..8u64 {
            let draw = |salt: u32| {
                (rng::hash(self.seed, salt, epoch * 8 + attempt, Stream::Demand) as usize) % self.pairs.len()
            };
            let (a, b) = (draw(0xC0), draw(0xC1));
            let (sa, sb) = (&self.pairs[a], &self.pairs[b]);
            if a == b
                || sa.class != sb.class
                || sa.origin == sb.origin
                || !idle(self, sa)
                || !idle(self, sb)
                || !reach.reachable(sa.origin, sb.dest)
                || !reach.reachable(sb.origin, sa.dest)
            {
                continue;
            }
            let (ao, ar) = (sa.origin, sa.base_rate);
            let (bo, br) = (sb.origin, sb.base_rate);
            (self.pairs[a].origin, self.pairs[a].base_rate) = (bo, br);
            (self.pairs[b].origin, self.pairs[b].base_rate) = (ao, ar);
            return;
        }
    }

    /// Attempt to admit trip `id` of stream `stream` at its origin: destination-routed
    /// via the live flow-field when the router knows the destination, else on a
    /// cost-routed link path. `false` if the entrance is occupied (try again later) or
    /// no route exists. Driver and entry speed are derived from `id`, so a queued trip
    /// keeps its identity across the wait.
    fn launch(&self, world: &mut NetWorld, costs: &[u64], stream: usize, id: u32) -> bool {
        let (origin, dest) = (self.pairs[stream].origin, self.pairs[stream].dest);
        let highway = !self.pairs[stream].surface;
        let driver = class_of(self.seed, id, highway, self.rush_clock).driver().sample(self.seed, id);
        let speed = self.launch_speed(&world.network, stream, origin, &driver);
        if world.router_knows(dest) {
            world.spawn_to(id, origin, dest, speed, driver)
        } else if let Some(route) = world.network.route_links_with_costs(origin, dest, costs) {
            world.spawn_routed(id, route, speed, driver)
        } else {
            false
        }
    }

    /// The speed a trip enters at: its origin road's free-flow speed off-peak, but eased
    /// down to a congested crawl on a freeway gateway whose current per-lane volume is deep
    /// in the rush-hour peak. The lower speed shrinks the admission gap (`spawn_to`), so
    /// peak traffic packs onto the freeway densely — many slow cars at once — instead of a
    /// fast, sparse stream. Always bounded by the UI start-speed cap.
    fn launch_speed(&self, net: &Network, stream: usize, origin: LinkId, driver: &super::config::DriverConfig) -> f64 {
        let free_flow = entry_speed(net, origin, driver);
        let speed = match (self.rush_clock, self.pairs[stream].rush) {
            (Some(t), Some(RushMode::Freeway(r))) => {
                rush_hour::congested_entry_speed(free_flow, r.scale * rush_hour::freeway_flow(r.profile, self.weekend(), t))
            }
            _ => free_flow,
        };
        speed.min(self.entry_speed_cap)
    }
}

/// Boundary-aware OD pairs: traffic passing through the map (gateway→gateway),
/// arriving (gateway→interior), leaving (interior→gateway), and staying within
/// (interior→interior). Deterministic in `seed`; falls back to any-link pairs
/// when the map has no gateways to anchor the categories.
pub fn boundary_od_pairs(net: &Network, seed: u64, target: usize) -> Vec<OdPair> {
    let reach = Reachability::new(net);
    let entries = boundary::entry_links(net);
    let exits = boundary::exit_links(net);
    let interior = boundary::interior_links(net);
    let categories: [(&[LinkId], &[LinkId], f64, SurfaceClass); 4] = [
        (&entries, &exits, 0.45, SurfaceClass::Through),
        (&entries, &interior, 0.20, SurfaceClass::Inbound),
        (&interior, &exits, 0.20, SurfaceClass::Outbound),
        (&interior, &interior, 0.15, SurfaceClass::Internal),
    ];

    let mut pairs = Vec::new();
    for (cat, (origins, dests, share, class)) in categories.iter().enumerate() {
        if origins.is_empty() || dests.is_empty() {
            continue;
        }
        let want = ((target as f64 * share).round() as usize).max(1);
        sample_pairs(net, &reach, seed, cat as u32, origins, dests, want, *class, DemandTuning::DEFAULT, &mut pairs);
    }
    if pairs.is_empty() {
        let all: Vec<LinkId> = (0..net.links.len() as u32).map(LinkId).collect();
        sample_pairs(net, &reach, seed, 99, &all, &all, target, SurfaceClass::Through, DemandTuning::DEFAULT, &mut pairs);
    }
    pairs
}

/// Freeway traffic: each trip enters at a highway gateway (from outside the map) and
/// is bound for — by likelihood — the far end of the *same* highway (through-traffic),
/// another highway exit (an interchange), or a surface street it leaves the freeway
/// for. Never a mid-freeway segment. Same-highway matching uses the OSM route `ref`;
/// a map without refs or freeway gateways contributes nothing (the caller falls back).
pub fn highway_od_pairs(net: &Network, seed: u64, target: usize, t: DemandTuning, out: &mut Vec<OdPair>) {
    let reach = Reachability::new(net);
    let hw_in = boundary::highway_entry_links(net);
    if hw_in.is_empty() {
        return;
    }
    let hw_out = boundary::highway_exit_links(net);
    let surface = boundary::surface_interior_links(net);

    // Per highway entry, its routable highway exits split into same-highway (route
    // ref matches) and other. Precomputed once; surface destinations are sampled
    // per-attempt (there are too many to route-check up front).
    let pools: Vec<(LinkId, Vec<LinkId>, Vec<LinkId>)> = hw_in
        .iter()
        .map(|&e| {
            let r = net.link_ref(e);
            let (mut same, mut other) = (Vec::new(), Vec::new());
            for &d in &hw_out {
                if d == e || !reach.reachable(e, d) {
                    continue;
                }
                if !r.is_empty() && same_highway(net.link_ref(d), r) {
                    // Same highway: a through-trip only if the exit continues in the
                    // entry's direction of travel. The opposite carriageway shares the
                    // route ref but heads back the other way — a U-turn onto the same
                    // freeway, which is impossible without leaving the freeway system —
                    // so it's dropped, not offered as through-traffic.
                    if continues_forward(net, e, d) {
                        same.push(d);
                    }
                } else {
                    // A different highway (either direction is fine) or an unrefed exit.
                    other.push(d);
                }
            }
            (e, same, other)
        })
        .collect();

    // Reserve a share of freeway demand for on-ramp loading (city → freeway →
    // off-map) so the mainline is fed at the interchanges, not only at the two
    // map-edge gateways; the rest is gateway-origin through/interchange traffic.
    let onramp_target = (target as f64 * t.on_ramp_share).round() as usize;
    let gateway_target = target.saturating_sub(onramp_target);

    let mut found = 0usize;
    let mut attempt = 0u64;
    while found < gateway_target && attempt < gateway_target as u64 * 40 + 400 {
        let (e, same, other) = &pools[(rng::hash(seed, 60, attempt, Stream::RouteChoice) as usize) % pools.len()];
        let r = rng::uniform01(seed, e.0, attempt, Stream::RouteChoice);
        // Weighted by category, cascading when a pool is empty so the majority still
        // lands on the same highway wherever refs make it possible. The through
        // share is set by AADT continuity: a freeway's counted volume barely
        // decays across a city-sized box (US-101 carries ~219k the whole way
        // through Millbrae), so ~85% of gateway trips ride the mainline out the
        // far end and only the remainder leaves for an interchange or the city.
        let dest = if r < 0.85 {
            pick(same, seed, attempt)
                .or_else(|| pick(other, seed, attempt))
                .or_else(|| pick_routable(net, &reach, *e, &surface, seed, attempt, SurfaceClass::Through, t))
        } else if r < 0.93 {
            pick(other, seed, attempt)
                .or_else(|| pick(same, seed, attempt))
                .or_else(|| pick_routable(net, &reach, *e, &surface, seed, attempt, SurfaceClass::Through, t))
        } else {
            pick_routable(net, &reach, *e, &surface, seed, attempt, SurfaceClass::Through, t)
                .or_else(|| pick(same, seed, attempt))
                .or_else(|| pick(other, seed, attempt))
        };
        attempt += 1;
        if let Some(d) = dest {
            if d != *e {
                out.push(OdPair { origin: *e, dest: d, rate_per_sec: capacity_rate(net, *e), class: SurfaceClass::Through, anchored: false });
                found += 1;
            }
        }
    }

    // On-ramp loading: local trips that get on the freeway and ride it out of the
    // map. Origin on a surface street (production-weighted — where drivers actually
    // start), destination a highway-exit gateway reachable ONLY through an on-ramp,
    // so the router sends them local → on-ramp → mainline → off the map. This feeds
    // the freeway at every interchange the map contains, not just the two edge
    // gateways — a freeway loaded by the city it runs through. It fires only where
    // the map has on-ramps (a surface→freeway route exists); otherwise it finds
    // nothing and the freeway stays gateway-fed. Destinations are drawn uniformly
    // among reachable highway exits (the freeway-zeroing `gravity_pick` endpoint
    // term doesn't apply — a highway gateway is a legitimate trip end).
    if onramp_target > 0 && !surface.is_empty() && !hw_out.is_empty() {
        let mut ocum = Vec::with_capacity(surface.len());
        let mut ototal = 0.0;
        for &o in &surface {
            ototal += net.link_res_weight(o) * class_production(net.link(o).kind, t);
            ocum.push(ototal);
        }
        let mut found_on = 0usize;
        let mut attempt_on = 0u64;
        while ototal > 0.0 && found_on < onramp_target && attempt_on < onramp_target as u64 * 40 + 400 {
            let r = rng::uniform01(seed, 63, attempt_on, Stream::RouteChoice) * ototal;
            let o = surface[ocum.partition_point(|&c| c < r).min(surface.len() - 1)];
            let mut dest = None;
            for k in 0..8u64 {
                let cand = hw_out[(rng::hash(seed, 64, attempt_on.wrapping_mul(8).wrapping_add(k), Stream::RouteChoice) as usize) % hw_out.len()];
                if cand != o && reach.reachable(o, cand) {
                    dest = Some(cand);
                    break;
                }
            }
            attempt_on += 1;
            if let Some(d) = dest {
                out.push(OdPair { origin: o, dest: d, rate_per_sec: capacity_rate(net, o), class: SurfaceClass::Outbound, anchored: false });
                found_on += 1;
            }
        }
    }
}

/// Surface (local/arterial) traffic: the boundary mix over non-freeway gateways and
/// interior streets — through the city, inbound, outbound, and internal trips.
pub fn surface_od_pairs(net: &Network, seed: u64, target: usize, t: DemandTuning, out: &mut Vec<OdPair>) {
    let reach = Reachability::new(net);
    let entries = boundary::surface_entry_links(net);
    let exits = boundary::surface_exit_links(net);
    let interior = boundary::surface_interior_links(net);
    // Observed corridors first: a counted state route (El Camino's CA-82) mostly
    // *carries its measured volume through* — the count at every screenline is
    // dominated by traffic riding the corridor, not by trips scattering off it. Each
    // counted, named gateway anchors a through stream to the far end of the same-
    // named road, PLUS mid-corridor access streams at its cross-streets, so the
    // corridor loads and unloads along its length (a longitudinal volume profile)
    // instead of one flat end-to-end OD. The entry's total anchored inflow is
    // preserved (through-share + egress-shares ≈ the old 0.9), so screenline volume
    // is not inflated; the access streams add realistic local mid-corridor traffic.
    let corridor_through_share = t.corridor_through_share;
    let corridor_access_share = t.corridor_access_share;
    const MAX_CORRIDOR_ACCESS: usize = 2;
    for &e in &entries {
        if net.link_aadt(e) <= 0.0 {
            continue;
        }
        let name = net.link_names[e.idx()].as_str();
        if name.is_empty() {
            continue;
        }
        let ep = link_centroid(net, e);
        let far = exits
            .iter()
            .filter(|&&d| d != e && net.link_names[d.idx()] == name && reach.reachable(e, d))
            .max_by(|&&a, &&b| {
                let da = link_centroid(net, a);
                let db = link_centroid(net, b);
                (da[0] - ep[0]).hypot(da[1] - ep[1]).total_cmp(&(db[0] - ep[0]).hypot(db[1] - ep[1]))
            });
        let Some(&d) = far else { continue };
        let rate = capacity_rate(net, e);
        out.push(OdPair {
            origin: e,
            dest: d,
            rate_per_sec: corridor_through_share * rate,
            class: SurfaceClass::Through,
            anchored: true,
        });
        // Same-named interior links between the entry and the far exit — the
        // corridor's mid-block cross-streets — become intermediate trip ends,
        // spread along its length by distance from the entry.
        let mut mids: Vec<LinkId> = interior
            .iter()
            .copied()
            .filter(|&m| {
                m != e && m != d && net.link_names[m.idx()] == name && reach.reachable(e, m) && reach.reachable(m, d)
            })
            .collect();
        if mids.is_empty() {
            continue;
        }
        mids.sort_by(|&a, &b| {
            let ca = link_centroid(net, a);
            let cb = link_centroid(net, b);
            (ca[0] - ep[0]).hypot(ca[1] - ep[1]).total_cmp(&(cb[0] - ep[0]).hypot(cb[1] - ep[1]))
        });
        let n = mids.len();
        let k = MAX_CORRIDOR_ACCESS.min(n);
        for j in 0..k {
            let m = mids[(((j + 1) * n) / (k + 1)).min(n - 1)];
            // Egress: through-riders peeling off at this cross-street (shares the
            // entry origin, so it counts against the entry's preserved inflow).
            out.push(OdPair { origin: e, dest: m, rate_per_sec: corridor_access_share * rate, class: SurfaceClass::Through, anchored: true });
            // Access: local traffic joining the corridor here, bound for its far end
            // (new mid-corridor volume — the corridor filling as it runs).
            out.push(OdPair { origin: m, dest: d, rate_per_sec: corridor_access_share * capacity_rate(net, m), class: SurfaceClass::Through, anchored: true });
        }
    }
    let categories: [(&[LinkId], &[LinkId], f64, SurfaceClass); 4] = [
        (&entries, &exits, 0.35, SurfaceClass::Through),      // through the city
        (&entries, &interior, 0.25, SurfaceClass::Inbound),   // arriving to a local destination
        (&interior, &exits, 0.20, SurfaceClass::Outbound),    // leaving town
        (&interior, &interior, 0.20, SurfaceClass::Internal), // internal local trips
    ];
    for (cat, (origins, dests, share, class)) in categories.iter().enumerate() {
        if origins.is_empty() || dests.is_empty() {
            continue;
        }
        let want = ((target as f64 * share).round() as usize).max(1);
        sample_pairs(net, &reach, seed, 80 + cat as u32, origins, dests, want, *class, t, out);
    }
}

/// Pick a random link from a small, pre-route-checked pool.
fn pick(pool: &[LinkId], seed: u64, attempt: u64) -> Option<LinkId> {
    (!pool.is_empty()).then(|| pool[(rng::hash(seed, 61, attempt, Stream::RouteChoice) as usize) % pool.len()])
}

/// Gravity-model distance-decay exponent: trip attraction falls as `1/dist^β`.
/// Fitted to real commute destination choice by maximum likelihood over the LODES
/// OD data (`tools/lodes/fit_gravity.py`): β = 0.54–0.75 across the five shipped
/// maps, commuter-weighted 0.68 — urban commuters barely distance-minimize within
/// a metro box once the opportunity geography is controlled for. Errand/shopping
/// trips decay faster in the literature (~1.5–2); a per-class β is a future
/// refinement.
const GRAVITY_BETA: f64 = 0.7;
/// Distance floor (m) so co-located destinations don't get an unbounded weight.
const GRAVITY_MIN_DIST: f64 = 40.0;

/// A link's midpoint, the point trips gravitate toward / from.
fn link_centroid(net: &Network, link: LinkId) -> [f64; 2] {
    let l = net.link(link);
    let (a, b) = (net.node(l.from).position, net.node(l.to).position);
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
}

/// Distance-decay exponent β by trip purpose. Commute/through trips barely
/// distance-minimize within a metro box (LODES max-likelihood fit β≈0.7); local
/// errand/shopping trips (`Internal`) decay far faster (~1.8, the NHTS shopping
/// range) so they stay in the neighborhood instead of scattering across the map.
/// Through trips are the longest (cross-map), so they decay least. This is the
/// per-class β the global-`GRAVITY_BETA` comment flagged as a future refinement.
fn class_beta(class: SurfaceClass, t: DemandTuning) -> f64 {
    match class {
        SurfaceClass::Internal => t.internal_beta,
        SurfaceClass::Inbound | SurfaceClass::Outbound | SurfaceClass::Through => t.gravity_beta,
    }
}

/// Gravity draw from `pool`: destination `d` is chosen with probability ∝
/// `link_capacity(d) × class_attraction(d) / dist(o, d)^β` — the classic gravity
/// model with a road-function endpoint term, so trips prefer bigger, nearer, and
/// endpoint-suitable roads over a uniform scatter. `β` is the trip purpose's
/// distance decay ([`class_beta`]). `None` only for an empty pool.
fn gravity_pick(net: &Network, o: LinkId, pool: &[LinkId], seed: u64, salt: u32, attempt: u64, beta: f64, t: DemandTuning) -> Option<LinkId> {
    if pool.is_empty() {
        return None;
    }
    let op = link_centroid(net, o);
    let mut total = 0.0;
    let mut cum = Vec::with_capacity(pool.len());
    for &d in pool {
        let dp = link_centroid(net, d);
        let dist = (op[0] - dp[0]).hypot(op[1] - dp[1]).max(GRAVITY_MIN_DIST);
        // Attraction = road-size × land-use × road-FUNCTION endpoint-suitability, all
        // over distance^β. The class term (`class_attraction`) demotes big through-
        // arterials as trip *endpoints* and promotes the local/collector network, so
        // destinations spread onto the streets where trips actually end rather than
        // piling onto the highest-capacity corridors. Land-use is neutral 1.0 without
        // the scraper's pass; the class term shapes the draw from topology alone.
        total += link_capacity(net, d) * net.link_attr_weight(d) * class_attraction(net.link(d).kind, t)
            / dist.powf(beta);
        cum.push(total);
    }
    if total <= 0.0 {
        return None;
    }
    let r = rng::uniform01(seed, salt, attempt, Stream::RouteChoice) * total;
    Some(pool[cum.partition_point(|&c| c < r).min(pool.len() - 1)])
}

/// Gravity-weighted pick from `pool` that is routable from `e` (a few candidate
/// tries), for trips of purpose `class` (sets the distance decay).
fn pick_routable(net: &Network, reach: &Reachability, e: LinkId, pool: &[LinkId], seed: u64, attempt: u64, class: SurfaceClass, t: DemandTuning) -> Option<LinkId> {
    for k in 0..8u64 {
        let d = gravity_pick(net, e, pool, seed, 62, attempt.wrapping_mul(8).wrapping_add(k), class_beta(class, t), t)?;
        if d != e && reach.reachable(e, d) {
            return Some(d);
        }
    }
    None
}

/// Whether two OSM route refs designate the same highway — sharing any route token
/// ("I 280;CA 35" and "I 280" match on "I 280").
fn same_highway(a: &str, b: &str) -> bool {
    a.split(';').any(|t| !t.is_empty() && b.split(';').any(|u| u == t))
}

/// Unit travel direction of `link` (from its `from` node toward its `to` node). For a
/// boundary link this points the way traffic crosses the map edge: inward for an entry,
/// outward for an exit — so an entry and its through-exit on the same carriageway share
/// a heading, while the opposite carriageway's exit heads the other way.
fn travel_dir(net: &Network, link: LinkId) -> [f64; 2] {
    let l = net.link(link);
    let (a, b) = (net.node(l.from).position, net.node(l.to).position);
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let m = dx.hypot(dy).max(1e-6);
    [dx / m, dy / m]
}

/// Whether a highway `exit` continues in the `entry`'s direction of travel rather than
/// doubling back on the opposite carriageway — a same-highway exit is genuine
/// through-traffic only when the two headings agree (positive dot product).
fn continues_forward(net: &Network, entry: LinkId, exit: LinkId) -> bool {
    let (e, x) = (travel_dir(net, entry), travel_dir(net, exit));
    e[0] * x[0] + e[1] * x[1] > 0.0
}

fn sample_pairs(
    net: &Network,
    reach: &Reachability,
    seed: u64,
    salt: u32,
    origins: &[LinkId],
    dests: &[LinkId],
    want: usize,
    class: SurfaceClass,
    t: DemandTuning,
    out: &mut Vec<OdPair>,
) {
    // Origins draw by land-use production weight × road-function production — trips
    // start where people live (residential/local streets), scaled by land use where
    // the scraper supplied it. On a map without land-use data the class term alone
    // still biases origins onto the local/collector network instead of uniformly.
    let mut ocum = Vec::with_capacity(origins.len());
    let mut ototal = 0.0;
    for &o in origins {
        ototal += net.link_res_weight(o) * class_production(net.link(o).kind, t);
        ocum.push(ototal);
    }
    // Degenerate pool (e.g. all-arterial origins zeroed by class): fall back to
    // uniform so the category still produces its trips.
    if ototal <= 0.0 {
        ototal = origins.len() as f64;
        ocum.clear();
        for i in 0..origins.len() {
            ocum.push((i + 1) as f64);
        }
    }
    let mut found = 0;
    let mut attempt = 0u64;
    while found < want && attempt < want as u64 * 40 + 200 {
        let r = rng::uniform01(seed, salt * 2, attempt, Stream::RouteChoice) * ototal;
        let o = origins[ocum.partition_point(|&c| c < r).min(origins.len() - 1)];
        // Destination by gravity (bigger/nearer/endpoint-suitable roads win), not a
        // uniform scatter; distance decay is set by the trip purpose (`class_beta`).
        let d = gravity_pick(net, o, dests, seed, salt * 2 + 1, attempt, class_beta(class, t), t);
        attempt += 1;
        if let Some(d) = d {
            // `o != d` reachable ⟺ a route of ≥ 2 links exists, so this matches the old
            // `route_links(o, d).is_some_and(|r| r.len() >= 2)` gate without the search.
            if o != d && reach.reachable(o, d) {
                out.push(OdPair { origin: o, dest: d, rate_per_sec: capacity_rate(net, o), class, anchored: false });
                found += 1;
            }
        }
    }
}

/// Trip-PRODUCTION multiplier by road function. Trips overwhelmingly begin on the
/// local/collector network — homes, driveways, parking — and only rarely off an
/// arterial's frontage; you never start a trip mid-freeway or on a ramp. This
/// multiplies the land-use production weight, so a map WITHOUT the scraper's
/// land-use pass (every `link_res_weight` == 1.0) still originates traffic from the
/// street network's *function* rather than uniformly over every interior link.
/// Grounded in `RoadKind`'s own doc: Local = "where trips actually start and end",
/// Arterial = "a through-corridor, not a typical trip endpoint".
fn class_production(kind: RoadKind, t: DemandTuning) -> f64 {
    if !t.road_function_weighting {
        return 1.0;
    }
    match kind {
        RoadKind::Local => 1.0,
        RoadKind::Collector => 0.7,
        RoadKind::Arterial => 0.25,
        RoadKind::Ramp | RoadKind::Freeway => 0.0,
    }
}

/// Trip-ATTRACTION (endpoint-suitability) multiplier by road function. Trips END on
/// the local/collector network and along commercial arterial frontage — not as
/// through-movements on the highest-capacity corridors, and never on a freeway or
/// ramp. Demotes the raw-capacity bias in [`gravity_pick`] that otherwise piles
/// destinations onto the biggest through-arterials (whose measured volume is carried
/// by the anchored through-streams, not by trips terminating there).
fn class_attraction(kind: RoadKind, t: DemandTuning) -> f64 {
    if !t.road_function_weighting {
        // Neutral for surface pools (which exclude freeways/ramps anyway); the
        // capacity term alone drives attraction, the pre-lever behavior.
        return 1.0;
    }
    match kind {
        RoadKind::Local => 1.0,
        RoadKind::Collector => 0.9,
        RoadKind::Arterial => 0.5,
        RoadKind::Ramp | RoadKind::Freeway => 0.0,
    }
}

/// Trip-attraction weight of a link: its *observed* AADT when a real count is
/// attached (scaled so a 30k-AADT arterial lands near the proxy's arterial range),
/// else the `lanes × speed_limit` proxy that correlates with volume — so demand
/// favours the roads that actually carry traffic (a freeway ramp / arterial like
/// El Camino Real over a residential street), preferring measurement to proxy.
fn link_capacity(net: &Network, link: LinkId) -> f64 {
    let aadt = net.link_aadt(link);
    if aadt > 0.0 {
        return aadt / 1000.0;
    }
    let l = net.link(link);
    l.lane_count as f64 * net.lane(l.lane_start).speed_limit
}

/// Share of a road's two-way AADT flowing in one directed link's direction.
const DIRECTIONAL_SHARE: f64 = 0.5;

/// Spawn rate for a stream originating on `origin`. With an observed count it is
/// the road's real daily-mean directional flow (`AADT/2` over 24 h — the diurnal
/// multiplier redistributes it through the day); otherwise the road-capacity proxy
/// relative to a typical arterial. Clamped so no single road dominates or starves.
fn capacity_rate(net: &Network, origin: LinkId) -> f64 {
    const BASE: f64 = 0.2;
    const REFERENCE: f64 = 30.0; // ~ a 2-lane, 15 m/s arterial
    let aadt = net.link_aadt(origin);
    let rate = if aadt > 0.0 {
        aadt * DIRECTIONAL_SHARE / 86_400.0
    } else {
        BASE * link_capacity(net, origin) / REFERENCE
    };
    rate.clamp(0.04, 0.9)
}

/// The speed a vehicle enters the map at: the free-flow speed of its origin road
/// (its posted limit, capped by the driver's own desired speed), so a car arriving
/// on a freeway is already moving at freeway speed instead of crawling up from a
/// standstill. Off-peak traffic entering from US-101 / I-280 comes in fast.
fn entry_speed(net: &Network, origin: LinkId, driver: &super::config::DriverConfig) -> f64 {
    let limit = net.lane(net.link(origin).lane_start).speed_limit;
    limit.min(driver.desired_speed)
}

/// Hourly truck-share multiplier (×100): trucks avoid the commute peaks and are
/// overrepresented overnight and around midday — the standard FHWA
/// Traffic-Monitoring-Guide shape for urban truck traffic.
const TRUCK_HOURLY_X100: [u16; 24] = [
    150, 155, 160, 160, 150, 120, 80, 60, 70, 100, 120, 125, 125, 120, 115, 100, 70, 55, 65, 80,
    95, 110, 125, 140,
];

/// Vehicle-class mix: mostly cars, some trucks, a few buses. Shares follow the
/// Caltrans truck AADT for the peninsula corridors (US-101 in San Mateo County
/// runs ≈5% trucks; surface streets less), not the generic urban ~12% the sim
/// once used; when the day clock runs, the truck share follows
/// [`TRUCK_HOURLY_X100`] — thin at the commute peaks, heavier overnight/midday.
fn class_of(seed: u64, id: u32, highway: bool, day_secs: Option<f64>) -> VehicleClass {
    // Buses are no longer a random background draw — the named transit lines
    // (`set_transit_lines`) run them on real routes and schedules.
    let (mut truck, bus) = if highway { (0.05, 0.0) } else { (0.03, 0.0) };
    if let Some(t) = day_secs {
        let h = (t.rem_euclid(86_400.0) / 3600.0) as usize % 24;
        truck = (truck * TRUCK_HOURLY_X100[h] as f64 / 100.0).min(0.3);
    }
    match rng::uniform01(seed, id, 0, Stream::DriverProfile) {
        u if u < 1.0 - truck - bus => VehicleClass::Car,
        u if u < 1.0 - bus => VehicleClass::Truck,
        _ => VehicleClass::Bus,
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::{DriverConfig, SimConfig};
    use super::super::map::{LinkSpec, NodeSpec, OsmMap};
    use super::super::network::LinkId;
    use super::*;

    fn corridor() -> OsmMap {
        OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::uncontrolled(2, 300.0, 0.0),
                NodeSpec::uncontrolled(3, 600.0, 0.0),
            ],
            links: vec![LinkSpec::oneway(1, 2, 1, 20.0), LinkSpec::oneway(2, 3, 1, 20.0)],
        }
    }

    #[test]
    fn generates_vehicles_at_roughly_the_requested_rate() {
        let net = corridor().build();
        let mut world = NetWorld::new(net, SimConfig::default_config());
        let pairs = [OdPair { origin: LinkId(0), dest: LinkId(1), rate_per_sec: 0.4, class: SurfaceClass::Through, anchored: false }];
        let mut demand = DemandGenerator::new(&world, &pairs, 7);
        world.install_router(&demand.destinations());

        for _ in 0..300 {
            demand.step(&mut world, 0.2);
            world.step();
        }

        // A class mix (slow trucks/buses) throttles a single entrance, so a
        // sustained-but-below-demand stream is the realistic outcome.
        assert!(demand.spawned() >= 8 && demand.spawned() <= 36,
            "≈0.4/s over 60 s, throttled by the entrance, got {}", demand.spawned());
        assert!(world.exited() > 0, "spawned vehicles should reach the destination");
    }

    #[test]
    fn flow_calibration_boosts_an_underfed_observed_corridor() {
        // "Route 9" carries an observed count far above what its configured
        // demand delivers; with calibration on, its stream's spawn rate is
        // nudged up (bounded), so the corridor carries measurably more traffic
        // than the uncalibrated baseline. An unnamed corridor is untouched.
        let build = |calibrate: bool| -> u32 {
            let mut map = corridor();
            for l in &mut map.links {
                l.name = "Route 9".into();
                l.aadt = 30_000.0; // demands ~vph the 0.02/s stream can't deliver
            }
            let net = map.build();
            let pairs = [OdPair { origin: LinkId(0), dest: LinkId(1), rate_per_sec: 0.02, class: SurfaceClass::Through, anchored: false }];
            let mut world = NetWorld::new(net, SimConfig::default_config());
            let mut gen = DemandGenerator::new(&world, &pairs, 7);
            world.install_router(&gen.destinations());
            if calibrate {
                gen.enable_flow_calibration(&world, 0.09);
            }
            // Long enough for several 900-day-second windows at 1:1 time.
            for _ in 0..22_000 {
                gen.step(&mut world, 0.2);
                world.step();
            }
            gen.spawned()
        };
        let (base, calibrated) = (build(false), build(true));
        assert!(
            calibrated as f64 > base as f64 * 1.4,
            "calibration lifts the underfed corridor: {base} → {calibrated}"
        );
    }

    #[test]
    fn rate_scale_gates_spawning() {
        let net = corridor().build();
        let pairs = [OdPair { origin: LinkId(0), dest: LinkId(1), rate_per_sec: 0.5, class: SurfaceClass::Through, anchored: false }];
        let mut world = NetWorld::new(net, SimConfig::default_config());
        let mut gen = DemandGenerator::new(&world, &pairs, 7);
        world.install_router(&gen.destinations());
        gen.set_rate_scale(0.0);
        for _ in 0..200 {
            gen.step(&mut world, 0.2);
            world.step();
        }
        assert_eq!(gen.spawned(), 0, "rate scale 0 stops all spawning");
        gen.set_rate_scale(2.0);
        for _ in 0..200 {
            gen.step(&mut world, 0.2);
            world.step();
        }
        assert!(gen.spawned() > 0, "restoring the rate resumes spawning");
    }

    #[test]
    fn demand_rate_scales_with_road_capacity() {
        // A 4-way with a high-capacity entry (3 lanes, 30 m/s — a freeway-ramp-like
        // road) and a low-capacity one (1 lane, 11 m/s — a local street). Demand
        // originating on the big road spawns far faster, reflecting real volumes.
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -200.0, 0.0), // H (high-capacity entry)
                NodeSpec::uncontrolled(2, 0.0, -200.0), // L (low-capacity entry)
                NodeSpec::uncontrolled(3, 0.0, 0.0),    // M (junction)
                NodeSpec::uncontrolled(4, 200.0, 0.0),  // E (exit)
                NodeSpec::uncontrolled(5, 0.0, 200.0),  // N (exit)
            ],
            links: vec![
                LinkSpec::oneway(1, 3, 3, 30.0), // 0: H→M, big road
                LinkSpec::oneway(2, 3, 1, 11.0), // 1: L→M, local
                LinkSpec::oneway(3, 4, 2, 20.0), // 2: M→E
                LinkSpec::oneway(3, 5, 2, 20.0), // 3: M→N
            ],
        }
        .build();
        assert!(capacity_rate(&net, LinkId(0)) > 3.0 * capacity_rate(&net, LinkId(1)),
            "the big road spawns far faster: {} vs {}", capacity_rate(&net, LinkId(0)), capacity_rate(&net, LinkId(1)));
        // boundary demand applies the capacity-scaled rate to each stream.
        for p in &boundary_od_pairs(&net, 5, 20) {
            assert_eq!(p.rate_per_sec, capacity_rate(&net, p.origin));
        }
    }

    /// A cross with two freeway gateways (fast entry/exit) and a slow surface
    /// street, all meeting a central junction — the shape needed to exercise the
    /// highway-biased mode's freeway origins.
    fn freeway_and_street() -> Network {
        OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -400.0, 0.0), // freeway gateway (W)
                NodeSpec::uncontrolled(2, 0.0, 0.0),    // interior junction
                NodeSpec::uncontrolled(3, 400.0, 0.0),  // freeway gateway (E)
                NodeSpec::uncontrolled(4, 0.0, 200.0),  // interior junction
                NodeSpec::uncontrolled(5, 0.0, 300.0),  // surface-street gateway (N)
            ],
            links: vec![
                LinkSpec::oneway(1, 2, 3, 29.0), // 0: freeway W→interior (entry)
                LinkSpec::oneway(2, 3, 3, 29.0), // 1: freeway interior→E (exit)
                LinkSpec::oneway(2, 4, 1, 13.0), // 2: interior surface link
                LinkSpec::oneway(4, 5, 1, 13.0), // 3: surface interior→N (slow exit)
            ],
        }
        .build()
    }

    #[test]
    fn highway_mode_originates_traffic_on_the_freeway() {
        let net = freeway_and_street();
        // Sanity: link 0 is a freeway entry, link 3 a surface exit.
        assert!(boundary::is_highway_link(&net, LinkId(0)));
        assert!(!boundary::is_highway_link(&net, LinkId(3)));

        let pairs = od_pairs(&net, 4, 40, DemandSources::new(true, false));
        assert!(!pairs.is_empty(), "highway mode yields demand");
        let hw_origin = pairs.iter().filter(|p| boundary::is_highway_link(&net, p.origin)).count();
        assert!(
            hw_origin * 2 > pairs.len(),
            "most trips originate on a freeway: {hw_origin} of {}",
            pairs.len()
        );
    }

    #[test]
    fn highway_entrants_enter_at_freeway_speed() {
        // A car entering on the freeway (link 0, 29 m/s) comes in near free-flow;
        // one entering on the surface street (link 2, 13 m/s) enters far slower —
        // so off-peak highway traffic streams in fast instead of crawling from 5 m/s.
        let net = freeway_and_street();
        let fast = entry_speed(&net, LinkId(0), &DriverConfig::car());
        let slow = entry_speed(&net, LinkId(2), &DriverConfig::car());
        assert!(fast > 25.0, "freeway entrants come in fast: {fast}");
        assert!(slow <= 13.0, "surface entrants enter at street speed: {slow}");
        assert!(fast > slow * 1.8, "highway traffic enters much faster than surface: {fast} vs {slow}");
    }

    #[test]
    fn highway_mode_falls_back_when_there_are_no_freeways() {
        // The plain corridor has no highway gateway, so highway mode degrades to the
        // balanced boundary mix rather than producing nothing.
        let net = corridor().build();
        let pairs = od_pairs(&net, 3, 10, DemandSources::new(true, false));
        assert!(!pairs.is_empty(), "no freeways → fall back to the balanced mix");
    }

    /// US-101 crossing the map (entry gateway → mid-freeway → exit gateway, all ref
    /// "US 101") with a surface off-ramp to a local street.
    fn freeway_corridor_with_ref() -> Network {
        let hw = |a, b, r: &str| LinkSpec {
            road_class: "motorway".into(),
            highway_ref: r.into(),
            ..LinkSpec::oneway(a, b, 3, 29.0)
        };
        OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -900.0, 0.0), // W freeway gateway
                NodeSpec::uncontrolled(2, -300.0, 0.0), // interior (mid-freeway)
                NodeSpec::uncontrolled(3, 300.0, 0.0),  // interior interchange
                NodeSpec::uncontrolled(4, 900.0, 0.0),  // E freeway gateway
                NodeSpec::uncontrolled(5, 300.0, 300.0), // interior surface node
                NodeSpec::uncontrolled(6, 300.0, 600.0), // S surface gateway
            ],
            links: vec![
                hw(1, 2, "US 101"),              // 0: entry W→interior
                hw(2, 3, "US 101"),              // 1: mid-freeway interior segment
                hw(3, 4, "US 101"),              // 2: interior→E exit (same highway)
                LinkSpec::oneway(3, 5, 1, 13.0), // 3: surface interior segment
                LinkSpec::oneway(5, 6, 1, 13.0), // 4: surface exit
            ],
        }
        .build()
    }

    #[test]
    fn highway_trips_run_same_highway_or_surface_never_midfreeway() {
        let net = freeway_corridor_with_ref();
        assert_eq!(boundary::highway_entry_links(&net), vec![LinkId(0)]);
        assert_eq!(boundary::highway_exit_links(&net), vec![LinkId(2)]);
        let surface: std::collections::HashSet<u32> =
            boundary::surface_interior_links(&net).iter().map(|l| l.0).collect();
        assert!(surface.contains(&3) && !surface.contains(&1), "link 1 is mid-freeway, not a surface dest");

        let mut pairs = Vec::new();
        highway_od_pairs(&net, 7, 200, DemandTuning::DEFAULT, &mut pairs);
        assert!(!pairs.is_empty(), "yields freeway demand");
        let mut same = 0;
        for p in &pairs {
            assert_eq!(p.origin, LinkId(0), "every highway trip enters at the freeway gateway (from outside)");
            assert!(
                p.dest == LinkId(2) || surface.contains(&p.dest.0),
                "destination is the highway exit or a surface street, never mid-freeway: {:?}",
                p.dest
            );
            assert_ne!(p.dest, LinkId(1), "no destination on the mid-freeway segment");
            same += (p.dest == LinkId(2)) as usize;
        }
        assert!(same * 2 > pairs.len(), "majority reach the far end of the same highway: {same}/{}", pairs.len());
    }

    /// A divided freeway: two one-way carriageways (NB, SB) sharing a route ref, joined
    /// by crossovers so the opposing exit is *routable* from either entry. A same-ref
    /// exit is only through-traffic when it continues the entry's direction — the
    /// opposing carriageway's exit is a U-turn and must never be a destination.
    fn divided_freeway_with_crossover() -> Network {
        let hw = |a, b| LinkSpec {
            road_class: "motorway".into(),
            highway_ref: "US 101".into(),
            ..LinkSpec::oneway(a, b, 3, 29.0)
        };
        OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, -900.0), // NB south gateway (enters heading N)
                NodeSpec::uncontrolled(2, 0.0, 0.0),    // NB interior
                NodeSpec::uncontrolled(3, 0.0, 900.0),  // NB north gateway (exits heading N)
                NodeSpec::uncontrolled(4, 50.0, 900.0), // SB north gateway (enters heading S)
                NodeSpec::uncontrolled(5, 50.0, 0.0),   // SB interior
                NodeSpec::uncontrolled(6, 50.0, -900.0), // SB south gateway (exits heading S)
            ],
            links: vec![
                hw(1, 2),                        // NB entry (heading +y)
                hw(2, 3),                        // NB exit  (heading +y)
                hw(4, 5),                        // SB entry (heading -y)
                hw(5, 6),                        // SB exit  (heading -y)
                LinkSpec::oneway(2, 5, 1, 13.0), // crossover NB→SB (makes the SB exit routable from the NB entry)
                LinkSpec::oneway(5, 2, 1, 13.0), // crossover SB→NB (and the NB exit from the SB entry)
            ],
        }
        .build()
    }

    #[test]
    fn highway_through_trips_never_u_turn_onto_the_opposing_carriageway() {
        let net = divided_freeway_with_crossover();
        let entries = boundary::highway_entry_links(&net);
        let exits: std::collections::HashSet<u32> =
            boundary::highway_exit_links(&net).iter().map(|l| l.0).collect();
        assert_eq!(entries.len(), 2, "two entry carriageways (NB, SB)");
        assert_eq!(exits.len(), 2, "two exit carriageways (NB, SB)");

        let dir = |l: LinkId| {
            let lk = net.link(l);
            let (a, b) = (net.node(lk.from).position, net.node(lk.to).position);
            [b[0] - a[0], b[1] - a[1]]
        };

        let mut pairs = Vec::new();
        highway_od_pairs(&net, 11, 300, DemandTuning::DEFAULT, &mut pairs);
        assert!(!pairs.is_empty(), "yields freeway demand");

        // Two origin classes now: gateway through/interchange traffic (freeway
        // carriageway origin) and on-ramp loading (a surface street origin routing
        // onto the freeway). The U-turn invariant concerns freeway-to-freeway
        // through-traffic, so it is checked only for gateway-origin pairs.
        let mut through = 0;
        let mut gateway_origin = 0;
        for p in &pairs {
            if entries.contains(&p.origin) {
                gateway_origin += 1;
                if exits.contains(&p.dest.0) {
                    let (e, x) = (dir(p.origin), dir(p.dest));
                    assert!(
                        e[0] * x[0] + e[1] * x[1] > 0.0,
                        "U-turn: entered heading {e:?} but exits heading {x:?} on the same freeway",
                    );
                    through += 1;
                }
            }
        }
        assert!(gateway_origin > 0, "gateway freeway traffic still runs");
        assert!(through > 0, "same-carriageway through-traffic still runs");
    }

    #[test]
    fn rush_hour_drives_gateways_from_the_real_profile() {
        // The 3-lane US-101 gateway (link 0, ref "US 101") should feed in exactly
        // 3 lanes × the route's per-lane hourly volume in aggregate — regardless of how
        // many destination pairs it fans out to — and follow the real curve over the day.
        let net = freeway_corridor_with_ref();
        let world = NetWorld::new(net, SimConfig::default_config());
        let mut pairs = Vec::new();
        highway_od_pairs(&world.network, 7, 60, DemandTuning::DEFAULT, &mut pairs);
        let mut gen = DemandGenerator::new(&world, &pairs, 7);

        gen.set_rush_hour(&world.network, true);
        assert!(gen.rush_hour_active());
        assert_eq!(gen.rush_hour_day_secs(), RUSH_START_HOUR * 3600.0, "clock starts pre-peak");

        let aggregate_at = |g: &DemandGenerator, secs: f64| {
            g.pairs
                .iter()
                .filter(|s| s.origin == LinkId(0))
                .map(|s| match s.rush.expect("freeway stream is on the profile") {
                    RushMode::Freeway(r) => r.lanes * rush_hour::interp(r.profile, secs) / 3600.0 * r.share,
                    RushMode::Surface => unreachable!("a freeway gateway stream is Freeway mode"),
                })
                .sum::<f64>()
        };
        let peak = aggregate_at(&gen, 7.0 * 3600.0);
        // The gateway (link 0) runs W→E, so its north-ward component is zero and the
        // direction resolves to southbound → the US-101 SB curve (1507 at 07:00).
        let expected = 3.0 * rush_hour::interp(&rush_hour::US101_S, 7.0 * 3600.0) / 3600.0;
        assert!((peak - expected).abs() < 1e-6, "AM-peak gateway inflow = 3 lanes × US-101 SB curve, got {peak}");
        // The curve varies through the day: the 3am trough is far below the 7am peak.
        assert!(aggregate_at(&gen, 3.0 * 3600.0) < peak * 0.2, "pre-dawn is a small fraction of the peak");

        gen.set_rush_hour(&world.network, false);
        assert!(!gen.rush_hour_active());
        assert!(gen.pairs.iter().all(|s| s.rush.is_none()), "toggling off restores the generic rate");
    }

    #[test]
    fn rush_hour_enters_the_freeway_slow_and_dense_at_the_peak() {
        // At the AM peak a freeway gateway should inject a slow, tightly packed stream — not
        // the off-peak fast, sparse trickle — because that is how a real congested freeway
        // meters in. Since a car is admitted only with `min_gap + speed·headway` of room,
        // the lower peak entry speed also shrinks the entry gap, so many more pack in at once.
        let net = freeway_corridor_with_ref();
        let world = NetWorld::new(net, SimConfig::default_config());
        let mut pairs = Vec::new();
        highway_od_pairs(&world.network, 7, 60, DemandTuning::DEFAULT, &mut pairs);
        let mut gen = DemandGenerator::new(&world, &pairs, 7);
        gen.set_rush_hour(&world.network, true);
        let stream = gen
            .pairs
            .iter()
            .position(|s| s.origin == LinkId(0) && matches!(s.rush, Some(RushMode::Freeway(_))))
            .expect("a freeway rush stream leaves the gateway");
        let driver = DriverConfig::car();
        let free_flow = entry_speed(&world.network, LinkId(0), &driver);

        gen.rush_clock = Some(3.0 * 3600.0); // pre-dawn trough
        let night = gen.launch_speed(&world.network, stream, LinkId(0), &driver);
        gen.rush_clock = Some(7.0 * 3600.0); // AM peak
        let peak = gen.launch_speed(&world.network, stream, LinkId(0), &driver);

        assert!((night - free_flow).abs() < 0.5, "off-peak enters at free-flow ({night} vs {free_flow})");
        assert!(peak < free_flow * 0.6, "the peak enters far slower than free-flow ({peak} vs {free_flow})");
        let gap = |v: f64| driver.min_gap + v * driver.time_headway;
        assert!(
            gap(peak) < gap(night) * 0.6,
            "the peak entry gap is far tighter, so many more cars pack in at once ({:.1} m vs {:.1} m)",
            gap(peak),
            gap(night),
        );
    }

    #[test]
    fn rush_hour_makes_surface_streams_breathe_with_the_arterial_curve() {
        // On a plain surface corridor (no freeway ref), rush hour puts every stream in
        // Surface mode — its base rate scaled by the arterial diurnal — so the surface
        // network peaks with the commute too, not just the freeways.
        let net = corridor().build(); // 1-lane, 20 m/s → surface, not highway
        let world = NetWorld::new(net, SimConfig::default_config());
        let pairs = boundary_od_pairs(&world.network, 3, 10);
        assert!(!pairs.is_empty(), "the corridor yields surface streams");
        let mut gen = DemandGenerator::new(&world, &pairs, 3);

        gen.set_rush_hour(&world.network, true);
        assert!(
            gen.pairs.iter().all(|s| matches!(s.rush, Some(RushMode::Surface))),
            "surface streams track the arterial diurnal under rush hour"
        );
        // The multiplier genuinely swings the rate: the PM peak is several× pre-dawn.
        assert!(rush_hour::arterial_factor(17.0 * 3600.0) > rush_hour::arterial_factor(3.0 * 3600.0) * 5.0);

        gen.set_rush_hour(&world.network, false);
        assert!(gen.pairs.iter().all(|s| s.rush.is_none()), "toggling off restores the flat rate");
    }

    #[test]
    fn gravity_favours_bigger_destinations_over_a_uniform_scatter() {
        // From one origin, two destinations equidistant but very different capacity: a
        // 3-lane, 25 m/s arterial vs a 1-lane, 11 m/s local street. The gravity draw
        // should pick the arterial the large majority of the time (capacity 75 vs 11 →
        // ~87%), where a uniform draw would be 50/50.
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(0, 0.0, 0.0),    // hub
                NodeSpec::uncontrolled(1, -200.0, 0.0), // origin start
                NodeSpec::uncontrolled(2, 0.0, 200.0),  // arterial dest end
                NodeSpec::uncontrolled(3, 0.0, -200.0), // local dest end (equidistant)
            ],
            links: vec![
                LinkSpec::oneway(1, 0, 1, 15.0), // origin approach
                LinkSpec::oneway(0, 2, 3, 25.0), // arterial exit — high capacity
                LinkSpec::oneway(0, 3, 1, 11.0), // local exit — low capacity
            ],
        }
        .build();
        let by = |lanes: u32, to_north: bool| {
            (0..net.links.len() as u32).map(LinkId).find(|&l| {
                let link = net.link(l);
                link.lane_count == lanes && (net.node(link.to).position[1] > 0.0) == to_north && net.node(link.from).position[0].abs() < 1.0
            })
        };
        let origin = (0..net.links.len() as u32).map(LinkId).find(|&l| net.node(net.link(l).from).position[0] < -100.0).unwrap();
        let (arterial, local) = (by(3, true).unwrap(), by(1, false).unwrap());
        let pool = vec![arterial, local];
        let big = (0..2000u64).filter(|&a| gravity_pick(&net, origin, &pool, 7, 5, a, GRAVITY_BETA, DemandTuning::DEFAULT) == Some(arterial)).count();
        assert!(big > 1600, "gravity strongly favours the higher-capacity arterial: {big}/2000");
    }

    #[test]
    fn excess_demand_queues_at_the_gateway_and_drains() {
        // A 1-lane gateway fed far faster than one lane can admit: trips that can't
        // enter wait in the gateway queue (conserved, not dropped) and the backlog
        // drains once demand eases — a real metered gateway queue.
        let net = corridor().build(); // 1-lane entry (link 0) → exit (link 1)
        let mut world = NetWorld::new(net, SimConfig::default_config());
        let pairs = vec![OdPair { origin: LinkId(0), dest: LinkId(1), rate_per_sec: 5.0, class: SurfaceClass::Through, anchored: false }];
        let mut gen = DemandGenerator::new(&world, &pairs, 1);
        world.install_router(&gen.destinations());
        for _ in 0..300 {
            gen.step(&mut world, 0.5);
            world.step();
        }
        assert!(gen.queued() > 0, "excess demand backs up at the gateway, got {}", gen.queued());

        // Ease off: with no new demand the backlog drains as the entrance clears, and
        // those waiting trips are admitted (spawned rises) rather than lost.
        gen.set_rate_scale(0.0);
        let (backlog, spawned_before) = (gen.queued(), gen.spawned());
        for _ in 0..600 {
            gen.step(&mut world, 0.5);
            world.step();
        }
        assert!(gen.queued() < backlog, "the queue drains once demand eases: {backlog} -> {}", gen.queued());
        assert!(gen.spawned() > spawned_before, "queued trips get admitted, not dropped");
    }

    /// A one-way five-node line: entry gateway link, three interior links, exit
    /// gateway link — the smallest network with all four boundary categories.
    fn line() -> OsmMap {
        OsmMap {
            nodes: (1..=6)
                .map(|i| NodeSpec::uncontrolled(i, (i as f64 - 1.0) * 250.0, 0.0))
                .collect(),
            links: (1..=5).map(|i| LinkSpec::oneway(i, i + 1, 1, 15.0)).collect(),
        }
    }

    #[test]
    fn boundary_categories_carry_their_diurnal_class() {
        let net = line().build();
        let mut pairs = Vec::new();
        surface_od_pairs(&net, 3, 40, DemandTuning::DEFAULT, &mut pairs);
        assert!(!pairs.is_empty());
        let (entry, exit) = (LinkId(0), LinkId(4));
        for p in &pairs {
            let expect = match (p.origin == entry, p.dest == exit) {
                (true, true) => SurfaceClass::Through,
                (true, false) => SurfaceClass::Inbound,
                (false, true) => SurfaceClass::Outbound,
                (false, false) => SurfaceClass::Internal,
            };
            assert_eq!(p.class, expect, "{:?}→{:?}", p.origin, p.dest);
        }
        let classes: std::collections::HashSet<_> = pairs.iter().map(|p| format!("{:?}", p.class)).collect();
        assert!(classes.len() >= 3, "the line yields several categories: {classes:?}");
    }

    #[test]
    fn platoons_bunch_arrivals_without_inflating_volume() {
        // One surface stream at 0.3 veh/s for 2000 s. Platooning must reshape
        // arrivals into occasional multi-vehicle firings while total demand stays
        // the calibrated volume.
        let net = corridor().build();
        let mut world = NetWorld::new(net, SimConfig::default_config());
        let pairs = [OdPair { origin: LinkId(0), dest: LinkId(1), rate_per_sec: 0.3, class: SurfaceClass::Through, anchored: false }];
        let mut gen = DemandGenerator::new(&world, &pairs, 11);
        world.install_router(&gen.destinations());

        let mut bursts = 0;
        for _ in 0..4000 {
            let before = gen.next_id;
            gen.step(&mut world, 0.5);
            world.step();
            if gen.next_id - before >= 2 {
                bursts += 1;
            }
        }
        assert!(bursts > 20, "platoon firings issue several ids at once: {bursts}");
        // Total fired demand ≈ rate × time (0.3 × 2000 = 600), despite the bunching.
        let fired = gen.next_id as f64;
        assert!((420.0..780.0).contains(&fired), "volume is conserved: fired {fired}, expected ≈600");
    }

    #[test]
    fn rate_modulation_wanders_but_averages_one() {
        let net = corridor().build();
        let world = NetWorld::new(net, SimConfig::default_config());
        let pairs = [OdPair { origin: LinkId(0), dest: LinkId(1), rate_per_sec: 0.3, class: SurfaceClass::Through, anchored: false }];
        let mut gen = DemandGenerator::new(&world, &pairs, 5);
        let mut vals = Vec::new();
        for k in 0..400 {
            gen.sim_secs = k as f64 * MODULATION_EPOCH_SECS * 0.5;
            vals.push(gen.modulation(0));
        }
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        assert!((mean - 1.0).abs() < 0.1, "long-run mean ≈ 1, got {mean}");
        let (lo, hi) = vals.iter().fold((f64::MAX, f64::MIN), |(l, h), &v| (l.min(v), h.max(v)));
        assert!(hi / lo > 1.5, "the level genuinely wanders: {lo:.2}..{hi:.2}");
    }

    #[test]
    fn daily_factor_varies_day_to_day_around_one() {
        let vals: Vec<f64> = (0..200).map(|d| daily_factor(9, d)).collect();
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        assert!((mean - 1.0).abs() < 0.05, "mean ≈ 1, got {mean}");
        assert!(vals.iter().any(|&v| v < 0.95) && vals.iter().any(|&v| v > 1.05), "days differ by ~10%");
    }

    #[test]
    fn rush_clock_rolls_days_into_weekends() {
        let net = corridor().build();
        let mut world = NetWorld::new(net, SimConfig::default_config());
        let pairs = [OdPair { origin: LinkId(0), dest: LinkId(1), rate_per_sec: 0.0, class: SurfaceClass::Through, anchored: false }];
        let mut gen = DemandGenerator::new(&world, &pairs, 5);
        world.install_router(&gen.destinations());
        gen.set_rush_hour(&world.network, true);
        assert!(!gen.weekend(), "the clock starts on a Monday");
        gen.rush_clock = Some(86_390.0);
        gen.step(&mut world, 0.5); // 0.5 s × 60 compression = 30 day-secs → wraps midnight
        assert_eq!(gen.day, 1, "midnight wrap advances the day");
        gen.day = 5;
        assert!(gen.weekend(), "day 5 is Saturday");
        gen.day = 7;
        assert!(!gen.weekend(), "day 7 is Monday again");
    }

    #[test]
    fn day_compression_is_parameterized_and_scales_only_the_day_clock() {
        let net = corridor().build();
        let mut world = NetWorld::new(net, SimConfig::default_config());
        let pairs = [OdPair { origin: LinkId(0), dest: LinkId(1), rate_per_sec: 0.0, class: SurfaceClass::Through, anchored: false }];
        let mut gen = DemandGenerator::new(&world, &pairs, 5);
        world.install_router(&gen.destinations());
        gen.set_rush_hour(&world.network, true);

        assert_eq!(gen.day_compression(), DEFAULT_DAY_COMPRESSION);
        gen.set_day_compression(0.01);
        assert_eq!(gen.day_compression(), MIN_DAY_COMPRESSION);
        gen.set_day_compression(1e9);
        assert_eq!(gen.day_compression(), MAX_DAY_COMPRESSION);

        // 1×: the day clock advances exactly with sim time; 60×: sixty day-seconds
        // per sim second. Dynamics-time state (tick, sim_secs) advances identically.
        gen.set_day_compression(1.0);
        let t0 = gen.rush_hour_day_secs();
        gen.step(&mut world, 0.5);
        assert!((gen.rush_hour_day_secs() - t0 - 0.5).abs() < 1e-9);
        gen.set_day_compression(60.0);
        let t1 = gen.rush_hour_day_secs();
        gen.step(&mut world, 0.5);
        assert!((gen.rush_hour_day_secs() - t1 - 30.0).abs() < 1e-9);
    }

    #[test]
    fn resume_clock_carries_the_day_across_a_rebuild() {
        let net = corridor().build();
        let world = NetWorld::new(net, SimConfig::default_config());
        let pairs = [OdPair { origin: LinkId(0), dest: LinkId(1), rate_per_sec: 0.0, class: SurfaceClass::Through, anchored: false }];
        let mut gen = DemandGenerator::new(&world, &pairs, 5);
        // Off-mode: nothing to resume onto (the clock stays off).
        gen.resume_clock(43_200.0, 3);
        assert!(!gen.rush_hour_active());
        gen.set_rush_hour(&world.network, true);
        gen.resume_clock(43_200.0, 3);
        assert_eq!(gen.rush_hour_day_secs(), 43_200.0, "a rebuilt generator picks the day clock back up");
        assert_eq!(gen.day(), 3);
    }

    #[test]
    fn origin_churn_swaps_same_class_surface_origins() {
        let net = line().build();
        let mut world = NetWorld::new(net, SimConfig::default_config());
        // Two Internal streams with distinct origins and distinct (cross-reachable)
        // destinations further down the line.
        let pairs = [
            OdPair { origin: LinkId(1), dest: LinkId(3), rate_per_sec: 0.2, class: SurfaceClass::Internal, anchored: false },
            OdPair { origin: LinkId(2), dest: LinkId(4), rate_per_sec: 0.4, class: SurfaceClass::Internal, anchored: false },
        ];
        let mut gen = DemandGenerator::new(&world, &pairs, 5);
        world.install_router(&gen.destinations());
        gen.set_rate_scale(0.0); // isolate churn from spawning
        for _ in 0..100 {
            gen.step(&mut world, 0.5); // 50 s crosses one churn window
            world.step();
        }
        assert_eq!(gen.pairs[0].origin, LinkId(2), "stream 0 took stream 1's origin");
        assert_eq!(gen.pairs[1].origin, LinkId(1), "stream 1 took stream 0's origin");
        assert_eq!(gen.pairs[0].base_rate, 0.4, "the calibrated rate travels with the origin");
        assert_eq!(gen.pairs[0].dest, LinkId(3), "destinations never move, so no new flow field");
    }

    #[test]
    fn vehicle_mix_varies_by_origin_type_and_hour() {
        let share = |highway, day_secs| {
            let n = 20_000u32;
            let trucks = (0..n).filter(|&id| class_of(3, id, highway, day_secs) == VehicleClass::Truck).count();
            trucks as f64 / n as f64
        };
        let surface = share(false, None);
        let freeway = share(true, None);
        assert!((0.02..0.045).contains(&surface), "surface trucks ≈ 3%: {surface}");
        assert!((0.035..0.065).contains(&freeway), "freeway trucks ≈ 5% (Caltrans truck AADT, US-101 SM): {freeway}");
        // Trucks thin out at the PM commute peak and thicken overnight.
        let peak = share(false, Some(17.0 * 3600.0));
        let night = share(false, Some(3.0 * 3600.0));
        assert!(peak < surface * 0.75, "peak-hour trucks are scarce: {peak}");
        assert!(night > surface * 1.3, "overnight trucks are heavy: {night}");
    }

    /// The [`line`] with both directions drivable, so commute streams can run
    /// home→work AND the evening reverse.
    fn twoway_line() -> OsmMap {
        OsmMap {
            nodes: (1..=6)
                .map(|i| NodeSpec::uncontrolled(i, (i as f64 - 1.0) * 250.0, 0.0))
                .collect(),
            links: (1..=5).flat_map(|i| LinkSpec::twoway(i, i + 1, 1, 15.0)).collect(),
        }
    }

    /// One measured flow: 1000 commuters/day from a home cell (x≈375) to a work
    /// cell (x≈875).
    fn commute_fixture() -> CommuteOd {
        CommuteOd {
            grid_m: 250.0,
            cells: vec![[375.0, 0.0], [875.0, 0.0]],
            flows: vec![(0, 1, 1000.0)],
        }
    }

    #[test]
    fn commute_flows_seed_am_pm_paired_streams() {
        let net = twoway_line().build();
        let od = commute_fixture();
        let mut pairs = Vec::new();
        commute_od_pairs(&net, &od, 7, 4, DemandTuning::DEFAULT, &mut pairs);
        assert_eq!(pairs.len(), 4, "each sampled flow yields an AM and a PM stream");

        let x = |l: LinkId| link_centroid(&net, l)[0];
        let mut inbound_rate = 0.0;
        for p in &pairs {
            assert!(p.anchored, "commute streams are pinned to measured geography");
            // Anchors now spread across each cell's candidate streets (class-
            // weighted) rather than pinning to the single nearest link, so the
            // invariant is the trip *direction* (home is west, work is east), not a
            // strict midpoint threshold.
            match p.class {
                SurfaceClass::Inbound => {
                    assert!(x(p.origin) <= x(p.dest), "AM runs home(west)→work(east): {} → {}", x(p.origin), x(p.dest));
                    inbound_rate += p.rate_per_sec;
                }
                SurfaceClass::Outbound => {
                    assert!(x(p.origin) >= x(p.dest), "PM runs work(east)→home(west): {} → {}", x(p.origin), x(p.dest));
                }
                c => panic!("commute stream carries a commute shape, got {c:?}"),
            }
        }
        // The sampled set carries the measured volume: 1000 trips/day toward work.
        assert!((inbound_rate - 1000.0 / 86_400.0).abs() < 1e-9, "measured volume survives sampling: {inbound_rate}");

        // The generator accepts them (both directions routable on the two-way line)
        // and never churns their origins away from the measured cells.
        let world = NetWorld::new(net, SimConfig::default_config());
        let gen = DemandGenerator::new(&world, &pairs, 7);
        assert_eq!(gen.pairs.len(), 4, "all commute streams are reachable");
        assert!(gen.pairs.iter().all(|s| s.anchored && s.surface));
    }

    #[test]
    fn commute_displaces_sampled_volume_never_coverage() {
        let net = twoway_line().build();
        let od = commute_fixture();
        let plain = od_pairs(&net, 3, 40, DemandSources::new(false, true));
        let pairs = od_pairs_with_commute(&net, 3, 40, DemandSources::new(false, true), Some(&od));
        let (anchored, sampled): (Vec<_>, Vec<_>) = pairs.iter().partition(|p| p.anchored);
        assert!(!anchored.is_empty(), "measured commute streams are present");
        assert_eq!(sampled.len(), plain.len(), "every sampled stream survives — coverage is never thinned");
        let vol = |ps: &[&OdPair]| ps.iter().map(|p| p.rate_per_sec).sum::<f64>();
        let plain_vol: f64 = plain.iter().map(|p| p.rate_per_sec).sum();
        let (sampled_vol, measured_vol) = (vol(&sampled), vol(&anchored));
        assert!(sampled_vol < plain_vol, "sampled rates yield the measured volume's share");
        assert!(sampled_vol >= 0.3 * plain_vol - 1e-12, "floored so non-work trips never vanish");
        assert!(plain_vol - sampled_vol <= measured_vol + 1e-12, "displacement never exceeds the measured volume");
        // Without commute data the same call is the plain category mix.
        assert!(plain.iter().all(|p| !p.anchored));
    }

    #[cfg(feature = "import")]
    #[test]
    fn commute_od_parses_and_validates_json() {
        let doc = r#"{
            "meta": {"state": "ri", "year": 2022, "grid_m": 500.0, "jobs": 77},
            "cells": [[100.0, 200.0], [600.0, -100.0]],
            "flows": [[0, 1, 50], [1, 0, 20], [0, 9, 7], [1, 1, 0]]
        }"#;
        let od = CommuteOd::from_json(doc).expect("valid commute json");
        assert_eq!(od.grid_m, 500.0);
        assert_eq!(od.cells.len(), 2);
        assert_eq!(od.flows, vec![(0, 1, 50.0), (1, 0, 20.0)], "out-of-range and zero flows dropped");
        assert!(CommuteOd::from_json("not json").is_err());
    }

    #[test]
    fn land_use_weights_tilt_origins_and_attractions() {
        // Two equal entry roads — one in residential fabric, one not — and two
        // equal exits — one by a commercial strip, one plain. Sampling must send
        // most trips home→shops instead of a uniform scatter.
        let mut spec = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -300.0, 100.0),
                NodeSpec::uncontrolled(2, -300.0, -100.0),
                NodeSpec::uncontrolled(3, 0.0, 0.0),
                NodeSpec::uncontrolled(4, 300.0, 100.0),
                NodeSpec::uncontrolled(5, 300.0, -100.0),
            ],
            links: vec![
                LinkSpec::oneway(1, 3, 1, 15.0), // 0: residential entry
                LinkSpec::oneway(2, 3, 1, 15.0), // 1: plain entry
                LinkSpec::oneway(3, 4, 1, 15.0), // 2: commercial exit
                LinkSpec::oneway(3, 5, 1, 15.0), // 3: plain exit
            ],
        };
        spec.links[0].res_weight = 1.7;
        spec.links[1].res_weight = 0.3;
        spec.links[2].attr_weight = 3.0;
        spec.links[3].attr_weight = 1.0;
        let net = spec.build();
        let reach = Reachability::new(&net);
        let mut pairs = Vec::new();
        sample_pairs(&net, &reach, 7, 40, &[LinkId(0), LinkId(1)], &[LinkId(2), LinkId(3)], 300, SurfaceClass::Through, DemandTuning::DEFAULT, &mut pairs);
        let n = pairs.len();
        assert!(n >= 250, "sampling fills the request: {n}");
        let res_origins = pairs.iter().filter(|p| p.origin == LinkId(0)).count();
        let comm_dests = pairs.iter().filter(|p| p.dest == LinkId(2)).count();
        assert!(res_origins * 3 > n * 2, "the residential origin dominates: {res_origins}/{n}");
        assert!(comm_dests * 3 > n * 2, "the commercial destination dominates: {comm_dests}/{n}");
        // Neutral maps (no land-use pass) keep every weight at 1.0.
        assert_eq!(corridor().build().link_res_weight(LinkId(0)), 1.0);
    }

    #[test]
    fn observed_aadt_calibrates_rate_and_attraction() {
        let mut spec = corridor();
        spec.links[0].aadt = 26_500.0; // an El Camino-scale count attached to the entry
        let net = spec.build();
        // The origin's spawn rate becomes the road's real daily-mean directional flow…
        let rate = capacity_rate(&net, LinkId(0));
        assert!((rate - 26_500.0 * 0.5 / 86_400.0).abs() < 1e-9, "observed rate = AADT/2 over 24 h: {rate}");
        // …and its attraction weight comes from the count, not the lanes×speed proxy.
        assert!((link_capacity(&net, LinkId(0)) - 26.5).abs() < 1e-9);
        // Uncounted links keep the proxy behaviour.
        assert_eq!(net.link_aadt(LinkId(1)), 0.0);
        assert_eq!(link_capacity(&net, LinkId(1)), 20.0);
    }

    #[test]
    fn boundary_pairs_run_from_entry_to_exit_gateways() {
        let net = corridor().build();
        let pairs = boundary_od_pairs(&net, 3, 10);
        assert!(!pairs.is_empty(), "the corridor's two gateways yield through demand");
        for p in &pairs {
            assert_eq!(p.origin, LinkId(0), "origin is the entry gateway link");
            assert_eq!(p.dest, LinkId(1), "destination is the exit gateway link");
        }
    }
}
