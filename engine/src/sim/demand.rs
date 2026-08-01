//! Origin–destination travel demand: the "who goes where, when" that carries
//! most of the perceived realism at city scale. Boundary-aware categories
//! (through / inbound / outbound / internal) place origins and destinations at
//! the map's gateways and interior, and each spawn is a Bernoulli draw on the
//! stateless RNG. Vehicles are handed a *destination* and routed live by the
//! world's flow field (rerouting around jams), falling back to a precomputed
//! route when the world has no router for that destination.

use super::boundary;
use super::config::VehicleClass;
use super::net_world::NetWorld;
use super::network::{LinkId, Network};
use super::rng::{self, Stream};
use super::rush_hour;

#[derive(Debug)]
pub struct OdPair {
    pub origin: LinkId,
    pub dest: LinkId,
    pub rate_per_sec: f64,
}

/// Which streams of traffic to spawn — the independent toggles the UI exposes, so
/// they compose. Freeway traffic enters at highway gateways bound for the far end of
/// its highway (or another highway exit, or a surface street it leaves the freeway
/// for); surface traffic is the local/arterial boundary mix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DemandSources {
    pub highway: bool,
    pub surface: bool,
    /// Drive the whole map by the simulated time of day: freeway gateways at their
    /// real per-lane PeMS volumes and surface streets by the arterial diurnal shape,
    /// so the network builds and fades the way a real peak commute does.
    pub rush_hour: bool,
}

impl DemandSources {
    pub const fn new(highway: bool, surface: bool) -> Self {
        Self { highway, surface, rush_hour: false }
    }

    pub const fn with_rush_hour(highway: bool, surface: bool, rush_hour: bool) -> Self {
        Self { highway, surface, rush_hour }
    }
}

/// OD demand for the enabled sources — the union of the freeway and surface streams,
/// with `target` trips split across whichever are on. Falls back to the plain
/// boundary mix only if the enabled sources yield nothing (e.g. no gateways at all).
pub fn od_pairs(net: &Network, seed: u64, target: usize, sources: DemandSources) -> Vec<OdPair> {
    let n = (sources.highway as usize) + (sources.surface as usize);
    if n == 0 {
        return Vec::new();
    }
    let per = (target / n).max(1);
    let mut pairs = Vec::new();
    if sources.highway {
        highway_od_pairs(net, seed, per, &mut pairs);
    }
    if sources.surface {
        surface_od_pairs(net, seed, per, &mut pairs);
    }
    if pairs.is_empty() {
        pairs = boundary_od_pairs(net, seed, target);
    }
    calibrate_origin_inflow(&mut pairs);
    pairs
}

/// Hold each origin's *aggregate* off-peak inflow to one `capacity_rate` by
/// splitting it across the pairs sharing that origin (each already carries the full
/// rate). Without this, a gateway that happens to seed many OD pairs would inject
/// `pairs × rate` — harmless while injection stacked everything on one lane and the
/// entrance throttled it, but a flood once inflow fills all lanes. Mirrors the
/// rush-hour path, which already divides a gateway's volume by its pair `share`.
fn calibrate_origin_inflow(pairs: &mut [OdPair]) {
    let mut count: std::collections::HashMap<LinkId, usize> = std::collections::HashMap::new();
    for p in pairs.iter() {
        *count.entry(p.origin).or_insert(0) += 1;
    }
    for p in pairs.iter_mut() {
        p.rate_per_sec /= count[&p.origin] as f64;
    }
}

/// Hour of day (0–24) the rush-hour clock starts at — mid pre-peak build-up, so the
/// morning ramp is imminent when the mode is switched on.
const RUSH_START_HOUR: f64 = 5.5;
/// Simulated day-seconds elapsed per second of sim time: the 24 h profile plays over
/// ~24 min of sim time, fast enough to watch the peak build and fade while vehicles
/// still have real time to form and clear the queues it creates.
const RUSH_DAY_COMPRESSION: f64 = 60.0;

/// One origin→destination stream and how fast it spawns. Off-peak it fires at a
/// fixed `base_rate`; under rush hour `rush` makes it follow the time of day.
struct OdStream {
    origin: LinkId,
    dest: LinkId,
    /// Generic (off-peak) spawn rate, veh/sec.
    base_rate: f64,
    /// Set when rush hour is on: how this stream tracks the simulated time of day.
    rush: Option<RushMode>,
}

/// How a stream's spawn rate varies through the day under rush hour.
#[derive(Clone, Copy)]
enum RushMode {
    /// Freeway gateway: its lane count × the route's real per-lane PeMS hourly volume,
    /// split across the gateway's pairs so the aggregate stays calibrated to real data.
    Freeway(RushRate),
    /// Surface street: the calibrated `base_rate` scaled by the arterial diurnal
    /// multiplier (see [`rush_hour::arterial_factor`]) so it breathes with the commute.
    Surface,
}

#[derive(Clone, Copy)]
struct RushRate {
    lanes: f64,
    profile: &'static [u16; 24],
    /// This stream's fraction (1 / pairs-sharing-origin) of its gateway's inflow.
    share: f64,
}

pub struct DemandGenerator {
    /// OD streams with at least one valid route.
    pairs: Vec<OdStream>,
    seed: u64,
    tick: u64,
    next_id: u32,
    spawned: u32,
    /// Global multiplier on every stream's spawn rate (the UI frequency control).
    rate_scale: f64,
    /// Cap (m/s) on the speed a vehicle enters the map at, applied on top of the
    /// origin road's limit and the driver's desired speed (the UI start-speed control).
    entry_speed_cap: f64,
    /// Simulated seconds-into-day while the rush-hour profile is driving the freeway
    /// streams; `None` off-peak. Advances by [`RUSH_DAY_COMPRESSION`] each sim second.
    rush_clock: Option<f64>,
    /// Per-gateway (origin link) backlog of fired-but-not-yet-admitted trips
    /// `(stream, id)`. When demand outruns what the entrance can accept, trips wait
    /// here instead of being dropped, and are released FIFO as lanes clear — a real
    /// metered gateway queue that conserves demand and grows/dissipates with the peak.
    queues: std::collections::BTreeMap<u32, std::collections::VecDeque<(usize, u32)>>,
}

/// Max backlog held per gateway before further trips spill (are dropped) — the
/// off-map storage a gateway queue can represent before it backs out of the region.
const MAX_QUEUE: usize = 400;

impl DemandGenerator {
    pub fn new(world: &NetWorld, pairs: &[OdPair], seed: u64) -> Self {
        let pairs = pairs
            .iter()
            .filter(|p| world.network.route_links(p.origin, p.dest).is_some())
            .map(|p| OdStream { origin: p.origin, dest: p.dest, base_rate: p.rate_per_sec, rush: None })
            .collect();
        Self {
            pairs, seed, tick: 0, next_id: 0, spawned: 0,
            rate_scale: 1.0, entry_speed_cap: f64::INFINITY, rush_clock: None,
            queues: std::collections::BTreeMap::new(),
        }
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
            self.pairs[i].rush = Some(if boundary::is_highway_link(net, o) {
                let n = self.pairs.iter().filter(|s| s.origin == o).count();
                // The gateway's travel direction picks the matching directional curve,
                // so an inbound and an outbound freeway gate peak at different times.
                let northbound = net.node(net.link(o).to).position[1] > net.node(net.link(o).from).position[1];
                RushMode::Freeway(RushRate {
                    lanes: net.link(o).lane_count as f64,
                    profile: rush_hour::profile_for(net.link_ref(o), northbound),
                    share: 1.0 / n as f64,
                })
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

    /// Scale every stream's spawn rate (1.0 = as configured; 0.0 = no spawning).
    pub fn set_rate_scale(&mut self, scale: f64) {
        self.rate_scale = scale.max(0.0);
    }

    /// Cap the entry speed (m/s); vehicles still never exceed the origin road's limit
    /// or the driver's desired speed. `f64::INFINITY` = enter at the road's limit.
    pub fn set_entry_speed_cap(&mut self, cap: f64) {
        self.entry_speed_cap = cap.max(0.0);
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
        let costs = world.live_link_costs();
        let day = self.rush_clock;

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
        //    can't admit joins its gateway's queue rather than vanishing.
        for i in 0..self.pairs.len() {
            let s = &self.pairs[i];
            let rate = match (day, s.rush) {
                (Some(t), Some(RushMode::Freeway(r))) => r.lanes * rush_hour::interp(r.profile, t) / 3600.0 * r.share,
                (Some(t), Some(RushMode::Surface)) => s.base_rate * rush_hour::arterial_factor(t),
                _ => s.base_rate,
            };
            if rng::uniform01(self.seed, i as u32, self.tick, Stream::RouteChoice) >= (rate * self.rate_scale * dt).min(1.0) {
                continue;
            }
            let (origin, id) = (s.origin, self.next_id);
            self.next_id += 1;
            if self.launch(world, &costs, i, id) {
                self.spawned += 1;
            } else {
                let q = self.queues.entry(origin.0).or_default();
                if q.len() < MAX_QUEUE {
                    q.push_back((i, id));
                }
            }
        }

        if let Some(t) = &mut self.rush_clock {
            *t = (*t + dt * RUSH_DAY_COMPRESSION).rem_euclid(86_400.0);
        }
        self.tick += 1;
    }

    /// Attempt to admit trip `id` of stream `stream` at its origin: destination-routed
    /// via the live flow-field when the router knows the destination, else on a
    /// cost-routed link path. `false` if the entrance is occupied (try again later) or
    /// no route exists. Driver and entry speed are derived from `id`, so a queued trip
    /// keeps its identity across the wait.
    fn launch(&self, world: &mut NetWorld, costs: &[u64], stream: usize, id: u32) -> bool {
        let (origin, dest) = (self.pairs[stream].origin, self.pairs[stream].dest);
        let driver = class_of(self.seed, id).driver().sample(self.seed, id);
        let speed = entry_speed(&world.network, origin, &driver).min(self.entry_speed_cap);
        if world.router_knows(dest) {
            world.spawn_to(id, origin, dest, speed, driver)
        } else if let Some(route) = world.network.route_links_with_costs(origin, dest, costs) {
            world.spawn_routed(id, route, speed, driver)
        } else {
            false
        }
    }
}

/// Boundary-aware OD pairs: traffic passing through the map (gateway→gateway),
/// arriving (gateway→interior), leaving (interior→gateway), and staying within
/// (interior→interior). Deterministic in `seed`; falls back to any-link pairs
/// when the map has no gateways to anchor the categories.
pub fn boundary_od_pairs(net: &Network, seed: u64, target: usize) -> Vec<OdPair> {
    let entries = boundary::entry_links(net);
    let exits = boundary::exit_links(net);
    let interior = boundary::interior_links(net);
    let categories: [(&[LinkId], &[LinkId], f64); 4] = [
        (&entries, &exits, 0.45),
        (&entries, &interior, 0.20),
        (&interior, &exits, 0.20),
        (&interior, &interior, 0.15),
    ];

    let mut pairs = Vec::new();
    for (cat, (origins, dests, share)) in categories.iter().enumerate() {
        if origins.is_empty() || dests.is_empty() {
            continue;
        }
        let want = ((target as f64 * share).round() as usize).max(1);
        sample_pairs(net, seed, cat as u32, origins, dests, want, &mut pairs);
    }
    if pairs.is_empty() {
        let all: Vec<LinkId> = (0..net.links.len() as u32).map(LinkId).collect();
        sample_pairs(net, seed, 99, &all, &all, target, &mut pairs);
    }
    pairs
}

/// Freeway traffic: each trip enters at a highway gateway (from outside the map) and
/// is bound for — by likelihood — the far end of the *same* highway (through-traffic),
/// another highway exit (an interchange), or a surface street it leaves the freeway
/// for. Never a mid-freeway segment. Same-highway matching uses the OSM route `ref`;
/// a map without refs or freeway gateways contributes nothing (the caller falls back).
pub fn highway_od_pairs(net: &Network, seed: u64, target: usize, out: &mut Vec<OdPair>) {
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
                if d == e || net.route_links(e, d).is_none() {
                    continue;
                }
                if !r.is_empty() && same_highway(net.link_ref(d), r) {
                    same.push(d);
                } else {
                    other.push(d);
                }
            }
            (e, same, other)
        })
        .collect();

    let mut found = 0usize;
    let mut attempt = 0u64;
    while found < target && attempt < target as u64 * 40 + 400 {
        let (e, same, other) = &pools[(rng::hash(seed, 60, attempt, Stream::RouteChoice) as usize) % pools.len()];
        let r = rng::uniform01(seed, e.0, attempt, Stream::RouteChoice);
        // Weighted by category, cascading when a pool is empty so the majority still
        // lands on the same highway wherever refs make it possible.
        let dest = if r < 0.60 {
            pick(same, seed, attempt)
                .or_else(|| pick(other, seed, attempt))
                .or_else(|| pick_routable(net, *e, &surface, seed, attempt))
        } else if r < 0.75 {
            pick(other, seed, attempt)
                .or_else(|| pick(same, seed, attempt))
                .or_else(|| pick_routable(net, *e, &surface, seed, attempt))
        } else {
            pick_routable(net, *e, &surface, seed, attempt)
                .or_else(|| pick(same, seed, attempt))
                .or_else(|| pick(other, seed, attempt))
        };
        attempt += 1;
        if let Some(d) = dest {
            if d != *e {
                out.push(OdPair { origin: *e, dest: d, rate_per_sec: capacity_rate(net, *e) });
                found += 1;
            }
        }
    }
}

/// Surface (local/arterial) traffic: the boundary mix over non-freeway gateways and
/// interior streets — through the city, inbound, outbound, and internal trips.
pub fn surface_od_pairs(net: &Network, seed: u64, target: usize, out: &mut Vec<OdPair>) {
    let entries = boundary::surface_entry_links(net);
    let exits = boundary::surface_exit_links(net);
    let interior = boundary::surface_interior_links(net);
    let categories: [(&[LinkId], &[LinkId], f64); 4] = [
        (&entries, &exits, 0.35),     // through the city
        (&entries, &interior, 0.25),  // arriving to a local destination
        (&interior, &exits, 0.20),    // leaving town
        (&interior, &interior, 0.20), // internal local trips
    ];
    for (cat, (origins, dests, share)) in categories.iter().enumerate() {
        if origins.is_empty() || dests.is_empty() {
            continue;
        }
        let want = ((target as f64 * share).round() as usize).max(1);
        sample_pairs(net, seed, 80 + cat as u32, origins, dests, want, out);
    }
}

/// Pick a random link from a small, pre-route-checked pool.
fn pick(pool: &[LinkId], seed: u64, attempt: u64) -> Option<LinkId> {
    (!pool.is_empty()).then(|| pool[(rng::hash(seed, 61, attempt, Stream::RouteChoice) as usize) % pool.len()])
}

/// Gravity-model distance-decay exponent: trip attraction falls as `1/dist^β`. 1.0 is
/// at the gentle end of the standard urban range (~1.0–2.0) — it still favours nearer,
/// bigger roads, but a steeper decay over-concentrates trips onto a handful of nearby
/// destinations, bursting demand into merges hard enough to expose collisions.
const GRAVITY_BETA: f64 = 1.0;
/// Distance floor (m) so co-located destinations don't get an unbounded weight.
const GRAVITY_MIN_DIST: f64 = 40.0;

/// A link's midpoint, the point trips gravitate toward / from.
fn link_centroid(net: &Network, link: LinkId) -> [f64; 2] {
    let l = net.link(link);
    let (a, b) = (net.node(l.from).position, net.node(l.to).position);
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
}

/// Gravity draw from `pool`: destination `d` is chosen with probability ∝
/// `link_capacity(d) / dist(o, d)^β` — the classic gravity model, so trips prefer
/// bigger (higher-capacity, AADT-correlated) roads and nearer ones over a uniform
/// scatter. `None` only for an empty pool.
fn gravity_pick(net: &Network, o: LinkId, pool: &[LinkId], seed: u64, salt: u32, attempt: u64) -> Option<LinkId> {
    if pool.is_empty() {
        return None;
    }
    let op = link_centroid(net, o);
    let mut total = 0.0;
    let mut cum = Vec::with_capacity(pool.len());
    for &d in pool {
        let dp = link_centroid(net, d);
        let dist = (op[0] - dp[0]).hypot(op[1] - dp[1]).max(GRAVITY_MIN_DIST);
        total += link_capacity(net, d) / dist.powf(GRAVITY_BETA);
        cum.push(total);
    }
    if total <= 0.0 {
        return None;
    }
    let r = rng::uniform01(seed, salt, attempt, Stream::RouteChoice) * total;
    Some(pool[cum.partition_point(|&c| c < r).min(pool.len() - 1)])
}

/// Gravity-weighted pick from `pool` that is routable from `e` (a few candidate tries).
fn pick_routable(net: &Network, e: LinkId, pool: &[LinkId], seed: u64, attempt: u64) -> Option<LinkId> {
    for k in 0..8u64 {
        let d = gravity_pick(net, e, pool, seed, 62, attempt.wrapping_mul(8).wrapping_add(k))?;
        if d != e && net.route_links(e, d).is_some() {
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

fn sample_pairs(
    net: &Network,
    seed: u64,
    salt: u32,
    origins: &[LinkId],
    dests: &[LinkId],
    want: usize,
    out: &mut Vec<OdPair>,
) {
    let mut found = 0;
    let mut attempt = 0u64;
    while found < want && attempt < want as u64 * 40 + 200 {
        let o = origins[(rng::hash(seed, salt * 2, attempt, Stream::RouteChoice) as usize) % origins.len()];
        // Destination by gravity (bigger/nearer roads win), not a uniform scatter.
        let d = gravity_pick(net, o, dests, seed, salt * 2 + 1, attempt);
        attempt += 1;
        if let Some(d) = d {
            if o != d && net.route_links(o, d).is_some_and(|r| r.len() >= 2) {
                out.push(OdPair { origin: o, dest: d, rate_per_sec: capacity_rate(net, o) });
                found += 1;
            }
        }
    }
}

/// Road capacity of a link as `lanes × speed_limit` (m/s) — a proxy for AADT that
/// correlates with observed volumes (Caltrans counts), so demand favours the
/// high-capacity roads (a freeway ramp / arterial like El Camino Real carries far
/// more than a residential street). Calibratable against real counts later.
fn link_capacity(net: &Network, link: LinkId) -> f64 {
    let l = net.link(link);
    l.lane_count as f64 * net.lane(l.lane_start).speed_limit
}

/// Spawn rate for a stream originating on `origin`, scaled by its road capacity
/// relative to a typical arterial and clamped so no single road dominates or
/// starves.
fn capacity_rate(net: &Network, origin: LinkId) -> f64 {
    const BASE: f64 = 0.2;
    const REFERENCE: f64 = 30.0; // ~ a 2-lane, 15 m/s arterial
    (BASE * link_capacity(net, origin) / REFERENCE).clamp(0.04, 0.9)
}

/// The speed a vehicle enters the map at: the free-flow speed of its origin road
/// (its posted limit, capped by the driver's own desired speed), so a car arriving
/// on a freeway is already moving at freeway speed instead of crawling up from a
/// standstill. Off-peak traffic entering from US-101 / I-280 comes in fast.
fn entry_speed(net: &Network, origin: LinkId, driver: &super::config::DriverConfig) -> f64 {
    let limit = net.lane(net.link(origin).lane_start).speed_limit;
    limit.min(driver.desired_speed)
}

/// Vehicle-class mix: mostly cars, some trucks, a few buses.
fn class_of(seed: u64, id: u32) -> VehicleClass {
    match rng::uniform01(seed, id, 0, Stream::DriverProfile) {
        u if u < 0.85 => VehicleClass::Car,
        u if u < 0.97 => VehicleClass::Truck,
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
        let pairs = [OdPair { origin: LinkId(0), dest: LinkId(1), rate_per_sec: 0.4 }];
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
    fn rate_scale_gates_spawning() {
        let net = corridor().build();
        let pairs = [OdPair { origin: LinkId(0), dest: LinkId(1), rate_per_sec: 0.5 }];
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
        highway_od_pairs(&net, 7, 200, &mut pairs);
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

    #[test]
    fn rush_hour_drives_gateways_from_the_real_profile() {
        // The 3-lane US-101 gateway (link 0, ref "US 101") should feed in exactly
        // 3 lanes × the route's per-lane hourly volume in aggregate — regardless of how
        // many destination pairs it fans out to — and follow the real curve over the day.
        let net = freeway_corridor_with_ref();
        let world = NetWorld::new(net, SimConfig::default_config());
        let mut pairs = Vec::new();
        highway_od_pairs(&world.network, 7, 60, &mut pairs);
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
        let big = (0..2000u64).filter(|&a| gravity_pick(&net, origin, &pool, 7, 5, a) == Some(arterial)).count();
        assert!(big > 1600, "gravity strongly favours the higher-capacity arterial: {big}/2000");
    }

    #[test]
    fn excess_demand_queues_at_the_gateway_and_drains() {
        // A 1-lane gateway fed far faster than one lane can admit: trips that can't
        // enter wait in the gateway queue (conserved, not dropped) and the backlog
        // drains once demand eases — a real metered gateway queue.
        let net = corridor().build(); // 1-lane entry (link 0) → exit (link 1)
        let mut world = NetWorld::new(net, SimConfig::default_config());
        let pairs = vec![OdPair { origin: LinkId(0), dest: LinkId(1), rate_per_sec: 5.0 }];
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
