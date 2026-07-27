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

pub struct OdPair {
    pub origin: LinkId,
    pub dest: LinkId,
    pub rate_per_sec: f64,
}

/// How origins and destinations are distributed across the map — the "traffic
/// mode" the UI exposes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DemandMode {
    /// Boundary-balanced: through / inbound / outbound / internal in fixed shares.
    Balanced,
    /// Freeway-dominated: most trips originate at a highway gateway (the US-101 /
    /// I-280 on-ramps into the map), the rest keep the surface streets alive.
    HighwayBiased,
}

impl DemandMode {
    /// Parse the UI's mode name; unknown names fall back to [`Balanced`].
    pub fn from_name(name: &str) -> Self {
        match name {
            "highway" | "highways" | "highway-biased" => DemandMode::HighwayBiased,
            _ => DemandMode::Balanced,
        }
    }
}

/// OD demand for a mode. [`DemandMode::Balanced`] is the boundary mix;
/// [`DemandMode::HighwayBiased`] anchors most origins on the freeway gateways.
/// Both fall back to the balanced mix when the map lacks the gateways a mode needs.
pub fn od_pairs(net: &Network, seed: u64, target: usize, mode: DemandMode) -> Vec<OdPair> {
    match mode {
        DemandMode::Balanced => boundary_od_pairs(net, seed, target),
        DemandMode::HighwayBiased => highway_biased_od_pairs(net, seed, target),
    }
}

pub struct DemandGenerator {
    /// `(rate_per_sec, origin, dest)` for OD pairs with at least one valid route.
    pairs: Vec<(f64, LinkId, LinkId)>,
    seed: u64,
    tick: u64,
    next_id: u32,
    spawned: u32,
}

impl DemandGenerator {
    pub fn new(world: &NetWorld, pairs: &[OdPair], seed: u64) -> Self {
        let pairs = pairs
            .iter()
            .filter(|p| world.network.route_links(p.origin, p.dest).is_some())
            .map(|p| (p.rate_per_sec, p.origin, p.dest))
            .collect();
        Self { pairs, seed, tick: 0, next_id: 0, spawned: 0 }
    }

    pub fn spawned(&self) -> u32 {
        self.spawned
    }

    /// The distinct destinations demanded — the destination set to build a
    /// [`NetWorld`] flow-field router over.
    pub fn destinations(&self) -> Vec<LinkId> {
        let mut dests: Vec<LinkId> = self.pairs.iter().map(|&(_, _, d)| d).collect();
        dests.sort_by_key(|l| l.0);
        dests.dedup();
        dests
    }

    pub fn step(&mut self, world: &mut NetWorld, dt: f64) {
        let costs = world.live_link_costs();
        for i in 0..self.pairs.len() {
            let (rate, origin, dest) = self.pairs[i];
            if rng::uniform01(self.seed, i as u32, self.tick, Stream::RouteChoice) >= (rate * dt).min(1.0) {
                continue;
            }
            let driver = class_of(self.seed, self.next_id).driver().sample(self.seed, self.next_id);
            let speed = entry_speed(&world.network, origin, &driver);
            let spawned = if world.router_knows(dest) {
                world.spawn_to(self.next_id, origin, dest, speed, driver)
            } else if let Some(route) = world.network.route_links_with_costs(origin, dest, &costs) {
                world.spawn_routed(self.next_id, route, speed, driver)
            } else {
                false
            };
            if spawned {
                self.spawned += 1;
            }
            self.next_id += 1;
        }
        self.tick += 1;
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

/// Freeway-dominated OD demand: most trips originate on a highway gateway (off
/// the freeway into town, or through to another gateway), with a minority leaving
/// town onto the freeway or staying local, so the surface network still lives.
/// Falls back to the balanced mix on a map with no highway gateways.
pub fn highway_biased_od_pairs(net: &Network, seed: u64, target: usize) -> Vec<OdPair> {
    let hw_in = boundary::highway_entry_links(net);
    let hw_out = boundary::highway_exit_links(net);
    let exits = boundary::exit_links(net);
    let interior = boundary::interior_links(net);
    if hw_in.is_empty() {
        return boundary_od_pairs(net, seed, target);
    }
    let categories: [(&[LinkId], &[LinkId], f64); 4] = [
        (&hw_in, &interior, 0.45),    // off the freeway into the city
        (&hw_in, &exits, 0.30),       // through: freeway → any gateway (incl. the other freeway)
        (&interior, &hw_out, 0.15),   // city onto the freeway
        (&interior, &interior, 0.10), // residual local trips
    ];
    let mut pairs = Vec::new();
    for (cat, (origins, dests, share)) in categories.iter().enumerate() {
        if origins.is_empty() || dests.is_empty() {
            continue;
        }
        let want = ((target as f64 * share).round() as usize).max(1);
        sample_pairs(net, seed, 50 + cat as u32, origins, dests, want, &mut pairs);
    }
    if pairs.is_empty() {
        return boundary_od_pairs(net, seed, target);
    }
    pairs
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
        let d = dests[(rng::hash(seed, salt * 2 + 1, attempt, Stream::RouteChoice) as usize) % dests.len()];
        attempt += 1;
        if o != d && net.route_links(o, d).is_some_and(|r| r.len() >= 2) {
            out.push(OdPair { origin: o, dest: d, rate_per_sec: capacity_rate(net, o) });
            found += 1;
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

        let pairs = od_pairs(&net, 4, 40, DemandMode::HighwayBiased);
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
        let pairs = od_pairs(&net, 3, 10, DemandMode::HighwayBiased);
        assert!(!pairs.is_empty(), "no freeways → fall back to the balanced mix");
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
