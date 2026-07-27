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
            let spawned = if world.router_knows(dest) {
                world.spawn_to(self.next_id, origin, dest, 5.0, driver)
            } else if let Some(route) = world.network.route_links_with_costs(origin, dest, &costs) {
                world.spawn_routed(self.next_id, route, 5.0, driver)
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
            out.push(OdPair { origin: o, dest: d, rate_per_sec: 0.2 });
            found += 1;
        }
    }
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
    use super::super::config::SimConfig;
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
