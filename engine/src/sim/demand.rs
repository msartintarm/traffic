//! Origin–destination travel demand: the "who goes where, when" that carries
//! most of the perceived realism at city scale. Each OD pair injects vehicles
//! along its precomputed fastest route at a mean rate, using the stateless RNG
//! so spawns are Bernoulli per tick without threaded state. Congestion-reactive
//! re-routing (recomputing the route against live travel times) plugs in later.

use super::config::VehicleClass;
use super::net_world::NetWorld;
use super::network::LinkId;
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

    pub fn step(&mut self, world: &mut NetWorld, dt: f64) {
        // Route each new vehicle against *current* congestion, so demand steers
        // around jams as they form (each spawn re-plans).
        let costs = world.live_link_costs();
        for i in 0..self.pairs.len() {
            let (rate, origin, dest) = self.pairs[i];
            if rng::uniform01(self.seed, i as u32, self.tick, Stream::RouteChoice) < (rate * dt).min(1.0) {
                if let Some(route) = world.network.route_links_with_costs(origin, dest, &costs) {
                    let driver = class_of(self.seed, self.next_id).driver().sample(self.seed, self.next_id);
                    if world.spawn_routed(self.next_id, route, 5.0, driver) {
                        self.spawned += 1;
                    }
                    self.next_id += 1;
                }
            }
        }
        self.tick += 1;
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
}
