//! Routing-cost assessment on the SF map: field counts, install cost, live-cost
//! sweep cost, recompute cadence, and where the fleet actually drives.
use std::time::Instant;

use engine::sim::config::SimConfig;
use engine::sim::demand::{self, DemandGenerator, DemandSources};
use engine::sim::net_world::NetWorld;
use engine::sim::network::NodeControl;
use engine::sim::OsmMap;

fn main() {
    let path = format!("{}/../web/public/sf.json", env!("CARGO_MANIFEST_DIR"));
    let t = Instant::now();
    let net = OsmMap::from_json(&std::fs::read_to_string(&path).unwrap()).unwrap().build();
    println!("build: {:.1}s, {} links {} lanes {} movements", t.elapsed().as_secs_f64(), net.links.len(), net.lanes.len(), net.movements.len());

    let arterial = net.links.iter().filter(|l| {
        use engine::sim::network::RoadKind;
        !matches!(l.kind, RoadKind::Local)
    }).count();
    println!("arterial-ish links (non-residential): {} of {} ({:.0}%)", arterial, net.links.len(), 100.0 * arterial as f64 / net.links.len() as f64);

    let cfg = SimConfig::default_config();
    let mut world = NetWorld::new(net, cfg);
    let t = Instant::now();
    let pairs = demand::od_pairs(&world.network, 0, 400, DemandSources::new(true, true));
    println!("od_pairs: {:.2}s, {} pairs", t.elapsed().as_secs_f64(), pairs.len());
    let mut gen = DemandGenerator::new(&world, &pairs, 0);
    let dests = gen.destinations();
    println!("destination fields: {}", dests.len());
    let t = Instant::now();
    world.install_router(&dests);
    println!("install_router (all fields, free-flow): {:.2}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let costs = world.live_link_costs();
    println!("live_link_costs sweep: {:.2} ms ({} links)", t.elapsed().as_secs_f64() * 1000.0, costs.len());

    // ramp demand, then measure steady state composition
    let t = Instant::now();
    for _ in 0..2400 {
        gen.step(&mut world, cfg.dt);
        world.step();
    }
    println!("2400-tick run: {:.1}s ({:.1} ms/tick), fleet {}", t.elapsed().as_secs_f64(), t.elapsed().as_secs_f64() * 1000.0 / 2400.0, world.vehicles().len());

    // fleet distribution by road class + free-flow share
    use engine::sim::network::RoadKind;
    let (mut res, mut art) = (0u32, 0u32);
    let mut near_stop = 0u32;
    let mut free = 0u32;
    for v in world.vehicles() {
        let lane = world.network.lane(v.lane);
        let l = world.network.link(lane.link);
        if matches!(l.kind, RoadKind::Local) { res += 1 } else { art += 1 }
        let to = world.network.node(l.to);
        if matches!(to.control, NodeControl::Stop) && lane.length - v.position < 60.0 { near_stop += 1 }
        if v.speed > 0.9 * v.driver.desired_speed.min(lane.speed_limit) { free += 1 }
    }
    println!("fleet: {} on residential, {} on arterial; {} approaching a stop node; {} at free-flow speed", res, art, near_stop, free);
}
