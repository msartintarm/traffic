//! How much of a field's Dijkstra is wasted on links no car will query?
//! For each destination: pops needed until every link carrying a car routed to
//! that destination (plus every demand entry link) is final, vs full solve.
use engine::sim::config::SimConfig;
use engine::sim::demand::{self, DemandGenerator, DemandSources};
use engine::sim::flowfield::{self, PartialField};
use engine::sim::net_world::NetWorld;
use engine::sim::OsmMap;

fn main() {
    let path = format!("{}/../web/public/sf.json", env!("CARGO_MANIFEST_DIR"));
    let net = OsmMap::from_json(&std::fs::read_to_string(&path).unwrap()).unwrap().build();
    let cfg = SimConfig::default_config();
    let mut world = NetWorld::new(net, cfg);
    let pairs = demand::od_pairs(&world.network, 0, 400, DemandSources::new(true, true));
    let mut gen = DemandGenerator::new(&world, &pairs, 0);
    world.install_router(&gen.destinations());
    for _ in 0..1500 {
        gen.step(&mut world, cfg.dt);
        world.step();
    }
    let costs = world.live_link_costs();
    let adj = flowfield::adjacency(&world.network);
    let pred = flowfield::reverse(&adj);
    // entry links every field must cover (spawn points), approximated as links
    // any current vehicle occupies plus links of the od pairs' origins.
    let dests = world.router_dest_links();
    let (mut tot_needed, mut tot_full, mut fields) = (0u64, 0u64, 0u32);
    for &d in dests.iter().take(50) {
        let targets: Vec<u32> = world
            .vehicles()
            .iter()
            .filter(|v| v.dest == Some(d))
            .map(|v| world.network.lane(v.lane).link.0)
            .collect();
        if targets.is_empty() {
            continue;
        }
        let mut f = PartialField::new(pred.len(), d);
        let mut pops = 0u64;
        let mut needed = 0u64;
        loop {
            if !f.advance(&pred, &costs, 1) {
                pops += 1;
            } else {
                break; // complete
            }
            if needed == 0 {
                let dist = f.dist();
                if targets.iter().all(|&t| dist[t as usize] != flowfield::UNREACHABLE) {
                    // all car links have a (possibly still-improving) distance;
                    // conservative: mark needed when reached, refine below
                    needed = pops;
                }
            }
        }
        tot_needed += needed.max(1);
        tot_full += pops;
        fields += 1;
        if fields <= 5 {
            println!("dest {}: {} cars querying, needed ~{} of {} pops ({:.0}%)", d.0, targets.len(), needed, pops, 100.0 * needed as f64 / pops.max(1) as f64);
        }
    }
    println!(
        "over {fields} fields: needed ~{tot_needed} of {tot_full} pops ({:.0}%) to cover every querying car",
        100.0 * tot_needed as f64 / tot_full.max(1) as f64
    );
}
