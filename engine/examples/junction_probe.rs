//! Signalized-cluster autopsy: throughput, green shares, stuck-state, and crash
//! sites at the ECR × Millbrae Ave split junction (and map-wide storage census).
use engine::sim::config::SimConfig;
use engine::sim::demand::{self, DemandGenerator, DemandSources};
use engine::sim::net_world::{CrashKind, NetWorld};
use engine::sim::network::{LinkId, MovementId};
use engine::sim::signal::SignalState;
use engine::sim::OsmMap;

fn main() {
    let path = format!("{}/../web/public/map.json", env!("CARGO_MANIFEST_DIR"));
    let net = OsmMap::from_json(&std::fs::read_to_string(&path).unwrap()).unwrap().build();
    // The big split junction: the cluster containing both names.
    let (ji, j) = net
        .junctions
        .iter()
        .enumerate()
        .find(|(_, j)| {
            let names: Vec<&str> = j.approaches.iter().map(|l| net.link_names[l.idx()].as_str()).collect();
            names.iter().any(|n| n.contains("El Camino")) && names.iter().any(|n| n.contains("Millbrae Avenue"))
        })
        .expect("ECR x Millbrae cluster");
    println!(
        "cluster {ji}: {} nodes, {} approaches {} exits, center ({:.0},{:.0}), signalized={}",
        j.nodes.len(), j.approaches.len(), j.exits.len(), j.center[0], j.center[1], j.program.is_some()
    );
    for &a in &j.approaches {
        println!("  approach {:4} {:24} lanes {}", a.0, net.link_names[a.idx()], net.link(a).lane_count);
    }
    let center = j.center;
    let approaches = j.approaches.clone();
    let exits = j.exits.clone();
    let member_nodes = j.nodes.clone();

    let cfg = SimConfig::default_config();
    let mut world = NetWorld::new(net, cfg);
    world.set_wreck_clear_secs(60.0); // browser-like
    let pairs = demand::od_pairs(&world.network, 0, 600, DemandSources::new(true, true));
    let mut gen = DemandGenerator::new(&world, &pairs, 0);
    world.install_router(&gen.destinations());
    for _ in 0..1500 {
        gen.step(&mut world, cfg.dt);
        world.step();
    }
    // movements whose node is in the cluster, keyed by approach
    let mvs: Vec<(MovementId, LinkId)> = (0..world.network.movements.len())
        .filter_map(|i| {
            let mv = world.network.movement(MovementId(i as u32));
            member_nodes.contains(&mv.node).then(|| {
                (MovementId(i as u32), world.network.lane(mv.from_lane).link)
            })
        })
        .collect();
    let base_entries: Vec<u32> = exits.iter().map(|l| world.link_entry_counts()[l.idx()]).collect();
    let base_crash = world.crash_counts();
    let mut green_ticks = vec![0u32; approaches.len()];
    let mut queue_sum = vec![0u64; approaches.len()];
    let mut wreck_ticks_near = 0u64;
    const AUDIT: usize = 3000; // 10 sim-min
    for _ in 0..AUDIT {
        gen.step(&mut world, cfg.dt);
        world.step();
        for (ai, &a) in approaches.iter().enumerate() {
            // any movement from this approach green?
            let any_green = mvs.iter().any(|&(m, fl)| fl == a && matches!(world.movement_state(m), SignalState::Green));
            green_ticks[ai] += any_green as u32;
            let q = world
                .vehicles()
                .iter()
                .filter(|v| world.network.lane(v.lane).link == a && v.speed < 1.0)
                .count();
            queue_sum[ai] += q as u64;
        }
        wreck_ticks_near += world
            .vehicles()
            .iter()
            .filter(|v| {
                v.is_wrecked() && {
                    let p = world.vehicle_arc_pose(v);
                    (p[0] - center[0]).hypot(p[1] - center[1]) < 60.0
                }
            })
            .count() as u64;
    }
    let mins = AUDIT as f64 * cfg.dt / 60.0;
    // Per-movement autopsy: green share + whether its from-link is internal.
    println!("\n== movements at the cluster (green%% over the window):");
    let mut mv_green: Vec<(MovementId, u32)> = mvs.iter().map(|&(m, _)| (m, 0u32)).collect();
    let internal: Vec<LinkId> = (0..world.network.links.len() as u32)
        .map(LinkId)
        .filter(|&l| {
            let lk = world.network.link(l);
            member_nodes.contains(&lk.from) && member_nodes.contains(&lk.to)
        })
        .collect();
    let base_int: Vec<u32> = internal.iter().map(|l| world.link_entry_counts()[l.idx()]).collect();
    for _ in 0..1500 {
        gen.step(&mut world, cfg.dt);
        world.step();
        for e in mv_green.iter_mut() {
            if matches!(world.movement_state(e.0), SignalState::Green) {
                e.1 += 1;
            }
        }
    }
    for (m, g) in &mv_green {
        let mv = world.network.movement(*m);
        let (fl, tl) = (world.network.lane(mv.from_lane).link, world.network.lane(mv.to_lane).link);
        let int_from = internal.contains(&fl);
        println!(
            "  mv {:5} n{:5} {:18} -> {:18} {} green {:3.0}%",
            m.0, mv.node.0,
            &world.network.link_names[fl.idx()][..world.network.link_names[fl.idx()].len().min(18)],
            &world.network.link_names[tl.idx()][..world.network.link_names[tl.idx()].len().min(18)],
            if int_from { "INT" } else { "ext" },
            100.0 * *g as f64 / 1500.0,
        );
    }
    println!("internal-link entries over 5 more sim-min:");
    for (ii, &l) in internal.iter().enumerate() {
        let n = world.link_entry_counts()[l.idx()] - base_int[ii];
        println!("  int {:4} {:20} +{}", l.0, world.network.link_names[l.idx()], n);
    }
    println!("\n== {mins:.0} sim-min under saturated demand:");
    for (ai, &a) in approaches.iter().enumerate() {
        println!(
            "  approach {:4} {:24} green {:4.0}% avg-queue {:5.1}",
            a.0,
            world.network.link_names[a.idx()],
            100.0 * green_ticks[ai] as f64 / AUDIT as f64,
            queue_sum[ai] as f64 / AUDIT as f64,
        );
    }
    println!("throughput per exit arm (veh/h):");
    for (ei, &e) in exits.iter().enumerate() {
        let n = world.link_entry_counts()[e.idx()] - base_entries[ei];
        println!("  exit {:4} {:24} {:6.0}", e.0, world.network.link_names[e.idx()], n as f64 * 60.0 / mins);
    }
    println!("\n== stuck-state autopsy:");
    for &a in &approaches {
        let mut cars: Vec<_> = world
            .vehicles()
            .iter()
            .filter(|v| world.network.lane(v.lane).link == a)
            .collect();
        cars.sort_by(|x, y| y.position.total_cmp(&x.position));
        if let Some(f) = cars.first() {
            let lane = world.network.lane(f.lane);
            println!(
                "  approach {:4} {:20} front id{:5} lane{:4} pos {:6.1}/{:6.1} v={:4.2} X={} n_on_link={}",
                a.0, world.network.link_names[a.idx()], f.id, f.lane.0, f.position, lane.length, f.speed, f.is_crossing() as u8, cars.len()
            );
        }
    }
    for &l in &internal {
        let cars: Vec<_> = world.vehicles().iter().filter(|v| world.network.lane(v.lane).link == l).collect();
        if cars.is_empty() { println!("  int {:4} {:20} EMPTY", l.0, world.network.link_names[l.idx()]); continue; }
        for f in &cars {
            let lane = world.network.lane(f.lane);
            println!(
                "  int {:4} {:20} id{:5} lane{:4} pos {:6.1}/{:6.1} v={:4.2} X={} wreck={}",
                l.0, world.network.link_names[l.idx()], f.id, f.lane.0, f.position, lane.length, f.speed, f.is_crossing() as u8, f.is_wrecked() as u8
            );
        }
    }
    let c = world.crash_counts();
    println!("crashes in window: rear-end {} junction {}", c[0] - base_crash[0], c[1] - base_crash[1]);
    let near = world
        .crash_log()
        .iter()
        .filter(|r| (r.pos[0] as f64 - center[0]).hypot(r.pos[1] as f64 - center[1]) < 80.0)
        .count();
    let jx_near = world
        .crash_log()
        .iter()
        .filter(|r| matches!(r.kind, CrashKind::Junction) && (r.pos[0] as f64 - center[0]).hypot(r.pos[1] as f64 - center[1]) < 80.0)
        .count();
    println!("crash-log entries within 80 m of the cluster: {near} (junction-kind {jx_near}) of {}", world.crash_log().len());
    println!("wreck-ticks within 60 m: {wreck_ticks_near} ({:.1} wreck-min blocking)", wreck_ticks_near as f64 * cfg.dt / 60.0);
    println!("\n== crash sites (kind @ pos, nearest node):");
    for r in world.crash_log() {
        let nearest = (0..world.network.nodes.len())
            .min_by(|&x, &y| {
                let d = |i: usize| {
                    let p = world.network.nodes[i].position;
                    (p[0] - r.pos[0] as f64).powi(2) + (p[1] - r.pos[1] as f64).powi(2)
                };
                d(x).total_cmp(&d(y))
            })
            .unwrap();
        let names: Vec<&str> = world.network.links.iter().enumerate()
            .filter(|(_, l)| l.from.0 as usize == nearest || l.to.0 as usize == nearest)
            .map(|(i, _)| world.network.link_names[i].as_str())
            .take(2)
            .collect();
        println!("  {:?} ({:6.0},{:6.0}) closing {:4.1} near n{} {:?} ctrl {:?}", r.kind, r.pos[0], r.pos[1], r.closing_speed, nearest, names, world.network.nodes[nearest].control);
    }
    // and: how short are internal lanes across ALL signalized clusters map-wide?
    let mut short = 0;
    let mut total = 0;
    for j in &world.network.junctions {
        if j.program.is_none() { continue; }
        for l in 0..world.network.links.len() as u32 {
            let lk = world.network.link(LinkId(l));
            if j.nodes.contains(&lk.from) && j.nodes.contains(&lk.to) {
                total += 1;
                if world.network.lane(lk.lane_start).length < 7.0 { short += 1; }
            }
        }
    }
    println!("\nmap-wide: {short} of {total} signalized-cluster internal links have <7 m of storage");
}
