//! Gridlock diagnostic: load a real map the way the browser does (per-driver
//! local routing), drive it to the rush-hour peak, and report whether traffic
//! is truly locking up (mean speed → 0, jam fraction stuck high) and *where*
//! (the most-occupied named links, and spillback chains). Run with:
//!
//! ```sh
//! cargo run --release --features import --example gridlock -- \
//!   --map ../web/public/mvsv.json --lodes ../web/public/mvsv.lodes.json
//! ```

use engine::sim::config::SimConfig;
use engine::sim::demand::{self, DemandGenerator, DemandSources};
use engine::sim::map::OsmMap;
use engine::sim::net_world::NetWorld;
use std::collections::HashMap;

fn main() {
    let mut map_path = String::new();
    let mut lodes_path = None;
    let mut start_hour = 7.5f64;
    let mut rush = true;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| panic!("missing value for {a}"));
        match a.as_str() {
            "--map" => map_path = val(),
            "--lodes" => lodes_path = Some(val()),
            "--start-hour" => start_hour = val().parse().expect("hour"),
            "--no-rush" => rush = false,
            other => panic!("unknown arg {other}"),
        }
    }
    let raw = std::fs::read_to_string(&map_path).expect("read map");
    let map = OsmMap::from_json(&raw).expect("parse map");
    let mut world = NetWorld::new(map.build(), SimConfig::default_config());
    world.set_local_routing(true); // the browser default

    let commute = lodes_path.as_ref().map(|p| {
        let raw = std::fs::read_to_string(p).expect("read lodes");
        demand::CommuteOd::from_json(&raw).expect("parse lodes")
    });
    let sources = DemandSources::with_rush_hour(true, true, rush);
    let pairs = demand::od_pairs_with_commute(&world.network, 7, 48, sources, commute.as_ref());
    let mut gen = DemandGenerator::new(&world, &pairs, 7);
    gen.set_rush_hour(&world.network, rush);
    gen.set_day_compression(60.0); // browser default
    gen.resume_clock(start_hour * 3600.0, 0);
    world.install_router(&gen.destinations());

    let dt = SimConfig::default_config().dt;
    println!("map={map_path} start={start_hour}h rush={rush} links={} lanes={}", world.network.links.len(), world.network.lanes.len());

    // Warm up ~20 min of sim time, sampling the jam fraction to see if it
    // stabilises (heavy flow) or ratchets toward a lock (gridlock).
    let steps = (20.0 * 60.0 / dt) as usize;
    let sample_every = (60.0 / dt) as usize; // ~1 sim-minute
    for k in 0..steps {
        gen.step(&mut world, dt);
        world.step();
        if k % sample_every == 0 {
            let (fleet, mean, jam) = flow_stats(&world);
            println!(
                "  t={:>4.1}min fleet={:>6} mean={:>5.1} mph jam={:>4.1}%",
                k as f64 * dt / 60.0,
                fleet,
                mean * 2.237,
                jam * 100.0,
            );
        }
    }

    // Localise: the most-occupied links (vehicles per jam-capacity), named.
    let mut per_link: HashMap<u32, (u32, f64, f64)> = HashMap::new(); // link -> (n, sum_speed, cap)
    for v in world.vehicles() {
        let lane = world.network.lane(v.lane);
        let e = per_link.entry(lane.link.0).or_insert((0, 0.0, 0.0));
        e.0 += 1;
        e.1 += v.speed;
    }
    // capacity per link = total drivable lane length / 7 m jam spacing
    for (li, cap) in per_link.iter_mut().map(|(k, e)| (*k, &mut e.2)) {
        let link = world.network.link(engine::sim::network::LinkId(li));
        let mut total = 0.0;
        for lane_i in 0..link.lane_count {
            let l = world.network.lane(engine::sim::network::LaneId(link.lane_start.0 + lane_i));
            total += l.length;
        }
        *cap = (total / 7.0).max(1.0);
    }
    let mut ranked: Vec<(u32, f64, u32, f64)> = per_link
        .iter()
        .map(|(&li, &(n, ss, cap))| (li, n as f64 / cap, n, if n > 0 { ss / n as f64 } else { 0.0 }))
        .collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    // Downstream links per node, to trace spillback.
    use engine::sim::network::LinkId;
    let net = &world.network;
    let mut out_of: Vec<Vec<u32>> = vec![Vec::new(); net.nodes.len()];
    for li in 0..net.links.len() {
        out_of[net.link(LinkId(li as u32)).from.idx()].push(li as u32);
    }
    let occ_of = |li: u32| per_link.get(&li).map(|&(n, _, cap)| n as f64 / cap).unwrap_or(0.0);
    println!("\nTop occupied links (occupancy = vehicles / jam-capacity):");
    for (li, occ, n, mean) in ranked.iter().take(14) {
        let link = net.link(LinkId(*li));
        let name = net.link_names.get(*li as usize).map(|s| s.as_str()).unwrap_or("");
        let kind = format!("{:?}", link.kind);
        let len = net.lane(link.lane_start).length;
        let ctrl = format!("{:?}", net.node(link.to).control);
        // Worst downstream link occupancy (is it blocked from draining?).
        let down_occ = out_of[link.to.idx()].iter().map(|&d| occ_of(d)).fold(0.0, f64::max);
        println!(
            "  occ={:>4.0}% n={:>3} mean={:>4.1}mph len={:>4.0}m lanes={} {:<10} down_occ={:>3.0}% ctrl={:<12} {}",
            occ * 100.0, n, mean * 2.237, len, link.lane_count, kind, down_occ * 100.0, ctrl,
            if name.is_empty() { "(unnamed)" } else { name },
        );
    }

    let (fleet, mean, jam) = flow_stats(&world);
    println!("\nFINAL fleet={fleet} mean={:.1}mph jam={:.1}% asleep={}", mean * 2.237, jam * 100.0, world.asleep_count());

    // Deadlock check: how long have stopped cars been stopped? A realistic queue
    // clears within a cycle or two (< ~60 s); cars stuck for minutes are a lock.
    let mut waits: Vec<f64> = world.vehicles().iter().map(|v| v.wait_ticks() as f64 * dt).collect();
    waits.sort_by(f64::total_cmp);
    let pct = |p: f64| waits.get(((waits.len() as f64 * p) as usize).min(waits.len().saturating_sub(1))).copied().unwrap_or(0.0);
    let over = |s: f64| waits.iter().filter(|&&w| w > s).count();
    println!(
        "wait-time (s): p50={:.0} p95={:.0} max={:.0} · stuck>60s={} stuck>180s={}",
        pct(0.5), pct(0.95), waits.last().copied().unwrap_or(0.0), over(60.0), over(180.0),
    );
    println!("trips: exited={} leaked={}", world.exited(), world.leaked());

    // Localise the long-stuck cars (> 120 s stationary): which links hold them?
    let mut stuck_by_link: HashMap<u32, u32> = HashMap::new();
    for v in world.vehicles() {
        if v.wait_ticks() as f64 * dt > 120.0 {
            *stuck_by_link.entry(net.lane(v.lane).link.0).or_insert(0) += 1;
        }
    }
    let mut sranked: Vec<(u32, u32)> = stuck_by_link.into_iter().collect();
    sranked.sort_by(|a, b| b.1.cmp(&a.1));
    println!("\nLinks holding the most >120s-stuck cars:");
    for (li, c) in sranked.iter().take(12) {
        let link = net.link(LinkId(*li));
        let name = net.link_names.get(*li as usize).map(|s| s.as_str()).unwrap_or("");
        let down_occ = out_of[link.to.idx()].iter().map(|&d| occ_of(d)).fold(0.0, f64::max);
        let cyc = match net.node(link.to).control {
            engine::sim::network::NodeControl::Signalized(pid) => {
                let prog = &net.programs[pid.0 as usize];
                format!("cycle={:.0}s phases={}", prog.cycle_length(), prog.phases.len())
            }
            other => format!("{other:?}"),
        };
        println!(
            "  stuck={:>3} {:<10} len={:>4.0}m down_occ={:>3.0}% {:<24} {}",
            c, format!("{:?}", link.kind), net.lane(link.lane_start).length, down_occ * 100.0, cyc,
            if name.is_empty() { "(unnamed)" } else { name },
        );
    }
}

/// (fleet size, mean speed m/s, fraction with speed < 0.5 m/s).
fn flow_stats(world: &NetWorld) -> (usize, f64, f64) {
    let vs = world.vehicles();
    if vs.is_empty() {
        return (0, 0.0, 0.0);
    }
    let mut sum = 0.0;
    let mut stopped = 0;
    for v in vs {
        sum += v.speed;
        if v.speed < 0.5 {
            stopped += 1;
        }
    }
    (vs.len(), sum / vs.len() as f64, stopped as f64 / vs.len() as f64)
}
