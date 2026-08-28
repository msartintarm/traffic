//! Spin audit: run the committed Millbrae map under whole-city demand and hunt
//! for physically-impossible rotation — cars whose heading swings faster than
//! any real vehicle can yaw. Classifies each event (one-tick flip, sustained
//! spin, impossibly tight arc), tallies them per street, and dumps short pose
//! traces of the worst offenders so the responsible code path can be read off.
//!
//! ```text
//! cargo run --release --example spin_audit --features import [-- <ticks>]
//! ```

use std::collections::HashMap;

use engine::sim::config::SimConfig;
use engine::sim::demand::{self, DemandGenerator, DemandSources};
use engine::sim::net_world::NetWorld;
use engine::sim::OsmMap;

/// Signed shortest rotation a → b, in (-π, π].
fn shortest_angle(a: f64, b: f64) -> f64 {
    let mut d = (b - a) % std::f64::consts::TAU;
    if d > std::f64::consts::PI {
        d -= std::f64::consts::TAU;
    }
    if d <= -std::f64::consts::PI {
        d += std::f64::consts::TAU;
    }
    d
}

const TRACE_LEN: usize = 20;
/// Sliding rotation window (ticks) for spin detection: 2 s at dt = 0.2.
const WINDOW: usize = 10;

#[derive(Clone, Copy)]
struct Tick {
    pose: [f64; 3],
    dtheta: f64,
    speed: f64,
    crossing: bool,
    link: u32,
}

#[derive(Default)]
struct Tally {
    flips: usize,
    spins: usize,
    tight: usize,
}

fn main() {
    let ticks: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(3000);
    let path = format!("{}/../web/public/map.json", env!("CARGO_MANIFEST_DIR"));
    let json = std::fs::read_to_string(&path).expect("web/public/map.json (Millbrae) must be present");
    let net = OsmMap::from_json(&json).expect("map parses").build();
    let cfg = SimConfig::default_config();
    let mut world = NetWorld::new(net, cfg);
    let pairs = demand::od_pairs(&world.network, 0, 600, DemandSources::new(true, true));
    let mut gen = DemandGenerator::new(&world, &pairs, 0);
    world.install_router(&gen.destinations());

    // Populate the city before auditing.
    for _ in 0..1500 {
        gen.step(&mut world, cfg.dt);
        world.step();
    }

    let mut hist: HashMap<u32, Vec<Tick>> = HashMap::new();
    let mut per_link: HashMap<u32, Tally> = HashMap::new();
    let mut worst: Vec<(f64, u32, usize, Vec<Tick>)> = Vec::new(); // (score, id, tick, trace)
    let (mut flips, mut spins, mut tight) = (0usize, 0usize, 0usize);

    for t in 0..ticks {
        gen.step(&mut world, cfg.dt);
        world.step();
        let mut alive: HashMap<u32, Vec<Tick>> = HashMap::with_capacity(hist.len());
        for v in world.vehicles() {
            let pose = world.vehicle_world_pose(v);
            let link = world.network.lane(v.lane).link.0;
            let mut h = hist.remove(&v.id).unwrap_or_default();
            let dtheta = h.last().map_or(0.0, |p| shortest_angle(p.pose[2], pose[2]));
            let ds = h.last().map_or(0.0, |p| ((pose[0] - p.pose[0]).powi(2) + (pose[1] - p.pose[1]).powi(2)).sqrt());
            h.push(Tick { pose, dtheta, speed: v.speed, crossing: v.is_crossing(), link });
            if h.len() > TRACE_LEN {
                h.remove(0);
            }

            let tally = per_link.entry(link).or_default();
            let mut score = 0.0f64;
            let deg = dtheta.to_degrees().abs();
            if deg >= 100.0 {
                flips += 1;
                tally.flips += 1;
                score = score.max(deg);
            }
            if deg >= 20.0 && ds > 0.05 && ds / dtheta.abs() < 2.0 {
                tight += 1;
                tally.tight += 1;
                score = score.max(deg);
            }
            let wsum: f64 = h.iter().rev().take(WINDOW).map(|p| p.dtheta).sum();
            if wsum.abs().to_degrees() >= 300.0 {
                spins += 1;
                tally.spins += 1;
                score = score.max(wsum.abs().to_degrees() + 1000.0); // rank true spins first
            }
            if score > 0.0 {
                worst.push((score, v.id, t, h.clone()));
            }
            alive.insert(v.id, h);
        }
        hist = alive;
    }

    println!("audited {ticks} ticks ({:.1} sim-min), fleet ~{} cars", ticks as f64 * cfg.dt / 60.0, world.vehicles().len());
    println!("events: {flips} one-tick flips (>=100 deg), {spins} spin-window hits (>=300 deg / 2 s), {tight} tight arcs (r < 2 m at >=20 deg)");

    let mut by_link: Vec<_> = per_link.into_iter().filter(|(_, t)| t.flips + t.spins + t.tight > 0).collect();
    by_link.sort_by_key(|(_, t)| std::cmp::Reverse(t.flips + t.spins + t.tight));
    println!("\ntop streets by event count:");
    for (link, t) in by_link.iter().take(15) {
        let name = &world.network.link_names[*link as usize];
        println!("  link {link:5} {name:40} flips={} spins={} tight={}", t.flips, t.spins, t.tight);
    }

    worst.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut seen = std::collections::HashSet::new();
    println!("\nworst offender traces (newest last; heading deg, dtheta deg, speed, link, X=crossing):");
    for (score, id, tick, trace) in worst.iter().filter(|(_, id, ..)| seen.insert(*id)).take(6) {
        println!("-- car {id} at tick {tick} (score {score:.0}):");
        for p in trace {
            let name = &world.network.link_names[p.link as usize];
            println!(
                "   ({:8.1},{:8.1}) h={:7.1} dh={:7.1} v={:5.2} {}{} {}",
                p.pose[0],
                p.pose[1],
                p.pose[2].to_degrees(),
                p.dtheta.to_degrees(),
                p.speed,
                if p.crossing { "X " } else { "  " },
                p.link,
                name,
            );
        }
    }
}
