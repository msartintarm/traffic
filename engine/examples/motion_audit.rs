//! Motion audit: run the committed Millbrae map under whole-city demand and
//! check every vehicle's motion against what a steered four-wheeled car can
//! physically do. Beyond the original spin hunts (one-tick flips, spin windows,
//! impossibly tight arcs) it audits the bicycle-model invariants: **crab**
//! (velocity direction versus heading — four wheels cannot slide sideways),
//! **curvature** (|Δθ|/Δs bounded by steering geometry), and the tracker's
//! divergence-guard count (clusters mark geometry the model cannot follow).
//!
//! ```text
//! cargo run --release --example motion_audit --features import [-- <ticks>]
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
    rear: [f64; 2],
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
    crab: usize,
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

    let base_div = world.kinematic_divergences();
    let mut hist: HashMap<u32, Vec<Tick>> = HashMap::new();
    let mut per_link: HashMap<u32, Tally> = HashMap::new();
    let mut worst: Vec<(f64, u32, usize, Vec<Tick>)> = Vec::new(); // (score, id, tick, trace)
    let (mut flips, mut spins, mut tight, mut crabs) = (0usize, 0usize, 0usize, 0usize);
    let (mut worst_crab, mut worst_kappa) = (0.0f64, 0.0f64);
    let mut moving_ticks = 0u64;

    for t in 0..ticks {
        gen.step(&mut world, cfg.dt);
        world.step();
        let mut alive: HashMap<u32, Vec<Tick>> = HashMap::with_capacity(hist.len());
        for v in world.vehicles() {
            let pose = world.vehicle_world_pose(v);
            let link = world.network.lane(v.lane).link.0;
            let mut h = hist.remove(&v.id).unwrap_or_default();
            let dtheta = h.last().map_or(0.0, |p| shortest_angle(p.pose[2], pose[2]));
            // Displacements at the rear axle: the bicycle model's ground truth.
            // The front bumper legitimately sweeps sideways during a turn (its
            // velocity tilts by atan(kappa * axle_offset) from the heading), so
            // crab judged there would flag every real corner.
            let rear = v.rear_axle();
            let (dx, dy) = h.last().map_or((0.0, 0.0), |p| (rear[0] - p.rear[0], rear[1] - p.rear[1]));
            let ds = dx.hypot(dy);
            h.push(Tick { pose, rear, dtheta, speed: v.speed, crossing: v.is_crossing(), link });
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
            // Bicycle-model invariants, judged only while genuinely moving.
            if v.speed > 3.0 && ds > 0.3 {
                moving_ticks += 1;
                let crab = shortest_angle(pose[2], dy.atan2(dx)).abs().to_degrees();
                worst_crab = worst_crab.max(crab);
                if crab >= 15.0 {
                    crabs += 1;
                    tally.crab += 1;
                    score = score.max(crab + 500.0);
                }
                worst_kappa = worst_kappa.max(dtheta.abs() / ds);
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
    println!(
        "kinematics over {moving_ticks} moving car-ticks: {crabs} crab events (>=15 deg), worst crab {worst_crab:.1} deg, worst curvature {worst_kappa:.3} 1/m (steering limit ~0.26)",
    );
    println!("divergence-guard activations: {}", world.kinematic_divergences() - base_div);

    let mut by_link: Vec<_> = per_link.into_iter().filter(|(_, t)| t.flips + t.spins + t.tight + t.crab > 0).collect();
    by_link.sort_by_key(|(_, t)| std::cmp::Reverse(t.flips + t.spins + t.tight + t.crab));
    if !by_link.is_empty() {
        println!("\ntop streets by event count:");
    }
    for (link, t) in by_link.iter().take(15) {
        let name = &world.network.link_names[*link as usize];
        println!("  link {link:5} {name:40} flips={} spins={} tight={} crab={}", t.flips, t.spins, t.tight, t.crab);
    }

    worst.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut seen = std::collections::HashSet::new();
    if !worst.is_empty() {
        println!("\nworst offender traces (newest last; heading deg, dtheta deg, speed, link, X=crossing):");
    }
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
