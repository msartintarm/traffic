//! Load tests: saturate the real Millbrae map and profile the per-frame CPU cost
//! the browser incurs — demand spawning, the full sim step, and render marshaling —
//! as vehicle density climbs into gridlock. These reproduce the FPS degradation
//! seen in the browser and show which phase scales super-linearly. Expensive and
//! `#[ignore]`d; run with `--features import -- --ignored --nocapture`.

#![cfg(feature = "import")]

use std::collections::HashMap;
use std::time::Instant;

use engine::sim::config::SimConfig;
use engine::sim::demand::{self, DemandGenerator, DemandSources};
use engine::sim::map::OsmMap;
use engine::sim::net_world::{prof_take, AccelBackend, NetWorld, PHASE_NAMES, STEP_PHASES};
use engine::sim::network::Network;

fn real_map() -> Option<Network> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
    Some(OsmMap::from_json(&std::fs::read_to_string(path).ok()?).ok()?.build())
}

struct Sample {
    cars: usize,
    step_ms: f64,
    marshal_ms: f64,
    phases: [f64; STEP_PHASES],
}

fn ramp(world: &mut NetWorld, gen: &mut DemandGenerator, dt: f64, ticks: usize) {
    for _ in 0..ticks {
        gen.step(world, dt);
        world.step();
    }
}

/// Time `frames` full browser frames — demand, sim step, and render marshaling (a
/// world pose per vehicle plus the id→pose interpolation map, as the bridge does).
fn window(world: &mut NetWorld, gen: &mut DemandGenerator, dt: f64, frames: usize) -> Sample {
    prof_take();
    let (mut step, mut marshal) = (0.0, 0.0);
    for _ in 0..frames {
        gen.step(world, dt);
        let t = Instant::now();
        world.step();
        step += t.elapsed().as_secs_f64() * 1000.0;

        let t = Instant::now();
        let mut prev = HashMap::with_capacity(world.vehicles().len());
        for v in world.vehicles() {
            let p = world.vehicle_world_pose(v);
            prev.insert(v.id, [p[0], p[1], p[2], v.speed]);
        }
        std::hint::black_box(&prev);
        marshal += t.elapsed().as_secs_f64() * 1000.0;
    }
    let phases = prof_take().map(|p| p / frames as f64);
    Sample {
        cars: world.vehicles().len(),
        step_ms: step / frames as f64,
        marshal_ms: marshal / frames as f64,
        phases,
    }
}

fn scaling_curve(sources: DemandSources, target: usize, external_reroute: bool, label: &str) {
    let Some(net) = real_map() else { return };
    let cfg = SimConfig::default_config();
    let mut world = NetWorld::new(net, cfg);
    // Request CPU threads: engages rayon under `--features parallel`, and cleanly
    // falls back to serial otherwise (so this line is a no-op without the feature).
    world.set_accel_backend(AccelBackend::Threads);
    let pairs = demand::od_pairs(&world.network, 1, target, sources);
    let mut gen = DemandGenerator::new(&world, &pairs, 1);
    world.install_router(&gen.destinations());
    world.set_external_reroute(external_reroute);

    eprintln!("\n=== {label} ===");
    eprintln!("{:>7} {:>8} {:>8} {:>8}   phases ms [{}]", "cars", "step_ms", "marshl", "ns/car", PHASE_NAMES.join(" "));
    let mut peak = 0usize;
    let mut prev_cars = 0usize;
    for _ in 0..30 {
        ramp(&mut world, &mut gen, cfg.dt, 150);
        let s = window(&mut world, &mut gen, cfg.dt, 60);
        let ns_car = if s.cars > 0 { s.step_ms * 1e6 / s.cars as f64 } else { 0.0 };
        let phases = s.phases.iter().map(|p| format!("{p:.2}")).collect::<Vec<_>>().join(" ");
        eprintln!("{:>7} {:>8.2} {:>8.2} {:>8.1}   {phases}", s.cars, s.step_ms, s.marshal_ms, ns_car);
        assert_eq!(world.leaked(), 0, "no vehicle disappears at an intersection under load");
        peak = peak.max(s.cars);
        if s.cars < prev_cars + 50 {
            break; // saturated / gridlocked — density is no longer climbing
        }
        prev_cars = s.cars;
    }
    eprintln!("peak {peak} cars, {} crashed over the run", world.crashed());
    assert!(peak > 1500, "the load test saturates the map (peaked at {peak})");
}

#[test]
#[ignore]
fn load_highway_saturation() {
    // Off-peak-style freeway inflow driven hard until the surface network jams —
    // the regime where node queues make the per-vehicle constraint work blow up.
    scaling_curve(DemandSources::new(true, false), 600, false, "Millbrae highway saturation (CPU routing)");
}

#[test]
#[ignore]
fn load_balanced_gridlock() {
    // Boundary demand from every gateway, pushed into citywide gridlock.
    scaling_curve(DemandSources::new(true, true), 600, false, "Millbrae balanced gridlock (CPU routing)");
}

/// The scraped 101/280 ramps must be wired tangentially: no grade-separated movement
/// crosses another (a car never has to cross the freeway to reach a ramp), and every
/// mainline off-ramp hangs off the curb (rightmost) lane while the mainline continues
/// straight-through. Guards the freeway interchange lane-wiring in `OsmMap::build`.
#[test]
fn real_map_ramps_are_wired_tangentially_never_crossing() {
    use engine::sim::network::{MovementId, RoadKind};
    let Some(net) = real_map() else { return };
    let n_mv = net.movements.len() as u32;
    let link_of = |lane| net.lane(lane).link;
    let kind_of = |lane| net.link(link_of(lane)).kind;

    // No grade-separated movement crosses another grade-separated movement.
    for a in 0..n_mv {
        if !net.is_interchange_movement(MovementId(a)) { continue; }
        for b in (a + 1)..n_mv {
            assert!(
                !(net.is_interchange_movement(MovementId(b)) && net.movements_conflict(MovementId(a), MovementId(b))),
                "grade-separated movements {a} and {b} cross — a ramp forces crossing the freeway",
            );
        }
    }

    // Every off-ramp *from a freeway mainline* is fed only from the curb-most lanes,
    // and a multi-lane off-ramp has every one of its lanes fed (no dead lane), so a
    // wide exit runs at full capacity instead of funnelling through one lane.
    use std::collections::BTreeSet;
    let mut ramp_lanes_fed: std::collections::HashMap<u32, BTreeSet<u32>> = std::collections::HashMap::new();
    for m in 0..n_mv {
        let mv = net.movement(MovementId(m));
        if kind_of(mv.to_lane) == RoadKind::Ramp && net.link(link_of(mv.from_lane)).kind == RoadKind::Freeway {
            let src = net.link(link_of(mv.from_lane));
            let ramp = net.link(link_of(mv.to_lane));
            let from_idx = mv.from_lane.0 - src.lane_start.0;
            // Fed only from the curb-most `ramp.lane_count` freeway lanes.
            assert!(
                from_idx >= src.lane_count - ramp.lane_count.min(src.lane_count),
                "off-ramp fed from an inner freeway lane {from_idx} (would force a crossing)",
            );
            ramp_lanes_fed.entry(link_of(mv.to_lane).0).or_default().insert(mv.to_lane.0 - ramp.lane_start.0);
        }
    }
    for (ramp_link, fed) in &ramp_lanes_fed {
        let ramp = net.link(engine::sim::network::LinkId(*ramp_link));
        assert_eq!(
            fed.len() as u32, ramp.lane_count,
            "off-ramp {ramp_link} ({} lanes) has a dead lane: only {:?} fed", ramp.lane_count, fed,
        );
    }
}

/// Two guards on freeway realism, both regressions we just fixed:
/// 1. No grade-separated drivable segment is a sub-vehicle sliver (they were chopping
///    ramps into micro-links); the shortest must clear a small floor.
/// 2. Cars don't phantom-stop when they traverse a segment boundary — the stale
///    cross-frame reaction-delay lookup was dead-stopping ~279 freeway cars per 1500
///    ticks; with the history reset + settled gate it's a handful of genuine
///    close-leader brakes, far below any regression.
#[test]
fn freeway_traffic_flows_without_phantom_stops() {
    use engine::sim::network::LinkId;
    let net = real_map().unwrap();
    let min_len = (0..net.links.len() as u32)
        .filter(|&i| net.link(LinkId(i)).kind.is_grade_separated())
        .map(|i| net.lane(net.link(LinkId(i)).lane_start).length)
        .fold(f64::MAX, f64::min);
    assert!(min_len > 18.0, "grade-separated sliver segment survived: {min_len:.1} m");

    // A one-tick drop from >8 m/s to <1 m/s is > ~32 m/s^2 — non-physical, the phantom
    // stop's signature. Count them on freeway mainline under highway demand.
    let cfg = SimConfig::default_config();
    let mut world = NetWorld::new(net, cfg);
    let pairs = demand::od_pairs(&world.network, 1, 600, DemandSources::new(true, false));
    let mut gen = DemandGenerator::new(&world, &pairs, 1);
    world.install_router(&gen.destinations());
    let mut prev: std::collections::HashMap<u32, f64> = Default::default();
    let mut freeway = 0u32;
    for _ in 0..1500 {
        gen.step(&mut world, cfg.dt);
        world.step();
        for v in world.vehicles() {
            let k = world.network.link(world.network.lane(v.lane).link).kind;
            if format!("{k:?}") == "Freeway" && prev.get(&v.id).copied().unwrap_or(v.speed) > 8.0 && v.speed < 1.0 {
                freeway += 1;
            }
        }
        prev.clear();
        for v in world.vehicles() { prev.insert(v.id, v.speed); }
    }
    assert!(freeway < 40, "freeway phantom stops regressed: {freeway} (was ~279 before the fix)");
}

#[test]
fn browser_default_external_reroute_does_not_panic() {
    // The exact browser default: freeway + surface demand with routing owned
    // externally (GPU flow-field default-on), which the balanced-gridlock CPU test
    // does not cover. Steps enough frames to spawn and route real traffic.
    let Some(net) = real_map() else { return };
    let cfg = SimConfig::default_config();
    let mut world = NetWorld::new(net, cfg);
    world.set_accel_backend(AccelBackend::Serial);
    let pairs = demand::od_pairs(&world.network, 1, 600, DemandSources::new(true, true));
    let mut gen = DemandGenerator::new(&world, &pairs, 1);
    world.install_router(&gen.destinations());
    world.set_external_reroute(true);
    for _ in 0..600 {
        gen.step(&mut world, cfg.dt);
        world.step();
    }
    assert_eq!(world.leaked(), 0);
}

#[test]
#[ignore]
fn load_highway_saturation_gpu_routing() {
    // Same load with routing owned externally (the browser's ?gpu=1 path), so the
    // step cost excludes the CPU flow-field recompute.
    scaling_curve(DemandSources::new(true, false), 600, true, "Millbrae highway saturation (external/GPU routing)");
}
