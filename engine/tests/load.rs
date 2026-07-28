//! Load tests: saturate the real Millbrae map and profile the per-frame CPU cost
//! the browser incurs — demand spawning, the full sim step, and render marshaling —
//! as vehicle density climbs into gridlock. These reproduce the FPS degradation
//! seen in the browser and show which phase scales super-linearly. Expensive and
//! `#[ignore]`d; run with `--features import -- --ignored --nocapture`.

#![cfg(feature = "import")]

use std::collections::HashMap;
use std::time::Instant;

use engine::sim::config::SimConfig;
use engine::sim::demand::{self, DemandGenerator, DemandMode};
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

fn scaling_curve(mode: DemandMode, target: usize, external_reroute: bool, label: &str) {
    let Some(net) = real_map() else { return };
    let cfg = SimConfig::default_config();
    let mut world = NetWorld::new(net, cfg);
    // Request CPU threads: engages rayon under `--features parallel`, and cleanly
    // falls back to serial otherwise (so this line is a no-op without the feature).
    world.set_accel_backend(AccelBackend::Threads);
    let pairs = demand::od_pairs(&world.network, 1, target, mode);
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
    scaling_curve(DemandMode::HighwayBiased, 600, false, "Millbrae highway saturation (CPU routing)");
}

#[test]
#[ignore]
fn load_balanced_gridlock() {
    // Boundary demand from every gateway, pushed into citywide gridlock.
    scaling_curve(DemandMode::Balanced, 600, false, "Millbrae balanced gridlock (CPU routing)");
}

#[test]
#[ignore]
fn load_highway_saturation_gpu_routing() {
    // Same load with routing owned externally (the browser's ?gpu=1 path), so the
    // step cost excludes the CPU flow-field recompute.
    scaling_curve(DemandMode::HighwayBiased, 600, true, "Millbrae highway saturation (external/GPU routing)");
}
