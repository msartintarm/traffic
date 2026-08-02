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

fn map_from(file: &str) -> Option<Network> {
    let path = format!("{}/../web/public/{}", env!("CARGO_MANIFEST_DIR"), file);
    Some(OsmMap::from_json(&std::fs::read_to_string(path).ok()?).ok()?.build())
}

fn real_map() -> Option<Network> {
    map_from("map.json")
}

struct Sample {
    cars: usize,
    step_ms: f64,
    marshal_ms: f64,
    phases: [f64; STEP_PHASES],
    /// Sleep-scheduler ROI census: [free, frozen, deciding] over the fleet.
    census: [usize; 3],
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
        census: world.state_census(),
    }
}

/// Controlled 2×2: {serial, threads} × {idle-car scheduler off, on}, ramped to a matched
/// car count and measured back-to-back in one process (so machine state is shared). Isolates
/// the accel phase via external routing. Answers whether the scheduler compounds with CPU
/// threads or competes. Run: `--features import,parallel -- --ignored --nocapture`.
#[test]
#[ignore]
fn bench_scheduler_vs_threads() {
    let Some(net) = real_map() else { return };
    let cfg0 = SimConfig::default_config();
    let target = 5500usize;
    // Two passes (order reversed the 2nd time) so thermal drift across the sequence shows up.
    for pass in 0..2 {
        eprintln!("\n=== pass {pass} ===");
        eprintln!("{:>14} {:>6} {:>8} {:>9} {:>8}", "config", "cars", "step_ms", "accel_ms", "ns/car");
        let mut cases = [
            ("serial off", AccelBackend::Serial, false),
            ("serial ON", AccelBackend::Serial, true),
            ("threads off", AccelBackend::Threads, false),
            ("threads ON", AccelBackend::Threads, true),
        ];
        if pass == 1 {
            cases.reverse();
        }
        for (label, backend, sleep) in cases {
            let mut world = NetWorld::new(net.clone(), SimConfig { sleep_scheduler: sleep, ..cfg0 });
            world.set_accel_backend(backend);
            world.set_scheduler_thread_limit(usize::MAX); // keep the scheduler on under threads, so threads+scheduler is measurable
            let pairs = demand::od_pairs(&world.network, 1, 600, DemandSources::new(true, true));
            let mut gen = DemandGenerator::new(&world, &pairs, 1);
            world.install_router(&gen.destinations());
            world.set_external_reroute(true); // exclude the single-threaded flow-field
            let mut t = 0;
            while world.vehicles().len() < target && t < 6000 {
                gen.step(&mut world, cfg0.dt);
                world.step();
                t += 1;
            }
            let s = window(&mut world, &mut gen, cfg0.dt, 120);
            let ns = if s.cars > 0 { s.step_ms * 1e6 / s.cars as f64 } else { 0.0 };
            eprintln!("{label:>14} {:>6} {:>8.2} {:>9.2} {:>8.1}", s.cars, s.step_ms, s.phases[4], ns);
        }
    }
}

/// Install the GPU accel solver, or report that no adapter is available. A no-op
/// (always `false`) without the `gpu` feature, so the harness compiles either way.
#[cfg(feature = "gpu")]
fn enable_gpu_accel(world: &mut NetWorld) -> bool {
    world.enable_gpu_accel()
}
#[cfg(not(feature = "gpu"))]
fn enable_gpu_accel(_world: &mut NetWorld) -> bool {
    false
}

fn scaling_curve(sources: DemandSources, target: usize, external_reroute: bool, accel: AccelBackend, sleep: bool, label: &str) {
    let Some(net) = real_map() else { return };
    let cfg = SimConfig { sleep_scheduler: sleep, ..SimConfig::default_config() };
    let mut world = NetWorld::new(net, cfg);
    // GPU accel needs a solver installed first; without an adapter, fall back rather
    // than skewing the curve with a silent serial run under a GPU label.
    if accel == AccelBackend::Gpu && !enable_gpu_accel(&mut world) {
        eprintln!("\n=== {label}: no GPU adapter, skipping ===");
        return;
    }
    // Requesting Threads engages rayon under `--features parallel`, and cleanly falls
    // back to serial otherwise (so it's a no-op without the feature).
    world.set_accel_backend(accel);
    let pairs = demand::od_pairs(&world.network, 1, target, sources);
    let mut gen = DemandGenerator::new(&world, &pairs, 1);
    world.install_router(&gen.destinations());
    world.set_external_reroute(external_reroute);

    eprintln!("\n=== {label} ===");
    eprintln!("{:>7} {:>8} {:>8} {:>8} {:>10}   phases ms [{}]", "cars", "step_ms", "marshl", "ns/car", "sleepable", PHASE_NAMES.join(" "));
    let mut peak = 0usize;
    let mut prev_cars = 0usize;
    for _ in 0..30 {
        ramp(&mut world, &mut gen, cfg.dt, 150);
        let s = window(&mut world, &mut gen, cfg.dt, 60);
        let ns_car = if s.cars > 0 { s.step_ms * 1e6 / s.cars as f64 } else { 0.0 };
        let phases = s.phases.iter().map(|p| format!("{p:.2}")).collect::<Vec<_>>().join(" ");
        // sleepable = free + frozen, the fraction the active-set scheduler could skip.
        let [free, frozen, deciding] = s.census;
        let sleepable = if s.cars > 0 { 100.0 * (free + frozen) as f64 / s.cars as f64 } else { 0.0 };
        let census = format!("{sleepable:.0}% (F{free}/Z{frozen}/D{deciding})");
        eprintln!("{:>7} {:>8.2} {:>8.2} {:>8.1} {census:>10}   {phases}", s.cars, s.step_ms, s.marshal_ms, ns_car);
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
    scaling_curve(DemandSources::new(true, false), 600, false, AccelBackend::Threads, false, "Millbrae highway saturation (CPU routing)");
}

#[test]
#[ignore]
fn load_balanced_gridlock() {
    // Boundary demand from every gateway, pushed into citywide gridlock.
    scaling_curve(DemandSources::new(true, true), 600, false, AccelBackend::Threads, false, "Millbrae balanced gridlock (CPU routing)");
}

#[test]
#[ignore]
fn load_balanced_gridlock_scheduler() {
    // Same citywide gridlock with the active-set scheduler on: queued sleepers skip the
    // full gather, so the accel (and lane_changes/advance) phase-ms should track the
    // *deciding* count rather than the fleet. Compare step_ms / phase columns against
    // `load_balanced_gridlock` (scheduler off) at matching car counts.
    scaling_curve(DemandSources::new(true, true), 600, false, AccelBackend::Threads, true, "Millbrae balanced gridlock (active-set scheduler)");
}

#[test]
#[ignore]
#[cfg(feature = "gpu")]
fn load_balanced_gridlock_gpu_accel() {
    // Same citywide gridlock, but the per-vehicle evaluate pass runs on the GPU
    // (`accel.wgsl`). Compare its step_ms/phase-4 (accel) column against the CPU
    // curve to see where the GPU fold starts paying off. Run with `--features
    // gpu,import -- --ignored --nocapture`.
    scaling_curve(DemandSources::new(true, true), 600, false, AccelBackend::Gpu, false, "Millbrae balanced gridlock (GPU accel)");
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
    // Rate, not absolute count: multi-lane injection now loads the freeway to real
    // density (before, all inflow stacked on one lane, so the freeway ran nearly
    // empty and any absolute count was tiny). Phantom stops are a *per-car* defect,
    // so measure them per 1000 freeway car-observations — load-invariant, and what
    // actually distinguishes the reaction-delay bug (endemic, ~1-in-5 cars) from the
    // rare genuine emergency brake under congestion.
    let mut prev: std::collections::HashMap<u32, f64> = Default::default();
    let (mut stops, mut freeway_obs) = (0u64, 0u64);
    for _ in 0..2500 {
        gen.step(&mut world, cfg.dt);
        world.step();
        for v in world.vehicles() {
            let k = world.network.link(world.network.lane(v.lane).link).kind;
            if format!("{k:?}") != "Freeway" {
                continue;
            }
            freeway_obs += 1;
            if prev.get(&v.id).copied().unwrap_or(v.speed) > 8.0 && v.speed < 1.0 {
                stops += 1;
            }
        }
        prev.clear();
        for v in world.vehicles() { prev.insert(v.id, v.speed); }
    }
    let per_1k = stops as f64 / freeway_obs.max(1) as f64 * 1000.0;
    // The bug dead-stopped ~279 of a near-empty freeway (a huge per-car rate); genuine
    // congestion braking is well under 5 per 1000 observations.
    assert!(per_1k < 5.0, "freeway phantom stops regressed: {stops} in {freeway_obs} obs = {per_1k:.2}/1000");
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
    scaling_curve(DemandSources::new(true, false), 600, true, AccelBackend::Threads, false, "Millbrae highway saturation (external/GPU routing)");
}

/// The committed Millbrae `map.json` must build *and* render without panicking —
/// the exact path the browser runs at load (`assemble` → world/marking meshes plus
/// the cached signal-head placements). Cheap, so it runs by default rather than
/// `#[ignore]`d; a geometry panic on the real map (e.g. a sub-metre link tripping a
/// clamp) fails here instead of blanking the page.
#[test]
fn real_map_builds_and_renders_without_panicking() {
    let Some(net) = real_map() else { return }; // map.json may be absent in a bare checkout
    let _ = engine::render::geometry::world_mesh(&net);
    let _ = engine::render::geometry::marking_mesh(&net);
    let _ = engine::render::geometry::signal_head_placements(&net);
}

/// The whole-city San Francisco map (~10k nodes, ~66k movements) must build and
/// render without panicking — the browser's load path, and the stress case the O(n)
/// build passes and O(1) conflict lookup were added for. Skipped if `sf.json` isn't
/// present. (A short sim run, gated behind `--ignored` since the full-city step is
/// heavy, checks it also carries traffic without leaking.)
#[test]
fn san_francisco_map_builds_and_renders_without_panicking() {
    let Some(net) = map_from("sf.json") else { return };
    assert!(net.movements.len() > 20_000, "the SF extract is a whole-city map, got {} movements", net.movements.len());
    let _ = engine::render::geometry::world_mesh(&net);
    let _ = engine::render::geometry::marking_mesh(&net);
    let _ = engine::render::geometry::signal_head_placements(&net);
}

#[test]
#[ignore]
fn san_francisco_map_flows_without_leaking() {
    let Some(net) = map_from("sf.json") else { return };
    let cfg = SimConfig::default_config();
    let pairs = demand::od_pairs(&net, 0, 400, DemandSources::new(true, true));
    let mut world = NetWorld::new(net, cfg);
    let mut gen = DemandGenerator::new(&world, &pairs, 0);
    world.install_router(&gen.destinations());
    for _ in 0..200 {
        gen.step(&mut world, cfg.dt);
        world.step();
    }
    eprintln!("san francisco: {} vehicles, {} crashed, {} leaked", world.vehicles().len(), world.crashed(), world.leaked());
    assert_eq!(world.leaked(), 0, "no vehicle vanishes at an SF intersection");
    assert!(world.vehicles().len() > 0, "traffic spawns and routes on the SF map");
}

#[test]
fn san_carlos_map_builds_renders_and_flows_without_panicking() {
    let Some(net) = map_from("sancarlos.json") else { return };
    let _ = engine::render::geometry::world_mesh(&net);
    let _ = engine::render::geometry::marking_mesh(&net);
    let _ = engine::render::geometry::signal_head_placements(&net);

    let cfg = SimConfig::default_config();
    let pairs = demand::od_pairs(&net, 0, 600, DemandSources::new(true, true));
    let mut world = NetWorld::new(net, cfg);
    let mut gen = DemandGenerator::new(&world, &pairs, 0);
    world.install_router(&gen.destinations());
    for _ in 0..1500 {
        gen.step(&mut world, cfg.dt);
        world.step();
    }
    eprintln!("san carlos: {} exited, {} crashed, {} leaked", world.exited(), world.crashed(), world.leaked());
    assert!(world.exited() > 0, "the San Carlos map carries traffic to its destinations");
    assert_eq!(world.leaked(), 0, "no vehicle vanishes at a San Carlos intersection");
    assert!(
        world.crashed() < world.exited() / 20,
        "collisions stay rare relative to throughput: {} crashed vs {} exited",
        world.crashed(),
        world.exited(),
    );
}

/// The Bay Area peninsula freeway extract (motorway/trunk + ramps only): it should
/// build, render, and carry mainline-and-ramp traffic to its exits without leaking.
/// A freeways-only network exercises the interchange (free-flow merge/diverge) path
/// rather than the signalized grid. Skipped if `peninsula.json` isn't present.
#[test]
fn peninsula_freeway_map_builds_renders_and_flows_without_panicking() {
    let Some(net) = map_from("peninsula.json") else { return };
    let _ = engine::render::geometry::world_mesh(&net);
    let _ = engine::render::geometry::marking_mesh(&net);
    let _ = engine::render::geometry::signal_head_placements(&net);

    let cfg = SimConfig::default_config();
    let pairs = demand::od_pairs(&net, 0, 600, DemandSources::new(true, true));
    assert!(!pairs.is_empty(), "the freeway network yields routable OD demand");
    let mut world = NetWorld::new(net, cfg);
    let mut gen = DemandGenerator::new(&world, &pairs, 0);
    world.install_router(&gen.destinations());
    for _ in 0..1500 {
        gen.step(&mut world, cfg.dt);
        world.step();
    }
    eprintln!("peninsula: {} exited, {} crashed, {} leaked", world.exited(), world.crashed(), world.leaked());
    assert!(world.exited() > 0, "the peninsula freeways carry traffic to their exits");
    assert_eq!(world.leaked(), 0, "no vehicle vanishes at a peninsula interchange");
}

/// The routing field must be acyclic: following `next_hop` from any link toward
/// any destination must reach it without revisiting a link. This is structural
/// (`next_hop = argmin(cost+dist)` with positive costs strictly decreases the
/// distance-to-destination each hop), and it rules out routing as a source of the
/// "cars running in circles" report — a car never chases a looping next-hop chain.
#[test]
fn routing_field_is_acyclic() {
    use engine::sim::network::LinkId;
    use engine::sim::router::FieldRouter;
    let Some(net) = real_map() else { return };
    let pairs = demand::od_pairs(&net, 0, 600, DemandSources::new(true, true));
    let mut dests: Vec<LinkId> = pairs.iter().map(|p| p.dest).collect();
    dests.sort_by_key(|d| d.0);
    dests.dedup();
    let cost: Vec<u64> = (0..net.links.len() as u32).map(|i| net.link_travel_time_ms(LinkId(i))).collect();
    let router = FieldRouter::new(&net, &dests, &cost);
    for &dest in router.destinations() {
        for from in (0..net.links.len() as u32).map(LinkId) {
            let mut cur = from;
            let mut seen = std::collections::HashSet::new();
            loop {
                if cur == dest { break; } // reached the destination
                assert!(seen.insert(cur.0), "routing cycle toward {dest:?} from {from:?} at {cur:?}");
                match router.next_hop(dest, cur) { Some(n) => cur = n, None => break } // dead-end/unreachable
            }
        }
    }
}

/// Under a fixed field (the browser's external-reroute default) and heavy
/// rush-hour congestion, circling must stay negligible. A single revisit (max 2)
/// is realistic missed-turn recovery — a car kept out of its turn lane by traffic
/// goes around the block once and retries. At most one car may double-loop
/// (revisit a link 3 times) and none may circle beyond that; a flood of either is
/// the regression this guards against.
#[test]
fn real_map_traffic_does_not_circle() {
    use std::collections::HashMap;
    let Some(net) = real_map() else { return };
    let cfg = SimConfig::default_config();
    let mut world = NetWorld::new(net, cfg);
    let pairs = demand::od_pairs(&world.network, 0, 600, DemandSources::new(true, true));
    let mut gen = DemandGenerator::new(&world, &pairs, 0);
    world.install_router(&gen.destinations());
    world.set_external_reroute(true); // browser default: field stays fixed, worst case for looping
    gen.set_rush_hour(&world.network, true); // heavy congestion keeps turn lanes blocked
    let mut hist: HashMap<u32, Vec<u32>> = HashMap::new();
    for _ in 0..1500 {
        gen.step(&mut world, cfg.dt);
        world.step();
        for v in world.vehicles() {
            let link = world.network.lane(v.lane).link.0;
            let h = hist.entry(v.id).or_default();
            if h.last() != Some(&link) { h.push(link); } // dedup consecutive: keep genuine revisits
        }
    }
    let (mut looped, mut recovered, mut worst) = (0u64, 0u64, 0i32);
    for h in hist.values() {
        let mut counts: HashMap<u32, i32> = HashMap::new();
        let mut m = 0;
        for &l in h {
            let c = counts.entry(l).or_insert(0);
            *c += 1;
            m = m.max(*c);
        }
        if m >= 3 { looped += 1; }
        if m == 2 { recovered += 1; }
        worst = worst.max(m);
    }
    assert!(looped <= 1 && worst <= 3, "persistent circling: {looped} vehicles revisit a link 3+ times (worst {worst})");
    // Missed-turn recovery is realistic but should stay rare — guard against a
    // regression that floods it (baseline ≈ 0.3% of tracked vehicles).
    assert!(
        recovered < hist.len() as u64 / 50,
        "missed-turn recovery spiked: {recovered} of {} vehicles looped back once",
        hist.len(),
    );
}
