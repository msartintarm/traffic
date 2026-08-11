//! Validation-harness unit tier: score the sim's emergent output against
//! traffic-engineering theory on synthetic fixtures — the yardsticks every
//! recalibration phase is measured by. The CI tier (`examples/scorecard.rs`)
//! scores real maps against real counts; these run under plain `cargo test
//! --features import` in seconds.

#![cfg(feature = "import")]

use engine::sim::config::{DriverConfig, SimConfig};
use engine::sim::map::{LinkSpec, NodeSpec, OsmMap};
use engine::sim::measure::Measurement;
use engine::sim::net_world::NetWorld;
use engine::sim::network::LinkId;
use engine::sim::world::World;

/// Peak flow (veh/h/lane) of the closed-ring fundamental diagram for `driver`,
/// scanned across the capacity-region densities.
fn ring_capacity_vph(driver: DriverConfig) -> f64 {
    let length = 1000.0;
    let mut peak: f64 = 0.0;
    for count in [22u32, 26, 28, 30, 32, 36, 40] {
        let mut w = World::uniform_ring(SimConfig::default_config(), length, count, driver);
        w.run_ticks(6000);
        peak = peak.max(w.flow() * 3600.0);
    }
    peak
}

/// The ring's capacity must sit in the observed US-101 queue-discharge range —
/// the P2.2 calibration target (T = 1.2 s / a = 1.5 / b = 2.0 puts the IDM
/// theoretical maximum ≈ 2,130 veh/h/ln).
#[test]
fn ring_capacity_sits_in_the_calibrated_band() {
    let capacity = ring_capacity_vph(DriverConfig::car());
    assert!(
        (1900.0..=2300.0).contains(&capacity),
        "ring capacity {capacity:.0} veh/h/ln outside the US-101 discharge band"
    );
}

/// Priority T-intersection fixture: a two-way major road (primary, 15 m/s)
/// crossed by a give-way minor street (residential, 10 m/s). Returns
/// `(minor capacity veh/h, major windowed flow veh/h)` under a saturated minor
/// approach and the given per-direction major demand (veh/h).
fn priority_cross_capacity(major_vph_per_dir: f64) -> (f64, f64) {
    let (cap, major, _) = priority_cross_capacity_full(major_vph_per_dir);
    (cap, major)
}

fn priority_cross_capacity_full(major_vph_per_dir: f64) -> (f64, f64, [u32; 2]) {
    let nodes = vec![
        NodeSpec::uncontrolled(1, -250.0, 0.0),
        NodeSpec::give_way(2, 0.0, 0.0),
        NodeSpec::uncontrolled(3, 250.0, 0.0),
        NodeSpec::uncontrolled(4, 0.0, 250.0),
        NodeSpec::uncontrolled(5, 0.0, -250.0),
    ];
    let link = |a: i64, b: i64, class: &str, speed: f64| {
        let mut l = LinkSpec::oneway(a, b, 1, speed);
        l.road_class = class.to_string();
        l.name = format!("{class} {a}-{b}");
        l
    };
    let links = vec![
        link(1, 2, "primary", 15.0),     // 0: W→C
        link(2, 3, "primary", 15.0),     // 1: C→E
        link(3, 2, "primary", 15.0),     // 2: E→C
        link(2, 1, "primary", 15.0),     // 3: C→W
        link(4, 2, "residential", 10.0), // 4: N→C
        link(2, 5, "residential", 10.0), // 5: C→S
        link(5, 2, "residential", 10.0), // 6: S→C
        link(2, 4, "residential", 10.0), // 7: C→N
    ];
    let map = OsmMap { nodes, links };
    let mut world = NetWorld::new(map.build(), SimConfig::default_config());

    let dt = SimConfig::default_config().dt;
    let driver = DriverConfig::car();
    // Poisson arrivals (per-tick Bernoulli), independent per direction — the
    // arrival process HCM's gap-acceptance capacity assumes. A deterministic
    // synchronized metronome leaves *no* headway above the critical gap and
    // (wrongly) proves the minor street can never cross.
    let p_spawn = (major_vph_per_dir / 3600.0 * dt).min(1.0);
    let rand01 = |a: u64, b: u64| {
        let mut x = a.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(b);
        x ^= x >> 33;
        x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        x ^= x >> 33;
        (x >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut id = 0u32;
    let spawn = |world: &mut NetWorld, id: &mut u32, route: Vec<LinkId>, speed: f64| {
        if world.spawn_routed(*id, route, speed, driver) {
            *id += 1;
        }
    };
    // Warm up 200 s, then measure 900 s.
    let mut meas = None;
    for tick in 0..(5500u64) {
        if rand01(tick, 1) < p_spawn {
            spawn(&mut world, &mut id, vec![LinkId(0), LinkId(1)], 15.0);
        }
        if rand01(tick, 2) < p_spawn {
            spawn(&mut world, &mut id, vec![LinkId(2), LinkId(3)], 15.0);
        }
        // Saturated minor approach: refill whenever the entrance clears.
        spawn(&mut world, &mut id, vec![LinkId(4), LinkId(5)], 8.0);
        world.step();
        if tick == 1000 {
            meas = Some(Measurement::begin(&world));
        }
        if let Some(m) = meas.as_mut() {
            m.sample(&world, dt);
        }
    }
    let m = meas.unwrap();
    let flows = m.link_flows(&world);
    // Minor capacity = completed crossings onto the departure leg; major flow =
    // arrivals on both conflicting approaches.
    (flows[5], flows[0] + flows[2], world.crash_counts())
}

/// The give-way minor street's capacity must be finite (gap-limited), fall as the
/// conflicting major volume rises, and leave the major stream essentially
/// unimpeded — the qualitative HCM two-way-stop-control envelope, asserted in the
/// regimes the current parameters get right (≤600 veh/h/dir).
///
/// Known deviation, deliberately *not* asserted yet: above ~900 veh/h/dir the
/// too-short critical gap (4.0 s shrinking to 1.5 s with impatience) lets minor
/// cars nose into the box, majors brake for the occupied box, and the priority
/// road's throughput collapses to ~⅓ — a priority inversion real TWSC junctions
/// don't show. P2.1 (HCM per-movement gaps) takes "major stays unimpeded at
/// 900 veh/h/dir" as its acceptance criterion and tightens this fixture to ±15%
/// of the HCM capacity formula.
#[test]
fn minor_street_capacity_is_gap_limited_and_falls_with_major_volume() {
    let (cap_low, major_low) = priority_cross_capacity(300.0);
    let (cap_mid, major_mid) = priority_cross_capacity(600.0);
    assert!(cap_low > 50.0, "minor street crosses at all: {cap_low:.0} veh/h");
    assert!(cap_low < 1400.0, "minor capacity is gap-limited, got {cap_low:.0} veh/h");
    assert!(
        cap_mid < cap_low * 0.85,
        "capacity falls with conflicting volume: {cap_mid:.0} !< 0.85×{cap_low:.0}"
    );
    assert!(major_low > 300.0 * 2.0 * 0.7, "the major stream is essentially unimpeded: {major_low:.0}");
    assert!(major_mid > 600.0 * 2.0 * 0.7, "the major stream stays unimpeded at volume: {major_mid:.0}");
}

/// Diagnostic scan (ignored): print the minor-capacity curve across major volumes.
#[test]
#[ignore]
fn print_priority_capacity_curve() {
    for v in [100.0, 300.0, 600.0, 900.0, 1200.0] {
        let (cap, major, crashes) = priority_cross_capacity_full(v);
        println!("major demand {v:>6.0}/dir → minor {cap:>5.0} veh/h, major flow {major:>6.0} veh/h, crashes {crashes:?}");
    }
}

/// Diagnostic (ignored): per-lane state at the US-101 gateway lane-drop seam.
#[test]
#[ignore]
fn diag_gateway_seam_lanes() {
    use engine::sim::demand::{self, DemandGenerator, DemandSources};
    use engine::sim::net_world::NetWorld;
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
    let Ok(text) = std::fs::read_to_string(path) else { return };
    let net = engine::sim::OsmMap::from_json(&text).expect("map json").build();
    let mut world = NetWorld::new(net, SimConfig::default_config());
    let sources = DemandSources::with_rush_hour(true, false, true);
    let pairs = demand::od_pairs_with_commute(&world.network, 0xC0FFEE, 48, sources, None);
    let mut gen = DemandGenerator::new(&world, &pairs, 0xC0FFEE);
    gen.set_rush_hour(&world.network, true);
    gen.set_day_compression(12.0);
    gen.resume_clock(7.0 * 3600.0, 0);
    world.install_router(&gen.destinations());
    for _ in 0..4000 {
        gen.step(&mut world, 0.2);
        world.step();
    }
    for lid in [990u32, 991, 1380, 988] {
        let link = *world.network.link(LinkId(lid));
        println!("link {lid} ({} lanes):", link.lane_count);
        for k in 0..link.lane_count {
            let lane = engine::sim::network::LaneId(link.lane_start.0 + k);
            let ln = world.network.lane(lane);
            let cars: Vec<&engine::sim::NetVehicle> =
                world.vehicles().iter().filter(|v| v.lane == lane).collect();
            let n = cars.len();
            let mean_v = if n > 0 { cars.iter().map(|v| v.speed).sum::<f64>() / n as f64 } else { f64::NAN };
            let near_end = cars.iter().filter(|v| ln.length - v.position < 60.0).count();
            println!("  lane {k}: {n} cars, mean v {mean_v:.1}, {near_end} in last 60 m (len {:.0})", ln.length);
        }
    }
}

/// Diagnostic (ignored): movement wiring of the gateway seam lanes.
#[test]
#[ignore]
fn diag_gateway_seam_movements() {
    use engine::sim::network::LaneId;
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
    let Ok(text) = std::fs::read_to_string(path) else { return };
    let net = engine::sim::OsmMap::from_json(&text).expect("map json").build();
    for lid in [990u32, 1380] {
        let link = *net.link(LinkId(lid));
        println!("link {lid} ({} lanes) turn_lanes={:?}:", link.lane_count, net.link_turn_lanes[lid as usize]);
        for k in 0..link.lane_count {
            let lane = LaneId(link.lane_start.0 + k);
            let mvs: Vec<String> = net
                .movements_of(lane)
                .iter()
                .map(|m| {
                    let to = net.lane(m.to_lane);
                    format!("→link{} lane{}", to.link.0, to.index_in_link)
                })
                .collect();
            println!("  lane {k}: {} movements {:?}", mvs.len(), mvs);
        }
    }
}

/// Diagnostic (ignored): gate-relevant flags on the gateway seam movements.
#[test]
#[ignore]
fn diag_gateway_seam_flags() {
    use engine::sim::network::{LaneId, MovementId};
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
    let Ok(text) = std::fs::read_to_string(path) else { return };
    let net = engine::sim::OsmMap::from_json(&text).expect("map json").build();
    for lid in [990u32, 1380] {
        let link = *net.link(LinkId(lid));
        println!("link {lid}:");
        for k in 0..link.lane_count {
            let lane = LaneId(link.lane_start.0 + k);
            let ln = net.lane(lane);
            for j in 0..ln.movement_count {
                let mid = MovementId(ln.movement_start.0 + j);
                let conflicts = net.conflicts.iter().filter(|c| c.a == mid || c.b == mid).count();
                println!(
                    "  lane {k} mv{}: interchange={} continuation={} conflicts={} node={}",
                    mid.0,
                    net.is_interchange_movement(mid),
                    net.is_continuation_seam(mid),
                    conflicts,
                    net.movement(mid).node.0,
                );
            }
        }
    }
}

/// Queue-discharge saturation flow: park a standing queue at a red, release it,
/// and measure the between-vehicle headways crossing the line. US signalized
/// approaches discharge at ~1,800–1,900 veh/h/lane (headway ≈ 1.9–2.0 s).
#[test]
#[ignore]
fn diag_queue_discharge_headway() {
    use engine::sim::map::SignalPlan;
    let plan = SignalPlan { green_secs: 60.0, yellow_secs: 4.0, offset: 30.0 };
    let net = OsmMap {
        nodes: vec![
            NodeSpec::uncontrolled(1, -600.0, 0.0),
            NodeSpec::signalized(2, 0.0, 0.0, plan),
            NodeSpec::uncontrolled(3, 400.0, 0.0),
            NodeSpec::uncontrolled(4, 0.0, -200.0),
            NodeSpec::uncontrolled(5, 0.0, 200.0),
        ],
        links: {
            let mut v = vec![LinkSpec::oneway(1, 2, 1, 15.0), LinkSpec::oneway(2, 3, 1, 15.0)];
            v.extend(LinkSpec::twoway(4, 2, 1, 10.0));
            v.extend(LinkSpec::twoway(2, 5, 1, 10.0));
            v
        },
    }
    .build();
    let mut w = NetWorld::new(net, SimConfig { red_run_prob: 0.0, ..SimConfig::default_config() });
    let mut id = 0u32;
    // Fill a standing queue during red.
    for _ in 0..1500 {
        if w.spawn_routed(id, vec![LinkId(0), LinkId(1)], 10.0, DriverConfig { accel_noise: 0.0, ..DriverConfig::car() }) {
            id += 1;
        }
        w.step();
    }
    // Now record crossing times over several cycles.
    let mut crossings: Vec<f64> = Vec::new();
    let mut t = 0.0f64;
    let mut last_count = w.link_entry_counts()[1];
    for _ in 0..9000 {
        if w.spawn_routed(id, vec![LinkId(0), LinkId(1)], 10.0, DriverConfig { accel_noise: 0.0, ..DriverConfig::car() }) {
            id += 1;
        }
        w.step();
        t += 0.2;
        let c = w.link_entry_counts()[1];
        for _ in 0..(c - last_count) {
            crossings.push(t);
        }
        last_count = c;
    }
    let mut headways: Vec<f64> = crossings.windows(2).map(|w| w[1] - w[0]).collect();
    headways.sort_by(|a, b| a.total_cmp(b));
    let pick = |q: f64| headways[((headways.len() - 1) as f64 * q) as usize];
    println!(
        "{} crossings; headway p10 {:.1} p25 {:.1} p50 {:.1} p75 {:.1} p90 {:.1}",
        crossings.len(),
        pick(0.1),
        pick(0.25),
        pick(0.5),
        pick(0.75),
        pick(0.9),
    );
    // Watch the line for 15 s: approach front car vs receiver tail.
    let l0 = *world_link(&w, 0);
    for t in 0..75 {
        w.step();
        let front = w
            .vehicles()
            .iter()
            .filter(|v| w_lane_link(&w, v) == 0 && !v.is_crossing())
            .max_by(|a, b| a.position.total_cmp(&b.position));
        let crossing = w.vehicles().iter().find(|v| v.is_crossing());
        let tail = w
            .vehicles()
            .iter()
            .filter(|v| w_lane_link(&w, v) == 1)
            .min_by(|a, b| a.position.total_cmp(&b.position));
        if t % 3 == 0 {
            println!(
                "t{:>4.1} front {:?} crossing {:?} tail {:?}",
                t as f64 * 0.2,
                front.map(|v| (v.id, format!("{:.1}/{:.1}", v.position, l0), format!("v{:.1}", v.speed))),
                crossing.map(|v| (v.id, format!("v{:.1}", v.speed))),
                tail.map(|v| (v.id, format!("{:.1}", v.position), format!("v{:.1}", v.speed))),
            );
        }
    }
}

fn world_link(w: &NetWorld, link: u32) -> &f64 {
    Box::leak(Box::new(w.network.lane(w.network.link(LinkId(link)).lane_start).length))
}

fn w_lane_link(w: &NetWorld, v: &engine::sim::NetVehicle) -> u32 {
    w.network.lane(v.lane).link.0
}

/// The scraped bus-route traces resolve into usable link chains on the real map.
#[test]
fn real_map_bus_routes_resolve_to_chains() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
    let Ok(text) = std::fs::read_to_string(path) else { return };
    let routes = engine::sim::map::bus_routes_from_json(&text);
    if routes.is_empty() {
        return; // map predates transit scraping
    }
    let net = engine::sim::OsmMap::from_json(&text).expect("map json").build();
    let resolved: Vec<usize> = routes
        .iter()
        .filter_map(|(_, pts)| net.resolve_route_chain(pts).map(|c| c.len()))
        .collect();
    assert!(
        resolved.len() * 2 >= routes.len(),
        "most scraped lines resolve ({} of {})",
        resolved.len(),
        routes.len()
    );
    assert!(resolved.iter().all(|&n| n >= 3), "chains are real routes, not stubs");
}
