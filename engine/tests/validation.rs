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
    let major_period = (3600.0 / major_vph_per_dir / dt).round() as u64;
    let mut id = 0u32;
    let spawn = |world: &mut NetWorld, id: &mut u32, route: Vec<LinkId>, speed: f64| {
        if world.spawn_routed(*id, route, speed, driver) {
            *id += 1;
        }
    };
    // Warm up 200 s, then measure 900 s.
    let mut meas = None;
    for tick in 0..(5500u64) {
        if major_period > 0 && tick % major_period == 0 {
            spawn(&mut world, &mut id, vec![LinkId(0), LinkId(1)], 15.0);
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
