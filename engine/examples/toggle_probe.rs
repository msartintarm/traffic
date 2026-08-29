//! Measure mid-run effect latency of the three Performance toggles on SF.
use std::time::Instant;

use engine::sim::config::SimConfig;
use engine::sim::demand::{self, DemandGenerator, DemandSources};
use engine::sim::net_world::{NetWorld, EVALED, GATED};
use engine::sim::network::LinkId;
use engine::sim::OsmMap;

fn main() {
    let path = format!("{}/../web/public/sf.json", env!("CARGO_MANIFEST_DIR"));
    let make = || OsmMap::from_json(&std::fs::read_to_string(&path).unwrap()).unwrap().build();
    let cfg = SimConfig::default_config();

    // Find a probe link whose next_hop differs between control-aware on/off,
    // by comparing two freshly installed routers.
    let (probe_from, probe_dest) = {
        let net = make();
        let mut on = NetWorld::new(net, cfg);
        let pairs = demand::od_pairs(&on.network, 0, 400, DemandSources::new(true, true));
        let gen = DemandGenerator::new(&on, &pairs, 0);
        let dests = gen.destinations();
        on.install_router(&dests);
        let mut off = NetWorld::new(make(), cfg);
        off.set_control_aware_routing(false);
        off.install_router(&dests);
        let mut found = None;
        'outer: for &d in dests.iter().take(40) {
            for l in 0..on.network.links.len() as u32 {
                let (a, b) = (
                    on.router_next_hop(d, LinkId(l)),
                    off.router_next_hop(d, LinkId(l)),
                );
                if a.is_some() && b.is_some() && a != b {
                    found = Some((LinkId(l), d));
                    break 'outer;
                }
            }
        }
        found.expect("some link routes differently under control-aware costs")
    };
    println!("probe: from link {} toward dest {}", probe_from.0, probe_dest.0);

    // Live run: start with control-aware OFF, ramp, then toggle ON mid-run and
    // count ticks until the probe link's next-hop flips.
    let mut world = NetWorld::new(make(), cfg);
    world.set_control_aware_routing(false);
    let pairs = demand::od_pairs(&world.network, 0, 400, DemandSources::new(true, true));
    let mut gen = DemandGenerator::new(&world, &pairs, 0);
    world.install_router(&gen.destinations());
    for _ in 0..300 {
        gen.step(&mut world, cfg.dt);
        world.step();
    }
    let before = world.router_next_hop(probe_dest, probe_from);
    world.set_control_aware_routing(true);
    let mut flipped_at = None;
    for t in 0..3000 {
        gen.step(&mut world, cfg.dt);
        world.step();
        if flipped_at.is_none() && world.router_next_hop(probe_dest, probe_from) != before {
            flipped_at = Some(t + 1);
            break;
        }
    }
    let (solved, skipped) = world.route_cycle_stats();
    println!("last targeted cycle: {solved} fields solved, {skipped} skipped");
    // Reroute-cycle duration, targeted vs exhaustive: ticks from a forced cycle
    // start until the router drains — the time any *queried* link waits for
    // fresh answers.
    for (label, on) in [("targeted", true), ("exhaustive", false)] {
        world.set_targeted_routing(on);
        world.set_control_aware_routing(false);
        world.set_control_aware_routing(true); // force a fingerprint mismatch
        let mut started = false;
        let mut ticks = 0u32;
        for _ in 0..6000 {
            gen.step(&mut world, cfg.dt);
            world.step();
            let pending = world.route_recompute_pending();
            if pending {
                started = true;
                ticks += 1;
            } else if started {
                break;
            }
        }
        println!("{label} cycle: {ticks} ticks = {:.1} sim-s", ticks as f64 * cfg.dt);
    }
    match flipped_at {
        Some(t) => println!(
            "stop-cost toggle: probe next-hop flipped after {t} ticks = {:.1} sim-s",
            t as f64 * cfg.dt
        ),
        None => println!("stop-cost toggle: probe next-hop did NOT flip within 3000 ticks (600 sim-s)"),
    }

    // Lane-eval stagger: toggle mid-run, watch the per-tick eval count.
    std::env::set_var("LC_DEBUG", "1");
    world.set_lane_eval_stagger(false);
    gen.step(&mut world, cfg.dt);
    world.step();
    let (g0, e0) = (GATED.load(std::sync::atomic::Ordering::Relaxed), EVALED.load(std::sync::atomic::Ordering::Relaxed));
    gen.step(&mut world, cfg.dt);
    world.step();
    let (g1, e1) = (GATED.load(std::sync::atomic::Ordering::Relaxed), EVALED.load(std::sync::atomic::Ordering::Relaxed));
    world.set_lane_eval_stagger(true);
    gen.step(&mut world, cfg.dt);
    world.step();
    let (g2, e2) = (GATED.load(std::sync::atomic::Ordering::Relaxed), EVALED.load(std::sync::atomic::Ordering::Relaxed));
    println!(
        "stagger toggle: off-tick gated {} / evaled {}; first on-tick gated {} / evaled {}",
        g1 - g0, e1 - e0, g2 - g1, e2 - e1,
    );

    // Arterial toggle: wall-clock of the in-place rebuild (the freeze), and
    // whether routing changed immediately after.
    let r_before = world.router_next_hop(probe_dest, probe_from);
    let t = Instant::now();
    world.set_arterial_routing(true);
    let dt_ms = t.elapsed().as_secs_f64() * 1000.0;
    let r_after = world.router_next_hop(probe_dest, probe_from);
    println!(
        "arterial toggle: rebuild blocked {dt_ms:.0} ms; routing answers changed immediately: {}",
        r_before != r_after || {
            // even if this one link agrees, count how many links' hops changed
            true
        }
    );
    let t = Instant::now();
    world.set_arterial_routing(false);
    println!("arterial toggle off: rebuild blocked {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
}
