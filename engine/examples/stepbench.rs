//! Step-time bench for the scheduler × backend matrix on the real map.
//!
//! Loads the region at rush-hour demand until the fleet is heavy, then measures
//! mean `step()` time under each {backend, sleep} configuration in interleaved
//! rounds (so drifting load cancels out of the comparison). Run with:
//!
//! ```sh
//! cargo run --release --features "import parallel" --example stepbench -- \
//!   --map ../web/public/map.json --lodes ../web/public/map.lodes.json
//! ```

use engine::sim::config::SimConfig;
use engine::sim::demand::{self, DemandGenerator, DemandSources};
use engine::sim::map::OsmMap;
use engine::sim::net_world::{prof_take, AccelBackend, NetWorld, PHASE_NAMES, STEP_PHASES};
use std::time::Instant;

fn main() {
    let mut map_path = String::new();
    let mut lodes_path = None;
    let mut target_fleet = 5000usize;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| panic!("missing value for {a}"));
        match a.as_str() {
            "--map" => map_path = val(),
            "--lodes" => lodes_path = Some(val()),
            "--fleet" => target_fleet = val().parse().expect("fleet"),
            other => panic!("unknown arg {other}"),
        }
    }
    let raw = std::fs::read_to_string(&map_path).expect("read map");
    let map = OsmMap::from_json(&raw).expect("parse map");
    let mut world = NetWorld::new(map.build(), SimConfig::default_config());

    let commute = lodes_path.as_ref().map(|p| {
        let raw = std::fs::read_to_string(p).expect("read lodes");
        demand::CommuteOd::from_json(&raw).expect("parse lodes")
    });
    let sources = DemandSources::with_rush_hour(true, true, true);
    let pairs = demand::od_pairs_with_commute(&world.network, 7, 48, sources, commute.as_ref());
    let mut gen = DemandGenerator::new(&world, &pairs, 7);
    gen.set_rush_hour(&world.network, true);
    gen.set_day_compression(12.0);
    gen.set_rate_scale(3.0); // push hard so the fleet climbs fast and queues form
    gen.resume_clock(7.0 * 3600.0, 0);
    world.install_router(&gen.destinations());

    let dt = SimConfig::default_config().dt;
    print!("loading fleet");
    let mut warm = 0u32;
    while world.vehicles().len() < target_fleet && warm < 40_000 {
        gen.step(&mut world, dt);
        world.step();
        warm += 1;
        if warm % 2000 == 0 {
            print!(" {}", world.vehicles().len());
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
    }
    println!("\nfleet {} after {warm} ticks", world.vehicles().len());

    let configs: &[(&str, AccelBackend, bool)] = &[
        ("serial sleep-off ", AccelBackend::Serial, false),
        ("serial sleep-on  ", AccelBackend::Serial, true),
        ("threads sleep-off", AccelBackend::Threads, false),
        ("threads sleep-on ", AccelBackend::Threads, true),
    ];
    const ROUNDS: usize = 4;
    const SETTLE: usize = 20;
    const MEASURE: usize = 120;
    let mut total_us = vec![0u128; configs.len()];
    let mut total_asleep = vec![0usize; configs.len()];
    let mut samples = vec![0u32; configs.len()];
    let mut phases = vec![[0.0f64; STEP_PHASES]; configs.len()];
    for _ in 0..ROUNDS {
        for (ci, &(_, backend, sleep)) in configs.iter().enumerate() {
            world.set_accel_backend(backend);
            world.set_sleep_scheduler(sleep);
            for _ in 0..SETTLE {
                gen.step(&mut world, dt);
                world.step();
            }
            prof_take();
            for _ in 0..MEASURE {
                gen.step(&mut world, dt);
                let t0 = Instant::now();
                world.step();
                total_us[ci] += t0.elapsed().as_micros();
                total_asleep[ci] += world.asleep_count();
                samples[ci] += 1;
            }
            let ph = prof_take();
            for k in 0..STEP_PHASES {
                phases[ci][k] += ph[k];
            }
        }
    }
    println!("fleet at end: {}", world.vehicles().len());
    for (ci, &(name, ..)) in configs.iter().enumerate() {
        println!(
            "{name}  mean step {:>7.0} us   asleep {:>5}",
            total_us[ci] as f64 / samples[ci] as f64,
            total_asleep[ci] / samples[ci] as usize,
        );
        let per: Vec<String> = (0..STEP_PHASES)
            .map(|k| format!("{}={:.0}us", PHASE_NAMES[k], phases[ci][k] * 1000.0 / samples[ci] as f64))
            .collect();
        println!("    {}", per.join(" "));
    }
}
