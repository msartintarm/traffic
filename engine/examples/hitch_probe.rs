//! Reproduce the browser's periodic hitches natively: run the real map under
//! the browser's configuration (LOCAL routing — stepbench measures the field
//! router instead), record per-tick step+demand times, and print the spikes
//! with their tick phase so the period identifies the mechanism
//! (15 ticks = reroute interval, 16 = locality reorder, 75 = calibration
//! window at 60× day compression, 225 = demand churn epoch).
//!
//! ```sh
//! cargo run --release --features "import parallel" --example hitch_probe -- \
//!   --map ../web/public/santaclara.json --lodes ../web/public/santaclara.lodes.json
//! ```

use engine::sim::config::SimConfig;
use engine::sim::demand::{self, DemandGenerator, DemandSources};
use engine::sim::map::OsmMap;
use engine::sim::net_world::{prof_take, NetWorld, PHASE_NAMES, STEP_PHASES};
use std::time::Instant;

fn main() {
    let mut map_path = String::new();
    let mut lodes_path = None;
    let mut warm_ticks = 3000usize;
    let mut measure_ticks = 450usize;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| panic!("missing value for {a}"));
        match a.as_str() {
            "--map" => map_path = val(),
            "--lodes" => lodes_path = Some(val()),
            "--warm" => warm_ticks = val().parse().expect("warm"),
            "--measure" => measure_ticks = val().parse().expect("measure"),
            other => panic!("unknown arg {other}"),
        }
    }
    let raw = std::fs::read_to_string(&map_path).expect("read map");
    let map = OsmMap::from_json(&raw).expect("parse map");
    let cfg = SimConfig { sleep_scheduler: true, ..SimConfig::default_config() };
    let mut world = NetWorld::new(map.build(), cfg);
    world.set_local_routing(true); // the browser default — the config that hitches

    let commute = lodes_path.as_ref().map(|p| {
        let raw = std::fs::read_to_string(p).expect("read lodes");
        demand::CommuteOd::from_json(&raw).expect("parse lodes")
    });
    let sources = DemandSources::new(true, true);
    let pairs = demand::od_pairs_with_commute(&world.network, 7, 48, sources, commute.as_ref());
    let mut gen = DemandGenerator::new(&world, &pairs, 7);
    world.install_router(&gen.destinations());

    let dt = cfg.dt;
    print!("warming");
    for t in 0..warm_ticks {
        gen.step(&mut world, dt);
        world.step();
        if t % 500 == 0 {
            print!(" {}", world.vehicles().len());
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
    }
    println!("\nfleet {} — measuring {measure_ticks} ticks", world.vehicles().len());

    prof_take();
    let mut rows: Vec<(usize, f64, f64, [f64; STEP_PHASES])> = Vec::with_capacity(measure_ticks);
    for t in 0..measure_ticks {
        let tg = Instant::now();
        gen.step(&mut world, dt);
        let gen_ms = tg.elapsed().as_secs_f64() * 1e3;
        let t0 = Instant::now();
        world.step();
        let step_ms = t0.elapsed().as_secs_f64() * 1e3;
        rows.push((t, step_ms, gen_ms, prof_take()));
    }

    let mut sorted: Vec<f64> = rows.iter().map(|r| r.1 + r.2).collect();
    sorted.sort_by(f64::total_cmp);
    let med = sorted[sorted.len() / 2];
    let p99 = sorted[sorted.len() * 99 / 100];
    println!("per-tick total: median {med:.1} ms  p99 {p99:.1} ms  max {:.1} ms", sorted.last().unwrap());
    println!("spikes (> 2× median), with tick mod 15/16/75/225 to expose the period:");
    for (t, step_ms, gen_ms, ph) in &rows {
        if step_ms + gen_ms > 2.0 * med {
            let per: Vec<String> =
                (0..STEP_PHASES).map(|k| format!("{}={:.1}", PHASE_NAMES[k], ph[k])).collect();
            println!(
                "  tick {t:4}  total {:6.1} ms (step {:6.1} + gen {:5.1})  mod15={:2} mod16={:2} mod75={:2} mod225={:3}  {}",
                step_ms + gen_ms, step_ms, gen_ms, t % 15, t % 16, t % 75, t % 225,
                per.join(" ")
            );
        }
    }
}
