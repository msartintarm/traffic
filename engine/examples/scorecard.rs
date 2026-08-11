//! Headless validation scorecard: run a scraped map through a simulated peak and
//! score the sim's *output* against independent data — windowed link flows vs
//! Caltrans AADT targets (GEH), corridor travel times vs free flow, and the
//! crash rate per 100M VMT. The CI tier of the validation harness.
//!
//! Count targets come from AADT *embedded in the map itself*
//! (`tools/counts/attach_counts.py --write-map`), which the network builder
//! carries through its topology transforms — raw-JSON link indices do not
//! survive the build, so an external `counts.json` cannot be joined by index.
//! Each directed link with an observed AADT is scored against `AADT·K·D`
//! (peak direction) or `AADT·K·(1−D)` (off-peak), whichever fits better.
//!
//! ```text
//! cargo run --release --example scorecard --features import -- \
//!   --map map-with-aadt.json --lodes map.lodes.json \
//!   --compression 12 --start-hour 6.5 --end-hour 9.5 [--assert]
//! ```

use std::collections::HashMap;

use engine::sim::config::SimConfig;
use engine::sim::demand::{self, DemandGenerator, DemandSources};
use engine::sim::measure::{self, Measurement};
use engine::sim::network::LinkId;
use engine::sim::{boundary, NetWorld, OsmMap};

struct Args {
    map: String,
    lodes: Option<String>,
    compression: f64,
    start_hour: f64,
    end_hour: f64,
    warmup_day_mins: f64,
    k_factor: f64,
    d_factor: f64,
    seed: u64,
    assert_gate: bool,
    dump_ref: Option<String>,
}

fn parse_args() -> Args {
    let mut args = Args {
        map: String::new(),
        lodes: None,
        compression: 1.0,
        start_hour: 6.5,
        end_hour: 9.5,
        warmup_day_mins: 30.0,
        k_factor: 0.09,
        d_factor: 0.55,
        seed: 0xC0FFEE,
        assert_gate: false,
        dump_ref: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| panic!("missing value for {a}"));
        match a.as_str() {
            "--map" => args.map = val(),
            "--lodes" => args.lodes = Some(val()),
            "--compression" => args.compression = val().parse().expect("compression"),
            "--start-hour" => args.start_hour = val().parse().expect("start-hour"),
            "--end-hour" => args.end_hour = val().parse().expect("end-hour"),
            "--warmup-day-mins" => args.warmup_day_mins = val().parse().expect("warmup"),
            "--k" => args.k_factor = val().parse().expect("k"),
            "--d" => args.d_factor = val().parse().expect("d"),
            "--seed" => args.seed = val().parse().expect("seed"),
            "--assert" => args.assert_gate = true,
            "--dump-ref" => args.dump_ref = Some(val()),
            other => panic!("unknown arg {other}"),
        }
    }
    if args.map.is_empty() {
        panic!("--map <path> is required");
    }
    args
}

/// Auto corridor probes: each highway gateway paired with its farthest-reachable
/// highway exit — the "drive the freeway through the box" ETAs.
fn auto_probes(world: &NetWorld) -> Vec<(LinkId, LinkId)> {
    let entries = boundary::highway_entry_links(&world.network);
    let exits = boundary::highway_exit_links(&world.network);
    let mut probes = Vec::new();
    for &e in entries.iter().take(6) {
        let mut best: Option<(f64, LinkId)> = None;
        for &x in &exits {
            if x == e {
                continue;
            }
            if let Some(t) = measure::free_flow_travel_secs(world, e, x) {
                if best.is_none_or(|(bt, _)| t > bt) {
                    best = Some((t, x));
                }
            }
        }
        if let Some((_, x)) = best {
            probes.push((e, x));
        }
    }
    probes
}

fn main() {
    let args = parse_args();
    let raw = std::fs::read_to_string(&args.map).expect("read map");
    let map = OsmMap::from_json(&raw).expect("parse map");
    let net = map.build();
    let mut world = NetWorld::new(net, SimConfig { seed: args.seed, ..SimConfig::default_config() });

    let commute = args.lodes.as_ref().map(|p| {
        let raw = std::fs::read_to_string(p).expect("read lodes");
        demand::CommuteOd::from_json(&raw).expect("parse lodes")
    });
    let sources = DemandSources::with_rush_hour(true, true, true);
    let pairs = demand::od_pairs_with_commute(&world.network, args.seed, 48, sources, commute.as_ref());
    let mut gen = DemandGenerator::new(&world, &pairs, args.seed);
    gen.set_rush_hour(&world.network, true);
    gen.set_day_compression(args.compression);
    gen.resume_clock(args.start_hour * 3600.0 - args.warmup_day_mins * 60.0, 0);
    world.install_router(&gen.destinations());

    let dt = SimConfig::default_config().dt;
    let day = |gen: &DemandGenerator| gen.rush_hour_day_secs();
    while day(&gen) < args.start_hour * 3600.0 {
        gen.step(&mut world, dt);
        world.step();
    }
    let mut meas = Measurement::begin(&world);
    while day(&gen) < args.end_hour * 3600.0 {
        gen.step(&mut world, dt);
        world.step();
        meas.sample(&world, dt);
    }

    let flows = meas.link_flows(&world);
    let speeds = meas.link_speeds();
    let mut geh_rows = Vec::new();
    let (mut obs_total, mut obs_pass) = (0usize, 0usize);
    for i in 0..flows.len() {
        let link = LinkId(i as u32);
        let aadt = world.network.link_aadt(link);
        if aadt <= 0.0 {
            continue;
        }
        // Two-way AADT splits D / (1−D) across the carriageways; judge each
        // directed link against whichever split it fits better. Freeway targets
        // take their peak-hour share from the PeMS diurnal shape itself (max
        // hour / day total of the directional curve, ≈0.058) — the arterial
        // K·D grossly overstates a congested freeway's hourly share.
        let (peak, off) = if boundary::is_highway_link(&world.network, link) {
            let shape = engine::sim::rush_hour::FALLBACK;
            let k_dir = shape.iter().map(|&v| v as f64).fold(0.0, f64::max)
                / shape.iter().map(|&v| v as f64).sum::<f64>();
            let t = aadt * 0.5 * k_dir;
            (t, t)
        } else {
            // K·D describes the arterial's daily *peak* hour (PM for urban
            // arterials); a window elsewhere in the day is scored against the
            // diurnal shape's ratio to that peak, or an AM run would demand
            // PM volumes.
            let shape = |t: f64| engine::sim::rush_hour::arterial_factor(t);
            let window = {
                let (a, b) = (args.start_hour * 3600.0, args.end_hour * 3600.0);
                let n = 8;
                (0..n).map(|k| shape(a + (b - a) * k as f64 / (n - 1) as f64)).sum::<f64>() / n as f64
            };
            let day_peak = (0..96).map(|k| shape(k as f64 * 900.0)).fold(0.0f64, f64::max);
            let tod = (window / day_peak.max(1e-9)).min(1.0);
            (
                aadt * args.k_factor * args.d_factor * tod,
                aadt * args.k_factor * (1.0 - args.d_factor) * tod,
            )
        };
        let g = measure::geh(flows[i], peak).min(measure::geh(flows[i], off));
        obs_total += 1;
        if g < 5.0 {
            obs_pass += 1;
        }
        let road = if world.network.link_ref(link).is_empty() {
            world.network.link_names[i].as_str()
        } else {
            world.network.link_ref(link)
        };
        geh_rows.push(serde_json::json!({
            "link": i,
            "road": road,
            "sim_vph": flows[i],
            "target_vph": [off, peak],
            "geh": g,
            "speed_mps": if speeds[i].is_nan() { serde_json::Value::Null } else { speeds[i].into() },
        }));
    }
    let geh_share = if obs_total > 0 { obs_pass as f64 / obs_total as f64 } else { f64::NAN };

    let corridors: Vec<serde_json::Value> = auto_probes(&world)
        .into_iter()
        .filter_map(|(from, to)| {
            let live = measure::corridor_travel_secs(&world, from, to)?;
            let free = measure::free_flow_travel_secs(&world, from, to)?;
            Some(serde_json::json!({
                "from": from.0, "to": to.0,
                "ref": world.network.link_ref(from),
                "live_secs": live, "free_secs": free,
                "ratio": live / free.max(1e-9),
            }))
        })
        .collect();

    // Windowed flow histogram by road ref, a quick per-corridor volume readout.
    let mut by_ref: HashMap<&str, f64> = HashMap::new();
    for i in 0..flows.len() {
        let r = world.network.link_ref(LinkId(i as u32));
        if !r.is_empty() {
            *by_ref.entry(r).or_default() += flows[i];
        }
    }

    // Gateway audit: is the demanded inflow actually being admitted?
    let gateways: Vec<serde_json::Value> = boundary::highway_entry_links(&world.network)
        .iter()
        .map(|&e| {
            serde_json::json!({
                "link": e.0,
                "ref": world.network.link_ref(e),
                "lanes": world.network.link(e).lane_count,
                "aadt": world.network.link_aadt(e),
                "sim_vph": flows[e.idx()],
            })
        })
        .collect();

    // Optional corridor trace: every link whose ref matches, with its flow — the
    // where-does-the-volume-go debugging view.
    let ref_dump: Vec<serde_json::Value> = args
        .dump_ref
        .as_deref()
        .map(|want| {
            (0..flows.len())
                .filter(|&i| world.network.link_ref(LinkId(i as u32)).contains(want))
                .map(|i| {
                    let link = world.network.link(LinkId(i as u32));
                    serde_json::json!({
                        "link": i,
                        "kind": format!("{:?}", link.kind),
                        "lanes": link.lane_count,
                        "from": link.from.0,
                        "to": link.to.0,
                        "aadt": world.network.link_aadt(LinkId(i as u32)),
                        "sim_vph": flows[i],
                        "speed_mps": if speeds[i].is_nan() { serde_json::Value::Null } else { speeds[i].into() },
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let report = serde_json::json!({
        "meta": {
            "map": args.map,
            "compression": args.compression,
            "window_hours": [args.start_hour, args.end_hour],
            "seed": args.seed,
            "links": flows.len(),
            "spawned": gen.spawned(),
            "exited": world.exited(),
            "queued_at_gateways": gen.queued(),
            "dropped_at_gateways": gen.dropped(),
        },
        "totals": {
            "vkt": meas.vkt(),
            "vmt": meas.vmt(),
            "crashes": meas.crashes(&world),
            "crashes_per_100m_vmt": if meas.vmt() > 0.0 { meas.crashes_per_100m_vmt(&world).into() } else { serde_json::Value::Null },
        },
        "geh": {
            "observed_links": obs_total,
            "observed_pass": obs_pass,
            "share_under_5": if geh_share.is_nan() { serde_json::Value::Null } else { geh_share.into() },
            "links": geh_rows,
        },
        "corridors": corridors,
        "flow_by_ref": by_ref,
        "gateways": gateways,
        "ref_dump": ref_dump,
    });
    println!("{}", serde_json::to_string_pretty(&report).expect("serialize"));

    if args.assert_gate && obs_total > 0 && geh_share < 0.85 {
        eprintln!("GEH gate failed: {obs_pass}/{obs_total} observed links under 5 ({geh_share:.2} < 0.85)");
        std::process::exit(1);
    }
}
