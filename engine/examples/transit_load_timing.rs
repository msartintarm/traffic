//! Times the pieces of the SF load path (run with --release).
fn main() {
    let map_path = std::env::args().nth(1).expect("map json path");
    let transit_path = std::env::args().nth(2).expect("transit json path");
    let text = std::fs::read_to_string(&map_path).unwrap();
    let ttext = std::fs::read_to_string(&transit_path).unwrap();
    let t0 = std::time::Instant::now();
    let map = engine::sim::map::OsmMap::from_json(&text).unwrap();
    println!("parse+simplify map: {:?}", t0.elapsed());
    let t = std::time::Instant::now();
    let mut net = map.build();
    println!("network build: {:?}", t.elapsed());
    let t = std::time::Instant::now();
    net.rail = engine::sim::rail::RailNetwork::from_map_json(&text);
    println!("rail parse: {:?} ({} lines)", t.elapsed(), net.rail.lines.len());
    let t = std::time::Instant::now();
    net.attach_bus_stops(&engine::sim::map::bus_stops_from_json(&text));
    println!("attach_bus_stops: {:?} ({} stops)", t.elapsed(), net.bus_stops.len());
    let t = std::time::Instant::now();
    let routes = engine::sim::map::bus_routes_from_json(&text);
    let chains: Vec<_> = routes.iter().filter_map(|(_, pts)| net.resolve_route_chain(pts)).collect();
    println!("osm bus routes: {:?} ({} of {} resolve)", t.elapsed(), chains.len(), routes.len());
    let t = std::time::Instant::now();
    let (rail_specs, bus_specs) = engine::sim::rail::transit_from_json(&ttext).unwrap();
    println!("transit parse: {:?} ({} rail, {} bus)", t.elapsed(), rail_specs.len(), bus_specs.len());
    let t = std::time::Instant::now();
    let (tt, dropped) = engine::sim::rail::build_timetable(&net.rail, &rail_specs);
    println!("build_timetable: {:?} ({} kept, {} dropped)", t.elapsed(), tt.trips.len(), dropped);
    let t = std::time::Instant::now();
    let mut resolved_stops = 0usize;
    let mut cache: std::collections::HashMap<(i64, i64), bool> = std::collections::HashMap::new();
    for spec in &bus_specs {
        for st in &spec.stops {
            let key = ((st.pos[0] * 4.0).round() as i64, (st.pos[1] * 4.0).round() as i64);
            let hit = *cache.entry(key).or_insert_with(|| {
                net.nearest_surface_link(st.pos).is_some_and(|(_, _, d)| d <= 30.0)
            });
            resolved_stops += hit as usize;
        }
    }
    println!(
        "bus stop resolution (memoized): {:?} ({} stops, {} unique)",
        t.elapsed(),
        resolved_stops,
        cache.len()
    );
    let t = std::time::Instant::now();
    // Line-name fallback: resolve one representative trace per unmatched line.
    use std::collections::BTreeMap;
    let mut rep: BTreeMap<&str, Vec<[f64; 2]>> = BTreeMap::new();
    for spec in &bus_specs {
        let e = rep.entry(spec.line.as_str()).or_default();
        if spec.stops.len() > e.len() {
            *e = spec.stops.iter().map(|s| s.pos).collect();
        }
    }
    let mut ok = 0usize;
    for (_, pts) in &rep {
        if net.resolve_route_chain(pts).is_some() {
            ok += 1;
        }
    }
    println!("bus line fallback: {:?} ({} of {} lines resolve)", t.elapsed(), ok, rep.len());
}
