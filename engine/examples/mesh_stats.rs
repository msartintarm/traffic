//! Prints static-render payload sizes for a map (assessing browser memory).
//! cargo run --release --example mesh_stats --features import -- ../web/public/columbus.json
use engine::render::geometry;
use engine::sim::map::OsmMap;

fn main() {
    // Comma-separated paths merge into one network (the combined-scenario path).
    let path = std::env::args().nth(1).expect("map path");
    let parts: Vec<_> = path
        .split(',')
        .map(|p| {
            let raw = std::fs::read_to_string(p).expect("read map");
            let origin = engine::sim::map::json_origin(&raw).expect("meta.origin");
            (OsmMap::from_json(&raw).expect("parse map"), origin)
        })
        .collect();
    let net = engine::sim::map::ImportedMap::merge(parts).build();
    let t = std::time::Instant::now();
    let g = geometry::world_geometry(&net);
    let vsize = std::mem::size_of::<engine::render::StaticVertex>();
    let vb = |m: &engine::render::StaticMesh| m.vertices.len() * vsize + m.indices.len() * 4;
    println!("bake: {:?}", t.elapsed());
    println!(
        "world: {} verts, {} idx ({} MB)   marking: {} verts, {} idx ({} MB)   dir: {} u32",
        g.world.vertices.len(),
        g.world.indices.len(),
        vb(&g.world) / 1_000_000,
        g.marking.vertices.len(),
        g.marking.indices.len(),
        vb(&g.marking) / 1_000_000,
        g.directory.len()
    );
}
