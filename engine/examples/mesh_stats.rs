//! Prints static-render payload sizes for a map (assessing browser memory).
//! cargo run --release --example mesh_stats --features import -- ../web/public/columbus.json
use engine::render::geometry;
use engine::sim::map::OsmMap;

fn main() {
    let path = std::env::args().nth(1).expect("map path");
    let raw = std::fs::read_to_string(&path).expect("read map");
    let map = OsmMap::from_json(&raw).expect("parse map");
    let net = map.build();
    let t = std::time::Instant::now();
    let g = geometry::world_geometry(&net);
    let vb = |m: &engine::render::StaticMesh| m.vertices.len() * 32 + m.indices.len() * 4;
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
