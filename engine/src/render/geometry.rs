//! Static world geometry, built once from the [`Network`]: carriageway ribbons,
//! intersection polygons, lane markings, and turn-arc splines. Emits
//! [`StaticMesh`] (`center + offset` vertices) so the shader can hold every
//! road/line to a minimum on-screen width at any zoom.

use crate::sim::network::{LinkId, Network, LANE_WIDTH};

use super::{mass, StaticMesh};

pub const ROAD_COLOR: [f32; 3] = [0.16, 0.18, 0.21];
pub const JUNCTION_COLOR: [f32; 3] = [0.19, 0.21, 0.24];
pub const LANE_LINE_COLOR: [f32; 3] = [0.72, 0.72, 0.66];
pub const EDGE_LINE_COLOR: [f32; 3] = [0.85, 0.85, 0.8];

/// Filled carriageway ribbons, one per directed link.
pub fn road_mesh(net: &Network) -> StaticMesh {
    let mut mesh = StaticMesh::default();
    for s in net.road_strips() {
        mesh.push_ribbon([s[0], s[1]], [s[2], s[3]], s[4] / 2.0, ROAD_COLOR, 0.0);
    }
    mesh
}

/// Filled polygons covering each junction.
pub fn junction_mesh(net: &Network) -> StaticMesh {
    let mut widest = vec![0.0f64; net.nodes.len()];
    let mut degree = vec![0u32; net.nodes.len()];
    for link in &net.links {
        let w = link.lane_count as f64 * LANE_WIDTH;
        for node in [link.from, link.to] {
            widest[node.idx()] = widest[node.idx()].max(w);
            degree[node.idx()] += 1;
        }
    }
    let mut mesh = StaticMesh::default();
    for (i, node) in net.nodes.iter().enumerate() {
        if degree[i] >= 2 {
            mesh.push_disc(node.position, widest[i] * 0.6 + 0.5, JUNCTION_COLOR);
        }
    }
    mesh
}

/// Lane dividers (dashed) plus solid carriageway edge lines. Drawn only when
/// zoomed in (the renderer skips this mesh past a zoom threshold).
pub fn marking_mesh(net: &Network) -> StaticMesh {
    let mut mesh = StaticMesh::default();
    for d in net.lane_dividers() {
        dashed_line(&mut mesh, [d[0], d[1]], [d[2], d[3]], 3.0, 3.0, 0.15, LANE_LINE_COLOR);
    }
    for s in net.road_strips() {
        let (a, b, half) = ([s[0], s[1]], [s[2], s[3]], s[4] / 2.0);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len = dx.hypot(dy).max(1e-9);
        let n = [dy / len, -dx / len];
        for side in [-1.0, 1.0] {
            let off = [n[0] * half * side, n[1] * half * side];
            mesh.push_ribbon([a[0] + off[0], a[1] + off[1]], [b[0] + off[0], b[1] + off[1]], 0.1, EDGE_LINE_COLOR, 0.0);
        }
    }
    mesh
}

/// Translucent congestion overlay: shade every polyline segment of a link busy
/// enough to matter (`counts[link]` = vehicles on it). `light = 3` triggers the
/// shader's translucent branch. Iterates *links* (a curved link is several
/// segments), so it's independent of the strip count.
pub fn congestion_mesh(net: &Network, counts: &[u32]) -> StaticMesh {
    let mut mesh = StaticMesh::default();
    for i in 0..net.links.len() {
        let link = net.link(LinkId(i as u32));
        let lane = net.lane(link.lane_start);
        let jam = (lane.length / 7.0 * link.lane_count as f64).max(1.0);
        let ratio = mass::occupancy_ratio(counts[i] as f64, jam);
        if ratio < 0.2 {
            continue;
        }
        let c = mass::congestion_color(ratio);
        let half = link.lane_count as f64 * LANE_WIDTH / 2.0;
        for seg in net.polylines[i].windows(2) {
            mesh.push_ribbon(seg[0], seg[1], half, [c[0], c[1], c[2]], 3.0);
        }
    }
    mesh
}

/// Quadratic Bézier point; `t` in `[0,1]`.
pub fn bezier(a: [f64; 2], ctrl: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    let u = 1.0 - t;
    [
        u * u * a[0] + 2.0 * u * t * ctrl[0] + t * t * b[0],
        u * u * a[1] + 2.0 * u * t * ctrl[1] + t * t * b[1],
    ]
}

/// A smooth turn path from `entry` through `node` to `exit`.
pub fn turn_arc(entry: [f64; 2], node: [f64; 2], exit: [f64; 2], samples: usize) -> Vec<[f64; 2]> {
    (0..=samples).map(|i| bezier(entry, node, exit, i as f64 / samples as f64)).collect()
}

fn dashed_line(mesh: &mut StaticMesh, a: [f64; 2], b: [f64; 2], dash: f64, gap: f64, width: f64, color: [f32; 3]) {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len = dx.hypot(dy);
    if len < 1e-6 {
        return;
    }
    let (ux, uy) = (dx / len, dy / len);
    let mut s = 0.0;
    while s < len {
        let e = (s + dash).min(len);
        mesh.push_ribbon([a[0] + ux * s, a[1] + uy * s], [a[0] + ux * e, a[1] + uy * e], width / 2.0, color, 0.0);
        s += dash + gap;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::map::{self, LinkSpec, NodeSpec, OsmMap};

    #[test]
    fn road_mesh_has_a_quad_per_link() {
        let net = map::corridor_with_signal();
        let mesh = road_mesh(&net);
        assert_eq!(mesh.vertices.len(), net.links.len() * 4);
        assert_eq!(mesh.indices.len(), net.links.len() * 6);
    }

    #[test]
    fn ribbon_center_plus_offset_reconstructs_the_quad() {
        let mut mesh = StaticMesh::default();
        mesh.push_ribbon([0.0, 0.0], [10.0, 0.0], 2.0, ROAD_COLOR, 0.0);
        // segment is +x, so offsets are ±y of magnitude 2.
        for v in &mesh.vertices {
            assert!((v.offset[1].abs() - 2.0).abs() < 1e-6 && v.offset[0].abs() < 1e-6);
        }
    }

    #[test]
    fn junction_mesh_covers_multi_link_nodes_only() {
        assert!(!junction_mesh(&map::corridor_with_signal()).is_empty());
    }

    #[test]
    fn markings_are_produced_for_a_multi_lane_road() {
        assert!(!marking_mesh(&map::corridor_with_signal()).is_empty());
    }

    #[test]
    fn congestion_mesh_handles_curved_links_with_valid_indices() {
        // Regression: `road_strips` is per-segment, so density must iterate links,
        // not strips. A curved (multi-segment) link must produce in-range indices.
        let net = OsmMap {
            nodes: vec![NodeSpec::uncontrolled(1, 0.0, 0.0), NodeSpec::uncontrolled(2, 200.0, 100.0)],
            links: vec![LinkSpec { from_osm: 1, to_osm: 2, lanes: 2, speed_limit: 20.0, geometry: vec![[100.0, 0.0], [150.0, 50.0]] }],
        }
        .build();
        let mesh = congestion_mesh(&net, &[999]); // link 0 heavily congested
        assert!(!mesh.is_empty(), "a congested curved link should shade");
        assert!(mesh.indices.iter().all(|&i| (i as usize) < mesh.vertices.len()), "indices in range");
        // an empty count → nothing shaded
        assert!(congestion_mesh(&net, &[0]).is_empty());
    }

    #[test]
    fn bezier_hits_endpoints_and_bends_through_control() {
        let (a, ctrl, b) = ([0.0, 0.0], [10.0, 10.0], [20.0, 0.0]);
        assert_eq!(bezier(a, ctrl, b, 0.0), a);
        assert_eq!(bezier(a, ctrl, b, 1.0), b);
        assert!(bezier(a, ctrl, b, 0.5)[1] > 0.0);
    }
}
