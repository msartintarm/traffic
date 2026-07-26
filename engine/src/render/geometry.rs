//! Static world geometry, built once from the [`Network`]: carriageway ribbons,
//! intersection polygons, lane markings, and turn-arc splines. Pure functions of
//! the network; the wgpu layer uploads the result to one vertex buffer.

use crate::sim::network::{Network, LANE_WIDTH};

use super::{Mesh, Vertex};

pub const ROAD_COLOR: [f32; 3] = [0.16, 0.18, 0.21];
pub const JUNCTION_COLOR: [f32; 3] = [0.19, 0.21, 0.24];
pub const LANE_LINE_COLOR: [f32; 3] = [0.72, 0.72, 0.66];
pub const EDGE_LINE_COLOR: [f32; 3] = [0.85, 0.85, 0.8];

fn unit_perp(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len = dx.hypot(dy).max(1e-9);
    [dy / len, -dx / len]
}

fn corners(a: [f64; 2], b: [f64; 2], half_w: f64) -> [[f32; 2]; 4] {
    let n = unit_perp(a, b);
    let off = [n[0] * half_w, n[1] * half_w];
    [
        [(a[0] - off[0]) as f32, (a[1] - off[1]) as f32],
        [(b[0] - off[0]) as f32, (b[1] - off[1]) as f32],
        [(b[0] + off[0]) as f32, (b[1] + off[1]) as f32],
        [(a[0] + off[0]) as f32, (a[1] + off[1]) as f32],
    ]
}

/// Filled carriageway quads, one per directed link.
pub fn road_mesh(net: &Network) -> Mesh {
    let mut mesh = Mesh::default();
    for s in net.road_strips() {
        let (a, b, w) = ([s[0], s[1]], [s[2], s[3]], s[4]);
        mesh.push_quad(corners(a, b, w / 2.0), ROAD_COLOR);
    }
    mesh
}

/// Filled polygons covering each junction, so overlapping approaches read as one
/// paved intersection rather than crossing ribbons.
pub fn junction_mesh(net: &Network) -> Mesh {
    let mut widest = vec![0.0f64; net.nodes.len()];
    let mut degree = vec![0u32; net.nodes.len()];
    for link in &net.links {
        let w = link.lane_count as f64 * LANE_WIDTH;
        for node in [link.from, link.to] {
            widest[node.idx()] = widest[node.idx()].max(w);
            degree[node.idx()] += 1;
        }
    }
    let mut mesh = Mesh::default();
    for (i, node) in net.nodes.iter().enumerate() {
        if degree[i] < 2 {
            continue;
        }
        push_disc(&mut mesh, node.position, widest[i] * 0.75 + 1.0, JUNCTION_COLOR);
    }
    mesh
}

/// Lane dividers (dashed) plus solid carriageway edge lines.
pub fn marking_mesh(net: &Network) -> Mesh {
    let mut mesh = Mesh::default();
    for d in net.lane_dividers() {
        dashed_line(&mut mesh, [d[0], d[1]], [d[2], d[3]], 3.0, 3.0, 0.15, LANE_LINE_COLOR);
    }
    for s in net.road_strips() {
        let (a, b, half) = ([s[0], s[1]], [s[2], s[3]], s[4] / 2.0);
        let n = unit_perp(a, b);
        for side in [-1.0, 1.0] {
            let off = [n[0] * half * side, n[1] * half * side];
            let a2 = [a[0] + off[0], a[1] + off[1]];
            let b2 = [b[0] + off[0], b[1] + off[1]];
            solid_line(&mut mesh, a2, b2, 0.2, EDGE_LINE_COLOR);
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

/// A smooth turn path from `entry` (arriving at the node) to `exit` (leaving
/// it), bending through the node — the visual path a vehicle follows through an
/// intersection so turns curve instead of snapping.
pub fn turn_arc(entry: [f64; 2], node: [f64; 2], exit: [f64; 2], samples: usize) -> Vec<[f64; 2]> {
    (0..=samples).map(|i| bezier(entry, node, exit, i as f64 / samples as f64)).collect()
}

fn solid_line(mesh: &mut Mesh, a: [f64; 2], b: [f64; 2], width: f64, color: [f32; 3]) {
    mesh.push_quad(corners(a, b, width / 2.0), color);
}

fn dashed_line(mesh: &mut Mesh, a: [f64; 2], b: [f64; 2], dash: f64, gap: f64, width: f64, color: [f32; 3]) {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len = dx.hypot(dy);
    if len < 1e-6 {
        return;
    }
    let (ux, uy) = (dx / len, dy / len);
    let stride = dash + gap;
    let mut s = 0.0;
    while s < len {
        let e = (s + dash).min(len);
        let p0 = [a[0] + ux * s, a[1] + uy * s];
        let p1 = [a[0] + ux * e, a[1] + uy * e];
        solid_line(mesh, p0, p1, width, color);
        s += stride;
    }
}

fn push_disc(mesh: &mut Mesh, center: [f64; 2], radius: f64, color: [f32; 3]) {
    const SIDES: usize = 12;
    let base = mesh.vertices.len() as u32;
    mesh.vertices.push(Vertex::body([center[0] as f32, center[1] as f32], color));
    for k in 0..SIDES {
        let a = std::f64::consts::TAU * k as f64 / SIDES as f64;
        mesh.vertices.push(Vertex::body(
            [(center[0] + radius * a.cos()) as f32, (center[1] + radius * a.sin()) as f32],
            color,
        ));
    }
    for k in 0..SIDES {
        let next = (k + 1) % SIDES;
        mesh.indices.extend([base, base + 1 + k as u32, base + 1 + next as u32]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::map;

    #[test]
    fn road_mesh_has_a_quad_per_link() {
        let net = map::corridor_with_signal();
        let mesh = road_mesh(&net);
        assert_eq!(mesh.vertices.len(), net.links.len() * 4);
        assert_eq!(mesh.indices.len(), net.links.len() * 6);
    }

    #[test]
    fn junction_mesh_covers_multi_link_nodes_only() {
        let net = map::corridor_with_signal();
        let mesh = junction_mesh(&net);
        assert!(!mesh.is_empty(), "the signalized crossing should be paved");
    }

    #[test]
    fn markings_are_produced_for_a_multi_lane_road() {
        let net = map::corridor_with_signal();
        assert!(!marking_mesh(&net).is_empty());
    }

    #[test]
    fn bezier_hits_its_endpoints_and_bends_through_the_control() {
        let a = [0.0, 0.0];
        let ctrl = [10.0, 10.0];
        let b = [20.0, 0.0];
        assert_eq!(bezier(a, ctrl, b, 0.0), a);
        assert_eq!(bezier(a, ctrl, b, 1.0), b);
        let mid = bezier(a, ctrl, b, 0.5);
        assert!(mid[1] > 0.0 && mid[1] < 10.0, "arc bends toward the control point");
    }

    #[test]
    fn turn_arc_starts_and_ends_where_asked() {
        let arc = turn_arc([0.0, -10.0], [0.0, 0.0], [10.0, 0.0], 8);
        assert_eq!(arc.len(), 9);
        assert_eq!(arc[0], [0.0, -10.0]);
        assert_eq!(*arc.last().unwrap(), [10.0, 0.0]);
    }
}
