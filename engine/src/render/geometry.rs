//! Static world geometry, built once from the [`Network`]: carriageway ribbons,
//! intersection polygons, lane markings, and turn-arc splines. Emits
//! [`StaticMesh`] (`center + offset` vertices) so the shader can hold every
//! road/line to a minimum on-screen width at any zoom.

use crate::sim::network::{LaneId, LinkId, MovementId, Network, NodeControl, NodeId, TurnType, LANE_WIDTH};

use super::{mass, StaticMesh};

pub const ROAD_COLOR: [f32; 3] = [0.16, 0.18, 0.21];
// The junction is the same asphalt as the carriageways, so it reads as one
// continuous surface (no lighter disc) when zoomed in.
// Kept dimmer than the vehicle body colour so cars read clearly against them.
pub const LANE_LINE_COLOR: [f32; 3] = [0.50, 0.50, 0.46]; // same-direction dashed dividers
pub const EDGE_LINE_COLOR: [f32; 3] = [0.55, 0.55, 0.50]; // outer road edge
pub const CENTER_LINE_COLOR: [f32; 3] = [0.55, 0.46, 0.13]; // dimmed yellow, luminance like the white lines

/// The complete static world surface — at-grade carriageways, junction fills,
/// then overpasses on top — as one mesh. The single source of truth every
/// render backend draws (the browser GPU feed and the ASCII rasteriser both use
/// this), so what the tests rasterise is exactly what the browser shows.
pub fn world_mesh(net: &Network) -> StaticMesh {
    let mut mesh = road_mesh(net);
    mesh.extend(&junction_mesh(net));
    mesh.extend(&overpass_mesh(net));
    mesh
}

/// Filled carriageway ribbons for at-grade and tunnel links (`layer <= 0`),
/// drawn low to high so a tunnel sits under the surface.
pub fn road_mesh(net: &Network) -> StaticMesh {
    road_ribbons(net, i32::MIN, 0)
}

/// Filled carriageway ribbons for overpasses (`layer >= 1`), drawn last so a
/// bridge renders on top of the road it crosses.
pub fn overpass_mesh(net: &Network) -> StaticMesh {
    road_ribbons(net, 1, i32::MAX)
}

/// Shift a segment to the right of its travel direction by `d` — a directed
/// link's carriageway sits on the right of the shared centreline, where its
/// lanes are (lane `k` centre is `(k+0.5)·LANE_WIDTH` to the right).
fn offset_right(a: [f64; 2], b: [f64; 2], d: f64) -> ([f64; 2], [f64; 2]) {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len = dx.hypot(dy).max(1e-9);
    let (nx, ny) = (dy / len, -dx / len);
    ([a[0] + nx * d, a[1] + ny * d], [b[0] + nx * d, b[1] + ny * d])
}

/// Filled carriageways drawn along each link's *full* centreline (node to node),
/// offset to the right. Because every carriageway runs to the node centre, the
/// approaches overlap there and pave the intersection continuously — no separate
/// junction fill, no seams or dark blocks. Markings (which trim at the box edge)
/// stop before the intersection, so the box reads as unmarked pavement.
fn road_ribbons(net: &Network, layer_min: i32, layer_max: i32) -> StaticMesh {
    let mut idx: Vec<usize> = (0..net.links.len())
        .filter(|&i| (layer_min..=layer_max).contains(&net.links[i].layer))
        .collect();
    idx.sort_by_key(|&i| net.links[i].layer);
    let mut mesh = StaticMesh::default();
    for i in idx {
        let half = net.links[i].lane_count as f64 * LANE_WIDTH / 2.0;
        for seg in net.polylines[i].windows(2) {
            let (a, b) = offset_right(seg[0], seg[1], half);
            mesh.push_ribbon(a, b, half, ROAD_COLOR, 0.0);
        }
    }
    mesh
}

/// Filled junction pavement. A real intersection is a *solid* paved surface
/// spanning where the approaches meet, so for every node we fan-fill the region
/// bounded by its incident carriageways' edge corners as seen from the node —
/// including the node itself as a corner — giving gap-free pavement that reaches
/// each approach mouth. Done per node (not a global hull) so it stays bounded to
/// that junction's own arms. `road_mesh` already paves the approaches; this fills
/// the box and the re-entrant corners between them.
pub fn junction_mesh(net: &Network) -> StaticMesh {
    let mut mesh = StaticMesh::default();
    for n in 0..net.nodes.len() as u32 {
        let node = NodeId(n);
        let node_pos = net.node(node).position;
        let mut corners: Vec<[f64; 2]> = vec![node_pos];
        for link in 0..net.links.len() as u32 {
            let l = net.link(LinkId(link));
            if l.layer != 0 {
                continue; // grade-separated crossings aren't one junction
            }
            let poly = &net.polylines[link as usize];
            let (seg_a, seg_b) = if l.to == node {
                (poly[poly.len() - 2], poly[poly.len() - 1])
            } else if l.from == node {
                (poly[0], poly[1])
            } else {
                continue;
            };
            let half = l.lane_count as f64 * LANE_WIDTH / 2.0;
            let (a, b) = offset_right(seg_a, seg_b, half);
            let center = if l.to == node { b } else { a };
            let (dx, dy) = (seg_b[0] - seg_a[0], seg_b[1] - seg_a[1]);
            let len = dx.hypot(dy).max(1e-9);
            let perp = [dy / len * half, -dx / len * half];
            corners.push([center[0] + perp[0], center[1] + perp[1]]);
            corners.push([center[0] - perp[0], center[1] - perp[1]]);
        }
        let hull = convex_hull(corners);
        if hull.len() >= 3 {
            mesh.push_polygon(&round_corners(&hull, CURB_RADIUS), ROAD_COLOR);
        }
    }
    mesh
}

/// Curb-return radius (m): California curb returns run ~5–10 m; 5 keeps corners
/// crisp without eating into small junctions.
const CURB_RADIUS: f64 = 5.0;

/// Round each convex-polygon corner with a quadratic fillet, so the junction
/// pavement reads with curb returns instead of sharp points. The fillet radius is
/// clamped to a fraction of the adjacent edges so short edges don't over-round.
fn round_corners(poly: &[[f64; 2]], radius: f64) -> Vec<[f64; 2]> {
    let n = poly.len();
    if n < 3 {
        return poly.to_vec();
    }
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        let (p, v, q) = (poly[(i + n - 1) % n], poly[i], poly[(i + 1) % n]);
        let (in_len, out_len) = (norm(sub(v, p)), norm(sub(q, v)));
        let r = radius.min(in_len * 0.4).min(out_len * 0.4);
        let din = norm2(sub(v, p));
        let dout = norm2(sub(q, v));
        let t1 = [v[0] - din[0] * r, v[1] - din[1] * r];
        let t2 = [v[0] + dout[0] * r, v[1] + dout[1] * r];
        for s in 0..=3 {
            out.push(bezier(t1, v, t2, s as f64 / 3.0));
        }
    }
    out
}

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn norm(v: [f64; 2]) -> f64 {
    v[0].hypot(v[1])
}

fn norm2(v: [f64; 2]) -> [f64; 2] {
    let n = v[0].hypot(v[1]).max(1e-9);
    [v[0] / n, v[1] / n]
}

/// Andrew's monotone chain convex hull (counter-clockwise); few points per node.
fn convex_hull(mut pts: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    pts.sort_by(|p, q| p[0].total_cmp(&q[0]).then(p[1].total_cmp(&q[1])));
    pts.dedup_by(|p, q| (p[0] - q[0]).abs() < 1e-9 && (p[1] - q[1]).abs() < 1e-9);
    if pts.len() < 3 {
        return pts;
    }
    let cross = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    let mut lower: Vec<[f64; 2]> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<[f64; 2]> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
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
        // Inner edge (side -1) borders opposing traffic → yellow centre line;
        // outer edge (side +1) is the road edge.
        for (side, color) in [(-1.0, CENTER_LINE_COLOR), (1.0, EDGE_LINE_COLOR)] {
            let off = [n[0] * half * side, n[1] * half * side];
            mesh.push_ribbon([a[0] + off[0], a[1] + off[1]], [b[0] + off[0], b[1] + off[1]], 0.1, color, 0.0);
        }
    }
    // A left-turn arrow painted on any lane that carries a protected (signalized)
    // left, near its stop line — the on-road cue a real turn lane shows.
    lane_use_arrows(net, &mut mesh);
    crosswalks(net, &mut mesh);
    mesh
}

pub const ARROW_COLOR: [f32; 3] = [0.88, 0.88, 0.82];
pub const CROSSWALK_COLOR: [f32; 3] = [0.82, 0.82, 0.78];

/// California lane-use pavement arrows: on each signalized-approach lane, paint an
/// arrow for every turn its movements allow (left / through / right), so the
/// permitted turns per lane read the way they do on a real approach.
fn lane_use_arrows(net: &Network, mesh: &mut StaticMesh) {
    for lane_id in 0..net.lanes.len() as u32 {
        let lane = LaneId(lane_id);
        let node = net.link(net.lane(lane).link).to;
        if !matches!(net.node(node).control, NodeControl::Signalized(_)) {
            continue;
        }
        let start = net.lane(lane).movement_start.0;
        let mut turns: Vec<TurnType> = (0..net.lane(lane).movement_count)
            .map(|k| net.movement_turn(MovementId(start + k)))
            .collect();
        turns.dedup();
        if turns.is_empty() {
            continue;
        }
        let len = net.lane(lane).length;
        let p = net.lane_point(lane, (len - 8.0).max(len * 0.5));
        for turn in turns {
            lane_arrow(mesh, [p[0], p[1]], p[2], turn);
        }
    }
}

/// Paint a lane-use arrow for `turn`, centred at `p`, aligned to lane heading `h`.
fn lane_arrow(mesh: &mut StaticMesh, p: [f64; 2], h: f64, turn: TurnType) {
    let fwd = [h.cos(), h.sin()];
    let side = match turn {
        TurnType::Left => 1.0,
        TurnType::Right => -1.0,
        TurnType::Through => 0.0,
    };
    let left = [-fwd[1] * side, fwd[0] * side];
    let at = |fx: f64, fy: f64| [p[0] + fwd[0] * fx + left[0] * fy, p[1] + fwd[1] * fx + left[1] * fy];
    let bar = |mesh: &mut StaticMesh, a: [f64; 2], b: [f64; 2]| mesh.push_ribbon(a, b, 0.18, ARROW_COLOR, 0.0);
    if turn == TurnType::Through {
        bar(mesh, at(-2.5, 0.0), at(1.6, 0.0)); // shaft
        bar(mesh, at(1.6, 0.0), at(0.9, 0.55)); // arrowhead
        bar(mesh, at(1.6, 0.0), at(0.9, -0.55));
    } else {
        bar(mesh, at(-2.5, 0.0), at(1.0, 0.0)); // shaft
        bar(mesh, at(1.0, 0.0), at(1.0, 1.5)); // bend toward the turn
        bar(mesh, at(1.0, 1.5), at(0.45, 1.0)); // arrowhead barbs
        bar(mesh, at(1.0, 1.5), at(1.55, 1.0));
    }
}

/// California-style continental crosswalks across every approach to a signalized
/// junction: bars parallel to travel, filling the stop-line-to-box margin. (The
/// convention is isolated here so other regions' markings can slot in later.)
fn crosswalks(net: &Network, mesh: &mut StaticMesh) {
    const DEPTH: f64 = 2.4; // crosswalk depth (m), ~ the stop-bar/box margin
    for n in 0..net.nodes.len() as u32 {
        if !matches!(net.node(NodeId(n)).control, NodeControl::Signalized(_)) {
            continue;
        }
        for link in 0..net.links.len() as u32 {
            if net.link(LinkId(link)).to != NodeId(n) {
                continue;
            }
            for lane in net.lanes_of(LinkId(link)) {
                let p = net.lane_point(lane, net.lane(lane).length); // stop-line point
                let (d, perp) = ([p[2].cos(), p[2].sin()], [p[2].sin(), -p[2].cos()]);
                for k in [-1.0, 1.0] {
                    let off = k * LANE_WIDTH * 0.28;
                    let a = [p[0] + perp[0] * off, p[1] + perp[1] * off];
                    let b = [a[0] + d[0] * DEPTH, a[1] + d[1] * DEPTH];
                    mesh.push_ribbon(a, b, 0.28, CROSSWALK_COLOR, 0.0);
                }
            }
        }
    }
}

/// Translucent congestion overlay: shade every polyline segment of a link busy
/// enough to matter (`counts[link]` = vehicles on it). `light = 3` triggers the
/// shader's translucent branch. Iterates *links* (a curved link is several
/// segments), so it's independent of the strip count.
/// Bright overlay colour for the user-selected link.
pub const HIGHLIGHT_COLOR: [f32; 3] = [0.25, 0.85, 1.0];

pub fn congestion_mesh(net: &Network, counts: &[u32], selected: Option<usize>) -> StaticMesh {
    let mut mesh = StaticMesh::default();
    for i in 0..net.links.len() {
        let is_selected = selected == Some(i);
        let link = net.link(LinkId(i as u32));
        let lane = net.lane(link.lane_start);
        let ratio = mass::occupancy_ratio(counts[i] as f64, (lane.length / 7.0 * link.lane_count as f64).max(1.0));
        if !is_selected && ratio < 0.2 {
            continue;
        }
        let color = if is_selected { HIGHLIGHT_COLOR } else { let c = mass::congestion_color(ratio); [c[0], c[1], c[2]] };
        let half = link.lane_count as f64 * LANE_WIDTH / 2.0;
        for seg in net.drivable_polyline(LinkId(i as u32)).windows(2) {
            let (a, b) = offset_right(seg[0], seg[1], half);
            mesh.push_ribbon(a, b, half, color, 3.0);
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
    fn carriageways_run_through_the_intersection() {
        // Fill uses full centrelines (node to node), so a link's carriageway
        // reaches its downstream node — the overlap that paves intersections.
        let net = map::corridor_with_signal();
        let node = net.node(net.link(LinkId(1)).from).position; // node 2 (signalized centre)
        let touches_node = road_mesh(&net)
            .vertices
            .iter()
            .any(|v| (v.center[0] as f64 - node[0]).hypot(v.center[1] as f64 - node[1]) < LANE_WIDTH);
        assert!(touches_node, "carriageway fill reaches the intersection node");
    }

    #[test]
    fn bridges_draw_in_the_overpass_mesh_not_the_road_mesh() {
        // A layer-1 bridge over a surface road: the surface goes in road_mesh, the
        // bridge in overpass_mesh (drawn later, on top).
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -100.0, 0.0),
                NodeSpec::uncontrolled(2, 100.0, 0.0),
                NodeSpec::uncontrolled(3, 0.0, -100.0),
                NodeSpec::uncontrolled(4, 0.0, 100.0),
            ],
            links: vec![
                LinkSpec { from_osm: 1, to_osm: 2, lanes: 1, speed_limit: 20.0, geometry: Vec::new(), layer: 0, name: String::new(), road_class: String::new(), highway_ref: String::new() },
                LinkSpec { from_osm: 3, to_osm: 4, lanes: 1, speed_limit: 25.0, geometry: Vec::new(), layer: 1, name: String::new(), road_class: String::new(), highway_ref: String::new() },
            ],
        }
        .build();
        assert_eq!(road_mesh(&net).vertices.len(), 4, "only the surface link is at grade");
        assert_eq!(overpass_mesh(&net).vertices.len(), 4, "the bridge draws on top");
    }

    #[test]
    fn junction_mesh_fills_the_box_covering_the_node() {
        // A four-way's junction polygon should be a real filled area that contains
        // the node centre, so the crossing is paved rather than a gap between arms.
        let net = map::arterial_intersection();
        let mesh = junction_mesh(&net);
        assert!(!mesh.is_empty(), "a junction is filled");
        assert!(mesh.vertices.iter().all(|v| v.offset == [0.0, 0.0]), "fill vertices are a real area, not min-width");
        // The hull around the centre node spans both axes (it's a box, not a sliver).
        let cxs: Vec<f32> = mesh.vertices.iter().map(|v| v.center[0]).collect();
        let cys: Vec<f32> = mesh.vertices.iter().map(|v| v.center[1]).collect();
        let span = |v: &[f32]| v.iter().cloned().fold(f32::MIN, f32::max) - v.iter().cloned().fold(f32::MAX, f32::min);
        assert!(span(&cxs) > LANE_WIDTH as f32 && span(&cys) > LANE_WIDTH as f32, "the box has width on both axes");
    }

    #[test]
    fn markings_are_produced_for_a_multi_lane_road() {
        assert!(!marking_mesh(&map::corridor_with_signal()).is_empty());
    }

    #[test]
    fn signalized_approaches_get_crosswalks() {
        let signal = marking_mesh(&map::arterial_intersection()).vertices.iter().filter(|v| v.color == CROSSWALK_COLOR).count();
        assert!(signal > 0, "a signalized junction paints crosswalks");
        // The unsignalized bridge map has no signal, so no crosswalks.
        let none = marking_mesh(&OsmMap {
            nodes: vec![NodeSpec::uncontrolled(1, -100.0, 0.0), NodeSpec::uncontrolled(2, 100.0, 0.0)],
            links: vec![LinkSpec::oneway(1, 2, 1, 20.0)],
        }
        .build())
        .vertices
        .iter()
        .filter(|v| v.color == CROSSWALK_COLOR)
        .count();
        assert_eq!(none, 0, "an uncontrolled road has no crosswalks");
    }

    #[test]
    fn protected_left_lanes_get_a_turn_arrow() {
        // The arterial has protected lefts; their approach lanes should carry a
        // left-turn arrow, so the marking mesh has arrow-coloured vertices that a
        // plain crossroad without protected lefts does not.
        let arrows = |net: &Network| {
            marking_mesh(net).vertices.iter().filter(|v| v.color == ARROW_COLOR).count()
        };
        assert!(arrows(&map::arterial_intersection()) > 0, "protected-left lanes are marked");
    }

    #[test]
    fn congestion_mesh_handles_curved_links_with_valid_indices() {
        // Regression: `road_strips` is per-segment, so density must iterate links,
        // not strips. A curved (multi-segment) link must produce in-range indices.
        let net = OsmMap {
            nodes: vec![NodeSpec::uncontrolled(1, 0.0, 0.0), NodeSpec::uncontrolled(2, 200.0, 100.0)],
            links: vec![LinkSpec { from_osm: 1, to_osm: 2, lanes: 2, speed_limit: 20.0, geometry: vec![[100.0, 0.0], [150.0, 50.0]], layer: 0, name: String::new(), road_class: String::new(), highway_ref: String::new() }],
        }
        .build();
        let mesh = congestion_mesh(&net, &[999], None); // link 0 heavily congested
        assert!(!mesh.is_empty(), "a congested curved link should shade");
        assert!(mesh.indices.iter().all(|&i| (i as usize) < mesh.vertices.len()), "indices in range");
        // an empty count → nothing shaded, unless selected
        assert!(congestion_mesh(&net, &[0], None).is_empty());
        assert!(!congestion_mesh(&net, &[0], Some(0)).is_empty(), "a selected link is highlighted even when empty");
    }

    #[test]
    fn bezier_hits_endpoints_and_bends_through_control() {
        let (a, ctrl, b) = ([0.0, 0.0], [10.0, 10.0], [20.0, 0.0]);
        assert_eq!(bezier(a, ctrl, b, 0.0), a);
        assert_eq!(bezier(a, ctrl, b, 1.0), b);
        assert!(bezier(a, ctrl, b, 0.5)[1] > 0.0);
    }
}
