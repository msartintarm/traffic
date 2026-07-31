//! Static world geometry, built once from the [`Network`]: carriageway ribbons,
//! intersection polygons, lane markings, and turn-arc splines. Emits
//! [`StaticMesh`] (`center + offset` vertices) so the shader can hold every
//! road/line to a minimum on-screen width at any zoom.

use crate::sim::network::{LaneId, LinkId, MovementId, Network, NodeControl, NodeId, TurnType, LANE_WIDTH};

use super::{mass, StaticMesh, StaticVertex};

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
/// offset to the right, so approaches pave right up to their nodes. The distinct
/// [`junction_mesh`] region then ties the arms of each intersection together into
/// one crossing box (and covers the short interior links absorbed by it). Markings
/// trim at the box edge, so the crossing reads as unmarked pavement.
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

/// Two intersection nodes belong to the same junction when the road between them
/// is barely longer than the two junction boxes it spans — i.e. there's almost no
/// open carriageway between them, so they're really one crossing (the case a
/// divided arterial's split signal nodes fall into). Metres of open road below
/// which the boxes are merged.
const MERGE_GAP: f64 = 14.0;

/// Distinct number of road neighbours at each node. A pass-through vertex on a
/// two-way road has 2 (a link each way to each side counts once per neighbour);
/// a real intersection has ≥ 3.
fn node_neighbours(net: &Network) -> Vec<std::collections::BTreeSet<u32>> {
    let mut nb = vec![std::collections::BTreeSet::new(); net.nodes.len()];
    for l in &net.links {
        if l.layer != 0 {
            continue;
        }
        nb[l.from.idx()].insert(l.to.0);
        nb[l.to.idx()].insert(l.from.0);
    }
    nb
}

/// Group nodes into junction *clusters*. Only true intersection nodes (≥ 3
/// neighbours) get a cluster; adjacent ones joined by a near-zero-length link are
/// merged into one (so a divided arterial's split crossing reads as a single
/// intersection). Returns `(node → Some(cluster) | None, cluster_count)`.
fn intersection_clusters(net: &Network) -> (Vec<Option<usize>>, usize) {
    let nb = node_neighbours(net);
    let is_ix = |n: usize| nb[n].len() >= 3;
    let mut parent: Vec<usize> = (0..net.nodes.len()).collect();
    fn find(parent: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while parent[r] != r {
            r = parent[r];
        }
        let mut c = x;
        while parent[c] != r {
            let n = parent[c];
            parent[c] = r;
            c = n;
        }
        r
    }
    // Merge two nodes when the open road between them is negligible and at least
    // one end is a real intersection — this pulls the short, wide approach stubs
    // that flare a junction (turn-lane widenings on degree-2 nodes right at the
    // crossing) into the junction, instead of leaving them as stray wide slabs.
    for i in 0..net.links.len() {
        let l = net.link(LinkId(i as u32));
        if l.layer != 0 {
            continue;
        }
        let (a, b) = (l.from.idx(), l.to.idx());
        if !is_ix(a) && !is_ix(b) {
            continue;
        }
        let full: f64 = net.polylines[i].windows(2).map(|w| norm(sub(w[1], w[0]))).sum();
        let gap = full - net.render_setback.get(a).copied().unwrap_or(0.0) - net.render_setback.get(b).copied().unwrap_or(0.0);
        if gap < MERGE_GAP {
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
            parent[ra] = rb;
        }
    }
    // A component is a junction cluster iff it contains a real intersection node.
    let mut has_ix: std::collections::HashMap<usize, bool> = std::collections::HashMap::new();
    for n in 0..net.nodes.len() {
        let root = find(&mut parent, n);
        *has_ix.entry(root).or_insert(false) |= is_ix(n);
    }
    let mut id = vec![None; net.nodes.len()];
    let mut map: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut next = 0;
    for n in 0..net.nodes.len() {
        let root = find(&mut parent, n);
        if has_ix[&root] {
            let ci = *map.entry(root).or_insert_with(|| {
                let v = next;
                next += 1;
                v
            });
            id[n] = Some(ci);
        }
    }
    (id, next)
}

/// The two edge corners (median-side, then outer-side) where an approach link
/// meets a junction box at node `end_is_to ? l.to : l.from`, taken at the box
/// boundary (the drivable clip).
fn arm_mouth(net: &Network, link: LinkId, end_is_to: bool) -> ([f64; 2], [f64; 2]) {
    let l = net.link(link);
    let dp = net.drivable_polyline(link);
    let k = dp.len();
    let (end, travel) = if end_is_to {
        (dp[k - 1], norm2(sub(dp[k - 1], dp[k - 2]))) // heading into the node
    } else {
        (dp[0], norm2(sub(dp[0], dp[1]))) // heading back toward the node from inside
    };
    let right = [travel[1], -travel[0]];
    let full = l.lane_count as f64 * LANE_WIDTH;
    let outer = [end[0] + right[0] * full, end[1] + right[1] * full];
    (end, outer) // median edge, outer edge
}

/// Whether each link is *interior* to a junction cluster (both ends in the same
/// cluster). Such links lie inside the junction region, so their lane markings
/// (arrows, crosswalks) are suppressed — they'd otherwise paint across the box.
fn interior_links(net: &Network) -> Vec<bool> {
    let (cluster, _) = intersection_clusters(net);
    (0..net.links.len())
        .map(|i| {
            let l = net.link(LinkId(i as u32));
            let (a, b) = (cluster[l.from.idx()], cluster[l.to.idx()]);
            a.is_some() && a == b
        })
        .collect()
}

/// Distinct junction pavement, one region per intersection *cluster* rather than
/// per node. A real intersection box is the area bounded by where each approach
/// stops, so we build it as the intersection of half-planes: for every external
/// arm, slice a generous starting square back to that arm's stop line (across the
/// mouth) and to its outer edge. The result is the convex crossing box — a clean
/// rectangle where two roads meet at right angles, a parallelogram when skewed —
/// aligned to the car stop points. Interior links (both ends in the cluster) fall
/// inside it and are absorbed, so a divided-arterial crossing reads as one box.
pub fn junction_mesh(net: &Network) -> StaticMesh {
    let rings = junction_rings(net);
    let mut mesh = StaticMesh::default();
    for r in &rings.rings {
        if r.ring.len() >= 3 {
            fill_fan(&mut mesh, r.apex, &r.ring, ROAD_COLOR);
        }
    }
    mesh
}

/// One junction cluster's paved region: the rounded boundary `ring` (fanned from
/// `apex`).
struct ClusterRing {
    apex: [f64; 2],
    ring: Vec<[f64; 2]>,
}

/// Every cluster's boundary ring plus the node→cluster map, so both the fill and
/// the marking/signal-head placement work from one shared boundary.
struct JunctionRings {
    cluster: Vec<Option<usize>>,
    rings: Vec<ClusterRing>,
}

fn junction_rings(net: &Network) -> JunctionRings {
    let (cluster, ncl) = intersection_clusters(net);
    let nb = node_neighbours(net);
    let mut arms: Vec<Vec<([f64; 2], [f64; 2])>> = vec![Vec::new(); ncl];
    // Fan apex per cluster: the busiest (highest-degree) node — the actual
    // crossing point — not the average position, which for an asymmetric cluster
    // can sit off to one side and thin the pavement over the real crossing.
    let mut apex = vec![[0.0f64; 2]; ncl];
    let mut apex_deg = vec![0usize; ncl];
    let mut ccount = vec![0usize; ncl];
    for n in 0..net.nodes.len() {
        if let Some(ci) = cluster[n] {
            ccount[ci] += 1;
            if nb[n].len() >= apex_deg[ci] {
                apex_deg[ci] = nb[n].len();
                apex[ci] = net.node(NodeId(n as u32)).position;
            }
        }
    }
    for i in 0..net.links.len() as u32 {
        let l = net.link(LinkId(i));
        if l.layer != 0 {
            continue;
        }
        let (ca, cb) = (cluster[l.from.idx()], cluster[l.to.idx()]);
        if ca.is_some() && ca == cb {
            continue; // interior link — absorbed by the region
        }
        if let Some(ci) = ca {
            let (m, o) = arm_mouth(net, LinkId(i), false);
            arms[ci].push((m, o));
        }
        if let Some(ci) = cb {
            let (m, o) = arm_mouth(net, LinkId(i), true);
            arms[ci].push((m, o));
        }
    }
    let mut rings = Vec::with_capacity(ncl);
    for ci in 0..ncl {
        let c = apex[ci];
        // A lone crossing gets the clean rectilinear box (the intersection of its
        // approach half-planes). A multi-node cluster — a divided arterial or a
        // sprawling interchange whose arms stagger — can't be one convex box
        // without leaving arms or the core unpaved, so it uses the arm-mouth fan.
        let ring = if arms[ci].len() < 2 || apex_deg[ci] == 0 {
            Vec::new()
        } else if ccount[ci] == 1 {
            round_corners(&junction_box(&arms[ci], c), CURB_RADIUS)
        } else {
            round_corners(&junction_fan_ring(&arms[ci], c), CURB_RADIUS)
        };
        rings.push(ClusterRing { apex: c, ring });
    }
    JunctionRings { cluster, rings }
}

/// A stop-bar/crosswalk margin (m) held back from the junction box, matching the
/// sim's `set_junction_setbacks` so markings sit just behind the crossing.
const STOP_MARGIN: f64 = 2.5;

/// The lane-arc position of each link's stop line, snapped to its junction
/// cluster's outer boundary. For a lone crossing this is the node's own stop line
/// (`lane.length`); for a multi-node cluster it's pulled out to where the approach
/// crosses the cluster boundary — so markings and heads land at the real mouth of
/// the junction, not inside it at some interior node's box edge.
fn stop_positions(net: &Network, rings: &JunctionRings) -> Vec<f64> {
    (0..net.links.len())
        .map(|i| {
            let link = LinkId(i as u32);
            let lane0 = net.link(link).lane_start;
            let default = net.lane(lane0).length;
            let Some(ci) = rings.cluster[net.link(link).to.idx()] else { return default };
            let ring = &rings.rings[ci].ring;
            match boundary_crossing(net, link, ring) {
                // `default` (the lane length) can be sub-metre on tiny links, so the
                // lower bound must be 0, never a fixed 1.0 (which would be > default).
                Some(s) => (s - net.lane(lane0).start_offset - STOP_MARGIN).clamp(0.0, default),
                None => default,
            }
        })
        .collect()
}

/// Arc-length along `link`'s centreline (from its upstream end) where it leaves
/// the ring on its way out from the downstream node — the cluster boundary the
/// approach crosses. `None` if the node isn't inside the ring.
fn boundary_crossing(net: &Network, link: LinkId, ring: &[[f64; 2]]) -> Option<f64> {
    if ring.len() < 3 {
        return None;
    }
    let poly = &net.polylines[link.idx()];
    let full: f64 = poly.windows(2).map(|w| norm(sub(w[1], w[0]))).sum();
    const STEP: f64 = 0.5;
    let mut s = full;
    while s > 0.0 {
        if !point_in_ring(ring, point_on_polyline(poly, s)) {
            return Some((s + STEP).min(full));
        }
        s -= STEP;
    }
    None
}

fn point_on_polyline(poly: &[[f64; 2]], s: f64) -> [f64; 2] {
    let mut acc = 0.0;
    for w in poly.windows(2) {
        let seg = norm(sub(w[1], w[0]));
        if acc + seg >= s {
            let t = (s - acc) / seg.max(1e-9);
            return [w[0][0] + (w[1][0] - w[0][0]) * t, w[0][1] + (w[1][1] - w[0][1]) * t];
        }
        acc += seg;
    }
    poly[poly.len() - 1]
}

/// Even-odd ray-cast point-in-polygon.
fn point_in_ring(ring: &[[f64; 2]], p: [f64; 2]) -> bool {
    let (n, mut inside) = (ring.len(), false);
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (ring[i], ring[j]);
        if (a[1] > p[1]) != (b[1] > p[1]) {
            let x = a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]);
            if p[0] < x {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// The rectilinear box of a lone crossing: a generous square clipped by each
/// arm's stop line (across its mouth) and outer edge, so the surviving convex
/// polygon is bounded by the lines where cars stop — a clean rectangle for a
/// right-angle crossing, a parallelogram when skewed.
fn junction_box(arms: &[([f64; 2], [f64; 2])], c: [f64; 2]) -> Vec<[f64; 2]> {
    let r = arms.iter().map(|&(m, o)| norm(sub(m, c)).max(norm(sub(o, c)))).fold(0.0, f64::max) + 20.0;
    let mut poly = vec![[c[0] - r, c[1] - r], [c[0] + r, c[1] - r], [c[0] + r, c[1] + r], [c[0] - r, c[1] + r]];
    for &(m, o) in arms {
        let mc = [(m[0] + o[0]) * 0.5, (m[1] + o[1]) * 0.5];
        poly = clip_halfplane(&poly, mc, norm2(sub(c, mc))); // stop line (across the mouth)
        let along = norm2(sub(o, m)); // median → outer, i.e. across the arm
        let sign = if (c[0] - o[0]) * along[0] + (c[1] - o[1]) * along[1] >= 0.0 { 1.0 } else { -1.0 };
        poly = clip_halfplane(&poly, o, [along[0] * sign, along[1] * sign]); // outer edge (along the arm)
    }
    poly
}

/// The crossing region of a multi-node cluster: each external arm's two mouth
/// corners, ordered around the centre, with a curb return between neighbours that
/// are close and a dip back through the centre where they're far apart (so open
/// spans between arms aren't paved as one giant wedge). Reaches every arm, which a
/// single convex box can't when the arms stagger.
fn junction_fan_ring(arms: &[([f64; 2], [f64; 2])], c: [f64; 2]) -> Vec<[f64; 2]> {
    let ang = |p: &[f64; 2]| (p[1] - c[1]).atan2(p[0] - c[0]);
    let mut ordered: Vec<([f64; 2], [f64; 2])> =
        arms.iter().map(|&(m, o)| if ang(&m) <= ang(&o) { (m, o) } else { (o, m) }).collect();
    ordered.sort_by(|a, b| {
        let ca = ang(&[(a.0[0] + a.1[0]) * 0.5, (a.0[1] + a.1[1]) * 0.5]);
        let cb = ang(&[(b.0[0] + b.1[0]) * 0.5, (b.0[1] + b.1[1]) * 0.5]);
        ca.total_cmp(&cb)
    });
    let n = ordered.len();
    let mut ring: Vec<[f64; 2]> = Vec::with_capacity(n * 3);
    for k in 0..n {
        let (m, o) = ordered[k];
        ring.push(m);
        ring.push(o);
        if norm(sub(o, ordered[(k + 1) % n].0)) > CURB_MAX {
            ring.push(c);
        }
    }
    ring
}

/// Clip a convex polygon to the half-plane `{p : (p − a)·n ≥ 0}`
/// (Sutherland–Hodgman); intersecting several of these builds the junction box.
fn clip_halfplane(poly: &[[f64; 2]], a: [f64; 2], n: [f64; 2]) -> Vec<[f64; 2]> {
    let side = |p: [f64; 2]| (p[0] - a[0]) * n[0] + (p[1] - a[1]) * n[1];
    let k = poly.len();
    let mut out = Vec::with_capacity(k + 2);
    for i in 0..k {
        let (p, q) = (poly[i], poly[(i + 1) % k]);
        let (dp, dq) = (side(p), side(q));
        if dp >= 0.0 {
            out.push(p);
        }
        if (dp >= 0.0) != (dq >= 0.0) {
            let t = dp / (dp - dq);
            out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
        }
    }
    out
}

/// Fill a star-shaped ring (as seen from `center`) as a triangle fan, so a
/// concave crossing region fills without the self-overlap a convex fan would hit.
fn fill_fan(mesh: &mut StaticMesh, center: [f64; 2], ring: &[[f64; 2]], color: [f32; 3]) {
    if ring.len() < 3 {
        return;
    }
    let base = mesh.vertices.len() as u32;
    mesh.vertices.push(StaticVertex { center: [center[0] as f32, center[1] as f32], offset: [0.0, 0.0], color, light: 0.0 });
    for p in ring {
        mesh.vertices.push(StaticVertex { center: [p[0] as f32, p[1] as f32], offset: [0.0, 0.0], color, light: 0.0 });
    }
    let k = ring.len() as u32;
    for j in 0..k {
        mesh.indices.extend([base, base + 1 + j, base + 1 + (j + 1) % k]);
    }
}

/// Curb-return radius (m): California curb returns run ~5–10 m; 5 keeps corners
/// crisp without eating into small junctions.
const CURB_RADIUS: f64 = 5.0;

/// Max gap (m) between neighbouring arms of a multi-node cluster that still reads
/// as one curb-return corner; beyond it the fan boundary dips back through the
/// centre instead of paving the open span between the arms.
const CURB_MAX: f64 = 26.0;

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
    // Arrows/crosswalks/stop-lines go at each approach's stop line, snapped to the
    // junction-cluster boundary (`stop`) so on a multi-node crossing they sit at
    // its real mouth, not inside it. Interior links are inside the box and skipped.
    let interior = interior_links(net);
    let stop = stop_positions(net, &junction_rings(net));
    lane_use_arrows(net, &interior, &stop, &mut mesh);
    crosswalks(net, &interior, &stop, &mut mesh);
    stop_yield_markings(net, &interior, &stop, &mut mesh);
    mesh
}

pub const ARROW_COLOR: [f32; 3] = [0.88, 0.88, 0.82];
pub const CROSSWALK_COLOR: [f32; 3] = [0.82, 0.82, 0.78];
pub const SIGN_RED: [f32; 3] = [0.72, 0.11, 0.11]; // stop/yield sign face

/// Map one OSM `turn:lanes` token to a turn direction; unknown / `none` tokens
/// (which paint no arrow) return `None`.
fn parse_turn_token(t: &str) -> Option<TurnType> {
    match t.trim() {
        "left" | "sharp_left" | "slight_left" | "merge_to_left" | "reverse" => Some(TurnType::Left),
        "through" => Some(TurnType::Through),
        "right" | "sharp_right" | "slight_right" | "merge_to_right" => Some(TurnType::Right),
        _ => None,
    }
}

/// The per-lane turn sets a link's OSM `turn:lanes` prescribes (median lane
/// first, matching `index_in_link`), or `None` when it's absent or its lane count
/// doesn't match the link — in which case the renderer derives turns from the
/// lane's movements instead.
fn turn_lanes_spec(net: &Network, link: LinkId) -> Option<Vec<Vec<TurnType>>> {
    let s = net.link_turn_lanes.get(link.idx())?;
    if s.is_empty() {
        return None;
    }
    let per: Vec<Vec<TurnType>> = s.split('|').map(|lane| lane.split(';').filter_map(parse_turn_token).collect()).collect();
    (per.len() == net.link(link).lane_count as usize).then_some(per)
}

/// California lane-use pavement arrows: on each signalized-approach lane, paint an
/// arrow for every turn it allows. Turns come from OSM `turn:lanes` when the road
/// carries that tag, else from the lane's actual movements — so the striping
/// reads the way it does on a real approach.
fn lane_use_arrows(net: &Network, interior: &[bool], stop: &[f64], mesh: &mut StaticMesh) {
    for lane_id in 0..net.lanes.len() as u32 {
        let lane = LaneId(lane_id);
        let link = net.lane(lane).link;
        if interior[link.idx()] {
            continue;
        }
        let node = net.link(link).to;
        if !matches!(net.node(node).control, NodeControl::Signalized(_)) {
            continue;
        }
        let mut turns: Vec<TurnType> = match turn_lanes_spec(net, link) {
            Some(spec) => spec[net.lane(lane).index_in_link as usize].clone(),
            None => {
                let start = net.lane(lane).movement_start.0;
                (0..net.lane(lane).movement_count).map(|k| net.movement_turn(MovementId(start + k))).collect()
            }
        };
        turns.dedup();
        if turns.is_empty() {
            continue;
        }
        let s = stop[link.idx()];
        let p = net.lane_point(lane, (s - 8.0).max(s * 0.5)); // ~8 m back from the stop line
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

/// California-style crossing markings on every approach to a signalized junction:
/// a solid transverse **limit line** (stop bar) where cars halt, then a
/// continental **crosswalk** (bars parallel to travel) in front of it, filling
/// the stop-line-to-box margin. (Isolated here so other regions can slot in.)
fn crosswalks(net: &Network, interior: &[bool], stop: &[f64], mesh: &mut StaticMesh) {
    const BAR_GAP: f64 = 0.6; // limit line → crosswalk gap (m)
    const DEPTH: f64 = 2.0; // crosswalk depth (m)
    for n in 0..net.nodes.len() as u32 {
        if !matches!(net.node(NodeId(n)).control, NodeControl::Signalized(_)) {
            continue;
        }
        for link in 0..net.links.len() as u32 {
            if net.link(LinkId(link)).to != NodeId(n) || interior[link as usize] {
                continue;
            }
            for lane in net.lanes_of(LinkId(link)) {
                let p = net.lane_point(lane, stop[link as usize]); // stop-line point
                let (d, perp) = ([p[2].cos(), p[2].sin()], [p[2].sin(), -p[2].cos()]);
                // Limit line: one lane-wide bar across the stop point.
                let e = [perp[0] * LANE_WIDTH * 0.5, perp[1] * LANE_WIDTH * 0.5];
                mesh.push_ribbon([p[0] + e[0], p[1] + e[1]], [p[0] - e[0], p[1] - e[1]], 0.2, CROSSWALK_COLOR, 0.0);
                // Crosswalk: bars parallel to travel, just past the limit line.
                let base = [p[0] + d[0] * BAR_GAP, p[1] + d[1] * BAR_GAP];
                for k in [-1.0, 1.0] {
                    let off = k * LANE_WIDTH * 0.28;
                    let a = [base[0] + perp[0] * off, base[1] + perp[1] * off];
                    let b = [a[0] + d[0] * DEPTH, a[1] + d[1] * DEPTH];
                    mesh.push_ribbon(a, b, 0.28, CROSSWALK_COLOR, 0.0);
                }
            }
        }
    }
}

/// Stop- and yield-controlled approaches, rendered so each reads at a glance: a
/// stop approach gets a solid transverse limit line plus a roadside red octagon;
/// a yield approach gets a "shark-tooth" yield line (triangles pointing back at
/// the driver) plus a roadside inverted red triangle. Interior links are inside
/// the box, so they're skipped.
fn stop_yield_markings(net: &Network, interior: &[bool], stop: &[f64], mesh: &mut StaticMesh) {
    for n in 0..net.nodes.len() as u32 {
        let control = net.node(NodeId(n)).control;
        let is_stop = matches!(control, NodeControl::Stop);
        let is_yield = matches!(control, NodeControl::Yield);
        if !is_stop && !is_yield {
            continue;
        }
        for link in 0..net.links.len() as u32 {
            if net.link(LinkId(link)).to != NodeId(n) || interior[link as usize] {
                continue;
            }
            let spos = stop[link as usize];
            for lane in net.lanes_of(LinkId(link)) {
                let p = net.lane_point(lane, spos); // stop point
                let (d, perp) = ([p[2].cos(), p[2].sin()], [p[2].sin(), -p[2].cos()]);
                if is_stop {
                    let e = [perp[0] * LANE_WIDTH * 0.5, perp[1] * LANE_WIDTH * 0.5];
                    mesh.push_ribbon([p[0] + e[0], p[1] + e[1]], [p[0] - e[0], p[1] - e[1]], 0.25, CROSSWALK_COLOR, 0.0);
                } else {
                    yield_teeth(mesh, [p[0], p[1]], d, perp);
                }
            }
            let (sp, fwd) = approach_curb(net, LinkId(link), spos);
            if is_stop {
                sign_octagon(mesh, sp, 1.6);
            } else {
                sign_triangle(mesh, sp, fwd, 1.9);
            }
        }
    }
}

/// A "shark's teeth" yield line across one lane at `p`: triangles with their base
/// on the stop line and apex pointing upstream (−`d`), at the yielding driver.
fn yield_teeth(mesh: &mut StaticMesh, p: [f64; 2], d: [f64; 2], perp: [f64; 2]) {
    const N: usize = 4;
    let w = LANE_WIDTH / N as f64;
    for i in 0..N {
        let off = (i as f64 + 0.5 - N as f64 / 2.0) * w;
        let c = [p[0] + perp[0] * off, p[1] + perp[1] * off];
        let a = [c[0] + perp[0] * w * 0.4, c[1] + perp[1] * w * 0.4];
        let b = [c[0] - perp[0] * w * 0.4, c[1] - perp[1] * w * 0.4];
        let apex = [c[0] - d[0] * 0.8, c[1] - d[1] * 0.8];
        mesh.push_polygon(&[a, b, apex], CROSSWALK_COLOR);
    }
}

/// A roadside point just past the right edge of an approach at stop position
/// `spos`, plus the approach heading — where a stop/yield sign (or signal pole)
/// stands.
fn approach_curb(net: &Network, link: LinkId, spos: f64) -> ([f64; 2], [f64; 2]) {
    let outer = net.lanes_of(link).last().unwrap_or(net.link(link).lane_start);
    let p = net.lane_point(outer, spos);
    let dir = net.arrival_dir(link);
    let right = [dir[1], -dir[0]];
    let off = LANE_WIDTH * 0.5 + 1.8;
    ([p[0] + right[0] * off, p[1] + right[1] * off], dir)
}

/// A red stop-sign octagon of radius `r` centred at `c`.
fn sign_octagon(mesh: &mut StaticMesh, c: [f64; 2], r: f64) {
    let pts: Vec<[f64; 2]> = (0..8)
        .map(|k| {
            let a = std::f64::consts::TAU * (k as f64 + 0.5) / 8.0;
            [c[0] + r * a.cos(), c[1] + r * a.sin()]
        })
        .collect();
    mesh.push_polygon(&pts, SIGN_RED);
}

/// A red yield-sign triangle (apex pointing upstream, toward the driver) of size
/// `r` at `c`, aligned to the approach heading `fwd`.
fn sign_triangle(mesh: &mut StaticMesh, c: [f64; 2], fwd: [f64; 2], r: f64) {
    let left = [-fwd[1], fwd[0]];
    let p1 = [c[0] + left[0] * r, c[1] + left[1] * r];
    let p2 = [c[0] - left[0] * r, c[1] - left[1] * r];
    let apex = [c[0] - fwd[0] * r * 1.2, c[1] - fwd[1] * r * 1.2];
    mesh.push_polygon(&[p1, apex, p2], SIGN_RED);
}

/// Curbside signal-head placement: for each signal group, the world position and
/// heading of its head. Like a real corner pole, the head sits *beside* the
/// carriageway — just past the right-hand edge of the approach it controls — at
/// the stop line, so it stands alongside the front stopped car instead of on the
/// pavement. Returns `(group_index, [x, y], heading)`; pure, state-independent
/// geometry the bridge pairs with each group's live colour. Groups sharing one
/// approach step outward by their local ordinal so their heads don't overlap.
pub fn signal_head_placements(net: &Network) -> Vec<(usize, [f32; 2], f32)> {
    const POLE_MARGIN: f64 = 1.6; // road edge → pole (m)
    const HEAD_SPACING: f64 = 2.4; // lateral step between heads on one approach (m)
    const HALF_HEAD: f64 = 2.8; // half the housing length, so its green end lands on the stop line
    let stop = stop_positions(net, &junction_rings(net));
    let mut rep = vec![None; net.groups.len()];
    for mv in &net.movements {
        if let Some(g) = mv.signal_group {
            rep[g.idx()].get_or_insert(mv.from_lane);
        }
    }
    let mut out = Vec::new();
    let mut per_approach: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for (gi, lane) in rep.into_iter().enumerate() {
        let Some(lane) = lane else { continue };
        let link = net.lane(lane).link;
        let ord = *per_approach.entry(link.0).and_modify(|c| *c += 1).or_insert(0) as f64;
        // Anchor to the outermost (right-edge) lane so every group on this
        // approach lands at the same curb, stepped out by its ordinal, at the
        // cluster-snapped stop line.
        let outer = net.lanes_of(link).last().unwrap_or(lane);
        let p = net.lane_point(outer, (stop[link.idx()] - HALF_HEAD).max(0.0));
        let dir = net.arrival_dir(link);
        let right = [dir[1], -dir[0]]; // right-hand normal of travel
        let off = LANE_WIDTH * 0.5 + POLE_MARGIN + ord * HEAD_SPACING;
        out.push((gi, [(p[0] + right[0] * off) as f32, (p[1] + right[1] * off) as f32], dir[1].atan2(dir[0]) as f32));
    }
    out
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
                LinkSpec { from_osm: 1, to_osm: 2, lanes: 1, speed_limit: 20.0, geometry: Vec::new(), layer: 0, name: String::new(), road_class: String::new(), highway_ref: String::new(), turn_lanes: String::new() },
                LinkSpec { from_osm: 3, to_osm: 4, lanes: 1, speed_limit: 25.0, geometry: Vec::new(), layer: 1, name: String::new(), road_class: String::new(), highway_ref: String::new(), turn_lanes: String::new() },
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
    fn a_lone_signal_is_one_cluster_that_absorbs_nothing() {
        // A plain four-way is a single-node cluster with no interior links — the
        // whole junction is the crossing box the four arms plug into.
        let net = map::arterial_intersection();
        let (_, ncl) = intersection_clusters(&net);
        assert_eq!(ncl, 1, "one junction");
        assert!(interior_links(&net).iter().all(|&x| !x), "nothing to absorb");
        assert!(!junction_mesh(&net).is_empty(), "the box is paved");
    }

    #[cfg(feature = "import")]
    #[test]
    fn a_split_arterial_crossing_collapses_to_one_cluster() {
        // Millbrae junction 0 is a divided arterial whose crossing splits across
        // several OSM nodes joined by short wide stubs. They must read as ONE
        // intersection: a single cluster spanning them, with the interior stubs
        // marked interior (so their arrows/crosswalks don't paint across the box)
        // and the whole crossing paved as one region.
        let net = map::millbrae_junction(0);
        let (_, ncl) = intersection_clusters(&net);
        assert_eq!(ncl, 1, "the split crossing is one junction");
        assert!(interior_links(&net).iter().any(|&x| x), "the stubs between the split nodes are interior");
        assert!(!junction_mesh(&net).is_empty(), "the cluster is paved as one region");
    }

    #[test]
    fn signal_heads_sit_beside_the_carriageway_not_on_it() {
        // Every head must lie laterally beyond the approach's carriageway — off the
        // pavement, at the curb — which is the whole point of the curbside pole.
        let net = map::arterial_intersection();
        let placements = signal_head_placements(&net);
        assert!(!placements.is_empty(), "a signalized junction has heads");
        let mut rep = vec![None; net.groups.len()];
        for mv in &net.movements {
            if let Some(g) = mv.signal_group {
                rep[g.idx()].get_or_insert(mv.from_lane);
            }
        }
        for (gi, pos, _h) in placements {
            let link = net.lane(rep[gi].unwrap()).link;
            let dir = net.arrival_dir(link);
            let node = net.node(net.link(link).to).position; // on the shared centreline
            let rel = [pos[0] as f64 - node[0], pos[1] as f64 - node[1]];
            let lateral = rel[0] * dir[1] - rel[1] * dir[0]; // signed right-of-centreline distance
            let carriageway = net.link(link).lane_count as f64 * LANE_WIDTH;
            assert!(lateral >= carriageway, "group {gi}: head only {lateral:.1}m right of centre — inside the {carriageway:.1}m carriageway");
        }
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
    fn turn_lanes_tag_drives_the_arrows() {
        assert_eq!(parse_turn_token("left"), Some(TurnType::Left));
        assert_eq!(parse_turn_token("through"), Some(TurnType::Through));
        assert_eq!(parse_turn_token("slight_right"), Some(TurnType::Right));
        assert_eq!(parse_turn_token("none"), None);
        // A link tagged `turn:lanes=left|through;right` yields, median-first, a
        // left-only lane then a through+right lane — used verbatim for the arrows.
        let net = OsmMap {
            nodes: vec![NodeSpec::uncontrolled(1, -100.0, 0.0), NodeSpec::uncontrolled(2, 0.0, 0.0)],
            links: vec![LinkSpec { turn_lanes: "left|through;right".into(), ..LinkSpec::oneway(1, 2, 2, 15.0) }],
        }
        .build();
        assert_eq!(
            turn_lanes_spec(&net, LinkId(0)),
            Some(vec![vec![TurnType::Left], vec![TurnType::Through, TurnType::Right]])
        );
        // A tag whose lane count disagrees with the link is ignored (fall back).
        let bad = OsmMap {
            nodes: vec![NodeSpec::uncontrolled(1, -100.0, 0.0), NodeSpec::uncontrolled(2, 0.0, 0.0)],
            links: vec![LinkSpec { turn_lanes: "left".into(), ..LinkSpec::oneway(1, 2, 2, 15.0) }],
        }
        .build();
        assert_eq!(turn_lanes_spec(&bad, LinkId(0)), None);
    }

    #[test]
    fn stop_and_yield_junctions_are_marked_distinctly() {
        // A stop or yield node paints a roadside red sign; an uncontrolled node
        // paints none — so the three control kinds read differently on the map.
        let cross = |center: NodeSpec| {
            let mut links = Vec::new();
            links.extend(LinkSpec::twoway(1, 0, 2, 15.0));
            links.extend(LinkSpec::twoway(0, 2, 2, 15.0));
            links.extend(LinkSpec::twoway(3, 0, 1, 12.0));
            links.extend(LinkSpec::twoway(0, 4, 1, 12.0));
            OsmMap {
                nodes: vec![
                    center,
                    NodeSpec::uncontrolled(1, -150.0, 0.0),
                    NodeSpec::uncontrolled(2, 150.0, 0.0),
                    NodeSpec::uncontrolled(3, 0.0, -150.0),
                    NodeSpec::uncontrolled(4, 0.0, 150.0),
                ],
                links,
            }
            .build()
        };
        let signs = |net: &Network| marking_mesh(net).vertices.iter().filter(|v| v.color == SIGN_RED).count();
        assert!(signs(&cross(NodeSpec::stop(0, 0.0, 0.0))) > 0, "a stop junction gets stop signs");
        assert!(signs(&cross(NodeSpec::give_way(0, 0.0, 0.0))) > 0, "a yield junction gets yield signs");
        assert_eq!(signs(&cross(NodeSpec::uncontrolled(0, 0.0, 0.0))), 0, "an uncontrolled junction gets no signs");
    }

    #[test]
    fn congestion_mesh_handles_curved_links_with_valid_indices() {
        // Regression: `road_strips` is per-segment, so density must iterate links,
        // not strips. A curved (multi-segment) link must produce in-range indices.
        let net = OsmMap {
            nodes: vec![NodeSpec::uncontrolled(1, 0.0, 0.0), NodeSpec::uncontrolled(2, 200.0, 100.0)],
            links: vec![LinkSpec { from_osm: 1, to_osm: 2, lanes: 2, speed_limit: 20.0, geometry: vec![[100.0, 0.0], [150.0, 50.0]], layer: 0, name: String::new(), road_class: String::new(), highway_ref: String::new(), turn_lanes: String::new() }],
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
