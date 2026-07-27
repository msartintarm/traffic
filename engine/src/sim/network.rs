//! Runtime road network: an index-addressed graph built from OSM import
//! ([`super::map`]) and consumed by [`super::net_world`].
//!
//! Everything is a flat `Vec` addressed by a small integer newtype rather than
//! pointers or nested ownership — the structure a GPU storage buffer wants, and
//! what lets the network scale to a whole city without per-object allocation.
//! A directed [`Link`] is one carriageway between two nodes holding one or more
//! parallel [`Lane`]s; a [`Movement`] is a permitted lane-to-lane transition
//! across a node, optionally gated by a [`SignalGroup`].

use super::signal::{SignalProgram, SignalState};

macro_rules! index_type {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub u32);
        impl $name {
            pub fn idx(self) -> usize {
                self.0 as usize
            }
        }
    };
}

index_type!(NodeId);
index_type!(LinkId);
index_type!(LaneId);
index_type!(MovementId);
index_type!(SignalGroupId);
index_type!(ProgramId);

pub const LANE_WIDTH: f64 = 3.5;

/// Turn-pocket bay geometry: the bay is fully open (turn lane at its own offset)
/// for [`POCKET_OPEN`] metres before the stop line, having diverged from the
/// adjacent through lane over the bay taper (a per-lane length) upstream of that.
pub const POCKET_OPEN: f64 = 12.0;
/// Default bay-taper length assigned to a qualifying turn lane.
pub const POCKET_TAPER: f64 = 18.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeControl {
    Uncontrolled,
    Stop,
    Yield,
    Signalized(ProgramId),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Node {
    pub position: [f64; 2],
    pub control: NodeControl,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Link {
    pub from: NodeId,
    pub to: NodeId,
    pub lane_start: LaneId,
    pub lane_count: u32,
    /// Grade-separation level for render z-order (see `LinkSpec::layer`).
    pub layer: i32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lane {
    pub link: LinkId,
    pub index_in_link: u32,
    /// Drivable length, i.e. the link centreline between the two junction
    /// boundaries — the polyline shortened by each end node's setback.
    pub length: f64,
    /// Arc-length along the link polyline where this lane's usable span begins
    /// (the upstream junction boundary); `position` is measured from here.
    pub start_offset: f64,
    pub speed_limit: f64,
    pub movement_start: MovementId,
    pub movement_count: u32,
    /// Turn-pocket bay-taper length (m), `0.0` for a normal full-width lane. When
    /// set, this is a dedicated turn lane whose lateral offset is its neighbour's
    /// upstream (the bay is "closed", merged into the adjacent through lane) and
    /// diverges over this taper to its own offset by the stop line (the bay
    /// "opens"). Makes a left/right-turn pocket a real feature of the geometry, so
    /// turners queue in the bay instead of the through lane.
    pub pocket_taper: f64,
}

/// A movement's turn direction, from the angle between the arriving and
/// departing road directions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TurnType {
    Through,
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Movement {
    pub from_lane: LaneId,
    pub to_lane: LaneId,
    pub node: NodeId,
    pub signal_group: Option<SignalGroupId>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SignalGroup {
    pub program: ProgramId,
    pub bit: u8,
}

/// The path a vehicle drives *inside* a node while executing a movement: a cubic
/// Bézier from the arrival lane's stop point (`entry`) to the departure lane's
/// start (`exit`), with control handles `c1`/`c2` set along the arrival and
/// departure road directions. That makes the path *leave* aligned with the road
/// it came from and *arrive* aligned with the road it enters — straight when the
/// roads are collinear, a smooth arc for a turn, a smooth shift when staggered —
/// so a vehicle never jerks sideways onto a diagonal. `len` is its arc length.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interior {
    pub entry: [f64; 2],
    pub c1: [f64; 2],
    pub c2: [f64; 2],
    pub exit: [f64; 2],
    pub len: f64,
}

/// Where two movements' interior paths cross at a node, as the arc-length along
/// each path (`sa` on `a`, `sb` on `b`). Precomputed once; at runtime a vehicle
/// near `sa` on `a` and another near `sb` on `b` are occupying the same spot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConflictPoint {
    pub node: NodeId,
    pub a: MovementId,
    pub sa: f64,
    pub b: MovementId,
    pub sb: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Network {
    pub nodes: Vec<Node>,
    pub links: Vec<Link>,
    pub lanes: Vec<Lane>,
    pub movements: Vec<Movement>,
    pub groups: Vec<SignalGroup>,
    pub programs: Vec<SignalProgram>,
    /// Centreline polyline per link (from-node … intermediate bends … to-node);
    /// a straight link is just its two endpoints. Vehicle placement and road
    /// geometry follow this, so real OSM curves render and drive as curves.
    pub polylines: Vec<Vec<[f64; 2]>>,
    /// Interior crossing path per movement (index-aligned with `movements`).
    pub interiors: Vec<Interior>,
    /// Crossing points between conflicting movements, over the whole network.
    pub conflicts: Vec<ConflictPoint>,
    /// Per-node render setback (metres): how far short of the node the drawn
    /// carriageway/markings stop — the junction-box edge. Smaller than a lane's
    /// stop-line setback (which adds a crosswalk margin), so lines run up to the
    /// box instead of breaking off early.
    pub render_setback: Vec<f64>,
    /// OSM road name per link (index-aligned with `links`), for browser labelling.
    pub link_names: Vec<String>,
}

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn norm(v: [f64; 2]) -> f64 {
    v[0].hypot(v[1])
}

fn unit(v: [f64; 2]) -> [f64; 2] {
    let n = norm(v).max(1e-9);
    [v[0] / n, v[1] / n]
}

/// Cubic Bézier point at parameter `t` in `[0,1]`.
fn bezier3(a: [f64; 2], c1: [f64; 2], c2: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    let u = 1.0 - t;
    let (w0, w1, w2, w3) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
    [
        w0 * a[0] + w1 * c1[0] + w2 * c2[0] + w3 * b[0],
        w0 * a[1] + w1 * c1[1] + w2 * c2[1] + w3 * b[1],
    ]
}

const INTERIOR_SAMPLES: usize = 16;

/// Sampled polyline of an interior Bézier and its cumulative arc length.
fn interior_polyline(it: &Interior) -> Vec<([f64; 2], f64)> {
    let mut out = Vec::with_capacity(INTERIOR_SAMPLES + 1);
    let mut acc = 0.0;
    let mut prev = it.entry;
    out.push((prev, 0.0));
    for i in 1..=INTERIOR_SAMPLES {
        let p = bezier3(it.entry, it.c1, it.c2, it.exit, i as f64 / INTERIOR_SAMPLES as f64);
        acc += norm(sub(p, prev));
        out.push((p, acc));
        prev = p;
    }
    out
}

fn point2(p: [f64; 3]) -> [f64; 2] {
    [p[0], p[1]]
}

/// The sub-polyline of `poly` between arc-lengths `s0` and `s1`, with the two
/// endpoints interpolated exactly onto those cuts.
fn clip_polyline(poly: &[[f64; 2]], s0: f64, s1: f64) -> Vec<[f64; 2]> {
    let lerp = |a: [f64; 2], b: [f64; 2], t: f64| [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
    let mut out = Vec::new();
    let mut acc = 0.0;
    for w in poly.windows(2) {
        let seg = norm(sub(w[1], w[0])).max(1e-9);
        let (a, b) = (acc, acc + seg);
        if b >= s0 && a <= s1 {
            if a <= s0 {
                out.push(lerp(w[0], w[1], (s0 - a) / seg));
            } else {
                out.push(w[0]);
            }
            if b >= s1 {
                out.push(lerp(w[0], w[1], (s1 - a) / seg));
            }
        }
        acc = b;
    }
    if out.len() < 2 {
        let (p0, _) = point_along(poly, s0);
        let (p1, _) = point_along(poly, s1);
        return vec![p0, p1];
    }
    out
}

/// Whether two segments cross, as parametric `(t, u)` along each; `None` if
/// parallel or non-overlapping.
fn segment_intersection(p1: [f64; 2], p2: [f64; 2], p3: [f64; 2], p4: [f64; 2]) -> Option<(f64, f64)> {
    let r = sub(p2, p1);
    let s = sub(p4, p3);
    let denom = r[0] * s[1] - r[1] * s[0];
    if denom.abs() < 1e-12 {
        return None;
    }
    let qp = sub(p3, p1);
    let t = (qp[0] * s[1] - qp[1] * s[0]) / denom;
    let u = (qp[0] * r[1] - qp[1] * r[0]) / denom;
    ((0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u)).then_some((t, u))
}

/// The arc-lengths where two interior paths conflict: an exact segment crossing
/// if one exists (their paths truly intersect, whatever the sampling), otherwise
/// the closest approach if it's within `clearance` (a near-miss like two
/// same-direction curves). Segment-exact so long interiors at big intersections
/// can't slip a real crossing between samples.
fn nearest_crossing(a: &[([f64; 2], f64)], b: &[([f64; 2], f64)], clearance: f64) -> Option<(f64, f64)> {
    let lerp = |x: f64, y: f64, k: f64| x + (y - x) * k;
    let mut best = f64::MAX;
    let mut at = (0.0, 0.0);
    for wa in a.windows(2) {
        for wb in b.windows(2) {
            if let Some((t, u)) = segment_intersection(wa[0].0, wa[1].0, wb[0].0, wb[1].0) {
                return Some((lerp(wa[0].1, wa[1].1, t), lerp(wb[0].1, wb[1].1, u)));
            }
            let d = norm(sub(wa[0].0, wb[0].0));
            if d < best {
                best = d;
                at = (wa[0].1, wb[0].1);
            }
        }
    }
    (best < clearance).then_some(at)
}

/// The point and unit direction at arc-length `s` along a polyline (clamped).
fn point_along(poly: &[[f64; 2]], s: f64) -> ([f64; 2], [f64; 2]) {
    if poly.len() < 2 {
        return (poly.first().copied().unwrap_or([0.0, 0.0]), [1.0, 0.0]);
    }
    let mut acc = 0.0;
    for w in poly.windows(2) {
        let seg = norm(sub(w[1], w[0]));
        if s <= acc + seg {
            let t = ((s - acc) / seg.max(1e-9)).clamp(0.0, 1.0);
            let dir = unit(sub(w[1], w[0]));
            return ([w[0][0] + (w[1][0] - w[0][0]) * t, w[0][1] + (w[1][1] - w[0][1]) * t], dir);
        }
        acc += seg;
    }
    let n = poly.len();
    (poly[n - 1], unit(sub(poly[n - 1], poly[n - 2])))
}

impl Network {
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.idx()]
    }

    pub fn link(&self, id: LinkId) -> &Link {
        &self.links[id.idx()]
    }

    pub fn lane(&self, id: LaneId) -> &Lane {
        &self.lanes[id.idx()]
    }

    pub fn movement(&self, id: MovementId) -> &Movement {
        &self.movements[id.idx()]
    }

    pub fn movements_of(&self, lane: LaneId) -> &[Movement] {
        let l = self.lane(lane);
        let start = l.movement_start.idx();
        &self.movements[start..start + l.movement_count as usize]
    }

    pub fn interior(&self, mid: MovementId) -> &Interior {
        &self.interiors[mid.idx()]
    }

    /// Whether two movements have a crossing conflict point.
    pub fn movements_conflict(&self, a: MovementId, b: MovementId) -> bool {
        self.conflicts.iter().any(|c| (c.a == a && c.b == b) || (c.a == b && c.b == a))
    }

    /// World `[x, y, heading]` of a point `s` metres along a movement's interior
    /// crossing path (clamped to its length).
    pub fn interior_point(&self, mid: MovementId, s: f64) -> [f64; 3] {
        let it = self.interior(mid);
        let t = (s / it.len.max(1e-9)).clamp(0.0, 1.0);
        let p = bezier3(it.entry, it.c1, it.c2, it.exit, t);
        // Clamped centred difference, so the tangent is valid at both endpoints.
        let a = bezier3(it.entry, it.c1, it.c2, it.exit, (t - 0.02).max(0.0));
        let b = bezier3(it.entry, it.c1, it.c2, it.exit, (t + 0.02).min(1.0));
        let d = unit(sub(b, a));
        [p[0], p[1], d[1].atan2(d[0])]
    }

    /// Compute each movement's interior path and all cross-movement conflict
    /// points. Called once at build time; `interiors` is index-aligned with
    /// `movements`. Two movements conflict when they arrive from different links,
    /// depart to different links, and their interior paths pass within a lane
    /// width of each other — a genuine crossing, not a merge or a diverge.
    pub fn build_interiors(&mut self) {
        self.interiors = (0..self.movements.len() as u32)
            .map(|m| {
                let mv = self.movement(MovementId(m));
                let entry = point2(self.lane_point(mv.from_lane, self.lane(mv.from_lane).length));
                let exit = point2(self.lane_point(mv.to_lane, 0.0));
                // Control handles lie along the arrival and departure road
                // directions, a third of the chord out, so the path leaves and
                // arrives tangent to each road (straight when collinear).
                let arr = self.arrival_dir(self.lane(mv.from_lane).link);
                let dep = self.departure_dir(self.lane(mv.to_lane).link);
                let k = norm(sub(exit, entry)) / 3.0;
                let c1 = [entry[0] + arr[0] * k, entry[1] + arr[1] * k];
                let c2 = [exit[0] - dep[0] * k, exit[1] - dep[1] * k];
                let mut it = Interior { entry, c1, c2, exit, len: 0.0 };
                it.len = interior_polyline(&it).last().map_or(0.0, |&(_, s)| s);
                it
            })
            .collect();

        // ~ a vehicle width: paths that pass farther apart than this share the box
        // safely (e.g. opposing protected lefts), so they neither collide nor force
        // separate signal phases; only genuine crossings (near-zero separation) do.
        const CLEARANCE: f64 = 2.0;
        let polys: Vec<Vec<([f64; 2], f64)>> = self.interiors.iter().map(interior_polyline).collect();
        let mut conflicts = Vec::new();
        for i in 0..self.movements.len() {
            for j in (i + 1)..self.movements.len() {
                let (a, b) = (self.movements[i], self.movements[j]);
                if a.node != b.node {
                    continue;
                }
                if self.lane(a.from_lane).link == self.lane(b.from_lane).link
                    || self.lane(a.to_lane).link == self.lane(b.to_lane).link
                {
                    continue;
                }
                if let Some((sa, sb)) = nearest_crossing(&polys[i], &polys[j], CLEARANCE) {
                    conflicts.push(ConflictPoint { node: a.node, a: MovementId(i as u32), sa, b: MovementId(j as u32), sb });
                }
            }
        }
        self.conflicts = conflicts;
    }

    pub fn lanes_of(&self, link: LinkId) -> impl Iterator<Item = LaneId> {
        let l = self.link(link);
        let start = l.lane_start.0;
        (start..start + l.lane_count).map(LaneId)
    }

    /// The colour a movement shows at `sim_time`; unsignalized movements are
    /// always green (priority/stop-yield handling is a later layer).
    pub fn movement_state(&self, movement: MovementId, sim_time: f64) -> SignalState {
        match self.movement(movement).signal_group {
            None => SignalState::Green,
            Some(g) => {
                let group = self.groups[g.idx()];
                self.programs[group.program.idx()].state_of(group.bit, sim_time)
            }
        }
    }

    pub fn link_travel_time_ms(&self, link: LinkId) -> u64 {
        let lane = self.lane(self.link(link).lane_start);
        ((lane.length / lane.speed_limit.max(0.1)) * 1000.0) as u64
    }

    pub fn outgoing_links(&self, link: LinkId) -> Vec<LinkId> {
        let mut set = std::collections::BTreeSet::new();
        for lane in self.lanes_of(link) {
            for m in self.movements_of(lane) {
                set.insert(self.lane(m.to_lane).link.0);
            }
        }
        set.into_iter().map(LinkId).collect()
    }

    /// Fastest link-by-link route from `from` to `to` (inclusive) under
    /// free-flow travel times.
    pub fn route_links(&self, from: LinkId, to: LinkId) -> Option<Vec<LinkId>> {
        self.route_links_weighted(from, to, |l| self.link_travel_time_ms(l))
    }

    /// As [`route_links`] but with live per-link travel-time costs (ms) indexed
    /// by link id — the congestion-reactive path: feed it estimates derived from
    /// the mass layer's occupancy and vehicles route around jams.
    pub fn route_links_with_costs(&self, from: LinkId, to: LinkId, cost_ms: &[u64]) -> Option<Vec<LinkId>> {
        self.route_links_weighted(from, to, |l| cost_ms[l.idx()])
    }

    fn route_links_weighted(
        &self,
        from: LinkId,
        to: LinkId,
        cost: impl Fn(LinkId) -> u64,
    ) -> Option<Vec<LinkId>> {
        use std::cmp::Reverse;
        use std::collections::{BinaryHeap, HashMap};

        let mut dist: HashMap<u32, u64> = HashMap::from([(from.0, 0)]);
        let mut prev: HashMap<u32, u32> = HashMap::new();
        let mut heap = BinaryHeap::from([Reverse((0u64, from.0))]);

        while let Some(Reverse((d, link))) = heap.pop() {
            if link == to.0 {
                break;
            }
            if d > dist.get(&link).copied().unwrap_or(u64::MAX) {
                continue;
            }
            for next in self.outgoing_links(LinkId(link)) {
                let nd = d + cost(next);
                if nd < dist.get(&next.0).copied().unwrap_or(u64::MAX) {
                    dist.insert(next.0, nd);
                    prev.insert(next.0, link);
                    heap.push(Reverse((nd, next.0)));
                }
            }
        }

        if from == to {
            return Some(vec![from]);
        }
        dist.get(&to.0)?;
        let mut path = vec![to.0];
        while let Some(&p) = prev.get(path.last().unwrap()) {
            path.push(p);
            if p == from.0 {
                break;
            }
        }
        path.reverse();
        Some(path.into_iter().map(LinkId).collect())
    }

    /// Lateral offset (m, right of the centreline) of `lane` at `position`. A
    /// normal lane sits at a constant `(index + 0.5)·WIDTH`; a turn-pocket lane
    /// (`pocket_taper > 0`) instead merges into its adjacent through lane far from
    /// the stop line and diverges to its own offset over the bay taper — so the
    /// bay opens near the junction and closes upstream.
    pub fn lane_lateral_offset(&self, l: &Lane, position: f64) -> f64 {
        self.lane_offset_at(l, l.length - position)
    }

    /// Lateral offset of `lane` a given `to_line` metres upstream of its stop line
    /// — the pocket taper expressed in the quantity it actually depends on, so both
    /// vehicle placement and the divider markings sample the same bay geometry.
    pub fn lane_offset_at(&self, l: &Lane, to_line: f64) -> f64 {
        let own = (l.index_in_link as f64 + 0.5) * LANE_WIDTH;
        if l.pocket_taper <= 0.0 {
            return own;
        }
        // A left pocket (lane 0) opens toward the centreline by merging into the
        // lane on its right; a right pocket (outermost lane) merges into its left.
        let blend_right = l.index_in_link == 0;
        let neighbour = own + if blend_right { LANE_WIDTH } else { -LANE_WIDTH };
        if to_line <= POCKET_OPEN {
            own
        } else if to_line >= POCKET_OPEN + l.pocket_taper {
            neighbour
        } else {
            let t = (POCKET_OPEN + l.pocket_taper - to_line) / l.pocket_taper; // 0 closed → 1 open
            neighbour + (own - neighbour) * t
        }
    }

    /// World `[x, y, heading]` of a point `position` metres along `lane`,
    /// laterally offset for the lane's index. Pure geometry the renderer uses to
    /// place vehicle instances.
    pub fn lane_point(&self, lane: LaneId, position: f64) -> [f64; 3] {
        let l = self.lane(lane);
        let poly = &self.polylines[l.link.idx()];
        let (pt, dir) = point_along(poly, l.start_offset + position.clamp(0.0, l.length));
        let off = self.lane_lateral_offset(l, position.clamp(0.0, l.length));
        let n = [dir[1], -dir[0]]; // right-hand normal
        [pt[0] + n[0] * off, pt[1] + n[1] * off, dir[1].atan2(dir[0])]
    }

    /// Unit direction of a link's final segment (heading as it reaches its
    /// downstream node).
    pub fn arrival_dir(&self, link: LinkId) -> [f64; 2] {
        let poly = &self.polylines[link.idx()];
        unit(sub(poly[poly.len() - 1], poly[poly.len() - 2]))
    }

    /// Unit direction of a link's first segment (heading as it leaves its
    /// upstream node).
    pub fn departure_dir(&self, link: LinkId) -> [f64; 2] {
        let poly = &self.polylines[link.idx()];
        unit(sub(poly[1], poly[0]))
    }

    /// Whether a movement goes straight, left, or right, from the signed angle
    /// between the arriving and departing directions.
    pub fn movement_turn(&self, mid: MovementId) -> TurnType {
        let m = self.movement(mid);
        let a = self.arrival_dir(self.lane(m.from_lane).link);
        let b = self.departure_dir(self.lane(m.to_lane).link);
        let ang = (a[0] * b[1] - a[1] * b[0]).atan2(a[0] * b[0] + a[1] * b[1]);
        if ang > 0.5 {
            TurnType::Left
        } else if ang < -0.5 {
            TurnType::Right
        } else {
            TurnType::Through
        }
    }

    /// Smallest turn radius (m) the lane's centreline reaches between `position`
    /// and `position + lookahead` — `f64::INFINITY` on a straight run.
    pub fn min_radius_ahead(&self, lane: LaneId, position: f64, lookahead: f64) -> f64 {
        let poly = &self.polylines[self.lane(lane).link.idx()];
        if poly.len() < 3 {
            return f64::INFINITY;
        }
        let from = self.lane(lane).start_offset + position;
        let mut acc = 0.0;
        let mut best = f64::INFINITY;
        for i in 1..poly.len() - 1 {
            let seg_in = norm(sub(poly[i], poly[i - 1]));
            acc += seg_in;
            if acc < from {
                continue;
            }
            if acc > from + lookahead {
                break;
            }
            let a = unit(sub(poly[i], poly[i - 1]));
            let b = unit(sub(poly[i + 1], poly[i]));
            let cross = a[0] * b[1] - a[1] * b[0];
            let dot = (a[0] * b[0] + a[1] * b[1]).clamp(-1.0, 1.0);
            let angle = cross.atan2(dot).abs();
            if angle > 1e-4 {
                let seg_out = norm(sub(poly[i + 1], poly[i]));
                best = best.min(0.5 * (seg_in + seg_out) / angle);
            }
        }
        best
    }

    /// The link centreline clipped to its lanes' drivable span (between the two
    /// junction boundaries), so carriageways and markings stop at the
    /// intersection rather than crossing into it.
    pub fn drivable_polyline(&self, link: LinkId) -> Vec<[f64; 2]> {
        let l = self.link(link);
        let poly = &self.polylines[link.idx()];
        let full: f64 = poly.windows(2).map(|w| norm(sub(w[1], w[0]))).sum();
        let s0 = self.render_setback.get(l.from.idx()).copied().unwrap_or(0.0);
        let s1 = full - self.render_setback.get(l.to.idx()).copied().unwrap_or(0.0);
        clip_polyline(poly, s0, s1.max(s0 + 0.5))
    }

    /// Filled carriageway quads `[cx0, cy0, cx1, cy1, width]`, one per polyline
    /// segment of each link (curved roads become several quads).
    pub fn road_strips(&self) -> Vec<[f64; 5]> {
        let mut out = Vec::new();
        for i in 0..self.links.len() {
            let w = self.links[i].lane_count as f64 * LANE_WIDTH;
            let c = w / 2.0;
            for seg in self.drivable_polyline(LinkId(i as u32)).windows(2) {
                let dir = unit(sub(seg[1], seg[0]));
                let n = [dir[1] * c, -dir[0] * c];
                out.push([seg[0][0] + n[0], seg[0][1] + n[1], seg[1][0] + n[0], seg[1][1] + n[1], w]);
            }
        }
        out
    }

    /// Interior lane-divider segments `[x0, y0, x1, y1]`, per polyline segment. A
    /// divider tracks the boundary between its two lanes; where a turn pocket
    /// widens an approach the adjacent boundary tapers (the bay-taper line),
    /// sampled finely so the marking follows the opening bay.
    pub fn lane_dividers(&self) -> Vec<[f64; 4]> {
        let mut out = Vec::new();
        for i in 0..self.links.len() {
            let link = self.links[i];
            let lanes = link.lane_count;
            let taper = |k: u32| self.lane(LaneId(link.lane_start.0 + k)).pocket_taper;
            let has_pocket = (0..lanes).any(|k| taper(k) > 0.0);
            let poly = self.drivable_polyline(LinkId(i as u32));
            if !has_pocket {
                for seg in poly.windows(2) {
                    let dir = unit(sub(seg[1], seg[0]));
                    let (nx, ny) = (dir[1], -dir[0]);
                    for k in 1..lanes {
                        let off = k as f64 * LANE_WIDTH;
                        out.push([seg[0][0] + nx * off, seg[0][1] + ny * off, seg[1][0] + nx * off, seg[1][1] + ny * off]);
                    }
                }
                continue;
            }
            // Pocket approach: subdivide so a divider follows the tapering boundary.
            // `to_line` (distance to the stop line) drives the pocket offset, so
            // track cumulative distance from the upstream end and subtract from the
            // total to get each sample's distance to the stop line.
            let total: f64 = poly.windows(2).map(|w| norm(sub(w[1], w[0]))).sum();
            let boundary = |k: u32, to_line: f64| -> f64 {
                let a = self.lane_offset_at(self.lane(LaneId(link.lane_start.0 + k - 1)), to_line);
                let b = self.lane_offset_at(self.lane(LaneId(link.lane_start.0 + k)), to_line);
                0.5 * (a + b)
            };
            let lat = |p: [f64; 2], dir: [f64; 2], o: f64| [p[0] + dir[1] * o, p[1] - dir[0] * o];
            const STEP: f64 = 2.0;
            let mut s0 = 0.0f64; // distance from the upstream end to the segment start
            for seg in poly.windows(2) {
                let seg_len = norm(sub(seg[1], seg[0]));
                if seg_len < 1e-6 {
                    continue;
                }
                let dir = unit(sub(seg[1], seg[0]));
                let steps = (seg_len / STEP).ceil().max(1.0) as usize;
                for k in 1..lanes {
                    for t in 0..steps {
                        let (f0, f1) = (t as f64 / steps as f64, (t + 1) as f64 / steps as f64);
                        let a = [seg[0][0] + (seg[1][0] - seg[0][0]) * f0, seg[0][1] + (seg[1][1] - seg[0][1]) * f0];
                        let b = [seg[0][0] + (seg[1][0] - seg[0][0]) * f1, seg[0][1] + (seg[1][1] - seg[0][1]) * f1];
                        let oa = boundary(k, total - (s0 + seg_len * f0));
                        let ob = boundary(k, total - (s0 + seg_len * f1));
                        let (pa, pb) = (lat(a, dir, oa), lat(b, dir, ob));
                        out.push([pa[0], pa[1], pb[0], pb[1]]);
                    }
                }
                s0 += seg_len;
            }
        }
        out
    }

    /// Axis-aligned world bounds `[min_x, min_y, max_x, max_y]` over all nodes.
    pub fn bounds(&self) -> [f64; 4] {
        let mut r = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
        for node in &self.nodes {
            r[0] = r[0].min(node.position[0]);
            r[1] = r[1].min(node.position[1]);
            r[2] = r[2].max(node.position[0]);
            r[3] = r[3].max(node.position[1]);
        }
        r
    }

    /// One O(#groups) pass evaluating every signal group at `sim_time`; vehicles
    /// then read the returned array in O(1). This is the per-tick scale story.
    pub fn signal_states(&self, sim_time: f64) -> Vec<SignalState> {
        self.groups
            .iter()
            .map(|g| self.programs[g.program.idx()].state_of(g.bit, sim_time))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::super::map::{self, LinkSpec, NodeSpec, OsmMap};
    use super::*;

    #[test]
    fn signal_states_cover_every_group_once() {
        let net = map::corridor_with_signal();
        let states = net.signal_states(0.0);
        assert_eq!(states.len(), net.groups.len());
        assert!(net.groups.len() >= 2);
    }

    #[test]
    fn movements_reference_valid_lanes() {
        let net = map::corridor_with_signal();
        for m in &net.movements {
            assert!(m.from_lane.idx() < net.lanes.len());
            assert!(m.to_lane.idx() < net.lanes.len());
            assert!(m.node.idx() < net.nodes.len());
        }
    }

    #[test]
    fn lane_point_interpolates_between_node_endpoints() {
        let map = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::uncontrolled(2, 100.0, 0.0),
            ],
            links: vec![LinkSpec::oneway(1, 2, 1, 20.0)],
        };
        let net = map.build();
        let lane = LaneId(0);
        let l = *net.lane(lane);
        let start = net.lane_point(lane, 0.0);
        let end = net.lane_point(lane, l.length);
        // Lanes are pulled back to the junction boundaries, so the span sits
        // inside the node endpoints by each node's setback.
        assert!((start[0] - l.start_offset).abs() < 1e-9);
        assert!((end[0] - (l.start_offset + l.length)).abs() < 1e-9);
        assert!(l.start_offset > 0.0 && l.length < 100.0, "setback shortens the drivable span");
        assert!((start[2]).abs() < 1e-9, "eastbound heading is 0 rad");
        assert!(start[1] < 0.0, "right-hand lane offset is negative-y for +x travel");
    }

    #[test]
    fn lane_point_and_length_follow_a_curved_polyline() {
        // L-shaped link (0,0) → bend (100,0) → (100,100).
        let net = OsmMap {
            nodes: vec![NodeSpec::uncontrolled(1, 0.0, 0.0), NodeSpec::uncontrolled(2, 100.0, 100.0)],
            links: vec![LinkSpec { from_osm: 1, to_osm: 2, lanes: 1, speed_limit: 20.0, geometry: vec![[100.0, 0.0]], layer: 0, name: String::new() }],
        }
        .build();
        let lane = LaneId(0);
        let l = *net.lane(lane);
        // Full arc is ~200; the drivable span is that minus the two setbacks.
        assert!((l.start_offset + l.length - (200.0 - l.start_offset)).abs() < 1.0, "span ends a setback short of the far node");
        let mid = net.lane_point(lane, 100.0 - l.start_offset);
        assert!((mid[0] - 100.0).abs() < 5.0 && mid[1].abs() < 5.0, "midpoint near the bend: {mid:?}");
        assert!(net.min_radius_ahead(lane, 0.0, 200.0).is_finite(), "a bend has finite radius");
    }

    #[test]
    fn movement_turns_are_classified() {
        // Corridor: link 1→2 (heading +x) with exits east (2→4, straight) and north
        // (2→5, a left turn). Across the two-lane approach both exist, and the left
        // is channelised onto the left lane (index increases leftward).
        let net = map::corridor_with_signal();
        let lanes: Vec<LaneId> = net.lanes_of(LinkId(0)).collect();
        let turns_of = |lane: LaneId| -> std::collections::HashSet<TurnType> {
            (0..net.lane(lane).movement_count)
                .map(|k| net.movement_turn(MovementId(net.lane(lane).movement_start.0 + k)))
                .collect()
        };
        let all: std::collections::HashSet<TurnType> = lanes.iter().flat_map(|&l| turns_of(l)).collect();
        assert!(all.contains(&TurnType::Through), "east exit is straight");
        assert!(all.contains(&TurnType::Left), "north exit is a left turn");
        // lane 0 sits next to the centreline (the left lane) and carries the left.
        assert!(turns_of(*lanes.first().unwrap()).contains(&TurnType::Left), "left turn is on the left lane");
    }

    #[test]
    fn straight_link_has_infinite_radius() {
        let net = map::corridor_with_signal();
        assert!(net.min_radius_ahead(LaneId(0), 0.0, 500.0).is_infinite());
    }

    #[test]
    fn road_geometry_matches_lane_counts() {
        let net = map::corridor_with_signal();
        assert_eq!(net.road_strips().len(), net.links.len());
        // One divider per interior lane boundary; a turn-pocket approach subdivides
        // its dividers to follow the tapering bay, so that only adds segments.
        let dividers = net.lane_dividers().len();
        let expected: u32 = net.links.iter().map(|l| l.lane_count.saturating_sub(1)).sum();
        assert!(dividers >= expected as usize, "at least one divider per lane boundary: {dividers} >= {expected}");
        let b = net.bounds();
        assert!(b[0] <= b[2] && b[1] <= b[3]);
    }

    #[test]
    fn drivable_polyline_stops_at_the_junction_boundaries() {
        // The trimmed centreline must sit inside the node endpoints by the
        // setback, so road fill and markings never cross into the intersection.
        let net = map::corridor_with_signal();
        for i in 0..net.links.len() {
            let link = net.link(LinkId(i as u32));
            let poly = net.drivable_polyline(LinkId(i as u32));
            let from = net.node(link.from).position;
            let to = net.node(link.to).position;
            let d0 = (poly[0][0] - from[0]).hypot(poly[0][1] - from[1]);
            let d1 = (poly[poly.len() - 1][0] - to[0]).hypot(poly[poly.len() - 1][1] - to[1]);
            // The drawn carriageway stops at the junction-box edge (render setback),
            // which is inside the stop line (that adds a crosswalk margin).
            assert!((d0 - net.render_setback[link.from.idx()]).abs() < 1e-6, "starts at the box edge");
            assert!(d1 > 0.5, "ends short of the downstream node: {d1}");
            assert!(net.render_setback[link.from.idx()] < net.lane(link.lane_start).start_offset, "box edge is inside the stop line");
        }
    }

    #[test]
    fn every_movement_has_a_nonzero_interior_path() {
        let net = map::corridor_with_signal();
        assert_eq!(net.interiors.len(), net.movements.len());
        for m in 0..net.movements.len() as u32 {
            let it = net.interior(MovementId(m));
            assert!(it.len > 0.0, "movement {m} interior has length");
            let start = net.interior_point(MovementId(m), 0.0);
            let end = net.interior_point(MovementId(m), it.len);
            assert!((start[0] - it.entry[0]).abs() < 1e-6 && (end[0] - it.exit[0]).abs() < 1e-6);
        }
    }

    #[test]
    fn turn_interior_is_tangent_to_the_lanes_it_joins() {
        // The crossing path should leave along the arrival heading and arrive along
        // the departure heading (a real turn trajectory), not pivot on the centre.
        let net = map::corridor_with_signal();
        let through_lane = net.lanes_of(LinkId(0)).next().unwrap(); // link 1->2 heading +x
        for k in 0..net.lane(through_lane).movement_count {
            let mid = MovementId(net.lane(through_lane).movement_start.0 + k);
            if net.movement_turn(mid) == TurnType::Through {
                continue;
            }
            let it = net.interior(mid);
            let arr = net.arrival_dir(net.lane(net.movement(mid).from_lane).link);
            let dep = net.departure_dir(net.lane(net.movement(mid).to_lane).link);
            let h0 = net.interior_point(mid, 0.0)[2];
            let h1 = net.interior_point(mid, it.len)[2];
            assert!((h0 - arr[1].atan2(arr[0])).abs() < 0.25, "leaves along the arrival heading");
            assert!((h1 - dep[1].atan2(dep[0])).abs() < 0.25, "arrives along the departure heading");
            // The first handle leads out of the entry along the arrival direction.
            let lead = [it.c1[0] - it.entry[0], it.c1[1] - it.entry[1]];
            assert!(lead[0] * arr[0] + lead[1] * arr[1] > 0.0, "first control handle leads along the arrival road");
        }
    }

    #[test]
    fn crossing_movements_produce_a_conflict_point() {
        // A four-way: west→east through vs south→north through must cross.
        let net = map::corridor_with_signal();
        assert!(!net.conflicts.is_empty(), "the crossing corridor has conflicts");
        for c in &net.conflicts {
            let (a, b) = (net.movement(c.a), net.movement(c.b));
            assert_ne!(net.lane(a.from_lane).link, net.lane(b.from_lane).link, "conflicts are between different approaches");
            let pa = net.interior_point(c.a, c.sa);
            let pb = net.interior_point(c.b, c.sb);
            assert!((pa[0] - pb[0]).hypot(pa[1] - pb[1]) < LANE_WIDTH, "the recorded points coincide");
        }
    }

    #[test]
    fn same_approach_movements_do_not_conflict() {
        // Two movements from the same approach (through vs left off the same link)
        // diverge — they must not be flagged as a crossing conflict.
        let net = map::corridor_with_signal();
        for c in &net.conflicts {
            assert_ne!(net.lane(net.movement(c.a).from_lane).link, net.lane(net.movement(c.b).from_lane).link);
        }
    }

    #[test]
    fn multi_lane_link_exposes_all_lanes() {
        let map = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::uncontrolled(2, 300.0, 0.0),
            ],
            links: vec![LinkSpec::oneway(1, 2, 3, 25.0)],
        };
        let net = map.build();
        let link = LinkId(0);
        assert_eq!(net.lanes_of(link).count(), 3);
    }
}
