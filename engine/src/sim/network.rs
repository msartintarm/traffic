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
index_type!(JunctionId);

pub const LANE_WIDTH: f64 = 3.5;

/// Alignment (dot of arrival and departure directions) above which an
/// interchange movement counts as a continuation seam rather than a turn —
/// cos 45°, wide enough to take in merge ramps at the gore.
const SEAM_ALIGN_DOT: f64 = 0.7;

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

/// A posted sign controlling one approach alone (OSM `highway=stop`/`give_way`
/// mapped on the way at its stop line, rather than on the junction node) —
/// how a two-way stop's minor street is surveyed. Ordered by control strength
/// so merges keep the stronger sign.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum LinkSign {
    #[default]
    None,
    Yield,
    Stop,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Node {
    pub position: [f64; 2],
    pub control: NodeControl,
    /// A railway level crossing: closed to road traffic while a train passes
    /// (the world gates movements here on the train timetable).
    pub rail_crossing: bool,
}

/// Road class, distilled from the OSM `highway` tag. The grade-separated freeway
/// system (mainline + ramps) has free-flow diverge/merge junctions; the at-grade
/// street classes (arterial / collector / local) are stop/signal/turn intersections,
/// and are graded by function so demand and junction defaults can differ by type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoadKind {
    /// Freeway/expressway mainline (`motorway`/`trunk`) — grade-separated, free-flow.
    Freeway,
    /// Freeway on/off ramp or interchange connector (`motorway_link`/`trunk_link`).
    Ramp,
    /// Major surface arterial (`primary`/`secondary`) — high-capacity signalized
    /// street (e.g. El Camino Real / a main avenue), a through-corridor, not a
    /// typical trip endpoint.
    Arterial,
    /// Collector (`tertiary`) — feeds local streets to the arterials.
    Collector,
    /// Local / residential street (`residential`, `unclassified`, service, …) — the
    /// streets where trips actually start and end.
    Local,
}

impl RoadKind {
    /// From an OSM `highway` class string. Ramp is *only* the freeway system's links
    /// (`motorway_link`/`trunk_link`); an arterial/collector `*_link` is an at-grade
    /// slip lane of that class, not a grade-separated ramp.
    pub fn from_osm(class: &str) -> Self {
        match class {
            "motorway" | "trunk" => RoadKind::Freeway,
            "motorway_link" | "trunk_link" => RoadKind::Ramp,
            "primary" | "primary_link" | "secondary" | "secondary_link" => RoadKind::Arterial,
            "tertiary" | "tertiary_link" => RoadKind::Collector,
            _ => RoadKind::Local,
        }
    }

    /// Whether this is part of the grade-separated freeway system (mainline or
    /// ramp) — where junctions are free-flow rather than at-grade crossings.
    pub fn is_grade_separated(self) -> bool {
        matches!(self, RoadKind::Freeway | RoadKind::Ramp)
    }

    /// An ordinary at-grade street (arterial, collector, or local).
    pub fn is_surface(self) -> bool {
        !self.is_grade_separated()
    }

    /// A major road — the freeway system or an arterial. Its trips are through-
    /// movements; local trips start/end off it. (Collector/Local are "minor".)
    pub fn is_major(self) -> bool {
        matches!(self, RoadKind::Freeway | RoadKind::Ramp | RoadKind::Arterial)
    }

    /// At-grade right-of-way rank of the functional class (higher wins): the
    /// primary determinant of priority between crossing streets — a collector
    /// beats a residential street even at the same posted speed and lane count
    /// (Hillcrest × Ashton: both 25 mph × 1, only the class separates them).
    /// Ramps rank below the surface classes they terminate on: a ramp mouth
    /// yields to the street it meets.
    pub fn at_grade_rank(self) -> u64 {
        match self {
            RoadKind::Freeway => 5,
            RoadKind::Arterial => 4,
            RoadKind::Collector => 3,
            RoadKind::Ramp => 2,
            RoadKind::Local => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Link {
    pub from: NodeId,
    pub to: NodeId,
    pub lane_start: LaneId,
    pub lane_count: u32,
    /// Grade-separation level for render z-order (see `LinkSpec::layer`).
    pub layer: i32,
    /// Road class from OSM, so freeway ramps read as free-flow interchanges.
    pub kind: RoadKind,
    /// OSM `motorway` *mainline* — a true freeway regardless of posted speed, so a slow-signed
    /// motorway (the Golden Gate Bridge, 45 mph) still counts as a highway for boundary demand.
    /// `motorway_link` (ramps) is deliberately excluded: ramps are slow (~11 m/s) and treating
    /// them as highway gateways floods the demand generator. `trunk` is excluded too — in a city
    /// it's often a signalized surface arterial (Van Ness, 19th Ave). See `boundary::is_highway_link`.
    pub motorway: bool,
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

/// Where one external arm plugs into a junction: the cross-section of the
/// carriageway at the box boundary. `anchor` is the median-edge corner, `outer`
/// the curb-edge corner (`lane_count · LANE_WIDTH` to the right of travel), and
/// `dir` the travel direction there — arrival for an inbound arm, departure for
/// an outbound one. The junction owns these; renderers and geometry passes read
/// them instead of re-deriving arm ends from node positions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mouth {
    pub link: LinkId,
    pub inbound: bool,
    pub anchor: [f64; 2],
    pub outer: [f64; 2],
    pub dir: [f64; 2],
}

#[derive(Clone, Debug, PartialEq)]
pub struct Junction {
    pub nodes: Vec<NodeId>,
    pub center: [f64; 2],
    pub footprint: [[f64; 2]; 4],
    pub approaches: Vec<LinkId>,
    pub exits: Vec<LinkId>,
    /// One mouth per external arm (every approach and exit), in arm order.
    pub mouths: Vec<Mouth>,
    pub program: Option<ProgramId>,
}

/// Lanelet2-style lane geometry for one link: shared boundary polylines instead
/// of centreline-plus-offset. `stations` are arc-lengths along the link polyline
/// covering the drivable span (box edge to box edge); `bounds[k]` is boundary
/// `k`'s world point at each station — boundary 0 the median edge, boundary
/// `lane_count` the curb edge, lane `k` the strip between `bounds[k]` and
/// `bounds[k+1]`. Neighbouring lanes *share* one boundary polyline, so they
/// cannot drift apart, and the seam stitch moves a boundary once for both its
/// lanes. Turn-pocket bays live here as edge geometry: the bay-side boundaries
/// converge onto the through edge upstream of the bay, so the median (or curb)
/// line tapers open the way a painted bay does.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LaneBounds {
    pub stations: Vec<f64>,
    pub bounds: Vec<Vec<[f64; 2]>>,
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
    /// OSM route ref per link (index-aligned with `links`), e.g. "US 101"; empty for
    /// unnumbered roads. Lets demand route freeway through-traffic along one highway.
    pub link_refs: Vec<String>,
    /// OSM `turn:lanes` per link (index-aligned with `links`), for this travel
    /// direction; empty when unmapped. Channelizes the lane→movement assignment
    /// (`map::turn_lane_exits`) and paints the renderer's lane-use arrows.
    pub link_turn_lanes: Vec<String>,
    /// OSM `hov:lanes` per link (index-aligned with `links`), this direction,
    /// median outward; empty when unmapped. Resolved to per-lane flags by
    /// [`build_hov_lanes`](Self::build_hov_lanes) — the US-101 express lanes.
    pub link_hov_lanes: Vec<String>,
    /// Per-lane HOV restriction, index-aligned with `lanes`.
    lane_hov: Vec<bool>,
    /// Bus service positions `(link, arc)` resolved from scraped stop points —
    /// buses dwell here (see `attach_bus_stops`).
    pub bus_stops: Vec<(LinkId, f64)>,
    /// Per-program green-wave offsets for the two daily plans: progression rides
    /// the corridor's walk direction in the AM and reverses for the PM commute
    /// (real corridors switch timing plans by time of day). Empty when no
    /// coordination ran; the runtime swaps `programs[p].offset` between them.
    pub am_offsets: Vec<f64>,
    pub pm_offsets: Vec<f64>,
    /// Observed AADT per link (index-aligned with `links`; both directions,
    /// vehicles/day; `0.0` = unobserved). Real counts joined at import — demand
    /// calibrates gateway inflow and gravity attraction against these.
    pub link_aadt: Vec<f64>,
    /// Land-use trip-production weight per link (`0.0` = no data → neutral 1.0):
    /// residential surroundings produce trips. From the scraper's `--landuse` pass.
    pub link_res_weight: Vec<f64>,
    /// Land-use trip-attraction weight per link (`0.0` = no data → neutral 1.0):
    /// shops/jobs/campuses attract trips.
    pub link_attr_weight: Vec<f64>,
    /// Per-approach sign per link (index-aligned with `links`): the OSM
    /// stop/give_way surveyed on the way into its downstream node.
    pub link_signs: Vec<LinkSign>,
    /// Whether each approach faces a stop line at its downstream node — the one
    /// authority the driver model reads. Node-level stop control (a sign on the
    /// junction node, or the all-way cluster promote) lines every approach;
    /// a per-link sign lines only its own. Empty (a hand-built network) = every
    /// approach at a Stop node serves the line, the historical behaviour.
    pub link_stop_line: Vec<bool>,
    /// Whether a Stop node is a genuine all-way stop (every approach lined) —
    /// gates the FIFO turn-taking protocol; at a two-way stop the minor street
    /// must keep gap-accepting against the major road instead. Empty = all-way.
    pub node_all_way: Vec<bool>,
    pub junctions: Vec<Junction>,
    pub node_junction: Vec<Option<JunctionId>>,
    /// O(1) membership index over `conflicts` (unordered movement-id pair packed into
    /// a `u64`), so [`Network::movements_conflict`] is a hash lookup rather than a scan
    /// of every conflict — the difference between a city-sized map building in seconds
    /// and in minutes. Rebuilt by [`Network::build_conflict_index`] whenever conflicts
    /// change.
    conflict_pairs: std::collections::HashSet<u64>,
    /// Cached `(departure, arrival)` unit direction per link, sampled at the
    /// junction boundaries rather than the raw polyline ends (see
    /// [`build_end_dirs`](Self::build_end_dirs)). Empty until built; the dir
    /// accessors fall back to the raw end segments.
    end_dirs: Vec<([f64; 2], [f64; 2])>,
    /// Shared lane-boundary polylines per link (index-aligned with `links`; see
    /// [`LaneBounds`]). Empty until [`build_lane_bounds`](Self::build_lane_bounds)
    /// runs; vehicle placement, dividers, strips, and mouths fall back to the
    /// centreline-offset model until then.
    pub lane_bounds: Vec<LaneBounds>,
}

fn movement_pair_key(a: MovementId, b: MovementId) -> u64 {
    let (lo, hi) = if a.0 <= b.0 { (a.0, b.0) } else { (b.0, a.0) };
    (lo as u64) << 32 | hi as u64
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

pub(crate) const JUNCTION_MERGE_GAP: f64 = 14.0;
const FOOTPRINT_RADIUS: f64 = 28.0;

/// ~ a vehicle width: interior paths passing farther apart than this share the box
/// safely (e.g. opposing protected lefts), so they neither collide nor force
/// separate signal phases; only genuine crossings (near-zero separation) conflict.
const CONFLICT_CLEARANCE: f64 = 2.0;

/// The road axis a junction footprint aligns to: the travel direction whose
/// near-parallel arms carry the most lanes — the dominant carriageway through the
/// junction. Aligning the rectangle to this, rather than to the min-area hull edge,
/// keeps its sides running with the main road instead of skewing across the pavement.
fn dominant_axis(arms: &[([f64; 2], f64)]) -> [f64; 2] {
    let mut best = ([1.0, 0.0], f64::NEG_INFINITY);
    for &(di, _) in arms {
        if norm(di) < 0.5 {
            continue;
        }
        let score: f64 = arms
            .iter()
            .filter(|(dj, _)| (di[0] * dj[0] + di[1] * dj[1]).abs() > 0.866)
            .map(|(_, w)| w)
            .sum();
        if score > best.1 {
            best = (di, score);
        }
    }
    best.0
}

/// The bounding parallelogram of `pts` with its sides along the two street axes
/// through the junction: each point decomposes as `a + u·e1 + v·e2` in the dual
/// basis and the corners take the extreme `(u, v)` pairs — so an oblique crossing
/// wears the skewed quad its pavement actually covers. Near-parallel axes fall
/// back to the axis-aligned rectangle.
fn oriented_parallelogram(pts: &[[f64; 2]], axis1: [f64; 2], axis2: [f64; 2]) -> [[f64; 2]; 4] {
    let (e1, e2) = (unit(axis1), unit(axis2));
    let det = e1[0] * e2[1] - e1[1] * e2[0];
    if det.abs() < 0.25 {
        return oriented_rect(pts, axis1);
    }
    let a = pts[0];
    let (mut lu, mut hu, mut lv, mut hv) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for &p in pts {
        let d = sub(p, a);
        let u = (d[0] * e2[1] - d[1] * e2[0]) / det;
        let v = (e1[0] * d[1] - e1[1] * d[0]) / det;
        lu = lu.min(u);
        hu = hu.max(u);
        lv = lv.min(v);
        hv = hv.max(v);
    }
    let corner = |u: f64, v: f64| [a[0] + u * e1[0] + v * e2[0], a[1] + u * e1[1] + v * e2[1]];
    [corner(lu, lv), corner(hu, lv), corner(hu, hv), corner(lu, hv)]
}

/// The bounding rectangle of `pts` with its sides parallel and perpendicular to
/// `axis`. `pts` must be non-empty.
fn oriented_rect(pts: &[[f64; 2]], axis: [f64; 2]) -> [[f64; 2]; 4] {
    let e = if norm(axis) > 1e-6 { unit(axis) } else { [1.0, 0.0] };
    let nrm = [-e[1], e[0]];
    let a = pts[0];
    let (mut lu, mut hu, mut lv, mut hv) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for &p in pts {
        let d = sub(p, a);
        let (u, v) = (d[0] * e[0] + d[1] * e[1], d[0] * nrm[0] + d[1] * nrm[1]);
        lu = lu.min(u);
        hu = hu.max(u);
        lv = lv.min(v);
        hv = hv.max(v);
    }
    let corner = |u: f64, v: f64| [a[0] + u * e[0] + v * nrm[0], a[1] + u * e[1] + v * nrm[1]];
    [corner(lu, lv), corner(hu, lv), corner(hu, hv), corner(lu, hv)]
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
/// Axis-aligned bounds `[min_x, min_y, max_x, max_y]` of an interior polyline, for a
/// cheap rejection test before the segment-exact crossing search.
fn poly_bbox(p: &[([f64; 2], f64)]) -> [f64; 4] {
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for &(pt, _) in p {
        lo = [lo[0].min(pt[0]), lo[1].min(pt[1])];
        hi = [hi[0].max(pt[0]), hi[1].max(pt[1])];
    }
    [lo[0], lo[1], hi[0], hi[1]]
}

/// Whether two bounding boxes come within `margin` of each other — if not, the two
/// polylines can't be within `margin`, so there's no need to test for a crossing.
fn bbox_overlap(a: [f64; 4], b: [f64; 4], margin: f64) -> bool {
    a[0] - margin <= b[2] && b[0] - margin <= a[2] && a[1] - margin <= b[3] && b[1] - margin <= a[3]
}

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

    /// OSM route ref of a link (e.g. "US 101"), or "" for an unnumbered road.
    pub fn link_ref(&self, id: LinkId) -> &str {
        self.link_refs.get(id.idx()).map_or("", String::as_str)
    }

    /// Observed AADT of a link (both directions, vehicles/day), or `0.0` when the
    /// road has no attached count.
    pub fn link_aadt(&self, id: LinkId) -> f64 {
        self.link_aadt.get(id.idx()).copied().unwrap_or(0.0)
    }

    /// Whether this approach faces a stop line at its downstream node. `true`
    /// on a network without the computed table (hand-built fixtures): every
    /// approach at a Stop node then serves the line.
    pub fn approach_stops(&self, id: LinkId) -> bool {
        self.link_stop_line.get(id.idx()).copied().unwrap_or(true)
    }

    /// Whether a Stop node is a genuine all-way stop (every approach lined);
    /// `true` without the computed table — the historical all-way assumption.
    pub fn all_way_stop(&self, node: NodeId) -> bool {
        self.node_all_way.get(node.idx()).copied().unwrap_or(true)
    }

    /// Land-use trip-production weight of a link; neutral 1.0 without data.
    pub fn link_res_weight(&self, id: LinkId) -> f64 {
        match self.link_res_weight.get(id.idx()).copied().unwrap_or(0.0) {
            w if w > 0.0 => w,
            _ => 1.0,
        }
    }

    /// Land-use trip-attraction weight of a link; neutral 1.0 without data.
    pub fn link_attr_weight(&self, id: LinkId) -> f64 {
        match self.link_attr_weight.get(id.idx()).copied().unwrap_or(0.0) {
            w if w > 0.0 => w,
            _ => 1.0,
        }
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

    /// Nearest surface (non-freeway) link to a point, with the arc along its
    /// polyline and the projection distance — the resolver behind bus stops
    /// and route traces.
    pub fn nearest_surface_link(&self, p: [f64; 2]) -> Option<(LinkId, f64, f64)> {
        let mut best: Option<(f64, LinkId, f64)> = None;
        for li in 0..self.links.len() {
            if matches!(self.links[li].kind, RoadKind::Freeway | RoadKind::Ramp) {
                continue;
            }
            let mut arc = 0.0;
            for w in self.polylines[li].windows(2) {
                let seg = [w[1][0] - w[0][0], w[1][1] - w[0][1]];
                let len2 = (seg[0] * seg[0] + seg[1] * seg[1]).max(1e-9);
                let t = (((p[0] - w[0][0]) * seg[0] + (p[1] - w[0][1]) * seg[1]) / len2).clamp(0.0, 1.0);
                let q = [w[0][0] + seg[0] * t, w[0][1] + seg[1] * t];
                let d = (q[0] - p[0]).hypot(q[1] - p[1]);
                if best.is_none_or(|(bd, ..)| d < bd) {
                    best = Some((d, LinkId(li as u32), arc + len2.sqrt() * t));
                }
                arc += len2.sqrt();
            }
        }
        best.map(|(d, l, a)| (l, a, d))
    }

    /// Resolve scraped bus-stop points onto surface links: each stop lands on
    /// the nearest non-freeway link's polyline (within 25 m) as a `(link, arc)`
    /// service position buses dwell at.
    pub fn attach_bus_stops(&mut self, pts: &[[f64; 2]]) {
        self.bus_stops.clear();
        for p in pts {
            if let Some((link, arc, d)) = self.nearest_surface_link(*p) {
                if d <= 25.0 {
                    self.bus_stops.push((link, arc));
                }
            }
        }
        self.bus_stops.sort_by(|a, b| a.0 .0.cmp(&b.0 .0).then(a.1.total_cmp(&b.1)));
        // Opposite-side stop pairs often project onto the same link a few metres
        // apart; one service position suffices (two would re-trap a bus).
        self.bus_stops.dedup_by(|b, a| b.0 == a.0 && (b.1 - a.1).abs() < 15.0);
    }

    /// Resolve a sampled route trace (a bus line's stitched way geometry) into a
    /// connected chain of links: nearest link per sample (within 30 m), dedup,
    /// then splice non-adjacent steps with short shortest-path repairs. `None`
    /// when too little of the trace lands on the network.
    pub fn resolve_route_chain(&self, pts: &[[f64; 2]]) -> Option<Vec<LinkId>> {
        // A junction-internal link (both ends in one cluster) is never a route
        // anchor: a trace sample inside a split intersection would otherwise snap
        // to a perpendicular median stub and break the chain. The shortest-path
        // repairs cover junction interiors.
        let internal = |l: LinkId| {
            let a = self.node_junction(self.link(l).from);
            a.is_some() && a == self.node_junction(self.link(l).to)
        };
        let mut raw: Vec<LinkId> = Vec::new();
        for p in pts {
            if let Some((link, _, d)) = self.nearest_surface_link(*p) {
                if d <= 30.0 && !internal(link) && raw.last() != Some(&link) {
                    raw.push(link);
                }
            }
        }
        if raw.len() < 3 {
            return None;
        }
        // Keep the longest connected run: trace noise (opposite-carriageway
        // snaps, out-of-box gaps) breaks a route into fragments — the service
        // the engine can actually run is the biggest one.
        let connected = |a: LinkId, b: LinkId| self.link(a).to == self.link(b).from;
        let mut best: Vec<LinkId> = Vec::new();
        let mut chain: Vec<LinkId> = vec![raw[0]];
        for &next in &raw[1..] {
            let cur = *chain.last().unwrap();
            if next == cur {
                continue;
            }
            if connected(cur, next) {
                chain.push(next);
                continue;
            }
            if let Some(path) = self.route_links(cur, next) {
                // Generous enough to thread the internal links of a split
                // multi-node junction, still local enough to reject trace jumps.
                if path.len() <= 12 {
                    chain.extend(path.into_iter().skip(1));
                    continue;
                }
            }
            if chain.len() > best.len() {
                best = std::mem::take(&mut chain);
            }
            chain = vec![next];
        }
        if chain.len() > best.len() {
            best = chain;
        }
        (best.len() >= 3).then_some(best)
    }

    /// Whether `lane` is HOV/express-restricted (OSM `hov:lanes`).
    pub fn lane_is_hov(&self, lane: LaneId) -> bool {
        self.lane_hov.get(lane.idx()).copied().unwrap_or(false)
    }

    /// Resolve the imported per-link `hov:lanes` strings into per-lane flags.
    /// Tokens run median-outward like `turn:lanes`; call once after every link
    /// and lane exists.
    pub fn build_hov_lanes(&mut self) {
        self.lane_hov = vec![false; self.lanes.len()];
        for li in 0..self.links.len() {
            let Some(spec) = self.link_hov_lanes.get(li) else { continue };
            if spec.is_empty() {
                continue;
            }
            let l = self.links[li];
            for (k, tok) in spec.split('|').take(l.lane_count as usize).enumerate() {
                if tok.trim() == "designated" {
                    self.lane_hov[l.lane_start.idx() + k] = true;
                }
            }
        }
    }

    /// Whether two movements have a crossing conflict point.
    pub fn movements_conflict(&self, a: MovementId, b: MovementId) -> bool {
        self.conflict_pairs.contains(&movement_pair_key(a, b))
    }

    /// Rebuild the O(1) conflict-lookup index from `conflicts`. Call after every pass
    /// that adds conflict points and before anything queries `movements_conflict`.
    pub fn build_conflict_index(&mut self) {
        self.conflict_pairs = self.conflicts.iter().map(|c| movement_pair_key(c.a, c.b)).collect();
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

    /// Smallest turn radius (m) along a movement's interior Bézier — analytic
    /// curvature `|B'×B''| / |B'|³` sampled across the curve, `INFINITY` for a
    /// straight path. What a curve-speed limit through the box reads, the way
    /// [`min_radius_ahead`](Self::min_radius_ahead) serves link curves.
    pub fn interior_min_radius(&self, mid: MovementId) -> f64 {
        let it = self.interior(mid);
        let (p0, p1, p2, p3) = (it.entry, it.c1, it.c2, it.exit);
        let mut best = f64::INFINITY;
        for i in 0..=16 {
            let t = i as f64 / 16.0;
            let u = 1.0 - t;
            let d1 = [
                3.0 * (u * u * (p1[0] - p0[0]) + 2.0 * u * t * (p2[0] - p1[0]) + t * t * (p3[0] - p2[0])),
                3.0 * (u * u * (p1[1] - p0[1]) + 2.0 * u * t * (p2[1] - p1[1]) + t * t * (p3[1] - p2[1])),
            ];
            let d2 = [
                6.0 * (u * (p2[0] - 2.0 * p1[0] + p0[0]) + t * (p3[0] - 2.0 * p2[0] + p1[0])),
                6.0 * (u * (p2[1] - 2.0 * p1[1] + p0[1]) + t * (p3[1] - 2.0 * p2[1] + p1[1])),
            ];
            let speed2 = d1[0] * d1[0] + d1[1] * d1[1];
            if speed2 < 1e-6 {
                continue;
            }
            let cross = (d1[0] * d2[1] - d1[1] * d2[0]).abs();
            if cross > 1e-9 {
                best = best.min(speed2.powf(1.5) / cross);
            }
        }
        best
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
                let arr = self.arrival_dir(self.lane(mv.from_lane).link);
                let dep = self.departure_dir(self.lane(mv.to_lane).link);
                // A continuation seam is not a place a car steers: it runs straight
                // through at its own lateral line, however far the target lane sits —
                // the landing blend eases it over on the next link. A curve here would
                // pack the whole lateral move into the node's ~1 m gap and double back
                // on itself whenever the chord is mostly sideways.
                if self.is_continuation_seam(MovementId(m)) {
                    let d = sub(exit, entry);
                    let gap = (d[0] * arr[0] + d[1] * arr[1]).max(0.25);
                    let exit = [entry[0] + arr[0] * gap, entry[1] + arr[1] * gap];
                    let c1 = [entry[0] + arr[0] * gap / 3.0, entry[1] + arr[1] * gap / 3.0];
                    let c2 = [entry[0] + arr[0] * gap * 2.0 / 3.0, entry[1] + arr[1] * gap * 2.0 / 3.0];
                    return Interior { entry, c1, c2, exit, len: gap };
                }
                // Control handles lie along the arrival and departure road
                // directions, a third of the chord out, so the path leaves and
                // arrives tangent to each road (straight when collinear). At a
                // genuine corner the handles are clamped to the tangent
                // intersection: the Bézier hull then stays inside the triangle
                // (entry, corner, exit), so a right turn hugs its curb return
                // instead of swinging metres outside the box when the two stop
                // lines sit far apart.
                let k = norm(sub(exit, entry)) / 3.0;
                let (mut k1, mut k2) = (k, k);
                let cross = arr[0] * dep[1] - arr[1] * dep[0];
                if cross.abs() > 0.05 {
                    let d = sub(exit, entry);
                    let t = (d[0] * dep[1] - d[1] * dep[0]) / cross;
                    let s = (arr[0] * d[1] - arr[1] * d[0]) / cross;
                    if t > 0.0 && s > 0.0 {
                        // Floored so a degenerate corner (a stop line nearly on
                        // the tangent crossing) can't collapse a handle to zero
                        // and rotate the end tangency off its road.
                        k1 = (t * 0.85).clamp(k * 0.45, k);
                        k2 = (s * 0.85).clamp(k * 0.45, k);
                    }
                }
                let c1 = [entry[0] + arr[0] * k1, entry[1] + arr[1] * k1];
                let c2 = [exit[0] - dep[0] * k2, exit[1] - dep[1] * k2];
                let mut it = Interior { entry, c1, c2, exit, len: 0.0 };
                it.len = interior_polyline(&it).last().map_or(0.0, |&(_, s)| s);
                it
            })
            .collect();

        let polys: Vec<Vec<([f64; 2], f64)>> = self.interiors.iter().map(interior_polyline).collect();
        let bboxes: Vec<[f64; 4]> = polys.iter().map(|p| poly_bbox(p)).collect();
        // Only movements sharing a node can cross, so group by node and pair within
        // each node — O(Σ node_movements²) instead of O(movements²), which is what
        // lets a whole-city map build in reasonable time.
        let mut by_node: Vec<Vec<usize>> = vec![Vec::new(); self.nodes.len()];
        for (i, mv) in self.movements.iter().enumerate() {
            by_node[mv.node.idx()].push(i);
        }
        let mut conflicts = Vec::new();
        for members in &by_node {
            for (x, &i) in members.iter().enumerate() {
                for &j in &members[x + 1..] {
                    if !bbox_overlap(bboxes[i], bboxes[j], CONFLICT_CLEARANCE) {
                        continue; // paths can't come within clearance — skip the crossing test
                    }
                    let (a, b) = (self.movements[i], self.movements[j]);
                    if self.lane(a.from_lane).link == self.lane(b.from_lane).link
                        || self.lane(a.to_lane).link == self.lane(b.to_lane).link
                    {
                        continue;
                    }
                    if let Some((sa, sb)) = nearest_crossing(&polys[i], &polys[j], CONFLICT_CLEARANCE) {
                        conflicts.push(ConflictPoint { node: a.node, a: MovementId(i as u32), sa, b: MovementId(j as u32), sb });
                    }
                }
            }
        }
        self.conflicts = conflicts;
    }

    /// Append conflict points between movements that belong to the same junction
    /// cluster but different OSM nodes. [`build_interiors`] only pairs movements
    /// sharing a node; where a real intersection is modelled as several close
    /// nodes, cross-arm paths that genuinely cross would otherwise be invisible to
    /// the collision and box-yield logic. Must run after [`build_junctions`].
    pub fn build_cross_junction_conflicts(&mut self) {
        let polys: Vec<Vec<([f64; 2], f64)>> = self.interiors.iter().map(interior_polyline).collect();
        // Group movements by junction so only same-cluster pairs are tested — O(Σ
        // junction_movements²) rather than O(movements²).
        let mut by_junction: Vec<Vec<usize>> = vec![Vec::new(); self.junctions.len()];
        for (i, mv) in self.movements.iter().enumerate() {
            if let Some(j) = self.node_junction(mv.node) {
                by_junction[j.idx()].push(i);
            }
        }
        let mut extra = Vec::new();
        for members in &by_junction {
            for (x, &i) in members.iter().enumerate() {
                for &j in &members[x + 1..] {
                    let (a, b) = (self.movements[i], self.movements[j]);
                    if a.node == b.node {
                        continue; // same-node pairs are handled by build_interiors
                    }
                    if self.lane(a.from_lane).link == self.lane(b.from_lane).link
                        || self.lane(a.to_lane).link == self.lane(b.to_lane).link
                    {
                        continue;
                    }
                    if a.to_lane == b.from_lane || b.to_lane == a.from_lane {
                        continue; // sequential segments of one path — traversed in turn, never at once
                    }
                    if let Some((sa, sb)) = nearest_crossing(&polys[i], &polys[j], CONFLICT_CLEARANCE) {
                        extra.push(ConflictPoint { node: a.node, a: MovementId(i as u32), sa, b: MovementId(j as u32), sb });
                    }
                }
            }
        }
        self.conflicts.extend(extra);
    }

    pub fn junction(&self, id: JunctionId) -> &Junction {
        &self.junctions[id.idx()]
    }

    pub fn node_junction(&self, node: NodeId) -> Option<JunctionId> {
        self.node_junction.get(node.idx()).copied().flatten()
    }

    /// The external approach link a path entered a junction through, tracing back
    /// across any interior links from `start`. For a lane outside a junction (or fed
    /// straight from outside) this is just its own link. Signal grouping keys on it so
    /// every movement of one through-path across a multi-node junction shares a group
    /// and shows one colour — the vehicle never meets a red between the interior nodes.
    pub(crate) fn entry_link(&self, start: LaneId, feeders_of: &[Vec<MovementId>]) -> LinkId {
        let mut lane = start;
        for _ in 0..8 {
            let link = self.lane(lane).link;
            let a = self.node_junction(self.link(link).from);
            if a.is_none() || a != self.node_junction(self.link(link).to) {
                return link;
            }
            // Follow only the *straightest* Through feeder: an internal lane can
            // catch Through-classified landings from more than one approach, and
            // the grouping walk must trace the geometric through-chain. A lane fed
            // only by turns is a turn-path's continuation, not part of any
            // approach's through-path — keying it to the turn's own approach would
            // seat a crossing movement in that approach's signal group and evict
            // the genuine throughs into conflicting groups. Stop and key on the
            // internal link itself instead.
            let feeders = &feeders_of[lane.idx()];
            let own = self.departure_dir(link);
            let align = |p: MovementId| {
                let d = self.arrival_dir(self.lane(self.movement(p).from_lane).link);
                d[0] * own[0] + d[1] * own[1]
            };
            let pred = feeders
                .iter()
                .copied()
                .filter(|&p| self.movement_turn(p) == TurnType::Through)
                .max_by(|&a, &b| align(a).total_cmp(&align(b)));
            match pred {
                Some(p) => lane = self.movement(p).from_lane,
                None => return link,
            }
        }
        self.lane(lane).link
    }

    /// Index of the movements landing on each lane (`to_lane`), for [`entry_link`].
    pub(crate) fn feeders_by_lane(&self) -> Vec<Vec<MovementId>> {
        let mut feeders = vec![Vec::new(); self.lanes.len()];
        for (m, mv) in self.movements.iter().enumerate() {
            feeders[mv.to_lane.idx()].push(MovementId(m as u32));
        }
        feeders
    }

    /// A stable id for the whole intersection `node` belongs to: every node in a
    /// junction cluster returns the same key (its cluster's first member), and a
    /// node outside any cluster returns its own id. Runtime neighbour maps key on
    /// this so box gating, priority yielding, and in-box collision detection treat a
    /// multi-node intersection as one, not as several independent mini-junctions.
    pub fn intersection_key(&self, node: NodeId) -> u32 {
        match self.node_junction(node) {
            Some(j) => self.junction(j).nodes[0].0,
            None => node.0,
        }
    }

    pub(crate) fn arm_mouth(&self, link: LinkId, end_is_to: bool) -> ([f64; 2], [f64; 2]) {
        let l = self.link(link);
        if let Some(lb) = self.link_bounds(link) {
            // The mouth *is* the boundary chart's end cross-section: median and
            // curb boundary endpoints, stitched corrections included.
            let si = if end_is_to { lb.stations.len() - 1 } else { 0 };
            return (lb.bounds[0][si], lb.bounds[l.lane_count as usize][si]);
        }
        let dp = self.drivable_polyline(link);
        let k = dp.len();
        if k < 2 {
            let p = self.node(if end_is_to { l.to } else { l.from }).position;
            return (p, p);
        }
        let (end, travel) = if end_is_to {
            (dp[k - 1], unit(sub(dp[k - 1], dp[k - 2])))
        } else {
            (dp[0], unit(sub(dp[0], dp[1])))
        };
        let right = [travel[1], -travel[0]];
        let full = l.lane_count as f64 * LANE_WIDTH;
        (end, [end[0] + right[0] * full, end[1] + right[1] * full])
    }

    pub fn build_junctions(&mut self) {
        let n = self.nodes.len();
        let mut nb: Vec<std::collections::BTreeSet<u32>> = vec![Default::default(); n];
        for l in &self.links {
            if l.layer != 0 {
                continue;
            }
            nb[l.from.idx()].insert(l.to.0);
            nb[l.to.idx()].insert(l.from.0);
        }
        let is_ix = |i: usize| nb[i].len() >= 3;

        let mut parent: Vec<usize> = (0..n).collect();
        fn find(p: &mut [usize], x: usize) -> usize {
            let mut r = x;
            while p[r] != r {
                r = p[r];
            }
            let mut c = x;
            while p[c] != r {
                let nx = p[c];
                p[c] = r;
                c = nx;
            }
            r
        }
        // Short-gap links (their boxes nearly touch) are junction-interior pavement.
        // Seeded at intersection nodes, then expanded to a fixpoint: a short link also
        // joins when either side's cluster already holds an intersection — so the stub
        // chain leading into a junction (the bend and signal nodes between the outer
        // stop line and the box) is annexed whole. Without the expansion, sliver links
        // straddle the boundary and cars stop at red on a few metres of pavement in
        // the middle of the drawn intersection.
        let short: Vec<(usize, usize)> = self
            .links
            .iter()
            .enumerate()
            .filter_map(|(i, l)| {
                if l.layer != 0 {
                    return None;
                }
                let full: f64 = self.polylines[i].windows(2).map(|w| norm(sub(w[1], w[0]))).sum();
                let gap = full
                    - self.render_setback.get(l.from.idx()).copied().unwrap_or(0.0)
                    - self.render_setback.get(l.to.idx()).copied().unwrap_or(0.0);
                (gap < JUNCTION_MERGE_GAP).then_some((l.from.idx(), l.to.idx()))
            })
            .collect();
        let mut root_ix: Vec<bool> = (0..n).map(is_ix).collect();
        loop {
            let mut changed = false;
            for &(a, b) in &short {
                let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                if ra != rb && (root_ix[ra] || root_ix[rb]) {
                    parent[ra] = rb;
                    root_ix[rb] |= root_ix[ra];
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let mut has_ix: std::collections::HashMap<usize, bool> = Default::default();
        for i in 0..n {
            let r = find(&mut parent, i);
            *has_ix.entry(r).or_insert(false) |= is_ix(i);
        }
        let mut root_index: std::collections::HashMap<usize, usize> = Default::default();
        let mut cluster_of = vec![None; n];
        for i in 0..n {
            let r = find(&mut parent, i);
            if has_ix[&r] {
                let next = root_index.len();
                let ci = *root_index.entry(r).or_insert(next);
                cluster_of[i] = Some(ci);
            }
        }
        let ncl = root_index.len();

        let mut members: Vec<Vec<NodeId>> = vec![Vec::new(); ncl];
        let mut apex = vec![[0.0f64; 2]; ncl];
        let mut apex_deg = vec![0usize; ncl];
        let mut arm_pts: Vec<Vec<[f64; 2]>> = vec![Vec::new(); ncl];
        let mut arm_dirs: Vec<Vec<([f64; 2], f64)>> = vec![Vec::new(); ncl];
        let mut approaches: Vec<Vec<LinkId>> = vec![Vec::new(); ncl];
        let mut exits: Vec<Vec<LinkId>> = vec![Vec::new(); ncl];
        let mut program: Vec<Option<ProgramId>> = vec![None; ncl];
        for i in 0..n {
            if let Some(ci) = cluster_of[i] {
                members[ci].push(NodeId(i as u32));
                if nb[i].len() >= apex_deg[ci] {
                    apex_deg[ci] = nb[i].len();
                    apex[ci] = self.nodes[i].position;
                }
                if let NodeControl::Signalized(p) = self.nodes[i].control {
                    program[ci].get_or_insert(p);
                }
            }
        }
        let mut mouths: Vec<Vec<Mouth>> = vec![Vec::new(); ncl];
        for i in 0..self.links.len() as u32 {
            let l = self.link(LinkId(i));
            if l.layer != 0 {
                continue;
            }
            let (ca, cb) = (cluster_of[l.from.idx()], cluster_of[l.to.idx()]);
            if ca.is_some() && ca == cb {
                continue;
            }
            let near_member = |m: [f64; 2], ci: usize| -> f64 {
                members[ci].iter().map(|&nd| norm(sub(m, self.nodes[nd.idx()].position))).fold(f64::MAX, f64::min)
            };
            if let Some(ci) = cb {
                approaches[ci].push(LinkId(i));
                let (m, o) = self.arm_mouth(LinkId(i), true);
                let dir = self.arrival_dir(LinkId(i));
                mouths[ci].push(Mouth { link: LinkId(i), inbound: true, anchor: m, outer: o, dir });
                if near_member(m, ci) < FOOTPRINT_RADIUS {
                    arm_pts[ci].push(m);
                    arm_pts[ci].push(o);
                    arm_dirs[ci].push((dir, l.lane_count as f64));
                }
            }
            if let Some(ci) = ca {
                exits[ci].push(LinkId(i));
                let (m, o) = self.arm_mouth(LinkId(i), false);
                let dir = self.departure_dir(LinkId(i));
                mouths[ci].push(Mouth { link: LinkId(i), inbound: false, anchor: m, outer: o, dir });
                if near_member(m, ci) < FOOTPRINT_RADIUS {
                    arm_pts[ci].push(m);
                    arm_pts[ci].push(o);
                    arm_dirs[ci].push((dir, l.lane_count as f64));
                }
            }
        }

        self.junctions = (0..ncl)
            .map(|ci| {
                // A lone crossing wears the skewed quad its two streets actually
                // cover; a sprawling multi-node cluster keeps the dominant-axis
                // rectangle, which bounds its scattered arms tightly.
                let footprint = if arm_pts[ci].len() >= 4 {
                    let e1 = dominant_axis(&arm_dirs[ci]);
                    let crossing: Vec<([f64; 2], f64)> = arm_dirs[ci]
                        .iter()
                        .filter(|(d, _)| (d[0] * e1[0] + d[1] * e1[1]).abs() <= 0.866)
                        .copied()
                        .collect();
                    if members[ci].len() == 1 && !crossing.is_empty() {
                        oriented_parallelogram(&arm_pts[ci], e1, dominant_axis(&crossing))
                    } else {
                        oriented_rect(&arm_pts[ci], e1)
                    }
                } else {
                    let c = apex[ci];
                    [[c[0] - 3.0, c[1] - 3.0], [c[0] + 3.0, c[1] - 3.0], [c[0] + 3.0, c[1] + 3.0], [c[0] - 3.0, c[1] + 3.0]]
                };
                Junction {
                    nodes: std::mem::take(&mut members[ci]),
                    center: apex[ci],
                    footprint,
                    approaches: std::mem::take(&mut approaches[ci]),
                    exits: std::mem::take(&mut exits[ci]),
                    mouths: std::mem::take(&mut mouths[ci]),
                    program: program[ci],
                }
            })
            .collect();
        self.node_junction = cluster_of.iter().map(|c| c.map(|ci| JunctionId(ci as u32))).collect();
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
    /// place vehicle instances. Once [`build_lane_bounds`](Self::build_lane_bounds)
    /// has run, the scalar offset is read *through the boundary chart*: the
    /// offset picks which pair of stored boundary polylines brackets the point
    /// and interpolates between them — so wherever a boundary was stitched or a
    /// bay tapers, every lane and every vehicle follows the same shared line.
    /// Before that (hand-built test networks), the raw centreline-offset model.
    pub fn lane_point(&self, lane: LaneId, position: f64) -> [f64; 3] {
        let l = self.lane(lane);
        let pos = position.clamp(0.0, l.length);
        let off = self.lane_lateral_offset(l, pos);
        if let Some(lb) = self.link_bounds(l.link) {
            let n = self.link(l.link).lane_count as usize;
            let s = l.start_offset + pos;
            let i = lb.stations.partition_point(|&x| x < s).clamp(1, lb.stations.len() - 1);
            let (sa, sb) = (lb.stations[i - 1], lb.stations[i]);
            let t = (s - sa) / (sb - sa).max(1e-9);
            let f = (off / LANE_WIDTH).clamp(0.0, n as f64);
            let j = (f as usize).min(n - 1);
            let frac = f - j as f64;
            let at = |si: usize| {
                let (a, b) = (lb.bounds[j][si], lb.bounds[j + 1][si]);
                [a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac]
            };
            let (pa, pb) = (at(i - 1), at(i));
            let d = unit(sub(pb, pa));
            return [pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t, d[1].atan2(d[0])];
        }
        let poly = &self.polylines[l.link.idx()];
        let (pt, dir) = point_along(poly, l.start_offset + pos);
        let n = [dir[1], -dir[0]]; // right-hand normal
        [pt[0] + n[0] * off, pt[1] + n[1] * off, dir[1].atan2(dir[0])]
    }

    /// Unit direction of a link as it reaches its downstream junction. Cached at
    /// the junction boundary once [`build_end_dirs`](Self::build_end_dirs) has
    /// run; before that, the raw final polyline segment.
    pub fn arrival_dir(&self, link: LinkId) -> [f64; 2] {
        if let Some(d) = self.end_dirs.get(link.idx()) {
            return d.1;
        }
        let poly = &self.polylines[link.idx()];
        unit(sub(poly[poly.len() - 1], poly[poly.len() - 2]))
    }

    /// Unit direction of a link as it leaves its upstream junction (cached like
    /// [`arrival_dir`](Self::arrival_dir)).
    pub fn departure_dir(&self, link: LinkId) -> [f64; 2] {
        if let Some(d) = self.end_dirs.get(link.idx()) {
            return d.0;
        }
        let poly = &self.polylines[link.idx()];
        unit(sub(poly[1], poly[0]))
    }

    /// Cache each link's end directions, sampled just inside the junction
    /// boundary (at least `MIN_DIR_SAMPLE` in from the polyline end) instead of
    /// on the raw end segment. The last few metres of an imported link can jag —
    /// a merged node's centroid re-point, dense survey noise at a crossing — and
    /// everything that reads an approach heading (turn classification, interior
    /// tangents, lane fans, footprint axes, stop-line bands) wants the direction
    /// the road actually holds at the stop line, not the jag's.
    pub fn build_end_dirs(&mut self) {
        const MIN_DIR_SAMPLE: f64 = 8.0;
        self.end_dirs = (0..self.links.len())
            .map(|i| {
                let l = &self.links[i];
                let poly = &self.polylines[i];
                let full: f64 = poly.windows(2).map(|w| norm(sub(w[1], w[0]))).sum();
                let sb = |n: NodeId| self.render_setback.get(n.idx()).copied().unwrap_or(0.0);
                let s0 = sb(l.from).max(MIN_DIR_SAMPLE).min(full * 0.45);
                let s1 = sb(l.to).max(MIN_DIR_SAMPLE).min(full * 0.45);
                (point_along(poly, s0).1, point_along(poly, full - s1).1)
            })
            .collect();
    }

    /// Build the shared lane-boundary polylines ([`LaneBounds`]) for every link
    /// from the settled axis, setbacks, and turn pockets. Stations cover the
    /// drivable span: its ends, every axis vertex inside it, and a fine
    /// subdivision across any pocket-bay taper so the converging edge is a curve
    /// rather than one long chord. Vertices offset along averaged (mitred)
    /// normals, so a lane line bends smoothly at an axis bend instead of
    /// stepping sideways.
    pub fn build_lane_bounds(&mut self) {
        const BAY_STEP: f64 = 2.0;
        self.lane_bounds = (0..self.links.len())
            .map(|i| {
                let l = self.links[i];
                let poly = &self.polylines[i];
                if poly.len() < 2 {
                    return LaneBounds::default();
                }
                let mut arcs = vec![0.0f64; poly.len()];
                for j in 1..poly.len() {
                    arcs[j] = arcs[j - 1] + norm(sub(poly[j], poly[j - 1]));
                }
                let full = arcs[poly.len() - 1];
                let n = l.lane_count as usize;
                let lane = |k: usize| &self.lanes[l.lane_start.idx() + k];
                // Cover the drivable render span *and* the lane span: a sliver
                // link squeezed by its setbacks can have its stop lines outside
                // the box-edge clip (the setback rescale), and every consumer —
                // vehicle placement, seam measurement, interiors — must sample
                // the chart there, never extrapolate off a kinked half-metre.
                let (l0, l1) = (lane(0).start_offset, lane(0).start_offset + lane(0).length);
                let s0 = self.render_setback.get(l.from.idx()).copied().unwrap_or(0.0).min(l0).max(0.0);
                let s1 = (full - self.render_setback.get(l.to.idx()).copied().unwrap_or(0.0)).max(l1).max(s0 + 0.5).min(full);
                // Edge bay blocks: consecutive pocket lanes from the median
                // (left bay) and from the curb (right bay); their boundaries
                // collapse onto the adjacent through edge when the bay closes.
                let ml = (0..n).take_while(|&k| lane(k).pocket_taper > 0.0).count();
                let mr = (0..n).rev().take_while(|&k| lane(k).pocket_taper > 0.0).count().min(n - ml);
                let taper_l = (0..ml).map(|k| lane(k).pocket_taper).fold(0.0f64, f64::max);
                let taper_r = (n - mr..n).map(|k| lane(k).pocket_taper).fold(0.0f64, f64::max);
                let stop = lane(0).start_offset + lane(0).length;

                let mut stations = vec![s0, s1];
                stations.extend(arcs.iter().copied().filter(|&a| a > s0 + 1e-6 && a < s1 - 1e-6));
                let taper_max = taper_l.max(taper_r);
                if taper_max > 0.0 {
                    let (za, zb) = ((stop - POCKET_OPEN - taper_max).max(s0), (stop - POCKET_OPEN).min(s1));
                    let mut s = za;
                    while s < zb {
                        stations.push(s);
                        s += BAY_STEP;
                    }
                    stations.push(zb);
                }
                stations.sort_by(|a, b| a.total_cmp(b));
                stations.dedup_by(|a, b| (*a - *b).abs() < 1e-6);

                // Point and mitred right-hand normal at each station.
                let seg_dir = |j: usize| unit(sub(poly[j + 1], poly[j]));
                let geom: Vec<([f64; 2], [f64; 2])> = stations
                    .iter()
                    .map(|&s| {
                        let j = arcs[..poly.len() - 1].partition_point(|&a| a <= s + 1e-9).saturating_sub(1);
                        let t = ((s - arcs[j]) / (arcs[j + 1] - arcs[j]).max(1e-9)).clamp(0.0, 1.0);
                        let pt = [
                            poly[j][0] + (poly[j + 1][0] - poly[j][0]) * t,
                            poly[j][1] + (poly[j + 1][1] - poly[j][1]) * t,
                        ];
                        let at_vertex = (s - arcs[j]).abs() < 1e-6 && j > 0;
                        let d = if at_vertex {
                            unit([seg_dir(j - 1)[0] + seg_dir(j)[0], seg_dir(j - 1)[1] + seg_dir(j)[1]])
                        } else {
                            seg_dir(j)
                        };
                        (pt, [d[1], -d[0]])
                    })
                    .collect();

                // The bay-open fraction at a station: 1 at the stop line, 0 once
                // the bay has merged into its through lane upstream.
                let open = |s: f64, taper: f64| -> f64 {
                    let to_line = stop - s;
                    if to_line <= POCKET_OPEN {
                        1.0
                    } else if to_line >= POCKET_OPEN + taper {
                        0.0
                    } else {
                        (POCKET_OPEN + taper - to_line) / taper
                    }
                };
                let bounds = (0..=n)
                    .map(|k| {
                        stations
                            .iter()
                            .zip(&geom)
                            .map(|(&s, &(pt, right))| {
                                let base = k as f64 * LANE_WIDTH;
                                let off = if k < ml {
                                    let closed = ml as f64 * LANE_WIDTH;
                                    closed + (base - closed) * open(s, taper_l)
                                } else if k > n - mr {
                                    let closed = (n - mr) as f64 * LANE_WIDTH;
                                    closed + (base - closed) * open(s, taper_r)
                                } else {
                                    base
                                };
                                [pt[0] + right[0] * off, pt[1] + right[1] * off]
                            })
                            .collect()
                    })
                    .collect();
                LaneBounds { stations, bounds }
            })
            .collect();
    }

    /// The stored boundary chart for a link, when built and usable.
    fn link_bounds(&self, link: LinkId) -> Option<&LaneBounds> {
        self.lane_bounds.get(link.idx()).filter(|lb| lb.stations.len() >= 2)
    }

    /// A movement is a *free-flow interchange* when both the road it leaves and the
    /// road it joins are grade-separated (freeway mainline or ramp) — a highway
    /// diverge, merge, or ramp-to-ramp connector. These carry no cross traffic, so
    /// they run at road speed rather than being throttled like an at-grade turn.
    pub fn is_interchange_movement(&self, mid: MovementId) -> bool {
        let mv = self.movement(mid);
        self.link(self.lane(mv.from_lane).link).kind.is_grade_separated()
            && self.link(self.lane(mv.to_lane).link).kind.is_grade_separated()
    }

    /// An interchange movement whose roads run on together — a segment seam, merge,
    /// or gore, not a real turn. A car takes it by continuing straight; whatever
    /// lane the movement targets is reached by a lateral ease *on* the next link
    /// (the seam-landing blend), never by swerving inside the node.
    pub fn is_continuation_seam(&self, mid: MovementId) -> bool {
        let mv = self.movement(mid);
        let (fl, tl) = (self.lane(mv.from_lane).link, self.lane(mv.to_lane).link);
        let a = self.arrival_dir(fl);
        let b = self.departure_dir(tl);
        self.is_interchange_movement(mid) && a[0] * b[0] + a[1] * b[1] > SEAM_ALIGN_DOT
    }

    /// Whether every carriageway meeting `node` is grade-separated — a pure highway
    /// interchange point (diverge/merge/connector), with no at-grade cross street.
    pub fn is_interchange_node(&self, node: NodeId) -> bool {
        let mut any = false;
        for l in &self.links {
            if l.from == node || l.to == node {
                any = true;
                if !l.kind.is_grade_separated() {
                    return false;
                }
            }
        }
        any
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
    /// segment of each link (curved roads become several quads). With lane
    /// bounds built, each quad spans the median and curb boundaries — so the
    /// carriageway (and the edge lines painted on it) follows a pocket bay's
    /// taper and any stitched seam correction.
    pub fn road_strips(&self) -> Vec<[f64; 5]> {
        let mut out = Vec::new();
        for i in 0..self.links.len() {
            if let Some(lb) = self.link_bounds(LinkId(i as u32)) {
                let n = self.links[i].lane_count as usize;
                for si in 1..lb.stations.len() {
                    let (m0, c0) = (lb.bounds[0][si - 1], lb.bounds[n][si - 1]);
                    let (m1, c1) = (lb.bounds[0][si], lb.bounds[n][si]);
                    let w = (norm(sub(c0, m0)) + norm(sub(c1, m1))) * 0.5;
                    if w < 1e-6 {
                        continue;
                    }
                    let mid = |a: [f64; 2], b: [f64; 2]| [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
                    let (a, b) = (mid(m0, c0), mid(m1, c1));
                    out.push([a[0], a[1], b[0], b[1], w]);
                }
                continue;
            }
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

    /// Interior lane-divider segments `[x0, y0, x1, y1]`. With lane bounds built
    /// these are the shared boundary polylines themselves — a bay's tapering
    /// edge included — minus any stretch where a boundary has collapsed onto the
    /// carriageway edge (a closed bay), which the edge line already draws.
    pub fn lane_dividers(&self) -> Vec<[f64; 4]> {
        let mut out = Vec::new();
        for i in 0..self.links.len() {
            let link = self.links[i];
            let lanes = link.lane_count;
            if let Some(lb) = self.link_bounds(LinkId(i as u32)) {
                let n = lanes as usize;
                for k in 1..n {
                    for si in 1..lb.stations.len() {
                        let (a, b) = (lb.bounds[k][si - 1], lb.bounds[k][si]);
                        let on_edge = |p: [f64; 2], q: [f64; 2]| norm(sub(p, q)) < 0.3;
                        if on_edge(a, lb.bounds[0][si - 1]) && on_edge(b, lb.bounds[0][si]) {
                            continue;
                        }
                        if on_edge(a, lb.bounds[n][si - 1]) && on_edge(b, lb.bounds[n][si]) {
                            continue;
                        }
                        if norm(sub(a, b)) > 1e-6 {
                            out.push([a[0], a[1], b[0], b[1]]);
                        }
                    }
                }
                continue;
            }
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
    fn build_junctions_makes_a_signalized_crossing_first_class() {
        let net = map::arterial_intersection();
        assert_eq!(net.junctions.len(), 1, "a lone crossing is one junction");
        let j = &net.junctions[0];
        assert!(j.program.is_some(), "the signalized crossing owns its signal program");
        assert!(!j.approaches.is_empty() && !j.exits.is_empty(), "it has approaches and exits");
        for &nd in &j.nodes {
            assert_eq!(net.node_junction(nd), Some(JunctionId(0)), "member nodes map back to the junction");
        }
        let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
        for c in j.footprint {
            lo = [lo[0].min(c[0]), lo[1].min(c[1])];
            hi = [hi[0].max(c[0]), hi[1].max(c[1])];
        }
        assert!(hi[0] - lo[0] > 5.0 && hi[1] - lo[1] > 5.0, "footprint spans the crossing");
        assert!(
            (lo[0] - 1.0..=hi[0] + 1.0).contains(&j.center[0]) && (lo[1] - 1.0..=hi[1] + 1.0).contains(&j.center[1]),
            "the crossing centre sits inside its footprint",
        );
    }

    #[test]
    #[cfg(feature = "import")]
    fn multi_node_junctions_progress_through_movements_together() {
        // Coordination check: at a junction modelled as several nodes, a through path
        // that crosses one interior node and then the next must find both greens open
        // at the same time — otherwise a vehicle would stop between them, inside the
        // box. Verified purely from the signal schedule.
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
        let Ok(txt) = std::fs::read_to_string(path) else { return };
        let net = OsmMap::from_json(&txt).unwrap().build();
        let internal = |l: LinkId| {
            let a = net.node_junction(net.link(l).from);
            a.is_some() && a == net.node_junction(net.link(l).to)
        };
        let (mut checked, mut ok) = (0, 0);
        for li in (0..net.links.len() as u32).map(LinkId).filter(|&l| internal(l)) {
            let mut pair = None;
            for a in (0..net.movements.len() as u32).map(MovementId) {
                if net.lane(net.movement(a).to_lane).link != li || net.movement_turn(a) != TurnType::Through {
                    continue;
                }
                for b in (0..net.movements.len() as u32).map(MovementId) {
                    if net.movement(b).from_lane == net.movement(a).to_lane && net.movement_turn(b) == TurnType::Through {
                        pair = Some((a, b));
                    }
                }
            }
            let Some((m_in, m_out)) = pair else { continue };
            checked += 1;
            let mut t = 0.0;
            let mut found = false;
            while t < 200.0 {
                if net.movement_state(m_in, t) == SignalState::Green && net.movement_state(m_out, t) == SignalState::Green {
                    ok += 1;
                    found = true;
                    break;
                }
                t += 0.5;
            }
            if !found {
                let prog = |m: MovementId| net.movement(m).signal_group.map(|g| (net.groups[g.idx()].program.0, net.groups[g.idx()].bit));
                let j = net.node_junction(net.link(li).from).unwrap();
                eprintln!(
                    "no joint green: link {} in junction {} ({} nodes, {} approaches): in {:?} out {:?}",
                    li.0, j.0, net.junction(j).nodes.len(), net.junction(j).approaches.len(),
                    prog(m_in), prog(m_out),
                );
            }
        }
        eprintln!("internal through-paths: {ok}/{checked} reach simultaneous green");
        assert!(checked >= 3, "too few internal through-paths to test ({checked})");
        assert!(ok == checked, "{}/{checked} internal through-paths never get simultaneous green — no progression", checked - ok);
    }

    #[test]
    #[cfg(feature = "import")]
    fn junction_footprints_are_angled_to_the_dominant_road_axis() {
        // Static check on how each intersection rectangle is oriented: its long edge
        // must run along the junction's dominant road axis (the lane-weighted heading
        // of the arms that feed it), never skewed to a min-area diagonal that belongs
        // to no road. Recomputes the axis from the public arm headings and compares it
        // to the built footprint edge.
        use std::f64::consts::PI;
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
        let Ok(txt) = std::fs::read_to_string(path) else { return };
        let net = OsmMap::from_json(&txt).unwrap().build();
        let axis_deg = |d: [f64; 2]| d[1].atan2(d[0]).rem_euclid(PI).to_degrees();
        let sep = |a: f64, b: f64| {
            let d = (a - b).rem_euclid(180.0);
            d.min(180.0 - d)
        };
        let (mut checked, mut worst, mut worst_wide) = (0, 0.0f64, 0.0f64);
        for j in &net.junctions {
            let node_pos: Vec<[f64; 2]> = j.nodes.iter().map(|&nd| net.nodes[nd.idx()].position).collect();
            let near = |m: [f64; 2]| node_pos.iter().map(|&p| norm(sub(m, p))).fold(f64::MAX, f64::min);
            // Reconstruct the arm set build_junctions used, in its link-index order (so
            // dominant_axis breaks ties on the same arm the build did).
            let mut indexed: Vec<(u32, [f64; 2], f64)> = Vec::new();
            for &l in &j.approaches {
                if near(net.arm_mouth(l, true).0) < FOOTPRINT_RADIUS {
                    indexed.push((l.0, net.arrival_dir(l), net.link(l).lane_count as f64));
                }
            }
            for &l in &j.exits {
                if near(net.arm_mouth(l, false).0) < FOOTPRINT_RADIUS {
                    indexed.push((l.0, net.departure_dir(l), net.link(l).lane_count as f64));
                }
            }
            indexed.sort_by_key(|&(i, _, _)| i);
            let arms: Vec<([f64; 2], f64)> = indexed.iter().map(|&(_, d, w)| (d, w)).collect();
            if arms.len() < 2 {
                continue; // fallback box, not arm-derived
            }
            let edge = axis_deg(sub(j.footprint[1], j.footprint[0]));
            // Consistency: the built edge equals the dominant road axis it was oriented to.
            let dev = sep(edge, axis_deg(dominant_axis(&arms)));
            worst = worst.max(dev);
            assert!(dev < 0.5, "junction footprint edge {edge:.1}° off its dominant road axis (Δ{dev:.1}°)");
            // Method-independent anti-skew: the long edge runs along a real arm heading,
            // never a bare corner-to-corner diagonal (the min-area failure mode).
            let to_arm = arms.iter().map(|&(d, _)| sep(edge, axis_deg(d))).fold(f64::MAX, f64::min);
            worst_wide = worst_wide.max(to_arm);
            checked += 1;
            assert!(to_arm < 1.0, "footprint edge {edge:.1}° matches no approach heading (nearest Δ{to_arm:.1}°) — skewed");
        }
        assert!(checked >= 20, "exercised only {checked} arm-derived footprints");
        eprintln!("footprint angles: {checked} checked, worst axis Δ {worst:.3}°, worst edge-vs-arm Δ {worst_wide:.3}°");
    }

    #[test]
    #[cfg(feature = "import")]
    fn cross_junction_conflicts_join_sibling_nodes_and_are_indexed() {
        use super::super::junction::Junctions;
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
        let Ok(txt) = std::fs::read_to_string(path) else { return };
        let net = OsmMap::from_json(&txt).unwrap().build();
        let junctions = Junctions::build(&net);
        let mut cross = 0usize;
        for (idx, c) in net.conflicts.iter().enumerate() {
            let (na, nb) = (net.movement(c.a).node, net.movement(c.b).node);
            if na == nb {
                continue;
            }
            cross += 1;
            let (ja, jb) = (net.node_junction(na), net.node_junction(nb));
            assert!(ja.is_some() && ja == jb, "a cross-node conflict joins two nodes of one junction");
            assert!(net.movements_conflict(c.a, c.b), "the conflict is visible to the collision test");
            assert!(junctions.conflict_ids(na).contains(&(idx as u32)), "indexed from one sibling node");
            assert!(junctions.conflict_ids(nb).contains(&(idx as u32)), "and from the other");
        }
        assert!(cross > 0, "multi-node Millbrae junctions produce cross-node conflicts");
    }

    #[test]
    #[cfg(feature = "import")]
    fn junction_interiors_hug_their_corners_and_never_swap_lanes() {
        // Two geometric guarantees of the stage-3 remodel, on the real map:
        // a cornered turn's interior stays inside its tangent triangle (entry,
        // tangent crossing, exit) — it hugs the curb return instead of swinging
        // outside the box — and parallel movements between one link pair land in
        // lateral order, so side-by-side cars never swap lanes across each other
        // mid-box. (One floored degenerate corner and one sliver-link dual-left
        // are tolerated.)
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
        let Ok(txt) = std::fs::read_to_string(path) else { return };
        let net = OsmMap::from_json(&txt).unwrap().build();

        let (mut checked, mut bulged) = (0, 0);
        let mut deepest = 0.0f64;
        for m in 0..net.movements.len() as u32 {
            let mid = MovementId(m);
            let mv = net.movement(mid);
            if net.node_junction(mv.node).is_none() || net.movement_turn(mid) == TurnType::Through {
                continue;
            }
            let it = net.interior(mid);
            let (fl, tl) = (net.lane(mv.from_lane).link, net.lane(mv.to_lane).link);
            let (arr, dep) = (net.arrival_dir(fl), net.departure_dir(tl));
            let cross = arr[0] * dep[1] - arr[1] * dep[0];
            if cross.abs() < 0.05 {
                continue;
            }
            let d = sub(it.exit, it.entry);
            let t = (d[0] * dep[1] - d[1] * dep[0]) / cross;
            let s = (arr[0] * d[1] - arr[1] * d[0]) / cross;
            if t <= 0.0 || s <= 0.0 {
                continue;
            }
            checked += 1;
            let c = [it.entry[0] + arr[0] * t, it.entry[1] + arr[1] * t];
            let tri = [it.entry, c, it.exit];
            let mut worst = 0.0f64;
            for i in 0..=16 {
                let p = net.interior_point(mid, it.len * i as f64 / 16.0);
                let mut outside = f64::MIN;
                for k in 0..3 {
                    let (a, b) = (tri[k], tri[(k + 1) % 3]);
                    let e = sub(b, a);
                    let sd = ((p[0] - a[0]) * e[1] - (p[1] - a[1]) * e[0]) / norm(e).max(1e-9);
                    outside = outside.max(if cross > 0.0 { sd } else { -sd });
                }
                worst = worst.max(outside);
            }
            if worst > 0.5 {
                bulged += 1;
            }
            deepest = deepest.max(worst);
        }
        eprintln!("cornered turns: {checked} checked, {bulged} bulge > 0.5 m, deepest {deepest:.2} m");
        assert!(checked > 1000, "enough cornered turns to exercise ({checked})");
        // The 0.45·k tangency floor lets a degenerate corner poke out by a
        // sub-metre sliver; what the clamp must guarantee is that no turn takes
        // the multi-metre swing across neighbouring lanes it used to.
        assert!(bulged <= 8, "{bulged} turn interiors swing outside their tangent triangle");
        assert!(deepest < 1.2, "a turn interior bulges {deepest:.2} m outside its tangent triangle");

        let mut groups: std::collections::HashMap<(u32, u32, u32), Vec<(u32, u32)>> = Default::default();
        for mv in &net.movements {
            let key = (net.lane(mv.from_lane).link.0, net.lane(mv.to_lane).link.0, mv.node.0);
            groups.entry(key).or_default().push((mv.from_lane.0, mv.to_lane.0));
        }
        for (key, mut g) in groups {
            g.sort();
            for w in g.windows(2) {
                assert!(
                    w[1].1 >= w[0].1,
                    "movements {key:?} pair out of lateral order: {:?} then {:?}",
                    w[0],
                    w[1],
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "import")]
    fn el_camino_through_lanes_stay_laterally_continuous() {
        // The Millbrae regression this build's geometry passes exist for: El Camino
        // Real's dual carriageways are one-way OSM ways whose lane counts churn
        // 3→4→5→3 around intersections. Recentred axes + turn:lanes pocket skew
        // must keep a through movement's exit laterally near its entry at plain
        // lane-count seams, and the divided crossings must keep their split nodes
        // (one junction cluster, real crossing points) instead of centroid-merging.
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
        let Ok(txt) = std::fs::read_to_string(path) else { return };
        let net = OsmMap::from_json(&txt).unwrap().build();
        let ecr = |l: LinkId| net.link_names[l.idx()].contains("El Camino");
        let mut seam: Vec<f64> = Vec::new();
        for m in 0..net.movements.len() as u32 {
            let mid = MovementId(m);
            let mv = net.movement(mid);
            let (fl, tl) = (net.lane(mv.from_lane).link, net.lane(mv.to_lane).link);
            if !ecr(fl) || !ecr(tl) || net.movement_turn(mid) != TurnType::Through || net.node_junction(mv.node).is_some() {
                continue;
            }
            // A movement landing in a turn pocket isn't a through-lane
            // continuation: the bay is deliberately merged into its neighbour at
            // the seam (the car rides the through lane until the bay opens), so
            // its designed lane-width offset would drown the drift this metric
            // guards against.
            if net.lane(mv.to_lane).pocket_taper > 0.0 {
                continue;
            }
            let it = net.interior(mid);
            let d = net.arrival_dir(fl);
            seam.push(((it.exit[0] - it.entry[0]) * d[1] - (it.exit[1] - it.entry[1]) * d[0]).abs());
        }
        assert!(seam.len() >= 20, "enough plain ECR seams to measure ({})", seam.len());
        let mean = seam.iter().sum::<f64>() / seam.len() as f64;
        let max = seam.iter().fold(0.0f64, |a, &b| a.max(b));
        eprintln!("ECR seam through jogs: n={} mean={mean:.2} max={max:.2}", seam.len());
        assert!(mean < 0.5, "seam through-lanes drift {mean:.2} m on average — seam stitching regressed");
        assert!(max < 2.5, "worst seam through-jog {max:.2} m — a through lane jumps lanes at a seam");

        let split = net
            .junctions
            .iter()
            .filter(|j| j.nodes.len() >= 2 && j.approaches.iter().any(|&l| ecr(l)))
            .count();
        assert!(split >= 8, "ECR's divided crossings keep their split nodes, got {split} multi-node junctions");
    }

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
        assert!(start[1].abs() < 1e-9, "a one-way carriageway is centred on its mapped line");
    }

    #[test]
    fn lane_point_and_length_follow_a_curved_polyline() {
        // L-shaped link (0,0) → bend (100,0) → (100,100).
        let net = OsmMap {
            nodes: vec![NodeSpec::uncontrolled(1, 0.0, 0.0), NodeSpec::uncontrolled(2, 100.0, 100.0)],
            links: vec![LinkSpec { from_osm: 1, to_osm: 2, lanes: 1, speed_limit: 20.0, geometry: vec![[100.0, 0.0]], layer: 0, name: String::new(), road_class: String::new(), highway_ref: String::new(), turn_lanes: String::new(), hov_lanes: String::new(), aadt: 0.0, res_weight: 0.0, attr_weight: 0.0, sign: LinkSign::None }],
        }
        .build();
        let lane = LaneId(0);
        let l = *net.lane(lane);
        // The drivable span is the (recentred) polyline's arc minus the two setbacks.
        let full: f64 = net.polylines[0].windows(2).map(|w| (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1])).sum();
        assert!((l.start_offset + l.length - (full - l.start_offset)).abs() < 1.0, "span ends a setback short of the far node");
        let mid = net.lane_point(lane, full / 2.0 - l.start_offset);
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
        // At least one strip per link; the boundary chart subdivides a link with
        // a tapering pocket bay into several quads.
        assert!(net.road_strips().len() >= net.links.len());
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
        // The trimmed centreline must sit inside the polyline ends by the
        // setback, so road fill and markings never cross into the intersection.
        let net = map::corridor_with_signal();
        for i in 0..net.links.len() {
            let link = net.link(LinkId(i as u32));
            let poly = net.drivable_polyline(LinkId(i as u32));
            let axis = &net.polylines[i];
            let from = axis[0];
            let to = *axis.last().unwrap();
            let d0 = (poly[0][0] - from[0]).hypot(poly[0][1] - from[1]);
            let d1 = (poly[poly.len() - 1][0] - to[0]).hypot(poly[poly.len() - 1][1] - to[1]);
            // The drawn carriageway stops at the junction-box edge (render setback),
            // which is inside the stop line (that adds a crosswalk margin).
            assert!((d0 - net.render_setback[link.from.idx()]).abs() < 1e-6, "starts at the box edge");
            assert!(d1 > 0.5, "ends short of the downstream end: {d1}");
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
