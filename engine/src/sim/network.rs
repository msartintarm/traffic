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
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lane {
    pub link: LinkId,
    pub index_in_link: u32,
    pub length: f64,
    pub speed_limit: f64,
    pub movement_start: MovementId,
    pub movement_count: u32,
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

    /// World `[x, y, heading]` of a point `position` metres along `lane`,
    /// laterally offset for the lane's index. Pure geometry the renderer uses to
    /// place vehicle instances.
    pub fn lane_point(&self, lane: LaneId, position: f64) -> [f64; 3] {
        let l = self.lane(lane);
        let poly = &self.polylines[l.link.idx()];
        let (pt, dir) = point_along(poly, position.clamp(0.0, l.length));
        let off = (l.index_in_link as f64 + 0.5) * LANE_WIDTH;
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
        let mut acc = 0.0;
        let mut best = f64::INFINITY;
        for i in 1..poly.len() - 1 {
            let seg_in = norm(sub(poly[i], poly[i - 1]));
            acc += seg_in;
            if acc < position {
                continue;
            }
            if acc > position + lookahead {
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

    /// Filled carriageway quads `[cx0, cy0, cx1, cy1, width]`, one per polyline
    /// segment of each link (curved roads become several quads).
    pub fn road_strips(&self) -> Vec<[f64; 5]> {
        let mut out = Vec::new();
        for i in 0..self.links.len() {
            let w = self.links[i].lane_count as f64 * LANE_WIDTH;
            let c = w / 2.0;
            for seg in self.polylines[i].windows(2) {
                let dir = unit(sub(seg[1], seg[0]));
                let n = [dir[1] * c, -dir[0] * c];
                out.push([seg[0][0] + n[0], seg[0][1] + n[1], seg[1][0] + n[0], seg[1][1] + n[1], w]);
            }
        }
        out
    }

    /// Interior lane-divider segments `[x0, y0, x1, y1]`, per polyline segment.
    pub fn lane_dividers(&self) -> Vec<[f64; 4]> {
        let mut out = Vec::new();
        for i in 0..self.links.len() {
            let lanes = self.links[i].lane_count;
            for seg in self.polylines[i].windows(2) {
                let dir = unit(sub(seg[1], seg[0]));
                let (nx, ny) = (dir[1], -dir[0]);
                for k in 1..lanes {
                    let off = k as f64 * LANE_WIDTH;
                    out.push([seg[0][0] + nx * off, seg[0][1] + ny * off, seg[1][0] + nx * off, seg[1][1] + ny * off]);
                }
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
        let start = net.lane_point(lane, 0.0);
        let end = net.lane_point(lane, net.lane(lane).length);
        assert!((start[0] - 0.0).abs() < 1e-9 && (end[0] - 100.0).abs() < 1e-9);
        assert!((start[2]).abs() < 1e-9, "eastbound heading is 0 rad");
        assert!(start[1] < 0.0, "right-hand lane offset is negative-y for +x travel");
    }

    #[test]
    fn lane_point_and_length_follow_a_curved_polyline() {
        // L-shaped link (0,0) → bend (100,0) → (100,100).
        let net = OsmMap {
            nodes: vec![NodeSpec::uncontrolled(1, 0.0, 0.0), NodeSpec::uncontrolled(2, 100.0, 100.0)],
            links: vec![LinkSpec { from_osm: 1, to_osm: 2, lanes: 1, speed_limit: 20.0, geometry: vec![[100.0, 0.0]] }],
        }
        .build();
        let lane = LaneId(0);
        let len = net.lane(lane).length;
        assert!((len - 200.0).abs() < 1.0, "arc length ~200, got {len}");
        let mid = net.lane_point(lane, 100.0);
        assert!((mid[0] - 100.0).abs() < 5.0 && mid[1].abs() < 5.0, "midpoint near the bend: {mid:?}");
        assert!(net.min_radius_ahead(lane, 0.0, 200.0).is_finite(), "a bend has finite radius");
    }

    #[test]
    fn movement_turns_are_classified() {
        // Corridor: through link 1→2 (heading +x) crossing at node 2, with exits
        // east (2→4, straight) and north (2→5, a left turn).
        let net = map::corridor_with_signal();
        let through_lane = net.lanes_of(LinkId(0)).next().unwrap(); // link 1->2
        let mut turns = std::collections::HashSet::new();
        for k in 0..net.lane(through_lane).movement_count {
            let mid = MovementId(net.lane(through_lane).movement_start.0 + k);
            turns.insert(net.movement_turn(mid));
        }
        assert!(turns.contains(&TurnType::Through), "east exit is straight");
        assert!(turns.contains(&TurnType::Left), "north exit is a left turn");
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
        let dividers = net.lane_dividers().len();
        let expected: u32 = net.links.iter().map(|l| l.lane_count.saturating_sub(1)).sum();
        assert_eq!(dividers, expected as usize);
        let b = net.bounds();
        assert!(b[0] <= b[2] && b[1] <= b[3]);
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
