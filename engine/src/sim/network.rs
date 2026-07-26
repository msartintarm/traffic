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
        let link = self.link(l.link);
        let a = self.node(link.from).position;
        let b = self.node(link.to).position;
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let seg = dx.hypot(dy).max(1e-6);
        let t = (position / l.length.max(1e-6)).clamp(0.0, 1.0);
        let off = (l.index_in_link as f64 + 0.5) * LANE_WIDTH;
        let (nx, ny) = (dy / seg, -dx / seg);
        [a[0] + dx * t + nx * off, a[1] + dy * t + ny * off, dy.atan2(dx)]
    }

    /// Right-hand offset unit normal of a link's travel direction (the axis its
    /// lanes are stacked along).
    fn link_normal(&self, link: LinkId) -> ([f64; 2], [f64; 2], [f64; 2]) {
        let lk = self.link(link);
        let a = self.node(lk.from).position;
        let b = self.node(lk.to).position;
        let seg = (b[0] - a[0]).hypot(b[1] - a[1]).max(1e-6);
        (a, b, [(b[1] - a[1]) / seg, -(b[0] - a[0]) / seg])
    }

    /// Filled carriageway quad per link as `[cx0, cy0, cx1, cy1, width]`, where
    /// the centreline is offset to cover all of the link's right-hand lanes.
    pub fn road_strips(&self) -> Vec<[f64; 5]> {
        (0..self.links.len() as u32)
            .map(|i| {
                let link = LinkId(i);
                let (a, b, n) = self.link_normal(link);
                let w = self.link(link).lane_count as f64 * LANE_WIDTH;
                let c = w / 2.0;
                [a[0] + n[0] * c, a[1] + n[1] * c, b[0] + n[0] * c, b[1] + n[1] * c, w]
            })
            .collect()
    }

    /// Interior lane-divider segments as `[x0, y0, x1, y1]`, one per boundary
    /// between adjacent lanes of a link.
    pub fn lane_dividers(&self) -> Vec<[f64; 4]> {
        let mut out = Vec::new();
        for i in 0..self.links.len() as u32 {
            let link = LinkId(i);
            let (a, b, n) = self.link_normal(link);
            for k in 1..self.link(link).lane_count {
                let off = k as f64 * LANE_WIDTH;
                out.push([a[0] + n[0] * off, a[1] + n[1] * off, b[0] + n[0] * off, b[1] + n[1] * off]);
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
