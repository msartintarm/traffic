//! OSM-facing import schema and the builder that compiles it into a runtime
//! [`Network`]. The scraper tool (`../tools/osm-scraper`) emits data shaped
//! exactly like [`OsmMap`] — nodes carrying projected coordinates and an
//! intersection control, plus already-directed links carrying lane count and
//! speed limit — so importing a real Millbrae extract is `OsmMap { .. }.build()`.
//!
//! The builder resolves signals into [`SignalGroup`]s (one per signalized
//! approach), and connects each incoming lane to the lanes of every onward link
//! at its downstream node, skipping U-turns. Turn restrictions and lane-level
//! `turn:lanes` from OSM prune these movements in a later pass.

use std::collections::{BTreeSet, HashMap};

use super::network::*;
use super::signal::{Phase, SignalProgram};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SignalPlan {
    pub green_secs: f64,
    pub yellow_secs: f64,
    pub offset: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MapControl {
    Uncontrolled,
    Stop,
    Yield,
    Signal(SignalPlan),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeSpec {
    pub osm_id: i64,
    pub x: f64,
    pub y: f64,
    pub control: MapControl,
}

impl NodeSpec {
    pub fn uncontrolled(osm_id: i64, x: f64, y: f64) -> Self {
        Self { osm_id, x, y, control: MapControl::Uncontrolled }
    }

    pub fn signalized(osm_id: i64, x: f64, y: f64, plan: SignalPlan) -> Self {
        Self { osm_id, x, y, control: MapControl::Signal(plan) }
    }

    pub fn stop(osm_id: i64, x: f64, y: f64) -> Self {
        Self { osm_id, x, y, control: MapControl::Stop }
    }

    pub fn give_way(osm_id: i64, x: f64, y: f64) -> Self {
        Self { osm_id, x, y, control: MapControl::Yield }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LinkSpec {
    pub from_osm: i64,
    pub to_osm: i64,
    pub lanes: u32,
    pub speed_limit: f64,
    /// Intermediate bend points (projected metres) between the from- and to-node;
    /// empty for a straight link.
    pub geometry: Vec<[f64; 2]>,
    /// Grade-separation level (OSM `layer`): 0 at grade, >0 a bridge/overpass,
    /// <0 a tunnel. Used only for render z-order — crossings at different layers
    /// share no node, so they already form no intersection.
    pub layer: i32,
    /// OSM road name, carried through the topology transforms so the browser can
    /// label the engine's own links (which no longer match the raw import).
    pub name: String,
    /// OSM `highway` class (e.g. "motorway", "motorway_link", "residential"),
    /// distilled into [`RoadKind`] at build; drives free-flow ramp interchanges.
    pub road_class: String,
    /// OSM `ref` — the route designation of a numbered road (e.g. "US 101",
    /// "I 280;CA 35"). Empty for unnumbered streets. Carried to the network so
    /// demand can send freeway through-traffic to the far end of the *same* highway.
    pub highway_ref: String,
    /// OSM `turn:lanes` (for this travel direction): a `|`-separated list, one
    /// entry per lane from the median outward, each a `;`-separated set of turns
    /// (e.g. `"left|through|through;right"`). Empty when unmapped — the renderer
    /// then falls back to arrows derived from the lane's actual movements.
    pub turn_lanes: String,
}

impl LinkSpec {
    pub fn oneway(from_osm: i64, to_osm: i64, lanes: u32, speed_limit: f64) -> Self {
        Self { from_osm, to_osm, lanes, speed_limit, ..Default::default() }
    }

    pub fn twoway(a: i64, b: i64, lanes: u32, speed_limit: f64) -> [Self; 2] {
        [Self::oneway(a, b, lanes, speed_limit), Self::oneway(b, a, lanes, speed_limit)]
    }
}

#[derive(Clone, Debug, Default)]
pub struct OsmMap {
    pub nodes: Vec<NodeSpec>,
    pub links: Vec<LinkSpec>,
}

impl OsmMap {
    /// Dissolve uncontrolled degree-2 pass-through nodes (where one road simply
    /// continues) into the adjacent link's polyline, so a road between two real
    /// junctions is a single link with bends rather than a chain of stub links and
    /// spurious junction boxes. Only merges segments sharing lanes, speed and
    /// layer; controlled nodes and attribute changes are left intact.
    pub fn collapse_pass_through_nodes(&self) -> OsmMap {
        let pos: HashMap<i64, [f64; 2]> = self.nodes.iter().map(|n| (n.osm_id, [n.x, n.y])).collect();
        let mut nodes = self.nodes.clone();
        let mut links = self.links.clone();
        while let Some((node, mut remove, add)) = next_collapse(&nodes, &links, &pos) {
            remove.sort_unstable();
            for &i in remove.iter().rev() {
                links.remove(i);
            }
            links.extend(add);
            nodes.retain(|n| n.osm_id != node);
        }
        OsmMap { nodes, links }
    }

    /// Merge intersections that OSM splits across several nodes a few metres apart
    /// (divided roads, staggered crossings) into one logical junction: cluster
    /// junction nodes joined by a short stub, collapse each cluster to its centroid,
    /// drop the now-internal stubs, and re-point the external approaches. `build`
    /// then forms a single box with one coordinated signal instead of two.
    pub fn merge_split_intersections(&self) -> OsmMap {
        const STUB_MAX: f64 = 25.0;
        // A link this short is junction interior (a turn slot, median crossing or
        // lane-change fragment), so merge it into the junction whatever its
        // endpoints' degree or lane count — otherwise it renders as a stray nub.
        const INTERIOR_MAX: f64 = 12.0;
        let pos: HashMap<i64, [f64; 2]> = self.nodes.iter().map(|n| (n.osm_id, [n.x, n.y])).collect();
        let mut neigh: HashMap<i64, BTreeSet<i64>> = HashMap::new();
        for l in &self.links {
            neigh.entry(l.from_osm).or_default().insert(l.to_osm);
            neigh.entry(l.to_osm).or_default().insert(l.from_osm);
        }
        let degree = |id: i64| neigh.get(&id).map_or(0, BTreeSet::len);

        let mut parent: HashMap<i64, i64> = self.nodes.iter().map(|n| (n.osm_id, n.osm_id)).collect();
        for l in &self.links {
            let d = distance(pos[&l.from_osm], pos[&l.to_osm]);
            let junctions = degree(l.from_osm) >= 3 && degree(l.to_osm) >= 3;
            if d < INTERIOR_MAX || (d < STUB_MAX && junctions) {
                let (ra, rb) = (uf_find(&mut parent, l.from_osm), uf_find(&mut parent, l.to_osm));
                if ra != rb {
                    parent.insert(ra, rb);
                }
            }
        }

        let mut clusters: HashMap<i64, Vec<i64>> = HashMap::new();
        for n in &self.nodes {
            let r = uf_find(&mut parent, n.osm_id);
            clusters.entry(r).or_default().push(n.osm_id);
        }

        let mut rep_of: HashMap<i64, i64> = HashMap::new();
        let mut nodes: Vec<NodeSpec> = Vec::new();
        for members in clusters.values() {
            let rep_id = *members.iter().min().unwrap();
            let (mut cx, mut cy) = (0.0, 0.0);
            let mut control = MapControl::Uncontrolled;
            for &m in members {
                rep_of.insert(m, rep_id);
                cx += pos[&m][0];
                cy += pos[&m][1];
                let c = self.nodes.iter().find(|n| n.osm_id == m).unwrap().control;
                if control_rank(c) > control_rank(control) {
                    control = c;
                }
            }
            let k = members.len() as f64;
            nodes.push(NodeSpec { osm_id: rep_id, x: cx / k, y: cy / k, control });
        }

        let mut links = Vec::new();
        for l in &self.links {
            let (from_osm, to_osm) = (rep_of[&l.from_osm], rep_of[&l.to_osm]);
            if from_osm != to_osm {
                links.push(LinkSpec { from_osm, to_osm, ..l.clone() });
            }
        }
        OsmMap { nodes, links }
    }

    pub fn build(&self) -> Network {
        let mut net = Network::default();
        let mut id_of: HashMap<i64, NodeId> = HashMap::new();

        for spec in &self.nodes {
            id_of.insert(spec.osm_id, NodeId(net.nodes.len() as u32));
            net.nodes.push(Node {
                position: [spec.x, spec.y],
                control: NodeControl::Uncontrolled,
            });
        }

        for spec in &self.links {
            let from = id_of[&spec.from_osm];
            let to = id_of[&spec.to_osm];
            let mut polyline = vec![net.nodes[from.idx()].position];
            polyline.extend(spec.geometry.iter().copied());
            polyline.push(net.nodes[to.idx()].position);
            let length = polyline.windows(2).map(|w| distance(w[0], w[1])).sum();
            let link_id = LinkId(net.links.len() as u32);
            let lane_start = LaneId(net.lanes.len() as u32);
            for i in 0..spec.lanes {
                net.lanes.push(Lane {
                    link: link_id,
                    index_in_link: i,
                    length,
                    start_offset: 0.0,
                    speed_limit: spec.speed_limit,
                    movement_start: MovementId(0),
                    movement_count: 0,
                    pocket_taper: 0.0,
                });
            }
            net.links.push(Link {
                from,
                to,
                lane_start,
                lane_count: spec.lanes,
                layer: spec.layer,
                kind: RoadKind::from_osm(&spec.road_class),
            });
            net.polylines.push(polyline);
            net.link_names.push(spec.name.clone());
            net.link_refs.push(spec.highway_ref.clone());
            net.link_turn_lanes.push(spec.turn_lanes.clone());
        }

        offset_ramps_to_curb(&mut net);
        set_junction_setbacks(&mut net);

        // Lane-channelised movements (California lane-use convention): rather than
        // wire every approach lane to every exit — which at a big merged junction
        // gives one lane six turns whose paths all fan out from the same point — we
        // sort each approach's exits by turn angle and give each lane a contiguous
        // *angular* slice, so the left lane serves left turns and the right lane
        // serves rights. `lane_point` places lane 0 adjacent to the centreline (the
        // left lane) with higher indices toward the curb, so exits are ordered
        // left-to-right and mapped in that same order onto lanes 0…n-1.
        let mut movements: Vec<Movement> = Vec::new();
        for in_li in 0..net.links.len() {
            let link = net.links[in_li];
            let node = link.to;
            let arr = net.arrival_dir(LinkId(in_li as u32));
            // Valid onward links (skip U-turns by node identity and by geometry),
            // as `(link, signed turn angle)`; ascending = rightmost turn first.
            let mut onward: Vec<(usize, f64)> = Vec::new();
            for out_li in 0..net.links.len() {
                let out = net.links[out_li];
                if out.from != node || out.to == link.from {
                    continue;
                }
                let dep = net.departure_dir(LinkId(out_li as u32));
                let dot = arr[0] * dep[0] + arr[1] * dep[1];
                if dot < -0.85 {
                    continue; // ~>148°: a near-reversal onto the opposing carriageway
                }
                onward.push((out_li, (arr[0] * dep[1] - arr[1] * dep[0]).atan2(dot)));
            }
            onward.sort_by(|a, b| b.1.total_cmp(&a.1)); // leftmost turn first → lane 0

            let (n, m) = (link.lane_count as usize, onward.len());
            // Map a position in [0, from] to the nearest in [0, to].
            let nearest = |x: usize, from: usize, to: usize| -> usize {
                if from == 0 { 0 } else { ((x as f64) * (to as f64) / (from as f64)).round() as usize }
            };
            // A freeway interchange (this approach is grade-separated and at least
            // one exit is a ramp) wires by side, not by angular slice: US ramps are
            // on the right (the highest lane index). The mainline keeps every lane;
            // an off-ramp hangs off the *curb* lane, which can still continue (a car
            // in it merges left rather than being forced to exit), so a car never
            // has to divert across the mainline to reach a ramp on the far side.
            let ramp_exit = |i: usize| net.links[onward[i].0].kind == RoadKind::Ramp;
            let is_interchange = link.kind.is_grade_separated() && (0..m).any(ramp_exit);
            let has_mainline = (0..m).any(|i| !ramp_exit(i));
            let mut lane_exits: Vec<std::collections::BTreeSet<usize>> = vec![Default::default(); n];
            if is_interchange && has_mainline {
                let mains: Vec<usize> = (0..m).filter(|&i| !ramp_exit(i)).collect();
                let mm = mains.len();
                // Mainline continuation(s) keep every lane (a freeway split still
                // fans left→right among the continuations).
                for (j, &i) in mains.iter().enumerate() {
                    lane_exits[nearest(j, mm - 1, n - 1)].insert(i);
                }
                for (k, exits) in lane_exits.iter_mut().enumerate() {
                    exits.insert(mains[nearest(k, n - 1, mm - 1)]);
                }
                // Off-ramps hang off the curb lanes: a k-lane ramp takes the curb-most
                // k freeway lanes, so a multi-lane exit is fed at full width instead of
                // being funnelled through one lane and backing up onto the mainline. The
                // curb lanes still carry the mainline too (option lanes, not exit-only).
                for i in 0..m {
                    if ramp_exit(i) {
                        let rl = net.links[onward[i].0].lane_count as usize;
                        for k in 0..rl.min(n) {
                            lane_exits[n - 1 - k].insert(i);
                        }
                    }
                }
            } else if m > 0 {
                for i in 0..m {
                    lane_exits[nearest(i, m - 1, n - 1)].insert(i); // every exit served
                }
                for (k, exits) in lane_exits.iter_mut().enumerate() {
                    exits.insert(nearest(k, n - 1, m - 1)); // every lane serves its nearest
                }
            }
            for (k, exits) in lane_exits.iter().enumerate() {
                let lane_id = link.lane_start.0 + k as u32;
                let start = MovementId(movements.len() as u32);
                for &exit_i in exits {
                    let out = net.links[onward[exit_i].0];
                    // An on-ramp merges onto the freeway's curb (rightmost) lane, not
                    // the same index it left (which would land it on the median). A
                    // multi-lane off-ramp maps the curb-most freeway lanes parallel onto
                    // the ramp's lanes (outermost-to-outermost), so every ramp lane is
                    // fed and none is a dead lane — and no path crosses the mainline.
                    let to_index = if link.kind == RoadKind::Ramp && out.kind == RoadKind::Freeway {
                        out.lane_count - 1
                    } else if link.kind == RoadKind::Freeway && out.kind == RoadKind::Ramp {
                        let from_curb = (link.lane_count - 1).saturating_sub(k as u32); // 0 at the curb lane
                        (out.lane_count - 1).saturating_sub(from_curb)
                    } else {
                        (k as u32).min(out.lane_count - 1)
                    };
                    movements.push(Movement {
                        from_lane: LaneId(lane_id),
                        to_lane: LaneId(out.lane_start.0 + to_index),
                        node,
                        signal_group: None,
                    });
                }
                net.lanes[lane_id as usize].movement_start = start;
                net.lanes[lane_id as usize].movement_count = movements.len() as u32 - start.0;
            }
        }
        net.movements = movements;
        assign_turn_pockets(&mut net);
        net.build_interiors();

        let plans = relocate_signals_to_junctions(&net, &self.nodes);
        for i in 0..self.nodes.len() {
            let control = match plans[i] {
                Some(plan) => assign_signal_program(&mut net, NodeId(i as u32), plan),
                None => match self.nodes[i].control {
                    MapControl::Stop => NodeControl::Stop,
                    MapControl::Yield => NodeControl::Yield,
                    _ => NodeControl::Uncontrolled,
                },
            };
            net.nodes[i].control = control;
        }
        coordinate_green_waves(&mut net);
        net
    }
}

/// Resolve where each signal actually controls. OSM commonly maps
/// `highway=traffic_signals` at each approach's stop line (a pass-through node
/// with one in and one out link), not at the junction centre — which would leave
/// the crossing itself uncontrolled and the stop-line "signal" stuck permanently
/// green. This walks each such pass-through signal one hop to the junction it
/// feeds and moves the signal there, so a real intersection collects all its
/// approaches into one program that actually cycles. Returns the effective
/// `SignalPlan` per node (`None` = not signalized).
fn relocate_signals_to_junctions(net: &Network, specs: &[NodeSpec]) -> Vec<Option<SignalPlan>> {
    let n = net.nodes.len();
    let (mut indeg, mut outdeg) = (vec![0u32; n], vec![0u32; n]);
    for link in &net.links {
        outdeg[link.from.idx()] += 1;
        indeg[link.to.idx()] += 1;
    }
    let is_junction = |node: usize| indeg[node] + outdeg[node] >= 3;
    let mut plans: Vec<Option<SignalPlan>> = specs
        .iter()
        .map(|s| if let MapControl::Signal(p) = s.control { Some(p) } else { None })
        .collect();
    for i in 0..n {
        let Some(plan) = plans[i] else { continue };
        // A pass-through signal (one approach in, one out) that isn't itself a
        // junction: hand its signal to the adjacent junction it protects.
        if is_junction(i) || indeg[i] != 1 || outdeg[i] != 1 {
            continue;
        }
        let downstream = net.links.iter().find(|l| l.from == NodeId(i as u32)).map(|l| l.to.idx());
        let upstream = net.links.iter().find(|l| l.to == NodeId(i as u32)).map(|l| l.from.idx());
        if let Some(j) = downstream.filter(|&j| is_junction(j)).or(upstream.filter(|&j| is_junction(j))) {
            plans[j].get_or_insert(plan);
            plans[i] = None;
        }
    }
    plans
}

/// Flag dedicated turn lanes as physical turn pockets. A lane qualifies when it
/// serves only one turn direction (left on the median-side lane 0, right on the
/// outermost lane), its neighbour is a through lane it can peel away from, and the
/// approach is long enough to hold a bay. The lane then gets a bay taper, so
/// [`Network::lane_lateral_offset`] opens the pocket near the stop line and merges
/// it into the through lane upstream — turners queue in the bay, not the through
/// lane, and the widened approach renders like a real intersection.
fn assign_turn_pockets(net: &mut Network) {
    const MIN_LEN: f64 = 45.0;
    let turns = |lane_id: u32| -> Vec<TurnType> {
        let lane = net.lanes[lane_id as usize];
        (0..lane.movement_count).map(|k| net.movement_turn(MovementId(lane.movement_start.0 + k))).collect()
    };
    let mut pockets: Vec<u32> = Vec::new();
    for li in 0..net.links.len() {
        let link = net.links[li];
        if link.lane_count < 2 {
            continue;
        }
        for (lane_idx, want) in [(0u32, TurnType::Left), (link.lane_count - 1, TurnType::Right)] {
            let lane_id = link.lane_start.0 + lane_idx;
            let ts = turns(lane_id);
            if net.lanes[lane_id as usize].length < MIN_LEN || ts.is_empty() || ts.iter().any(|&t| t != want) {
                continue;
            }
            let nb_idx = if lane_idx == 0 { 1 } else { link.lane_count - 2 };
            if !turns(link.lane_start.0 + nb_idx).contains(&TurnType::Through) {
                continue; // the bay must peel off a through lane, not another pocket
            }
            pockets.push(lane_id);
        }
    }
    for lane_id in pockets {
        net.lanes[lane_id as usize].pocket_taper = super::network::POCKET_TAPER;
    }
}

/// Build a signalized node's phase program from the conflict graph: each approach
/// contributes a through/right group and (if it has lefts) a protected-left group,
/// and groups that never conflict are served together. Because the phases come
/// from the same conflict data the collision model uses, the signal can never
/// green two movements that would crash — opposing throughs pair up, and a left
/// that crosses opposing traffic gets its own protected phase.
fn assign_signal_program(net: &mut Network, node: NodeId, plan: SignalPlan) -> NodeControl {
    // Group movements by (approach link, is-left); non-left groups first so the
    // greedy assignment pairs opposing throughs before protecting lefts.
    let mut group_movements: Vec<Vec<MovementId>> = Vec::new();
    let mut left_group: Vec<bool> = Vec::new();
    let mut key_index: HashMap<(u32, bool), usize> = HashMap::new();
    for want_left in [false, true] {
        for m in 0..net.movements.len() {
            let mv = net.movements[m];
            if mv.node != node {
                continue;
            }
            let is_left = net.movement_turn(MovementId(m as u32)) == TurnType::Left;
            if is_left != want_left {
                continue;
            }
            let key = (net.lane(mv.from_lane).link.0, is_left);
            let idx = *key_index.entry(key).or_insert_with(|| {
                group_movements.push(Vec::new());
                left_group.push(is_left);
                group_movements.len() - 1
            });
            group_movements[idx].push(MovementId(m as u32));
        }
    }
    if group_movements.is_empty() {
        return NodeControl::Uncontrolled;
    }

    let conflicts = |a: usize, b: usize| {
        group_movements[a]
            .iter()
            .any(|&x| group_movements[b].iter().any(|&y| net.movements_conflict(x, y)))
    };
    let mut phase_groups: Vec<Vec<usize>> = Vec::new();
    for g in 0..group_movements.len() {
        match phase_groups.iter_mut().find(|ph| ph.iter().all(|&h| !conflicts(g, h))) {
            Some(ph) => ph.push(g),
            None => phase_groups.push(vec![g]),
        }
    }

    let program = ProgramId(net.programs.len() as u32);
    let group_ids: Vec<SignalGroupId> = (0..group_movements.len())
        .map(|g| {
            let id = SignalGroupId(net.groups.len() as u32);
            net.groups.push(SignalGroup { program, bit: g as u8 });
            id
        })
        .collect();
    let phases = phase_groups
        .iter()
        .map(|gs| {
            let mask = gs.iter().fold(0u64, |m, &g| m | (1u64 << g));
            let green = if gs.iter().all(|&g| left_group[g]) {
                (plan.green_secs * 0.45).max(6.0)
            } else {
                plan.green_secs
            };
            let phase_movements: Vec<MovementId> =
                gs.iter().flat_map(|&g| group_movements[g].iter().copied()).collect();
            let (yellow, all_red) = change_and_clearance_intervals(net, &phase_movements);
            Phase::with_clearance(mask, green, yellow, all_red)
        })
        .collect();
    net.programs.push(SignalProgram::new(plan.offset, phases));
    for (g, mids) in group_movements.iter().enumerate() {
        for &mid in mids {
            net.movements[mid.idx()].signal_group = Some(group_ids[g]);
        }
    }
    NodeControl::Signalized(program)
}

const YELLOW_REACTION: f64 = 1.0;
const YELLOW_DECEL: f64 = 3.0;
const YELLOW_MIN: f64 = 3.0;
const YELLOW_MAX: f64 = 6.0;
const CLEARANCE_VEHICLE_LEN: f64 = 6.0;
const ALL_RED_MIN: f64 = 1.5;
const ALL_RED_MAX: f64 = 5.0;

fn change_and_clearance_intervals(net: &Network, movements: &[MovementId]) -> (f64, f64) {
    let approach = movements
        .iter()
        .map(|&m| net.lane(net.movement(m).from_lane).speed_limit)
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let yellow = (YELLOW_REACTION + approach / (2.0 * YELLOW_DECEL)).clamp(YELLOW_MIN, YELLOW_MAX);
    let crossing = movements
        .iter()
        .map(|&m| net.interior(m).len)
        .fold(0.0_f64, f64::max);
    let all_red = ((crossing + CLEARANCE_VEHICLE_LEN) / approach).clamp(ALL_RED_MIN, ALL_RED_MAX);
    (yellow, all_red)
}

/// Coordinate signalized arterial corridors into green waves. OSM carries no signal
/// timing, so every imported signal starts at `offset = 0` (all synchronized — the
/// worst case for progression). This walks each named arterial between consecutive
/// signals and offsets each program so its **through** phase opens just as a platoon
/// travelling the corridor at road speed arrives. Best-effort along one direction per
/// corridor; a mismatched cycle only degrades the coordination — the conflict-built
/// phases (and thus safety) are never touched.
fn coordinate_green_waves(net: &mut Network) {
    use std::collections::{BTreeMap, BTreeSet, VecDeque};
    const SPEED_FLOOR: f64 = 5.0; // m/s, so a slow arterial still gets a sane travel time
    const MAX_HOPS: usize = 24; // guard against a road that loops back on itself
    const MIN_ALIGN: f64 = 0.3; // a continuation link must head roughly the same way

    // Compute every offset while only *reading* the network, then apply them — so the
    // read-only walk/lookup closures don't clash with mutating `net.programs`.
    let offsets: Vec<(usize, f64)> = {
        let mut sig: BTreeMap<u32, ProgramId> = BTreeMap::new();
        for (i, n) in net.nodes.iter().enumerate() {
            if let NodeControl::Signalized(p) = n.control {
                sig.insert(i as u32, p);
            }
        }
        if sig.len() < 2 {
            return;
        }
        let mut out_links: Vec<Vec<u32>> = vec![Vec::new(); net.nodes.len()];
        for i in 0..net.links.len() {
            let l = net.link(LinkId(i as u32));
            if l.layer == 0 {
                out_links[l.from.idx()].push(i as u32);
            }
        }
        let link_len = |i: u32| net.polylines[i as usize].windows(2).map(|w| distance(w[0], w[1])).sum::<f64>();

        // Walk the same-named arterial from `start` (a link leaving a signal) to the
        // next signal; returns (that node, travel time, the link arriving there).
        let walk = |start: u32| -> Option<(u32, f64, u32)> {
            let name = &net.link_names[start as usize];
            if name.is_empty() {
                return None;
            }
            let mut cur = start;
            let mut cum = 0.0;
            for _ in 0..MAX_HOPS {
                let l = net.link(LinkId(cur));
                cum += link_len(cur) / net.lane(l.lane_start).speed_limit.max(SPEED_FLOOR);
                if sig.contains_key(&l.to.0) {
                    return Some((l.to.0, cum, cur));
                }
                let arr = net.arrival_dir(LinkId(cur));
                let mut best: Option<(u32, f64)> = None;
                for &nx in &out_links[l.to.idx()] {
                    if &net.link_names[nx as usize] != name || net.link(LinkId(nx)).to == l.from {
                        continue; // different road, or turning straight back
                    }
                    let dep = net.departure_dir(LinkId(nx));
                    let align = arr[0] * dep[0] + arr[1] * dep[1];
                    if align > MIN_ALIGN && best.map_or(true, |(_, a)| align > a) {
                        best = Some((nx, align));
                    }
                }
                cur = best?.0;
            }
            None
        };

        // Green-start time (into the cycle) of the corridor's through phase at `node`,
        // found via a straight-through movement along `corridor_link`.
        let through_start = |node: u32, corridor_link: u32| -> Option<f64> {
            let prog = &net.programs[sig.get(&node)?.idx()];
            for m in 0..net.movements.len() {
                let mv = net.movements[m];
                if mv.node.0 != node || net.movement_turn(MovementId(m as u32)) != TurnType::Through {
                    continue;
                }
                if net.lane(mv.from_lane).link.0 != corridor_link && net.lane(mv.to_lane).link.0 != corridor_link {
                    continue;
                }
                let bit = net.groups[mv.signal_group?.idx()].bit;
                let mut acc = 0.0;
                for ph in &prog.phases {
                    if ph.green_mask & (1u64 << bit) != 0 {
                        return Some(acc);
                    }
                    acc += ph.length();
                }
            }
            None
        };

        // Corridor adjacency: node → [(neighbour, travel time, link leaving here, link arriving there)].
        let mut adj: BTreeMap<u32, Vec<(u32, f64, u32, u32)>> = BTreeMap::new();
        for &s in sig.keys() {
            for &start in &out_links[s as usize] {
                if let Some((to, time, arriving)) = walk(start) {
                    adj.entry(s).or_default().push((to, time, start, arriving));
                }
            }
        }

        // BFS each corridor from its lowest-id signal (deterministic); cumulative
        // travel time sets each signal's offset so the through green opens on arrival.
        let mut out = Vec::new();
        let mut visited: BTreeSet<u32> = BTreeSet::new();
        for &root in sig.keys() {
            if !visited.insert(root) {
                continue;
            }
            let root_link = adj.get(&root).and_then(|e| e.first()).map(|&(_, _, start, _)| start);
            let mut members: Vec<(u32, f64, Option<u32>)> = Vec::new();
            let mut queue: VecDeque<(u32, f64, Option<u32>)> = VecDeque::from([(root, 0.0, root_link)]);
            while let Some(item) = queue.pop_front() {
                members.push(item);
                if let Some(neigh) = adj.get(&item.0) {
                    let mut ns = neigh.clone();
                    ns.sort_by_key(|&(m, ..)| m);
                    for (m, time, _, arriving) in ns {
                        if visited.insert(m) {
                            queue.push_back((m, item.1 + time, Some(arriving)));
                        }
                    }
                }
            }
            if members.len() < 2 {
                continue; // an isolated signal isn't a corridor
            }
            for (n, cum, link) in members {
                let pid = sig[&n];
                let cycle = net.programs[pid.idx()].cycle_length();
                if let (Some(link), true) = (link, cycle > 1.0) {
                    if let Some(sp) = through_start(n, link) {
                        out.push((pid.idx(), (sp - cum).rem_euclid(cycle)));
                    }
                }
            }
        }
        out
    };

    for (idx, offset) in offsets {
        net.programs[idx].offset = offset;
        net.programs[idx].coordinated = true;
    }
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn uf_find(parent: &mut HashMap<i64, i64>, x: i64) -> i64 {
    let p = parent[&x];
    if p == x {
        return x;
    }
    let root = uf_find(parent, p);
    parent.insert(x, root);
    root
}

/// Control precedence when merging nodes: a signal outranks a stop, a stop a
/// yield, so a merged junction keeps its strongest control.
fn control_rank(c: MapControl) -> u8 {
    match c {
        MapControl::Signal(_) => 3,
        MapControl::Stop => 2,
        MapControl::Yield => 1,
        MapControl::Uncontrolled => 0,
    }
}

/// Find one uncontrolled pass-through node to dissolve: returns its osm id, the
/// indices of the links to remove, and the merged link(s) to add. `None` when no
/// node is collapsible (the fixpoint).
fn next_collapse(
    nodes: &[NodeSpec],
    links: &[LinkSpec],
    pos: &HashMap<i64, [f64; 2]>,
) -> Option<(i64, Vec<usize>, Vec<LinkSpec>)> {
    for node in nodes.iter().filter(|n| n.control == MapControl::Uncontrolled) {
        let n = node.osm_id;
        let incident: Vec<usize> = links
            .iter()
            .enumerate()
            .filter(|(_, l)| l.from_osm == n || l.to_osm == n)
            .map(|(i, _)| i)
            .collect();
        let neigh: BTreeSet<i64> = incident
            .iter()
            .map(|&i| if links[i].from_osm == n { links[i].to_osm } else { links[i].from_osm })
            .collect();
        if neigh.len() != 2 {
            continue;
        }
        let mut nb = neigh.iter().copied();
        let (a, b) = (nb.next().unwrap(), nb.next().unwrap());
        let find = |from: i64, to: i64| links.iter().position(|l| l.from_osm == from && l.to_osm == to);
        let (a_in, a_out, b_in, b_out) = (find(a, n), find(n, a), find(b, n), find(n, b));

        let link_len = |l: &LinkSpec| -> f64 {
            let mut pts = vec![pos[&l.from_osm]];
            pts.extend(l.geometry.iter().copied());
            pts.push(pos[&l.to_osm]);
            pts.windows(2).map(|w| distance(w[0], w[1])).sum()
        };
        let joined = |s1: usize, s2: usize, from: i64, to: i64| -> Option<LinkSpec> {
            let (l1, l2) = (&links[s1], &links[s2]);
            if l1.speed_limit != l2.speed_limit || l1.layer != l2.layer {
                return None;
            }
            // Lane counts must match, except when one side is a grade-separated ramp
            // sliver too short to hold a vehicle — an OSM lane-count fragment on a ramp
            // approach (e.g. a 16 m 3-lane nub before a surface junction). Absorb it into
            // the substantive segment, adopting that segment's lane count, so the ramp
            // isn't chopped into micro-links that thrash the node-crossing logic.
            const SLIVER_MAX: f64 = 30.0;
            let lanes = if l1.lanes == l2.lanes {
                l1.lanes
            } else {
                let ramp = |l: &LinkSpec| RoadKind::from_osm(&l.road_class).is_grade_separated();
                let (len1, len2) = (link_len(l1), link_len(l2));
                if !(ramp(l1) && ramp(l2)) || len1.min(len2) >= SLIVER_MAX {
                    return None;
                }
                if len1 >= len2 { l1.lanes } else { l2.lanes }
            };
            let mut geometry = l1.geometry.clone();
            geometry.push(pos[&n]);
            geometry.extend(l2.geometry.iter().copied());
            let name = if l1.name.is_empty() { l2.name.clone() } else { l1.name.clone() };
            let road_class = if l1.road_class.is_empty() { l2.road_class.clone() } else { l1.road_class.clone() };
            let highway_ref = if l1.highway_ref.is_empty() { l2.highway_ref.clone() } else { l1.highway_ref.clone() };
            // The downstream segment (l2, ending at the merge's `to`) carries the
            // turn:lanes that matter at the stop line; fall back to l1 if it lacks them.
            let turn_lanes = if l2.turn_lanes.is_empty() { l1.turn_lanes.clone() } else { l2.turn_lanes.clone() };
            Some(LinkSpec { from_osm: from, to_osm: to, lanes, speed_limit: l1.speed_limit, geometry, layer: l1.layer, name, road_class, highway_ref, turn_lanes })
        };

        if incident.len() == 4 {
            if let (Some(ai), Some(ao), Some(bi), Some(bo)) = (a_in, a_out, b_in, b_out) {
                if let (Some(fwd), Some(rev)) = (joined(ai, bo, a, b), joined(bi, ao, b, a)) {
                    return Some((n, vec![ai, ao, bi, bo], vec![fwd, rev]));
                }
            }
        }
        if incident.len() == 2 {
            if let (Some(ai), None, None, Some(bo)) = (a_in, a_out, b_in, b_out) {
                if let Some(fwd) = joined(ai, bo, a, b) {
                    return Some((n, vec![ai, bo], vec![fwd]));
                }
            }
            if let (None, Some(ao), Some(bi), None) = (a_in, a_out, b_in, b_out) {
                if let Some(rev) = joined(bi, ao, b, a) {
                    return Some((n, vec![bi, ao], vec![rev]));
                }
            }
        }
    }
    None
}

/// Pull every lane back from its end nodes to the junction boundary so vehicles
/// stop and start at the edge of the intersection, leaving the interior (the box)
/// to the movements' crossing paths. A node's setback is half the widest
/// carriageway meeting it; clamped so short links keep a positive drivable span.
/// Slide each ramp's freeway end out to the curb. OSM attaches a ramp at the
/// freeway *centreline* node, but the renderer treats a carriageway's polyline as
/// its left (median) edge and offsets lanes rightward — so a narrow ramp sharing
/// that node is drawn on the freeway's inner lanes instead of peeling off the
/// outside. Shifting the ramp end laterally by the freeway's remaining width
/// (tapered back to its own alignment over `TRANSITION` m) makes the ramp diverge
/// from / merge onto the curb edge, matching how the lanes are wired.
fn offset_ramps_to_curb(net: &mut Network) {
    const TRANSITION: f64 = 45.0;
    // Freeway travel direction and width at each node it touches (through-direction).
    let mut freeway_at: HashMap<u32, ([f64; 2], f64)> = HashMap::new();
    for fi in 0..net.links.len() {
        let f = net.links[fi];
        if f.kind != RoadKind::Freeway {
            continue;
        }
        freeway_at.entry(f.to.0).or_insert((net.arrival_dir(LinkId(fi as u32)), f.lane_count as f64));
        freeway_at.entry(f.from.0).or_insert((net.departure_dir(LinkId(fi as u32)), f.lane_count as f64));
    }
    for li in 0..net.links.len() {
        if net.links[li].kind != RoadKind::Ramp {
            continue;
        }
        let ramp_lanes = net.links[li].lane_count as f64;
        for at_start in [true, false] {
            let node = if at_start { net.links[li].from } else { net.links[li].to };
            let Some(&(fdir, flanes)) = freeway_at.get(&node.0) else { continue };
            if flanes <= ramp_lanes {
                continue; // nothing wider to peel off from
            }
            let right = [fdir[1], -fdir[0]]; // curb side of the freeway
            let mag = (flanes - ramp_lanes) * LANE_WIDTH;
            let shift = [right[0] * mag, right[1] * mag];
            // Cumulative arc distance of each point from this (freeway-connected) end.
            let poly = net.polylines[li].clone();
            let count = poly.len();
            let order: Vec<usize> = if at_start { (0..count).collect() } else { (0..count).rev().collect() };
            let mut dist = vec![0.0f64; count];
            for w in 1..count {
                let (a, b) = (poly[order[w - 1]], poly[order[w]]);
                dist[order[w]] = dist[order[w - 1]] + (b[0] - a[0]).hypot(b[1] - a[1]);
            }
            for i in 0..count {
                let t = (1.0 - dist[i] / TRANSITION).max(0.0); // full at the end, 0 by TRANSITION
                net.polylines[li][i][0] += shift[0] * t;
                net.polylines[li][i][1] += shift[1] * t;
            }
        }
    }
}

fn set_junction_setbacks(net: &mut Network) {
    // A crosswalk/stop-bar margin so vehicles halt just behind the box, as at a
    // real signalized intersection, rather than nosing into the crossing.
    const STOP_MARGIN: f64 = 2.5;
    // Box radius per node = half the widest carriageway meeting it; the drawn
    // road stops here, and vehicles stop `STOP_MARGIN` further back.
    let mut box_r = vec![0.0f64; net.nodes.len()];
    for link in &net.links {
        let half = link.lane_count as f64 * LANE_WIDTH * 0.5;
        box_r[link.from.idx()] = box_r[link.from.idx()].max(half);
        box_r[link.to.idx()] = box_r[link.to.idx()].max(half);
    }
    // A pure freeway interchange (diverge/merge/connector) has no cross traffic and
    // no stop bar: the ramp peels off the mainline edge. Collapse its box so the
    // carriageways run together instead of pulling back into an intersection-like
    // gap, keeping only a hairline setback for numerical safety.
    let interchange: Vec<bool> = (0..net.nodes.len()).map(|n| net.is_interchange_node(NodeId(n as u32))).collect();
    for n in 0..net.nodes.len() {
        if interchange[n] {
            box_r[n] = box_r[n].min(0.5);
        }
    }
    net.render_setback = box_r.clone();
    let radius: Vec<f64> = box_r
        .iter()
        .enumerate()
        .map(|(n, &r)| if r <= 0.0 { 0.0 } else if interchange[n] { r } else { r + STOP_MARGIN })
        .collect();
    for i in 0..net.links.len() {
        let link = net.links[i];
        let full = net.polylines[i].windows(2).map(|w| distance(w[0], w[1])).sum::<f64>();
        let (mut r0, mut r1) = (radius[link.from.idx()], radius[link.to.idx()]);
        if r0 + r1 > full - 1.0 {
            let scale = ((full - 1.0).max(0.0)) / (r0 + r1).max(1e-9);
            r0 *= scale;
            r1 *= scale;
        }
        for lane in link.lane_start.0..link.lane_start.0 + link.lane_count {
            net.lanes[lane as usize].start_offset = r0;
            net.lanes[lane as usize].length = full - r0 - r1;
        }
    }
}

#[cfg(feature = "import")]
mod json {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct JsonSignal {
        green_secs: f64,
        yellow_secs: f64,
        offset: f64,
    }

    #[derive(Deserialize)]
    struct JsonNode {
        osm_id: i64,
        x: f64,
        y: f64,
        control: String,
        #[serde(default)]
        signal: Option<JsonSignal>,
    }

    #[derive(Deserialize)]
    struct JsonLink {
        from_osm: i64,
        to_osm: i64,
        lanes: u32,
        speed_limit: f64,
        /// Intermediate bend points `[[x, y], …]` between the endpoints.
        #[serde(default)]
        geometry: Vec<[f64; 2]>,
        #[serde(default)]
        layer: i32,
        #[serde(default)]
        name: String,
        #[serde(default)]
        road_class: String,
        #[serde(default, rename = "ref")]
        highway_ref: String,
        #[serde(default)]
        turn_lanes: String,
    }

    #[derive(Deserialize)]
    struct JsonMap {
        nodes: Vec<JsonNode>,
        links: Vec<JsonLink>,
    }

    pub fn parse(s: &str) -> Result<OsmMap, String> {
        let raw: JsonMap = serde_json::from_str(s).map_err(|e| e.to_string())?;
        let nodes = raw
            .nodes
            .into_iter()
            .map(|n| {
                let control = match n.control.as_str() {
                    "signal" => {
                        let s = n.signal.unwrap_or(JsonSignal {
                            green_secs: 25.0,
                            yellow_secs: 4.0,
                            offset: 0.0,
                        });
                        MapControl::Signal(SignalPlan {
                            green_secs: s.green_secs,
                            yellow_secs: s.yellow_secs,
                            offset: s.offset,
                        })
                    }
                    "stop" => MapControl::Stop,
                    "yield" => MapControl::Yield,
                    _ => MapControl::Uncontrolled,
                };
                NodeSpec { osm_id: n.osm_id, x: n.x, y: n.y, control }
            })
            .collect();
        let links = raw
            .links
            .into_iter()
            .map(|l| LinkSpec {
                from_osm: l.from_osm,
                to_osm: l.to_osm,
                lanes: l.lanes.max(1),
                speed_limit: l.speed_limit,
                geometry: l.geometry,
                layer: l.layer,
                name: l.name,
                road_class: l.road_class,
                highway_ref: l.highway_ref,
                turn_lanes: l.turn_lanes,
            })
            .collect();
        Ok(OsmMap { nodes, links })
    }
}

impl OsmMap {
    /// Parse the OSM scraper's JSON (`tools/osm-scraper`) into an `OsmMap`,
    /// simplifying spurious pass-through nodes. Requires the `import` feature.
    #[cfg(feature = "import")]
    pub fn from_json(s: &str) -> Result<OsmMap, String> {
        Ok(json::parse(s)?.collapse_pass_through_nodes().merge_split_intersections())
    }
}

#[cfg(all(test, feature = "import"))]
mod import_tests {
    use super::*;

    #[test]
    fn parses_scraper_json_into_a_buildable_map() {
        let doc = r#"{
            "meta": { "place": "test" },
            "nodes": [
                { "osm_id": 1, "x": 0.0, "y": 0.0, "control": "uncontrolled" },
                { "osm_id": 2, "x": 200.0, "y": 0.0, "control": "signal",
                  "signal": { "green_secs": 20.0, "yellow_secs": 3.0, "offset": 5.0 } },
                { "osm_id": 3, "x": 400.0, "y": 0.0, "control": "uncontrolled" },
                { "osm_id": 4, "x": 200.0, "y": -150.0, "control": "uncontrolled" }
            ],
            "links": [
                { "from_osm": 1, "to_osm": 2, "lanes": 2, "speed_limit": 20.0 },
                { "from_osm": 2, "to_osm": 3, "lanes": 2, "speed_limit": 20.0 },
                { "from_osm": 4, "to_osm": 2, "lanes": 1, "speed_limit": 15.0 }
            ]
        }"#;
        let map = OsmMap::from_json(doc).expect("valid json");
        let net = map.build();
        assert_eq!(net.nodes.len(), 4);
        assert_eq!(net.lanes.len(), 5);
        assert_eq!(net.programs.len(), 1, "node 2 is signalized");
        assert!(net.groups.len() >= 2);
    }

    #[test]
    #[ignore] // dev tool: emits a committable fixture from the local map.json
    fn extract_complex_junction_fixtures() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
        let Ok(text) = std::fs::read_to_string(&path) else { return };
        let raw = json::parse(&text).expect("map json");
        let pos: std::collections::HashMap<i64, [f64; 2]> = raw.nodes.iter().map(|n| (n.osm_id, [n.x, n.y])).collect();
        let mut neigh: std::collections::HashMap<i64, std::collections::BTreeSet<i64>> = Default::default();
        for l in &raw.links {
            neigh.entry(l.from_osm).or_default().insert(l.to_osm);
            neigh.entry(l.to_osm).or_default().insert(l.from_osm);
        }
        // Rank nodes by *neighbourhood* link density (links within 40 m) so the
        // complex clusters — divided carriageways crossing a street as several
        // nodes — surface, not just single high-degree nodes. Dedupe nearby centres.
        let density = |cc: [f64; 2]| raw.links.iter().filter(|l| {
            let (a, b) = (pos[&l.from_osm], pos[&l.to_osm]);
            (a[0] - cc[0]).hypot(a[1] - cc[1]) < 40.0 || (b[0] - cc[0]).hypot(b[1] - cc[1]) < 40.0
        }).count();
        let mut by_links: Vec<(i64, usize)> = raw.nodes.iter().map(|n| (n.osm_id, density([n.x, n.y]))).collect();
        by_links.sort_by(|a, b| b.1.cmp(&a.1));
        let mut centers: Vec<i64> = Vec::new();
        for &(id, _) in &by_links {
            if centers.iter().all(|&e| (pos[&id][0] - pos[&e][0]).hypot(pos[&id][1] - pos[&e][1]) > 90.0) {
                centers.push(id);
            }
            if centers.len() == 3 {
                break;
            }
        }

        for (rank, &center) in centers.iter().enumerate() {
            let deg = density(pos[&center]);
            const R: f64 = 70.0;
            let c = pos[&center];
            let dist = |id: i64| (pos[&id][0] - c[0]).hypot(pos[&id][1] - c[1]);
            let mut keep: std::collections::BTreeSet<i64> = raw.nodes.iter().map(|n| n.osm_id).filter(|&id| dist(id) < R).collect();
            let links: Vec<&LinkSpec> = raw.links.iter().filter(|l| keep.contains(&l.from_osm) || keep.contains(&l.to_osm)).collect();
            for l in &links { keep.insert(l.from_osm); keep.insert(l.to_osm); } // arm termini
            let ctrl = |c: MapControl| match c { MapControl::Signal(_) => "signal", MapControl::Stop => "stop", MapControl::Yield => "yield", _ => "uncontrolled" };
            let mut out = String::from("{\n  \"nodes\": [\n");
            let nodes: Vec<&NodeSpec> = raw.nodes.iter().filter(|n| keep.contains(&n.osm_id)).collect();
            for (i, n) in nodes.iter().enumerate() {
                out.push_str(&format!("    {{ \"osm_id\": {}, \"x\": {:.1}, \"y\": {:.1}, \"control\": \"{}\"{} }}{}\n",
                    n.osm_id, n.x - c[0], n.y - c[1], ctrl(n.control),
                    if let MapControl::Signal(p) = n.control { format!(", \"signal\": {{ \"green_secs\": {}, \"yellow_secs\": {}, \"offset\": 0.0 }}", p.green_secs, p.yellow_secs) } else { String::new() },
                    if i + 1 < nodes.len() { "," } else { "" }));
            }
            out.push_str("  ],\n  \"links\": [\n");
            for (i, l) in links.iter().enumerate() {
                let geom: Vec<String> = l.geometry.iter().map(|g| format!("[{:.1},{:.1}]", g[0] - c[0], g[1] - c[1])).collect();
                out.push_str(&format!("    {{ \"from_osm\": {}, \"to_osm\": {}, \"lanes\": {}, \"speed_limit\": {}, \"layer\": {}, \"geometry\": [{}] }}{}\n",
                    l.from_osm, l.to_osm, l.lanes, l.speed_limit, l.layer, geom.join(","), if i + 1 < links.len() { "," } else { "" }));
            }
            out.push_str("  ]\n}\n");
            let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/sim/fixtures");
            std::fs::create_dir_all(dir).unwrap();
            std::fs::write(format!("{dir}/junction_{rank}.json"), &out).unwrap();
            println!("\n===== FIXTURE {rank} (center {center}, {deg} nearby links, {} nodes / {} links) written", nodes.len(), links.len());
        }
    }

    #[test]
    fn exposed_link_data_stays_aligned_with_the_engines_links() {
        // The browser hit-tests clicks against the engine's exposed link names and
        // polylines and then asks the engine for that link's stats. If the import
        // transforms (collapse + merge) left names/geometry misaligned with the
        // link set, a click would select the wrong road — the glitch this guards.
        // A named road with a pass-through node (collapses) and a split junction
        // 20 m wide (merges) exercises both transforms.
        let doc = r#"{
            "nodes": [
                { "osm_id": 1, "x": -100.0, "y": 0.0, "control": "uncontrolled" },
                { "osm_id": 2, "x": 0.0, "y": 0.0, "control": "uncontrolled" },
                { "osm_id": 3, "x": 100.0, "y": 0.0, "control": "uncontrolled" },
                { "osm_id": 4, "x": 120.0, "y": 0.0, "control": "uncontrolled" },
                { "osm_id": 5, "x": 220.0, "y": 0.0, "control": "uncontrolled" },
                { "osm_id": 6, "x": 100.0, "y": -100.0, "control": "uncontrolled" },
                { "osm_id": 7, "x": 120.0, "y": 100.0, "control": "uncontrolled" }
            ],
            "links": [
                { "from_osm": 1, "to_osm": 2, "lanes": 1, "speed_limit": 15.0, "name": "Broadway" },
                { "from_osm": 2, "to_osm": 3, "lanes": 1, "speed_limit": 15.0, "name": "Broadway" },
                { "from_osm": 4, "to_osm": 5, "lanes": 1, "speed_limit": 15.0, "name": "Broadway" },
                { "from_osm": 3, "to_osm": 4, "lanes": 1, "speed_limit": 15.0, "name": "" },
                { "from_osm": 3, "to_osm": 6, "lanes": 1, "speed_limit": 15.0, "name": "Oak Street" },
                { "from_osm": 4, "to_osm": 7, "lanes": 1, "speed_limit": 15.0, "name": "Oak Street" }
            ]
        }"#;
        let net = OsmMap::from_json(doc).expect("valid json").build();

        assert_eq!(net.link_names.len(), net.links.len(), "one name per link");
        assert_eq!(net.polylines.len(), net.links.len(), "one polyline per link");
        assert!(net.links.len() < 6, "collapse + merge reduced the link count, got {}", net.links.len());

        for i in 0..net.links.len() {
            let link = net.link(LinkId(i as u32));
            let poly = &net.polylines[i];
            assert_eq!(poly[0], net.node(link.from).position, "polyline starts at its link's from-node");
            assert_eq!(*poly.last().unwrap(), net.node(link.to).position, "polyline ends at its link's to-node");
        }

        // Names survive the transforms and land on the right links.
        let broadway = (0..net.links.len())
            .find(|&i| net.node(net.link(LinkId(i as u32)).from).position == [-100.0, 0.0])
            .expect("the west Broadway approach exists");
        assert_eq!(net.link_names[broadway], "Broadway", "the collapsed+merged road keeps its name");
        assert!(net.link_names.iter().any(|n| n == "Oak Street"), "the cross street name is preserved too");
    }

    #[test]
    fn malformed_json_is_an_error_not_a_panic() {
        assert!(OsmMap::from_json("{ not json").is_err());
    }

    #[test]
    fn real_map_signals_are_linked_and_not_all_green() {
        // Load the committed Millbrae map and verify its signalized nodes behave:
        // no conflicting movements are ever green together (orthogonal approaches
        // are linked), and greens/reds coexist (they aren't stuck all-green).
        use crate::sim::signal::SignalState;
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
        let Ok(text) = std::fs::read_to_string(path) else { return }; // skip if absent
        let net = OsmMap::from_json(&text).expect("valid map json").build();
        if net.programs.is_empty() {
            return;
        }
        let mut saw_red = false;
        for step in 0..600 {
            let t = step as f64 * 0.5;
            let states = net.signal_states(t);
            saw_red |= states.iter().any(|s| *s == SignalState::Red);
            for c in &net.conflicts {
                let (Some(a), Some(b)) = (net.movement(c.a).signal_group, net.movement(c.b).signal_group) else { continue };
                let ga = net.movement_state(c.a, t);
                let gb = net.movement_state(c.b, t);
                assert!(
                    !(ga == SignalState::Green && gb == SignalState::Green),
                    "conflicting movements both green at a real signalized node, t={t}"
                );
                let _ = (a, b);
            }
        }
        assert!(saw_red, "real-map signals are not stuck all-green");
    }
}

/// A signalized four-way: a two-lane through corridor (west→east) crossed by a
/// one-lane street (south→north), the centre node on a two-phase signal.
pub fn corridor_with_signal() -> Network {
    let plan = SignalPlan { green_secs: 15.0, yellow_secs: 3.0, offset: 0.0 };
    OsmMap {
        nodes: vec![
            NodeSpec::uncontrolled(1, 0.0, 0.0),
            NodeSpec::signalized(2, 200.0, 0.0, plan),
            NodeSpec::uncontrolled(4, 400.0, 0.0),
            NodeSpec::uncontrolled(3, 200.0, -200.0),
            NodeSpec::uncontrolled(5, 200.0, 200.0),
        ],
        links: vec![
            LinkSpec::oneway(1, 2, 2, 25.0),
            LinkSpec::oneway(2, 4, 2, 25.0),
            LinkSpec::oneway(3, 2, 1, 15.0),
            LinkSpec::oneway(2, 5, 1, 15.0),
        ],
    }
    .build()
}

/// A deliberate bottleneck for demonstrating the adaptive mass layer: a long
/// three-lane arterial approaches a signal and drops to a single-lane exit, so under
/// steady demand the approach saturates and backs up — exactly the congested link
/// the mesoscopic layer aggregates. A cross street shares the signal so the arterial
/// never gets full green.
pub fn gridlock() -> Network {
    let plan = SignalPlan { green_secs: 16.0, yellow_secs: 3.0, offset: 0.0 };
    OsmMap {
        nodes: vec![
            NodeSpec::uncontrolled(1, -600.0, 0.0),
            NodeSpec::signalized(2, 0.0, 0.0, plan),
            NodeSpec::uncontrolled(3, 350.0, 0.0),
            NodeSpec::uncontrolled(4, 0.0, -350.0),
            NodeSpec::uncontrolled(5, 0.0, 350.0),
        ],
        links: vec![
            LinkSpec::oneway(1, 2, 3, 25.0), // wide approach (jams behind the drop + signal)
            LinkSpec::oneway(2, 3, 1, 20.0), // single-lane bottleneck exit
            LinkSpec::oneway(4, 2, 2, 18.0), // cross approach
            LinkSpec::oneway(2, 5, 2, 18.0), // cross exit
        ],
    }
    .build()
}

/// Real complex junctions lifted from the scraped Millbrae map (re-centred to the
/// origin, names stripped — geometry only), as deterministic test fixtures. #0 is
/// a divided arterial whose crossing OSM splits across four signal nodes; the
/// import pipeline (collapse + merge) then reduces it to one junction. Requires
/// the `import` feature (JSON parse). Regenerate via the `extract_complex_junction
/// _fixtures` dev test.
#[cfg(feature = "import")]
pub fn millbrae_junction(n: usize) -> Network {
    let json = match n {
        0 => include_str!("fixtures/junction_0.json"),
        1 => include_str!("fixtures/junction_1.json"),
        _ => include_str!("fixtures/junction_2.json"),
    };
    OsmMap::from_json(json).expect("fixture json is valid").build()
}

/// A single Millbrae-complexity intersection: a two-way, two-lanes-each-way
/// arterial (like El Camino Real) crossing a two-way one-lane-each-way cross
/// street, centre node signalized. Every approach carries through, left and
/// right movements — the realistic case the intersection model must handle.
pub fn arterial_intersection() -> Network {
    let plan = SignalPlan { green_secs: 18.0, yellow_secs: 4.0, offset: 0.0 };
    let mut links = Vec::new();
    links.extend(LinkSpec::twoway(1, 0, 2, 20.0)); // west arm, arterial
    links.extend(LinkSpec::twoway(0, 2, 2, 20.0)); // east arm, arterial
    links.extend(LinkSpec::twoway(3, 0, 1, 13.0)); // south arm, cross street
    links.extend(LinkSpec::twoway(0, 4, 1, 13.0)); // north arm, cross street
    OsmMap {
        nodes: vec![
            NodeSpec::signalized(0, 0.0, 0.0, plan),
            NodeSpec::uncontrolled(1, -220.0, 0.0),
            NodeSpec::uncontrolled(2, 220.0, 0.0),
            NodeSpec::uncontrolled(3, 0.0, -220.0),
            NodeSpec::uncontrolled(4, 0.0, 220.0),
        ],
        links,
    }
    .build()
}

/// A small hand-built stand-in for scraped Millbrae, CA geometry: three
/// El Camino Real blocks (north↔south, two lanes each way) with signalized
/// cross streets, offset for a green wave. Replaced wholesale once the scraper
/// emits a real extract.
pub fn millbrae_sample() -> Network {
    let plan = |offset| SignalPlan { green_secs: 25.0, yellow_secs: 4.0, offset };
    let mut links = Vec::new();
    links.extend(LinkSpec::twoway(10, 11, 2, 18.0));
    links.extend(LinkSpec::twoway(11, 12, 2, 18.0));
    links.extend(LinkSpec::twoway(12, 13, 2, 18.0));
    links.extend(LinkSpec::twoway(20, 11, 1, 13.0));
    links.extend(LinkSpec::twoway(21, 12, 1, 13.0));
    OsmMap {
        nodes: vec![
            NodeSpec::uncontrolled(10, 0.0, 0.0),
            NodeSpec::signalized(11, 0.0, 220.0, plan(0.0)),
            NodeSpec::signalized(12, 0.0, 470.0, plan(12.0)),
            NodeSpec::uncontrolled(13, 0.0, 700.0),
            NodeSpec::uncontrolled(20, -180.0, 220.0),
            NodeSpec::uncontrolled(21, 180.0, 470.0),
        ],
        links,
    }
    .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signalized_corridor_is_coordinated_into_a_green_wave() {
        use super::super::signal::SignalState;
        // Three signals on one arterial, 300 m apart at 15 m/s (20 s travel each). After
        // coordination their through greens should open in a ~20 s progression — a green
        // wave — where the raw import leaves them all synchronized at offset 0.
        let plan = SignalPlan { green_secs: 20.0, yellow_secs: 4.0, offset: 0.0 };
        let road = |a, b, name: &str, lanes, sp| {
            let mut v = LinkSpec::twoway(a, b, lanes, sp).to_vec();
            for l in &mut v {
                l.name = name.to_string();
            }
            v
        };
        let mut links = Vec::new();
        for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 4)] {
            links.extend(road(a, b, "Main Street", 2, 15.0)); // the arterial
        }
        for (n, e, name) in [(1, 10, "Cross A"), (2, 11, "Cross B"), (3, 12, "Cross C")] {
            links.extend(road(n, e, name, 1, 12.0)); // a cross street at each signal
        }
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(0, -300.0, 0.0),
                NodeSpec::signalized(1, 0.0, 0.0, plan),
                NodeSpec::signalized(2, 300.0, 0.0, plan),
                NodeSpec::signalized(3, 600.0, 0.0, plan),
                NodeSpec::uncontrolled(4, 900.0, 0.0),
                NodeSpec::uncontrolled(10, 0.0, -200.0),
                NodeSpec::uncontrolled(11, 300.0, -200.0),
                NodeSpec::uncontrolled(12, 600.0, -200.0),
            ],
            links,
        }
        .build();

        // Absolute time each signal's Main-Street through movement turns green.
        let through_open = |node: NodeId| -> f64 {
            let NodeControl::Signalized(p) = net.node(node).control else { panic!("signal") };
            let prog = &net.programs[p.idx()];
            let bit = (0..net.movements.len() as u32)
                .find_map(|m| {
                    let mv = net.movement(MovementId(m));
                    let (fl, tl) = (net.lane(mv.from_lane).link, net.lane(mv.to_lane).link);
                    (mv.node == node
                        && net.movement_turn(MovementId(m)) == TurnType::Through
                        && net.link_names[fl.idx()] == "Main Street"
                        && net.link_names[tl.idx()] == "Main Street")
                        .then(|| net.groups[mv.signal_group.unwrap().idx()].bit)
                })
                .expect("a Main-Street through movement");
            let cycle = prog.cycle_length();
            let mut t = 0.0;
            while t < cycle {
                if prog.state_of(bit, t) == SignalState::Green
                    && prog.state_of(bit, (t - 0.1).rem_euclid(cycle)) != SignalState::Green
                {
                    return t;
                }
                t += 0.1;
            }
            0.0
        };

        // The three signals, in corridor order.
        let mut sigs: Vec<NodeId> = (0..net.nodes.len() as u32)
            .map(NodeId)
            .filter(|&n| matches!(net.node(n).control, NodeControl::Signalized(_)))
            .collect();
        sigs.sort_by(|&a, &b| net.node(a).position[0].total_cmp(&net.node(b).position[0]));
        assert_eq!(sigs.len(), 3, "three signals on the corridor");

        let cycle = {
            let NodeControl::Signalized(p) = net.node(sigs[0]).control else { panic!() };
            net.programs[p.idx()].cycle_length()
        };
        let opens: Vec<f64> = sigs.iter().map(|&n| through_open(n)).collect();
        // Coordination happened (raw import is offset 0 everywhere → all opens equal).
        assert!(opens[0] != opens[1] || opens[1] != opens[2], "the signals are staggered, not synchronized: {opens:?}");
        // Consecutive through greens open ~20 s apart (the travel time) around the cycle.
        let circ = |a: f64, b: f64| {
            let d = (a - b).rem_euclid(cycle);
            d.min(cycle - d)
        };
        for w in opens.windows(2) {
            assert!((circ(w[0], w[1]) - 20.0).abs() < 3.0, "greens progress by the ~20 s travel time: {opens:?} (cycle {cycle})");
        }
    }

    #[test]
    fn actuated_controller_honors_coordination_offsets_at_runtime() {
        use super::super::junction::SignalController;
        use super::super::signal::SignalState;
        use std::collections::HashSet;

        let plan = SignalPlan { green_secs: 20.0, yellow_secs: 4.0, offset: 0.0 };
        let road = |a, b, name: &str, lanes, sp| {
            let mut v = LinkSpec::twoway(a, b, lanes, sp).to_vec();
            for l in &mut v {
                l.name = name.to_string();
            }
            v
        };
        let mut links = Vec::new();
        for (a, b) in [(0, 1), (1, 2), (2, 3)] {
            links.extend(road(a, b, "Main Street", 2, 15.0));
        }
        for (n, e, name) in [(1, 10, "Cross A"), (2, 11, "Cross B")] {
            links.extend(road(n, e, name, 1, 12.0));
        }
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(0, -300.0, 0.0),
                NodeSpec::signalized(1, 0.0, 0.0, plan),
                NodeSpec::signalized(2, 300.0, 0.0, plan),
                NodeSpec::uncontrolled(3, 600.0, 0.0),
                NodeSpec::uncontrolled(10, 0.0, -200.0),
                NodeSpec::uncontrolled(11, 300.0, -200.0),
            ],
            links,
        }
        .build();

        let coordinated = net.programs.iter().filter(|p| p.coordinated).count();
        assert_eq!(coordinated, 2, "both corridor signals are coordinated");

        let through = |node: NodeId| -> (MovementId, u8, usize) {
            (0..net.movements.len() as u32)
                .find_map(|m| {
                    let mv = net.movement(MovementId(m));
                    let (fl, tl) = (net.lane(mv.from_lane).link, net.lane(mv.to_lane).link);
                    (mv.node == node
                        && net.movement_turn(MovementId(m)) == TurnType::Through
                        && net.link_names[fl.idx()] == "Main Street"
                        && net.link_names[tl.idx()] == "Main Street")
                        .then(|| {
                            let g = net.groups[mv.signal_group.unwrap().idx()];
                            (MovementId(m), g.bit, g.program.idx())
                        })
                })
                .expect("a Main-Street through movement")
        };
        let up = through(NodeId(1));
        let down = through(NodeId(2));
        assert_ne!(net.programs[up.2].offset, net.programs[down.2].offset, "offsets are staggered");

        let cycle = net.programs[up.2].cycle_length();
        let steps = (cycle / 0.1).ceil() as usize + 4;
        let mut ctrl = SignalController::build(&net);
        let empty = HashSet::new();
        let mut clock = 0.0;
        let mut staggered_instant = false;
        for _ in 0..steps {
            for (mid, bit, pid) in [up, down] {
                assert_eq!(
                    ctrl.movement_state(&net, mid),
                    net.programs[pid].state_of(bit, clock),
                    "coordinated runtime state must follow the offset schedule at t={clock}",
                );
            }
            let (us, ds) = (ctrl.movement_state(&net, up.0), ctrl.movement_state(&net, down.0));
            if us == SignalState::Green && ds != SignalState::Green {
                staggered_instant = true;
            }
            ctrl.advance(&net, &empty, 0.1);
            clock += 0.1;
        }
        assert!(staggered_instant, "the wave reaches the upstream green before the downstream one");
    }

    /// A signalized four-way with every in/out leg, centre node on a signal.
    fn signalized_cross() -> Network {
        let plan = SignalPlan { green_secs: 15.0, yellow_secs: 3.0, offset: 0.0 };
        OsmMap {
            nodes: vec![
                NodeSpec::signalized(0, 0.0, 0.0, plan),
                NodeSpec::uncontrolled(1, -100.0, 0.0),
                NodeSpec::uncontrolled(2, 100.0, 0.0),
                NodeSpec::uncontrolled(3, 0.0, -100.0),
                NodeSpec::uncontrolled(4, 0.0, 100.0),
            ],
            links: vec![
                LinkSpec::oneway(1, 0, 1, 15.0),
                LinkSpec::oneway(2, 0, 1, 15.0),
                LinkSpec::oneway(3, 0, 1, 15.0),
                LinkSpec::oneway(4, 0, 1, 15.0),
                LinkSpec::oneway(0, 1, 1, 15.0),
                LinkSpec::oneway(0, 2, 1, 15.0),
                LinkSpec::oneway(0, 3, 1, 15.0),
                LinkSpec::oneway(0, 4, 1, 15.0),
            ],
        }
        .build()
    }

    fn signalized_cross_at(speed: f64) -> Network {
        let plan = SignalPlan { green_secs: 15.0, yellow_secs: 3.0, offset: 0.0 };
        OsmMap {
            nodes: vec![
                NodeSpec::signalized(0, 0.0, 0.0, plan),
                NodeSpec::uncontrolled(1, -100.0, 0.0),
                NodeSpec::uncontrolled(2, 100.0, 0.0),
                NodeSpec::uncontrolled(3, 0.0, -100.0),
                NodeSpec::uncontrolled(4, 0.0, 100.0),
            ],
            links: vec![
                LinkSpec::oneway(1, 0, 1, speed),
                LinkSpec::oneway(2, 0, 1, speed),
                LinkSpec::oneway(3, 0, 1, speed),
                LinkSpec::oneway(4, 0, 1, speed),
                LinkSpec::oneway(0, 1, 1, speed),
                LinkSpec::oneway(0, 2, 1, speed),
                LinkSpec::oneway(0, 3, 1, speed),
                LinkSpec::oneway(0, 4, 1, speed),
            ],
        }
        .build()
    }

    #[test]
    fn signal_change_and_clearance_intervals_scale_with_approach_speed() {
        let intervals = |speed: f64| -> (f64, f64) {
            let net = signalized_cross_at(speed);
            let prog = &net.programs[0];
            (
                prog.phases.iter().map(|p| p.yellow_secs).fold(0.0, f64::max),
                prog.phases.iter().map(|p| p.all_red_secs).fold(0.0, f64::max),
            )
        };
        let (slow_y, slow_ar) = intervals(11.0);
        let (fast_y, fast_ar) = intervals(29.0);
        assert!(fast_y > slow_y, "faster approaches get longer yellows: {slow_y} vs {fast_y}");
        assert!((3.0..=6.0).contains(&slow_y) && (3.0..=6.0).contains(&fast_y), "yellows stay in the ITE range");
        assert!((fast_y - (1.0 + 29.0 / 6.0)).abs() < 1e-9, "yellow follows the ITE kinematic formula");
        for ar in [slow_ar, fast_ar] {
            assert!((1.5..=5.0).contains(&ar), "clearance stays in a realistic band: {ar}");
        }
    }

    #[test]
    fn signal_phasing_never_greens_conflicting_movements_together() {
        use crate::sim::signal::SignalState;
        let net = signalized_cross();
        let cycle = net.programs[0].cycle_length();
        let conflicts = net.conflicts.clone();
        assert!(!conflicts.is_empty(), "a four-way has crossing conflicts");
        let mut steps = 0;
        while steps as f64 * 0.5 < cycle + 1.0 {
            let t = steps as f64 * 0.5;
            for c in &conflicts {
                let a = net.movement_state(c.a, t);
                let b = net.movement_state(c.b, t);
                assert!(!(a == SignalState::Green && b == SignalState::Green), "conflicting greens at t={t}");
            }
            steps += 1;
        }
    }

    #[test]
    fn arterial_intersection_phasing_is_safe_and_complete() {
        // A realistic multi-lane, two-way signalized crossing (El Camino × cross
        // street): every conflicting movement pair is kept out of simultaneous
        // green, and every movement is served green at some point in the cycle.
        use crate::sim::signal::SignalState;
        let net = arterial_intersection();
        assert!(net.programs.len() == 1 && !net.conflicts.is_empty());
        let cycle = net.programs[0].cycle_length();
        let signalized: Vec<u32> = (0..net.movements.len() as u32)
            .filter(|&m| net.movement(MovementId(m)).signal_group.is_some())
            .collect();
        let mut ever_green = vec![false; net.movements.len()];
        let mut steps = 0;
        while steps as f64 * 0.5 < cycle + 1.0 {
            let t = steps as f64 * 0.5;
            for c in &net.conflicts {
                assert!(
                    !(net.movement_state(c.a, t) == SignalState::Green && net.movement_state(c.b, t) == SignalState::Green),
                    "conflicting greens at t={t}"
                );
            }
            for &m in &signalized {
                if net.movement_state(MovementId(m), t) == SignalState::Green {
                    ever_green[m as usize] = true;
                }
            }
            steps += 1;
        }
        assert!(signalized.iter().all(|&m| ever_green[m as usize]), "every movement gets a green in the cycle");
    }

    #[test]
    fn opposing_left_gets_its_own_protected_phase() {
        // A left turn that crosses opposing through traffic must be in a different
        // signal group than that through (a protected, not permissive, left).
        let net = signalized_cross();
        let mut found = false;
        for c in &net.conflicts {
            let (a, b) = (net.movement(c.a), net.movement(c.b));
            if net.movement_turn(c.a) == TurnType::Left || net.movement_turn(c.b) == TurnType::Left {
                assert_ne!(a.signal_group, b.signal_group, "a left is grouped apart from what it conflicts with");
                found = true;
            }
        }
        assert!(found, "the four-way has a left-turn conflict to protect");
    }

    #[test]
    fn a_bridge_crossing_is_not_an_intersection() {
        // Two roads that cross geometrically but share no OSM node (an overpass):
        // they must form no movements between each other and no conflict, and the
        // bridge link carries its layer for render z-order.
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -100.0, 0.0),
                NodeSpec::uncontrolled(2, 100.0, 0.0),
                NodeSpec::uncontrolled(3, 0.0, -100.0),
                NodeSpec::uncontrolled(4, 0.0, 100.0),
            ],
            links: vec![
                LinkSpec::oneway(1, 2, 1, 20.0), // surface road, crosses origin
                LinkSpec { from_osm: 3, to_osm: 4, lanes: 1, speed_limit: 25.0, geometry: Vec::new(), layer: 1, name: String::new(), road_class: String::new(), highway_ref: String::new(), turn_lanes: String::new() }, // bridge over it
            ],
        }
        .build();
        assert!(net.movements.is_empty(), "no shared node → no movements between the crossing roads");
        assert!(net.conflicts.is_empty(), "a grade-separated crossing has no conflict point");
        assert_eq!(net.link(LinkId(1)).layer, 1, "the bridge carries its layer");
    }

    #[test]
    fn corridor_builds_expected_counts() {
        let net = corridor_with_signal();
        assert_eq!(net.nodes.len(), 5);
        assert_eq!(net.links.len(), 4);
        assert_eq!(net.lanes.len(), 2 + 2 + 1 + 1);
        // West approach: a through/right group + a protected-left group (2→5);
        // south approach: one through/right group. Phasing is per-conflict, so
        // the left is separated from the opposing through.
        assert_eq!(net.groups.len(), 3);
        assert_eq!(net.programs.len(), 1);
    }

    #[test]
    fn through_lanes_are_gated_by_a_signal_group() {
        let net = corridor_with_signal();
        for lane_id in net.lanes_of(LinkId(0)) {
            let ms = net.movements_of(lane_id);
            assert!(!ms.is_empty());
            assert!(ms.iter().all(|m| m.signal_group.is_some()));
        }
    }

    #[test]
    fn exit_lanes_are_sinks() {
        let net = corridor_with_signal();
        for lane_id in net.lanes_of(LinkId(1)) {
            assert_eq!(net.movements_of(lane_id).len(), 0);
        }
    }

    #[test]
    fn no_uturn_movements() {
        let net = corridor_with_signal();
        for m in &net.movements {
            let from_link = net.lane(m.from_lane).link;
            let to_link = net.lane(m.to_lane).link;
            assert_ne!(
                net.link(from_link).from,
                net.link(to_link).to,
                "movement reverses down the arrival link"
            );
        }
    }

    #[test]
    fn turns_are_channelised_left_to_right() {
        // A 3-lane west approach into a 4-way: the LEFT turn must land on the
        // left-hand lane, the RIGHT turn on the right-hand lane, with through in
        // between — otherwise turning cars weave across the approach. Regression
        // guard: lane 0 sits toward the centreline (`lane_point` places it there),
        // so exits are ordered left→right onto ascending lane indices.
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(0, 0.0, 0.0),
                NodeSpec::uncontrolled(1, -200.0, 0.0), // W, travel +x (east)
                NodeSpec::uncontrolled(2, 200.0, 0.0),  // E (through)
                NodeSpec::uncontrolled(3, 0.0, 200.0),  // N (left, i.e. +y)
                NodeSpec::uncontrolled(4, 0.0, -200.0), // S (right)
            ],
            links: vec![
                LinkSpec::oneway(1, 0, 3, 15.0), // 0: W→junction, three lanes
                LinkSpec::oneway(0, 2, 1, 15.0), // 1: →E through
                LinkSpec::oneway(0, 3, 1, 15.0), // 2: →N left
                LinkSpec::oneway(0, 4, 1, 15.0), // 3: →S right
            ],
        }
        .build();
        let (mut lefts, mut throughs, mut rights) = (vec![], vec![], vec![]);
        for lane in net.lanes_of(LinkId(0)) {
            let idx = net.lane(lane).index_in_link;
            for k in 0..net.lane(lane).movement_count {
                match net.movement_turn(MovementId(net.lane(lane).movement_start.0 + k)) {
                    TurnType::Left => lefts.push(idx),
                    TurnType::Through => throughs.push(idx),
                    TurnType::Right => rights.push(idx),
                }
            }
        }
        let min = |v: &[u32]| *v.iter().min().unwrap();
        let max = |v: &[u32]| *v.iter().max().unwrap();
        assert!(max(&lefts) <= min(&throughs), "left is left of through: {lefts:?} vs {throughs:?}");
        assert!(max(&throughs) <= min(&rights), "through is left of right: {throughs:?} vs {rights:?}");

        // Geometry: lower lane index really is the left lane. Travel is +x, so the
        // left side is +y — lane 0 must sit farther +y than the last lane. Sample
        // at the stop line, where the turn-pocket bays are fully open (upstream the
        // left/right bays merge into the through lane and coincide).
        let lanes: Vec<LaneId> = net.lanes_of(LinkId(0)).collect();
        let stop = net.lane(lanes[0]).length;
        let p0 = net.lane_point(lanes[0], stop);
        let pn = net.lane_point(*lanes.last().unwrap(), net.lane(*lanes.last().unwrap()).length);
        assert!(p0[1] > pn[1], "lane 0 sits to the left (toward the centreline)");
    }

    #[test]
    fn a_dedicated_turn_lane_becomes_a_pocket_that_opens_at_the_stop_line() {
        // The same 3-lane approach: its dedicated left lane (0) is a real bay that
        // merges into the through lane upstream and diverges to its own offset by
        // the stop line. A single-lane road never gets a pocket.
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(0, 0.0, 0.0),
                NodeSpec::uncontrolled(1, -200.0, 0.0),
                NodeSpec::uncontrolled(2, 200.0, 0.0),
                NodeSpec::uncontrolled(3, 0.0, 200.0),
                NodeSpec::uncontrolled(4, 0.0, -200.0),
            ],
            links: vec![
                LinkSpec::oneway(1, 0, 3, 15.0),
                LinkSpec::oneway(0, 2, 1, 15.0),
                LinkSpec::oneway(0, 3, 1, 15.0),
                LinkSpec::oneway(0, 4, 1, 15.0),
            ],
        }
        .build();
        let left = net.lanes_of(LinkId(0)).next().unwrap();
        let l = net.lane(left);
        assert!(l.pocket_taper > 0.0, "the dedicated left lane is a pocket");
        // At the stop line the bay is open (own offset); far upstream it has merged
        // into the through lane a full lane-width over; the divergence is monotonic.
        let off = |pos: f64| net.lane_lateral_offset(l, pos);
        let (open, closed) = (off(l.length), off(0.0));
        assert!((open - 0.5 * LANE_WIDTH).abs() < 1e-9, "open bay sits at its own offset");
        assert!((closed - 1.5 * LANE_WIDTH).abs() < 1e-9, "closed bay merges into the through lane");
        assert!(off(l.length - 6.0) < off(l.length - 24.0), "the bay tapers monotonically");

        // A one-lane approach can't have a bay.
        assert_eq!(net.lane(net.lanes_of(LinkId(1)).next().unwrap()).pocket_taper, 0.0);
    }

    #[test]
    fn freeway_ramps_wire_to_the_curb_lane() {
        // A lane-drop off-ramp like US-101's: a 6-lane freeway drops to a 5-lane
        // mainline plus a 1-lane off-ramp. The dropped 6th (curb) lane is consumed
        // by the ramp — exit-only — the mainline keeps lanes 0-4, and a downstream
        // on-ramp merges back onto the curb lane, not the median.
        let hw = |a, b, lanes, sp| LinkSpec { road_class: "motorway".into(), ..LinkSpec::oneway(a, b, lanes, sp) };
        let ramp = |a, b, lanes, sp| LinkSpec { road_class: "motorway_link".into(), ..LinkSpec::oneway(a, b, lanes, sp) };
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -600.0, 0.0),    // freeway in
                NodeSpec::uncontrolled(2, -200.0, 0.0),    // diverge
                NodeSpec::uncontrolled(3, 200.0, 0.0),     // merge
                NodeSpec::uncontrolled(6, 600.0, 0.0),     // freeway out
                NodeSpec::uncontrolled(4, 100.0, -260.0),  // off-ramp exit
                NodeSpec::uncontrolled(5, -100.0, -260.0), // on-ramp origin
            ],
            links: vec![
                hw(1, 2, 6, 29.0),   // 0: 6-lane freeway → diverge
                hw(2, 3, 5, 29.0),   // 1: 5-lane mainline (a lane drops to the ramp)
                ramp(2, 4, 1, 25.0), // 2: off-ramp (diverge from link 0)
                hw(3, 6, 6, 29.0),   // 3: 6-lane freeway out (a lane merges in)
                ramp(5, 3, 1, 25.0), // 4: on-ramp (merge into link 3)
            ],
        }
        .build();

        // Off-ramp: fed only by the curb lane (index 5) of the 6-lane approach.
        let ramp_froms: Vec<u32> = (0..net.movements.len() as u32)
            .map(MovementId)
            .filter(|&m| net.lane(net.movement(m).to_lane).link == LinkId(2))
            .map(|m| net.lane(net.movement(m).from_lane).index_in_link)
            .collect();
        assert_eq!(ramp_froms, vec![5], "the off-ramp diverges from the curb lane only, got {ramp_froms:?}");

        // The curb lane can continue OR exit (it is not forced onto the ramp), so a
        // through car merges left instead of stopping; the inner lanes only continue.
        let curb_lane = net.lanes_of(LinkId(0)).last().unwrap();
        let mut curb_dests: Vec<u32> = (0..net.lane(curb_lane).movement_count)
            .map(|j| net.lane(net.movement(MovementId(net.lane(curb_lane).movement_start.0 + j)).to_lane).link.0)
            .collect();
        curb_dests.sort();
        assert_eq!(curb_dests, vec![1, 2], "the curb lane continues on the freeway and feeds the ramp, got {curb_dests:?}");
        for k in 0..5u32 {
            let lane = net.lanes_of(LinkId(0)).nth(k as usize).unwrap();
            let dests: Vec<u32> = (0..net.lane(lane).movement_count)
                .map(|j| net.lane(net.movement(MovementId(net.lane(lane).movement_start.0 + j)).to_lane).link.0)
                .collect();
            assert_eq!(dests, vec![1], "inner lane {k} continues on the freeway only, got {dests:?}");
        }

        // On-ramp: the ramp (link 4) merges onto the curb lane (index 5) of link 3.
        let merge_tos: Vec<u32> = (0..net.movements.len() as u32)
            .map(MovementId)
            .filter(|&m| net.lane(net.movement(m).from_lane).link == LinkId(4))
            .map(|m| net.lane(net.movement(m).to_lane).index_in_link)
            .collect();
        assert_eq!(merge_tos, vec![5], "the on-ramp merges onto the curb lane, got {merge_tos:?}");
    }

    #[test]
    fn an_off_ramp_peels_off_the_freeway_curb_edge() {
        // A 6-lane freeway heading +x; its curb (right) side is -y. A diverging
        // off-ramp's start must be slid out toward that curb edge, not left sitting
        // on the median-side node where it would overlap the inner lanes.
        let hw = |a, b, lanes, sp| LinkSpec { road_class: "motorway".into(), ..LinkSpec::oneway(a, b, lanes, sp) };
        let ramp = |a, b, lanes, sp| LinkSpec { road_class: "motorway_link".into(), ..LinkSpec::oneway(a, b, lanes, sp) };
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -200.0, 0.0),
                NodeSpec::uncontrolled(2, 0.0, 0.0),      // diverge (freeway centreline node)
                NodeSpec::uncontrolled(3, 200.0, 0.0),
                NodeSpec::uncontrolled(4, 160.0, -200.0), // off-ramp exit
            ],
            links: vec![hw(1, 2, 6, 29.0), hw(2, 3, 5, 29.0), ramp(2, 4, 1, 25.0)],
        }
        .build();
        // The freeway carriageway spans y ∈ [-6·W, 0] (median at 0, curb at -21).
        // The ramp start began at the node (y=0) and must be pushed toward the curb.
        let ramp_start = net.polylines[2][0];
        assert!(ramp_start[1] < -10.0, "the off-ramp peels off the curb edge, start y = {}", ramp_start[1]);
    }

    #[test]
    fn collapse_dissolves_a_oneway_pass_through_into_one_link() {
        let map = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::uncontrolled(2, 100.0, 0.0),
                NodeSpec::uncontrolled(3, 200.0, 0.0),
            ],
            links: vec![LinkSpec::oneway(1, 2, 1, 20.0), LinkSpec::oneway(2, 3, 1, 20.0)],
        }
        .collapse_pass_through_nodes();
        assert_eq!(map.nodes.len(), 2, "the middle pass-through node is dissolved");
        assert_eq!(map.links.len(), 1);
        let l = &map.links[0];
        assert_eq!((l.from_osm, l.to_osm), (1, 3));
        assert_eq!(l.geometry, vec![[100.0, 0.0]], "the dissolved node becomes a bend point");
    }

    #[test]
    fn collapse_dissolves_a_twoway_pass_through_both_directions() {
        let mut links = LinkSpec::twoway(1, 2, 2, 18.0).to_vec();
        links.extend(LinkSpec::twoway(2, 3, 2, 18.0));
        let map = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::uncontrolled(2, 100.0, 0.0),
                NodeSpec::uncontrolled(3, 200.0, 0.0),
            ],
            links,
        }
        .collapse_pass_through_nodes();
        assert_eq!(map.nodes.len(), 2);
        assert_eq!(map.links.len(), 2, "one link each way remains");
        assert!(map.links.iter().any(|l| (l.from_osm, l.to_osm) == (1, 3)));
        assert!(map.links.iter().any(|l| (l.from_osm, l.to_osm) == (3, 1)));
    }

    #[test]
    fn collapse_chains_multiple_pass_through_nodes() {
        let map = OsmMap {
            nodes: (1..=4).map(|i| NodeSpec::uncontrolled(i, (i - 1) as f64 * 100.0, 0.0)).collect(),
            links: vec![
                LinkSpec::oneway(1, 2, 1, 20.0),
                LinkSpec::oneway(2, 3, 1, 20.0),
                LinkSpec::oneway(3, 4, 1, 20.0),
            ],
        }
        .collapse_pass_through_nodes();
        assert_eq!(map.nodes.len(), 2);
        assert_eq!(map.links.len(), 1);
        assert_eq!(map.links[0].geometry, vec![[100.0, 0.0], [200.0, 0.0]]);
    }

    #[test]
    fn collapse_preserves_real_junctions_signals_and_attribute_changes() {
        let plan = SignalPlan { green_secs: 15.0, yellow_secs: 3.0, offset: 0.0 };
        let signal_node = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::signalized(2, 100.0, 0.0, plan),
                NodeSpec::uncontrolled(3, 200.0, 0.0),
            ],
            links: vec![LinkSpec::oneway(1, 2, 1, 20.0), LinkSpec::oneway(2, 3, 1, 20.0)],
        }
        .collapse_pass_through_nodes();
        assert_eq!(signal_node.nodes.len(), 3, "a signalized pass-through is kept");

        let lane_drop = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::uncontrolled(2, 100.0, 0.0),
                NodeSpec::uncontrolled(3, 200.0, 0.0),
            ],
            links: vec![LinkSpec::oneway(1, 2, 2, 20.0), LinkSpec::oneway(2, 3, 1, 20.0)],
        }
        .collapse_pass_through_nodes();
        assert_eq!(lane_drop.nodes.len(), 3, "a lane-count change is a real transition, kept");

        let cross = signalized_cross_map();
        let collapsed = cross.collapse_pass_through_nodes();
        assert_eq!(collapsed.nodes.len(), cross.nodes.len(), "a real 4-way is untouched");
    }

    /// A divided-road crossing OSM splits into two junctions 20 m apart (west
    /// half `1`, east half `2`, joined by a stub), each carrying a cross arm.
    fn split_crossing() -> OsmMap {
        let mut links = LinkSpec::twoway(1, 2, 2, 20.0).to_vec(); // the stub between halves
        links.extend(LinkSpec::twoway(10, 1, 2, 20.0)); // west arm at half 1
        links.extend(LinkSpec::twoway(1, 11, 1, 13.0)); // south arm at half 1
        links.extend(LinkSpec::twoway(2, 12, 2, 20.0)); // east arm at half 2
        links.extend(LinkSpec::twoway(2, 13, 1, 13.0)); // north arm at half 2
        OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::uncontrolled(2, 20.0, 0.0),
                NodeSpec::uncontrolled(10, -120.0, 0.0),
                NodeSpec::uncontrolled(11, 0.0, -120.0),
                NodeSpec::uncontrolled(12, 140.0, 0.0),
                NodeSpec::uncontrolled(13, 20.0, 120.0),
            ],
            links,
        }
    }

    #[test]
    fn merge_collapses_a_split_crossing_into_one_junction() {
        let merged = split_crossing().merge_split_intersections();
        assert_eq!(merged.nodes.len(), 5, "the two halves become one node (plus the four arm ends)");
        assert!(!merged.links.iter().any(|l| (l.from_osm, l.to_osm) == (1, 2) || (l.from_osm, l.to_osm) == (2, 1)),
            "the internal stub is dropped");
        let rep = merged.nodes.iter().find(|n| ![10, 11, 12, 13].contains(&n.osm_id)).unwrap();
        assert!((rep.x - 10.0).abs() < 1e-6 && rep.y.abs() < 1e-6, "merged node sits at the cluster centroid");
        for arm in [10, 11, 12, 13] {
            assert!(merged.links.iter().any(|l| l.from_osm == arm && l.to_osm == rep.osm_id),
                "arm {arm} now approaches the merged junction");
        }
    }

    #[test]
    fn merged_split_crossing_is_one_signalized_intersection() {
        let plan = SignalPlan { green_secs: 20.0, yellow_secs: 4.0, offset: 0.0 };
        let mut map = split_crossing();
        map.nodes[1].control = MapControl::Signal(plan); // signalize the east half only
        let net = map.merge_split_intersections().build();
        assert_eq!(net.programs.len(), 1, "the merged junction has a single coordinated signal program");
        assert!(!net.conflicts.is_empty(), "the merged 4-way has crossing conflict points");
    }

    #[test]
    fn merge_leaves_ordinary_blocks_untouched() {
        // Nodes a normal block apart (>STUB_MAX) are not merged.
        let net_nodes = signalized_cross_map().merge_split_intersections();
        assert_eq!(net_nodes.nodes.len(), 5, "a real 100 m four-way is left alone");
    }

    fn signalized_cross_map() -> OsmMap {
        let plan = SignalPlan { green_secs: 15.0, yellow_secs: 3.0, offset: 0.0 };
        OsmMap {
            nodes: vec![
                NodeSpec::signalized(0, 0.0, 0.0, plan),
                NodeSpec::uncontrolled(1, -100.0, 0.0),
                NodeSpec::uncontrolled(2, 100.0, 0.0),
                NodeSpec::uncontrolled(3, 0.0, -100.0),
                NodeSpec::uncontrolled(4, 0.0, 100.0),
            ],
            links: vec![
                LinkSpec::oneway(1, 0, 1, 15.0),
                LinkSpec::oneway(2, 0, 1, 15.0),
                LinkSpec::oneway(3, 0, 1, 15.0),
                LinkSpec::oneway(4, 0, 1, 15.0),
                LinkSpec::oneway(0, 1, 1, 15.0),
                LinkSpec::oneway(0, 2, 1, 15.0),
                LinkSpec::oneway(0, 3, 1, 15.0),
                LinkSpec::oneway(0, 4, 1, 15.0),
            ],
        }
    }

    #[test]
    fn millbrae_sample_builds_and_signals_are_consistent() {
        let net = millbrae_sample();
        assert!(net.programs.len() >= 2);
        let states = net.signal_states(30.0);
        assert_eq!(states.len(), net.groups.len());
    }
}
