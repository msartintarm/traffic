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
}

#[derive(Clone, Debug, PartialEq)]
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
}

impl LinkSpec {
    pub fn oneway(from_osm: i64, to_osm: i64, lanes: u32, speed_limit: f64) -> Self {
        Self { from_osm, to_osm, lanes, speed_limit, geometry: Vec::new(), layer: 0, name: String::new() }
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
        let pos: HashMap<i64, [f64; 2]> = self.nodes.iter().map(|n| (n.osm_id, [n.x, n.y])).collect();
        let mut neigh: HashMap<i64, BTreeSet<i64>> = HashMap::new();
        for l in &self.links {
            neigh.entry(l.from_osm).or_default().insert(l.to_osm);
            neigh.entry(l.to_osm).or_default().insert(l.from_osm);
        }
        let degree = |id: i64| neigh.get(&id).map_or(0, BTreeSet::len);

        let mut parent: HashMap<i64, i64> = self.nodes.iter().map(|n| (n.osm_id, n.osm_id)).collect();
        for l in &self.links {
            if distance(pos[&l.from_osm], pos[&l.to_osm]) < STUB_MAX
                && degree(l.from_osm) >= 3
                && degree(l.to_osm) >= 3
            {
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
                });
            }
            net.links.push(Link { from, to, lane_start, lane_count: spec.lanes, layer: spec.layer });
            net.polylines.push(polyline);
            net.link_names.push(spec.name.clone());
        }

        set_junction_setbacks(&mut net);

        let mut movements: Vec<Movement> = Vec::new();
        for lane_id in 0..net.lanes.len() {
            let lane = net.lanes[lane_id];
            let link = net.links[lane.link.idx()];
            let node = link.to;
            let start = MovementId(movements.len() as u32);
            for out_li in 0..net.links.len() {
                let out = net.links[out_li];
                if out.from != node || out.to == link.from {
                    continue;
                }
                let to_index = lane.index_in_link.min(out.lane_count - 1);
                let to_lane = LaneId(out.lane_start.0 + to_index);
                movements.push(Movement { from_lane: LaneId(lane_id as u32), to_lane, node, signal_group: None });
            }
            net.lanes[lane_id].movement_start = start;
            net.lanes[lane_id].movement_count = movements.len() as u32 - start.0;
        }
        net.movements = movements;
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
            // A protected-left-only phase (a left arrow) is short; through phases
            // get the full green. This keeps the cycle from ballooning when an
            // intersection needs several protected-left stages.
            let green = if gs.iter().all(|&g| left_group[g]) {
                (plan.green_secs * 0.45).max(6.0)
            } else {
                plan.green_secs
            };
            Phase::new(mask, green, plan.yellow_secs)
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

        let joined = |s1: usize, s2: usize, from: i64, to: i64| -> Option<LinkSpec> {
            let (l1, l2) = (&links[s1], &links[s2]);
            if l1.lanes != l2.lanes || l1.speed_limit != l2.speed_limit || l1.layer != l2.layer {
                return None;
            }
            let mut geometry = l1.geometry.clone();
            geometry.push(pos[&n]);
            geometry.extend(l2.geometry.iter().copied());
            let name = if l1.name.is_empty() { l2.name.clone() } else { l1.name.clone() };
            Some(LinkSpec { from_osm: from, to_osm: to, lanes: l1.lanes, speed_limit: l1.speed_limit, geometry, layer: l1.layer, name })
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
    net.render_setback = box_r.clone();
    let radius: Vec<f64> = box_r.iter().map(|&r| if r > 0.0 { r + STOP_MARGIN } else { 0.0 }).collect();
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

    #[test]
    fn signal_phasing_never_greens_conflicting_movements_together() {
        // The phases are built from the conflict graph, so at no instant in the
        // cycle are two crossing movements both green — the guarantee that keeps
        // signalized intersections collision-free.
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
                LinkSpec { from_osm: 3, to_osm: 4, lanes: 1, speed_limit: 25.0, geometry: Vec::new(), layer: 1, name: String::new() }, // bridge over it
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
