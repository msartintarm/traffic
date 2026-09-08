//! OSM-facing import schema and the builder that compiles it into a runtime
//! [`Network`]. The scraper tool (`../tools/osm-scraper`) emits data shaped
//! exactly like [`OsmMap`] — nodes carrying projected coordinates and an
//! intersection control, plus already-directed links carrying lane count and
//! speed limit — so importing a real Millbrae extract is `OsmMap { .. }.build()`.
//!
//! The builder resolves signals into [`SignalGroup`]s (one per signalized
//! approach), and connects each incoming lane to onward links at its downstream
//! node, skipping U-turns. A link carrying OSM `turn:lanes` channelizes its lanes
//! from that tag (`turn_lane_exits`); the rest split their exits by angular slice.

use std::collections::{BTreeSet, HashMap, HashSet};

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
    /// A railway level crossing (OSM `railway=level_crossing`): the engine
    /// closes it to road traffic on the train timetable.
    pub rail_crossing: bool,
}

impl NodeSpec {
    pub fn uncontrolled(osm_id: i64, x: f64, y: f64) -> Self {
        Self { osm_id, x, y, control: MapControl::Uncontrolled, rail_crossing: false }
    }

    pub fn signalized(osm_id: i64, x: f64, y: f64, plan: SignalPlan) -> Self {
        Self { osm_id, x, y, control: MapControl::Signal(plan), rail_crossing: false }
    }

    pub fn stop(osm_id: i64, x: f64, y: f64) -> Self {
        Self { osm_id, x, y, control: MapControl::Stop, rail_crossing: false }
    }

    pub fn give_way(osm_id: i64, x: f64, y: f64) -> Self {
        Self { osm_id, x, y, control: MapControl::Yield, rail_crossing: false }
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
    /// OSM `hov:lanes` (this direction, median outward, e.g. `"designated|no|no"`):
    /// lanes restricted to HOV/express-eligible vehicles. Empty when unmapped.
    pub hov_lanes: String,
    /// Observed Annual Average Daily Traffic for the road (both directions,
    /// vehicles/day), joined from real counts by `tools/counts attach_counts.py
    /// --write-map`; `0.0` = no observation. Calibrates demand to measured volumes
    /// instead of the lanes×speed proxy.
    pub aadt: f64,
    /// Trip-production weight from the scraper's land-use pass (`--landuse`):
    /// how residential the link's surroundings are, ~0.3–1.7 with 1.0 neutral;
    /// `0.0` = no data (treated as neutral). Tilts demand origins toward homes.
    pub res_weight: f64,
    /// Trip-attraction weight from the land-use pass: shops/jobs/campuses around
    /// the link, ~0.3–4 with 1.0 neutral; `0.0` = no data (treated as neutral).
    /// Multiplies gravity destination choice.
    pub attr_weight: f64,
    /// Per-approach sign (OSM `highway=stop`/`give_way` surveyed on the way,
    /// this travel direction): the approach stops/yields at its downstream
    /// node while the cross street keeps rolling — a two-way stop.
    pub sign: LinkSign,
}

impl LinkSpec {
    pub fn oneway(from_osm: i64, to_osm: i64, lanes: u32, speed_limit: f64) -> Self {
        Self { from_osm, to_osm, lanes, speed_limit, ..Default::default() }
    }

    pub fn twoway(a: i64, b: i64, lanes: u32, speed_limit: f64) -> [Self; 2] {
        [Self::oneway(a, b, lanes, speed_limit), Self::oneway(b, a, lanes, speed_limit)]
    }
}

/// One OSM turn restriction, resolved by the scraper to emitted-link node-id
/// pairs meeting at the via node: `from` enters it, `to` leaves it (`from.1 ==
/// to.0` for a via-node relation; a via-way relation's two ends differ until
/// the junction merge unifies them, and the restriction is inert if it never
/// does). `only` inverts the sense: the from-link may use *only* the to-link.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RestrictionSpec {
    pub from: (i64, i64),
    pub to: (i64, i64),
    pub only: bool,
}

#[derive(Clone, Debug, Default)]
pub struct OsmMap {
    pub nodes: Vec<NodeSpec>,
    pub links: Vec<LinkSpec>,
}

/// A parsed scraper extract: the topology-transformed map plus the turn
/// restrictions rewritten alongside it (their node pairs track every collapse
/// and merge, so they always reference the transformed map's link identities).
#[derive(Clone, Debug, Default)]
pub struct ImportedMap {
    pub map: OsmMap,
    pub restrictions: Vec<RestrictionSpec>,
}

impl ImportedMap {
    pub fn build(&self) -> Network {
        self.map.build_with_restrictions(&self.restrictions)
    }

    /// [`build`] with live stage reporting (see [`OsmMap::build_with_progress`]).
    pub fn build_with_progress(&self, cb: &mut dyn FnMut(&str, u32, u32)) -> Network {
        self.map.build_with_progress(&self.restrictions, cb)
    }

    /// Merge several extracts into one map in the FIRST entry's frame (each
    /// paired with its scraper projection origin). Boundary-clipped county
    /// scrapes keep the real OSM node at each county-line crossing, so deduping
    /// nodes by id stitches the road network back together across the line;
    /// duplicate links (a segment lying on the boundary in both extracts)
    /// collapse by their (from, to) pair, first occurrence winning.
    pub fn merge(parts: Vec<(ImportedMap, [f64; 2])>) -> ImportedMap {
        // A merged region is memory-bound end to end (network, lane bounds,
        // render mesh all scale with bend points), so link polylines are
        // thinned to this tolerance — invisible at any zoom the region is
        // viewed at.
        const GEOMETRY_TOL_M: f64 = 2.0;
        let frame = parts[0].1;
        let mut out = ImportedMap::default();
        let mut seen_nodes: HashSet<i64> = HashSet::new();
        let mut seen_links: HashSet<(i64, i64)> = HashSet::new();
        for (part, origin) in parts {
            let pos: HashMap<i64, [f64; 2]> = part.map.nodes.iter().map(|n| (n.osm_id, [n.x, n.y])).collect();
            for mut n in part.map.nodes {
                if !seen_nodes.insert(n.osm_id) {
                    continue;
                }
                [n.x, n.y] = reframe(n.x, n.y, origin, frame);
                out.map.nodes.push(n);
            }
            for mut l in part.map.links {
                if !seen_links.insert((l.from_osm, l.to_osm)) {
                    continue;
                }
                if let (Some(&a), Some(&b)) = (pos.get(&l.from_osm), pos.get(&l.to_osm)) {
                    l.geometry = thin_polyline(a, l.geometry, b, GEOMETRY_TOL_M);
                }
                for g in &mut l.geometry {
                    *g = reframe(g[0], g[1], origin, frame);
                }
                out.map.links.push(l);
            }
            out.restrictions.extend(part.restrictions);
        }
        out
    }
}

#[cfg(test)]
mod merge_tests {
    use super::*;

    #[test]
    fn merge_stitches_shared_boundary_nodes_and_reframes() {
        // Map A: 0→1→2 heading east. Map B (origin 0.01° north of A's): its
        // local frame sees the shared node 2 and continues 2→3. After the
        // merge, node 2 exists once, B's nodes land re-framed into A's metres,
        // and a route runs A-side to B-side across the stitch.
        let a = ImportedMap {
            map: OsmMap {
                nodes: vec![
                    NodeSpec::uncontrolled(0, 0.0, 0.0),
                    NodeSpec::uncontrolled(1, 100.0, 0.0),
                    NodeSpec::uncontrolled(2, 200.0, 0.0),
                ],
                links: vec![LinkSpec::oneway(0, 1, 1, 15.0), LinkSpec::oneway(1, 2, 1, 15.0)],
            },
            restrictions: vec![],
        };
        let rad = std::f64::consts::PI / 180.0;
        let dy = 0.01 * rad * 6371000.0; // B's origin sits this many metres north
        let b = ImportedMap {
            map: OsmMap {
                nodes: vec![
                    NodeSpec::uncontrolled(2, 200.0, -dy), // the shared boundary node, in B's frame
                    NodeSpec::uncontrolled(3, 300.0, -dy),
                ],
                links: vec![LinkSpec::oneway(2, 3, 1, 15.0)],
            },
            restrictions: vec![],
        };
        let origin_a = [37.0, -122.0];
        let origin_b = [37.01, -122.0];
        let merged = ImportedMap::merge(vec![(a, origin_a), (b, origin_b)]);
        assert_eq!(merged.map.nodes.len(), 4, "shared node 2 deduped");
        let n3 = merged.map.nodes.iter().find(|n| n.osm_id == 3).unwrap();
        assert!((n3.y - 0.0).abs() < 1.0 && (n3.x - 300.0).abs() < 1.0, "B re-framed into A's metres: {:?}", (n3.x, n3.y));
        let net = merged.build();
        let route = net.route_links(LinkId(0), LinkId(2));
        assert!(route.is_some(), "routes cross the county stitch");
    }
}

/// Douglas–Peucker over a link's interior bend points, anchored at its node
/// endpoints so bends near a junction are judged against the true chord.
fn thin_polyline(from: [f64; 2], bends: Vec<[f64; 2]>, to: [f64; 2], tol: f64) -> Vec<[f64; 2]> {
    if bends.len() < 2 {
        return bends;
    }
    let mut line = Vec::with_capacity(bends.len() + 2);
    line.push(from);
    line.extend(bends);
    line.push(to);
    let mut keep = vec![false; line.len()];
    keep[0] = true;
    *keep.last_mut().unwrap() = true;
    let mut stack = vec![(0usize, line.len() - 1)];
    while let Some((i, j)) = stack.pop() {
        let [ax, ay] = line[i];
        let [bx, by] = line[j];
        let (dx, dy) = (bx - ax, by - ay);
        let len = dx.hypot(dy).max(1e-9);
        let (mut far, mut fd) = (0usize, tol);
        for k in i + 1..j {
            let d = ((line[k][0] - ax) * dy - (line[k][1] - ay) * dx).abs() / len;
            if d > fd {
                (far, fd) = (k, d);
            }
        }
        if far > 0 {
            keep[far] = true;
            stack.push((i, far));
            stack.push((far, j));
        }
    }
    (1..line.len() - 1).filter(|&k| keep[k]).map(|k| line[k]).collect()
}

/// A point in one scraper extract's local frame, re-projected into another's
/// (local metres → GPS via `from`'s origin → local metres via `to`'s).
fn reframe(x: f64, y: f64, from: [f64; 2], to: [f64; 2]) -> [f64; 2] {
    const R: f64 = 6371000.0;
    let rad = std::f64::consts::PI / 180.0;
    let lat = from[0] + y / (rad * R);
    let lon = from[1] + x / (rad * R * (from[0] * rad).cos());
    [(lon - to[1]) * rad * (to[0] * rad).cos() * R, (lat - to[0]) * rad * R]
}

/// The scraper projection origin (`meta.origin`, [lat, lon]) of a raw map
/// JSON, read from the document head (meta leads the file) so a merge doesn't
/// re-lex a 30 MB body per part.
pub fn json_origin(s: &str) -> Option<[f64; 2]> {
    let head = &s[..s.len().min(4096)];
    let rest = &head[head.find("\"origin\"")?..];
    let coords = &rest[rest.find('[')? + 1..rest.find(']')?];
    let mut it = coords.split(',').map(|t| t.trim().parse::<f64>());
    Some([it.next()?.ok()?, it.next()?.ok()?])
}

impl OsmMap {
    /// Dissolve uncontrolled degree-2 pass-through nodes (where one road simply
    /// continues) into the adjacent link's polyline, so a road between two real
    /// junctions is a single link with bends rather than a chain of stub links and
    /// spurious junction boxes. Only merges segments sharing lanes, speed and
    /// layer; controlled nodes and attribute changes are left intact.
    pub fn collapse_pass_through_nodes(&self) -> OsmMap {
        self.collapse_pass_through_nodes_with(&mut Vec::new())
    }

    /// [`collapse_pass_through_nodes`], also rewriting `restrictions` so each
    /// node pair keeps naming its (possibly merged) link: a pair absorbed into
    /// a longer chain takes the chain's endpoints, and a restriction whose via
    /// node dissolved (it gated a single continuation — nothing to restrict)
    /// is dropped.
    pub fn collapse_pass_through_nodes_with(&self, restrictions: &mut Vec<RestrictionSpec>) -> OsmMap {
        use std::collections::VecDeque;
        let pos: HashMap<i64, [f64; 2]> = self.nodes.iter().map(|n| (n.osm_id, [n.x, n.y])).collect();
        // Tombstoned link table with a node→incident-link index, so each candidate is
        // examined in O(degree) and collapses drive off a work-queue — near-linear,
        // instead of re-scanning every link for every node on every pass.
        let mut links: Vec<Option<LinkSpec>> = self.links.iter().cloned().map(Some).collect();
        // Where each tombstoned link's traffic went: the merged link that absorbed
        // it. Chased transitively, this resolves any original link to its final
        // merged identity — how the restriction pairs are kept current.
        let mut redirect: Vec<Option<usize>> = vec![None; links.len()];
        let mut inc: HashMap<i64, Vec<usize>> = HashMap::new();
        for (i, l) in links.iter().enumerate() {
            let l = l.as_ref().unwrap();
            inc.entry(l.from_osm).or_default().push(i);
            inc.entry(l.to_osm).or_default().push(i);
        }
        let uncontrolled: std::collections::HashSet<i64> =
            self.nodes.iter().filter(|n| n.control == MapControl::Uncontrolled && !n.rail_crossing).map(|n| n.osm_id).collect();
        let mut removed: std::collections::HashSet<i64> = Default::default();
        let mut queue: VecDeque<i64> =
            self.nodes.iter().filter(|n| n.control == MapControl::Uncontrolled && !n.rail_crossing).map(|n| n.osm_id).collect();
        while let Some(n) = queue.pop_front() {
            if removed.contains(&n) || !uncontrolled.contains(&n) {
                continue;
            }
            let incident: Vec<usize> =
                inc.get(&n).map_or(Vec::new(), |v| v.iter().copied().filter(|&i| links[i].is_some()).collect());
            let neigh: BTreeSet<i64> = incident
                .iter()
                .map(|&i| {
                    let l = links[i].as_ref().unwrap();
                    if l.from_osm == n { l.to_osm } else { l.from_osm }
                })
                .collect();
            if neigh.len() != 2 {
                continue;
            }
            let mut nb = neigh.iter().copied();
            let (a, b) = (nb.next().unwrap(), nb.next().unwrap());
            let seg = |from: i64, to: i64| {
                incident.iter().copied().find(|&i| {
                    let l = links[i].as_ref().unwrap();
                    l.from_osm == from && l.to_osm == to
                })
            };
            let (a_in, a_out, b_in, b_out) = (seg(a, n), seg(n, a), seg(b, n), seg(n, b));
            let merge: Option<Vec<(Vec<usize>, LinkSpec)>> = match (incident.len(), a_in, a_out, b_in, b_out) {
                (4, Some(ai), Some(ao), Some(bi), Some(bo)) => match (
                    join_pass_through(&links, ai, bo, a, b, n, &pos),
                    join_pass_through(&links, bi, ao, b, a, n, &pos),
                ) {
                    (Some(fwd), Some(rev)) => Some(vec![(vec![ai, bo], fwd), (vec![bi, ao], rev)]),
                    _ => None,
                },
                (2, Some(ai), None, None, Some(bo)) => {
                    join_pass_through(&links, ai, bo, a, b, n, &pos).map(|fwd| vec![(vec![ai, bo], fwd)])
                }
                (2, None, Some(ao), Some(bi), None) => {
                    join_pass_through(&links, bi, ao, b, a, n, &pos).map(|rev| vec![(vec![bi, ao], rev)])
                }
                _ => None,
            };
            if let Some(groups) = merge {
                for (old, l) in groups {
                    let id = links.len();
                    for i in old {
                        links[i] = None;
                        redirect[i] = Some(id);
                    }
                    inc.entry(l.from_osm).or_default().push(id);
                    inc.entry(l.to_osm).or_default().push(id);
                    links.push(Some(l));
                    redirect.push(None);
                }
                removed.insert(n);
                // Re-examine the neighbours: the merge may have changed the joined link's
                // lane count (the ramp-sliver rule), which can make an adjacent
                // pass-through now collapsible where it wasn't. The `uncontrolled` guard
                // on pop keeps this from ever dissolving a real (controlled) junction.
                queue.push_back(a);
                queue.push_back(b);
            }
        }
        if !restrictions.is_empty() {
            let orig: HashMap<(i64, i64), usize> =
                self.links.iter().enumerate().map(|(i, l)| ((l.from_osm, l.to_osm), i)).collect();
            let live = |mut i: usize| {
                while let Some(n) = redirect[i] {
                    i = n;
                }
                i
            };
            restrictions.retain_mut(|r| {
                if removed.contains(&r.from.1) || removed.contains(&r.to.0) {
                    return false;
                }
                let (Some(&f), Some(&t)) = (orig.get(&r.from), orig.get(&r.to)) else {
                    return false;
                };
                let (Some(fl), Some(tl)) = (&links[live(f)], &links[live(t)]) else {
                    return false;
                };
                r.from = (fl.from_osm, fl.to_osm);
                r.to = (tl.from_osm, tl.to_osm);
                true
            });
        }
        let nodes = self.nodes.iter().filter(|n| !removed.contains(&n.osm_id)).cloned().collect();
        OsmMap { nodes, links: links.into_iter().flatten().collect() }
    }

    /// Merge intersections that OSM splits across several nodes a few metres apart
    /// (divided roads, staggered crossings) into one logical junction: cluster
    /// junction nodes joined by a short stub, collapse each cluster to its centroid,
    /// drop the now-internal stubs, and re-point the external approaches. `build`
    /// then forms a single box with one coordinated signal instead of two.
    pub fn merge_split_intersections(&self, cap_extent: bool) -> OsmMap {
        self.merge_split_intersections_with(cap_extent, &mut Vec::new())
    }

    /// [`merge_split_intersections`], also remapping `restrictions` through the
    /// cluster collapse. A restriction whose approach or exit link became
    /// cluster interior (both endpoints merged away) is dropped; a via-way
    /// restriction whose two via ends merged into one node becomes applicable.
    pub fn merge_split_intersections_with(
        &self,
        cap_extent: bool,
        restrictions: &mut Vec<RestrictionSpec>,
    ) -> OsmMap {
        const STUB_MAX: f64 = 25.0;
        // A link this short is junction interior (a turn slot, median crossing or
        // lane-change fragment), so merge it into the junction whatever its
        // endpoints' degree or lane count — otherwise it renders as a stray nub.
        const INTERIOR_MAX: f64 = 12.0;
        // `cap_extent` (experimental) bounds how far a merged *surface* junction may span from
        // its centre, so a divided boulevard's carriageways stay separate and aligned instead of
        // collapsing to one off-centre point that kinks every approach. It fixes the geometry but
        // disrupts flow (the movement model wants complex junctions merged), so it's opt-in.
        // Grade-separated (freeway/ramp) junctions are exempt — splitting those breaks freeways.
        const MAX_RADIUS: f64 = 15.0;
        let pos: HashMap<i64, [f64; 2]> = self.nodes.iter().map(|n| (n.osm_id, [n.x, n.y])).collect();
        // Control by osm id, so the cluster loop looks up each member's control in O(1)
        // instead of a linear `self.nodes.iter().find` — an O(nodes²) scan (≈3 billion on a
        // whole-city map) that native's cache hides but wasm runs ~20× slower, dominating load.
        let control_of: HashMap<i64, MapControl> = self.nodes.iter().map(|n| (n.osm_id, n.control)).collect();
        let rail_of: HashMap<i64, bool> = self.nodes.iter().map(|n| (n.osm_id, n.rail_crossing)).collect();
        let mut neigh: HashMap<i64, BTreeSet<i64>> = HashMap::new();
        for l in &self.links {
            neigh.entry(l.from_osm).or_default().insert(l.to_osm);
            neigh.entry(l.to_osm).or_default().insert(l.from_osm);
        }
        let degree = |id: i64| neigh.get(&id).map_or(0, BTreeSet::len);

        let mut parent: HashMap<i64, i64> = self.nodes.iter().map(|n| (n.osm_id, n.osm_id)).collect();
        if cap_extent {
            // Nodes on a grade-separated road (motorway/trunk/ramp) — exempt from the cap so
            // freeway interchanges keep merging freely.
            let mut grade_sep: std::collections::HashSet<i64> = std::collections::HashSet::new();
            for l in &self.links {
                if RoadKind::from_osm(&l.road_class).is_grade_separated() {
                    grade_sep.insert(l.from_osm);
                    grade_sep.insert(l.to_osm);
                }
            }
            // Mergeable links, shortest first with a stable tie-break so the greedy is deterministic.
            // Only sub-`INTERIOR_MAX` fragments merge here: a longer stub joining two
            // junction nodes (a divided road's median crossing, ~16-19 m on El Camino
            // Real) is a real piece of carriageway whose end nodes are the true
            // crossing points — collapsing them to a centroid kinks every approach.
            // The junction clustering in `build_junctions` already unifies such
            // crossings for signals, box gating, and conflicts, so geometry can keep
            // the split nodes.
            let mut cand: Vec<(f64, i64, i64)> = self
                .links
                .iter()
                .filter_map(|l| {
                    let d = distance(pos[&l.from_osm], pos[&l.to_osm]);
                    (d < INTERIOR_MAX).then_some((d, l.from_osm, l.to_osm))
                })
                .collect();
            cand.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
            // Merge two clusters only if the union still fits within `MAX_RADIUS` of its centroid
            // (unless a grade-separated node is involved, which merges freely).
            let mut members: HashMap<i64, Vec<i64>> = self.nodes.iter().map(|n| (n.osm_id, vec![n.osm_id])).collect();
            for (_, a, b) in cand {
                let (ra, rb) = (uf_find(&mut parent, a), uf_find(&mut parent, b));
                if ra == rb {
                    continue;
                }
                let capped = !grade_sep.contains(&a) && !grade_sep.contains(&b);
                let within = !capped || {
                    let (ma, mb) = (&members[&ra], &members[&rb]);
                    let n = (ma.len() + mb.len()) as f64;
                    let (mut cx, mut cy) = (0.0, 0.0);
                    for &m in ma.iter().chain(mb) {
                        cx += pos[&m][0];
                        cy += pos[&m][1];
                    }
                    let centre = [cx / n, cy / n];
                    ma.iter().chain(mb).all(|&m| distance(centre, pos[&m]) <= MAX_RADIUS)
                };
                if within {
                    parent.insert(ra, rb);
                    let moved = members.remove(&ra).unwrap();
                    members.get_mut(&rb).unwrap().extend(moved);
                }
            }
        } else {
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
                let c = control_of[&m];
                if control_rank(c) > control_rank(control) {
                    control = c;
                }
            }
            let k = members.len() as f64;
            let rail_crossing = members.iter().any(|m| rail_of.get(m).copied().unwrap_or(false));
            nodes.push(NodeSpec { osm_id: rep_id, x: cx / k, y: cy / k, control, rail_crossing });
        }
        // `clusters` is a std HashMap, so its `.values()` order is randomly seeded per
        // process. The output node order fixes every NodeId (and downstream: junction cluster
        // ids, signal phase/green-mask assignment), so an unsorted emit makes the whole build
        // — and thus the sim — non-reproducible run to run. Sort by the stable rep osm_id.
        nodes.sort_by_key(|n| n.osm_id);

        let rep_pos: HashMap<i64, [f64; 2]> = nodes.iter().map(|n| (n.osm_id, [n.x, n.y])).collect();
        let mut links = Vec::new();
        for l in &self.links {
            let (from_osm, to_osm) = (rep_of[&l.from_osm], rep_of[&l.to_osm]);
            if from_osm == to_osm {
                continue;
            }
            let mut geometry = l.geometry.clone();
            // A surface link keeps its true end position as a geometry vertex when its
            // node moves to a cluster centroid: the road runs straight to the real
            // crossing point and only the final jag (clipped inside the junction box)
            // reaches the merged node, instead of the whole last segment veering.
            if RoadKind::from_osm(&l.road_class).is_surface() {
                if distance(pos[&l.from_osm], rep_pos[&from_osm]) > 2.0 {
                    geometry.insert(0, pos[&l.from_osm]);
                }
                if distance(pos[&l.to_osm], rep_pos[&to_osm]) > 2.0 {
                    geometry.push(pos[&l.to_osm]);
                }
            }
            links.push(LinkSpec { from_osm, to_osm, geometry, ..l.clone() });
        }
        restrictions.retain_mut(|r| {
            let rep = |id: i64| rep_of.get(&id).copied();
            let (Some(f0), Some(f1), Some(t0), Some(t1)) =
                (rep(r.from.0), rep(r.from.1), rep(r.to.0), rep(r.to.1))
            else {
                return false;
            };
            r.from = (f0, f1);
            r.to = (t0, t1);
            f0 != f1 && t0 != t1
        });
        OsmMap { nodes, links }
    }

    pub fn build(&self) -> Network {
        self.build_with_restrictions(&[])
    }

    /// [`build`] honoring OSM turn restrictions: each restriction filters the
    /// movement wiring at its via node — `no_*` removes the named exit from the
    /// approach link, `only_*` removes every other exit. Fail-open: a
    /// restriction that would leave an approach with no exit at all (a mapping
    /// error, or exits lost to bbox clipping) is ignored, because stranding
    /// every driver on the link models nothing real.
    pub fn build_with_restrictions(&self, restrictions: &[RestrictionSpec]) -> Network {
        self.build_with_progress(restrictions, &mut |_, _, _| {})
    }

    /// [`build_with_restrictions`], reporting `(stage, done, total)` at the
    /// build's internal boundaries (and inside its two longest loops) so a
    /// county-scale load can show live progress instead of one silent stall.
    /// `total == 0` marks an indeterminate boundary.
    pub fn build_with_progress(
        &self,
        restrictions: &[RestrictionSpec],
        cb: &mut dyn FnMut(&str, u32, u32),
    ) -> Network {
        let mut net = Network::default();
        let mut id_of: HashMap<i64, NodeId> = HashMap::new();

        for spec in &self.nodes {
            id_of.insert(spec.osm_id, NodeId(net.nodes.len() as u32));
            net.nodes.push(Node {
                position: [spec.x, spec.y],
                control: NodeControl::Uncontrolled,
                rail_crossing: spec.rail_crossing,
            });
        }

        // Cap "lane fans": a link far wider than every road it connects to isn't real capacity
        // but an artifact — a toll plaza modelled as many lanes (the Golden Gate toll plaza is
        // 8 lanes between 1-lane segments). Left alone it collapses 8→1 downstream and dog-piles
        // the merge. Pull such an outlier to its neighbours' width + a small margin; a genuinely
        // wide road (whose neighbours are also wide, e.g. a 6-lane arterial) is untouched.
        const FAN_MARGIN: u32 = 2;
        let mut incident: HashMap<i64, Vec<(u32, usize)>> = HashMap::new();
        for (i, l) in self.links.iter().enumerate() {
            incident.entry(l.from_osm).or_default().push((l.lanes, i));
            incident.entry(l.to_osm).or_default().push((l.lanes, i));
        }
        let capped: Vec<u32> = self
            .links
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let others_max = incident[&l.from_osm]
                    .iter()
                    .chain(&incident[&l.to_osm])
                    .filter(|&&(_, j)| j != i)
                    .map(|&(ln, _)| ln)
                    .max()
                    .unwrap_or(l.lanes);
                l.lanes.min(others_max + FAN_MARGIN).max(1)
            })
            .collect();

        // Fill lane "pinches" on the freeway mainline. OSM splits a motorway into segments and
        // often drops the lane tag on some (defaulting to 1), so a wide freeway funnels through
        // 1-lane artifacts — the Golden Gate toll approach runs 3->1->1->1->8->1->1->4. Follow
        // each mainline link's *straightest* continuation up- and downstream (across the toll's
        // fan-out/fan-in, which a strict single-successor chain would stop at) and take the
        // widest segment reached on each side; a link narrower than both flanks is a wedged
        // artifact, widened to the narrower flank. A genuine lane drop reaches nothing wider on
        // one side, so it stays. Ramps (motorway_link) are excluded — a 1-lane ramp is real.
        let mainline: Vec<bool> =
            self.links.iter().map(|l| RoadKind::from_osm(&l.road_class) == RoadKind::Freeway).collect();
        let pos: HashMap<i64, [f64; 2]> = self.nodes.iter().map(|n| (n.osm_id, [n.x, n.y])).collect();
        let dir = |a: [f64; 2], b: [f64; 2]| {
            let d = [b[0] - a[0], b[1] - a[1]];
            let n = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-9);
            [d[0] / n, d[1] / n]
        };
        let mut dep = vec![[0.0; 2]; self.links.len()];
        let mut arr = vec![[0.0; 2]; self.links.len()];
        for (i, l) in self.links.iter().enumerate() {
            let (fp, tp) = (pos[&l.from_osm], pos[&l.to_osm]);
            dep[i] = dir(fp, *l.geometry.first().unwrap_or(&tp));
            arr[i] = dir(*l.geometry.last().unwrap_or(&fp), tp);
        }
        let mut leaving: HashMap<i64, Vec<usize>> = HashMap::new();
        let mut arriving: HashMap<i64, Vec<usize>> = HashMap::new();
        for (i, l) in self.links.iter().enumerate() {
            leaving.entry(l.from_osm).or_default().push(i);
            arriving.entry(l.to_osm).or_default().push(i);
        }
        // The straightest mainline continuation downstream (tnext) and upstream (tprev) of each
        // link. Straightest, not single: at the toll's diverge/merge this stays on the mainline
        // (a >45° turn onto a ramp is rejected) instead of dead-ending at the fan.
        let straightest = |cands: &[usize], into: [f64; 2], reverse_of: i64, forward: bool| {
            let mut best: Option<(f64, usize)> = None;
            for &j in cands {
                if !mainline[j] {
                    continue;
                }
                let other = if forward { self.links[j].to_osm } else { self.links[j].from_osm };
                if other == reverse_of {
                    continue;
                }
                let out = if forward { dep[j] } else { arr[j] };
                let d = into[0] * out[0] + into[1] * out[1];
                if d > 0.7 && best.is_none_or(|(b, _)| d > b) {
                    best = Some((d, j));
                }
            }
            best.map(|(_, j)| j)
        };
        let tnext: Vec<Option<usize>> = (0..self.links.len())
            .map(|i| {
                if !mainline[i] {
                    return None;
                }
                let l = &self.links[i];
                straightest(leaving.get(&l.to_osm).map_or(&[][..], |v| v), arr[i], l.from_osm, true)
            })
            .collect();
        let tprev: Vec<Option<usize>> = (0..self.links.len())
            .map(|i| {
                if !mainline[i] {
                    return None;
                }
                let l = &self.links[i];
                straightest(arriving.get(&l.from_osm).map_or(&[][..], |v| v), dep[i], l.to_osm, false)
            })
            .collect();
        const MAX_HOPS: usize = 16;
        let flank_max = |start: usize, chain: &[Option<usize>]| {
            let (mut cur, mut m) = (start, 0u32);
            for _ in 0..MAX_HOPS {
                match chain[cur] {
                    Some(n) if n != start => {
                        m = m.max(capped[n]);
                        cur = n;
                    }
                    _ => break,
                }
            }
            m
        };
        // Only fill a *dramatic* pinch — at most half the surrounding width. A 6->5 lane drop
        // where a ramp genuinely consumes the lane is left alone; a 3->1 collapse with no ramp
        // to explain it (the toll) is the artifact we widen.
        let eff_lanes: Vec<u32> = (0..self.links.len())
            .map(|i| {
                if !mainline[i] {
                    return capped[i];
                }
                let fill = flank_max(i, &tprev).min(flank_max(i, &tnext));
                if fill >= capped[i] * 2 {
                    fill
                } else {
                    capped[i]
                }
            })
            .collect();

        for (li, spec) in self.links.iter().enumerate() {
            if li % 8192 == 0 {
                cb("links", li as u32, self.links.len() as u32);
            }
            let lanes = eff_lanes[li];
            let from = id_of[&spec.from_osm];
            let to = id_of[&spec.to_osm];
            let mut polyline = vec![net.nodes[from.idx()].position];
            polyline.extend(spec.geometry.iter().copied());
            polyline.push(net.nodes[to.idx()].position);
            let polyline = fillet_polyline(polyline, FILLET_RADIUS);
            let length = polyline.windows(2).map(|w| distance(w[0], w[1])).sum();
            let link_id = LinkId(net.links.len() as u32);
            let lane_start = LaneId(net.lanes.len() as u32);
            for i in 0..lanes {
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
                lane_count: lanes,
                layer: spec.layer,
                kind: RoadKind::from_osm(&spec.road_class),
                motorway: spec.road_class == "motorway",
            });
            net.polylines.push(polyline);
            net.link_names.push(spec.name.clone());
            net.link_refs.push(spec.highway_ref.clone());
            net.link_turn_lanes.push(spec.turn_lanes.clone());
            net.link_hov_lanes.push(spec.hov_lanes.clone());
            net.link_aadt.push(spec.aadt);
            net.link_res_weight.push(spec.res_weight);
            net.link_attr_weight.push(spec.attr_weight);
        }

        center_oneway_axes(&mut net);
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
        // Index links by their upstream node so each approach scans only the roads
        // actually leaving its downstream node — O(links · degree) instead of O(links²),
        // the dominant build cost on a city-sized map.
        let mut links_from: Vec<Vec<usize>> = vec![Vec::new(); net.nodes.len()];
        for (li, l) in net.links.iter().enumerate() {
            links_from[l.from.idx()].push(li);
        }
        // Turn restrictions as (approach link → exit link) filters at their via
        // node, keyed by link index. Node-pair matching covers parallel links —
        // duplicates of one carriageway are the same street, so the sign binds
        // them all.
        let mut link_at: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        for (i, l) in self.links.iter().enumerate() {
            link_at.entry((l.from_osm, l.to_osm)).or_default().push(i);
        }
        let mut only_of: HashMap<usize, HashSet<usize>> = HashMap::new();
        let mut banned: HashSet<(usize, usize)> = HashSet::new();
        for r in restrictions {
            if r.from.1 != r.to.0 {
                continue; // a via-way restriction whose interior never merged: spans two junctions
            }
            let (Some(ins), Some(outs)) = (link_at.get(&r.from), link_at.get(&r.to)) else { continue };
            for &i in ins {
                for &o in outs {
                    if r.only {
                        only_of.entry(i).or_default().insert(o);
                    } else {
                        banned.insert((i, o));
                    }
                }
            }
        }
        let mut movements: Vec<Movement> = Vec::new();
        for in_li in 0..net.links.len() {
            if in_li % 8192 == 0 {
                cb("movements", in_li as u32, net.links.len() as u32);
            }
            let link = net.links[in_li];
            let node = link.to;
            let arr = net.arrival_dir(LinkId(in_li as u32));
            // Valid onward links (skip U-turns by node identity and by geometry),
            // as `(link, signed turn angle)`; ascending = rightmost turn first.
            let mut onward: Vec<(usize, f64)> = Vec::new();
            for &out_li in &links_from[node.idx()] {
                let out = net.links[out_li];
                if out.to == link.from {
                    continue; // a U-turn back onto the road we came from
                }
                let dep = net.departure_dir(LinkId(out_li as u32));
                let dot = arr[0] * dep[0] + arr[1] * dep[1];
                if dot < -0.85 {
                    continue; // ~>148°: a near-reversal onto the opposing carriageway
                }
                onward.push((out_li, (arr[0] * dep[1] - arr[1] * dep[0]).atan2(dot)));
            }
            // Apply this approach's turn restrictions (`only_*` first — it is the
            // stronger claim — then `no_*`), each fail-open so no filter may
            // strand the approach with zero exits.
            if let Some(allowed) = only_of.get(&in_li) {
                let kept: Vec<(usize, f64)> =
                    onward.iter().copied().filter(|&(o, _)| allowed.contains(&o)).collect();
                if !kept.is_empty() {
                    onward = kept;
                }
            }
            if !banned.is_empty() {
                let kept: Vec<(usize, f64)> =
                    onward.iter().copied().filter(|&(o, _)| !banned.contains(&(in_li, o))).collect();
                if !kept.is_empty() {
                    onward = kept;
                }
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
                // Off-ramps claim the curb-most lanes as their exit — and *only* their
                // exit. Past the gore the ramp is physically separated from the mainline,
                // so an exit lane no longer continues on the freeway: a k-lane ramp takes
                // the curb-most k lanes, exit-only.
                let mut exit_only = vec![false; n];
                for i in 0..m {
                    if ramp_exit(i) {
                        let rl = net.links[onward[i].0].lane_count as usize;
                        for k in 0..rl.min(n) {
                            lane_exits[n - 1 - k].insert(i);
                            exit_only[n - 1 - k] = true;
                        }
                    }
                }
                // The mainline continuation(s) fan left→right across the *remaining*
                // through lanes, so each continuation lane has exactly one feeder — no
                // lane-drop choke. A through driver caught in an exit lane reaches the
                // freeway by changing lanes upstream (a mandatory change before the gore),
                // not by a movement that merges three lanes into one at the node.
                let through: Vec<usize> = (0..n).filter(|&k| !exit_only[k]).collect();
                let tn = through.len();
                if tn == 0 {
                    // The ramp(s) claimed every lane: keep them all continuing so no
                    // through car is stranded.
                    for (k, exits) in lane_exits.iter_mut().enumerate() {
                        exits.insert(mains[nearest(k, n - 1, mm - 1)]);
                    }
                } else {
                    for (j, &i) in mains.iter().enumerate() {
                        lane_exits[through[nearest(j, mm - 1, tn - 1)]].insert(i);
                    }
                    for (idx, &k) in through.iter().enumerate() {
                        lane_exits[k].insert(mains[nearest(idx, tn - 1, mm - 1)]);
                    }
                }
            } else if m > 0 {
                match turn_lane_exits(&net, in_li, n, &onward) {
                    Some(sets) => lane_exits = sets,
                    None => {
                        for i in 0..m {
                            lane_exits[nearest(i, m - 1, n - 1)].insert(i); // every exit served
                        }
                        for (k, exits) in lane_exits.iter_mut().enumerate() {
                            exits.insert(nearest(k, n - 1, m - 1)); // every lane serves its nearest
                        }
                    }
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
                    } else if let Some(t) = through_targets(&net, onward[exit_i].0, onward[exit_i].1) {
                        // A through arrival lands on the receiver's *through-marked* lanes
                        // (its `turn:lanes`), so a widening that opens a turn pocket feeds
                        // the pocket only from turning traffic. A sender with its own tag
                        // maps by rank *within its through set* — its through lanes continue
                        // onto the receiver's through lanes 1:1, and a tagged turn lane
                        // continuing anyway (its exit is at a later node of the cluster)
                        // holds its lateral index, landing in the receiver's matching bay
                        // instead of swerving onto a through lane mid-box.
                        match through_marked_lanes(&net, in_li) {
                            Some(s) => match s.iter().position(|&x| x == k as u32) {
                                Some(rank) => t[rank.min(t.len() - 1)],
                                None => (k as u32).min(out.lane_count - 1),
                            },
                            None => t[(k).min(t.len() - 1)],
                        }
                    } else {
                        // A lane drop maps excess lanes onto the curb (the rightmost exit lane
                        // ends and merges left — a realistic single-lane drop). The pathological
                        // *multi*-lane drop is the toll-plaza fan, which the lane-fan cap in
                        // `build` pulls down first, so it never reaches here as an 8→1 dog-pile.
                        // A tagged sender's through lanes count from their *rank*, so a
                        // pocket-flanked core narrows onto an untagged receiver from its
                        // median edge rather than dog-piling the curb lane.
                        let rank = through_marked_lanes(&net, in_li)
                            .and_then(|s| s.iter().position(|&x| x == k as u32))
                            .unwrap_or(k);
                        (rank as u32).min(out.lane_count - 1)
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
        spread_merge_feeders(&mut net);
        untangle_parallel_movements(&mut net);
        assign_turn_pockets(&mut net);
        retarget_pocket_landings(&mut net);
        untangle_parallel_movements(&mut net);
        align_through_seams(&mut net);
        cb("lane-bounds", 0, 0);
        net.build_lane_bounds();
        stitch_seam_bounds(&mut net);
        enforce_mouth_ordering(&mut net);
        cb("interiors", 0, 0);
        net.build_interiors();

        let mut plans = relocate_signals_to_junctions(&net, &self.nodes);
        // Fixed (non-signal) controls now; the signal programs are built after the
        // junctions and cross-node conflicts exist, so each multi-node junction is
        // timed as one coordinated signal rather than several independent ones.
        for i in 0..self.nodes.len() {
            if plans[i].is_none() {
                net.nodes[i].control = match self.nodes[i].control {
                    MapControl::Stop => NodeControl::Stop,
                    MapControl::Yield => NodeControl::Yield,
                    _ => NodeControl::Uncontrolled,
                };
            }
        }
        net.build_hov_lanes();
        cb("junctions", 0, 0);
        net.build_junctions();
        relax_junction_interior_storage(&mut net);
        // Stop control is a property of the *intersection*: OSM surveys the sign
        // per approach, so a multi-node cluster ends up with Stop on some member
        // nodes and Uncontrolled on the rest — and the uncontrolled street then
        // streams through the "all-way stop" while the signed street waits
        // forever. Promote: any unsignalized cluster with a Stop member is
        // stop-controlled at every member.
        let mut touches_freeway = vec![false; net.nodes.len()];
        for l in &net.links {
            if matches!(l.kind, RoadKind::Freeway | RoadKind::Ramp) {
                touches_freeway[l.from.idx()] = true;
                touches_freeway[l.to.idx()] = true;
            }
        }
        for ji in 0..net.junctions.len() {
            let members = net.junctions[ji].nodes.clone();
            let signalized =
                members.iter().any(|&n| matches!(net.nodes[n.idx()].control, NodeControl::Signalized(_)));
            let any_stop = members.iter().any(|&n| matches!(net.nodes[n.idx()].control, NodeControl::Stop));
            if !signalized && any_stop {
                for &n in &members {
                    // The stop sign is an at-grade artifact: a freeway seam node
                    // clustered nearby (a frontage stop beside the gateway) must
                    // not put a phantom stop on the mainline.
                    if !touches_freeway[n.idx()] {
                        net.nodes[n.idx()].control = NodeControl::Stop;
                    }
                }
            }
        }
        // Nodes whose stop control came from the node level (a sign surveyed on
        // the junction node, or the all-way cluster promote above): every
        // approach there serves a line. Snapshotted before the per-approach
        // signs upgrade any further nodes, because a link-signed node lines
        // *only* its signed approaches — that is what makes a two-way stop.
        let node_stop_all: Vec<bool> =
            net.nodes.iter().map(|n| matches!(n.control, NodeControl::Stop)).collect();
        net.link_signs = self.links.iter().map(|l| l.sign).collect();
        for li in 0..net.links.len() {
            let n = net.links[li].to.idx();
            if touches_freeway[n] {
                continue;
            }
            match (net.link_signs[li], net.nodes[n].control) {
                (LinkSign::Stop, NodeControl::Uncontrolled | NodeControl::Yield) => {
                    net.nodes[n].control = NodeControl::Stop;
                }
                (LinkSign::Yield, NodeControl::Uncontrolled) => {
                    net.nodes[n].control = NodeControl::Yield;
                }
                _ => {}
            }
        }
        net.link_stop_line = (0..net.links.len())
            .map(|li| {
                let to = net.links[li].to.idx();
                matches!(net.nodes[to].control, NodeControl::Stop)
                    && (node_stop_all[to] || net.link_signs[li] == LinkSign::Stop)
            })
            .collect();
        net.node_all_way = {
            let mut aw: Vec<bool> =
                net.nodes.iter().map(|n| matches!(n.control, NodeControl::Stop)).collect();
            for (li, l) in net.links.iter().enumerate() {
                if !net.link_stop_line[li] {
                    aw[l.to.idx()] = false;
                }
            }
            aw
        };
        cb("conflicts", 0, 0);
        net.build_cross_junction_conflicts();
        net.build_conflict_index();
        // Real traffic engineering signalizes crossings of two major roads; OSM
        // frequently omits the `traffic_signals` tag on them, leaving an at-grade
        // major×major crossing uncontrolled. Modelled uncontrolled, both major
        // streams gap-accept against each other and starve under heavy flow — a
        // false gridlock. Promote those (only) to a default-timed signal here,
        // where the conflict index exists to identify a genuine crossing.
        cb("signals", 0, 0);
        promote_major_crossings(&net, &mut plans);
        coordinate_junction_signals(&mut net, &plans);
        coordinate_green_waves(&mut net);
        // Render-only grade layering: infer overpass/underpass occlusion for
        // crossings OSM left untagged, so a road passing over another is drawn on
        // top of it. Purely visual (see `Network::render_layer`); after all
        // geometry is final.
        net.build_render_layers();
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
    // The single onward / incoming node, captured in the degree pass so a pass-through node
    // (in/out-degree 1) resolves its neighbours in O(1) instead of scanning every link — an
    // O(nodes · links) walk on a city map.
    let (mut down_of, mut up_of) = (vec![usize::MAX; n], vec![usize::MAX; n]);
    for link in &net.links {
        outdeg[link.from.idx()] += 1;
        indeg[link.to.idx()] += 1;
        down_of[link.from.idx()] = link.to.idx();
        up_of[link.to.idx()] = link.from.idx();
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
        let downstream = (down_of[i] != usize::MAX).then_some(down_of[i]);
        let upstream = (up_of[i] != usize::MAX).then_some(up_of[i]);
        if let Some(j) = downstream.filter(|&j| is_junction(j)).or(upstream.filter(|&j| is_junction(j))) {
            plans[j].get_or_insert(plan);
            plans[i] = None;
        }
    }
    plans
}

/// Per-lane exit sets from the link's OSM `turn:lanes` (lanes listed left→right,
/// matching lane 0 at the median): each marked direction claims the exits whose
/// signed-angle turn class matches; unmarked (`none`) and merge lanes carry the
/// through class; `reverse` has no modelled movement, so a `reverse;left` lane
/// serves its left. Data gaps degrade to geometry: an absent tag or a lane-count
/// mismatch falls back to the angular-slice channelization, a marked direction
/// with no matching exit takes the lane's geometric nearest, and every exit keeps
/// a serving lane (a side street too minor for an arrow still connects).
fn turn_lane_exits(
    net: &Network,
    in_li: usize,
    n: usize,
    onward: &[(usize, f64)],
) -> Option<Vec<std::collections::BTreeSet<usize>>> {
    let spec = net.link_turn_lanes.get(in_li)?.as_str();
    if spec.is_empty() {
        return None;
    }
    let entries: Vec<&str> = spec.split('|').collect();
    if entries.len() != n {
        return None;
    }
    let m = onward.len();
    let class_of = |ang: f64| {
        if ang > 0.5 {
            TurnType::Left
        } else if ang < -0.5 {
            TurnType::Right
        } else {
            TurnType::Through
        }
    };
    let lane_to_exit = |k: usize| if n <= 1 { 0 } else { ((k as f64) * ((m - 1) as f64) / ((n - 1) as f64)).round() as usize };
    let exit_to_lane = |i: usize| if m <= 1 { 0 } else { ((i as f64) * ((n - 1) as f64) / ((m - 1) as f64)).round() as usize };
    let mut sets: Vec<std::collections::BTreeSet<usize>> = vec![Default::default(); n];
    for (k, entry) in entries.iter().enumerate() {
        for part in entry.split(';') {
            let want = match part.trim() {
                "left" | "slight_left" | "sharp_left" => Some(TurnType::Left),
                "right" | "slight_right" | "sharp_right" => Some(TurnType::Right),
                "through" | "none" | "" | "merge_to_left" | "merge_to_right" => Some(TurnType::Through),
                _ => None,
            };
            if let Some(t) = want {
                sets[k].extend((0..m).filter(|&i| class_of(onward[i].1) == t));
            }
        }
        if sets[k].is_empty() {
            sets[k].insert(lane_to_exit(k));
        }
    }
    for i in 0..m {
        if !sets.iter().any(|s| s.contains(&i)) {
            sets[exit_to_lane(i)].insert(i);
        }
    }
    Some(sets)
}

/// A link's through-marked lane indices from its own `turn:lanes` tag. `None`
/// when the tag is absent, malformed, or marks no through lane.
fn through_marked_lanes(net: &Network, li: usize) -> Option<Vec<u32>> {
    let spec = net.link_turn_lanes.get(li)?.as_str();
    if spec.is_empty() {
        return None;
    }
    let entries: Vec<&str> = spec.split('|').collect();
    if entries.len() != net.links[li].lane_count as usize {
        return None;
    }
    let through: Vec<u32> = entries
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            e.split(';').any(|p| matches!(p.trim(), "through" | "none" | "" | "merge_to_left" | "merge_to_right"))
        })
        .map(|(k, _)| k as u32)
        .collect();
    (!through.is_empty()).then_some(through)
}

/// The receiving link's through-marked lane indices, for landing a *through*
/// arrival (`|ang| ≤ 0.5`) on its through lanes: a receiver whose `turn:lanes`
/// opens turn pockets keeps them for turning traffic. `None` (no tag, no
/// through-marked lanes, or a turning arrival) keeps the plain index mapping.
fn through_targets(net: &Network, out_li: usize, ang: f64) -> Option<Vec<u32>> {
    if ang.abs() > 0.5 {
        return None;
    }
    through_marked_lanes(net, out_li)
}

/// Re-land through movements that were wired into a turn pocket. Pockets are
/// assigned *after* movements exist, so an untagged receiver's bay lane can be
/// holding a plain index-mapped through stream — and a bay is merged shut at
/// the seam, so that stream would sidestep a lane width onto its neighbour's
/// centreline (two uncoupled streams sharing one physical line). Landing it on
/// the nearest genuine through lane instead makes the confluence an explicit,
/// modeled merge; bay users reach the pocket the way real drivers do, by
/// changing into it where it opens. A *tagged* turn lane continuing across a
/// cluster into its matching bay keeps its lateral index (the stage-2 bay→bay
/// chains); only tagged-through and untagged senders re-land.
fn retarget_pocket_landings(net: &mut Network) {
    let mut moves: Vec<(usize, LaneId)> = Vec::new();
    for (mi, mv) in net.movements.iter().enumerate() {
        let to = net.lane(mv.to_lane);
        if to.pocket_taper <= 0.0
            || net.lane(mv.from_lane).pocket_taper > 0.0
            || net.movement_turn(MovementId(mi as u32)) != TurnType::Through
        {
            continue;
        }
        let from = net.lane(mv.from_lane);
        if let Some(marked) = through_marked_lanes(net, from.link.idx()) {
            if !marked.contains(&from.index_in_link) {
                continue; // a tagged turn lane continuing into its matching bay
            }
        }
        let link = net.link(to.link);
        let nearest = (0..link.lane_count)
            .filter(|&k| net.lane(LaneId(link.lane_start.0 + k)).pocket_taper <= 0.0)
            .min_by_key(|&k| k.abs_diff(to.index_in_link));
        if let Some(k) = nearest {
            moves.push((mi, LaneId(link.lane_start.0 + k)));
        }
    }
    for (mi, to) in moves {
        net.movements[mi].to_lane = to;
    }
}

/// Make each movement group between one link pair *monotone*: the movements from
/// link A to link B, ordered by their approach lane, land on non-decreasing exit
/// lanes. Any wiring pass can leave a crossed pair (lane 3→4 beside lane 4→3),
/// and two side-by-side cars would then swap lanes across each other inside the
/// box — parallel paths through a junction never cross on a real road. Keeps
/// each group's multiset of exit lanes; only the pairing is straightened.
fn untangle_parallel_movements(net: &mut Network) {
    let mut groups: HashMap<(u32, u32, u32), Vec<usize>> = HashMap::new();
    for (mi, mv) in net.movements.iter().enumerate() {
        let key = (net.lane(mv.from_lane).link.0, net.lane(mv.to_lane).link.0, mv.node.0);
        groups.entry(key).or_default().push(mi);
    }
    for members in groups.values_mut() {
        if members.len() < 2 {
            continue;
        }
        members.sort_by_key(|&mi| net.movements[mi].from_lane.0);
        let mut tos: Vec<LaneId> = members.iter().map(|&mi| net.movements[mi].to_lane).collect();
        tos.sort_by_key(|l| l.0);
        for (&mi, &to) in members.iter().zip(&tos) {
            net.movements[mi].to_lane = to;
        }
    }
}

/// Flag dedicated turn lanes as physical turn pockets. A contiguous block of
/// lanes serving only one turn direction — lefts from the median side, rights
/// from the curb (a dual left is a two-lane block) — qualifies when the first
/// lane beyond the block is a through lane it peels away from and the approach
/// is long enough to hold a bay. Each bay lane gets a taper, so
/// [`Network::lane_lateral_offset`] opens the pocket near the stop line and
/// merges it into the through lane upstream — turners queue in the bay, not the
/// through lane, and the widened approach renders like a real intersection.
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
        let dedicated = |lane_idx: u32, want: TurnType| -> bool {
            let lane_id = link.lane_start.0 + lane_idx;
            let ts = turns(lane_id);
            net.lanes[lane_id as usize].length >= MIN_LEN && !ts.is_empty() && ts.iter().all(|&t| t == want)
        };
        for (start, step, want) in [(0i64, 1i64, TurnType::Left), (link.lane_count as i64 - 1, -1, TurnType::Right)] {
            let mut block = Vec::new();
            let mut idx = start;
            while (0..link.lane_count as i64).contains(&idx) && dedicated(idx as u32, want) {
                block.push(idx as u32);
                idx += step;
            }
            if block.is_empty() || !(0..link.lane_count as i64).contains(&idx) {
                continue; // no bay, or the whole link turns (a turn roadway, not a pocket)
            }
            if !turns(link.lane_start.0 + idx as u32).contains(&TurnType::Through) {
                continue; // the bay must peel off a through lane, not another pocket
            }
            pockets.extend(block.into_iter().map(|k| link.lane_start.0 + k));
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
fn assign_signal_program(
    net: &mut Network,
    nodes: &[NodeId],
    plan: SignalPlan,
    feeders: &[Vec<MovementId>],
    movements_by_node: &[Vec<MovementId>],
) -> NodeControl {
    // Group movements by (approach link, is-left); non-left groups first so the
    // greedy assignment pairs opposing throughs before protecting lefts. When
    // `nodes` is a whole junction cluster the groups span its member nodes, so the
    // conflict-phased program coordinates them: through movements across the junction
    // never conflict, land in one phase, and go green together (progression).
    let junction_movements: Vec<MovementId> =
        nodes.iter().flat_map(|n| movements_by_node[n.idx()].iter().copied()).collect();
    let mut group_movements: Vec<Vec<MovementId>> = Vec::new();
    let mut left_group: Vec<bool> = Vec::new();
    let mut group_link: Vec<u32> = Vec::new();
    let mut key_index: HashMap<(u32, bool), usize> = HashMap::new();
    for want_left in [false, true] {
        for &mid in &junction_movements {
            let mv = net.movement(mid);
            let is_left = net.movement_turn(mid) == TurnType::Left;
            if is_left != want_left {
                continue;
            }
            // Key by the *entry* approach so a whole through-path across the junction
            // shares one group and goes green together — but never fold a movement into
            // a group it conflicts with (a phase can't split movements within a group),
            // so a mis-traced or genuinely crossing movement gets its own group instead.
            let link = net.entry_link(mv.from_lane, feeders).0;
            let idx = match key_index.get(&(link, is_left)).copied() {
                Some(i) if !group_movements[i].iter().any(|&x| net.movements_conflict(mid, x)) => i,
                _ => {
                    group_movements.push(Vec::new());
                    left_group.push(is_left);
                    group_link.push(link);
                    let i = group_movements.len() - 1;
                    key_index.entry((link, is_left)).or_insert(i);
                    i
                }
            };
            group_movements[idx].push(mid);
        }
    }
    if group_movements.is_empty() || group_movements.len() > 64 {
        return NodeControl::Uncontrolled; // no movements, or too many groups for a 64-bit phase mask
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
    let mut phase_mask: Vec<u64> = phase_groups
        .iter()
        .map(|gs| gs.iter().fold(0u64, |m, &g| m | (1u64 << g)))
        .collect();
    for g in 0..group_movements.len() {
        if !left_group[g] {
            continue;
        }
        let through = (0..group_movements.len()).find(|&h| !left_group[h] && group_link[h] == group_link[g]);
        if let Some(p) = through.and_then(|h| phase_groups.iter().position(|gs| gs.contains(&h))) {
            phase_mask[p] |= 1u64 << g;
        }
    }
    // Pedestrian green floor where the land-use pass marks street activity
    // (shops/jobs around the node): the walk runs parallel to a through phase,
    // so that phase must hold green ≥ walk interval + crossing the widest road
    // it runs beside at 1.1 m/s (the MUTCD clearance speed). Downtown cycles
    // lengthen toward real signal timing; quiet residential crossings don't.
    let commercial = group_link.iter().any(|&l| net.link_attr_weight(LinkId(l)) > 1.05);
    let phases = phase_groups
        .iter()
        .enumerate()
        .map(|(p, gs)| {
            let mut green = if gs.iter().all(|&g| left_group[g]) {
                (plan.green_secs * 0.45).max(6.0)
            } else {
                plan.green_secs
            };
            if commercial && !gs.iter().all(|&g| left_group[g]) {
                let crossing = group_link
                    .iter()
                    .enumerate()
                    .filter(|(h, _)| !gs.contains(h))
                    .map(|(_, &l)| net.link(LinkId(l)).lane_count as f64 * LANE_WIDTH * 2.0)
                    .fold(0.0, f64::max);
                green = green.max(5.0 + crossing / 1.1);
            }
            let phase_movements: Vec<MovementId> =
                gs.iter().flat_map(|&g| group_movements[g].iter().copied()).collect();
            let (yellow, all_red) = change_and_clearance_intervals(net, &phase_movements);
            Phase::with_clearance(phase_mask[p], green, yellow, all_red)
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

/// Signalize uncontrolled junctions where two *different* major roads (expressway
/// or arterial) actually cross — a conflict between streams whose approaches are
/// both major and on distinct streets. Such a crossing left uncontrolled starves
/// (each major stream perpetually gap-accepting against the other); real ones are
/// signalized. Conservative by construction: a major road meeting a minor one (one
/// major approach) keeps its priority/gap-acceptance, an already-controlled
/// junction (signal/stop/yield) is untouched, and a mere merge/continuation (no
/// cross-conflict) is skipped. Writes a default-timed plan the signal builder then
/// programs and coordinates like any other.
fn promote_major_crossings(net: &Network, plans: &mut [Option<SignalPlan>]) {
    const GREEN: f64 = 20.0;
    const YELLOW: f64 = 3.5;
    let mut mvs_by_node: Vec<Vec<MovementId>> = vec![Vec::new(); net.nodes.len()];
    for (m, mv) in net.movements.iter().enumerate() {
        mvs_by_node[mv.node.idx()].push(MovementId(m as u32));
    }
    let is_major = |mid: MovementId| {
        matches!(net.link(net.lane(net.movement(mid).from_lane).link).kind, RoadKind::Expressway | RoadKind::Arterial)
    };
    let approach = |mid: MovementId| net.lane(net.movement(mid).from_lane).link;
    for ji in 0..net.junctions.len() {
        let members = &net.junctions[ji].nodes;
        if members.iter().any(|&n| {
            plans[n.idx()].is_some() || matches!(net.nodes[n.idx()].control, NodeControl::Stop | NodeControl::Yield)
        }) {
            continue; // already controlled — leave it
        }
        let mvs: Vec<MovementId> = members.iter().flat_map(|&n| mvs_by_node[n.idx()].iter().copied()).collect();
        let major_crossing = mvs.iter().enumerate().any(|(i, &a)| {
            is_major(a)
                && mvs[i + 1..].iter().any(|&b| {
                    is_major(b) && approach(a) != approach(b) && net.movements_conflict(a, b)
                })
        });
        if major_crossing {
            plans[members[0].idx()] = Some(SignalPlan { green_secs: GREEN, yellow_secs: YELLOW, offset: 0.0 });
        }
    }
}

/// Give each multi-node junction a single coordinated signal spanning all its member
/// nodes, so through movements across the junction share one green phase and vehicles
/// never stop between the interior nodes. Runs after the cross-node conflict graph is
/// built, so genuinely conflicting cross-node movements still land in separate phases.
/// Signalized nodes outside any junction keep an independent per-node program.
fn coordinate_junction_signals(net: &mut Network, plans: &[Option<SignalPlan>]) {
    let feeders = net.feeders_by_lane();
    let mut movements_by_node: Vec<Vec<MovementId>> = vec![Vec::new(); net.nodes.len()];
    for (m, mv) in net.movements.iter().enumerate() {
        movements_by_node[mv.node.idx()].push(MovementId(m as u32));
    }
    for ji in 0..net.junctions.len() {
        let mut members = net.junctions[ji].nodes.clone();
        members.sort_by_key(|nd| nd.0);
        let Some(plan) = members.iter().find_map(|&nd| plans[nd.idx()]) else {
            continue;
        };
        let control = assign_signal_program(net, &members, plan, &feeders, &movements_by_node);
        if let NodeControl::Signalized(pid) = control {
            for &nd in &members {
                net.nodes[nd.idx()].control = control;
            }
            net.junctions[ji].program = Some(pid);
        } else {
            // Too many groups for one 64-bit phase mask — fall back to per-node signals.
            for &nd in &members {
                if let Some(p) = plans[nd.idx()] {
                    net.nodes[nd.idx()].control = assign_signal_program(net, &[nd], p, &feeders, &movements_by_node);
                }
            }
            net.junctions[ji].program = members.iter().find_map(|&nd| match net.nodes[nd.idx()].control {
                NodeControl::Signalized(p) => Some(p),
                _ => None,
            });
        }
    }
    for i in 0..net.nodes.len() {
        let nd = NodeId(i as u32);
        if plans[i].is_some() && net.node_junction(nd).is_none() {
            net.nodes[i].control = assign_signal_program(net, &[nd], plans[i].unwrap(), &feeders, &movements_by_node);
        }
    }
}

const YELLOW_REACTION: f64 = 1.0;
const YELLOW_DECEL: f64 = 3.0;
const YELLOW_MIN: f64 = 3.0;
const YELLOW_MAX: f64 = 6.0;
const CLEARANCE_VEHICLE_LEN: f64 = 6.0;
const ALL_RED_MIN: f64 = 1.5;
const ALL_RED_MAX: f64 = 5.0;
/// Straggler speed through a compound junction's internal path (m/s) and the larger
/// all-red cap such junctions may need. A movement that lands on a cluster-internal
/// link hasn't cleared anything: the car still has slivers and further interiors to
/// traverse at box speed, and an all-red sized to one interior strands it mid-box
/// every cycle — the cross phase then flows around a parked straggler.
const CLUSTER_CLEAR_SPEED: f64 = 6.0;
const ALL_RED_MAX_CLUSTER: f64 = 8.0;

/// Seconds a car landing off movement `m` still needs to reach the cluster exit —
/// zero when the movement leaves the junction directly.
fn internal_continuation_secs(net: &Network, m: MovementId) -> f64 {
    let mv = net.movement(m);
    let Some(j) = net.node_junction(mv.node) else { return 0.0 };
    let mut lane_id = mv.to_lane;
    let mut dist = 0.0;
    for _ in 0..8 {
        let lane = net.lane(lane_id);
        if net.node_junction(net.link(lane.link).to) != Some(j) {
            break; // this link leaves the cluster
        }
        dist += lane.length;
        let start = lane.movement_start.0;
        let onward = (0..net.movements_of(lane_id).len())
            .map(|k| MovementId(start + k as u32))
            .max_by(|&a, &b| net.interior(a).len.total_cmp(&net.interior(b).len));
        let Some(next) = onward else { break };
        dist += net.interior(next).len;
        lane_id = net.movement(next).to_lane;
    }
    dist / CLUSTER_CLEAR_SPEED
}

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
    let internal = movements
        .iter()
        .map(|&m| internal_continuation_secs(net, m))
        .fold(0.0_f64, f64::max);
    let cap = if internal > 0.0 { ALL_RED_MAX_CLUSTER } else { ALL_RED_MAX };
    let all_red = ((crossing + CLEARANCE_VEHICLE_LEN) / approach + internal).clamp(ALL_RED_MIN, cap);
    (yellow, all_red)
}

/// Coordinate signalized arterial corridors into green waves. OSM carries no signal
/// timing, so every imported signal starts at `offset = 0` (all synchronized — the
/// worst case for progression). This walks each named arterial between consecutive
/// signals and offsets each program so its **through** phase opens just as a platoon
/// travelling the corridor at road speed arrives. Best-effort along one direction per
/// corridor; a mismatched cycle only degrades the coordination — the conflict-built
/// phases (and thus safety) are never touched.
/// Stretch every signal on a named street to the street's longest cycle: a green
/// wave only repeats when cycle lengths match, so a corridor whose members run
/// different cycles (protected-left stages, pedestrian floors) drifts in and out
/// of progression each round. Greens scale up (never down) and conflict masks
/// are untouched, so safety is unaffected — the offsets computed afterwards then
/// hold every cycle.
fn harmonize_corridor_cycles(net: &mut Network, sig: &std::collections::BTreeMap<u32, ProgramId>) {
    use std::collections::{BTreeMap, BTreeSet};
    let mut families: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
    for (i, l) in net.links.iter().enumerate() {
        let name = net.link_names[i].as_str();
        if name.is_empty() {
            continue;
        }
        for nd in [l.from, l.to] {
            if let Some(pid) = sig.get(&nd.0) {
                families.entry(name.to_string()).or_default().insert(pid.idx());
            }
        }
    }
    for pids in families.values() {
        if pids.len() < 2 {
            continue;
        }
        let target = pids.iter().map(|&p| net.programs[p].cycle_length()).fold(0.0, f64::max);
        for &p in pids {
            let prog = &mut net.programs[p];
            let cycle = prog.cycle_length();
            if target - cycle < 0.1 {
                continue;
            }
            let fixed: f64 = prog.phases.iter().map(|ph| ph.yellow_secs + ph.all_red_secs).sum();
            let green: f64 = prog.phases.iter().map(|ph| ph.green_secs).sum();
            if green <= 0.0 {
                continue;
            }
            let f = (target - fixed) / green;
            for ph in &mut prog.phases {
                ph.green_secs *= f;
            }
        }
    }
}

fn coordinate_green_waves(net: &mut Network) {
    use std::collections::{BTreeMap, BTreeSet, VecDeque};
    const SPEED_FLOOR: f64 = 5.0; // m/s, so a slow arterial still gets a sane travel time
    const MAX_HOPS: usize = 24; // guard against a road that loops back on itself
    const MIN_ALIGN: f64 = 0.3; // a continuation link must head roughly the same way

    // Compute every offset while only *reading* the network, then apply them — so the
    // read-only walk/lookup closures don't clash with mutating `net.programs`.
    let offsets: Vec<(usize, f64, f64, usize)> = {
        let mut sig: BTreeMap<u32, ProgramId> = BTreeMap::new();
        for (i, n) in net.nodes.iter().enumerate() {
            if let NodeControl::Signalized(p) = n.control {
                sig.insert(i as u32, p);
            }
        }
        if sig.len() < 2 {
            return;
        }
        harmonize_corridor_cycles(net, &sig);
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

        // Green-start time (into the cycle) and phase index of the corridor's
        // through phase at `node`, found via a straight-through movement along
        // `corridor_link`. The index is stored on the program so the
        // semi-actuated controller knows which phase the progression anchors to.
        let through_start = |node: u32, corridor_link: u32| -> Option<(f64, usize)> {
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
                for (pi, ph) in prog.phases.iter().enumerate() {
                    if ph.green_mask & (1u64 << bit) != 0 {
                        return Some((acc, pi));
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
                    if let Some((sp, phase)) = through_start(n, link) {
                        // AM plan: green opens `cum` later downstream (progression
                        // along the walk); PM plan: the mirror (progression back).
                        out.push((pid.idx(), (sp - cum).rem_euclid(cycle), (sp + cum).rem_euclid(cycle), phase));
                    }
                }
            }
        }
        out
    };

    net.am_offsets = net.programs.iter().map(|p| p.offset).collect();
    net.pm_offsets = net.am_offsets.clone();
    for (idx, am, pm, phase) in offsets {
        net.programs[idx].offset = am;
        net.programs[idx].coordinated = true;
        net.programs[idx].coordinated_phase = phase;
        net.am_offsets[idx] = am;
        net.pm_offsets[idx] = pm;
    }
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// Target corner radius for [`fillet_polyline`] — a comfortable urban curb-return
/// scale, well above any car's minimum turning radius.
const FILLET_RADIUS: f64 = 8.0;

/// Round each interior bend of a link centreline into a short curve, so a mapped
/// corner is something a car steers through (bounded heading change per metre)
/// rather than a point where the heading jumps instantaneously. The fillet aims
/// for `radius`, shrunk where the adjacent segments are short — each vertex may
/// consume at most half of each neighbouring segment, so consecutive fillets
/// never overlap. Endpoints (the node positions) and every sub-segment direction
/// outside the curves are preserved, so junction arrival/departure directions
/// and downstream placement passes see the same road axes.
fn fillet_polyline(poly: Vec<[f64; 2]>, radius: f64) -> Vec<[f64; 2]> {
    const MIN_TURN: f64 = 0.10; // ~6°: already smooth enough to steer through
    // Leave the link's ends untouched: the junction model (end-direction sampling,
    // carriageway bands, stop-line clearance, arm mouths) owns the geometry there,
    // and rounding a vertex under a junction box perturbs all of it. The fillet's
    // job is mid-corridor steering realism; near a node cars are slow and the
    // interior paths carry them.
    const END_KEEP: f64 = 15.0;
    if poly.len() < 3 {
        return poly;
    }
    let mut arc = vec![0.0; poly.len()];
    for i in 1..poly.len() {
        arc[i] = arc[i - 1] + distance(poly[i - 1], poly[i]);
    }
    let total = arc[poly.len() - 1];
    let mut out: Vec<[f64; 2]> = Vec::with_capacity(poly.len() * 3);
    out.push(poly[0]);
    let push = |out: &mut Vec<[f64; 2]>, p: [f64; 2]| {
        if distance(*out.last().unwrap(), p) > 1e-9 {
            out.push(p);
        }
    };
    for i in 1..poly.len() - 1 {
        if arc[i] < END_KEEP || total - arc[i] < END_KEEP {
            let v = poly[i];
            push(&mut out, v);
            continue;
        }
        let (p, v, n) = (poly[i - 1], poly[i], poly[i + 1]);
        let (la, lb) = (distance(p, v), distance(v, n));
        if la < 1e-6 || lb < 1e-6 {
            push(&mut out, v);
            continue;
        }
        let a = [(v[0] - p[0]) / la, (v[1] - p[1]) / la];
        let b = [(n[0] - v[0]) / lb, (n[1] - v[1]) / lb];
        let theta = (a[0] * b[0] + a[1] * b[1]).clamp(-1.0, 1.0).acos();
        // Leave near-reversal spikes alone: a "fillet" across one would cut the
        // corner into a different road entirely; they don't survive import anyway.
        if theta < MIN_TURN || theta > 2.8 {
            push(&mut out, v);
            continue;
        }
        let t = (radius * (theta / 2.0).tan()).min(0.5 * la.min(lb));
        let p1 = [v[0] - a[0] * t, v[1] - a[1] * t];
        let p2 = [v[0] + b[0] * t, v[1] + b[1] * t];
        // Quadratic Bézier with its control at the vertex — tangent to both
        // segments at p1/p2 and within a few percent of the inscribed arc.
        let steps = ((theta / 0.26).ceil() as usize).max(2); // a point every ~15°
        for k in 0..=steps {
            let s = k as f64 / steps as f64;
            let u = 1.0 - s;
            push(&mut out, [
                u * u * p1[0] + 2.0 * u * s * v[0] + s * s * p2[0],
                u * u * p1[1] + 2.0 * u * s * v[1] + s * s * p2[1],
            ]);
        }
    }
    push(&mut out, poly[poly.len() - 1]);
    out
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

/// Merge the two link segments meeting at a dissolved pass-through node `n` into one
/// link `from`→`to` (segment `s1` ends at `n`, `s2` starts at `n`), threading the
/// merged geometry through `n`. `None` if the segments' speed, layer or lane count
/// don't match (so the node is a real attribute change and stays a junction).
fn join_pass_through(
    links: &[Option<LinkSpec>],
    s1: usize,
    s2: usize,
    from: i64,
    to: i64,
    n: i64,
    pos: &HashMap<i64, [f64; 2]>,
) -> Option<LinkSpec> {
    let (l1, l2) = (links[s1].as_ref().unwrap(), links[s2].as_ref().unwrap());
    let link_len = |l: &LinkSpec| -> f64 {
        let mut pts = vec![pos[&l.from_osm]];
        pts.extend(l.geometry.iter().copied());
        pts.push(pos[&l.to_osm]);
        pts.windows(2).map(|w| distance(w[0], w[1])).sum()
    };
    // Attributes must match across the joint — except a grade-separated fragment
    // too short to hold a vehicle, which is survey noise (a maxspeed, layer, or
    // lane-count change tagged onto a few metres of ramp approach): it adopts the
    // substantive segment's attributes, so the freeway system never carries
    // micro-links that thrash the node-crossing logic.
    const SLIVER_MAX: f64 = 30.0;
    let (len1, len2) = (link_len(l1), link_len(l2));
    let grade_sep = |l: &LinkSpec| RoadKind::from_osm(&l.road_class).is_grade_separated();
    let sliver = grade_sep(l1) && grade_sep(l2) && len1.min(len2) < SLIVER_MAX;
    let big = if len1 >= len2 { l1 } else { l2 };
    if !sliver && (l1.speed_limit != l2.speed_limit || l1.layer != l2.layer || l1.lanes != l2.lanes) {
        return None;
    }
    let (speed_limit, layer) = if sliver { (big.speed_limit, big.layer) } else { (l1.speed_limit, l1.layer) };
    let lanes = if l1.lanes == l2.lanes { l1.lanes } else { big.lanes };
    let mut geometry = l1.geometry.clone();
    geometry.push(pos[&n]);
    geometry.extend(l2.geometry.iter().copied());
    let name = if l1.name.is_empty() { l2.name.clone() } else { l1.name.clone() };
    let road_class = if l1.road_class.is_empty() { l2.road_class.clone() } else { l1.road_class.clone() };
    let highway_ref = if l1.highway_ref.is_empty() { l2.highway_ref.clone() } else { l1.highway_ref.clone() };
    // The downstream segment (l2, ending at the merge's `to`) carries the turn:lanes
    // that matter at the stop line; fall back to l1 if it lacks them.
    let turn_lanes = if l2.turn_lanes.is_empty() { l1.turn_lanes.clone() } else { l2.turn_lanes.clone() };
    let hov_lanes = if l2.hov_lanes.is_empty() { l1.hov_lanes.clone() } else { l2.hov_lanes.clone() };
    let nonzero = |a: f64, b: f64| if a > 0.0 { a } else { b };
    let (aadt, res_weight, attr_weight) = (
        nonzero(l1.aadt, l2.aadt),
        nonzero(l1.res_weight, l2.res_weight),
        nonzero(l1.attr_weight, l2.attr_weight),
    );
    // A sign on the upstream segment referred to the dissolved node — which had a
    // single continuation, so it really protects the junction downstream: taking
    // the stronger sign relocates it there, however many pass-throughs dissolve.
    let sign = l1.sign.max(l2.sign);
    Some(LinkSpec { from_osm: from, to_osm: to, lanes, speed_limit, geometry, layer, name, road_class, highway_ref, turn_lanes, hov_lanes, aadt, res_weight, attr_weight, sign })
}

/// Pull every lane back from its end nodes to the junction boundary so vehicles
/// stop and start at the edge of the intersection, leaving the interior (the box)
/// to the movements' crossing paths. A node's setback is half the widest
/// carriageway meeting it; clamped so short links keep a positive drivable span.
/// Slide each ramp's freeway end out to the curb. OSM attaches a ramp at the
/// freeway *centreline* node, so a narrow ramp sharing that node is drawn across
/// the freeway's middle lanes instead of peeling off the outside. Shifting the
/// ramp end laterally by half the width difference (tapered back to its own
/// alignment over `TRANSITION` m) makes the ramp diverge from / merge onto the
/// curb edge, matching how the lanes are wired.
/// Spread the feeders of a grade-separated merge across the exit link's lanes by their lateral
/// order, so N parallel feeders — toll-booth lanes — land in N distinct lanes (left-to-right)
/// instead of every mainline feeder independently mapping onto lane 0 (the median). That
/// per-link `min(k, out-1)` mapping was what made the Golden Gate toll plaza's lanes all
/// converge onto one lane. On-ramp feeders are handled separately: a ramp takes a genuinely
/// added exit lane if one exists, else it merges into (shares) its nearest mainline feeder's
/// lane. Scoped to freeway/ramp exits; surface intersections keep their turn-angle
/// channelisation (feeders there come from different directions, where a lateral spread would
/// be wrong).
fn spread_merge_feeders(net: &mut Network) {
    let mut by_exit: HashMap<u32, Vec<usize>> = HashMap::new();
    for (mi, m) in net.movements.iter().enumerate() {
        by_exit.entry(net.lane(m.to_lane).link.0).or_default().push(mi);
    }
    for (elid, mvs) in by_exit {
        let exit = *net.link(LinkId(elid));
        if !exit.kind.is_grade_separated() || exit.lane_count <= 1 {
            continue;
        }
        // Distinct feeder lanes converging into this exit link.
        let mut feeders: Vec<u32> = mvs.iter().map(|&mi| net.movements[mi].from_lane.0).collect();
        feeders.sort_unstable();
        feeders.dedup();
        if feeders.len() <= 1 {
            continue; // a single feeder isn't a merge
        }
        // Order the feeders left→right by their lateral position at the node (their lane end
        // projected onto the perpendicular of the exit's direction of travel).
        let dep = net.departure_dir(LinkId(elid));
        let right = [dep[1], -dep[0]];
        let mut ordered: Vec<(f64, u32)> = feeders
            .iter()
            .map(|&fl| {
                let p = net.lane_point(LaneId(fl), net.lane(LaneId(fl)).length);
                (p[0] * right[0] + p[1] * right[1], fl)
            })
            .collect();
        ordered.sort_by(|a, b| a.0.total_cmp(&b.0));
        // The continuing carriageway (non-ramp feeders) spreads across the exit
        // lanes; a merging on-ramp only gets a lane of its own when the exit
        // genuinely adds one (an acceleration lane) — otherwise it shares its
        // nearest mainline feeder's target. A ramp must never claim an exit lane
        // exclusively by lateral accident: single-fed, that lane would read as a
        // seamless corridor continuation and the merge would lose its yield.
        let is_ramp = |fl: u32| net.link(net.lane(LaneId(fl)).link).kind == RoadKind::Ramp;
        let mains: Vec<(f64, u32)> = ordered.iter().copied().filter(|&(_, fl)| !is_ramp(fl)).collect();
        let ramps: Vec<(f64, u32)> = ordered.iter().copied().filter(|&(_, fl)| is_ramp(fl)).collect();
        let spread = |ranked: &[(f64, u32)], lane_of: &mut HashMap<u32, u32>| {
            let f = ranked.len();
            for (rank, &(_, fl)) in ranked.iter().enumerate() {
                let idx = if f == 1 {
                    0
                } else {
                    ((rank as f64) * (exit.lane_count - 1) as f64 / (f - 1) as f64).round() as u32
                };
                lane_of.insert(fl, idx);
            }
        };
        let mut lane_of: HashMap<u32, u32> = HashMap::new();
        if mains.is_empty() || ramps.is_empty() || mains.len() > exit.lane_count as usize {
            spread(&ordered, &mut lane_of);
        } else {
            // The continuing carriageway lands each lane on its laterally nearest
            // exit lane (monotone, collision-free), so the added lane a merge
            // brings in stays open exactly where the pavement puts it.
            let exit_proj: Vec<f64> = (0..exit.lane_count)
                .map(|k| {
                    let p = net.lane_point(LaneId(exit.lane_start.0 + k), 0.0);
                    p[0] * right[0] + p[1] * right[1]
                })
                .collect();
            let n = exit.lane_count as i64;
            let mut prev: i64 = -1;
            for (rank, &(p, fl)) in mains.iter().enumerate() {
                let hi = n - (mains.len() - rank) as i64;
                let idx = ((prev + 1)..=hi)
                    .min_by(|&a, &b| (exit_proj[a as usize] - p).abs().total_cmp(&(exit_proj[b as usize] - p).abs()))
                    .unwrap();
                lane_of.insert(fl, idx as u32);
                prev = idx;
            }
            let taken: std::collections::BTreeSet<u32> = lane_of.values().copied().collect();
            let mut leftover: Vec<u32> = (0..exit.lane_count).filter(|k| !taken.contains(k)).collect();
            for &(p, fl) in &ramps {
                let target = if leftover.is_empty() {
                    let (_, nearest) = mains
                        .iter()
                        .copied()
                        .min_by(|a, b| (a.0 - p).abs().total_cmp(&(b.0 - p).abs()))
                        .unwrap();
                    lane_of[&nearest]
                } else {
                    let k = (0..leftover.len())
                        .min_by(|&a, &b| {
                            (exit_proj[leftover[a] as usize] - p)
                                .abs()
                                .total_cmp(&(exit_proj[leftover[b] as usize] - p).abs())
                        })
                        .unwrap();
                    leftover.remove(k)
                };
                lane_of.insert(fl, target);
            }
        }
        for &mi in &mvs {
            let fl = net.movements[mi].from_lane.0;
            net.movements[mi].to_lane = LaneId(exit.lane_start.0 + lane_of[&fl]);
        }
    }
}

/// Whether every `;`-separated part of a `turn:lanes` token is the given kind of
/// turn — a lane serving *only* that direction (a pocket candidate).
fn pure_turn_token(tok: &str, kinds: &[&str]) -> bool {
    let mut any = false;
    for p in tok.split(';') {
        if !kinds.contains(&p.trim()) {
            return false;
        }
        any = true;
    }
    any
}

/// Median-side (`left`) and curb-side (`right`) dedicated-turn lane counts from a
/// link's `turn:lanes`, or `(0, 0)` when the tag is absent, doesn't match the
/// lane count, or leaves no through core to anchor on.
fn pocket_counts(turn_lanes: &str, lane_count: u32) -> (u32, u32) {
    if turn_lanes.is_empty() {
        return (0, 0);
    }
    let toks: Vec<&str> = turn_lanes.split('|').collect();
    if toks.len() != lane_count as usize {
        return (0, 0);
    }
    let lefts = ["left", "slight_left", "sharp_left", "reverse"];
    let rights = ["right", "slight_right", "sharp_right"];
    let l = toks.iter().take_while(|t| pure_turn_token(t, &lefts)).count() as u32;
    let r = toks.iter().rev().take_while(|t| pure_turn_token(t, &rights)).count() as u32;
    if l + r >= lane_count {
        return (0, 0); // a pure turn roadway, not a widened approach
    }
    (l, r)
}

/// Laterally offset a polyline to the *left* of its travel direction, using
/// averaged vertex normals so bends stay connected. `mag(d_start, d_end)` gives
/// the shift at a vertex from its arc distances to the two ends, so the offset
/// can taper where the mapped line is already correct (a divided→undivided seam).
fn shift_polyline_left(poly: &mut [[f64; 2]], mag: impl Fn(f64, f64) -> f64) {
    let k = poly.len();
    if k < 2 {
        return;
    }
    let seg_dir = |a: [f64; 2], b: [f64; 2]| -> [f64; 2] {
        let d = [b[0] - a[0], b[1] - a[1]];
        let n = (d[0] * d[0] + d[1] * d[1]).sqrt();
        if n < 1e-9 {
            [0.0, 0.0]
        } else {
            [d[0] / n, d[1] / n]
        }
    };
    let dirs: Vec<[f64; 2]> = (0..k - 1).map(|i| seg_dir(poly[i], poly[i + 1])).collect();
    let mut arc = vec![0.0f64; k];
    for i in 1..k {
        arc[i] = arc[i - 1] + distance(poly[i - 1], poly[i]);
    }
    let full = arc[k - 1];
    let shifted: Vec<[f64; 2]> = (0..k)
        .map(|i| {
            let a = if i == 0 { [0.0, 0.0] } else { dirs[i - 1] };
            let b = if i == k - 1 { [0.0, 0.0] } else { dirs[i] };
            let sum = [a[0] + b[0], a[1] + b[1]];
            let n = (sum[0] * sum[0] + sum[1] * sum[1]).sqrt();
            let d = if n < 1e-9 { [1.0, 0.0] } else { [sum[0] / n, sum[1] / n] };
            let s = mag(arc[i], full - arc[i]);
            [poly[i][0] - d[1] * s, poly[i][1] + d[0] * s]
        })
        .collect();
    poly.copy_from_slice(&shifted);
}

/// Recentre every one-way carriageway on its OSM way line. OSM draws a way down
/// the middle of its pavement; the engine's placement convention is polyline =
/// the carriageway's left (median) edge with lanes offset rightward — exact for
/// the two directions of a two-way road sharing one centreline, but half a
/// carriageway off for a one-way link (a divided arterial like El Camino Real, a
/// freeway mainline, a ramp). Shift those polylines left by half their width so
/// the drawn and driven lanes straddle the mapped line. `turn:lanes` skews the
/// shift by the dedicated-turn lanes on each side, because a left-turn pocket
/// widens the carriageway on the median side, not the curb: the through core
/// then stays laterally continuous across the lane-count change instead of
/// jogging a full lane width at every pocket.
fn center_oneway_axes(net: &mut Network) {
    /// Length over which an end-of-link shift correction blends back to the
    /// link's own mid-block shift.
    const TAPER: f64 = 40.0;
    let two_way: HashSet<(u32, u32)> = net.links.iter().map(|l| (l.to.0, l.from.0)).collect();
    let is_two_way = |li: usize| two_way.contains(&(net.links[li].from.0, net.links[li].to.0));
    // Raw end headings, taken before any polyline moves.
    let end_dir = |li: usize, at_from: bool| -> [f64; 2] {
        let p = &net.polylines[li];
        let d = if at_from {
            [p[1][0] - p[0][0], p[1][1] - p[0][1]]
        } else {
            let k = p.len();
            [p[k - 1][0] - p[k - 2][0], p[k - 1][1] - p[k - 2][1]]
        };
        let n = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-9);
        [d[0] / n, d[1] / n]
    };
    let mut at_node: HashMap<u32, Vec<usize>> = HashMap::new();
    for (li, l) in net.links.iter().enumerate() {
        at_node.entry(l.from.0).or_default().push(li);
        at_node.entry(l.to.0).or_default().push(li);
    }
    // A one-way link's end continues into a two-way road when some other link at
    // the node runs on in (roughly) the same direction and has a reverse twin.
    // There the mapped line is already the median edge — the ways split/join at
    // that node — so the recentring shift must fade out or it kinks the seam.
    let seam_end = |li: usize, node: u32, own: [f64; 2]| -> bool {
        at_node[&node].iter().any(|&lj| {
            if lj == li || !is_two_way(lj) {
                return false;
            }
            let l = net.links[lj];
            let d = end_dir(lj, l.from.0 == node);
            (own[0] * d[0] + own[1] * d[1]).abs() > 0.7
        })
    };
    let shifts: Vec<Option<(f64, f64, f64)>> = (0..net.links.len())
        .map(|li| {
            let l = net.links[li];
            if is_two_way(li) {
                return None;
            }
            let (lp, rp) = pocket_counts(&net.link_turn_lanes[li], l.lane_count);
            let s = (l.lane_count + lp - rp) as f64 * LANE_WIDTH * 0.5;
            let s_from = if seam_end(li, l.from.0, end_dir(li, true)) { 0.0 } else { s };
            let s_to = if seam_end(li, l.to.0, end_dir(li, false)) { 0.0 } else { s };
            Some((s, s_from, s_to))
        })
        .collect();
    for (li, sh) in shifts.into_iter().enumerate() {
        let Some((s, s_from, s_to)) = sh else { continue };
        shift_polyline_left(&mut net.polylines[li], |d_start, d_end| {
            let f = s + (s_from - s) * (1.0 - d_start / TAPER).max(0.0);
            let t = s + (s_to - s) * (1.0 - d_end / TAPER).max(0.0);
            // Whichever end correction reaches this vertex dominates; on a short
            // link both do, and the smaller (more corrected) shift wins.
            f.min(t)
        });
    }
}

/// Reconcile through-lane geometry across seams and junction boxes: where a
/// road's through movements land laterally off the line they left (an OSM way
/// redrawn mid-carriageway at a width change, an unmapped `turn:lanes` falling
/// back to the symmetric shift), nudge the two link ends toward each other with
/// tapered lateral shifts so the through path runs straight. The junction —
/// not either link alone — is what knows both sides of the box, which makes
/// this the geometry-owning step of the intersection-as-entity model. Bounded:
/// sub-lane offsets always correct; up to half a carriageway corrects only when
/// both links carry the same road name (the road itself continuing); anything
/// larger is a genuine dogleg or a parallel service way and stays.
/// Mutual-primary through-seam partners: ordered one-way link pairs joined by
/// aligned through movements where *each* end elects the other as its main
/// continuation (most through movements, then widest carriageway, lowest id on
/// ties) — so a parallel service way merging in can't drag the mainline. Both
/// the axis-level seam alignment and the exact boundary stitch reconcile the
/// same partnerships.
fn through_seam_partners(net: &Network) -> Vec<(u32, u32, Vec<MovementId>)> {
    let two_way: HashSet<(u32, u32)> = net.links.iter().map(|l| (l.to.0, l.from.0)).collect();
    let one_way = |li: u32| !two_way.contains(&(net.links[li as usize].from.0, net.links[li as usize].to.0));
    let mut pairs: HashMap<(u32, u32), Vec<MovementId>> = HashMap::new();
    for m in 0..net.movements.len() as u32 {
        let mid = MovementId(m);
        let mv = *net.movement(mid);
        let (a, b) = (net.lane(mv.from_lane).link, net.lane(mv.to_lane).link);
        if a == b || net.movement_turn(mid) != TurnType::Through || !one_way(a.0) || !one_way(b.0) {
            continue;
        }
        let (da, db) = (net.arrival_dir(a), net.departure_dir(b));
        if da[0] * db[0] + da[1] * db[1] < 0.7 {
            continue; // an angled "through" at a Y — not a seam to straighten
        }
        pairs.entry((a.0, b.0)).or_default().push(mid);
    }
    let mut best_out: HashMap<u32, (u32, (usize, u32))> = HashMap::new();
    let mut best_in: HashMap<u32, (u32, (usize, u32))> = HashMap::new();
    for (&(a, b), mids) in &pairs {
        let score = (mids.len(), net.links[a as usize].lane_count.min(net.links[b as usize].lane_count));
        for (map, key, partner) in [(&mut best_out, a, b), (&mut best_in, b, a)] {
            match map.get(&key) {
                Some(&(p, s)) if s > score || (s == score && p < partner) => {}
                _ => {
                    map.insert(key, (partner, score));
                }
            }
        }
    }
    let mut out: Vec<(u32, u32, Vec<MovementId>)> = best_in
        .iter()
        .filter(|&(&b, &(a, _))| best_out.get(&a).is_some_and(|&(bb, _)| bb == b))
        .map(|(&b, &(a, _))| (a, b, pairs[&(a, b)].clone()))
        .collect();
    out.sort_by_key(|&(a, b, _)| (a, b));
    out
}

/// Whether a measured seam offset is small enough to correct: sub-noise always;
/// up to ~a carriageway's reach when the same named road continues across its
/// own box (a way redrawn mid-pavement at a width change stacks on the pocket
/// skew — mapping noise, not a real dogleg).
fn seam_shift_allowed(net: &Network, a: u32, b: u32, delta: f64) -> bool {
    const NOISE_BOUND: f64 = 0.6 * LANE_WIDTH;
    if delta.abs() <= NOISE_BOUND {
        return true;
    }
    let (na, nb) = (&net.link_names[a as usize], &net.link_names[b as usize]);
    let wide = net.links[a as usize].lane_count.max(net.links[b as usize].lane_count) as f64 * LANE_WIDTH;
    !na.is_empty() && na == nb && delta.abs() <= wide * 0.6
}

/// Midpoint-extrapolated leftward jog from point `pa` (leaving along `da`) to
/// point `pb` (continuing along `db`): each is run straight to their midpoint,
/// so the longitudinal gap (and half its curvature) drops out and a collinear
/// continuation measures zero at any distance.
fn seam_jog(pa: [f64; 2], da: [f64; 2], pb: [f64; 2], db: [f64; 2]) -> f64 {
    let mid_pt = [(pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5];
    let project = |p: [f64; 2], d: [f64; 2]| {
        let t = (mid_pt[0] - p[0]) * d[0] + (mid_pt[1] - p[1]) * d[1];
        [p[0] + d[0] * t, p[1] + d[1] * t]
    };
    let (qa, qb) = (project(pa, da), project(pb, db));
    let mean = [da[0] + db[0], da[1] + db[1]];
    let n = (mean[0] * mean[0] + mean[1] * mean[1]).sqrt().max(1e-9);
    let left = [-mean[1] / n, mean[0] / n];
    (qb[0] - qa[0]) * left[0] + (qb[1] - qa[1]) * left[1]
}

fn align_through_seams(net: &mut Network) {
    const TAPER: f64 = 40.0;

    // Each seam splits its correction between the two ends, so chains of
    // segments settle toward each other instead of one link absorbing the
    // whole error at both of its ends.
    let mut end_shift: HashMap<(u32, bool), f64> = HashMap::new(); // (link, at_start) → leftward shift
    for (a, b, mids) in through_seam_partners(net) {
        let (da, db) = (net.arrival_dir(LinkId(a)), net.departure_dir(LinkId(b)));
        let delta = mids
            .iter()
            .map(|&mid| {
                let mv = net.movement(mid);
                let entry = net.lane_point(mv.from_lane, net.lane(mv.from_lane).length);
                let exit = net.lane_point(mv.to_lane, 0.0);
                seam_jog([entry[0], entry[1]], da, [exit[0], exit[1]], db)
            })
            .sum::<f64>()
            / mids.len() as f64;
        if seam_shift_allowed(net, a, b, delta) {
            end_shift.insert((a, false), delta * 0.5);
            end_shift.insert((b, true), -delta * 0.5);
        }
    }

    if end_shift.is_empty() {
        return;
    }
    for li in 0..net.links.len() {
        let s_start = end_shift.get(&(li as u32, true)).copied().unwrap_or(0.0);
        let s_end = end_shift.get(&(li as u32, false)).copied().unwrap_or(0.0);
        if s_start == 0.0 && s_end == 0.0 {
            continue;
        }
        shift_polyline_left(&mut net.polylines[li], |d_start, d_end| {
            s_start * (1.0 - d_start / TAPER).max(0.0) + s_end * (1.0 - d_end / TAPER).max(0.0)
        });
    }
    // Arc lengths and box-edge headings moved a little; recompute the setbacks,
    // drivable spans, and the end-direction cache from the settled geometry.
    set_junction_setbacks(net);
}

/// Close every through seam *exactly*, at the boundary level. The axis pass
/// (`align_through_seams`) reconciles carriageways in the mean; what survives it
/// is per-lane: lanes of one seam disagreeing after a width change, corrections
/// truncated by its bounds. Here each shared lane boundary crossing a
/// mutual-primary seam is measured by the same midpoint extrapolation and the
/// two ends are pulled onto each other — half each, tapered upstream — directly
/// in the stored [`LaneBounds`]. The axis polylines stay put (arc lengths,
/// headings, and routing are untouched); the boundary chart, which is what
/// vehicles, markings, and mouths read, is what closes.
fn stitch_seam_bounds(net: &mut Network) {
    const MAX_SHIFT: f64 = 0.75 * LANE_WIDTH; // per boundary; more means "not the same lane"
    const PINCH: f64 = 0.8 * LANE_WIDTH; // adjacent corrections disagreeing more: distrust the wiring

    // (link, at downstream end, per-boundary leftward shift, shift direction).
    let mut plan: Vec<(usize, bool, Vec<f64>, [f64; 2])> = Vec::new();
    for (a, b, mids) in through_seam_partners(net) {
        let (ai, bi) = (a as usize, b as usize);
        let (na, nb) = (net.links[ai].lane_count as usize, net.links[bi].lane_count as usize);
        let (Some(lba), Some(lbb)) = (net.lane_bounds.get(ai), net.lane_bounds.get(bi)) else { continue };
        if lba.stations.len() < 2 || lbb.stations.len() < 2 {
            continue;
        }
        let (da, db) = (net.arrival_dir(LinkId(a)), net.departure_dir(LinkId(b)));
        let mean = {
            let m = [da[0] + db[0], da[1] + db[1]];
            let n = (m[0] * m[0] + m[1] * m[1]).sqrt().max(1e-9);
            [m[0] / n, m[1] / n]
        };
        let left = [-mean[1], mean[0]];

        // Measure against the *base* (unpocketed) cross-section at each end's
        // stop line — the arc where the interiors attach and where the applied
        // basis is pinned at full strength; the box-edge stations beyond it sit
        // past a curve's last sweep and would misstate the jog the vehicles see.
        // Base, because a closed bay collapses its boundaries onto the through
        // edge, and pairing those points would read the bay's shape as a seam
        // error (a rigid half-lane offset came out as ±a lane width, defeating
        // the guards). The through anchors are the bay-free boundaries; base
        // points respace from them at a lane width apiece. The applied shift
        // then moves the stored chart — bay geometry and all — rigidly onto the
        // corrected base.
        let base_cross = |li: usize, at_end: bool| -> Vec<[f64; 2]> {
            let l = &net.links[li];
            let n = l.lane_count as usize;
            let pocket = |k: usize| net.lanes[l.lane_start.idx() + k].pocket_taper > 0.0;
            let ml = (0..n).take_while(|&k| pocket(k)).count();
            let mr = (0..n).rev().take_while(|&k| pocket(k)).count().min(n - ml);
            let lane = &net.lanes[l.lane_start.idx()];
            let stop = if at_end { lane.start_offset + lane.length } else { lane.start_offset };
            let lb = &net.lane_bounds[li];
            let i = lb.stations.partition_point(|&x| x < stop).clamp(1, lb.stations.len() - 1);
            let t = ((stop - lb.stations[i - 1]) / (lb.stations[i] - lb.stations[i - 1]).max(1e-9)).clamp(0.0, 1.0);
            let sample = |k: usize| {
                let (p, q) = (lb.bounds[k][i - 1], lb.bounds[k][i]);
                [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]
            };
            let (lo, hi) = (sample(ml), sample(n - mr));
            let right = {
                let d = [hi[0] - lo[0], hi[1] - lo[1]];
                let len = d[0].hypot(d[1]).max(1e-9);
                [d[0] / len, d[1] / len]
            };
            (0..=n)
                .map(|k| {
                    let off = (k as f64 - ml as f64) * LANE_WIDTH;
                    [lo[0] + right[0] * off, lo[1] + right[1] * off]
                })
                .collect()
        };
        let (base_a, base_b) = (base_cross(ai, true), base_cross(bi, false));

        // Per-boundary jogs, keyed independently on each side's boundary index
        // (the through wiring maps lane i → j, so boundaries i→j and i+1→j+1).
        // A *cleanly* wired movement — no merge sharing its target, no fork
        // splitting its source — is authoritative for its boundaries; merge and
        // fork correspondences (which disagree with the core by construction)
        // only speak where no clean movement does.
        let (mut up, mut dn) = (vec![[(0.0, 0u32); 2]; na + 1], vec![[(0.0, 0u32); 2]; nb + 1]);
        for &mid in &mids {
            let mv = net.movement(mid);
            let clean = !mids.iter().any(|&o| {
                o != mid && (net.movement(o).to_lane == mv.to_lane || net.movement(o).from_lane == mv.from_lane)
            });
            let tier = if clean { 0 } else { 1 };
            let (i, j) = (net.lane(mv.from_lane).index_in_link as usize, net.lane(mv.to_lane).index_in_link as usize);
            for (bi_k, bj_k) in [(i, j), (i + 1, j + 1)] {
                if bi_k > na || bj_k > nb {
                    continue;
                }
                let d = seam_jog(base_a[bi_k], da, base_b[bj_k], db);
                up[bi_k][tier] = (up[bi_k][tier].0 + d, up[bi_k][tier].1 + 1);
                dn[bj_k][tier] = (dn[bj_k][tier].0 + d, dn[bj_k][tier].1 + 1);
            }
        }
        let resolve = |acc: Vec<[(f64, u32); 2]>| -> Option<Vec<f64>> {
            let known: Vec<Option<f64>> = acc
                .iter()
                .map(|&[(cs, cc), (ms, mc)]| {
                    if cc > 0 {
                        Some(cs / cc as f64)
                    } else if mc > 0 {
                        Some(ms / mc as f64)
                    } else {
                        None
                    }
                })
                .collect();
            known.iter().any(Option::is_some).then(|| {
                // Boundaries the wiring doesn't cover (bay edges, dropped lanes)
                // follow their nearest stitched neighbour.
                let nearest = |k: usize| {
                    (0..known.len())
                        .filter_map(|m| known[m].map(|v| (k.abs_diff(m), v)))
                        .min_by_key(|&(dist, _)| dist)
                        .map(|(_, v)| v)
                        .unwrap()
                };
                let filled: Vec<f64> = (0..known.len()).map(|k| known[k].unwrap_or_else(|| nearest(k))).collect();
                // The rigid part of the correction is vetted by
                // `seam_shift_allowed` below; per-boundary *deviation* from it
                // (lane fan-out at a width change) is capped, and wild adjacent
                // disagreement means the wiring can't be trusted lane-by-lane —
                // stitch rigidly then.
                let mean = filled.iter().sum::<f64>() / filled.len() as f64;
                if filled.windows(2).any(|w| (w[1] - w[0]).abs() > PINCH) {
                    return vec![mean; filled.len()];
                }
                filled.iter().map(|&v| mean + (v - mean).clamp(-MAX_SHIFT, MAX_SHIFT)).collect()
            })
        };
        let (Some(up), Some(dn)) = (resolve(up), resolve(dn)) else { continue };
        let pair_mean = up.iter().sum::<f64>() / up.len() as f64;
        let gated = !seam_shift_allowed(net, a, b, pair_mean);
        if std::env::var("STITCH_DEBUG").is_ok() && (gated || pair_mean.abs() > 0.05) {
            eprintln!(
                "ST {a}->{b} mean={pair_mean:.2} up={:?} gated={gated} ({} / {})",
                up.iter().map(|d| (d * 100.0).round() / 100.0).collect::<Vec<_>>(),
                net.link_names[ai],
                net.link_names[bi],
            );
        }
        if gated {
            continue;
        }
        plan.push((ai, true, up.iter().map(|d| d * 0.5).collect(), left));
        plan.push((bi, false, dn.iter().map(|d| -d * 0.5).collect(), left));
    }

    // Each end's correction rides an affine basis over the link: full strength
    // from its stop line out to its box edge, fading linearly to exactly zero
    // at the *opposite* stop line. The two ends of a link therefore decouple —
    // a link corrected at both ends shears gently between two exact values —
    // where a taper-with-hold field made overlapping setbacks fight and
    // oscillate on short links. Interiors attach at the stop lines, which is
    // exactly where the basis is pinned. One pass closes every seam; a second
    // would misread the shear this pass legitimately leaves in a merge seam's
    // cross-section.
    for (li, at_end, shifts, left) in plan {
        let lane = net.lanes[net.links[li].lane_start.idx()];
        let (stop0, stop1) = (lane.start_offset, lane.start_offset + lane.length);
        let lb = &mut net.lane_bounds[li];
        for si in 0..lb.stations.len() {
            let u = ((lb.stations[si] - stop0) / (stop1 - stop0).max(1e-9)).clamp(0.0, 1.0);
            let w = if at_end { u } else { 1.0 - u };
            if w <= 0.0 {
                continue;
            }
            for (k, &shift) in shifts.iter().enumerate() {
                lb.bounds[k][si][0] += left[0] * shift * w;
                lb.bounds[k][si][1] += left[1] * shift * w;
            }
        }
    }
}

fn offset_ramps_to_curb(net: &mut Network) {
    const TRANSITION: f64 = 45.0;
    // Freeway travel direction and width at each node it touches (through-direction). A
    // node can have both a wide mainline and a narrower continuation; the ramp peels off
    // the *widest* one it meets, so keep that — taking whichever freeway happened to be
    // indexed first would under-shift the ramp onto the mainline's inner lanes.
    let mut freeway_at: HashMap<u32, ([f64; 2], f64)> = HashMap::new();
    let widen = |map: &mut HashMap<u32, ([f64; 2], f64)>, node: u32, dir: [f64; 2], lanes: f64| {
        let e = map.entry(node).or_insert((dir, lanes));
        if lanes > e.1 {
            *e = (dir, lanes);
        }
    };
    for fi in 0..net.links.len() {
        let f = net.links[fi];
        if f.kind != RoadKind::Freeway {
            continue;
        }
        let lanes = f.lane_count as f64;
        widen(&mut freeway_at, f.to.0, net.arrival_dir(LinkId(fi as u32)), lanes);
        widen(&mut freeway_at, f.from.0, net.departure_dir(LinkId(fi as u32)), lanes);
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
            // Both polylines are centred on their own carriageways (see
            // `center_oneway_axes`), so aligning the ramp's pavement with the
            // freeway's curb-most lanes takes half the width difference.
            let mag = (flanes - ramp_lanes) * LANE_WIDTH / 2.0;
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

/// Guarantee every movement's exit mouth sits *ahead* of its entry mouth along
/// the arrival direction, by pulling an offending approach's stop line further
/// upstream. At a handful of shallow-angle or tightly-packed nodes the chart
/// puts an incoming lane's end past (or nearly on top of) the outgoing lane's
/// start; the interior connecting them is then physically undrivable — a car
/// lands behind its own bumper, rotated, and has to drive a full-lock recovery
/// loop through the box (the Richmond → Laurel pattern: landed 5 m behind-left,
/// 95° off). Trimming the from-link's lanes moves its mouth back along its own
/// chart line, so the corner Bézier built afterwards is a forward, steerable
/// path. Runs on trimmed lane lengths only — the chart, polylines, and end
/// directions are untouched; must run before [`Network::build_interiors`].
fn enforce_mouth_ordering(net: &mut Network) {
    // A turn needs at least this much forward room between mouths to be
    // steerable; and no stop line moves more than this per pass (a defective
    // cluster shouldn't relocate a stop line halfway up the block).
    const MARGIN: f64 = 1.0;
    const MAX_TRIM: f64 = 18.0;
    const MIN_LANE: f64 = 3.0;
    for _ in 0..4 {
        let mut end_trim = vec![0.0f64; net.links.len()];
        let mut start_push = vec![0.0f64; net.links.len()];
        for mv in &net.movements {
            let (fl, tl) = (net.lane(mv.from_lane).link, net.lane(mv.to_lane).link);
            let arr = net.arrival_dir(fl);
            let dep = net.departure_dir(tl);
            let dot = arr[0] * dep[0] + arr[1] * dep[1];
            // Only genuine *turns* need forward room between their mouths — an
            // undrivable turn handoff dumps a car behind itself, rotated.
            // Reversals legitimately have backward chords, and aligned seams
            // (freeway continuations, corridor bends) are already served by the
            // straight-seam interior + landing rebase; trimming those shifted
            // freeway seam geometry and broke merges.
            if !(-0.5..0.7).contains(&dot) {
                continue;
            }
            let entry = net.lane_point(mv.from_lane, net.lane(mv.from_lane).length);
            let exit = net.lane_point(mv.to_lane, 0.0);
            let fwd = (exit[0] - entry[0]) * arr[0] + (exit[1] - entry[1]) * arr[1];
            if fwd < MARGIN {
                end_trim[fl.idx()] = end_trim[fl.idx()].max(MARGIN - fwd);
                if dot > 0.3 {
                    start_push[tl.idx()] = start_push[tl.idx()].max((MARGIN - fwd) / dot);
                }
            }
        }
        let mut changed = false;
        for li in 0..net.links.len() {
            let link = net.links[li];
            for lane in link.lane_start.0..link.lane_start.0 + link.lane_count {
                let l = &mut net.lanes[lane as usize];
                let trim = end_trim[li].min(MAX_TRIM).min(l.length - MIN_LANE);
                if trim > 0.01 {
                    l.length -= trim;
                    changed = true;
                }
            }
        }
        // Second lever, applied only where the approach could not make room.
        let mut deficit = vec![0.0f64; net.links.len()];
        for mv in &net.movements {
            let (fl, tl) = (net.lane(mv.from_lane).link, net.lane(mv.to_lane).link);
            let arr = net.arrival_dir(fl);
            let dep = net.departure_dir(tl);
            if !(0.3..0.7).contains(&(arr[0] * dep[0] + arr[1] * dep[1])) {
                continue;
            }
            let entry = net.lane_point(mv.from_lane, net.lane(mv.from_lane).length);
            let exit = net.lane_point(mv.to_lane, 0.0);
            let fwd = (exit[0] - entry[0]) * arr[0] + (exit[1] - entry[1]) * arr[1];
            if fwd < MARGIN {
                deficit[tl.idx()] = deficit[tl.idx()].max(start_push[tl.idx()].min(MARGIN - fwd + 2.0));
            }
        }
        for li in 0..net.links.len() {
            let link = net.links[li];
            for lane in link.lane_start.0..link.lane_start.0 + link.lane_count {
                let l = &mut net.lanes[lane as usize];
                let push = deficit[li].min(MAX_TRIM).min(l.length - MIN_LANE);
                if push > 0.01 {
                    l.start_offset += push;
                    l.length -= push;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
}

/// Give back a junction cluster's internal storage. `set_junction_setbacks`
/// (which runs before clusters exist) applies stop-line setbacks to both ends of
/// every link; for the short links *between* one cluster's member nodes the two
/// setbacks exceed the link and the clamp leaves a ~1 m lane — so no car ever
/// fits any interior of a split junction (admission needs a vehicle length plus
/// a gap), and El Camino × Millbrae Ave served ~66 veh/h across 19 approach
/// lanes while its signals sat green. With the *final* clustering known, cap
/// interior link ends at a small mouth and rebuild the trim-derived geometry
/// (bounds → seam stitching → mouth ordering → interiors) so every consumer —
/// dividers, box decomposition, admission — agrees on what is interior.
fn relax_junction_interior_storage(net: &mut Network) {
    const INTERIOR_MOUTH: f64 = 2.5;
    let mut changed = false;
    for i in 0..net.links.len() {
        if !net.link_is_junction_interior(LinkId(i as u32)) {
            continue;
        }
        let link = net.links[i];
        let full: f64 = net.polylines[i].windows(2).map(|w| distance(w[0], w[1])).sum();
        for lane in link.lane_start.0..link.lane_start.0 + link.lane_count {
            let l = &mut net.lanes[lane as usize];
            let r1 = (full - l.start_offset - l.length).min(INTERIOR_MOUTH).max(0.0);
            let r0 = l.start_offset.min(INTERIOR_MOUTH);
            if (l.start_offset - r0).abs() > 0.01 || (full - r0 - r1 - l.length).abs() > 0.01 {
                l.start_offset = r0;
                l.length = full - r0 - r1;
                changed = true;
            }
        }
    }
    if changed {
        net.build_lane_bounds();
        stitch_seam_bounds(net);
        enforce_mouth_ordering(net);
        net.build_interiors();
    }
}

fn set_junction_setbacks(net: &mut Network) {
    // A crosswalk/stop-bar margin so vehicles halt just behind the box, as at a
    // real signalized intersection, rather than nosing into the crossing.
    const STOP_MARGIN: f64 = 2.5;
    // Crossings shallower than this (|approach · band normal|) don't gate the stop
    // line: a parallel road — the boulevard's other carriageway, this road's own
    // continuation — never crosses the approach.
    const MIN_CROSS: f64 = 0.25;
    // Ceiling on one band's required clearance, so a near-parallel skew crossing
    // in bad OSM data can't push a stop line absurdly far up the road.
    const CLEAR_CAP: f64 = 45.0;
    // How far past its anchor node's box a band still counts as junction sweep
    // (wide-band corner overhang); beyond it the crossing is an infinite-strip
    // fiction and demands no clearance.
    const REACH_SLACK: f64 = 6.0;
    // Box radius per node = half the widest carriageway meeting it — the drawn
    // road stops here. The *driving* stop line per approach is computed below
    // against the actual crossing carriageways and is usually further back.
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
    let interchange: Vec<bool> = crate::sim::network::pmap(net.nodes.len(), |n| net.is_interchange_node(NodeId(n as u32)));
    for n in 0..net.nodes.len() {
        if interchange[n] {
            box_r[n] = box_r[n].min(0.5);
        }
    }
    net.render_setback = box_r.clone();
    // Approach headings sampled at the box edge, past any end-of-link jag, so the
    // bands below (and every later consumer) see the road's real stop-line direction.
    net.build_end_dirs();

    let full_len: Vec<f64> = crate::sim::network::pmap(net.polylines.len(), |i| net.polylines[i].windows(2).map(|w| distance(w[0], w[1])).sum());
    let n = net.nodes.len();

    // Cluster nodes exactly as `build_junctions` will (short internal links between
    // intersection nodes are one junction), so an approach's stop line clears every
    // carriageway of its whole junction — a split boulevard's far half included —
    // not just the roads sharing its own node.
    let mut nb: Vec<BTreeSet<u32>> = vec![Default::default(); n];
    for l in &net.links {
        if l.layer != 0 {
            continue;
        }
        nb[l.from.idx()].insert(l.to.0);
        nb[l.to.idx()].insert(l.from.0);
    }
    let is_ix = |i: usize| nb[i].len() >= 3;
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut Vec<usize>, mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    // Same short-gap fixpoint as `build_junctions`: the stub chain leading into a
    // junction (bend/signal nodes between the outer stop line and the box) is part
    // of the cluster, so stop lines are computed against the same junction the
    // behavioral model uses.
    let short: Vec<(usize, usize)> = net
        .links
        .iter()
        .enumerate()
        .filter_map(|(i, l)| {
            (l.layer == 0 && full_len[i] - box_r[l.from.idx()] - box_r[l.to.idx()] < JUNCTION_MERGE_GAP)
                .then_some((l.from.idx(), l.to.idx()))
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
    let root: Vec<usize> = (0..n).map(|i| find(&mut parent, i)).collect();
    let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &r) in root.iter().enumerate() {
        members.entry(r).or_default().push(i);
    }

    // Carriageway bands at each node: for every link end, the strip its lanes occupy —
    // anchored at the polyline end (the pavement's median edge, wherever the axis
    // was shifted), along the link's local direction, `lanes · LANE_WIDTH` wide on
    // the right of travel. `side` marks which longitudinal half-plane is the
    // band's *junction* side (+1: traffic continues past the anchor into the box —
    // an arriving end; −1: the box is behind — a departing end); the street side is
    // the other half. (link, anchor, travel dir, width, layer, side).
    let mut bands: Vec<Vec<(usize, [f64; 2], [f64; 2], f64, i32, f64)>> = vec![Vec::new(); n];
    for (i, l) in net.links.iter().enumerate() {
        let w = l.lane_count as f64 * LANE_WIDTH;
        let lid = LinkId(i as u32);
        bands[l.from.idx()].push((i, net.polylines[i][0], net.departure_dir(lid), w, l.layer, -1.0));
        bands[l.to.idx()].push((i, *net.polylines[i].last().unwrap(), net.arrival_dir(lid), w, l.layer, 1.0));
    }

    let setbacks: Vec<(f64, f64)> = {
        // The clearance one link end needs: the smallest distance from the node along
        // the link past which its whole lane band sits outside every crossing band of
        // the cluster — where a real stop bar sits, clear of the cross street's full
        // width. Solved on the band's linear lateral coordinate (`away` tilts it by g,
        // the own band's breadth by al; corners suffice by linearity): exact for
        // straight crossings at any angle — a perpendicular road demands its full
        // width, a skew one proportionally more.
        let clear_of = |link: usize, node: usize, p: [f64; 2], away: [f64; 2], n_tr: [f64; 2], w_own: f64| -> f64 {
            if !is_ix(node) {
                return 0.0; // a bend or lane-count change crosses nothing
            }
            let mut need = 0.0f64;
            for &m in &members[&root[node]] {
                for &(bl, q, d, w, layer, side) in &bands[m] {
                    if bl == link || layer != net.links[link].layer {
                        continue;
                    }
                    let ni = [d[1], -d[0]];
                    let g = away[0] * ni[0] + away[1] * ni[1];
                    if g.abs() < MIN_CROSS {
                        continue;
                    }
                    let l0 = (p[0] - q[0]) * ni[0] + (p[1] - q[1]) * ni[1];
                    let al = n_tr[0] * ni[0] + n_tr[1] * ni[1];
                    let u = if g > 0.0 {
                        (0.0f64).max((w - l0) / g).max((w - l0 - w_own * al) / g)
                    } else {
                        (0.0f64).max(l0 / -g).max((l0 + w_own * al) / -g)
                    };
    // A band is only real where its link's pavement actually lies: on the
                    // street side (against `side`) up to the link's own length, and on
                    // the junction side just across the anchor node's box — beyond
                    // that, other links' own bands cover the pavement. Treating the
                    // strip as infinite let distant cluster members (a split
                    // junction's signal stubs, internal segments) demand clearances at
                    // crossing points their road never reaches, capping stop lines
                    // ~45 m up the approach — the "cars halt mid-intersection"
                    // artifact at big divided junctions (El Camino × Millbrae Ave).
                    let s = (p[0] + u * away[0] - q[0]) * d[0] + (p[1] + u * away[1] - q[1]) * d[1];
                    let along = s * side; // >0: junction side of the anchor; <0: up the link's own street
                    let reach = if along > 0.0 { box_r[m] } else { full_len[bl] };
                    if along.abs() > reach + REACH_SLACK {
                        continue;
                    }
                    need = need.max(u.min(CLEAR_CAP));
                }
            }
            need
        };
        (0..net.links.len())
            .map(|i| {
                let link = net.links[i];
                let w_own = link.lane_count as f64 * LANE_WIDTH;
                let lid = LinkId(i as u32);
                let dep = net.departure_dir(lid);
                let arr = net.arrival_dir(lid);
                let (fi, ti) = (link.from.idx(), link.to.idx());
                let mut r0 = box_r[fi];
                let mut r1 = box_r[ti];
                if !interchange[fi] {
                    r0 = r0.max(clear_of(i, fi, net.polylines[i][0], dep, [dep[1], -dep[0]], w_own));
                }
                if !interchange[ti] {
                    r1 = r1.max(clear_of(i, ti, *net.polylines[i].last().unwrap(), [-arr[0], -arr[1]], [arr[1], -arr[0]], w_own));
                }
                (r0, r1)
            })
            .collect()
    };

    let radius = |r: f64, node: usize| if r <= 0.0 { 0.0 } else if interchange[node] { r } else { r + STOP_MARGIN };
    for i in 0..net.links.len() {
        let link = net.links[i];
        let full = full_len[i];
        let (mut r0, mut r1) = (radius(setbacks[i].0, link.from.idx()), radius(setbacks[i].1, link.to.idx()));
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
        #[serde(default)]
        rail_crossing: bool,
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
        #[serde(default)]
        hov_lanes: String,
        #[serde(default)]
        aadt: f64,
        #[serde(default)]
        res_weight: f64,
        #[serde(default)]
        attr_weight: f64,
        /// `"stop"` / `"yield"`: a per-approach sign on this directed link.
        #[serde(default)]
        sign: String,
    }

    #[derive(Deserialize)]
    struct JsonRestriction {
        from: [i64; 2],
        to: [i64; 2],
        kind: String,
    }

    #[derive(Deserialize)]
    struct JsonMap {
        nodes: Vec<JsonNode>,
        links: Vec<JsonLink>,
        #[serde(default)]
        restrictions: Vec<JsonRestriction>,
    }

    pub fn parse(s: &str) -> Result<(OsmMap, Vec<RestrictionSpec>), String> {
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
                NodeSpec { osm_id: n.osm_id, x: n.x, y: n.y, control, rail_crossing: n.rail_crossing }
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
                hov_lanes: l.hov_lanes,
                aadt: l.aadt,
                res_weight: l.res_weight,
                attr_weight: l.attr_weight,
                sign: match l.sign.as_str() {
                    "stop" => LinkSign::Stop,
                    "yield" => LinkSign::Yield,
                    _ => LinkSign::None,
                },
            })
            .collect();
        let restrictions = raw
            .restrictions
            .into_iter()
            .map(|r| RestrictionSpec {
                from: (r.from[0], r.from[1]),
                to: (r.to[0], r.to[1]),
                only: r.kind.starts_with("only_"),
            })
            .collect();
        Ok((OsmMap { nodes, links }, restrictions))
    }
}

/// Scraped bus-stop points (projected metres) from the map JSON's top-level
/// `bus_stops`, for [`Network::attach_bus_stops`]; empty when absent. Separate
/// from [`OsmMap`] so the many hand-built map literals stay untouched.
#[cfg(feature = "import")]
pub fn bus_stops_from_json(s: &str) -> Vec<[f64; 2]> {
    use serde::Deserialize;
    #[derive(Deserialize)]
    struct Doc {
        #[serde(default)]
        bus_stops: Vec<[f64; 2]>,
    }
    serde_json::from_str::<Doc>(s).map(|d| d.bus_stops).unwrap_or_default()
}

/// Scraped bus-line traces from the map JSON's top-level `bus_routes`
/// (`{name, pts}` per line), for [`Network::resolve_route_chain`].
#[cfg(feature = "import")]
pub fn bus_routes_from_json(s: &str) -> Vec<(String, Vec<[f64; 2]>)> {
    use serde::Deserialize;
    #[derive(Deserialize)]
    struct Route {
        name: String,
        pts: Vec<[f64; 2]>,
    }
    #[derive(Deserialize)]
    struct Doc {
        #[serde(default)]
        bus_routes: Vec<Route>,
    }
    serde_json::from_str::<Doc>(s)
        .map(|d| d.bus_routes.into_iter().map(|r| (r.name, r.pts)).collect())
        .unwrap_or_default()
}

impl OsmMap {
    /// Parse the OSM scraper's JSON (`tools/osm-scraper`) into an [`ImportedMap`]
    /// (the map plus its turn restrictions, which `.build()` honors), simplifying
    /// spurious pass-through nodes. Requires the `import` feature.
    #[cfg(feature = "import")]
    pub fn from_json(s: &str) -> Result<ImportedMap, String> {
        Self::from_json_opts(s, true)
    }

    /// [`from_json`] with the experimental extent-capped junction merge (`split_junctions`):
    /// large surface junctions stay split into aligned sub-nodes (see `merge_split_intersections`).
    #[cfg(feature = "import")]
    pub fn from_json_opts(s: &str, split_junctions: bool) -> Result<ImportedMap, String> {
        Self::from_json_opts_with_progress(s, split_junctions, &mut |_, _, _| {})
    }

    /// [`from_json_opts`] reporting each preprocessing pass, so a big map's
    /// parse phase shows movement instead of one silent stall.
    #[cfg(feature = "import")]
    pub fn from_json_opts_with_progress(
        s: &str,
        split_junctions: bool,
        cb: &mut dyn FnMut(&str, u32, u32),
    ) -> Result<ImportedMap, String> {
        let (map, mut restrictions) = json::parse(s)?;
        cb("parse", 1, 1);
        let map = map.relocate_sign_nodes();
        cb("simplify", 1, 3);
        let map = map.collapse_pass_through_nodes_with(&mut restrictions);
        cb("simplify", 2, 3);
        let map = map.merge_split_intersections_with(split_junctions, &mut restrictions);
        cb("simplify", 3, 3);
        Ok(ImportedMap { map, restrictions })
    }

    /// OSM surveys stop/give_way where the sign stands — often a stand-alone
    /// node on the way rather than on the junction it protects (the same
    /// convention as stop-line signals). A controlled 1-in/1-out node is such a
    /// stop line: demote it to a per-approach sign on its incoming link and
    /// free the node, so the collapse dissolves it and the sign rides the
    /// merged link to the real junction (`join_pass_through` keeps the
    /// stronger sign). Two-way pass-throughs (2-in/2-out) stay: with both
    /// directions controlled the protected junction is ambiguous.
    #[cfg(feature = "import")]
    fn relocate_sign_nodes(&self) -> OsmMap {
        let (mut indeg, mut outdeg): (HashMap<i64, u32>, HashMap<i64, u32>) = (HashMap::new(), HashMap::new());
        for l in &self.links {
            *outdeg.entry(l.from_osm).or_default() += 1;
            *indeg.entry(l.to_osm).or_default() += 1;
        }
        let mut map = self.clone();
        let mut demoted: HashMap<i64, LinkSign> = HashMap::new();
        for n in &mut map.nodes {
            let kind = match n.control {
                MapControl::Stop => LinkSign::Stop,
                MapControl::Yield => LinkSign::Yield,
                _ => continue,
            };
            if indeg.get(&n.osm_id) == Some(&1) && outdeg.get(&n.osm_id) == Some(&1) {
                demoted.insert(n.osm_id, kind);
                n.control = MapControl::Uncontrolled;
            }
        }
        for l in &mut map.links {
            if let Some(&k) = demoted.get(&l.to_osm) {
                l.sign = l.sign.max(k);
            }
        }
        map
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
                { "from_osm": 1, "to_osm": 2, "lanes": 2, "speed_limit": 20.0, "aadt": 26500,
                  "res_weight": 1.4, "attr_weight": 2.1 },
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
        // An embedded count (attach_counts.py --write-map) survives into the network,
        // as do the land-use weights from the scraper's --landuse pass.
        let counted = (0..net.links.len() as u32)
            .map(LinkId)
            .find(|&l| net.link_aadt(l) > 0.0)
            .expect("the counted link keeps its aadt");
        assert_eq!(net.link_aadt(counted), 26_500.0);
        assert_eq!(net.link_res_weight(counted), 1.4);
        assert_eq!(net.link_attr_weight(counted), 2.1);
        let plain = (0..net.links.len() as u32).map(LinkId).find(|&l| l != counted).unwrap();
        assert_eq!(net.link_res_weight(plain), 1.0, "no data → neutral weight");
    }

    /// A four-way at node 5 whose west approach runs through pass-through node 1
    /// (dissolved by the collapse, so any restriction naming the `[1, 5]` block
    /// must be rewritten onto the merged `[10, 5]` link to survive).
    fn cross_doc(restrictions: &str) -> String {
        let node = |id: i64, x: f64, y: f64| {
            format!(r#"{{ "osm_id": {id}, "x": {x:.1}, "y": {y:.1}, "control": "uncontrolled" }}"#)
        };
        let arm = |a: i64, b: i64| {
            format!(
                r#"{{ "from_osm": {a}, "to_osm": {b}, "lanes": 1, "speed_limit": 13.4 }},
                   {{ "from_osm": {b}, "to_osm": {a}, "lanes": 1, "speed_limit": 13.4 }}"#
            )
        };
        format!(
            r#"{{
                "nodes": [{}, {}, {}, {}, {}, {}],
                "links": [{}, {}, {}, {}, {}],
                "restrictions": [{restrictions}]
            }}"#,
            node(10, -400.0, 0.0),
            node(1, -200.0, 0.0),
            node(2, 0.0, 200.0),
            node(3, 200.0, 0.0),
            node(4, 0.0, -200.0),
            node(5, 0.0, 0.0),
            arm(10, 1),
            arm(1, 5),
            arm(2, 5),
            arm(3, 5),
            arm(4, 5),
        )
    }

    /// The link running `from`→`to`, located by node positions.
    fn link_between(net: &Network, from: [f64; 2], to: [f64; 2]) -> LinkId {
        LinkId(
            (0..net.links.len())
                .find(|&i| {
                    let l = &net.links[i];
                    net.node(l.from).position == from && net.node(l.to).position == to
                })
                .expect("link present") as u32,
        )
    }

    /// The exit-node positions (rounded to metres) reachable from the link
    /// `from`→`via`, straight off the movement wiring.
    fn exit_targets(net: &Network, from: [f64; 2], via: [f64; 2]) -> BTreeSet<(i64, i64)> {
        let li = link_between(net, from, via).0 as usize;
        net.outgoing_links(LinkId(li as u32))
            .into_iter()
            .map(|ol| {
                let p = net.node(net.link(ol).to).position;
                (p[0].round() as i64, p[1].round() as i64)
            })
            .collect()
    }

    #[test]
    fn turn_restrictions_prune_movements_across_the_collapse() {
        let west = [-400.0, 0.0];
        let center = [0.0, 0.0];
        let all = exit_targets(&OsmMap::from_json(&cross_doc("")).unwrap().build(), west, center);
        assert_eq!(all, BTreeSet::from([(0, 200), (200, 0), (0, -200)]), "unrestricted baseline");

        let banned = r#"{ "from": [1, 5], "to": [5, 2], "kind": "no_left_turn" }"#;
        let net = OsmMap::from_json(&cross_doc(banned)).unwrap().build();
        assert_eq!(
            exit_targets(&net, west, center),
            BTreeSet::from([(200, 0), (0, -200)]),
            "the left exit is gone; the restriction survived node 1's dissolution"
        );

        let only = r#"{ "from": [2, 5], "to": [5, 4], "kind": "only_straight_on" }"#;
        let net = OsmMap::from_json(&cross_doc(only)).unwrap().build();
        assert_eq!(
            exit_targets(&net, [0.0, 200.0], center),
            BTreeSet::from([(0, -200)]),
            "only_ keeps just the named exit"
        );
    }

    #[test]
    fn stranding_restrictions_fail_open() {
        let all_banned = r#"
            { "from": [1, 5], "to": [5, 2], "kind": "no_left_turn" },
            { "from": [1, 5], "to": [5, 3], "kind": "no_straight_on" },
            { "from": [1, 5], "to": [5, 4], "kind": "no_right_turn" }"#;
        let net = OsmMap::from_json(&cross_doc(all_banned)).unwrap().build();
        assert_eq!(
            exit_targets(&net, [-400.0, 0.0], [0.0, 0.0]),
            BTreeSet::from([(0, 200), (200, 0), (0, -200)]),
            "a restriction set that would strand the approach is ignored wholesale"
        );
    }

    /// A plain four-way at node 5 whose north/south (minor) approach links can
    /// carry per-approach signs; every node uncontrolled unless overridden.
    fn signed_cross_doc(center_control: &str, minor_sign: &str) -> String {
        let sign = |s: &str| if s.is_empty() { String::new() } else { format!(r#", "sign": "{s}""#) };
        format!(
            r#"{{
                "nodes": [
                    {{ "osm_id": 1, "x": -200.0, "y": 0.0, "control": "uncontrolled" }},
                    {{ "osm_id": 2, "x": 0.0, "y": 200.0, "control": "uncontrolled" }},
                    {{ "osm_id": 3, "x": 200.0, "y": 0.0, "control": "uncontrolled" }},
                    {{ "osm_id": 4, "x": 0.0, "y": -200.0, "control": "uncontrolled" }},
                    {{ "osm_id": 5, "x": 0.0, "y": 0.0, "control": "{center_control}" }}
                ],
                "links": [
                    {{ "from_osm": 1, "to_osm": 5, "lanes": 1, "speed_limit": 13.4 }},
                    {{ "from_osm": 5, "to_osm": 1, "lanes": 1, "speed_limit": 13.4 }},
                    {{ "from_osm": 2, "to_osm": 5, "lanes": 1, "speed_limit": 13.4{s} }},
                    {{ "from_osm": 5, "to_osm": 2, "lanes": 1, "speed_limit": 13.4 }},
                    {{ "from_osm": 3, "to_osm": 5, "lanes": 1, "speed_limit": 13.4 }},
                    {{ "from_osm": 5, "to_osm": 3, "lanes": 1, "speed_limit": 13.4 }},
                    {{ "from_osm": 4, "to_osm": 5, "lanes": 1, "speed_limit": 13.4{s} }},
                    {{ "from_osm": 5, "to_osm": 4, "lanes": 1, "speed_limit": 13.4 }}
                ]
            }}"#,
            s = sign(minor_sign),
        )
    }

    #[test]
    fn way_mapped_stop_signs_make_a_two_way_stop() {
        let net = OsmMap::from_json(&signed_cross_doc("uncontrolled", "stop")).unwrap().build();
        let center = [0.0, 0.0];
        let c = net.nodes.iter().position(|n| n.position == center).unwrap();
        assert!(matches!(net.nodes[c].control, NodeControl::Stop), "signed approaches make the node stop-controlled");
        assert!(net.approach_stops(link_between(&net, [0.0, 200.0], center)), "north minor stops");
        assert!(net.approach_stops(link_between(&net, [0.0, -200.0], center)), "south minor stops");
        assert!(!net.approach_stops(link_between(&net, [-200.0, 0.0], center)), "west major rolls");
        assert!(!net.approach_stops(link_between(&net, [200.0, 0.0], center)), "east major rolls");
        assert!(!net.all_way_stop(NodeId(c as u32)), "a two-way stop must not run the all-way FIFO");
    }

    #[test]
    fn junction_node_stops_stay_all_way() {
        let net = OsmMap::from_json(&signed_cross_doc("stop", "")).unwrap().build();
        let center = [0.0, 0.0];
        let c = net.nodes.iter().position(|n| n.position == center).unwrap();
        assert!(matches!(net.nodes[c].control, NodeControl::Stop));
        for arm in [[-200.0, 0.0], [0.0, 200.0], [200.0, 0.0], [0.0, -200.0]] {
            assert!(net.approach_stops(link_between(&net, arm, center)), "node-level stop lines every approach");
        }
        assert!(net.all_way_stop(NodeId(c as u32)), "node-level stop keeps the all-way protocol");
    }

    #[test]
    fn stop_line_nodes_relocate_to_their_junction() {
        // The minor street enters one-way through a stand-alone stop node 9 (the
        // OSM stop-line convention). It must dissolve, its sign riding the merged
        // approach to the junction: a two-way stop there, not a mid-block halt.
        let doc = r#"{
            "nodes": [
                { "osm_id": 1, "x": -200.0, "y": 0.0, "control": "uncontrolled" },
                { "osm_id": 2, "x": 0.0, "y": 300.0, "control": "uncontrolled" },
                { "osm_id": 9, "x": 0.0, "y": 220.0, "control": "stop" },
                { "osm_id": 3, "x": 200.0, "y": 0.0, "control": "uncontrolled" },
                { "osm_id": 5, "x": 0.0, "y": 0.0, "control": "uncontrolled" }
            ],
            "links": [
                { "from_osm": 1, "to_osm": 5, "lanes": 1, "speed_limit": 13.4 },
                { "from_osm": 5, "to_osm": 1, "lanes": 1, "speed_limit": 13.4 },
                { "from_osm": 2, "to_osm": 9, "lanes": 1, "speed_limit": 13.4 },
                { "from_osm": 9, "to_osm": 5, "lanes": 1, "speed_limit": 13.4 },
                { "from_osm": 3, "to_osm": 5, "lanes": 1, "speed_limit": 13.4 },
                { "from_osm": 5, "to_osm": 3, "lanes": 1, "speed_limit": 13.4 }
            ]
        }"#;
        let net = OsmMap::from_json(doc).unwrap().build();
        let center = [0.0, 0.0];
        assert!(
            !net.nodes.iter().any(|n| n.position == [0.0, 220.0]),
            "the stand-alone stop node dissolves"
        );
        let c = net.nodes.iter().position(|n| n.position == center).unwrap();
        assert!(matches!(net.nodes[c].control, NodeControl::Stop));
        assert!(net.approach_stops(link_between(&net, [0.0, 300.0], center)), "the sign rode down to the junction");
        assert!(!net.approach_stops(link_between(&net, [-200.0, 0.0], center)), "the cross street keeps rolling");
        assert!(!net.all_way_stop(NodeId(c as u32)));
    }

    #[test]
    fn via_way_restrictions_bind_once_the_junction_merges() {
        // A divided crossing: median stub 5–6 (8 m, merged as cluster interior),
        // west approach at 5, east/south arms at 6, north arm at 5. The via-way
        // restriction's ends land on the merged cluster's one node, where it bans
        // the through movement.
        let doc = r#"{
            "nodes": [
                { "osm_id": 1, "x": -200.0, "y": 0.0, "control": "uncontrolled" },
                { "osm_id": 2, "x": 0.0, "y": 200.0, "control": "uncontrolled" },
                { "osm_id": 3, "x": 208.0, "y": 0.0, "control": "uncontrolled" },
                { "osm_id": 4, "x": 8.0, "y": -200.0, "control": "uncontrolled" },
                { "osm_id": 5, "x": 0.0, "y": 0.0, "control": "uncontrolled" },
                { "osm_id": 6, "x": 8.0, "y": 0.0, "control": "uncontrolled" }
            ],
            "links": [
                { "from_osm": 1, "to_osm": 5, "lanes": 1, "speed_limit": 13.4 },
                { "from_osm": 5, "to_osm": 1, "lanes": 1, "speed_limit": 13.4 },
                { "from_osm": 5, "to_osm": 6, "lanes": 1, "speed_limit": 13.4 },
                { "from_osm": 6, "to_osm": 5, "lanes": 1, "speed_limit": 13.4 },
                { "from_osm": 2, "to_osm": 5, "lanes": 1, "speed_limit": 13.4 },
                { "from_osm": 5, "to_osm": 2, "lanes": 1, "speed_limit": 13.4 },
                { "from_osm": 3, "to_osm": 6, "lanes": 1, "speed_limit": 13.4 },
                { "from_osm": 6, "to_osm": 3, "lanes": 1, "speed_limit": 13.4 },
                { "from_osm": 4, "to_osm": 6, "lanes": 1, "speed_limit": 13.4 },
                { "from_osm": 6, "to_osm": 4, "lanes": 1, "speed_limit": 13.4 }
            ],
            "restrictions": [
                { "from": [1, 5], "to": [6, 3], "kind": "no_straight_on" }
            ]
        }"#;
        let net = OsmMap::from_json(doc).unwrap().build();
        let exits = exit_targets(&net, [-200.0, 0.0], [4.0, 0.0]);
        assert!(!exits.contains(&(208, 0)), "the through exit across the median is banned: {exits:?}");
        assert!(exits.contains(&(0, 200)) && exits.contains(&(8, -200)), "the turns survive: {exits:?}");
    }

    /// The junction-end setback of every approach into and exit out of `j`, in
    /// metres: how far the drivable lane span stops short of (or starts past)
    /// the junction node — where cars actually halt and resume.
    fn junction_end_setbacks(net: &Network, j: &crate::sim::network::Junction) -> Vec<(u32, f64)> {
        let full = |l: LinkId| -> f64 { net.polylines[l.idx()].windows(2).map(|w| distance(w[0], w[1])).sum() };
        let mut out = Vec::new();
        for &l in &j.approaches {
            let lane = net.lane(net.link(l).lane_start);
            out.push((l.0, full(l) - lane.start_offset - lane.length));
        }
        for &l in &j.exits {
            out.push((l.0, net.lane(net.link(l).lane_start).start_offset));
        }
        out
    }

    /// Regression for cars halting far from (or inside) complex intersections:
    /// junction fixture #0 is the real El Camino Real × Millbrae Avenue crossing —
    /// two divided roads whose OSM form splits the junction over four crossing
    /// nodes plus signal stubs ~20 m up each approach. Treating carriageway bands
    /// as infinite strips let stubs demand clearances at crossing points their road
    /// never reaches, pinning stop lines at the 45 m cap — cars visibly stopped in
    /// the middle of the road/box. Bounded bands keep every stop line and exit
    /// mouth near the box edge, at right angles to its own lanes.
    #[test]
    fn complex_junction_stop_lines_sit_at_the_box_edge() {
        for n in 0..3 {
            let net = millbrae_junction(n);
            for (ji, j) in net.junctions.iter().enumerate() {
                for (l, setback) in junction_end_setbacks(&net, j) {
                    println!("fixture {n} junction {ji} link {l}: setback {setback:.1}");
                    assert!(
                        setback < 33.0,
                        "fixture {n} junction {ji} link {l}: stop line/exit mouth {setback:.1} m from the node — \
                         far beyond any crossing carriageway's width (infinite-strip clearance regression)",
                    );
                }
            }
        }
    }

    #[test]
    #[ignore] // dev tool: emits a committable fixture from the local map.json
    fn extract_complex_junction_fixtures() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
        let Ok(text) = std::fs::read_to_string(&path) else { return };
        let (raw, _) = json::parse(&text).expect("map json");
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
            // A one-way link's axis is recentred on its carriageway, so its ends sit
            // beside the node laterally — never further than the carriageway's width.
            let near = |p: [f64; 2], nd: NodeId| {
                let n = net.node(nd).position;
                let w = link.lane_count as f64 * LANE_WIDTH;
                (p[0] - n[0]).hypot(p[1] - n[1]) <= w
            };
            assert!(near(poly[0], link.from), "polyline starts beside its link's from-node");
            assert!(near(*poly.last().unwrap(), link.to), "polyline ends beside its link's to-node");
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
        // within a single signal program no conflicting movements are ever green
        // together (orthogonal approaches are linked), and greens/reds coexist (they
        // aren't stuck all-green). Cross-node conflicts spanning two programs at a
        // multi-node junction can't be separated by either signal — the runtime box
        // logic serializes those (see the collision-free real-map tests).
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
                if net.groups[a.idx()].program != net.groups[b.idx()].program {
                    continue; // separate programs; no single signal governs the pair
                }
                let ga = net.movement_state(c.a, t);
                let gb = net.movement_state(c.b, t);
                let permissive = net.movement_turn(c.a) == TurnType::Left || net.movement_turn(c.b) == TurnType::Left;
                assert!(
                    permissive || !(ga == SignalState::Green && gb == SignalState::Green),
                    "conflicting non-permissive movements both green under one program, t={t}"
                );
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

/// A small hand-built sample of Millbrae, CA geometry: three El Camino Real
/// blocks (north↔south, two lanes each way) with signalized cross streets,
/// offset for a green wave — the fallback scene when no scraped map loads.
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
    #[ignore] // diagnostic dump
    #[cfg(feature = "import")]
    fn dump_seam_residuals() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
        let Ok(txt) = std::fs::read_to_string(path) else { return };
        let net = OsmMap::from_json(&txt).unwrap().build();
        let partners = through_seam_partners(&net);
        println!("partners: {}", partners.len());
        let mut residuals: Vec<(f64, u32, u32, usize)> = Vec::new();
        for (a, b, mids) in &partners {
            let (ai, bi) = (*a as usize, *b as usize);
            let (lba, lbb) = (&net.lane_bounds[ai], &net.lane_bounds[bi]);
            let (da, db) = (net.arrival_dir(LinkId(*a)), net.departure_dir(LinkId(*b)));
            let (na, nb) = (net.links[ai].lane_count as usize, net.links[bi].lane_count as usize);
            for &mid in mids {
                let mv = net.movement(mid);
                let (i, j) = (net.lane(mv.from_lane).index_in_link as usize, net.lane(mv.to_lane).index_in_link as usize);
                for (bk, bj) in [(i, j), (i + 1, j + 1)] {
                    if bk > na || bj > nb {
                        continue;
                    }
                    let d = seam_jog(*lba.bounds[bk].last().unwrap(), da, lbb.bounds[bj][0], db);
                    residuals.push((d.abs(), *a, *b, bk));
                }
            }
        }
        residuals.sort_by(|x, y| y.0.total_cmp(&x.0));
        let mean = residuals.iter().map(|r| r.0).sum::<f64>() / residuals.len() as f64;
        println!("boundary residuals: n={} mean={mean:.3}", residuals.len());
        for (d, a, b, k) in residuals.iter().take(15) {
            println!(
                "  {d:.2} m  {a}->{b} boundary {k}  ({} / {})",
                net.link_names[*a as usize], net.link_names[*b as usize]
            );
        }
        // The el_camino metric's own movement set, midpoint-projected.
        let ecr = |l: LinkId| net.link_names[l.idx()].contains("El Camino");
        for m in 0..net.movements.len() as u32 {
            let mid = MovementId(m);
            let mv = net.movement(mid);
            let (fl, tl) = (net.lane(mv.from_lane).link, net.lane(mv.to_lane).link);
            if !ecr(fl) || !ecr(tl) || net.movement_turn(mid) != TurnType::Through || net.node_junction(mv.node).is_some() {
                continue;
            }
            let it = net.interior(mid);
            let (da, db) = (net.arrival_dir(fl), net.departure_dir(tl));
            let jog = seam_jog(it.entry, da, it.exit, db);
            let chord = ((it.exit[0] - it.entry[0]) * da[1] - (it.exit[1] - it.entry[1]) * da[0]).abs();
            let stitched = partners.iter().any(|&(a, b, _)| a == fl.0 && b == tl.0);
            println!("ECR seam {}->{} chord={chord:.2} midjog={jog:.2} stitched={stitched}", fl.0, tl.0);
        }
    }

    #[test]
    #[cfg(feature = "import")]
    fn through_seams_are_stitched_shut_across_the_map() {
        // The lane-boundary stitch's contract: wherever a one-way road continues
        // into its mutual-primary partner, a *uniquely wired* through movement
        // (no merge sharing its target, no bay landing) leaves one link's lane
        // and enters the next with no lateral step — measured by midpoint
        // extrapolation, so curvature and the junction gap drop out. This is the
        // "misalignment is unrepresentable" guarantee over the whole real map,
        // not just El Camino.
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/map.json");
        let Ok(txt) = std::fs::read_to_string(path) else { return };
        let net = OsmMap::from_json(&txt).unwrap().build();
        let mut jogs: Vec<f64> = Vec::new();
        for (a, b, mids) in through_seam_partners(&net) {
            let (da, db) = (net.arrival_dir(LinkId(a)), net.departure_dir(LinkId(b)));
            for &mid in &mids {
                let mv = net.movement(mid);
                let shared = mids.iter().any(|&o| o != mid && net.movement(o).to_lane == mv.to_lane);
                if shared || net.lane(mv.to_lane).pocket_taper > 0.0 || net.lane(mv.from_lane).pocket_taper > 0.0 {
                    continue;
                }
                let it = net.interior(mid);
                jogs.push(seam_jog(it.entry, da, it.exit, db).abs());
            }
        }
        jogs.sort_by(f64::total_cmp);
        let mean = jogs.iter().sum::<f64>() / jogs.len() as f64;
        let share = |bound: f64| jogs.iter().filter(|&&j| j < bound).count() as f64 / jogs.len() as f64;
        eprintln!(
            "stitched seams: n={} mean={mean:.3} <5cm={:.2} <20cm={:.2} <50cm={:.2} p95={:.2} max={:.2}",
            jogs.len(),
            share(0.05),
            share(0.2),
            share(0.5),
            jogs[(jogs.len() * 95) / 100],
            jogs.last().copied().unwrap_or(0.0)
        );
        assert!(jogs.len() >= 300, "enough uniquely-wired through seams to measure ({})", jogs.len());
        // Not every seam may stitch — a genuine dogleg past `seam_shift_allowed`
        // stays put by design — but the overwhelming share must close to nothing
        // (at this writing: 81% under 5 cm, 98% under 20 cm, p95 = 0.12 m).
        assert!(share(0.05) > 0.75, "only {:.2} of through seams are stitched shut — the boundary stitch regressed", share(0.05));
        assert!(share(0.2) > 0.9, "only {:.2} of through seams within 20 cm — the boundary stitch regressed", share(0.2));
    }

    #[test]
    fn through_streams_never_land_in_a_closed_bay() {
        // A 2-lane road continues into a 3-lane approach whose flanking lanes are
        // dedicated turn pockets (left/through/right exits channelise them). The
        // through streams must land on the genuine through lane — a bay is merged
        // shut at the seam, so landing there is a lane-width sidestep onto its
        // neighbour's line — and the confluence becomes an explicit merge.
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::uncontrolled(2, 300.0, 0.0),
                NodeSpec::uncontrolled(3, 400.0, 0.0),
                NodeSpec::uncontrolled(4, 400.0, 200.0),  // left exit
                NodeSpec::uncontrolled(5, 500.0, 0.0),    // through exit
                NodeSpec::uncontrolled(6, 400.0, -200.0), // right exit
            ],
            links: vec![
                LinkSpec::oneway(1, 2, 2, 15.0),
                LinkSpec::oneway(2, 3, 3, 15.0),
                LinkSpec::oneway(3, 4, 1, 15.0),
                LinkSpec::oneway(3, 5, 1, 15.0),
                LinkSpec::oneway(3, 6, 1, 15.0),
            ],
        }
        .build();
        let bay_count = net
            .lanes_of(LinkId(1))
            .filter(|&l| net.lane(l).pocket_taper > 0.0)
            .count();
        assert_eq!(bay_count, 2, "the flanking lanes are turn pockets");
        for m in 0..net.movements.len() as u32 {
            let mv = net.movement(MovementId(m));
            let (fl, tl) = (net.lane(mv.from_lane).link, net.lane(mv.to_lane).link);
            if fl == LinkId(0) && tl == LinkId(1) {
                assert_eq!(
                    net.lane(mv.to_lane).pocket_taper,
                    0.0,
                    "a through stream lands on the through lane, not a closed bay"
                );
            }
        }
    }

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
    fn pm_plan_reverses_the_green_wave() {
        // The build stores both plans: AM offsets progress along the corridor
        // walk; the PM set is its mirror, so the evening flush rides the other
        // direction the way real corridor timing plans rotate.
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
            links.extend(road(a, b, "Main Street", 2, 15.0));
        }
        for (n, e, name) in [(1, 10, "Cross A"), (2, 11, "Cross B"), (3, 12, "Cross C")] {
            links.extend(road(n, e, name, 1, 12.0));
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
        assert_eq!(net.am_offsets.len(), net.programs.len());
        assert_eq!(net.pm_offsets.len(), net.programs.len());
        let coordinated: Vec<usize> =
            (0..net.programs.len()).filter(|&p| net.programs[p].coordinated).collect();
        assert!(coordinated.len() >= 3, "the corridor coordinates");
        // The two plans genuinely differ (the corridor's non-root members carry
        // mirrored offsets), and the loaded plan is the AM one.
        assert!(
            coordinated.iter().any(|&p| (net.am_offsets[p] - net.pm_offsets[p]).abs() > 1.0),
            "AM and PM plans differ"
        );
        for &p in &coordinated {
            assert!((net.programs[p].offset - net.am_offsets[p]).abs() < 1e-9);
        }
    }

    #[test]
    fn corridor_cycles_harmonize_to_the_longest_member() {
        // Three signals on one named street with deliberately different plans:
        // without harmonization their cycles differ and any offset progression
        // drifts out of phase every round. The build stretches greens (never
        // masks) so every member shares the corridor's longest cycle.
        let plan = |g| SignalPlan { green_secs: g, yellow_secs: 4.0, offset: 0.0 };
        let road = |a, b, name: &str, lanes, sp| {
            let mut v = LinkSpec::twoway(a, b, lanes, sp).to_vec();
            for l in &mut v {
                l.name = name.to_string();
            }
            v
        };
        let mut links = Vec::new();
        for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 4)] {
            links.extend(road(a, b, "Main Street", 2, 15.0));
        }
        for (n, e, name) in [(1, 10, "Cross A"), (2, 11, "Cross B"), (3, 12, "Cross C")] {
            links.extend(road(n, e, name, 1, 12.0));
        }
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(0, -300.0, 0.0),
                NodeSpec::signalized(1, 0.0, 0.0, plan(14.0)),
                NodeSpec::signalized(2, 300.0, 0.0, plan(26.0)),
                NodeSpec::signalized(3, 600.0, 0.0, plan(18.0)),
                NodeSpec::uncontrolled(4, 900.0, 0.0),
                NodeSpec::uncontrolled(10, 0.0, -200.0),
                NodeSpec::uncontrolled(11, 300.0, -200.0),
                NodeSpec::uncontrolled(12, 600.0, -200.0),
            ],
            links,
        }
        .build();
        let cycles: Vec<f64> = net
            .nodes
            .iter()
            .filter_map(|n| match n.control {
                NodeControl::Signalized(p) => Some(net.programs[p.idx()].cycle_length()),
                _ => None,
            })
            .collect();
        assert_eq!(cycles.len(), 3);
        let max = cycles.iter().fold(0.0f64, |a, &b| a.max(b));
        for c in &cycles {
            assert!((c - max).abs() < 0.1, "every corridor member shares the longest cycle: {cycles:?}");
        }
        for p in &net.programs {
            assert!(p.coordinated, "harmonized corridors carry coordinated offsets");
        }
    }

    /// A two-signal Main-Street corridor with a cross street at each signal —
    /// coordination's minimal fixture.
    fn coordinated_corridor_net() -> Network {
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
        OsmMap {
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
        .build()
    }

    /// The upstream signal's Main-Street through group: (program, bit, movement).
    fn corridor_through(net: &Network) -> (usize, u8, MovementId) {
        (0..net.movements.len() as u32)
            .find_map(|m| {
                let mv = net.movement(MovementId(m));
                let (fl, tl) = (net.lane(mv.from_lane).link, net.lane(mv.to_lane).link);
                (mv.node == NodeId(1)
                    && net.movement_turn(MovementId(m)) == TurnType::Through
                    && net.link_names[fl.idx()] == "Main Street"
                    && net.link_names[tl.idx()] == "Main Street")
                    .then(|| {
                        let g = net.groups[mv.signal_group.unwrap().idx()];
                        (g.program.idx(), g.bit, MovementId(m))
                    })
            })
            .expect("a Main-Street through movement")
    }

    #[test]
    fn actuated_controller_honors_coordination_offsets_at_runtime() {
        use super::super::junction::SignalController;
        use super::super::signal::SignalState;

        let net = coordinated_corridor_net();
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

        // Semi-actuated coordination's contract: whenever the offset schedule has
        // the through phase green (well inside the window, clear of clearance
        // rounding), the live controller must be green too — side phases may
        // borrow or return the rest of the cycle, but the progression window is
        // inviolate. Full demand keeps every side phase competing, and the
        // staggered offsets must still show as staggered green instants.
        let cycle = net.programs[up.2].cycle_length();
        let steps = 2 * (cycle / 0.1).ceil() as usize + 4;
        let mut ctrl = SignalController::build(&net);
        let demand = super::super::junction::LaneSet::of(net.lanes.len(), 0..net.lanes.len() as u32);
        let mut clock = 0.0;
        // One warm-up cycle so the seeded runtime settles onto the schedule.
        while clock < cycle {
            ctrl.advance(&net, &demand, 0.1, &Default::default());
            clock += 0.1;
        }
        let mut staggered_instant = false;
        for _ in 0..steps {
            for (mid, bit, pid) in [up, down] {
                let sched = |t: f64| net.programs[pid].state_of(bit, t) == SignalState::Green;
                if sched(clock - 0.5) && sched(clock) && sched(clock + 0.5) {
                    assert_eq!(
                        ctrl.movement_state(&net, mid),
                        SignalState::Green,
                        "the progression window is guaranteed green at t={clock}",
                    );
                }
            }
            let (us, ds) = (ctrl.movement_state(&net, up.0), ctrl.movement_state(&net, down.0));
            if (clock * 10.0).round() as i64 % 20 == 0 {
                eprintln!("t={clock:.1} up={us:?} down={ds:?}");
            }
            if us == SignalState::Green && ds != SignalState::Green {
                staggered_instant = true;
            }
            ctrl.advance(&net, &demand, 0.1, &Default::default());
            clock += 0.1;
        }
        assert!(staggered_instant, "the wave reaches the upstream green before the downstream one");
    }

    #[test]
    fn a_protected_left_is_called_by_the_bay_not_the_through_queue() {
        // Per-lane detection: cars in the through lanes of the same approach
        // must not call the protected-left window; a car in the left's own lane
        // must. (With per-link detectors, any through queue rang the left bell.)
        use super::super::junction::SignalController;
        use super::super::signal::SignalState;
        let net = coordinated_corridor_net();
        let left = (0..net.movements.len() as u32)
            .find_map(|m| {
                let mv = net.movement(MovementId(m));
                (mv.node == NodeId(1)
                    && net.movement_turn(MovementId(m)) == TurnType::Left
                    && net.link_names[net.lane(mv.from_lane).link.idx()] == "Main Street")
                    .then_some(MovementId(m))
            })
            .expect("a Main-Street protected left at the upstream signal");
        let left_lane = net.movement(left).from_lane;
        let (pid, _bit, through_mid) = corridor_through(&net);
        let cycle = net.programs[pid].cycle_length();
        let steps = 3 * (cycle / 0.1).ceil() as usize;

        // Every through lane of the same link occupied — but not the left's lane.
        let link = net.lane(left_lane).link;
        let through_lanes: Vec<u32> = net.lanes_of(link).filter(|&l| l != left_lane).map(|l| l.0).collect();
        let run = |demand: &super::super::junction::LaneSet| -> bool {
            let mut ctrl = SignalController::build(&net);
            let mut left_green = false;
            for _ in 0..steps {
                ctrl.advance(&net, demand, 0.1, &Default::default());
                left_green |= ctrl.movement_state(&net, left) == SignalState::Green
                    && ctrl.movement_state(&net, through_mid) != SignalState::Green;
            }
            left_green
        };
        let lanes = net.lanes.len();
        assert!(
            !run(&super::super::junction::LaneSet::of(lanes, through_lanes.iter().copied())),
            "a through queue alone never opens the protected-left window"
        );
        let with_bay = through_lanes.iter().copied().chain([left_lane.0]);
        assert!(
            run(&super::super::junction::LaneSet::of(lanes, with_bay)),
            "a car in the bay calls and receives its protected window"
        );
    }

    #[test]
    fn a_coordinated_signal_rests_in_green_without_side_demand() {
        // The actuated half of semi-actuated coordination: with nobody on the
        // side street, the corridor keeps its green (the side phase is skipped)
        // instead of cycling on the fixed schedule; a real side arrival is still
        // served within a cycle.
        use super::super::junction::SignalController;
        use super::super::signal::SignalState;
        let net = coordinated_corridor_net();
        let (pid, _bit, mid) = corridor_through(&net);
        let cycle = net.programs[pid].cycle_length();
        let mut ctrl = SignalController::build(&net);
        // Demand only in the through movement's own lane: detection is per-lane,
        // so this places no call for any protected-left phase.
        let main = super::super::junction::LaneSet::of(net.lanes.len(), [net.movement(mid).from_lane.0]);
        let mut clock = 0.0;
        // Warm-up cycle, then measure green share over two cycles.
        while clock < cycle {
            ctrl.advance(&net, &main, 0.1, &Default::default());
            clock += 0.1;
        }
        let steps = 2 * (cycle / 0.1).ceil() as usize;
        let mut green = 0usize;
        for _ in 0..steps {
            if ctrl.movement_state(&net, mid) == SignalState::Green {
                green += 1;
            }
            ctrl.advance(&net, &main, 0.1, &Default::default());
        }
        let share = green as f64 / steps as f64;
        let scheduled = net.programs[pid].phases[net.programs[pid].coordinated_phase].green_secs / cycle;
        assert!(
            share > scheduled + 0.15,
            "resting in green beats the fixed schedule: live {share:.2} vs scheduled {scheduled:.2}"
        );
        assert_eq!(net.programs[pid].coordinated, true, "the corridor program is coordinated");
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
    fn conflicting_non_permissive_movements_never_green_together() {
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
                let permissive = net.movement_turn(c.a) == TurnType::Left || net.movement_turn(c.b) == TurnType::Left;
                assert!(
                    permissive || !(a == SignalState::Green && b == SignalState::Green),
                    "conflicting non-permissive greens at t={t}",
                );
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
                let permissive = net.movement_turn(c.a) == TurnType::Left || net.movement_turn(c.b) == TurnType::Left;
                assert!(
                    permissive
                        || !(net.movement_state(c.a, t) == SignalState::Green && net.movement_state(c.b, t) == SignalState::Green),
                    "conflicting non-permissive greens at t={t}"
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
    fn left_turns_also_show_a_permissive_green_with_the_through_phase() {
        use crate::sim::signal::SignalState;
        let net = signalized_cross();
        let cycle = net.programs[0].cycle_length();
        let mut saw_permissive = false;
        let mut t = 0.0;
        while t < cycle {
            for c in &net.conflicts {
                let has_left = net.movement_turn(c.a) == TurnType::Left || net.movement_turn(c.b) == TurnType::Left;
                if has_left
                    && net.movement_state(c.a, t) == SignalState::Green
                    && net.movement_state(c.b, t) == SignalState::Green
                {
                    saw_permissive = true;
                }
            }
            t += 0.1;
        }
        assert!(saw_permissive, "a left shows green concurrent with the through it must yield to");
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
                LinkSpec { from_osm: 3, to_osm: 4, lanes: 1, speed_limit: 25.0, geometry: Vec::new(), layer: 1, name: String::new(), road_class: String::new(), highway_ref: String::new(), turn_lanes: String::new(), hov_lanes: String::new(), aadt: 0.0, res_weight: 0.0, attr_weight: 0.0, sign: LinkSign::None }, // bridge over it
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

        // The curb lane is exit-only at the diverge: past the gore the ramp is physically
        // separated from the mainline, so the curb lane feeds only the ramp. A through car
        // reaches the freeway by changing lanes upstream, not by a merge at the node. The
        // inner lanes only continue.
        let curb_lane = net.lanes_of(LinkId(0)).last().unwrap();
        let mut curb_dests: Vec<u32> = (0..net.lane(curb_lane).movement_count)
            .map(|j| net.lane(net.movement(MovementId(net.lane(curb_lane).movement_start.0 + j)).to_lane).link.0)
            .collect();
        curb_dests.sort();
        assert_eq!(curb_dests, vec![2], "the curb lane is exit-only (feeds the ramp, not the mainline), got {curb_dests:?}");
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
    fn mainline_lane_pinch_is_widened_but_a_ramp_drop_is_not() {
        // OSM sometimes drops the lane tag on a mainline segment (defaulting to 1), pinching a
        // wide freeway to a single lane between full-width segments — the Golden Gate toll
        // approach runs 3->1->1->1->8. A wedged 1-lane artifact with no ramp to explain the
        // drop is widened back to the through width; a modest drop where a ramp takes the lane
        // (covered by `freeway_ramps_wire_to_the_curb_lane`) is left alone.
        let hw = |a, b, lanes| LinkSpec { road_class: "motorway".into(), ..LinkSpec::oneway(a, b, lanes, 29.0) };
        let ramp = |a, b, lanes| LinkSpec { road_class: "motorway_link".into(), ..LinkSpec::oneway(a, b, lanes, 25.0) };
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::uncontrolled(2, 200.0, 0.0),
                NodeSpec::uncontrolled(3, 300.0, 0.0),
                NodeSpec::uncontrolled(4, 500.0, 0.0),
                NodeSpec::uncontrolled(5, 250.0, -200.0), // off-ramp exit, well clear of the mainline
            ],
            links: vec![
                hw(1, 2, 3),   // 0: 3-lane freeway
                hw(2, 3, 1),   // 1: 1-lane artifact — the lanes vanish, no ramp explains it
                hw(3, 4, 3),   // 2: 3-lane freeway
                ramp(2, 5, 1), // 3: a real 1-lane off-ramp diverging at node 2
            ],
        }
        .build();
        assert_eq!(
            net.link(LinkId(1)).lane_count,
            3,
            "the wedged 1-lane mainline artifact is widened to the through width"
        );
        assert_eq!(net.link(LinkId(3)).lane_count, 1, "the real off-ramp stays one lane");
    }

    #[test]
    fn stop_lines_clear_the_full_crossing_carriageway() {
        // Two-way 4-way: E-W three lanes per direction, N-S two. An approach's stop
        // line must sit behind the *whole* width of the crossing carriageway on its
        // side — where a real stop bar is — not half of it: cars queued at the old
        // half-width line were parked inside the cross traffic's path and got hit.
        let two_way = |a: i64, b: i64, lanes| vec![LinkSpec::oneway(a, b, lanes, 15.0), LinkSpec::oneway(b, a, lanes, 15.0)];
        let mut links = Vec::new();
        links.extend(two_way(1, 5, 3)); // west arm
        links.extend(two_way(2, 5, 3)); // east arm
        links.extend(two_way(3, 5, 2)); // south arm
        links.extend(two_way(4, 5, 2)); // north arm
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -300.0, 0.0),
                NodeSpec::uncontrolled(2, 300.0, 0.0),
                NodeSpec::uncontrolled(3, 0.0, -300.0),
                NodeSpec::uncontrolled(4, 0.0, 300.0),
                NodeSpec::uncontrolled(5, 0.0, 0.0),
            ],
            links,
        }
        .build();
        // Eastbound approach (link 0, 1→5): the southbound carriageway (two lanes,
        // west of the N-S centreline) crosses its path — the stop line clears it.
        let east_lane = net.link(LinkId(0)).lane_start;
        let east_stop = net.lane_point(east_lane, net.lane(east_lane).length);
        assert!(east_stop[0] <= -2.0 * LANE_WIDTH, "eastbound stops behind the southbound carriageway: x={:.2}", east_stop[0]);
        // Northbound approach (link 4, 3→5): the eastbound carriageway (three lanes,
        // south of the E-W centreline) crosses its path.
        let north_lane = net.link(LinkId(4)).lane_start;
        let north_stop = net.lane_point(north_lane, net.lane(north_lane).length);
        assert!(north_stop[1] <= -3.0 * LANE_WIDTH, "northbound stops behind the eastbound carriageway: y={:.2}", north_stop[1]);
    }

    #[test]
    fn seam_interiors_run_straight_whatever_lane_the_movement_targets() {
        // A lane-count change at a mainline segment boundary can map a lane onto a
        // non-adjacent index (here lane 1 of 2 → lane 2 of 3). The crossing path
        // must not pack that lateral move into the node's ~1 m gap — the curve
        // doubles back and the car sweeps sideways pointing the wrong way. It runs
        // straight through on the entry lane's own line; the car eases onto its
        // target line on the next link via the seam-landing blend.
        let hw = |a, b, lanes| LinkSpec { road_class: "motorway".into(), ..LinkSpec::oneway(a, b, lanes, 29.0) };
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -400.0, 0.0),
                NodeSpec::uncontrolled(2, 0.0, 0.0),
                NodeSpec::uncontrolled(3, 400.0, 0.0),
            ],
            links: vec![hw(1, 2, 2), hw(2, 3, 3)],
        }
        .build();
        assert!(!net.movements.is_empty());
        for m in 0..net.movements.len() as u32 {
            let mid = MovementId(m);
            let it = net.interior(mid);
            assert!(it.len < 3.0, "a seam interior spans the node gap, not a lateral detour: len={}", it.len);
            for i in 0..=8 {
                let p = net.interior_point(mid, it.len * i as f64 / 8.0);
                assert!(
                    p[2].abs() < 0.05,
                    "the crossing path points down the road the whole way, got {:.1}°",
                    p[2].to_degrees()
                );
            }
        }
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
        // The recentred freeway carriageway spans y ∈ [-10.5, 10.5] (node on the
        // mapped mid-carriageway line). The ramp start began near the node and must
        // be slid into the curb-side lanes — around the curb lane's own span, well
        // clear of the median half.
        let ramp_start = net.polylines[2][0];
        assert!(
            (-11.0..=-6.0).contains(&ramp_start[1]),
            "the off-ramp peels off within the curb lane's span, start y = {}",
            ramp_start[1]
        );
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
        let merged = split_crossing().merge_split_intersections(false);
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
        let net = map.merge_split_intersections(false).build();
        assert_eq!(net.programs.len(), 1, "the merged junction has a single coordinated signal program");
        assert!(!net.conflicts.is_empty(), "the merged 4-way has crossing conflict points");
    }

    #[test]
    fn merge_leaves_ordinary_blocks_untouched() {
        // Nodes a normal block apart (>STUB_MAX) are not merged.
        let net_nodes = signalized_cross_map().merge_split_intersections(false);
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

    fn cross_with_turn_lanes(lanes: u32, turn_lanes: &str) -> Network {
        // Eastbound approach into a 4-way; left exit north (2 lanes, so a dual
        // left can fan), through east (3), right south (1).
        OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -200.0, 0.0),
                NodeSpec::uncontrolled(2, 0.0, 0.0),
                NodeSpec::uncontrolled(3, 200.0, 0.0),
                NodeSpec::uncontrolled(4, 0.0, 200.0),
                NodeSpec::uncontrolled(5, 0.0, -200.0),
            ],
            links: vec![
                LinkSpec { turn_lanes: turn_lanes.into(), ..LinkSpec::oneway(1, 2, lanes, 15.0) },
                LinkSpec::oneway(2, 3, 3, 15.0),
                LinkSpec::oneway(2, 4, 2, 15.0),
                LinkSpec::oneway(2, 5, 1, 15.0),
            ],
        }
        .build()
    }

    fn lane_turns(net: &Network, lane: LaneId) -> Vec<TurnType> {
        let l = net.lane(lane);
        (0..l.movement_count).map(|k| net.movement_turn(MovementId(l.movement_start.0 + k))).collect()
    }

    #[test]
    fn turn_lanes_channelize_movements_and_open_pockets() {
        let net = cross_with_turn_lanes(4, "left|through|through|right");
        let lanes: Vec<LaneId> = net.lanes_of(LinkId(0)).collect();
        assert!(lane_turns(&net, lanes[0]).iter().all(|&t| t == TurnType::Left), "median lane turns left only");
        assert!(lane_turns(&net, lanes[3]).iter().all(|&t| t == TurnType::Right), "curb lane turns right only");
        for &l in &lanes[1..3] {
            assert!(lane_turns(&net, l).iter().all(|&t| t == TurnType::Through), "middle lanes run through");
        }
        assert!(net.lane(lanes[0]).pocket_taper > 0.0, "the left bay tapers open");
        assert!(net.lane(lanes[3]).pocket_taper > 0.0, "the right bay tapers open");
        assert_eq!(net.lane(lanes[1]).pocket_taper, 0.0, "through lanes run full length");
    }

    #[test]
    fn dual_left_block_channelizes_and_pockets_both_lanes() {
        let net = cross_with_turn_lanes(5, "left|left|none|none|none");
        let lanes: Vec<LaneId> = net.lanes_of(LinkId(0)).collect();
        for &l in &lanes[..2] {
            assert!(lane_turns(&net, l).iter().all(|&t| t == TurnType::Left), "both bay lanes turn left only");
            assert!(net.lane(l).pocket_taper > 0.0, "both bay lanes taper open");
        }
        for &l in &lanes[2..4] {
            assert!(lane_turns(&net, l).iter().all(|&t| t == TurnType::Through));
        }
        // The south exit carries no arrow in the tag, so the curb `none` lane picks
        // it up alongside its through — every exit keeps a serving lane.
        assert!(lane_turns(&net, lanes[4]).contains(&TurnType::Through));
        for &l in &lanes[2..] {
            assert_eq!(net.lane(l).pocket_taper, 0.0);
        }
        // The dual left fans onto both receiving lanes rather than piling into one.
        let to_lane = |lane: LaneId| net.movements_of(lane)[0].to_lane;
        assert_ne!(to_lane(lanes[0]), to_lane(lanes[1]));
    }

    #[test]
    fn through_traffic_lands_on_through_lanes_across_a_widening() {
        // A 3-lane road widens to a 4-lane bay segment ("left|||") before the
        // junction — the OSM way split at the physical bay start. Through lanes
        // continue onto the through-marked lanes; the pocket fills from turners.
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -400.0, 0.0),
                NodeSpec::uncontrolled(2, -120.0, 0.0),
                NodeSpec::uncontrolled(3, 0.0, 0.0),
                NodeSpec::uncontrolled(4, 200.0, 0.0),
                NodeSpec::uncontrolled(5, 0.0, 200.0),
            ],
            links: vec![
                LinkSpec::oneway(1, 2, 3, 15.0),
                LinkSpec { turn_lanes: "left|none|none|none".into(), ..LinkSpec::oneway(2, 3, 4, 15.0) },
                LinkSpec::oneway(3, 4, 3, 15.0),
                LinkSpec::oneway(3, 5, 2, 15.0),
            ],
        }
        .build();
        let bay_lanes: Vec<LaneId> = net.lanes_of(LinkId(1)).collect();
        for lane in net.lanes_of(LinkId(0)) {
            for m in net.movements_of(lane) {
                assert_ne!(m.to_lane, bay_lanes[0], "no through lane continues into the left pocket");
            }
        }
    }

    #[test]
    fn malformed_turn_lanes_fall_back_to_angular_channelization() {
        // Two entries on a three-lane approach: the tag disagrees with the model,
        // so the angular slice takes over and every exit stays reachable.
        let net = cross_with_turn_lanes(3, "left|through");
        for exit in [LinkId(1), LinkId(2), LinkId(3)] {
            let served = net
                .lanes_of(LinkId(0))
                .flat_map(|l| net.movements_of(l).iter())
                .any(|m| net.lane(m.to_lane).link == exit);
            assert!(served, "exit {exit:?} keeps a serving lane");
        }
    }
}
