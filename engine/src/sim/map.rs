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

use std::collections::HashMap;

use super::network::*;
use super::signal::SignalProgram;

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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinkSpec {
    pub from_osm: i64,
    pub to_osm: i64,
    pub lanes: u32,
    pub speed_limit: f64,
}

impl LinkSpec {
    pub fn oneway(from_osm: i64, to_osm: i64, lanes: u32, speed_limit: f64) -> Self {
        Self { from_osm, to_osm, lanes, speed_limit }
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
            let length = distance(net.nodes[from.idx()].position, net.nodes[to.idx()].position);
            let link_id = LinkId(net.links.len() as u32);
            let lane_start = LaneId(net.lanes.len() as u32);
            for i in 0..spec.lanes {
                net.lanes.push(Lane {
                    link: link_id,
                    index_in_link: i,
                    length,
                    speed_limit: spec.speed_limit,
                    movement_start: MovementId(0),
                    movement_count: 0,
                });
            }
            net.links.push(Link { from, to, lane_start, lane_count: spec.lanes });
        }

        let mut group_of_link: HashMap<u32, SignalGroupId> = HashMap::new();
        for (i, spec) in self.nodes.iter().enumerate() {
            let MapControl::Signal(plan) = spec.control else {
                net.nodes[i].control = match spec.control {
                    MapControl::Stop => NodeControl::Stop,
                    MapControl::Yield => NodeControl::Yield,
                    _ => NodeControl::Uncontrolled,
                };
                continue;
            };
            let node = NodeId(i as u32);
            let incoming: Vec<u32> = (0..net.links.len() as u32)
                .filter(|&li| net.links[li as usize].to == node)
                .collect();
            let program = ProgramId(net.programs.len() as u32);
            net.programs.push(SignalProgram::round_robin(
                incoming.len().max(1),
                plan.green_secs,
                plan.yellow_secs,
                plan.offset,
            ));
            for (bit, li) in incoming.iter().enumerate() {
                let g = SignalGroupId(net.groups.len() as u32);
                net.groups.push(SignalGroup { program, bit: bit as u8 });
                group_of_link.insert(*li, g);
            }
            net.nodes[i].control = NodeControl::Signalized(program);
        }

        let mut movements: Vec<Movement> = Vec::new();
        for lane_id in 0..net.lanes.len() {
            let lane = net.lanes[lane_id];
            let link = net.links[lane.link.idx()];
            let node = link.to;
            let start = MovementId(movements.len() as u32);
            let group = group_of_link.get(&lane.link.0).copied();
            for out_li in 0..net.links.len() {
                let out = net.links[out_li];
                if out.from != node || out.to == link.from {
                    continue;
                }
                let to_index = lane.index_in_link.min(out.lane_count - 1);
                let to_lane = LaneId(out.lane_start.0 + to_index);
                movements.push(Movement { from_lane: LaneId(lane_id as u32), to_lane, node, signal_group: group });
            }
            net.lanes[lane_id].movement_start = start;
            net.lanes[lane_id].movement_count = movements.len() as u32 - start.0;
        }
        net.movements = movements;
        net
    }
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
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
            })
            .collect();
        Ok(OsmMap { nodes, links })
    }
}

impl OsmMap {
    /// Parse the OSM scraper's JSON (`tools/osm-scraper`) into an `OsmMap`.
    /// Requires the `import` feature.
    #[cfg(feature = "import")]
    pub fn from_json(s: &str) -> Result<OsmMap, String> {
        json::parse(s)
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
    fn malformed_json_is_an_error_not_a_panic() {
        assert!(OsmMap::from_json("{ not json").is_err());
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
    fn corridor_builds_expected_counts() {
        let net = corridor_with_signal();
        assert_eq!(net.nodes.len(), 5);
        assert_eq!(net.links.len(), 4);
        assert_eq!(net.lanes.len(), 2 + 2 + 1 + 1);
        assert_eq!(net.groups.len(), 2);
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
    fn millbrae_sample_builds_and_signals_are_consistent() {
        let net = millbrae_sample();
        assert!(net.programs.len() >= 2);
        let states = net.signal_states(30.0);
        assert_eq!(states.len(), net.groups.len());
    }
}
