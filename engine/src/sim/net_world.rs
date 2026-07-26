//! Vehicles driving a graph [`Network`]: IDM car-following within a lane, a red
//! signal (or unsatisfied movement) acting as a stationary virtual leader at
//! the stop line, and lane hand-off across a node when the movement is served.
//! Accelerations read committed pre-step state and apply in a second pass, the
//! same double-buffered shape the GPU mass layer will use.

use std::collections::HashMap;

use super::config::{DriverConfig, SimConfig};
use super::constraint::{self, LongContext, Obstacle, SpeedTarget};
use super::network::{LaneId, LinkId, MovementId, Network, NodeControl, NodeId};

#[derive(Clone, Debug, PartialEq)]
pub struct NetVehicle {
    pub id: u32,
    pub lane: LaneId,
    pub position: f64,
    pub speed: f64,
    pub driver: DriverConfig,
    pub route: Vec<LinkId>,
    pub route_idx: usize,
    /// The stop-controlled node this vehicle has already halted at, so a stop
    /// sign is enforced once rather than forever.
    stopped_at: Option<NodeId>,
}

pub struct NetWorld {
    pub network: Network,
    cfg: SimConfig,
    vehicles: Vec<NetVehicle>,
    time: f64,
    exited: u32,
    /// Downstream lanes fed by more than one lane — the merge points; value is
    /// the list of feeding (from) lane ids.
    merges: HashMap<u32, Vec<u32>>,
}

impl NetWorld {
    pub fn new(network: Network, cfg: SimConfig) -> Self {
        let mut merges: HashMap<u32, Vec<u32>> = HashMap::new();
        for m in &network.movements {
            let entry = merges.entry(m.to_lane.0).or_default();
            if !entry.contains(&m.from_lane.0) {
                entry.push(m.from_lane.0);
            }
        }
        merges.retain(|_, froms| froms.len() > 1);
        Self { network, cfg, vehicles: Vec::new(), time: 0.0, exited: 0, merges }
    }

    pub fn spawn(&mut self, id: u32, lane: LaneId, position: f64, speed: f64, driver: DriverConfig) {
        self.vehicles.push(NetVehicle {
            id, lane, position, speed, driver, route: Vec::new(), route_idx: 0, stopped_at: None,
        });
    }

    /// Spawn at the start of a precomputed link route; the vehicle takes the
    /// route-consistent movement at each intersection and exits on the last link.
    /// Returns `false` (spawn refused) if the route is empty or the entrance is
    /// still occupied, so demand can't stack vehicles on top of each other.
    pub fn spawn_routed(&mut self, id: u32, route: Vec<LinkId>, speed: f64, driver: DriverConfig) -> bool {
        let Some(&first) = route.first() else { return false };
        let Some(lane) = self.network.lanes_of(first).next() else { return false };
        if !self.entrance_clear(lane, driver.vehicle_length + driver.min_gap) {
            return false;
        }
        self.vehicles.push(NetVehicle {
            id, lane, position: 0.0, speed, driver, route, route_idx: 0, stopped_at: None,
        });
        true
    }

    pub fn entrance_clear(&self, lane: LaneId, clearance: f64) -> bool {
        self.vehicles.iter().filter(|v| v.lane == lane).all(|v| v.position > clearance)
    }

    pub fn time(&self) -> f64 {
        self.time
    }

    pub fn exited(&self) -> u32 {
        self.exited
    }

    pub fn vehicles(&self) -> &[NetVehicle] {
        &self.vehicles
    }

    pub fn vehicle(&self, id: u32) -> Option<&NetVehicle> {
        self.vehicles.iter().find(|v| v.id == id)
    }

    fn intended_movement(&self, veh: &NetVehicle) -> Option<MovementId> {
        let lane = self.network.lane(veh.lane);
        if lane.movement_count == 0 {
            return None;
        }
        if !veh.route.is_empty() {
            if veh.route_idx + 1 >= veh.route.len() {
                return None;
            }
            let next_link = veh.route[veh.route_idx + 1];
            for (k, m) in self.network.movements_of(veh.lane).iter().enumerate() {
                if self.network.lane(m.to_lane).link == next_link {
                    return Some(MovementId(lane.movement_start.0 + k as u32));
                }
            }
            return None;
        }
        Some(lane.movement_start)
    }

    fn neighbors(&self) -> Neighbors {
        let mut by_lane: HashMap<u32, Vec<usize>> = HashMap::new();
        let mut approaching: HashMap<u32, Vec<usize>> = HashMap::new();
        for (i, v) in self.vehicles.iter().enumerate() {
            by_lane.entry(v.lane.0).or_default().push(i);
            approaching.entry(self.downstream_node(v.lane).0).or_default().push(i);
        }
        let mut leader_of = vec![None; self.vehicles.len()];
        let mut lane_front = HashMap::new();
        for members in by_lane.values_mut() {
            members.sort_by(|&a, &b| {
                self.vehicles[a].position.total_cmp(&self.vehicles[b].position)
            });
            for w in members.windows(2) {
                leader_of[w[0]] = Some(w[1]);
            }
            let front = *members.first().unwrap();
            lane_front.insert(self.vehicles[front].lane.0, front);
        }
        Neighbors { leader_of, lane_front, by_lane, approaching }
    }

    fn downstream_node(&self, lane: LaneId) -> NodeId {
        self.network.link(self.network.lane(lane).link).to
    }

    /// A strict priority order over links (higher wins): faster road first, then
    /// more lanes, then link id as a deterministic tie-break so opposing yields
    /// can never deadlock.
    fn priority_key(&self, link: LinkId) -> u64 {
        let l = self.network.link(link);
        let lane = self.network.lane(l.lane_start);
        ((lane.speed_limit * 1000.0) as u64) << 40 | (l.lane_count as u64) << 24 | (link.0 as u64 & 0xFF_FFFF)
    }

    pub fn step(&mut self) {
        let dt = self.cfg.dt;
        let now = self.time;
        let nb = self.neighbors();

        let accels: Vec<f64> = self
            .vehicles
            .iter()
            .enumerate()
            .map(|(i, veh)| {
                let lane = *self.network.lane(veh.lane);
                let driver = veh.driver.capped_to(lane.speed_limit);
                let intended = self.intended_movement(veh);
                let node = self.downstream_node(veh.lane);
                let control = self.network.node(node).control;
                let to_line = (lane.length - veh.position).max(0.05);

                let leader = if let Some(li) = nb.leader_of[i] {
                    let lead = &self.vehicles[li];
                    Some(Obstacle {
                        gap: lead.position - veh.position - driver.vehicle_length,
                        speed: lead.speed,
                    })
                } else if let Some(mid) = intended {
                    let to_lane = self.network.movement(mid).to_lane;
                    nb.lane_front.get(&to_lane.0).map(|&front| {
                        let lead = &self.vehicles[front];
                        Obstacle {
                            gap: (lane.length - veh.position) + lead.position - driver.vehicle_length,
                            speed: lead.speed,
                        }
                    })
                } else {
                    None
                };

                let stop_line = match intended {
                    Some(mid) if !self.network.movement_state(mid, now).is_go() => Some(to_line),
                    _ => None,
                };

                let speed_target = intended.and_then(|mid| {
                    let to = self.network.lane(self.network.movement(mid).to_lane);
                    let target = veh.driver.desired_speed.min(to.speed_limit);
                    (target < driver.desired_speed).then_some(SpeedTarget { speed: target, distance: to_line })
                });

                let stop_sign = (matches!(control, NodeControl::Stop) && veh.stopped_at != Some(node))
                    .then_some(to_line);

                let yield_line = matches!(control, NodeControl::Stop | NodeControl::Yield)
                    .then(|| self.conflicting_priority_traffic(i, veh.lane, node, &nb))
                    .flatten()
                    .map(|()| to_line);

                let merge = self.merge_conflict(veh, lane.length, intended, &nb);

                let ctx = LongContext {
                    driver: &driver,
                    speed: veh.speed,
                    leader,
                    stop_line,
                    speed_target,
                    stop_sign,
                    yield_line,
                    merge,
                };
                constraint::binding_acceleration(&ctx, constraint::DEFAULT)
            })
            .collect();

        let mut survivors = Vec::with_capacity(self.vehicles.len());
        for (mut veh, a) in std::mem::take(&mut self.vehicles).into_iter().zip(accels) {
            integrate(&mut veh, a, dt);
            let lane = *self.network.lane(veh.lane);
            let node = self.network.link(lane.link).to;
            if matches!(self.network.node(node).control, NodeControl::Stop)
                && veh.speed < 0.3
                && (lane.length - veh.position) < veh.driver.vehicle_length + veh.driver.min_gap + 1.0
            {
                veh.stopped_at = Some(node);
            }
            if self.advance_across_nodes(&mut veh, now) {
                survivors.push(veh);
            } else {
                self.exited += 1;
            }
        }
        self.vehicles = survivors;
        self.time += dt;
    }

    /// `Some(())` when a higher-priority vehicle is approaching `node` from a
    /// different link and will arrive within the critical gap — the signal to
    /// give way. `None` means clear to proceed.
    fn conflicting_priority_traffic(&self, i: usize, lane: LaneId, node: NodeId, nb: &Neighbors) -> Option<()> {
        const CRITICAL_GAP: f64 = 4.0;
        let my_link = self.network.lane(lane).link;
        let my_key = self.priority_key(my_link);
        for &j in nb.approaching.get(&node.0)? {
            if j == i {
                continue;
            }
            let o = &self.vehicles[j];
            let o_lane = *self.network.lane(o.lane);
            if o_lane.link == my_link || o.speed < 0.5 {
                continue;
            }
            if self.priority_key(o_lane.link) <= my_key {
                continue;
            }
            if (o_lane.length - o.position) / o.speed.max(0.1) < CRITICAL_GAP {
                return Some(());
            }
        }
        None
    }

    /// The nearest-to-the-merge conflicting vehicle on a converging lane, as an
    /// obstacle to follow. `None` when this vehicle is first to the merge (the
    /// other yields) or there is no merge.
    fn merge_conflict(&self, veh: &NetVehicle, lane_len: f64, intended: Option<MovementId>, nb: &Neighbors) -> Option<Obstacle> {
        let to_lane = self.network.movement(intended?).to_lane;
        let froms = self.merges.get(&to_lane.0)?;
        let my_dist = lane_len - veh.position;
        let mut best: Option<Obstacle> = None;
        for &from in froms {
            if from == veh.lane.0 {
                continue;
            }
            for &j in nb.by_lane.get(&from).into_iter().flatten() {
                let o = &self.vehicles[j];
                if o.speed < 0.5 {
                    continue;
                }
                let o_dist = self.network.lane(o.lane).length - o.position;
                if o_dist < my_dist {
                    let gap = my_dist - o_dist;
                    if best.is_none_or(|b| gap < b.gap) {
                        best = Some(Obstacle { gap, speed: o.speed });
                    }
                }
            }
        }
        best
    }

    fn advance_across_nodes(&self, veh: &mut NetVehicle, now: f64) -> bool {
        for _ in 0..8 {
            let lane = *self.network.lane(veh.lane);
            if veh.position < lane.length {
                return true;
            }
            match self.intended_movement(veh) {
                Some(mid) if self.network.movement_state(mid, now).is_go() => {
                    let to_lane = self.network.movement(mid).to_lane;
                    veh.position -= lane.length;
                    veh.lane = to_lane;
                    veh.stopped_at = None;
                    let to_link = self.network.lane(to_lane).link;
                    if veh.route_idx + 1 < veh.route.len()
                        && veh.route[veh.route_idx + 1] == to_link
                    {
                        veh.route_idx += 1;
                    }
                }
                Some(_) => {
                    veh.position = lane.length;
                    veh.speed = 0.0;
                    return true;
                }
                None => return false,
            }
        }
        true
    }

    pub fn run_ticks(&mut self, ticks: u32) {
        for _ in 0..ticks {
            self.step();
        }
    }
}

struct Neighbors {
    leader_of: Vec<Option<usize>>,
    lane_front: HashMap<u32, usize>,
    by_lane: HashMap<u32, Vec<usize>>,
    approaching: HashMap<u32, Vec<usize>>,
}

fn integrate(v: &mut NetVehicle, accel: f64, dt: f64) {
    if v.speed + accel * dt < 0.0 {
        v.position += -0.5 * v.speed * v.speed / accel;
        v.speed = 0.0;
    } else {
        v.position += v.speed * dt + 0.5 * accel * dt * dt;
        v.speed += accel * dt;
    }
}

#[cfg(test)]
mod tests {
    use super::super::map::*;
    use super::super::network::{LaneId, LinkId};
    use super::*;

    fn cfg() -> SimConfig {
        SimConfig::default_config()
    }

    fn approach_lane(net: &Network) -> LaneId {
        net.lanes_of(LinkId(0)).next().unwrap()
    }

    fn signal_at(offset: f64) -> Network {
        let plan = SignalPlan { green_secs: 15.0, yellow_secs: 3.0, offset };
        OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::signalized(2, 150.0, 0.0, plan),
                NodeSpec::uncontrolled(3, 300.0, 0.0),
                NodeSpec::uncontrolled(4, 150.0, -120.0),
            ],
            links: vec![
                LinkSpec::oneway(1, 2, 1, 20.0),
                LinkSpec::oneway(2, 3, 1, 20.0),
                LinkSpec::oneway(4, 2, 1, 15.0),
            ],
        }
        .build()
    }

    #[test]
    fn vehicle_stops_before_a_red_light() {
        let net = signal_at(18.0); // through movement red across t∈[0,18)
        let lane = approach_lane(&net);
        let length = net.lane(lane).length;
        let mut world = NetWorld::new(net, cfg());
        world.spawn(1, lane, 0.0, 15.0, DriverConfig::car());

        world.run_ticks(85); // 17 s: braked to rest, still within the red window

        let v = world.vehicle(1).expect("still on the approach");
        assert_eq!(v.lane, lane, "must not cross a red light");
        assert!(v.speed < 0.5, "should be stopped, speed={}", v.speed);
        assert!(v.position < length && v.position > length - 8.0,
            "should be halted at the stop line, pos={} len={length}", v.position);
        assert_eq!(world.exited(), 0);
    }

    #[test]
    fn vehicle_proceeds_through_a_green_light() {
        let net = signal_at(0.0); // through movement green from t=0
        let lane = approach_lane(&net);
        let mut world = NetWorld::new(net, cfg());
        world.spawn(1, lane, 0.0, 15.0, DriverConfig::car());

        world.run_ticks(150); // 30 s: cross and reach the sink

        assert_eq!(world.exited(), 1, "vehicle should clear the intersection");
        assert!(world.vehicle(1).is_none());
    }

    #[test]
    fn follower_settles_behind_a_slower_leader_without_colliding() {
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::uncontrolled(2, 6000.0, 0.0),
            ],
            links: vec![LinkSpec::oneway(1, 2, 1, 30.0)],
        }
        .build();
        let lane = LaneId(0);
        let mut world = NetWorld::new(net, cfg());
        let slow = DriverConfig { desired_speed: 10.0, ..DriverConfig::car() };
        world.spawn(1, lane, 200.0, 10.0, slow);
        world.spawn(2, lane, 100.0, 10.0, DriverConfig::car());

        world.run_ticks(1500); // 300 s

        let leader = world.vehicle(1).unwrap();
        let follower = world.vehicle(2).unwrap();
        assert!((follower.speed - 10.0).abs() < 0.5, "follower speed={}", follower.speed);
        let gap = leader.position - follower.position - follower.driver.vehicle_length;
        assert!(gap > 0.0, "no collision, gap={gap}");
    }

    #[test]
    fn a_standing_queue_discharges_on_green() {
        let net = signal_at(18.0); // starts red, turns green at t=18
        let lane = approach_lane(&net);
        let length = net.lane(lane).length;
        let mut world = NetWorld::new(net, cfg());
        for i in 0..6u32 {
            world.spawn(i, lane, length - 8.0 - i as f64 * 7.0, 0.0, DriverConfig::car());
        }

        world.run_ticks(60); // settle into a stopped queue while red
        assert_eq!(world.exited(), 0, "nobody moves on red");

        world.run_ticks(500); // enough green cycles to serve the whole queue
        assert_eq!(world.exited(), 6, "the whole queue should discharge");
    }

    fn diamond() -> Network {
        OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(0, 0.0, 0.0),
                NodeSpec::uncontrolled(1, 100.0, 0.0),
                NodeSpec::uncontrolled(2, 200.0, -10.0),
                NodeSpec::uncontrolled(3, 200.0, 300.0),
                NodeSpec::uncontrolled(4, 300.0, 0.0),
                NodeSpec::uncontrolled(5, 400.0, 0.0),
            ],
            links: vec![
                LinkSpec::oneway(0, 1, 1, 20.0),
                LinkSpec::oneway(1, 2, 1, 20.0),
                LinkSpec::oneway(1, 3, 1, 20.0),
                LinkSpec::oneway(2, 4, 1, 20.0),
                LinkSpec::oneway(3, 4, 1, 20.0),
                LinkSpec::oneway(4, 5, 1, 20.0),
            ],
        }
        .build()
    }

    #[test]
    fn router_prefers_the_faster_path() {
        let net = diamond();
        let route = net.route_links(LinkId(0), LinkId(5)).expect("reachable");
        assert_eq!(route, vec![LinkId(0), LinkId(1), LinkId(3), LinkId(5)]);
    }

    #[test]
    fn router_reroutes_around_a_congested_link() {
        let net = diamond();
        let mut costs: Vec<u64> =
            (0..net.links.len() as u32).map(|i| net.link_travel_time_ms(LinkId(i))).collect();
        assert_eq!(net.route_links(LinkId(0), LinkId(5)).unwrap()[1], LinkId(1));
        costs[1] = 10_000_000; // link (1,2) is now jammed
        let rerouted = net.route_links_with_costs(LinkId(0), LinkId(5), &costs).unwrap();
        assert_eq!(rerouted, vec![LinkId(0), LinkId(2), LinkId(4), LinkId(5)]);
    }

    #[test]
    fn vehicles_follow_their_assigned_routes_to_exit() {
        let net = diamond();
        let mut world = NetWorld::new(net, cfg());
        let short = world.network.route_links(LinkId(0), LinkId(5)).unwrap();
        let long = vec![LinkId(0), LinkId(2), LinkId(4), LinkId(5)];
        assert!(world.spawn_routed(1, short, 12.0, DriverConfig::car()));
        world.run_ticks(50); // clear the shared entrance before the next spawn
        assert!(world.spawn_routed(2, long, 12.0, DriverConfig::car()));

        world.run_ticks(600);

        assert_eq!(world.exited(), 2, "both routed vehicles reach their destination");
    }

    #[test]
    fn brakes_in_advance_for_a_slower_road_ahead() {
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::uncontrolled(2, 200.0, 0.0),
                NodeSpec::uncontrolled(3, 400.0, 0.0),
            ],
            links: vec![
                LinkSpec::oneway(1, 2, 1, 25.0), // fast approach
                LinkSpec::oneway(2, 3, 1, 8.0),  // slow road ahead
            ],
        }
        .build();
        let fast = net.lanes_of(LinkId(0)).next().unwrap();
        let slow = net.lanes_of(LinkId(1)).next().unwrap();
        let mut world = NetWorld::new(net, cfg());
        world.spawn(1, fast, 0.0, 24.0, DriverConfig::car());

        let mut entry_speed = None;
        for _ in 0..300 {
            let before = world.vehicle(1).map(|v| (v.lane, v.speed));
            world.step();
            let after = world.vehicle(1).map(|v| v.lane);
            if let (Some((lane_before, s)), Some(lane_after)) = (before, after) {
                if lane_before == fast && lane_after == slow {
                    entry_speed = Some(s);
                    break;
                }
            }
        }

        let s = entry_speed.expect("vehicle should cross onto the slow road");
        assert!(s < 11.0, "should have slowed toward 8 m/s before crossing, entered at {s}");
    }

    #[test]
    fn a_vehicle_fully_stops_at_a_stop_sign_then_proceeds() {
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec { osm_id: 2, x: 150.0, y: 0.0, control: MapControl::Stop },
                NodeSpec::uncontrolled(3, 300.0, 0.0),
            ],
            links: vec![LinkSpec::oneway(1, 2, 1, 15.0), LinkSpec::oneway(2, 3, 1, 15.0)],
        }
        .build();
        let approach = net.lanes_of(LinkId(0)).next().unwrap();
        let mut world = NetWorld::new(net, cfg());
        world.spawn(1, approach, 0.0, 14.0, DriverConfig::car());

        let mut min_speed_on_approach = f64::MAX;
        for _ in 0..250 {
            world.step();
            if let Some(v) = world.vehicle(1) {
                if v.lane == approach {
                    min_speed_on_approach = min_speed_on_approach.min(v.speed);
                }
            }
        }
        assert!(min_speed_on_approach < 0.5, "must come to a full stop, min {min_speed_on_approach}");
        assert_eq!(world.exited(), 1, "and then continue through");
    }

    #[test]
    fn minor_road_yields_to_the_major_road_then_goes() {
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -200.0, 0.0),
                NodeSpec { osm_id: 2, x: 0.0, y: 0.0, control: MapControl::Yield },
                NodeSpec::uncontrolled(3, 200.0, 0.0),
                NodeSpec::uncontrolled(4, 0.0, -200.0),
                NodeSpec::uncontrolled(5, 0.0, 200.0),
            ],
            links: vec![
                LinkSpec::oneway(1, 2, 1, 25.0), // major approach
                LinkSpec::oneway(2, 3, 1, 25.0), // major exit
                LinkSpec::oneway(4, 2, 1, 10.0), // minor approach
                LinkSpec::oneway(2, 5, 1, 10.0), // minor exit
            ],
        }
        .build();
        let major_app = net.lanes_of(LinkId(0)).next().unwrap();
        let minor_app = net.lanes_of(LinkId(2)).next().unwrap();
        let mut world = NetWorld::new(net, cfg());
        world.spawn(10, major_app, 100.0, 20.0, DriverConfig::car()); // major, farther
        world.spawn(20, minor_app, 180.0, 9.0, DriverConfig::car()); // minor, closer to line

        let (mut major_cross, mut minor_cross) = (None, None);
        for t in 0..400 {
            let mj_before = world.vehicle(10).map(|v| v.lane);
            let mn_before = world.vehicle(20).map(|v| v.lane);
            world.step();
            let mj_after = world.vehicle(10).map(|v| v.lane);
            let mn_after = world.vehicle(20).map(|v| v.lane);
            if major_cross.is_none() && mj_before == Some(major_app) && mj_after != Some(major_app) {
                major_cross = Some(t);
            }
            if minor_cross.is_none() && mn_before == Some(minor_app) && mn_after != Some(minor_app) {
                minor_cross = Some(t);
            }
        }
        let (mj, mn) = (major_cross.expect("major crosses"), minor_cross.expect("minor crosses"));
        assert!(mn > mj, "minor arrived first but must yield: minor@{mn} major@{mj}");
    }

    #[test]
    fn two_lanes_zipper_merge_without_colliding() {
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -100.0, 50.0),
                NodeSpec::uncontrolled(2, -100.0, -50.0),
                NodeSpec::uncontrolled(3, 0.0, 0.0), // merge point
                NodeSpec::uncontrolled(4, 150.0, 0.0),
            ],
            links: vec![
                LinkSpec::oneway(1, 3, 1, 15.0),
                LinkSpec::oneway(2, 3, 1, 15.0),
                LinkSpec::oneway(3, 4, 1, 15.0),
            ],
        }
        .build();
        let branch_a = net.lanes_of(LinkId(0)).next().unwrap();
        let branch_b = net.lanes_of(LinkId(1)).next().unwrap();
        let mut world = NetWorld::new(net, cfg());
        world.spawn(1, branch_a, 10.0, 10.0, DriverConfig::car()); // slightly ahead
        world.spawn(2, branch_b, 0.0, 10.0, DriverConfig::car());

        let mut min_gap = f64::MAX;
        for _ in 0..300 {
            world.step();
            let mut by_lane: std::collections::HashMap<u32, Vec<f64>> = std::collections::HashMap::new();
            for v in world.vehicles() {
                by_lane.entry(v.lane.0).or_default().push(v.position);
            }
            for positions in by_lane.values_mut() {
                positions.sort_by(f64::total_cmp);
                for w in positions.windows(2) {
                    min_gap = min_gap.min(w[1] - w[0]);
                }
            }
        }
        assert!(min_gap > 2.0, "vehicles overlapped at the merge: min gap {min_gap}");
        assert_eq!(world.exited(), 2, "both should clear the merge");
    }

    #[test]
    fn lanes_are_independent() {
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::uncontrolled(2, 6000.0, 0.0),
            ],
            links: vec![LinkSpec::oneway(1, 2, 2, 30.0)],
        }
        .build();
        let lanes: Vec<LaneId> = net.lanes_of(LinkId(0)).collect();
        let mut world = NetWorld::new(net, cfg());
        let slow = DriverConfig { desired_speed: 8.0, ..DriverConfig::car() };
        world.spawn(1, lanes[0], 200.0, 8.0, slow);
        world.spawn(2, lanes[0], 100.0, 8.0, DriverConfig::car());
        world.spawn(3, lanes[1], 100.0, 8.0, DriverConfig::car());

        world.run_ticks(1000);

        let blocked = world.vehicle(2).unwrap();
        let free = world.vehicle(3).unwrap();
        assert!(blocked.speed < 9.0, "lane-0 follower is held back, speed={}", blocked.speed);
        assert!(free.speed > 20.0, "lane-1 vehicle is unaffected, speed={}", free.speed);
    }
}
