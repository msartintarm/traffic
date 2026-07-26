//! Vehicles driving a graph [`Network`]: IDM car-following within a lane, a red
//! signal (or unsatisfied movement) acting as a stationary virtual leader at
//! the stop line, and lane hand-off across a node when the movement is served.
//! Accelerations read committed pre-step state and apply in a second pass, the
//! same double-buffered shape the GPU mass layer will use.

use std::collections::HashMap;

use super::config::{DriverConfig, SimConfig};
use super::constraint::{self, LongContext, Obstacle, SpeedTarget};
use super::idm;
use super::mobil::{self, MobilParams};
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
    /// Recent `(position, speed)` samples (newest last) so a follower can react
    /// to the leader's state as of its reaction time ago.
    history: Vec<(f64, f64)>,
}

/// How many `(position, speed)` samples to retain — enough for the largest
/// plausible reaction delay at the fixed timestep.
const HISTORY_LEN: usize = 8;

impl NetVehicle {
    /// This vehicle's `(position, speed)` `ticks` steps ago (clamped to the
    /// oldest sample available).
    fn delayed(&self, ticks: usize) -> (f64, f64) {
        let n = self.history.len();
        self.history[n - 1 - ticks.min(n - 1)]
    }

    fn record(&mut self) {
        self.history.push((self.position, self.speed));
        if self.history.len() > HISTORY_LEN {
            self.history.remove(0);
        }
    }
}

pub struct NetWorld {
    pub network: Network,
    cfg: SimConfig,
    vehicles: Vec<NetVehicle>,
    time: f64,
    tick: u64,
    exited: u32,
    crashed: u32,
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
        Self { network, cfg, vehicles: Vec::new(), time: 0.0, tick: 0, exited: 0, crashed: 0, merges }
    }

    pub fn spawn(&mut self, id: u32, lane: LaneId, position: f64, speed: f64, driver: DriverConfig) {
        self.vehicles.push(NetVehicle {
            id, lane, position, speed, driver, route: Vec::new(), route_idx: 0,
            stopped_at: None, history: vec![(position, speed)],
        });
    }

    /// Spawn at the start of a precomputed link route; the vehicle takes the
    /// route-consistent movement at each intersection and exits on the last link.
    /// Returns `false` (spawn refused) if the route is empty or the entrance is
    /// still occupied, so demand can't stack vehicles on top of each other.
    pub fn spawn_routed(&mut self, id: u32, route: Vec<LinkId>, speed: f64, driver: DriverConfig) -> bool {
        let Some(&first) = route.first() else { return false };
        let Some(lane) = self.network.lanes_of(first).next() else { return false };
        if !self.entrance_clear(lane, driver.min_gap) {
            return false;
        }
        self.vehicles.push(NetVehicle {
            id, lane, position: 0.0, speed, driver, route, route_idx: 0,
            stopped_at: None, history: vec![(0.0, speed)],
        });
        true
    }

    /// Whether a vehicle can be placed at the start of `lane` without overlapping
    /// one already there — measured against each occupant's *rear* (position minus
    /// its own length), so long vehicles are accounted for.
    pub fn entrance_clear(&self, lane: LaneId, min_gap: f64) -> bool {
        self.vehicles
            .iter()
            .filter(|v| v.lane == lane)
            .all(|v| v.position - v.driver.vehicle_length > min_gap)
    }

    pub fn time(&self) -> f64 {
        self.time
    }

    pub fn exited(&self) -> u32 {
        self.exited
    }

    /// Vehicles removed from the road after a collision.
    pub fn crashed(&self) -> u32 {
        self.crashed
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

    /// MOBIL lane changes: evaluated on committed positions, applied before the
    /// longitudinal update. Discretionary (overtake a slow leader into a freer
    /// lane) and mandatory (move to a lane that serves the route's next link).
    fn lane_changes(&mut self) {
        let mut by_lane: HashMap<u32, Vec<usize>> = HashMap::new();
        for (i, v) in self.vehicles.iter().enumerate() {
            by_lane.entry(v.lane.0).or_default().push(i);
        }
        for m in by_lane.values_mut() {
            m.sort_by(|&a, &b| self.vehicles[a].position.total_cmp(&self.vehicles[b].position));
        }
        let mut changes: Vec<(usize, LaneId)> = Vec::new();
        for i in 0..self.vehicles.len() {
            if let Some(t) = self.best_lane_change(i, &by_lane) {
                changes.push((i, t));
            }
        }
        for (i, target) in changes {
            let (pos, len) = (self.vehicles[i].position, self.vehicles[i].driver.vehicle_length);
            if self.lane_slot_clear(target, pos, len, i) {
                self.vehicles[i].lane = target;
            }
        }
    }

    fn best_lane_change(&self, i: usize, by_lane: &HashMap<u32, Vec<usize>>) -> Option<LaneId> {
        let v = &self.vehicles[i];
        let lane = *self.network.lane(v.lane);
        let link = *self.network.link(lane.link);
        let idx = lane.index_in_link as i64;
        let cur_leader = self.nearest_ahead(v.lane, v.position, by_lane, i);
        let a_self_cur = idm_follow(v, lane.speed_limit, v.position, v.speed, cur_leader.map(|j| &self.vehicles[j]));

        let mut best: Option<(f64, LaneId)> = None;
        for delta in [-1i64, 1] {
            let ti = idx + delta;
            if ti < 0 || ti >= link.lane_count as i64 {
                continue;
            }
            let target = LaneId(link.lane_start.0 + ti as u32);
            let limit = self.network.lane(target).speed_limit;

            let a_self_new = idm_follow(
                v,
                limit,
                v.position,
                v.speed,
                self.nearest_ahead(target, v.position, by_lane, i).map(|j| &self.vehicles[j]),
            );

            let (a_nf_cur, a_nf_new) = match self.nearest_behind(target, v.position, by_lane, i) {
                Some(fj) => {
                    let f = &self.vehicles[fj];
                    let fl = self.nearest_ahead(target, f.position, by_lane, i).map(|j| &self.vehicles[j]);
                    (
                        idm_follow(f, limit, f.position, f.speed, fl),
                        idm_follow(f, limit, f.position, f.speed, Some(v)),
                    )
                }
                None => (0.0, 0.0),
            };

            let mandatory = self.mandatory_change(v, v.lane, target);
            let params = MobilParams::new(v.driver.politeness);
            if mobil::should_change(&params, a_self_cur, a_self_new, a_nf_cur, a_nf_new, mandatory) {
                let gain = (a_self_new - a_self_cur) + if mandatory { 100.0 } else { 0.0 };
                if best.is_none_or(|(g, _)| gain > g) {
                    best = Some((gain, target));
                }
            }
        }
        best.map(|(_, t)| t)
    }

    fn nearest_ahead(&self, lane: LaneId, pos: f64, by_lane: &HashMap<u32, Vec<usize>>, exclude: usize) -> Option<usize> {
        by_lane
            .get(&lane.0)?
            .iter()
            .copied()
            .filter(|&j| j != exclude && self.vehicles[j].position > pos)
            .min_by(|&a, &b| self.vehicles[a].position.total_cmp(&self.vehicles[b].position))
    }

    fn nearest_behind(&self, lane: LaneId, pos: f64, by_lane: &HashMap<u32, Vec<usize>>, exclude: usize) -> Option<usize> {
        by_lane
            .get(&lane.0)?
            .iter()
            .copied()
            .filter(|&j| j != exclude && self.vehicles[j].position < pos)
            .max_by(|&a, &b| self.vehicles[a].position.total_cmp(&self.vehicles[b].position))
    }

    /// Whether the current lane can't serve the route's next link but `target` can.
    fn mandatory_change(&self, veh: &NetVehicle, current: LaneId, target: LaneId) -> bool {
        matches!(
            (self.lane_serves_route(veh, current), self.lane_serves_route(veh, target)),
            (Some(false), Some(true))
        )
    }

    fn lane_serves_route(&self, veh: &NetVehicle, lane: LaneId) -> Option<bool> {
        if veh.route.is_empty() || veh.route_idx + 1 >= veh.route.len() {
            return None;
        }
        let next = veh.route[veh.route_idx + 1];
        Some(self.network.movements_of(lane).iter().any(|m| self.network.lane(m.to_lane).link == next))
    }

    fn lane_slot_clear(&self, target: LaneId, pos: f64, len: f64, exclude: usize) -> bool {
        self.vehicles
            .iter()
            .enumerate()
            .filter(|(j, o)| *j != exclude && o.lane == target)
            .all(|(_, o)| {
                if o.position > pos {
                    o.position - o.driver.vehicle_length - pos > 0.5
                } else {
                    pos - len - o.position > 0.5
                }
            })
    }

    pub fn step(&mut self) {
        let dt = self.cfg.dt;
        let now = self.time;
        self.lane_changes();
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

                // Reaction delay: perceive the leader's pose/speed as of the
                // driver's reaction time ago. The perceived gap is capped by the
                // true current gap (`min`), so IDM can never *under*-brake — it's
                // collision-free in-lane. The delay instead shows up as realistic
                // start-up lag: a follower is slow to notice the leader pulling
                // away, so queues discharge with lost time.
                let leader = if let Some(li) = nb.leader_of[i] {
                    let lead = &self.vehicles[li];
                    let delay = (driver.reaction_time / dt).round() as usize;
                    let (my_p, _) = veh.delayed(delay);
                    let (lead_p, lead_v) = lead.delayed(delay);
                    let delayed_gap = lead_p - my_p - lead.driver.vehicle_length;
                    let current_gap = lead.position - veh.position - lead.driver.vehicle_length;
                    Some(Obstacle { gap: delayed_gap.min(current_gap), speed: lead_v })
                } else if let Some(mid) = intended {
                    let to_lane = self.network.movement(mid).to_lane;
                    nb.lane_front.get(&to_lane.0).map(|&front| {
                        let lead = &self.vehicles[front];
                        Obstacle {
                            gap: (lane.length - veh.position) + lead.position - lead.driver.vehicle_length,
                            speed: lead.speed,
                        }
                    })
                } else {
                    None
                };

                // Don't-block-the-box: about to cross, but the downstream lane's
                // entrance is occupied — hold at the line rather than land on top
                // of a stopped vehicle (the main source of intersection crashes).
                let downstream_blocked = intended.is_some_and(|mid| {
                    let to_lane = self.network.movement(mid).to_lane;
                    nb.lane_front
                        .get(&to_lane.0)
                        .is_some_and(|&f| self.vehicles[f].position < driver.vehicle_length + driver.min_gap)
                });
                let stop_line = match intended {
                    Some(mid) if !self.network.movement_state(mid, now).is_go() => Some(to_line),
                    _ if downstream_blocked => Some(to_line),
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
                let binding = constraint::binding_acceleration(&ctx, constraint::DEFAULT);
                binding + constraint::accel_noise(driver.accel_noise, self.cfg.seed, veh.id, self.tick)
            })
            .collect();

        // Integrate every vehicle in place (no crossing yet).
        let mut moved: Vec<NetVehicle> = std::mem::take(&mut self.vehicles)
            .into_iter()
            .zip(accels)
            .map(|(mut veh, a)| {
                integrate(&mut veh, a, dt);
                let lane = *self.network.lane(veh.lane);
                let node = self.network.link(lane.link).to;
                if matches!(self.network.node(node).control, NodeControl::Stop)
                    && veh.speed < 0.3
                    && (lane.length - veh.position) < veh.driver.vehicle_length + veh.driver.min_gap + 1.0
                {
                    veh.stopped_at = Some(node);
                }
                veh
            })
            .collect();

        // Occupancy-aware crossing: process in id order, tracking each lane's
        // frontmost (entrance-nearest) position, so a vehicle never crosses a
        // node onto a spot another vehicle already holds. This makes
        // intersection/merge overlaps impossible rather than after-the-fact.
        // `front[lane]` = the nearest occupied point to the lane entrance (the
        // rear = position − length of the closest-in vehicle), so a crosser never
        // lands on top of a long vehicle already there.
        moved.sort_by_key(|v| v.id);
        let mut front: HashMap<u32, f64> = HashMap::new();
        for v in &moved {
            if v.position < self.network.lane(v.lane).length {
                let e = front.entry(v.lane.0).or_insert(f64::MAX);
                *e = e.min(v.position - v.driver.vehicle_length);
            }
        }

        let mut survivors = Vec::with_capacity(moved.len());
        for mut veh in moved {
            let alive = self.advance_across_nodes(&mut veh, now, &mut front);
            if alive {
                veh.record();
                let e = front.entry(veh.lane.0).or_insert(f64::MAX);
                *e = e.min(veh.position - veh.driver.vehicle_length);
                survivors.push(veh);
            } else {
                self.exited += 1;
            }
        }
        self.vehicles = survivors;
        self.remove_crashes();
        self.time += dt;
        self.tick += 1;
    }

    /// Detect rear-end overlaps within a lane (a follower's front past the
    /// leader's rear) and take the crashed vehicles off the road, counting them.
    /// Reaction delay makes this physically possible; behaviour should minimize
    /// it over time. Crashes are resolved deterministically (scan by position).
    fn remove_crashes(&mut self) {
        const OVERLAP_TOL: f64 = 0.5;
        let mut by_lane: HashMap<u32, Vec<usize>> = HashMap::new();
        for (i, v) in self.vehicles.iter().enumerate() {
            by_lane.entry(v.lane.0).or_default().push(i);
        }
        let mut crashed = vec![false; self.vehicles.len()];
        for members in by_lane.values_mut() {
            members.sort_by(|&a, &b| self.vehicles[a].position.total_cmp(&self.vehicles[b].position));
            for w in members.windows(2) {
                let (rear, front) = (&self.vehicles[w[0]], &self.vehicles[w[1]]);
                let gap = front.position - rear.position - front.driver.vehicle_length;
                if gap < -OVERLAP_TOL {
                    crashed[w[0]] = true;
                    crashed[w[1]] = true;
                }
            }
        }
        if crashed.iter().any(|&c| c) {
            let mut i = 0;
            self.vehicles.retain(|_| {
                let keep = !crashed[i];
                i += 1;
                if !keep {
                    self.crashed += 1;
                }
                keep
            });
        }
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
                    let gap = my_dist - o_dist - o.driver.vehicle_length;
                    if best.is_none_or(|b| gap < b.gap) {
                        best = Some(Obstacle { gap, speed: o.speed });
                    }
                }
            }
        }
        best
    }

    fn advance_across_nodes(&self, veh: &mut NetVehicle, now: f64, front: &mut HashMap<u32, f64>) -> bool {
        for _ in 0..8 {
            let lane = *self.network.lane(veh.lane);
            if veh.position < lane.length {
                return true;
            }
            match self.intended_movement(veh) {
                Some(mid) if self.network.movement_state(mid, now).is_go() => {
                    let to_lane = self.network.movement(mid).to_lane;
                    // `front` holds the nearest occupied rear; the crosser lands at
                    // ~0, so require that rear to be at least min_gap ahead.
                    if front.get(&to_lane.0).is_some_and(|&rear| rear < veh.driver.min_gap) {
                        veh.position = lane.length; // entrance occupied — hold at the line
                        veh.speed = 0.0;
                        return true;
                    }
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

/// IDM acceleration for a vehicle placed at `pos`/`speed` on a lane with the
/// given speed limit, following `leader` (or free road if none). Used to score
/// hypothetical lane placements for MOBIL.
fn idm_follow(follower: &NetVehicle, lane_speed_limit: f64, pos: f64, speed: f64, leader: Option<&NetVehicle>) -> f64 {
    let d = follower.driver.capped_to(lane_speed_limit);
    match leader {
        Some(l) => idm::acceleration(&d, speed, speed - l.speed, (l.position - pos - l.driver.vehicle_length).max(0.05)),
        None => idm::free_acceleration(&d, speed),
    }
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

    fn straight_link(length: f64) -> Network {
        OsmMap {
            nodes: vec![NodeSpec::uncontrolled(1, 0.0, 0.0), NodeSpec::uncontrolled(2, length, 0.0)],
            links: vec![LinkSpec::oneway(1, 2, 1, 40.0)],
        }
        .build()
    }

    #[test]
    fn reaction_delay_causes_start_up_lag() {
        // A follower behind a leader that accelerates away travels less over the
        // same window when it has a reaction delay (it's slow to notice the gap
        // opening) — realistic start-up lost time — and never crashes.
        let distance_travelled = |reaction: f64| {
            let mut w = NetWorld::new(straight_link(6000.0), cfg());
            let base = DriverConfig { accel_noise: 0.0, reaction_time: reaction, desired_speed: 25.0, ..DriverConfig::car() };
            w.spawn(1, LaneId(0), 20.0, 0.0, base); // leader just ahead, from rest
            w.spawn(2, LaneId(0), 0.0, 0.0, base); // follower from rest
            for _ in 0..200 {
                w.step();
            }
            (w.vehicle(2).map(|v| v.position).unwrap_or(0.0), w.vehicle(1).is_some() && w.vehicle(2).is_some())
        };
        let (delayed, ok_d) = distance_travelled(0.8);
        let (instant, ok_i) = distance_travelled(0.0);
        assert!(ok_d && ok_i, "neither should crash");
        assert!(delayed > 0.0 && instant > 0.0);
        assert!(delayed < instant, "reaction delay should lag start-up: {delayed} vs {instant}");
    }

    #[test]
    fn acceleration_noise_keeps_speed_just_below_desired() {
        let steady = |noise: f64| {
            let mut w = NetWorld::new(straight_link(20000.0), cfg());
            let d = DriverConfig { desired_speed: 20.0, accel_noise: noise, reaction_time: 0.0, ..DriverConfig::car() };
            w.spawn(1, LaneId(0), 0.0, 20.0, d);
            let (mut sum, mut n) = (0.0, 0);
            for t in 0..1000 {
                w.step();
                if t >= 800 {
                    if let Some(v) = w.vehicle(1) {
                        sum += v.speed;
                        n += 1;
                    }
                }
            }
            sum / n as f64
        };
        let without = steady(0.0);
        let with = steady(0.4);
        assert!((without - 20.0).abs() < 0.1, "noiseless reaches desired: {without}");
        assert!(with < without - 0.1 && with > 15.0, "noise slows a little: {with}");
    }

    #[test]
    fn overlapping_vehicles_crash_and_leave_the_road() {
        let mut w = NetWorld::new(straight_link(1000.0), cfg());
        w.spawn(1, LaneId(0), 100.0, 5.0, DriverConfig::car());
        w.spawn(2, LaneId(0), 99.0, 5.0, DriverConfig::car()); // deep overlap into the leader
        w.step();
        assert_eq!(w.crashed(), 2);
        assert_eq!(w.vehicles().len(), 0);
    }

    #[test]
    fn faster_vehicle_changes_lanes_to_overtake() {
        let net = OsmMap {
            nodes: vec![NodeSpec::uncontrolled(1, 0.0, 0.0), NodeSpec::uncontrolled(2, 6000.0, 0.0)],
            links: vec![LinkSpec::oneway(1, 2, 2, 30.0)],
        }
        .build();
        let lanes: Vec<LaneId> = net.lanes_of(LinkId(0)).collect();
        let mut w = NetWorld::new(net, cfg());
        let slow = DriverConfig { desired_speed: 8.0, accel_noise: 0.0, ..DriverConfig::car() };
        let fast = DriverConfig { desired_speed: 30.0, accel_noise: 0.0, ..DriverConfig::car() };
        w.spawn(1, lanes[0], 300.0, 8.0, slow); // slow leader
        w.spawn(2, lanes[0], 100.0, 20.0, fast); // fast follower, same lane

        let mut used_other_lane = false;
        for _ in 0..500 {
            w.step();
            if w.vehicle(2).is_some_and(|v| v.lane == lanes[1]) {
                used_other_lane = true;
            }
        }
        let f = w.vehicle(2).unwrap();
        let s = w.vehicle(1).unwrap();
        assert!(used_other_lane, "fast vehicle should use the adjacent lane");
        assert!(f.position > s.position, "and overtake the slow one: {} vs {}", f.position, s.position);
    }

    #[test]
    fn mixed_class_traffic_does_not_crash_under_sustained_demand() {
        // Regression for the length-convention bug: a follower must reserve the
        // *leader's* length, and spawns/crossings must respect a long vehicle's
        // rear — otherwise cars overlap trucks/buses. Runs a busy grid with a
        // car/truck/bus mix and asserts nobody crashes.
        use super::super::demand::{DemandGenerator, OdPair};
        let net = super::super::map::millbrae_sample();
        let mut w = NetWorld::new(net, cfg());
        let n = w.network.links.len();
        let mut pairs = Vec::new();
        for o in 0..n {
            for d in 0..n {
                if o != d {
                    if let Some(r) = w.network.route_links(LinkId(o as u32), LinkId(d as u32)) {
                        if r.len() >= 3 {
                            pairs.push(OdPair { origin: LinkId(o as u32), dest: LinkId(d as u32), rate_per_sec: 0.3 });
                        }
                    }
                }
            }
        }
        let mut demand = DemandGenerator::new(&w, &pairs, 1);
        for _ in 0..1500 {
            demand.step(&mut w, 0.2);
            w.step();
        }
        assert!(demand.spawned() > 100, "scenario should be busy: {}", demand.spawned());
        assert_eq!(w.crashed(), 0, "mixed-class traffic should not crash");
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
        // A slow vehicle in lane 0 must not slow a vehicle in lane 1. (No lane-0
        // follower here — that one would rightly change lanes; see the overtake
        // test — this isolates cross-lane car-following independence.)
        let slow = DriverConfig { desired_speed: 8.0, accel_noise: 0.0, ..DriverConfig::car() };
        world.spawn(1, lanes[0], 200.0, 8.0, slow);
        world.spawn(3, lanes[1], 100.0, 8.0, DriverConfig { accel_noise: 0.0, ..DriverConfig::car() });

        world.run_ticks(1000);

        let slow_v = world.vehicle(1).unwrap();
        let free = world.vehicle(3).unwrap();
        assert!(slow_v.speed < 9.0, "lane-0 vehicle stays slow, speed={}", slow_v.speed);
        assert!(free.speed > 20.0, "lane-1 vehicle is unaffected, speed={}", free.speed);
    }
}
