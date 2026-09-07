//! Junctions as index-addressed views over the flat network arrays. A
//! [`Junctions`] table is built once and gives O(degree) access to each node's
//! movements and conflict points — the per-intersection grouping the runtime's
//! in-box conflict avoidance and permissive-left checks read each tick.
//!
//! It stays data-oriented: CSR (`offsets` + concatenated ids) over the existing
//! `Vec`s, no per-object ownership, so the GPU-friendly SoA layout is preserved.

use super::network::{LaneId, MovementId, Network, NodeControl, NodeId, ProgramId};
use super::signal::{SignalProgram, SignalState, DEFAULT_ALL_RED};

/// Actuation timing: green is held at least `MIN_GREEN`, extended while vehicles
/// keep arriving (gap-out) up to `MAX_GREEN`, and only terminated when a
/// conflicting approach is waiting. `DETECT` is how far back a stop-line detector
/// senses demand.
pub const DETECT: f64 = 35.0;
/// Typical actuated controllers hold 8–15 s minimum green; 6 s cycled phases
/// faster than any real intersection.
const MIN_GREEN: f64 = 8.0;
const MAX_GREEN: f64 = 45.0;

fn all_red_of(program: &SignalProgram, phase: usize) -> f64 {
    let ar = program.phases[phase].all_red_secs;
    if ar > 0.0 {
        ar
    } else {
        DEFAULT_ALL_RED
    }
}

/// Reusable flat set of demanded lane ids: unhashed insert/contains over a
/// lane-indexed bit array, cleared via the touched list so a steady-state tick
/// allocates nothing. Replaces a per-tick `HashSet` whose O(cars) hashed
/// inserts were a flat tax at county fleet sizes.
#[derive(Default)]
pub struct LaneSet {
    bits: Vec<bool>,
    touched: Vec<u32>,
}

impl LaneSet {
    /// Empty the set and (idempotently) size the bit array for lane ids `0..cap`.
    pub fn reset(&mut self, cap: usize) {
        for &l in &self.touched {
            self.bits[l as usize] = false;
        }
        self.touched.clear();
        if self.bits.len() < cap {
            self.bits.resize(cap, false);
        }
    }

    #[inline]
    pub fn insert(&mut self, lane: u32) {
        let b = &mut self.bits[lane as usize];
        if !*b {
            *b = true;
            self.touched.push(lane);
        }
    }

    #[inline]
    pub fn contains(&self, lane: u32) -> bool {
        self.bits.get(lane as usize).copied().unwrap_or(false)
    }

    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.touched.iter().copied()
    }

    /// A fresh set over lane ids `0..cap` containing `lanes`.
    pub fn of(cap: usize, lanes: impl IntoIterator<Item = u32>) -> Self {
        let mut s = Self::default();
        s.reset(cap);
        for l in lanes {
            s.insert(l);
        }
        s
    }
}

/// Live state of one actuated signal program: which phase is running, how long
/// it has, and whether it's in the phase's yellow or all-red clearance.
#[derive(Clone, Copy, Debug)]
struct SignalRuntime {
    phase: usize,
    elapsed: f64,
    yellow: bool,
    all_red: f64,
}

impl SignalRuntime {
    fn state_of(&self, bit: u8, program: &SignalProgram) -> SignalState {
        if program.phases.is_empty() || self.all_red > 0.0 {
            return SignalState::Red;
        }
        let served = program.phases[self.phase].green_mask & (1u64 << bit) != 0;
        if !served {
            SignalState::Red
        } else if self.yellow {
            SignalState::Yellow
        } else {
            SignalState::Green
        }
    }
}

/// The actuated signal system: per-program runtime plus the approach links each
/// signal group serves (for demand detection). Owns all signal timing so the
/// vehicle world only has to supply which links currently have waiting demand.
pub struct SignalController {
    signals: Vec<SignalRuntime>,
    /// Reverse of `approaches` in CSR form (`lane_prog_off[lane]..` slices
    /// `lane_prog`): each feeder lane → the programs it can call, so `advance`
    /// builds the active set from the (small) demand set in O(demand) and skips
    /// resting programs instead of scanning all of them. Lane-indexed, so the
    /// hot lookup is two array reads, no hashing.
    lane_prog_off: Vec<u32>,
    lane_prog: Vec<u32>,
    /// Scratch for `advance`'s active-program set, reused across ticks.
    active: Vec<bool>,
    /// Approach *lanes* per program, per group bit — the from-lanes of each
    /// group's movements. Lane-grained like a real stop-line detector, so a
    /// through queue never calls the adjacent bay's protected-left phase.
    approaches: Vec<Vec<Vec<LaneId>>>,
    time: f64,
}

impl SignalController {
    pub fn build(net: &Network) -> Self {
        // A coordinated program boots on its *scheduled* phase (offset-aware), so
        // corridor progression is aligned from the first cycle instead of every
        // signal starting at phase 0 in unison.
        let signals = net
            .programs
            .iter()
            .map(|p| {
                let mut rt = SignalRuntime { phase: 0, elapsed: 0.0, yellow: false, all_red: 0.0 };
                if p.coordinated && p.cycle_length() > 0.0 {
                    let mut t = p.offset.rem_euclid(p.cycle_length());
                    for (i, ph) in p.phases.iter().enumerate() {
                        if t < ph.length() {
                            rt.phase = i;
                            break;
                        }
                        t -= ph.length();
                    }
                }
                rt
            })
            .collect();
        let mut approaches: Vec<Vec<Vec<LaneId>>> = vec![Vec::new(); net.programs.len()];
        for g in &net.groups {
            let bits = &mut approaches[g.program.idx()];
            if bits.len() <= g.bit as usize {
                bits.resize(g.bit as usize + 1, Vec::new());
            }
        }
        for mv in &net.movements {
            if let Some(gid) = mv.signal_group {
                let g = net.groups[gid.idx()];
                let feeders = &mut approaches[g.program.idx()][g.bit as usize];
                if !feeders.contains(&mv.from_lane) {
                    feeders.push(mv.from_lane);
                }
            }
        }
        let mut lane_programs: std::collections::HashMap<u32, Vec<usize>> = std::collections::HashMap::new();
        for (pid, bits) in approaches.iter().enumerate() {
            for lanes in bits {
                for l in lanes {
                    let e = lane_programs.entry(l.0).or_default();
                    if !e.contains(&pid) {
                        e.push(pid);
                    }
                }
            }
        }
        let mut lane_prog_off = vec![0u32; net.lanes.len() + 1];
        for (&l, pids) in &lane_programs {
            lane_prog_off[l as usize + 1] = pids.len() as u32;
        }
        for i in 0..net.lanes.len() {
            lane_prog_off[i + 1] += lane_prog_off[i];
        }
        let mut lane_prog = vec![0u32; lane_prog_off[net.lanes.len()] as usize];
        for (&l, pids) in &lane_programs {
            let start = lane_prog_off[l as usize] as usize;
            for (k, &pid) in pids.iter().enumerate() {
                lane_prog[start + k] = pid as u32;
            }
        }
        let active = vec![false; net.programs.len()];
        Self { signals, approaches, time: 0.0, lane_prog_off, lane_prog, active }
    }

    fn group_state(&self, net: &Network, program: ProgramId, bit: u8) -> SignalState {
        let prog = &net.programs[program.idx()];
        self.signals[program.idx()].state_of(bit, prog)
    }

    /// Signal state of a movement (`Green` if it carries no signal group).
    pub fn movement_state(&self, net: &Network, mid: MovementId) -> SignalState {
        match net.movement(mid).signal_group {
            None => SignalState::Green,
            Some(g) => {
                let group = net.groups[g.idx()];
                self.group_state(net, group.program, group.bit)
            }
        }
    }

    /// Colour of one signal group — O(1), so a viewport-culled render pass can
    /// query only the visible heads instead of building the all-groups vector.
    pub fn state_of_group(&self, net: &Network, gid: usize) -> SignalState {
        let g = net.groups[gid];
        self.group_state(net, g.program, g.bit)
    }

    /// Colour of every signal group, indexed by group id — for rendering.
    pub fn states(&self, net: &Network) -> Vec<SignalState> {
        net.groups
            .iter()
            .map(|g| self.group_state(net, g.program, g.bit))
            .collect()
    }

    pub fn green_elapsed(&self, net: &Network, mid: MovementId) -> f64 {
        let Some(gid) = net.movement(mid).signal_group else { return f64::INFINITY };
        let group = net.groups[gid.idx()];
        let program = &net.programs[group.program.idx()];
        let mask = 1u64 << group.bit;
        let rt = self.signals[group.program.idx()];
        let served = program.phases.get(rt.phase).is_some_and(|ph| ph.green_mask & mask != 0);
        if served && !rt.yellow && rt.all_red <= 0.0 {
            rt.elapsed
        } else {
            0.0
        }
    }

    /// Advance actuated signals: hold green while its approaches keep demand,
    /// terminate on max-green or a gap-out with a conflicting approach waiting.
    /// `demand` is the set of *lane* ids with a vehicle within [`DETECT`] of a line.
    /// `forced` preempts a program to a specific phase (rail preemption): the
    /// current phase terminates after a short grace, clearance runs as normal,
    /// and the controller jumps to — and holds — the forced phase.
    pub fn advance(
        &mut self,
        net: &Network,
        demand: &LaneSet,
        dt: f64,
        forced: &std::collections::HashMap<usize, usize>,
    ) {
        /// Preemption grace before terminating the current green (real
        /// controllers abbreviate, never instantly kill, a conflicting phase).
        const PREEMPT_GRACE: f64 = 3.0;
        self.time += dt;
        // Skip resting programs cheaply: a program not called by any demanded
        // lane and not mid-transition/preempted/coordinated only holds green, so
        // its expensive per-approach demand checks are pointless — advance its
        // timer and move on. Cuts the per-tick signal work to O(demanded
        // programs) + a cheap O(programs) skip scan.
        let mut active = std::mem::take(&mut self.active);
        active.iter_mut().for_each(|b| *b = false);
        for lane in demand.iter() {
            let (s, e) = (self.lane_prog_off[lane as usize], self.lane_prog_off[lane as usize + 1]);
            for &pid in &self.lane_prog[s as usize..e as usize] {
                active[pid as usize] = true;
            }
        }
        for pid in 0..self.signals.len() {
            let rt0 = self.signals[pid];
            if !active[pid]
                && !rt0.yellow
                && rt0.all_red <= 0.0
                && !forced.contains_key(&pid)
                && !net.programs[pid].coordinated
            {
                self.signals[pid].elapsed += dt;
                continue;
            }
            let (n_phases, green_mask, yellow_dur) = {
                let program = &net.programs[pid];
                if program.phases.is_empty() {
                    continue;
                }
                let ph = program.phases[self.signals[pid].phase];
                (program.phases.len(), ph.green_mask, ph.yellow_secs)
            };
            let phase_demand = |mask: u64| {
                self.approaches[pid].iter().enumerate().any(|(bit, links)| {
                    mask & (1u64 << bit) != 0 && links.iter().any(|l| demand.contains(l.0))
                })
            };
            let bit_has_demand = |served: bool| {
                self.approaches[pid].iter().enumerate().any(|(bit, links)| {
                    (green_mask & (1u64 << bit) != 0) == served && links.iter().any(|l| demand.contains(l.0))
                })
            };
            // Semi-actuated coordination: the progression phase is guaranteed its
            // scheduled window on the cycle clock (force-on: side phases must
            // clear before it opens; force-off: it only yields once the window
            // has passed and a side phase is waiting), and time no side phase
            // claims returns to it (rest-in-green). Everything else — clearance
            // sequencing, preemption, gap-out — is the shared machinery below.
            let coord = {
                let program = &net.programs[pid];
                (program.coordinated && n_phases > 1 && program.cycle_length() > 0.0).then(|| {
                    let cp = program.coordinated_phase.min(n_phases - 1);
                    let cycle = program.cycle_length();
                    let t_c = (self.time + program.offset).rem_euclid(cycle);
                    let start_c = program.phase_start(cp);
                    let green_c = program.phases[cp].green_secs;
                    (cp, cycle, t_c, start_c, green_c)
                })
            };
            let force = forced.get(&pid).copied().filter(|&t| t < n_phases);
            let mut rt = self.signals[pid];
            rt.elapsed += dt;
            if rt.all_red > 0.0 {
                rt.all_red -= dt;
                if rt.all_red <= 0.0 {
                    rt.all_red = 0.0;
                    rt.phase = force.unwrap_or_else(|| match coord {
                        // Inside (or at) the progression window: the coordinated
                        // phase, unconditionally. Otherwise the next demanded
                        // side phase in ring order, falling back to coordinated
                        // rest-in-green when nothing is waiting.
                        Some((cp, cycle, t_c, start_c, green_c)) => {
                            let in_window = (t_c - start_c).rem_euclid(cycle) < green_c;
                            if in_window {
                                cp
                            } else {
                                (1..=n_phases)
                                    .map(|k| (rt.phase + k) % n_phases)
                                    .find(|&p| p == cp || phase_demand(net.programs[pid].phases[p].green_mask))
                                    .unwrap_or(cp)
                            }
                        }
                        None => (rt.phase + 1) % n_phases,
                    });
                    rt.elapsed = 0.0;
                }
            } else if rt.yellow {
                if rt.elapsed >= yellow_dur {
                    rt.yellow = false;
                    rt.all_red = all_red_of(&net.programs[pid], rt.phase);
                    rt.elapsed = 0.0;
                }
            } else if force.is_some_and(|t| t == rt.phase) {
                // Held by preemption: the track-clearing phase stays green.
            } else if force.is_some_and(|t| t != rt.phase) && rt.elapsed >= PREEMPT_GRACE {
                rt.yellow = true;
                rt.elapsed = 0.0;
            } else if let Some((cp, cycle, t_c, start_c, green_c)) = coord {
                let terminate = if rt.phase == cp {
                    // Hold green through the scheduled window; past it, yield
                    // only to a phase somebody is actually waiting for (else
                    // rest in green). Demand is judged per *phase mask*, not by
                    // served-vs-unserved bits: in a protected-permissive program
                    // the coordinated phase serves every group permissively, and
                    // a protected-left phase is a strict subset of it — the
                    // waiting left is "served", but its protected window is
                    // still owed.
                    let in_window = (t_c - start_c).rem_euclid(cycle) < green_c;
                    let side_demand = (0..n_phases)
                        .any(|p| p != cp && phase_demand(net.programs[pid].phases[p].green_mask));
                    !in_window && rt.elapsed >= MIN_GREEN && side_demand
                } else {
                    // A side phase gaps out like any actuated phase, and must be
                    // clear (yellow + all-red) by the progression window's start.
                    let clearance = yellow_dur + all_red_of(&net.programs[pid], rt.phase);
                    let force_off = (start_c - t_c).rem_euclid(cycle) <= clearance;
                    rt.elapsed >= MIN_GREEN
                        && (force_off || rt.elapsed >= MAX_GREEN || !bit_has_demand(true))
                };
                if terminate {
                    rt.yellow = true;
                    rt.elapsed = 0.0;
                }
            } else if n_phases > 1
                && rt.elapsed >= MIN_GREEN
                && bit_has_demand(false)
                && (rt.elapsed >= MAX_GREEN || !bit_has_demand(true))
            {
                rt.yellow = true;
                rt.elapsed = 0.0;
            }
            self.signals[pid] = rt;
        }
        self.active = active;
    }
}

pub struct Junctions {
    mv_off: Vec<u32>,
    mv: Vec<MovementId>,
    cf_off: Vec<u32>,
    cf: Vec<u32>,
    program: Vec<Option<ProgramId>>,
    node_bucket: Vec<u32>,
}

fn csr<T: Copy>(n: usize, key: impl Fn(&T) -> usize, items: &[T], id: impl Fn(usize) -> u32) -> (Vec<u32>, Vec<u32>) {
    let mut off = vec![0u32; n + 1];
    for it in items {
        off[key(it) + 1] += 1;
    }
    for i in 0..n {
        off[i + 1] += off[i];
    }
    let mut out = vec![0u32; items.len()];
    let mut cursor = off.clone();
    for (i, it) in items.iter().enumerate() {
        let slot = &mut cursor[key(it)];
        out[*slot as usize] = id(i);
        *slot += 1;
    }
    (off, out)
}

impl Junctions {
    pub fn build(net: &Network) -> Self {
        let n = net.nodes.len();
        // Movements and conflicts are grouped per *intersection*, not per OSM node:
        // every node in a junction cluster shares one bucket, so a conflict stored
        // under one member node is found when querying any of its siblings. Nodes
        // outside a cluster get a private singleton bucket after the junctions.
        let nj = net.junctions.len();
        let bucket = |node: NodeId| -> usize {
            match net.node_junction(node) {
                Some(j) => j.idx(),
                None => nj + node.idx(),
            }
        };
        let n_buckets = nj + n;
        let (mv_off, mv_raw) = csr(n_buckets, |m: &super::network::Movement| bucket(m.node), &net.movements, |i| i as u32);
        let (cf_off, cf) = csr(n_buckets, |c: &super::network::ConflictPoint| bucket(c.node), &net.conflicts, |i| i as u32);
        let mv = mv_raw.into_iter().map(MovementId).collect();
        let node_bucket = (0..n as u32).map(|i| bucket(NodeId(i)) as u32).collect();
        let mut program = vec![None; n_buckets];
        for (ji, j) in net.junctions.iter().enumerate() {
            program[ji] = j.program;
        }
        for (i, node) in net.nodes.iter().enumerate() {
            if net.node_junction(NodeId(i as u32)).is_none() {
                program[nj + i] = match node.control {
                    NodeControl::Signalized(p) => Some(p),
                    _ => None,
                };
            }
        }
        Self { mv_off, mv, cf_off, cf, program, node_bucket }
    }

    /// The movements crossing `node`'s intersection (the whole junction cluster).
    pub fn movements(&self, node: NodeId) -> &[MovementId] {
        let b = self.node_bucket[node.idx()] as usize;
        &self.mv[self.mv_off[b] as usize..self.mv_off[b + 1] as usize]
    }

    /// Indices into [`Network::conflicts`] of the conflict points at `node`'s
    /// intersection — aggregated across every OSM node in the junction cluster.
    pub fn conflict_ids(&self, node: NodeId) -> &[u32] {
        let b = self.node_bucket[node.idx()] as usize;
        &self.cf[self.cf_off[b] as usize..self.cf_off[b + 1] as usize]
    }

    /// The signal program controlling `node`'s intersection, if any.
    pub fn program(&self, node: NodeId) -> Option<ProgramId> {
        self.program[self.node_bucket[node.idx()] as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::map::arterial_intersection;

    #[test]
    fn signal_controller_cycles_phases_under_sustained_demand() {
        // With demand on every approach, an actuated multi-phase signal must not
        // sit on one phase — it cycles, so more than one green pattern appears.
        let net = arterial_intersection();
        let mut ctrl = SignalController::build(&net);
        let demand = LaneSet::of(net.lanes.len(), 0..net.lanes.len() as u32);
        let mut patterns: std::collections::HashSet<Vec<bool>> = std::collections::HashSet::new();
        for _ in 0..2000 {
            ctrl.advance(&net, &demand, 0.2, &Default::default());
            patterns.insert(ctrl.states(&net).iter().map(|s| *s == SignalState::Green).collect());
        }
        assert!(patterns.len() > 1, "the actuated signal cycles through phases, got {} pattern(s)", patterns.len());
    }

    #[test]
    fn indexes_movements_and_conflicts_per_node() {
        let net = arterial_intersection();
        let junctions = Junctions::build(&net);
        let center = NodeId(0);

        let indexed: Vec<u32> = junctions.movements(center).iter().map(|m| m.0).collect();
        let expected: Vec<u32> =
            (0..net.movements.len() as u32).filter(|&m| net.movement(MovementId(m)).node == center).collect();
        assert_eq!(indexed, expected, "every movement at the node is indexed");

        let cf_count = net.conflicts.iter().filter(|c| c.node == center).count();
        assert_eq!(junctions.conflict_ids(center).len(), cf_count);
        for &ci in junctions.conflict_ids(center) {
            assert_eq!(net.conflicts[ci as usize].node, center);
        }
        assert!(junctions.program(center).is_some(), "the arterial centre is signalized");
    }

    #[test]
    fn every_movement_and_conflict_is_indexed_exactly_once() {
        let net = arterial_intersection();
        let junctions = Junctions::build(&net);
        let mvs: usize = (0..net.nodes.len() as u32).map(|i| junctions.movements(NodeId(i)).len()).sum();
        let cfs: usize = (0..net.nodes.len() as u32).map(|i| junctions.conflict_ids(NodeId(i)).len()).sum();
        assert_eq!(mvs, net.movements.len());
        assert_eq!(cfs, net.conflicts.len());
    }
}
