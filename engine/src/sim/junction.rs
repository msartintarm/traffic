//! Junctions as index-addressed views over the flat network arrays. A
//! [`Junctions`] table is built once and gives O(degree) access to each node's
//! movements and conflict points — the per-intersection grouping that the runtime
//! (box gates, in-box conflict avoidance, crash detection) would otherwise get by
//! scanning every movement or conflict each tick.
//!
//! It stays data-oriented: CSR (`offsets` + concatenated ids) over the existing
//! `Vec`s, no per-object ownership, so the GPU-friendly SoA layout is preserved.

use std::collections::HashSet;

use super::network::{LinkId, MovementId, Network, NodeControl, NodeId, ProgramId};
use super::signal::{SignalProgram, SignalState, DEFAULT_ALL_RED};

/// Actuation timing: green is held at least `MIN_GREEN`, extended while vehicles
/// keep arriving (gap-out) up to `MAX_GREEN`, and only terminated when a
/// conflicting approach is waiting. `DETECT` is how far back a stop-line detector
/// senses demand.
pub const DETECT: f64 = 35.0;
const MIN_GREEN: f64 = 6.0;
const MAX_GREEN: f64 = 45.0;

fn all_red_of(program: &SignalProgram, phase: usize) -> f64 {
    let ar = program.phases[phase].all_red_secs;
    if ar > 0.0 {
        ar
    } else {
        DEFAULT_ALL_RED
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
    /// Approach links per program, per group bit — the links feeding each group.
    approaches: Vec<Vec<Vec<LinkId>>>,
    time: f64,
}

impl SignalController {
    pub fn build(net: &Network) -> Self {
        let signals = vec![SignalRuntime { phase: 0, elapsed: 0.0, yellow: false, all_red: 0.0 }; net.programs.len()];
        let mut approaches: Vec<Vec<Vec<LinkId>>> = vec![Vec::new(); net.programs.len()];
        for g in &net.groups {
            let bits = &mut approaches[g.program.idx()];
            if bits.len() <= g.bit as usize {
                bits.resize(g.bit as usize + 1, Vec::new());
            }
        }
        for mv in &net.movements {
            if let Some(gid) = mv.signal_group {
                let g = net.groups[gid.idx()];
                let link = net.lane(mv.from_lane).link;
                let feeders = &mut approaches[g.program.idx()][g.bit as usize];
                if !feeders.contains(&link) {
                    feeders.push(link);
                }
            }
        }
        Self { signals, approaches, time: 0.0 }
    }

    fn group_state(&self, net: &Network, program: ProgramId, bit: u8) -> SignalState {
        let prog = &net.programs[program.idx()];
        if prog.coordinated {
            prog.state_of(bit, self.time)
        } else {
            self.signals[program.idx()].state_of(bit, prog)
        }
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
        if program.coordinated {
            let cycle = program.cycle_length();
            if cycle <= 0.0 {
                return 0.0;
            }
            let mut t = (self.time + program.offset).rem_euclid(cycle);
            for phase in &program.phases {
                if t < phase.length() {
                    return if phase.green_mask & mask != 0 && t < phase.green_secs { t } else { 0.0 };
                }
                t -= phase.length();
            }
            0.0
        } else {
            let rt = self.signals[group.program.idx()];
            let served = program.phases.get(rt.phase).is_some_and(|ph| ph.green_mask & mask != 0);
            if served && !rt.yellow && rt.all_red <= 0.0 {
                rt.elapsed
            } else {
                0.0
            }
        }
    }

    /// Advance actuated signals: hold green while its approaches keep demand,
    /// terminate on max-green or a gap-out with a conflicting approach waiting.
    /// `demand` is the set of link ids with a vehicle within [`DETECT`] of a line.
    pub fn advance(&mut self, net: &Network, demand: &HashSet<u32>, dt: f64) {
        self.time += dt;
        for pid in 0..self.signals.len() {
            let (n_phases, green_mask, yellow_dur) = {
                let program = &net.programs[pid];
                if program.phases.is_empty() || program.coordinated {
                    continue;
                }
                let ph = program.phases[self.signals[pid].phase];
                (program.phases.len(), ph.green_mask, ph.yellow_secs)
            };
            let bit_has_demand = |served: bool| {
                self.approaches[pid].iter().enumerate().any(|(bit, links)| {
                    (green_mask & (1u64 << bit) != 0) == served && links.iter().any(|l| demand.contains(&l.0))
                })
            };
            let mut rt = self.signals[pid];
            rt.elapsed += dt;
            if rt.all_red > 0.0 {
                rt.all_red -= dt;
                if rt.all_red <= 0.0 {
                    rt.all_red = 0.0;
                    rt.phase = (rt.phase + 1) % n_phases;
                    rt.elapsed = 0.0;
                }
            } else if rt.yellow {
                if rt.elapsed >= yellow_dur {
                    rt.yellow = false;
                    rt.all_red = all_red_of(&net.programs[pid], rt.phase);
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
    }
}

pub struct Junctions {
    mv_off: Vec<u32>,
    mv: Vec<MovementId>,
    cf_off: Vec<u32>,
    cf: Vec<u32>,
    program: Vec<Option<ProgramId>>,
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
        let (mv_off, mv_raw) = csr(n, |m: &super::network::Movement| m.node.idx(), &net.movements, |i| i as u32);
        let (cf_off, cf) = csr(n, |c: &super::network::ConflictPoint| c.node.idx(), &net.conflicts, |i| i as u32);
        let mv = mv_raw.into_iter().map(MovementId).collect();
        let program = net
            .nodes
            .iter()
            .map(|node| match node.control {
                NodeControl::Signalized(p) => Some(p),
                _ => None,
            })
            .collect();
        Self { mv_off, mv, cf_off, cf, program }
    }

    /// The movements crossing `node`.
    pub fn movements(&self, node: NodeId) -> &[MovementId] {
        &self.mv[self.mv_off[node.idx()] as usize..self.mv_off[node.idx() + 1] as usize]
    }

    /// Indices into [`Network::conflicts`] of the conflict points at `node`.
    pub fn conflict_ids(&self, node: NodeId) -> &[u32] {
        &self.cf[self.cf_off[node.idx()] as usize..self.cf_off[node.idx() + 1] as usize]
    }

    /// The signal program controlling `node`, if any.
    pub fn program(&self, node: NodeId) -> Option<ProgramId> {
        self.program[node.idx()]
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
        let demand: HashSet<u32> = (0..net.links.len() as u32).collect();
        let mut patterns: HashSet<Vec<bool>> = HashSet::new();
        for _ in 0..2000 {
            ctrl.advance(&net, &demand, 0.2);
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
