//! Spatial domain decomposition of the step (parallelism "Option A"), built so
//! the two later options fall out of it: functional overlap of global phases
//! ("Option B") schedules over the same phase contract, and worker-per-region
//! with separate memories ("Option C") reuses the partition, with the
//! cross-shard reads below becoming its ghost set and the fate scatter its
//! wire protocol.
//!
//! # Partition
//! The network is split into shards that own whole junction clusters; a car is
//! acted on by the shard owning the node at its lane's end (`lane_home`), and
//! every occupancy entry is owned by the shard of the node where it is written
//! (`lane_entry` — a lane's `front`/`inbound`/slot state is only ever touched
//! by cars entering it at its upstream node). Clusters being shard-atomic is
//! what makes multi-node box FIFO state single-owner too. The partition is
//! plain data (two `Vec<u32>` + a count): serializable, so a future region
//! worker can be handed it verbatim.
//!
//! # Phase contract
//! Every phase of `NetWorld::step` declares what it reads and writes; SPMD
//! phases run on all shards between barriers, and a phase may read ANY
//! committed row (shared memory makes that free — under Option C those become
//! replicated ghosts) but may write only state its shard owns:
//!
//! | phase              | reads                                   | writes                      | execution |
//! |--------------------|-----------------------------------------|-----------------------------|-----------|
//! | refresh_routes     | committed fleet, network                | router tables               | global, amortized (B candidate) |
//! | advance_signals    | signal state, clock                     | signal state                | global (small) |
//! | lane-change scan   | committed rows, groups                  | per-car decision            | fork-join parallel |
//! | lane-change apply  | corridor occupancy (mutating)           | changed rows                | serial (corridors cross shards) |
//! | neighbors          | committed rows, groups                  | neighbor lists              | global |
//! | accel gather/eval  | committed rows, neighbors, signals      | per-car accel               | fork-join parallel |
//! | SPMD seed          | committed rows                          | every shard's `ShardMaps` (routed) | serial pass¹ |
//! | SPMD integrate     | own roster rows, accels                 | own rows, own deferred list | parallel, fused² |
//! | SPMD resolve       | own rows + own maps + own box FIFO      | own rows, own fates         | parallel, fused² |
//! | fate scatter       | per-shard fates                         | global fate vec             | serial — THE exchange point |
//! | crash detect       | post-resolve rows (all), fates          | per-car verdicts            | fork-join parallel |
//! | assembly           | fates, rows                             | fleet rebuild               | serial |
//!
//! ¹ Seed was first built as an all-shards broadcast between barriers; measured
//! on the loaded real map that lost ~1 ms/tick (K redundant fleet scans + two
//! full-pool barrier waits), so it runs as one serial routed pass instead —
//! same owned-writes contract, ~0.2 ms.
//! ² Integrate and resolve share no cross-shard state (a shard writes only its
//! own rows and maps throughout), so they fuse into one barrier-free task per
//! shard on the shared rayon pool (one engine-wide thread budget — never a
//! second pool). Without the `parallel` feature the same task bodies run in
//! shard order, bit-identically: they are data-independent by the table above,
//! so execution order cannot matter.

use super::*;

pub(super) struct Sharding {
    pub(super) count: usize,
    /// Entry shard per lane — the shard of its link's *upstream* node, which is
    /// where cars enter it: owns the lane's `front`/`inbound`/slot entries.
    pub(super) lane_entry: Vec<u32>,
    /// Home shard per lane — the shard of its link's *downstream* node: owns
    /// the cars currently on it (their boundary interaction happens there).
    pub(super) lane_home: Vec<u32>,
    /// Indivisible-unit (junction cluster / lone node) id per node. Kept so the
    /// partition can be re-balanced by a fresh per-unit weight without recomputing
    /// clusters. See [`rebalance`](Self::rebalance).
    pub(super) unit_of: Vec<u32>,
    /// Number of distinct units.
    pub(super) units: usize,
}

impl Sharding {
    /// Deterministic partition: each junction cluster (or lone node) is one
    /// indivisible unit. Built balanced by node count; the runtime re-balances it
    /// by live car load (see [`rebalance`](Self::rebalance)).
    pub(super) fn build(net: &Network, target: usize) -> Self {
        let n = net.nodes.len();
        // Unit id per node: its junction cluster index, or a fresh unit for a
        // clusterless node.
        let mut unit_of = vec![u32::MAX; n];
        for (ji, j) in net.junctions.iter().enumerate() {
            for &nd in &j.nodes {
                unit_of[nd.idx()] = ji as u32;
            }
        }
        let mut next = net.junctions.len() as u32;
        for u in unit_of.iter_mut() {
            if *u == u32::MAX {
                *u = next;
                next += 1;
            }
        }
        let units = next as usize;
        // Seed weight: node count per unit (a car-free network still balances).
        let mut size = vec![0.0f64; units];
        for &u in &unit_of {
            size[u as usize] += 1.0;
        }
        let count = Self::clamp_count(target);
        let mut s = Self { count, lane_entry: vec![0; net.lanes.len()], lane_home: vec![0; net.lanes.len()], unit_of, units };
        s.assign(net, &size);
        s
    }

    /// Re-partition the units onto shards by `unit_weight` (e.g. live car counts),
    /// keeping cluster atomicity. The SPMD span is a barrier — it waits on the
    /// slowest shard — so balancing the *work* (cars), not the node count, is what
    /// lifts parallel efficiency once traffic concentrates. Cheap: O(units + lanes),
    /// deterministic (greedy least-loaded in unit order).
    pub(super) fn rebalance(&mut self, net: &Network, unit_weight: &[f64]) {
        self.assign(net, unit_weight);
    }

    /// Greedy least-loaded assignment of units → shards by `weight`, then the
    /// per-lane entry/home shard tables that follow from it.
    fn assign(&mut self, net: &Network, weight: &[f64]) {
        let mut load = vec![0.0f64; self.count];
        let mut shard_of_unit = vec![0u32; self.units];
        for u in 0..self.units {
            let s = (0..self.count).min_by(|&a, &b| load[a].total_cmp(&load[b])).unwrap();
            shard_of_unit[u] = s as u32;
            load[s] += weight[u];
        }
        let node_shard: Vec<u32> = self.unit_of.iter().map(|&u| shard_of_unit[u as usize]).collect();
        for li in 0..net.links.len() {
            let l = net.links[li];
            for k in 0..l.lane_count {
                let lane = (l.lane_start.0 + k) as usize;
                self.lane_entry[lane] = node_shard[l.from.idx()];
                self.lane_home[lane] = node_shard[l.to.idx()];
            }
        }
    }

    /// The SPMD span barriers `count` tasks that must all be running at once,
    /// so the shard count can never exceed the rayon pool (a larger count
    /// would deadlock the barrier). Serial builds have no barrier — any count
    /// works phase-major.
    fn clamp_count(target: usize) -> usize {
        #[cfg(feature = "parallel")]
        {
            target.max(1).min(rayon::current_num_threads().max(1))
        }
        #[cfg(not(feature = "parallel"))]
        {
            target.max(1)
        }
    }

    /// The shard that resolves a vehicle this tick: the one owning the node at
    /// the end of its current lane (a crossing car's movement node is that same
    /// node, so this is stable through a crossing).
    pub(super) fn home_of(&self, lane: LaneId) -> u32 {
        self.lane_home[lane.idx()]
    }

    pub(super) fn entry_of(&self, lane: LaneId) -> u32 {
        self.lane_entry[lane.idx()]
    }
}

/// Per-shard occupancy maps for one tick's resolution, persisted across ticks
/// (cleared, capacity kept) so sharded stepping allocates nothing at steady
/// state.
#[derive(Default)]
pub(super) struct ShardMaps {
    pub(super) front: IntMap<f64>,
    pub(super) front_speed: IntMap<f64>,
    pub(super) interior_occ: IntMap<u32>,
    pub(super) inbound: IntMap<f64>,
}

impl ShardMaps {
    pub(super) fn clear(&mut self) {
        self.front.clear();
        self.front_speed.clear();
        self.interior_occ.clear();
        self.inbound.clear();
    }
}

/// A shard task's window onto the fleet: write only rows the shard owns. This
/// is the seam Option C swaps, and both backends exist today:
///
/// - `Shared` — one fleet buffer, all threads share memory (Option A). The
///   deref is a raw index.
/// - `Regions` — each shard's rows live in a private buffer, indexed through a
///   global→(shard, slot) map (Option C's dry-run: the SPMD span runs with the
///   regions genuinely unable to see each other's rows, proving no phase
///   smuggles a foreign access; the copy-in/merge-out around it is exactly the
///   migration a real worker-per-region split performs over the wire).
///
/// Soundness of the aliasing in both: writes go only to roster-owned indices
/// (each fleet index is in exactly one shard's roster), so no row is ever
/// accessed by two tasks.
pub(super) enum ShardView<'a> {
    Shared(&'a mut [NetVehicle]),
    Regions { bufs: Vec<*mut NetVehicle>, index: &'a [(u32, u32)] },
}

unsafe impl Sync for ShardView<'_> {}
unsafe impl Send for ShardView<'_> {}

impl<'a> ShardView<'a> {
    /// Region-isolated backend over per-shard buffers (`index` maps a global
    /// fleet index to its shard and slot).
    pub(super) fn regions(buffers: &'a mut [Vec<NetVehicle>], index: &'a [(u32, u32)]) -> Self {
        Self::Regions { bufs: buffers.iter_mut().map(|b| b.as_mut_ptr()).collect(), index }
    }

    /// Mutable access to a row this shard owns.
    ///
    /// # Safety
    /// `i` must be owned by the calling shard's roster this tick.
    #[allow(clippy::mut_from_ref)]
    pub(super) unsafe fn row(&self, i: usize) -> &mut NetVehicle {
        match self {
            Self::Shared(rows) => unsafe { &mut *(rows.as_ptr().add(i) as *mut NetVehicle) },
            Self::Regions { bufs, index } => {
                let (s, l) = index[i];
                unsafe { &mut *bufs[s as usize].add(l as usize) }
            }
        }
    }
}

/// Disjoint per-index `&mut` access to a slice from simultaneously running
/// shard tasks. Sound because the rosters partition the index space — each
/// index is written by exactly one task.
#[cfg(feature = "parallel")]
pub(super) struct SyncSlice<T>(*mut T);

#[cfg(feature = "parallel")]
unsafe impl<T: Send> Sync for SyncSlice<T> {}

#[cfg(feature = "parallel")]
impl<T> SyncSlice<T> {
    pub(super) fn new(s: &mut [T]) -> Self {
        Self(s.as_mut_ptr())
    }

    /// # Safety
    /// `i` must be in bounds and written by only the calling task this phase.
    #[allow(clippy::mut_from_ref)]
    pub(super) unsafe fn at(&self, i: usize) -> &mut T {
        unsafe { &mut *self.0.add(i) }
    }
}

/// Reusable per-shard parallel region — the primitive the tick pipeline moves
/// work into (Amdahl: anything a shard can compute over its owned cars/nodes
/// belongs here, off the serial path). Runs `f(shard_index, roster)` for every
/// shard, in parallel on the shared pool under the `parallel` feature and in
/// shard order without it; the closure reads any committed state and returns a
/// per-shard result, merged by the caller (the well-defined sync point). Pure
/// by contract — no shard writes another shard's owned state — so the parallel
/// and serial executions are equivalent.
#[cfg(feature = "parallel")]
pub(super) fn shard_map<T, F>(rosters: &[Vec<u32>], f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize, &[u32]) -> T + Sync,
{
    use rayon::prelude::*;
    rosters.par_iter().enumerate().map(|(s, r)| f(s, r)).collect()
}

#[cfg(not(feature = "parallel"))]
pub(super) fn shard_map<T, F>(rosters: &[Vec<u32>], f: F) -> Vec<T>
where
    F: Fn(usize, &[u32]) -> T,
{
    rosters.iter().enumerate().map(|(s, r)| f(s, r)).collect()
}
