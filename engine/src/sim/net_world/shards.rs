//! Spatial domain decomposition for the step's boundary resolution: the road
//! network is partitioned into shards that own whole junction clusters, and a
//! car is resolved by the shard owning the node it is interacting with. Every
//! piece of shared occupancy state (`front`/`inbound`/interior slots keyed by
//! the lane being *entered*, box FIFO keyed by intersection) is written only at
//! that node — so with clusters atomic per shard, each shard's resolution is
//! fully independent of the others: coarse tasks with real work per core, the
//! grain Amdahl (and this codebase's measurements) demand, with no cross-shard
//! phase at all.

use super::*;

pub(super) struct Sharding {
    pub(super) count: usize,
    /// Entry shard per lane — the shard of its link's *upstream* node, which is
    /// where cars enter it: owns the lane's `front`/`inbound`/slot entries.
    pub(super) lane_entry: Vec<u32>,
    /// Home shard per lane — the shard of its link's *downstream* node: owns
    /// the cars currently on it (their boundary interaction happens there).
    pub(super) lane_home: Vec<u32>,
}

impl Sharding {
    /// Deterministic partition: each junction cluster (or lone node) is one
    /// indivisible unit; units, in stable order, are placed greedily onto the
    /// least-loaded shard (by node count).
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
        // Unit sizes, then greedy balance in unit order (stable, deterministic).
        let mut size = vec![0u32; next as usize];
        for &u in &unit_of {
            size[u as usize] += 1;
        }
        let count = target.max(1);
        let mut load = vec![0u32; count];
        let mut shard_of_unit = vec![0u32; next as usize];
        for u in 0..next as usize {
            let s = (0..count).min_by_key(|&s| load[s]).unwrap();
            shard_of_unit[u] = s as u32;
            load[s] += size[u];
        }
        let node_shard: Vec<u32> = unit_of.iter().map(|&u| shard_of_unit[u as usize]).collect();
        let mut lane_entry = vec![0u32; net.lanes.len()];
        let mut lane_home = vec![0u32; net.lanes.len()];
        for li in 0..net.links.len() {
            let l = net.links[li];
            for k in 0..l.lane_count {
                let lane = (l.lane_start.0 + k) as usize;
                lane_entry[lane] = node_shard[l.from.idx()];
                lane_home[lane] = node_shard[l.to.idx()];
            }
        }
        Self { count, lane_entry, lane_home }
    }

    /// The shard that resolves a vehicle this tick: the one owning the node at
    /// the end of its current lane (a crossing car's movement node is that same
    /// node, so this is stable through a crossing).
    pub(super) fn home_of(&self, lane: LaneId) -> u32 {
        self.lane_home[lane.idx()]
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

/// A `&mut [NetVehicle]` that shard tasks index concurrently. Safety rests on
/// the roster invariant: every fleet index appears in exactly one shard's
/// roster (each is built in a single pass bucketing by home shard), so no two
/// tasks ever touch the same row.
pub(super) struct SharedRows<'a>(pub(super) &'a mut [NetVehicle]);

unsafe impl Sync for SharedRows<'_> {}

impl SharedRows<'_> {
    /// # Safety
    /// `i` must be owned by the calling shard's roster this tick.
    #[allow(clippy::mut_from_ref)]
    pub(super) unsafe fn row(&self, i: usize) -> &mut NetVehicle {
        unsafe { &mut *(self.0.as_ptr().add(i) as *mut NetVehicle) }
    }
}
