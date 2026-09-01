//! Flow-field routing over a fixed set of destinations.
//!
//! For each destination in use we keep a next-hop field: `next_hop[dest][link]`
//! is the outgoing link to take from `link` to reach `dest` fastest. A vehicle
//! routes by an O(1) lookup per intersection ([`FieldRouter::next_hop`]) instead
//! of a per-vehicle path search.
//!
//! Fields are rebuilt from live per-link travel times, so in-flight vehicles pick
//! fresh next-hops as jams form. Computed by [`flowfield`] (Bellman–Ford Jacobi);
//! `flowfield_gpu` runs the same relaxation on a GPU device.

use std::collections::HashMap;

use super::flowfield;
use super::network::{LinkId, Network};

pub struct FieldRouter {
    adj: Vec<Vec<u32>>,
    /// Reverse adjacency (predecessor lists), built once so each field's Dijkstra
    /// doesn't rebuild it — the difference between a city router installing in ~1 s
    /// and ~40 s.
    pred: Vec<Vec<u32>>,
    dests: Vec<LinkId>,
    slot: HashMap<u32, usize>,
    /// `next_hop[slot][from_link]` toward `dests[slot]` (`None` = unreachable).
    next_hop: Vec<Vec<Option<LinkId>>>,
    /// `dist[slot][from_link]` = travel cost from `from_link` to `dests[slot]`,
    /// kept alongside the next-hop field so a car that can't reach its routed lane
    /// can fall back onto the *forward-most* movement it can take (nearest to the
    /// destination) instead of an arbitrary one. `u64::MAX` = unreachable.
    dist: Vec<Vec<u64>>,
    /// Next slot a recompute cycle starts from, so successive cycles round-robin the fields.
    cursor: usize,
    /// An in-flight budgeted recompute: refresh every field once against a cost snapshot, one
    /// bounded Dijkstra slice per [`advance_recompute`] call, so a whole-map rebuild spreads
    /// across frames instead of stalling one. `None` when routing is up to date.
    rc: Option<Recompute>,
    /// Arterial ("trunk") routing mask — `None` = whole-graph fields. When set,
    /// each destination's field is solved only over trunk links plus the locals
    /// within the destination's access neighborhood; a car anywhere else follows
    /// the shared ascent field below ("drive to a main road first"), the way a
    /// real driver plans: locals at the ends, arterials in between.
    trunk: Option<Vec<bool>>,
    /// Shared to-nearest-trunk field (next hop + cost), the fallback for links no
    /// destination field covers. Empty in whole-graph mode.
    ascent_next: Vec<Option<LinkId>>,
    ascent_dist: Vec<u64>,
    /// Cycles since each destination field was last solved — dirty-gated slots
    /// age until [`AGE_FORCE`] forces a full refresh, bounding staleness.
    age: Vec<u32>,
    /// `(solved, skipped)` of the last targeted cycle, for probes and tests.
    last_cycle: (usize, usize),
}

/// A clean (not-dirty) field may skip at most this many cycles before a full
/// solve is forced — the staleness bound for links no querying car currently
/// touches (a better route opening up elsewhere is invisible to the dirty walk,
/// which only re-prices the field's *own* current paths).
const AGE_FORCE: u32 = 3;
/// How many of a slot's target links the dirty walk re-prices (cars bound for
/// one destination share path tails, so a handful of walks covers the tree).
const DIRTY_SAMPLES: usize = 8;

/// How far off the trunk a destination's access neighborhood reaches (ms of
/// travel on the unrestricted graph): enough local fabric that a trip's last
/// leg descends sensibly, small enough that fields stay ~trunk-sized.
const NEIGHBORHOOD_MS: u64 = 90_000;
/// Bias added to ascent-based distance estimates so any candidate a real field
/// covers always outranks one routed via the ascent fallback.
const ASCENT_BIAS: u64 = 1 << 40;

struct Recompute {
    /// Cost snapshot taken when the cycle began, so every field in it is mutually consistent.
    cost: Vec<u64>,
    /// How many fields to solve concurrently (the batch width): 1 serial, else one per core.
    width: usize,
    /// The cycle's work list: `(slot, targets)` — `targets` Some = a targeted
    /// early-terminating solve published by merge, None = a full solve published
    /// by replacement.
    plan: Vec<(usize, Option<Vec<u32>>)>,
    /// Next unstarted entry in `plan`.
    started: usize,
    /// Fields *published* (swapped into the live buffers) so far; the cycle ends at `total`.
    published: usize,
    /// Fields in this cycle.
    total: usize,
    /// The batch of in-flight fields `(slot, targeted?, resumable Dijkstra, subgraph mask)`,
    /// each advanced per tick — concurrently across cores when `width > 1`.
    batch: Vec<(usize, bool, flowfield::PartialField, Option<Vec<bool>>)>,
}

impl FieldRouter {
    /// Build a router for `dests` (duplicates ignored), with every field
    /// initialised against `cost`.
    pub fn new(net: &Network, dests: &[LinkId], cost: &[u64]) -> Self {
        Self::new_with_trunk(net, dests, cost, None)
    }

    /// [`new`](Self::new) with an optional arterial mask: `trunk[link]` marks the
    /// through-network fields are solved over (see the `trunk` field docs).
    pub fn new_with_trunk(net: &Network, dests: &[LinkId], cost: &[u64], trunk: Option<Vec<bool>>) -> Self {
        let adj = flowfield::adjacency(net);
        let pred = flowfield::reverse(&adj);
        let mut unique = Vec::new();
        let mut slot = HashMap::new();
        for &d in dests {
            slot.entry(d.0).or_insert_with(|| {
                unique.push(d);
                unique.len() - 1
            });
        }
        let next_hop = vec![Vec::new(); unique.len()];
        let dist = vec![Vec::new(); unique.len()];
        let mut router = Self {
            adj, pred, dests: unique, slot, next_hop, dist, cursor: 0, rc: None,
            trunk, ascent_next: Vec::new(), ascent_dist: Vec::new(),
            age: Vec::new(), last_cycle: (0, 0),
        };
        router.age = vec![0; router.dests.len()];
        router.recompute(cost);
        router
    }

    /// A destination field's subgraph in trunk mode: the trunk plus this
    /// destination's local access neighborhood. `None` in whole-graph mode.
    fn field_mask(&self, dest: LinkId, cost: &[u64]) -> Option<Vec<bool>> {
        let trunk = self.trunk.as_ref()?;
        let mut allowed = trunk.clone();
        for l in flowfield::PartialField::neighborhood(&self.pred, dest, cost, NEIGHBORHOOD_MS) {
            allowed[l as usize] = true;
        }
        Some(allowed)
    }

    /// Total fields a recompute cycle covers: every destination, plus the shared
    /// ascent field in trunk mode.
    fn cycle_total(&self) -> usize {
        self.dests.len() + self.trunk.is_some() as usize
    }

    /// The resumable solve for cycle slot `i`: a destination field, or (last slot,
    /// trunk mode only) the multi-source ascent field.
    fn start_field(&self, i: usize, cost: &[u64]) -> (flowfield::PartialField, Option<Vec<bool>>) {
        if i < self.dests.len() {
            (flowfield::PartialField::new(self.pred.len(), self.dests[i]), self.field_mask(self.dests[i], cost))
        } else {
            let trunk = self.trunk.as_ref().unwrap();
            let seeds = trunk.iter().enumerate().filter(|(_, &t)| t).map(|(l, _)| l as u32);
            (flowfield::PartialField::new_multi(self.pred.len(), seeds), None)
        }
    }

    /// Merge an early-terminated solve over the live field: within the settled
    /// region the new distances are exact and next-hops are chosen among
    /// *settled* neighbors only (a settled link's true successor is always
    /// settled first, so this loses nothing) — which makes the fresh region
    /// closed under its own hops: once a route enters it, it descends fresh
    /// distances straight to the destination. Outside it the previous field is
    /// kept whole, an internally consistent older tree. The two can meet only
    /// stale→fresh, never alternate, so the merged field cannot cycle.
    fn publish_merged(&mut self, slot: usize, dist: Vec<u64>, settled: Vec<bool>, cost: &[u64]) {
        if self.next_hop[slot].len() != dist.len() {
            return self.publish(slot, dist, cost); // no previous field to merge over
        }
        let (hops, live) = (&mut self.next_hop[slot], &mut self.dist[slot]);
        for a in 0..dist.len() {
            if !settled[a] {
                continue;
            }
            live[a] = dist[a];
            hops[a] = self.adj[a]
                .iter()
                .copied()
                .filter(|&b| settled[b as usize] && dist[b as usize] != u64::MAX)
                .min_by_key(|&b| cost[b as usize].saturating_add(dist[b as usize]))
                .map(LinkId);
        }
    }

    /// Publish a finished field into the live buffers.
    fn publish(&mut self, slot: usize, dist: Vec<u64>, cost: &[u64]) {
        let next = flowfield::next_hops(&self.adj, &dist, cost);
        if slot < self.dests.len() {
            self.next_hop[slot] = next;
            self.dist[slot] = dist;
        } else {
            self.ascent_next = next;
            self.ascent_dist = dist;
        }
    }

    /// Number of distinct destination fields maintained.
    pub fn destination_count(&self) -> usize {
        self.dests.len()
    }

    /// Whether a budgeted recompute cycle is in flight.
    pub fn recompute_pending(&self) -> bool {
        self.rc.is_some()
    }

    /// Begin a cycle that refreshes every field once against `cost` (snapshotted for
    /// consistency), resumable via [`advance_recompute`]. `width` fields are solved
    /// concurrently (1 = serial; set it to the core count to spread the rebuild across cores).
    /// Continues from `cursor` so successive cycles round-robin. No-op with no destinations.
    pub fn begin_recompute(&mut self, cost: Vec<u64>, width: usize) {
        if self.dests.is_empty() {
            return;
        }
        let total = self.cycle_total();
        let base = self.cursor;
        self.cursor = (self.cursor + 1) % total; // rotate the start so freshness spreads over cycles
        let plan: Vec<(usize, Option<Vec<u32>>)> = (0..total).map(|i| ((base + i) % total, None)).collect();
        self.begin_plan(cost, width, plan);
    }

    /// [`begin_recompute`] restricted to what will actually be read: `targets`
    /// maps a destination link to the links that will query its field this
    /// cycle (cars bound for it + its spawn gateways). Per destination:
    /// - **clean & young** (its current paths still price within tolerance of
    ///   the field's own distance claims): skipped entirely;
    /// - **dirty**: a targeted solve, early-terminated once every target is
    ///   settled and merge-published (the untouched region keeps its previous
    ///   values — stale-but-valid hops, the same staleness any link has
    ///   mid-cycle);
    /// - **aged past [`AGE_FORCE`]** (or with no targets to judge by): a full
    ///   solve, so links no car currently touches are refreshed on a bounded
    ///   cadence too.
    pub fn begin_recompute_targeted(&mut self, cost: Vec<u64>, width: usize, targets: &HashMap<u32, Vec<u32>>) {
        if self.dests.is_empty() {
            return;
        }
        let mut plan: Vec<(usize, Option<Vec<u32>>)> = Vec::new();
        let (mut solved, mut skipped) = (0usize, 0usize);
        for s in 0..self.dests.len() {
            let t = targets.get(&self.dests[s].0).map(Vec::as_slice).unwrap_or(&[]);
            if self.age[s] >= AGE_FORCE {
                // Full refresh. Ages count cycles since the last *full* solve —
                // a targeted solve leaves the unqueried region stale, so it must
                // not reset the clock (a busy field would otherwise never
                // refresh its far side at all).
                plan.push((s, None));
                self.age[s] = 0;
                solved += 1;
            } else if t.is_empty() || !self.dirty(s, t, &cost) {
                self.age[s] += 1;
                skipped += 1;
            } else {
                plan.push((s, Some(t.to_vec())));
                self.age[s] += 1;
                solved += 1;
            }
        }
        self.last_cycle = (solved, skipped);
        if plan.is_empty() {
            return; // nothing moved — no cycle at all
        }
        if self.trunk.is_some() {
            plan.push((self.dests.len(), None)); // the shared ascent field rides every real cycle
        }
        // Rotate so the same slots aren't always freshest-first.
        let total = plan.len();
        plan.rotate_left(self.cursor % total);
        self.cursor = (self.cursor + 1) % self.cycle_total().max(1);
        self.begin_plan(cost, width, plan);
    }

    /// Whether a field's answers still price correctly: re-walk (a sample of)
    /// its querying links' current next-hop chains under the *new* costs and
    /// compare with the field's stored distances. A broken chain or a >10 %
    /// (and >5 s) drift means the routes cars are actually following have
    /// materially changed — solve; otherwise the field is still telling the
    /// truth and can skip the cycle.
    fn dirty(&self, slot: usize, targets: &[u32], cost: &[u64]) -> bool {
        let (hops, dist) = (&self.next_hop[slot], &self.dist[slot]);
        if hops.is_empty() {
            return true;
        }
        let dest = self.dests[slot].0;
        for &t in targets.iter().take(DIRTY_SAMPLES) {
            let claimed = dist[t as usize];
            if claimed == u64::MAX {
                continue; // ascent-covered or unreachable — nothing to compare
            }
            let (mut cur, mut acc, mut steps) = (t, 0u64, 0u32);
            let ok = loop {
                if cur == dest {
                    break true;
                }
                let Some(next) = hops[cur as usize] else { break false };
                acc = acc.saturating_add(cost[next.idx()]);
                cur = next.0;
                steps += 1;
                if steps > 600 {
                    break false;
                }
            };
            if !ok || (acc.abs_diff(claimed) > 5_000 && acc.abs_diff(claimed) * 10 > claimed) {
                return true;
            }
        }
        false
    }

    fn begin_plan(&mut self, cost: Vec<u64>, width: usize, plan: Vec<(usize, Option<Vec<u32>>)>) {
        let total = plan.len();
        let width = width.clamp(1, total);
        let mut rc = Recompute { cost, width, plan, started: 0, published: 0, total, batch: Vec::with_capacity(width) };
        self.fill_batch(&mut rc);
        self.rc = Some(rc);
    }

    /// `(solved, skipped)` destination counts of the most recent targeted cycle.
    pub fn last_cycle_stats(&self) -> (usize, usize) {
        self.last_cycle
    }

    /// Top the in-flight batch back up to its width with the next unstarted fields.
    fn fill_batch(&self, rc: &mut Recompute) {
        while rc.batch.len() < rc.width && rc.started < rc.total {
            let (slot, targets) = rc.plan[rc.started].clone();
            let (mut pf, mask) = self.start_field(slot, &rc.cost);
            let targeted = targets.is_some();
            if let Some(t) = targets {
                pf = pf.with_targets(&t);
            }
            rc.batch.push((slot, targeted, pf, mask));
            rc.started += 1;
        }
    }

    /// Advance the in-flight recompute by up to `total_budget` settled links this call, split
    /// evenly across the batch — a bounded slice of each field's Dijkstra — so a whole-map
    /// rebuild is spread across frames and no frame does the full O(links log links) sweep. The
    /// batch's fields are independent Dijkstras, so with `width > 1` (and the `parallel`
    /// feature) they advance **concurrently across cores**, cutting the per-frame wall time by
    /// ~the core count at the same total work. A field is published (next-hop + distances
    /// swapped into the live buffers) only once its Dijkstra completes — until then the previous
    /// field stays live, so routing is never read mid-solve. No-op when nothing is pending.
    pub fn advance_recompute(&mut self, total_budget: usize) {
        let Some(mut rc) = self.rc.take() else {
            return;
        };
        let per_field = (total_budget / rc.batch.len().max(1)).max(1);
        // Advance each in-flight field; concurrently when the batch is wide enough to be worth it.
        let (pred, cost) = (&self.pred, &rc.cost);
        #[cfg(feature = "parallel")]
        let done: Vec<bool> = if rc.batch.len() > 1 {
            use rayon::prelude::*;
            rc.batch.par_iter_mut().map(|(_, _, pf, m)| pf.advance_masked(pred, cost, per_field, m.as_deref())).collect()
        } else {
            rc.batch.iter_mut().map(|(_, _, pf, m)| pf.advance_masked(pred, cost, per_field, m.as_deref())).collect()
        };
        #[cfg(not(feature = "parallel"))]
        let done: Vec<bool> = rc.batch.iter_mut().map(|(_, _, pf, m)| pf.advance_masked(pred, cost, per_field, m.as_deref())).collect();

        // Publish every field that finished this tick, then refill the batch from the remaining
        // slots. Iterate back-to-front so `swap_remove` doesn't skip an entry.
        for i in (0..rc.batch.len()).rev() {
            if done[i] {
                let (slot, targeted, mut pf, _) = rc.batch.swap_remove(i);
                let (dist, settled) = pf.take_parts();
                if targeted {
                    self.publish_merged(slot, dist, settled, &rc.cost);
                } else {
                    self.publish(slot, dist, &rc.cost);
                }
                rc.published += 1;
            }
        }
        if rc.published >= rc.total {
            return; // cycle done — `rc` dropped, so `recompute_pending` is false
        }
        self.fill_batch(&mut rc);
        self.rc = Some(rc);
    }

    /// The destinations this router routes to.
    /// The predecessor graph the field solves walk (reverse adjacency) — an
    /// external/overlapped solver runs `flowfield::distances_to_with` on it.
    pub fn pred(&self) -> &[Vec<u32>] {
        &self.pred
    }

    pub fn destinations(&self) -> &[LinkId] {
        &self.dests
    }

    /// Whether this router maintains a field for `dest`.
    pub fn knows(&self, dest: LinkId) -> bool {
        self.slot.contains_key(&dest.0)
    }

    /// Rebuild every next-hop field against the current per-link `cost` (ms).
    /// Call periodically so routing tracks live congestion. The fields are
    /// independent, so with the `parallel` feature (the browser's wasm-thread pool)
    /// they're computed across cores — the whole-city install, hundreds of fields,
    /// otherwise dominates the first frame.
    pub fn recompute(&mut self, cost: &[u64]) {
        let total = self.cycle_total();
        let field = |i: usize| {
            let (mut pf, mask) = self.start_field(i, cost);
            pf.advance_masked(&self.pred, cost, usize::MAX, mask.as_deref());
            pf.take_dist()
        };
        #[cfg(feature = "parallel")]
        let dists: Vec<_> = {
            use rayon::prelude::*;
            (0..total).into_par_iter().map(field).collect()
        };
        #[cfg(not(feature = "parallel"))]
        let dists: Vec<_> = (0..total).map(field).collect();
        for (s, dist) in dists.into_iter().enumerate() {
            self.publish(s, dist, cost);
        }
    }

    /// The destinations in slot order — the order [`recompute_from_distances`]
    /// expects its `dist_per_slot` in.
    pub fn dests_in_slot_order(&self) -> &[LinkId] {
        &self.dests
    }

    /// Adjacency (CSR-flattenable) for an external solver (e.g. the GPU backend)
    /// to run the same relaxation the pure [`recompute`] does.
    pub fn adjacency(&self) -> &[Vec<u32>] {
        &self.adj
    }

    /// Rebuild the next-hop fields from reverse distances computed elsewhere
    /// (the GPU backend), `dist_per_slot[s]` being the distances to
    /// `dests_in_slot_order()[s]`. Equivalent to [`recompute`] but with the
    /// expensive Bellman–Ford done off-core.
    pub fn recompute_from_distances(&mut self, cost: &[u64], dist_per_slot: &[Vec<u64>]) {
        // A batch whose slot count differs from the current field set is stale — the
        // destinations changed (the router was reinstalled) after it was dispatched.
        // Applying it would index past the fields or bind distances to the wrong
        // destinations, so drop it wholesale; the next dispatch matches the new set.
        if dist_per_slot.len() != self.next_hop.len() {
            return;
        }
        // Rebuild every destination's next-hop field from the GPU-computed distances.
        // O(dests × links), so on a city map parallelize it across cores (with the
        // `parallel` feature / browser thread pool) to keep it off the frame's critical
        // path when a readback lands.
        let adj = &self.adj;
        let field = |dist: &Vec<u64>| (flowfield::next_hops(adj, dist, cost), dist.clone());
        #[cfg(feature = "parallel")]
        let fields: Vec<_> = {
            use rayon::prelude::*;
            dist_per_slot.par_iter().map(field).collect()
        };
        #[cfg(not(feature = "parallel"))]
        let fields: Vec<_> = dist_per_slot.iter().map(field).collect();
        for (s, (next_hop, dist)) in fields.into_iter().enumerate() {
            self.next_hop[s] = next_hop;
            self.dist[s] = dist;
        }
    }

    /// The next link to take from `from` toward `dest`: `None` if `from` already
    /// is the destination, the destination is unreachable, or it is not one this
    /// router tracks.
    pub fn next_hop(&self, dest: LinkId, from: LinkId) -> Option<LinkId> {
        if from == dest {
            return None;
        }
        let &s = self.slot.get(&dest.0)?;
        let hop = self.next_hop[s].get(from.idx()).copied().flatten();
        if hop.is_some() {
            return hop;
        }
        // Trunk mode: a link outside this destination's field isn't a dead end —
        // it's a local street no through-plan covers. Drive toward the nearest
        // trunk link (the ascent field); once on the trunk, the field takes over.
        // A *trunk* link with no hop is genuinely unreachable, so it still exits.
        match &self.trunk {
            Some(t) if !t.get(from.idx()).copied().unwrap_or(false) => {
                self.ascent_next.get(from.idx()).copied().flatten()
            }
            _ => None,
        }
    }

    /// Travel cost from `from` to `dest` under the field's current costs — used to
    /// rank a wrong-lane car's *reachable* movements by forward progress. `None` if
    /// `dest` isn't tracked or `from` can't reach it.
    pub fn distance(&self, dest: LinkId, from: LinkId) -> Option<u64> {
        let &s = self.slot.get(&dest.0)?;
        match self.dist[s].get(from.idx()).copied() {
            Some(d) if d != u64::MAX => Some(d),
            // Trunk mode: rank uncovered locals by their distance to the trunk,
            // biased so any candidate a real field covers always outranks them —
            // a wrong-lane car outside the field still makes arterial-ward
            // progress instead of treating every option as equally unreachable.
            _ => match self.trunk.as_ref() {
                Some(t) if !t.get(from.idx()).copied().unwrap_or(false) => {
                    match self.ascent_dist.get(from.idx()).copied() {
                        Some(d) if d != u64::MAX => Some(ASCENT_BIAS + d),
                        _ => None,
                    }
                }
                _ => None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::map::{LinkSpec, NodeSpec, OsmMap};

    /// Two parallel arms (via node 2 or node 3) from node 1 to node 4, so a
    /// congested arm can be rerouted around.
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

    fn free_costs(net: &Network) -> Vec<u64> {
        (0..net.links.len() as u32).map(|i| net.link_travel_time_ms(LinkId(i))).collect()
    }

    #[test]
    fn next_hop_agrees_with_dijkstra() {
        let net = diamond();
        let dest = LinkId(5);
        let router = FieldRouter::new(&net, &[dest], &free_costs(&net));
        // Walk the field from link 0 and compare to the direct search.
        let mut path = vec![LinkId(0)];
        while let Some(n) = router.next_hop(dest, *path.last().unwrap()) {
            path.push(n);
        }
        assert_eq!(path, net.route_links(LinkId(0), dest).unwrap());
    }

    #[test]
    fn recompute_reroutes_around_a_costly_link() {
        let net = diamond();
        let dest = LinkId(5);
        let mut cost = free_costs(&net);
        let mut router = FieldRouter::new(&net, &[dest], &cost);
        assert_eq!(router.next_hop(dest, LinkId(0)), Some(LinkId(1)), "baseline uses the short arm");
        // Jam the short arm and rebuild the field: routing swings to the other arm.
        cost[1] = 10_000_000;
        router.recompute(&cost);
        assert_eq!(router.next_hop(dest, LinkId(0)), Some(LinkId(2)), "reroutes onto the free arm");
    }

    #[test]
    fn destination_and_unknown_have_no_next_hop() {
        let net = diamond();
        let dest = LinkId(5);
        let router = FieldRouter::new(&net, &[dest], &free_costs(&net));
        assert_eq!(router.next_hop(dest, dest), None, "already at the destination");
        assert_eq!(router.next_hop(LinkId(3), LinkId(0)), None, "no field for an untracked destination");
        assert!(router.knows(dest) && !router.knows(LinkId(3)));
    }

    #[test]
    fn recompute_from_distances_matches_the_builtin_recompute() {
        // Feeding externally-computed reverse distances (the GPU backend's job)
        // must produce the same next-hop field as the pure CPU recompute.
        let net = diamond();
        let cost = free_costs(&net);
        let reference = FieldRouter::new(&net, &[LinkId(5), LinkId(4)], &cost);

        let mut fed = FieldRouter::new(&net, &[LinkId(5), LinkId(4)], &cost);
        let dists: Vec<Vec<u64>> = fed
            .dests_in_slot_order()
            .to_vec()
            .iter()
            .map(|&d| flowfield::distances_to(fed.adjacency(), d, &cost))
            .collect();
        fed.recompute_from_distances(&cost, &dists);

        for &dest in &[LinkId(5), LinkId(4)] {
            for from in 0..net.links.len() as u32 {
                assert_eq!(
                    fed.next_hop(dest, LinkId(from)),
                    reference.next_hop(dest, LinkId(from)),
                    "next hop to {dest:?} from {from} must match the builtin recompute"
                );
            }
        }
    }

    #[test]
    fn stale_distance_batch_is_ignored_not_applied() {
        // A GPU readback dispatched for a larger destination set can land after the
        // router is reinstalled with fewer destinations (the checkbox-toggle path).
        // Feeding that oversized batch must not panic (index past the fields) nor
        // corrupt the current fields — it is simply dropped.
        let net = diamond();
        let cost = free_costs(&net);
        let mut router = FieldRouter::new(&net, &[LinkId(5)], &cost);
        let before = router.next_hop(LinkId(5), LinkId(0));
        // Two slots' worth of distances fed into a one-slot router.
        let stale = vec![
            flowfield::distances_to(router.adjacency(), LinkId(5), &cost),
            flowfield::distances_to(router.adjacency(), LinkId(4), &cost),
        ];
        router.recompute_from_distances(&cost, &stale); // must not panic
        assert_eq!(router.next_hop(LinkId(5), LinkId(0)), before, "stale batch left the field untouched");
    }

    /// A long corridor where a faster local shortcut parallels the arterial
    /// trunk, and the destination sits on far local fabric — the layout that
    /// separates whole-graph routing (takes the shortcut) from trunk-mode
    /// routing (stays on the arterial; the shortcut is outside both the trunk
    /// and the destination's access neighborhood).
    fn trunk_corridor() -> Network {
        let arterial = |a: i64, b: i64| {
            let mut l = LinkSpec::oneway(a, b, 1, 15.0);
            l.road_class = "primary".into();
            l
        };
        let local = |a: i64, b: i64, v: f64| LinkSpec::oneway(a, b, 1, v);
        OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::uncontrolled(2, 200.0, 0.0),
                NodeSpec::uncontrolled(3, 2200.0, 0.0),
                NodeSpec::uncontrolled(4, 4200.0, 0.0),
                NodeSpec::uncontrolled(5, 4400.0, 0.0),
                NodeSpec::uncontrolled(6, 2200.0, 100.0), // local shortcut midpoint
            ],
            links: vec![
                local(1, 2, 10.0),      // 0: origin access
                arterial(2, 3),         // 1: trunk
                arterial(3, 4),         // 2: trunk
                local(4, 5, 10.0),      // 3: destination link
                local(2, 6, 25.0),      // 4: fast local shortcut (would win on time)
                local(6, 4, 25.0),      // 5
            ],
        }
        .build()
    }

    #[test]
    fn trunk_mode_keeps_through_traffic_on_the_arterial() {
        let net = trunk_corridor();
        let cost = free_costs(&net);
        let dest = LinkId(3);
        let full = FieldRouter::new(&net, &[dest], &cost);
        assert_eq!(full.next_hop(dest, LinkId(0)), Some(LinkId(4)), "whole-graph routing takes the faster local shortcut");

        let trunk: Vec<bool> = net.links.iter().map(|l| !matches!(l.kind, crate::sim::network::RoadKind::Local)).collect();
        let router = FieldRouter::new_with_trunk(&net, &[dest], &cost, Some(trunk));
        // Follow next hops from the origin access link all the way to the
        // destination: the path must ride the trunk, never the shortcut.
        let (mut cur, mut path) = (LinkId(0), vec![LinkId(0)]);
        while cur != dest {
            let hop = router.next_hop(dest, cur).expect("every link still routes somewhere");
            assert!(!path.contains(&hop), "no routing cycle");
            path.push(hop);
            cur = hop;
        }
        assert!(path.contains(&LinkId(1)) && path.contains(&LinkId(2)), "the route rides the arterial: {path:?}");
        assert!(!path.contains(&LinkId(4)), "and skips the local shortcut: {path:?}");
        // A local outside the field (the origin access link, 4 km from the
        // destination) is ranked by arterial-ward progress via the ascent
        // fallback, and always behind anything a real field covers.
        let d_origin = router.distance(dest, LinkId(0)).expect("uncovered local still comparable via ascent");
        let d_trunk = router.distance(dest, LinkId(1)).expect("trunk link in field");
        assert!(d_origin > d_trunk, "field coverage outranks ascent fallback");
    }

    /// Drive a targeted cycle to completion.
    fn run_targeted(r: &mut FieldRouter, cost: &[u64], targets: &HashMap<u32, Vec<u32>>) {
        r.begin_recompute_targeted(cost.to_vec(), 1, targets);
        while r.recompute_pending() {
            r.advance_recompute(64);
        }
    }

    #[test]
    fn targeted_cycle_updates_queried_links_and_merges_the_rest() {
        // Install against free-flow, then congest the lower diamond arm and run a
        // *targeted* refresh for a querying car at the entrance: the car's hop
        // must flip to the clear arm, and every link — settled or merely
        // merge-preserved — must still route to the destination without cycles.
        let net = diamond();
        let free = free_costs(&net);
        let dest = LinkId(5);
        let mut r = FieldRouter::new(&net, &[dest], &free);
        let mut jammed = free.clone();
        let lower = r.next_hop(dest, LinkId(0)).unwrap();
        jammed[lower.idx()] = jammed[lower.idx()] * 40; // the arm it liked is now parked
        let targets = HashMap::from([(dest.0, vec![0u32])]);
        run_targeted(&mut r, &jammed, &targets);
        assert_eq!(r.last_cycle_stats(), (1, 0), "the congested field solved, nothing skipped");
        let hop = r.next_hop(dest, LinkId(0)).unwrap();
        assert_ne!(hop, lower, "the querying link reroutes around the jam");
        for l in 0..net.links.len() as u32 {
            if r.next_hop(dest, LinkId(l)).is_none() && LinkId(l) != dest {
                continue; // genuinely unreachable
            }
            let (mut cur, mut steps) = (LinkId(l), 0);
            while cur != dest {
                cur = r.next_hop(dest, cur).expect("merged field never dead-ends");
                steps += 1;
                assert!(steps < 64, "merged field never cycles (from link {l})");
            }
        }
    }

    #[test]
    fn clean_fields_skip_until_age_forces_a_refresh() {
        let net = diamond();
        let free = free_costs(&net);
        let dest = LinkId(5);
        let mut r = FieldRouter::new(&net, &[dest], &free);
        let targets = HashMap::from([(dest.0, vec![0u32])]);
        // Nothing changed: the field prices its own paths correctly and skips.
        for _ in 0..AGE_FORCE {
            run_targeted(&mut r, &free, &targets);
            assert_eq!(r.last_cycle_stats(), (0, 1), "a clean field skips the cycle");
        }
        // The skips age it out: the next cycle force-solves in full.
        run_targeted(&mut r, &free, &targets);
        assert_eq!(r.last_cycle_stats(), (1, 0), "aging bounds staleness with a forced full solve");
        // And a real cost change is dirty immediately, no aging needed.
        let mut jammed = free.clone();
        let liked = r.next_hop(dest, LinkId(0)).unwrap();
        jammed[liked.idx()] *= 40;
        run_targeted(&mut r, &jammed, &targets);
        assert_eq!(r.last_cycle_stats(), (1, 0), "a repriced route is dirty at once");
    }

    #[test]
    fn duplicate_destinations_share_one_field() {
        let net = diamond();
        let router = FieldRouter::new(&net, &[LinkId(5), LinkId(5), LinkId(4)], &free_costs(&net));
        assert_eq!(router.destinations().len(), 2, "duplicates collapse to one slot");
    }
}
