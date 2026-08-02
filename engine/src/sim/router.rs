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
    /// Next slot for [`recompute_incremental`] to refresh — spreads the rebuild.
    cursor: usize,
}

impl FieldRouter {
    /// Build a router for `dests` (duplicates ignored), with every field
    /// initialised against `cost`.
    pub fn new(net: &Network, dests: &[LinkId], cost: &[u64]) -> Self {
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
        let mut router = Self { adj, pred, dests: unique, slot, next_hop, dist, cursor: 0 };
        router.recompute(cost);
        router
    }

    /// Number of distinct destination fields maintained.
    pub fn destination_count(&self) -> usize {
        self.dests.len()
    }

    /// Refresh only `count` destination fields this call, cycling through them, so
    /// the full rebuild is spread across many ticks instead of a single spike (a
    /// ~half-second freeze at city scale). Over `dests.len()/count` calls every
    /// field is refreshed against the latest `cost`.
    pub fn recompute_incremental(&mut self, cost: &[u64], count: usize) {
        let total = self.dests.len();
        if total == 0 {
            return;
        }
        let count = count.min(total);
        let slots: Vec<usize> = (0..count).map(|k| (self.cursor + k) % total).collect();
        let (pred, adj, dests) = (&self.pred, &self.adj, &self.dests);
        let field = |&s: &usize| {
            let dist = flowfield::distances_to_with(pred, dests[s], cost);
            (s, flowfield::next_hops(adj, &dist, cost), dist)
        };
        #[cfg(feature = "parallel")]
        let fields: Vec<_> = {
            use rayon::prelude::*;
            slots.par_iter().map(field).collect()
        };
        #[cfg(not(feature = "parallel"))]
        let fields: Vec<_> = slots.iter().map(field).collect();
        for (s, next_hop, dist) in fields {
            self.next_hop[s] = next_hop;
            self.dist[s] = dist;
        }
        self.cursor = (self.cursor + count) % total;
    }

    /// The destinations this router routes to.
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
        let (pred, adj) = (&self.pred, &self.adj);
        let field = |&dest: &LinkId| {
            let dist = flowfield::distances_to_with(pred, dest, cost);
            let next_hop = flowfield::next_hops(adj, &dist, cost);
            (next_hop, dist)
        };
        #[cfg(feature = "parallel")]
        let fields: Vec<_> = {
            use rayon::prelude::*;
            self.dests.par_iter().map(field).collect()
        };
        #[cfg(not(feature = "parallel"))]
        let fields: Vec<_> = self.dests.iter().map(field).collect();
        for (s, (next_hop, dist)) in fields.into_iter().enumerate() {
            self.next_hop[s] = next_hop;
            self.dist[s] = dist;
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
        self.next_hop[s].get(from.idx()).copied().flatten()
    }

    /// Travel cost from `from` to `dest` under the field's current costs — used to
    /// rank a wrong-lane car's *reachable* movements by forward progress. `None` if
    /// `dest` isn't tracked or `from` can't reach it.
    pub fn distance(&self, dest: LinkId, from: LinkId) -> Option<u64> {
        let &s = self.slot.get(&dest.0)?;
        match self.dist[s].get(from.idx()).copied() {
            Some(d) if d != u64::MAX => Some(d),
            _ => None,
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

    #[test]
    fn duplicate_destinations_share_one_field() {
        let net = diamond();
        let router = FieldRouter::new(&net, &[LinkId(5), LinkId(5), LinkId(4)], &free_costs(&net));
        assert_eq!(router.destinations().len(), 2, "duplicates collapse to one slot");
    }
}
