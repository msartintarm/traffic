//! Flow-field routing over a fixed set of destinations.
//!
//! For each destination in use we keep a next-hop field: `next_hop[dest][link]`
//! is the outgoing link to take from `link` to reach `dest` fastest. A vehicle
//! then routes by an O(1) lookup per intersection ([`FieldRouter::next_hop`])
//! instead of a per-vehicle path search — the layout that scales to 1M+.
//!
//! Because the fields are rebuilt from **live** per-link travel times
//! ([`FieldRouter::recompute`]), in-flight vehicles pick fresh next-hops as jams
//! form, so rerouting is automatic and shared across every car heading to the
//! same place. The fields are computed with [`flowfield`] — the Bellman–Ford
//! Jacobi reference the GPU `flowfield.wgsl` kernel mirrors — so the recompute is
//! GPU-swappable wholesale at scale.

use std::collections::HashMap;

use super::flowfield;
use super::network::{LinkId, Network};

pub struct FieldRouter {
    adj: Vec<Vec<u32>>,
    dests: Vec<LinkId>,
    slot: HashMap<u32, usize>,
    /// `next_hop[slot][from_link]` toward `dests[slot]` (`None` = unreachable).
    next_hop: Vec<Vec<Option<LinkId>>>,
}

impl FieldRouter {
    /// Build a router for `dests` (duplicates ignored), with every field
    /// initialised against `cost`.
    pub fn new(net: &Network, dests: &[LinkId], cost: &[u64]) -> Self {
        let adj = flowfield::adjacency(net);
        let mut unique = Vec::new();
        let mut slot = HashMap::new();
        for &d in dests {
            slot.entry(d.0).or_insert_with(|| {
                unique.push(d);
                unique.len() - 1
            });
        }
        let next_hop = vec![Vec::new(); unique.len()];
        let mut router = Self { adj, dests: unique, slot, next_hop };
        router.recompute(cost);
        router
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
    /// Call periodically so routing tracks live congestion.
    pub fn recompute(&mut self, cost: &[u64]) {
        for (s, &dest) in self.dests.iter().enumerate() {
            let dist = flowfield::distances_to(&self.adj, dest, cost);
            self.next_hop[s] = flowfield::next_hops(&self.adj, &dist, cost);
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
        for (s, dist) in dist_per_slot.iter().enumerate() {
            self.next_hop[s] = flowfield::next_hops(&self.adj, dist, cost);
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
    fn duplicate_destinations_share_one_field() {
        let net = diamond();
        let router = FieldRouter::new(&net, &[LinkId(5), LinkId(5), LinkId(4)], &free_costs(&net));
        assert_eq!(router.destinations().len(), 2, "duplicates collapse to one slot");
    }
}
