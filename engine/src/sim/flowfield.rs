//! Flow-field (vector-field) routing over the link graph.
//!
//! Instead of a per-vehicle Dijkstra, we precompute, for a destination, a
//! shortest-path *tree*: `next_hop[link]` = the outgoing link to take from each
//! link to reach the destination fastest. A vehicle then routes by an O(1)
//! lookup per intersection — what scales to 1M+. The tree is computed by
//! **parallel Bellman–Ford** (Jacobi relaxation: read committed distances, write
//! new ones), which maps directly onto a GPU compute kernel (see `flowfield.wgsl`
//! / `flowfield_gpu.rs`); this is the CPU reference the GPU is validated against.
//!
//! Edge weights are `cost[link]` = the traversal cost (ms) of *entering* a link,
//! so feeding live travel times makes routing congestion-reactive for free.

use super::network::{LinkId, Network};

pub const UNREACHABLE: u64 = u64::MAX;

/// Precomputed outgoing adjacency (link → its onward links), so relaxation
/// doesn't rebuild it every pass.
pub fn adjacency(net: &Network) -> Vec<Vec<u32>> {
    (0..net.links.len() as u32)
        .map(|i| net.outgoing_links(LinkId(i)).into_iter().map(|l| l.0).collect())
        .collect()
}

/// Flatten adjacency to CSR (`offsets` of length n+1, `targets` concatenated) —
/// the layout the GPU kernel indexes.
pub fn csr(adj: &[Vec<u32>]) -> (Vec<u32>, Vec<u32>) {
    let mut offsets = Vec::with_capacity(adj.len() + 1);
    let mut targets = Vec::new();
    offsets.push(0u32);
    for outs in adj {
        targets.extend_from_slice(outs);
        offsets.push(targets.len() as u32);
    }
    (offsets, targets)
}

/// Reverse shortest-path distances (ms) from every link to `dest` under
/// `cost`, by Bellman–Ford Jacobi relaxation to convergence. `UNREACHABLE` where
/// no path exists.
pub fn distances_to(adj: &[Vec<u32>], dest: LinkId, cost: &[u64]) -> Vec<u64> {
    let n = adj.len();
    let mut dist = vec![UNREACHABLE; n];
    dist[dest.idx()] = 0;
    for _ in 0..=n {
        let mut changed = false;
        let mut next = dist.clone();
        for (a, outs) in adj.iter().enumerate() {
            if a == dest.idx() {
                continue;
            }
            let mut best = UNREACHABLE;
            for &b in outs {
                let db = dist[b as usize];
                if db != UNREACHABLE {
                    best = best.min(cost[b as usize].saturating_add(db));
                }
            }
            if best < next[a] {
                next[a] = best;
                changed = true;
            }
        }
        dist = next;
        if !changed {
            break;
        }
    }
    dist
}

/// The next link to take from each link toward the destination whose distances
/// these are (`None` if the destination is unreachable from that link).
pub fn next_hops(adj: &[Vec<u32>], dist: &[u64], cost: &[u64]) -> Vec<Option<LinkId>> {
    adj.iter()
        .map(|outs| {
            outs.iter()
                .copied()
                .filter(|&b| dist[b as usize] != UNREACHABLE)
                .min_by_key(|&b| cost[b as usize].saturating_add(dist[b as usize]))
                .map(LinkId)
        })
        .collect()
}

/// Follow the next-hop field from `from` to `to` (inclusive), for validating the
/// field against a direct shortest-path search.
pub fn route_via_field(next_hop: &[Option<LinkId>], from: LinkId, to: LinkId) -> Option<Vec<LinkId>> {
    let mut path = vec![from];
    let mut cur = from;
    while cur != to {
        cur = next_hop[cur.idx()]?;
        if path.contains(&cur) {
            return None; // guard against a cycle
        }
        path.push(cur);
    }
    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::map::{LinkSpec, NodeSpec, OsmMap};

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

    fn costs(net: &Network) -> Vec<u64> {
        (0..net.links.len() as u32).map(|i| net.link_travel_time_ms(LinkId(i))).collect()
    }

    #[test]
    fn field_route_matches_dijkstra() {
        let net = diamond();
        let adj = adjacency(&net);
        let cost = costs(&net);
        let dest = LinkId(5);
        let dist = distances_to(&adj, dest, &cost);
        let next = next_hops(&adj, &dist, &cost);
        let via_field = route_via_field(&next, LinkId(0), dest).unwrap();
        let dijkstra = net.route_links(LinkId(0), dest).unwrap();
        assert_eq!(via_field, dijkstra, "flow field must agree with Dijkstra");
    }

    #[test]
    fn reroutes_when_a_link_is_expensive() {
        let net = diamond();
        let adj = adjacency(&net);
        let mut cost = costs(&net);
        let dest = LinkId(5);
        // Baseline goes via link 1 (the short arm).
        let base = next_hops(&adj, &distances_to(&adj, dest, &cost), &cost);
        assert_eq!(route_via_field(&base, LinkId(0), dest).unwrap()[1], LinkId(1));
        // Make link 1 hugely costly → the field steers around it via link 2.
        cost[1] = 10_000_000;
        let rerouted = next_hops(&adj, &distances_to(&adj, dest, &cost), &cost);
        assert_eq!(route_via_field(&rerouted, LinkId(0), dest).unwrap()[1], LinkId(2));
    }

    #[test]
    fn csr_matches_adjacency() {
        let net = diamond();
        let adj = adjacency(&net);
        let (offsets, targets) = csr(&adj);
        assert_eq!(offsets.len(), adj.len() + 1);
        assert_eq!(targets.len(), adj.iter().map(|o| o.len()).sum::<usize>());
        for (a, outs) in adj.iter().enumerate() {
            assert_eq!(&targets[offsets[a] as usize..offsets[a + 1] as usize], outs.as_slice());
        }
    }

    #[test]
    fn wgsl_kernel_parses_and_validates() {
        let src = include_str!("flowfield.wgsl");
        let module = naga::front::wgsl::parse_str(src).expect("flowfield.wgsl should parse");
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .expect("flowfield.wgsl should type-check");
    }

    #[test]
    fn unreachable_destination_has_no_next_hop() {
        // Two disconnected one-way links: 0->1 and 2->3.
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(0, 0.0, 0.0),
                NodeSpec::uncontrolled(1, 100.0, 0.0),
                NodeSpec::uncontrolled(2, 0.0, 500.0),
                NodeSpec::uncontrolled(3, 100.0, 500.0),
            ],
            links: vec![LinkSpec::oneway(0, 1, 1, 20.0), LinkSpec::oneway(2, 3, 1, 20.0)],
        }
        .build();
        let adj = adjacency(&net);
        let cost = costs(&net);
        // Links: 0 = (0->1), 1 = (2->3). Route everything to link 1.
        let dist = distances_to(&adj, LinkId(1), &cost);
        assert_eq!(dist[LinkId(1).idx()], 0, "destination distance is zero");
        assert_eq!(dist[LinkId(0).idx()], UNREACHABLE, "disconnected link can't reach it");
    }
}
