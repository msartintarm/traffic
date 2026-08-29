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

/// Reverse shortest-path distances (ms) from every link to `dest` under `cost`
/// (`dist[a] = min over outgoing b of cost[b] + dist[b]`). `UNREACHABLE` where no
/// path exists. Computed by a reverse Dijkstra — identical to the Bellman–Ford
/// relaxation the GPU runs, but O(E log V) rather than O(V·E), so it stays fast on a
/// whole-city graph. Builds the predecessor lists itself; use [`distances_to_with`]
/// with a shared reverse index when computing many fields over one graph.
pub fn distances_to(adj: &[Vec<u32>], dest: LinkId, cost: &[u64]) -> Vec<u64> {
    distances_to_with(&reverse(adj), dest, cost)
}

/// Predecessor lists (`pred[b]` = links that have `b` as an outgoing link) — the
/// reverse graph a field's Dijkstra pulls along. Build once, reuse per destination.
pub fn reverse(adj: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let mut pred: Vec<Vec<u32>> = vec![Vec::new(); adj.len()];
    for (a, outs) in adj.iter().enumerate() {
        for &b in outs {
            pred[b as usize].push(a as u32);
        }
    }
    pred
}

/// [`distances_to`] with a prebuilt reverse index (see [`reverse`]).
pub fn distances_to_with(pred: &[Vec<u32>], dest: LinkId, cost: &[u64]) -> Vec<u64> {
    let mut field = PartialField::new(pred.len(), dest);
    field.advance(pred, cost, usize::MAX); // run to completion
    field.dist
}

/// A resumable reverse-Dijkstra toward one destination, so a whole-map field solve can be
/// **spread across frames** — advance a bounded number of settled links per call — instead of
/// blocking one frame for the full O(links log links) sweep (~40 ms in wasm on a city map).
/// Popped (settled) distances are final by the Dijkstra invariant, but a partial field is not
/// yet a valid routing field, so callers keep the previous complete field live and swap the
/// finished distances in only once [`advance`](Self::advance) reports completion.
pub struct PartialField {
    dist: Vec<u64>,
    heap: std::collections::BinaryHeap<std::cmp::Reverse<(u64, u32)>>,
    /// Which links this solve has *settled* (popped with a final distance) — the
    /// region a targeted solve is exact over, and the merge boundary when a
    /// partial result is published over a previous field.
    settled: Vec<bool>,
    /// Early-termination targets: the solve is complete once every target is
    /// settled (Dijkstra settles in distance order, so everything a querying
    /// car can read is final by then). `None` = run to exhaustion.
    targets: Option<Vec<bool>>,
    targets_left: usize,
}

impl PartialField {
    pub fn new(n: usize, dest: LinkId) -> Self {
        Self::new_multi(n, std::iter::once(dest.0))
    }

    /// A field seeded at several zero-cost sources at once — the "distance to the
    /// nearest of these" solve the arterial-ascent field uses (every trunk link
    /// is a source).
    pub fn new_multi(n: usize, seeds: impl Iterator<Item = u32>) -> Self {
        let mut dist = vec![UNREACHABLE; n];
        let mut heap = std::collections::BinaryHeap::new();
        for s in seeds {
            dist[s as usize] = 0;
            heap.push(std::cmp::Reverse((0u64, s)));
        }
        Self { dist, heap, settled: vec![false; n], targets: None, targets_left: 0 }
    }

    /// Restrict this solve to early-terminate once every link in `targets` is
    /// settled — the links that will actually query the field this cycle.
    pub fn with_targets(mut self, targets: &[u32]) -> Self {
        let mut set = vec![false; self.dist.len()];
        let mut left = 0;
        for &t in targets {
            if !set[t as usize] {
                set[t as usize] = true;
                left += 1;
            }
        }
        self.targets = Some(set);
        self.targets_left = left;
        self
    }

    pub fn settled(&self) -> &[bool] {
        &self.settled
    }

    /// Settle up to `budget` links; returns `true` once the field is complete (heap drained).
    pub fn advance(&mut self, pred: &[Vec<u32>], cost: &[u64], budget: usize) -> bool {
        self.advance_masked(pred, cost, budget, None)
    }

    /// [`advance`](Self::advance), restricted: with a mask, relaxation never
    /// enters a link outside it — the field is solved over a subgraph (arterial
    /// trunk + a destination's access neighborhood) and every excluded link
    /// simply stays `UNREACHABLE`, which the router answers with its ascent
    /// fallback. This is what makes an arterial-mode field ~a third of the work
    /// of a whole-graph one.
    pub fn advance_masked(&mut self, pred: &[Vec<u32>], cost: &[u64], budget: usize, allowed: Option<&[bool]>) -> bool {
        let mut settled = 0usize;
        while settled < budget {
            let Some(std::cmp::Reverse((d, b))) = self.heap.pop() else {
                return true;
            };
            if d > self.dist[b as usize] {
                continue; // a stale, superseded entry — not a settle
            }
            settled += 1;
            self.settled[b as usize] = true;
            if let Some(ts) = &self.targets {
                if ts[b as usize] {
                    self.targets_left -= 1;
                    if self.targets_left == 0 {
                        return true; // every querying link is final — the rest is unread this cycle
                    }
                }
            }
            let step = cost[b as usize].saturating_add(d);
            for &a in &pred[b as usize] {
                if allowed.is_some_and(|m| !m[a as usize]) {
                    continue;
                }
                if step < self.dist[a as usize] {
                    self.dist[a as usize] = step;
                    self.heap.push(std::cmp::Reverse((step, a)));
                }
            }
        }
        self.heap.is_empty() // budget spent; done iff nothing is left to settle
    }

    /// The links within `cap` cost of `dest` on the *unrestricted* graph — a
    /// destination's local access neighborhood, merged into the trunk mask so a
    /// trip can descend from the arterials to its actual endpoint.
    pub fn neighborhood(pred: &[Vec<u32>], dest: LinkId, cost: &[u64], cap: u64) -> Vec<u32> {
        let mut f = Self::new(pred.len(), dest);
        let mut out = Vec::new();
        while let Some(std::cmp::Reverse((d, b))) = f.heap.pop() {
            if d > f.dist[b as usize] {
                continue;
            }
            if d > cap {
                break;
            }
            out.push(b);
            let step = cost[b as usize].saturating_add(d);
            for &a in &pred[b as usize] {
                if step < f.dist[a as usize] {
                    f.dist[a as usize] = step;
                    f.heap.push(std::cmp::Reverse((step, a)));
                }
            }
        }
        out
    }

    pub fn dist(&self) -> &[u64] {
        &self.dist
    }

    /// Move the settled distances out (leaving the field empty), to publish into a live buffer.
    pub fn take_dist(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.dist)
    }

    /// Move both the distances and the settled map out, for a merge-publish.
    pub fn take_parts(&mut self) -> (Vec<u64>, Vec<bool>) {
        (std::mem::take(&mut self.dist), std::mem::take(&mut self.settled))
    }
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
        for src in [include_str!("flowfield.wgsl"), include_str!("flowfield_batch.wgsl")] {
            let module = naga::front::wgsl::parse_str(src).expect("flow-field kernel should parse");
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .expect("flow-field kernel should type-check");
        }
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
