//! Boundary (gateway) classification: which links let traffic enter or leave the
//! simulated area. OSM data clipped to a bounding box leaves dead-end nodes where
//! roads were cut at the edge; a node with a single distinct neighbour is such a
//! **gateway** (one-way stubs have one incident link, two-way roads two
//! anti-parallel ones — both connect to the same neighbour). Traffic *entering*
//! the map originates on a link leaving a gateway; traffic *leaving* ends on a
//! link arriving at one; everything else is interior.
//!
//! Pure topology over `&Network`, so it is unit-tested standalone and reused by
//! the demand generator to build boundary-aware origin/destination streams.

use super::network::{LinkId, Network, NodeId};

/// Distinct neighbour nodes one link from `node` (either direction); size 1 for
/// a map-edge dead-end.
fn neighbours(net: &Network, node: NodeId) -> Vec<NodeId> {
    let mut out = Vec::new();
    for link in &net.links {
        let other = if link.from == node {
            Some(link.to)
        } else if link.to == node {
            Some(link.from)
        } else {
            None
        };
        if let Some(n) = other {
            if !out.contains(&n) {
                out.push(n);
            }
        }
    }
    out
}

/// Whether `node` is a gateway: a map-edge dead-end with exactly one distinct
/// neighbour, where clipped roads enter/leave the simulated area.
pub fn is_gateway(net: &Network, node: NodeId) -> bool {
    neighbours(net, node).len() == 1
}

/// Gateway flag per node — a gateway has exactly one distinct neighbour (a stub where a road
/// leaves the clipped map). Computed in one O(links) pass, so the list filters below are
/// O(links) rather than O(links²): a per-node `is_gateway` re-scans every link, which on a
/// whole-city map (152k links) was ~50 s — the dominant Columbus demand-build cost in wasm.
fn gateway_mask(net: &Network) -> Vec<bool> {
    let mut nbrs: Vec<Vec<u32>> = vec![Vec::new(); net.nodes.len()];
    for link in &net.links {
        nbrs[link.from.idx()].push(link.to.0);
        nbrs[link.to.idx()].push(link.from.0);
    }
    nbrs.iter_mut()
        .map(|v| {
            v.sort_unstable();
            v.dedup();
            v.len() == 1
        })
        .collect()
}

/// Every gateway node in the network.
pub fn gateways(net: &Network) -> Vec<NodeId> {
    let g = gateway_mask(net);
    (0..net.nodes.len() as u32).map(NodeId).filter(|&n| g[n.idx()]).collect()
}

/// Links leaving a gateway (`from` is a gateway): where external traffic *enters*
/// the map.
pub fn entry_links(net: &Network) -> Vec<LinkId> {
    let g = gateway_mask(net);
    (0..net.links.len() as u32).map(LinkId).filter(|&l| g[net.link(l).from.idx()]).collect()
}

/// Links arriving at a gateway (`to` is a gateway): where traffic *leaves* the
/// map.
pub fn exit_links(net: &Network) -> Vec<LinkId> {
    let g = gateway_mask(net);
    (0..net.links.len() as u32).map(LinkId).filter(|&l| g[net.link(l).to.idx()]).collect()
}

/// Links that neither leave nor arrive at a gateway — the genuinely internal
/// segments, the candidate interior origins/destinations.
pub fn interior_links(net: &Network) -> Vec<LinkId> {
    let g = gateway_mask(net);
    (0..net.links.len() as u32)
        .map(LinkId)
        .filter(|&l| !g[net.link(l).from.idx()] && !g[net.link(l).to.idx()])
        .collect()
}

/// Speed (m/s) at or above which a road counts as a grade-separated freeway.
/// Around Millbrae US-101 and I-280 run ~29 m/s (65 mph) while arterials top out
/// near 18; anything faster than this threshold is one of the freeways.
pub const HIGHWAY_SPEED: f64 = 22.0;

/// Whether `link` is a freeway/highway (by its posted speed).
pub fn is_highway_link(net: &Network, link: LinkId) -> bool {
    net.lane(net.link(link).lane_start).speed_limit >= HIGHWAY_SPEED
}

/// Entry links (leaving a gateway) that are freeways — where highway traffic
/// enters the map (the US-101 / I-280 on-ramps at the map edge).
pub fn highway_entry_links(net: &Network) -> Vec<LinkId> {
    entry_links(net).into_iter().filter(|&l| is_highway_link(net, l)).collect()
}

/// Exit links (arriving at a gateway) that are freeways — where traffic leaves
/// the map onto a highway.
pub fn highway_exit_links(net: &Network) -> Vec<LinkId> {
    exit_links(net).into_iter().filter(|&l| is_highway_link(net, l)).collect()
}

/// Entry links that are ordinary surface streets (not freeways) — where local /
/// arterial traffic enters the map.
pub fn surface_entry_links(net: &Network) -> Vec<LinkId> {
    entry_links(net).into_iter().filter(|&l| !is_highway_link(net, l)).collect()
}

/// Exit links that are ordinary surface streets (not freeways).
pub fn surface_exit_links(net: &Network) -> Vec<LinkId> {
    exit_links(net).into_iter().filter(|&l| !is_highway_link(net, l)).collect()
}

/// Interior links that are ordinary surface streets — the "non-major road" trip
/// ends a highway traveller may leave the freeway for (never a mid-freeway segment).
pub fn surface_interior_links(net: &Network) -> Vec<LinkId> {
    interior_links(net).into_iter().filter(|&l| !is_highway_link(net, l)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::map::{LinkSpec, NodeSpec, OsmMap};

    /// A one-way corridor 1 → 2 → 3: nodes 1 and 3 are dead-ends (gateways),
    /// node 2 is interior.
    fn corridor() -> Network {
        OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::uncontrolled(2, 300.0, 0.0),
                NodeSpec::uncontrolled(3, 600.0, 0.0),
            ],
            links: vec![LinkSpec::oneway(1, 2, 1, 20.0), LinkSpec::oneway(2, 3, 1, 20.0)],
        }
        .build()
    }

    #[test]
    fn dead_end_nodes_are_gateways() {
        let net = corridor();
        let g = gateways(&net);
        assert_eq!(g.len(), 2, "both ends of the corridor are gateways, the middle is not");
        assert!(g.contains(&NodeId(0)) && g.contains(&NodeId(2)));
        assert!(!g.contains(&NodeId(1)), "the interior junction is not a gateway");
    }

    #[test]
    fn entry_and_exit_links_are_the_boundary_links() {
        let net = corridor();
        assert_eq!(entry_links(&net), vec![LinkId(0)], "link leaving the start gateway is an entry");
        assert_eq!(exit_links(&net), vec![LinkId(1)], "link into the end gateway is an exit");
        assert!(interior_links(&net).is_empty(), "a 2-link corridor has no interior link");
    }

    #[test]
    fn freeway_gateways_are_classified_by_speed() {
        // A fast link into the map (freeway on-ramp) and a slow surface street, both
        // cut at the edge. Only the fast one is a highway entry.
        let net = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, -400.0, 0.0),
                NodeSpec::uncontrolled(2, 0.0, 0.0),
                NodeSpec::uncontrolled(3, 0.0, 300.0),
            ],
            links: vec![
                LinkSpec::oneway(1, 2, 3, 29.0), // freeway entry
                LinkSpec::oneway(2, 3, 1, 13.0), // surface exit
            ],
        }
        .build();
        assert!(is_highway_link(&net, LinkId(0)) && !is_highway_link(&net, LinkId(1)));
        assert_eq!(highway_entry_links(&net), vec![LinkId(0)], "only the freeway is a highway entry");
        assert!(highway_exit_links(&net).is_empty(), "the surface exit is not a highway exit");
    }

    #[test]
    fn two_way_dead_end_is_still_one_gateway() {
        // A two-way road cut at the edge: node 1 has two anti-parallel links to
        // node 2, but only one distinct neighbour, so it is a single gateway.
        let net = OsmMap {
            nodes: vec![NodeSpec::uncontrolled(1, 0.0, 0.0), NodeSpec::uncontrolled(2, 300.0, 0.0)],
            links: vec![LinkSpec::oneway(1, 2, 1, 20.0), LinkSpec::oneway(2, 1, 1, 20.0)],
        }
        .build();
        assert!(is_gateway(&net, NodeId(0)) && is_gateway(&net, NodeId(1)));
        // Each node is both an entry (a link leaves it) and an exit (a link enters it).
        assert_eq!(entry_links(&net).len(), 2);
        assert_eq!(exit_links(&net).len(), 2);
    }
}
