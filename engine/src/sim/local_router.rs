//! Per-driver local routing (an O(1)-in-map-size alternative to the flow-field).
//!
//! The flow-field router precomputes, per destination, a shortest-path distance
//! field over the whole graph — O(links × destinations) to maintain, so a bigger
//! map costs more every reroute cycle even though a driver only ever needs the
//! *next turn or two*. This router instead answers `next_hop(dest, from)` by a
//! bounded local decision at the current node: of the links leaving it, pick the
//! one whose estimated remaining distance to the destination is smallest.
//!
//! The estimate is the **ALT** lower bound (A*, Landmarks, Triangle inequality):
//! at load we run one reverse-Dijkstra from each of `K` landmark links, giving
//! `dist(link → landmark)` for every link. For any pair the triangle inequality
//! then gives a tight lower bound `dist(from → dest) ≥ maxₗ dist(from→l) −
//! dist(dest→l)` with no per-destination field. Precompute is O(K · links) ONCE
//! (K a small constant, ~16), not per tick and not per destination; each
//! `next_hop` is O(out-degree · K) ≈ O(1), independent of map size. The
//! bounded ALT-guided search (`SEARCH_CAP`) with live congestion penalties is
//! stateless and re-queried from wherever the car is, so it self-heals against
//! lane-level divergence the way the flow-field does. This is the browser
//! default (`set_local_routing(true)` in the bridge); the engine default stays
//! the flow-field for tests and native baselines.

use super::network::{LinkId, Network};

/// Per-thread reusable A* scratch, so `next_hop` allocates nothing per call
/// (the bounded search is on the hot path once per link entry).
struct Scratch {
    g: super::hash::IntMap<u64>,
    first: super::hash::IntMap<u32>,
    heap: std::collections::BinaryHeap<std::cmp::Reverse<(u64, u32)>>,
}
/// Cap on the per-thread next-hop memo before a full clear (entries are
/// (dest,from)→next; bounded by traffic in practice, this only guards a runaway).
const MEMO_CAP: usize = 1 << 20;

/// Cached hops expire in rotating cohorts as the congestion generation
/// advances: each bump re-searches 1/`MEMO_SPREAD` of the cache instead of
/// flushing it wholesale (which stalls a tick on thousands of simultaneous ALT
/// searches at scale). A new jam reaches every driver within `MEMO_SPREAD`
/// reroute intervals — staggered discovery, the way real drivers learn of a
/// backup.
const MEMO_SPREAD: u64 = 4;

struct Memo {
    /// `(dest, from)` key → (hop, generation at compute); `u32::MAX` = no hop.
    map: std::collections::HashMap<u64, (u32, u64), std::hash::BuildHasherDefault<super::hash::FxHasher>>,
}

thread_local! {
    static MEMO: std::cell::RefCell<Memo> =
        std::cell::RefCell::new(Memo { map: std::collections::HashMap::default() });
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch {
        g: super::hash::IntMap::default(),
        first: super::hash::IntMap::default(),
        heap: std::collections::BinaryHeap::new(),
    });
}

pub struct LocalRouter {
    /// Links reachable in one hop from each link (link-level successors).
    adj: Vec<Vec<u32>>,
    /// Free-flow travel time per link (ms) — the edge weight.
    cost: Vec<u32>,
    /// `dist[l][link]` = free-flow distance (ms) from `link` to landmark `l`;
    /// `u32::MAX` if the landmark is unreachable from it.
    dist: Vec<Vec<u32>>,
    /// Link midpoints, for the geographic tiebreak when landmarks are silent.
    pos: Vec<[f32; 2]>,
    /// Live congestion addend per link (ms), refreshed on the reroute interval
    /// from occupancy so the bounded search steers around jams (as the flow-field
    /// does), updated in O(occupied) ≈ O(drivers), not O(map).
    penalty: Vec<u32>,
    /// Links currently penalized — for an O(occupied) reset each refresh.
    jammed: Vec<u32>,
    /// Jam capacity (vehicles at jam density) per link.
    jam: Vec<f64>,
    /// Bumped when `penalty` changes; drives the memo's rotating cohort
    /// expiry (see [`MEMO_SPREAD`] — staleness is bounded, not zero).
    generation: u64,
}

/// How many landmarks. More = tighter bounds (fewer detours) at linear precompute
/// and memory cost; 16 is the usual sweet spot on road networks.
const LANDMARKS: usize = 16;

/// How many links a driver's turn decision may explore — the "bounded
/// neighbourhood" that makes routing O(1) in map size. Large enough that a
/// destination a few intersections away is found exactly; a farther one is
/// approached via the ALT heuristic and refined at the next intersection.
const SEARCH_CAP: usize = 64;

impl LocalRouter {
    pub fn build(net: &Network) -> Self {
        Self::build_with_progress(net, &mut |_, _, _| {})
    }

    /// [`build`] reporting each landmark's whole-graph Dijkstra as it lands —
    /// the longest stretch of a big-map load, and the one that reads as "stuck"
    /// without a counter.
    pub fn build_with_progress(net: &Network, cb: &mut dyn FnMut(&str, u32, u32)) -> Self {
        let n = net.links.len();
        let adj = super::flowfield::adjacency(net);
        let cost: Vec<u32> = (0..n as u32).map(|i| net.link_travel_time_ms(LinkId(i)) as u32).collect();
        let pos: Vec<[f32; 2]> = (0..n as u32)
            .map(|i| {
                let l = net.link(LinkId(i));
                let (a, b) = (net.node(l.from).position, net.node(l.to).position);
                [((a[0] + b[0]) * 0.5) as f32, ((a[1] + b[1]) * 0.5) as f32]
            })
            .collect();
        let pred = super::flowfield::reverse(&adj);
        let landmarks = pick_landmarks(&pos, LANDMARKS);
        // dist(link → landmark) for each landmark = reverse-Dijkstra toward it.
        let cost64 = to_u64(&cost);
        let dist: Vec<Vec<u32>> = landmarks
            .iter()
            .enumerate()
            .map(|(k, &lm)| {
                cb("landmarks", k as u32, landmarks.len() as u32);
                super::flowfield::distances_to_with(&pred, LinkId(lm), &cost64).iter().map(clamp_u32).collect()
            })
            .collect();
        let jam: Vec<f64> = (0..n as u32)
            .map(|i| {
                let l = net.link(LinkId(i));
                (net.lane(l.lane_start).length / 7.0 * l.lane_count as f64).max(1.0)
            })
            .collect();
        Self { adj, cost, dist, pos, penalty: vec![0; n], jammed: Vec::new(), jam, generation: 0 }
    }

    /// Refresh congestion penalties from per-link vehicle counts (`counts`: link
    /// id → vehicles; occupied links only). O(occupied) reset+set — no map sweep.
    /// The generation (which rotates the memo's expiry cohorts — see
    /// [`MEMO_SPREAD`]) bumps only when the banded penalty set actually moved,
    /// so a quiet map keeps every cached route indefinitely.
    pub fn update_congestion(&mut self, counts: &super::hash::IntMap<u32>) {
        let mut old: Vec<(u32, u32)> =
            self.jammed.iter().map(|&l| (l, self.penalty[l as usize])).filter(|&(_, p)| p != 0).collect();
        let mut next: Vec<(u32, u32)> = counts
            .iter()
            .filter_map(|(&link, &c)| {
                let l = link as usize;
                let band = ((c as f64 / self.jam[l]).min(3.0) * 4.0) as u32; // quarter-jam bands
                let pen = ((self.cost[l] as u64 * band as u64) / 3).min(u32::MAX as u64) as u32;
                (band > 0 && pen > 0).then_some((link, pen))
            })
            .collect();
        old.sort_unstable();
        next.sort_unstable();
        if old == next {
            return; // same jams, same prices — cached routes all still hold
        }
        for &l in &self.jammed {
            self.penalty[l as usize] = 0;
        }
        self.jammed.clear();
        for &(link, pen) in &next {
            self.penalty[link as usize] = pen;
            self.jammed.push(link);
        }
        self.generation = self.generation.wrapping_add(1);
    }

    /// The next link to take from `from` toward `dest`, decided by a driver who
    /// searches only a *bounded neighbourhood* — an ALT-guided A* capped at
    /// `SEARCH_CAP` settled links. If `dest` falls inside that horizon the first
    /// hop is exact; if it is farther, the driver heads toward the searched link
    /// closest to the destination (by the landmark bound) and refines at the
    /// next intersection. Cost is O(SEARCH_CAP · log) — independent of total map
    /// size — and it never loops (A* does not re-settle a link). `None` = arrived
    /// or dead end. This is the O(1)-in-map replacement for the flow-field's
    /// global next-hop, meant to be recomputed once per link entry (not per tick).
    ///
    /// Memoized per thread: local routing uses static free-flow costs, so
    /// `next_hop(dest, from)` is a pure deterministic function — the first query
    /// for a pair runs the bounded search, every later one (from any car, any
    /// caller — lane positioning, the crossing next-hop gate) is an O(1) lookup.
    /// This is what makes the lookahead call sites cheap without threading a
    /// cache through each of them. Active (dest, from) pairs are bounded by
    /// traffic, not map size; a size cap guards a pathological blow-up.
    pub fn next_hop(&self, dest: LinkId, from: LinkId) -> Option<LinkId> {
        let key = ((dest.0 as u64) << 32) | from.0 as u64;
        MEMO.with(|c| {
            let mut m = c.borrow_mut();
            if let Some(&(v, g)) = m.map.get(&key) {
                // The entry expires at the first generation bump whose index
                // lands on this key's rotating cohort — exactly one cohort per
                // bump, so each bump re-searches 1/MEMO_SPREAD of the cache.
                let cohort = key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 62; // 0..MEMO_SPREAD-1
                let until_expiry = 1 + (cohort.wrapping_sub(g).wrapping_sub(1) % MEMO_SPREAD);
                if self.generation.wrapping_sub(g) < until_expiry {
                    return (v != u32::MAX).then(|| LinkId(v));
                }
            }
            let ans = self.next_hop_search(dest, from);
            if m.map.len() >= MEMO_CAP {
                // Evict only pairs no query has refreshed for several
                // generations — a blanket clear would recreate the re-search
                // burst the cohorts exist to prevent.
                let gen = self.generation;
                m.map.retain(|_, &mut (_, g)| gen.wrapping_sub(g) <= MEMO_SPREAD);
            }
            m.map.insert(key, (ans.map_or(u32::MAX, |l| l.0), self.generation));
            ans
        })
    }

    fn next_hop_search(&self, dest: LinkId, from: LinkId) -> Option<LinkId> {
        use std::cmp::Reverse;
        if from == dest {
            return None;
        }
        if self.adj[from.idx()].is_empty() {
            return None;
        }
        let dest_i = dest.idx();
        SCRATCH.with(|c| {
        let mut sc = c.borrow_mut();
        let Scratch { g, first, heap } = &mut *sc;
        g.clear();
        first.clear();
        heap.clear();
        g.insert(from.0, 0);
        heap.push(Reverse((self.heuristic(from.idx(), dest_i).unwrap_or(0), from.0)));
        let mut settled = 0usize;
        // Best frontier fallback: the closest-to-dest link seen (by ALT bound),
        // for when dest is beyond the horizon.
        let mut best_front: Option<(u64, u32)> = None;
        while let Some(Reverse((_f, u))) = heap.pop() {
            if u == dest.0 {
                return first.get(&u).copied().map(LinkId);
            }
            settled += 1;
            if settled > SEARCH_CAP {
                break;
            }
            let gu = g.get(&u).copied().unwrap_or(u64::MAX);
            for &c in &self.adj[u as usize] {
                let ng = gu.saturating_add((self.cost[c as usize] + self.penalty[c as usize]) as u64);
                if ng < g.get(&c).copied().unwrap_or(u64::MAX) {
                    g.insert(c, ng);
                    let fh = if u == from.0 { c } else { first.get(&u).copied().unwrap_or(c) };
                    first.insert(c, fh);
                    let h = self.heuristic(c as usize, dest_i).unwrap_or_else(|| self.geo(c as usize, self.pos[dest_i]));
                    if best_front.is_none_or(|(bh, _)| h < bh) {
                        best_front = Some((h, fh));
                    }
                    heap.push(Reverse((ng.saturating_add(h), c)));
                }
            }
        }
        best_front.map(|(_, fh)| LinkId(fh))
        })
    }

    /// Plan a full route `from → dest` once, ALT-guided A* (the landmark lower
    /// bound as the admissible heuristic). Returns the link sequence *excluding*
    /// `from` (the hops to take), or `None` if unreachable. This is the O(1)-per-
    /// tick model: a driver plans its route at spawn, then follows the hop list —
    /// no per-tick search, no per-map field. A* cost is O(path · log) once per
    /// trip, independent of total map size for a given trip length.
    pub fn route(&self, from: LinkId, dest: LinkId) -> Option<Vec<LinkId>> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        if from == dest {
            return Some(Vec::new());
        }
        let n = self.adj.len();
        // g-cost so far and predecessor, sparse via HashMap (touched links only —
        // A* explores a corridor, not the whole map).
        let mut g: super::hash::IntMap<u64> = super::hash::IntMap::default();
        let mut prev: super::hash::IntMap<u32> = super::hash::IntMap::default();
        let mut heap: BinaryHeap<Reverse<(u64, u32)>> = BinaryHeap::new();
        g.insert(from.0, 0);
        let h0 = self.heuristic(from.idx(), dest.idx()).unwrap_or(0);
        heap.push(Reverse((h0, from.0)));
        let _ = n;
        while let Some(Reverse((_f, u))) = heap.pop() {
            if u == dest.0 {
                // Reconstruct hop list (excluding `from`).
                let mut hops = vec![LinkId(dest.0)];
                let mut cur = dest.0;
                while let Some(&p) = prev.get(&cur) {
                    if p == from.0 {
                        break;
                    }
                    hops.push(LinkId(p));
                    cur = p;
                }
                hops.reverse();
                return Some(hops);
            }
            let gu = g.get(&u).copied().unwrap_or(u64::MAX);
            for &c in &self.adj[u as usize] {
                let ng = gu.saturating_add((self.cost[c as usize] + self.penalty[c as usize]) as u64);
                if ng < g.get(&c).copied().unwrap_or(u64::MAX) {
                    g.insert(c, ng);
                    prev.insert(c, u);
                    let h = self.heuristic(c as usize, dest.idx()).unwrap_or(0);
                    heap.push(Reverse((ng.saturating_add(h), c)));
                }
            }
        }
        None
    }

    /// A distance *estimate* (ms) from `from` to `dest` — the ALT lower bound,
    /// for ranking fallbacks (e.g. a wrong-lane car's forward-most movement).
    /// Not the exact field distance the flow-field returns, but monotone toward
    /// the goal, which is all the ranking needs.
    pub fn distance(&self, dest: LinkId, from: LinkId) -> Option<u64> {
        if from == dest {
            return Some(0);
        }
        self.heuristic(from.idx(), dest.idx()).or_else(|| Some(self.geo(from.idx(), self.pos[dest.idx()])))
    }

    /// Whether a route to `dest` is believed to exist from anywhere — the field
    /// router answers this from its field; here a destination is "known" if it is
    /// a real link (local routing discovers the path as it goes).
    pub fn knows(&self, dest: LinkId) -> bool {
        dest.idx() < self.adj.len()
    }

    /// ALT lower bound on `dist(from → dest)` in ms, or `None` if no landmark is
    /// reachable from both (fall back to geography).
    fn heuristic(&self, from: usize, dest: usize) -> Option<u64> {
        let mut best = 0u32;
        let mut any = false;
        for dl in &self.dist {
            let (df, dd) = (dl[from], dl[dest]);
            if df != u32::MAX && dd != u32::MAX {
                any = true;
                best = best.max(df.saturating_sub(dd));
            }
        }
        any.then_some(best as u64)
    }

    /// Geographic fallback: straight-line link→dest distance scaled to a rough ms
    /// (only used when landmarks give no bound, e.g. across an unreachable gap).
    fn geo(&self, from: usize, dpos: [f32; 2]) -> u64 {
        let p = self.pos[from];
        let d = ((p[0] - dpos[0]).powi(2) + (p[1] - dpos[1]).powi(2)).sqrt();
        (d * 100.0) as u64 // ~0.1s per metre; only a tiebreak, magnitude irrelevant
    }
}

/// Farthest-point (maxmin) landmark selection: spread `k` landmarks across the
/// map so their bounds cover every direction. Deterministic (starts at link 0).
fn pick_landmarks(pos: &[[f32; 2]], k: usize) -> Vec<u32> {
    let n = pos.len();
    if n == 0 {
        return Vec::new();
    }
    let mut chosen = vec![0u32];
    let mut mind: Vec<f32> = pos.iter().map(|p| dist2(*p, pos[0])).collect();
    while chosen.len() < k.min(n) {
        // The link farthest from every chosen landmark.
        let next = (0..n).max_by(|&a, &b| mind[a].total_cmp(&mind[b])).unwrap() as u32;
        chosen.push(next);
        for i in 0..n {
            mind[i] = mind[i].min(dist2(pos[i], pos[next as usize]));
        }
    }
    chosen
}

fn dist2(a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

fn to_u64(c: &[u32]) -> Vec<u64> {
    c.iter().map(|&x| x as u64).collect()
}

fn clamp_u32(d: &u64) -> u32 {
    (*d).min(u32::MAX as u64) as u32
}
