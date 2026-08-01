// Batched flow-field relaxation with in-place atomicMin and GPU-side convergence.
// `dist` holds `slot_count` fields of `link_count` each ([slot0 links…][slot1
// links…]…), so many destination fields over a large graph solve in one dispatch
// stream instead of K. Each iteration is two passes the host encodes back-to-back:
//
//   * `relax` lowers every (slot, link) toward its onward neighbours via
//     `atomicMin`, flagging `flags[0]` (changed) whenever it moves a value;
//   * `gate` latches `flags[1]` (done) as soon as a relax pass changed nothing,
//     and clears `changed` for the next pass.
//
// Once `done` latches, further relax passes early-out — so the solve costs about
// the graph's diameter in iterations, not the `link_count` Bellman-Ford worst case
// the host loop still bounds it by. In-place `atomicMin` (rather than ping-ponging
// two buffers) is what makes a converged pass free: it touches no memory and needs
// no copy to keep the "current" buffer straight.

struct Params {
    link_count: u32,
    slot_count: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> offsets: array<u32>;   // len link_count+1
@group(0) @binding(2) var<storage, read> targets: array<u32>;   // len = #edges
@group(0) @binding(3) var<storage, read> cost: array<u32>;      // per link (entry cost)
@group(0) @binding(4) var<storage, read> dests: array<u32>;     // dest link per slot
@group(0) @binding(5) var<storage, read_write> dist: array<atomic<u32>>;  // slot_count * link_count
@group(0) @binding(6) var<storage, read_write> flags: array<atomic<u32>>; // [changed, done]

const INF: u32 = 4294967295u;

@compute @workgroup_size(64)
fn relax(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (atomicLoad(&flags[1]) == 1u) {
        return; // already converged — no memory touched
    }
    let g = gid.x;
    let total = p.slot_count * p.link_count;
    if (g >= total) {
        return;
    }
    let slot = g / p.link_count;
    let a = g % p.link_count;
    if (a == dests[slot]) {
        return; // the destination link stays pinned at 0 (set by the host init)
    }
    let base = slot * p.link_count;
    var best: u32 = INF;
    let start = offsets[a];
    let end = offsets[a + 1u];
    for (var e = start; e < end; e = e + 1u) {
        let b = targets[e];
        let db = atomicLoad(&dist[base + b]);
        if (db != INF) {
            let c = cost[b];
            if (c <= INF - db) {
                best = min(best, c + db);
            }
        }
    }
    if (best < INF) {
        let old = atomicMin(&dist[g], best);
        if (best < old) {
            atomicStore(&flags[0], 1u);
        }
    }
}

@compute @workgroup_size(1)
fn gate() {
    if (atomicLoad(&flags[0]) == 0u) {
        atomicStore(&flags[1], 1u); // nothing changed last pass ⇒ converged
    }
    atomicStore(&flags[0], 0u); // reset for the next relax pass
}
