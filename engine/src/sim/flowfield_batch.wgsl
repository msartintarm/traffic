// Batched flow-field relaxation: one Bellman-Ford pass over *all* destination
// fields at once. `dist` holds `slot_count` fields of `link_count` each, laid out
// [slot0 links…][slot1 links…]…. One invocation per (slot, link) relaxes that
// link within its slot's field, reading committed distances (dist_in) and writing
// dist_out; the host ping-pongs to convergence. This is what makes many
// destinations over a large graph a single parallel dispatch instead of K.

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
@group(0) @binding(5) var<storage, read> dist_in: array<u32>;   // slot_count * link_count
@group(0) @binding(6) var<storage, read_write> dist_out: array<u32>;

const INF: u32 = 4294967295u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    let total = p.slot_count * p.link_count;
    if (g >= total) {
        return;
    }
    let slot = g / p.link_count;
    let a = g % p.link_count;
    let base = slot * p.link_count;
    if (a == dests[slot]) {
        dist_out[g] = 0u;
        return;
    }
    var best: u32 = INF;
    let start = offsets[a];
    let end = offsets[a + 1u];
    for (var e = start; e < end; e = e + 1u) {
        let b = targets[e];
        let db = dist_in[base + b];
        if (db != INF) {
            let c = cost[b];
            if (c <= INF - db) {
                best = min(best, c + db);
            }
        }
    }
    dist_out[g] = best;
}
