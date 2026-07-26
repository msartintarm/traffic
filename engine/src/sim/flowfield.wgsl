// One parallel Bellman-Ford relaxation pass for flow-field routing, GPU mirror
// of `flowfield::distances_to`. One invocation per link relaxes it from its
// onward links (CSR adjacency), reading committed distances (dist_in) and
// writing the new distance (dist_out); the host ping-pongs until convergence.
// Distances are milliseconds as u32; INF marks unreachable.

struct Params {
    link_count: u32,
    dest: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> offsets: array<u32>;   // len link_count+1
@group(0) @binding(2) var<storage, read> targets: array<u32>;   // len = #edges
@group(0) @binding(3) var<storage, read> cost: array<u32>;      // per link (entry cost)
@group(0) @binding(4) var<storage, read> dist_in: array<u32>;
@group(0) @binding(5) var<storage, read_write> dist_out: array<u32>;

const INF: u32 = 4294967295u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let a = gid.x;
    if (a >= p.link_count) {
        return;
    }
    if (a == p.dest) {
        dist_out[a] = 0u;
        return;
    }
    var best: u32 = INF;
    let start = offsets[a];
    let end = offsets[a + 1u];
    for (var e = start; e < end; e = e + 1u) {
        let b = targets[e];
        let db = dist_in[b];
        if (db != INF) {
            let c = cost[b];
            if (c <= INF - db) {
                best = min(best, c + db);
            }
        }
    }
    dist_out[a] = best;
}
