// GPU mirror of `MesoCorridor::step` (open boundary). One invocation per cell;
// every read is from the committed input buffer and the sole write is this
// cell's next occupancy, so ping-ponging cells_in/cells_out is race-free by
// construction. This is the kernel that carries the 1M+ mass layer.

struct Params {
    cell_max: f32,
    capacity: f32,
    backward_ratio: f32,
    inflow_demand: f32,
    cell_count: u32,
    blocked_exit: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> cells_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> cells_out: array<f32>;

fn sending(n: f32) -> f32 {
    return min(n, p.capacity);
}

fn receiving(n: f32) -> f32 {
    return min(p.capacity, max(p.backward_ratio * (p.cell_max - n), 0.0));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.cell_count) {
        return;
    }
    let n = cells_in[i];

    var inflow: f32;
    if (i == 0u) {
        inflow = min(p.inflow_demand, receiving(n));
    } else {
        inflow = min(sending(cells_in[i - 1u]), receiving(n));
    }

    var outflow: f32;
    if (i == p.cell_count - 1u) {
        outflow = select(sending(n), 0.0, p.blocked_exit == 1u);
    } else {
        outflow = min(sending(n), receiving(cells_in[i + 1u]));
    }

    cells_out[i] = n + inflow - outflow;
}
