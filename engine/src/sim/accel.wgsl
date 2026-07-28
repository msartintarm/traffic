// GPU mirror of the CPU accel evaluate (`net_world::VehicleContext::binding`): the
// binding (minimum) of the IDM/constraint set, per vehicle, in f32. The `accel_noise`
// term is added on the CPU after readback — its RNG is u64 (SplitMix64), which WGSL
// has no type for. Absent optional constraints arrive as the `BIG` sentinel and are
// skipped, matching the CPU's `+∞`-is-non-binding convention.

// Field order and types must match `net_world::VehicleContextGpu`.
struct Ctx {
    desired_speed: f32,
    accel_exponent: f32,
    min_gap: f32,
    time_headway: f32,
    max_accel: f32,
    comfort_decel: f32,
    speed: f32,
    leader_gap: f32,
    leader_speed: f32,
    stop_line: f32,
    speed_target_speed: f32,
    speed_target_dist: f32,
    stop_sign: f32,
    yield_line: f32,
    curve_speed: f32,
    curve_dist: f32,
    merge_gap: f32,
    merge_speed: f32,
};

@group(0) @binding(0) var<storage, read> contexts: array<Ctx>;
@group(0) @binding(1) var<storage, read_write> out_accel: array<f32>;

const BIG: f32 = 1e30;

// IDM acceleration (Treiber): `s` bumper-to-bumper gap, `dv = v - v_lead`.
fn idm_accel(v: f32, dv: f32, gap: f32, v0: f32, delta: f32, s0: f32, tt: f32, a: f32, b: f32) -> f32 {
    let s = max(gap, 1e-3);
    let free = 1.0 - pow(v / v0, delta);
    let s_star = s0 + max(v * tt + (v * dv) / (2.0 * sqrt(a * b)), 0.0);
    let ratio = s_star / s;
    return a * (free - ratio * ratio);
}

fn free_accel(v: f32, v0: f32, delta: f32, a: f32) -> f32 {
    return a * (1.0 - pow(v / v0, delta));
}

// Constant-deceleration braking to reach `tspeed` by `tdist` ahead; non-binding
// (returns BIG) when already at or below the target.
fn brake_to_target(v: f32, tspeed: f32, tdist: f32) -> f32 {
    if (v > tspeed) {
        return (tspeed * tspeed - v * v) / (2.0 * max(tdist, 0.05));
    }
    return BIG;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&contexts)) {
        return;
    }
    let c = contexts[i];
    let v = c.speed;
    let v0 = c.desired_speed;
    let delta = c.accel_exponent;
    let s0 = c.min_gap;
    let tt = c.time_headway;
    let a = c.max_accel;
    let b = c.comfort_decel;

    // desired_speed always binds (the free-road term); fold the rest with min().
    var acc = free_accel(v, v0, delta, a);
    if (c.leader_gap < BIG) {
        acc = min(acc, idm_accel(v, v - c.leader_speed, c.leader_gap, v0, delta, s0, tt, a, b));
    }
    // stop_line / stop_sign / give_way: brake to a stationary line (dv = v).
    if (c.stop_line < BIG) {
        acc = min(acc, idm_accel(v, v, max(c.stop_line, 0.05), v0, delta, s0, tt, a, b));
    }
    if (c.speed_target_dist < BIG) {
        acc = min(acc, brake_to_target(v, c.speed_target_speed, c.speed_target_dist));
    }
    if (c.curve_dist < BIG) {
        acc = min(acc, brake_to_target(v, c.curve_speed, c.curve_dist));
    }
    if (c.stop_sign < BIG) {
        acc = min(acc, idm_accel(v, v, max(c.stop_sign, 0.05), v0, delta, s0, tt, a, b));
    }
    if (c.yield_line < BIG) {
        acc = min(acc, idm_accel(v, v, max(c.yield_line, 0.05), v0, delta, s0, tt, a, b));
    }
    if (c.merge_gap < BIG) {
        acc = min(acc, idm_accel(v, v - c.merge_speed, c.merge_gap, v0, delta, s0, tt, a, b));
    }
    out_accel[i] = acc;
}
