// One shader, three entry points: `vs_static` for the baked road/marking mesh,
// `vs_instanced` for vehicles and signal heads (with GPU-side prev→current
// interpolation by the `alpha` uniform), and `fs_main` for both — a matte body
// term plus an emissive term for lamp vertices (brake/tail/head lights).

struct Camera {
    view_proj: mat4x4<f32>,
    alpha: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> cam: Camera;

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) light: f32,
    @location(2) brake: f32,
};

@vertex
fn vs_static(
    @location(0) pos: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) light: f32,
) -> VOut {
    var o: VOut;
    o.clip = cam.view_proj * vec4<f32>(pos, 0.0, 1.0);
    o.color = color;
    o.light = light;
    o.brake = 0.0;
    return o;
}

@vertex
fn vs_instanced(
    @location(0) v_pos: vec2<f32>,
    @location(1) v_color: vec3<f32>,
    @location(2) v_light: f32,
    @location(3) i_pos: vec2<f32>,
    @location(4) i_prev_pos: vec2<f32>,
    @location(5) i_scale: vec2<f32>,
    @location(6) i_color: vec3<f32>,
    @location(7) i_heading: f32,
    @location(8) i_prev_heading: f32,
    @location(9) i_brake: f32,
) -> VOut {
    let pos = mix(i_prev_pos, i_pos, cam.alpha);
    let heading = mix(i_prev_heading, i_heading, cam.alpha);
    let c = cos(heading);
    let s = sin(heading);
    let local = v_pos * i_scale;
    let rotated = vec2<f32>(local.x * c - local.y * s, local.x * s + local.y * c);
    let world = pos + rotated;

    var o: VOut;
    o.clip = cam.view_proj * vec4<f32>(world, 0.0, 1.0);
    o.color = v_color * i_color;
    o.light = v_light;
    o.brake = i_brake;
    return o;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    if (in.light < 0.5) {
        return vec4<f32>(in.color * 0.92, 1.0);
    }
    if (in.light < 1.5) {
        let e = 0.15 + 0.85 * in.brake;
        return vec4<f32>(in.color * e, 1.0);
    }
    if (in.light < 2.5) {
        return vec4<f32>(in.color * 0.7, 1.0);
    }
    return vec4<f32>(in.color, 0.55); // congestion overlay (translucent)
}
