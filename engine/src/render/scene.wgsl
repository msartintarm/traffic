// One shader, three entry points: `vs_static` for the baked road/marking mesh,
// `vs_instanced` for vehicles and signal heads (with GPU-side prev→current
// interpolation by the `alpha` uniform), and `fs_main` for both — a matte body
// term plus an emissive term for lamp vertices (brake/tail/head lights).

struct Camera {
    view_proj: mat4x4<f32>,
    alpha: f32,
    meters_per_pixel: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> cam: Camera;

// Keep roads/lines at least this many world-metres of half-width per pixel, so a
// thin ribbon never collapses below ~1.5px on screen and rasterizes cleanly.
const MIN_HALF_PIXELS: f32 = 0.75;

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) light: f32,
    @location(2) brake: f32,
    // 1.0 when this vertex is a turn-signal lamp on the side the instance is
    // signalling (and the blink phase is lit); 0.0 otherwise.
    @location(3) blink: f32,
};

@vertex
fn vs_static(
    @location(0) center: vec2<f32>,
    @location(1) offset: vec2<f32>,
    @location(2) color: vec3<f32>,
    @location(3) light: f32,
) -> VOut {
    let min_off = MIN_HALF_PIXELS * cam.meters_per_pixel;
    let len = length(offset);
    var world = center;
    if (len > 1e-6) {
        world = center + offset * (max(len, min_off) / len);
    }

    var o: VOut;
    o.clip = cam.view_proj * vec4<f32>(world, 0.0, 1.0);
    o.color = color;
    o.light = light;
    o.brake = 0.0;
    o.blink = 0.0;
    return o;
}

@vertex
fn vs_instanced(
    @location(0) v_pos: vec2<f32>,
    @location(1) v_color: vec3<f32>,
    @location(2) v_light: f32,
    @location(3) i_pos: vec2<f32>,
    @location(4) i_prev_pos: vec2<f32>,
    @location(5) i_control: vec2<f32>,
    @location(6) i_scale: vec2<f32>,
    @location(7) i_color: vec3<f32>,
    @location(8) i_heading: f32,
    @location(9) i_prev_heading: f32,
    @location(10) i_brake: f32,
    @location(11) i_blinker: f32,
) -> VOut {
    // Quadratic Bézier prev → control → current: a straight line when the
    // control is the midpoint, a corner-hugging arc when it's an intersection.
    let t = cam.alpha;
    let u = 1.0 - t;
    let pos = u * u * i_prev_pos + 2.0 * u * t * i_control + t * t * i_pos;
    let heading = mix(i_prev_heading, i_heading, cam.alpha);
    let c = cos(heading);
    let s = sin(heading);
    let local = v_pos * i_scale;
    let rotated = vec2<f32>(local.x * c - local.y * s, local.x * s + local.y * c);
    let world = pos + rotated;

    // Light 4 = left signal lamp, 5 = right. Lit when the instance is signalling
    // that side (i_blinker < 0 left, > 0 right); the bridge already gated i_blinker
    // by the blink phase, so a non-zero value means "on this frame".
    var blink = 0.0;
    if ((v_light > 3.5 && v_light < 4.5 && i_blinker < -0.5) || (v_light > 4.5 && i_blinker > 0.5)) {
        blink = 1.0;
    }

    var o: VOut;
    o.clip = cam.view_proj * vec4<f32>(world, 0.0, 1.0);
    o.color = v_color * i_color;
    o.light = v_light;
    o.brake = i_brake;
    o.blink = blink;
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
    if (in.light < 3.5) {
        return vec4<f32>(in.color, 0.45); // congestion overlay (translucent)
    }
    // Turn-signal lamp (light 4/5): amber when blinking on, else fades to the body.
    let amber = vec3<f32>(1.0, 0.62, 0.0);
    return vec4<f32>(mix(in.color * 0.92, amber, in.blink), 1.0);
}
