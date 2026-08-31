// Composites the cached static-world texture into the frame. The cache is the
// world rendered at the current zoom over an expanded viewport; `a`/`b` map the
// live frame's NDC into the cache's NDC (pure scale + translate — the camera
// never rotates), so a pan inside the cached margin is just a shifted sample.

struct BlitParams {
    a: vec2<f32>,
    b: vec2<f32>,
}

@group(0) @binding(0) var<uniform> blit: BlitParams;
@group(0) @binding(1) var cache_tex: texture_2d<f32>;
@group(0) @binding(2) var cache_samp: sampler;

struct BlitOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_blit(@builtin(vertex_index) vi: u32) -> BlitOut {
    // Fullscreen triangle: (-1,-1), (3,-1), (-1,3).
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi >> 1u) * 4 - 1);
    var out: BlitOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    let c = blit.a * vec2<f32>(x, y) + blit.b;
    out.uv = vec2<f32>((c.x + 1.0) * 0.5, (1.0 - c.y) * 0.5);
    return out;
}

@fragment
fn fs_blit(in: BlitOut) -> @location(0) vec4<f32> {
    return textureSample(cache_tex, cache_samp, in.uv);
}
