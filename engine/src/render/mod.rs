//! Rendering.
//!
//! Mirrors `../plant`: the geometry/scene/camera math is pure, dependency-free,
//! and native-`cargo test`-able (`camera`, `geometry`, `scene`, `mass`); only
//! the wgpu device + frame loop is `wasm32`-gated. Correctness lives in the
//! pure layer so visual behaviour is verified without a browser, exactly as the
//! sim is. The shader (`scene.wgsl`) is `naga`-validated under `cargo test`.

pub mod ascii;
pub mod camera;
pub mod geometry;
pub mod interp;
pub mod mass;
pub mod raster;
pub mod scene;

#[cfg(target_arch = "wasm32")]
pub mod gpu;

use bytemuck::{Pod, Zeroable};

use crate::sim::network::Network;

/// A render backend the engine can drive. Both the browser GPU renderer and the
/// ASCII rasteriser are `RenderTarget`s: [`draw_world`] feeds them identical
/// geometry from the one [`geometry::world_mesh`] builder, so an ASCII regression
/// test exercises exactly the scene the browser draws — the parity guarantee.
pub trait RenderTarget {
    /// The static world surface (carriageways + junction fills + overpasses).
    fn world(&mut self, mesh: &StaticMesh);
    /// A vehicle at world pose `[x, y, heading]`.
    fn vehicle(&mut self, pose: [f64; 3]);
    /// Lane markings; ignored by default (the ASCII view omits them for clarity,
    /// the GPU renderer draws them).
    fn markings(&mut self, _mesh: &StaticMesh) {}
}

/// Drive a [`RenderTarget`] with the current world and vehicle poses. Draws in
/// painter's-order render bands (grade layer, then road-class priority): each
/// band's fill then its markings, bottom to top, so an overpass band's fill
/// covers the road and lane lines it crosses over, and same-grade overlaps
/// resolve by road class. The single path the raster/ascii backends go through;
/// the GPU backend consumes the same [`geometry::world_bands`].
pub fn draw_world<R: RenderTarget>(net: &Network, vehicle_poses: &[[f64; 3]], target: &mut R) {
    for band in geometry::world_bands(net) {
        target.world(&band.fill);
        target.markings(&band.marking);
    }
    for &pose in vehicle_poses {
        target.vehicle(pose);
    }
}

/// A mesh vertex. `light` selects emissive behaviour in the shader:
/// 0 = matte body, 1 = brake/tail lamp, 2 = headlamp.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 2],
    pub color: [f32; 3],
    pub light: f32,
}

impl Vertex {
    pub const fn body(pos: [f32; 2], color: [f32; 3]) -> Self {
        Self { pos, color, light: 0.0 }
    }
    pub const fn lamp(pos: [f32; 2], color: [f32; 3], kind: f32) -> Self {
        Self { pos, color, light: kind }
    }
}

/// One rendered vehicle (or signal head). Carries both the previous and current
/// tick pose so the vertex shader interpolates by a single `alpha` uniform —
/// the CPU never lerps per car.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Instance {
    pub pos: [f32; 2],
    pub prev_pos: [f32; 2],
    /// Quadratic-Bézier control point for prev→current motion: the segment
    /// midpoint for a straight move (curve degenerates to a line), or the
    /// intersection node for a turn (the path bulges through the corner).
    pub control: [f32; 2],
    pub scale: [f32; 2],
    pub color: [f32; 3],
    pub heading: f32,
    pub prev_heading: f32,
    /// 0 = coasting, 1 = braking; scales the rear-lamp emissive.
    pub brake: f32,
    /// Turn-signal side, gated by the blink phase: `-1` left lamp lit, `+1` right
    /// lamp lit, `0` none. The bridge toggles it on/off so the shader stays timeless.
    pub blinker: f32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl Mesh {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Append `other`, offsetting its indices to the merged vertex range.
    pub fn extend(&mut self, other: &Mesh) {
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&other.vertices);
        self.indices.extend(other.indices.iter().map(|i| i + base));
    }

    /// A convex fan `[center, ring...]` as triangles, or a quad from 4 CCW
    /// corners — the two primitives road/marking geometry is built from.
    pub fn push_quad(&mut self, corners: [[f32; 2]; 4], color: [f32; 3]) {
        let base = self.vertices.len() as u32;
        for &c in &corners {
            self.vertices.push(Vertex::body(c, color));
        }
        self.indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// A static-geometry vertex stored as `center + offset` rather than a baked
/// absolute position, so the vertex shader can expand `offset` (always
/// perpendicular to a road/line) to a minimum on-screen width — keeping roads at
/// least ~1px wide at any zoom instead of rasterizing to choppy sub-pixel slivers.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct StaticVertex {
    pub center: [f32; 2],
    pub offset: [f32; 2],
    pub color: [f32; 3],
    /// Fragment-shader tag. 0 = flat static geometry; 1–5 are vehicle-lamp tiers
    /// (see `scene.wgsl`). For a DASHED marking it instead carries
    /// `DASH_LIGHT_BASE + arc-length` (world metres along the line), interpolated
    /// across the ribbon so the shader paints the dash pattern itself — one quad
    /// per whole divider instead of one per 6 m dash — with no extra vertex field.
    pub light: f32,
    /// Road-fill only: signed lateral position (world metres) across the
    /// carriageway — `-hw` at the median edge, `+hw` at the curb edge, interpolated
    /// so the fragment paints the solid centre (median, yellow) and edge (curb,
    /// white) lines itself. That removes the per-segment edge/centre ribbons — the
    /// bulk of the marking mesh — at the cost of these two floats on every vertex.
    pub edge: f32,
    /// Road-fill only: the carriageway half-width (metres); `0` on all other
    /// geometry, which the fragment reads as "no edge lines".
    pub hw: f32,
}

/// A `light` at or above this marks a dashed marking; the excess is the arc-length
/// (metres) along the line. Well clear of the 0–5 lamp/overlay tiers.
pub(crate) const DASH_LIGHT_BASE: f32 = 100.0;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct StaticMesh {
    pub vertices: Vec<StaticVertex>,
    pub indices: Vec<u32>,
}

fn unit_perp(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len = dx.hypot(dy).max(1e-9);
    [dy / len, -dx / len]
}

impl StaticMesh {
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    pub fn extend(&mut self, other: &StaticMesh) {
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&other.vertices);
        self.indices.extend(other.indices.iter().map(|i| i + base));
    }

    /// A width-`2·half_w` ribbon quad along the centre segment `a→b`; the two
    /// side offsets are perpendicular so the shader can widen them per-zoom.
    pub fn push_ribbon(&mut self, a: [f64; 2], b: [f64; 2], half_w: f64, color: [f32; 3], light: f32) {
        let n = unit_perp(a, b);
        let off = [(n[0] * half_w) as f32, (n[1] * half_w) as f32];
        let neg = [-off[0], -off[1]];
        let (af, bf) = ([a[0] as f32, a[1] as f32], [b[0] as f32, b[1] as f32]);
        let base = self.vertices.len() as u32;
        for (center, offset) in [(af, neg), (bf, neg), (bf, off), (af, off)] {
            self.vertices.push(StaticVertex { center, offset, color, light, edge: 0.0, hw: 0.0 });
        }
        self.indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    /// A carriageway-fill ribbon that also carries the solid centre/edge line
    /// markings, painted by the fragment shader from the signed lateral position
    /// `edge` (∈ `[-half_w, +half_w]`): a yellow median line at `-half_w`, a white
    /// curb line at `+half_w`. Replaces per-segment edge/centre marking ribbons —
    /// the bulk of the marking mesh — with the road fill it already draws. The
    /// `-half_w` side is the median (left of travel for right-hand traffic), the
    /// `+half_w` side the curb, matching the perpendicular from [`unit_perp`].
    pub fn push_road_fill(&mut self, a: [f64; 2], b: [f64; 2], half_w: f64, color: [f32; 3]) {
        let n = unit_perp(a, b);
        let off = [(n[0] * half_w) as f32, (n[1] * half_w) as f32];
        let neg = [-off[0], -off[1]];
        let (af, bf) = ([a[0] as f32, a[1] as f32], [b[0] as f32, b[1] as f32]);
        let hw = half_w as f32;
        let base = self.vertices.len() as u32;
        // Order mirrors `push_ribbon`: the `neg` (−n) side is the median (`edge = -hw`).
        for (center, offset, edge) in [(af, neg, -hw), (bf, neg, -hw), (bf, off, hw), (af, off, hw)] {
            self.vertices.push(StaticVertex { center, offset, color, light: 0.0, edge, hw });
        }
        self.indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    /// One ribbon quad along `a→b` whose dash pattern (3 m on / 3 m off) is painted
    /// by the fragment shader from the interpolated arc-length carried in `light`
    /// (offset by [`DASH_LIGHT_BASE`]). Replaces the old per-dash quads: a lane
    /// divider is now a single quad, not `length/6` of them — and it reuses the
    /// existing `light` field, so nothing grows. `a`-end vertices carry arc-length
    /// 0, `b`-end vertices the full length.
    pub fn push_dashed(&mut self, a: [f64; 2], b: [f64; 2], half_w: f64, color: [f32; 3]) {
        let n = unit_perp(a, b);
        let off = [(n[0] * half_w) as f32, (n[1] * half_w) as f32];
        let neg = [-off[0], -off[1]];
        let (af, bf) = ([a[0] as f32, a[1] as f32], [b[0] as f32, b[1] as f32]);
        let len = ((b[0] - a[0]).hypot(b[1] - a[1])) as f32;
        let base = self.vertices.len() as u32;
        // Order mirrors `push_ribbon`: a, b, b, a — so a-end verts get arc-length 0.
        for (center, offset, arc) in [(af, neg, 0.0), (bf, neg, len), (bf, off, len), (af, off, 0.0)] {
            self.vertices.push(StaticVertex { center, offset, color, light: DASH_LIGHT_BASE + arc, edge: 0.0, hw: 0.0 });
        }
        self.indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    /// Fill a convex polygon (vertices in order) as a triangle fan. Vertices
    /// carry zero offset — a real paved area, not a min-width line.
    pub fn push_polygon(&mut self, points: &[[f64; 2]], color: [f32; 3]) {
        if points.len() < 3 {
            return;
        }
        let base = self.vertices.len() as u32;
        for p in points {
            self.vertices.push(StaticVertex { center: [p[0] as f32, p[1] as f32], offset: [0.0, 0.0], color, light: 0.0, edge: 0.0, hw: 0.0 });
        }
        for k in 1..points.len() as u32 - 1 {
            self.indices.extend([base, base + k, base + k + 1]);
        }
    }

    pub fn push_disc(&mut self, center: [f64; 2], radius: f64, color: [f32; 3]) {
        const SIDES: u32 = 12;
        let c = [center[0] as f32, center[1] as f32];
        let base = self.vertices.len() as u32;
        self.vertices.push(StaticVertex { center: c, offset: [0.0, 0.0], color, light: 0.0, edge: 0.0, hw: 0.0 });
        for k in 0..SIDES {
            let a = std::f64::consts::TAU * k as f64 / SIDES as f64;
            self.vertices.push(StaticVertex {
                center: c,
                offset: [(radius * a.cos()) as f32, (radius * a.sin()) as f32],
                color,
                light: 0.0,
                edge: 0.0,
                hw: 0.0,
            });
        }
        for k in 0..SIDES {
            self.indices.extend([base, base + 1 + k, base + 1 + (k + 1) % SIDES]);
        }
    }
}

/// Level-of-detail tier, selected by on-screen distance. This is what makes
/// 1M+ tractable: individual meshes only [`Lod::Near`], flat quads [`Lod::Mid`],
/// and no per-car draw at all [`Lod::Far`] (the mass layer's density shading
/// stands in — see [`mass`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lod {
    Near,
    Mid,
    Far,
}

impl Lod {
    /// `near`/`mid` are world-distance thresholds from the camera focus.
    pub fn for_distance(d: f64, near: f64, mid: f64) -> Lod {
        if d <= near {
            Lod::Near
        } else if d <= mid {
            Lod::Mid
        } else {
            Lod::Far
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_extend_offsets_indices() {
        let mut a = Mesh::default();
        a.push_quad([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], [1.0, 1.0, 1.0]);
        let mut b = Mesh::default();
        b.push_quad([[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0]], [1.0, 0.0, 0.0]);
        let n = a.vertices.len() as u32;
        a.extend(&b);
        assert_eq!(a.vertices.len(), 8);
        assert_eq!(a.indices.len(), 12);
        assert!(a.indices[6..].iter().all(|&i| i >= n));
    }

    #[test]
    fn lod_tiers_split_by_distance() {
        assert_eq!(Lod::for_distance(10.0, 50.0, 200.0), Lod::Near);
        assert_eq!(Lod::for_distance(120.0, 50.0, 200.0), Lod::Mid);
        assert_eq!(Lod::for_distance(500.0, 50.0, 200.0), Lod::Far);
    }

    #[test]
    fn instance_layout_matches_the_shader_stride() {
        // The instanced vertex buffer packs these fields back-to-back; the GPU
        // attribute offsets and scene.wgsl's `@location`s assume this layout.
        assert_eq!(std::mem::size_of::<Instance>(), 60);
        assert_eq!(std::mem::offset_of!(Instance, prev_pos), 8);
        assert_eq!(std::mem::offset_of!(Instance, control), 16);
        assert_eq!(std::mem::offset_of!(Instance, scale), 24);
        assert_eq!(std::mem::offset_of!(Instance, color), 32);
        assert_eq!(std::mem::offset_of!(Instance, heading), 44);
        assert_eq!(std::mem::offset_of!(Instance, prev_heading), 48);
        assert_eq!(std::mem::offset_of!(Instance, brake), 52);
        assert_eq!(std::mem::offset_of!(Instance, blinker), 56);
    }

    #[test]
    fn scene_wgsl_parses_and_validates() {
        let src = include_str!("scene.wgsl");
        let module = naga::front::wgsl::parse_str(src).expect("scene.wgsl should parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("scene.wgsl should type-check");
    }

    #[test]
    fn blit_wgsl_parses_and_validates() {
        let src = include_str!("blit.wgsl");
        let module = naga::front::wgsl::parse_str(src).expect("blit.wgsl should parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("blit.wgsl should type-check");
    }
}
