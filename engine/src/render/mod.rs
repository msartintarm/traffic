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

/// Drive a [`RenderTarget`] with the current world and vehicle poses. The single
/// path both backends go through, so they can't diverge.
pub fn draw_world<R: RenderTarget>(net: &Network, vehicle_poses: &[[f64; 3]], target: &mut R) {
    target.world(&geometry::world_mesh(net));
    target.markings(&geometry::marking_mesh(net));
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
    pub light: f32,
}

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
            self.vertices.push(StaticVertex { center, offset, color, light });
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
            self.vertices.push(StaticVertex { center: [p[0] as f32, p[1] as f32], offset: [0.0, 0.0], color, light: 0.0 });
        }
        for k in 1..points.len() as u32 - 1 {
            self.indices.extend([base, base + k, base + k + 1]);
        }
    }

    pub fn push_disc(&mut self, center: [f64; 2], radius: f64, color: [f32; 3]) {
        const SIDES: u32 = 12;
        let c = [center[0] as f32, center[1] as f32];
        let base = self.vertices.len() as u32;
        self.vertices.push(StaticVertex { center: c, offset: [0.0, 0.0], color, light: 0.0 });
        for k in 0..SIDES {
            let a = std::f64::consts::TAU * k as f64 / SIDES as f64;
            self.vertices.push(StaticVertex {
                center: c,
                offset: [(radius * a.cos()) as f32, (radius * a.sin()) as f32],
                color,
                light: 0.0,
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
}
