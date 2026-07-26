//! Rendering.
//!
//! Mirrors `../plant`: the geometry/scene/camera math is pure, dependency-free,
//! and native-`cargo test`-able (`camera`, `geometry`, `scene`, `mass`); only
//! the wgpu device + frame loop is `wasm32`-gated. Correctness lives in the
//! pure layer so visual behaviour is verified without a browser, exactly as the
//! sim is. The shader (`scene.wgsl`) is `naga`-validated under `cargo test`.

pub mod camera;
pub mod geometry;
pub mod mass;
pub mod scene;

#[cfg(target_arch = "wasm32")]
pub mod gpu;

use bytemuck::{Pod, Zeroable};

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
    pub scale: [f32; 2],
    pub color: [f32; 3],
    pub heading: f32,
    pub prev_heading: f32,
    /// 0 = coasting, 1 = braking; scales the rear-lamp emissive.
    pub brake: f32,
    pub _pad: f32,
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
