//! Turning sim state into render [`Instance`]s: per-class vehicle sizing, brake
//! lights from acceleration, signal heads from signal state, and LOD selection.
//! Pure functions — the shader-facing counterpart of the sim's `scene`-style math.

use crate::sim::signal::SignalState;

use super::{Instance, Mesh, Vertex};

/// A unit car mesh (extent ±0.5, scaled to metres per-instance): a matte body
/// plus a red rear-lamp strip flagged `light = 1` so the shader makes it
/// emissive under braking.
pub fn unit_car_mesh() -> Mesh {
    let mut m = Mesh::default();
    m.push_quad([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], [1.0, 1.0, 1.0]);
    let base = m.vertices.len() as u32;
    let red = [1.0, 0.25, 0.18];
    for c in [[-0.5, -0.5], [-0.36, -0.5], [-0.36, 0.5], [-0.5, 0.5]] {
        m.vertices.push(Vertex::lamp(c, red, 1.0));
    }
    m.indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    m
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VehicleClass {
    Car,
    Truck,
    Bus,
}

impl VehicleClass {
    /// `[length, width]` in metres.
    pub fn dims(self) -> [f32; 2] {
        match self {
            VehicleClass::Car => [4.6, 2.0],
            VehicleClass::Truck => [10.0, 2.5],
            VehicleClass::Bus => [12.0, 2.55],
        }
    }

    pub fn body_color(self) -> [f32; 3] {
        match self {
            VehicleClass::Car => [0.80, 0.82, 0.86],
            VehicleClass::Truck => [0.86, 0.55, 0.24],
            VehicleClass::Bus => [0.30, 0.52, 0.86],
        }
    }
}

/// A vehicle as the renderer sees it: current and previous pose (for GPU
/// interpolation), longitudinal acceleration (for brake lights), and class.
#[derive(Clone, Copy, Debug)]
pub struct VehicleView {
    pub pos: [f32; 2],
    pub prev_pos: [f32; 2],
    pub heading: f32,
    pub prev_heading: f32,
    pub accel: f32,
    pub class: VehicleClass,
}

/// Brake-light intensity in `[0,1]`: off while coasting, ramping in with harder
/// deceleration.
pub fn brake_intensity(accel: f32) -> f32 {
    (((-accel) - 0.3) / 2.0).clamp(0.0, 1.0)
}

pub fn vehicle_instance(v: &VehicleView) -> Instance {
    Instance {
        pos: v.pos,
        prev_pos: v.prev_pos,
        scale: v.class.dims(),
        color: v.class.body_color(),
        heading: v.heading,
        prev_heading: v.prev_heading,
        brake: brake_intensity(v.accel),
        _pad: 0.0,
    }
}

/// A unit emissive disc (radius 1, scaled per-instance) for signal heads;
/// `light = 1` so the shader renders it fully lit in the instance colour.
pub fn signal_head_mesh() -> Mesh {
    const SIDES: usize = 10;
    let white = [1.0, 1.0, 1.0];
    let mut m = Mesh::default();
    m.vertices.push(Vertex::lamp([0.0, 0.0], white, 1.0));
    for k in 0..SIDES {
        let a = std::f64::consts::TAU * k as f64 / SIDES as f64;
        m.vertices.push(Vertex::lamp([a.cos() as f32, a.sin() as f32], white, 1.0));
    }
    for k in 0..SIDES as u32 {
        m.indices.extend([0, 1 + k, 1 + (k + 1) % SIDES as u32]);
    }
    m
}

pub fn signal_color(state: SignalState) -> [f32; 3] {
    match state {
        SignalState::Red => [0.90, 0.11, 0.11],
        SignalState::Yellow => [0.95, 0.75, 0.12],
        SignalState::Green => [0.20, 0.85, 0.28],
    }
}

/// A signal head as an emissive instance at `pos`, sized `radius`.
pub fn signal_instance(pos: [f32; 2], radius: f32, state: SignalState) -> Instance {
    Instance {
        pos,
        prev_pos: pos,
        scale: [radius, radius],
        color: signal_color(state),
        heading: 0.0,
        prev_heading: 0.0,
        brake: 1.0, // fully emissive
        _pad: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brake_lights_off_while_coasting_on_while_stopping() {
        assert_eq!(brake_intensity(1.0), 0.0);
        assert_eq!(brake_intensity(-0.1), 0.0);
        assert!(brake_intensity(-1.5) > 0.0);
        assert_eq!(brake_intensity(-100.0), 1.0);
    }

    #[test]
    fn class_sizes_are_ordered() {
        assert!(VehicleClass::Truck.dims()[0] > VehicleClass::Car.dims()[0]);
        assert!(VehicleClass::Bus.dims()[0] > VehicleClass::Truck.dims()[0]);
    }

    #[test]
    fn vehicle_instance_carries_pose_and_cues() {
        let v = VehicleView {
            pos: [10.0, 5.0],
            prev_pos: [9.0, 5.0],
            heading: 0.5,
            prev_heading: 0.4,
            accel: -2.0,
            class: VehicleClass::Car,
        };
        let inst = vehicle_instance(&v);
        assert_eq!(inst.pos, [10.0, 5.0]);
        assert_eq!(inst.prev_pos, [9.0, 5.0]);
        assert_eq!(inst.scale, VehicleClass::Car.dims());
        assert!(inst.brake > 0.0);
    }

    #[test]
    fn unit_car_mesh_has_a_body_and_an_emissive_rear_lamp() {
        let m = unit_car_mesh();
        assert!(m.vertices.iter().any(|v| v.light == 0.0), "has matte body");
        assert!(m.vertices.iter().any(|v| v.light == 1.0), "has a brake lamp");
        assert_eq!(m.indices.len(), 12); // two quads
    }

    #[test]
    fn signal_head_mesh_is_an_emissive_disc() {
        let m = signal_head_mesh();
        assert!(m.vertices.iter().all(|v| v.light == 1.0));
        assert!(m.indices.len() >= 3 && m.indices.len() % 3 == 0);
    }

    #[test]
    fn signal_colors_are_distinct() {
        let r = signal_color(SignalState::Red);
        let g = signal_color(SignalState::Green);
        let y = signal_color(SignalState::Yellow);
        assert!(r != g && g != y && r != y);
        assert!(r[0] > g[0] && g[1] > r[1]);
    }
}
