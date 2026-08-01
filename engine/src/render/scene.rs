//! Turning sim state into render [`Instance`]s: per-class vehicle sizing, brake
//! lights from acceleration, signal heads from signal state, and LOD selection.
//! Pure functions — the shader-facing counterpart of the sim's `scene`-style math.

use crate::sim::config::VehicleClass;
use crate::sim::signal::SignalState;

use super::{Instance, Mesh, Vertex};

/// Render `[length, width]` in metres for a vehicle class.
pub fn class_dims(c: VehicleClass) -> [f32; 2] {
    [c.driver().vehicle_length as f32, c.width() as f32]
}

pub fn class_color(c: VehicleClass) -> [f32; 3] {
    match c {
        VehicleClass::Car => [0.80, 0.82, 0.86],
        VehicleClass::Truck => [0.86, 0.55, 0.24],
        VehicleClass::Bus => [0.30, 0.52, 0.86],
    }
}

/// A unit car mesh (extent ±0.5, scaled to metres per-instance): a matte body, a
/// red rear-lamp strip flagged `light = 1` (emissive under braking), and two
/// front-corner turn-signal lamps flagged `light = 4` (left, +y) / `light = 5`
/// (right, −y) that the shader lights amber when the instance's `blinker` matches.
pub fn unit_car_mesh() -> Mesh {
    let mut m = Mesh::default();
    m.push_quad([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], [1.0, 1.0, 1.0]);
    let base = m.vertices.len() as u32;
    let red = [1.0, 0.25, 0.18];
    for c in [[-0.5, -0.5], [-0.36, -0.5], [-0.36, 0.5], [-0.5, 0.5]] {
        m.vertices.push(Vertex::lamp(c, red, 1.0));
    }
    m.indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    // Turn-signal lamps at the two front corners. `light` (4/5) selects the side;
    // the plain body colour lets the fragment fade them to the body when unlit.
    let body = [1.0, 1.0, 1.0];
    for (corners, kind) in [
        ([[0.30, 0.34], [0.5, 0.34], [0.5, 0.5], [0.30, 0.5]], 4.0),   // left (+y)
        ([[0.30, -0.5], [0.5, -0.5], [0.5, -0.34], [0.30, -0.34]], 5.0), // right (−y)
    ] {
        let b = m.vertices.len() as u32;
        for c in corners {
            m.vertices.push(Vertex::lamp(c, body, kind));
        }
        m.indices.extend([b, b + 1, b + 2, b, b + 2, b + 3]);
    }
    m
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
        control: [(v.pos[0] + v.prev_pos[0]) * 0.5, (v.pos[1] + v.prev_pos[1]) * 0.5],
        scale: class_dims(v.class),
        color: class_color(v.class),
        heading: v.heading,
        prev_heading: v.prev_heading,
        brake: brake_intensity(v.accel),
        blinker: 0.0,
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

/// Dark housing colour of a signal head.
pub const SIGNAL_HOUSING: [f32; 3] = [0.05, 0.05, 0.06];

fn emissive(pos: [f32; 2], scale: [f32; 2], color: [f32; 3], heading: f32) -> Instance {
    Instance { pos, prev_pos: pos, control: pos, scale, color, heading, prev_heading: heading, brake: 1.0, blinker: 0.0 }
}

/// A miniature three-lamp signal head (a dark housing with red/yellow/green
/// lamps, the current state's lamp lit and the others dimmed) at `pos`, aligned
/// to the approach `heading` — the on-street look of a real Bay Area signal.
pub fn signal_head_instances(pos: [f32; 2], heading: f32, state: SignalState) -> [Instance; 4] {
    // Only the active lamp shows colour; the other two are dark (unlit) lenses,
    // so a red head reads clearly as red and a green as green — no phantom colour.
    const UNLIT: [f32; 3] = [0.03, 0.03, 0.035];
    let fwd = [heading.cos(), heading.sin()];
    let lamp = |slot: f32, on: bool, base: [f32; 3]| {
        let p = [pos[0] + fwd[0] * slot, pos[1] + fwd[1] * slot];
        emissive(p, [0.9, 0.9], if on { base } else { UNLIT }, heading)
    };
    [
        emissive(pos, [5.6, 2.3], SIGNAL_HOUSING, heading), // housing
        lamp(-1.6, state == SignalState::Red, signal_color(SignalState::Red)),
        lamp(0.0, state == SignalState::Yellow, signal_color(SignalState::Yellow)),
        lamp(1.6, state == SignalState::Green, signal_color(SignalState::Green)),
    ]
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
        assert!(class_dims(VehicleClass::Truck)[0] > class_dims(VehicleClass::Car)[0]);
        assert!(class_dims(VehicleClass::Bus)[0] > class_dims(VehicleClass::Truck)[0]);
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
        assert_eq!(inst.scale, class_dims(VehicleClass::Car));
        assert!(inst.brake > 0.0);
    }

    #[test]
    fn unit_car_mesh_has_a_body_brake_and_turn_signal_lamps() {
        let m = unit_car_mesh();
        assert!(m.vertices.iter().any(|v| v.light == 0.0), "has matte body");
        assert!(m.vertices.iter().any(|v| v.light == 1.0), "has a brake lamp");
        assert!(m.vertices.iter().any(|v| v.light == 4.0), "has a left turn signal");
        assert!(m.vertices.iter().any(|v| v.light == 5.0), "has a right turn signal");
        assert_eq!(m.indices.len(), 24); // body + brake strip + two blinker quads
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
