//! Far-LOD visualization of the mass layer: instead of drawing a million cars,
//! shade each road by its Cell Transmission Model occupancy — free-flowing green
//! through to jammed red, opacity rising with density. Per-link, so it costs
//! nothing next to per-car rendering. Pure color math.

/// `[r, g, b, a]` for an occupancy ratio in `[0,1]` (occupancy / jam density):
/// green → amber → red, fading in as the road fills.
pub fn congestion_color(ratio: f64) -> [f32; 4] {
    let r = ratio.clamp(0.0, 1.0);
    let (red, green) = if r < 0.5 {
        (2.0 * r, 1.0)
    } else {
        (1.0, 1.0 - 2.0 * (r - 0.5))
    };
    let alpha = (0.15 + 0.7 * r).min(0.85);
    [red as f32, green as f32, 0.05, alpha as f32]
}

/// Occupancy ratio for a link given its current and jam vehicle counts.
pub fn occupancy_ratio(vehicles: f64, jam: f64) -> f64 {
    if jam <= 0.0 {
        0.0
    } else {
        (vehicles / jam).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_flow_is_green_jam_is_red() {
        let free = congestion_color(0.0);
        let jam = congestion_color(1.0);
        assert!(free[1] > free[0], "free flow leans green");
        assert!(jam[0] > jam[1], "jam leans red");
        assert!(jam[3] > free[3], "denser roads are more opaque");
    }

    #[test]
    fn midpoint_is_amber() {
        let mid = congestion_color(0.5);
        assert!(mid[0] > 0.8 && mid[1] > 0.8, "amber has strong red and green");
    }

    #[test]
    fn occupancy_ratio_is_clamped() {
        assert_eq!(occupancy_ratio(50.0, 10.0), 1.0);
        assert_eq!(occupancy_ratio(0.0, 10.0), 0.0);
        assert_eq!(occupancy_ratio(5.0, 0.0), 0.0);
    }
}
