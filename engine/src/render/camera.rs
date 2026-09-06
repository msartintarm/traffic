//! Orthographic pan/zoom camera over the world plane. Pure math: view-projection
//! for the shader, world↔screen mapping for picking/pan, and the visible world
//! rectangle for frustum culling.

/// Column-major 4×4, the layout wgpu/WGSL expect.
pub type Mat4 = [f32; 16];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    /// World point at the centre of the viewport.
    pub center: [f64; 2],
    /// World metres per screen pixel; smaller = more zoomed in.
    pub meters_per_pixel: f64,
    /// Viewport size in pixels.
    pub viewport: [f64; 2],
}

impl Camera {
    pub fn new(center: [f64; 2], meters_per_pixel: f64, viewport: [f64; 2]) -> Self {
        Self { center, meters_per_pixel, viewport }
    }

    /// Fit `[min_x, min_y, max_x, max_y]` world bounds into the viewport with a
    /// pixel margin.
    pub fn fit_bounds(bounds: [f64; 4], viewport: [f64; 2], margin_px: f64) -> Self {
        let (w, h) = (bounds[2] - bounds[0], bounds[3] - bounds[1]);
        let usable = [(viewport[0] - 2.0 * margin_px).max(1.0), (viewport[1] - 2.0 * margin_px).max(1.0)];
        let mpp = (w / usable[0]).max(h / usable[1]).max(1e-6);
        Self::new([(bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0], mpp, viewport)
    }

    pub fn world_to_screen(&self, p: [f64; 2]) -> [f64; 2] {
        [
            self.viewport[0] / 2.0 + (p[0] - self.center[0]) / self.meters_per_pixel,
            self.viewport[1] / 2.0 - (p[1] - self.center[1]) / self.meters_per_pixel,
        ]
    }

    pub fn screen_to_world(&self, s: [f64; 2]) -> [f64; 2] {
        [
            self.center[0] + (s[0] - self.viewport[0] / 2.0) * self.meters_per_pixel,
            self.center[1] - (s[1] - self.viewport[1] / 2.0) * self.meters_per_pixel,
        ]
    }

    /// Pan by a screen-pixel delta (drag).
    pub fn pan_pixels(&mut self, dx: f64, dy: f64) {
        self.center[0] -= dx * self.meters_per_pixel;
        self.center[1] += dy * self.meters_per_pixel;
    }

    /// Zoom by `factor` (<1 zooms in) keeping the world point under `anchor`
    /// (a screen pixel) fixed — the standard scroll-to-cursor zoom.
    pub fn zoom_at(&mut self, factor: f64, anchor: [f64; 2]) {
        let before = self.screen_to_world(anchor);
        self.meters_per_pixel = (self.meters_per_pixel * factor).clamp(0.02, 10_000.0);
        let after = self.screen_to_world(anchor);
        self.center[0] += before[0] - after[0];
        self.center[1] += before[1] - after[1];
    }

    /// Ease the centre toward a world `target` for a follow camera: a frame-rate
    /// independent exponential approach (time constant `tau` seconds), then a
    /// hard clamp so the target never trails farther than `max_offset_px` screen
    /// pixels from the centre — the car stays on screen however fast it moves.
    pub fn ease_toward(&mut self, target: [f64; 2], frame_dt: f64, tau: f64, max_offset_px: f64) {
        let k = 1.0 - (-frame_dt.max(0.0) / tau.max(1e-6)).exp();
        let mut c = [
            self.center[0] + (target[0] - self.center[0]) * k,
            self.center[1] + (target[1] - self.center[1]) * k,
        ];
        let max_off = max_offset_px * self.meters_per_pixel;
        let (dx, dy) = (target[0] - c[0], target[1] - c[1]);
        let d = dx.hypot(dy);
        if d > max_off {
            let s = (d - max_off) / d;
            c = [c[0] + dx * s, c[1] + dy * s];
        }
        self.center = c;
    }

    /// Visible world rectangle `[min_x, min_y, max_x, max_y]` for culling.
    pub fn visible_world_rect(&self) -> [f64; 4] {
        let hw = self.viewport[0] / 2.0 * self.meters_per_pixel;
        let hh = self.viewport[1] / 2.0 * self.meters_per_pixel;
        [self.center[0] - hw, self.center[1] - hh, self.center[0] + hw, self.center[1] + hh]
    }

    /// Orthographic world→NDC, column-major for wgpu (y up in NDC).
    pub fn view_proj(&self) -> Mat4 {
        let sx = (2.0 / (self.viewport[0] * self.meters_per_pixel)) as f32;
        let sy = (2.0 / (self.viewport[1] * self.meters_per_pixel)) as f32;
        let (cx, cy) = (self.center[0] as f32, self.center[1] as f32);
        [
            sx, 0.0, 0.0, 0.0,
            0.0, sy, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            -cx * sx, -cy * sy, 0.0, 1.0,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam() -> Camera {
        Camera::new([100.0, 50.0], 0.5, [800.0, 600.0])
    }

    #[test]
    fn center_maps_to_viewport_centre() {
        let s = cam().world_to_screen([100.0, 50.0]);
        assert!((s[0] - 400.0).abs() < 1e-9 && (s[1] - 300.0).abs() < 1e-9);
    }

    #[test]
    fn world_screen_round_trips() {
        let c = cam();
        for p in [[0.0, 0.0], [250.0, -30.0], [100.0, 50.0]] {
            let back = c.screen_to_world(c.world_to_screen(p));
            assert!((back[0] - p[0]).abs() < 1e-6 && (back[1] - p[1]).abs() < 1e-6);
        }
    }

    #[test]
    fn zoom_at_keeps_the_anchor_world_point_fixed() {
        let mut c = cam();
        let anchor = [650.0, 120.0];
        let before = c.screen_to_world(anchor);
        c.zoom_at(0.5, anchor);
        let after = c.screen_to_world(anchor);
        assert!((before[0] - after[0]).abs() < 1e-6 && (before[1] - after[1]).abs() < 1e-6);
    }

    #[test]
    fn view_proj_sends_center_to_ndc_origin() {
        let m = cam().view_proj();
        let (cx, cy) = (100.0f32, 50.0f32);
        let ndc_x = m[0] * cx + m[12];
        let ndc_y = m[5] * cy + m[13];
        assert!(ndc_x.abs() < 1e-5 && ndc_y.abs() < 1e-5);
    }

    #[test]
    fn visible_rect_scales_with_zoom() {
        let mut c = cam();
        let w0 = c.visible_world_rect();
        c.meters_per_pixel *= 2.0;
        let w1 = c.visible_world_rect();
        assert!((w1[2] - w1[0]) > (w0[2] - w0[0]));
    }

    #[test]
    fn ease_toward_converges_on_a_still_target() {
        let mut c = cam();
        let target = [400.0, -200.0];
        for _ in 0..600 {
            c.ease_toward(target, 1.0 / 60.0, 0.18, 90.0);
        }
        assert!((c.center[0] - target[0]).abs() < 1e-3 && (c.center[1] - target[1]).abs() < 1e-3);
    }

    #[test]
    fn ease_toward_moves_partway_in_one_frame() {
        // A single frame should close some of the gap but not all of it (the
        // smoothing) when the target sits within the pixel clamp.
        let mut c = Camera::new([0.0, 0.0], 0.5, [800.0, 600.0]);
        let target = [10.0, 0.0]; // 20 px away, inside the 90 px window
        c.ease_toward(target, 1.0 / 60.0, 0.18, 90.0);
        assert!(c.center[0] > 0.0 && c.center[0] < 10.0, "eased partway, got {}", c.center[0]);
    }

    #[test]
    fn ease_toward_clamps_a_fast_target_within_the_pixel_window() {
        // A far jump (target way outside the window) is clamped so the target is
        // at most max_offset_px * mpp from the centre — the car can't run off.
        let mut c = Camera::new([0.0, 0.0], 2.0, [800.0, 600.0]);
        let target = [100_000.0, 0.0];
        c.ease_toward(target, 1.0 / 60.0, 0.18, 90.0);
        let off = (target[0] - c.center[0]).hypot(target[1] - c.center[1]);
        assert!(off <= 90.0 * 2.0 + 1e-6, "target within the clamp window, off={off}");
    }

    #[test]
    fn fit_bounds_centers_and_contains() {
        let c = Camera::fit_bounds([0.0, 0.0, 400.0, 200.0], [800.0, 600.0], 20.0);
        assert!((c.center[0] - 200.0).abs() < 1e-9 && (c.center[1] - 100.0).abs() < 1e-9);
        let r = c.visible_world_rect();
        assert!(r[0] <= 0.0 && r[1] <= 0.0 && r[2] >= 400.0 && r[3] >= 200.0);
    }
}
