//! Intelligent Driver Model (Treiber, Hennecke & Helbing, 2000), the pure
//! collision-free car-following law of the micro layer:
//!
//! ```text
//!   s*(v, Δv) = s0 + max(0, v·T + v·Δv / (2·√(a·b)))
//!   dv/dt     = a · ( 1 − (v/v0)^δ − (s*/s)² )
//! ```
//!
//! `s` is the bumper-to-bumper gap and `Δv = v − v_lead`. No state or RNG here,
//! so it is unit-tested alone and reused unchanged by a future WGSL port.

use super::config::DriverConfig;

pub fn acceleration(cfg: &DriverConfig, v: f64, delta_v: f64, gap: f64) -> f64 {
    let s = gap.max(1e-3);
    let free = 1.0 - (v / cfg.desired_speed).powf(cfg.accel_exponent);
    let s_star = cfg.min_gap
        + (v * cfg.time_headway
            + (v * delta_v) / (2.0 * (cfg.max_accel * cfg.comfort_decel).sqrt()))
            .max(0.0);
    let interaction = (s_star / s).powi(2);
    cfg.max_accel * (free - interaction)
}

/// Free-road acceleration: the limiting behaviour when the leader is
/// infinitely far away. Handy for tests and for the leaderless case.
pub fn free_acceleration(cfg: &DriverConfig, v: f64) -> f64 {
    cfg.max_accel * (1.0 - (v / cfg.desired_speed).powf(cfg.accel_exponent))
}

/// Steady-state gap where IDM acceleration is zero at `Δv = 0`; the
/// microscopic root of the fundamental diagram.
pub fn equilibrium_gap(cfg: &DriverConfig, v: f64) -> f64 {
    let free = 1.0 - (v / cfg.desired_speed).powf(cfg.accel_exponent);
    if free <= 0.0 {
        return f64::INFINITY;
    }
    let s_star = cfg.min_gap + v * cfg.time_headway;
    s_star / free.sqrt()
}

/// Inverse of [`equilibrium_gap`] by bisection: the uniform-flow speed a
/// homogeneous stream settles to when every vehicle holds bumper-to-bumper `gap`.
pub fn equilibrium_speed(cfg: &DriverConfig, gap: f64) -> f64 {
    if gap <= cfg.min_gap {
        return 0.0;
    }
    let (mut lo, mut hi) = (0.0, cfg.desired_speed);
    if equilibrium_gap(cfg, hi - 1e-6) <= gap {
        return cfg.desired_speed;
    }
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if equilibrium_gap(cfg, mid) < gap {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn car() -> DriverConfig {
        DriverConfig::car()
    }

    #[test]
    fn accelerates_from_rest_on_a_free_road() {
        let a = free_acceleration(&car(), 0.0);
        assert!((a - car().max_accel).abs() < 1e-9, "a={a}");
    }

    #[test]
    fn no_acceleration_at_desired_speed_on_free_road() {
        let a = free_acceleration(&car(), car().desired_speed);
        assert!(a.abs() < 1e-9, "a={a}");
    }

    #[test]
    fn brakes_hard_when_gap_collapses() {
        // Fast follower almost touching a stopped leader must decelerate.
        let a = acceleration(&car(), 20.0, 20.0, 1.0);
        assert!(a < 0.0, "expected braking, got a={a}");
    }

    #[test]
    fn free_road_recovered_at_huge_gap() {
        let cfg = car();
        let a_far = acceleration(&cfg, 10.0, 0.0, 1.0e6);
        let a_free = free_acceleration(&cfg, 10.0);
        assert!((a_far - a_free).abs() < 1e-3, "a_far={a_far} a_free={a_free}");
    }

    #[test]
    fn equilibrium_gap_zeroes_the_acceleration() {
        let cfg = car();
        for &v in &[1.0, 5.0, 10.0, 20.0, 29.0] {
            let s = equilibrium_gap(&cfg, v);
            let a = acceleration(&cfg, v, 0.0, s);
            assert!(a.abs() < 1e-6, "v={v} s={s} a={a}");
        }
    }

    #[test]
    fn equilibrium_gap_grows_with_speed() {
        let cfg = car();
        let mut last = 0.0;
        for &v in &[1.0, 5.0, 10.0, 15.0, 20.0, 25.0] {
            let s = equilibrium_gap(&cfg, v);
            assert!(s > last, "gap should increase with speed: v={v} s={s}");
            last = s;
        }
    }

    #[test]
    fn equilibrium_gap_infinite_at_desired_speed() {
        assert!(equilibrium_gap(&car(), car().desired_speed).is_infinite());
    }

    #[test]
    fn equilibrium_speed_inverts_equilibrium_gap() {
        let cfg = car();
        for &v in &[2.0, 8.0, 15.0, 25.0] {
            let s = equilibrium_gap(&cfg, v);
            assert!((equilibrium_speed(&cfg, s) - v).abs() < 1e-3);
        }
    }

    #[test]
    fn equilibrium_speed_pins_at_zero_below_jam_gap() {
        assert_eq!(equilibrium_speed(&car(), car().min_gap - 0.1), 0.0);
    }
}
