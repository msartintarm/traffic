//! Composable longitudinal constraints.
//!
//! A vehicle's acceleration is the *binding* (minimum) of candidate
//! accelerations produced by an ordered list of independent constraints — the
//! standard microsimulation pattern. Each constraint is a pure function of an
//! explicit [`LongContext`]; a non-binding constraint returns `+∞` so the fold
//! ignores it. This is the engine's accuracy dial: adding a behaviour (yield,
//! merge, curve speed, bus stop, cooperative gap) is appending one function,
//! and since a constraint can only ever *lower* the chosen acceleration, adding
//! constraints refines behaviour monotonically without touching the rest.
//!
//! Everything here is a pure `fn` with no state, heap, or dispatch, so the same
//! set inlines into a WGSL kernel as a sequence of `min()` calls when the micro
//! layer moves onto the GPU.

use super::config::DriverConfig;
use super::idm;
use super::rng::{self, Stream};

pub fn accel_noise(sigma: f64, seed: u64, agent_id: u32, tick: u64) -> f64 {
    sigma * (2.0 * rng::uniform01(seed, agent_id, tick, Stream::AccelNoise) - 1.0)
}

/// A leader (moving or stationary) ahead in the vehicle's path.
#[derive(Clone, Copy, Debug)]
pub struct Obstacle {
    pub gap: f64,
    pub speed: f64,
}

/// A speed the vehicle must be down to by `distance` metres ahead — a slower
/// downstream road, a curve advisory, a school zone.
#[derive(Clone, Copy, Debug)]
pub struct SpeedTarget {
    pub speed: f64,
    pub distance: f64,
}

/// Everything the constraints read. Extend it as new constraints need new
/// inputs; existing constraints are unaffected.
pub struct LongContext<'a> {
    /// Driver params already capped to the current road's speed limit.
    pub driver: &'a DriverConfig,
    pub speed: f64,
    pub leader: Option<Obstacle>,
    /// Distance to a stop line the vehicle must halt at (red signal).
    pub stop_line: Option<f64>,
    pub speed_target: Option<SpeedTarget>,
    /// Distance to a not-yet-satisfied stop sign's line.
    pub stop_sign: Option<f64>,
    /// Distance to a give-way line when priority traffic is too close to enter.
    pub yield_line: Option<f64>,
    /// A conflicting vehicle on a converging lane, projected onto this path.
    pub merge: Option<Obstacle>,
    /// Speed to slow to for an upcoming curve (lateral-acceleration limit).
    pub curve: Option<SpeedTarget>,
}

pub type Constraint = fn(&LongContext) -> f64;

/// The full constraint set, in no particular order (the fold is commutative).
/// Append to this as behaviours are added.
pub const DEFAULT: &[Constraint] = &[
    desired_speed,
    car_following,
    stop_line,
    downstream_speed,
    curve_speed,
    stop_sign,
    give_way,
    merge_yield,
];

/// The binding acceleration: the most restrictive constraint wins.
pub fn binding_acceleration(ctx: &LongContext, constraints: &[Constraint]) -> f64 {
    constraints.iter().map(|c| c(ctx)).fold(f64::INFINITY, f64::min)
}

/// Free-road pursuit of the (already speed-limited) desired speed. Always binds,
/// so the fold is never empty.
pub fn desired_speed(ctx: &LongContext) -> f64 {
    idm::free_acceleration(ctx.driver, ctx.speed)
}

pub fn car_following(ctx: &LongContext) -> f64 {
    match ctx.leader {
        Some(o) => idm::acceleration(ctx.driver, ctx.speed, ctx.speed - o.speed, o.gap),
        None => f64::INFINITY,
    }
}

pub fn stop_line(ctx: &LongContext) -> f64 {
    brake_to_line(ctx, ctx.stop_line)
}

/// Anticipatory braking so the vehicle reaches a target speed by the time it
/// arrives at the target distance — the constant-deceleration kinematics. Not
/// binding when already at or below the target.
fn brake_to_target(speed: f64, target: Option<SpeedTarget>) -> f64 {
    match target {
        Some(t) if speed > t.speed => (t.speed * t.speed - speed * speed) / (2.0 * t.distance.max(0.05)),
        _ => f64::INFINITY,
    }
}

/// Slow for a slower road ahead (speed-limit change / downstream link).
pub fn downstream_speed(ctx: &LongContext) -> f64 {
    brake_to_target(ctx.speed, ctx.speed_target)
}

/// Slow for an upcoming curve to the lateral-acceleration limit.
pub fn curve_speed(ctx: &LongContext) -> f64 {
    brake_to_target(ctx.speed, ctx.curve)
}

/// Mandatory halt at a stop sign: brake to a stationary line until the world
/// marks the sign satisfied (the vehicle has actually stopped there).
pub fn stop_sign(ctx: &LongContext) -> f64 {
    brake_to_line(ctx, ctx.stop_sign)
}

/// Priority yielding: hold at the give-way line while conflicting priority
/// traffic is within the critical gap.
pub fn give_way(ctx: &LongContext) -> f64 {
    brake_to_line(ctx, ctx.yield_line)
}

/// Cooperative (zipper) merge: follow the conflicting vehicle that reaches the
/// merge point first as if it were a leader, opening a gap rather than colliding.
pub fn merge_yield(ctx: &LongContext) -> f64 {
    match ctx.merge {
        Some(o) => idm::acceleration(ctx.driver, ctx.speed, ctx.speed - o.speed, o.gap),
        None => f64::INFINITY,
    }
}

fn brake_to_line(ctx: &LongContext, line: Option<f64>) -> f64 {
    match line {
        Some(d) => idm::acceleration(ctx.driver, ctx.speed, ctx.speed, d.max(0.05)),
        None => f64::INFINITY,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(speed: f64) -> LongContext<'static> {
        LongContext {
            driver: &CAR,
            speed,
            leader: None,
            stop_line: None,
            speed_target: None,
            stop_sign: None,
            yield_line: None,
            merge: None,
            curve: None,
        }
    }

    static CAR: DriverConfig = DriverConfig::car();

    #[test]
    fn desired_speed_always_binds_and_accelerates_from_rest() {
        assert!((desired_speed(&ctx(0.0)) - CAR.max_accel).abs() < 1e-9);
        assert!(car_following(&ctx(0.0)).is_infinite());
    }

    #[test]
    fn car_following_binds_below_free_road_when_closing() {
        let mut c = ctx(20.0);
        c.leader = Some(Obstacle { gap: 5.0, speed: 0.0 });
        assert!(car_following(&c) < desired_speed(&c));
    }

    #[test]
    fn stop_line_brakes_when_present() {
        let mut c = ctx(15.0);
        c.stop_line = Some(8.0);
        assert!(stop_line(&c) < 0.0);
        c.stop_line = None;
        assert!(stop_line(&c).is_infinite());
    }

    #[test]
    fn downstream_speed_decelerates_toward_a_slower_target() {
        let mut c = ctx(20.0);
        c.speed_target = Some(SpeedTarget { speed: 8.0, distance: 40.0 });
        let a = downstream_speed(&c);
        assert!(a < 0.0, "should brake, got {a}");
        // (8² − 20²)/(2·40) = −4.2 m/s²
        assert!((a - (-4.2)).abs() < 1e-9);
        c.speed = 5.0; // already slower than target
        assert!(downstream_speed(&c).is_infinite());
    }

    #[test]
    fn curve_speed_brakes_for_a_tight_bend() {
        let mut c = ctx(20.0);
        c.curve = Some(SpeedTarget { speed: 8.0, distance: 40.0 });
        assert!(curve_speed(&c) < 0.0, "should slow for the curve");
        c.speed = 6.0; // already under the curve speed
        assert!(curve_speed(&c).is_infinite());
    }

    #[test]
    fn stop_sign_and_give_way_brake_to_their_lines() {
        let mut c = ctx(12.0);
        c.stop_sign = Some(6.0);
        assert!(stop_sign(&c) < 0.0);
        c.stop_sign = None;
        assert!(stop_sign(&c).is_infinite());

        c.yield_line = Some(10.0);
        assert!(give_way(&c) < 0.0);
        c.yield_line = None;
        assert!(give_way(&c).is_infinite());
    }

    #[test]
    fn merge_yield_follows_the_conflicting_vehicle() {
        let mut c = ctx(18.0);
        assert!(merge_yield(&c).is_infinite());
        c.merge = Some(Obstacle { gap: 6.0, speed: 6.0 });
        assert!(merge_yield(&c) < desired_speed(&c));
    }

    #[test]
    fn accel_noise_is_zero_mean_two_sided_and_bounded() {
        let sigma = 0.3;
        let (mut sum, mut saw_pos, mut saw_neg) = (0.0, false, false);
        let n = 20_000u64;
        for tick in 0..n {
            let x = accel_noise(sigma, 7, 42, tick);
            assert!(x.abs() <= sigma + 1e-9, "bounded by sigma: {x}");
            sum += x;
            saw_pos |= x > 0.0;
            saw_neg |= x < 0.0;
        }
        assert!((sum / n as f64).abs() < 0.02, "zero-mean over time: {}", sum / n as f64);
        assert!(saw_pos && saw_neg, "noise is two-sided, not one-sided drag");
        assert_eq!(accel_noise(sigma, 7, 42, 5), accel_noise(sigma, 7, 42, 5), "reproducible");
    }

    #[test]
    fn binding_acceleration_takes_the_most_restrictive() {
        let mut c = ctx(20.0);
        c.leader = Some(Obstacle { gap: 4.0, speed: 0.0 });
        c.stop_line = Some(60.0);
        let a = binding_acceleration(&c, DEFAULT);
        assert!(a <= car_following(&c) && a <= stop_line(&c) && a <= desired_speed(&c));
    }
}
