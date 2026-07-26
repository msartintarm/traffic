//! Tunables as plain data, injected into the sim rather than hardcoded, so
//! scenarios, Millbrae calibration, and per-driver heterogeneity are all just
//! different values. Mirrors `../plant`'s `config.rs`.

use super::rng::{self, Stream};

/// IDM car-following parameters. Fields carry SI units: speeds m/s,
/// accelerations m/s², headway s, gaps/length m. Defaults are Treiber's
/// passenger-car values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DriverConfig {
    pub desired_speed: f64,
    pub time_headway: f64,
    pub max_accel: f64,
    pub comfort_decel: f64,
    pub accel_exponent: f64,
    pub min_gap: f64,
    pub vehicle_length: f64,
}

impl DriverConfig {
    pub const fn car() -> Self {
        Self {
            desired_speed: 30.0,
            time_headway: 1.5,
            max_accel: 1.0,
            comfort_decel: 1.5,
            accel_exponent: 4.0,
            min_gap: 2.0,
            vehicle_length: 5.0,
        }
    }

    /// Treats `self` as the population mean and draws a heterogeneous driver
    /// via the stateless per-agent RNG; physical length is not jittered.
    pub fn sample(&self, seed: u64, agent_id: u32) -> Self {
        let jitter = |lo: f64, hi: f64| {
            rng::uniform_range(seed, agent_id, 0, Stream::DriverProfile, lo, hi)
        };
        Self {
            desired_speed: self.desired_speed * jitter(0.85, 1.15),
            time_headway: self.time_headway * jitter(0.7, 1.3),
            max_accel: self.max_accel * jitter(0.75, 1.25),
            comfort_decel: self.comfort_decel * jitter(0.75, 1.25),
            ..*self
        }
    }

    pub fn capped_to(&self, speed_limit: f64) -> Self {
        Self {
            desired_speed: self.desired_speed.min(speed_limit),
            ..*self
        }
    }
}

/// `dt` is the fixed timestep in seconds (IDM is stable around 0.1–0.25 s);
/// `seed` feeds the counter-based RNG.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimConfig {
    pub dt: f64,
    pub seed: u64,
}

impl SimConfig {
    pub const fn default_config() -> Self {
        Self { dt: 0.2, seed: 0xC0FFEE }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampled_driver_stays_in_expected_envelope() {
        let base = DriverConfig::car();
        for id in 0..5000u32 {
            let d = base.sample(1, id);
            assert!(d.desired_speed >= base.desired_speed * 0.85);
            assert!(d.desired_speed <= base.desired_speed * 1.15);
            assert!(d.time_headway >= base.time_headway * 0.7);
            assert!(d.time_headway <= base.time_headway * 1.3);
            assert!(d.max_accel > 0.0 && d.comfort_decel > 0.0);
            // Length is not jittered.
            assert_eq!(d.vehicle_length, base.vehicle_length);
        }
    }

    #[test]
    fn population_mean_tracks_the_base() {
        let base = DriverConfig::car();
        let n = 20_000u32;
        let mean_v0: f64 =
            (0..n).map(|id| base.sample(9, id).desired_speed).sum::<f64>() / n as f64;
        // Symmetric ±15% jitter → population mean ≈ base desired speed.
        assert!((mean_v0 - base.desired_speed).abs() < 0.2, "mean_v0={mean_v0}");
    }

    #[test]
    fn capped_to_never_raises_desired_speed() {
        let slow = DriverConfig { desired_speed: 10.0, ..DriverConfig::car() };
        assert_eq!(slow.capped_to(20.0).desired_speed, 10.0);
        let fast = DriverConfig { desired_speed: 40.0, ..DriverConfig::car() };
        assert_eq!(fast.capped_to(20.0).desired_speed, 20.0);
    }
}
