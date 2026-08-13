//! Tunables as plain data, injected into the sim rather than hardcoded, so
//! scenarios, Millbrae calibration, and per-driver heterogeneity are all just
//! different values. Mirrors `../plant`'s `config.rs`.

use super::rng::{self, Stream};

/// IDM car-following parameters. Fields carry SI units: speeds m/s,
/// accelerations m/s², headway s, gaps/length m. Defaults are Treiber's
/// passenger-car values recalibrated to observed US freeway queue discharge:
/// T = 1.2 s / a = 1.5 / b = 2.0 puts lane capacity at ≈ 2,130 veh/h/ln
/// (US-101 discharges 2,000–2,300), where the original T = 1.5 s capped it
/// near 1,780 — a road that jammed ~20% too early. Reaction 0.7 s is the
/// empirical brake PRT mean; a and b also govern the urban launch and the
/// stop-line service time the junction fixtures measure.
// `#[repr(C)]` + `Pod` so it can be embedded in the flat, GPU-uploadable
// per-vehicle accel context (`net_world::VehicleContext`); all fields are `f64`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DriverConfig {
    pub desired_speed: f64,
    pub time_headway: f64,
    pub max_accel: f64,
    pub comfort_decel: f64,
    pub accel_exponent: f64,
    pub min_gap: f64,
    pub vehicle_length: f64,
    /// Perception–reaction delay in seconds: the driver responds to the traffic
    /// situation as it was this long ago (0 = the idealized instantaneous IDM).
    pub reaction_time: f64,
    /// Half-amplitude (m/s²) of zero-mean acceleration jitter modelling imperfect
    /// throttle; fluctuates speed around the desired value without biasing it.
    pub accel_noise: f64,
    /// MOBIL politeness: how much this driver weighs the braking it would impose
    /// on others when changing lanes (0 = selfish, 1 = very considerate).
    pub politeness: f64,
    /// Critical gap (s): the smallest time-to-arrival of priority traffic this
    /// driver will accept when entering an unsignalized intersection. Shrinks with
    /// waiting (impatience).
    pub critical_gap: f64,
}

impl DriverConfig {
    pub const fn car() -> Self {
        Self {
            desired_speed: 30.0,
            time_headway: 1.2,
            // Mid comfortable range (≈1.5–2.5): tuned against the signalized
            // queue-discharge test (`queue_discharge_hits_real_saturation_flow`)
            // — launch kinematics are the accel-sensitive half of saturation
            // flow, and 1.5 measurably under-discharged every stop line.
            max_accel: 2.0,
            comfort_decel: 2.0,
            accel_exponent: 4.0,
            min_gap: 2.0,
            vehicle_length: 5.0,
            reaction_time: 0.7,
            accel_noise: 0.2,
            politeness: 0.3,
            critical_gap: 4.0,
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
            reaction_time: self.reaction_time * jitter(0.7, 1.3),
            critical_gap: self.critical_gap * jitter(0.8, 1.2),
            ..*self
        }
    }

    /// The driver's desired speed on a road with this posted limit. Real free-flow
    /// traffic runs *over* the limit — Californian freeway means sit ~3–7% above
    /// posted with aggressive tails near +15 mph — so instead of a hard clamp the
    /// limit scales by the driver's aggression (their sampled desired-speed ratio,
    /// clamped 0.90–1.20), bounded by the absolute +15 mph envelope and the
    /// driver's own open-road preference. Slow classes (governed trucks) land at
    /// ~90% of the limit.
    pub fn capped_to(&self, speed_limit: f64) -> Self {
        let target = speed_limit * (1.05 * (self.desired_speed / 30.0)).clamp(0.90, 1.20);
        Self {
            desired_speed: self.desired_speed.min(target.min(speed_limit + 6.7)),
            ..*self
        }
    }
}

/// Vehicle class: distinct physical size and driving envelope. Behaviour is a
/// [`DriverConfig`] preset; larger classes accelerate/brake more gently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VehicleClass {
    Car,
    Truck,
    Bus,
}

impl VehicleClass {
    pub fn driver(self) -> DriverConfig {
        match self {
            VehicleClass::Car => DriverConfig::car(),
            VehicleClass::Truck => DriverConfig {
                desired_speed: 25.0,
                time_headway: 1.8,
                max_accel: 0.8,
                comfort_decel: 1.3,
                vehicle_length: 10.0,
                ..DriverConfig::car()
            },
            VehicleClass::Bus => DriverConfig {
                desired_speed: 22.0,
                time_headway: 1.6,
                max_accel: 0.9,
                comfort_decel: 1.4,
                vehicle_length: 12.0,
                ..DriverConfig::car()
            },
        }
    }

    pub fn width(self) -> f64 {
        match self {
            VehicleClass::Car => 2.0,
            VehicleClass::Truck => 2.5,
            VehicleClass::Bus => 2.55,
        }
    }

    /// Classify a vehicle by its length so the renderer can size/colour it from
    /// the per-vehicle `DriverConfig` alone (no extra stored field).
    pub fn from_length(length: f64) -> VehicleClass {
        if length >= 11.0 {
            VehicleClass::Bus
        } else if length >= 8.0 {
            VehicleClass::Truck
        } else {
            VehicleClass::Car
        }
    }
}

/// `dt` is the fixed timestep in seconds (IDM is stable around 0.1–0.25 s);
/// `seed` feeds the counter-based RNG.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimConfig {
    pub dt: f64,
    pub seed: u64,
    /// Active-set scheduler: when set, vehicles queued behind a stopped leader skip the
    /// full gather each tick (their accel is leader-dominated), so per-tick work tracks
    /// the *deciding* fraction rather than the whole fleet. Off by default; the all-cars
    /// step stays the behavioural reference it is A/B'd against.
    pub sleep_scheduler: bool,
    /// How long a wreck stays on the road blocking traffic before it is cleared
    /// (seconds). Zero — the default — removes crashed vehicles instantly; setting
    /// it opts in to post-crash obstruction (queues form behind the wreck).
    pub wreck_clear_secs: f64,
    /// Base probability that a driver is blind to one particular signal (a
    /// distraction model, decided once per vehicle–node pair) and so runs its red.
    /// Scaled by driver aggression; zero disables red-running entirely.
    pub red_run_prob: f64,
}

impl SimConfig {
    pub const fn default_config() -> Self {
        Self { dt: 0.2, seed: 0xC0FFEE, sleep_scheduler: false, wreck_clear_secs: 0.0, red_run_prob: 0.002 }
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
    fn heavier_classes_are_slower_and_brake_more_gently() {
        let car = VehicleClass::Car.driver();
        let truck = VehicleClass::Truck.driver();
        let bus = VehicleClass::Bus.driver();
        assert!(truck.vehicle_length > car.vehicle_length);
        assert!(bus.vehicle_length > truck.vehicle_length);
        assert!(truck.max_accel < car.max_accel && truck.comfort_decel < car.comfort_decel);
        assert!(truck.desired_speed < car.desired_speed);
    }

    #[test]
    fn class_is_recovered_from_length() {
        assert_eq!(VehicleClass::from_length(VehicleClass::Car.driver().vehicle_length), VehicleClass::Car);
        assert_eq!(VehicleClass::from_length(VehicleClass::Truck.driver().vehicle_length), VehicleClass::Truck);
        assert_eq!(VehicleClass::from_length(VehicleClass::Bus.driver().vehicle_length), VehicleClass::Bus);
    }

    #[test]
    fn trucks_need_more_distance_to_stop() {
        // Steady-following gap at 20 m/s is larger for a truck (gentler braking).
        let v = 20.0;
        let car_gap = super::super::idm::equilibrium_gap(&VehicleClass::Car.driver(), v);
        let truck_gap = super::super::idm::equilibrium_gap(&VehicleClass::Truck.driver(), v);
        assert!(truck_gap > car_gap, "truck {truck_gap} vs car {car_gap}");
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
    fn capped_to_speeds_realistically_over_the_limit() {
        // A slow driver keeps their own preference.
        let slow = DriverConfig { desired_speed: 10.0, ..DriverConfig::car() };
        assert_eq!(slow.capped_to(20.0).desired_speed, 10.0);
        // The mean driver runs ~5% over the posted limit, never at the hard clamp.
        let mean = DriverConfig::car().capped_to(20.0).desired_speed;
        assert!((mean - 21.0).abs() < 0.01, "mean driver ≈ +5% over, got {mean}");
        // An aggressive driver exceeds further but stays inside +15 mph absolute.
        let fast = DriverConfig { desired_speed: 40.0, ..DriverConfig::car() };
        let v = fast.capped_to(29.0).desired_speed;
        assert!(v > 29.0 && v <= 29.0 + 6.7 + 1e-9, "aggressive over the limit within the envelope: {v}");
        // The population mean on a freeway limit lands ~3–7% over posted.
        let base = DriverConfig::car();
        let n = 10_000u32;
        let mean_v: f64 = (0..n).map(|id| base.sample(4, id).capped_to(29.0).desired_speed).sum::<f64>() / n as f64;
        assert!((29.6..30.9).contains(&mean_v), "population free-flow mean ≈ +3–7% over 29 m/s, got {mean_v}");
        // Governed heavy classes sit under the limit.
        let truck = VehicleClass::Truck.driver().capped_to(29.0).desired_speed;
        assert!(truck < 29.0 * 0.95, "trucks run below the limit: {truck}");
    }
}
