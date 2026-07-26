//! Single closed-ring micro world used to validate emergent flow/density
//! against theory (the fundamental-diagram test). Accelerations read committed
//! state and apply in a second pass, matching the GPU port's double buffering.
//! Graph-based multi-lane roads with signals live in [`super::net_world`].

use super::clock::SimClock;
use super::config::{DriverConfig, SimConfig};
use super::idm;
use super::vehicle::Vehicle;

pub struct World {
    pub cfg: SimConfig,
    pub clock: SimClock,
    ring_length: f64,
    vehicles: Vec<Vehicle>,
}

impl World {
    pub fn ring_road(cfg: SimConfig, ring_length: f64, vehicles: Vec<Vehicle>) -> Self {
        let clock = SimClock::new(&cfg);
        let mut world = Self { cfg, clock, ring_length, vehicles };
        world.sort_by_position();
        world
    }

    pub fn uniform_ring(cfg: SimConfig, ring_length: f64, count: u32, driver: DriverConfig) -> Self {
        let spacing = ring_length / count as f64;
        let vehicles = (0..count)
            .map(|i| Vehicle::new(i, i as f64 * spacing, 0.0, driver))
            .collect();
        Self::ring_road(cfg, ring_length, vehicles)
    }

    pub fn vehicles(&self) -> &[Vehicle] {
        &self.vehicles
    }

    pub fn ring_length(&self) -> f64 {
        self.ring_length
    }

    fn sort_by_position(&mut self) {
        self.vehicles.sort_by(|a, b| a.position.total_cmp(&b.position));
    }

    pub fn step(&mut self) {
        let dt = self.cfg.dt;
        let n = self.vehicles.len();
        if n == 0 {
            return;
        }
        let accels = self.accelerations();
        for (v, a) in self.vehicles.iter_mut().zip(accels) {
            integrate(v, a, dt, self.ring_length);
        }
        self.sort_by_position();
    }

    fn accelerations(&self) -> Vec<f64> {
        let n = self.vehicles.len();
        (0..n)
            .map(|i| {
                let me = &self.vehicles[i];
                if n == 1 {
                    return idm::free_acceleration(&me.driver, me.speed);
                }
                let leader = &self.vehicles[(i + 1) % n];
                let raw = leader.position - me.position;
                let center_to_center = if raw > 0.0 { raw } else { raw + self.ring_length };
                let gap = center_to_center - me.driver.vehicle_length;
                idm::acceleration(&me.driver, me.speed, me.speed - leader.speed, gap)
            })
            .collect()
    }

    pub fn density(&self) -> f64 {
        self.vehicles.len() as f64 / self.ring_length
    }

    pub fn mean_speed(&self) -> f64 {
        if self.vehicles.is_empty() {
            return 0.0;
        }
        self.vehicles.iter().map(|v| v.speed).sum::<f64>() / self.vehicles.len() as f64
    }

    pub fn flow(&self) -> f64 {
        self.density() * self.mean_speed()
    }

    pub fn run_ticks(&mut self, ticks: u32) {
        for _ in 0..ticks {
            self.step();
        }
    }
}

fn integrate(v: &mut Vehicle, accel: f64, dt: f64, ring_length: f64) {
    if v.speed + accel * dt < 0.0 {
        v.position += -0.5 * v.speed * v.speed / accel;
        v.speed = 0.0;
    } else {
        v.position += v.speed * dt + 0.5 * accel * dt * dt;
        v.speed += accel * dt;
    }
    v.position = v.position.rem_euclid(ring_length);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> SimConfig {
        SimConfig::default_config()
    }

    #[test]
    fn lone_vehicle_reaches_desired_speed() {
        let mut w = World::uniform_ring(cfg(), 100_000.0, 1, DriverConfig::car());
        w.run_ticks(2000);
        assert!((w.mean_speed() - DriverConfig::car().desired_speed).abs() < 0.1);
    }

    #[test]
    fn no_collisions_form_on_a_dense_ring() {
        let driver = DriverConfig::car();
        let mut w = World::uniform_ring(cfg(), 500.0, 40, driver);
        w.run_ticks(5000);
        let vs = w.vehicles();
        for i in 0..vs.len() {
            let a = &vs[i];
            let b = &vs[(i + 1) % vs.len()];
            let raw = b.position - a.position;
            let c2c = if raw > 0.0 { raw } else { raw + w.ring_length() };
            assert!(c2c >= driver.vehicle_length - 1e-6, "overlap: c2c={c2c}");
        }
    }

    #[test]
    fn steady_state_matches_theoretical_equilibrium_speed() {
        let driver = DriverConfig::car();
        let length = 1000.0;
        for &count in &[10u32, 30, 60, 100] {
            let mut w = World::uniform_ring(cfg(), length, count, driver);
            w.run_ticks(8000);
            let spacing = length / count as f64;
            let expected = idm::equilibrium_speed(&driver, spacing - driver.vehicle_length);
            assert!(
                (w.mean_speed() - expected).abs() < 0.3,
                "count={count} mean={} expected={expected}",
                w.mean_speed()
            );
        }
    }

    #[test]
    fn fundamental_diagram_has_free_flow_capacity_and_jam_branches() {
        let driver = DriverConfig::car();
        let length = 1000.0;
        let mut curve = Vec::new();
        for count in (5..=180).step_by(5) {
            let mut w = World::uniform_ring(cfg(), length, count, driver);
            w.run_ticks(8000);
            curve.push((w.density(), w.flow(), w.mean_speed()));
        }

        let (low_k, _, low_v) = curve[0];
        assert!(low_v > driver.desired_speed * 0.9, "expected free flow at low density");

        let (_, peak_q, _) = curve
            .iter()
            .copied()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        let (_, first_q, _) = curve[0];
        let (_, last_q, last_v) = *curve.last().unwrap();
        assert!(peak_q > first_q * 1.5, "capacity peak should exceed free-flow-tail flow");
        assert!(peak_q > last_q * 1.5, "capacity peak should exceed jammed flow");
        assert!(last_v < low_v * 0.5, "stream should be congested at high density");

        let capacity_k = curve.iter().copied().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
        assert!(capacity_k > low_k, "capacity occurs at an interior density");
    }
}
