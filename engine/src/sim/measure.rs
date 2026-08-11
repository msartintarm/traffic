//! Measurement windows and corridor probes — the sim's own instruments, so its
//! output can be scored against independent data (Caltrans counts, PeMS speeds,
//! travel times) instead of only feeding real data in. A [`Measurement`] samples
//! a [`NetWorld`] from outside (no coupling into the step), accumulating windowed
//! per-link flows and speeds plus vehicle-kilometres and crashes; corridor probes
//! turn live link costs into an expected end-to-end travel time.

use super::net_world::NetWorld;
use super::network::LinkId;

/// One open measurement window over a world: snapshot at `begin`, then call
/// [`sample`](Self::sample) once per sim tick. Flows come from differencing the
/// world's cumulative entry counters; speeds and VKT from per-tick fleet sums.
pub struct Measurement {
    start_time: f64,
    entries0: Vec<u32>,
    crashed0: u32,
    speed_sum: Vec<f64>,
    speed_n: Vec<u64>,
    vkt_m: f64,
}

impl Measurement {
    pub fn begin(world: &NetWorld) -> Self {
        let n = world.network.links.len();
        Self {
            start_time: world.time(),
            entries0: world.link_entry_counts().to_vec(),
            crashed0: world.crashed(),
            speed_sum: vec![0.0; n],
            speed_n: vec![0; n],
            vkt_m: 0.0,
        }
    }

    /// Accumulate one tick: per-link speed samples and fleet distance (Σ v·dt).
    pub fn sample(&mut self, world: &NetWorld, dt: f64) {
        for v in world.vehicles() {
            let link = world.network.lane(v.lane).link.idx();
            self.speed_sum[link] += v.speed;
            self.speed_n[link] += 1;
            self.vkt_m += v.speed * dt;
        }
    }

    pub fn elapsed_hours(&self, world: &NetWorld) -> f64 {
        (world.time() - self.start_time) / 3600.0
    }

    /// Windowed flow (vehicles/hour) per link: entries during the window over its
    /// duration — the number a peak-hour count target compares against.
    pub fn link_flows(&self, world: &NetWorld) -> Vec<f64> {
        let hours = self.elapsed_hours(world).max(1e-9);
        world
            .link_entry_counts()
            .iter()
            .zip(&self.entries0)
            .map(|(&now, &then)| (now.saturating_sub(then)) as f64 / hours)
            .collect()
    }

    /// Time-mean speed (m/s) per link over the window; `NAN` where nothing drove.
    pub fn link_speeds(&self) -> Vec<f64> {
        self.speed_sum
            .iter()
            .zip(&self.speed_n)
            .map(|(&s, &n)| if n > 0 { s / n as f64 } else { f64::NAN })
            .collect()
    }

    /// Vehicle-kilometres travelled during the window.
    pub fn vkt(&self) -> f64 {
        self.vkt_m / 1000.0
    }

    /// Vehicle-miles travelled — the unit US crash rates are quoted in.
    pub fn vmt(&self) -> f64 {
        self.vkt_m / 1609.344
    }

    pub fn crashes(&self, world: &NetWorld) -> u32 {
        world.crashed().saturating_sub(self.crashed0)
    }

    /// Crashes per 100 million vehicle-miles — comparable to published (SWITRS /
    /// NHTSA) rates. `NAN` until some distance has accumulated.
    pub fn crashes_per_100m_vmt(&self, world: &NetWorld) -> f64 {
        let vmt = self.vmt();
        if vmt <= 0.0 {
            return f64::NAN;
        }
        self.crashes(world) as f64 / (vmt / 1.0e8)
    }
}

/// Expected travel time (seconds) from the start of `from` to the end of `to`
/// along the currently cheapest route, using the live congestion-inflated link
/// costs — the corridor probe a navigation ETA is. `None` if unroutable.
pub fn corridor_travel_secs(world: &NetWorld, from: LinkId, to: LinkId) -> Option<f64> {
    let costs = world.live_link_costs();
    let route = world.network.route_links_with_costs(from, to, &costs)?;
    Some(route.iter().map(|l| costs[l.idx()] as f64).sum::<f64>() / 1000.0)
}

/// Free-flow travel time (seconds) over the same routing, with uncongested link
/// times as both the route weights and the sum — the probe's denominator.
pub fn free_flow_travel_secs(world: &NetWorld, from: LinkId, to: LinkId) -> Option<f64> {
    let base: Vec<u64> =
        (0..world.network.links.len() as u32).map(|i| world.network.link_travel_time_ms(LinkId(i))).collect();
    let route = world.network.route_links_with_costs(from, to, &base)?;
    Some(route.iter().map(|l| base[l.idx()] as f64).sum::<f64>() / 1000.0)
}

/// The GEH statistic comparing a modelled hourly flow `m` against a counted one
/// `c` — the standard calibration acceptance measure (GEH < 5 on ≥ 85% of links).
pub fn geh(m: f64, c: f64) -> f64 {
    if m + c <= 0.0 {
        return 0.0;
    }
    (2.0 * (m - c).powi(2) / (m + c)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::config::{DriverConfig, SimConfig};
    use crate::sim::map::{LinkSpec, NodeSpec, OsmMap};

    fn two_link_world() -> NetWorld {
        let map = OsmMap {
            nodes: vec![
                NodeSpec::uncontrolled(1, 0.0, 0.0),
                NodeSpec::uncontrolled(2, 500.0, 0.0),
                NodeSpec::uncontrolled(3, 1000.0, 0.0),
            ],
            links: vec![LinkSpec::oneway(1, 2, 1, 20.0), LinkSpec::oneway(2, 3, 1, 20.0)],
        };
        NetWorld::new(map.build(), SimConfig::default_config())
    }

    #[test]
    fn windowed_flow_counts_only_the_window() {
        let mut world = two_link_world();
        let d = DriverConfig::car();
        // One vehicle enters before the window opens; two after.
        assert!(world.spawn_routed(0, vec![LinkId(0), LinkId(1)], 15.0, d));
        let m = Measurement::begin(&world);
        for _ in 0..40 {
            world.step();
        }
        assert!(world.spawn_routed(1, vec![LinkId(0), LinkId(1)], 15.0, d));
        for _ in 0..40 {
            world.step();
        }
        assert!(world.spawn_routed(2, vec![LinkId(0), LinkId(1)], 15.0, d));
        let flows = m.link_flows(&world);
        let hours = m.elapsed_hours(&world);
        assert!((flows[0] * hours - 2.0).abs() < 1e-9, "two entries inside the window, got {}", flows[0] * hours);
    }

    #[test]
    fn speeds_and_vkt_accumulate_from_samples() {
        let mut world = two_link_world();
        // Desired 18 = the speeding model's steady cruise for this driver on the
        // fixture's 20 m/s limit, so the measured mean sits exactly there.
        let d = DriverConfig { desired_speed: 18.0, accel_noise: 0.0, ..DriverConfig::car() };
        assert!(world.spawn_routed(0, vec![LinkId(0), LinkId(1)], 18.0, d));
        let mut m = Measurement::begin(&world);
        let dt = SimConfig::default_config().dt;
        for _ in 0..50 {
            world.step();
            m.sample(&world, dt);
        }
        let speeds = m.link_speeds();
        assert!((speeds[0] - 18.0).abs() < 0.5, "cruise speed measured, got {}", speeds[0]);
        assert!(speeds[1].is_nan(), "nothing drove link 1 yet");
        // 50 ticks × 0.2 s × 18 m/s = 180 m.
        assert!((m.vkt() - 0.18).abs() < 0.01, "vkt {}", m.vkt());
        assert!(m.crashes(&world) == 0 && m.crashes_per_100m_vmt(&world) == 0.0);
    }

    #[test]
    fn corridor_probe_reads_free_flow_when_empty_and_inflates_when_jammed() {
        let world = two_link_world();
        let free = free_flow_travel_secs(&world, LinkId(0), LinkId(1)).unwrap();
        // 1000 m at 20 m/s = 50 s.
        assert!((free - 50.0).abs() < 1.0, "free-flow probe {free}");
        let live = corridor_travel_secs(&world, LinkId(0), LinkId(1)).unwrap();
        assert!((live - free).abs() < 1.0, "empty road probes at free flow");
    }

    #[test]
    fn geh_matches_the_standard_formula() {
        assert_eq!(geh(0.0, 0.0), 0.0);
        assert!((geh(1000.0, 1000.0)).abs() < 1e-12);
        // Canonical example: 700 vs 1000 → GEH ≈ 10.3.
        assert!((geh(700.0, 1000.0) - 10.29).abs() < 0.01);
        assert!(geh(950.0, 1000.0) < 5.0, "within-5 flows accept");
    }
}
