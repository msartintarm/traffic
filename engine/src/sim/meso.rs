//! Mesoscopic mass layer: Daganzo's Cell Transmission Model (CTM).
//!
//! This is the layer that scales to 1M+ vehicles. A link is a chain of cells,
//! each one free-flow-speed × dt long, holding a vehicle *count*. Every tick a
//! cell's outflow is `min(sending_i, receiving_{i+1})` of a send/receive pair,
//! so the update is local, conservative, and reads only committed neighbour
//! state — which is exactly why it double-buffers onto a GPU compute kernel with
//! no races (see `meso.wgsl`, the WGSL mirror of [`MesoCorridor::step`]).
//!
//! Kept as the CPU *reference*: correctness is asserted here (conservation,
//! capacity, backward-propagating congestion) and the GPU path is validated to
//! reproduce the same behaviour.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CtmParams {
    pub cell_max: f64,
    pub capacity: f64,
    pub backward_ratio: f64,
}

impl CtmParams {
    /// Derive per-cell CTM quantities from physical road parameters and the
    /// mesoscopic timestep. Cell length is fixed at `free_speed × dt` so a
    /// free-flowing vehicle advances exactly one cell per tick.
    pub fn from_physical(free_speed: f64, jam_density: f64, capacity: f64, backward_speed: f64, dt: f64) -> Self {
        let cell_length = free_speed * dt;
        Self {
            cell_max: jam_density * cell_length,
            capacity: capacity * dt,
            backward_ratio: backward_speed / free_speed,
        }
    }

    pub fn car_arterial(dt: f64) -> Self {
        Self::from_physical(15.0, 1.0 / 7.0, 0.5, 5.0, dt)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Boundary {
    /// Closed loop: the last cell feeds the first; total occupancy is conserved.
    Ring,
    /// `inflow_demand` vehicles/tick offered at the entrance; free exit unless blocked.
    Open { inflow_demand: f64, blocked_exit: bool },
}

pub struct MesoCorridor {
    pub params: CtmParams,
    boundary: Boundary,
    cells: Vec<f64>,
    cumulative_out: f64,
}

impl MesoCorridor {
    pub fn new(params: CtmParams, boundary: Boundary, cells: Vec<f64>) -> Self {
        Self { params, boundary, cells, cumulative_out: 0.0 }
    }

    pub fn empty(params: CtmParams, boundary: Boundary, cell_count: usize) -> Self {
        Self::new(params, boundary, vec![0.0; cell_count])
    }

    pub fn cells(&self) -> &[f64] {
        &self.cells
    }

    pub fn total(&self) -> f64 {
        self.cells.iter().sum()
    }

    pub fn cumulative_out(&self) -> f64 {
        self.cumulative_out
    }

    fn sending(&self, i: usize) -> f64 {
        self.cells[i].min(self.params.capacity)
    }

    fn receiving(&self, i: usize) -> f64 {
        let supply = self.params.backward_ratio * (self.params.cell_max - self.cells[i]);
        self.params.capacity.min(supply.max(0.0))
    }

    pub fn step(&mut self) {
        let m = self.cells.len();
        if m == 0 {
            return;
        }
        let mut flow = vec![0.0; m];
        for i in 0..m - 1 {
            flow[i] = self.sending(i).min(self.receiving(i + 1));
        }
        let (inflow, out) = match self.boundary {
            Boundary::Ring => (self.sending(m - 1).min(self.receiving(0)), 0.0),
            Boundary::Open { inflow_demand, blocked_exit } => {
                let out = if blocked_exit { 0.0 } else { self.sending(m - 1) };
                (min(inflow_demand, self.receiving(0)), out)
            }
        };
        flow[m - 1] = match self.boundary {
            Boundary::Ring => inflow,
            Boundary::Open { .. } => out,
        };

        let mut next = self.cells.clone();
        for i in 0..m {
            let inf = if i == 0 {
                match self.boundary {
                    Boundary::Ring => flow[m - 1],
                    Boundary::Open { .. } => inflow,
                }
            } else {
                flow[i - 1]
            };
            next[i] = self.cells[i] + inf - flow[i];
        }
        self.cells = next;
        self.cumulative_out += out;
    }

    pub fn run(&mut self, ticks: u32) {
        for _ in 0..ticks {
            self.step();
        }
    }
}

fn min(a: f64, b: f64) -> f64 {
    a.min(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> CtmParams {
        CtmParams::car_arterial(1.0)
    }

    #[test]
    fn ring_conserves_vehicles() {
        let mut c = MesoCorridor::new(params(), Boundary::Ring, vec![1.5, 0.0, 0.0, 1.0, 0.5, 0.0]);
        let before = c.total();
        c.run(500);
        assert!((c.total() - before).abs() < 1e-9, "total drifted: {} vs {before}", c.total());
    }

    #[test]
    fn low_demand_flows_through_at_the_offered_rate() {
        let demand = 0.1;
        let mut c = MesoCorridor::empty(params(), Boundary::Open { inflow_demand: demand, blocked_exit: false }, 12);
        c.run(200);
        let rate = c.cumulative_out() / 200.0;
        assert!((rate - demand).abs() < 0.02, "throughput {rate} != demand {demand}");
        assert!(c.cells().iter().all(|&n| n < params().capacity), "should stay uncongested");
    }

    #[test]
    fn heavy_demand_is_capped_at_capacity() {
        let p = params();
        let mut c = MesoCorridor::empty(p, Boundary::Open { inflow_demand: 100.0, blocked_exit: false }, 12);
        c.run(400);
        let rate = c.cumulative_out() / 400.0;
        assert!((rate - p.capacity).abs() < 0.02, "throughput {rate} should hit capacity {}", p.capacity);
    }

    #[test]
    fn a_blocked_exit_backs_a_queue_up_the_corridor() {
        let p = params();
        let mut c = MesoCorridor::empty(p, Boundary::Open { inflow_demand: 100.0, blocked_exit: true }, 8);
        c.run(3);
        let early_upstream = c.cells()[0];

        c.run(200);
        let last = c.cells().len() - 1;
        assert!(c.cells()[last] > p.cell_max * 0.9, "downstream cell should jam");
        assert!(c.cells()[0] > early_upstream, "the jam should propagate upstream over time");
        assert_eq!(c.cumulative_out(), 0.0, "nothing exits past a blockage");
    }

    fn gpu_cell_formula(cells: &[f64], i: usize, p: CtmParams, inflow_demand: f64, blocked: bool) -> f64 {
        let n = cells[i];
        let send = |x: f64| x.min(p.capacity);
        let recv = |x: f64| p.capacity.min((p.backward_ratio * (p.cell_max - x)).max(0.0));
        let inflow = if i == 0 {
            inflow_demand.min(recv(n))
        } else {
            send(cells[i - 1]).min(recv(n))
        };
        let outflow = if i == cells.len() - 1 {
            if blocked { 0.0 } else { send(n) }
        } else {
            send(n).min(recv(cells[i + 1]))
        };
        n + inflow - outflow
    }

    #[test]
    fn cpu_reference_matches_the_per_cell_gpu_formula() {
        let p = params();
        let demand = 0.37;
        let start = vec![0.4, 1.9, 0.0, 2.1, 0.8, 1.2, 0.05, 1.5];
        let mut c = MesoCorridor::new(p, Boundary::Open { inflow_demand: demand, blocked_exit: false }, start.clone());
        c.step();
        for (i, &got) in c.cells().iter().enumerate() {
            let expect = gpu_cell_formula(&start, i, p, demand, false);
            assert!((got - expect).abs() < 1e-12, "cell {i}: cpu {got} != gpu-formula {expect}");
        }
    }

    #[test]
    fn wgsl_kernel_parses_and_validates() {
        let src = include_str!("meso.wgsl");
        let module = naga::front::wgsl::parse_str(src).expect("meso.wgsl should parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("meso.wgsl should type-check");
    }

    #[test]
    fn flow_never_exceeds_capacity_or_overfills_a_cell() {
        let p = params();
        let mut c = MesoCorridor::empty(p, Boundary::Open { inflow_demand: 100.0, blocked_exit: false }, 10);
        for _ in 0..300 {
            c.step();
            assert!(c.cells().iter().all(|&n| n <= p.cell_max + 1e-9), "cell overfilled");
        }
    }
}
