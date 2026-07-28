//! Congestion-triggered level-of-detail for the per-car step.
//!
//! Cars are always individual and keep their real positions — this only chooses,
//! per link, *how much work* each car's decision costs. A link that stays saturated
//! switches to [`Mode::Queue`], where a follower (a car with a leader ahead in its
//! lane) is advanced by cheap leader-only car-following instead of the full
//! constraint gather, and lane changes are skipped. The car at the front of a lane
//! keeps the full model, so signals, yielding and junctions are still handled
//! correctly at the point they matter. Nothing is aggregated and no car ever leaves
//! the fleet, so behaviour stays close to per-car and overlaps remain impossible.

/// Settings for the congestion LOD. `engage`/`release` are occupancy ratios
/// (cars ÷ jam) with a gap between them, and `dwell_ticks` a sustain count, so a
/// link hovering at the threshold does not thrash between modes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CongestionConfig {
    pub enabled: bool,
    pub engage_occ: f64,
    pub release_occ: f64,
    pub dwell_ticks: u32,
}

impl CongestionConfig {
    pub const fn disabled() -> Self {
        Self { enabled: false, engage_occ: 0.85, release_occ: 0.55, dwell_ticks: 15 }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Full,
    Queue,
}

/// Per-link LOD mode with hysteresis.
pub struct CongestionLod {
    mode: Vec<Mode>,
    /// Consecutive ticks the link has held past the threshold that would flip it.
    dwell: Vec<u32>,
}

impl CongestionLod {
    pub fn new(link_count: usize) -> Self {
        Self { mode: vec![Mode::Full; link_count], dwell: vec![0; link_count] }
    }

    pub fn is_queue(&self, link: usize) -> bool {
        self.mode[link] == Mode::Queue
    }

    pub fn active_count(&self) -> u32 {
        self.mode.iter().filter(|&&m| m == Mode::Queue).count() as u32
    }

    /// All links revert to full detail — used when the LOD is switched off.
    pub fn reset(&mut self) {
        for (m, d) in self.mode.iter_mut().zip(self.dwell.iter_mut()) {
            *m = Mode::Full;
            *d = 0;
        }
    }

    /// Flip links between full and queue detail under `cfg`, given each link's live
    /// occupancy ratio. A link must hold past a threshold for `dwell_ticks` before it
    /// flips, and the engage/release gap keeps it from oscillating.
    pub fn update_modes(&mut self, occ: &[f64], cfg: &CongestionConfig) {
        for i in 0..self.mode.len() {
            let (past, target) = match self.mode[i] {
                Mode::Full => (occ[i] >= cfg.engage_occ, Mode::Queue),
                Mode::Queue => (occ[i] <= cfg.release_occ, Mode::Full),
            };
            if past {
                self.dwell[i] += 1;
                if self.dwell[i] >= cfg.dwell_ticks {
                    self.mode[i] = target;
                    self.dwell[i] = 0;
                }
            } else {
                self.dwell[i] = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> CongestionConfig {
        CongestionConfig { enabled: true, engage_occ: 0.8, release_occ: 0.4, dwell_ticks: 5 }
    }

    #[test]
    fn sustained_saturation_engages_after_the_dwell() {
        let mut lod = CongestionLod::new(2);
        let cfg = cfg();
        for _ in 0..4 {
            lod.update_modes(&[0.95, 0.0], &cfg);
            assert!(!lod.is_queue(0), "must hold for the full dwell before flipping");
        }
        lod.update_modes(&[0.95, 0.0], &cfg);
        assert!(lod.is_queue(0));
        assert!(!lod.is_queue(1), "an uncongested link stays full detail");
        assert_eq!(lod.active_count(), 1);
    }

    #[test]
    fn a_flapping_occupancy_never_engages() {
        let mut lod = CongestionLod::new(1);
        let cfg = cfg();
        for t in 0..100 {
            lod.update_modes(&[if t % 2 == 0 { 0.95 } else { 0.0 }], &cfg);
            assert!(!lod.is_queue(0), "a signal that never sustains must not flip");
        }
    }

    #[test]
    fn clears_back_to_full_with_hysteresis() {
        let mut lod = CongestionLod::new(1);
        let cfg = cfg();
        for _ in 0..5 {
            lod.update_modes(&[0.95], &cfg);
        }
        assert!(lod.is_queue(0));
        // Between release and engage: neither flips (the hysteresis band).
        for _ in 0..20 {
            lod.update_modes(&[0.6], &cfg);
            assert!(lod.is_queue(0), "occupancy in the hysteresis band holds the current mode");
        }
        // Below release, sustained: reverts.
        for _ in 0..5 {
            lod.update_modes(&[0.2], &cfg);
        }
        assert!(!lod.is_queue(0));
    }
}
