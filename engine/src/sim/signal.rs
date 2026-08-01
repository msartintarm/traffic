//! Fixed-time traffic-signal timing as a pure function of clock time.
//!
//! Designed for scale: a signal *program* is a compact cycle of phases, and a
//! movement references a `SignalGroup` (a bit within a program). Evaluating
//! every group's colour for a given instant is one O(#groups) pass whose result
//! is an array vehicles index in O(1) — no per-vehicle branching over phase
//! tables, and a pure function of time, so it ports directly to a GPU kernel.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignalState {
    Red,
    Yellow,
    Green,
}

impl SignalState {
    /// Whether a vehicle may proceed through the movement this instant.
    pub fn is_go(self) -> bool {
        matches!(self, SignalState::Green)
    }
}

pub const DEFAULT_ALL_RED: f64 = 2.5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Phase {
    pub green_mask: u64,
    pub green_secs: f64,
    pub yellow_secs: f64,
    pub all_red_secs: f64,
}

impl Phase {
    pub fn new(green_mask: u64, green_secs: f64, yellow_secs: f64) -> Self {
        Self { green_mask, green_secs, yellow_secs, all_red_secs: 0.0 }
    }

    pub fn with_clearance(green_mask: u64, green_secs: f64, yellow_secs: f64, all_red_secs: f64) -> Self {
        Self { green_mask, green_secs, yellow_secs, all_red_secs }
    }

    pub fn length(&self) -> f64 {
        self.green_secs + self.yellow_secs + self.all_red_secs
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SignalProgram {
    pub offset: f64,
    pub phases: Vec<Phase>,
    pub coordinated: bool,
}

impl SignalProgram {
    pub fn new(offset: f64, phases: Vec<Phase>) -> Self {
        Self { offset, phases, coordinated: false }
    }

    /// Two non-overlapping approaches (bit 0 and bit 1) alternating, a common
    /// isolated-intersection default.
    pub fn two_phase(green_secs: f64, yellow_secs: f64, offset: f64) -> Self {
        Self::round_robin(2, green_secs, yellow_secs, offset)
    }

    /// `bits` non-conflicting groups each served once per cycle, one green per
    /// phase. `bits == 1` still yields a real red window (a lone approach cycles
    /// green→yellow→red against an otherwise empty stage).
    pub fn round_robin(bits: usize, green_secs: f64, yellow_secs: f64, offset: f64) -> Self {
        let phases = if bits <= 1 {
            vec![
                Phase::new(0b1, green_secs, yellow_secs),
                Phase::new(0, green_secs, 0.0),
            ]
        } else {
            (0..bits)
                .map(|b| Phase::new(1u64 << b, green_secs, yellow_secs))
                .collect()
        };
        Self::new(offset, phases)
    }

    pub fn cycle_length(&self) -> f64 {
        self.phases.iter().map(Phase::length).sum()
    }

    pub fn state_of(&self, bit: u8, sim_time: f64) -> SignalState {
        let cycle = self.cycle_length();
        if cycle <= 0.0 {
            return SignalState::Red;
        }
        let mut t = (sim_time + self.offset).rem_euclid(cycle);
        let mask = 1u64 << bit;
        for phase in &self.phases {
            if t < phase.length() {
                if phase.green_mask & mask == 0 {
                    return SignalState::Red;
                }
                return if t < phase.green_secs {
                    SignalState::Green
                } else if t < phase.green_secs + phase.yellow_secs {
                    SignalState::Yellow
                } else {
                    SignalState::Red
                };
            }
            t -= phase.length();
        }
        SignalState::Red
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_phase_alternates_and_conflicts_never_both_green() {
        let p = SignalProgram::two_phase(20.0, 4.0, 0.0);
        assert_eq!(p.cycle_length(), 48.0);
        for step in 0..480 {
            let t = step as f64 * 0.1;
            let a = p.state_of(0, t);
            let b = p.state_of(1, t);
            assert!(!(a.is_go() && b.is_go()), "conflicting greens at t={t}");
        }
    }

    #[test]
    fn colours_follow_the_schedule() {
        let p = SignalProgram::two_phase(20.0, 4.0, 0.0);
        assert_eq!(p.state_of(0, 0.0), SignalState::Green);
        assert_eq!(p.state_of(0, 19.9), SignalState::Green);
        assert_eq!(p.state_of(0, 20.1), SignalState::Yellow);
        assert_eq!(p.state_of(0, 23.9), SignalState::Yellow);
        assert_eq!(p.state_of(0, 24.1), SignalState::Red);
        assert_eq!(p.state_of(1, 24.1), SignalState::Green);
    }

    #[test]
    fn cycle_repeats() {
        let p = SignalProgram::two_phase(20.0, 4.0, 0.0);
        for bit in 0..2u8 {
            for t in [0.0, 5.0, 22.0, 40.0] {
                assert_eq!(p.state_of(bit, t), p.state_of(bit, t + p.cycle_length()));
            }
        }
    }

    #[test]
    fn offset_shifts_the_phase() {
        let base = SignalProgram::two_phase(20.0, 4.0, 0.0);
        let shifted = SignalProgram::two_phase(20.0, 4.0, 24.0);
        assert_eq!(base.state_of(0, 0.0), SignalState::Green);
        assert_eq!(shifted.state_of(0, 0.0), base.state_of(0, 24.0));
    }

    #[test]
    fn all_red_clearance_reds_every_group_at_the_phase_boundary() {
        let all_red = 3.0;
        let p = SignalProgram::new(
            0.0,
            vec![
                Phase::with_clearance(0b01, 20.0, 4.0, all_red),
                Phase::with_clearance(0b10, 20.0, 4.0, all_red),
            ],
        );
        assert_eq!(p.cycle_length(), 2.0 * (20.0 + 4.0 + 3.0));
        assert_eq!(p.state_of(0, 19.9), SignalState::Green);
        assert_eq!(p.state_of(0, 22.0), SignalState::Yellow);
        assert_eq!(p.state_of(0, 25.0), SignalState::Red);
        assert_eq!(p.state_of(1, 25.0), SignalState::Red);
        assert_eq!(p.state_of(1, 27.1), SignalState::Green);
        for step in 0..((p.cycle_length() * 10.0) as u32) {
            let t = step as f64 * 0.1;
            assert!(!(p.state_of(0, t).is_go() && p.state_of(1, t).is_go()), "conflict at t={t}");
        }
    }

    #[test]
    fn programs_are_uncoordinated_by_default() {
        assert!(!SignalProgram::two_phase(20.0, 4.0, 0.0).coordinated);
    }
}
