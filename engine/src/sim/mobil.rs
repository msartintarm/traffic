//! MOBIL lane-change decision (Treiber & Helbing), as a pure function of the
//! accelerations a change would produce — the lateral companion to IDM. A change
//! is taken when it's *safe* for the prospective new follower and the net
//! acceleration incentive (weighted by politeness toward others) clears a
//! threshold. Mandatory changes (needed to reach a route's turn lane) accept a
//! near-zero incentive as long as they're safe.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MobilParams {
    /// How much a driver weighs the disadvantage imposed on the new follower.
    pub politeness: f64,
    /// Minimum acceleration gain (m/s²) to bother with a discretionary change.
    pub threshold: f64,
    /// The new follower must not be forced to brake harder than this (m/s²).
    pub safe_braking: f64,
}

impl MobilParams {
    pub fn new(politeness: f64) -> Self {
        Self { politeness, threshold: 0.2, safe_braking: 4.0 }
    }
}

/// Whether to move into the candidate lane. `a_self_*` is this vehicle's IDM
/// acceleration in its current vs. the target lane; `a_follower_*` is the
/// prospective new follower's acceleration before/after being cut in front of
/// (pass equal values, e.g. both 0, when there is no new follower).
pub fn should_change(
    p: &MobilParams,
    a_self_current: f64,
    a_self_target: f64,
    a_follower_current: f64,
    a_follower_target: f64,
    mandatory: bool,
    bias: f64,
) -> bool {
    if a_follower_target < -p.safe_braking {
        return false; // would force the new follower to brake unsafely
    }
    let incentive =
        (a_self_target - a_self_current) - p.politeness * (a_follower_current - a_follower_target) + bias;
    // A mandatory change proceeds unless it's much worse; a discretionary one
    // needs a real gain.
    let threshold = if mandatory { -1.0 } else { p.threshold };
    incentive > threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p() -> MobilParams {
        MobilParams::new(0.3)
    }

    #[test]
    fn refuses_an_unsafe_change() {
        // Big self-gain, but the new follower would brake at 6 m/s² (> safe 4).
        assert!(!should_change(&p(), 0.0, 3.0, 0.0, -6.0, false, 0.0));
    }

    #[test]
    fn takes_a_clearly_beneficial_change() {
        assert!(should_change(&p(), 0.0, 2.0, 0.0, -0.5, false, 0.0));
    }

    #[test]
    fn declines_a_marginal_discretionary_change() {
        assert!(!should_change(&p(), 0.0, 0.1, 0.0, 0.0, false, 0.0));
    }

    #[test]
    fn accepts_a_marginal_mandatory_change_if_safe() {
        // No self-gain, slight cost, but required for routing → proceed (safe).
        assert!(should_change(&p(), 0.0, -0.2, 0.0, -1.0, true, 0.0));
        // still refused if unsafe
        assert!(!should_change(&p(), 0.0, -0.2, 0.0, -5.0, true, 0.0));
    }

    #[test]
    fn politeness_can_block_a_selfish_change() {
        // Modest self-gain but a big imposition on the follower; high politeness.
        let polite = MobilParams { politeness: 1.0, ..p() };
        assert!(!should_change(&polite, 0.0, 0.3, 0.0, -1.0, false, 0.0));
    }

    #[test]
    fn keep_right_bias_moves_an_unobstructed_car_toward_the_curb() {
        // Equal conditions in both lanes: a positive (rightward) bias clears the
        // threshold, a negative (leftward) one does not.
        assert!(should_change(&p(), 0.0, 0.0, 0.0, 0.0, false, 0.3));
        assert!(!should_change(&p(), 0.0, 0.0, 0.0, 0.0, false, -0.3));
    }
}
