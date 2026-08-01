//! Real diurnal freeway volume profiles, so "rush hour" is sampled data rather than
//! a made-up constant. The tables are per-lane hourly mainline flow (vehicles per
//! hour per lane) for US-101 and I-280 on the San Mateo peninsula around Millbrae,
//! averaged over Caltrans PeMS *typical-weekday* detector summaries (District 04,
//! 2024; mainline stations within lat 37.50–37.70, lon −122.50…−122.25, each with
//! ≥10 observed days). Source dataset: BayAreaMetro/pems-typical-weekday.
//!
//! Each corridor is split by **direction of travel** (northbound / southbound),
//! because the commute is directional: I-280 southbound peaks in the AM (1455 at
//! 07:00 — the run to the Silicon Valley job centres) while northbound peaks in the
//! PM (1467 at 16:00 — the evening return); US-101 is AM-dominant both ways but its
//! southbound crest is sharper. Averaging the two directions (as a single curve
//! would) erases exactly this asymmetry — the thing that makes one side of a freeway
//! jam while the other flows. Index is hour of day, 0 = midnight.

/// US-101 mainline **northbound**, per-lane vehicles/hour by hour (27 stations).
pub const US101_N: [u16; 24] = [
    288, 179, 141, 181, 358, 767, 1197, 1467, 1438, 1305, 1247, 1234, 1215, 1213, 1319, 1386,
    1452, 1432, 1360, 1200, 1067, 931, 688, 448,
];

/// US-101 mainline **southbound**, per-lane vehicles/hour by hour (21 stations).
pub const US101_S: [u16; 24] = [
    281, 164, 119, 144, 293, 621, 1051, 1507, 1476, 1361, 1288, 1228, 1244, 1238, 1338, 1343,
    1340, 1310, 1200, 1027, 922, 865, 734, 488,
];

/// I-280 mainline **northbound**, per-lane vehicles/hour by hour (6 stations). PM peak.
pub const I280_N: [u16; 24] = [
    128, 69, 44, 40, 80, 211, 495, 957, 1125, 927, 808, 770, 765, 808, 1003, 1327, 1467, 1457,
    1324, 914, 648, 503, 338, 217,
];

/// I-280 mainline **southbound**, per-lane vehicles/hour by hour (7 stations). AM peak.
pub const I280_S: [u16; 24] = [
    98, 54, 50, 89, 169, 401, 826, 1455, 1395, 1328, 1035, 933, 920, 943, 1128, 1246, 1274, 1293,
    1050, 718, 565, 463, 338, 200,
];

/// Generic freeway fallback (the mean of the four directional curves) for a gateway
/// whose route ref matches neither corridor, or whose travel is not clearly N/S —
/// any other grade-separated highway in the loaded map.
pub const FALLBACK: [u16; 24] = [
    199, 117, 89, 114, 225, 500, 892, 1346, 1359, 1230, 1095, 1041, 1036, 1051, 1197, 1325, 1383,
    1373, 1234, 965, 800, 691, 525, 338,
];

/// Typical-weekday **urban-arterial** hourly shape (relative weight per hour of
/// day). Unlike the freeway curves above, this is *modeled*, not sensor data:
/// Caltrans PeMS covers freeway mainline only, so there's no per-arterial diurnal
/// curve to lift. The shape is the standard urban-arterial form — a broad AM
/// shoulder, a sustained midday plateau (errand/shopping trips freeways lack), and
/// a dominant PM peak — and its *peakiness* is pinned to the real Caltrans
/// K-factor for these routes (peak hour ≈ 9% of the day; see `tools/counts`,
/// `k_factor = 0.09`). Used only via [`arterial_factor`] as a normalized multiplier,
/// so absolute magnitude is irrelevant — only the relative shape matters.
pub const ARTERIAL: [u16; 24] = [
    110, 65, 50, 50, 90, 190, 370, 560, 600, 540, 520, 540, 580, 580, 620, 720, 850, 900, 760, 540,
    400, 320, 230, 160,
];

const DAY_SECS: f64 = 86_400.0;

/// The hourly per-lane profile for a freeway gateway, chosen by its OSM route ref
/// **and travel direction**: US-101 or I-280 (northbound vs southbound), or the
/// generic fallback. `northbound` is the sign of the gateway's north-ward travel
/// component; on the peninsula both corridors run N–S, so a gateway that isn't
/// clearly N/S (an E–W highway) should use the fallback rather than this split.
pub fn profile_for(route_ref: &str, northbound: bool) -> &'static [u16; 24] {
    if route_ref.contains("101") {
        if northbound {
            &US101_N
        } else {
            &US101_S
        }
    } else if route_ref.contains("280") {
        if northbound {
            &I280_N
        } else {
            &I280_S
        }
    } else {
        &FALLBACK
    }
}

/// Per-lane flow (veh/hour/lane) at `seconds_into_day`, linearly interpolated between
/// the bracketing hourly samples and wrapping across midnight, so the volume slides
/// smoothly through the peak instead of stepping hour to hour.
pub fn interp(profile: &[u16; 24], seconds_into_day: f64) -> f64 {
    let hour = seconds_into_day.rem_euclid(DAY_SECS) / 3600.0;
    let i = hour.floor() as usize % 24;
    let frac = hour - hour.floor();
    let a = profile[i] as f64;
    let b = profile[(i + 1) % 24] as f64;
    a + (b - a) * frac
}

/// Per-lane flow for a route ref + direction at `seconds_into_day` — `interp` over
/// the resolved profile, the one-call form the UI readout uses.
pub fn per_lane(route_ref: &str, northbound: bool, seconds_into_day: f64) -> f64 {
    interp(profile_for(route_ref, northbound), seconds_into_day)
}

/// Time-of-day multiplier for surface-street demand: the [`ARTERIAL`] shape scaled
/// so its *daily mean is 1.0*. So a surface stream's calibrated `base_rate` (which
/// represents its average-day volume) breathes with the commute — ~2× at the PM
/// peak, ~0.1× pre-dawn — instead of firing flat around the clock.
pub fn arterial_factor(seconds_into_day: f64) -> f64 {
    let mean = ARTERIAL.iter().map(|&v| v as f64).sum::<f64>() / 24.0;
    interp(&ARTERIAL, seconds_into_day) / mean
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ref_and_direction_select_the_matching_curve() {
        assert_eq!(profile_for("US 101", true), &US101_N);
        assert_eq!(profile_for("US 101;CA 82", false), &US101_S);
        assert_eq!(profile_for("I 280", true), &I280_N);
        assert_eq!(profile_for("I 280", false), &I280_S);
        assert_eq!(profile_for("CA 92", true), &FALLBACK);
    }

    #[test]
    fn interpolation_hits_the_samples_and_bridges_them() {
        // On the hour → exactly the sampled value; half past → the midpoint.
        assert_eq!(interp(&US101_N, 7.0 * 3600.0), 1467.0);
        assert_eq!(interp(&US101_N, 8.0 * 3600.0), 1438.0);
        assert_eq!(interp(&US101_N, 7.5 * 3600.0), (1467.0 + 1438.0) / 2.0);
    }

    #[test]
    fn wraps_across_midnight() {
        // 23:30 interpolates between hour 23 and hour 0, not off the end of the table.
        assert_eq!(interp(&US101_N, 23.5 * 3600.0), (448.0 + 288.0) / 2.0);
        assert_eq!(interp(&I280_S, 9.0 * 3600.0), interp(&I280_S, (24.0 + 9.0) * 3600.0));
    }

    #[test]
    fn arterial_factor_breathes_with_the_commute_around_a_daily_mean_of_one() {
        // Averaged over the day the multiplier is 1 (so it only redistributes the
        // calibrated base rate through time, never inflates the total).
        let mean = (0..24).map(|h| arterial_factor(h as f64 * 3600.0)).sum::<f64>() / 24.0;
        assert!((mean - 1.0).abs() < 0.02, "daily mean ≈ 1, got {mean}");
        // Urban arterials are PM-dominant, and night is a small fraction of the peak.
        let (am, pm, night) = (arterial_factor(8.0 * 3600.0), arterial_factor(17.0 * 3600.0), arterial_factor(3.0 * 3600.0));
        assert!(pm > am && am > 1.0, "PM peak dominates and both peaks exceed the mean: am={am} pm={pm}");
        assert!(night < 0.2, "pre-dawn is a small fraction of the day: {night}");
        // Peakiness matches the real Caltrans K-factor: peak hour ≈ 9% of the day.
        let k = *ARTERIAL.iter().max().unwrap() as f64 / ARTERIAL.iter().map(|&v| v as f64).sum::<f64>();
        assert!((k - 0.09).abs() < 0.01, "peak-hour share ≈ K=0.09, got {k:.3}");
    }

    #[test]
    fn directions_show_the_real_commute_asymmetry() {
        // I-280 is the clean commute corridor: southbound peaks in the morning (toward
        // the job centres), northbound in the evening (the return). The whole point of
        // splitting by direction — a single averaged curve would hide it.
        let argmax = |p: &[u16; 24]| (0..24).max_by_key(|&h| p[h]).unwrap();
        assert_eq!(argmax(&I280_S), 7, "I-280 SB peaks in the AM");
        assert_eq!(argmax(&I280_N), 16, "I-280 NB peaks in the PM");
        // And at the AM peak, SB carries far more than NB (jammed vs flowing).
        assert!(I280_S[7] > I280_N[7] + 400, "I-280 AM is southbound-heavy");
        // US-101 is AM-dominant both ways, but the southbound crest is the sharper one.
        assert_eq!(argmax(&US101_S), 7, "US-101 SB AM peak");
        assert!(US101_S[7] > US101_N[7], "US-101 AM peak is southbound-heavier");
    }
}
