//! Real diurnal freeway volume profiles, so "rush hour" is sampled data rather than
//! a made-up constant. The tables are per-lane hourly mainline flow (vehicles per
//! hour per lane) for US-101 and I-280 on the San Mateo peninsula around Millbrae,
//! averaged over Caltrans PeMS *typical-weekday* detector summaries (District 04,
//! 2024; 61 mainline stations within lat 37.50–37.70, lon −122.50…−122.25, each with
//! ≥16 observed days). Source dataset: BayAreaMetro/pems-typical-weekday.
//!
//! Both curves show the real commute double-peak: US-101 crests in the AM
//! (1484 veh/h/lane at 07:00), I-280 in the PM (1369 at 17:00), with the familiar
//! pre-dawn trough and midday plateau between. Index is hour of day, 0 = midnight.

/// US-101 mainline, per-lane vehicles/hour by hour of day (PeMS typical weekday).
pub const US101: [u16; 24] = [
    285, 173, 132, 165, 330, 703, 1133, 1484, 1455, 1329, 1265, 1231, 1228, 1224, 1327, 1367,
    1403, 1379, 1290, 1124, 1005, 902, 708, 465,
];

/// I-280 mainline, per-lane vehicles/hour by hour of day (PeMS typical weekday).
pub const I280: [u16; 24] = [
    112, 61, 47, 67, 128, 313, 673, 1225, 1270, 1143, 930, 858, 849, 881, 1070, 1283, 1363, 1369,
    1176, 808, 603, 482, 338, 208,
];

/// Generic freeway fallback (the mean of the two) for a gateway whose route ref
/// matches neither — any other grade-separated highway in the loaded map.
pub const FALLBACK: [u16; 24] = [
    198, 117, 90, 116, 229, 508, 903, 1354, 1362, 1236, 1098, 1044, 1038, 1052, 1198, 1325, 1383,
    1374, 1233, 966, 804, 692, 523, 336,
];

const DAY_SECS: f64 = 86_400.0;

/// The hourly per-lane profile for a freeway, chosen by its OSM route ref: US-101,
/// I-280, or the generic freeway fallback. Matches on the bare route number so
/// "US 101", "US 101;CA 82" and the like all resolve to the 101 curve.
pub fn profile_for(route_ref: &str) -> &'static [u16; 24] {
    if route_ref.contains("101") {
        &US101
    } else if route_ref.contains("280") {
        &I280
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

/// Per-lane flow for a route ref at `seconds_into_day` — `interp` over the ref's
/// profile, the one-call form the UI readout uses.
pub fn per_lane(route_ref: &str, seconds_into_day: f64) -> f64 {
    interp(profile_for(route_ref), seconds_into_day)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ref_selects_the_matching_route_curve() {
        assert_eq!(profile_for("US 101"), &US101);
        assert_eq!(profile_for("US 101;CA 82"), &US101);
        assert_eq!(profile_for("I 280"), &I280);
        assert_eq!(profile_for("CA 92"), &FALLBACK);
    }

    #[test]
    fn interpolation_hits_the_samples_and_bridges_them() {
        // On the hour → exactly the sampled value; half past → the midpoint.
        assert_eq!(interp(&US101, 7.0 * 3600.0), 1484.0);
        assert_eq!(interp(&US101, 8.0 * 3600.0), 1455.0);
        assert_eq!(interp(&US101, 7.5 * 3600.0), (1484.0 + 1455.0) / 2.0);
    }

    #[test]
    fn wraps_across_midnight() {
        // 23:30 interpolates between hour 23 and hour 0, not off the end of the table.
        assert_eq!(interp(&US101, 23.5 * 3600.0), (465.0 + 285.0) / 2.0);
        // A full day later is the same point.
        assert_eq!(interp(&I280, 9.0 * 3600.0), interp(&I280, (24.0 + 9.0) * 3600.0));
    }

    #[test]
    fn commute_peaks_land_in_the_right_period() {
        // US-101 peaks in the morning commute, I-280 in the evening — the real
        // asymmetry between the two corridors.
        let argmax = |p: &[u16; 24]| (0..24).max_by_key(|&h| p[h]).unwrap();
        assert_eq!(argmax(&US101), 7, "US-101 AM peak");
        assert_eq!(argmax(&I280), 17, "I-280 PM peak");
    }
}
