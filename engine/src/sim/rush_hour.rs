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
/// `k_factor = 0.09`). Used only via [`arterial_factor`] / [`surface_factor`] as a
/// normalized multiplier, so absolute magnitude is irrelevant — only the relative
/// shape matters.
pub const ARTERIAL: [u16; 24] = [
    110, 65, 50, 50, 90, 190, 370, 560, 600, 540, 520, 540, 580, 580, 620, 720, 850, 900, 760, 540,
    400, 320, 230, 160,
];

/// Which kind of surface trip a demand stream is, by its boundary geometry. Each
/// class breathes on its own diurnal shape — the surface analog of the per-direction
/// freeway curves: a city's inbound side jams in the AM while outbound flows, and
/// swaps in the PM. Shapes are modeled on the NHTS purpose mix (work trips dominate
/// the commute classes; shopping/errand trips give Internal its midday plateau).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceClass {
    /// Gateway→gateway across the map: the generic arterial mix.
    Through,
    /// Gateway→interior — arriving commuters; AM-dominant. Also carried by
    /// measured home→work commute streams (LODES), which want the same shape.
    Inbound,
    /// Interior→gateway — the evening return; PM-dominant. Also the work→home
    /// direction of measured commute streams.
    Outbound,
    /// Interior→interior — local errands; broad midday plateau.
    Internal,
}

/// Inbound (arriving) surface shape: the home→work departure surge, sharp 07–08
/// crest, then a long moderate tail (midday arrivals, a secondary PM shoulder).
pub const SURFACE_INBOUND: [u16; 24] = [
    70, 45, 35, 40, 90, 260, 560, 850, 800, 560, 460, 450, 460, 460, 480, 520, 560, 570, 480, 360,
    270, 210, 150, 100,
];

/// Outbound (departing) surface shape: the work→home return, dominant 16–18 crest
/// with a modest AM shoulder — the mirror of [`SURFACE_INBOUND`].
pub const SURFACE_OUTBOUND: [u16; 24] = [
    100, 60, 45, 40, 60, 130, 280, 420, 460, 430, 440, 470, 520, 560, 640, 780, 900, 920, 760, 540,
    400, 320, 240, 160,
];

/// Internal (local) surface shape: shopping/errand/school trips — a broad
/// 10:00–15:00 plateau between softer commute shoulders.
pub const SURFACE_INTERNAL: [u16; 24] = [
    80, 50, 40, 40, 60, 120, 260, 420, 520, 570, 620, 650, 670, 660, 650, 640, 610, 570, 500, 410,
    330, 270, 200, 130,
];

/// Weekend surface shape, all classes: no commute structure — a slow morning rise
/// into a single early-afternoon hump, with evenings staying livelier than a weekday.
pub const SURFACE_WEEKEND: [u16; 24] = [
    150, 100, 70, 50, 50, 70, 120, 200, 320, 450, 560, 640, 690, 700, 680, 640, 600, 560, 500, 430,
    370, 320, 260, 200,
];

/// The hourly shape a surface stream of `class` follows. Weekends collapse every
/// class onto the single weekend hump — the commute asymmetry is a weekday thing.
pub fn surface_profile(class: SurfaceClass, weekend: bool) -> &'static [u16; 24] {
    if weekend {
        return &SURFACE_WEEKEND;
    }
    match class {
        SurfaceClass::Through => &ARTERIAL,
        SurfaceClass::Inbound => &SURFACE_INBOUND,
        SurfaceClass::Outbound => &SURFACE_OUTBOUND,
        SurfaceClass::Internal => &SURFACE_INTERNAL,
    }
}

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

/// Per-lane volume (veh/h/lane) up to which a freeway still admits cars at its posted
/// free-flow speed; below the morning build-up there is no congestion to slow entry.
const FREE_FLOW_VOLUME: f64 = 700.0;
/// Per-lane volume at which entry is fully in the congested rush-hour regime — around the
/// real peninsula corridors' peak (I-280/US-101 crest ~1450–1500 veh/h/lane).
const CONGESTED_VOLUME: f64 = 1400.0;
/// Speed (m/s, ≈ 20 mph) cars enter a freeway at once it is fully congested: a dense crawl
/// whose short safe following gap lets many pack onto the entry at once.
const CONGESTED_ENTRY_SPEED: f64 = 9.0;

/// The speed a car should enter a freeway at for the current per-lane volume. Off-peak it
/// is the road's free-flow speed; as volume climbs toward the peak it eases down to a
/// congested crawl. Because a vehicle is admitted only with `min_gap + speed·headway` of
/// clearance, a lower entry speed means a much shorter entry gap — so rush hour enters
/// slower *and* far more tightly packed, many more cars at once, the way a real freeway
/// meters into a jam rather than injecting a fast, sparse trickle.
pub fn congested_entry_speed(free_flow: f64, per_lane_volume: f64) -> f64 {
    let floor = CONGESTED_ENTRY_SPEED.min(free_flow);
    let t = ((per_lane_volume - FREE_FLOW_VOLUME) / (CONGESTED_VOLUME - FREE_FLOW_VOLUME)).clamp(0.0, 1.0);
    free_flow + (floor - free_flow) * t
}

/// Mean of an hourly shape — the normalizer that turns a shape into a
/// daily-mean-1.0 multiplier, so it redistributes a calibrated base rate through
/// the day without inflating the total.
fn shape_mean(shape: &[u16; 24]) -> f64 {
    shape.iter().map(|&v| v as f64).sum::<f64>() / 24.0
}

/// Time-of-day multiplier for surface-street demand: the [`ARTERIAL`] shape scaled
/// so its *daily mean is 1.0*. So a surface stream's calibrated `base_rate` (which
/// represents its average-day volume) breathes with the commute — ~2× at the PM
/// peak, ~0.1× pre-dawn — instead of firing flat around the clock.
pub fn arterial_factor(seconds_into_day: f64) -> f64 {
    interp(&ARTERIAL, seconds_into_day) / shape_mean(&ARTERIAL)
}

/// Time-of-day multiplier for a surface stream of `class` (daily mean 1.0): the
/// class-specific weekday shape, or the flat-topped weekend hump. This is what
/// makes the inbound side of town build in the AM while outbound builds in the PM
/// — the boundary categories stop breathing in lockstep.
pub fn surface_factor(class: SurfaceClass, weekend: bool, seconds_into_day: f64) -> f64 {
    let shape = surface_profile(class, weekend);
    interp(shape, seconds_into_day) / shape_mean(shape)
}

/// Weekend share of a freeway's weekday volume (roughly the observed 10–20% drop).
const WEEKEND_FREEWAY_LEVEL: f64 = 0.85;
/// How far the weekend flattens the weekday freeway curve toward its own mean —
/// commute peaks mostly vanish; a single soft midday crest remains.
const WEEKEND_FREEWAY_FLATTEN: f64 = 0.5;

/// Per-lane freeway flow at `seconds_into_day` adjusted for the day of week. The
/// PeMS curves are *typical weekday*; on a weekend the volume drops ~15% and the
/// commute peaks collapse, so the curve is blended toward its daily mean rather
/// than replayed verbatim.
pub fn freeway_flow(profile: &[u16; 24], weekend: bool, seconds_into_day: f64) -> f64 {
    let v = interp(profile, seconds_into_day);
    if !weekend {
        return v;
    }
    let mean = shape_mean(profile);
    WEEKEND_FREEWAY_LEVEL * (v + (mean - v) * WEEKEND_FREEWAY_FLATTEN)
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
    fn entry_speed_eases_down_and_tightens_toward_the_peak() {
        let vf = 29.0;
        // Off-peak volume: still free-flow entry.
        assert_eq!(congested_entry_speed(vf, 300.0), vf);
        // Peak volume: the congested crawl (clamped at the floor).
        assert_eq!(congested_entry_speed(vf, 1600.0), CONGESTED_ENTRY_SPEED);
        // Mid build-up sits strictly between and monotonically below free-flow.
        let mid = congested_entry_speed(vf, 1050.0);
        assert!(mid < vf && mid > CONGESTED_ENTRY_SPEED, "mid-volume entry is between: {mid}");
        // A slow road whose free-flow is already below the congested floor never speeds up.
        assert!(congested_entry_speed(6.0, 1600.0) <= 6.0);
        // Lower entry speed ⇒ shorter admission gap ⇒ denser entry (min_gap 2, headway 1.5).
        let gap = |v: f64| 2.0 + v * 1.5;
        assert!(gap(congested_entry_speed(vf, 1600.0)) < gap(vf) * 0.5, "peak entry packs at least twice as dense");
    }

    #[test]
    fn surface_classes_peak_at_their_own_hours() {
        let argmax = |p: &[u16; 24]| (0..24).max_by_key(|&h| p[h]).unwrap();
        // The commute classes mirror each other; internal errand traffic crests midday.
        assert_eq!(argmax(&SURFACE_INBOUND), 7, "inbound is the AM arrival surge");
        assert_eq!(argmax(&SURFACE_OUTBOUND), 17, "outbound is the PM return");
        assert!((10..=15).contains(&argmax(&SURFACE_INTERNAL)), "internal peaks midday");
        assert!((11..=14).contains(&argmax(&SURFACE_WEEKEND)), "weekend is one early-afternoon hump");

        // Every class factor is a daily-mean-1 multiplier, so base rates stay calibrated.
        for class in [SurfaceClass::Through, SurfaceClass::Inbound, SurfaceClass::Outbound, SurfaceClass::Internal] {
            for weekend in [false, true] {
                let mean = (0..24).map(|h| surface_factor(class, weekend, h as f64 * 3600.0)).sum::<f64>() / 24.0;
                assert!((mean - 1.0).abs() < 0.02, "{class:?} weekend={weekend} daily mean ≈ 1, got {mean}");
            }
        }

        // At the AM peak the inbound side of town far outdraws outbound; swapped by PM.
        let am = 7.5 * 3600.0;
        let pm = 17.5 * 3600.0;
        let f = |c, t| surface_factor(c, false, t);
        assert!(f(SurfaceClass::Inbound, am) > f(SurfaceClass::Outbound, am) * 1.5, "AM is inbound-heavy");
        assert!(f(SurfaceClass::Outbound, pm) > f(SurfaceClass::Inbound, pm) * 1.3, "PM is outbound-heavy");
    }

    #[test]
    fn weekends_lighten_and_flatten_the_freeway() {
        // The weekday AM crest mostly collapses on a weekend, and the whole day
        // carries ~15% less volume.
        let weekday_peak = freeway_flow(&I280_S, false, 7.0 * 3600.0);
        let weekend_peak = freeway_flow(&I280_S, true, 7.0 * 3600.0);
        assert!(weekend_peak < weekday_peak * 0.75, "weekend blunts the commute peak: {weekend_peak} vs {weekday_peak}");
        let day_total = |weekend| (0..24).map(|h| freeway_flow(&I280_S, weekend, h as f64 * 3600.0)).sum::<f64>();
        let ratio = day_total(true) / day_total(false);
        assert!((ratio - 0.85).abs() < 0.02, "weekend carries ~85% of weekday volume, got {ratio}");
        // Overnight the flattening *raises* flow toward the mean — quiet hours are less dead.
        assert!(freeway_flow(&I280_S, true, 3.0 * 3600.0) > freeway_flow(&I280_S, false, 3.0 * 3600.0));
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
