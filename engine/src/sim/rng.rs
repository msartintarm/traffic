//! Stateless counter-based randomness: every draw is a pure hash of
//! `(seed, agent_id, tick, stream)`, so CPU and a future GPU kernel agree
//! regardless of thread order, and fixing the seed reproduces a run on demand.
//! Mixer is SplitMix64.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u64)]
pub enum Stream {
    DriverProfile = 0,
    GapAcceptance = 1,
    RouteChoice = 2,
    AccelNoise = 3,
}

#[inline]
pub fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
pub fn hash(seed: u64, agent_id: u32, tick: u64, stream: Stream) -> u64 {
    let mut key = seed;
    key = key.wrapping_mul(0xD1B5_4A32_D192_ED03).wrapping_add(agent_id as u64);
    key = key.wrapping_mul(0x00A0_761D_6478_BD64).wrapping_add(tick);
    key = key.wrapping_mul(0xE703_7ED1_A0B4_28DB).wrapping_add(stream as u64);
    mix64(key)
}

#[inline]
pub fn uniform01(seed: u64, agent_id: u32, tick: u64, stream: Stream) -> f64 {
    (hash(seed, agent_id, tick, stream) >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

#[inline]
pub fn uniform_range(seed: u64, agent_id: u32, tick: u64, stream: Stream, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * uniform01(seed, agent_id, tick, stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform01_is_in_unit_interval() {
        for id in 0..10_000u32 {
            let u = uniform01(1234, id, 7, Stream::DriverProfile);
            assert!((0.0..1.0).contains(&u), "u={u} out of range for id={id}");
        }
    }

    #[test]
    fn same_coordinates_reproduce_exactly() {
        let a = uniform01(42, 99, 3, Stream::GapAcceptance);
        let b = uniform01(42, 99, 3, Stream::GapAcceptance);
        assert_eq!(a.to_bits(), b.to_bits());
    }

    #[test]
    fn streams_are_decorrelated() {
        let a = uniform01(42, 99, 3, Stream::DriverProfile);
        let b = uniform01(42, 99, 3, Stream::GapAcceptance);
        assert_ne!(a.to_bits(), b.to_bits());
    }

    #[test]
    fn different_agents_and_ticks_differ() {
        let base = uniform01(42, 99, 3, Stream::RouteChoice);
        assert_ne!(base, uniform01(42, 100, 3, Stream::RouteChoice));
        assert_ne!(base, uniform01(42, 99, 4, Stream::RouteChoice));
        assert_ne!(base, uniform01(43, 99, 3, Stream::RouteChoice));
    }

    #[test]
    fn mean_is_near_a_half() {
        let n = 200_000u32;
        let sum: f64 = (0..n)
            .map(|id| uniform01(7, id, 0, Stream::DriverProfile))
            .sum();
        let mean = sum / n as f64;
        assert!((mean - 0.5).abs() < 0.01, "mean={mean}");
    }

    #[test]
    fn uniform_range_respects_bounds() {
        for id in 0..1000u32 {
            let v = uniform_range(0, id, 0, Stream::DriverProfile, -3.0, 5.0);
            assert!((-3.0..5.0).contains(&v));
        }
    }
}
