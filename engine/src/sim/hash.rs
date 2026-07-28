//! Cheap hashing for dense integer ids. The per-tick maps and the browser pose
//! buffer key on small `u32` ids, where the default SipHash is needless overhead.

use std::collections::HashMap;
use std::hash::BuildHasherDefault;

/// FxHash-style hasher for dense integer ids — far cheaper than the default SipHash.
#[derive(Default)]
pub struct FxHasher(u64);

const FX_K: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl std::hash::Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(5) ^ b as u64).wrapping_mul(FX_K);
        }
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.0 = (self.0.rotate_left(5) ^ i as u64).wrapping_mul(FX_K);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

/// Map keyed by a dense integer id, hashed with [`FxHasher`].
pub type IntMap<V> = HashMap<u32, V, BuildHasherDefault<FxHasher>>;
