//! A reusable index multimap for the per-tick vehicle groupings (per-lane,
//! per-corridor, per-intersection). The step rebuilds these groups every tick;
//! allocating fresh maps and vectors each time made the group phases
//! allocation-bound at city scale. A `GroupMap` instead persists across ticks:
//! `begin_tick` clears only the groups touched last tick (keeping every
//! vector's capacity), so a steady-state tick allocates nothing.
//!
//! Keys that are dense small integers (lane ids, corridor ids, intersection
//! keys) can skip the hash entirely: `reserve_dense(cap)` switches the store to
//! a `cap`-sized bucket array indexed directly by key. The per-tick grouping is
//! the hottest serial loop in the step, and its cost is the per-car push; a
//! direct index removes the hash + probe (measured cheaper than `IntMap` at
//! city fleet sizes). Instances with sparse/large keys stay on the hash map.

use crate::sim::hash::IntMap;

pub(super) struct GroupMap<T = usize> {
    /// Hash-backed store, used unless `dense` is non-empty.
    map: IntMap<Vec<T>>,
    /// Direct-indexed store: `dense[key]` is the group. Non-empty only after
    /// `reserve_dense`; then it is authoritative and `map` is unused.
    dense: Vec<Vec<T>>,
    /// Keys with a non-empty group this tick — the only ones `begin_tick` must
    /// clear and the only ones the group iterators visit (either store keeps
    /// one empty, capacity-bearing vector per key ever seen).
    touched: Vec<u32>,
}

impl<T> Default for GroupMap<T> {
    fn default() -> Self {
        Self { map: IntMap::default(), dense: Vec::new(), touched: Vec::new() }
    }
}

impl<T> GroupMap<T> {
    /// Switch to a direct-indexed bucket array sized for keys in `0..cap`
    /// (idempotent-safe to call once at construction). Keys `< cap` use the array;
    /// keys `>= cap` (e.g. the `u32::MAX` "no corridor" sentinel a lane in a cycle
    /// / unreachable component carries) fall back to the hash map, so the dense
    /// store is never a *narrower* key domain than the hash map it replaced.
    pub(super) fn reserve_dense(&mut self, cap: usize) {
        self.dense = (0..cap).map(|_| Vec::new()).collect();
    }

    /// Whether `key` is served by the direct-indexed array (dense mode and in
    /// range); otherwise it lives in the hash map.
    #[inline]
    fn dense_key(&self, key: u32) -> bool {
        (key as usize) < self.dense.len()
    }

    /// The bucket for `key`, from whichever store owns it.
    #[inline]
    fn bucket_mut(&mut self, key: u32) -> &mut Vec<T> {
        if self.dense_key(key) {
            &mut self.dense[key as usize]
        } else {
            self.map.entry(key).or_default()
        }
    }

    /// Reset for a new tick: clear last tick's groups, keep their capacity.
    pub(super) fn begin_tick(&mut self) {
        for k in self.touched.drain(..) {
            if (k as usize) < self.dense.len() {
                self.dense[k as usize].clear();
            } else if let Some(v) = self.map.get_mut(&k) {
                v.clear();
            }
        }
    }

    /// This tick's non-empty groups, keyed.
    pub(super) fn iter(&self) -> impl Iterator<Item = (u32, &Vec<T>)> {
        self.touched.iter().filter_map(move |&k| Some((k, self.group_by_key(k)?)))
    }

    fn group_by_key(&self, key: u32) -> Option<&Vec<T>> {
        if self.dense_key(key) {
            Some(&self.dense[key as usize])
        } else {
            self.map.get(&key)
        }
    }

    pub(super) fn push(&mut self, key: u32, value: T) {
        // Inlined (not via `bucket_mut`) so the bucket borrow (`self.dense`/`self.map`)
        // and `self.touched` are seen as disjoint fields.
        if (key as usize) < self.dense.len() {
            let g = &mut self.dense[key as usize];
            if g.is_empty() {
                self.touched.push(key);
            }
            g.push(value);
        } else {
            let g = self.map.entry(key).or_default();
            if g.is_empty() {
                self.touched.push(key);
            }
            g.push(value);
        }
    }

    /// The group under `key` this tick, if any member was pushed.
    pub(super) fn get(&self, key: &u32) -> Option<&Vec<T>> {
        self.group_by_key(*key).filter(|v| !v.is_empty())
    }

    /// Number of non-empty groups this tick.
    pub(super) fn len(&self) -> usize {
        self.touched.len()
    }

    /// The `i`-th touched group (arbitrary but stable within a tick).
    pub(super) fn group_at(&self, i: usize) -> &Vec<T> {
        self.group_by_key(self.touched[i]).expect("touched key has a group")
    }

    pub(super) fn group_mut_at(&mut self, i: usize) -> (u32, &mut Vec<T>) {
        let k = self.touched[i];
        (k, self.bucket_mut(k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_groupmap_handles_out_of_range_keys() {
        // A dense-backed map must still accept keys >= cap (e.g. the u32::MAX
        // "no corridor" sentinel) via the hash-map fallback — not index-panic.
        let mut g: GroupMap<usize> = GroupMap::default();
        g.reserve_dense(4);
        g.begin_tick();
        g.push(1, 10); // in the dense array
        g.push(u32::MAX, 20); // out of range -> hash map
        g.push(1, 11);
        g.push(u32::MAX, 21);
        assert_eq!(g.get(&1), Some(&vec![10, 11]));
        assert_eq!(g.get(&u32::MAX), Some(&vec![20, 21]));
        assert_eq!(g.len(), 2, "two touched groups across both stores");
        let mut seen: Vec<(u32, usize)> = g.iter().map(|(k, v)| (k, v.len())).collect();
        seen.sort();
        assert_eq!(seen, vec![(1, 2), (u32::MAX, 2)]);
        // A second tick clears both stores' touched buckets.
        g.begin_tick();
        assert_eq!(g.get(&1), None);
        assert_eq!(g.get(&u32::MAX), None);
        assert_eq!(g.len(), 0);
    }
}
