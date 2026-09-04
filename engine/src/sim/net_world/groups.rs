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
    /// (idempotent-safe to call once at construction). Every key ever pushed
    /// must be `< cap`.
    pub(super) fn reserve_dense(&mut self, cap: usize) {
        self.dense = (0..cap).map(|_| Vec::new()).collect();
    }

    #[inline]
    fn is_dense(&self) -> bool {
        !self.dense.is_empty()
    }

    /// Reset for a new tick: clear last tick's groups, keep their capacity.
    pub(super) fn begin_tick(&mut self) {
        if self.is_dense() {
            for &k in &self.touched {
                self.dense[k as usize].clear();
            }
            self.touched.clear();
        } else {
            for k in self.touched.drain(..) {
                if let Some(v) = self.map.get_mut(&k) {
                    v.clear();
                }
            }
        }
    }

    /// This tick's non-empty groups, keyed.
    pub(super) fn iter(&self) -> impl Iterator<Item = (u32, &Vec<T>)> {
        let dense = self.is_dense();
        self.touched.iter().filter_map(move |&k| {
            let g = if dense { &self.dense[k as usize] } else { self.map.get(&k)? };
            Some((k, g))
        })
    }

    pub(super) fn push(&mut self, key: u32, value: T) {
        let group = if self.is_dense() {
            &mut self.dense[key as usize]
        } else {
            self.map.entry(key).or_default()
        };
        if group.is_empty() {
            self.touched.push(key);
        }
        group.push(value);
    }

    /// The group under `key` this tick, if any member was pushed.
    pub(super) fn get(&self, key: &u32) -> Option<&Vec<T>> {
        if self.is_dense() {
            self.dense.get(*key as usize).filter(|v| !v.is_empty())
        } else {
            self.map.get(key).filter(|v| !v.is_empty())
        }
    }

    /// Number of non-empty groups this tick.
    pub(super) fn len(&self) -> usize {
        self.touched.len()
    }

    /// The `i`-th touched group (arbitrary but stable within a tick).
    pub(super) fn group_at(&self, i: usize) -> &Vec<T> {
        let k = self.touched[i];
        if self.is_dense() {
            &self.dense[k as usize]
        } else {
            &self.map[&k]
        }
    }

    pub(super) fn group_mut_at(&mut self, i: usize) -> (u32, &mut Vec<T>) {
        let k = self.touched[i];
        let g = if self.is_dense() {
            &mut self.dense[k as usize]
        } else {
            self.map.get_mut(&k).unwrap()
        };
        (k, g)
    }
}
