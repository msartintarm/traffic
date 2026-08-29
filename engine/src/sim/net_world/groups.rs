//! A reusable index multimap for the per-tick vehicle groupings (per-lane,
//! per-corridor, per-intersection). The step rebuilds these groups every tick;
//! allocating fresh maps and vectors each time made the group phases
//! allocation-bound at city scale. A `GroupMap` instead persists across ticks:
//! `begin_tick` clears only the groups touched last tick (keeping every
//! vector's capacity), so a steady-state tick allocates nothing.

use crate::sim::hash::IntMap;

pub(super) struct GroupMap<T = usize> {
    map: IntMap<Vec<T>>,
    /// Keys with a non-empty group this tick — the only ones `begin_tick` must
    /// clear and the only ones the group iterators visit (the map itself keeps
    /// one empty, capacity-bearing vector per key ever seen).
    touched: Vec<u32>,
}

impl<T> Default for GroupMap<T> {
    fn default() -> Self {
        Self { map: IntMap::default(), touched: Vec::new() }
    }
}

impl<T> GroupMap<T> {
    /// Reset for a new tick: clear last tick's groups, keep their capacity.
    pub(super) fn begin_tick(&mut self) {
        for k in self.touched.drain(..) {
            if let Some(v) = self.map.get_mut(&k) {
                v.clear();
            }
        }
    }

    pub(super) fn push(&mut self, key: u32, value: T) {
        let group = self.map.entry(key).or_default();
        if group.is_empty() {
            self.touched.push(key);
        }
        group.push(value);
    }

    /// The group under `key` this tick, if any member was pushed.
    pub(super) fn get(&self, key: &u32) -> Option<&Vec<T>> {
        self.map.get(key).filter(|v| !v.is_empty())
    }

    /// Number of non-empty groups this tick.
    pub(super) fn len(&self) -> usize {
        self.touched.len()
    }

    /// The `i`-th touched group (arbitrary but stable within a tick).
    pub(super) fn group_at(&self, i: usize) -> &Vec<T> {
        &self.map[&self.touched[i]]
    }

    pub(super) fn group_mut_at(&mut self, i: usize) -> (u32, &mut Vec<T>) {
        let k = self.touched[i];
        (k, self.map.get_mut(&k).unwrap())
    }
}
