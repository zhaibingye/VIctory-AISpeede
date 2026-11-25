//! 2Q with priority and recording/submission pins, independent of page ownership.
//! The adapter must also satisfy coherence before performing an eviction.

use std::collections::{BTreeMap, BTreeSet};
use crate::{coherence::Fence, guest::PageId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Key<P> { priority: u8, stamp: u64, page: P }

struct Entry<P> {
    key: Key<P>,
    protected: bool,
    recording: u32,
    until: Option<Fence>,
}

pub struct Residency<P = PageId> {
    entries: BTreeMap<P, Entry<P>>,
    probation: BTreeSet<Key<P>>,
    protected: BTreeSet<Key<P>>,
    clock: u64,
}

impl<P> Default for Residency<P> {
    fn default() -> Self {
        Self { entries: BTreeMap::new(), probation: BTreeSet::new(), protected: BTreeSet::new(), clock: 0 }
    }
}

impl<P: Copy + Ord> Residency<P> {
    pub fn access(&mut self, page: P, priority: u8) {
        let key = Key { priority, stamp: self.clock, page };
        self.clock = self.clock.checked_add(1).expect("2Q epoch overflow");
        if let Some(entry) = self.entries.get_mut(&page) {
            if entry.protected {
                self.protected.remove(&entry.key);
            } else {
                self.probation.remove(&entry.key);
            }
            entry.key = key;
            entry.protected = true;
            self.protected.insert(key);
        } else {
            self.entries.insert(page, Entry { key, protected: false, recording: 0, until: None });
            self.probation.insert(key);
        }
        while self.protected.len() > self.entries.len() / 2 {
            let cold = self.protected.pop_first().expect("over target");
            self.entries.get_mut(&cold.page).expect("resident page").protected = false;
            self.probation.insert(cold);
        }
    }

    pub fn pin_recording(&mut self, page: P) -> bool {
        let Some(entry) = self.entries.get_mut(&page) else { return false; };
        let Some(pins) = entry.recording.checked_add(1) else { return false; };
        entry.recording = pins;
        true
    }

    pub fn end_recording(&mut self, page: P, submitted: Option<Fence>) -> bool {
        let Some(entry) = self.entries.get_mut(&page) else { return false; };
        let Some(pins) = entry.recording.checked_sub(1) else { return false; };
        entry.recording = pins;
        entry.until = entry.until.max(submitted);
        true
    }

    pub fn candidate(&self, completed: Fence) -> Option<P> {
        self.candidate_where(completed, |_| true)
    }

    pub fn candidate_where(&self, completed: Fence, eligible: impl Fn(P) -> bool) -> Option<P> {
        self.probation.iter().chain(&self.protected)
            .find(|key| self.evictable(key.page, completed) && eligible(key.page)).map(|key| key.page)
    }

    /// Call after coherence writeback and D3D12 unmapping have completed.
    pub fn remove(&mut self, page: P, completed: Fence) -> bool {
        if !self.evictable(page, completed) { return false; }
        let entry = self.entries.remove(&page).expect("evictable resident page");
        if entry.protected { self.protected.remove(&entry.key); }
        else { self.probation.remove(&entry.key); }
        true
    }

    pub(crate) fn evictable(&self, page: P, completed: Fence) -> bool {
        self.entries.get(&page).is_some_and(|e| {
            e.recording == 0 && e.until.is_none_or(|until| until <= completed)
        })
    }
}
