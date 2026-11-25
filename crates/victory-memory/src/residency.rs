//! 2Q with priority and recording/submission pins, independent of page ownership.
//! The adapter must also satisfy coherence before performing an eviction.

use std::collections::{BTreeMap, BTreeSet};
use crate::{coherence::Fence, guest::PageId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Key { priority: u8, stamp: u64, page: PageId }

struct Entry {
    key: Key,
    protected: bool,
    recording: u32,
    until: Option<Fence>,
}

#[derive(Default)]
pub struct Residency {
    entries: BTreeMap<PageId, Entry>,
    probation: BTreeSet<Key>,
    protected: BTreeSet<Key>,
    clock: u64,
}

impl Residency {
    pub fn access(&mut self, page: PageId, priority: u8) {
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

    pub fn pin_recording(&mut self, page: PageId) -> bool {
        let Some(entry) = self.entries.get_mut(&page) else { return false; };
        let Some(pins) = entry.recording.checked_add(1) else { return false; };
        entry.recording = pins;
        true
    }

    pub fn end_recording(&mut self, page: PageId, submitted: Option<Fence>) -> bool {
        let Some(entry) = self.entries.get_mut(&page) else { return false; };
        let Some(pins) = entry.recording.checked_sub(1) else { return false; };
        entry.recording = pins;
        entry.until = entry.until.max(submitted);
        true
    }

    pub fn candidate(&self, completed: Fence) -> Option<PageId> {
        self.probation.iter().chain(&self.protected)
            .find(|key| self.evictable(key.page, completed)).map(|key| key.page)
    }

    /// Call after coherence writeback and D3D12 unmapping have completed.
    pub fn remove(&mut self, page: PageId, completed: Fence) -> bool {
        if !self.evictable(page, completed) { return false; }
        let entry = self.entries.remove(&page).expect("evictable resident page");
        if entry.protected { self.protected.remove(&entry.key); }
        else { self.probation.remove(&entry.key); }
        true
    }

    pub(crate) fn evictable(&self, page: PageId, completed: Fence) -> bool {
        self.entries.get(&page).is_some_and(|e| {
            e.recording == 0 && e.until.is_none_or(|until| until <= completed)
        })
    }
}
