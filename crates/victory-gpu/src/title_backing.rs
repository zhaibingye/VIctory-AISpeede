//! Boundary between command preparation and the title's actual XMem backing.
//! A pin retains a physical allocation, not the title's virtual mapping.

use std::sync::OnceLock;

use victory_memory::physical::{BackingPins, PhysicalPage};

type Range = (usize, usize);
pub type BackingPinner = fn(&[Range], &mut BackingPins);

pub type WritebackPage = fn(usize, Option<PhysicalPage>, &mut dyn FnMut()) -> bool;

static BACKING: OnceLock<(BackingPinner, WritebackPage)> = OnceLock::new();

/// Installed by the runtime before the title creates its D3D12 device.
pub fn use_backing(pinner: BackingPinner, writeback: WritebackPage) {
    let _ = BACKING.set((pinner, writeback));
}

#[derive(Default)]
pub(crate) struct SubmissionBacking {
    pending: Vec<Range>,
    pins: BackingPins,
}

impl SubmissionBacking {
    pub(crate) fn include(&mut self, start: usize, end: usize) {
        if start < end && BACKING.get().is_some() {
            self.pending.push((start, end));
        }
    }

    pub(crate) fn flush(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        merge(&mut self.pending);
        if let Some((pin, _)) = BACKING.get() {
            // Called without a GPU page table/descriptor/shadow registry lock.
            pin(&self.pending, &mut self.pins);
        }
        self.pending.clear();
    }
    /// Run the read/merge/write on one 64 KiB page without a concurrent
    /// XMem remap. A stale destination is left alone.
    pub(crate) fn write_page<T>(&self, address: usize, write: impl FnOnce() -> T) -> Option<T> {
        let Some((_, with_page)) = BACKING.get() else {
            return Some(write());
        };
        let mut write = Some(write);
        let mut result = None;
        with_page(address, self.pins.expected(address), &mut || {
            result = write.take().map(|write| write());
        });
        result
    }

}

fn merge(ranges: &mut Vec<Range>) {
    ranges.sort_unstable();
    ranges.dedup_by(|next, current| {
        if next.0 > current.1 {
            return false;
        }
        current.1 = current.1.max(next.1);
        true
    });
}

#[cfg(test)]
#[test]
fn submission_backing_merges_overlapping_and_touching_ranges_only() {
    let mut ranges = vec![(40, 52), (8, 16), (4, 6), (51, 60), (16, 20), (44, 48), (4, 6)];
    merge(&mut ranges);
    assert_eq!(ranges, [(4, 6), (8, 20), (40, 60)]);
}
