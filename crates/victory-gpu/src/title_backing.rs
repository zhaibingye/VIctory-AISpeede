//! Boundary between command preparation and the title's actual XMem backing.
//! A pin retains a physical allocation, not the title's virtual mapping.

use std::sync::OnceLock;

use victory_memory::physical::BackingPins;

type Range = (usize, usize);
pub type BackingPinner = fn(&[Range], &mut BackingPins);

#[derive(Clone, Copy)]
pub struct WritebackTarget {
    pub address: usize,
    /// True only if the title VA still maps the captured page.
    pub original_va: bool,
}

pub type WritebackPage =
    fn(usize, usize, &BackingPins, bool, &mut dyn FnMut(WritebackTarget)) -> bool;

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
    /// For marked writes without a stable before-image, retain the stale
    /// destination guard. Only a computed byte delta can be redirected.
    pub(crate) fn write_page<T>(&self, address: usize, write: impl FnOnce() -> T) -> Option<T> {
        self.write_target(address, 1, false, |_| write())
    }

    pub(crate) fn write_delta<T>(
        &self, address: usize, len: usize, write: impl FnOnce(WritebackTarget) -> T,
    ) -> Option<T> {
        self.write_target(address, len, true, write)
    }

    fn write_target<T>(
        &self, address: usize, len: usize, redirect: bool, write: impl FnOnce(WritebackTarget) -> T,
    ) -> Option<T> {
        let Some((_, with_page)) = BACKING.get() else {
            return Some(write(WritebackTarget { address, original_va: true }));
        };
        let mut write = Some(write);
        let mut result = None;
        with_page(address, len, &self.pins, redirect, &mut |target| {
            result = write.take().map(|write| write(target));
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
