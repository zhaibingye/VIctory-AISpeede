//! Boundary between command preparation and the title's actual XMem backing.
//! A pin retains a physical allocation, not the title's virtual mapping.

use std::sync::OnceLock;

use victory_memory::physical::BackingPins;

type Range = (usize, usize);
pub type BackingPinner = fn(&[Range], &mut BackingPins);

static PINNER: OnceLock<BackingPinner> = OnceLock::new();

/// Installed by the runtime before the title creates its D3D12 device.
pub fn use_pinner(pinner: BackingPinner) {
    let _ = PINNER.set(pinner);
}

#[derive(Default)]
pub(crate) struct SubmissionBacking {
    pending: Vec<Range>,
    pins: BackingPins,
}

impl SubmissionBacking {
    pub(crate) fn include(&mut self, start: usize, end: usize) {
        if start < end && PINNER.get().is_some() {
            self.pending.push((start, end));
        }
    }

    pub(crate) fn flush(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        merge(&mut self.pending);
        if let Some(pin) = PINNER.get() {
            // Called without a GPU page table/descriptor/shadow registry lock.
            pin(&self.pending, &mut self.pins);
        }
        self.pending.clear();
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
