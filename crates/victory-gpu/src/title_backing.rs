//! Boundary between command preparation and the title's actual XMem backing.
//! A pin retains a physical allocation, not the title's virtual mapping.

use std::sync::OnceLock;

use victory_memory::{
    PAGE_BYTES,
    physical::{BackingPins, PhysicalPage},
};

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

/// A visitor receives (title start, source address or unmapped, bytes,
/// same physical binding). A temporary source is valid only in the visitor.
pub type ReadRuns =
    fn(usize, usize, &BackingPins, &mut dyn FnMut(usize, Option<usize>, usize, bool));

type BackingApi = (BackingPinner, WritebackPage, ReadRuns);
static BACKING: OnceLock<BackingApi> = OnceLock::new();

type UploadApi = (fn(usize) -> usize, fn(usize));
static UPLOAD: OnceLock<UploadApi> = OnceLock::new();

/// List-owned upload allocations use the same RAM ledger and page IDs
/// as the runtime's private XMem backing.
pub fn use_upload_backing(alloc: fn(usize) -> usize, free: fn(usize)) {
    let _ = UPLOAD.set((alloc, free));
}

#[cfg(feature = "paged-title")]
pub(crate) fn allocate_upload(bytes: usize) -> Option<(usize, fn(usize))> {
    let &(alloc, free) = UPLOAD.get()?;
    let at = alloc(bytes);
    (at != 0).then_some((at, free))
}

/// Installed by the runtime before the title creates its D3D12 device.
pub fn use_backing(pinner: BackingPinner, writeback: WritebackPage, read: ReadRuns) {
    let _ = BACKING.set((pinner, writeback, read));
}

#[derive(Default)]
pub(crate) struct SubmissionBacking {
    pending: Vec<Range>,
    pins: BackingPins,
    #[cfg(test)]
    test_api: Option<BackingApi>,
}

impl SubmissionBacking {
    fn api(&self) -> Option<BackingApi> {
        #[cfg(test)]
        if self.test_api.is_some() {
            return self.test_api;
        }
        BACKING.get().copied()
    }

    pub(crate) fn include(&mut self, start: usize, end: usize) {
        if start < end && self.api().is_some() {
            self.pending.push((start, end));
        }
    }

    pub(crate) fn flush(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        merge(&mut self.pending);
        if let Some((pin, _, _)) = self.api() {
            // Called without a GPU page table/descriptor/shadow registry lock.
            pin(&self.pending, &mut self.pins);
        }
        self.pending.clear();
    }
    pub(crate) fn read_runs(
        &self,
        start: usize,
        end: usize,
        mut visit: impl FnMut(usize, Option<usize>, usize, bool),
    ) {
        if let Some((_, _, read)) = self.api() {
            read(start, end, &self.pins, &mut visit);
        } else {
            visit(start, Some(start), end - start, true);
        }
    }

    /// Captured identities for renderer batch planning after `flush`.
    pub(crate) fn pins(&self) -> &BackingPins {
        &self.pins
    }

    /// Snapshot a complete page, never a replacement at the same VA.
    /// `destination` is the retained, budgeted upload page and doubles as
    /// the before-image for the dirty-page writeback following the fence.
    pub(crate) fn stage_page(
        &self,
        va: usize,
        physical: PhysicalPage,
        destination: &mut [u8],
    ) -> bool {
        const PAGE: usize = PAGE_BYTES as usize;
        if !va.is_multiple_of(PAGE)
            || destination.len() != PAGE
            || self.pins.expected(va) != Some(physical)
        {
            return false;
        }
        let Some(end) = va.checked_add(PAGE) else {
            return false;
        };
        let mut read = 0;
        let mut ok = true;
        self.read_runs(va, end, |at, source, count, _| {
            ok &= source.is_some_and(|source| {
                crate::host::read_into(
                    source as *const u8,
                    &mut destination[at - va..at - va + count],
                )
            });
            read += count;
        });
        ok && read == PAGE
    }

    /// Publish proven GPU overwrites and other observed GPU changes,
    /// preserving independent title writes outside those byte ranges.
    /// Moving a mapping never invalidates its pinned original section.
    /// The caller may retire dirty residency only after a true result.
    pub(crate) fn merge_page(
        &self,
        va: usize,
        physical: PhysicalPage,
        before: &[u8],
        gpu: &[u8],
        scratch: &mut [u8],
        overwrites: &[std::ops::Range<usize>],
    ) -> bool {
        const PAGE: usize = PAGE_BYTES as usize;
        if !va.is_multiple_of(PAGE)
            || [before.len(), gpu.len(), scratch.len()] != [PAGE; 3]
            || self.pins.expected(va) != Some(physical)
        {
            return false;
        }
        if before == gpu && overwrites.is_empty() {
            return true;
        }
        self.write_delta(va, PAGE, |target| {
            if !crate::host::read_into(target.address as *const u8, scratch) {
                return false;
            }
            let mut changed = merge_page_delta(scratch, before, gpu);
            for span in overwrites {
                let current = &mut scratch[span.clone()];
                let result = &gpu[span.clone()];
                changed |= current != result;
                current.copy_from_slice(result);
            }
            if !changed {
                return true;
            }
            let _writing = target.original_va.then(|| crate::watch::writing(va, PAGE));
            crate::host::write_mapped_exact(target.address as *mut u8, scratch)
        })
        .unwrap_or(false)
    }

    /// For marked writes without a stable before-image, retain the stale
    /// destination guard. Only a computed byte delta can be redirected.
    pub(crate) fn write_page<T>(&self, address: usize, write: impl FnOnce() -> T) -> Option<T> {
        self.write_target(address, 1, false, |_| write())
    }

    pub(crate) fn write_delta<T>(
        &self,
        address: usize,
        len: usize,
        write: impl FnOnce(WritebackTarget) -> T,
    ) -> Option<T> {
        self.write_target(address, len, true, write)
    }

    fn write_target<T>(
        &self,
        address: usize,
        len: usize,
        redirect: bool,
        write: impl FnOnce(WritebackTarget) -> T,
    ) -> Option<T> {
        let Some((_, with_page, _)) = self.api() else {
            return Some(write(WritebackTarget {
                address,
                original_va: true,
            }));
        };
        let mut write = Some(write);
        let mut result = None;
        with_page(address, len, &self.pins, redirect, &mut |target| {
            result = write.take().map(|write| write(target));
        });
        result
    }
}

fn merge_page_delta(cpu: &mut [u8], before: &[u8], gpu: &[u8]) -> bool {
    let mut changed = 0u8;
    for ((current, &original), &result) in cpu.iter_mut().zip(before).zip(gpu) {
        let mask = 0u8.wrapping_sub(u8::from(result != original));
        let delta = (*current ^ result) & mask;
        *current ^= delta;
        changed |= delta;
    }
    changed != 0
}

#[cfg(test)]
#[test]
fn physical_page_delta_preserves_unwritten_cpu_bytes() {
    let baseline = [0xa0, 0xa0, 0xa0, 0xa0];
    let result = [0xa0, 0x77, 0xa0, 0x10];
    let mut cpu = [0x11, 0x22, 0x33, 0x10];
    assert!(merge_page_delta(&mut cpu, &baseline, &result));
    assert_eq!(cpu, [0x11, 0x77, 0x33, 0x10]);
    assert!(!merge_page_delta(&mut cpu, &baseline, &result));
}

#[cfg(test)]
pub(crate) fn test_capture(
    id: PhysicalPage,
    aliases: &[usize],
) -> (
    SubmissionBacking,
    std::sync::Arc<std::sync::Mutex<Box<[u8]>>>,
) {
    use std::sync::{Arc, Mutex};
    const PAGE: usize = PAGE_BYTES as usize;
    type Section = Mutex<Box<[u8]>>;

    // Simulate an old section surviving with no original title VA, while
    // retaining validated pointers to it only inside the callbacks.
    let read: ReadRuns = |at, end, pins, visit| {
        let Some((old, _)) = pins.expected(at).and_then(|id| pins.section::<Section>(id)) else {
            visit(at, None, end - at, false);
            return;
        };
        let old = old.lock().unwrap();
        visit(at, Some(old.as_ptr() as usize), end - at, false);
    };
    let write: WritebackPage = |at, _, pins, redirect, visit| {
        if !redirect {
            return false;
        }
        let Some((old, _)) = pins.expected(at).and_then(|id| pins.section::<Section>(id)) else {
            return false;
        };
        let mut old = old.lock().unwrap();
        visit(WritebackTarget {
            address: old.as_mut_ptr() as usize,
            original_va: false,
        });
        true
    };
    let mut submission = SubmissionBacking {
        test_api: Some((|_, _| {}, write, read)),
        ..Default::default()
    };
    let section = Arc::new(Mutex::new(vec![0x11u8; PAGE].into_boxed_slice()));
    submission.pins.keep(id, &section);
    for &va in aliases {
        submission.pins.remember(va, va + PAGE, Some(id));
    }
    (submission, section)
}

#[cfg(test)]
#[test]
fn captured_page_snapshot_and_merge_use_original_section() {
    use victory_memory::physical::PageNumbers;
    const PAGE: usize = PAGE_BYTES as usize;
    const VA: usize = PAGE;
    let mut ids = PageNumbers::default();
    let first = ids.allocate(1).unwrap();
    let replacement = ids.allocate(1).unwrap();
    let (submission, section) = test_capture(first, &[VA]);
    let mut before = vec![0; PAGE];
    assert!(!submission.stage_page(VA, replacement, &mut before));
    assert!(submission.stage_page(VA, first, &mut before));
    assert!(before.iter().all(|&byte| byte == 0x11));
    let mut gpu = before.clone();
    gpu[2] = 0x77;
    let mut scratch = vec![0; PAGE];
    section.lock().unwrap()[1] = 0xab;
    assert!(!submission.merge_page(VA, replacement, &before, &gpu, &mut scratch, &[]));
    assert!(submission.merge_page(VA, first, &before, &gpu, &mut scratch, &[]));
    assert_eq!(&section.lock().unwrap()[..4], &[0x11, 0xab, 0x77, 0x11]);

    // A GPU overwrite equal to its before-image still owns that byte;
    // the neighboring CPU byte was not part of the operation.
    assert!(submission.merge_page(VA, first, &before, &before, &mut scratch, &[2..3]));
    assert_eq!(&section.lock().unwrap()[..4], &[0x11, 0xab, 0x11, 0x11]);
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
    let mut ranges = vec![
        (40, 52),
        (8, 16),
        (4, 6),
        (51, 60),
        (16, 20),
        (44, 48),
        (4, 6),
    ];
    merge(&mut ranges);
    assert_eq!(ranges, [(4, 6), (8, 20), (40, 60)]);
}
