//! Boundary between command preparation and the title's actual XMem backing.
//! A pin retains a physical allocation, not the title's virtual mapping.

use std::sync::OnceLock;

use victory_memory::{
    PAGE_BYTES,
    physical::{BackingPins, PhysicalPage},
};

#[cfg(all(feature = "paged-title", feature = "diagnostics"))]
mod dstorage;
#[cfg(all(feature = "paged-title", feature = "diagnostics"))]
pub use dstorage::{DstorageVaWrite, note_dstorage_section_write, probe_dstorage_va_write};
#[cfg(all(feature = "paged-title", feature = "diagnostics"))]
pub(crate) use dstorage::trace_dstorage_write_history;
#[cfg(all(feature = "paged-title", feature = "diagnostics"))]
mod section_history;
#[cfg(all(feature = "paged-title", feature = "diagnostics"))]
pub use section_history::{note_xmem_section_mapping, prepare_xmem_section_history};
#[cfg(all(feature = "paged-title", feature = "diagnostics"))]
pub(crate) use section_history::trace_color_section_history;

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

#[cfg(feature = "paged-title")]
static PRIVATE_HEAP_SPAN: OnceLock<fn(usize, usize) -> bool> = OnceLock::new();

/// 仅识别由本运行时实际分配且仍存活的系统堆范围。
#[cfg(feature = "paged-title")]
pub fn use_private_heap_span(prove: fn(usize, usize) -> bool) {
    let _ = PRIVATE_HEAP_SPAN.set(prove);
}

#[cfg(feature = "paged-title")]
pub(crate) fn is_private_heap_span(start: usize, end: usize) -> bool {
    PRIVATE_HEAP_SPAN.get().is_some_and(|prove| prove(start, end))
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
        // Explicit copies, clears and markers commonly account for every
        // changed byte. Publish their spans without reading or rewriting
        // neighboring CPU bytes, even if they changed during retirement.
        let exact_only = !overwrites.is_empty() && changes_contained(before, gpu, overwrites);
        let mut mixed = [const { 0..0 }; SMALL_WRITEBACK];
        let spans = if exact_only {
            Some(overwrites)
        } else {
            sparse_writeback(before, gpu, overwrites, &mut mixed).map(|count| &mixed[..count])
        };
        self.write_delta(va, PAGE, |target| {
            if let Some(spans) = spans {
                let _writing = target.original_va.then(|| crate::watch::writing(va, PAGE));
                return spans.iter().all(|span| crate::host::write_mapped_exact(
                    (target.address + span.start) as *mut u8, &gpu[span.clone()],
                ));
            }
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

/// The planner keeps these intervals sorted, nonempty and disjoint.
fn changes_contained(before: &[u8], gpu: &[u8], overwrites: &[std::ops::Range<usize>]) -> bool {
    let mut cursor = 0;
    for span in overwrites {
        debug_assert!(span.start >= cursor && span.start < span.end && span.end <= before.len());
        if before[cursor..span.start] != gpu[cursor..span.start] { return false; }
        cursor = span.end;
    }
    before[cursor..] == gpu[cursor..]
}

/// Bound extra kernel copies for pages with fragmented shader writes.
const SMALL_WRITEBACK: usize = 8;

/// Combine declared intervals and observed differences. A gap is never
/// included in a direct store just to stay under the syscall budget.
fn sparse_writeback(
    before: &[u8], gpu: &[u8], overwrites: &[std::ops::Range<usize>],
    output: &mut [std::ops::Range<usize>],
) -> Option<usize> {
    let mut count = 0;
    let mut next = overwrites.iter().peekable();
    let mut at = 0;
    while at < before.len() {
        let end = if let Some(span) = next.next_if(|span| span.start == at) {
            span.end
        } else if before[at] != gpu[at] {
            at + 1
        } else {
            at += 1;
            continue;
        };
        if count > 0 && output[count - 1].end == at {
            output[count - 1].end = end;
        } else {
            *output.get_mut(count)? = at..end;
            count += 1;
        }
        at = end;
    }
    Some(count)
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
fn sparse_writeback_keeps_gaps_and_adjacent_writes_exact() {
    let mut output = [0..0, 0..0];
    assert_eq!(sparse_writeback(&[0; 8], &[0, 1, 0, 0, 0, 0, 1, 0], &[2..3], &mut output), Some(2));
    assert_eq!(output, [1..3, 6..7]);
    assert_eq!(sparse_writeback(&[0; 8], &[0, 1, 0, 1, 0, 1, 0, 0], &[], &mut output), None);
}

#[cfg(test)]
#[test]
fn exact_overwrites_require_all_other_bytes_to_match() {
    let before = [1, 2, 3, 4, 5, 6];
    let gpu = [1, 9, 3, 4, 8, 6];
    assert!(changes_contained(&before, &gpu, &[1..2, 4..5]));
    assert!(!changes_contained(&before, &gpu, &[1..2]));
    assert!(!changes_contained(&before, &gpu, &[4..5]));
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
