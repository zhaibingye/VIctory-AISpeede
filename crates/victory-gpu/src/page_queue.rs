//! Native direct-queue submission for the unified-memory engine. The
//! caller records only commands whose views use the same physical pages.
use std::collections::BTreeMap;
use std::ptr::NonNull;

use victory_memory::{
    PAGE_BYTES,
    coherence::Fence,
    gpu_table::RawPageTable,
    physical::{HostBudget, PhysicalPage},
};
use windows::Win32::Foundation::{E_FAIL, E_OUTOFMEMORY};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{Error, Interface, Result};

use crate::descriptor::host::HostHeap;
use crate::host::{Host, HostList, fail, transition};
use crate::page_slabs::{BatchError, CapturedUploads, PinnedBatch, Transfer, download};
use crate::title_backing::SubmissionBacking;

#[derive(Clone, Copy)]
pub(crate) struct PageRoots {
    pub(crate) table: u64,
    pub(crate) dirty: u64,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct PageStats {
    pub(crate) uploaded: usize,
    pub(crate) downloaded: usize,
}

struct Completion {
    fence: Fence,
    stats: PageStats,
    digests: BTreeMap<PhysicalPage, blake3::Hash>,
}

/// One heap's reserved descriptor slots. They are released only after the
/// direct-queue fence, or when a recording was never executed.
struct Pairs<'a> {
    heap: &'a HostHeap,
    slots: Vec<u32>,
}

impl Pairs<'_> {
    fn reserve(&mut self, slab: Option<&ID3D12Resource>, bytes: u64) -> Result<u32> {
        let slot = self.heap.pair().ok_or_else(oom)?;
        self.heap.raw_pair_at(slot, slab, bytes);
        self.slots.push(slot);
        Ok(slot)
    }
}

impl Drop for Pairs<'_> {
    fn drop(&mut self) {
        for slot in self.slots.drain(..) {
            self.heap.raw_pair_at(slot, None, 0);
            self.heap.free_pair(slot);
        }
    }
}

fn oom() -> Error {
    Error::from(E_OUTOFMEMORY)
}

impl Host {
    /// Owns the native page working set under the same direct-queue lock
    /// as title submissions. There is no legacy fallback if the working
    /// set, snapshot or descriptor plan is incomplete. Recording failures
    /// before submission leave the title untouched; failures after native
    /// dispatch terminate rather than continue with a partial GPU result.
    pub(crate) fn execute_paged(
        &self,
        heap: &HostHeap,
        backing: &mut SubmissionBacking,
        ranges: &[(u64, u64)],
        record: impl FnOnce(&HostList, PageRoots) -> Result<()>,
    ) -> std::result::Result<PageStats, BatchError> {
        let mut signal = self.direct_queue();
        let mut slabs = self.paged.lock().unwrap_or_else(|e| e.into_inner());
        let batch = PinnedBatch::capture(self, &mut slabs, backing, ranges)?;
        let result = run(self, &mut signal, heap, backing, &batch, record);
        match result {
            Ok(completed) => {
                // The readback was merged before the pins are relinquished.
                // The next submission revalidates its own CPU snapshot.
                batch.submitted(&mut slabs, completed.fence, |_| (true, false));
                slabs.publish_completed_content(completed.digests);
                Ok(completed.stats)
            }
            Err(e) => {
                batch.cancel(&mut slabs);
                Err(e)
            }
        }
    }
}

fn run(
    host: &Host,
    signal: &mut u64,
    heap: &HostHeap,
    backing: &SubmissionBacking,
    batch: &PinnedBatch,
    record: impl FnOnce(&HostList, PageRoots) -> Result<()>,
) -> std::result::Result<Completion, BatchError> {
    let native = BatchError::Native;
    let mut pairs = Pairs {
        heap,
        slots: Vec::new(),
    };
    let mut raw = RawPageTable::new(pairs.reserve(None, 0).map_err(native)?);
    let leaves = batch.bind(&mut raw, |pin| {
        pairs.reserve(Some(pin.buffer()), crate::page_slabs::SLAB_BYTES)
    })?;
    let (words, blocks) = raw.into_parts();
    let table_bytes = words.len() * 4;
    let bitmap_words = blocks.len() * victory_memory::gpu_table::BLOCK / 32;
    let (table, table_ptr) = host
        .upload_buffer(table_bytes as u64)
        .ok_or_else(|| native(oom()))?;
    let (bitmap, bitmap_ptr) = host
        .shared_buffer((bitmap_words * 4) as u64)
        .ok_or_else(|| native(oom()))?;
    let table_ptr = NonNull::new(table_ptr).ok_or_else(|| native(oom()))?;
    let bitmap_ptr = NonNull::new(bitmap_ptr).ok_or_else(|| native(oom()))?;
    let uploads = CapturedUploads::stage(host, batch, backing)?;
    let list = host.new_list().map_err(native)?;
    let slabs: BTreeMap<_, _> = batch
        .pages()
        .map(|(_, _, page)| (page.slab(), page.buffer()))
        .collect();
    for buffer in slabs.values() {
        list.keep(*buffer);
    }
    list.keep(&bitmap);
    list.keep(&table);
    list.keep(&heap.heap);
    // SAFETY: mapped backings with the allocated sizes; the queue has not
    // run. No CPU slice into bitmap data exists while commands run.
    let roots = unsafe {
        std::ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), table_ptr.as_ptr(), table_bytes);
        bitmap_ptr.as_ptr().write_bytes(0, bitmap_words * 4);
        PageRoots {
            table: table.GetGPUVirtualAddress(),
            dirty: bitmap.GetGPUVirtualAddress(),
        }
    };
    let uploaded = uploads.as_ref().map_or(0, |stage| stage.record(&list));
    barrier(&list, slabs.values().copied(), false);
    record(&list, roots).map_err(native)?;
    barrier(&list, slabs.values().copied(), true);
    list.close();
    let command = list.list.cast().map_err(native)?;
    let rendered = host
        .run_native_group(signal, &[Some(command)])
        .unwrap_or_else(|e| fail(format_args!("native paged render submission: {e}")));
    if host.completed_fence().is_none() {
        fail(format_args!("native paged device was removed"));
    }

    // SAFETY: the rendering fence completed; this bitmap is fully
    // initialized and no other submission uses it.
    let bitmap_slice =
        unsafe { std::slice::from_raw_parts(bitmap_ptr.as_ptr().cast::<u32>(), bitmap_words) };
    let dirty = leaves.dirty_pages(bitmap_slice).unwrap_or_else(|leaf| {
        fail(format_args!(
            "paged dispatch accessed unbound page-table leaf {leaf}"
        ))
    });
    let mut digests = uploads
        .as_ref()
        .map_or_else(BTreeMap::new, CapturedUploads::digests);
    let stats = PageStats {
        uploaded,
        downloaded: dirty.len(),
    };
    if dirty.is_empty() {
        return Ok(Completion {
            fence: rendered,
            stats,
            digests,
        });
    }

    let readback = (|| -> Result<_> {
        let data = host
            .shared_buffer(dirty.len() as u64 * PAGE_BYTES)
            .ok_or_else(oom)?;
        let mapped = NonNull::new(data.1).ok_or_else(oom)?;
        let transfer = host.new_list()?;
        let entries: Vec<_> = batch
            .pages()
            .filter(|(id, _, _)| dirty.contains(id))
            .enumerate()
            .map(|(i, (_, _, page))| Transfer {
                page,
                staging: &data.0,
                offset: i as u64 * PAGE_BYTES,
            })
            .collect();
        download(&transfer, &entries);
        transfer.close();
        let completed = host.run_native_group(signal, &[Some(transfer.list.cast()?)])?;
        if host.completed_fence().is_none() {
            return Err(E_FAIL.into());
        }
        Ok((data.0, mapped, completed))
    })()
    .unwrap_or_else(|e| fail(format_args!("native paged dirty readback: {e}")));
    let (_keep_readback, mapped, completed) = readback;
    let snapshots = uploads.expect("dirty leaf requires an uploaded page");
    let _scratch_charge = HostBudget::process()
        .reserve_ram(PAGE_BYTES)
        .unwrap_or_else(|_| fail(format_args!("no RAM for paged writeback scratch")));
    let mut scratch = vec![0; PAGE_BYTES as usize];
    for (i, (page, _, _)) in batch
        .pages()
        .filter(|(id, _, _)| dirty.contains(id))
        .enumerate()
    {
        // SAFETY: one initialized dirty page in the completed readback.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                mapped.as_ptr().add(i * PAGE_BYTES as usize),
                PAGE_BYTES as usize,
            )
        };
        if !snapshots.merge(backing, page, bytes, &mut scratch) {
            fail(format_args!(
                "paged writeback failed for physical page {}",
                page.number()
            ));
        }
        digests.insert(page, blake3::hash(bytes));
    }
    Ok(Completion {
        fence: completed,
        stats,
        digests,
    })
}

fn barrier<'a>(
    list: &HostList,
    resources: impl Iterator<Item = &'a ID3D12Resource>,
    restore: bool,
) {
    let common = D3D12_RESOURCE_STATE_COMMON;
    let unordered = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    let (before, after) = if restore {
        (unordered, common)
    } else {
        (common, unordered)
    };
    let barriers: Vec<_> = resources
        .map(|res| transition(res, before, after))
        .collect();
    if !barriers.is_empty() {
        // SAFETY: one barrier per live slab retained by the list.
        unsafe { list.list.ResourceBarrier(&barriers) };
    }
}

#[cfg(test)]
#[test]
fn paged_direct_queue_merges_only_marked_physical_pages() {
    use victory_memory::physical::PageNumbers;
    const PAGE: usize = PAGE_BYTES as usize;
    const VA: usize = PAGE;
    const ALIAS: usize = 0x91_0001_0000;
    let host = crate::host::host().expect("D3D12 renderer host (CI selects WARP)");
    let heap = crate::memory::heap().unwrap();
    let id = PageNumbers::default().allocate(1).unwrap();
    let (mut backing, section) = crate::title_backing::test_capture(id, &[VA, ALIAS]);
    const SOURCE: &str = r#"
#define RS "DescriptorTable(UAV(u0, numDescriptors=unbounded, space=1)), SRV(t0, space=2), UAV(u0), RootConstants(num32BitConstants=2, b0)"
RWByteAddressBuffer slabs[] : register(u0, space1);
ByteAddressBuffer table : register(t0, space2);
RWByteAddressBuffer dirty : register(u0);
cbuffer c : register(b0) { uint lo; uint hi; }
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main(uint3 thread : SV_DispatchThreadID) {
    uint block = table.Load(((hi << 4) | (lo >> 28)) * 4);
    uint leaf = (block << 12) | ((lo >> 16) & 4095);
    uint2 binding = table.Load2((4096 + 2 * leaf) * 4);
    uint view = NonUniformResourceIndex(binding.x + 1);
    uint offset = lo - binding.y;
    uint original = slabs[view].Load(offset);
    slabs[view].Store(offset, original ^ 0x00002200);
    uint ignored;
    dirty.InterlockedOr((leaf >> 5) * 4, 1u << (leaf & 31), ignored);
}
"#;
    let (root, state) = crate::memory::tests::pipeline(SOURCE);
    let ranges = [
        (VA as u64, (VA + PAGE) as u64),
        (ALIAS as u64, (ALIAS + PAGE) as u64),
    ];
    let before = host.completed_fence().unwrap();
    let first = host
        .execute_paged(heap, &mut backing, &ranges, |list, roots| {
            // A CPU write lands after snapshot; the GPU never touched byte 0.
            section.lock().unwrap()[0] = 0xa5;
            // SAFETY: paged queue keeps heap, table, bitmap and slab pins
            // through this dispatch and native readback.
            unsafe {
                let cmd = &list.list;
                cmd.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
                cmd.SetComputeRootSignature(&root);
                cmd.SetComputeRootDescriptorTable(0, heap.gpu(0));
                cmd.SetComputeRootShaderResourceView(1, roots.table);
                cmd.SetComputeRootUnorderedAccessView(2, roots.dirty);
                let args = [ALIAS as u32, (ALIAS as u64 >> 32) as u32];
                cmd.SetComputeRoot32BitConstants(3, 2, args.as_ptr().cast(), 0);
                cmd.SetPipelineState(&state);
                cmd.Dispatch(1, 1, 1);
            }
            Ok(())
        })
        .unwrap();
    assert_eq!(&section.lock().unwrap()[..4], &[0xa5, 0x33, 0x11, 0x11]);
    assert!(host.completed_fence().unwrap().0 >= before.0 + 2);
    assert_eq!(
        first,
        PageStats {
            uploaded: 1,
            downloaded: 1
        }
    );
    // The first upload only saw 0x11 at byte 0; the previous dispatch
    // preserved the CPU's 0xa5, so update the GPU once more, then reuse it.
    let refresh = host
        .execute_paged(heap, &mut backing, &ranges, |_, _| Ok(()))
        .unwrap();
    assert_eq!(
        refresh,
        PageStats {
            uploaded: 1,
            downloaded: 0
        }
    );
    let reused = host
        .execute_paged(heap, &mut backing, &ranges, |_, _| Ok(()))
        .unwrap();
    assert_eq!(
        reused,
        PageStats {
            uploaded: 0,
            downloaded: 0
        }
    );
}
