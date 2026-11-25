//! Native direct-queue submission for the unified-memory engine. The
//! caller records only commands whose views use the same physical pages.
use std::collections::{BTreeMap, BTreeSet};
use std::ptr::NonNull;

use victory_memory::{
    PAGE_BYTES,
    coherence::Fence,
    gpu_table::RawPageTable,
    physical::{BackingPins, HostBudget, PhysicalPage},
};
use windows::Win32::Foundation::{E_FAIL, E_OUTOFMEMORY};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{Error, Interface, Result};

use crate::copy_plan::{GuestCopy, Unbound};
use crate::descriptor::host::HostHeap;
use crate::host::{Host, HostList, fail, transition};
use crate::page_view::InputView;
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

#[derive(Clone)]
pub(crate) enum PageOp {
    Copy(GuestCopy),
    Input(InputView),
    Draw(std::sync::Arc<crate::page_draw::ClosedDraw>),
    Dispatch(std::sync::Arc<crate::page_dispatch::ClosedDispatch>),
    Clear(std::sync::Arc<crate::page_draw::ColorClear>),
    ClearDepth(std::sync::Arc<crate::page_draw::DepthClear>),
    Barrier(std::sync::Arc<crate::page_resource::PageBarrier>),
    ClearRaw(std::sync::Arc<crate::page_uav::RawClear>),
}

struct Completion {
    fence: Fence,
    stats: PageStats,
    digests: BTreeMap<PhysicalPage, blake3::Hash>,
}

/// One heap's reserved descriptor slots. They are released only after the
/// direct-queue fence, or when a recording was never executed.
pub(crate) struct PageDescriptors<'a> {
    pub(crate) heap: &'a HostHeap,
    slots: Vec<u32>,
    textures: Vec<(u32, windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT)>,
    raw_uavs: Vec<u32>,
}

impl PageDescriptors<'_> {
    fn reserve(&mut self, slab: Option<&ID3D12Resource>, bytes: u64) -> Result<u32> {
        let slot = self.heap.pair().ok_or_else(oom)?;
        self.heap.raw_pair_at(slot, slab, bytes);
        self.slots.push(slot);
        Ok(slot)
    }

    pub(crate) fn reserve_raw_uav(
        &mut self, buffer: &ID3D12Resource, offset: u64, bytes: u64,
    ) -> Result<u32> {
        let slot = self.heap.dynamic(1).ok_or_else(oom)?;
        self.heap.raw_view_at(slot, Some(buffer), offset, bytes, true);
        self.raw_uavs.push(slot);
        Ok(slot)
    }

    pub(crate) fn reserve_texture_run(
        &mut self,
        views: &[Option<(ID3D12Resource, windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT)>],
    ) -> Result<u32> {
        let n = u32::try_from(views.len()).map_err(|_| oom())?;
        let first = self.heap.dynamic(n).ok_or_else(oom)?;
        for (offset, view) in views.iter().enumerate() {
            use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM;
            let slot = first + offset as u32;
            let (resource, format) = view.as_ref().map_or(
                (None, DXGI_FORMAT_R8G8B8A8_UNORM), |(tex, f)| (Some(tex), *f),
            );
            self.heap.paged_texture_at(slot, resource, format);
            self.textures.push((slot, format));
        }
        Ok(first)
    }

    pub(crate) fn reserve_texture(
        &mut self,
        texture: &ID3D12Resource,
        format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    ) -> Result<u32> {
        let slot = self.heap.dynamic(1).ok_or_else(oom)?;
        self.heap.paged_texture_at(slot, Some(texture), format);
        self.textures.push((slot, format));
        Ok(slot)
    }
}

impl Drop for PageDescriptors<'_> {
    fn drop(&mut self) {
        for slot in self.raw_uavs.drain(..) {
            self.heap.raw_view_at(slot, None, 0, 0, true);
            self.heap.release(slot, 1);
        }
        for (slot, format) in self.textures.drain(..) {
            self.heap.paged_texture_at(slot, None, format);
            self.heap.release(slot, 1);
        }
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
        self.execute_paged_work(heap, backing, ranges, |list, roots, _, _, _, _| {
            record(list, roots).map_err(BatchError::Native)?;
            Ok(BTreeSet::new())
        })
    }

    /// Encoder boundary for a closed queue submission recorded by the
    /// title list handlers. Reject the whole submission before touching
    /// backing if any list still contains operations we cannot lower.
    pub(crate) fn execute_paged_title(
        &self,
        heap: &HostHeap,
        backing: &mut SubmissionBacking,
        lists: &[&HostList],
    ) -> std::result::Result<PageStats, BatchError> {
        let mut ops = Vec::new();
        for (index, list) in lists.iter().enumerate() {
            if !list.append_closed_page_ops(&mut ops) {
                return Err(BatchError::UnloweredList(index));
            }
        }
        self.execute_paged_ops(heap, backing, &ops)
    }

    /// Run a closed sequence of buffer copies against one capture and
    /// perform a single dirty-page merge after the whole sequence.
    pub(crate) fn execute_paged_copies(
        &self,
        heap: &HostHeap,
        backing: &mut SubmissionBacking,
        copies: &[GuestCopy],
    ) -> std::result::Result<PageStats, BatchError> {
        let ops: Vec<_> = copies.iter().copied().map(PageOp::Copy).collect();
        self.execute_paged_ops(heap, backing, &ops)
    }

    fn execute_paged_ops(
        &self,
        heap: &HostHeap,
        backing: &mut SubmissionBacking,
        ops: &[PageOp],
    ) -> std::result::Result<PageStats, BatchError> {
        if ops.is_empty() {
            return Ok(PageStats::default());
        }
        let mut ranges = Vec::new();
        let mut max_copy = 0;
        for op in ops {
            match op {
                PageOp::Copy(copy) => {
                    ranges.extend(copy.ranges());
                    max_copy = max_copy.max(copy.len);
                }
                PageOp::Input(input) => ranges.push(input.range()?),
                PageOp::Draw(draw) => draw.ranges(&mut ranges)?,
                PageOp::ClearRaw(clear) => ranges.push(clear.range()?),
                PageOp::Dispatch(dispatch) => dispatch.ranges(&mut ranges)?,
                PageOp::Clear(_) | PageOp::ClearDepth(_) | PageOp::Barrier(_) => {},
            }
        }
        self.execute_paged_work(heap, backing, &ranges, |list, _, batch, pins, descriptors, targets| {
            let scratch = if max_copy > 0 {
                let buffer = self.default_buffer(max_copy as u64)
                    .ok_or_else(|| BatchError::Native(oom()))?;
                list.keep(&buffer);
                Some(buffer)
            } else {
                None
            };
            let mut dirty = BTreeSet::new();
            let mut indirect_budget = None;
            for op in ops {
                match op {
                    PageOp::Copy(copy) => {
                        let pieces = copy.bind(pins).map_err(|miss| match miss {
                            Unbound::AddressWidth => BatchError::Range(copy.src as u64, copy.dst as u64),
                            Unbound::Source(va) | Unbound::Destination(va) => BatchError::NotCaptured(va as u64),
                        })?;
                        record_copy(list, batch, scratch.as_ref().expect("copy reserved scratch"), &pieces, &mut dirty)
                            .map_err(BatchError::Native)?;
                    }
                    PageOp::Input(input) => input.encode(self, list, batch)?,
                    PageOp::Dispatch(dispatch) => dispatch.encode(self, list, batch, &mut indirect_budget, descriptors)?,
                    PageOp::Draw(draw) => draw.encode(self, list, batch, &mut indirect_budget, descriptors, targets)?,
                    PageOp::Clear(clear) => clear.encode(self, list, targets)?,
                    PageOp::ClearDepth(clear) => clear.encode(self, list, targets)?,
                    PageOp::Barrier(barrier) => barrier.encode(list, targets)?,
                    PageOp::ClearRaw(clear) => clear.encode(self, list, batch, descriptors, targets, &mut dirty)?,
                }
            }
            Ok(dirty)
        })
    }

    fn execute_paged_work(
        &self,
        heap: &HostHeap,
        backing: &mut SubmissionBacking,
        ranges: &[(u64, u64)],
        record: impl FnOnce(
            &HostList,
            PageRoots,
            &PinnedBatch,
            &BackingPins,
            &mut PageDescriptors<'_>,
            &mut crate::page_resource::ResourcePlan,
        ) -> std::result::Result<BTreeSet<PhysicalPage>, BatchError>,
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
    record: impl FnOnce(
        &HostList,
        PageRoots,
        &PinnedBatch,
        &BackingPins,
        &mut PageDescriptors<'_>,
        &mut crate::page_resource::ResourcePlan,
    ) -> std::result::Result<BTreeSet<PhysicalPage>, BatchError>,
) -> std::result::Result<Completion, BatchError> {
    let native = BatchError::Native;
    let mut pairs = PageDescriptors {
        heap,
        slots: Vec::new(),
        textures: Vec::new(),
        raw_uavs: Vec::new(),
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
    let mut targets = crate::page_resource::ResourcePlan::default();
    let declared_dirty = record(&list, roots, batch, backing.pins(), &mut pairs, &mut targets)?;
    barrier(&list, slabs.values().copied(), true);
    list.close();
    let command = list.recording().cast().map_err(native)?;
    let rendered = host
        .run_native_group(signal, &[Some(command)])
        .unwrap_or_else(|e| fail(format_args!("native paged render submission: {e}")));
    if host.completed_fence().is_none() {
        fail(format_args!("native paged device was removed"));
    }

    // Publish authority only after the native queue really executed it.
    targets.commit();

    // SAFETY: the rendering fence completed; this bitmap is fully
    // initialized and no other submission uses it.
    let bitmap_slice =
        unsafe { std::slice::from_raw_parts(bitmap_ptr.as_ptr().cast::<u32>(), bitmap_words) };
    let mut dirty = leaves.dirty_pages(bitmap_slice).unwrap_or_else(|leaf| {
        fail(format_args!(
            "paged dispatch accessed unbound page-table leaf {leaf}"
        ))
    });
    dirty.extend(declared_dirty);
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
        let completed = host.run_native_group(signal, &[Some(transfer.recording().cast()?)])?;
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

/// Gather all source fragments into disjoint scratch before scattering.
/// A D3D12 buffer cannot be COPY_SOURCE and COPY_DEST simultaneously,
/// and otherwise an earlier write could corrupt a later alias read.
fn record_copy(
    list: &HostList,
    batch: &PinnedBatch,
    scratch: &ID3D12Resource,
    pieces: &[crate::copy_plan::CopyPiece],
    dirty: &mut BTreeSet<PhysicalPage>,
) -> Result<()> {
    let mut source = BTreeMap::new();
    let mut dest = BTreeMap::new();
    for piece in pieces {
        let s = batch.pinned(piece.src).ok_or_else(|| Error::from(E_FAIL))?;
        source.insert(s.slab(), s.buffer());
        let d = batch.pinned(piece.dst).ok_or_else(|| Error::from(E_FAIL))?;
        dest.insert(d.slab(), d.buffer());
    }
    let uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    let copy_src = D3D12_RESOURCE_STATE_COPY_SOURCE;
    let copy_dst = D3D12_RESOURCE_STATE_COPY_DEST;
    // SAFETY: native buffers are owned by the same captured batch; all
    // byte spans are bounded by its guest copy plan.
    unsafe {
        let cmd = list.recording();
        cmd.ResourceBarrier(
            &source
                .values()
                .map(|s| transition(s, uav, copy_src))
                .collect::<Vec<_>>(),
        );
        let mut offset = 0;
        for piece in pieces {
            let s = batch.pinned(piece.src).expect("validated source");
            cmd.CopyBufferRegion(
                scratch,
                offset,
                s.buffer(),
                s.offset() + u64::from(piece.src_offset),
                u64::from(piece.count),
            );
            offset += u64::from(piece.count);
        }
        let restore_source: Vec<_> = source
            .values()
            .map(|s| transition(s, copy_src, uav))
            .collect();
        cmd.ResourceBarrier(&restore_source);
        let scatter: Vec<_> = dest
            .values()
            .map(|d| transition(d, uav, copy_dst))
            .chain([transition(scratch, copy_dst, copy_src)])
            .collect();
        cmd.ResourceBarrier(&scatter);
        offset = 0;
        for piece in pieces {
            let d = batch.pinned(piece.dst).expect("validated destination");
            cmd.CopyBufferRegion(
                d.buffer(),
                d.offset() + u64::from(piece.dst_offset),
                scratch,
                offset,
                u64::from(piece.count),
            );
            offset += u64::from(piece.count);
            dirty.insert(piece.dst);
        }
        cmd.ResourceBarrier(
            &dest
                .values()
                .map(|d| transition(d, copy_dst, uav))
                .chain([transition(scratch, copy_src, copy_dst)])
                .collect::<Vec<_>>(),
        );
    }
    Ok(())
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
        unsafe { list.recording().ResourceBarrier(&barriers) };
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
                let cmd = list.recording();
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

#[cfg(test)]
#[test]
fn paged_buffer_copy_gathers_overlapping_physical_aliases() {
    use victory_memory::physical::PageNumbers;
    const VA: usize = PAGE_BYTES as usize;
    const ALIAS: usize = 0x81_0000_0000;
    let host = crate::host::host().expect("D3D12 renderer host (CI selects WARP)");
    let heap = crate::memory::heap().unwrap();
    let id = PageNumbers::default().allocate(2).unwrap().offset(1);
    let (mut backing, section) = crate::title_backing::test_capture(id, &[VA, ALIAS]);
    section.lock().unwrap()[..6].copy_from_slice(&[10, 20, 30, 40, 50, 60]);
    let copy = GuestCopy::new(ALIAS + 1, VA, 5).unwrap();
    let stats = host.execute_paged_copies(heap, &mut backing, &[copy]).unwrap();
    assert_eq!(
        stats,
        PageStats {
            uploaded: 1,
            downloaded: 1
        }
    );
    assert_eq!(&section.lock().unwrap()[..6], &[10, 10, 20, 30, 40, 50]);
}

#[cfg(test)]
#[test]
fn paged_buffer_copy_batch_orders_alias_dependencies_and_rejects_partial_work() {
    use victory_memory::physical::PageNumbers;
    const VA: usize = PAGE_BYTES as usize;
    const ALIAS: usize = 0x82_0000_0000;
    let host = crate::host::host().expect("D3D12 renderer host (CI selects WARP)");
    let heap = crate::memory::heap().unwrap();
    let id = PageNumbers::default().allocate(3).unwrap().offset(2);
    let (mut backing, section) = crate::title_backing::test_capture(id, &[VA, ALIAS]);
    let initial = [10, 20, 30, 40, 50, 60];
    section.lock().unwrap()[..6].copy_from_slice(&initial);
    let first = GuestCopy::new(ALIAS + 1, VA, 5).unwrap();
    let missing = GuestCopy::new(VA, VA + PAGE_BYTES as usize, 2).unwrap();
    assert!(matches!(
        host.execute_paged_copies(heap, &mut backing, &[first, missing]),
        Err(BatchError::NotCaptured(_))
    ));
    assert_eq!(&section.lock().unwrap()[..6], &initial);
    let second = GuestCopy::new(VA, ALIAS + 2, 2).unwrap();
    let stats = host
        .execute_paged_copies(heap, &mut backing, &[first, second])
        .unwrap();
    assert_eq!(
        stats,
        PageStats {
            uploaded: 1,
            downloaded: 1,
        }
    );
    assert_eq!(&section.lock().unwrap()[..6], &[20, 30, 20, 30, 40, 50]);
}

#[cfg(test)]
#[test]
fn paged_title_recording_rejects_mixed_lists_before_alias_copy() {
    use victory_memory::physical::PageNumbers;
    const VA: usize = PAGE_BYTES as usize;
    const ALIAS: usize = 0x83_0000_0000;
    let host = crate::host::host().expect("D3D12 renderer host (CI selects WARP)");
    let heap = crate::memory::heap().unwrap();
    let id = PageNumbers::default().allocate(4).unwrap().offset(3);
    let (mut backing, section) = crate::title_backing::test_capture(id, &[VA, ALIAS]);
    let first = host.new_list().unwrap();
    let second = host.new_list().unwrap();
    first.defer_copy((ALIAS + 1) as *mut u8, VA as *const u8, 5);
    second.defer_copy(VA as *mut u8, (ALIAS + 2) as *const u8, 2);
    first.close();
    let initial = [10, 20, 30, 40, 50, 60];
    section.lock().unwrap()[..6].copy_from_slice(&initial);
    assert!(matches!(
        host.execute_paged_title(heap, &mut backing, &[&first, &second]),
        Err(BatchError::UnloweredList(1))
    ));
    // The native recorder is exposed; even a no-op native body is not
    // provably all copies. No early list should be submitted.
    let _ = second.recording();
    second.close();
    assert!(matches!(
        host.execute_paged_title(heap, &mut backing, &[&first, &second]),
        Err(BatchError::UnloweredList(1))
    ));
    assert_eq!(&section.lock().unwrap()[..6], &initial);
    second.reset();
    let vertex = crate::page_view::VertexView { address: VA as u64, size: 4, stride: 4 };
    second.record_page_input(InputView::Vertex(32, vertex));
    second.close();
    assert!(matches!(
        host.execute_paged_title(heap, &mut backing, &[&first, &second]),
        Err(BatchError::Range(..))
    ));
    assert_eq!(&section.lock().unwrap()[..6], &initial);
    second.reset();
    second.defer_copy(VA as *mut u8, (ALIAS + 2) as *const u8, 2);
    second.record_page_input(InputView::Vertex(0, vertex));
    second.close();
    let stats = host.execute_paged_title(heap, &mut backing, &[&first, &second]).unwrap();
    assert_eq!(stats, PageStats { uploaded: 1, downloaded: 1 });
    assert_eq!(&section.lock().unwrap()[..6], &[20, 30, 20, 30, 40, 50]);
}

#[cfg(test)]
#[test]
fn paged_input_views_gather_cross_page_gpu_writes() {
    use crate::page_view::{IndexView, InputView, VertexView};
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R16_UINT;
    use victory_memory::physical::PageNumbers;
    const PAGE: usize = PAGE_BYTES as usize;
    const VA: usize = PAGE;
    let host = crate::host::host().expect("D3D12 renderer host (CI selects WARP)");
    let heap = crate::memory::heap().unwrap();
    let id = PageNumbers::default().allocate(5).unwrap().offset(4);
    let (mut backing, section) = crate::title_backing::test_capture(id, &[VA, VA + PAGE]);
    {
        let mut data = section.lock().unwrap();
        data[..4].copy_from_slice(&[1, 2, 3, 4]);
        data[PAGE - 2..].copy_from_slice(&[8, 9]);
    }
    let (out, ptr) = host.shared_buffer(8).unwrap();
    let copy = GuestCopy::new(VA + PAGE, VA + 2, 2).unwrap();
    let vertex = VertexView { address: (VA + PAGE - 2) as u64, size: 4, stride: 4 };
    let view = InputView::Vertex(0, vertex);
    let index = IndexView {
        address: vertex.address,
        size: vertex.size,
        format: DXGI_FORMAT_R16_UINT.0 as u32,
    };
    let indexed = InputView::Index(index);
    assert_eq!(view.range().unwrap(), indexed.range().unwrap());
    let [src, dst] = copy.ranges();
    let ranges = [src, dst, view.range().unwrap()];
    let stats = host.execute_paged_work(heap, &mut backing, &ranges, |list, _, batch, pins, _, _| {
        let scratch = host.default_buffer(copy.len as u64).unwrap();
        list.keep(&scratch);
        let mut dirty = BTreeSet::new();
        record_copy(list, batch, &scratch, &copy.bind(pins).unwrap(), &mut dirty)
            .map_err(BatchError::Native)?;
        let gathered = view.gather(host, list, batch)?;
        let indexed = indexed.gather(host, list, batch)?;
        list.keep(&out);
        // SAFETY: the view is in a read-compatible state and the live
        // test output promotes to COPY_DEST.
        unsafe {
            let native = vertex.native(gathered.GetGPUVirtualAddress());
            list.recording().IASetVertexBuffers(0, Some(&[native]));
            list.recording().CopyBufferRegion(&out, 0, &gathered, 0, 4);
            let native = index.native(indexed.GetGPUVirtualAddress());
            list.recording().IASetIndexBuffer(Some(&native));
            list.recording().CopyBufferRegion(&out, 4, &indexed, 0, 4);
        }
        Ok(dirty)
    }).unwrap();
    assert_eq!(stats, PageStats { uploaded: 1, downloaded: 1 });
    // SAFETY: the output owns the map and the real queue fence completed.
    assert_eq!(unsafe { std::slice::from_raw_parts(ptr, 8) }, &[8, 9, 3, 4, 8, 9, 3, 4]);
}

#[cfg(all(test, feature = "diagnostics"))]
#[test]
fn paged_linear_texture_uses_ordered_gpu_writes_and_pitched_rows() {
    use crate::page_texture::LinearTexture;
    use victory_memory::physical::PageNumbers;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM;

    const PAGE: usize = PAGE_BYTES as usize;
    const VA: usize = PAGE;
    let host = crate::host::host().expect("D3D12 renderer host (CI selects WARP)");
    let heap = crate::memory::heap().unwrap();
    let id = PageNumbers::default().allocate(7).unwrap().offset(6);
    let (mut backing, section) = crate::title_backing::test_capture(id, &[VA, VA + PAGE]);
    {
        let mut bytes = section.lock().unwrap();
        bytes[PAGE - 4..].copy_from_slice(&[1, 2, 3, 4]);
        bytes[..4].copy_from_slice(&[0, 0, 0, 0]);
        bytes[12..20].copy_from_slice(&[9, 10, 11, 12, 13, 14, 15, 16]);
        bytes[40..44].copy_from_slice(&[5, 6, 7, 8]);
    }
    let copy = GuestCopy::new(VA + PAGE, VA + 40, 4).unwrap();
    let texture = LinearTexture {
        address: (VA + PAGE - 4) as u64, row_pitch: 16, width: 2, height: 2,
        format: DXGI_FORMAT_R8G8B8A8_UNORM,
    };
    let ranges: Vec<_> = copy.ranges().into_iter().chain(texture.rows().unwrap()).collect();
    let mut output = None;
    let stats = host.execute_paged_work(heap, &mut backing, &ranges, |list, _, batch, pins, _, _| {
        let scratch = host.default_buffer(copy.len as u64).unwrap();
        list.keep(&scratch);
        let mut dirty = BTreeSet::new();
        record_copy(list, batch, &scratch, &copy.bind(pins).unwrap(), &mut dirty)
            .map_err(BatchError::Native)?;
        let texture = texture.encode(host, list, batch)?;
        let (readback, footprint) = crate::capture::read_back(&host.device, list.recording(), &texture)
            .ok_or_else(|| BatchError::Native(E_FAIL.into()))?;
        list.keep(&readback);
        output = Some((readback, footprint));
        Ok(dirty)
    }).unwrap();
    assert_eq!(stats, PageStats { uploaded: 1, downloaded: 1 });
    let (readback, footprint) = output.unwrap();
    // SAFETY: the queue fence completed; both padded rows are mapped.
    unsafe {
        let mut ptr = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut ptr)).unwrap();
        let first = ptr.cast::<u8>().add(footprint.Offset as usize);
        let second = first.add(footprint.Footprint.RowPitch as usize);
        assert_eq!(std::slice::from_raw_parts(first, 8), &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(std::slice::from_raw_parts(second, 8), &[9, 10, 11, 12, 13, 14, 15, 16]);
        readback.Unmap(0, None);
    }
}
