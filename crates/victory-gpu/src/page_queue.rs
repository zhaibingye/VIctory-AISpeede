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
use crate::page_slabs::{BatchError, BoundLeaves, CapturedUploads, PageSlabs, PinnedBatch, Transfer, download};
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
    CopyTexture(std::sync::Arc<crate::page_resource::TextureCopy>),
    ReadbackTexture(std::sync::Arc<crate::page_resource::TextureReadback>),
    UploadTexture(std::sync::Arc<crate::page_resource::TextureUpload>),
    Input(InputView),
    Draw(std::sync::Arc<crate::page_draw::ClosedDraw>),
    Dispatch(std::sync::Arc<crate::page_dispatch::ClosedDispatch>),
    Clear(std::sync::Arc<crate::page_draw::ColorClear>),
    ClearDepth(std::sync::Arc<crate::page_draw::DepthClear>),
    Barrier(std::sync::Arc<crate::page_resource::PageBarrier>),
    ClearRaw(std::sync::Arc<crate::page_uav::RawClear>),
    ClearUavTexture(std::sync::Arc<crate::page_uav::TextureClear>),
}

#[cfg(feature = "paged-title")]
fn physical_footprint(
    pins: &BackingPins, ranges: &[(u64, u64)],
) -> std::result::Result<BTreeSet<PhysicalPage>, BatchError> {
    let mut physical = BTreeSet::new();
    for &(start, end) in ranges {
        let mut page = start & !(PAGE_BYTES - 1);
        while page < end {
            physical.insert(pins.expected(page as usize).ok_or(BatchError::NotCaptured(page))?);
            page += PAGE_BYTES;
        }
    }
    Ok(physical)
}

#[cfg(feature = "paged-title")]
impl PageOp {
    fn preserved_colors<'a>(&'a self, out: &mut Vec<&'a std::sync::Arc<crate::resource::HostTexture>>) {
        let target = match self {
            Self::CopyTexture(copy) => copy.preserves_color(),
            Self::UploadTexture(upload) => upload.preserves_color(),
            Self::Draw(draw) => return draw.preserved_colors(out),
            Self::Dispatch(dispatch) => return dispatch.written_targets(out),
            Self::Clear(clear) => clear.preserves_color(),
            Self::ClearUavTexture(clear) => clear.preserves_color(),
            _ => None,
        };
        out.extend(target);
    }

    fn writes_native<'a>(&'a self, out: &mut Vec<&'a std::sync::Arc<crate::resource::HostTexture>>) {
        match self {
            Self::CopyTexture(copy) => out.push(copy.target()),
            Self::UploadTexture(upload) => out.push(upload.target()),
            Self::Draw(draw) => draw.written_targets(out),
            Self::Dispatch(dispatch) => dispatch.written_targets(out),
            Self::Clear(clear) => clear.written_targets(out),
            Self::ClearDepth(clear) => clear.written_targets(out),
            Self::ClearUavTexture(clear) => out.push(clear.target()),
            _ => {},
        }
    }
}

/// Snapshot of a complete title submission, independent of guest command
/// list lifetime and reset. Guest backing pins are captured later, when
/// this work is actually eligible to execute.
pub(crate) struct ClosedTitleBatch(Vec<PageOp>);

const _: fn() = || {
    fn require_send<T: Send>() {}
    require_send::<ClosedTitleBatch>();
};

impl ClosedTitleBatch {
    pub(crate) fn capture(lists: &[&HostList]) -> std::result::Result<Self, BatchError> {
        let mut ops = Vec::new();
        for (index, list) in lists.iter().enumerate() {
            if !list.append_closed_page_ops(&mut ops) {
                return Err(BatchError::UnloweredList(index));
            }
        }
        Ok(Self(ops))
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn native_window_candidate(&self) -> bool {
        // A cheap scheduler hint. The footprint is resolved only under
        // the direct queue lock, which also orders pointer publication.
        self.0.iter().all(|op| matches!(op,
            PageOp::Clear(_) | PageOp::ClearDepth(_) | PageOp::ClearUavTexture(_)
            | PageOp::Barrier(_) | PageOp::CopyTexture(_) | PageOp::Draw(_)
            | PageOp::Dispatch(_)))
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn execute_adjacent(
        host: &Host, heap: &HostHeap, batches: Vec<Self>,
    ) -> std::result::Result<(), BatchError> {
        let mut signal = host.direct_queue();
        if batches.len() > 1 {
            return host.execute_native_window(&mut signal, heap, batches);
        }
        if let Some(batch) = batches.into_iter().next() {
            let mut backing = SubmissionBacking::default();
            host.execute_paged_ops_locked(&mut signal, heap, &mut backing, &batch.0)?;
        }
        Ok(())
    }

    pub(crate) fn execute(
        &self, host: &Host, heap: &HostHeap, backing: &mut SubmissionBacking,
    ) -> std::result::Result<PageStats, BatchError> {
        host.execute_paged_ops(heap, backing, &self.0)
    }
}

#[cfg(feature = "paged-title")]
type VisibleTexture = (usize, crate::page_resource::TextureReadback, std::ops::Range<usize>);

/// One byte-visibility plan for native recording and scheduler admission.
/// In addition to explicit shader/copy ranges it includes preserving old
/// CPU bytes on partial writes and all automatic CPU-visible readbacks.
struct ClosedFootprint {
    ranges: Vec<(u64, u64)>,
    max_copy: u64,
    #[cfg(feature = "paged-title")]
    ordinary_count: usize,
    #[cfg(feature = "paged-title")]
    preserved: BTreeMap<usize, std::ops::Range<usize>>,
    #[cfg(feature = "paged-title")]
    visible: Vec<VisibleTexture>,
}

impl ClosedFootprint {
    #[cfg(feature = "paged-title")]
    fn is_page_free(&self) -> bool {
        self.ranges.is_empty() && self.preserved.is_empty() && self.visible.is_empty()
    }

    fn capture(ops: &[PageOp]) -> std::result::Result<Self, BatchError> {
        let mut ranges = Vec::new();
        let mut max_copy = 0;
        for op in ops {
            match op {
                PageOp::Copy(copy) => {
                    ranges.extend(copy.ranges());
                    max_copy = max_copy.max(copy.len as u64);
                }
                PageOp::Input(input) => ranges.push(input.range()?),
                PageOp::ReadbackTexture(readback) => readback.ranges(&mut ranges)?,
                PageOp::UploadTexture(upload) => upload.ranges(&mut ranges)?,
                PageOp::Draw(draw) => draw.ranges(&mut ranges)?,
                PageOp::ClearRaw(clear) => ranges.push(clear.range()?),
                PageOp::Dispatch(dispatch) => dispatch.ranges(&mut ranges)?,
                PageOp::CopyTexture(copy) => copy.ranges(&mut ranges)?,
                PageOp::Clear(_) | PageOp::ClearDepth(_) | PageOp::ClearUavTexture(_) | PageOp::Barrier(_) => {},
            }
        }
        #[cfg(feature = "paged-title")]
        let (ordinary_count, preserved, visible) = {
            let ordinary_count = ranges.len();
            let mut preserved = BTreeMap::new();
            let mut preserve_targets = Vec::new();
            for op in ops {
                op.preserved_colors(&mut preserve_targets);
            }
            for target in preserve_targets {
                if let Some(layout) = target.compatible_texture_source() {
                    let key = std::sync::Arc::as_ptr(target) as usize;
                    if let std::collections::btree_map::Entry::Vacant(entry) = preserved.entry(key) {
                        let start = ranges.len();
                        ranges.extend(layout.rows()?);
                        entry.insert(start..ranges.len());
                    }
                }
            }
            let mut targets = Vec::new();
            for op in ops {
                op.writes_native(&mut targets);
            }
            let mut seen = BTreeSet::new();
            let mut visible = Vec::new();
            for texture in targets {
                if seen.insert(std::sync::Arc::as_ptr(texture))
                    && let Some(layout) = texture.visible_linear_writeback()?
                {
                    let readback = crate::page_resource::TextureReadback::capture(texture.clone(), layout)
                        .ok_or(BatchError::UnsupportedTextureMap)?;
                    let start = ranges.len();
                    readback.ranges(&mut ranges)?;
                    visible.push((std::sync::Arc::as_ptr(texture) as usize, readback, start..ranges.len()));
                }
            }
            (ordinary_count, preserved, visible)
        };
        Ok(Self {
            ranges, max_copy,
            #[cfg(feature = "paged-title")]
            ordinary_count,
            #[cfg(feature = "paged-title")]
            preserved,
            #[cfg(feature = "paged-title")]
            visible,
        })
    }
}

#[derive(Clone, Copy)]
enum WriteSet {
    /// The closed title recorder proves every GPU-written physical page.
    Declared,
    /// Low-level page shader work learns its writes from the GPU bitmap.
    ShaderBitmap,
}

struct PageDownload {
    buffer: ID3D12Resource,
    bytes: NonNull<u8>,
}

impl PageDownload {
    fn record(host: &Host, list: &HostList, batch: &PinnedBatch, dirty: &BTreeSet<PhysicalPage>) -> Result<Self> {
        let (buffer, ptr) = host.shared_buffer(dirty.len() as u64 * PAGE_BYTES).ok_or_else(oom)?;
        let bytes = NonNull::new(ptr).ok_or_else(oom)?;
        let entries: Vec<_> = batch.pages().filter(|(id, _, _)| dirty.contains(id))
            .enumerate().map(|(i, (_, _, page))| Transfer {
                page, staging: &buffer, offset: i as u64 * PAGE_BYTES,
            }).collect();
        if entries.len() != dirty.len() {
            return Err(E_FAIL.into());
        }
        download(list, &entries);
        Ok(Self { buffer, bytes })
    }

    fn bytes(&self, i: usize) -> &[u8] {
        // SAFETY: this is used only after the corresponding native fence
        // completed. Allocation and entry count match the captured set.
        let _keep_alive = &self.buffer;
        unsafe { std::slice::from_raw_parts(self.bytes.as_ptr().add(i * PAGE_BYTES as usize), PAGE_BYTES as usize) }
    }
}

struct Completion {
    fence: Fence,
    stats: PageStats,
    digests: BTreeMap<PhysicalPage, blake3::Hash>,
}

/// Holds physical residency and retirement work for one native submission.
#[must_use]
struct InFlightPages<'a> {
    batch: PinnedBatch,
    render: SubmittedRender<'a>,
}

impl InFlightPages<'_> {
    fn retire(
        self, host: &Host, signal: &mut u64, backing: &SubmissionBacking, slabs: &mut PageSlabs,
    ) -> PageStats {
        let completed = self.render.retire(host, signal, backing, &self.batch);
        self.batch.submitted(slabs, completed.fence, |_| (true, false));
        slabs.publish_completed_content(completed.digests);
        completed.stats
    }
}

/// GPU work, descriptors and byte publication retained to the native fence.
#[must_use]
struct SubmittedRender<'a> {
    fence: Fence,
    list: HostList,
    descriptors: PageDescriptors<'a>,
    targets: crate::page_resource::ResourcePlan,
    tracking: Option<BitmapTracking>,
    declared_dirty: BTreeSet<PhysicalPage>,
    planned: Option<PageDownload>,
    uploads: Option<CapturedUploads>,
    uploaded: usize,
    reusable: bool,
}

struct BitmapTracking {
    leaves: BoundLeaves,
    mapped: NonNull<u8>,
    words: usize,
}

/// One heap's reserved descriptor slots. They are released only after the
/// direct-queue fence, or when a recording was never executed.
pub(crate) struct PageDescriptors<'a> {
    pub(crate) heap: &'a HostHeap,
    slots: Vec<u32>,
    textures: Vec<(u32, windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT)>,
    raw_views: Vec<(u32, bool)>,
    structured_views: Vec<(u32, u32)>,
    texture_uavs: Vec<(u32, windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT)>,
}

pub(crate) enum TableDescriptor {
    Texture(ID3D12Resource, windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT),
    TextureUav(ID3D12Resource, windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT),
    Raw(ID3D12Resource, u64, u64, bool),
    Structured(ID3D12Resource, u64, u32, u32),
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
        self.raw_views.push((slot, true));
        Ok(slot)
    }

    pub(crate) fn reserve_table_run(
        &mut self,
        views: &[Option<TableDescriptor>],
    ) -> Result<u32> {
        let n = u32::try_from(views.len()).map_err(|_| oom())?;
        let first = self.heap.dynamic(n).ok_or_else(oom)?;
        for (offset, view) in views.iter().enumerate() {
            use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM;
            let slot = first + offset as u32;
            match view {
                Some(TableDescriptor::TextureUav(texture, format)) => {
                    self.heap.paged_texture_uav_at(slot, Some(texture), *format);
                    self.texture_uavs.push((slot, *format));
                }
                Some(TableDescriptor::Structured(buffer, at, elements, stride)) => {
                    self.heap.paged_structured_at(slot, Some(buffer), *at, *elements, *stride);
                    self.structured_views.push((slot, *stride));
                }
                Some(TableDescriptor::Raw(buffer, at, len, unordered)) => {
                    self.heap.raw_view_at(slot, Some(buffer), *at, *len, *unordered);
                    self.raw_views.push((slot, *unordered));
                }
                _ => {
                    let (resource, format) = match view {
                        Some(TableDescriptor::Texture(tex, format)) => (Some(tex), *format),
                        _ => (None, DXGI_FORMAT_R8G8B8A8_UNORM),
                    };
                    self.heap.paged_texture_at(slot, resource, format);
                    self.textures.push((slot, format));
                }
            }
        }
        Ok(first)
    }

    pub(crate) fn reserve_texture_uav(
        &mut self, texture: &ID3D12Resource, format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    ) -> Result<u32> {
        let slot = self.heap.dynamic(1).ok_or_else(oom)?;
        self.heap.paged_texture_uav_at(slot, Some(texture), format);
        self.texture_uavs.push((slot, format));
        Ok(slot)
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
        for (slot, format) in self.texture_uavs.drain(..) {
            self.heap.paged_texture_uav_at(slot, None, format);
            self.heap.release(slot, 1);
        }
        for (slot, stride) in self.structured_views.drain(..) {
            self.heap.paged_structured_at(slot, None, 0, 1, stride);
            self.heap.release(slot, 1);
        }
        for (slot, unordered) in self.raw_views.drain(..) {
            self.heap.raw_view_at(slot, None, 0, 0, unordered);
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
        self.execute_paged_work(heap, backing, ranges, WriteSet::ShaderBitmap, |list, roots, _, _, _, _| {
            record(list, roots.expect("bitmap mode provides shader roots")).map_err(BatchError::Native)?;
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
        ClosedTitleBatch::capture(lists)?.execute(self, heap, backing)
    }

    /// Materialize a completed native texture at an actual CPU Map
    /// demand, using the same physical-page writeback as title work.
    #[cfg(feature = "paged-title")]
    pub(crate) fn materialize_texture_bytes_locked(
        &self, signal: &mut u64, heap: &HostHeap,
        source: std::sync::Arc<crate::resource::HostTexture>,
        target: crate::page_texture::LinearTexture,
    ) -> std::result::Result<(), BatchError> {
        let ranges: Vec<_> = target.rows()?.collect();
        if source.reusable_byte_mirror(&ranges) {
            return Ok(());
        }
        let readback = crate::page_resource::TextureReadback::capture(source, target)
            .ok_or(BatchError::UnsupportedTextureMap)?;
        let mut backing = SubmissionBacking::default();
        let mut publication = None;
        self.execute_paged_work_locked(signal, heap, &mut backing, &ranges, WriteSet::Declared, |list, _, batch, pins, _, plan| {
            publication = Some(readback.prepare_mirror(&ranges, pins)?);
            let mut dirty = BTreeSet::new();
            readback.encode(self, list, batch, plan, &mut dirty)?;
            Ok(dirty)
        })?;
        readback.publish_mirror(publication.expect("readback reserved its publication"));
        Ok(())
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
        let mut signal = self.direct_queue();
        self.execute_paged_ops_locked(&mut signal, heap, backing, ops)
    }

    fn execute_paged_ops_locked(
        &self,
        signal: &mut u64,
        heap: &HostHeap,
        backing: &mut SubmissionBacking,
        ops: &[PageOp],
    ) -> std::result::Result<PageStats, BatchError> {
        if ops.is_empty() {
            return Ok(PageStats::default());
        }
        let footprint = ClosedFootprint::capture(ops)?;
        self.execute_paged_plan_locked(signal, heap, backing, ops, footprint)
    }

    fn execute_paged_plan_locked(
        &self,
        signal: &mut u64,
        heap: &HostHeap,
        backing: &mut SubmissionBacking,
        ops: &[PageOp],
        footprint: ClosedFootprint,
    ) -> std::result::Result<PageStats, BatchError> {
        let ClosedFootprint {
            ranges, max_copy,
            #[cfg(feature = "paged-title")]
            ordinary_count,
            #[cfg(feature = "paged-title")]
            preserved,
            #[cfg(feature = "paged-title")]
            visible,
        } = footprint;
        #[cfg(feature = "paged-title")]
        let mut publications = Vec::with_capacity(visible.len());
        let stats = self.execute_paged_work_locked(signal, heap, backing, &ranges, WriteSet::Declared, |list, _, batch, pins, descriptors, targets| {
            #[cfg(feature = "paged-title")]
            if !visible.is_empty() || !preserved.is_empty() {
                // Own preservation and own tail publication are an
                // ordered pair. Any other overlap still needs alias lowering.
                let ordinary = physical_footprint(pins, &ranges[..ordinary_count])?;
                let mut assigned = BTreeMap::<PhysicalPage, usize>::new();
                for (&resource, span) in &preserved {
                    let pages = physical_footprint(pins, &ranges[span.clone()])?;
                    for page in pages {
                        if ordinary.contains(&page) || assigned.insert(page, resource).is_some() {
                            return Err(BatchError::UnsupportedTextureMap);
                        }
                    }
                }
                for (resource, _, span) in &visible {
                    let pages = physical_footprint(pins, &ranges[span.clone()])?;
                    for page in pages {
                        if ordinary.contains(&page)
                            || assigned.insert(page, *resource).is_some_and(|old| old != *resource)
                        {
                            return Err(BatchError::UnsupportedTextureMap);
                        }
                    }
                }
            }
            let mut dirty = self.encode_closed_ops(list, batch, pins, descriptors, targets, ops, max_copy)?;
            #[cfg(feature = "paged-title")]
            for (_, readback, span) in &visible {
                publications.push(readback.prepare_mirror(&ranges[span.clone()], pins)?);
                readback.encode(self, list, batch, targets, &mut dirty)?;
            }
            Ok(dirty)
        })?;
        #[cfg(feature = "paged-title")]
        for ((_, readback, _), publication) in visible.into_iter().zip(publications) {
            readback.publish_mirror(publication);
        }
        Ok(stats)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_closed_ops(
        &self, list: &HostList, batch: &PinnedBatch, pins: &BackingPins,
        descriptors: &mut PageDescriptors<'_>, targets: &mut crate::page_resource::ResourcePlan,
        ops: &[PageOp], max_copy: u64,
    ) -> std::result::Result<BTreeSet<PhysicalPage>, BatchError> {
        let scratch = if max_copy > 0 {
            let buffer = self.default_buffer(max_copy)
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
                PageOp::CopyTexture(copy) => copy.encode(self, list, batch, targets)?,
                PageOp::ReadbackTexture(readback) => readback.encode(self, list, batch, targets, &mut dirty)?,
                PageOp::UploadTexture(upload) => upload.encode(self, list, batch, targets)?,
                PageOp::Input(input) => input.encode(self, list, batch)?,
                PageOp::Dispatch(dispatch) => dispatch.encode(self, list, batch, &mut indirect_budget, descriptors, targets, &mut dirty)?,
                PageOp::Draw(draw) => draw.encode(self, list, batch, &mut indirect_budget, descriptors, targets, &mut dirty)?,
                PageOp::Clear(clear) => clear.encode(self, list, batch, targets)?,
                PageOp::ClearDepth(clear) => clear.encode(self, list, targets)?,
                PageOp::Barrier(barrier) => barrier.encode(list, targets)?,
                PageOp::ClearRaw(clear) => clear.encode(self, list, batch, descriptors, targets, &mut dirty)?,
                PageOp::ClearUavTexture(clear) => clear.encode(self, list, batch, descriptors, targets)?,
            }
        }
        Ok(dirty)
    }

    #[cfg(feature = "paged-title")]
    fn execute_native_window(
        &self, signal: &mut u64, heap: &HostHeap, batches: Vec<ClosedTitleBatch>,
    ) -> std::result::Result<(), BatchError> {
        use crate::page_resource::{ResourcePlan, SubmittedTail};

        let mut tail = SubmittedTail::default();
        let mut empty_backing = SubmissionBacking::default();
        let mut in_flight = Vec::with_capacity(batches.len());
        let enqueue = (|| {
            for ops in batches.into_iter().map(|batch| batch.0).filter(|ops| !ops.is_empty()) {
                let mut footprint = ClosedFootprint::capture(&ops)?;
                if !footprint.is_page_free() {
                    // Any guest bytes are an execution boundary. Publish
                    // earlier work before reading current authority or
                    // taking a new CPU snapshot of those physical pages.
                    if !in_flight.is_empty() {
                        self.drain_native_window(signal, &empty_backing, &mut in_flight);
                        tail = SubmittedTail::default();
                        footprint = ClosedFootprint::capture(&ops)?;
                    }
                    let mut backing = SubmissionBacking::default();
                    self.execute_paged_plan_locked(signal, heap, &mut backing, &ops, footprint)?;
                    continue;
                }
                let mut slabs = self.paged.lock().unwrap_or_else(|e| e.into_inner());
                let batch = PinnedBatch::capture(self, &mut slabs, &mut empty_backing, &[])?;
                let render = submit_work(
                    self, signal, heap, &empty_backing, &batch, WriteSet::Declared, ResourcePlan::after(&tail),
                    |list, _, batch, pins, descriptors, targets| {
                        let dirty = self.encode_closed_ops(list, batch, pins, descriptors, targets, &ops, 0)?;
                        if !dirty.is_empty() {
                            return Err(BatchError::UnsupportedTextureMap);
                        }
                        Ok(dirty)
                    },
                );
                match render {
                    Ok(render) => {
                        render.targets.advance_submitted_tail(&mut tail);
                        in_flight.push(InFlightPages { batch, render });
                    }
                    Err(error) => {
                        batch.cancel(&mut slabs);
                        return Err(error);
                    }
                }
            }
            Ok(())
        })();
        // Always retire submitted GPU work, even on a later recording
        // error. The logical fence and present boundaries come afterward.
        self.drain_native_window(signal, &empty_backing, &mut in_flight);
        enqueue
    }

    #[cfg(feature = "paged-title")]
    fn drain_native_window(
        &self, signal: &mut u64, backing: &SubmissionBacking, in_flight: &mut Vec<InFlightPages<'_>>,
    ) {
        if !in_flight.is_empty() {
            let mut slabs = self.paged.lock().unwrap_or_else(|e| e.into_inner());
            for work in in_flight.drain(..) {
                work.retire(self, signal, backing, &mut slabs);
            }
        }
    }

    fn execute_paged_work(
        &self,
        heap: &HostHeap,
        backing: &mut SubmissionBacking,
        ranges: &[(u64, u64)],
        writes: WriteSet,
        record: impl FnOnce(
            &HostList,
            Option<PageRoots>,
            &PinnedBatch,
            &BackingPins,
            &mut PageDescriptors<'_>,
            &mut crate::page_resource::ResourcePlan,
        ) -> std::result::Result<BTreeSet<PhysicalPage>, BatchError>,
    ) -> std::result::Result<PageStats, BatchError> {
        let mut signal = self.direct_queue();
        self.execute_paged_work_locked(&mut signal, heap, backing, ranges, writes, record)
    }

    fn execute_paged_work_locked(
        &self,
        signal: &mut u64,
        heap: &HostHeap,
        backing: &mut SubmissionBacking,
        ranges: &[(u64, u64)],
        writes: WriteSet,
        record: impl FnOnce(
            &HostList,
            Option<PageRoots>,
            &PinnedBatch,
            &BackingPins,
            &mut PageDescriptors<'_>,
            &mut crate::page_resource::ResourcePlan,
        ) -> std::result::Result<BTreeSet<PhysicalPage>, BatchError>,
    ) -> std::result::Result<PageStats, BatchError> {
        let mut slabs = self.paged.lock().unwrap_or_else(|e| e.into_inner());
        let batch = PinnedBatch::capture(self, &mut slabs, backing, ranges)?;
        let render = match submit_work(
            self, signal, heap, backing, &batch, writes, crate::page_resource::ResourcePlan::default(), record,
        ) {
            Ok(render) => render,
            Err(e) => {
                batch.cancel(&mut slabs);
                return Err(e);
            }
        };
        // Keep retirement ordered here until the scheduler has separate
        // submitted resource state and byte-publication dependency gates.
        Ok(InFlightPages { batch, render }.retire(self, signal, backing, &mut slabs))
    }
}

fn submit_work<'a>(
    host: &Host,
    signal: &mut u64,
    heap: &'a HostHeap,
    backing: &SubmissionBacking,
    batch: &PinnedBatch,
    writes: WriteSet,
    mut targets: crate::page_resource::ResourcePlan,
    record: impl FnOnce(
        &HostList,
        Option<PageRoots>,
        &PinnedBatch,
        &BackingPins,
        &mut PageDescriptors<'_>,
        &mut crate::page_resource::ResourcePlan,
    ) -> std::result::Result<BTreeSet<PhysicalPage>, BatchError>,
) -> std::result::Result<SubmittedRender<'a>, BatchError> {
    let native = BatchError::Native;
    let mut pairs = PageDescriptors {
        heap,
        slots: Vec::new(),
        textures: Vec::new(),
        raw_views: Vec::new(),
        structured_views: Vec::new(),
        texture_uavs: Vec::new(),
    };
    let reusable = matches!(writes, WriteSet::Declared);
    let list = if reusable { host.acquire_paged_list() } else { host.new_list() }.map_err(native)?;
    // Only the low-level page shader API uses a shader-visible virtual
    // page table. Closed title ops bind physical resources directly and
    // provide exact or conservative write sets, including translated UAVs.
    let (roots, tracking) = if matches!(writes, WriteSet::ShaderBitmap) {
        let mut raw = RawPageTable::new(pairs.reserve(None, 0).map_err(native)?);
        let leaves = batch.bind(&mut raw, |pin| {
            pairs.reserve(Some(pin.buffer()), crate::page_slabs::SLAB_BYTES)
        })?;
        let (words, blocks) = raw.into_parts();
        let table_bytes = words.len() * 4;
        let bitmap_words = blocks.len() * victory_memory::gpu_table::BLOCK / 32;
        let (table, table_ptr) = host.upload_buffer(table_bytes as u64).ok_or_else(|| native(oom()))?;
        let (bitmap, bitmap_ptr) = host.shared_buffer((bitmap_words * 4) as u64).ok_or_else(|| native(oom()))?;
        let table_ptr = NonNull::new(table_ptr).ok_or_else(|| native(oom()))?;
        let bitmap_ptr = NonNull::new(bitmap_ptr).ok_or_else(|| native(oom()))?;
        list.keep(&table);
        list.keep(&bitmap);
        // SAFETY: the resources are mapped for their allocated sizes
        // and have not been submitted to the GPU.
        let roots = unsafe {
            std::ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), table_ptr.as_ptr(), table_bytes);
            bitmap_ptr.as_ptr().write_bytes(0, bitmap_words * 4);
            PageRoots {
                table: table.GetGPUVirtualAddress(),
                dirty: bitmap.GetGPUVirtualAddress(),
            }
        };
        (Some(roots), Some(BitmapTracking { leaves, mapped: bitmap_ptr, words: bitmap_words }))
    } else {
        (None, None)
    };
    let uploads = CapturedUploads::stage(host, batch, backing)?;
    let slabs: BTreeMap<_, _> = batch
        .pages()
        .map(|(_, _, page)| (page.slab(), page.buffer()))
        .collect();
    for buffer in slabs.values() {
        list.keep(*buffer);
    }
    list.keep(&heap.heap);
    let uploaded = uploads.as_ref().map_or(0, |stage| stage.record(&list, batch));
    barrier(&list, slabs.values().copied(), false);
    let declared_dirty = record(&list, roots, batch, backing.pins(), &mut pairs, &mut targets)?;
    barrier(&list, slabs.values().copied(), true);
    // Closed work knows every page it may write before native enqueue.
    // Append its copies here, after all dispatches and slab transitions,
    // instead of waiting on the CPU and driving a second submission.
    let planned = match writes {
        WriteSet::Declared if !declared_dirty.is_empty() => {
            Some(PageDownload::record(host, &list, batch, &declared_dirty).map_err(native)?)
        }
        _ => None,
    };
    list.close();
    let command = list.recording().cast().map_err(native)?;
    let rendered = host
        .submit_native_group(signal, &[Some(command)])
        .unwrap_or_else(|e| fail(format_args!("native paged render submission: {e}")));
    Ok(SubmittedRender {
        fence: rendered, list, descriptors: pairs, targets, tracking,
        declared_dirty, planned, uploads, uploaded, reusable,
    })
}

impl SubmittedRender<'_> {
    fn retire(
        self, host: &Host, signal: &mut u64, backing: &SubmissionBacking, batch: &PinnedBatch,
    ) -> Completion {
        let observed = host.await_native_group(self.fence)
            .unwrap_or_else(|e| fail(format_args!("native paged render fence: {e}")));
        let mut dirty = self.declared_dirty;
        if let Some(tracking) = self.tracking {
            // SAFETY: the fence has completed, and the retained list
            // exclusively owns this submission's bitmap mapping.
            let words = unsafe {
                std::slice::from_raw_parts(tracking.mapped.as_ptr().cast(), tracking.words)
            };
            dirty.extend(tracking.leaves.dirty_pages(words).unwrap_or_else(|leaf| {
                fail(format_args!("paged dispatch accessed unbound page-table leaf {leaf}"))
            }));
        }
        let mut digests = self.uploads.as_ref().map_or_else(BTreeMap::new, CapturedUploads::digests);
        let stats = PageStats { uploaded: self.uploaded, downloaded: dirty.len() };
        let mut completed = self.fence;
        if !dirty.is_empty() {
            let readback = if let Some(planned) = self.planned {
                planned
            } else {
                // Diagnostics discover their writes after rendering, so
                // only they require a second native submission.
                let (readback, fence) = (|| -> Result<_> {
                    let transfer = host.new_list()?;
                    let readback = PageDownload::record(host, &transfer, batch, &dirty)?;
                    transfer.close();
                    let fence = host.run_native_group(signal, &[Some(transfer.recording().cast()?)])?;
                    Ok((readback, fence))
                })().unwrap_or_else(|e| fail(format_args!("native paged dirty readback: {e}")));
                completed = fence;
                readback
            };
            let snapshots = self.uploads.expect("dirty leaf requires an uploaded page");
            let _scratch_charge = HostBudget::process()
                .reserve_ram(PAGE_BYTES)
                .unwrap_or_else(|_| fail(format_args!("no RAM for paged writeback scratch")));
            let mut scratch = vec![0; PAGE_BYTES as usize];
            for (i, (page, _, _)) in batch.pages().filter(|(id, _, _)| dirty.contains(id)).enumerate() {
                let bytes = readback.bytes(i);
                if !snapshots.merge(backing, page, bytes, &mut scratch) {
                    fail(format_args!("paged writeback failed for physical page {}", page.number()));
                }
                digests.insert(page, blake3::hash(bytes));
            }
        }
        // Native readiness is not guest byte visibility. Publish the
        // completed plan only after the required CPU merge succeeded.
        self.targets.commit(self.fence, observed);
        drop(self.descriptors);
        if self.reusable {
            host.recycle_paged_list(self.list, self.fence, observed);
        }
        Completion { fence: completed, stats, digests }
    }
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
    let stats = host.execute_paged_work(heap, &mut backing, &ranges, WriteSet::ShaderBitmap, |list, _, batch, pins, _, _| {
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
    let stats = host.execute_paged_work(heap, &mut backing, &ranges, WriteSet::ShaderBitmap, |list, _, batch, pins, _, _| {
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
