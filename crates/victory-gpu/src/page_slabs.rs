//! Renderer-owned physical page storage. Native slabs are charged to the same
//! ledger as XMem. A slab is one D3D12 buffer, so one descriptor pair can bind
//! every slot. The queue must transition a slab as a unit, not per slot.
use std::collections::{BTreeMap, BTreeSet};

use victory_memory::{
    PAGE_BYTES,
    coherence::Fence,
    gpu_table::{ADDRESS_LIMIT, BindError, RawPageTable},
    physical::{BackingPins, PhysicalPage},
    residency::Residency,
};
use windows::Win32::Foundation::{E_INVALIDARG, E_OUTOFMEMORY};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::{Common::DXGI_SAMPLE_DESC, DXGI_ERROR_DEVICE_REMOVED};
use windows::core::{Error, Interface, Result};

use crate::host::{Host, HostList, transition};

const SLOTS: u8 = 16;
pub(crate) const SLAB_BYTES: u64 = PAGE_BYTES * SLOTS as u64;

struct Slab {
    resource: ID3D12Resource,
    free: u16,
}

#[derive(Clone, Copy)]
struct Location {
    slab: u64,
    slot: u8,
}

struct Page {
    loc: Location,
    initialized: bool,
    digest: Option<blake3::Hash>,
    dirty: Option<Fence>,
}

/// One recording pin. Keep it until the containing list is discarded or
/// handed to the queue. Only the allocator may assign physical identity.
#[must_use]
pub(crate) struct PinnedPage {
    page: PhysicalPage,
    loc: Location,
    buffer: ID3D12Resource,
    initialized: bool,
    digest: Option<blake3::Hash>,
}

impl PinnedPage {
    pub(crate) fn physical(&self) -> PhysicalPage {
        self.page
    }

    pub(crate) fn slab(&self) -> u64 {
        self.loc.slab
    }
    pub(crate) fn buffer(&self) -> &ID3D12Resource {
        &self.buffer
    }
    pub(crate) fn offset(&self) -> u64 {
        u64::from(self.loc.slot) * PAGE_BYTES
    }
    pub(crate) fn needs_upload(&self) -> bool {
        !self.initialized
    }

    pub(crate) fn bind(
        &self,
        table: &mut RawPageTable,
        va: u64,
        pair: u32,
    ) -> std::result::Result<usize, BindError> {
        table.bind_page(va, pair, self.offset() as u32)
    }
}

#[derive(Default)]
pub(crate) struct PageSlabs {
    slabs: BTreeMap<u64, Slab>,
    pages: BTreeMap<PhysicalPage, Page>,
    residency: Residency<PhysicalPage>,
    next: u64,
}

impl PageSlabs {
    pub(crate) fn pin(&mut self, host: &Host, page: PhysicalPage) -> Result<PinnedPage> {
        let completed = host
            .completed_fence()
            .ok_or_else(|| Error::from(DXGI_ERROR_DEVICE_REMOVED))?;
        self.pin_with(page, completed, || make_slab(host))
    }

    fn pin_with(
        &mut self,
        id: PhysicalPage,
        completed: Fence,
        make: impl FnOnce() -> Result<ID3D12Resource>,
    ) -> Result<PinnedPage> {
        if !self.pages.contains_key(&id) {
            let loc = self.allocate(completed, make)?;
            self.pages.insert(
                id,
                Page {
                    loc,
                    initialized: false,
                    digest: None,
                    dirty: None,
                },
            );
        }
        self.residency.access(id, 0);
        assert!(self.residency.pin_recording(id));
        let page = &self.pages[&id];
        Ok(PinnedPage {
            page: id,
            loc: page.loc,
            buffer: self.slabs[&page.loc.slab].resource.clone(),
            initialized: page.initialized,
            digest: page.digest,
        })
    }

    fn allocate(
        &mut self,
        completed: Fence,
        make: impl FnOnce() -> Result<ID3D12Resource>,
    ) -> Result<Location> {
        if let Some((&slab, entry)) = self.slabs.iter_mut().find(|(_, s)| s.free != 0) {
            let slot = entry.free.trailing_zeros() as u8;
            entry.free &= !(1 << slot);
            return Ok(Location { slab, slot });
        }
        let resource = match make() {
            Ok(resource) => resource,
            Err(e) if e.code() == E_OUTOFMEMORY => {
                // Leave dirty pages intact until the caller has downloaded
                // and merged them into their captured physical backing.
                let old = self
                    .residency
                    .candidate_where(completed, |id| self.pages[&id].dirty.is_none())
                    .ok_or(e)?;
                assert!(self.residency.remove(old, completed));
                return Ok(self.pages.remove(&old).expect("resident page").loc);
            }
            Err(e) => return Err(e),
        };
        let slab = self.next;
        self.next = self
            .next
            .checked_add(1)
            .ok_or_else(|| Error::from(E_OUTOFMEMORY))?;
        self.slabs.insert(
            slab,
            Slab {
                resource,
                free: u16::MAX & !1,
            },
        );
        Ok(Location { slab, slot: 0 })
    }

    /// No execution occurred. An abandoned recording changes no ownership.
    pub(crate) fn cancel(&mut self, pin: PinnedPage) {
        assert!(self.residency.end_recording(pin.page, None));
    }

    /// Called once after queue submission with its actual signal value. A
    /// freshly allocated slot may only become readable after a full upload
    /// or full overwrite in this ordered queue.
    pub(crate) fn submitted(
        &mut self,
        pin: PinnedPage,
        fence: Fence,
        initialized: bool,
        written: bool,
    ) {
        assert!(fence.0 != 0, "zero is not a submission signal");
        let page = self.pages.get_mut(&pin.page).expect("pinned page");
        assert!(
            !written || initialized || page.initialized,
            "partial write to an uninitialized slot"
        );
        page.initialized |= initialized;
        page.digest = None; // callers that know the completed bytes publish their digest later
        if written {
            page.dirty = Some(page.dirty.map_or(fence, |prior| prior.max(fence)));
        }
        assert!(self.residency.end_recording(pin.page, Some(fence)));
    }

    /// Publish the GPU's bytes (not merged CPU bytes) after this queue's
    /// actual completion. Eviction drops the digest with the physical ID.
    pub(crate) fn publish_completed_content(
        &mut self,
        digests: BTreeMap<PhysicalPage, blake3::Hash>,
    ) {
        for (id, digest) in digests {
            let page = self
                .pages
                .get_mut(&id)
                .expect("content for pinned resident page");
            debug_assert!(page.initialized && page.dirty.is_none());
            page.digest = Some(digest);
        }
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn completed_digest(&self, page: PhysicalPage) -> Option<blake3::Hash> {
        self.pages.get(&page).and_then(|entry| entry.digest)
    }

    /// Mark only the exact GPU version that was successfully merged back
    /// into RAM. A later GPU write may have superseded an older download.
    pub(crate) fn wrote_back(
        &mut self,
        id: PhysicalPage,
        written: Fence,
        completed: Fence,
    ) -> bool {
        let Some(page) = self.pages.get_mut(&id) else {
            return false;
        };
        if page.dirty != Some(written) || written > completed {
            return false;
        }
        page.dirty = None;
        true
    }

    /// Retire a clean page after its last pin and real fence, releasing the
    /// allocation when its last slot leaves. Drop descriptor references
    /// before expecting COM-resident bytes to be returned to the ledger.
    pub(crate) fn retire(&mut self, id: PhysicalPage, completed: Fence) -> bool {
        if self.pages.get(&id).is_none_or(|p| p.dirty.is_some())
            || !self.residency.remove(id, completed)
        {
            return false;
        }
        let Location { slab, slot } = self.pages.remove(&id).expect("resident page").loc;
        let entry = self.slabs.get_mut(&slab).expect("resident slab");
        entry.free |= 1 << slot;
        if entry.free == u16::MAX {
            self.slabs.remove(&slab);
        }
        true
    }
}

/// One direct-queue batch. Distinct VAs can have separate leaf/dirty bits;
/// the physical page is pinned and transferred only once.
pub(crate) struct PinnedBatch {
    pages: BTreeMap<PhysicalPage, (u64, PinnedPage)>,
    aliases: BTreeMap<u64, PhysicalPage>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ListRejection {
    NotClosed,
    NativeRecording(&'static std::panic::Location<'static>),
    UnsupportedCommand(&'static std::panic::Location<'static>),
    Shader {
        operation: &'static str,
        reason: crate::page_shader::UnloweredShader,
    },
    NativeState,
    DeferredEffects,
}

#[derive(Debug)]
pub(crate) enum BatchError {
    UnloweredList { index: usize, reason: ListRejection },
    Operation {
        phase: &'static str,
        index: usize,
        operation: &'static str,
        cause: Box<BatchError>,
    },
    TargetNotInitialized,
    UnsupportedDepth,
    UnsupportedBarrier,
    UnsupportedTextureMap,
    Range(u64, u64),
    NotCaptured(u64),
    Native(Error),
    Table(BindError),
}

#[cfg(feature = "paged-title")]
impl BatchError {
    pub(crate) fn is_out_of_memory(&self) -> bool {
        match self {
            Self::Native(error) => error.code() == windows::Win32::Foundation::E_OUTOFMEMORY,
            Self::Operation { cause, .. } => cause.is_out_of_memory(),
            _ => false,
        }
    }
}

/// Translation back from the GPU's per-VA mark bitmap. One missing leaf
/// taints the whole batch; the caller must not merge writes computed
/// from a partial/missing address space.
pub(crate) struct BoundLeaves(BTreeMap<usize, PhysicalPage>);

impl BoundLeaves {
    pub(crate) fn dirty_pages(
        &self,
        bitmap: &[u32],
    ) -> std::result::Result<BTreeSet<PhysicalPage>, usize> {
        let mut dirty = BTreeSet::new();
        for (word, &bits) in bitmap.iter().enumerate() {
            let mut bits = bits;
            while bits != 0 {
                let leaf = word * 32 + bits.trailing_zeros() as usize;
                let page = self.0.get(&leaf).copied().ok_or(leaf)?;
                dirty.insert(page);
                bits &= bits - 1;
            }
        }
        Ok(dirty)
    }
}

impl PinnedBatch {
    /// `ranges` are half-open GPU virtual addresses. Until ordinary
    /// allocations receive their own stable identity, unknown pages fail
    /// this path as a batch rather than becoming readable nulls.
    pub(crate) fn from_capture(
        host: &Host,
        slabs: &mut PageSlabs,
        backing: &BackingPins,
        ranges: &[(u64, u64)],
    ) -> std::result::Result<Self, BatchError> {
        let mut aliases = BTreeMap::new();
        for &(start, end) in ranges {
            if start > end || end > ADDRESS_LIMIT {
                return Err(BatchError::Range(start, end));
            }
            if start == end {
                continue;
            }
            let mut va = start & !(PAGE_BYTES - 1);
            while va < end {
                let (stop, physical) = backing.run_at(va as usize, end as usize);
                let physical = physical.ok_or(BatchError::NotCaptured(va))?;
                for (i, addr) in (va..stop as u64).step_by(PAGE_BYTES as usize).enumerate() {
                    aliases.insert(addr, physical.offset(i));
                }
                va = stop as u64;
            }
        }
        let mut pages = BTreeMap::new();
        for (&va, &id) in &aliases {
            use std::collections::btree_map::Entry;
            if let Entry::Vacant(slot) = pages.entry(id) {
                match slabs.pin(host, id) {
                    Ok(pin) => {
                        slot.insert((va, pin));
                    }
                    Err(e) => {
                        for (_, pin) in pages.into_values() {
                            slabs.cancel(pin);
                        }
                        return Err(BatchError::Native(e));
                    }
                }
            }
        }
        Ok(Self { pages, aliases })
    }

    /// Capture and pin in the same submission object that will own later
    /// writeback. Duplicate draw/range requests are folded by physical ID.
    pub(crate) fn capture(
        host: &Host,
        slabs: &mut PageSlabs,
        backing: &mut crate::title_backing::SubmissionBacking,
        ranges: &[(u64, u64)],
    ) -> std::result::Result<Self, BatchError> {
        Self::capture_backing(backing, ranges)?;
        Self::from_capture(host, slabs, backing.pins(), ranges)
    }

    pub(crate) fn capture_backing(
        backing: &mut crate::title_backing::SubmissionBacking, ranges: &[(u64, u64)],
    ) -> std::result::Result<(), BatchError> {
        for &(start, end) in ranges {
            if start > end || end > ADDRESS_LIMIT {
                return Err(BatchError::Range(start, end));
            }
            backing.include(start as usize, end as usize);
        }
        backing.flush();
        Ok(())
    }

    /// The first captured VA is a source handle, not a new physical ID.
    pub(crate) fn pages(&self) -> impl Iterator<Item = (PhysicalPage, u64, &PinnedPage)> {
        self.pages.iter().map(|(&id, (va, pin))| (id, *va, pin))
    }

    pub(crate) fn pinned_at(&self, va: u64) -> Option<&PinnedPage> {
        self.pinned(*self.aliases.get(&(va & !(PAGE_BYTES - 1)))?)
    }

    pub(crate) fn pinned(&self, page: PhysicalPage) -> Option<&PinnedPage> {
        self.pages.get(&page).map(|(_, pin)| pin)
    }

    pub(crate) fn bind(
        &self,
        table: &mut RawPageTable,
        mut pair: impl FnMut(&PinnedPage) -> Result<u32>,
    ) -> std::result::Result<BoundLeaves, BatchError> {
        let mut leaves = BTreeMap::new();
        let mut pairs = BTreeMap::new();
        for (&va, &physical) in &self.aliases {
            let pin = &self.pages[&physical].1;
            use std::collections::btree_map::Entry;
            let pair = match pairs.entry(pin.slab()) {
                Entry::Occupied(e) => *e.get(),
                Entry::Vacant(e) => *e.insert(pair(pin).map_err(BatchError::Native)?),
            };
            let index = pin.bind(table, va, pair).map_err(BatchError::Table)?;
            leaves.insert(index, physical);
        }
        Ok(BoundLeaves(leaves))
    }

    pub(crate) fn cancel(self, slabs: &mut PageSlabs) {
        for (_, pin) in self.pages.into_values() {
            slabs.cancel(pin);
        }
    }

    pub(crate) fn submitted(
        self,
        slabs: &mut PageSlabs,
        signal: Fence,
        mut access: impl FnMut(PhysicalPage) -> (bool, bool),
    ) {
        for (physical, (_, pin)) in self.pages {
            let (initialized, written) = access(physical);
            slabs.submitted(pin, signal, initialized, written);
        }
    }
}

/// Budgeted CPU-visible before-images. The same allocation supplies GPU
/// copies and post-fence byte comparison, avoiding a second full CPU copy.
pub(crate) struct CapturedUploads {
    staging: ID3D12Resource,
    mapped: std::ptr::NonNull<u8>,
    snapshots: std::sync::Arc<BTreeMap<PhysicalPage, PageSnapshot>>,
}

struct PageSnapshot {
    va: u64,
    offset: usize,
    digest: blake3::Hash,
}

#[cfg(feature = "paged-title")]
pub(crate) struct SnapshotDigests(std::sync::Arc<BTreeMap<PhysicalPage, PageSnapshot>>);

#[cfg(feature = "paged-title")]
impl SnapshotDigests {
    pub(crate) fn get(&self, page: PhysicalPage) -> Option<blake3::Hash> {
        self.0.get(&page).map(|page| page.digest)
    }
}

impl CapturedUploads {
    pub(crate) fn stage(
        host: &Host,
        batch: &PinnedBatch,
        backing: &crate::title_backing::SubmissionBacking,
    ) -> std::result::Result<Option<Self>, BatchError> {
        const PAGE: usize = PAGE_BYTES as usize;
        if batch.pages.is_empty() {
            return Ok(None);
        }
        let bytes = batch
            .pages
            .len()
            .checked_mul(PAGE)
            .ok_or_else(|| BatchError::Native(Error::from(E_OUTOFMEMORY)))?;
        let (staging, data) = host
            .shared_buffer(bytes as u64)
            .ok_or_else(|| BatchError::Native(Error::from(E_OUTOFMEMORY)))?;
        let mapped = std::ptr::NonNull::new(data)
            .ok_or_else(|| BatchError::Native(Error::from(E_INVALIDARG)))?;
        let mut snapshots = BTreeMap::new();
        for (i, (id, va, _)) in batch.pages().enumerate() {
            let offset = i * PAGE;
            // SAFETY: staging is still exclusively CPU-owned; one slot in
            // its committed mapping and no other reference to the bytes.
            let dest = unsafe {
                let at = mapped.as_ptr().add(offset);
                at.write_bytes(0, PAGE);
                std::slice::from_raw_parts_mut(at, PAGE)
            };
            if !backing.stage_page(va as usize, id, dest) {
                return Err(BatchError::NotCaptured(va));
            }
            snapshots.insert(id, PageSnapshot { va, offset, digest: blake3::hash(dest) });
        }
        Ok(Some(Self {
            staging,
            mapped,
            snapshots: std::sync::Arc::new(snapshots),
        }))
    }

    pub(crate) fn record(&self, list: &HostList, batch: &PinnedBatch) -> usize {
        let entries: Vec<_> = batch
            .pages()
            .filter_map(|(id, _, page)| {
                let snapshot = &self.snapshots[&id];
                (page.digest != Some(snapshot.digest)).then_some(Transfer {
                    page,
                    staging: &self.staging,
                    offset: snapshot.offset as u64,
                })
            })
            .collect();
        upload(list, &entries);
        entries.len()
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn snapshot_digests(&self) -> SnapshotDigests {
        SnapshotDigests(self.snapshots.clone())
    }

    pub(crate) fn digests(&self) -> BTreeMap<PhysicalPage, blake3::Hash> {
        self.snapshots
            .iter()
            .map(|(&id, snapshot)| (id, snapshot.digest))
            .collect()
    }

    /// Called after actual readback completion. A failed merge leaves
    /// dirty residency pinned for recovery; it must not recycle the slot.
    pub(crate) fn merge(
        &self,
        backing: &crate::title_backing::SubmissionBacking,
        page: PhysicalPage,
        gpu: &[u8],
        scratch: &mut [u8],
    ) -> bool {
        let Some(snapshot) = self.snapshots.get(&page) else {
            return false;
        };
        // SAFETY: fully initialized snapshot; our GPU lists only read it.
        let before = unsafe {
            std::slice::from_raw_parts(self.mapped.as_ptr().add(snapshot.offset), PAGE_BYTES as usize)
        };
        backing.merge_page(snapshot.va as usize, page, before, gpu, scratch)
    }
}

/// A full-page transfer. Staging offsets belong to the containing batch;
/// staging allocations need not be contiguous across pages.
pub(crate) struct Transfer<'a> {
    pub(crate) page: &'a PinnedPage,
    pub(crate) staging: &'a ID3D12Resource,
    pub(crate) offset: u64,
}

pub(crate) fn upload(list: &HostList, pages: &[Transfer<'_>]) {
    transfer(list, pages, true);
}

pub(crate) fn download(list: &HostList, pages: &[Transfer<'_>]) {
    transfer(list, pages, false);
}

/// Between operations slabs are in COMMON. A renderer may transition
/// them for shader access between transfers, but must restore COMMON
/// before calling again. Staging is already in a compatible copy state.
/// The list keeps native resources; the submitter must additionally hold
/// the logical page pins and descriptor pairs through the queue signal.
fn transfer(list: &HostList, pages: &[Transfer<'_>], upload: bool) {
    if pages.is_empty() {
        return;
    }
    let state = if upload {
        D3D12_RESOURCE_STATE_COPY_DEST
    } else {
        D3D12_RESOURCE_STATE_COPY_SOURCE
    };
    let slabs: BTreeMap<_, _> = pages
        .iter()
        .map(|copy| (copy.page.buffer().as_raw() as usize, copy.page.buffer()))
        .collect();
    let to_copy: Vec<_> = slabs
        .values()
        .map(|r| transition(r, D3D12_RESOURCE_STATE_COMMON, state))
        .collect();
    let restore: Vec<_> = slabs
        .values()
        .map(|r| transition(r, state, D3D12_RESOURCE_STATE_COMMON))
        .collect();
    for buffer in slabs.values() {
        list.keep(*buffer);
    }
    let mut staging = BTreeSet::new();
    // SAFETY: borrowed resources remain live through the list; each
    // transfer references one allocated slot and an in-range staging page.
    unsafe {
        list.recording().ResourceBarrier(&to_copy);
        for copy in pages {
            if staging.insert(copy.staging.as_raw() as usize) {
                list.keep(copy.staging);
            }
            if upload {
                list.recording().CopyBufferRegion(
                    copy.page.buffer(),
                    copy.page.offset(),
                    copy.staging,
                    copy.offset,
                    PAGE_BYTES,
                );
            } else {
                list.recording().CopyBufferRegion(
                    copy.staging,
                    copy.offset,
                    copy.page.buffer(),
                    copy.page.offset(),
                    PAGE_BYTES,
                );
            }
        }
        list.recording().ResourceBarrier(&restore);
    }
}

fn make_slab(host: &Host) -> Result<ID3D12Resource> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: SLAB_BYTES,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    // SAFETY: one initialized buffer description on a live device.
    let bytes = unsafe { host.device.GetResourceAllocationInfo(0, &[desc]) }.SizeInBytes;
    host.managed_page_slab(bytes, |heap| {
        let mut resource = None;
        // SAFETY: initialized descriptions and writable typed output.
        unsafe {
            host.device.CreateCommittedResource(
                heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut resource,
            )?;
        }
        resource.ok_or_else(|| Error::from(E_INVALIDARG))
    })
}

#[cfg(test)]
#[test]
fn renderer_page_slab_roundtrip_uses_direct_queue_fence() {
    use victory_memory::physical::{BackingPins, PageNumbers};

    let host = crate::host::host().expect("D3D12 renderer host (CI selects WARP)");
    let mut ids = PageNumbers::default();
    let id = ids.allocate(2).unwrap();
    let remapped = ids.allocate(1).unwrap();
    let low = PAGE_BYTES as usize;
    let high = 0x91_0001_0000;
    let mut backing = BackingPins::default();
    backing.remember(low, low * 3, Some(id));
    backing.remember(high, high + low, Some(id.offset(1)));
    backing.remember(high, high + low, Some(remapped)); // first identity wins in this submission

    let mut slabs = PageSlabs::default();
    let before = host.completed_fence().unwrap();
    let batch = PinnedBatch::from_capture(
        host,
        &mut slabs,
        &backing,
        &[
            (low as u64, (low * 3) as u64),
            (high as u64 + 4, (high + low) as u64),
        ],
    )
    .unwrap();
    let pages: Vec<_> = batch.pages().collect();
    let [(first, first_va, a), (second, second_va, b)] = pages.as_slice() else {
        panic!("dedup by physical page")
    };
    assert_eq!((*first, *first_va), (id, low as u64));
    assert_eq!((*second, *second_va), (id.offset(1), (low * 2) as u64));
    assert_eq!(b.slab(), a.slab());
    assert_eq!(b.offset(), PAGE_BYTES);
    assert!(a.needs_upload() && b.needs_upload());
    let heap = crate::memory::heap().unwrap();
    let null = heap.pair().unwrap();
    heap.raw_pair_at(null, None, 0);
    let mut table = RawPageTable::new(null);
    let mut pairs = Vec::new();
    let leaves = batch
        .bind(&mut table, |pin| {
            let pair = heap.pair().ok_or_else(|| Error::from(E_OUTOFMEMORY))?;
            heap.raw_pair_at(pair, Some(pin.buffer()), SLAB_BYTES);
            pairs.push(pair);
            Ok(pair)
        })
        .unwrap();
    assert_eq!(pairs.len(), 1, "one descriptor pair per native slab");
    let alias_leaves: Vec<_> = leaves
        .0
        .iter()
        .filter_map(|(&leaf, &p)| (p == id.offset(1)).then_some(leaf))
        .collect();
    assert_eq!(alias_leaves.len(), 2);
    let mut bitmap = vec![0; alias_leaves[1] / 32 + 1];
    for leaf in alias_leaves {
        bitmap[leaf / 32] |= 1 << (leaf % 32);
    }
    assert_eq!(
        leaves.dirty_pages(&bitmap).unwrap(),
        BTreeSet::from([id.offset(1)])
    );
    bitmap[0] |= 1;
    assert_eq!(leaves.dirty_pages(&bitmap), Err(0));
    let (words, _) = table.into_parts();
    let table_bytes = words.len() * 4;
    let (table_resource, table_ptr) = host.upload_buffer(table_bytes as u64).unwrap();
    let (out, out_ptr) = host.shared_buffer(32).unwrap();
    const SOURCE: &str = r#"
#define RS "DescriptorTable(UAV(u0, numDescriptors=unbounded, space=1)), SRV(t0, space=2), UAV(u0), RootConstants(num32BitConstants=3, b0)"
RWByteAddressBuffer slabs[] : register(u0, space1);
ByteAddressBuffer table : register(t0, space2);
RWByteAddressBuffer outbuf : register(u0);
cbuffer c : register(b0) { uint lo; uint hi; uint at; }
[RootSignature(RS)]
[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint va = lo + id.x * 4;
    uint block = table.Load(((hi << 4) | (va >> 28)) * 4);
    uint leaf = (block << 12) | ((va >> 16) & 4095);
    uint2 binding = table.Load2((4096 + 2 * leaf) * 4);
    uint view = NonUniformResourceIndex(binding.x + 1);
    uint offset = va - binding.y;
    outbuf.Store(at + id.x * 4, slabs[view].Load(offset));
    if (at == 0) slabs[view].Store(offset, 0x5a5a5a5a);
}
"#;
    let (root, pso) = crate::memory::tests::pipeline(SOURCE);
    // SAFETY: the table is a mapped D3D12 upload buffer of the copied size.
    unsafe { std::ptr::copy_nonoverlapping(words.as_ptr().cast::<u8>(), table_ptr, table_bytes) };
    // Later submissions see a different fresh ID at the same address.
    let mut next = BackingPins::default();
    next.remember(high, high + low, Some(remapped));
    let distinct = slabs.pin(host, next.expected(high).unwrap()).unwrap();
    assert_eq!(distinct.slab(), b.slab());
    assert_ne!(distinct.offset(), b.offset());
    slabs.cancel(distinct);
    assert!(slabs.retire(remapped, before));
    assert!(!slabs.retire(id.offset(1), before), "recording pin");

    let (upload, up) = host.upload_buffer(PAGE_BYTES * 2).unwrap();
    let (download, down) = host.shared_buffer(PAGE_BYTES * 2).unwrap();
    let list = host.new_list().unwrap();
    // SAFETY: the two-page upload allocation is mapped for CPU writes.
    unsafe {
        std::ptr::write_bytes(up, 0x31, low);
        std::ptr::write_bytes(up.add(low), 0x72, low);
    }
    self::upload(
        &list,
        &[
            Transfer {
                page: a,
                staging: &upload,
                offset: 0,
            },
            Transfer {
                page: b,
                staging: &upload,
                offset: PAGE_BYTES,
            },
        ],
    );
    // SAFETY: all descriptors, pipeline, table, and output stay valid
    // until the direct-queue fence; both draw dispatches use UAV state.
    unsafe {
        let cmd = list.recording();
        cmd.ResourceBarrier(&[transition(
            a.buffer(),
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )]);
        cmd.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
        cmd.SetComputeRootSignature(&root);
        cmd.SetComputeRootDescriptorTable(0, heap.gpu(0));
        cmd.SetComputeRootShaderResourceView(1, table_resource.GetGPUVirtualAddress());
        cmd.SetComputeRootUnorderedAccessView(2, out.GetGPUVirtualAddress());
        cmd.SetPipelineState(&pso);
        for (i, va) in [(low * 2) as u64, high as u64].into_iter().enumerate() {
            let args = [va as u32, (va >> 32) as u32, (i * 16) as u32];
            cmd.SetComputeRoot32BitConstants(3, 3, args.as_ptr().cast(), 0);
            cmd.Dispatch(1, 1, 1);
            if i == 0 {
                cmd.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                            // Borrow the resource without creating a COM ownership edge.
                            pResource: std::mem::transmute_copy(a.buffer()),
                        }),
                    },
                }]);
            }
        }
        cmd.ResourceBarrier(&[transition(
            a.buffer(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        )]);
    }
    self::download(
        &list,
        &[
            Transfer {
                page: a,
                staging: &download,
                offset: 0,
            },
            Transfer {
                page: b,
                staging: &download,
                offset: PAGE_BYTES,
            },
        ],
    );
    list.close();
    host.execute(&[&list]);
    // Descriptor pairs stay immutable through the same native queue fence.
    for pair in pairs.into_iter().chain([null]) {
        heap.raw_pair_at(pair, None, 0);
        heap.free_pair(pair);
    }
    // SAFETY: the direct queue has completed both writes to mapped output.
    let result = unsafe { std::slice::from_raw_parts(out_ptr, 32) };
    assert_eq!(&result[..16], &[0x72; 16], "first alias read before write");
    assert_eq!(
        &result[16..],
        &[0x5a; 16],
        "second alias observed the same UAV slot"
    );
    let signal = host.completed_fence().unwrap();
    assert!(signal > before);
    batch.submitted(&mut slabs, signal, |page| (true, page == id.offset(1)));
    assert!(!slabs.retire(id, before), "fence protects clean slots");
    assert!(
        !slabs.retire(id.offset(1), signal),
        "dirty slots still need writeback"
    );
    assert!(!slabs.wrote_back(id.offset(1), signal, before));
    // SAFETY: the direct queue has finished the copy into mapped shared memory.
    let content = unsafe { std::slice::from_raw_parts(down, low * 2) };
    assert!(content[..low].iter().all(|&byte| byte == 0x31));
    assert_eq!(&content[low..low + 16], &[0x5a; 16]);
    assert!(content[low + 16..].iter().all(|&byte| byte == 0x72));
    // Exhaust the slab without another allocation. Before the actual
    // fence arrives every slot is recording-pinned, dirty or in flight.
    let others = ids.allocate(SLOTS as usize - 2).unwrap();
    let pins: Vec<_> = (0..SLOTS as usize - 2)
        .map(|i| slabs.pin(host, others.offset(i)).unwrap())
        .collect();
    let replacement = ids.allocate(1).unwrap();
    let no_space = || Err(E_OUTOFMEMORY.into());
    assert!(slabs.pin_with(replacement, before, no_space).is_err());
    let replaced = slabs.pin_with(replacement, signal, no_space).unwrap();
    assert_eq!(
        replaced.offset(),
        0,
        "only the clean completed slot can be reused"
    );
    assert!(
        replaced.needs_upload(),
        "new physical page may not see stale bytes"
    );
    assert!(
        !slabs.retire(id, signal),
        "old physical identity was evicted"
    );
    slabs.cancel(replaced);
    assert!(slabs.retire(replacement, signal));
    for (i, pin) in pins.into_iter().enumerate() {
        slabs.cancel(pin);
        assert!(slabs.retire(others.offset(i), signal));
    }
    assert!(slabs.wrote_back(id.offset(1), signal, signal));

    let again = slabs.pin(host, id.offset(1)).unwrap();
    assert!(!again.needs_upload());
    slabs.cancel(again);
    assert!(slabs.retire(id.offset(1), signal));
    assert!(slabs.slabs.is_empty());
}
