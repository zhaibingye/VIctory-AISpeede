//! Renderer-owned physical page storage. Native slabs are charged to the same
//! ledger as XMem. A slab is one D3D12 buffer, so one descriptor pair can bind
//! every slot. The queue must transition a slab as a unit, not per slot.
use std::collections::BTreeMap;

use victory_memory::{
    PAGE_BYTES,
    coherence::Fence,
    gpu_table::{BindError, RawPageTable},
    physical::PhysicalPage,
    residency::Residency,
};
use windows::Win32::Foundation::{E_INVALIDARG, E_OUTOFMEMORY};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::{Common::DXGI_SAMPLE_DESC, DXGI_ERROR_DEVICE_REMOVED};
use windows::core::{Error, Result};

use crate::host::Host;

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
}

impl PinnedPage {
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
        if written {
            page.dirty = Some(page.dirty.map_or(fence, |prior| prior.max(fence)));
        }
        assert!(self.residency.end_recording(pin.page, Some(fence)));
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
    use crate::host::transition;
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
    let a = slabs.pin(host, backing.expected(low).unwrap()).unwrap();
    let b = slabs.pin(host, backing.expected(low * 2).unwrap()).unwrap();
    let alias = slabs.pin(host, backing.expected(high).unwrap()).unwrap();
    assert_eq!(b.slab(), a.slab());
    assert_eq!(alias.slab(), b.slab());
    assert_eq!(alias.offset(), b.offset());
    assert_eq!(b.offset(), PAGE_BYTES);
    assert!(a.needs_upload() && b.needs_upload());
    let mut table = RawPageTable::new(2);
    assert_ne!(
        b.bind(&mut table, (low * 2) as u64, 8).unwrap(),
        alias.bind(&mut table, high as u64, 8).unwrap(),
    );
    // Later submissions see a different fresh ID at the same address.
    let mut next = BackingPins::default();
    next.remember(high, high + low, Some(remapped));
    let distinct = slabs.pin(host, next.expected(high).unwrap()).unwrap();
    assert_eq!(distinct.slab(), b.slab());
    assert_ne!(distinct.offset(), b.offset());
    slabs.cancel(distinct);
    assert!(slabs.retire(remapped, before));
    slabs.cancel(alias);
    assert!(!slabs.retire(id.offset(1), before), "recording pin");

    let (upload, up) = host.upload_buffer(PAGE_BYTES * 2).unwrap();
    let (download, down) = host.shared_buffer(PAGE_BYTES * 2).unwrap();
    let list = host.new_list().unwrap();
    // SAFETY: both mapped two-page buffers and the native slab are live;
    // this command list owns execution and no CPU uses the download before its fence.
    unsafe {
        std::ptr::write_bytes(up, 0x31, low);
        std::ptr::write_bytes(up.add(low), 0x72, low);
        list.list.ResourceBarrier(&[transition(
            a.buffer(),
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )]);
        list.list
            .CopyBufferRegion(a.buffer(), a.offset(), &upload, 0, PAGE_BYTES);
        list.list
            .CopyBufferRegion(b.buffer(), b.offset(), &upload, PAGE_BYTES, PAGE_BYTES);
        list.list.ResourceBarrier(&[transition(
            a.buffer(),
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        list.list
            .CopyBufferRegion(&download, 0, a.buffer(), a.offset(), PAGE_BYTES);
        list.list
            .CopyBufferRegion(&download, PAGE_BYTES, b.buffer(), b.offset(), PAGE_BYTES);
    }
    list.close();
    host.execute(&[&list]);
    let signal = host.completed_fence().unwrap();
    assert!(signal > before);
    slabs.submitted(a, signal, true, false);
    slabs.submitted(b, signal, true, true);
    assert!(!slabs.retire(id, before), "fence protects clean slots");
    assert!(
        !slabs.retire(id.offset(1), signal),
        "dirty slots still need writeback"
    );
    assert!(!slabs.wrote_back(id.offset(1), signal, before));
    // SAFETY: the direct queue has finished the copy into mapped shared memory.
    let content = unsafe { std::slice::from_raw_parts(down, low * 2) };
    assert!(content[..low].iter().all(|&byte| byte == 0x31));
    assert!(content[low..].iter().all(|&byte| byte == 0x72));
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
