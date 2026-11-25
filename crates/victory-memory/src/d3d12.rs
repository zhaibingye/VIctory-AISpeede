//! Serialized D3D12 page transport for the new memory engine. This entry point
//! owns its copy queue and resources. It is not yet the game's renderer backend.

mod copy;

use std::collections::BTreeMap;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Error as DxError;

use crate::PAGE_BYTES;
use crate::budget::{Charge, NoSpace, Pool};
use crate::coherence::{Access, Invalid, Need};
use crate::engine::{Engine, Error as PolicyError};
use crate::guest::PageId;
use copy::CopyQueue;

pub const PAGE_LEN: usize = PAGE_BYTES as usize;

#[derive(Debug)]
pub enum Error {
    Policy(PolicyError),
    Coherence(Invalid),
    Device(DxError),
    SamePage,
    QueueLost,
    FenceExhausted,
}

impl From<PolicyError> for Error {
    fn from(e: PolicyError) -> Self {
        Self::Policy(e)
    }
}

impl From<Invalid> for Error {
    fn from(e: Invalid) -> Self {
        Self::Coherence(e)
    }
}

impl From<DxError> for Error {
    fn from(e: DxError) -> Self {
        Self::Device(e)
    }
}

struct Backing {
    ram: Box<[u8; PAGE_LEN]>,
    local: Option<ID3D12Resource>,
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Transfers {
    pub uploaded_pages: u64,
    pub downloaded_pages: u64,
}

pub struct Pages {
    queue: CopyQueue,
    upload: ID3D12Resource,
    readback: ID3D12Resource,
    backing: BTreeMap<PageId, Backing>,
    policy: Engine,
    _staging_charge: Charge,
    transfers: Transfers,
}

impl Pages {
    /// The caller supplies the current local budget; the engine also enforces
    /// its fixed Series S cap. The staging buffers are reused and charged to RAM.
    pub fn new(device: ID3D12Device, local_budget: u64, unmanaged: u64) -> Result<Self, Error> {
        let mut policy = Engine::default();
        policy.set_local_budget(local_budget, unmanaged);
        let staging_charge = policy.charge_staging(2 * PAGE_BYTES)?;
        let queue = CopyQueue::new(device)?;
        let upload = queue.buffer(D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ)?;
        let readback = queue.buffer(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST)?;
        Ok(Self {
            queue,
            upload,
            readback,
            backing: BTreeMap::new(),
            policy,
            _staging_charge: staging_charge,
            transfers: Transfers::default(),
        })
    }

    pub fn used(&self, pool: Pool) -> u64 {
        self.policy.used(pool)
    }

    pub fn transfers(&self) -> Transfers {
        self.transfers
    }

    /// Shrink the actual working set, not just future admissions.
    pub fn set_local_budget(&mut self, bytes: u64, unmanaged: u64) -> Result<(), Error> {
        self.queue.check()?;
        self.policy.set_local_budget(bytes, unmanaged);
        while self.used(Pool::Local) > self.policy.local_limit() {
            let candidate = self.policy.candidate(self.queue.completed())
                .ok_or(PolicyError::NotEvictable)?;
            self.evict(candidate)?;
        }
        Ok(())
    }

    pub fn commit(&mut self, address: u64) -> Result<PageId, Error> {
        self.queue.check()?;
        let id = self.policy.commit(address)?;
        // Allocate on the heap rather than placing a 64 KiB array on the stack.
        let ram = vec![0; PAGE_LEN].into_boxed_slice().try_into().expect("page-sized box");
        self.backing.insert(id, Backing { ram, local: None });
        Ok(id)
    }

    pub fn write(&mut self, id: PageId, bytes: &[u8; PAGE_LEN]) -> Result<(), Error> {
        self.queue.check()?;
        let state = self.policy.page(id).ok_or(PolicyError::UnknownPage)?;
        state.cpu_written(Access::FullOverwrite, self.queue.completed())?;
        self.backing.get_mut(&id).expect("committed page").ram.copy_from_slice(bytes);
        Ok(())
    }

    pub fn read(&mut self, id: PageId) -> Result<&[u8; PAGE_LEN], Error> {
        self.queue.check()?;
        self.sync_ram(id)?;
        Ok(&self.backing[&id].ram)
    }

    /// A real GPU-side write from a source page to a distinct destination.
    /// The destination's previous RAM contents are deliberately not uploaded.
    pub fn copy_page(&mut self, source: PageId, dest: PageId) -> Result<(), Error> {
        self.queue.check()?;
        if source == dest {
            return Err(Error::SamePage);
        }
        if !self.backing.contains_key(&source) || !self.backing.contains_key(&dest) {
            return Err(PolicyError::UnknownPage.into());
        }
        let capacity = self.policy.local_limit();
        if capacity < 2 * PAGE_BYTES {
            return Err(PolicyError::Space(NoSpace {
                pool: Pool::Local,
                shortfall: 2 * PAGE_BYTES - capacity,
            }).into());
        }

        // Protect the source from being chosen as a victim while the
        // destination acquires a slot. A currently resident destination
        // is likewise protected while bringing in the source.
        let dest_resident = self.policy.pin(dest);
        let result = (|| {
            self.admit(source)?;
            assert!(self.policy.pin(source));
            let result = self.copy_pinned(source, dest, dest_resident);
            assert!(self.policy.end_pin(source, Some(self.queue.completed())));
            result
        })();
        if dest_resident {
            assert!(self.policy.end_pin(dest, Some(self.queue.completed())));
        }
        result
    }

    fn copy_pinned(&mut self, source: PageId, dest: PageId, dest_pinned: bool) -> Result<(), Error> {
        self.admit(dest)?;
        if !dest_pinned {
            assert!(self.policy.pin(dest));
        }
        let result = self.transfer_page(source, dest);
        if !dest_pinned {
            assert!(self.policy.end_pin(dest, Some(self.queue.completed())));
        }
        result
    }

    fn transfer_page(&mut self, source: PageId, dest: PageId) -> Result<(), Error> {
        self.sync_local(source)?;
        let before = self.queue.completed();
        let fence = self.queue.copy(
            self.backing[&source].local.as_ref().expect("admitted source"),
            self.backing[&dest].local.as_ref().expect("admitted destination"),
        )?;
        self.policy.page(source).expect("committed source").gpu_submitted(Access::Read, before, fence)?;
        self.policy.page(dest).expect("committed destination").gpu_submitted(Access::FullOverwrite, before, fence)?;
        Ok(())
    }

    /// A dirty local page is written back before its allocation is released.
    pub fn evict(&mut self, id: PageId) -> Result<(), Error> {
        self.queue.check()?;
        let need = self.policy.eviction_need(id, self.queue.completed())?;
        if need == Need::Download {
            self.sync_ram(id)?;
        } else if need != Need::Ready {
            return Err(PolicyError::TransferRequired(need).into());
        }
        drop(self.backing.get_mut(&id).expect("committed page").local.take());
        self.policy.complete_eviction(id, self.queue.completed())?;
        Ok(())
    }

    fn admit(&mut self, id: PageId) -> Result<(), Error> {
        loop {
            match self.policy.admit(id, 0) {
                Ok(()) => break,
                Err(error @ PolicyError::Space(_)) => {
                    let victim = self.policy.candidate(self.queue.completed()).ok_or(error)?;
                    self.evict(victim)?;
                }
                Err(error) => return Err(error.into()),
            }
        }
        if self.backing[&id].local.is_some() {
            return Ok(());
        }
        match self.queue.buffer(D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON) {
            Ok(local) => self.backing.get_mut(&id).expect("committed page").local = Some(local),
            Err(e) => {
                self.policy.complete_eviction(id, self.queue.completed())?;
                return Err(e);
            }
        }
        Ok(())
    }

    fn sync_local(&mut self, id: PageId) -> Result<(), Error> {
        let before = self.queue.completed();
        let state = self.policy.page(id).ok_or(PolicyError::UnknownPage)?;
        match state.gpu_need(Access::Read, before) {
            Need::Ready => Ok(()),
            Need::Upload => {
                let page = &self.backing[&id];
                copy::write_upload(&self.upload, &page.ram)?;
                let fence = self.queue.copy(&self.upload, page.local.as_ref().expect("admitted page"))?;
                state.upload_submitted(before, fence)?;
                self.transfers.uploaded_pages += 1;
                Ok(())
            }
            need => Err(PolicyError::TransferRequired(need).into()),
        }
    }

    fn sync_ram(&mut self, id: PageId) -> Result<(), Error> {
        let before = self.queue.completed();
        let state = self.policy.page(id).ok_or(PolicyError::UnknownPage)?;
        match state.cpu_need(Access::Read, before) {
            Need::Ready => Ok(()),
            Need::Download => {
                let page = self.backing.get_mut(&id).expect("committed page");
                let fence = self.queue.copy(page.local.as_ref().expect("GPU-owned page"), &self.readback)?;
                // Never publish Both before the CPU mapping and byte copy succeed.
                copy::read_back(&self.readback, &mut page.ram)?;
                state.download_submitted(before, fence)?;
                self.transfers.downloaded_pages += 1;
                Ok(())
            }
            need => Err(PolicyError::TransferRequired(need).into()),
        }
    }
}
