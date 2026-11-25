//! Joined guest, ownership, residency, and budget policy. The D3D12 adapter
//! must serialize these transitions with the corresponding physical operations.

use std::collections::BTreeMap;
use crate::{
    PAGE_BYTES,
    budget::{Budget, Charge, NoSpace, Pool},
    coherence::{Access, Fence, Need, Page},
    guest::{Guest, MapError, PageId},
    residency::Residency,
};

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Address(MapError),
    Space(NoSpace),
    UnknownPage,
    StillMapped,
    NotEvictable,
    TransferRequired(Need),
}

struct Backing {
    ram: Charge,
    local: Option<Charge>,
    state: Page,
    mapped: bool,
}

#[derive(Default)]
pub struct Engine {
    guest: Guest,
    budget: Budget,
    residency: Residency,
    pages: BTreeMap<PageId, Backing>,
}

impl Engine {
    pub fn set_local_budget(&mut self, dxgi: u64, unmanaged: u64) -> u64 {
        self.budget.observe_local(dxgi, unmanaged)
    }

    pub fn used(&self, pool: Pool) -> u64 { self.budget.used(pool) }

    #[cfg(windows)]
    pub(crate) fn charge_staging(&mut self, bytes: u64) -> Result<Charge, Error> {
        self.budget.charge(Pool::Ram, bytes).map_err(Error::Space)
    }

    pub fn commit(&mut self, address: u64) -> Result<PageId, Error> {
        let ram = self.budget.charge(Pool::Ram, PAGE_BYTES).map_err(Error::Space)?;
        match self.guest.commit_page(address) {
            Ok(id) => {
                self.pages.insert(id, Backing { ram, local: None, state: Page::default(), mapped: true });
                Ok(id)
            }
            Err(e) => { self.budget.release(ram); Err(Error::Address(e)) }
        }
    }

    pub fn alias(&mut self, address: u64, id: PageId) -> Result<(), Error> {
        self.guest.alias_page(address, id).map_err(Error::Address)
    }

    pub fn resolve(&self, address: u64) -> Option<(PageId, u64)> {
        self.guest.resolve(address)
    }

    pub fn unmap(&mut self, address: u64) -> Result<Option<PageId>, Error> {
        let last = self.guest.unmap(address).map_err(Error::Address)?;
        if let Some(id) = last {
            self.pages.get_mut(&id).expect("physical backing exists").mapped = false;
        }
        Ok(last)
    }

    /// Reserve bytes before requesting a physical local mapping. If the
    /// physical request fails, release the reservation via complete_eviction.
    pub fn admit(&mut self, id: PageId, priority: u8) -> Result<(), Error> {
        let page = self.pages.get_mut(&id).ok_or(Error::UnknownPage)?;
        if !page.mapped { return Err(Error::UnknownPage); }
        if page.local.is_none() {
            page.local = Some(self.budget.charge(Pool::Local, PAGE_BYTES).map_err(Error::Space)?);
        }
        self.residency.access(id, priority);
        Ok(())
    }

    pub fn pin(&mut self, id: PageId) -> bool { self.residency.pin_recording(id) }
    pub fn end_pin(&mut self, id: PageId, fence: Option<Fence>) -> bool {
        self.residency.end_recording(id, fence)
    }

    pub fn candidate(&self, completed: Fence) -> Option<PageId> {
        self.residency.candidate(completed)
    }

    pub fn page(&mut self, id: PageId) -> Option<&mut Page> {
        self.pages.get_mut(&id).map(|p| &mut p.state)
    }

    /// Preparation check. A dirty GPU page reports its required writeback;
    /// the mapping, charges and old bytes remain live until completion.
    pub fn eviction_need(&mut self, id: PageId, completed: Fence) -> Result<Need, Error> {
        let page = self.pages.get_mut(&id).ok_or(Error::UnknownPage)?;
        if !self.residency.evictable(id, completed) { return Err(Error::NotEvictable); }
        Ok(page.state.cpu_need(Access::Read, completed))
    }

    /// Call after the physical mapping has been removed and the unmap
    /// signal completed. The caller must prevent new recording after the
    /// readiness check and through the physical operation.
    pub fn complete_eviction(&mut self, id: PageId, completed: Fence) -> Result<(), Error> {
        let need = self.eviction_need(id, completed)?;
        if need != Need::Ready { return Err(Error::TransferRequired(need)); }
        let page = self.pages.get_mut(&id).expect("existing page");
        page.state.discard_gpu(completed).expect("checked CPU authority");
        assert!(self.residency.remove(id, completed));
        self.budget.release(page.local.take().expect("resident page is charged"));
        Ok(())
    }

    /// Last VA is already removed. The adapter also guarantees no CPU
    /// handler or recorded GPU reference still borrows the RAM backing.
    pub fn complete_retirement(&mut self, id: PageId, completed: Fence) -> Result<(), Error> {
        let page = self.pages.get_mut(&id).ok_or(Error::UnknownPage)?;
        if page.mapped { return Err(Error::StillMapped); }
        if page.local.is_some() { self.complete_eviction(id, completed)?; }
        let page = self.pages.remove(&id).expect("existing page");
        self.budget.release(page.ram);
        Ok(())
    }
}
