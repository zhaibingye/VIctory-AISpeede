use std::collections::BTreeMap;
use crate::PAGE_BYTES;

/// IDs are never reused for a different allocation at the original VA.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PageId(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapError { Alignment, AlreadyMapped, Missing, IdExhausted }

#[derive(Default)]
pub struct Guest {
    virt: BTreeMap<u64, PageId>,
    aliases: BTreeMap<PageId, u64>,
    next: u64,
}

impl Guest {
    pub fn commit_page(&mut self, address: u64) -> Result<PageId, MapError> {
        self.vacant(address)?;
        let id = PageId(self.next);
        self.next = self.next.checked_add(1).ok_or(MapError::IdExhausted)?;
        self.virt.insert(address, id);
        self.aliases.insert(id, 1);
        Ok(id)
    }

    pub fn alias_page(&mut self, address: u64, id: PageId) -> Result<(), MapError> {
        self.vacant(address)?;
        *self.aliases.get_mut(&id).ok_or(MapError::Missing)? += 1;
        self.virt.insert(address, id);
        Ok(())
    }

    pub fn resolve(&self, address: u64) -> Option<(PageId, u64)> {
        let offset = address % PAGE_BYTES;
        Some((*self.virt.get(&(address - offset))?, offset))
    }

    /// Last CPU mapping: GPU references and data ownership still gate retirement.
    pub fn unmap(&mut self, address: u64) -> Result<Option<PageId>, MapError> {
        if address % PAGE_BYTES != 0 {
            return Err(MapError::Alignment);
        }
        let id = self.virt.remove(&address).ok_or(MapError::Missing)?;
        let refs = self.aliases.get_mut(&id).expect("mapped physical page");
        *refs -= 1;
        if *refs == 0 {
            self.aliases.remove(&id);
            return Ok(Some(id));
        }
        Ok(None)
    }

    fn vacant(&self, address: u64) -> Result<(), MapError> {
        if address % PAGE_BYTES != 0 {
            return Err(MapError::Alignment);
        }
        if self.virt.contains_key(&address) {
            return Err(MapError::AlreadyMapped);
        }
        Ok(())
    }
}
