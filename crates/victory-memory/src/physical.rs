//! Physical allocations and alias identity, independent of host address.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::{
    PAGE_BYTES,
    budget::{Budget, Charge, NoSpace, Pool},
};

/// Admission for CPU backing whose lifetime can extend beyond its page handles.
#[derive(Clone, Default)]
pub struct RamPool(Arc<Mutex<Budget>>);

impl RamPool {
    pub fn reserve(&self, bytes: u64) -> Result<RamReservation, NoSpace> {
        let charge = self
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .charge(Pool::Ram, bytes)?;
        Ok(RamReservation {
            budget: Arc::clone(&self.0),
            charge: Some(charge),
        })
    }

    pub fn used(&self) -> u64 {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .used(Pool::Ram)
    }
}

/// Retain this lease with the entire host allocation, not with individual views.
pub struct RamReservation {
    budget: Arc<Mutex<Budget>>,
    charge: Option<Charge>,
}

impl Drop for RamReservation {
    fn drop(&mut self) {
        self.budget
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .release(self.charge.take().expect("one charge per reservation"));
    }
}

/// Physical page identity in the title's ABI: a 21-bit index packed as a
/// number of 4 KiB frames. Monotonic across allocations, regardless of VA.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PhysicalPage(usize);

const FRAMES: usize = PAGE_BYTES as usize / 4096;
const FIRST: usize = 0x10_0000;
const END: usize = (1 << 21) * FRAMES;

impl PhysicalPage {
    pub fn number(self) -> usize {
        self.0
    }

    /// Caller keeps the index within the allocated extent.
    pub fn offset(self, index: usize) -> Self {
        Self(self.0 + index * FRAMES)
    }
}

pub struct PageNumbers {
    next: usize,
}

impl Default for PageNumbers {
    fn default() -> Self {
        Self { next: FIRST }
    }
}

impl PageNumbers {
    pub fn allocate(&mut self, count: usize) -> Option<PhysicalPage> {
        let span = count.checked_mul(FRAMES)?;
        let end = self.next.checked_add(span)?;
        if count == 0 || end > END {
            return None;
        }
        let first = self.next;
        self.next = end;
        Some(PhysicalPage(first))
    }
}

/// Reverse mapping for the OS-backed views. Only successful host maps are
/// recorded here; splitting a view does not create a new physical identity.
#[derive(Default)]
pub struct Aliases(HashMap<PhysicalPage, Vec<usize>>);

impl Aliases {
    pub fn addresses(&self, page: PhysicalPage) -> &[usize] {
        self.0.get(&page).map_or(&[], Vec::as_slice)
    }

    pub fn mapped(&mut self, page: PhysicalPage, address: usize) {
        self.0.entry(page).or_default().push(address);
    }

    pub fn unmapped(&mut self, page: PhysicalPage, address: usize) {
        if let std::collections::hash_map::Entry::Occupied(mut entry) = self.0.entry(page) {
            entry.get_mut().retain(|&a| a != address);
            if entry.get().is_empty() {
                entry.remove();
            }
        }
    }
}
