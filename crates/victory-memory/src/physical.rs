//! Physical allocations and alias identity, independent of host address.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::{
    PAGE_BYTES,
    budget::{Budget, Charge, NoSpace, Pool},
};

/// Thread-safe ledger for the host's managed physical allocations. XMem and
/// graphics use the same process ledger; isolated fixtures can own another.
#[derive(Clone, Default)]
pub struct HostBudget(Arc<Mutex<Budget>>);

impl HostBudget {
    pub fn process() -> &'static Self {
        static HOST: std::sync::LazyLock<HostBudget> =
            std::sync::LazyLock::new(HostBudget::default);
        &HOST
    }

    pub fn reserve_ram(&self, bytes: u64) -> Result<Reservation, NoSpace> {
        let charge = self.lock().charge(Pool::Ram, bytes)?;
        Ok(self.reservation(charge))
    }

    /// DXGI reports our process's total local use, including managed
    /// resources, so count excess over the ledger rather than adding both.
    pub fn reserve_local(
        &self, bytes: u64, observed: u64, dxgi_budget: u64, headroom: u64,
    ) -> Result<Reservation, NoSpace> {
        let mut budget = self.lock();
        let outside = observed.saturating_sub(budget.used(Pool::Local));
        budget.observe_local(dxgi_budget, outside.saturating_add(headroom));
        let charge = budget.charge(Pool::Local, bytes)?;
        Ok(self.reservation(charge))
    }

    pub fn used(&self, pool: Pool) -> u64 {
        self.lock().used(pool)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Budget> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn reservation(&self, charge: Charge) -> Reservation {
        Reservation {
            budget: self.clone(),
            charge: Some(charge),
        }
    }
}

/// Retain this lease with the host allocation, not with individual views.
pub struct Reservation {
    budget: HostBudget,
    charge: Option<Charge>,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.budget.lock().release(self.charge.take().expect("one charge per reservation"));
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

/// Retains real host backing across a GPU submission. Aliases and repeated
/// ranges can name one allocation; there is only one retained reference to it.
#[derive(Default)]
pub struct BackingPins(HashMap<PhysicalPage, Arc<dyn Send + Sync>>);

impl BackingPins {
    /// `section` identifies the first physical page of this allocation.
    pub fn keep<T: Send + Sync + 'static>(&mut self, section: PhysicalPage, backing: &Arc<T>) {
        self.0.entry(section).or_insert_with(|| backing.clone());
    }
}
