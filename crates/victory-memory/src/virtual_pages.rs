//! Lifetime identities for ordinary title virtual allocations. The host
//! can commit/decommit individual 4 KiB units. Admit only complete 64 KiB
//! pages, and never let a recommit at the same VA reuse an identity.
use std::collections::{BTreeMap, BTreeSet};

use crate::{PAGE_BYTES, physical::PhysicalPage};

const PAGE: usize = PAGE_BYTES as usize;
const GRANULE: usize = 4096;

#[derive(Clone, Copy)]
struct Slot {
    allocation: usize,
    committed: u16,
    id: Option<PhysicalPage>,
}

#[derive(Default)]
pub struct VirtualPages {
    pages: BTreeMap<usize, Slot>,
    reservations: BTreeSet<usize>,
    next: usize,
    committed_bytes: u64,
}

impl VirtualPages {
    pub fn committed_bytes(&self) -> u64 {
        self.committed_bytes
    }

    /// Bytes the host may add to an ordinary commit. With no allocation
    /// identity, reserve the entire rounded request (new allocation).
    pub fn additional_bytes(
        &self,
        allocation: Option<usize>,
        address: usize,
        len: usize,
    ) -> Option<u64> {
        if len == 0 {
            return Some(0);
        }
        let end = address
            .checked_add(len)?
            .checked_next_multiple_of(GRANULE)?;
        let start = address & !(GRANULE - 1);
        let mut bytes = (end - start) as u64;
        if let Some(allocation) = allocation {
            for (&base, slot) in self.pages.range((start & !(PAGE - 1))..end) {
                if slot.allocation == allocation {
                    let overlapping =
                        bits(base, base.max(start), base.saturating_add(PAGE).min(end))
                            & slot.committed;
                    bytes -= u64::from(overlapping.count_ones()) * GRANULE as u64;
                }
            }
        }
        Some(bytes)
    }

    pub fn page(&self, address: usize) -> Option<PhysicalPage> {
        self.pages.get(&(address & !(PAGE - 1)))?.id
    }

    /// 只标记明确由运行时新建的独立虚拟内存保留区，而不是现成堆上的提交。
    pub fn reserved(&mut self, allocation: usize) {
        self.released(allocation);
        self.reservations.insert(allocation);
    }

    pub fn has_own_reservation(&self, address: usize) -> bool {
        self.pages.get(&(address & !(PAGE - 1))).is_some_and(|slot| {
            slot.id.is_some() && self.reservations.contains(&slot.allocation)
        })
    }

    /// Runs of private backing, or gaps. A known private ID spans one
    /// physical page; long unknown ranges do not force per-page scanning
    /// in the renderer's compatibility path.
    pub fn run_at(&self, address: usize, end: usize) -> (usize, Option<PhysicalPage>) {
        let base = address & !(PAGE - 1);
        if let Some(page) = self.page(address) {
            return (end.min(base.saturating_add(PAGE)), Some(page));
        }
        let next = self
            .pages
            .range(base.saturating_add(PAGE)..end.max(base.saturating_add(PAGE)))
            .find(|(_, slot)| slot.id.is_some())
            .map_or(end, |(&va, _)| va);
        (next, None)
    }

    /// Mark host-verified commit only. A repeated commit of an existing
    /// byte range does not give a page a new identity.
    pub fn committed(&mut self, allocation: usize, address: usize, len: usize) {
        let Some(end) = address.checked_add(len) else {
            return;
        };
        if address >= end {
            return;
        }
        let start = address & !(GRANULE - 1);
        let Some(end) = end.checked_next_multiple_of(GRANULE) else {
            return;
        };
        let mut at = start;
        while at < end {
            let base = at & !(PAGE - 1);
            let stop = end.min(base.saturating_add(PAGE));
            let slot = self.pages.entry(base).or_insert(Slot {
                allocation,
                committed: 0,
                id: None,
            });
            let previous = slot.committed.count_ones();
            if slot.allocation != allocation {
                *slot = Slot {
                    allocation,
                    committed: 0,
                    id: None,
                };
            }
            slot.committed |= bits(base, at, stop);
            self.committed_bytes -= u64::from(previous) * GRANULE as u64;
            self.committed_bytes += u64::from(slot.committed.count_ones()) * GRANULE as u64;
            if slot.committed == u16::MAX && slot.id.is_none() {
                slot.id = PhysicalPage::private(self.next);
                if slot.id.is_some() {
                    self.next += 1;
                }
            }
            at = stop;
        }
    }

    pub fn decommitted(&mut self, address: usize, len: usize) {
        let Some(end) = address.checked_add(len) else {
            return;
        };
        if address >= end {
            return;
        }
        let start = address & !(GRANULE - 1);
        let Some(end) = end.checked_next_multiple_of(GRANULE) else {
            return;
        };
        let mut empty = Vec::new();
        for (&base, slot) in self.pages.range_mut((start & !(PAGE - 1))..end) {
            let from = base.max(start);
            let to = base.saturating_add(PAGE).min(end);
            let lost = bits(base, from, to);
            let removed = slot.committed & lost;
            if removed != 0 {
                self.committed_bytes -= u64::from(removed.count_ones()) * GRANULE as u64;
                slot.committed &= !lost;
                slot.id = None;
            }
            if slot.committed == 0 {
                empty.push(base);
            }
        }
        for base in empty {
            self.pages.remove(&base);
        }
    }

    /// Host-confirmed MEM_RELEASE of the whole allocation.
    pub fn released(&mut self, allocation: usize) {
        self.reservations.remove(&allocation);
        self.pages.retain(|_, slot| {
            if slot.allocation != allocation {
                return true;
            }
            self.committed_bytes -= u64::from(slot.committed.count_ones()) * GRANULE as u64;
            false
        });
    }
}

fn bits(base: usize, start: usize, end: usize) -> u16 {
    let first = (start - base) / GRANULE;
    let count = (end - start) / GRANULE;
    (((1u32 << count) - 1) << first) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn own_reservation_is_not_inferred_from_commit_and_does_not_survive_release() {
        let mut v = VirtualPages::default();
        let va = PAGE;
        v.committed(va, va, PAGE);
        let unproven = v.page(va).unwrap();
        assert!(!v.has_own_reservation(va));
        v.reserved(va);
        assert_eq!(v.page(va), None);
        v.committed(va, va, PAGE);
        assert_ne!(v.page(va), Some(unproven));
        assert!(v.has_own_reservation(va));
        v.decommitted(va, PAGE);
        assert!(!v.has_own_reservation(va));
        v.committed(va, va, PAGE);
        assert!(v.has_own_reservation(va));
        v.released(va);
        v.committed(va, va, PAGE);
        assert!(!v.has_own_reservation(va));
    }

    #[test]
    fn private_commit_preflight_counts_only_new_granules() {
        let mut v = VirtualPages::default();
        let va = PAGE;
        v.committed(va, va, GRANULE);
        assert_eq!(v.committed_bytes(), GRANULE as u64);
        assert_eq!(
            v.additional_bytes(Some(va), va + 5, GRANULE),
            Some(GRANULE as u64)
        );
        assert_eq!(
            v.additional_bytes(None, va + 5, GRANULE),
            Some(2 * GRANULE as u64)
        );
        v.committed(va, va + 5, GRANULE);
        assert_eq!(v.committed_bytes(), 2 * GRANULE as u64);
        v.decommitted(va, GRANULE);
        assert_eq!(v.committed_bytes(), GRANULE as u64);
        v.released(va);
        assert_eq!(v.committed_bytes(), 0);
    }

    #[test]
    fn tracked_ram_charge_grows_and_shrinks_without_losing_budget_on_failure() {
        use crate::{
            budget::{Pool, RAM_LIMIT},
            physical::HostBudget,
        };
        let budget = HostBudget::default();
        let mut private = budget.reserve_ram(0).unwrap();
        let _section = budget.reserve_ram(RAM_LIMIT - PAGE_BYTES).unwrap();
        private.resize(PAGE_BYTES).unwrap();
        assert!(private.resize(PAGE_BYTES + 1).is_err());
        assert_eq!(budget.used(Pool::Ram), RAM_LIMIT);
        private.resize(4096).unwrap();
        assert_eq!(budget.used(Pool::Ram), RAM_LIMIT - PAGE_BYTES + 4096);
        drop(private);
        assert_eq!(budget.used(Pool::Ram), RAM_LIMIT - PAGE_BYTES);
    }

    #[test]
    fn complete_private_pages_change_identity_on_decommit_and_reuse() {
        let mut v = VirtualPages::default();
        let va = PAGE;
        v.committed(va, va, GRANULE);
        assert_eq!(v.page(va), None);
        v.committed(va, va + GRANULE, 2 * PAGE - GRANULE);
        let before = v.page(va).unwrap();
        assert!(before.is_private());
        let neighbor = v.page(va + PAGE).unwrap();
        assert_eq!(v.run_at(0x1000, va * 8), (va, None));
        assert_eq!(v.run_at(va + 12, va * 8), (va * 2, Some(before)));
        assert_ne!(neighbor, before);
        v.committed(va, va + GRANULE, GRANULE);
        assert_eq!(v.page(va), Some(before));
        v.decommitted(va + GRANULE, GRANULE);
        assert_eq!(v.page(va), None);
        assert_eq!(v.page(va + PAGE), Some(neighbor));
        v.committed(va, va + GRANULE, GRANULE);
        assert_ne!(v.page(va), Some(before));
        v.released(va);
        assert_eq!(v.page(va + PAGE), None);
        v.committed(va, va, 2 * PAGE);
        assert_ne!(v.page(va + PAGE), Some(neighbor));
    }
}
