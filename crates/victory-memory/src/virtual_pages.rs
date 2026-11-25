//! Lifetime identities for ordinary title virtual allocations. The host
//! can commit/decommit individual 4 KiB units. Admit only complete 64 KiB
//! pages, and never let a recommit at the same VA reuse an identity.
use std::collections::BTreeMap;

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
    next: usize,
}

impl VirtualPages {
    pub fn page(&self, address: usize) -> Option<PhysicalPage> {
        self.pages.get(&(address & !(PAGE - 1)))?.id
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
            if slot.allocation != allocation {
                *slot = Slot {
                    allocation,
                    committed: 0,
                    id: None,
                };
            }
            slot.committed |= bits(base, at, stop);
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
            if slot.committed & lost != 0 {
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
        self.pages.retain(|_, slot| slot.allocation != allocation);
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
