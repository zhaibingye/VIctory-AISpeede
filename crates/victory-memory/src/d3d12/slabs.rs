//! Local D3D12 backing. Each slot is a persistent buffer at a disjoint heap
//! offset. Eviction reassigns the *same* buffer after its fence, not an aliased
//! replacement resource; no stale resource descriptor can see a new page.

use std::collections::BTreeMap;

use windows::Win32::Graphics::Direct3D12::*;

use crate::{
    PAGE_BYTES,
    budget::{NoSpace, Pool},
    engine::Error as PolicyError,
};

use super::{Error, copy::CopyQueue};

const SLOTS_PER_SLAB: usize = 16;

pub(super) struct Slot {
    slab: u64,
    index: usize,
}

struct Slab {
    buffers: Vec<ID3D12Resource>,
    free: Vec<usize>,
}

#[derive(Default)]
pub(super) struct Slabs {
    slabs: BTreeMap<u64, Slab>,
    next: u64,
    bytes: u64,
}

impl Slabs {
    pub(super) fn bytes(&self) -> u64 {
        self.bytes
    }

    pub(super) fn allocate(&mut self, queue: &CopyQueue, cap: u64) -> Result<Slot, Error> {
        for (&slab, candidate) in &mut self.slabs {
            if let Some(index) = candidate.free.pop() {
                return Ok(Slot { slab, index });
            }
        }

        let headroom = cap.saturating_sub(self.bytes);
        let count = (headroom / PAGE_BYTES).min(SLOTS_PER_SLAB as u64) as usize;
        if count == 0 {
            return Err(PolicyError::Space(NoSpace {
                pool: Pool::Local,
                shortfall: PAGE_BYTES - headroom,
            }).into());
        }
        let slab = self.next;
        self.next = self.next.checked_add(1).expect("local slab identity exhausted");
        let buffers = queue.local_slab(count)?;
        self.bytes += count as u64 * PAGE_BYTES;
        self.slabs.insert(slab, Slab {
            buffers,
            free: (1..count).rev().collect(),
        });
        Ok(Slot { slab, index: 0 })
    }

    pub(super) fn buffer(&self, slot: &Slot) -> &ID3D12Resource {
        &self.slabs[&slot.slab].buffers[slot.index]
    }

    /// No list using the slot may still be running. The last logical page
    /// gives the entire heap back; idle unused slots never outlive a slab.
    pub(super) fn release(&mut self, slot: Slot) {
        let slab = self.slabs.get_mut(&slot.slab).expect("live local slab");
        debug_assert!(!slab.free.contains(&slot.index));
        slab.free.push(slot.index);
        if slab.free.len() == slab.buffers.len() {
            let slab = self.slabs.remove(&slot.slab).expect("empty local slab");
            let bytes = slab.buffers.len() as u64 * PAGE_BYTES;
            drop(slab); // Release the native heap before reducing the charge.
            self.bytes -= bytes;
        }
    }
}
