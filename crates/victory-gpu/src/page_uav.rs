//! Raw UAV clears write directly to captured slabs, with no separate
//! authoritative copy. Join aliases before creating native range views.
use std::collections::{BTreeMap, BTreeSet};

use victory_memory::{PAGE_BYTES, gpu_table::ADDRESS_LIMIT, physical::PhysicalPage};

use crate::host::{Host, HostList};
use crate::page_queue::PageDescriptors;
use crate::page_resource::{PageBarrier, ResourcePlan};
use crate::page_slabs::{BatchError, PinnedBatch, PinnedPage};

pub(crate) struct RawClear {
    pub(crate) start: u64,
    pub(crate) end: u64,
    pub(crate) values: [u32; 4],
}

impl RawClear {
    pub(crate) fn range(&self) -> Result<(u64, u64), BatchError> {
        if self.start == 0 || self.start >= self.end || self.end > ADDRESS_LIMIT
            || !self.start.is_multiple_of(16) || !self.end.is_multiple_of(4)
        {
            return Err(BatchError::Range(self.start, self.end));
        }
        Ok((self.start, self.end))
    }

    pub(crate) fn encode(
        &self,
        host: &Host,
        list: &HostList,
        pages: &PinnedBatch,
        descriptors: &mut PageDescriptors<'_>,
        resources: &mut ResourcePlan,
        dirty: &mut BTreeSet<PhysicalPage>,
    ) -> Result<(), BatchError> {
        let (mut va, end) = self.range()?;
        let mut spans: BTreeMap<_, (&PinnedPage, Vec<(u64, u64)>)> = BTreeMap::new();
        while va < end {
            let page = pages.pinned_at(va).ok_or(BatchError::NotCaptured(va))?;
            let offset = va % PAGE_BYTES;
            let len = (end - va).min(PAGE_BYTES - offset);
            spans.entry(page.physical()).or_insert_with(|| (page, Vec::new()))
                .1.push((offset, offset + len));
            va += len;
        }
        PageBarrier::Uav.encode(list, resources)?;
        let cmd = list.recording();
        // SAFETY: this heap stays bound for the clear and owned by the batch.
        unsafe { cmd.SetDescriptorHeaps(&[Some(descriptors.heap.heap.clone())]) };
        for (physical, (page, mut ranges)) in spans {
            ranges.sort_unstable();
            let mut ranges = ranges.into_iter().peekable();
            while let Some((start, mut end)) = ranges.next() {
                while let Some(&(next, last)) = ranges.peek() {
                    if next > end { break; }
                    ranges.next();
                    end = end.max(last);
                }
                let offset = page.offset() + start;
                let bytes = end - start;
                let gpu_slot = descriptors.reserve_raw_uav(page.buffer(), offset, bytes)
                    .map_err(BatchError::Native)?;
                let cpu = host.cpu_view();
                let view = crate::descriptor::host::raw_uav_desc(offset / 4, (bytes / 4) as u32);
                // SAFETY: page is pinned in UNORDERED_ACCESS, GPU/CPU
                // descriptors describe the exact same validated slot span.
                unsafe {
                    host.device.CreateUnorderedAccessView(page.buffer(), None, Some(&view), cpu);
                    cmd.ClearUnorderedAccessViewUint(descriptors.heap.gpu(gpu_slot), cpu,
                        page.buffer(), &self.values, &[]);
                }
            }
            dirty.insert(physical);
        }
        PageBarrier::Uav.encode(list, resources)
    }
}
