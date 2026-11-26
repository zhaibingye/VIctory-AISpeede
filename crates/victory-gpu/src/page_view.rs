//! Materialize contiguous fixed-function input views from captured GPU
//! pages at the point of use, including changes from earlier commands.
use std::collections::{BTreeMap, BTreeSet};

use victory_memory::{PAGE_BYTES, gpu_table::ADDRESS_LIMIT, physical::PhysicalPage};
use windows::Win32::Foundation::E_OUTOFMEMORY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R32_UINT,
};
use windows::core::Error;

use crate::host::{Host, HostList, transition};
use crate::page_slabs::{BatchError, PinnedBatch};

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VertexView {
    pub(crate) address: u64,
    pub(crate) size: u32,
    pub(crate) stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IndexView {
    pub(crate) address: u64,
    pub(crate) size: u32,
    pub(crate) format: u32,
}

impl VertexView {
    pub(crate) fn native(self, gpu: u64) -> D3D12_VERTEX_BUFFER_VIEW {
        D3D12_VERTEX_BUFFER_VIEW {
            BufferLocation: gpu,
            SizeInBytes: self.size,
            StrideInBytes: self.stride,
        }
    }
}

impl IndexView {
    pub(crate) fn native(self, gpu: u64) -> D3D12_INDEX_BUFFER_VIEW {
        D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: gpu,
            SizeInBytes: self.size,
            Format: DXGI_FORMAT(self.format as i32),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum InputView {
    Vertex(u32, VertexView),
    Index(IndexView),
}

impl InputView {
    pub(crate) fn range(self) -> Result<(u64, u64), BatchError> {
        let (start, size, alignment) = match self {
            Self::Vertex(slot, view) if slot < 32 => (view.address, view.size, 1),
            Self::Vertex(_, view) => return Err(BatchError::Range(view.address, view.address)),
            Self::Index(view) => {
                let alignment = match DXGI_FORMAT(view.format as i32) {
                    DXGI_FORMAT_R16_UINT => 2,
                    DXGI_FORMAT_R32_UINT => 4,
                    _ => return Err(BatchError::Range(view.address, view.address)),
                };
                (view.address, view.size, alignment)
            }
        };
        let end = start.saturating_add(u64::from(size));
        if start == 0 || size == 0 || end > ADDRESS_LIMIT
            || start % alignment != 0 || u64::from(size) % alignment != 0
        {
            return Err(BatchError::Range(start, end));
        }
        Ok((start, end))
    }

    pub(crate) fn encode(
        self,
        host: &Host,
        list: &HostList,
        batch: &PinnedBatch,
    ) -> Result<(), BatchError> {
        let buffer = self.gather(host, list, batch)?;
        let native = list.recording();
        // SAFETY: gather validated the slot/format and prepared a live
        // contiguous resource in the corresponding read-only state.
        unsafe {
            let gpu = buffer.GetGPUVirtualAddress();
            match self {
                Self::Vertex(slot, view) => native.IASetVertexBuffers(slot, Some(&[view.native(gpu)])),
                Self::Index(view) => native.IASetIndexBuffer(Some(&view.native(gpu))),
            }
        }
        Ok(())
    }

    /// All slabs are in UAV state at this point. Transitions expose
    /// preceding GPU writes; the resulting native buffer is budgeted
    /// and kept to the queue fence.
    pub(crate) fn gather(
        self,
        host: &Host,
        list: &HostList,
        batch: &PinnedBatch,
    ) -> Result<ID3D12Resource, BatchError> {
        let span = self.range()?;
        let buffer = host
            .default_buffer(span.1 - span.0)
            .ok_or_else(|| BatchError::Native(Error::from(E_OUTOFMEMORY)))?;
        let binding = match self {
            Self::Vertex(..) => D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            Self::Index(_) => D3D12_RESOURCE_STATE_INDEX_BUFFER,
        };
        gather_span(list, batch, span, &buffer, D3D12_RESOURCE_STATE_COPY_DEST,
            binding | D3D12_RESOURCE_STATE_COPY_SOURCE)?;
        Ok(buffer)
    }
}

/// Copy a checked admitted guest span to a separate, sufficiently sized
/// native buffer. The destination is retained until the list's fence.
pub(crate) fn gather_span(
    list: &HostList,
    batch: &PinnedBatch,
    (start, end): (u64, u64),
    buffer: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> Result<(), BatchError> {
    gather_regions(list, batch, [((start, end), 0)], buffer, before, after)
}

/// Gather multiple pitched regions while transitioning each slab once.
pub(crate) fn gather_regions(
    list: &HostList,
    batch: &PinnedBatch,
    regions: impl IntoIterator<Item = ((u64, u64), u64)>,
    buffer: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> Result<(), BatchError> {
    transfer_regions(list, batch, regions, buffer, before, after, None)
}

/// Scatter row bytes, not footprint padding, back into physical pages.
/// Only explicit guest copies establish unconditional byte ownership;
/// internal texture mirrors preserve independent CPU writes.
pub(crate) fn scatter_regions(
    list: &HostList,
    batch: &PinnedBatch,
    regions: impl IntoIterator<Item = ((u64, u64), u64)>,
    buffer: &ID3D12Resource,
    dirty: &mut BTreeSet<PhysicalPage>,
    exact: Option<&mut crate::page_resource::ResourcePlan>,
) -> Result<(), BatchError> {
    transfer_regions(list, batch, regions, buffer, D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_COPY_DEST, Some((dirty, exact)))
}

fn transfer_regions(
    list: &HostList,
    batch: &PinnedBatch,
    regions: impl IntoIterator<Item = ((u64, u64), u64)>,
    buffer: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
    output: Option<(&mut BTreeSet<PhysicalPage>, Option<&mut crate::page_resource::ResourcePlan>)>,
) -> Result<(), BatchError> {
    let scatter = output.is_some();
    let mut touched = BTreeMap::new();
    let mut writes = BTreeMap::<PhysicalPage, BTreeMap<u64, u64>>::new();
    let mut pieces = Vec::new();
    // SAFETY: query a live buffer descriptor.
    let capacity = unsafe { buffer.GetDesc() }.Width;
    for ((start, end), staging) in regions {
        if start >= end || end > ADDRESS_LIMIT
            || staging.checked_add(end - start).is_none_or(|stop| stop > capacity)
        {
            return Err(BatchError::Range(start, end));
        }
        let mut va = start;
        while va < end {
            let page = batch.pinned_at(va).ok_or(BatchError::NotCaptured(va))?;
            let offset = va % PAGE_BYTES;
            let count = (end - va).min(PAGE_BYTES - offset);
            if scatter {
                // A guest footprint may alias itself via different VAs.
                // Reject overlaps instead of racing copies to the same bytes.
                let intervals = writes.entry(page.physical()).or_default();
                if intervals.range(..offset + count).next_back()
                    .is_some_and(|(_, &end)| end > offset)
                {
                    return Err(BatchError::Range(va, va + count));
                }
                intervals.insert(offset, offset + count);
            }
            touched.insert(page.slab(), page.buffer());
            pieces.push((page.buffer(), page.offset() + offset, staging + (va - start), count));
            va += count;
        }
    }
    if pieces.is_empty() {
        return Err(BatchError::Range(0, 0));
    }
    list.keep(buffer);
    let cmd = list.recording();
    let uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    let read = D3D12_RESOURCE_STATE_COPY_SOURCE;
    let dest = D3D12_RESOURCE_STATE_COPY_DEST;
    let (page_state, stage_state) = if scatter { (dest, read) } else { (read, dest) };
    let mut begin: Vec<_> = touched.values().map(|page| transition(page, uav, page_state)).collect();
    if before != stage_state {
        begin.push(transition(buffer, before, stage_state));
    }
    let mut finish: Vec<_> = touched.values().map(|page| transition(page, page_state, uav)).collect();
    if stage_state != after {
        finish.push(transition(buffer, stage_state, after));
    }
    // SAFETY: every page segment fits a pinned slot, and the staging
    // buffer is separate and sized to each validated transfer.
    unsafe {
        cmd.ResourceBarrier(&begin);
        for (page, at, staging, count) in pieces {
            if scatter {
                cmd.CopyBufferRegion(page, at, buffer, staging, count);
            } else {
                cmd.CopyBufferRegion(buffer, staging, page, at, count);
            }
        }
        cmd.ResourceBarrier(&finish);
    }
    if let Some((dirty, mut exact)) = output {
        for (page, intervals) in writes {
            dirty.insert(page);
            if let Some(plan) = exact.as_deref_mut() {
                for (start, end) in intervals {
                    plan.record_guest_overwrite(page, start as usize, (end - start) as usize);
                }
            }
        }
    }
    Ok(())
}
