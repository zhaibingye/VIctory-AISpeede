//! Materialize contiguous fixed-function input views from captured GPU
//! pages at the point of use, including changes from earlier commands.
use std::collections::BTreeMap;

use victory_memory::{PAGE_BYTES, gpu_table::ADDRESS_LIMIT};
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
        let (start, end) = self.range()?;
        let mut source = BTreeMap::new();
        let mut pieces = Vec::new();
        let mut va = start;
        while va < end {
            let page = batch.pinned_at(va).ok_or(BatchError::NotCaptured(va))?;
            let offset = va % PAGE_BYTES;
            let count = (end - va).min(PAGE_BYTES - offset);
            source.insert(page.slab(), page.buffer());
            pieces.push((page.buffer(), page.offset() + offset, va - start, count));
            va += count;
        }
        let buffer = host
            .default_buffer(end - start)
            .ok_or_else(|| BatchError::Native(Error::from(E_OUTOFMEMORY)))?;
        list.keep(&buffer);
        let cmd = list.recording();
        let uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        let read = D3D12_RESOURCE_STATE_COPY_SOURCE;
        let binding = match self {
            Self::Vertex(..) => D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
            Self::Index(_) => D3D12_RESOURCE_STATE_INDEX_BUFFER,
        };
        let gather: Vec<_> = source.values().map(|src| transition(src, uav, read)).collect();
        let finish: Vec<_> = source
            .values()
            .map(|src| transition(src, read, uav))
            .chain([transition(&buffer, D3D12_RESOURCE_STATE_COPY_DEST, binding | read)])
            .collect();
        // SAFETY: all sources are pinned and each piece is within one
        // slot. The separate destination fits the span and stays live.
        unsafe {
            cmd.ResourceBarrier(&gather);
            for (src, from, to, count) in pieces {
                cmd.CopyBufferRegion(&buffer, to, src, from, count);
            }
            cmd.ResourceBarrier(&finish);
        }
        Ok(buffer)
    }
}
