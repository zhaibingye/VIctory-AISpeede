//! Indirect arguments are gathered from current GPU backing and clamped
//! in a temporary UAV. No GPU-to-CPU synchronization or mutation of the
//! title's original arguments is necessary.
use victory_memory::gpu_table::ADDRESS_LIMIT;
use windows::Win32::Foundation::{E_FAIL, E_OUTOFMEMORY};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Error;

use crate::host::{Host, HostList};
use crate::page_slabs::{BatchError, PinnedBatch};
use crate::page_view::gather_span;

#[derive(Clone, Copy)]
pub(crate) struct IndirectDraw {
    pub(crate) indexed: bool,
    pub(crate) stride: u32,
    pub(crate) max: u32,
    pub(crate) arguments: u64,
    pub(crate) count: Option<u64>,
}

type Span = (u64, u64);

impl IndirectDraw {
    fn spans(self) -> Result<(Span, Option<Span>), BatchError> {
        let width = if self.indexed { 20 } else { 16 };
        let length = u64::from(self.stride) * u64::from(self.max);
        if self.stride < width || !self.stride.is_multiple_of(4)
            || self.max == 0 || self.max > 65536 || length > u64::from(u32::MAX)
        {
            return Err(BatchError::Range(self.arguments, self.arguments.saturating_add(length)));
        }
        fn span(start: u64, len: u64) -> Result<Span, BatchError> {
            let end = start.saturating_add(len);
            if start == 0 || !start.is_multiple_of(4) || end > ADDRESS_LIMIT {
                return Err(BatchError::Range(start, end));
            }
            Ok((start, end))
        }
        Ok((span(self.arguments, length)?, self.count.map(|c| span(c, 4)).transpose()?))
    }

    pub(crate) fn ranges(self, ranges: &mut Vec<Span>) -> Result<(), BatchError> {
        let (arguments, count) = self.spans()?;
        ranges.push(arguments);
        ranges.extend(count);
        Ok(())
    }

    pub(crate) fn prepare(
        self,
        host: &Host,
        list: &HostList,
        pages: &PinnedBatch,
        budget: &mut Option<ID3D12Resource>,
    ) -> Result<Prepared, BatchError> {
        let (arguments, count) = self.spans()?;
        let kind = u32::from(self.indexed);
        let oom = || BatchError::Native(Error::from(E_OUTOFMEMORY));
        let signature = crate::draw::host_signature(kind, self.stride)
            .ok_or_else(|| BatchError::Native(Error::from(E_FAIL)))?;
        let (args, _) = host.shared_buffer(arguments.1 - arguments.0).ok_or_else(oom)?;
        gather_span(list, pages, arguments, &args, D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS)?;
        let counter = if let Some(span) = count {
            let c = host.default_buffer(4).ok_or_else(oom)?;
            gather_span(list, pages, span, &c, D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT)?;
            Some(c)
        } else {
            None
        };
        if budget.is_none() {
            let (word, ptr) = host.shared_buffer(4).ok_or_else(oom)?;
            // SAFETY: mapped fresh 4-byte storage before submission.
            unsafe { ptr.cast::<u32>().write(0) };
            list.keep(&word);
            *budget = Some(word);
        }
        list.keep(&signature);
        // SAFETY: both buffers are live and ready for the GPU clamp.
        let gpu = unsafe { args.GetGPUVirtualAddress() };
        let meter = unsafe { budget.as_ref().unwrap().GetGPUVirtualAddress() };
        if !crate::indirect::bound_with_budget(
            list, &args, gpu, (kind, self.stride, self.max), meter,
        ) {
            return Err(BatchError::Native(Error::from(E_FAIL)));
        }
        Ok(Prepared { signature, args, counter, max: self.max })
    }
}

pub(crate) struct Prepared {
    signature: ID3D12CommandSignature,
    args: ID3D12Resource,
    counter: Option<ID3D12Resource>,
    max: u32,
}

impl Prepared {
    pub(crate) fn draw(&self, list: &ID3D12GraphicsCommandList) {
        // SAFETY: the signature requires only draw/indexed draw records;
        // all buffers are live in the indirect-argument state.
        unsafe {
            list.ExecuteIndirect(&self.signature, self.max, &self.args, 0, self.counter.as_ref(), 0);
        }
    }
}

#[cfg(test)]
#[test]
fn paged_indirect_checks_stride_and_entire_counted_range() {
    let mut draw = IndirectDraw { indexed: true, stride: 20, max: 2, arguments: 0x10000, count: Some(0x20000) };
    assert!(matches!(draw.spans(), Ok(((0x10000, 0x10028), Some((0x20000, 0x20004))))));
    draw.stride = 16;
    assert!(draw.spans().is_err());
    draw.stride = 20;
    draw.arguments = ADDRESS_LIMIT - 20;
    assert!(draw.spans().is_err());
}
