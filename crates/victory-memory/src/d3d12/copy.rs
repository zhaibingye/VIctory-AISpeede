use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::DXGI_SAMPLE_DESC};
use windows::core::Interface;

use super::{Error, PAGE_LEN};
use crate::{PAGE_BYTES, coherence::Fence};

pub(super) struct CopyQueue {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    completed: Fence,
    lost: bool,
}

impl CopyQueue {
    pub(super) fn new(device: ID3D12Device) -> Result<Self, Error> {
        // SAFETY: valid descriptors; this queue and fence belong to one device.
        let (queue, fence) = unsafe {
            (
                device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_COPY,
                    ..Default::default()
                })?,
                device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?,
            )
        };
        Ok(Self { device, queue, fence, completed: Fence(0), lost: false })
    }

    pub(super) fn check(&self) -> Result<(), Error> {
        if self.lost { Err(Error::QueueLost) } else { Ok(()) }
    }

    pub(super) fn completed(&self) -> Fence {
        self.completed
    }

    pub(super) fn buffer(
        &self,
        heap: D3D12_HEAP_TYPE,
        state: D3D12_RESOURCE_STATES,
    ) -> Result<ID3D12Resource, Error> {
        self.check()?;
        let mut out = None;
        // SAFETY: valid single-node heap and 64 KiB row-major buffer; writable COM output.
        unsafe {
            self.device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: heap,
                    CreationNodeMask: 1,
                    VisibleNodeMask: 1,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: PAGE_BYTES,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                },
                state,
                None,
                &mut out,
            )?;
        }
        Ok(out.expect("successful D3D12 resource creation"))
    }

    pub(super) fn copy(&mut self, src: &ID3D12Resource, dst: &ID3D12Resource) -> Result<Fence, Error> {
        self.check()?;
        let value = self.completed.0.checked_add(1).filter(|&v| v != u64::MAX).ok_or(Error::FenceExhausted)?;
        // SAFETY: distinct live, page-sized buffers on this device. Default
        // buffers promote from COMMON and decay after each copy submission.
        // We retain the list, allocator and resources through its fence.
        unsafe {
            let allocator: ID3D12CommandAllocator =
                self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None,
            )?;
            list.CopyBufferRegion(dst, 0, src, 0, PAGE_BYTES);
            list.Close()?;
            let submission = [Some(list.cast::<ID3D12CommandList>()?)];
            self.queue.ExecuteCommandLists(&submission);
            if let Err(e) = self.queue.Signal(&self.fence, value) {
                self.lost = true;
                if self.device.GetDeviceRemovedReason().is_ok() {
                    // No completion marker was queued. Preserve these COM
                    // references rather than freeing potentially in-flight work.
                    std::mem::forget((allocator, list, src.clone(), dst.clone(), self.queue.clone()));
                }
                return Err(e.into());
            }
            if self.fence.SetEventOnCompletion(value, Default::default()).is_err() {
                // Registering the wait can itself run out of memory. A signal
                // was queued, so wait without allocating another event.
                while self.fence.GetCompletedValue() < value {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
            if self.fence.GetCompletedValue() == u64::MAX {
                self.lost = true;
                self.device.GetDeviceRemovedReason()?;
                return Err(Error::QueueLost);
            }
        }
        self.completed = Fence(value);
        Ok(self.completed)
    }
}

pub(super) fn write_upload(resource: &ID3D12Resource, bytes: &[u8; PAGE_LEN]) -> Result<(), Error> {
    let no_read = D3D12_RANGE { Begin: 0, End: 0 };
    let written = D3D12_RANGE { Begin: 0, End: PAGE_LEN };
    let mut ptr = std::ptr::null_mut();
    // SAFETY: the caller owns this 64 KiB upload buffer and prior copies have
    // completed. Map supplies writable storage; Unmap advertises our writes.
    unsafe {
        resource.Map(0, Some(&no_read), Some(&mut ptr))?;
        debug_assert!(!ptr.is_null());
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast(), PAGE_LEN);
        resource.Unmap(0, Some(&written));
    }
    Ok(())
}

pub(super) fn read_back(resource: &ID3D12Resource, bytes: &mut [u8; PAGE_LEN]) -> Result<(), Error> {
    let read = D3D12_RANGE { Begin: 0, End: PAGE_LEN };
    let no_write = D3D12_RANGE { Begin: 0, End: 0 };
    let mut ptr = std::ptr::null_mut();
    // SAFETY: the copy fence completed. The caller owns the entire writable
    // CPU page and the completed readback buffer is mapped for CPU reads.
    unsafe {
        resource.Map(0, Some(&read), Some(&mut ptr))?;
        debug_assert!(!ptr.is_null());
        std::ptr::copy_nonoverlapping(ptr.cast(), bytes.as_mut_ptr(), PAGE_LEN);
        resource.Unmap(0, Some(&no_write));
    }
    Ok(())
}
