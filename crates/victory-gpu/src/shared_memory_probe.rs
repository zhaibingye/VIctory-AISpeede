//! Small title-free test of the first in-place buffer primitive.
//! See fxceless/victory-fork#772 and XCL's coherent system-memory window.
//! Does not open game pages, change the page pool, or allocate a VRAM mirror.

use std::ptr::NonNull;
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::DXGI_SAMPLE_DESC};
use windows::core::Interface;
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
};

const BYTES: usize = 64 * 1024;

struct Committed(NonNull<u8>);

impl Committed {
    fn new() -> Self {
        // SAFETY: a fresh, zero-filled allocation owned by this fixture.
        let data = unsafe {
            VirtualAlloc(std::ptr::null(), BYTES, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE)
        };
        Self(NonNull::new(data.cast()).expect("commit one allocation-granularity block"))
    }
}

impl Drop for Committed {
    fn drop(&mut self) {
        // SAFETY: the fixture drops the resource and heap first; this is
        // the original base of the one owned reservation.
        let freed = unsafe { VirtualFree(self.0.as_ptr().cast(), 0, MEM_RELEASE) };
        assert_ne!(freed, 0, "host heap did not release its CPU allocation");
    }
}

fn description(flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: BYTES as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    }
}

#[test]
#[ignore = "requires a D3D12 adapter with opened system-memory heaps; no game files"]
fn gpu_reads_cpu_writes_from_one_committed_page_without_upload_staging() {
    let host = crate::host::host().expect("D3D12 device");
    // Declaration order makes both GPU objects drop before this allocation.
    let cpu = Committed::new();
    let dev3: ID3D12Device3 = host.device.cast().expect("D3D12 device 3");
    // SAFETY: the address is the beginning of our fully committed block;
    // we keep it alive until the placed buffer and heap have been released.
    let heap: ID3D12Heap = unsafe { dev3.OpenExistingHeapFromAddress(cpu.0.as_ptr().cast()) }
        .expect("adapter should open CPU-owned memory");
    let mut source: Option<ID3D12Resource> = None;
    // SAFETY: the buffer fits at offset zero of the live opened heap.
    unsafe {
        host.device.CreatePlacedResource(
            &heap,
            0,
            &description(D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER),
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut source,
        )
    }
    .expect("place one cross-adapter read-only buffer");
    let source = source.expect("placed buffer");
    let mut readback: Option<ID3D12Resource> = None;
    // SAFETY: initialized descriptors and a writable COM output slot.
    unsafe {
        host.device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_READBACK,
                CreationNodeMask: 1,
                VisibleNodeMask: 1,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &description(D3D12_RESOURCE_FLAG_NONE),
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut readback,
        )
    }
    .expect("allocate the test oracle's readback");
    let readback = readback.expect("readback buffer");
    for want in [0x1234_abcd_u32, 0x8765_4567_u32] {
        let list = host.new_list().expect("fresh command list");
        // SAFETY: write an aligned word in our own memory before submitting
        // the copy that reads it. Both buffers cover the four addressed bytes.
        unsafe {
            cpu.0.as_ptr().cast::<u32>().write_volatile(want);
            list.list.CopyBufferRegion(&readback, 0, &source, 0, 4);
        }
        list.close();
        host.execute(&[&list]);
        let mut got = std::ptr::null_mut();
        // SAFETY: the fence for this list has completed. Only the first
        // mapped word is read and the readback lives until after Unmap.
        unsafe {
            readback.Map(0, None, Some(&mut got)).expect("map completed readback");
            assert!(!got.is_null());
            assert_eq!(got.cast::<u32>().read_volatile(), want);
            readback.Unmap(0, None);
        }
    }
}
