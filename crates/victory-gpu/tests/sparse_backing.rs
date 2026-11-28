//! Small native D3D12 capability probe; no game, shaders, or presentation.
//! Backing size describes the explicit tile heap, not total driver memory.
//! Set `VICTORY_TEST_SPARSE_BACKING=1`, then run:
//! `cargo test -p victory-gpu --test sparse_backing reserved_buffer_two_tiles -- --exact --nocapture`
#![cfg(windows)]

use std::mem::ManuallyDrop;
use std::ptr::{copy_nonoverlapping, null_mut};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::Graphics::Dxgi::*;
use windows::core::Interface;

const TILE_BYTES: u32 = D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
const VIRTUAL_BYTES: u32 = 64 << 20;
const VIRTUAL_TILES: u32 = VIRTUAL_BYTES / TILE_BYTES;
const TILE_INDICES: [u32; 2] = [1, VIRTUAL_TILES - 1];
const BACKING_BYTES: u64 = TILE_INDICES.len() as u64 * TILE_BYTES as u64;
const PATTERN_BYTES: usize = 16;
const PAYLOAD: &[u8; 2 * PATTERN_BYTES] = b"0123456789ABCDEFqrstuvwxyz!@#$%^";
const FENCE_VALUE: u64 = 1;
const FENCE_TIMEOUT: Duration = Duration::from_millis(1500);

#[test]
fn reserved_buffer_two_tiles() {
    if !std::env::var_os("VICTORY_TEST_SPARSE_BACKING").is_some_and(|v| v == "1") {
        eprintln!("CAPABILITY sparse_backing status=skipped opt_in=VICTORY_TEST_SPARSE_BACKING=1");
        return;
    }
    if let Err(e) = probe() {
        panic!("CAPABILITY sparse_backing status=fail error={e:#}");
    }
}

fn nvidia_adapter() -> Result<(IDXGIAdapter1, DXGI_ADAPTER_DESC1)> {
    const NVIDIA_VENDOR_ID: u32 = 0x10de;
    const EXCLUDED_FLAGS: u32 =
        DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 | DXGI_ADAPTER_FLAG_REMOTE.0 as u32;
    // SAFETY: ordinary DXGI enumeration; no default or WARP fallback.
    unsafe {
        let factory: IDXGIFactory4 =
            CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).context("CreateDXGIFactory2")?;
        for i in 0.. {
            let adapter = match factory.EnumAdapters1(i) {
                Ok(adapter) => adapter,
                Err(e) if e.code() == DXGI_ERROR_NOT_FOUND => break,
                Err(e) => return Err(e).context("EnumAdapters1"),
            };
            let desc = adapter.GetDesc1().context("GetDesc1")?;
            if desc.VendorId == NVIDIA_VENDOR_ID
                && desc.DedicatedVideoMemory != 0
                && desc.Flags & EXCLUDED_FLAGS == 0
            {
                return Ok((adapter, desc));
            }
        }
    }
    bail!("no local non-software NVIDIA adapter with dedicated video memory")
}

fn buffer_desc(width: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: width,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    }
}

fn staging(
    device: &ID3D12Device,
    kind: D3D12_HEAP_TYPE,
    state: D3D12_RESOURCE_STATES,
) -> Result<ID3D12Resource> {
    let mut resource = None;
    // SAFETY: the caller supplies the heap type and its required initial state.
    unsafe {
        device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: kind,
                CreationNodeMask: 1,
                VisibleNodeMask: 1,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &buffer_desc(PAYLOAD.len() as u64),
            state,
            None,
            &mut resource,
        )?;
    }
    resource.context("CreateCommittedResource returned no buffer")
}

fn record_copies(
    device: &ID3D12Device,
    allocator: &ID3D12CommandAllocator,
    src: &ID3D12Resource,
    src_offsets: [u64; 2],
    dst: &ID3D12Resource,
    dst_offsets: [u64; 2],
) -> Result<ID3D12CommandList> {
    // SAFETY: the two buffers are live; every range fits its buffer and each
    // sparse range fits a single tile, mapped before the lists are executed.
    unsafe {
        let list: ID3D12GraphicsCommandList = device
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, None)
            .context("CreateCommandList")?;
        for (src_at, dst_at) in src_offsets.into_iter().zip(dst_offsets) {
            list.CopyBufferRegion(dst, dst_at, src, src_at, PATTERN_BYTES as u64);
        }
        list.Close().context("Close")?;
        list.cast().context("query ID3D12CommandList")
    }
}

fn wait_for_gpu(device: &ID3D12Device, fence: &ID3D12Fence) -> Result<u64> {
    let deadline = Instant::now() + FENCE_TIMEOUT;
    loop {
        // SAFETY: querying live native D3D12 objects.
        let completed = unsafe { fence.GetCompletedValue() };
        if completed != 0 || Instant::now() >= deadline {
            // SAFETY: the device is kept live alongside the fence.
            unsafe { device.GetDeviceRemovedReason() }.context("device lost during GPU work")?;
            // A removed device may make GetCompletedValue return UINT64_MAX;
            // it must not be mistaken for a completed submission.
            ensure!(
                completed == FENCE_VALUE,
                "queue fence expected {FENCE_VALUE}, observed {completed} (limit {FENCE_TIMEOUT:?})"
            );
            return Ok(completed);
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn probe() -> Result<()> {
    let (adapter, desc) = nvidia_adapter()?;
    let name = String::from_utf16_lossy(&desc.Description);
    let name = name.trim_end_matches('\0');
    eprintln!(
        "CAPABILITY sparse_backing status=adapter name={name:?} vendor={:#06x} device={:#06x}",
        desc.VendorId, desc.DeviceId
    );

    // SAFETY: the probe owns all objects and CPU mappings; only the two
    // mapped tiles are copied, and readback waits for the native queue fence.
    unsafe {
        let mut device = None;
        D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)
            .context("D3D12CreateDevice on the selected adapter")?;
        let device: ID3D12Device = device.context("D3D12CreateDevice returned no device")?;
        ensure!(
            device.GetNodeCount() == 1,
            "probe requires a single D3D12 node"
        );
        let mut arch = D3D12_FEATURE_DATA_ARCHITECTURE::default();
        device
            .CheckFeatureSupport(
                D3D12_FEATURE_ARCHITECTURE,
                (&raw mut arch).cast(),
                size_of_val(&arch) as u32,
            )
            .context("CheckFeatureSupport(ARCHITECTURE)")?;
        ensure!(!arch.UMA.as_bool(), "selected adapter is UMA, not discrete");
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
            .context("CheckFeatureSupport(D3D12_OPTIONS)")?;
        let tier = options.TiledResourcesTier.0;
        eprintln!(
            "CAPABILITY sparse_backing status=features uma=false tiled_resources_tier={tier}"
        );
        ensure!(
            tier >= D3D12_TILED_RESOURCES_TIER_2.0,
            "TiledResourcesTier={tier}; need >=2"
        );

        let mut heap = None;
        device
            .CreateHeap(
                &D3D12_HEAP_DESC {
                    SizeInBytes: BACKING_BYTES,
                    Properties: D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        CreationNodeMask: 1,
                        VisibleNodeMask: 1,
                        ..Default::default()
                    },
                    Alignment: 0,
                    Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
                },
                &mut heap,
            )
            .context("CreateHeap for two 64 KiB tiles")?;
        let heap: ID3D12Heap = heap.context("CreateHeap returned no heap")?;
        let mut sparse = None;
        device
            .CreateReservedResource(
                &buffer_desc(u64::from(VIRTUAL_BYTES)),
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut sparse,
            )
            .context("CreateReservedResource for 64 MiB buffer")?;
        let sparse: ID3D12Resource = sparse.context("CreateReservedResource returned no buffer")?;
        let mut tiles = 0;
        device.GetResourceTiling(&sparse, Some(&mut tiles), None, None, None, 0, null_mut());
        ensure!(
            tiles == VIRTUAL_TILES,
            "reserved buffer reported {tiles} tiles; expected {VIRTUAL_TILES}"
        );

        let upload = staging(
            &device,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )
        .context("create upload staging buffer")?;
        let readback = staging(
            &device,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )
        .context("create readback staging buffer")?;
        let payload_range = D3D12_RANGE {
            Begin: 0,
            End: PAYLOAD.len(),
        };
        let empty_range = D3D12_RANGE::default();
        let mut data = null_mut();
        upload
            .Map(0, Some(&empty_range), Some(&mut data))
            .context("Map upload")?;
        ensure!(!data.is_null(), "Map upload returned no CPU pointer");
        copy_nonoverlapping(PAYLOAD.as_ptr(), data.cast::<u8>(), PAYLOAD.len());
        upload.Unmap(0, Some(&payload_range));

        let queue: ID3D12CommandQueue = device
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
            .context("CreateCommandQueue")?;
        let allocator: ID3D12CommandAllocator = device
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .context("CreateCommandAllocator")?;
        let fence: ID3D12Fence = device
            .CreateFence(0, D3D12_FENCE_FLAG_NONE)
            .context("CreateFence")?;
        let staging_offsets = [0, PATTERN_BYTES as u64];
        let sparse_offsets = TILE_INDICES.map(|tile| u64::from(tile) * u64::from(TILE_BYTES));
        let up = record_copies(
            &device,
            &allocator,
            &upload,
            staging_offsets,
            &sparse,
            sparse_offsets,
        )
        .context("record upload copies")?;
        let down = record_copies(
            &device,
            &allocator,
            &sparse,
            sparse_offsets,
            &readback,
            staging_offsets,
        )
        .context("record readback copies")?;

        let regions = TILE_INDICES.map(|x| D3D12_TILED_RESOURCE_COORDINATE {
            X: x,
            ..Default::default()
        });
        let sizes = [D3D12_TILE_REGION_SIZE {
            NumTiles: 1,
            ..Default::default()
        }; 2];
        let flags = [D3D12_TILE_RANGE_FLAG_NONE; 2];
        let heap_offsets = [0, 1];
        let counts = [1, 1];
        // Keep all objects live even on an error after queuing work. If the
        // wait fails these small allocations are left to test-process teardown.
        let in_flight = ManuallyDrop::new((
            up, down, allocator, upload, readback, sparse, heap, fence, queue, device,
        ));
        let (up, down, _, _, _, sparse, heap, fence, queue, device) = &*in_flight;
        queue.UpdateTileMappings(
            sparse,
            regions.len() as u32,
            Some(regions.as_ptr()),
            Some(sizes.as_ptr()),
            heap,
            heap_offsets.len() as u32,
            Some(flags.as_ptr()),
            Some(heap_offsets.as_ptr()),
            Some(counts.as_ptr()),
            D3D12_TILE_MAPPING_FLAG_NONE,
        );
        // UpdateTileMappings is void; check removal now and verify both
        // mappings via the copies and the fence on this same queue.
        device
            .GetDeviceRemovedReason()
            .context("device lost after UpdateTileMappings")?;
        // Two ExecuteCommandLists calls: the buffer promotes COMMON ->
        // COPY_DEST for upload, decays to COMMON, then promotes to COPY_SOURCE.
        queue.ExecuteCommandLists(&[Some(up.clone())]);
        queue.ExecuteCommandLists(&[Some(down.clone())]);
        queue.Signal(fence, FENCE_VALUE).context("queue Signal")?;
        let completed = wait_for_gpu(device, fence)?;

        let finished = ManuallyDrop::into_inner(in_flight);
        let (_, _, _, _, readback, ..) = &finished;
        let mut data = null_mut();
        readback
            .Map(0, Some(&payload_range), Some(&mut data))
            .context("Map readback")?;
        ensure!(!data.is_null(), "Map readback returned no CPU pointer");
        let mut actual = [0u8; PAYLOAD.len()];
        copy_nonoverlapping(data.cast::<u8>(), actual.as_mut_ptr(), actual.len());
        readback.Unmap(0, Some(&empty_range));
        for (i, (got, want)) in actual
            .chunks_exact(PATTERN_BYTES)
            .zip(PAYLOAD.chunks_exact(PATTERN_BYTES))
            .enumerate()
        {
            ensure!(
                got == want,
                "readback mismatch at virtual tile {}",
                TILE_INDICES[i]
            );
        }
        eprintln!(
            "CAPABILITY sparse_backing status=pass tiled_resources_tier={tier} virtual_bytes={VIRTUAL_BYTES} backing_heap_bytes={BACKING_BYTES} mapped_tiles={} verified_bytes={} fence_completed={completed}",
            TILE_INDICES.len(),
            PAYLOAD.len()
        );
    }
    Ok(())
}
