//! Native D3D12 sparse tile reuse probe; no game, shaders, or presentation.
//! Backing size describes the explicit tile heap, not total driver memory.
//! Set `VICTORY_TEST_SPARSE_BACKING=1`, then run:
//! `cargo test -p victory-gpu --test sparse_backing reserved_buffer_two_tiles -- --exact --nocapture`
#![cfg(windows)]

use std::mem::ManuallyDrop;
use std::ptr::{copy_nonoverlapping, null_mut};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use victory_gpu::sparse_backing::{SparseBacking, TILE_BYTES};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::Graphics::Dxgi::*;
use windows::core::Interface;

const VIRTUAL_BYTES: u32 = 64 << 20;
const VIRTUAL_TILES: u32 = VIRTUAL_BYTES / TILE_BYTES as u32;
const PHYSICAL_TILES: u32 = 2;
const BACKING_BYTES: u64 = PHYSICAL_TILES as u64 * TILE_BYTES as u64;
const RETIRED_TILE: u32 = 1;
const KEPT_TILE: u32 = VIRTUAL_TILES - 1;
const REPLACEMENT_TILE: u32 = VIRTUAL_TILES / 2;
const PATTERN_BYTES: usize = 16;
const READBACK_BYTES: usize = 2 * PATTERN_BYTES;
const PAYLOAD: &[u8; 3 * PATTERN_BYTES] = b"0123456789ABCDEFqrstuvwxyz!@#$%^recycled-tile-03";
const FIRST_FENCE: u64 = 1;
const SECOND_FENCE: u64 = 2;
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
    bytes: usize,
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
            &buffer_desc(bytes as u64),
            state,
            None,
            &mut resource,
        )?;
    }
    resource.context("CreateCommittedResource returned no buffer")
}

fn record_copies<const N: usize>(
    device: &ID3D12Device,
    allocator: &ID3D12CommandAllocator,
    src: &ID3D12Resource,
    src_offsets: [u64; N],
    dst: &ID3D12Resource,
    dst_offsets: [u64; N],
) -> Result<ID3D12CommandList> {
    // SAFETY: both buffers are live; every range fits its buffer and each
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

fn submit_copies(queue: &ID3D12CommandQueue, lists: &[ID3D12CommandList; 2]) {
    // SAFETY: the probe keeps the copy lists, allocator and resources live
    // until completion. Execute separately so the sparse buffer promotes
    // COMMON -> COPY_DEST, decays to COMMON, then promotes to COPY_SOURCE.
    unsafe {
        for list in lists {
            queue.ExecuteCommandLists(&[Some(list.clone())]);
        }
    }
}

fn wait_for_gpu(device: &ID3D12Device, fence: &ID3D12Fence, expected: u64) -> Result<u64> {
    let deadline = Instant::now() + FENCE_TIMEOUT;
    loop {
        // SAFETY: querying live native D3D12 objects.
        let completed = unsafe { fence.GetCompletedValue() };
        if completed >= expected || Instant::now() >= deadline {
            // SAFETY: the device is kept live alongside the fence.
            unsafe { device.GetDeviceRemovedReason() }.context("device lost during GPU work")?;
            // Device removal can return u64::MAX, not a completed signal.
            ensure!(
                completed == expected,
                "queue fence expected {expected}, observed {completed} (limit {FENCE_TIMEOUT:?})"
            );
            return Ok(completed);
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn verify_readback(readback: &ID3D12Resource, tiles: [u32; 2], expected: &[u8]) -> Result<()> {
    ensure!(expected.len() == READBACK_BYTES, "invalid readback fixture");
    let range = D3D12_RANGE {
        Begin: 0,
        End: READBACK_BYTES,
    };
    let mut data = null_mut();
    let mut actual = [0u8; READBACK_BYTES];
    // SAFETY: the phase's fence has completed; the buffer is live and no GPU
    // work accesses it while we map and read the committed staging range.
    unsafe {
        readback
            .Map(0, Some(&range), Some(&mut data))
            .context("Map readback")?;
        if !data.is_null() {
            copy_nonoverlapping(data.cast::<u8>(), actual.as_mut_ptr(), actual.len());
        }
        readback.Unmap(0, Some(&D3D12_RANGE::default()));
    }
    ensure!(!data.is_null(), "Map readback returned no CPU pointer");
    for (i, (got, want)) in actual
        .chunks_exact(PATTERN_BYTES)
        .zip(expected.chunks_exact(PATTERN_BYTES))
        .enumerate()
    {
        ensure!(
            got == want,
            "readback mismatch at virtual tile {}",
            tiles[i]
        );
    }
    Ok(())
}

fn probe() -> Result<()> {
    let (adapter, desc) = nvidia_adapter()?;
    let name = String::from_utf16_lossy(&desc.Description);
    let name = name.trim_end_matches('\0');
    eprintln!(
        "CAPABILITY sparse_backing status=adapter name={name:?} vendor={:#06x} device={:#06x}",
        desc.VendorId, desc.DeviceId
    );

    // SAFETY: the probe owns all objects and CPU mappings; only mapped tiles
    // are copied in each phase, and readback waits for the native queue fence.
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

        let upload = staging(
            &device,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            PAYLOAD.len(),
        )
        .context("create upload staging buffer")?;
        let readback = staging(
            &device,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
            READBACK_BYTES,
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
        let backing = SparseBacking::new(&device, &queue, &fence, VIRTUAL_TILES, PHYSICAL_TILES)
            .context("create sparse backing")?;
        let sparse = backing.buffer();
        let initial_tiles = [RETIRED_TILE, KEPT_TILE];
        let replacement_tiles = [KEPT_TILE, REPLACEMENT_TILE];
        let tile_offset = |tile| u64::from(tile) * TILE_BYTES as u64;
        let staging_offsets = [0, PATTERN_BYTES as u64];
        let initial = [
            record_copies(
                &device,
                &allocator,
                &upload,
                staging_offsets,
                sparse,
                initial_tiles.map(tile_offset),
            )
            .context("record initial upload copies")?,
            record_copies(
                &device,
                &allocator,
                sparse,
                initial_tiles.map(tile_offset),
                &readback,
                staging_offsets,
            )
            .context("record initial readback copies")?,
        ];
        let replacement = [
            record_copies(
                &device,
                &allocator,
                &upload,
                [READBACK_BYTES as u64],
                sparse,
                [tile_offset(REPLACEMENT_TILE)],
            )
            .context("record replacement upload copy")?,
            // Read the kept tile into a different slot so stale readback bytes
            // from phase one cannot satisfy either phase-two comparison.
            record_copies(
                &device,
                &allocator,
                sparse,
                replacement_tiles.map(tile_offset),
                &readback,
                staging_offsets,
            )
            .context("record replacement readback copies")?,
        ];

        // Mapping itself queues work. Keep all GPU objects live on any error
        // from here until the final fence; teardown can reclaim them if needed.
        let mut in_flight = ManuallyDrop::new((
            initial,
            replacement,
            allocator,
            upload,
            readback,
            backing,
            fence,
            queue,
            device,
        ));
        let (initial, replacement, _, _, readback, backing, fence, queue, device) = &mut *in_flight;
        for tile in initial_tiles {
            ensure!(backing.map_tile(tile)?, "failed to map initial tile {tile}");
        }
        submit_copies(queue, initial);
        // The retired tile's only GPU uses have now been enqueued; its
        // Signal(FIRST_FENCE) will follow them on the same queue.
        backing
            .retire_tile(RETIRED_TILE, FIRST_FENCE)
            .context("retire the first sparse tile")?;
        ensure!(
            !backing.map_tile(REPLACEMENT_TILE)?,
            "reused retired backing before fence {FIRST_FENCE} was submitted"
        );
        queue
            .Signal(&*fence, FIRST_FENCE)
            .context("signal initial fence")?;
        let initial_completed = wait_for_gpu(device, fence, FIRST_FENCE)?;
        verify_readback(readback, initial_tiles, &PAYLOAD[..READBACK_BYTES])?;

        ensure!(
            backing.map_tile(REPLACEMENT_TILE)?,
            "retired backing was not reusable after fence {initial_completed} completed"
        );
        submit_copies(queue, replacement);
        queue
            .Signal(&*fence, SECOND_FENCE)
            .context("signal replacement fence")?;
        let completed = wait_for_gpu(device, fence, SECOND_FENCE)?;
        let finished = ManuallyDrop::into_inner(in_flight);
        let (_, _, _, _, readback, ..) = &finished;
        verify_readback(readback, replacement_tiles, &PAYLOAD[PATTERN_BYTES..])?;
        eprintln!(
            "CAPABILITY sparse_backing status=pass tiled_resources_tier={tier} virtual_bytes={VIRTUAL_BYTES} backing_heap_bytes={BACKING_BYTES} mapped_tiles={PHYSICAL_TILES} verified_bytes={} reuse_blocked_before_signal=true reused_tile={REPLACEMENT_TILE} first_fence_completed={initial_completed} fence_completed={completed}",
            2 * READBACK_BYTES
        );
    }
    Ok(())
}
