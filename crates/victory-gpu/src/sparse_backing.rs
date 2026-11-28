//! Fixed-capacity backing for a native D3D12 reserved buffer.

use std::collections::HashMap;
use std::ptr::null_mut;

use anyhow::{Context, Result, ensure};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::{IUnknown, Interface};

/// Bytes in one D3D12 tile (64 KiB).
pub const TILE_BYTES: u32 = D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;

#[derive(Clone, Copy)]
struct Mapping {
    heap_tile: u32,
    last_use: Option<u64>,
}

/// A reserved DEFAULT buffer and its physical tile heap on one native Direct queue.
///
/// Submit uses only on that queue, after mapping the tiles they access. Do not
/// change the buffer's mappings externally. This owner does not wait on drop:
/// keep it alive through completion of all uses and queued mapping updates.
pub struct SparseBacking {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    buffer: ID3D12Resource,
    heap: ID3D12Heap,
    virtual_tiles: u32,
    physical_tiles: u32,
    next_unused: u32,
    // One entry per assigned physical tile, including those awaiting retirement.
    mappings: HashMap<u32, Mapping>,
}

impl SparseBacking {
    /// Reserve virtual space in COMMON state and allocate only `physical_tiles`
    /// of backing. The native objects must come from the same single-node device.
    /// Use `fence` as a monotonically increasing timeline signaled only by `queue`.
    pub fn new(
        device: &ID3D12Device,
        queue: &ID3D12CommandQueue,
        fence: &ID3D12Fence,
        virtual_tiles: u32,
        physical_tiles: u32,
    ) -> Result<Self> {
        ensure!(
            virtual_tiles > 0,
            "sparse buffer needs at least one virtual tile"
        );
        ensure!(
            physical_tiles > 0,
            "sparse heap needs at least one physical tile"
        );

        // SAFETY: these are live D3D12 interfaces and all outputs are local.
        // The buffer description is tile-aligned and node masks are for one node.
        unsafe {
            device
                .GetDeviceRemovedReason()
                .context("sparse backing device lost")?;
            ensure!(
                device.GetNodeCount() == 1,
                "sparse backing requires a single D3D12 node"
            );
            ensure!(
                queue.GetDesc().Type == D3D12_COMMAND_LIST_TYPE_DIRECT,
                "sparse backing requires a Direct queue"
            );
            let mut queue_device: Option<ID3D12Device> = None;
            queue
                .GetDevice(&mut queue_device)
                .context("get sparse queue device")?;
            let mut fence_device: Option<ID3D12Device> = None;
            fence
                .GetDevice(&mut fence_device)
                .context("get sparse fence device")?;
            let identity = device.cast::<IUnknown>()?;
            for child_device in [queue_device, fence_device] {
                let child_device = child_device.context("GetDevice returned no device")?;
                ensure!(
                    child_device.cast::<IUnknown>()?.as_raw() == identity.as_raw(),
                    "sparse device, queue, and fence must share a D3D12 device"
                );
            }
            let mut buffer = None;
            device
                .CreateReservedResource(
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: u64::from(virtual_tiles) * u64::from(TILE_BYTES),
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .with_context(|| {
                    format!("CreateReservedResource for {virtual_tiles} virtual tiles")
                })?;
            let buffer: ID3D12Resource =
                buffer.context("CreateReservedResource returned no buffer")?;
            let mut tiles = 0;
            device.GetResourceTiling(&buffer, Some(&mut tiles), None, None, None, 0, null_mut());
            device
                .GetDeviceRemovedReason()
                .context("device lost after GetResourceTiling")?;
            ensure!(
                tiles == virtual_tiles,
                "sparse buffer reported {tiles} tiles; expected {virtual_tiles}"
            );

            let mut heap = None;
            device
                .CreateHeap(
                    &D3D12_HEAP_DESC {
                        SizeInBytes: u64::from(physical_tiles) * u64::from(TILE_BYTES),
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
                .with_context(|| format!("CreateHeap for {physical_tiles} physical tiles"))?;
            let backing = Self {
                device: device.clone(),
                queue: queue.clone(),
                fence: fence.clone(),
                buffer,
                heap: heap.context("CreateHeap returned no heap")?,
                virtual_tiles,
                physical_tiles,
                next_unused: 0,
                mappings: HashMap::new(),
            };
            backing.completed()?;
            Ok(backing)
        }
    }

    /// The reserved resource. Its size does not describe physical backing.
    pub fn buffer(&self) -> &ID3D12Resource {
        &self.buffer
    }

    /// Ensure an active mapping, ordered before later work on the same queue.
    ///
    /// Returns `false` if the pool is full and no retirement is complete. A
    /// request for a tile whose own retirement is pending is an error. A new
    /// mapping does not initialize storage; write it before reading it.
    ///
    /// # Safety
    /// Keep this owner alive until the Direct queue completes the mappings
    /// and all work using them. Use the buffer only on that queue; no other
    /// code may change its mappings.
    pub unsafe fn map_tile(&mut self, virtual_tile: u32) -> Result<bool> {
        self.check_bounds(virtual_tile)?;
        let completed = self.completed()?;
        let reclaim = match self.mappings.get(&virtual_tile).copied() {
            Some(Mapping { last_use: None, .. }) => return Ok(true),
            Some(Mapping {
                heap_tile,
                last_use: Some(last_use),
            }) => {
                ensure!(
                    completed >= last_use,
                    "sparse tile {virtual_tile} is still retiring at fence value {last_use}"
                );
                Some((virtual_tile, heap_tile))
            }
            None if self.next_unused < self.physical_tiles => None,
            None => {
                let ready = self.mappings.iter().find_map(|(&tile, mapping)| {
                    mapping
                        .last_use
                        .filter(|&value| value <= completed)
                        .map(|_| (tile, mapping.heap_tile))
                });
                let Some(ready) = ready else {
                    return Ok(false);
                };
                Some(ready)
            }
        };

        let heap_tile = if let Some((old_tile, heap_tile)) = reclaim {
            // The real queue fence covered every use of the retired mapping.
            // NULL it before assigning the same slot; never overwrite in flight.
            self.update_mapping(old_tile, None)?;
            heap_tile
        } else {
            self.next_unused
        };
        self.update_mapping(virtual_tile, Some(heap_tile))?;
        if let Some((old_tile, _)) = reclaim {
            self.mappings.remove(&old_tile);
        } else {
            self.next_unused += 1;
        }
        self.mappings.insert(
            virtual_tile,
            Mapping {
                heap_tile,
                last_use: None,
            },
        );
        Ok(true)
    }

    /// Mark an active tile for reuse after `last_use_fence`; does not unmap it yet.
    ///
    /// # Safety
    /// A successful signal of `last_use_fence` on this backing's Direct queue and
    /// fence must be ordered after every GPU use of the old mapping. Completion
    /// must imply those uses have finished: use fresh, increasing values and
    /// never signal from the CPU or another queue. Do not bind or otherwise
    /// access the old mapping after retirement; a later successful `map_tile`
    /// starts a new mapping. Keep the owner alive until all queued updates and
    /// GPU uses have finished.
    pub unsafe fn retire_tile(&mut self, virtual_tile: u32, last_use_fence: u64) -> Result<()> {
        self.check_bounds(virtual_tile)?;
        ensure!(
            last_use_fence != 0 && last_use_fence != u64::MAX,
            "sparse retirement requires a nonzero fence value, not the lost-device sentinel"
        );
        self.completed()?;
        let mapping = self
            .mappings
            .get_mut(&virtual_tile)
            .context("sparse tile is not mapped")?;
        ensure!(
            mapping.last_use.is_none(),
            "sparse tile {virtual_tile} is already retiring"
        );
        mapping.last_use = Some(last_use_fence);
        Ok(())
    }

    fn check_bounds(&self, virtual_tile: u32) -> Result<()> {
        ensure!(
            virtual_tile < self.virtual_tiles,
            "sparse tile {virtual_tile} is out of bounds"
        );
        Ok(())
    }

    fn completed(&self) -> Result<u64> {
        // SAFETY: both interfaces are live and from the same device.
        let completed = unsafe {
            let completed = self.fence.GetCompletedValue();
            self.device
                .GetDeviceRemovedReason()
                .context("sparse backing device lost")?;
            completed
        };
        ensure!(
            completed != u64::MAX,
            "sparse backing fence reports device loss"
        );
        Ok(completed)
    }

    fn update_mapping(&self, virtual_tile: u32, heap_tile: Option<u32>) -> Result<()> {
        let at = D3D12_TILED_RESOURCE_COORDINATE {
            X: virtual_tile,
            ..Default::default()
        };
        let size = D3D12_TILE_REGION_SIZE {
            NumTiles: 1,
            ..Default::default()
        };
        let (heap, flag) = match heap_tile {
            Some(_) => (Some(&self.heap), D3D12_TILE_RANGE_FLAG_NONE),
            None => (None, D3D12_TILE_RANGE_FLAG_NULL),
        };
        let count = 1;
        // SAFETY: one in-bounds tile in the buffer, mapped to one in-bounds tile
        // in the same-device heap or NULL. All argument pointers last the call.
        unsafe {
            self.queue.UpdateTileMappings(
                &self.buffer,
                1,
                Some(&raw const at),
                Some(&raw const size),
                heap,
                1,
                Some(&raw const flag),
                heap_tile.as_ref().map(|offset| offset as *const u32),
                Some(&raw const count),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
            // The update itself is void; preserve the native removal HRESULT.
            self.device
                .GetDeviceRemovedReason()
                .context("device lost after UpdateTileMappings")
        }
    }
}
