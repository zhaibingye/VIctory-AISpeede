//! Account committed graphics allocations for the entire COM object lifetime.
//! On discrete GPUs a non-local allocation spends the same RAM budget as XMem.

use victory_memory::budget::Pool;
use victory_memory::physical::{HostBudget, Reservation};
use windows::Win32::Foundation::{E_INVALIDARG, E_OUTOFMEMORY};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{GUID, IUnknown, Interface, Result};

pub(crate) use victory_memory::budget::LOCAL_LIMIT;
const LOCAL_RESERVE: u64 = 384 << 20;

// Attached to the actual D3D resource/heap, including when the title drops
// its object before the last descriptor or command-list reference disappears.
#[windows::core::implement()]
struct Charge {
    _reservation: Reservation,
}

fn allocate_charged<T: Interface>(lease: Reservation, make: impl FnOnce() -> Result<T>) -> Result<T> {
    let object = make()?;
    let d3d = object.cast::<ID3D12Object>()?;
    const KEY: GUID = GUID::from_u128(0x69c7c68e_0483_4a04_aef3_424ab673167c);
    let token: IUnknown = Charge { _reservation: lease }.into();
    // SAFETY: both objects are live. D3D retains the token on success.
    unsafe { d3d.SetPrivateDataInterface(&KEY, &token)? };
    Ok(object)
}

fn valid_size(bytes: u64) -> Result<()> {
    // D3D12 returns UINT64_MAX for an invalid allocation description.
    if bytes == 0 || bytes == u64::MAX {
        return Err(windows::core::Error::from(E_INVALIDARG));
    }
    Ok(())
}

pub(crate) fn counters() -> (u64, u64) {
    let budget = HostBudget::process();
    (budget.used(Pool::Local), budget.used(Pool::Ram))
}

impl crate::host::Host {
    pub(crate) fn managed_system<T: Interface>(
        &self, bytes: u64, make: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        valid_size(bytes)?;
        let lease = HostBudget::process().reserve_ram(bytes)
            .map_err(|_| windows::core::Error::from(E_OUTOFMEMORY))?;
        allocate_charged(lease, make)
    }

    pub(crate) fn managed_gpu<T: Interface>(
        &self,
        bytes: u64,
        mut make: impl FnMut(&D3D12_HEAP_PROPERTIES) -> Result<T>,
    ) -> Result<T> {
        valid_size(bytes)?;
        let default_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        // SAFETY: a plain query on the live device. UMA default heaps are
        // physically RAM; they get no independent local allowance.
        let props = unsafe { self.device.GetCustomHeapProperties(0, D3D12_HEAP_TYPE_DEFAULT) };
        if props.MemoryPoolPreference != D3D12_MEMORY_POOL_L1 {
            return self.managed_system(bytes, || make(&default_heap));
        }
        if let Some((observed, budget)) = self.local_memory_for_admission()
            && let Ok(lease) = HostBudget::process().reserve_local(bytes, observed, budget, LOCAL_RESERVE)
        {
            match allocate_charged(lease, || make(&default_heap)) {
                Ok(object) => return Ok(object),
                Err(e) if e.code() != E_OUTOFMEMORY => return Err(e),
                Err(_) => {} // Release the local reservation before trying RAM.
            }
        }
        let system_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
        };
        self.managed_system(bytes, || make(&system_heap))
    }
}

#[cfg(test)]
#[test]
fn residency_com_charge_stays_until_last_graphics_reference() {
    use windows::Win32::Graphics::{
        Direct3D::D3D_FEATURE_LEVEL_11_0,
        Dxgi::{CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter, IDXGIFactory4},
    };
    let budget = HostBudget::default();
    let size = victory_memory::PAGE_BYTES;
    let bytes = budget.reserve_ram(size).unwrap();
    // SAFETY: initialized factory flag, WARP adapter, and typed output slot.
    let device: ID3D12Device = unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter().unwrap();
        let mut device = None;
        D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut device).unwrap();
        device.unwrap()
    };
    let resource: ID3D12Heap = allocate_charged(bytes, || {
        let mut heap = None;
        // SAFETY: live software device, initialized 64 KiB heap description and output.
        unsafe {
            device.CreateHeap(&D3D12_HEAP_DESC {
                SizeInBytes: size,
                Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
                Properties: D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    CreationNodeMask: 1,
                    VisibleNodeMask: 1,
                    ..Default::default()
                },
                ..Default::default()
            }, &mut heap)?;
        }
        Ok(heap.unwrap())
    }).unwrap();
    let another_reference = resource.cast::<IUnknown>().unwrap();
    drop(resource);
    assert_eq!(budget.used(Pool::Ram), size);
    drop(another_reference);
    assert_eq!(budget.used(Pool::Ram), 0);

    let lease = budget.reserve_ram(size).unwrap();
    assert!(allocate_charged::<ID3D12Heap>(lease, || Err(E_OUTOFMEMORY.into())).is_err());
    assert_eq!(budget.used(Pool::Ram), 0);
}
