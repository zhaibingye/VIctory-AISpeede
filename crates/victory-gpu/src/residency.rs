//! Keep dedicated memory for GPU-facing resources without giving the host
//! permission to fill the entire adapter. Overflow has the same D3D12
//! resource shape, on non-local system memory, not a dummy resource.
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Foundation::E_OUTOFMEMORY;
use windows::core::{GUID, IUnknown, Interface, Result};

const MIB: u64 = 1 << 20;
pub(crate) const LOCAL_LIMIT: u64 = 5 << 30;
// Swap chain, descriptors, driver/runtime and imported resources are not in
// the managed ledger. They still count against DXGI's usage.
const LOCAL_RESERVE: u64 = 384 * MIB;
static LOCAL: AtomicU64 = AtomicU64::new(0);
static FALLBACK_BYTES: AtomicU64 = AtomicU64::new(0);

// A COM object keeps the reservation even if a descriptor or recorded
// list outlives the Rust object that first owned the D3D12 resource.
#[windows::core::implement()]
struct Charge {
    bytes: u64,
}

impl Drop for Charge {
    fn drop(&mut self) {
        LOCAL.fetch_sub(self.bytes, Relaxed);
    }
}

fn fits(existing: u64, observed: u64, budget: u64, bytes: u64) -> bool {
    let ceiling = LOCAL_LIMIT.min(budget).saturating_sub(LOCAL_RESERVE);
    bytes != 0 && existing.max(observed).checked_add(bytes).is_some_and(|n| n <= ceiling)
}

pub(crate) fn counters() -> (u64, u64) {
    (LOCAL.load(Relaxed), FALLBACK_BYTES.load(Relaxed))
}

impl crate::host::Host {
    pub(crate) fn managed_gpu<T: Interface>(
        &self,
        bytes: u64,
        mut make: impl FnMut(&D3D12_HEAP_PROPERTIES) -> Result<T>,
    ) -> Result<T> {
        let default_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        // SAFETY: a plain query on the live device. Only discrete/NUMA
        // adapters have a separate dedicated memory pool.
        let props = unsafe { self.device.GetCustomHeapProperties(0, D3D12_HEAP_TYPE_DEFAULT) };
        if props.MemoryPoolPreference != D3D12_MEMORY_POOL_L1 {
            return make(&default_heap);
        }
        let (observed, budget) = crate::working_set::local_sample()
            .unwrap_or((0, LOCAL_LIMIT));
        if LOCAL.fetch_update(Relaxed, Relaxed, |used| {
            fits(used, observed, budget, bytes).then(|| used + bytes)
        }).is_ok() {
            let token: IUnknown = Charge { bytes }.into();
            match make(&default_heap) {
                Ok(object) => {
                    let d3d = object.cast::<ID3D12Object>()?;
                    const KEY: GUID = GUID::from_u128(0x69c7c68e_0483_4a04_aef3_424ab673167c);
                    // SAFETY: both COM objects are live, the resource adds
                    // a reference to the token.
                    unsafe { d3d.SetPrivateDataInterface(&KEY, &token)? };
                    return Ok(object);
                }
                Err(e) if e.code() != E_OUTOFMEMORY => return Err(e),
                Err(_) => {} // release the local reservation before L0
            }
        }
        let system_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
        };
        let resource = make(&system_heap)?;
        FALLBACK_BYTES.fetch_add(bytes, Relaxed);
        Ok(resource)
    }
}

#[test]
fn local_admission_leaves_room_and_can_use_a_smaller_dxgi_budget() {
    assert!(fits(4 << 30, 0, 6 << 30, 512 * MIB));
    assert!(!fits(4 << 30, 0, 6 << 30, 1024 * MIB));
    assert!(!fits(2 << 30, 3 << 30, 4 << 30, 768 * MIB));
    assert!(!fits(0, 0, 128 * MIB, MIB));
    assert!(!fits(0, 0, u64::MAX, u64::MAX));
}
