//! Title query heaps. The VNME path admits actual direct-queue timestamps, visibility and pipeline counters.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, QUERY_HEAP, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

#[cfg(all(windows, feature = "paged-title"))]
use {
    std::{collections::BTreeMap, sync::{Arc, Mutex, OnceLock}},
    windows::Win32::Graphics::Direct3D12::*,
};

#[cfg(all(windows, feature = "paged-title"))]
pub(crate) struct NativeQueryHeap {
    pub(crate) native: ID3D12QueryHeap,
    pub(crate) count: u32,
    pub(crate) initialized: Mutex<BTreeMap<u32, D3D12_QUERY_TYPE>>,
}

const HEAP_IIDS: [Iid; 4] = [
    guid(
        0x0d9658ae,
        0xed45,
        0x469e,
        [0xa6, 0x1d, 0x97, 0x0e, 0xc5, 0x83, 0xca, 0xb4],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Desc {
    kind: u32,
    count: u32,
    nodes: u32,
}

#[repr(C)]
struct Heap {
    object: Object,
    desc: Desc,
    #[cfg(all(windows, feature = "paged-title"))]
    native: OnceLock<Option<Arc<NativeQueryHeap>>>,
}

#[cfg(all(windows, feature = "paged-title"))]
pub(crate) fn result_bytes(kind: D3D12_QUERY_TYPE) -> Option<u64> {
    const _: () = assert!(size_of::<D3D12_QUERY_DATA_PIPELINE_STATISTICS>() == 88);
    match kind {
        D3D12_QUERY_TYPE_TIMESTAMP | D3D12_QUERY_TYPE_OCCLUSION | D3D12_QUERY_TYPE_BINARY_OCCLUSION => Some(8),
        D3D12_QUERY_TYPE_PIPELINE_STATISTICS => Some(size_of::<D3D12_QUERY_DATA_PIPELINE_STATISTICS>() as u64),
        _ => None,
    }
}

#[cfg(all(windows, feature = "paged-title"))]
pub(crate) fn native_heap(this: This, kind: u32) -> Option<(Arc<NativeQueryHeap>, D3D12_QUERY_TYPE)> {
    if this.is_null() { return None; }
    let query_type = D3D12_QUERY_TYPE(i32::try_from(kind).ok()?);
    let heap_type = match query_type {
        D3D12_QUERY_TYPE_TIMESTAMP => D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
        D3D12_QUERY_TYPE_OCCLUSION | D3D12_QUERY_TYPE_BINARY_OCCLUSION => D3D12_QUERY_HEAP_TYPE_OCCLUSION,
        D3D12_QUERY_TYPE_PIPELINE_STATISTICS => D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS,
        _ => return None,
    };
    // SAFETY: the query API receives a retained title query heap.
    let heap = unsafe { &*this.cast::<Heap>() };
    if heap.desc.kind != heap_type.0 as u32
        || !matches!(heap.desc.nodes, 0 | 1) || !(1..=65536).contains(&heap.desc.count)
    {
        return None;
    }
    heap.native.get_or_init(|| {
        let host = crate::host::host()?;
        let mut native = None;
        // SAFETY: the native description names this host's direct node.
        unsafe { host.device.CreateQueryHeap(&D3D12_QUERY_HEAP_DESC {
            Type: heap_type, Count: heap.desc.count, NodeMask: 0,
        }, &mut native) }.ok()?;
        Some(Arc::new(NativeQueryHeap {
            native: native?, count: heap.desc.count, initialized: Mutex::default(),
        }))
    }).clone().map(|heap| (heap, query_type))
}

/// Device method.
pub(crate) extern "system" fn create(
    _: This,
    desc: *const Desc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "query heap");
    let heap = Box::new(Heap {
        object: Object::new(&FUNCTIONS),
        desc,
        #[cfg(all(windows, feature = "paged-title"))]
        native: OnceLock::new(),
    });
    object::hand_out(QUERY_HEAP, heap, iid, &HEAP_IIDS, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        QUERY_HEAP,
        iid,
        object::is(iid, &HEAP_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}

/// IUnknown 3, object 4, device child 1.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<QUERY_HEAP, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{E_NOINTERFACE, E_POINTER, S_FALSE, S_OK, slot};

    #[test]
    fn null_descriptions_are_rejected() {
        let mut heap = std::ptr::null();
        assert_eq!(
            create(std::ptr::null(), std::ptr::null(), &HEAP_IIDS[0], &mut heap),
            E_INVALIDARG
        );
        assert!(heap.is_null());
        // The description is checked first: a probe with no room for the
        // heap fails on it rather than answering S_FALSE.
        assert_eq!(
            create(
                std::ptr::null(),
                std::ptr::null(),
                &HEAP_IIDS[0],
                std::ptr::null_mut()
            ),
            E_INVALIDARG
        );
    }

    #[test]
    fn creation_without_an_out_pointer_only_probes() {
        let desc = Desc {
            kind: 1, // timestamps
            count: 64,
            nodes: 0,
        };
        // A known interface with no out pointer asks whether creation would
        // succeed; the heap is dropped instead of handed out.
        assert_eq!(
            create(std::ptr::null(), &desc, &HEAP_IIDS[0], std::ptr::null_mut()),
            S_FALSE
        );
        // A null interface id is tolerated: no interface, no heap.
        let mut heap = std::ptr::null();
        assert_eq!(
            create(std::ptr::null(), &desc, std::ptr::null(), &mut heap),
            E_NOINTERFACE
        );
        assert!(heap.is_null());
    }

    #[test]
    fn heaps_answer_their_interfaces_and_count_references() {
        let desc = Desc {
            kind: 0, // occlusion
            count: 8,
            nodes: 0,
        };
        let null = std::ptr::null();
        let mut heap = null;
        assert_eq!(create(null, &desc, &HEAP_IIDS[0], &mut heap), S_OK);
        assert!(!heap.is_null());
        let query = slot::<extern "system" fn(This, *const Iid, *mut This) -> HRESULT>(heap, 0x0);
        // Every interface the heap answers to hands out the same object.
        for iid in &HEAP_IIDS {
            let mut out = null;
            assert_eq!(query(heap, iid, &mut out), S_OK);
            assert_eq!(out as usize, heap as usize);
        }
        // An unknown interface writes null behind and reports it missing.
        let unknown = [0u8; 16];
        let mut out = heap;
        assert_eq!(query(heap, &unknown, &mut out), E_NOINTERFACE);
        assert!(out.is_null());
        // A null interface id is tolerated rather than matched.
        let mut out = heap;
        assert_eq!(query(heap, std::ptr::null(), &mut out), E_NOINTERFACE);
        assert!(out.is_null());
        // No room for the answer is a pointer error.
        assert_eq!(query(heap, &unknown, std::ptr::null_mut()), E_POINTER);
        // References: one from creation, four from the queries, one added;
        // each release hands back what is left, the last frees the heap.
        assert_eq!(slot::<extern "system" fn(This) -> u32>(heap, 0x8)(heap), 6);
        let release = slot::<extern "system" fn(This) -> u32>(heap, 0x10);
        for left in [5, 4, 3, 2, 1, 0] {
            assert_eq!(release(heap), left);
        }
    }
}
