#![cfg(windows)]

use victory_memory::{
    PAGE_BYTES,
    budget::Pool,
    d3d12::{Error, PAGE_LEN, Pages, Transfers},
    engine::Error as PolicyError,
};
use windows::Win32::Graphics::{
    Direct3D::D3D_FEATURE_LEVEL_11_0,
    Direct3D12::{D3D12CreateDevice, ID3D12Device},
    Dxgi::{CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter, IDXGIFactory4},
};

fn device() -> ID3D12Device {
    // SAFETY: the factory and WARP adapter are valid and the output slot is writable.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter().unwrap();
        let mut device = None;
        D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut device).unwrap();
        device.unwrap()
    }
}

fn pattern(seed: u8) -> Box<[u8; PAGE_LEN]> {
    (0..PAGE_LEN)
        .map(|i| (i ^ (i >> 7)) as u8 ^ seed)
        .collect::<Vec<u8>>()
        .into_boxed_slice()
        .try_into()
        .unwrap()
}

#[test]
fn round_trip_preserves_bytes_across_dirty_eviction_and_readmission() {
    let mut pages = Pages::new(device(), 2 * PAGE_BYTES, 0).unwrap();
    let a = pages.commit(0).unwrap();
    let b = pages.commit(PAGE_BYTES).unwrap();
    let c = pages.commit(2 * PAGE_BYTES).unwrap();
    assert_eq!(pages.used(Pool::Ram), 5 * PAGE_BYTES); // Three backing and two staging pages.
    assert_eq!(pages.used(Pool::Local), 0);

    let input = pattern(0x35);
    pages.write(a, &input).unwrap();
    pages.write(b, &pattern(0xa1)).unwrap();
    pages.copy_page(a, b).unwrap();
    pages.copy_page(a, b).unwrap();
    assert_eq!(pages.used(Pool::Local), 2 * PAGE_BYTES);
    assert_eq!(pages.transfers(), Transfers { uploaded_pages: 1, downloaded_pages: 0 });

    // B is GPU-authoritative and is the next source, so A has to go.
    // It is clean: eviction needs no download or duplicate upload of B.
    pages.copy_page(b, c).unwrap();
    assert_eq!(pages.used(Pool::Local), 2 * PAGE_BYTES);
    assert_eq!(pages.transfers(), Transfers { uploaded_pages: 1, downloaded_pages: 0 });

    // Shrinking to zero writes back both dirty pages before releasing the
    // physical buffers. Neither read below should request a new transfer.
    pages.set_local_budget(0, 0).unwrap();
    assert_eq!(pages.used(Pool::Local), 0);
    assert_eq!(pages.transfers(), Transfers { uploaded_pages: 1, downloaded_pages: 2 });
    assert_eq!(pages.read(b).unwrap(), input.as_ref());
    assert_eq!(pages.read(c).unwrap(), input.as_ref());
    assert!(matches!(pages.copy_page(b, c), Err(Error::Policy(PolicyError::Space(_)))));
    assert_eq!(pages.used(Pool::Local), 0);
    assert_eq!(pages.transfers().downloaded_pages, 2);

    pages.set_local_budget(2 * PAGE_BYTES, 0).unwrap();
    pages.copy_page(b, c).unwrap();
    assert_eq!(pages.read(c).unwrap(), input.as_ref());
    assert_eq!(pages.transfers(), Transfers { uploaded_pages: 2, downloaded_pages: 3 });

    let updated = pattern(0xf7);
    pages.write(b, &updated).unwrap();
    pages.copy_page(b, c).unwrap();
    assert_eq!(pages.read(c).unwrap(), updated.as_ref());
    assert_eq!(pages.transfers(), Transfers { uploaded_pages: 3, downloaded_pages: 4 });
}

#[test]
fn admission_evicts_a_dirty_non_operand_and_preserves_its_bytes() {
    let mut pages = Pages::new(device(), 2 * PAGE_BYTES, 0).unwrap();
    let a = pages.commit(0).unwrap();
    let b = pages.commit(PAGE_BYTES).unwrap();
    let c = pages.commit(2 * PAGE_BYTES).unwrap();
    let input = pattern(0x17);
    pages.write(a, &input).unwrap();
    pages.copy_page(a, b).unwrap();
    // Protect A as the source. B has newer GPU contents and has to be
    // downloaded before C can take its local allocation.
    pages.copy_page(a, c).unwrap();
    assert_eq!(pages.used(Pool::Local), 2 * PAGE_BYTES);
    assert_eq!(pages.transfers(), Transfers { uploaded_pages: 1, downloaded_pages: 1 });
    assert_eq!(pages.read(b).unwrap(), input.as_ref());
    assert_eq!(pages.transfers().downloaded_pages, 1);
    assert_eq!(pages.read(c).unwrap(), input.as_ref());
}

#[test]
fn slab_capacity_is_bounded_and_reused_until_the_last_slot_leaves() {
    let mut pages = Pages::new(device(), 4 * PAGE_BYTES, 0).unwrap();
    let a = pages.commit(0).unwrap();
    let b = pages.commit(PAGE_BYTES).unwrap();
    let c = pages.commit(2 * PAGE_BYTES).unwrap();
    let input = pattern(0x32);
    pages.write(a, &input).unwrap();
    pages.copy_page(a, b).unwrap();
    assert_eq!(pages.local_heap_bytes(), 4 * PAGE_BYTES);
    assert_eq!(pages.used(Pool::Local), 2 * PAGE_BYTES);

    pages.evict(b).unwrap(); // A still retains the underlying native heap.
    assert_eq!(pages.local_heap_bytes(), 4 * PAGE_BYTES);
    pages.copy_page(a, c).unwrap();
    assert_eq!(pages.local_heap_bytes(), 4 * PAGE_BYTES);
    assert_eq!(pages.read(c).unwrap(), input.as_ref());
    // The live pages fit in a nominal two-page cap, but the actual four-page
    // slab does not. Reclaim before admitting any more buffers.
    pages.set_local_budget(2 * PAGE_BYTES, 0).unwrap();
    assert_eq!(pages.local_heap_bytes(), 0);
    assert_eq!(pages.read(b).unwrap(), input.as_ref());
    pages.copy_page(a, c).unwrap();
    assert_eq!(pages.local_heap_bytes(), 2 * PAGE_BYTES);
    assert_eq!(pages.read(c).unwrap(), input.as_ref());
}
