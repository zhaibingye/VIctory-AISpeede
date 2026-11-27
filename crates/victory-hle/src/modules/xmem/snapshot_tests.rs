use super::*;
use victory_memory::physical::access::try_snapshot;

#[test]
fn physical_copy_claims_discontiguous_runs_together_before_writing() {
    use std::time::{Duration, Instant};

    let mut pages = [0usize; 2];
    let mut n = pages.len();
    assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
    assert_eq!(n, 2);
    let slot = XMemVirtualAlloc(
        std::ptr::null_mut(),
        2 * PAGE,
        MEM_RESERVE | PAGES_64K,
        0,
        PAGE_READWRITE,
    );
    assert!(!slot.is_null());
    assert_eq!(XMemMapPhysicalPages(slot, 2, pages.as_ptr()), slot);
    let va = slot as usize;
    let mut pins = BackingPins::default();
    pin_backing(&[(va, va + 2 * PAGE)], &mut pins);
    let second_run = pins.expected(va).unwrap();
    let first_run = pins.expected(va + PAGE).unwrap();
    let snapshot = try_snapshot([second_run].into_iter()).unwrap();
    // In reverse physical order, so the fallback copy must map two separate runs.
    let worker =
        std::thread::spawn(move || write_physical(&[pages[1], pages[0]], PAGE - 1, &[3, 7]));
    let deadline = Instant::now() + Duration::from_secs(5);
    let both_claimed = loop {
        if try_snapshot([first_run].into_iter()).is_err() {
            break true;
        }
        if worker.is_finished() || Instant::now() >= deadline {
            break false;
        }
        std::thread::sleep(Duration::from_millis(1));
    };
    drop(snapshot);
    assert!(worker.join().unwrap());
    assert!(
        both_claimed,
        "a conflict in the second run must keep the first run claimed too"
    );
    // SAFETY: both pages are still mapped; the sole writer has joined.
    assert_eq!(
        unsafe {
            [
                slot.cast::<u8>().read(),
                slot.cast::<u8>().add(2 * PAGE - 1).read(),
            ]
        },
        [7, 3]
    );
    assert_eq!(XMemMapPhysicalPages(slot, 2, std::ptr::null()), slot);
    assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    assert_ne!(private_virtual_free(slot, 0, MEM_RELEASE), 0);
}

#[test]
fn raw_va_write_claims_only_the_touched_pages_of_the_current_section() {
    const N: usize = 4;
    let allocate_section = || {
        let mut numbers = [0usize; N];
        let mut n = N;
        assert_eq!(
            XMemAllocatePhysicalPages(0, &mut n, numbers.as_mut_ptr()),
            1
        );
        assert_eq!(n, N);
        // Resolve test identities independently of the VA that we'll rebind.
        let physical = physical();
        let pages = numbers.map(|number| physical.pages[&number].page(0));
        (numbers, pages)
    };
    let (original, a) = allocate_section();
    let (replacement, b) = allocate_section();
    let slot = XMemVirtualAlloc(
        std::ptr::null_mut(),
        N * PAGE,
        MEM_RESERVE | PAGES_64K,
        0,
        PAGE_READWRITE,
    );
    assert!(!slot.is_null());
    let dest = slot.wrapping_byte_add(2 * PAGE - 1).cast::<[u8; 2]>();
    let fill = |numbers: &[usize; N],
                current: [PhysicalPage; N],
                other: [PhysicalPage; N],
                before: [u8; 2],
                after: [u8; 2]| {
        assert_eq!(XMemMapPhysicalPages(slot, N, numbers.as_ptr()), slot);
        let guard = raw_va_write(dest as usize, after.len()).expect("nonempty fill");
        // Probe the exclusive side used by both managed remap entry points. No
        // sleep, and no recursive lock acquisition on the fill thread.
        assert!(std::thread::spawn(|| matches!(
            TITLE_VIEW_GATE.try_write(), Err(std::sync::TryLockError::WouldBlock)
        )).join().unwrap());
        // Check separately: a batch conflict alone wouldn't prove *both* pages are claimed.
        for page in [current[1], current[2]] {
            assert_eq!(try_snapshot(std::iter::once(page)).unwrap_err(), page);
        }
        let unrelated = try_snapshot([current[0], current[3]].into_iter().chain(other))
            .expect("untouched pages and the other section must stay available");
        // SAFETY: the two bytes straddle committed pages in our live view. This test
        // owns both sections, and the snapshot covers only pages the fill does not touch.
        unsafe {
            assert_eq!(dest.read(), before);
            dest.write(after);
        }
        drop(unrelated);
        drop(guard);
        drop(try_snapshot(current.into_iter()).expect("completed fill released its pages"));
    };
    let marker = [0x53, 0xa7];
    fill(&original, a, b, [0; 2], marker);
    fill(&replacement, b, a, [0; 2], [0x19, 0x37]);
    fill(&original, a, b, marker, [0; 2]);

    assert_eq!(XMemMapPhysicalPages(slot, N, std::ptr::null()), slot);
    assert_eq!(XMemFreePhysicalPages(N, replacement.as_ptr()), 1);
    assert_eq!(XMemFreePhysicalPages(N, original.as_ptr()), 1);
    assert_ne!(private_virtual_free(slot, 0, MEM_RELEASE), 0);
}

#[test]
fn page_window_blocks_snapshots_of_both_pinned_pages_but_not_a_rebound_va() {
    const OFFSET: usize = PAGE - 1;
    const MARKER: [u8; 2] = [0x53, 0xa7];

    let allocate_pair = || {
        let mut pages = [0usize; 2];
        let mut n = pages.len();
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        assert_eq!(n, pages.len());
        pages
    };
    let slot = XMemVirtualAlloc(
        std::ptr::null_mut(),
        2 * PAGE,
        MEM_RESERVE | PAGES_64K,
        0,
        PAGE_READWRITE,
    );
    assert!(!slot.is_null());
    let start = slot as usize + OFFSET;
    let end = start + MARKER.len();
    let pin_pair = || {
        let mut pins = BackingPins::default();
        pin_backing(&[(start, end)], &mut pins);
        let pages = [start, end - 1].map(|va| pins.expected(va).expect("mapped XMem page"));
        (pins, pages)
    };

    let original = allocate_pair();
    assert_eq!(XMemMapPhysicalPages(slot, 2, original.as_ptr()), slot);
    let (pins, pages) = pin_pair();
    assert_eq!(pages.map(PhysicalPage::number), original);
    let mut window = page_window(&original, OFFSET, MARKER.len()).expect("cross-page window");

    let replacement = allocate_pair();
    assert_eq!(XMemMapPhysicalPages(slot, 2, replacement.as_ptr()), slot);
    assert_eq!(XMemFreePhysicalPages(2, original.as_ptr()), 1);
    let (_replacement_pins, current) = pin_pair();
    assert_eq!(current.map(PhysicalPage::number), replacement);

    // The worker still owns both original pages even after every guest handle and alias is gone.
    for page in pages {
        assert_eq!(try_snapshot(std::iter::once(page)).err(), Some(page));
    }
    let rebound = try_snapshot(current.into_iter()).expect("replacement pages have no writer");
    window.bytes().copy_from_slice(&MARKER);
    // SAFETY: the slot maps our fresh replacement; the worker writes through its original view.
    let untouched = unsafe { slot.wrapping_byte_add(OFFSET).cast::<[u8; 2]>().read() };
    assert_eq!(untouched, [0; 2]);
    drop(rebound);
    drop(window);

    let snapshot = try_snapshot(pages.into_iter()).expect("closed window released both pages");
    let mut bytes = Vec::new();
    read_backing_runs(start, end, &pins, &mut |at, source, len, same| {
        assert!(!same, "must read the pinned original");
        assert_eq!(at, start + bytes.len());
        let source = source.expect("original section view") as *const u8;
        // SAFETY: the callback keeps this view alive; our worker window is closed.
        bytes.extend_from_slice(unsafe { std::slice::from_raw_parts(source, len) });
    });
    assert_eq!(bytes, MARKER);
    drop(snapshot);

    assert_eq!(XMemMapPhysicalPages(slot, 2, std::ptr::null()), slot);
    assert_eq!(XMemFreePhysicalPages(2, replacement.as_ptr()), 1);
    assert_ne!(private_virtual_free(slot, 0, MEM_RELEASE), 0);
}
