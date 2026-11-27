use super::{SubmissionBacking, WritebackTarget, visit_writeback_spans};
use victory_memory::{PAGE_BYTES, physical::PageNumbers};

#[test]
fn fragmented_writes_preserve_a_later_cpu_store_in_a_gap() {
    let before = [0x11; 19];
    let mut gpu = before;
    for at in (0..19).step_by(2) {
        gpu[at] = 0xab;
    }
    let mut cpu = [0x33; 19];
    let mut stores = 0;
    assert!(visit_writeback_spans(&before, &gpu, &[], |span| {
        cpu[span.clone()].copy_from_slice(&gpu[span]);
        stores += 1;
        if stores == 1 {
            cpu[17] = 0; // A title store between the first and last GPU stores.
        }
        true
    }));
    assert_eq!(stores, 10);
    assert_eq!(
        cpu,
        [
            0xab, 0x33, 0xab, 0x33, 0xab, 0x33, 0xab, 0x33, 0xab, 0x33, 0xab, 0x33, 0xab, 0x33,
            0xab, 0x33, 0xab, 0, 0xab,
        ]
    );
}

#[test]
fn observed_and_declared_writes_form_maximal_runs_without_bridging_gaps() {
    let before = [0; 18];
    let mut gpu = before;
    for at in [1, 2, 4, 6, 7, 10, 11, 15] {
        gpu[at] = 1;
    }
    let declared = [0..2, 3..5, 5..6, 11..13, 14..15, 17..18];
    let mut spans = Vec::new();
    assert!(visit_writeback_spans(&before, &gpu, &declared, |span| {
        spans.push(span);
        true
    }));
    assert_eq!(spans, [0..8, 10..13, 14..16, 17..18]);
}

#[test]
fn equal_images_write_only_the_explicitly_owned_bytes() {
    let image = [0; 4];
    assert!(visit_writeback_spans(&image, &image, &[], |_| {
        panic!("nothing owns a store")
    }));
    let mut cpu = [9; 4];
    assert!(visit_writeback_spans(&image, &image, &[1..3], |span| {
        cpu[span.clone()].copy_from_slice(&image[span]);
        true
    }));
    assert_eq!(cpu, [9, 0, 0, 9]);
}

#[test]
fn failed_store_stops_the_visit_even_past_the_old_eight_span_limit() {
    let before = [0; 21];
    let mut gpu = before;
    for at in (0..21).step_by(2) {
        gpu[at] = 1;
    }
    let mut attempts = Vec::new();
    assert!(!visit_writeback_spans(&before, &gpu, &[], |span| {
        let ok = span.start != 16;
        attempts.push(span);
        ok
    }));
    assert_eq!(
        attempts,
        [0..1, 2..3, 4..5, 6..7, 8..9, 10..11, 12..13, 14..15, 16..17]
    );
}

#[test]
fn merge_page_fragmentation_does_not_write_an_unowned_host_page() {
    use crate::watch::tests::{free, pool};
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};

    const PAGE: usize = PAGE_BYTES as usize;
    let base = pool(1);
    let mut protection = 0;
    // SAFETY: this changes only the middle host page of our own committed
    // allocation. It stays readable, making even a same-value write detectable.
    assert_ne!(
        unsafe {
            VirtualProtect(
                (base + PAGE / 2) as *const _,
                1,
                PAGE_READONLY,
                &mut protection,
            )
        },
        0
    );

    let id = PageNumbers::default().allocate(1).unwrap();
    let mut backing = SubmissionBacking {
        test_api: Some((
            |_, _| {},
            |at, _, _, redirect, visit| {
                assert!(redirect);
                visit(WritebackTarget {
                    address: at,
                    original_va: false,
                });
                true
            },
            |_, _, _, _| unreachable!(),
        )),
        ..Default::default()
    };
    backing.pins.remember(base, base + PAGE, Some(id));
    let before = vec![0; PAGE];
    let mut gpu = before.clone();
    for at in (0..18).step_by(2) {
        gpu[at] = 0x77;
    }
    gpu[PAGE - 1] = 0xab;
    // The old whole-page read succeeds, but its write across the protected
    // gap fails. Exact stores reach both ends without needing the gap writable.
    let ok = backing.merge_page(base, id, &before, &gpu, &[]);
    // SAFETY: all bytes, including the protected host page, are still readable.
    let matches = unsafe { std::slice::from_raw_parts(base as *const u8, PAGE) } == &gpu[..];
    // Now the protected byte IS owned. Refuse retirement and don't publish
    // the later span, even though that span would have been writable.
    gpu[PAGE / 2] = 1;
    gpu[PAGE - 1] = 0xcc;
    let retry_ok = backing.merge_page(base, id, &before, &gpu, &[]);
    // SAFETY: the last byte of our allocation is still readable.
    let last = unsafe { ((base + PAGE - 1) as *const u8).read() };
    free(base);
    assert!(ok);
    assert!(matches);
    assert!(!retry_ok);
    assert_eq!(last, 0xab);
}
