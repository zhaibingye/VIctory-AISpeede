use super::{GuestCopy, Unbound};
use victory_memory::{
    PAGE_BYTES,
    physical::{BackingPins, PageNumbers, PhysicalPage},
};

const PAGE: usize = PAGE_BYTES as usize;
const SRC: usize = PAGE;
const DST: usize = 8 * PAGE;
const LAST: usize = DST + 2 * PAGE;

fn capture(first: PhysicalPage, last: PhysicalPage) -> BackingPins {
    let mut pins = BackingPins::default();
    pins.remember(SRC, SRC + 3 * PAGE, Some(first));
    pins.remember(DST, LAST, Some(first.offset(3)));
    pins.remember(LAST, LAST + PAGE, Some(last));
    pins
}

fn copy(last_end: usize) -> GuestCopy {
    // Source boundaries fall two bytes before destination boundaries.
    GuestCopy::new(DST + 3, SRC + 5, LAST + last_end - (DST + 3)).unwrap()
}

#[test]
fn destination_tail_and_aliased_prefix_must_be_disjoint() {
    let first = PageNumbers::default().allocate(5).unwrap();
    let pins = capture(first, first.offset(3)); // destination pages A, B, A
    for last_end in [2, 3, 4] {
        let op = copy(last_end);
        // Independent interval oracle: only A repeats, at offsets [3, PAGE) and [0, last_end).
        let overlap = 3 < last_end;
        if overlap {
            // At A:3 the two writes would disagree. Model each source page as a constant byte.
            let payload = [0x11, 0x22, 0x33];
            let byte_for_dst = |va| payload[(op.src + (va - op.dst) - SRC) / PAGE];
            assert_ne!(byte_for_dst(DST + 3), byte_for_dst(LAST + 3));
        }
        assert_eq!(
            op.bind(&pins).map(|_| ()),
            if overlap { Err(Unbound::OverlappingDestination) } else { Ok(()) },
            "prefix end {last_end}"
        );
    }
}

#[test]
fn destination_remap_only_resolves_the_collision_in_a_fresh_capture() {
    let first = PageNumbers::default().allocate(6).unwrap();
    let replacement = first.offset(5);
    let op = copy(4);
    let mut old = capture(first, first.offset(3));
    old.remember(LAST, LAST + PAGE, Some(replacement));
    assert_eq!(op.bind(&old), Err(Unbound::OverlappingDestination));

    let fresh = capture(first, replacement);
    let pieces = op.bind(&fresh).unwrap();
    assert_eq!(pieces.last().unwrap().dst, replacement);
}
