//! A title-buffer copy is retained as guest addresses until submission.
//! Splitting a range resolves both ends against the same backing capture,
//! never against a host buffer baked in during title command recording.

use victory_memory::{
    PAGE_BYTES,
    gpu_table::ADDRESS_LIMIT,
    physical::{BackingPins, PhysicalPage},
};

#[derive(Clone, Copy)]
pub(crate) struct GuestCopy {
    pub(crate) dst: usize,
    pub(crate) src: usize,
    pub(crate) len: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct CopyPiece {
    pub(crate) src: PhysicalPage,
    pub(crate) src_offset: u32,
    pub(crate) dst: PhysicalPage,
    pub(crate) dst_offset: u32,
    pub(crate) count: u32,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Unbound {
    AddressWidth,
    Source(usize),
    Destination(usize),
}

impl GuestCopy {
    pub(crate) fn new(dst: usize, src: usize, len: usize) -> Option<Self> {
        dst.checked_add(len)?;
        src.checked_add(len)?;
        (len != 0).then_some(Self { dst, src, len })
    }

    pub(crate) fn ranges(self) -> [(u64, u64); 2] {
        [
            (self.src as u64, (self.src + self.len) as u64),
            (self.dst as u64, (self.dst + self.len) as u64),
        ]
    }

    /// The native encoder must gather source bytes before scattering into
    /// destinations: two disjoint VAs may be aliases of one physical page
    /// (also two different pages may share the same native slab).
    pub(crate) fn bind(self, pins: &BackingPins) -> Result<Vec<CopyPiece>, Unbound> {
        const PAGE: usize = PAGE_BYTES as usize;
        if self.ranges().iter().any(|&(_, end)| end > ADDRESS_LIMIT) {
            return Err(Unbound::AddressWidth);
        }
        let mut pieces = Vec::new();
        let (mut src, mut dst, mut left) = (self.src, self.dst, self.len);
        while left != 0 {
            let source = pins.expected(src).ok_or(Unbound::Source(src))?;
            let dest = pins.expected(dst).ok_or(Unbound::Destination(dst))?;
            let (s, d) = (src % PAGE, dst % PAGE);
            let count = left.min(PAGE - s).min(PAGE - d);
            pieces.push(CopyPiece {
                src: source,
                src_offset: s as u32,
                dst: dest,
                dst_offset: d as u32,
                count: count as u32,
            });
            (src, dst, left) = (src + count, dst + count, left - count);
        }
        Ok(pieces)
    }
}

#[cfg(test)]
#[test]
fn guest_copy_splits_both_ends_and_preserves_physical_aliases() {
    use victory_memory::physical::PageNumbers;
    const PAGE: usize = PAGE_BYTES as usize;
    let first = PageNumbers::default().allocate(3).unwrap();
    let mut pins = BackingPins::default();
    pins.remember(PAGE, 3 * PAGE, Some(first));
    pins.remember(4 * PAGE, 5 * PAGE, Some(first.offset(1))); // alias of src's second page
    let copy = GuestCopy::new(4 * PAGE + 3, 2 * PAGE - 2, 5).unwrap();
    assert_eq!(
        copy.bind(&pins),
        Ok(vec![
            CopyPiece {
                src: first,
                src_offset: PAGE as u32 - 2,
                dst: first.offset(1),
                dst_offset: 3,
                count: 2
            },
            CopyPiece {
                src: first.offset(1),
                src_offset: 0,
                dst: first.offset(1),
                dst_offset: 5,
                count: 3
            },
        ])
    );
    let misses = GuestCopy::new(5 * PAGE - 1, PAGE, 2).unwrap();
    assert_eq!(misses.bind(&pins), Err(Unbound::Destination(5 * PAGE)));
    assert!(GuestCopy::new(usize::MAX, PAGE, 1).is_none());
}
