//! Two-level raw GPU address table shared by the renderer and shader fixtures.
//!
//! A leaf stores an SRV/UAV descriptor pair and a wrapping 32-bit address
//! bias. Independent virtual pages can bind the same descriptor and the
//! same offset in a slab, even when their addresses are above 4 GiB.
use crate::PAGE_BYTES;

pub const BLOCK: usize = 4096;
pub const LEVEL1: usize = 4096;
pub const ADDRESS_LIMIT: u64 = 1 << 40;
pub const CAPACITY: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindError {
    InvalidAddress,
    InvalidOffset,
    TableFull,
}

/// Block zero is permanently populated with null views. A binding returns
/// the leaf index so submission bookkeeping can associate writes with it.
pub struct RawPageTable {
    dwords: Vec<u32>,
    blocks: Vec<usize>,
    null_pair: u32,
}

impl RawPageTable {
    pub fn new(null_pair: u32) -> Self {
        let mut table = Self {
            dwords: vec![0; LEVEL1],
            blocks: vec![usize::MAX],
            null_pair,
        };
        table.append_null_block();
        table
    }

    fn append_null_block(&mut self) {
        self.dwords
            .extend((0..BLOCK).flat_map(|_| [self.null_pair, 0]));
    }

    pub fn bind_page(
        &mut self,
        page: u64,
        pair: u32,
        slab_offset: u32,
    ) -> Result<usize, BindError> {
        if !u64::from(slab_offset).is_multiple_of(PAGE_BYTES) || slab_offset >= (1 << 29) {
            return Err(BindError::InvalidOffset);
        }
        self.bind_bias_if(page, pair, (page as u32).wrapping_sub(slab_offset), |_| {
            true
        })
    }

    /// A page may also address part of a contiguous view whose start is
    /// not page aligned. The caller supplies the existing view's bias.
    /// `accept` receives the leaf index; if it returns false the existing
    /// entry is preserved. A new block still starts with all null views.
    pub fn bind_bias_if(
        &mut self,
        page: u64,
        pair: u32,
        bias: u32,
        accept: impl FnOnce(usize) -> bool,
    ) -> Result<usize, BindError> {
        if page >= ADDRESS_LIMIT || !page.is_multiple_of(PAGE_BYTES) {
            return Err(BindError::InvalidAddress);
        }
        let l1 = (page >> 28) as usize;
        if self.dwords[l1] == 0 {
            if self.blocks.len() > CAPACITY {
                return Err(BindError::TableFull);
            }
            self.dwords[l1] = self.blocks.len() as u32;
            self.blocks.push(l1);
            self.append_null_block();
        }
        let entry = self.dwords[l1] as usize * BLOCK + ((page >> 16) as usize & (BLOCK - 1));
        if accept(entry) {
            let at = LEVEL1 + 2 * entry;
            self.dwords[at..at + 2].copy_from_slice(&[pair, bias]);
        }
        Ok(entry)
    }

    pub fn into_parts(self) -> (Vec<u32>, Vec<usize>) {
        (self.dwords, self.blocks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn translate(table: &[u32], address: u64) -> (u32, u32) {
        let block = if address < ADDRESS_LIMIT {
            table[(address >> 28) as usize]
        } else {
            0
        };
        let page = block as usize * BLOCK + ((address >> 16) as usize & (BLOCK - 1));
        let at = LEVEL1 + 2 * page;
        (table[at], (address as u32).wrapping_sub(table[at + 1]))
    }

    #[test]
    fn aliases_address_the_same_descriptor_and_nonzero_slab_offset() {
        let mut t = RawPageTable::new(2);
        let low = 0x1_0000;
        let high = 0x91_fffe_0000;
        let offset = 0x8_0000;
        assert_ne!(
            t.bind_page(low, 22, offset).unwrap(),
            t.bind_page(high, 22, offset).unwrap()
        );
        t.bind_page(low + PAGE_BYTES, 24, 0).unwrap();
        let (words, _) = t.into_parts();
        assert_eq!(translate(&words, low + 0xfffc), (22, offset + 0xfffc));
        assert_eq!(translate(&words, high + 0xfffc), (22, offset + 0xfffc));
        assert_eq!(translate(&words, low + PAGE_BYTES), (24, 0));
        assert_eq!(translate(&words, high + PAGE_BYTES).0, 2);
        assert_eq!(translate(&words, ADDRESS_LIMIT).0, 2);
    }

    #[test]
    fn invalid_bindings_do_not_change_the_table() {
        let mut t = RawPageTable::new(2);
        assert_eq!(t.bind_page(1, 4, 0), Err(BindError::InvalidAddress));
        assert_eq!(
            t.bind_page(ADDRESS_LIMIT, 4, 0),
            Err(BindError::InvalidAddress)
        );
        assert_eq!(t.bind_page(0, 4, 4), Err(BindError::InvalidOffset));
        assert_eq!(t.bind_page(0, 4, 1 << 29), Err(BindError::InvalidOffset));
        let (words, blocks) = t.into_parts();
        assert_eq!(blocks.len(), 1);
        assert_eq!(translate(&words, 0), (2, 0));
    }
}
