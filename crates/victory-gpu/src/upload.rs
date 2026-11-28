//! Memory `allocate_upload` (`list.rs`) hands out: the title's CPU writes
//! it and its shaders read it at the same address (the structure builder's
//! instance descriptions lie there).
//!
//! The console's GPU addresses have 40 bits, and a translated shader that
//! reads by address has no more (`victory_shader`'s `Lower::memory`): memory
//! from the process heap lies above 2^40 and was read at another address, as
//! zeros. So this memory is taken in blocks of 64 KiB out of arenas in
//! [4 GiB, 2^37), as upstream VIctory 0f637570 and 6d021fe3 place it (there
//! an allocation of its own each). An arena is a title buffer to the host
//! copies ([`crate::shadow`]) and lives as long as the process: the directory
//! of memory by address ([`crate::memory`]) keeps the copy it found at an
//! address for good, so the same addresses must stay the same copy.
//! `VICTORY_UPLOAD_LOW=0`: from the process heap, as before.

use std::num::NonZeroUsize;
use std::sync::{Mutex, OnceLock};

/// What blocks are cut on: a page of the directory of memory by address.
const BLOCK: usize = 64 << 10;
/// An arena's size, if one request does not need more.
const ARENA: usize = 4 << 20;

struct Arena {
    at: usize,
    /// Per block whether it is handed out.
    taken: Vec<bool>,
}

static ARENAS: Mutex<Vec<Arena>> = Mutex::new(Vec::new());

/// Upload memory of one request, zeroed, given back when dropped.
pub(crate) struct Upload(Kind);

enum Kind {
    /// Address, bytes.
    Low(usize, usize),
    Heap(Vec<u8>),
}

fn low() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_UPLOAD_LOW").is_none_or(|v| v != "0"))
}

/// The first place of `n` free blocks side by side.
fn gap(taken: &[bool], n: usize) -> Option<usize> {
    let mut free = 0;
    for (i, &t) in taken.iter().enumerate() {
        free = if t { 0 } else { free + 1 };
        if free == n {
            return Some(i + 1 - n);
        }
    }
    None
}

/// Owns fresh zeroed, write-watched memory in [4 GiB, 2^37), within the
/// directory of memory by address. Dropping it releases the allocation
/// unless ownership was transferred to the process with [`Self::keep`].
pub(crate) struct Reservation(NonZeroUsize);

// In the release test, keep parallel users of this allocator from taking
// the same address between VirtualFree and VirtualQuery.
#[cfg(test)]
static RESERVATION_TEST_LOCK: Mutex<()> = Mutex::new(());

impl Reservation {
    pub(crate) fn new(len: usize) -> Option<Self> {
        #[cfg(test)]
        let _lock = RESERVATION_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        allocate(len).map(Self)
    }

    /// The allocation base, valid while this owner lives or after `keep`.
    pub(crate) fn address(&self) -> usize {
        self.0.get()
    }

    /// Keep this allocation at the same address for the life of the process.
    pub(crate) fn keep(self) -> usize {
        let at = self.address();
        std::mem::forget(self);
        at
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
        // SAFETY: this owner holds the original allocation base; MEM_RELEASE
        // with size zero releases the entire allocation.
        unsafe { VirtualFree(self.address() as *mut _, 0, MEM_RELEASE) };
    }
}

/// Fresh write-watched memory kept for the life of the process.
pub(crate) fn reserve(len: usize) -> Option<usize> {
    Reservation::new(len).map(Reservation::keep)
}

fn allocate(len: usize) -> Option<NonZeroUsize> {
    use windows_sys::Win32::System::Memory::{
        MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER, MEM_RESERVE, MemExtendedParameterAddressRequirements,
        PAGE_READWRITE, VirtualAlloc2,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: (1usize << 32) as *mut _,
        HighestEndingAddress: ((1usize << 37) - 1) as *mut _,
        Alignment: 0,
    };
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // SAFETY: a fresh allocation; the parameter points at `requirements`,
    // which outlives the call.
    let at = unsafe {
        VirtualAlloc2(
            std::ptr::null_mut(),
            std::ptr::null(),
            len,
            MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
            PAGE_READWRITE,
            &mut parameter,
            1,
        )
    } as usize;
    NonZeroUsize::new(at)
}

impl Upload {
    pub(crate) fn new(len: usize) -> Option<Upload> {
        if !low() {
            return Some(Upload(Kind::Heap(vec![0; len])));
        }
        let n = len.max(1).div_ceil(BLOCK);
        let mut arenas = ARENAS.lock().unwrap_or_else(|e| e.into_inner());
        let found = arenas.iter().enumerate().find_map(|(a, arena)| Some((a, gap(&arena.taken, n)?)));
        let (a, first) = match found {
            Some(place) => place,
            None => {
                let blocks = n.max(ARENA / BLOCK);
                let at = reserve(blocks * BLOCK)?;
                crate::shadow::register(at, at as u64, (blocks * BLOCK) as u64);
                tracing::info!(at = format_args!("{at:#x}"), bytes = blocks * BLOCK, "memory for lists' uploads, where shaders reach it by address");
                arenas.push(Arena { at, taken: vec![false; blocks] });
                (arenas.len() - 1, 0)
            }
        };
        let arena = &mut arenas[a];
        arena.taken[first..first + n].fill(true);
        let at = arena.at + first * BLOCK;
        // SAFETY: the blocks are the arena's and nobody else's now.
        unsafe { std::ptr::write_bytes(at as *mut u8, 0, len) };
        Some(Upload(Kind::Low(at, len)))
    }

    pub(crate) fn start(&mut self) -> *mut u8 {
        match &mut self.0 {
            Kind::Low(at, _) => *at as *mut u8,
            Kind::Heap(block) => block.as_mut_ptr(),
        }
    }
}

impl Drop for Upload {
    fn drop(&mut self) {
        let Kind::Low(at, len) = self.0 else { return };
        let mut arenas = ARENAS.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(arena) = arenas.iter_mut().find(|a| (a.at..a.at + a.taken.len() * BLOCK).contains(&at)) {
            let first = (at - arena.at) / BLOCK;
            arena.taken[first..first + len.max(1).div_ceil(BLOCK)].fill(false);
        }
    }
}

/// The block handed out that holds `address`: its start and bytes (whole
/// [`BLOCK`]s).
pub(crate) fn block(address: usize) -> Option<(usize, usize)> {
    let arenas = ARENAS.lock().unwrap_or_else(|e| e.into_inner());
    let arena = arenas.iter().find(|a| (a.at..a.at + a.taken.len() * BLOCK).contains(&address))?;
    let first = (address - arena.at) / BLOCK;
    let n = arena.taken[first..].iter().take_while(|t| **t).count();
    (n > 0).then_some((arena.at + first * BLOCK, n * BLOCK))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(windows)]
    #[test]
    fn unpublished_reservation_releases_pages_but_kept_reservation_does_not() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_FREE, MEM_RELEASE, MEMORY_BASIC_INFORMATION, VirtualFree, VirtualQuery,
        };

        fn query(at: usize) -> MEMORY_BASIC_INFORMATION {
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: VirtualQuery writes into `info` and does not dereference `at`.
            assert_ne!(unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) }, 0);
            info
        }

        let len = 2 * BLOCK;
        let unpublished = Reservation::new(len).unwrap();
        let kept = Reservation::new(len).unwrap();
        let at = unpublished.address();
        let before = query(at);
        assert_eq!(before.AllocationBase as usize, at);
        assert_eq!(before.State, MEM_COMMIT);
        assert!(before.RegionSize >= len);

        let _lock = RESERVATION_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // A failure before publication drops the owner.
        drop(unpublished);
        let after = query(at);
        assert_eq!(after.State, MEM_FREE);
        assert!(after.RegionSize >= len);

        // A successful publication consumes the owner without freeing the pages.
        let at = kept.address();
        assert_eq!(kept.keep(), at);
        let after = query(at);
        assert_eq!(after.AllocationBase as usize, at);
        assert_eq!(after.State, MEM_COMMIT);
        assert!(after.RegionSize >= len);
        // SAFETY: the kept sample was never shared outside this test.
        assert_ne!(unsafe { VirtualFree(at as *mut _, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn blocks_lie_below_40_bits_and_are_used_again_zeroed() {
        let mut a = Upload::new(100).unwrap();
        let mut b = Upload::new(BLOCK + 1).unwrap();
        let (pa, pb) = (a.start(), b.start());
        assert!((pa as usize) >= 1 << 32 && (pb as usize) + 2 * BLOCK <= 1 << 40);
        assert!((pa as usize).is_multiple_of(BLOCK) && pa != pb);
        // SAFETY: 100 bytes were handed out.
        unsafe { pa.write_bytes(0xab, 100) };
        drop(a);
        // The lowest free place (another test may have taken it meanwhile;
        // zeroed either way).
        let mut c = Upload::new(100).unwrap();
        // SAFETY: as above.
        assert!(unsafe { std::slice::from_raw_parts(c.start(), 100) }.iter().all(|&x| x == 0));
        // More than an arena.
        let mut d = Upload::new(ARENA + 1).unwrap();
        // SAFETY: the bytes were handed out.
        unsafe { d.start().add(ARENA).write(1) };
        drop(b);
    }

    #[test]
    fn a_gap_is_blocks_side_by_side() {
        assert_eq!(gap(&[true, false, true, false, false, true], 1), Some(1));
        assert_eq!(gap(&[true, false, true, false, false, true], 2), Some(3));
        assert_eq!(gap(&[true, false, true, false, false, true], 3), None);
    }
}
