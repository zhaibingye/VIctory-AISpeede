//! `xmem.dll`: console memory, served from host memory. The GPU side works
//! from CPU addresses (unified memory), so memory types and bandwidth
//! classes change nothing here; what constrains addresses (alignment,
//! 2 MiB pages, address-bit limits) is honoured.
#![allow(non_snake_case)]
// Title code calls these through its import table, passing pointers the real
// API would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::{Layout, alloc, dealloc};
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};
#[cfg(feature = "paged-title")]
use std::sync::{RwLock, RwLockReadGuard};

use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_COMMITMENT_LIMIT, ERROR_NOT_ENOUGH_MEMORY, GetLastError, HANDLE, INVALID_HANDLE_VALUE, SetLastError,
};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_DECOMMIT, MEM_EXTENDED_PARAMETER,
    MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE,
    MEM_RESERVE_PLACEHOLDER, MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3,
    MemExtendedParameterAddressRequirements, PAGE_NOACCESS, PAGE_READWRITE, SEC_COMMIT,
    SEC_RESERVE, UnmapViewOfFile2, VirtualAlloc, VirtualAlloc2, VirtualFree, VirtualQueryEx,
};
use windows_sys::Win32::System::ProcessStatus::{
    K32GetPerformanceInfo, K32GetProcessMemoryInfo, PERFORMANCE_INFORMATION,
    PROCESS_MEMORY_COUNTERS,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;
use windows_sys::core::BOOL;

use crate::{HleModule, hle_export};
use victory_memory::{budget::Pool, virtual_pages::VirtualPages};
use victory_memory::physical::{
    Aliases, BackingPins, DeferredRam, HostBudget, PageNumbers, PhysicalPage, Reservation,
};

const MIB: usize = 1 << 20;

/// Whether we run under Wine (its ntdll exports `wine_get_version`).
pub(crate) fn on_wine() -> bool {
    static WINE: LazyLock<bool> = LazyLock::new(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
        // SAFETY: plain lookups in ntdll, always loaded.
        unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            !ntdll.is_null() && GetProcAddress(ntdll, c"wine_get_version".as_ptr().cast()).is_some()
        }
    });
    *WINE
}

/// Standard memory the title has handed to the system.
static TRANSFERRED: AtomicUsize = AtomicUsize::new(0);

// Console additions to the host's flags.
/// 2 MiB pages; the host reads this bit as large pages, which need a
/// privilege.
const PAGES_2M: u32 = 0x2000_0000;
/// With `PAGES_2M`: 64 KiB pages; the host reads it as AWE memory.
const PAGES_64K_BIT: u32 = 0x40_0000;
const PAGES_64K: u32 = PAGES_2M | PAGES_64K_BIT;
/// The host's protection constants and `PAGE_GUARD`. Above them the console
/// adds GPU access bits (0x800-0x40000). Write-combining and uncached are
/// dropped too: our GPU side reads this memory through the CPU.
const HOST_PROTECTION: u32 = 0x1ff;
/// XMem flags of memory both processors reach: all of ours.
const GRAPHICS: u64 = 3;
/// XMem flag bit of 32-bit graphics memory.
const ADDRESS_32BIT: u64 = 4;
/// The host's write watching (not in windows-sys' memory module).
const MEM_WRITE_WATCH: u32 = 0x20_0000;

/// The addresses XMem `flags` allow, first and last: the low 4 GiB for
/// 32-bit memory; else from 4 GiB (kept for 32-bit requests, which the
/// title's big pools would otherwise crowd out) up to 46 down to 40 bits
/// from the field at bit 12, or 40 bits without one: the console keeps
/// title memory there, and its ray-tracing structures must be (Grand Theft
/// Auto VI's BVH builder refuses one above; the title asks for its GPU heap
/// with no field). Decision 0001.
fn address_range(flags: u64) -> (usize, usize) {
    let bits = match (flags & ADDRESS_32BIT, flags >> 12 & 7) {
        (0, 0) => 40,
        (0, field) => 47 - field,
        _ => return (0, u32::MAX as usize),
    };
    (1 << 32, (1 << bits) - 1)
}

#[derive(Default)]
struct PrivateMemory {
    pages: VirtualPages,
    reservation: Option<Reservation>,
}

impl PrivateMemory {
    fn use_budget(&mut self, budget: &HostBudget) -> Result<(), victory_memory::budget::NoSpace> {
        if self.reservation.is_none() {
            self.reservation = Some(budget.reserve_ram(self.pages.committed_bytes())?);
        }
        Ok(())
    }

    fn reserve_commit(&mut self, base: *const c_void, len: usize, kind: u32) -> bool {
        let Some(charge) = self.reservation.as_mut() else { return true; };
        if kind & MEM_COMMIT == 0 || len == 0 { return true; }
        let mut info = MEMORY_BASIC_INFORMATION::default();
        let original = base as usize;
        let (allocation, at) = if base.is_null() {
            (None, 0)
        } else if kind & MEM_RESERVE != 0 {
            (None, original & !(PAGE - 1))
        } else {
            // SAFETY: a query only, with a local output record.
            let found = unsafe { VirtualQueryEx(GetCurrentProcess(), base, &mut info, size_of_val(&info)) };
            if found != 0 && matches!(
                info.Type, windows_sys::Win32::System::Memory::MEM_MAPPED | windows_sys::Win32::System::Memory::MEM_IMAGE
            ) { return true; } // section pages are charged by their owner
            ((found != 0).then_some(info.AllocationBase as usize), original)
        };
        let Some(covered) = original.checked_sub(at).and_then(|extra| extra.checked_add(len)) else { return false; };
        let Some(total) = self.pages.additional_bytes(allocation, at, covered)
            .and_then(|bytes| self.pages.committed_bytes().checked_add(bytes)) else { return false; };
        charge.resize(total).is_ok()
    }

    fn settle(&mut self) {
        if let Some(charge) = self.reservation.as_mut() {
            charge.resize(self.pages.committed_bytes())
                .expect("host private commit bounded by its preflight reservation");
        }
    }
}

// Serialize wrapped host commit/release with GPU capture.
fn virtual_pages() -> MutexGuard<'static, PrivateMemory> {
    static PRIVATE: LazyLock<Mutex<PrivateMemory>> = LazyLock::new(Mutex::default);
    PRIVATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Activate when the unified renderer takes ownership, not while legacy
/// shadows duplicate RAM charges. Existing tracked commits count too.
#[allow(dead_code, reason = "enabled together with renderer cutover")]
pub(crate) fn activate_private_ram_budget() -> Result<(), victory_memory::budget::NoSpace> {
    virtual_pages().use_budget(HostBudget::process())
}

fn note_private_commit(pages: &mut VirtualPages, at: *mut c_void, bytes: usize, kind: u32) {
    if at.is_null() || kind & (MEM_RESERVE | MEM_COMMIT) == 0 || kind & MEM_RESERVE_PLACEHOLDER != 0 {
        return;
    }
    let mut info = MEMORY_BASIC_INFORMATION::default();
    // SAFETY: querying the successful allocation writes one local record.
    let found = unsafe { VirtualQueryEx(GetCurrentProcess(), at, &mut info, size_of_val(&info)) };
    if found == 0 || info.Type != windows_sys::Win32::System::Memory::MEM_PRIVATE { return; }
    let allocation = info.AllocationBase as usize;
    if kind & MEM_RESERVE != 0 && allocation == at as usize {
        pages.reserved(allocation);
    }
    if kind & MEM_COMMIT != 0 && info.State == MEM_COMMIT {
        pages.committed(allocation, at as usize, bytes);
    }
}

fn commit_private(
    private: &mut PrivateMemory,
    base: *const c_void, bytes: usize, kind: u32,
    allocate: impl FnOnce() -> *mut c_void,
) -> *mut c_void {
    if !private.reserve_commit(base, bytes, kind) {
        // SAFETY: sets this thread's Win32 status for the failed allocation.
        unsafe { SetLastError(ERROR_NOT_ENOUGH_MEMORY) };
        return std::ptr::null_mut();
    }
    let at = allocate();
    // SAFETY: only reads this thread's Win32 error after a failed call.
    let error = at.is_null().then(|| unsafe { GetLastError() });
    let committed = if base.is_null() || at.is_null() {
        bytes
    } else {
        (base as usize).saturating_add(bytes).saturating_sub(at as usize)
    };
    note_private_commit(&mut private.pages, at, committed, kind);
    private.settle();
    if let Some(error) = error {
        // SAFETY: restores the original failed allocation status.
        unsafe { SetLastError(error) };
    }
    at
}

pub(crate) fn private_virtual_alloc(base: *const c_void, size: usize, kind: u32, protect: u32) -> *mut c_void {
    commit_private(&mut virtual_pages(), base, size, kind, || {
        // SAFETY: title VirtualAlloc ABI; Windows validates all arguments.
        unsafe { VirtualAlloc(base, size, kind, protect) }
    })
}

fn free_private(private: &mut PrivateMemory, base: *mut c_void, size: usize, kind: u32) -> BOOL {
    // SAFETY: unchanged host VirtualFree ABI; only publish success.
    let freed = unsafe { VirtualFree(base, size, kind) };
    if freed != 0 {
        if kind & (MEM_RELEASE | MEM_DECOMMIT) != 0 && size == 0 {
            private.pages.released(base as usize);
        } else if kind & (MEM_RELEASE | MEM_DECOMMIT) != 0 {
            private.pages.decommitted(base as usize, size);
        }
        private.settle();
    }
    freed
}

pub(crate) fn private_virtual_free(base: *mut c_void, size: usize, kind: u32) -> BOOL {
    free_private(&mut virtual_pages(), base, size, kind)
}

pub(crate) fn allocate_list_upload(bytes: usize) -> usize {
    if activate_private_ram_budget().is_err() { return 0; }
    XMemVirtualAlloc(std::ptr::null_mut(), bytes, MEM_RESERVE | MEM_COMMIT, GRAPHICS, PAGE_READWRITE) as usize
}

pub(crate) fn free_list_upload(base: usize) {
    private_virtual_free(base as *mut c_void, 0, MEM_RELEASE);
}

pub extern "system" fn XMemVirtualAlloc(
    base: *mut c_void,
    size: usize,
    kind: u32,
    flags: u64,
    protect: u32,
) -> *mut c_void {
    let (lowest, highest) = address_range(flags);
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: lowest as *mut c_void,
        HighestEndingAddress: highest as *mut c_void,
        Alignment: if kind & (PAGES_2M | PAGES_64K_BIT) == PAGES_2M {
            2 * MIB
        } else {
            0
        },
    };
    let mut protect = match protect & HOST_PROTECTION {
        0 => PAGE_NOACCESS,
        p => p,
    };
    // A reservation of 64 KiB pages takes physical pages later
    // (`XMemMapPhysicalPages`); on the host only a placeholder can.
    // shortcut: committing part of such a reservation later would fail
    // (Grand Theft Auto VI's 23 calls never do); replace that part of the
    // placeholder then.
    let placeholder = kind & PAGES_64K == PAGES_64K && kind & MEM_COMMIT == 0;
    if placeholder {
        if kind & MEM_REPLACE_PLACEHOLDER != 0 {
            // Part of a placeholder the title split: it stays one.
            return base;
        }
        protect = PAGE_NOACCESS;
    }
    let mut kind = kind & !(PAGES_2M | PAGES_64K_BIT);
    if placeholder {
        kind |= MEM_RESERVE | MEM_RESERVE_PLACEHOLDER;
    }
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // Address requirements apply to reservations at an address of our choice.
    let constrained = base.is_null();
    if constrained {
        kind |= MEM_RESERVE;
        // The GPU side copies only the pages the title wrote since its last
        // copy (`victory_gpu`, decision 0007); placeholders cannot be
        // watched (their physical-page views are copied whole).
        if kind & MEM_RESERVE_PLACEHOLDER == 0 {
            kind |= MEM_WRITE_WATCH;
        }
    }
    // SAFETY: the title's own arguments, cleaned of console bits; the
    // parameter points at `requirements`, which outlives the call.
    let at = commit_private(&mut virtual_pages(), base, size, kind, || unsafe {
        VirtualAlloc2(
            GetCurrentProcess(),
            base,
            size,
            kind,
            protect,
            if constrained {
                &mut parameter
            } else {
                std::ptr::null_mut()
            },
            constrained.into(),
        )
    });
    if at.is_null() {
        let error = std::io::Error::last_os_error();
        if kind & MEM_COMMIT != 0 {
            commit_refused(size, "virtual memory", error);
        } else {
            tracing::warn!(
                size,
                kind = format_args!("{kind:#x}"),
                flags = format_args!("{flags:#x}"),
                %error,
                "XMemVirtualAlloc refused"
            );
        }
    }
    if base.is_null() {
        tracing::debug!(
            size,
            kind = format_args!("{kind:#x}"),
            flags = format_args!("{flags:#x}"),
            at = format_args!("{at:p}"),
            "XMemVirtualAlloc"
        );
    }
    at
}

/// `MEMORY_BASIC_INFORMATION` (the console's first 0x30 bytes are the
/// same), then the XMem flags and tag.
#[repr(C)]
pub struct BasicInformation {
    host: MEMORY_BASIC_INFORMATION,
    flags: u64,
    tag_code: u32,
    tag: [u8; 4],
}

// shortcut: every committed page reports graphics memory and no tag, since
// all of it is GPU-reachable here (Grand Theft Auto VI's BVH builder, the
// only caller, wants graphics memory); remember the flags of
// `XMemVirtualAlloc` regions if a title tells CPU-only memory apart.
pub extern "system" fn XMemVirtualQuery(
    process: HANDLE,
    address: *const c_void,
    out: *mut BasicInformation,
) -> BOOL {
    let size = size_of::<MEMORY_BASIC_INFORMATION>();
    // SAFETY: the title passes room for the whole structure; the host
    // fails on a null one.
    unsafe {
        if VirtualQueryEx(process, address, out.cast(), size) == 0 {
            return 0;
        }
        let info = &mut *out;
        info.flags = if info.host.State == MEM_COMMIT {
            GRAPHICS
        } else {
            0
        };
        info.tag_code = 0;
        info.tag = [0; 4];
    }
    1
}

// `XMemAlloc` attributes: alignment as log2 (5 bits at 24, 4 bytes and up
// from 2), then the memory type, then bit 33: a 32-bit address.
const ALIGNMENT_SHIFT: u64 = 24;
const ALLOC_32BIT: u64 = 1 << 33;
/// Our block header: the block's size and the pointer's offset in it,
/// just below the pointer handed out.
const HEADER: usize = 2 * size_of::<usize>();
/// The alignment the host heap gives without padding of its own.
const HEAP_ALIGN: usize = 16;

/// The alignment `attributes` ask for (at least 16, which leaves room for
/// the header), and the host block for `size` bytes at it: room for the
/// header and the alignment, at the host heap's own alignment. (A layout
/// at the asked alignment pays it twice: the host allocator pads an
/// over-aligned block by the alignment again, so a 64 KiB-aligned block
/// cost 192 KiB.)
fn layout(size: usize, attributes: u64) -> Option<(usize, Layout)> {
    let align = 1usize << (attributes >> ALIGNMENT_SHIFT & 0x1f).max(4);
    // From a 16-aligned block, the header and the step up to `align` take
    // at most `align` bytes (`HEADER` is 16).
    let layout = Layout::from_size_align(size.checked_add(align)?, HEAP_ALIGN).ok()?;
    Some((align, layout))
}

type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
type FreeRoutine = extern "system" fn(*mut c_void, u64);

/// The allocation and free routines the title's exe exports as
/// `__XMemAllocationHooks__` (Grand Theft Auto VI does): `XMemAlloc` and
/// `XMemFree` call them, for the title's own calls and its middleware's;
/// the `…Default` exports do not.
fn hooks() -> [usize; 2] {
    static AT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let at = match AT.get() {
        Some(&at) => at,
        // Looked up once the title is mapped, which is before it runs.
        None => match crate::runtime::main_image() {
            Some(base) => *AT.get_or_init(|| {
                use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
                // SAFETY: a mapped image and a NUL-terminated name.
                let at = unsafe {
                    GetProcAddress(base as _, c"__XMemAllocationHooks__".as_ptr().cast())
                }
                .map_or(0, |f| f as usize);
                tracing::info!(found = at != 0, "xmem allocation hooks");
                at
            }),
            None => 0,
        },
    };
    if at == 0 {
        return [0; 2];
    }
    // SAFETY: the export is two routine pointers.
    unsafe { (at as *const [usize; 2]).read() }
}

/// The title's general allocator, for middleware too: its hook if it has
/// one, else [`XMemAllocDefault`].
pub extern "system" fn XMemAlloc(size: usize, attributes: u64) -> *mut c_void {
    match hooks()[0] {
        0 => XMemAllocDefault(size, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let alloc: AllocRoutine = unsafe { std::mem::transmute(hook) };
            alloc(size, attributes)
        }
    }
}

/// See [`XMemAlloc`].
pub extern "system" fn XMemFree(address: *mut c_void, attributes: u64) {
    match hooks()[1] {
        0 => XMemFreeDefault(address, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let free: FreeRoutine = unsafe { std::mem::transmute(hook) };
            free(address, attributes)
        }
    }
}

/// 仅记录明确由默认分配器创建的图形堆块；不改 allocator hook，
/// 不用虚拟地址或 VirtualQuery 猜测第三方分配的身份。
#[cfg(feature = "paged-title")]
static GRAPHICS_HEAP: LazyLock<Mutex<BTreeMap<usize, usize>>> = LazyLock::new(Mutex::default);

#[cfg(feature = "paged-title")]
fn graphics_allocator(attributes: u64) -> bool {
    (attributes >> 16) & 0xff == 128
}

#[cfg(feature = "paged-title")]
pub(crate) fn is_graphics_heap_span(start: usize, end: usize) -> bool {
    if start >= end { return false; }
    GRAPHICS_HEAP.lock().unwrap_or_else(|e| e.into_inner())
        .range(..=start).next_back().is_some_and(|(_, &bound)| end <= bound)
}

/// The system allocator behind [`XMemAlloc`].
pub extern "system" fn XMemAllocDefault(size: usize, attributes: u64) -> *mut c_void {
    if attributes & ALLOC_32BIT != 0 {
        return XMemVirtualAlloc(
            std::ptr::null_mut(),
            size,
            MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | ADDRESS_32BIT,
            PAGE_READWRITE,
        );
    }
    let Some((align, layout)) = layout(size, attributes) else {
        return std::ptr::null_mut();
    };
    // shortcut: the host heap's addresses lie above 2^40, unlike the
    // console's; nothing seen builds ray-tracing structures in this memory
    // (Grand Theft Auto VI uses its GPU heap). Sub-allocate from a
    // reservation below 2^40 if one does.
    // SAFETY: the layout has a non-zero size.
    let block = unsafe { alloc(layout) };
    if block.is_null() {
        tracing::warn!(size, "XMemAllocDefault: the host heap is out of memory");
        return block.cast();
    }
    let offset = (block as usize + HEADER).next_multiple_of(align) - block as usize;
    // SAFETY: the block holds the header, the step to the alignment (at
    // most `align` bytes together) and then `size` bytes.
    unsafe {
        let user = block.add(offset);
        user.sub(HEADER)
            .cast::<[usize; 2]>()
            .write([layout.size(), offset]);
        #[cfg(feature = "paged-title")]
        if graphics_allocator(attributes) && let Some(end) = (user as usize).checked_add(size) {
            GRAPHICS_HEAP.lock().unwrap_or_else(|e| e.into_inner()).insert(user as usize, end);
        }
        user.cast()
    }
}

/// Frees what [`XMemAllocDefault`] returned, given the same attributes.
pub extern "system" fn XMemFreeDefault(address: *mut c_void, attributes: u64) {
    if address.is_null() {
        return;
    }
    // SAFETY: the address came from `XMemAlloc` with these attributes: a
    // whole reservation, or a header-prefixed block.
    unsafe {
        if attributes & ALLOC_32BIT != 0 {
            private_virtual_free(address, 0, MEM_RELEASE);
            return;
        }
        let user = address.cast::<u8>();
        #[cfg(feature = "paged-title")]
        if graphics_allocator(attributes) {
            GRAPHICS_HEAP.lock().unwrap_or_else(|e| e.into_inner()).remove(&(user as usize));
        }
        let [size, offset] = user.sub(HEADER).cast::<[usize; 2]>().read();
        dealloc(
            user.sub(offset),
            Layout::from_size_align_unchecked(size, HEAP_ALIGN),
        );
    }
}

/// A retail console has no tools memory.
pub extern "system" fn XMemIsToolsMemAvailable() -> BOOL {
    0
}

// Physical pages: on the console, 64 KiB frames a title maps into its own
// reservations, at more than one address if it likes. Here each allocation
// is a pagefile-backed section, its pages are 64 KiB slices of it, and a
// mapping is a view of consecutive slices in place of a host placeholder.

const PAGE: usize = victory_memory::PAGE_BYTES as usize;

struct Section {
    handle: HANDLE,
    first: PhysicalPage,
    pages: usize,
    backing: SectionBacking,
}

enum SectionBacking {
    Eager { _reservation: Reservation },
    Deferred(Mutex<DeferredRam>),
}

impl Section {
    fn commit(&self, index: usize, base: usize, len: usize) -> bool {
        let apply = || commit_section_view(base, len);
        match &self.backing {
            SectionBacking::Eager { .. } => apply(),
            SectionBacking::Deferred(state) => {
                let mut state = state.lock().unwrap_or_else(|e| e.into_inner());
                match state.commit(index, len / PAGE, apply) {
                    Ok(committed) => committed,
                    Err(error) => {
                        let (used, limits) = HostBudget::process().pool_state(Pool::Ram);
                        tracing::warn!(requested_mib = len / MIB, used_mib = used / MIB as u64,
                            hard_mib = limits.hard / MIB as u64, shortfall = error.shortfall,
                            "XMem 首次提交物理页超出宿主 RAM 预算");
                        false
                    }
                }
            }
        }
    }
}

// SAFETY: a section handle may be used from any thread.
unsafe impl Send for Section {}
// SAFETY: as above.
unsafe impl Sync for Section {}

impl Drop for Section {
    fn drop(&mut self) {
        // SAFETY: our handle; views keep the section alive on their own.
        if unsafe { CloseHandle(self.handle) } == 0 {
            tracing::error!(error = %std::io::Error::last_os_error(), "physical section close failed");
        }
    }
}

/// Where a page lives, or what a view starts with: a section and a page
/// index in it.
#[derive(Clone)]
struct Slice {
    section: Arc<Section>,
    index: usize,
}

#[derive(Default)]
struct Physical {
    numbers: PageNumbers,
    pages: HashMap<usize, Slice>,
    /// Mapped views by address: their length and what they show.
    views: BTreeMap<usize, (usize, Slice)>,
    shown: Aliases,
}

impl Slice {
    fn page(&self, offset: usize) -> PhysicalPage {
        self.section.first.offset(self.index + offset)
    }

    #[cfg(all(feature = "paged-title", feature = "diagnostics"))]
    fn note_dstorage_write(&self, within: usize, len: usize, confirmed: bool) {
        if len != 0 {
            victory_gpu::title_backing::note_dstorage_section_write(
                self.page(0),
                within,
                len,
                confirmed,
            );
        }
    }
}

impl Physical {
    /// All views intersecting a half-open interval. Views do not overlap.
    fn views_in(&self, start: usize, end: usize) -> impl Iterator<Item = (usize, &(usize, Slice))> {
        use std::ops::Bound::Excluded;
        let first = self.views.range(..=start).next_back()
            .filter(|&(&base, &(len, _))| base + len > start);
        first.into_iter()
            .chain(self.views.range((Excluded(start), Excluded(end))))
            .map(|(&base, view)| (base, view))
    }
}

// A raw VA fill may run through I/O or a decoder. Block even unrelated title
// remaps during that time: unmapping a subrange briefly removes the *whole*
// host view before remapping its sides. Concurrent fills can share the gate.
// Both sides take this gate before PHYSICAL. Snapshot readers must never
// need this gate: they acquire PHYSICAL to stage pages while a writer waits.
#[cfg(feature = "paged-title")]
static TITLE_VIEW_GATE: RwLock<()> = RwLock::new(());

#[cfg(feature = "paged-title")]
#[must_use = "keep the title views and physical writer until the raw write finishes"]
pub(crate) struct RawVaWrite {
    _writer: Option<victory_memory::physical::access::Writer>,
    _views: RwLockReadGuard<'static, ()>,
}

/// Stabilize the VA, then claim just the section pages intersecting the fill.
/// Gaps have no physical writer, but cannot become section mappings mid-fill.
#[cfg(feature = "paged-title")]
pub(crate) fn raw_va_write(at: usize, len: usize) -> Option<RawVaWrite> {
    let end = at.saturating_add(len);
    if at >= end {
        return None;
    }
    let views = TITLE_VIEW_GATE.read().unwrap_or_else(|e| e.into_inner());
    let runs: Vec<_> = {
        let physical = physical();
        physical.views_in(at, end).map(|(base, &(size, ref slice))| {
            let first = at.saturating_sub(base) / PAGE;
            let limit = (end - base).min(size).div_ceil(PAGE);
            (slice.page(first), limit - first)
        }).collect()
    };
    // Never hold PHYSICAL while waiting for snapshots or during the fill.
    let writer = (!runs.is_empty()).then(|| victory_memory::physical::access::writing(&runs));
    Some(RawVaWrite { _writer: writer, _views: views })
}

/// Resolve title ranges once with the mapping lock. The GPU holds the
/// returned section leases after the OS views or title handles are removed.
fn remember_private(
    private: &VirtualPages, pins: &mut BackingPins, start: usize, end: usize,
) {
    let mut at = start;
    while at < end {
        let (next, page) = private.run_at(at, end);
        pins.remember_with_origin(at, next, page, private.has_own_reservation(at));
        at = next;
    }
}

pub(crate) fn pin_backing(ranges: &[(usize, usize)], pins: &mut BackingPins) {
    let physical = physical();
    let private = virtual_pages();
    for &(start, end) in ranges {
        if start >= end {
            continue;
        }
        let start = start / PAGE * PAGE;
        let end = end.checked_next_multiple_of(PAGE).unwrap_or(usize::MAX / PAGE * PAGE);
        if start >= end {
            continue;
        }
        let mut at = start;
        for (base, &(size, ref slice)) in physical.views_in(start, end) {
            let from = base.max(start);
            let to = (base + size).min(end);
            if at < from {
                remember_private(&private.pages, pins, at, from);
            }
            pins.keep(slice.section.first, &slice.section);
            pins.remember(from, to, Some(slice.page((from - base) / PAGE)));
            at = to;
        }
        if at < end {
            remember_private(&private.pages, pins, at, end);
        }
    }
}

/// Read the same XMem backing the submission captured before preparation.
/// Live aliases can be read without remapping; changed ranges use a short
/// private section view. Never read an unrelated replacement page.
pub(crate) fn read_backing_runs(
    start: usize, end: usize, pins: &BackingPins,
    visit: &mut dyn FnMut(usize, Option<usize>, usize, bool),
) {
    let physical = physical();
    let private = virtual_pages();
    let mut at = start;
    while at < end {
        let (limit, expected) = pins.run_at(at, end);
        let current = physical.views.range(..=at).next_back()
            .filter(|&(&base, &(size, _))| at - base < size);
        let (limit, current) = match current {
            Some((&base, &(size, ref slice))) => (
                limit.min(base + size), Some(slice.page((at - base) / PAGE)),
            ),
            None => {
                let next = physical.views.range(at..).next()
                    .map_or(limit, |(&base, _)| limit.min(base));
                private.pages.run_at(at, next)
            }
        };
        let len = limit - at;
        if current == expected {
            visit(at, Some(at), len, true);
        } else if let Some((section, index)) = expected.and_then(|page| pins.section::<Section>(page)) {
            let offset = at % PAGE;
            let pages = (offset + len).div_ceil(PAGE);
            let view = (pages <= section.pages.saturating_sub(index))
                .then(|| View::map_section(section, index, pages)).flatten();
            let source = view.as_ref().map(|view| view.at.Value as usize + offset);
            visit(at, source, len, false);
        } else {
            visit(at, None, len, false);
        }
        at = limit;
    }
}

/// Verify the destination under the map lock. A byte delta also has a
/// stable before-image, so it can still reach the pinned original backing
/// through a temporary view, even when every title alias is gone.
pub(crate) fn with_writeback_page(
    address: usize,
    len: usize,
    pins: &BackingPins,
    redirect: bool,
    write: &mut dyn FnMut(victory_gpu::title_backing::WritebackTarget),
) -> bool {
    use victory_gpu::title_backing::WritebackTarget;

    let offset = address % PAGE;
    if len == 0 || len > PAGE - offset {
        return false;
    }
    let physical = physical();
    let private = virtual_pages();
    let expected = pins.expected(address);
    let current = physical.views.range(..=address).next_back()
        .filter(|&(&base, &(size, _))| address - base < size)
        .map(|(&base, (_, slice))| slice.page((address - base) / PAGE))
        .or_else(|| private.pages.page(address));
    if current == expected {
        write(WritebackTarget { address, original_va: true });
        return true;
    }
    if !redirect {
        return false;
    }
    let Some(page) = expected else { return false };
    let Some((section, index)) = pins.section::<Section>(page) else { return false };
    if index >= section.pages {
        return false;
    }
    let Some(view) = View::map_section(section, index, 1) else {
        return false;
    };
    // Writes through this temporary view do not fault at the title's VAs.
    // Notify its current aliases without ever touching the replacement VA.
    let _writers: Vec<_> = physical.shown.addresses(page).iter().map(|&alias| {
        victory_gpu::kernel_will_write(alias + offset, len)
    }).collect();
    write(WritebackTarget {
        address: view.at.Value as usize + offset,
        original_va: false,
    });
    true
}

/// Every address of a title view showing a page that title memory at
/// `[at, at + len)` shows, except the page's own address: pushed to `out`
/// as (page address, other address). The GPU side's change tracking
/// (`victory_gpu::watch`) marks writes there too.
pub fn aliases(at: usize, len: usize, out: &mut Vec<(usize, usize)>) {
    let end = at.saturating_add(len);
    if at >= end {
        return;
    }
    let physical = physical();
    for (base, &(size, ref slice)) in physical.views_in(at, end) {
        let (from, to) = (at.max(base), end.min(base + size));
        for page in (from / PAGE..to.div_ceil(PAGE)).map(|p| p * PAGE) {
            let k = (page - base) / PAGE;
            out.extend(
                physical.shown.addresses(slice.page(k)).iter()
                    .copied()
                    .filter(|&address| address != page)
                    .map(|address| (page, address)),
            );
        }
    }
}

/// The addresses of title views showing these physical slices: workers can
/// still report aliases after the original page handles have been released.
fn shown_at(runs: &[(Slice, usize)]) -> Vec<usize> {
    let physical = physical();
    runs.iter()
        .flat_map(|(slice, n)| (0..*n).map(move |k| slice.page(k)))
        .flat_map(|id| physical.shown.addresses(id))
        .copied()
        .collect()
}

/// Reports writes into physical pages through our own views at every title
/// view showing them (DirectStorage's page destinations).
fn report_written(shown: &[usize]) {
    for &page in shown {
        drop(victory_gpu::kernel_will_write(page, PAGE));
    }
}

static PHYSICAL: LazyLock<Mutex<Physical>> = LazyLock::new(|| Mutex::new(Physical::default()));

fn physical() -> MutexGuard<'static, Physical> {
    PHYSICAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// `count` is in and out (pages wanted, pages given); all or nothing here.
pub extern "system" fn XMemAllocatePhysicalPages(
    flags: u64,
    count: *mut usize,
    pages: *mut usize,
) -> BOOL {
    if count.is_null() || pages.is_null() {
        return 0;
    }
    // SAFETY: checked non-null.
    let n = unsafe { *count };
    if n == 0 || n > PageNumbers::MAX_SECTION_PAGES {
        // 安全：已检查 count 非空。
        unsafe { count.write(0) };
        return 0;
    }
    let size = n * PAGE;
    let ram = HostBudget::process();
    // 原生 Windows 的 SEC_RESERVE 只预留地址后备；旧后端和 Wine 保持原有准入。
    let backing = if cfg!(feature = "paged-title") && !on_wine() {
        DeferredRam::new(ram, n).map(|pages| SectionBacking::Deferred(Mutex::new(pages)))
    } else {
        ram.reserve_ram(size as u64).map(|lease| SectionBacking::Eager { _reservation: lease })
    };
    let backing = match backing {
        Ok(backing) => backing,
        Err(error) => {
            tracing::warn!(n, reserved = ram.used(Pool::Ram), shortfall = error.shortfall, "XMem 物理后备预留超出宿主 RAM 预算");
            // SAFETY: checked non-null; this ABI reports no pages on failure.
            unsafe { count.write(0) };
            return 0;
        }
    };
    // A pagefile-backed section of `size` bytes, its pages committed when
    // first mapped or written (`commit`): Grand Theft Auto VI asks for 6.1
    // GiB at once, which a busy host often cannot commit (error 1450).
    // SAFETY: plain section creation.
    let handle = unsafe {
        CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            std::ptr::null(),
            // Under Wine a committed section is a sparse file, free until
            // touched, and committing a reserved one page by page costs a
            // wineserver request each that scans every range committed so
            // far: commit it whole there.
            PAGE_READWRITE | if on_wine() { SEC_COMMIT } else { SEC_RESERVE },
            (size as u64 >> 32) as u32,
            size as u32,
            std::ptr::null(),
        )
    };
    if handle.is_null() {
        let error = std::io::Error::last_os_error();
        tracing::warn!(n, %error, "XMemAllocatePhysicalPages failed");
        // SAFETY: checked non-null.
        unsafe { count.write(0) };
        return 0;
    }
    tracing::debug!(
        n,
        flags = format_args!("{flags:#x}"),
        "XMemAllocatePhysicalPages"
    );
    #[cfg(all(feature = "paged-title", feature = "diagnostics"))]
    victory_gpu::title_backing::prepare_xmem_section_history();
    let mut physical = physical();
    let Some(first) = physical.numbers.allocate(n) else {
        // SAFETY: fresh owned section with no views, and a non-null count.
        unsafe {
            CloseHandle(handle);
            count.write(0);
        }
        tracing::warn!(n, "XMem physical page IDs exhausted");
        return 0;
    };
    let section = Arc::new(Section {
        handle,
        first,
        pages: n,
        backing,
    });
    for index in 0..n {
        let number = first.offset(index).number();
        physical.pages.insert(number, Slice {
            section: section.clone(),
            index,
        });
        // SAFETY: the title passes room for `n` page numbers.
        unsafe { pages.add(index).write(number) };
    }
    1
}

/// Pages still mapped stay readable through their views until unmapped.
pub extern "system" fn XMemFreePhysicalPages(n: usize, pages: *const usize) -> BOOL {
    if pages.is_null() {
        return 0;
    }
    let mut physical = physical();
    for i in 0..n {
        // SAFETY: the title passes `n` page numbers.
        physical.pages.remove(&unsafe { *pages.add(i) });
    }
    1
}

/// A view of our own on consecutive slices of one section, committed and
/// writable, unmapped when dropped.
struct View {
    at: MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
    section: Option<Arc<Section>>,
}

impl View {
    /// Maps `n` slices from `slice` on.
    fn map(slice: &Slice, n: usize) -> Option<View> {
        Self::map_section(slice.section.clone(), slice.index, n)
    }

    fn map_section(section: Arc<Section>, index: usize, n: usize) -> Option<View> {
        use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
        let (offset, len) = ((index * PAGE) as u64, n * PAGE);
        // SAFETY: slices of a section we own; the view is ours until dropped.
        let at = unsafe {
            MapViewOfFile(
                section.handle,
                FILE_MAP_WRITE,
                (offset >> 32) as u32,
                offset as u32,
                len,
            )
        };
        if at.Value.is_null() {
            return None;
        }
        let view = View { at, len, section: Some(section.clone()) };
        section.commit(index, at.Value as usize, len).then_some(view)
    }

    fn bytes(&mut self) -> &mut [u8] {
        // SAFETY: the view holds `len` committed read-write bytes, ours
        // alone while `self` lives (other views of the slices only alias
        // them, as the console's do).
        unsafe { std::slice::from_raw_parts_mut(self.at.Value.cast(), self.len) }
    }
}

impl Drop for View {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::UnmapViewOfFile;
        // SAFETY: our view.
        if unsafe { UnmapViewOfFile(self.at) } == 0 {
            // Keep its section and budget charge if the OS kept our view.
            tracing::error!(error = %std::io::Error::last_os_error(), "physical worker view unmap failed");
            std::mem::forget(self.section.take());
        }
    }
}

/// The pages behind bytes `[offset, offset + len)` of `pages` (64 KiB
/// each), as runs of consecutive slices of one section: each run's first
/// slice and its length in pages. One look at the page table, which the
/// title's own mapping calls share with every DirectStorage worker. `None`
/// if a page is unknown or `pages` ends early.
fn runs(pages: &[usize], offset: usize, len: usize) -> Option<Vec<(Slice, usize)>> {
    let end = offset.checked_add(len)?;
    let pages = pages.get(offset / PAGE..end.div_ceil(PAGE))?;
    let physical = physical();
    let mut runs: Vec<(Slice, usize)> = Vec::new();
    for page in pages {
        let slice = physical.pages.get(page)?;
        match runs.last_mut() {
            Some((first, n))
                if Arc::ptr_eq(&first.section, &slice.section)
                    && slice.index == first.index + *n =>
            {
                *n += 1;
            }
            _ => runs.push((slice.clone(), 1)),
        }
    }
    Some(runs)
}

/// Copies `data` into physical pages, from `offset` bytes into the first,
/// whether or not they are mapped anywhere (DirectStorage destinations).
/// One view per run of consecutive pages, not one per page: each view
/// costs a map, a commit and an unmap (which flushes every core's TLB).
/// False if a page is unknown or the host fails.
pub(crate) fn write_physical(pages: &[usize], offset: usize, data: &[u8]) -> bool {
    let Some(runs) = runs(pages, offset, data.len()) else {
        return false;
    };
    // `runs` has released the page-table lock and pinned the resolved sections.
    // Claim every destination together, before copying even the first run.
    #[cfg(feature = "paged-title")]
    let _writer = (!data.is_empty()).then(|| {
        let destinations: Vec<_> = runs.iter().map(|(slice, n)| (slice.page(0), *n)).collect();
        victory_memory::physical::access::writing(&destinations)
    });
    let (mut within, mut done) = (offset % PAGE, 0);
    for (slice, n) in &runs {
        let Some(mut view) = View::map(slice, *n) else {
            return false;
        };
        let k = (n * PAGE - within).min(data.len() - done);
        view.bytes()[within..within + k].copy_from_slice(&data[done..done + k]);
        // This run was copied even if a later run fails to map.
        #[cfg(all(feature = "paged-title", feature = "diagnostics"))]
        slice.note_dstorage_write(within, k, true);
        (within, done) = (0, done + k);
    }
    report_written(&shown_at(&runs));
    true
}

/// Bytes `[offset, offset + len)` of physical pages, writable in place
/// until dropped: DirectStorage decodes into it instead of a buffer it
/// would then copy.
pub(crate) struct PageWindow {
    view: View,
    within: usize,
    len: usize,
    /// Stable backing identity, even if the title releases its page numbers.
    run: (Slice, usize),
    #[cfg(feature = "paged-title")]
    _writer: Option<victory_memory::physical::access::Writer>,
}

impl Drop for PageWindow {
    fn drop(&mut self) {
        report_written(&shown_at(std::slice::from_ref(&self.run)));
    }
}

impl PageWindow {
    pub(crate) fn bytes(&mut self) -> &mut [u8] {
        let (within, len) = (self.within, self.len);
        &mut self.view.bytes()[within..within + len]
    }

    /// 失败时解码器或读入操作可能已修改窗口内的部分字节。
    #[cfg(all(feature = "paged-title", feature = "diagnostics"))]
    pub(crate) fn note_dstorage_fill(&self, produced: Option<usize>) {
        self.run.0.note_dstorage_write(
            self.within,
            produced.unwrap_or(self.len).min(self.len),
            produced.is_some_and(|len| len <= self.len),
        );
    }
}

/// A window on bytes `[offset, offset + len)` of `pages` when they are
/// consecutive pages of one allocation (how titles usually hand them out);
/// `None` otherwise (pages of several allocations, out of order, or
/// unknown, or a host failure), which `write_physical` then handles or
/// reports.
pub(crate) fn page_window(pages: &[usize], offset: usize, len: usize) -> Option<PageWindow> {
    let runs = runs(pages, offset, len)?;
    let [(slice, n)] = runs.as_slice() else {
        return None;
    };
    Some(PageWindow {
        view: View::map(slice, *n)?,
        within: offset % PAGE,
        len,
        run: (slice.clone(), *n),
        #[cfg(feature = "paged-title")]
        _writer: (len != 0).then(|| victory_memory::physical::access::writing(&[(slice.page(0), *n)])),
    })
}

/// Maps `n` pages at `address`, in place of what is mapped there; null
/// `pages` unmaps. Returns `address`, or null if a page or the host fails.
pub extern "system" fn XMemMapPhysicalPages(
    address: *mut c_void,
    n: usize,
    pages: *const usize,
) -> *mut c_void {
    if n == 0 {
        return address;
    }
    let base = address as usize;
    if base == 0 || !base.is_multiple_of(PAGE) {
        return std::ptr::null_mut();
    }
    let Some(len) = n.checked_mul(PAGE).filter(|&len| base.checked_add(len).is_some()) else {
        return std::ptr::null_mut();
    };
    #[cfg(feature = "paged-title")]
    let _views = TITLE_VIEW_GATE.write().unwrap_or_else(|e| e.into_inner());
    let mut physical = physical();
    if !unmap(&mut physical, base, len) {
        return std::ptr::null_mut();
    }
    if pages.is_null() {
        return address;
    }
    // SAFETY: the title passes `n` page numbers.
    let pages = unsafe { std::slice::from_raw_parts(pages, n) };
    let mut i = 0;
    while i < n {
        let Some(first) = physical.pages.get(&pages[i]).cloned() else {
            tracing::warn!(page = pages[i], "XMemMapPhysicalPages: unknown page");
            return std::ptr::null_mut();
        };
        // One view per run of consecutive slices of one section.
        let run = 1 + pages[i + 1..]
            .iter()
            .zip(1..)
            .take_while(|&(page, k)| {
                physical.pages.get(page).is_some_and(|s| {
                    Arc::ptr_eq(&s.section, &first.section) && s.index == first.index + k
                })
            })
            .count();
        if !map(&mut physical, base + i * PAGE, run * PAGE, first) {
            return std::ptr::null_mut();
        }
        i += run;
    }
    address
}

/// One page at each address; null `pages`, or a page number 0, unmaps
/// (Grand Theft Auto VI passes 0 for addresses to clear).
pub extern "system" fn XMemMapPhysicalPagesScatter(
    addresses: *const *mut c_void,
    n: usize,
    pages: *const usize,
) -> BOOL {
    if addresses.is_null() {
        return 0;
    }
    for i in 0..n {
        // SAFETY: the title passes `n` addresses and, unless null, `n` pages.
        let (address, page) = unsafe {
            (
                *addresses.add(i),
                if pages.is_null() || *pages.add(i) == 0 {
                    std::ptr::null()
                } else {
                    pages.add(i)
                },
            )
        };
        if XMemMapPhysicalPages(address, 1, page).is_null() {
            return 0;
        }
    }
    1
}

/// Maps the physical pages shown at `[from, from + len)` at `to` as well,
/// in place of what `to` shows; `from` 0 unmaps. The GPU's page-mapping
/// copies (`victory_gpu`, queue `CopyPageMappingsBatchX`): Grand Theft Auto
/// VI maps pages into a pool view, then has the GPU map them into its tiled
/// heap. Page by page; false if a source page is not mapped physical memory.
pub fn map_alias(to: u64, from: u64, len: u64) -> bool {
    let (to, from, len) = (to as usize, from as usize, len as usize);
    if len == 0 {
        return true;
    }
    if to == 0 || ![to, from, len].into_iter().all(|x| x.is_multiple_of(PAGE))
        || to.checked_add(len).is_none() || from.checked_add(len).is_none()
    {
        return false;
    }
    #[cfg(feature = "paged-title")]
    let _views = TITLE_VIEW_GATE.write().unwrap_or_else(|e| e.into_inner());
    let mut physical = physical();
    if !unmap(&mut physical, to, len) {
        return false;
    }
    for k in (0..len).step_by(PAGE) {
        if from == 0 {
            break;
        }
        let at = from + k;
        let Some(slice) = physical
            .views
            .range(..=at)
            .next_back()
            .filter(|(base, (size, _))| at < *base + size)
            .map(|(base, (_, s))| Slice {
                index: s.index + (at - base) / PAGE,
                section: s.section.clone(),
            })
        else {
            return false;
        };
        if !map(&mut physical, to + k, PAGE, slice) {
            return false;
        }
    }
    true
}

/// Maps `slice` and the slices after it over `[base, base + len)`, which
/// must lie in one placeholder.
fn map(physical: &mut Physical, base: usize, len: usize, slice: Slice) -> bool {
    // SAFETY: splits the placeholder around the range (failing harmlessly
    // when the range is the whole placeholder), then maps the section
    // there; both act only on placeholders.
    let view = unsafe {
        VirtualFree(
            base as *mut c_void,
            len,
            MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
        );
        MapViewOfFile3(
            slice.section.handle,
            GetCurrentProcess(),
            base as *const c_void,
            (slice.index * PAGE) as u64,
            len,
            MEM_REPLACE_PLACEHOLDER,
            PAGE_READWRITE,
            std::ptr::null_mut(),
            0,
        )
    };
    if view.Value as usize != base {
        tracing::warn!(
            address = format_args!("{base:#x}"),
            len,
            error = %std::io::Error::last_os_error(),
            "mapping physical pages failed (not a placeholder?)"
        );
        return false;
    }
    if !slice.section.commit(slice.index, base, len) {
        // SAFETY: the view just mapped, back to a placeholder.
        if unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) } == 0 {
            tracing::error!(base, len, "failed to roll back an uncommitted physical view");
            // No usable mapping was published, but the view still holds backing.
            std::mem::forget(slice);
        }
        return false;
    }
    // New contents, read-write: the GPU side's copies no longer guard it.
    victory_gpu::title_memory_remapped(base, len);
    for k in 0..len / PAGE {
        let page = slice.page(k);
        let address = base + k * PAGE;
        victory_gpu::title_memory_aliased(address, physical.shown.addresses(page));
        physical.shown.mapped(page, address);
    }
    #[cfg(all(feature = "paged-title", feature = "diagnostics"))]
    let binding = (slice.section.first, slice.index);
    physical.views.insert(base, (len, slice));
    #[cfg(all(feature = "paged-title", feature = "diagnostics"))]
    victory_gpu::title_backing::note_xmem_section_mapping(binding.0, binding.1, base, len, true);
    true
}

/// Commits the section pages a view shows (once per page, whichever view).
fn commit_section_view(base: usize, len: usize) -> bool {
    if on_wine() {
        return true; // sections are committed whole there
    }
    // SAFETY: `[base, base + len)` is a view of a reserved section.
    if !unsafe { VirtualAlloc(base as *const c_void, len, MEM_COMMIT, PAGE_READWRITE) }.is_null() {
        return true;
    }
    commit_refused(len, "physical pages", std::io::Error::last_os_error());
    false
}

/// Reports memory the host would not give: at the Windows commit limit to
/// `crashes.log`, since the title usually faults soon after on the null it
/// gets; otherwise as a warning.
fn commit_refused(len: usize, what: &str, error: std::io::Error) {
    if error.raw_os_error() == Some(ERROR_COMMITMENT_LIMIT as i32) {
        let mut info = PERFORMANCE_INFORMATION {
            cb: size_of::<PERFORMANCE_INFORMATION>() as u32,
            ..Default::default()
        };
        // SAFETY: the buffer and size describe a writable PERFORMANCE_INFORMATION.
        let queried = unsafe { K32GetPerformanceInfo(&mut info, info.cb) } != 0;
        victory_loader::log_crash(&commit_limit_message(
            len,
            what,
            &error,
            queried.then_some(&info),
        ));
    } else {
        tracing::warn!(len, %error, "committing {what} failed");
    }
}

fn commit_limit_message(
    len: usize,
    what: &str,
    error: &std::io::Error,
    info: Option<&PERFORMANCE_INFORMATION>,
) -> String {
    let mut message = format!(
        "VIctory: Windows commit limit reached while committing {len} bytes of {what}: {error}.\n"
    );
    if let Some(info) = info {
        use std::fmt::Write;

        let committed = info.CommitTotal * info.PageSize / MIB;
        let limit = info.CommitLimit * info.PageSize / MIB;
        let _ = writeln!(
            message,
            "System commit: {committed} MiB of {limit} MiB in use."
        );
    }
    message.push_str(
        "Close other applications or increase the Windows paging file size, then retry.\n",
    );
    message
}

/// Returns `[base, base + len)` to placeholders; views reaching past it
/// are mapped again outside it.
fn unmap(physical: &mut Physical, base: usize, len: usize) -> bool {
    // No bytes: no view to return. The search below would still find the
    // view holding `base` and cut it in two there, and its range of views
    // starting inside would run backwards (which `BTreeMap` panics on).
    if len == 0 {
        return true;
    }
    let end = base + len;
    let hit: Vec<usize> = physical.views_in(base, end).map(|(at, _)| at).collect();
    for at in hit {
        let size = physical.views[&at].0;
        // SAFETY: a live view we mapped; the OS must release it before
        // we retire its identity, aliases, and section lease.
        let unmapped = unsafe {
            UnmapViewOfFile2(
                GetCurrentProcess(),
                MEMORY_MAPPED_VIEW_ADDRESS { Value: at as *mut c_void },
                MEM_PRESERVE_PLACEHOLDER,
            )
        };
        if unmapped == 0 {
            tracing::warn!(at, size, error = %std::io::Error::last_os_error(), "unmapping physical pages failed");
            return false;
        }
        let (_, slice) = physical.views.remove(&at).expect("live physical view");
        for k in 0..size / PAGE {
            physical.shown.unmapped(slice.page(k), at + k * PAGE);
        }
        victory_gpu::title_memory_remapped(at, size);
        // 整个原视图已解绑。切分后各片段只有重新映射成功才登记。
        #[cfg(all(feature = "paged-title", feature = "diagnostics"))]
        victory_gpu::title_backing::note_xmem_section_mapping(slice.section.first, slice.index, at, size, false);
        let left = at >= base || map(physical, at, base - at, slice.clone());
        let right = at + size <= end || map(physical, end, at + size - end, Slice {
            index: slice.index + (end - at) / PAGE, ..slice
        });
        if !left || !right {
            return false;
        }
    }
    true
}

/// The oldest layout we know; newer titles pass other flag values (Minecraft
/// 0x20000001) and read the same leading fields.
#[repr(C)]
#[derive(Debug, Default)]
pub struct WorkingSet {
    pub title_limit: usize,
    pub title_used: usize,
    pub tools_limit: usize,
    pub tools_used: usize,
    pub stack_used: u32,
    pub physical_pages: u32,
    pub gpu_optimal_limit: usize,
    pub gpu_optimal_used: usize,
    pub tools_gpu_optimal_limit: usize,
    pub tools_gpu_optimal_used: usize,
    pub kernel_pool_used: u32,
    pub kernel_pool_limit: u32,
}

/// The title's budget and our process's commit as what it uses. No tools
/// memory: a retail console.
// shortcut: all fields are filled whatever the flags ask for, and use is
// not split by bandwidth class (GPU-optimal use reads 0); track
// allocations per class if a title budgets from that field.
pub extern "system" fn XMemGetWorkingSetStatistics(_which: u32, out: *mut WorkingSet) -> BOOL {
    if out.is_null() {
        return 0;
    }
    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    let stats = WorkingSet {
        title_limit: victory_gpu::console::TITLE_MEMORY - TRANSFERRED.load(Ordering::Relaxed),
        title_used: counters.PagefileUsage,
        ..Default::default()
    };
    // SAFETY: checked non-null; the title passes room for the structure.
    unsafe { out.write(stats) };
    1
}

/// Hands title memory (standard, in 2 MiB pages) to the system. The other
/// transfers involve tools memory, which a retail console has none of.
pub extern "system" fn XMemTransferMemory(kind: u32, size: usize) -> BOOL {
    const TITLE_TO_SYSTEM: u32 = 0;
    tracing::info!(kind, size, "XMemTransferMemory");
    if size == 0 {
        return 1;
    }
    let Some(size) = size.checked_next_multiple_of(2 * MIB) else {
        return 0;
    };
    let standard = victory_gpu::console::TITLE_MEMORY;
    let moved = kind == TITLE_TO_SYSTEM
        && TRANSFERRED
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| {
                t.checked_add(size).filter(|&t| t <= standard)
            })
            .is_ok();
    moved.into()
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "xmem",
        dlls: &["xmem.dll"],
        exports: vec![
            hle_export!(XMemAlloc),
            hle_export!(XMemFree),
            hle_export!(XMemAllocDefault),
            hle_export!(XMemFreeDefault),
            hle_export!(XMemIsToolsMemAvailable),
            hle_export!(XMemAllocatePhysicalPages),
            hle_export!(XMemFreePhysicalPages),
            hle_export!(XMemMapPhysicalPages),
            hle_export!(XMemMapPhysicalPagesScatter),
            hle_export!(XMemVirtualAlloc),
            hle_export!(XMemVirtualQuery),
            hle_export!(XMemGetWorkingSetStatistics),
            hle_export!(XMemTransferMemory),
        ],
    }
}

#[cfg(all(test, feature = "paged-title"))]
mod snapshot_tests;

#[cfg(test)]
mod tests {

    #[cfg(all(feature = "paged-title", feature = "diagnostics"))]
    #[test]
    fn graphics_heap_provenance_tracks_only_live_allocation_bounds() {
        const ATTR: u64 = (128 << 16) | (8 << ALIGNMENT_SHIFT);
        let allocation = XMemAllocDefault(512, ATTR);
        assert!(!allocation.is_null());
        let start = allocation as usize;
        assert!(is_graphics_heap_span(start, start + 512));
        assert!(is_graphics_heap_span(start + 256, start + 512));
        assert!(!is_graphics_heap_span(start + 256, start + 513));
        assert!(!is_graphics_heap_span(start, start));
        XMemFreeDefault(allocation, ATTR);
        assert!(!is_graphics_heap_span(start, start + 512));
    }

    use super::*;

    fn query(address: *const c_void) -> BasicInformation {
        // SAFETY: plain data; zero is a valid value.
        let mut info: BasicInformation = unsafe { std::mem::zeroed() };
        // SAFETY: a pseudo handle needs no closing.
        let process = unsafe { GetCurrentProcess() };
        assert_eq!(XMemVirtualQuery(process, address, &mut info), 1);
        info
    }

    #[test]
    fn ordinary_ram_budget_tracks_native_4k_commit_and_rolls_back_refusal() {
        use victory_memory::budget::RAM_LIMIT;
        const SMALL: usize = 4096;
        let budget = HostBudget::default();
        let _section = budget.reserve_ram(RAM_LIMIT - 2 * SMALL as u64).unwrap();
        let mut p = PrivateMemory::default();
        let base = commit_private(&mut p, std::ptr::null(), PAGE, MEM_RESERVE, || {
            // SAFETY: fresh private reservation, committed below.
            unsafe { VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE, PAGE_READWRITE) }
        });
        assert!(!base.is_null());
        fn one(p: &mut PrivateMemory, at: *const c_void) -> *mut c_void {
            commit_private(p, at, SMALL, MEM_COMMIT, || {
                // SAFETY: one granule in the native reservation.
                unsafe { VirtualAlloc(at, SMALL, MEM_COMMIT, PAGE_READWRITE) }
            })
        }
        let second = (base as usize + SMALL) as *mut c_void;
        assert_eq!(one(&mut p, base), base);
        p.use_budget(&budget).unwrap(); // includes already-committed bytes
        assert_eq!(budget.used(Pool::Ram), RAM_LIMIT - SMALL as u64);
        assert_eq!(one(&mut p, base), base); // no double charge
        assert!(commit_private(&mut p, second, SMALL, MEM_COMMIT, std::ptr::null_mut).is_null());
        assert_eq!(budget.used(Pool::Ram), RAM_LIMIT - SMALL as u64);
        assert!(commit_private(&mut p, second, 2 * SMALL, MEM_COMMIT, || {
            panic!("must not call Windows on over-budget commit")
        }).is_null());
        assert_eq!(one(&mut p, second), second);
        assert_eq!(budget.used(Pool::Ram), RAM_LIMIT);
        assert_ne!(free_private(&mut p, second, SMALL, MEM_DECOMMIT), 0);
        assert_eq!(budget.used(Pool::Ram), RAM_LIMIT - SMALL as u64);
        assert_ne!(free_private(&mut p, base, 0, MEM_RELEASE), 0);
        assert_eq!(budget.used(Pool::Ram), RAM_LIMIT - 2 * SMALL as u64);
    }

    #[test]
    fn wrapped_private_decommit_does_not_retarget_captured_gpu_page() {
        let p = super::super::kernel::VirtualAlloc(
            std::ptr::null(), 2 * PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE,
        );
        assert!(!p.is_null());
        let va = p as usize;
        let mut before = BackingPins::default();
        pin_backing(&[(va, va + 2 * PAGE)], &mut before);
        let original = before.expected(va).unwrap();
        assert!(original.is_private());
        assert!(before.run_with_origin(va, va + PAGE).2);
        let adjacent = before.expected(va + PAGE).unwrap();
        assert_ne!(original, adjacent);
        let granule = (va + 4096) as *mut c_void;
        assert_ne!(super::super::kernel::VirtualFree(granule, 4096, MEM_DECOMMIT), 0);
        assert_eq!(super::super::kernel::VirtualAlloc(granule, 4096, MEM_COMMIT, PAGE_READWRITE), granule);
        let mut after = BackingPins::default();
        pin_backing(&[(va, va + 2 * PAGE)], &mut after);
        let fresh = after.expected(va).unwrap();
        assert!(fresh.is_private() && fresh != original);
        assert!(after.run_with_origin(va, va + PAGE).2);
        assert_eq!(after.expected(va + PAGE), Some(adjacent));
        read_backing_runs(va, va + PAGE, &before, &mut |_, at, len, same| {
            assert_eq!(at, None, "do not return bytes of the replacement page");
            assert_eq!(len, PAGE);
            assert!(!same);
        });
        assert!(!with_writeback_page(va, 1, &before, true, &mut |_| panic!("stale write")));
        assert!(with_writeback_page(va, 1, &after, true, &mut |target| {
            assert_eq!(target.address, va);
            assert!(target.original_va);
        }));
        assert_ne!(super::super::kernel::VirtualFree(p, 0, MEM_RELEASE), 0);
        let mut released = BackingPins::default();
        pin_backing(&[(va, va + PAGE)], &mut released);
        assert_ne!(released.expected(va), Some(fresh));
        assert!(!released.run_with_origin(va, va + PAGE).2);
    }

    #[test]
    fn alloc_honours_alignment_and_frees() {
        for (field, align) in [(0, 16), (4, 16), (0xc, 4096), (0x10, 64 * 1024)] {
            let attributes = field << ALIGNMENT_SHIFT | 1 << 29; // a graphics type
            let p = XMemAlloc(100, attributes);
            assert!(!p.is_null());
            assert_eq!(p as usize % align, 0, "field {field:#x}");
            // SAFETY: 100 bytes were allocated.
            unsafe { p.cast::<u8>().write_bytes(0xab, 100) };
            XMemFree(p, attributes);
        }
        let p = XMemAlloc(4096, ALLOC_32BIT);
        assert!(!p.is_null() && (p as usize) < 1 << 32);
        XMemFree(p, ALLOC_32BIT);
    }

    /// An aligned block costs the host heap its size and the alignment
    /// once: the block is the heap's own (16-aligned), not one the host
    /// allocator padded for the alignment again.
    #[test]
    fn aligned_blocks_are_padded_once() {
        use windows_sys::Win32::System::Memory::{GetProcessHeap, HeapSize};
        for (field, align) in [(0, 16usize), (0xc, 4096), (0x10, 64 * 1024)] {
            let attributes = field << ALIGNMENT_SHIFT;
            let p = XMemAllocDefault(1000, attributes);
            assert_eq!(p as usize % align, 0);
            // SAFETY: (both) the header below the block we just made.
            let [size, offset] = unsafe { p.cast::<u8>().sub(HEADER).cast::<[usize; 2]>().read() };
            assert_eq!(size, 1000 + align);
            assert!(HEADER <= offset && offset <= align, "{offset}");
            // SAFETY: the host block, a block of the process heap.
            let heap = unsafe { HeapSize(GetProcessHeap(), 0, p.cast::<u8>().sub(offset).cast()) };
            assert_eq!(heap, 1000 + align, "align {align}");
            XMemFreeDefault(p, attributes);
        }
    }

    #[test]
    fn virtual_alloc_meets_console_constraints() {
        const GPU_READ_WRITE: u32 = 0x2000;
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * MIB,
            PAGES_2M | MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | 7 << 12, // 40-bit addresses
            PAGE_READWRITE | GPU_READ_WRITE,
        );
        assert!(!p.is_null());
        assert_eq!(p as usize % (2 * MIB), 0);
        assert!((p as usize) < 1 << 40);
        let info = query(p);
        assert_eq!(info.host.State, MEM_COMMIT);
        assert_eq!(info.host.Protect, PAGE_READWRITE);
        assert_eq!(info.flags & GRAPHICS, GRAPHICS);
        // Write-watched, for the GPU side's copies.
        let (mut found, mut count, mut granularity) = ([0usize; 4], 4, 0);
        // SAFETY: room for `count` addresses.
        let watched = unsafe {
            windows_sys::Win32::System::Memory::GetWriteWatch(
                0,
                p,
                4 * MIB,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        };
        assert_eq!(watched, 0);
        // SAFETY: our own reservation.
        assert_ne!(private_virtual_free(p, 0, MEM_RELEASE), 0);
        assert_eq!(query(p).flags, 0);
        // No address field: above the 32-bit range and below 2^40, where
        // the console keeps it.
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            16 << 30,
            MEM_RESERVE,
            GRAPHICS,
            PAGE_NOACCESS,
        );
        assert!(!p.is_null() && (p as usize) >= 1 << 32, "{p:p}");
        assert!((p as usize) + (16 << 30) <= 1 << 40, "{p:p}");
        // SAFETY: our own reservation.
        assert_ne!(private_virtual_free(p, 0, MEM_RELEASE), 0);
    }

    #[test]
    fn transfers_shrink_the_title_budget() {
        let mut before = WorkingSet::default();
        assert_eq!(XMemGetWorkingSetStatistics(0x2000_0001, &mut before), 1);
        assert!(before.title_used > 0);
        assert_eq!(XMemTransferMemory(0, 128 * MIB - 1), 1);
        assert_eq!(XMemTransferMemory(2, MIB), 0, "no tools memory");
        let standard = victory_gpu::console::TITLE_MEMORY;
        assert_eq!(XMemTransferMemory(0, standard), 0, "over budget");
        let mut after = WorkingSet::default();
        XMemGetWorkingSetStatistics(0x2000_0001, &mut after);
        assert_eq!(before.title_limit - after.title_limit, 128 * MIB);
    }

    #[test]
    fn statistics_have_the_console_layout() {
        assert_eq!(size_of::<WorkingSet>(), 0x50);
        assert_eq!(size_of::<BasicInformation>(), 0x40);
        assert_eq!(std::mem::offset_of!(WorkingSet, gpu_optimal_limit), 0x28);
    }

    #[test]
    fn commit_limit_report_includes_request_and_system_commit() {
        let error = std::io::Error::from_raw_os_error(ERROR_COMMITMENT_LIMIT as i32);
        let info = PERFORMANCE_INFORMATION {
            CommitTotal: 512,
            CommitLimit: 768,
            PageSize: 4096,
            ..Default::default()
        };
        let report = commit_limit_message(4 * MIB, "physical pages", &error, Some(&info));
        assert!(report.contains("4194304 bytes of physical pages"));
        assert!(report.contains("2 MiB of 3 MiB in use"));
        let fallback = commit_limit_message(4 * MIB, "virtual memory", &error, None);
        assert!(fallback.contains("4194304 bytes of virtual memory"));
        for report in [&report, &fallback] {
            assert!(report.contains("os error 1455"));
            assert!(report.contains("Windows paging file size"));
        }
        assert!(!fallback.contains("System commit:"));
    }

    #[test]
    fn physical_pages_map_alias_and_unmap() {
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        assert_eq!(n, 3);
        // Packed into 21 bits as `number >> 4`, as the title keeps them.
        for p in pages {
            assert_eq!((p >> 4 & 0x1f_ffff) << 4, p);
        }
        // Reservations of 64 KiB pages, as titles make them for mapping.
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);

        // The same page at a second address shows the same bytes.
        assert_eq!(
            XMemMapPhysicalPages(at(b, 1), 1, pages[2..].as_ptr()),
            at(b, 1)
        );
        // SAFETY: both pages are mapped read-write.
        unsafe {
            at(a, 2).cast::<u8>().write(7);
            assert_eq!(at(b, 1).cast::<u8>().read(), 7);
            a.cast::<u8>().write(1);
        }
        // Each mapping knows the other (the GPU side's change tracking).
        let others = |p: *mut c_void| {
            let mut out = Vec::new();
            aliases(p as usize, PAGE, &mut out);
            out.into_iter().map(|(_, a)| a).collect::<Vec<_>>()
        };
        assert_eq!(others(at(a, 2)), [at(b, 1) as usize]);
        assert_eq!(others(at(b, 1)), [at(a, 2) as usize]);
        assert!(others(a).is_empty());
        // Writes into pages (DirectStorage's) show through their views.
        assert!(write_physical(&pages[2..], 1, &[9]));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { at(b, 1).cast::<u8>().add(1).read() }, 9);

        // Unmapping the middle page keeps its neighbours.
        assert_eq!(
            XMemMapPhysicalPages(at(a, 1), 1, std::ptr::null()),
            at(a, 1)
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(at(a, 2)).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { at(a, 2).cast::<u8>().read() }, 7);

        // A scattered page lands in the hole.
        let addresses = [at(a, 1)];
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, pages.as_ptr()),
            1
        );
        // SAFETY: page 0 is now mapped at both `a` and `a + PAGE`.
        assert_eq!(unsafe { at(a, 1).cast::<u8>().read() }, 1);
        // Page number 0 unmaps.
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, [0usize].as_ptr()),
            1
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        // The GPU's page-mapping copy: what `a + 2 pages` shows, at `b`.
        assert!(map_alias(b as u64, at(a, 2) as u64, PAGE as u64));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { b.cast::<u8>().read() }, 7);
        let mut both = others(at(a, 2));
        both.sort_unstable();
        assert_eq!(both, [b as usize, at(b, 1) as usize]);
        assert!(map_alias(b as u64, 0, PAGE as u64));
        assert_eq!(others(at(a, 2)), [at(b, 1) as usize], "unmapped views are forgotten");
        assert_eq!(query(b).host.State, MEM_RESERVE);
        assert!(!map_alias(b as u64, at(a, 1) as u64, PAGE as u64));
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
        assert_eq!(XMemIsToolsMemAvailable(), 0);
    }

    #[cfg(feature = "paged-title")]
    #[test]
    fn managed_xmem_alias_shares_section_not_live_private_memory() {
        use std::ptr::{null, null_mut};
        use windows_sys::Win32::System::Memory::MEM_PRIVATE;
        if on_wine() { return; } // Exercise the native Windows placeholder protocol.

        let alloc_va = |kind| XMemVirtualAlloc(null_mut(), PAGE, MEM_RESERVE | kind, 0, PAGE_READWRITE);
        let (pool, tiled, private) = (alloc_va(PAGES_64K), alloc_va(PAGES_64K), alloc_va(MEM_COMMIT));
        assert!(!pool.is_null() && !tiled.is_null() && !private.is_null());
        let pages = allocate(1);
        assert_eq!(XMemMapPhysicalPages(pool, 1, pages.as_ptr()), pool);

        const ATTR: u64 = (128 << 16) | (16 << ALIGNMENT_SHIFT);
        let heap = XMemAllocDefault(PAGE, ATTR);
        assert!(!heap.is_null() && (heap as usize).is_multiple_of(PAGE));
        assert!(is_graphics_heap_span(heap as usize, heap as usize + PAGE));
        assert_eq!(query(heap).host.Type, MEM_PRIVATE);
        let copy = |from: *mut c_void| map_alias(tiled as u64, from as u64, PAGE as u64);
        // Alignment is not enough to make the default graphics heap a pool source.
        assert!(!copy(heap));
        assert_eq!(query(tiled).host.State, MEM_RESERVE);
        assert!(copy(pool));

        // Check what the submission would pin, as well as actual shared bytes.
        let ranges = [pool, tiled, heap, private].map(|at| (at as usize, at as usize + PAGE));
        let mut pins = BackingPins::default();
        pin_backing(&ranges, &mut pins);
        let section_page = pins.expected(pool as usize).expect("mapped section backing");
        assert!(pins.section::<Section>(section_page).is_some());
        assert_eq!(pins.expected(tiled as usize), Some(section_page));
        assert_eq!(pins.expected(heap as usize), None);
        assert_ne!(pins.expected(private as usize), Some(section_page));
        // SAFETY: all four pointers cover a live read-write byte.
        unsafe {
            heap.cast::<u8>().write_volatile(0x21);
            pool.cast::<u8>().write_volatile(0x42);
            assert_eq!(tiled.cast::<u8>().read_volatile(), 0x42);
            tiled.cast::<u8>().write_volatile(0x68);
            assert_eq!(pool.cast::<u8>().read_volatile(), 0x68);
            assert_eq!(heap.cast::<u8>().read_volatile(), 0x21);
            private.cast::<u8>().write_volatile(0x34);
        }

        // Exercise both destination paths on a whole private reservation,
        // never on the host allocator's live heap block.
        assert!(!map_alias(private as u64, pool as u64, PAGE as u64));
        assert!(XMemMapPhysicalPages(private, 1, pages.as_ptr()).is_null());
        let info = query(private).host;
        assert_eq!(info.Type, MEM_PRIVATE);
        assert_eq!(info.State, MEM_COMMIT);
        // SAFETY: still the original committed private allocation.
        assert_eq!(unsafe { private.cast::<u8>().read_volatile() }, 0x34);

        XMemFreeDefault(heap, ATTR);
        for at in [pool, tiled] {
            assert_eq!(XMemMapPhysicalPages(at, 1, null()), at);
            // SAFETY: now an unmapped, whole placeholder owned by this test.
            assert_ne!(unsafe { VirtualFree(at, 0, MEM_RELEASE) }, 0);
        }
        assert_ne!(private_virtual_free(private, 0, MEM_RELEASE), 0);
        assert_eq!(XMemFreePhysicalPages(1, pages.as_ptr()), 1);
    }

    #[test]
    fn physical_backing_survives_page_number_free_until_all_views_close() {
        let pages = allocate(1);
        let backing = Arc::downgrade(&physical().pages[&pages[0]].section);
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(), PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 1, pages.as_ptr()), a);
        assert!(map_alias(b as u64, a as u64, PAGE as u64));
        let mut worker = page_window(&pages, 0, 8).expect("mapped worker view");
        assert_eq!(XMemFreePhysicalPages(1, pages.as_ptr()), 1);
        assert!(page_window(&pages, 0, 8).is_none());

        worker.bytes().fill(0x6c);
        drop(worker);
        // SAFETY: the views remain valid after freeing their page numbers.
        unsafe {
            assert_eq!(a.cast::<u8>().read(), 0x6c);
            assert_eq!(b.cast::<u8>().read(), 0x6c);
        }
        let mut others = Vec::new();
        aliases(a as usize, PAGE, &mut others);
        assert_eq!(others, [(a as usize, b as usize)]);

        assert!(map_alias(a as u64, 0, PAGE as u64));
        assert_ne!(backing.strong_count(), 0);
        assert!(map_alias(b as u64, 0, PAGE as u64));
        assert_eq!(backing.strong_count(), 0);
        // SAFETY: each unmapped view is now an owned placeholder.
        unsafe {
            assert_ne!(VirtualFree(a, 0, MEM_RELEASE), 0);
            assert_ne!(VirtualFree(b, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn submitted_physical_backing_survives_cpu_unmap_and_handle_free() {
        let pages = allocate(1);
        let backing = Arc::downgrade(&physical().pages[&pages[0]].section);
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(), PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 1, pages.as_ptr()), a);
        assert!(map_alias(b as u64, a as u64, PAGE as u64));
        let mut pins = BackingPins::default();
        let (a, b) = (a as usize, b as usize);
        // The ranges need not start at a page boundary. Aliases retain one lease.
        pin_backing(&[(a + 1, a + 2), (b, b + PAGE)], &mut pins);
        assert_eq!(backing.strong_count(), 4); // page handle, two mappings and one pin
        assert_eq!(XMemFreePhysicalPages(1, pages.as_ptr()), 1);
        assert!(map_alias(a as u64, 0, PAGE as u64));
        assert!(map_alias(b as u64, 0, PAGE as u64));
        assert_eq!(backing.strong_count(), 1);
        drop(pins); // the host submission does this only after fence + copy-back
        assert_eq!(backing.strong_count(), 0);
        // SAFETY: our two reservations, now placeholders.
        unsafe {
            assert_ne!(VirtualFree(a as *mut c_void, 0, MEM_RELEASE), 0);
            assert_ne!(VirtualFree(b as *mut c_void, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn native_reserved_section_first_commit_is_zero_but_pinned_remap_keeps_bytes() {
        use windows_sys::Win32::System::Memory::{FILE_MAP_READ, MapViewOfFile};
        if on_wine() { return; } // Wine uses SEC_COMMIT instead.
        const LEN: usize = 2 * PAGE;
        let slot = XMemVirtualAlloc(
            std::ptr::null_mut(), LEN, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE,
        );
        assert!(!slot.is_null());
        let base = slot as usize;
        let map_fresh = |prewrite: bool| {
            let pages = allocate(2);
            let section = physical().pages[&pages[0]].section.clone();
            // SAFETY: inspect a fresh pagefile section without committing or reading its bytes.
            let at = unsafe { MapViewOfFile(section.handle, FILE_MAP_READ, 0, 0, LEN) };
            assert!(!at.Value.is_null());
            let probe = View { at, len: LEN, section: Some(section) };
            for offset in [0, PAGE] {
                assert_eq!(query(probe.at.Value.wrapping_byte_add(offset)).host.State, MEM_RESERVE);
            }
            drop(probe);
            // A physical writer can be the first committer, before any title VA is published.
            let mut expected = vec![0u8; LEN];
            if prewrite {
                expected[PAGE - 1..=PAGE].copy_from_slice(&[3, 7]);
                assert!(write_physical(&pages, PAGE - 1, &[3, 7]));
            }
            assert_eq!(XMemMapPhysicalPages(slot, 2, pages.as_ptr()), slot);
            for offset in [0, PAGE] {
                assert_eq!(query(slot.wrapping_byte_add(offset)).host.State, MEM_COMMIT);
            }
            // SAFETY: the first guest mapping now shows two committed pages.
            let bytes = unsafe { std::slice::from_raw_parts(slot.cast::<u8>(), LEN) };
            assert_eq!(bytes.iter().zip(expected).position(|(&a, b)| a != b), None,
                "first guest map must expose any earlier physical write");
            pages
        };
        let pages = map_fresh(true);
        let mut writer = page_window(&pages, 0, LEN).expect("physical writer on original section");
        let mut pins = BackingPins::default();
        pin_backing(&[(base, base + LEN)], &mut pins);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
        assert_eq!(XMemMapPhysicalPages(slot, 2, std::ptr::null()), slot);
        for offset in [0, PAGE] {
            assert_eq!(query(slot.wrapping_byte_add(offset)).host.State, MEM_RESERVE);
        }

        let replacement = map_fresh(false);
        // The pin captures identity, not a byte version. Finish a physical write on the old
        // section AFTER the same VA has been rebound; its worker view then goes away as well.
        let marker: Vec<u8> = (0..LEN).map(|i| (1 + i % 251) as u8).collect();
        writer.bytes().copy_from_slice(&marker);
        drop(writer);
        // SAFETY: slot still maps the entire replacement; no writer targets it.
        let current = unsafe { std::slice::from_raw_parts(slot.cast::<u8>(), LEN) };
        assert_eq!(current.iter().position(|&b| b != 0), None);

        // Only the submission still owns the original section.
        let mut checked = 0;
        read_backing_runs(base, base + LEN, &pins, &mut |address, source, len, same| {
            assert!(!same);
            assert_eq!(address, base + checked);
            let source = source.expect("pinned original section");
            // SAFETY: the temporary section view covers this run for the callback's lifetime.
            let old = unsafe { std::slice::from_raw_parts(source as *const u8, len) };
            let expected = &marker[checked..checked + len];
            assert_eq!(old.iter().zip(expected).position(|(a, b)| a != b), None, "pinned remap");
            checked += len;
        });
        assert_eq!(checked, LEN);
        assert_eq!(XMemMapPhysicalPages(slot, 2, std::ptr::null()), slot);
        assert_eq!(XMemFreePhysicalPages(2, replacement.as_ptr()), 1);
        assert_ne!(private_virtual_free(slot, 0, MEM_RELEASE), 0);
    }

    #[test]
    fn submitted_byte_writeback_reaches_the_original_backing_after_remap() {
        let pages = allocate(2);
        let ptr = XMemVirtualAlloc(
            std::ptr::null_mut(), PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE,
        );
        assert!(!ptr.is_null());
        assert_eq!(XMemMapPhysicalPages(ptr, 1, pages[1..].as_ptr()), ptr);
        let address = ptr as usize;
        let mut pins = BackingPins::default();
        pin_backing(&[(address + 1, address + 2)], &mut pins);
        let expected = pins.expected(address + 1);
        assert!(expected.is_some());
        let write = &mut |target: victory_gpu::title_backing::WritebackTarget| {
            // SAFETY: this closure is allowed only for the captured writable address.
            unsafe { (target.address as *mut u8).write(7) };
        };
        assert!(with_writeback_page(address + 1, 1, &pins, false, write));
        assert_eq!(XMemMapPhysicalPages(ptr, 1, pages.as_ptr()), ptr);
        pin_backing(&[(address, address + PAGE)], &mut pins);
        assert_eq!(pins.expected(address + 1), expected);
        assert!(!with_writeback_page(address + 1, 1, &pins, false, write));
        assert!(!with_writeback_page(address + 1, 1, &BackingPins::default(), true, write));
        assert!(with_writeback_page(address + 1, 1, &pins, true, &mut |target| {
            assert!(!target.original_va);
            // SAFETY: the callback has a view of the original page, not the rebound address.
            unsafe {
                let byte = target.address as *mut u8;
                assert_eq!(byte.read(), 7);
                byte.write(13);
            }
        }));
        let mut refresh = [0; 1];
        read_backing_runs(address + 1, address + 2, &pins, &mut |base, source, len, same| {
            assert_eq!((base, len, same), (address + 1, 1, false));
            // SAFETY: the section mapping remains live for this callback.
            refresh[0] = unsafe { (source.expect("original backing") as *const u8).read() };
        });
        assert_eq!(refresh, [13]);
        // SAFETY: the rebound page belongs to this test and is still mapped.
        assert_eq!(unsafe { ptr.cast::<u8>().add(1).read() }, 0);
        assert_eq!(XMemMapPhysicalPages(ptr, 1, std::ptr::null()), ptr);
        assert!(!with_writeback_page(address + 1, 1, &pins, false, write));
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
        // No original handles or views remain; the submission still owns
        // the section. The byte delta must land there, not in the placeholder.
        let mut called = false;
        assert!(with_writeback_page(address + 1, 1, &pins, true, &mut |target| {
            assert!(!target.original_va);
            // SAFETY: the callback owns a writable byte in the original section.
            unsafe {
                let byte = target.address as *mut u8;
                assert_eq!(byte.read(), 13);
                byte.write(29);
            }
            called = true;
        }));
        assert!(called);
        assert!(with_writeback_page(address + 1, 1, &pins, true, &mut |target| {
            // SAFETY: original physical page is still backed by the pin.
            assert_eq!(unsafe { (target.address as *const u8).read() }, 29);
        }));
        // SAFETY: the one unmapped page is again our placeholder.
        assert_ne!(unsafe { VirtualFree(ptr, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn physical_worker_view_keeps_backing_after_handles_are_freed() {
        let pages = allocate(1);
        let backing = Arc::downgrade(&physical().pages[&pages[0]].section);
        let mut worker = page_window(&pages, 0, 1).expect("mapped worker view");
        assert_eq!(XMemFreePhysicalPages(1, pages.as_ptr()), 1);
        assert_ne!(backing.strong_count(), 0);
        worker.bytes()[0] = 0x42;
        drop(worker);
        assert_eq!(backing.strong_count(), 0);
    }

    #[cfg(feature = "paged-title")]
    fn deferred_test_section(budget: &HostBudget, numbers: &mut PageNumbers, pages: usize) -> Arc<Section> {
        let first = numbers.allocate(pages).unwrap();
        let backing = SectionBacking::Deferred(Mutex::new(DeferredRam::new(budget, pages).unwrap()));
        let size = u32::try_from(pages * PAGE).unwrap();
        // SAFETY: small unnamed pagefile section; Section owns the returned handle.
        let handle = unsafe {
            CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE | SEC_RESERVE,
                0, size, std::ptr::null())
        };
        assert!(!handle.is_null(), "{}", std::io::Error::last_os_error());
        Arc::new(Section { handle, first, pages, backing })
    }

    #[cfg(feature = "paged-title")]
    #[test]
    fn deferred_native_section_charges_at_commit_and_reuses_alias() {
        use victory_memory::budget::RAM_LIMIT;
        use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
        if on_wine() { return; }
        let budget = HostBudget::default();
        let baseline = RAM_LIMIT - 4 * PAGE as u64;
        let _used = budget.reserve_ram(baseline).unwrap();
        let blocker = budget.reserve_ram(PAGE as u64).unwrap();
        let section = deferred_test_section(&budget, &mut PageNumbers::default(), 4);
        assert_eq!(budget.used(Pool::Ram), baseline + PAGE as u64);
        let mut first = View::map_section(section.clone(), 0, 2).expect("commit pages [0,2)");
        first.bytes()[PAGE] = 0x42;
        let before = baseline + 3 * PAGE as u64;
        assert_eq!(budget.used(Pool::Ram), before);

        // [1,4) overlaps by one page: two fresh pages are needed, only one fits.
        // SAFETY: keep a second view alive to query Windows before touching any fresh page.
        let at = unsafe { MapViewOfFile(section.handle, FILE_MAP_WRITE, 0, PAGE as u32, 3 * PAGE) };
        assert!(!at.Value.is_null());
        let mut overlap = View { at, len: 3 * PAGE, section: Some(section.clone()) };
        let base = overlap.at.Value;
        assert_ne!(base, first.at.Value.wrapping_byte_add(PAGE));
        assert!(!section.commit(1, base as usize, overlap.len));
        assert_eq!(budget.used(Pool::Ram), before);
        assert_eq!(query(base).host.State, MEM_COMMIT);
        for offset in [PAGE, 2 * PAGE] {
            assert_eq!(query(base.wrapping_byte_add(offset)).host.State, MEM_RESERVE);
        }

        // Real release, unchanged hard line. Exactly the two missing pages fit now.
        drop(blocker);
        assert!(section.commit(1, base as usize, overlap.len));
        assert_eq!(budget.used(Pool::Ram), RAM_LIMIT);
        for offset in [0, PAGE, 2 * PAGE] {
            assert_eq!(query(base.wrapping_byte_add(offset)).host.State, MEM_COMMIT);
        }
        assert_eq!(overlap.bytes()[0], 0x42);
        overlap.bytes()[2 * PAGE] = 0x75;
        let mut alias = View::map_section(section.clone(), 3, 1).expect("alias at hard limit");
        assert_eq!(alias.bytes()[0], 0x75);
        drop((section, first, overlap));
        assert_eq!(budget.used(Pool::Ram), RAM_LIMIT);
        drop(alias);
        assert_eq!(budget.used(Pool::Ram), baseline);
    }

    #[cfg(feature = "paged-title")]
    #[test]
    fn deferred_native_host_refusal_rolls_back_only_fresh_pages() {
        use windows_sys::Win32::System::Memory::{FILE_MAP_READ, MapViewOfFile, UnmapViewOfFile};
        if on_wine() { return; }
        let budget = HostBudget::default();
        let section = deferred_test_section(&budget, &mut PageNumbers::default(), 2);
        let mut first = View::map_section(section.clone(), 0, 1).expect("commit first page");
        first.bytes()[0] = 0x64;

        // There is budget for the second page, but Windows cannot commit PAGE_READWRITE
        // through a read-only view. No synthetic OOM or exhausted system commit needed.
        // SAFETY: map exactly the two pages in our section; only query this view.
        let ro = unsafe { MapViewOfFile(section.handle, FILE_MAP_READ, 0, 0, 2 * PAGE) };
        assert!(!ro.Value.is_null());
        let committed = section.commit(0, ro.Value as usize, 2 * PAGE);
        let second_state = query(ro.Value.wrapping_byte_add(PAGE)).host.State;
        // SAFETY: our temporary view, no references into it.
        assert_ne!(unsafe { UnmapViewOfFile(ro) }, 0);
        assert!(!committed);
        assert_eq!(second_state, MEM_RESERVE);
        assert_eq!(budget.used(Pool::Ram), PAGE as u64);

        let mut retry = View::map_section(section.clone(), 0, 2).expect("retry with write access");
        assert_eq!(budget.used(Pool::Ram), 2 * PAGE as u64);
        assert_eq!(query(retry.at.Value.wrapping_byte_add(PAGE)).host.State, MEM_COMMIT);
        assert_eq!(retry.bytes()[0], 0x64);
        drop((section, first, retry));
        assert_eq!(budget.used(Pool::Ram), 0);
    }

    #[cfg(feature = "paged-title")]
    #[test]
    fn deferred_native_rebind_retries_at_same_va_after_budget_refusal() {
        use victory_memory::budget::RAM_LIMIT;
        if on_wine() { return; }
        let budget = HostBudget::default();
        let baseline = RAM_LIMIT - 2 * PAGE as u64;
        let _used = budget.reserve_ram(baseline).unwrap();
        let blocker = budget.reserve_ram(PAGE as u64).unwrap();
        let mut p = Physical::default();
        let original = deferred_test_section(&budget, &mut p.numbers, 1);
        let replacement = deferred_test_section(&budget, &mut p.numbers, 1);
        let mut held = View::map_section(original.clone(), 0, 1).expect("original worker view");
        held.bytes()[0] = 0x5a;
        // SAFETY: single private placeholder; map/unmap preserve it until the final release.
        let slot = unsafe {
            VirtualAlloc2(GetCurrentProcess(), std::ptr::null_mut(), PAGE,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, std::ptr::null_mut(), 0)
        };
        assert!(!slot.is_null());
        let base = slot as usize;
        assert!(map(&mut p, base, PAGE, Slice { section: original, index: 0 }));
        // SAFETY: a live writable alias of the worker view.
        assert_eq!(unsafe { slot.cast::<u8>().read() }, 0x5a);
        assert!(unmap(&mut p, base, PAGE));
        assert_eq!(budget.used(Pool::Ram), RAM_LIMIT);

        let next = Slice { section: replacement, index: 0 };
        assert!(!map(&mut p, base, PAGE, next.clone()));
        assert!(p.views.is_empty());
        assert!(p.shown.addresses(next.page(0)).is_empty());
        assert_eq!(query(slot).host.State, MEM_RESERVE);
        assert_eq!(budget.used(Pool::Ram), RAM_LIMIT);
        drop(blocker);
        // Reusing the exact slot also proves rollback restored the placeholder.
        assert!(map(&mut p, base, PAGE, next));
        assert_eq!(budget.used(Pool::Ram), RAM_LIMIT);
        assert_eq!(query(slot).host.State, MEM_COMMIT);
        // SAFETY: map committed the replacement at this exact address.
        unsafe {
            assert_eq!(slot.cast::<u8>().read(), 0);
            slot.cast::<u8>().write(0x33);
        }
        assert_eq!(held.bytes()[0], 0x5a);
        drop(held);
        assert_eq!(budget.used(Pool::Ram), baseline + PAGE as u64);
        assert!(unmap(&mut p, base, PAGE));
        assert_eq!(budget.used(Pool::Ram), baseline);
        // SAFETY: no views remain over the placeholder.
        assert_ne!(unsafe { VirtualFree(slot, 0, MEM_RELEASE) }, 0);
    }

    #[cfg(not(feature = "paged-title"))]
    #[test]
    fn physical_backing_refuses_requests_over_ram_cap() {
        let mut n = victory_memory::budget::RAM_LIMIT as usize / PAGE + 1;
        let original = 123usize;
        let mut pages = vec![original; n];
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 0);
        assert_eq!(n, 0);
        assert!(pages.iter().all(|&p| p == original));
    }

    #[test]
    fn mapping_no_pages_changes_nothing() {
        let pages = allocate(2);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        // SAFETY: both pages are mapped read-write.
        unsafe { a.cast::<u8>().write(5) };
        let second = a.wrapping_byte_add(PAGE);

        // A count of 0 at the start of a view, inside it and past it: with
        // pages, without (an unmap), and as the GPU's page-mapping copy.
        for at in [a, second, a.wrapping_byte_add(2 * PAGE)] {
            assert_eq!(XMemMapPhysicalPages(at, 0, pages.as_ptr()), at);
            assert_eq!(XMemMapPhysicalPages(at, 0, std::ptr::null()), at);
            assert!(map_alias(at as u64, a as u64, 0));
            assert!(map_alias(at as u64, 0, 0));
        }

        // The view is still the one view it was.
        let view = physical().views.get(&(a as usize)).map(|v| v.0);
        assert_eq!(view, Some(2 * PAGE));
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(second).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { a.cast::<u8>().read() }, 5);

        // A count of 1 still unmaps that page alone.
        assert_eq!(XMemMapPhysicalPages(second, 1, std::ptr::null()), second);
        assert_eq!(query(second).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// `n` pages of one new allocation.
    fn allocate(n: usize) -> Vec<usize> {
        let (mut pages, mut given) = (vec![0; n], n);
        assert_eq!(
            XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()),
            1
        );
        assert_eq!(given, n);
        pages
    }

    /// The bytes of one page, through a view of our own.
    fn page_bytes(page: usize) -> Vec<u8> {
        let slice = physical().pages.get(&page).cloned().expect("known page");
        View::map(&slice, 1).expect("mapped").bytes().to_vec()
    }

    #[test]
    fn runs_group_consecutive_pages_of_one_allocation() {
        let (a, b) = (allocate(3), allocate(1));
        let lengths = |pages: &[usize], offset, len| {
            runs(pages, offset, len).map(|r| r.iter().map(|&(_, n)| n).collect::<Vec<_>>())
        };
        assert_eq!(lengths(&a, 0, 3 * PAGE), Some(vec![3]));
        // Only the pages the bytes reach.
        assert_eq!(lengths(&a, PAGE + 5, 10), Some(vec![1]));
        assert_eq!(lengths(&a, PAGE - 1, 2), Some(vec![2]));
        assert_eq!(lengths(&a, 0, 0), Some(vec![]));
        // Out of order, or across allocations: one run each.
        assert_eq!(
            lengths(&[a[0], a[2], a[1]], 0, 3 * PAGE),
            Some(vec![1, 1, 1])
        );
        assert_eq!(lengths(&[a[1], a[2], b[0]], 0, 3 * PAGE), Some(vec![2, 1]));
        // A slice that follows on in another section starts a run.
        assert_eq!(lengths(&[b[0], a[1]], 0, 2 * PAGE), Some(vec![1, 1]));
        // Too few pages for the bytes, or an unknown one.
        assert_eq!(lengths(&a, 0, 3 * PAGE + 1), None);
        assert_eq!(lengths(&[a[0], 0x10], 0, 2 * PAGE), None);
        assert_eq!(lengths(&a, usize::MAX, 2), None);
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn writes_cross_runs_and_allocations() {
        let (a, b) = (allocate(2), allocate(1));
        let pages = [a[0], a[1], b[0]];
        let data: Vec<u8> = (0..2 * PAGE + 64).map(|i| (i * 7 % 251) as u8).collect();
        let offset = PAGE - 32;
        // Past the third page: the second run is too short.
        assert!(!write_physical(&pages, offset, &data));
        assert!(write_physical(&pages, offset, &data[..2 * PAGE]));
        let seen: Vec<u8> = pages.iter().flat_map(|&p| page_bytes(p)).collect();
        assert!(seen[..offset].iter().all(|&b| b == 0), "before the offset");
        assert_eq!(seen[offset..offset + 2 * PAGE], data[..2 * PAGE]);
        assert!(seen[offset + 2 * PAGE..].iter().all(|&b| b == 0), "after");
        assert!(
            !write_physical(&[a[0], 0x10], 0, &data[..PAGE + 1]),
            "unknown page"
        );
        assert_eq!(XMemFreePhysicalPages(2, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn page_windows_cover_one_run_in_place() {
        let (a, b) = (allocate(3), allocate(1));
        let mut window = page_window(&a, PAGE + 3, PAGE).expect("one run");
        assert_eq!(window.bytes().len(), PAGE);
        window.bytes().fill(0x5a);
        drop(window);
        let (second, third) = (page_bytes(a[1]), page_bytes(a[2]));
        assert!(second[..3].iter().all(|&b| b == 0));
        assert!(second[3..].iter().all(|&b| b == 0x5a));
        assert!(third[..3].iter().all(|&b| b == 0x5a));
        assert!(third[3..].iter().all(|&b| b == 0));
        // Several runs, or an unknown page: no window.
        assert!(page_window(&[a[0], b[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[a[1], a[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[0x10], 0, 1).is_none());
        // One page of several is still one run.
        assert!(page_window(&[a[0], b[0]], 0, PAGE).is_some());
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }
}
