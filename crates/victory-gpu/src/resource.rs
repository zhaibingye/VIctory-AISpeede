//! Heaps, committed and placed resources, and resource sizes.
//!
//! Heaps and resources are host memory and a resource's GPU address is its
//! CPU address, as on the console; textures also get a host copy, and
//! buffers a host copy when a draw reads them (`shadow`).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ffi::c_void;
use std::ptr::NonNull;

use victory_formats::texture;

use crate::object::{
    self, E_INVALIDARG, E_OUTOFMEMORY, HEAP, HRESULT, Iid, Object, RESOURCE, S_OK, Table, This,
    guid,
};

/// The interfaces every device child also answers for.
pub(crate) const PAGEABLE_IIDS: [Iid; 3] = [
    guid(
        0x63ee58fb,
        0x1268,
        0x4835,
        [0x86, 0xda, 0xf0, 0x08, 0xce, 0x62, 0xf0, 0xd6],
    ),
    guid(
        0x905db94b,
        0xa00c,
        0x4140,
        [0x9d, 0xf5, 0x2b, 0x64, 0xca, 0x9e, 0xa3, 0x57],
    ),
    guid(
        0xc4fec28f,
        0x7966,
        0x4e95,
        [0x9f, 0x94, 0xf4, 0x31, 0xcb, 0x56, 0xc3, 0xb8],
    ),
];
const HEAP_IIDS: [Iid; 4] = [
    guid(
        0x6b3b2502,
        0x6e51,
        0x45b3,
        [0x90, 0xee, 0x98, 0x84, 0x26, 0x5e, 0x8d, 0xf3],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];
const RESOURCE_IIDS: [Iid; 4] = [
    guid(
        0x696442be,
        0xa72e,
        0x4059,
        [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

const KIB_64: u64 = 64 << 10;
const MIB_4: u64 = 4 << 20;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeapProperties {
    kind: u32,
    cpu_pages: u32,
    pool: u32,
    creation_nodes: u32,
    visible_nodes: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeapDesc {
    size: u64,
    properties: HeapProperties,
    alignment: u64,
    flags: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResourceDesc {
    dimension: u32,
    alignment: u64,
    width: u64,
    height: u32,
    depth_or_array_size: u16,
    mip_levels: u16,
    format: u32,
    sample_count: u32,
    sample_quality: u32,
    layout: u32,
    flags: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AllocationInfo {
    size: u64,
    alignment: u64,
}

const _: () = assert!(size_of::<HeapProperties>() == 20);
const _: () = assert!(size_of::<HeapDesc>() == 48);
const _: () = assert!(size_of::<ResourceDesc>() == 56);

const BUFFER: u32 = 1;
const TEXTURE_3D: u32 = 4;

/// Bytes per element and the element's width in texels (4 for block
/// compression), by DXGI format number. `None` for 1-bit and video
/// formats. The console adds 116/117 (10-bit float color), 118-120 (16-bit
/// depth with 8-bit stencil, two planes: 3 bytes per texel), 189 (signed
/// 10-bit color), 190 (two 4-bit channels) and 191 (24/8 color).
pub(crate) fn element(format: u32) -> Option<(u64, u64)> {
    Some(match format {
        1..=4 => (16, 1),
        5..=8 => (12, 1),
        9..=22 => (8, 1),
        23..=47 | 67 | 87..=93 | 116 | 117 | 189 | 191 => (4, 1),
        118..=120 => (3, 1),
        48..=59 | 68 | 69 | 85 | 86 | 115 => (2, 1),
        60..=65 | 190 => (1, 1),
        70..=72 | 79..=81 => (8, 4),
        73..=78 | 82..=84 | 94..=99 => (16, 4),
        _ => return None,
    })
}

/// A texture's layout in console memory, as the console's layout library
/// gives it (`victory-hle`'s `xg` serves the same): the description's
/// layout field is `0x100 | swizzle mode`, 1 linear, 0 the library's pick.
pub(crate) fn texture_layout(desc: &ResourceDesc) -> Option<texture::Texture> {
    use texture::{Dimension, Swizzle, TextureDesc};
    const TEXTURE_1D: u32 = 2;
    const TEXTURE_2D: u32 = 3;
    const RENDER_TARGET: u32 = 1;
    const DEPTH: u32 = 2;
    let dimension = match desc.dimension {
        TEXTURE_1D => Dimension::Tex1d,
        TEXTURE_2D => Dimension::Tex2d,
        TEXTURE_3D => Dimension::Tex3d,
        _ => return None,
    };
    let swizzle = match desc.layout {
        0 if dimension == Dimension::Tex1d => Swizzle::LINEAR,
        0 if desc.flags & DEPTH != 0 => Swizzle::SW_64KB_Z_X,
        0 if desc.flags & RENDER_TARGET != 0 => Swizzle::SW_64KB_R_X,
        0 => Swizzle::SW_4KB_S,
        1 => Swizzle::LINEAR,
        l if l & !0xff == 0x100 => Swizzle(l & 0xff),
        _ => return None,
    };
    texture::Texture::new(&TextureDesc {
        dimension,
        width: u32::try_from(desc.width).ok()?,
        height: desc.height,
        depth_or_array: u32::from(desc.depth_or_array_size),
        mips: u32::from(desc.mip_levels),
        format: desc.format,
        samples: desc.sample_count.max(1),
        swizzle,
    })
    .ok()
}

/// Bytes and alignment a resource takes: buffers their width (64 KiB
/// aligned), textures their console layout (formats it does not know:
/// every mip of every array slice and sample, packed, 64 KiB aligned),
/// rounded up to the alignment.
fn footprint(desc: &ResourceDesc) -> Option<AllocationInfo> {
    if desc.dimension != BUFFER
        && let Some(t) = texture_layout(desc)
    {
        // As the console: the layout's own alignment unless more is asked
        // (4 KiB for 4 KiB modes, where PC says 64 KiB). Grand Theft Auto
        // VI keeps these sizes in its texture tables, and its streaming
        // asserts when they disagree with the streamed layout.
        let alignment = desc.alignment.max(t.alignment);
        if !alignment.is_power_of_two() {
            return None;
        }
        return Some(AllocationInfo {
            size: t.size.checked_next_multiple_of(alignment)?,
            alignment,
        });
    }
    let bytes = if desc.dimension == BUFFER {
        desc.width
    } else {
        let (element, block) = element(desc.format)?;
        let (depth, slices) = if desc.dimension == TEXTURE_3D {
            (u64::from(desc.depth_or_array_size), 1)
        } else {
            (1, u64::from(desc.depth_or_array_size))
        };
        let height = u64::from(desc.height);
        let mips = mip_count(desc);
        let mut chain = 0u64;
        for mip in 0..mips.min(64) {
            let w = (desc.width >> mip).max(1).div_ceil(block);
            let h = (height >> mip).max(1).div_ceil(block);
            let d = (depth >> mip).max(1);
            chain = chain.checked_add(w.checked_mul(h)?.checked_mul(d)?.checked_mul(element)?)?;
        }
        chain
            .checked_mul(slices)?
            .checked_mul(u64::from(desc.sample_count.max(1)))?
    };
    if bytes == 0 {
        return None;
    }
    let minimum = if desc.sample_count > 1 { MIB_4 } else { KIB_64 };
    let alignment = desc.alignment.max(minimum);
    if !alignment.is_power_of_two() {
        return None;
    }
    Some(AllocationInfo {
        size: bytes.checked_next_multiple_of(KIB_64)?,
        alignment,
    })
}

/// A format as the host has it. The console's 16-bit depth with stencil
/// (and its typeless views) has no PC format: the 24-bit ones stand in.
pub(crate) fn pc_format(format: u32) -> u32 {
    match format {
        118 => 45,
        119 => 46,
        120 => 47,
        other => other,
    }
}

/// Shared-exponent RGB is sampleable on PC but cannot be a D3D12 render
/// target. Keep the packed format in title memory and on ordinary sampled
/// textures. For GPU-rendered colour, use float16 RGBA to preserve range.
pub(crate) const fn target_format(format: u32) -> u32 {
    match format {
        67 => 10,
        other => other,
    }
}

/// Format for a view whose console target has a different host representation.
#[cfg(windows)]
pub(crate) fn pc_view_format(this: This, format: u32) -> u32 {
    if format == 67 && stands_in(this) { target_format(format) } else { format }
}

#[cfg(windows)]
pub(crate) fn stands_in(this: This) -> bool {
    if this.is_null() || resource(this).desc.format != 67 {
        return false;
    }
    // SAFETY: the returned host resource lives with the title resource.
    host_resource(this).is_some_and(|r| unsafe { r.GetDesc() }.Format.0 as u32 == target_format(67))
}

#[cfg(windows)]
pub(crate) fn copy_formats_match(dst: This, src: This) -> bool {
    if !stands_in(dst) && !stands_in(src) {
        return true;
    }
    // A raw footprint or a texture with a different host format cannot
    // carry the stand-in's eight-byte texels.
    let compatible = host_resource(dst).zip(host_resource(src))
        // SAFETY: Both resource interfaces are live for these read-only description queries.
        .is_some_and(|(a, b)| unsafe { a.GetDesc().Format == b.GetDesc().Format });
    if !compatible {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!("copy between packed RGB9E5 and its host render target needs format conversion"));
    }
    compatible
}

/// The typeless format of a format's family (itself if it has none).
pub(crate) fn family(format: u32) -> u32 {
    const FAMILIES: [(u32, u32); 22] = [
        (1, 4),
        (5, 8),
        (9, 14),
        (15, 18),
        (19, 22),
        (23, 25),
        (27, 32),
        (33, 38),
        (39, 43),
        (44, 47),
        (48, 52),
        (53, 59),
        (60, 64),
        (70, 72),
        (73, 75),
        (76, 78),
        (79, 81),
        (82, 84),
        (90, 91),
        (92, 93),
        (94, 96),
        (97, 99),
    ];
    match format {
        87 => 90,
        88 => 92,
        f => FAMILIES
            .iter()
            .find(|&&(first, last)| (first..=last).contains(&f))
            .map_or(f, |&(first, _)| first),
    }
}

/// sRGB formats and their linear twins.
pub(crate) const SRGB: [(u32, u32); 3] = [(29, 28), (91, 87), (93, 88)];

/// The format a view of a title texture must name when its host copy is
/// typeless (see [`DEPTH_FORMATS`], [`SRGB`]): the title's own; `None` if
/// the host copy is typed.
#[cfg(windows)]
pub(crate) fn view_format(this: This) -> Option<u32> {
    let own = pc_format(resource(this).desc.format);
    let host = host_resource(this)?;
    // SAFETY: a live resource.
    let format = unsafe { host.GetDesc() }.Format.0 as u32;
    (format != own && !stands_in(this)).then_some(own)
}

/// Depth formats with their typeless family: depth textures that shaders
/// read are made typeless on the host, which only allows views of
/// typeless depth resources.
pub(crate) const DEPTH_FORMATS: [(u32, u32); 4] = [(20, 19), (40, 39), (45, 44), (55, 53)];

/// Mip levels, the full chain for 0.
fn mip_count(desc: &ResourceDesc) -> u32 {
    let depth = if desc.dimension == TEXTURE_3D {
        u64::from(desc.depth_or_array_size)
    } else {
        1
    };
    match desc.mip_levels {
        0 => {
            64 - desc
                .width
                .max(u64::from(desc.height))
                .max(depth)
                .max(1)
                .leading_zeros()
        }
        n => u32::from(n),
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlacedFootprint {
    pub(crate) offset: u64,
    format: u32,
    width: u32,
    height: u32,
    depth: u32,
    row_pitch: u32,
}

impl PlacedFootprint {
    #[cfg(windows)]
    pub(crate) fn linear_page_view(self, buffer: This) -> Option<crate::page_texture::LinearTexture> {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
        let (base, size) = buffer_bytes(buffer)?;
        if base.is_null() || self.depth != 1 {
            return None;
        }
        let address = (base as u64).checked_add(self.offset)?;
        let texture = crate::page_texture::LinearTexture {
            address, width: self.width, height: self.height, row_pitch: self.row_pitch,
            format: DXGI_FORMAT(self.format as i32),
        };
        let end = texture.rows().ok()?.last()?.1;
        (end.checked_sub(base as u64)? <= size).then_some(texture)
    }


    /// The bytes it spans in a buffer: every row pitched but the last.
    pub(crate) fn bytes(&self) -> Option<u64> {
        let (element, block) = element(self.format)?;
        let lines = u64::from(self.height).div_ceil(block) * u64::from(self.depth);
        let last = u64::from(self.width).div_ceil(block) * element;
        Some(u64::from(self.row_pitch) * lines.saturating_sub(1) + last)
    }
}
const _: () = assert!(size_of::<PlacedFootprint>() == 32);

const NO_FOOTPRINT: PlacedFootprint = PlacedFootprint {
    offset: u64::MAX,
    format: 0,
    width: 0,
    height: 0,
    depth: 0,
    row_pitch: 0,
};

/// One subresource's upload footprint (offset 0), row count and unpadded row size.
fn subresource(desc: &ResourceDesc, index: u32) -> Option<(PlacedFootprint, u32, u64)> {
    const PITCH: u64 = 256;
    if desc.dimension == BUFFER {
        let footprint = PlacedFootprint {
            offset: 0,
            format: 0,
            width: u32::try_from(desc.width).ok()?,
            height: 1,
            depth: 1,
            row_pitch: u32::try_from(desc.width.next_multiple_of(PITCH)).ok()?,
        };
        return (index == 0).then_some((footprint, 1, desc.width));
    }
    let mips = mip_count(desc);
    let slices = if desc.dimension == TEXTURE_3D {
        1
    } else {
        u32::from(desc.depth_or_array_size).max(1)
    };
    let mip = index % mips;
    let plane = index / mips / slices;
    let depth_stencil = matches!(desc.format, 19..=22 | 44..=47);
    // Each plane contains all the mip levels and array slices. Depth and
    // stencil copies use separate formats and 512-byte row alignment.
    let (format, pitch_alignment) = match (depth_stencil, plane) {
        (true, 0) => (39, 512), // DXGI_FORMAT_R32_TYPELESS
        (true, 1) => (60, 512), // DXGI_FORMAT_R8_TYPELESS
        (false, 0) => (desc.format, PITCH),
        _ => return None,
    };
    let (element, block) = element(format)?;
    let width = (desc.width >> mip).max(1).next_multiple_of(block);
    let height = (u64::from(desc.height) >> mip)
        .max(1)
        .next_multiple_of(block);
    let depth = if desc.dimension == TEXTURE_3D {
        (u32::from(desc.depth_or_array_size) >> mip).max(1)
    } else {
        1
    };
    let row_size = width / block * element;
    let footprint = PlacedFootprint {
        offset: 0,
        format,
        width: u32::try_from(width).ok()?,
        height: u32::try_from(height).ok()?,
        depth,
        row_pitch: u32::try_from(row_size.next_multiple_of(pitch_alignment)).ok()?,
    };
    Some((footprint, u32::try_from(height / block).ok()?, row_size))
}

/// Places subresources at 512-byte boundaries in an upload buffer.
/// Row pitch is aligned to 512 bytes for depth/stencil planes and 256 bytes
/// otherwise. Unknown sizes give the documented all-ones values.
pub(crate) extern "system" fn copyable_footprints(
    _: This,
    desc: *const ResourceDesc,
    first: u32,
    count: u32,
    base: u64,
    layouts: *mut PlacedFootprint,
    rows: *mut u32,
    row_sizes: *mut u64,
    total: *mut u64,
) {
    const PLACEMENT: u64 = 512;
    // SAFETY: the title passes a description (null is tolerated).
    let desc = unsafe { desc.as_ref() };
    let mut end = base;
    let mut known = desc.is_some();
    for i in 0..count {
        let one = desc.and_then(|d| subresource(d, first + i));
        let (layout, row_count, row_size) = match one {
            Some((mut footprint, row_count, row_size)) => {
                footprint.offset = end.next_multiple_of(PLACEMENT);
                let pitch = u64::from(footprint.row_pitch);
                let lines = u64::from(row_count) * u64::from(footprint.depth);
                end = footprint.offset + pitch * lines.saturating_sub(1) + row_size;
                (footprint, row_count, row_size)
            }
            None => {
                known = false;
                (NO_FOOTPRINT, u32::MAX, u64::MAX)
            }
        };
        let i = i as usize;
        // SAFETY: each output is null or holds `count` entries.
        unsafe {
            if !layouts.is_null() {
                layouts.add(i).write(layout);
            }
            if !rows.is_null() {
                rows.add(i).write(row_count);
            }
            if !row_sizes.is_null() {
                row_sizes.add(i).write(row_size);
            }
        }
    }
    if !known {
        tracing::warn!(?desc, "no footprint for this resource");
    }
    if !total.is_null() {
        // SAFETY: checked non-null.
        unsafe { total.write(if known { end - base } else { u64::MAX }) };
    }
}

/// Device method: the footprint of resources placed one after another.
/// Returns a 16-byte value, through a hidden first argument. Invalid input
/// gives the documented size `u64::MAX`.
pub(crate) extern "system" fn allocation_info(
    out: *mut AllocationInfo,
    _: This,
    _visible: u32,
    count: u32,
    descs: *const ResourceDesc,
) -> *mut AllocationInfo {
    let descs = if descs.is_null() {
        &[][..]
    } else {
        // SAFETY: the title passes `count` descriptions.
        unsafe { std::slice::from_raw_parts(descs, count as usize) }
    };
    // SAFETY: the caller passes room for the value.
    unsafe { out.write(place(descs.iter(), |_, _| ())) };
    out
}

/// The newer description: the plain one, then a sampler feedback mip
/// region (width, height, depth).
#[repr(C)]
pub(crate) struct ResourceDesc1 {
    desc: ResourceDesc,
    mip_region: [u32; 3],
}
const _: () = assert!(size_of::<ResourceDesc1>() == 72);

/// Where one resource goes in [`allocation_info2`]'s placement.
#[repr(C)]
pub(crate) struct AllocationInfo1 {
    offset: u64,
    alignment: u64,
    size: u64,
}

/// Device method 0x360 (`ID3D12Device8`): [`allocation_info`] for newer
/// descriptions, also telling where each resource goes.
pub(crate) extern "system" fn allocation_info2(
    out: *mut AllocationInfo,
    _: This,
    _visible: u32,
    count: u32,
    descs: *const ResourceDesc1,
    each: *mut AllocationInfo1,
) -> *mut AllocationInfo {
    let descs = if descs.is_null() {
        &[][..]
    } else {
        // SAFETY: the title passes `count` descriptions.
        unsafe { std::slice::from_raw_parts(descs, count as usize) }
    };
    let total = place(descs.iter().map(|d| &d.desc), |i, info| {
        if !each.is_null() {
            // SAFETY: the title passes room for `count` entries.
            unsafe { each.add(i).write(info) };
        }
    });
    // SAFETY: the caller passes room for the value.
    unsafe { out.write(total) };
    out
}

/// Resources placed one after another: the whole footprint, telling `each`
/// where every resource goes. Unknown sizes give the documented `u64::MAX`.
fn place<'a>(
    descs: impl Iterator<Item = &'a ResourceDesc>,
    mut each: impl FnMut(usize, AllocationInfo1),
) -> AllocationInfo {
    let mut total = AllocationInfo {
        size: 0,
        alignment: KIB_64,
    };
    for (i, desc) in descs.enumerate() {
        let Some(one) = footprint(desc) else {
            tracing::warn!(?desc, "no size for this resource");
            each(
                i,
                AllocationInfo1 {
                    offset: u64::MAX,
                    alignment: 0,
                    size: u64::MAX,
                },
            );
            return AllocationInfo {
                size: u64::MAX,
                alignment: 0,
            };
        };
        tracing::debug!(?desc, ?one, "allocation info");
        let offset = total.size.next_multiple_of(one.alignment);
        each(
            i,
            AllocationInfo1 {
                offset,
                alignment: one.alignment,
                size: one.size,
            },
        );
        total.size = offset + one.size;
        total.alignment = total.alignment.max(one.alignment);
    }
    total
}

/// Zeroed host memory standing in for GPU memory.
pub(crate) struct Memory {
    ptr: NonNull<u8>,
    layout: Layout,
    /// Set when the memory came from the title's allocator, with these
    /// allocation attributes (it goes back there).
    attributes: Option<u64>,
}

pub type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
pub type FreeRoutine = extern "system" fn(*mut c_void, u64);
static ALLOCATOR: std::sync::OnceLock<(AllocRoutine, FreeRoutine)> = std::sync::OnceLock::new();

/// Committed resources get their memory from the title's general
/// allocator, as on the console, where titles hook it to place graphics
/// memory themselves (Grand Theft Auto VI places its back buffers in a
/// block of its own and checks they fill it). The runtime names it before
/// the device exists.
pub fn use_allocator(alloc: AllocRoutine, free: FreeRoutine) {
    let _ = ALLOCATOR.set((alloc, free));
}

/// Maps the physical pages behind one range of title memory at another,
/// `(to, from, len)` (`from` 0 unmaps); false on failure.
pub type PageMapper = fn(u64, u64, u64) -> bool;
static PAGE_MAPPER: std::sync::OnceLock<PageMapper> = std::sync::OnceLock::new();

/// The runtime's page mapper, for the queues' page-mapping copies (memory
/// is the runtime's, as on the console).
pub fn use_page_mapper(mapper: PageMapper) {
    let _ = PAGE_MAPPER.set(mapper);
}

pub(crate) fn page_mapper() -> Option<PageMapper> {
    PAGE_MAPPER.get().copied()
}

/// Allocation attributes for resource memory: allocator id, page size
/// (bits 14-15: 2 MiB pages from 2 MiB up, else 64 KiB), alignment as log2
/// (bits 24-28) and memory type (bits 29-32: write-combined for GPU-only
/// heaps, cacheable for the CPU-visible ones).
/// shortcut: the console runtime's own choice of page size and memory type
/// is not known; this is the plausible one.
fn resource_attributes(info: AllocationInfo, heap_kind: u32) -> u64 {
    const GRAPHICS_ALLOCATOR: u64 = 128 << 16;
    const DEFAULT_HEAP: u32 = 1;
    let pages: u64 = if info.size >= 2 << 20 { 2 } else { 1 };
    let memory: u64 = if heap_kind == DEFAULT_HEAP { 1 } else { 3 };
    let alignment = u64::from(info.alignment.max(1).trailing_zeros());
    GRAPHICS_ALLOCATOR | pages << 14 | alignment << 24 | memory << 29
}

impl Memory {
    /// A committed resource's memory: from the title's allocator if the
    /// runtime named one, else our own; zeroed either way.
    fn for_resource(info: AllocationInfo, heap_kind: u32) -> Result<Memory, HRESULT> {
        let Some(&(alloc, _)) = ALLOCATOR.get() else {
            return Memory::new(info);
        };
        let layout = Layout::from_size_align(info.size as usize, info.alignment as usize)
            .map_err(|_| E_INVALIDARG)?;
        let attributes = resource_attributes(info, heap_kind);
        let ptr =
            NonNull::new(alloc(layout.size(), attributes).cast::<u8>()).ok_or(E_OUTOFMEMORY)?;
        // SAFETY: the allocator handed out `size` bytes.
        unsafe { ptr.as_ptr().write_bytes(0, layout.size()) };
        Ok(Memory {
            ptr,
            layout,
            attributes: Some(attributes),
        })
    }

    /// `size` zeroed bytes aligned to `alignment` (a power of two).
    pub(crate) fn zeroed(size: u64, alignment: u64) -> Result<Memory, HRESULT> {
        Memory::new(AllocationInfo { size, alignment })
    }

    pub(crate) fn address(&self) -> u64 {
        self.ptr.as_ptr() as u64
    }

    fn new(info: AllocationInfo) -> Result<Memory, HRESULT> {
        let layout = usize::try_from(info.size)
            .ok()
            .zip(usize::try_from(info.alignment).ok())
            .filter(|&(size, _)| size > 0)
            .and_then(|(size, align)| Layout::from_size_align(size, align).ok())
            .ok_or(E_INVALIDARG)?;
        // SAFETY: the size is non-zero.
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) }).ok_or(E_OUTOFMEMORY)?;
        Ok(Memory {
            ptr,
            layout,
            attributes: None,
        })
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        match (self.attributes, ALLOCATOR.get()) {
            (Some(attributes), Some(&(_, free))) => free(self.ptr.as_ptr().cast(), attributes),
            // SAFETY: allocated in `new` with this layout.
            _ => unsafe { dealloc(self.ptr.as_ptr(), self.layout) },
        }
    }
}

#[repr(C)]
struct Heap {
    object: Object,
    desc: HeapDesc,
    memory: Memory,
}

pub(crate) extern "system" fn create_heap(
    _: This,
    desc: *const HeapDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(
        size = desc.size,
        kind = desc.properties.kind,
        flags = format_args!("{:#x}", desc.flags),
        "heap"
    );
    if desc.alignment != 0 && !desc.alignment.is_power_of_two() {
        return E_INVALIDARG;
    }
    let info = AllocationInfo {
        size: desc.size,
        alignment: desc.alignment.max(KIB_64),
    };
    let memory = match Memory::new(info) {
        Ok(memory) => memory,
        Err(hr) => return hr,
    };
    let heap = Box::new(Heap {
        object: Object::new(&HEAP_FUNCTIONS),
        desc,
        memory,
    });
    object::hand_out(HEAP, heap, iid, &HEAP_IIDS, out)
}

extern "system" fn heap_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(HEAP, iid, object::is(iid, &HEAP_IIDS).then_some(this), out)
}
extern "system" fn heap_release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}
extern "system" fn heap_desc(this: This, out: *mut HeapDesc) {
    if !out.is_null() {
        // SAFETY: `this` is a heap; the title passes room for a description.
        unsafe { out.write((*this.cast::<Heap>()).desc) };
    }
}

/// A resource's memory: its own, or a range of a heap it holds a reference
/// to.
enum Backing {
    Committed(Memory),
    Placed {
        heap: This,
        offset: u64,
    },
    /// Memory the title mapped itself, at this address.
    Address(*mut u8),
}

impl Drop for Backing {
    fn drop(&mut self) {
        if let Backing::Placed { heap, .. } = *self {
            heap_release(heap);
        }
    }
}

#[repr(C)]
struct Resource {
    object: Object,
    desc: ResourceDesc,
    heap: HeapProperties,
    heap_flags: u32,
    backing: Backing,
    /// Textures also live on the host GPU (decision 0007); buffers do not
    /// yet.
    #[cfg(windows)]
    host: Option<std::sync::Arc<HostTexture>>,
}

/// Charged certificate storage. A generation is recorded only after
/// compatible bytes were merged into these exact physical identities.
#[cfg(all(windows, feature = "paged-title"))]
struct ByteMirror {
    version: Option<u64>,
    ranges: Box<[(u64, u64)]>,
    pages: Box<[(usize, victory_memory::physical::PhysicalPage)]>,
    _charge: victory_memory::physical::Reservation,
}

/// None means the existing charged certificate matches these pins.
#[cfg(all(windows, feature = "paged-title"))]
pub(crate) struct PreparedByteMirror(Option<ByteMirror>);

#[cfg(all(windows, feature = "paged-title"))]
#[derive(Default)]
struct MapEpoch {
    active: usize,
}

/// A title texture's host copy and the state it is in. The title's barriers
/// name console states (looser than the host's), so we transition from this
/// one. Lists that use the texture hold it until they are reset: the title
/// may release a resource whose commands it recorded but not yet submitted
/// (console memory outlives the object; a host texture must outlive the
/// lists that name it).
///
/// One per texture memory (decision 0007, "One host texture per texture
/// memory"): title textures at the same address with the same shape and
/// element size share it, each viewing it in its own format.
#[cfg(windows)]
pub(crate) struct HostTexture {
    pub(crate) sparse: Option<crate::sparse::Sparse>,
    pub(crate) texture: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    pub(crate) state: std::sync::atomic::AtomicU32,
    /// Formats views may take besides the texture's own family: the host
    /// took them as castable formats when it made the texture.
    pub(crate) castable: Vec<u32>,
    /// A host copy or resolve wrote it: its contents are the host's, not
    /// title memory's (`crate::texture`).
    pub(crate) host_written: std::sync::atomic::AtomicBool,
    byte_exposure: std::sync::atomic::AtomicU8,
    #[cfg(feature = "paged-title")]
    committed_linear: std::sync::OnceLock<crate::page_texture::LinearTexture>,
    #[cfg(feature = "paged-title")]
    mapped_pointers: std::sync::Mutex<MapEpoch>,
    #[cfg(feature = "paged-title")]
    byte_mirror: std::sync::Mutex<Option<ByteMirror>>,
    pub(crate) native_authority: std::sync::Mutex<Option<victory_memory::authority::CompletedVersion>>,
    /// Its uploads from title memory (`crate::texture`), made on first use.
    pub(crate) staging: std::sync::Mutex<Option<std::sync::Arc<crate::texture::Staging>>>,
    /// A depth texture's writable twin (`crate::twin`), made with its first
    /// unordered access view; `None` inside if it has none.
    pub(crate) twin: std::sync::OnceLock<Option<crate::twin::Twin>>,
    /// Bytes it takes on the host GPU.
    bytes: u64,
    /// A list used it (counted once).
    used: std::sync::atomic::AtomicBool,
}

#[cfg(windows)]
impl HostTexture {
    fn expose_bytes(&self, route: victory_memory::authority::ByteExposure) {
        self.byte_exposure.fetch_or(route as u8, std::sync::atomic::Ordering::AcqRel);
    }

    pub(crate) fn byte_observation(&self) -> victory_memory::authority::Observation {
        victory_memory::authority::Observation::from_exposures(
            self.byte_exposure.load(std::sync::atomic::Ordering::Acquire),
        )
    }

    pub(crate) fn uses_unproven_alias_backing(&self) -> bool {
        use victory_memory::authority::ByteExposure::*;
        self.byte_exposure.load(std::sync::atomic::Ordering::Acquire)
            & ((SharedHeap as u8) | (ExternalAddress as u8)) != 0
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn exact_committed_bytes(&self) -> Option<crate::page_texture::LinearTexture> {
        if self.uses_unproven_alias_backing() {
            return None;
        }
        self.committed_linear.get().copied()
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn compatible_texture_source(&self) -> Option<crate::page_texture::LinearTexture> {
        use victory_memory::authority::Representation;
        let value = self.native_authority.lock().unwrap_or_else(|e| e.into_inner());
        if value.as_ref()?.version.representation != Representation::CompatibleBytes {
            return None;
        }
        self.exact_committed_bytes()
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn has_pointer_exposure(&self) -> bool {
        use victory_memory::authority::ByteExposure::{GuestGpuAddress, MappedPointer};
        let flags = self.byte_exposure.load(std::sync::atomic::Ordering::Acquire);
        flags & ((MappedPointer as u8) | (GuestGpuAddress as u8)) != 0
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn visible_linear_writeback(&self) -> Result<Option<crate::page_texture::LinearTexture>, crate::page_slabs::BatchError> {
        if !self.has_pointer_exposure() {
            return Ok(None);
        }
        self.exact_committed_bytes().map(Some)
            .ok_or(crate::page_slabs::BatchError::UnsupportedTextureMap)
    }

    /// Called under ordered ownership of both the CPU epoch and the
    /// direct queue. A live mapping must never be overwritten by another
    /// readback; require a certificate for the latest version and backing.
    #[cfg(feature = "paged-title")]
    fn ensure_cpu_bytes_locked(
        self: &std::sync::Arc<Self>,
        host: &crate::host::Host,
        ordered: &mut u64,
        mapped: &MapEpoch,
        layout: Option<crate::page_texture::LinearTexture>,
    ) -> Result<(), crate::page_slabs::BatchError> {
        use crate::page_slabs::BatchError;
        if self.completed_native_version().is_none() {
            return Ok(());
        }
        let layout = layout.ok_or(BatchError::UnsupportedTextureMap)?;
        if mapped.active != 0 {
            let rows: Vec<_> = layout.rows()?.collect();
            return self.reusable_byte_mirror(&rows).then_some(())
                .ok_or(BatchError::UnsupportedTextureMap);
        }
        let heap = crate::memory::heap().ok_or(BatchError::UnsupportedTextureMap)?;
        host.materialize_texture_bytes_locked(ordered, heap, self.clone(), layout)
    }

    #[cfg(feature = "paged-title")]
    fn prepare_pointer_bytes_locked(
        self: &std::sync::Arc<Self>, host: &crate::host::Host, ordered: &mut u64,
        mapped: &MapEpoch, layout: Option<crate::page_texture::LinearTexture>,
    ) -> Result<(), crate::page_slabs::BatchError> {
        if self.completed_native_version().is_none() && let Some(layout) = layout {
            return self.reserve_pointer_mirror(layout);
        }
        self.ensure_cpu_bytes_locked(host, ordered, mapped, layout)
    }

    /// A guest GPU address is also observable as a CPU pointer.
    #[cfg(feature = "paged-title")]
    fn publish_guest_address(self: &std::sync::Arc<Self>) -> bool {
        use victory_memory::authority::ByteExposure::GuestGpuAddress;
        let Some(host) = crate::host::host() else { return false; };
        let mapped = self.mapped_pointers.lock().unwrap_or_else(|e| e.into_inner());
        let mut ordered = host.direct_queue();
        if self.byte_exposure.load(std::sync::atomic::Ordering::Acquire) & GuestGpuAddress as u8 == 0 {
            if let Err(error) = self.prepare_pointer_bytes_locked(host, &mut ordered, &mapped, self.exact_committed_bytes()) {
                tracing::warn!(?error, "VNME could not materialize a title texture address");
                return false;
            }
            self.expose_bytes(GuestGpuAddress);
        }
        true
    }

    #[cfg(feature = "paged-title")]
    fn completed_native_version(&self) -> Option<u64> {
        use victory_memory::authority::Representation;
        self.native_authority.lock().unwrap_or_else(|e| e.into_inner())
            .as_ref().filter(|v| v.version.representation == Representation::NativeGpu)
            .map(|v| v.version.number)
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn prepare_byte_mirror(
        &self, ranges: &[(u64, u64)], pins: &victory_memory::physical::BackingPins,
    ) -> Result<PreparedByteMirror, crate::page_slabs::BatchError> {
        use crate::page_slabs::BatchError;
        {
            let saved = self.byte_mirror.lock().unwrap_or_else(|e| e.into_inner());
            if saved.as_ref().is_some_and(|m| m.ranges.as_ref() == ranges && !m.pages.is_empty()
                && m.pages.iter().all(|&(va, id)| pins.expected(va) == Some(id)))
            {
                return Ok(PreparedByteMirror(None));
            }
        }
        let mut pages = std::collections::BTreeMap::new();
        for &(start, end) in ranges {
            let mut page = start & !(victory_memory::PAGE_BYTES - 1);
            while page < end {
                let at = page as usize;
                let physical = pins.expected(at).ok_or(BatchError::NotCaptured(page))?;
                pages.insert(at, physical);
                page += victory_memory::PAGE_BYTES;
            }
        }
        if pages.is_empty() {
            return Err(BatchError::UnsupportedTextureMap);
        }
        let bytes = std::mem::size_of_val(ranges) as u64
            + pages.len() as u64 * std::mem::size_of::<(usize, victory_memory::physical::PhysicalPage)>() as u64;
        let charge = victory_memory::physical::HostBudget::process().reserve_ram(bytes)
            .map_err(|_| BatchError::Native(windows::Win32::Foundation::E_OUTOFMEMORY.into()))?;
        Ok(PreparedByteMirror(Some(ByteMirror {
            version: None, ranges: ranges.into(), pages: pages.into_iter().collect(), _charge: charge,
        })))
    }

    /// Reserve before the first pointer to compatible backing escapes,
    /// even when the first native GPU write has not happened yet.
    #[cfg(feature = "paged-title")]
    fn reserve_pointer_mirror(
        &self, target: crate::page_texture::LinearTexture,
    ) -> Result<(), crate::page_slabs::BatchError> {
        let ranges: Vec<_> = target.rows()?.collect();
        let mut probe = crate::title_backing::SubmissionBacking::default();
        for &(start, end) in &ranges {
            probe.include(start as usize, end as usize);
        }
        probe.flush();
        if let PreparedByteMirror(Some(mirror)) = self.prepare_byte_mirror(&ranges, probe.pins())? {
            *self.byte_mirror.lock().unwrap_or_else(|e| e.into_inner()) = Some(mirror);
        }
        Ok(())
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn publish_byte_mirror(&self, prepared: PreparedByteMirror) {
        let version = self.completed_native_version().unwrap_or_else(|| {
            crate::host::fail(format_args!("texture bytes published without completed native authority"))
        });
        let mut saved = self.byte_mirror.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(mirror) = prepared.0 {
            *saved = Some(mirror);
        }
        saved.as_mut().expect("reserved byte mirror").version = Some(version);
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn reusable_byte_mirror(&self, ranges: &[(u64, u64)]) -> bool {
        let Some(version) = self.completed_native_version() else { return false; };
        let mirror = self.byte_mirror.lock().unwrap_or_else(|e| e.into_inner());
        let Some(mirror) = mirror.as_ref().filter(|m| m.version == Some(version) && m.ranges.as_ref() == ranges && !m.pages.is_empty())
            else { return false; };
        // Query physical identity without allocating GPU residency or
        // reading/uploading contents from guest pages.
        let mut probe = crate::title_backing::SubmissionBacking::default();
        for &(start, end) in ranges {
            probe.include(start as usize, end as usize);
        }
        probe.flush();
        mirror.pages.iter().all(|&(at, id)| probe.pins().expected(at) == Some(id))
    }

    #[cfg(feature = "paged-title")]
    fn record_cpu_bytes(&self) {
        use victory_memory::{authority::{CompletedVersion, Representation, Version}, coherence::Fence};
        let mut completed = self.native_authority.lock().unwrap_or_else(|e| e.into_inner());
        let old = *completed;
        let version = Version::successor(old.map(|v| v.version), Representation::CompatibleBytes,
            self.byte_observation()).unwrap_or_else(|| crate::host::fail(format_args!("VNME version exhausted")));
        *completed = Some(CompletedVersion { version, fence: old.map_or(Fence(0), |v| v.fence) });
        self.host_written.store(false, std::sync::atomic::Ordering::Release);
        use victory_memory::authority::ByteExposure::{GuestGpuAddress, MappedPointer};
        let mut mirror = self.byte_mirror.lock().unwrap_or_else(|e| e.into_inner());
        let flags = self.byte_exposure.load(std::sync::atomic::Ordering::Acquire);
        if flags & ((MappedPointer as u8) | (GuestGpuAddress as u8)) == 0 {
            *mirror = None;
        } else if let Some(mirror) = mirror.as_mut() {
            mirror.version = None;
        }
    }

    /// A host texture made outside the title's resources (twins).
    pub(crate) fn new(
        texture: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
        state: u32,
        castable: Vec<u32>,
    ) -> Self {
        Self::with_backing(texture, state, castable, None)
    }

    fn with_backing(
        texture: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
        state: u32,
        castable: Vec<u32>,
        sparse: Option<crate::sparse::Sparse>,
    ) -> Self {
        // Sparse backing keeps its own physical-byte counters and
        // updates them only after a successful tile mapping.
        let bytes = if sparse.is_some() {
            0
        } else {
            crate::host::host().map_or(0, |h| {
                // SAFETY: plain queries on a live resource.
                unsafe { h.device.GetResourceAllocationInfo(0, &[texture.GetDesc()]) }.SizeInBytes
            })
        };
        TEXTURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        TEXTURE_BYTES.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
        HostTexture {
            sparse,
            texture,
            state: state.into(),
            castable,
            host_written: false.into(),
            byte_exposure: 0.into(),
            #[cfg(feature = "paged-title")]
            committed_linear: std::sync::OnceLock::new(),
            #[cfg(feature = "paged-title")]
            mapped_pointers: std::sync::Mutex::default(),
            #[cfg(feature = "paged-title")]
            byte_mirror: std::sync::Mutex::default(),
            native_authority: std::sync::Mutex::new(None),
            staging: std::sync::Mutex::new(None),
            twin: std::sync::OnceLock::new(),
            bytes,
            used: false.into(),
        }
    }

    /// A list uses it: counted the first time (diagnostics of how much of
    /// the host textures' memory the GPU ever touches).
    pub(crate) fn note_used(&self) {
        if let Some(sparse) = &self.sparse {
            sparse.note_used();
        } else if !self.used.swap(true, std::sync::atomic::Ordering::Relaxed) {
            USED.fetch_add(self.bytes, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Bytes of host textures some list used (while they live).
#[cfg(windows)]
static USED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Host textures alive and the bytes they take (the fps line's companion).
#[cfg(windows)]
static TEXTURES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(windows)]
static TEXTURE_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Title textures that found their memory's host texture made already.
#[cfg(windows)]
static SHARED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Bytes the title textures that shared would have taken in host textures
/// of their own.
#[cfg(windows)]
static SAVED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Title textures whose memory's host texture could not take them (more
/// uses, a format it cannot be viewed in): a new one replaced it.
#[cfg(windows)]
static REPLACED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Host textures alive, their bytes and the bytes of those lists used,
/// title textures that shared one (and the bytes that saved), those that
/// replaced their memory's.
#[cfg(windows)]
pub(crate) fn texture_counts() -> (u64, u64, u64, u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let (mapped, sparse_used) = crate::sparse::counts();
    (
        TEXTURES.load(Relaxed),
        TEXTURE_BYTES.load(Relaxed) + mapped,
        USED.load(Relaxed) + sparse_used,
        SHARED.load(Relaxed),
        SAVED.load(Relaxed),
        REPLACED.load(Relaxed),
    )
}

#[cfg(windows)]
impl Drop for HostTexture {
    fn drop(&mut self) {
        TEXTURES.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        TEXTURE_BYTES.fetch_sub(self.bytes, std::sync::atomic::Ordering::Relaxed);
        if *self.used.get_mut() {
            USED.fetch_sub(self.bytes, std::sync::atomic::Ordering::Relaxed);
        }
        #[cfg(feature = "diagnostics")]
        {
            use windows::core::Interface;
            life::note(self.texture.as_raw() as usize, life::LET_GO, 0, life::holders(&self.texture));
        }
    }
}

/// Diagnostics: the last events in host textures' lives, for
/// device-removed reports that find a fault inside a freed one (host
/// textures are named with their address, `@0x...`).
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) mod life {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    /// The title released the resource the texture copies.
    pub(crate) const RELEASED: u8 = 1;
    /// A view heap slot (the value) stopped viewing it.
    pub(crate) const SLOT: u8 = 2;
    /// The shared host copy (title resource and lists) let go of it.
    pub(crate) const LET_GO: u8 = 3;
    /// A submission started (value: its number; no texture).
    pub(crate) const SUBMITTED: u8 = 4;
    /// That submission's lists have run.
    pub(crate) const RAN: u8 = 5;

    /// (texture, time, kind, value, references left after the event).
    type Event = (usize, std::time::Instant, u8, u64, u32);
    static EVENTS: Mutex<VecDeque<Event>> = Mutex::new(VecDeque::new());
    const KEPT: usize = 1 << 18;

    /// The references `texture` has besides the one about to go.
    pub(crate) fn holders(texture: &windows::Win32::Graphics::Direct3D12::ID3D12Resource) -> u32 {
        use windows::core::Interface;
        // SAFETY: a live COM object; one reference taken and given back.
        unsafe {
            let raw = texture.as_raw();
            let unknown = &*(raw as *const *const windows::core::IUnknown_Vtbl);
            ((**unknown).AddRef)(raw);
            ((**unknown).Release)(raw).saturating_sub(1)
        }
    }

    /// `left`: the texture's references once the event's own is gone.
    pub(crate) fn note(texture: usize, kind: u8, value: u64, left: u32) {
        let mut events = EVENTS.lock().unwrap_or_else(|e| e.into_inner());
        if events.len() >= KEPT {
            events.pop_front();
        }
        events.push_back((texture, std::time::Instant::now(), kind, value, left));
    }

    /// What happened to the texture named `name` (`...@0x<address>`), with
    /// the submissions around it, times relative to now.
    pub(crate) fn of(name: &str) -> String {
        let Some(at) = name
            .rsplit_once("@0x")
            .and_then(|(_, a)| usize::from_str_radix(a.split(|c: char| !c.is_ascii_hexdigit()).next()?, 16).ok())
        else {
            return String::new();
        };
        let now = std::time::Instant::now();
        let events = EVENTS.lock().unwrap_or_else(|e| e.into_inner());
        let mut text = Vec::new();
        let mut last_submission = None;
        for &(t, when, kind, value, left) in events.iter() {
            let ago = now.duration_since(when).as_secs_f64() * 1e3;
            match kind {
                SUBMITTED | RAN => last_submission = Some((kind, value, ago)),
                _ if t == at => {
                    if let Some((k, n, a)) = last_submission.take() {
                        let what = if k == SUBMITTED { "started" } else { "ran" };
                        text.push(format!("(submission {n} {what} {a:.1} ms ago)"));
                    }
                    let what = match kind {
                        RELEASED => "the title released its resource".to_owned(),
                        SLOT => format!("view heap slot {value} stopped viewing it"),
                        _ => "the shared host copy let go".to_owned(),
                    };
                    let freed = match (kind, left) {
                        (SLOT, 0) => ", its last holder (freed after the submission running then)",
                        (LET_GO, 0) => ", FREED",
                        _ => "",
                    };
                    text.push(format!("{ago:.1} ms ago {what}{freed}"));
                }
                _ => {}
            }
        }
        if let Some((k, n, a)) = last_submission {
            let what = if k == SUBMITTED { "started" } else { "ran" };
            text.push(format!("(last: submission {n} {what} {a:.1} ms ago)"));
        }
        format!(" [{name}: {}]", text.join("; "))
    }
}

/// Moves the host texture to `wanted` (a host texture state); the host
/// resource and the state it leaves, if that is a change.
#[cfg(windows)]
pub(crate) fn host_transition(this: This, wanted: u32) -> Option<(*mut c_void, u32)> {
    use windows::core::Interface;
    let host = resource(this).host.as_ref()?;
    let before = host
        .state
        .swap(wanted, std::sync::atomic::Ordering::Relaxed);
    (before != wanted).then(|| (host.texture.as_raw(), before))
}

/// The host GPU's copy of a title resource, if it has one.
#[cfg(windows)]
pub(crate) fn host_resource<'a>(
    this: This,
) -> Option<&'a windows::Win32::Graphics::Direct3D12::ID3D12Resource> {
    if this.is_null() {
        return None;
    }
    resource(this).host.as_deref().map(|h| &h.texture)
}

/// A title texture's host copy, shared, for a list to hold.
#[cfg(windows)]
pub(crate) fn host_copy(this: This) -> Option<std::sync::Arc<HostTexture>> {
    if this.is_null() {
        return None;
    }
    resource(this).host.clone()
}

/// The host description of a title texture: the same, without the
/// console's layout and flags, in a format the host takes for its use.
#[cfg(windows)]
fn host_desc(desc: &ResourceDesc) -> Option<ResourceDesc> {
    const PC_FLAGS: u32 = 0xf; // render target, depth, UAV, no shader access
    if desc.dimension == BUFFER {
        return None;
    }
    let mut d = *desc;
    d.layout = 0;
    d.alignment = 0;
    d.flags &= PC_FLAGS;
    // PC refuses depth targets that are also unordered-access (the console
    // allows both; Grand Theft Auto VI's shadow maps and material depth ask
    // for it): unordered views of such a texture write its twin
    // (`crate::twin`). It refuses unordered access on multisampled textures
    // too.
    if d.flags & DEPTH != 0 || d.sample_count > 1 {
        d.flags &= !UNORDERED;
    }
    d.format = pc_format(d.format);
    if d.flags & 0x1 != 0 {
        d.format = target_format(d.format);
    }
    const DENY_SHADER_RESOURCE: u32 = 8;
    if d.flags & DENY_SHADER_RESOURCE == 0
        && let Some(&(_, typeless)) = DEPTH_FORMATS.iter().find(|&&(f, _)| f == d.format)
    {
        d.format = typeless;
    }
    // The host has no sRGB unordered access: such textures are typeless,
    // their unordered access views plain UNORM.
    if d.flags & UNORDERED != 0 && SRGB.iter().any(|&(srgb, _)| srgb == d.format) {
        d.format = family(d.format);
    }
    Some(d)
}

const DEPTH: u32 = 2;
#[cfg(windows)]
const UNORDERED: u32 = 4;
/// Render target, depth, unordered access: what a host texture allows.
#[cfg(windows)]
const USES: u32 = 0x7;

/// What a texture memory is, for the one host texture it gets: its address
/// and shape (the console's layout resolved, mips counted, and the bytes
/// and texels of an element), and whether it is a depth texture (which
/// cannot share with colour ones on the host).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct MemoryKey {
    address: u64,
    dimension: u32,
    width: u64,
    height: u32,
    depth_or_array: u16,
    mips: u32,
    samples: u32,
    swizzle: u32,
    element: (u64, u64),
    depth: bool,
}

/// The texture memory `desc` describes at `address`; `None` for buffers
/// and formats without a known element.
pub(crate) fn memory_key(address: u64, desc: &ResourceDesc) -> Option<MemoryKey> {
    if desc.dimension == BUFFER {
        return None;
    }
    Some(MemoryKey {
        address,
        dimension: desc.dimension,
        width: desc.width,
        height: desc.height,
        depth_or_array: desc.depth_or_array_size,
        mips: mip_count(desc),
        samples: desc.sample_count.max(1),
        swizzle: texture_layout(desc).map_or(desc.layout, |t| t.planes[0].layout.desc.swizzle.0),
        element: element(desc.format)?,
        depth: desc.flags & DEPTH != 0,
    })
}

/// The formats a host texture of `format` is made castable to: every
/// format of the same element (bytes and block texels), as the host's
/// format list casting allows (uncompressed among uncompressed, block
/// formats among block formats of the same block size), but depth ones
/// (refused in lists on the RX 6700 XT and WARP alike) and typeless ones
/// (no view takes them).
pub(crate) fn castable(format: u32) -> Vec<u32> {
    const DEPTH_FAMILIES: [u32; 10] = [19, 20, 21, 22, 40, 44, 45, 46, 47, 55];
    let Some(e) = element(format) else {
        return Vec::new();
    };
    (1..=115)
        .filter(|&f| f != format && element(f) == Some(e) && !DEPTH_FAMILIES.contains(&f))
        .filter(|&f| !(family(f) == f && family(f + 1) == f))
        .collect()
}

/// Live host textures by texture memory: title textures (resource objects
/// and placed views) of one memory share one.
#[cfg(windows)]
static MEMORY: std::sync::Mutex<
    Option<std::collections::HashMap<MemoryKey, std::sync::Weak<HostTexture>>>,
> = std::sync::Mutex::new(None);

/// The host texture of the memory at `address` that `desc` describes: the
/// live one if it takes `desc` (its uses, a format it can be viewed in),
/// else a new one, which becomes the memory's.
#[cfg(windows)]
fn shared_texture(
    address: u64,
    desc: &ResourceDesc,
    state: u32,
    clear: *const c_void,
    sparse_prefix: u32,
) -> Option<std::sync::Arc<HostTexture>> {
    let d = host_desc(desc)?;
    let key = memory_key(address, desc);
    let mut map = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(Default::default);
    let live = key.and_then(|k| map.get(&k)?.upgrade());
    if live.is_none() && key.is_some() && sparse_prefix != 0 {
        use windows::Win32::Graphics::Direct3D12::*;
        // SAFETY: host_desc normalizes the repr(C) resource description.
        let mut pc = unsafe { std::ptr::from_ref(&d).cast::<D3D12_RESOURCE_DESC>().read_unaligned() };
        // A placed view may select sRGB or linear later. A BC1 typeless
        // resource supports both without extra physical allocations.
        pc.Format = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_BC1_TYPELESS;
        let initial = crate::host::legal_for(d.flags, crate::host::texture_state(state));
        if let Some((texture, sparse)) = crate::sparse::Sparse::try_new(
            pc, D3D12_RESOURCE_STATES(initial as i32), sparse_prefix,
        ) {
            return Some(std::sync::Arc::new(HostTexture::with_backing(
                texture, initial, Vec::new(), Some(sparse),
            )));
        }
    }
    if let Some(t) = &live
        && takes(t, &d)
    {
        SHARED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Some(host) = crate::host::host() {
            // The live texture's own description: `d` may ask for uses its
            // format does not take on this host (unordered access on
            // B5G5R5A1 on NVIDIA), which the query rejects by removing the
            // device. SAFETY: plain queries on a live resource.
            let info = unsafe { host.device.GetResourceAllocationInfo(0, &[t.texture.GetDesc()]) };
            SAVED.fetch_add(info.SizeInBytes, std::sync::atomic::Ordering::Relaxed);
        }
        return live;
    }
    // shortcut: title textures of a replaced host texture keep it (placed
    // views move to the new one, `texture_at`); copies (`alias_source`)
    // bridge the two where the shapes allow.
    let t = std::sync::Arc::new(make_texture(&d, state, clear)?);
    if let Some(k) = key {
        if live.is_some() {
            REPLACED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            tracing::debug!(?k, "a texture memory's host texture replaced by one with more uses or another format");
        }
        if map.len().is_power_of_two() {
            map.retain(|_, w| w.strong_count() > 0);
        }
        map.insert(k, std::sync::Arc::downgrade(&t));
    }
    Some(t)
}

/// Whether host texture `t` takes a title texture of host description
/// `d` at its memory: its uses, and a view format of its family or
/// castable list.
#[cfg(windows)]
fn takes(t: &HostTexture, d: &ResourceDesc) -> bool {
    // SAFETY: a live resource.
    let td = unsafe { t.texture.GetDesc() };
    let (flags, own) = (td.Flags.0 as u32, td.Format.0 as u32);
    d.flags & USES & !flags == 0
        && (family(d.format) == family(own)
            || t.castable.iter().any(|&c| family(c) == family(d.format)))
}

/// Whether `this`'s host texture is still its memory's (not replaced by one
/// with more uses).
#[cfg(windows)]
fn current(this: This) -> bool {
    let r = resource(this);
    let (Some(h), Some(k)) = (&r.host, memory_key(r.data() as u64, &r.desc)) else {
        return true;
    };
    // The private placed allocation is also the identity of already-made
    // descriptors. An ordinary resource appearing later cannot replace it;
    // the same-address GPU-writable alias path copies into it instead.
    if h.sparse.is_some() {
        return true;
    }
    let map = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    map.as_ref()
        .and_then(|m| m.get(&k)?.upgrade())
        .is_none_or(|live| std::sync::Arc::ptr_eq(&live, h))
}

/// Whether the host has `format` at all. A castable list naming one it does
/// not have is refused whole (vkd3d-proton has no
/// `R10G10B10_XR_BIAS_A2_UNORM`), and with it every castable texture.
#[cfg(windows)]
fn host_has(host: &crate::host::Host, format: u32) -> bool {
    use std::sync::atomic::{AtomicU8, Ordering::Relaxed};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
    // Per format: 0 not asked yet, 1 had, 2 not.
    static HAS: [AtomicU8; 128] = [const { AtomicU8::new(0) }; 128];
    let Some(known) = HAS.get(format as usize) else {
        return false;
    };
    match known.load(Relaxed) {
        1 => return true,
        2 => return false,
        _ => {}
    }
    let mut data = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: DXGI_FORMAT(format as i32),
        ..Default::default()
    };
    // SAFETY: the query writes its own structure of the given size.
    let asked = unsafe {
        host.device.CheckFeatureSupport(
            D3D12_FEATURE_FORMAT_SUPPORT,
            (&raw mut data).cast(),
            size_of_val(&data) as u32,
        )
    };
    let has = asked.is_ok() && data.Support1 != D3D12_FORMAT_SUPPORT1_NONE;
    known.store(if has { 1 } else { 2 }, Relaxed);
    has
}

/// A host texture of host description `d` in GPU memory, castable to the
/// formats of [`castable`] the host has (colour textures, through the
/// enhanced-barrier creation; it starts in the common state) where the host
/// takes that.
#[cfg(windows)]
fn make_texture(d: &ResourceDesc, state: u32, clear: *const c_void) -> Option<HostTexture> {
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
    use windows::core::Interface;
    let host = crate::host::host()?;
    #[cfg(test)]
    if REFUSE_NEXT_TEXTURE.with(|refuse| refuse.replace(false)) {
        note_texture_pressure();
        return None;
    }
    // The clear value names the format too: format, then color or depth
    // and stencil (5 words).
    let clear_value = (!clear.is_null()).then(|| {
        // SAFETY: the title passes a whole clear value.
        let mut value = unsafe { clear.cast::<[u32; 5]>().read_unaligned() };
        value[0] = pc_format(value[0]);
        if d.flags & 0x1 != 0 {
            value[0] = target_format(value[0]);
        }
        value
    });
    let clear_value = clear_value.as_ref().map(|v| std::ptr::from_ref(v).cast::<D3D12_CLEAR_VALUE>());
    // SAFETY: the host-compatible description layout is checked.
    let p = unsafe { std::ptr::from_ref(d).cast::<D3D12_RESOURCE_DESC>().read_unaligned() };
    // SAFETY: request the allocation size for this single live description.
    let bytes = unsafe { host.device.GetResourceAllocationInfo(0, &[p]) }.SizeInBytes;
    let mut castable_formats = if d.flags & DEPTH == 0 { castable(d.format) } else { Vec::new() };
    castable_formats.retain(|&f| host_has(host, f));
    let mut initial = crate::host::legal_for(d.flags, crate::host::texture_state(state));
    let texture = host.managed_gpu(bytes, |heap| {
        let mut texture: Option<ID3D12Resource> = None;
        if !castable_formats.is_empty()
            && let Ok(device) = host.device.cast::<ID3D12Device10>()
        {
            let desc1 = D3D12_RESOURCE_DESC1 {
                Dimension: p.Dimension, Alignment: p.Alignment, Width: p.Width,
                Height: p.Height, DepthOrArraySize: p.DepthOrArraySize,
                MipLevels: p.MipLevels, Format: p.Format, SampleDesc: p.SampleDesc,
                Layout: p.Layout, Flags: p.Flags, SamplerFeedbackMipRegion: Default::default(),
            };
            let formats: Vec<DXGI_FORMAT> = castable_formats.iter().map(|&f| DXGI_FORMAT(f as i32)).collect();
            // SAFETY: initialized descriptions and our out pointer.
            let made = unsafe {
                device.CreateCommittedResource3(
                    heap, D3D12_HEAP_FLAG_NONE, &desc1, D3D12_BARRIER_LAYOUT_COMMON,
                    clear_value, None::<&ID3D12ProtectedResourceSession>, Some(&formats), &mut texture,
                )
            };
            if made.is_ok() && let Some(texture) = texture {
                initial = 0;
                return Ok(texture);
            }
            texture = None;
        }
        // SAFETY: initialized descriptions, our out pointer.
        unsafe {
            host.device.CreateCommittedResource(
                heap, D3D12_HEAP_FLAG_NONE, &p, D3D12_RESOURCE_STATES(initial as i32),
                clear_value, &mut texture,
            )?;
        }
        castable_formats.clear();
        texture.ok_or_else(windows::core::Error::empty)
    });
    let texture = match texture {
        Ok(texture) => texture,
        Err(e) => {
            if e.code() == windows::Win32::Foundation::E_OUTOFMEMORY {
                note_texture_pressure();
            }
            tracing::warn!(error = %e, desc = ?d, "no host texture");
            return None;
        }
    };
    crate::host::name(&texture, || {
        // Its address, for the lifetime notes (`life`) of a freed one.
        format!("texture {}x{} f{} flags {:#x} @{:#x}", d.width, d.height, d.format, d.flags, texture.as_raw() as usize)
    });
    let initial = crate::host::legal_state(&texture, initial);
    Some(HostTexture::new(texture, initial, castable_formats))
}

/// A texture's console memory and description; `None` for buffers.
pub(crate) fn texture_memory(this: This) -> Option<(*const u8, ResourceDesc)> {
    if this.is_null() {
        return None;
    }
    let r = resource(this);
    (r.desc.dimension != BUFFER).then(|| (r.data().cast_const(), r.desc))
}

/// Resolve only title-owned, explicitly linear, fully typed textures.
#[cfg(windows)]
#[allow(dead_code, reason = "used by paged descriptor admission")]
pub(crate) fn paged_linear_texture(this: This) -> Option<crate::page_texture::LinearTexture> {
    let (data, desc) = texture_memory(this)?;
    page_texture_layout(data as u64, &desc)
}

#[cfg(windows)]
fn page_texture_layout(address: u64, desc: &ResourceDesc) -> Option<crate::page_texture::LinearTexture> {
    if desc.flags & 0xf != 0 {
        return None;
    }
    linear_texture_bytes(address, desc)
}

#[cfg(windows)]
fn linear_texture_bytes(address: u64, desc: &ResourceDesc) -> Option<crate::page_texture::LinearTexture> {
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
    if desc.dimension != 3 || !matches!(desc.layout, 1 | 0x100)
        || desc.depth_or_array_size != 1 || desc.mip_levels != 1
        || desc.sample_count != 1
        || !matches!(desc.format, 28 | 87)
    {
        return None;
    }
    let layout = texture_layout(desc)?;
    let [plane] = layout.planes.as_slice() else { return None };
    let [mip] = plane.layout.mips.as_slice() else { return None };
    if mip.tail.is_some() || mip.depth != 1 {
        return None;
    }
    let input = crate::page_texture::LinearTexture {
        address: address.checked_add(plane.offset)?.checked_add(mip.offset)?,
        row_pitch: mip.pitch.checked_mul(4)?,
        width: u32::try_from(desc.width).ok()?,
        height: desc.height,
        format: DXGI_FORMAT(desc.format as i32),
    };
    let _validated_rows = input.rows().ok()?;
    Some(input)
}

#[cfg(all(windows, test))]
#[test]
fn paged_linear_texture_uses_title_layout_pitch() {
    let mut desc = ResourceDesc {
        dimension: 3, alignment: 0, width: 2, height: 2, depth_or_array_size: 1,
        mip_levels: 1, format: 28, sample_count: 1, sample_quality: 0, layout: 1, flags: 0,
    };
    let texture = page_texture_layout(0x10000, &desc).unwrap();
    assert_eq!(texture.row_pitch, 256);
    assert_eq!(texture.rows().unwrap().collect::<Vec<_>>(), [(0x10000, 0x10008), (0x10100, 0x10108)]);
    desc.layout = 0; // The default for 2D is tiled.
    assert!(page_texture_layout(0x10000, &desc).is_none());
    desc.layout = 1;
    desc.flags = 4; // A writable texture is not a derived source.
    assert!(page_texture_layout(0x10000, &desc).is_none());
}

/// A texture's format as the title gave it.
pub(crate) fn title_format(this: This) -> Option<u32> {
    (!this.is_null()).then(|| resource(this).desc.format)
}

/// A buffer's host memory and size; `None` for textures.
pub(crate) fn buffer_bytes(this: This) -> Option<(*mut u8, u64)> {
    if this.is_null() {
        return None;
    }
    let r = resource(this);
    (r.desc.dimension == BUFFER).then(|| (r.data(), r.desc.width))
}

impl Resource {
    fn data(&self) -> *mut u8 {
        match self.backing {
            Backing::Committed(ref memory) => memory.ptr.as_ptr(),
            // SAFETY: placement checked the range lies inside the heap.
            Backing::Placed { heap, offset } => unsafe {
                (*heap.cast::<Heap>())
                    .memory
                    .ptr
                    .as_ptr()
                    .add(offset as usize)
            },
            Backing::Address(at) => at,
        }
    }
}

fn resource<'a>(this: This) -> &'a Resource {
    // SAFETY: called from resource methods only.
    unsafe { &*this.cast::<Resource>() }
}

// The creation methods' own arguments, passed on.
#[allow(clippy::too_many_arguments)]
fn hand_out_resource(
    desc: ResourceDesc,
    heap: HeapProperties,
    heap_flags: u32,
    backing: Backing,
    state: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
    sparse_prefix: u32,
) -> HRESULT {
    #[cfg(not(windows))]
    let _ = (state, clear, sparse_prefix);
    #[allow(unused_mut)] // the host texture is set on Windows only
    let mut resource = Box::new(Resource {
        object: Object::new(&RESOURCE_FUNCTIONS),
        desc,
        heap,
        heap_flags,
        backing,
        #[cfg(windows)]
        host: None,
    });
    #[cfg(windows)]
    {
        use victory_memory::authority::ByteExposure;
        resource.host = shared_texture(resource.data() as u64, &desc, state, clear, sparse_prefix);
        if let Some(texture) = &resource.host {
            #[cfg(feature = "paged-title")]
            let _ordered = crate::host::host().map(|host| host.direct_queue());
            #[cfg(feature = "paged-title")]
            if matches!(resource.backing, Backing::Committed(_))
                && let Some(layout) = linear_texture_bytes(resource.data() as u64, &desc)
            {
                let _ = texture.committed_linear.set(layout);
            }
            texture.expose_bytes(match resource.backing {
                Backing::Committed(_) => ByteExposure::TitleAllocation,
                Backing::Placed { .. } => ByteExposure::SharedHeap,
                Backing::Address(_) => ByteExposure::ExternalAddress,
            });
        }
    }
    #[cfg(windows)]
    if desc.dimension == BUFFER {
        let at = std::ptr::from_ref::<Resource>(&resource) as usize;
        crate::shadow::register(at, resource.data() as u64, desc.width);
    } else if resource.host.is_some() {
        let at = std::ptr::from_ref::<Resource>(&resource) as usize;
        let mut map = TEXTURES_AT.lock().unwrap_or_else(|e| e.into_inner());
        map.entry(resource.data() as u64).or_default().push(at);
    }
    object::hand_out(RESOURCE, resource, iid, &RESOURCE_IIDS, out)
}

/// Title textures with a host copy, by the console memory they start at.
/// Title textures of one memory share a host texture where the host can
/// view it in each one's format; where it cannot (block-compressed against
/// plain formats, a cast the host refused, a host texture replaced),
/// [`alias_source`] finds the one the GPU writes.
#[cfg(windows)]
static TEXTURES_AT: std::sync::Mutex<std::collections::BTreeMap<u64, Vec<usize>>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

/// For a texture made for a placed view (`texture_at`): a title texture
/// the GPU writes (render target, depth or unordered access) at the same
/// memory, in the same layout and with the same bytes per row and rows
/// (elements of the same size, block-compressed ones counted in blocks),
/// that has another host texture: what the GPU wrote there lives in that
/// one, to be copied over (`texture::alias_copy`).
/// shortcut: same start only; overlapping ranges, other mips or slices of
/// one memory, and views of buffers are not matched.
#[cfg(windows)]
pub(crate) fn alias_source(placed: This) -> Option<This> {
    const WRITTEN: u32 = 0x7; // render target, depth, unordered access
    let b = resource(placed);
    let map = TEXTURES_AT.lock().unwrap_or_else(|e| e.into_inner());
    let bytes = |d: &ResourceDesc| {
        let (size, block) = element(d.format)?;
        let swizzle = texture_layout(d).map_or(d.layout, |t| t.planes[0].layout.desc.swizzle.0);
        let rows = u64::from(d.height).div_ceil(block);
        Some((d.dimension, d.width.div_ceil(block) * size, rows, d.depth_or_array_size, d.sample_count, swizzle))
    };
    let want = bytes(&b.desc)?;
    map.get(&(b.data() as u64))?
        .iter()
        .map(|&r| r as This)
        .filter(|&r| r != placed)
        .find(|&r| {
            let a = resource(r);
            a.desc.flags & WRITTEN != 0
                && bytes(&a.desc) == Some(want)
                && matches!((&a.host, &b.host), (Some(x), Some(y)) if !std::sync::Arc::ptr_eq(x, y))
        })
}

#[cfg(windows)]
impl Drop for Resource {
    fn drop(&mut self) {
        #[cfg(feature = "diagnostics")]
        if self
            .host
            .as_ref()
            .is_some_and(|h| std::sync::Arc::strong_count(h) > 1)
        {
            // Before lists held their textures, this freed a texture that
            // lists not yet run still named (a GPU page fault).
            static OUTLIVED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = OUTLIVED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n.is_power_of_two() {
                tracing::info!(
                    n,
                    desc = ?self.desc,
                    "a title texture was released while a list not yet reset names its host copy"
                );
            }
        }
        #[cfg(feature = "diagnostics")]
        if let Some(h) = &self.host {
            use windows::core::Interface;
            life::note(h.texture.as_raw() as usize, life::RELEASED, 0, life::holders(&h.texture));
        }
        crate::bindless::forget(std::ptr::from_ref::<Resource>(self) as usize);
        if self.desc.dimension != BUFFER {
            let mut map = TEXTURES_AT.lock().unwrap_or_else(|e| e.into_inner());
            let at = self.data() as u64;
            if let Some(list) = map.get_mut(&at) {
                list.retain(|&r| r != std::ptr::from_ref::<Resource>(self) as usize);
                if list.is_empty() {
                    map.remove(&at);
                }
            }
        }
        crate::descriptor::forget(std::ptr::from_ref::<Resource>(self) as usize);
        if self.desc.dimension == BUFFER {
            crate::shadow::unregister(
                std::ptr::from_ref::<Resource>(self) as usize,
                self.data() as u64,
            );
        }
    }
}

pub(crate) extern "system" fn create_committed(
    _: This,
    heap: *const HeapProperties,
    heap_flags: u32,
    desc: *const ResourceDesc,
    state: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes both descriptions (null is tolerated).
    let (Some(&heap), Some(&desc)) = (unsafe { (heap.as_ref(), desc.as_ref()) }) else {
        return E_INVALIDARG;
    };
    let Some(info) = footprint(&desc) else {
        tracing::warn!(?desc, "no size for this resource");
        return E_INVALIDARG;
    };
    tracing::debug!(
        dimension = desc.dimension,
        size = info.size,
        heap = heap.kind,
        "committed resource"
    );
    let memory = match Memory::for_resource(info, heap.kind) {
        Ok(memory) => memory,
        Err(hr) => return hr,
    };
    hand_out_resource(
        desc,
        heap,
        heap_flags,
        Backing::Committed(memory),
        state,
        clear,
        iid,
        out,
        0,
    )
}

/// Device method 0x3f8, inferred from Grand Theft Auto VI's calls
/// (title+0x193e1b4: its back buffers, heap flag "allow display"): the
/// committed resource creation of the enhanced-barrier interfaces,
/// `(heap, heap flags, desc1, initial layout, clear value, protected
/// session, castable format count, castable formats, iid, out)`. The host
/// copy starts in the common state, as layouts are not tracked.
#[allow(clippy::too_many_arguments)] // the method's own arguments
pub(crate) extern "system" fn create_committed3(
    this: This,
    heap: *const HeapProperties,
    heap_flags: u32,
    desc: *const ResourceDesc,
    _layout: u32,
    clear: *const c_void,
    _protected: This,
    _castable_count: u32,
    _castable: *const u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    const COMMON: u32 = 0;
    create_committed(this, heap, heap_flags, desc, COMMON, clear, iid, out)
}

/// Registered page pools (64 KiB pages the title maps into tiled
/// resources by page index): first address and page count; a handle is an
/// index plus one, a removed pool leaves `None`.
static PAGE_POOLS: std::sync::Mutex<Vec<Option<(u64, u32)>>> = std::sync::Mutex::new(Vec::new());

fn page_pools() -> std::sync::MutexGuard<'static, Vec<Option<(u64, u32)>>> {
    PAGE_POOLS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Device method 0x250: `(address, page count, handle out)`.
pub(crate) extern "system" fn register_page_pool(
    _: This,
    address: u64,
    pages: u32,
    out: *mut usize,
) -> HRESULT {
    if out.is_null() || pages == 0 || !address.is_multiple_of(KIB_64) {
        return E_INVALIDARG;
    }
    tracing::debug!(address = format_args!("{address:#x}"), pages, "page pool");
    let mut pools = page_pools();
    pools.push(Some((address, pages)));
    // SAFETY: checked non-null; the title passes room for a handle.
    unsafe { out.write(pools.len()) };
    S_OK
}

/// Device method 0x258.
pub(crate) extern "system" fn unregister_page_pool(_: This, handle: usize) -> HRESULT {
    match page_pools().get_mut(handle.wrapping_sub(1)) {
        Some(pool @ Some(_)) => {
            *pool = None;
            S_OK
        }
        _ => E_INVALIDARG,
    }
}

/// Device method 0x428, one of the `ID3D12Device9` to `ID3D12Device11`
/// additions. From Grand Theft Auto VI's call: a resource at a virtual
/// address the title mapped itself, `(address, desc1, initial barrier
/// layout, clear value, iid, out)`, as `CreatePlacedResourceX` with
/// enhanced barriers. GPU addresses are CPU addresses, so the resource is
/// a view of that memory. `D3D12_RESOURCE_DESC1` starts like the plain
/// description; its sampler feedback region is not needed.
pub(crate) extern "system" fn create_at_address(
    _: This,
    address: u64,
    desc: *const ResourceDesc,
    layout: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    address_resource(address, desc, layout, clear, iid, out, 0)
}

fn address_resource(
    address: u64,
    desc: ResourceDesc,
    layout: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
    sparse_prefix: u32,
) -> HRESULT {
    if address == 0 {
        return E_INVALIDARG;
    }
    tracing::debug!(
        address = format_args!("{address:#x}"),
        dimension = desc.dimension,
        width = desc.width,
        layout,
        "resource at an address"
    );
    const COMMON: u32 = 0;
    const DEFAULT: HeapProperties = HeapProperties {
        kind: 1,
        cpu_pages: 0,
        pool: 0,
        creation_nodes: 1,
        visible_nodes: 1,
    };
    hand_out_resource(
        desc,
        DEFAULT,
        0,
        Backing::Address(address as *mut u8),
        COMMON,
        clear,
        iid,
        out,
        sparse_prefix,
    )
}

/// A cached placed texture, or a failed allocation waiting to be retried.
#[cfg(any(windows, test))]
#[derive(Clone, Copy, Default)]
struct PlacedSlot {
    ready: Option<usize>,
    retry: Option<(std::time::Instant, u8)>,
}

#[cfg(any(windows, test))]
impl PlacedSlot {
    fn waiting(self, now: std::time::Instant) -> bool {
        self.retry.is_some_and(|(until, _)| now < until)
    }

    fn defer(&mut self, now: std::time::Instant) {
        let attempts = self.retry.map_or(0, |(_, attempts)| attempts);
        let delay = std::time::Duration::from_millis(125 << attempts.min(5));
        self.retry = Some((now + delay, attempts.saturating_add(1)));
    }
}

/// A real device allocation refusal gives pending textures a short break,
/// without pausing updates to already backed textures.
#[cfg(any(windows, test))]
#[derive(Default)]
struct PlacementPressure {
    until: Option<std::time::Instant>,
}

#[cfg(any(windows, test))]
impl PlacementPressure {
    fn active(&self, now: std::time::Instant) -> bool {
        self.until.is_some_and(|until| now < until)
    }

    fn refuse(&mut self, now: std::time::Instant) {
        self.until = Some(now + std::time::Duration::from_millis(125));
    }
}

#[cfg(windows)]
static PLACEMENT_PRESSURE: std::sync::Mutex<PlacementPressure> =
    std::sync::Mutex::new(PlacementPressure { until: None });

#[cfg(windows)]
fn note_texture_pressure() {
    PLACEMENT_PRESSURE.lock().unwrap_or_else(|e| e.into_inner())
        .refuse(std::time::Instant::now());
}

#[cfg(windows)]
fn under_texture_pressure() -> bool {
    let mut pressure = PLACEMENT_PRESSURE.lock().unwrap_or_else(|e| e.into_inner());
    if pressure.until.is_none() {
        return false;
    }
    if pressure.active(std::time::Instant::now()) {
        return true;
    }
    pressure.until = None;
    false
}

#[cfg(all(windows, test))]
thread_local! {
    pub(crate) static REFUSE_NEXT_TEXTURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(all(windows, test))]
pub(crate) fn hold_placement_pressure_for_test() {
    let mut pressure = PLACEMENT_PRESSURE.lock().unwrap_or_else(|e| e.into_inner());
    assert!(pressure.until.is_some(), "the refused allocation must have signalled pressure");
    pressure.until = Some(std::time::Instant::now() + std::time::Duration::from_secs(60));
}

#[cfg(all(windows, test))]
pub(crate) fn allow_placement_retry() {
    PLACEMENT_PRESSURE.lock().unwrap_or_else(|e| e.into_inner()).until = None;
    let mut map = PLACED_TEXTURES.lock().unwrap_or_else(|e| e.into_inner());
    for s in map.values_mut() {
        s.retry = None;
    }
}

#[cfg(windows)]
type PlacedKey = (u64, [u32; 14]);

#[cfg(windows)]
fn placed_key(address: u64, desc: &[u32; 14]) -> PlacedKey {
    let mut desc = *desc;
    // Words 1 and 13 are padding (stack bytes in the title's records).
    (desc[1], desc[13]) = (0, 0);
    (address, desc)
}

#[cfg(windows)]
pub(crate) fn placement_pending(address: u64, desc: &[u32; 14]) -> bool {
    PLACED_TEXTURES.lock().unwrap_or_else(|e| e.into_inner())
        .get(&placed_key(address, desc))
        .is_some_and(|slot| slot.ready.is_none() && slot.retry.is_some())
}

/// Textures the title placed in its own memory and views without a
/// resource object (device 0x230, Grand Theft Auto VI's streamed and
/// interface textures), by address and description: a resource at that
/// address, made the first time a draw reads one.
/// shortcut: never released (one host texture per distinct address and
/// description viewed); release them when the title unmaps or reuses the
/// memory once streaming grows VRAM use.
#[cfg(windows)]
static PLACED_TEXTURES: std::sync::Mutex<std::collections::BTreeMap<PlacedKey, PlacedSlot>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

#[cfg(windows)]
/// Return the detailed mip prefix the initial SRV does not expose and
/// whose console pages are all absent. Never infer absence from protection
/// flags, since a guarded/NOACCESS committed page may still contain data.
fn sparse_missing_prefix(
    desc: &ResourceDesc,
    srv: &windows::Win32::Graphics::Direct3D12::D3D12_SHADER_RESOURCE_VIEW_DESC,
    address: usize,
) -> u32 {
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;
    if desc.dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D.0 as u32
        || desc.depth_or_array_size != 1
        || desc.sample_count != 1
        || desc.flags & 0xf != 0
        || desc.width < 512
        || desc.height < 512
        || srv.ViewDimension != D3D12_SRV_DIMENSION_TEXTURE2D
        || !matches!(srv.Format, DXGI_FORMAT_BC1_UNORM | DXGI_FORMAT_BC1_UNORM_SRGB)
        || !(desc.format == DXGI_FORMAT_BC1_TYPELESS.0 as u32 || desc.format == srv.Format.0 as u32)
    {
        return 0;
    }
    // SAFETY: the dimension was checked above.
    let view = unsafe { srv.Anonymous.Texture2D };
    let first = view.MostDetailedMip;
    let mips = u32::from(desc.mip_levels);
    if first == 0
        || first >= mips
        || view.MipLevels == 0
        || (view.MipLevels != u32::MAX && view.MipLevels > mips - first)
        || view.PlaneSlice != 0
    {
        return 0;
    }
    let Some(layout) = texture_layout(desc) else {
        return 0;
    };
    let layout = &layout.planes[0].layout;
    if layout.slices != 1 || layout.desc.swizzle.xor() {
        return 0;
    }
    let query = |at: usize| {
        use windows_sys::Win32::System::Memory::*;
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: VirtualQuery describes the address without reading the
        // texture data and writes only to the local information structure.
        if unsafe { VirtualQuery(at as _, &mut info, size_of_val(&info)) } != size_of_val(&info) {
            return None;
        }
        let base = info.BaseAddress as usize;
        (base <= at).then_some(())?;
        Some((base.checked_add(info.RegionSize)?, matches!(info.State, MEM_RESERVE | MEM_FREE)))
    };
    let mut end = layout.slice_size;
    let mut missing = 0;
    for mip in layout.mips.iter().take(first.min(layout.first_tail_mip) as usize) {
        let extent = (|| {
            let start = address.checked_add(usize::try_from(mip.offset).ok()?)?;
            let end = address.checked_add(usize::try_from(end).ok()?)?;
            Some((start, end))
        })();
        let Some((lo, hi)) = extent else { break };
        if !crate::sparse::wholly_absent(lo, hi, query) {
            break;
        }
        missing += 1;
        end = mip.offset;
    }
    missing
}

/// The resource for a texture placed at `address` with the resource
/// description `desc` (as words), made on first use.
#[cfg(windows)]
pub(crate) fn texture_at(
    address: u64,
    desc: &[u32; 14],
    view: &windows::Win32::Graphics::Direct3D12::D3D12_SHADER_RESOURCE_VIEW_DESC,
) -> Option<This> {
    let mut map = PLACED_TEXTURES.lock().unwrap_or_else(|e| e.into_inner());
    let key = placed_key(address, desc);
    let slot = map.get(&key).copied().unwrap_or_default();
    let old = slot.ready.map(|r| r as This);
    // A failed allocation yields without retaining an empty host object.
    // A still-live generation (including sparse backing) never waits for it.
    if old.is_some_and(current) {
        return old;
    }
    if slot.retry.is_some() && slot.waiting(std::time::Instant::now()) {
        return old;
    }
    if under_texture_pressure() {
        // Mark the view pending even if the global wait expires before
        // the heap records this attempt as a null descriptor.
        if old.is_none() {
            map.entry(key).or_default().retry.get_or_insert((std::time::Instant::now(), 0));
        }
        return old;
    }
    // SAFETY: 56 bytes in the description's layout.
    let d = unsafe { key.1.as_ptr().cast::<ResourceDesc>().read_unaligned() };
    if d.dimension == BUFFER {
        return None;
    }
    // Never select sparse backing where an existing title texture can
    // already have written this memory, or when replacing a generation.
    let sparse_prefix = if crate::sparse::enabled()
        && old.is_none()
        && !TEXTURES_AT.lock().unwrap_or_else(|e| e.into_inner()).contains_key(&address)
    {
        sparse_missing_prefix(&d, view, usize::try_from(address).ok()?)
    } else {
        0
    };
    let mut out = std::ptr::null();
    let made = address_resource(
        address, d, 0, std::ptr::null(), &RESOURCE_IIDS[0], &mut out, sparse_prefix,
    );
    if made != S_OK {
        return old;
    }
    if resource(out).host.is_none() {
        map.entry(key).or_default().defer(std::time::Instant::now());
        // No descriptor ever saw this failed object. Do not keep it (and
        // don't enter the bindless cleanup while holding the placed map).
        drop(map);
        resource_release(out);
        return old;
    }
    map.insert(key, PlacedSlot { ready: Some(out as usize), retry: None });
    Some(out)
}

pub(crate) extern "system" fn create_placed(
    _: This,
    heap: This,
    offset: u64,
    desc: *const ResourceDesc,
    state: u32,
    clear: *const c_void,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    if heap.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the title passes one of our heaps.
    let h = unsafe { &*heap.cast::<Heap>() };
    let fits = footprint(&desc).is_some_and(|info| {
        offset.is_multiple_of(info.alignment)
            && offset
                .checked_add(info.size)
                .is_some_and(|end| end <= h.desc.size)
    });
    if !fits {
        tracing::warn!(
            ?desc,
            offset,
            heap = h.desc.size,
            "placed resource does not fit"
        );
        return E_INVALIDARG;
    }
    object::add_ref(heap);
    let backing = Backing::Placed { heap, offset };
    hand_out_resource(
        desc,
        h.desc.properties,
        h.desc.flags,
        backing,
        state,
        clear,
        iid,
        out,
        0,
    )
}

extern "system" fn resource_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        RESOURCE,
        iid,
        object::is(iid, &RESOURCE_IIDS).then_some(this),
        out,
    )
}
pub(crate) extern "system" fn resource_release(this: This) -> u32 {
    object::release_boxed::<Resource>(this)
}
/// A null range means the entire resource, an empty range no access.
#[cfg(all(windows, feature = "paged-title"))]
fn accesses_bytes(range: *const c_void) -> Option<bool> {
    if range.is_null() { return Some(true); }
    // SAFETY: the guest Map/Unmap ABI supplies a readable D3D12_RANGE.
    let [start, end] = unsafe { range.cast::<[usize; 2]>().read_unaligned() };
    (start <= end).then_some(start != end)
}

/// Map is a CPU byte demand, not proof that a submitted GPU write has
/// completed. The title must still use the corresponding queue fence.
extern "system" fn map(this: This, _sub: u32, _read: *const c_void, out: *mut *mut u8) -> HRESULT {
    if !out.is_null() {
        let resource = resource(this);
        #[cfg(windows)]
        if let Some(texture) = &resource.host {
            use victory_memory::authority::ByteExposure::MappedPointer;
            #[cfg(not(feature = "paged-title"))]
            texture.expose_bytes(MappedPointer);
            #[cfg(feature = "paged-title")]
            {
                if accesses_bytes(_read).is_none() {
                    return E_INVALIDARG;
                }
                if _sub != 0 {
                    return crate::object::E_NOTIMPL;
                }
                let Some(host) = crate::host::host() else { return crate::object::E_NOTIMPL; };
                let mut mapped = texture.mapped_pointers.lock().unwrap_or_else(|e| e.into_inner());
                let mut ordered = host.direct_queue();
                let layout = texture.exact_committed_bytes();
                let Some(next) = mapped.active.checked_add(1) else { return E_INVALIDARG; };
                if let Err(error) = texture.prepare_pointer_bytes_locked(host, &mut ordered, &mapped, layout) {
                    tracing::warn!(?error, "VNME texture Map could not materialize bytes");
                    return crate::object::E_NOTIMPL;
                }
                // Exposure is committed only for a successful pointer return,
                // while GPU write planning is still excluded by the queue lock.
                mapped.active = next;
                texture.expose_bytes(MappedPointer);
                // SAFETY: the ABI supplied valid, non-null pointer output.
                unsafe { out.write(resource.data()) };
                return S_OK;
            }
        }
        // SAFETY: checked non-null; the guest provides room for a pointer.
        unsafe { out.write(resource.data()) };
    }
    S_OK
}

extern "system" fn unmap(_this: This, _sub: u32, _written: *const c_void) {
    #[cfg(all(windows, feature = "paged-title"))]
    if let Some(texture) = &resource(_this).host {
        let mut mapped = texture.mapped_pointers.lock().unwrap_or_else(|e| e.into_inner());
        if _sub != 0 || mapped.active == 0 {
            return;
        }
        mapped.active -= 1;
        if accesses_bytes(_written) != Some(false)
            && let Some(host) = crate::host::host()
        {
            let _ordered = host.direct_queue();
            texture.record_cpu_bytes();
        }
    }
}
// These CPU copy APIs do not hand the application a persistent pointer.
// All layout validation and queue ordering live in one coherency boundary.
#[cfg(all(windows, feature = "paged-title"))]
extern "system" fn write_subresource(
    this: This, sub: u32, area: *const [u32; 6], source: *const u8,
    row_pitch: u32, _depth_pitch: u32,
) -> HRESULT {
    cpu_texture_copy(this, sub, area, source.cast_mut(), row_pitch, true)
}

#[cfg(all(windows, feature = "paged-title"))]
extern "system" fn read_subresource(
    this: This, destination: *mut u8, row_pitch: u32, _depth_pitch: u32,
    sub: u32, area: *const [u32; 6],
) -> HRESULT {
    cpu_texture_copy(this, sub, area, destination, row_pitch, false)
}

#[cfg(all(windows, feature = "paged-title"))]
fn cpu_texture_copy(
    this: This, sub: u32, area: *const [u32; 6], data: *mut u8,
    row_pitch: u32, write: bool,
) -> HRESULT {
    use crate::object::E_NOTIMPL;
    let resource = resource(this);
    if sub != 0 {
        return E_NOTIMPL;
    }
    // CPU-visible custom heaps, not default GPU-only storage. The
    // compatible storage strategy currently admits exact committed 2D data.
    if !matches!(resource.backing, Backing::Committed(_))
        || resource.heap.kind != 4 || !matches!(resource.heap.cpu_pages, 2 | 3)
    {
        return E_NOTIMPL;
    }
    let Some(texture) = &resource.host else { return E_NOTIMPL; };
    let Some(host) = crate::host::host() else { return E_NOTIMPL; };
    // As in Unmap, take the mapping mutex before the direct-queue lock.
    let mapped = texture.mapped_pointers.lock().unwrap_or_else(|e| e.into_inner());
    let mut ordered = host.direct_queue();
    let Some(layout) = texture.exact_committed_bytes() else { return E_NOTIMPL; };
    // SAFETY: the resource ABI supplies six UINTs or a null box.
    let [left, top, front, right, bottom, back] = if area.is_null() {
        [0, 0, 0, layout.width, layout.height, 1]
    } else {
        unsafe { area.read_unaligned() }
    };
    if left >= right || top >= bottom || front >= back {
        return S_OK;
    }
    let copied = ((right - left) as usize) * 4;
    let pitch = row_pitch as usize;
    if data.is_null() || front != 0 || back != 1
        || right > layout.width || bottom > layout.height
        || pitch < copied
    {
        return E_INVALIDARG;
    }
    let rows = (bottom - top) as usize;
    let tex_pitch = layout.row_pitch as usize;
    let tex_start = layout.address as usize + top as usize * tex_pitch + left as usize * 4;
    let tex_end = tex_start + (rows - 1) * tex_pitch + copied;
    let Some(data_len) = (rows - 1).checked_mul(pitch).and_then(|n| n.checked_add(copied))
        .filter(|&n| n <= isize::MAX as usize)
    else { return E_INVALIDARG; };
    let Some(data_end) = (data as usize).checked_add(data_len) else { return E_INVALIDARG; };
    if (data as usize) < tex_end && tex_start < data_end {
        return E_INVALIDARG;
    }
    // Leave the actual public D3D12 resource state unchanged by any
    // internal transfer. The title still owns synchronization with fences.
    if texture.state.load(std::sync::atomic::Ordering::Acquire) != 0 {
        return E_INVALIDARG;
    }
    let replaces_all = write && left == 0 && top == 0 && right == layout.width && bottom == layout.height;
    if (!replaces_all || mapped.active != 0)
        && let Err(error) = texture.ensure_cpu_bytes_locked(host, &mut ordered, &mapped, Some(layout))
    {
        tracing::warn!(?error, "VNME CPU texture copy could not preserve native bytes");
        return E_NOTIMPL;
    }
    // SAFETY: the title provides a valid pitched input/output buffer.
    // The compatible texture layout and non-overlapping row spans were checked.
    unsafe {
        let tex = tex_start as *mut u8;
        for row in 0..rows {
            if write {
                std::ptr::copy_nonoverlapping(data.add(row * pitch), tex.add(row * tex_pitch), copied);
            } else {
                std::ptr::copy_nonoverlapping(tex.add(row * tex_pitch), data.add(row * pitch), copied);
            }
        }
    }
    if write {
        texture.record_cpu_bytes();
    }
    S_OK
}

extern "system" fn resource_desc(this: This, out: *mut ResourceDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(resource(this).desc) };
    }
}
/// Device method `GetCustomHeapProperties`, returning its struct through a
/// hidden first argument: what a standard heap type is as a custom heap.
/// Memory is unified and cache-coherent (as the architecture query says),
/// so every type is in pool L0, default heaps are not CPU-visible and the
/// others are write-back.
pub(crate) extern "system" fn custom_heap_properties(
    out: *mut HeapProperties,
    _: This,
    nodes: u32,
    kind: u32,
) -> *mut HeapProperties {
    const CUSTOM: u32 = 4;
    const NOT_AVAILABLE: u32 = 1;
    const WRITE_BACK: u32 = 3;
    const L0: u32 = 1;
    let nodes = nodes.max(1);
    let cpu_pages = match kind {
        1 => NOT_AVAILABLE,
        2 | 3 => WRITE_BACK,
        _ => 0,
    };
    let properties = HeapProperties {
        kind: CUSTOM,
        cpu_pages,
        pool: if cpu_pages == 0 { 0 } else { L0 },
        creation_nodes: nodes,
        visible_nodes: nodes,
    };
    // SAFETY: the title passes room for the result (null is tolerated).
    if let Some(out) = unsafe { out.as_mut() } {
        *out = properties;
    }
    out
}

extern "system" fn gpu_address(this: This) -> u64 {
    let r = resource(this);
    #[cfg(windows)]
    if let Some(texture) = &r.host {
        #[cfg(feature = "paged-title")]
        if !texture.publish_guest_address() {
            return 0;
        }
        #[cfg(not(feature = "paged-title"))]
        texture.expose_bytes(victory_memory::authority::ByteExposure::GuestGpuAddress);
    }
    let at = r.data() as u64;
    // The console's GPU addresses fit in 40 bits; its ray-tracing builder
    // refuses others (diagnostics: which memory breaks that).
    #[cfg(feature = "diagnostics")]
    if at >> 40 != 0 {
        static ONCE: [std::sync::Once; 2] = [std::sync::Once::new(), std::sync::Once::new()];
        let backing = match r.backing {
            Backing::Committed(ref m) => format!("committed, attributes {:?}", m.attributes),
            Backing::Placed { offset, .. } => format!("placed in a heap at {offset:#x}"),
            Backing::Address(_) => "at a title address".into(),
        };
        ONCE[usize::from(r.desc.dimension == BUFFER)].call_once(|| {
            tracing::warn!(address = format_args!("{at:#x}"), backing, desc = ?r.desc, "a GPU address above 40 bits")
        });
    }
    at
}
extern "system" fn heap_properties(
    this: This,
    properties: *mut HeapProperties,
    flags: *mut u32,
) -> HRESULT {
    let r = resource(this);
    // SAFETY: either pointer may be null; otherwise the title passes room.
    unsafe {
        if let Some(p) = properties.as_mut() {
            *p = r.heap;
        }
        if let Some(f) = flags.as_mut() {
            *f = r.heap_flags;
        }
    }
    S_OK
}

/// IUnknown 3, object 4, device child 1, heap 1.
static HEAP_FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<HEAP, 24>());
    f[0] = heap_query as *const ();
    f[2] = heap_release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = heap_desc as *const (); // 0x40
    Table(f)
};

/// IUnknown 3, object 4, device child 1, resource 10 (7 as on PC, then 3
/// console ones).
static RESOURCE_FUNCTIONS: Table<32> = {
    let mut f = object::d3d12_object(object::missing_table::<RESOURCE, 32>());
    f[0] = resource_query as *const ();
    f[2] = resource_release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = map as *const (); // 0x40
    f[9] = unmap as *const (); // 0x48
    f[10] = resource_desc as *const (); // 0x50
    f[11] = gpu_address as *const (); // 0x58
    #[cfg(all(windows, feature = "paged-title"))]
    {
        f[12] = write_subresource as *const (); // 0x60
        f[13] = read_subresource as *const (); // 0x68
    }
    f[14] = heap_properties as *const (); // 0x70
    Table(f)
};

#[cfg(test)]
pub(crate) fn test_buffer(width: u64) -> This {
    let heap = HeapProperties {
        kind: 1,
        cpu_pages: 0,
        pool: 0,
        creation_nodes: 0,
        visible_nodes: 0,
    };
    let desc = ResourceDesc {
        dimension: BUFFER,
        alignment: 0,
        width,
        height: 1,
        depth_or_array_size: 1,
        mip_levels: 1,
        format: 0,
        sample_count: 1,
        sample_quality: 0,
        layout: 1,
        flags: 0,
    };
    let mut out = std::ptr::null();
    let hr = create_committed(
        std::ptr::null(),
        &heap,
        0,
        &desc,
        0,
        std::ptr::null(),
        &RESOURCE_IIDS[0],
        &mut out,
    );
    assert_eq!(hr, S_OK);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_target_translation_does_not_change_title_texture_bytes() {
        assert_eq!(pc_format(67), 67);
        assert_eq!(element(67), Some((4, 1)));
        assert_eq!(element(target_format(67)), Some((8, 1)));
        assert_eq!(target_format(28), 28);
    }

    #[cfg(windows)]
    #[test]
    fn shared_exponent_changes_only_the_host_render_target() {
        let sampled = texture(64, 64, 67, 1);
        assert_eq!(host_desc(&sampled).unwrap().format, 67);
        let target = ResourceDesc { flags: 1, ..sampled };
        let host = host_desc(&target).unwrap();
        assert_eq!(host.format, 10);
        assert_eq!(host.flags, 1);
        assert_eq!(target.format, 67);
    }


    #[test]
    fn placed_allocation_refusal_yields_then_retries_without_affecting_other_slots() {
        use std::time::{Duration, Instant};
        let start = Instant::now();
        let mut refused = PlacedSlot::default();
        let live = PlacedSlot { ready: Some(1), retry: None };
        assert!(!refused.waiting(start));
        refused.defer(start);
        assert!(refused.waiting(start + Duration::from_millis(124)));
        let second = start + Duration::from_millis(125);
        assert!(!refused.waiting(second));
        refused.defer(second);
        assert!(refused.waiting(second + Duration::from_millis(249)));
        assert!(!refused.waiting(second + Duration::from_millis(250)));
        assert!(!live.waiting(start));
        assert_eq!(live.ready, Some(1));
        for _ in 0..100 {
            refused.defer(start);
        }
        assert!(!refused.waiting(start + Duration::from_secs(4)));
        assert!(refused.ready.is_none());
    }

    #[test]
    fn texture_pressure_expires_without_extending_on_deferred_work() {
        use std::time::{Duration, Instant};
        let now = Instant::now();
        let mut pressure = PlacementPressure::default();
        assert!(!pressure.active(now));
        pressure.refuse(now);
        for t in 0..125 {
            assert!(pressure.active(now + Duration::from_millis(t)));
        }
        assert!(!pressure.active(now + Duration::from_millis(125)));
        pressure.refuse(now + Duration::from_millis(125));
        assert!(pressure.active(now + Duration::from_millis(249)));
        assert!(!pressure.active(now + Duration::from_millis(250)));
    }

    use crate::object::slot;

    /// A render target with unordered views (the title's full-screen
    /// targets) is made castable to the formats of its element the host
    /// has, on every host: one format it lacks does not cost it the list.
    #[cfg(windows)]
    #[test]
    fn a_colour_target_is_castable_to_the_formats_the_host_has() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let d = ResourceDesc {
            dimension: 3,
            alignment: 0,
            width: 256,
            height: 256,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: 28,
            sample_count: 1,
            sample_quality: 0,
            layout: 0,
            flags: 5,
        };
        let t = make_texture(&d, 0, std::ptr::null()).unwrap();
        let had: Vec<u32> = castable(28)
            .into_iter()
            .filter(|&f| host_has(host, f))
            .collect();
        assert!(!had.is_empty());
        assert_eq!(t.castable, had);
    }

    #[test]
    fn a8_unorm_is_not_in_the_r8_typeless_family() {
        // DXGI_FORMAT values: 60 R8_TYPELESS, 61 R8_UNORM, 62 R8_UINT,
        // 63 R8_SNORM, 64 R8_SINT, 65 A8_UNORM. The R8 typeless family is
        // 60..=64; A8_UNORM has no typeless family and cannot be cast to R8,
        // so it must stand alone rather than fold into R8_TYPELESS.
        assert_eq!(family(60), 60);
        for f in 61..=64 {
            assert_eq!(family(f), 60);
        }
        assert_eq!(family(65), 65);
        // Boundaries: 59 belongs to the preceding family, 66 is unmapped.
        assert_eq!(family(59), 53);
        assert_eq!(family(66), 66);
    }

    type Release = extern "system" fn(This) -> u32;
    type Map = extern "system" fn(This, u32, *const c_void, *mut *mut u8) -> HRESULT;
    type Address = extern "system" fn(This) -> u64;

    const PROPERTIES: HeapProperties = HeapProperties {
        kind: 1,
        cpu_pages: 0,
        pool: 0,
        creation_nodes: 0,
        visible_nodes: 0,
    };

    fn buffer(width: u64) -> ResourceDesc {
        ResourceDesc {
            dimension: BUFFER,
            alignment: 0,
            width,
            height: 1,
            depth_or_array_size: 1,
            mip_levels: 1,
            format: 0,
            sample_count: 1,
            sample_quality: 0,
            layout: 1,
            flags: 0,
        }
    }

    fn texture(width: u64, height: u32, format: u32, mips: u16) -> ResourceDesc {
        ResourceDesc {
            dimension: 3,
            width,
            height,
            format,
            mip_levels: mips,
            layout: 0,
            ..buffer(0)
        }
    }

    fn heap(size: u64) -> This {
        let desc = HeapDesc {
            size,
            properties: PROPERTIES,
            alignment: 0,
            flags: 0,
        };
        let mut heap = std::ptr::null();
        assert_eq!(
            create_heap(std::ptr::null(), &desc, &HEAP_IIDS[0], &mut heap),
            S_OK
        );
        heap
    }

    /// Title textures of one memory share a key in any format of the same
    /// element; another address, shape, element size or a depth texture is
    /// another memory. The library's own layout pick resolves as spelled.
    #[test]
    fn one_memory_one_key_in_any_format_of_its_element() {
        let a = ResourceDesc { flags: 4, ..texture(192, 108, 42, 1) };
        let b = ResourceDesc { format: 67, flags: 0, ..a };
        let key = memory_key(0x1_0000, &a);
        assert!(key.is_some());
        assert_eq!(key, memory_key(0x1_0000, &b));
        assert_eq!(key, memory_key(0x1_0000, &ResourceDesc { layout: 0x105, ..b }));
        assert_ne!(key, memory_key(0x2_0000, &a));
        assert_ne!(key, memory_key(0x1_0000, &ResourceDesc { width: 96, ..a }));
        assert_ne!(key, memory_key(0x1_0000, &ResourceDesc { format: 10, ..a }));
        assert_ne!(key, memory_key(0x1_0000, &ResourceDesc { layout: 0x102, ..a }));
        assert_ne!(key, memory_key(0x1_0000, &ResourceDesc { flags: 2, format: 40, ..a }));
        assert_eq!(memory_key(0x1_0000, &buffer(256)), None);
    }

    /// Castable formats: every format of the same element but typeless and
    /// depth ones; block formats among block formats of their block size.
    #[test]
    fn castable_formats_share_the_element() {
        let c = castable(42);
        for f in [67, 28, 29, 41, 24, 26, 87, 35] {
            assert!(c.contains(&f), "{f}");
        }
        for f in [42, 39, 40, 45, 27, 10, 56] {
            assert!(!c.contains(&f), "{f}");
        }
        let bc1 = castable(71);
        assert!(bc1.contains(&72) && bc1.contains(&80) && bc1.contains(&81));
        assert!(!bc1.contains(&70) && !bc1.contains(&98) && !bc1.contains(&17));
        assert!(castable(98).contains(&95) && castable(98).contains(&77));
        assert!(castable(53).contains(&56) && !castable(53).contains(&55));
    }

    #[test]
    fn heaps_are_zeroed_aligned_memory() {
        let heap = heap(1 << 20);
        // SAFETY: a heap we just made.
        let memory = unsafe { &(*heap.cast::<Heap>()).memory };
        assert_eq!(memory.ptr.as_ptr() as u64 % KIB_64, 0);
        let mut back = std::mem::MaybeUninit::<HeapDesc>::uninit();
        slot::<extern "system" fn(This, *mut HeapDesc)>(heap, 0x40)(heap, back.as_mut_ptr());
        // SAFETY: written above.
        assert_eq!(unsafe { back.assume_init() }.size, 1 << 20);
        assert_eq!(slot::<Release>(heap, 0x10)(heap), 0);

        let bad = HeapDesc {
            size: 1 << 20,
            properties: PROPERTIES,
            alignment: 3,
            flags: 0,
        };
        let mut out = std::ptr::null();
        assert_eq!(
            create_heap(std::ptr::null(), &bad, &HEAP_IIDS[0], &mut out),
            E_INVALIDARG
        );
    }

    #[test]
    fn device_setup_heap_properties_and_resources_at_addresses() {
        let device = crate::device::device();
        // GetCustomHeapProperties (0xd0): the struct comes back through the
        // hidden first argument, which is also returned.
        type Custom =
            extern "system" fn(*mut HeapProperties, This, u32, u32) -> *mut HeapProperties;
        let mut p = PROPERTIES;
        let back = slot::<Custom>(device, 0xd0)(&mut p, device, 1, 2);
        assert_eq!(back, &raw mut p);
        assert_eq!(
            (
                p.kind,
                p.cpu_pages,
                p.pool,
                p.creation_nodes,
                p.visible_nodes
            ),
            (4, 3, 1, 1, 1),
            "upload: custom, write-back, L0"
        );
        slot::<Custom>(device, 0xd0)(&mut p, device, 1, 1);
        assert_eq!(p.cpu_pages, 1, "default: not CPU-visible");

        // 0x428: a buffer over memory the title owns, as GTA VI asks.
        type At = extern "system" fn(
            This,
            u64,
            *const ResourceDesc,
            u32,
            *const c_void,
            *const Iid,
            *mut This,
        ) -> HRESULT;
        let memory = vec![0u8; 0x1_0000];
        let address = memory.as_ptr() as u64;
        let mut r = std::ptr::null();
        let create = slot::<At>(device, 0x428);
        let desc = buffer(0x1_0000);
        let null = std::ptr::null();
        assert_eq!(
            create(
                device,
                address,
                &desc,
                u32::MAX,
                null,
                &RESOURCE_IIDS[0],
                &mut r
            ),
            0
        );
        assert_eq!(slot::<Address>(r, 0x58)(r), address);
        assert_eq!(slot::<Release>(r, 0x10)(r), 0);
        // The title's memory is still there after the release.
        assert_eq!(memory[0xffff], 0);
        assert_ne!(
            create(device, 0, &desc, u32::MAX, null, &RESOURCE_IIDS[0], &mut r),
            0
        );
    }

    #[test]
    fn sizes_cover_every_mip() {
        let size = |d: &ResourceDesc| footprint(d).map(|i| i.size);
        assert_eq!(size(&buffer(256)), Some(KIB_64));
        // Textures take their console layout and its alignment (4 KiB
        // standard swizzle when the description leaves it to the library):
        // 512x512 BC7 with 10 mips is 0x56000 bytes, 4 KiB aligned, as on
        // the console.
        let layout = |d: &ResourceDesc| texture_layout(d).unwrap().size;
        let rgba = texture(256, 256, 28, 0);
        assert_eq!(size(&rgba), Some(layout(&rgba).next_multiple_of(4096)));
        let bc7 = footprint(&texture(512, 512, 98, 10));
        let console = AllocationInfo {
            size: 0x56000,
            alignment: 0x1000,
        };
        assert_eq!(bc7, Some(console));
        assert!(layout(&rgba) >= 4 * (256 * 256 * 4) / 3, "every mip");
        // BC1 is 8 bytes per 4x4 block: 1024x1024, one mip = 512 KiB.
        assert_eq!(size(&texture(1024, 1024, 71, 1)), Some(512 << 10));
        assert_eq!(size(&texture(16, 16, 66, 1)), None, "1-bit format");
        // Linear layout (1) and an explicit mode (64 KiB display, 0x10a).
        let linear = ResourceDesc {
            layout: 1,
            ..texture(100, 60, 28, 1)
        };
        assert_eq!(
            texture_layout(&linear).unwrap().planes[0]
                .layout
                .desc
                .swizzle
                .0,
            0
        );
        let display = ResourceDesc {
            layout: 0x10a,
            ..texture(100, 60, 28, 1)
        };
        assert_eq!(
            texture_layout(&display).unwrap().planes[0]
                .layout
                .desc
                .swizzle
                .0,
            10
        );

        let mut info = AllocationInfo {
            size: 0,
            alignment: 0,
        };
        let descs = [buffer(100), texture(256, 256, 28, 0)];
        allocation_info(&mut info, std::ptr::null(), 0, 2, descs.as_ptr());
        assert_eq!(
            info,
            AllocationInfo {
                size: KIB_64 + size(&descs[1]).unwrap(),
                alignment: KIB_64
            }
        );
        allocation_info(&mut info, std::ptr::null(), 0, 1, &texture(1, 1, 999, 1));
        assert_eq!(info.size, u64::MAX);
    }

    #[test]
    fn upload_footprints_follow_the_pc_rules() {
        let null = std::ptr::null();
        let mut layouts = [NO_FOOTPRINT; 2];
        let (mut rows, mut sizes, mut total) = ([0; 2], [0; 2], 0);
        let mut footprints = |desc: &ResourceDesc, first, count| {
            let (l, r, s) = (layouts.as_mut_ptr(), rows.as_mut_ptr(), sizes.as_mut_ptr());
            copyable_footprints(null, desc, first, count, 0, l, r, s, &mut total);
            (layouts, rows, sizes, total)
        };
        // 100x60 RGBA8, 2 mips: 400-byte rows pitched to 512.
        let (l, r, s, t) = footprints(&texture(100, 60, 28, 2), 0, 2);
        assert_eq!((l[0].row_pitch, r[0], s[0]), (512, 60, 400));
        // Mip 1 (50x30, 200-byte rows) starts after 59 pitched rows and one
        // packed row, rounded up to 512.
        assert_eq!(l[1].offset, (512 * 59 + 400u64).next_multiple_of(512));
        assert_eq!((l[1].width, l[1].height, r[1]), (50, 30, 30));
        assert_eq!(t, l[1].offset + 256 * 29 + 200);
        // BC1: rows of 4x4 blocks, 8 bytes each.
        let (_, r, s, _) = footprints(&texture(64, 64, 71, 1), 0, 1);
        assert_eq!((r[0], s[0]), (16, 128));
        let (_, _, _, t) = footprints(&texture(64, 64, 71, 1), 1, 1);
        assert_eq!(t, u64::MAX, "no second subresource");
    }

    #[test]
    fn depth_stencil_footprints_follow_the_pc_plane_order() {
        // Native D3D12 reports the same layout for both depth/stencil families
        // and their view formats. Mips vary fastest, then array slices, then planes.
        let expected = [
            (512, 39, 16, 8),
            (4_608, 39, 8, 4),
            (6_656, 39, 16, 8),
            (10_752, 39, 8, 4),
            (12_800, 60, 16, 8),
            (16_896, 60, 8, 4),
            (18_944, 60, 16, 8),
            (23_040, 60, 8, 4),
        ];
        for format in (19..=22).chain(44..=47) {
            let desc = ResourceDesc {
                depth_or_array_size: 2,
                ..texture(16, 8, format, 2)
            };
            let mut layouts = [NO_FOOTPRINT; 8];
            let (mut rows, mut sizes, mut total) = ([0; 8], [0; 8], 0);
            copyable_footprints(
                std::ptr::null(),
                &desc,
                0,
                8,
                512,
                layouts.as_mut_ptr(),
                rows.as_mut_ptr(),
                sizes.as_mut_ptr(),
                &mut total,
            );

            assert_eq!(
                layouts.map(|l| (l.offset, l.format, l.width, l.height)),
                expected,
                "format {format}"
            );
            assert!(
                layouts.iter().all(|l| l.depth == 1 && l.row_pitch == 512),
                "format {format}"
            );
            assert_eq!(rows, [8, 4, 8, 4, 8, 4, 8, 4], "format {format}");
            assert_eq!(sizes, [64, 32, 64, 32, 16, 8, 16, 8], "format {format}");
            assert_eq!(total, 24_072, "format {format}");
            assert!(subresource(&desc, 8).is_none(), "format {format}");
        }
    }

    /// A 4x4 texture filled from a title buffer through our list and queue,
    /// then read back from the host GPU.
    #[cfg(windows)]
    #[test]
    fn buffer_data_reaches_the_host_texture() {
        use windows::Win32::Graphics::Direct3D12::*;
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let null = std::ptr::null();
        let iid = &RESOURCE_IIDS[0];
        const COPY_DEST: u32 = 0x400;
        let (mut image, mut source) = (null, null);
        let rgba = texture(4, 4, 28, 1);
        assert_eq!(
            create_committed(
                null,
                &PROPERTIES,
                0,
                &rgba,
                COPY_DEST,
                null.cast(),
                iid,
                &mut image
            ),
            S_OK
        );
        assert_eq!(
            create_committed(
                null,
                &PROPERTIES,
                0,
                &buffer(1024),
                0,
                null.cast(),
                iid,
                &mut source
            ),
            S_OK
        );
        let (data, _) = buffer_bytes(source).unwrap();
        // SAFETY: the buffer holds 1024 bytes.
        let bytes = unsafe { std::slice::from_raw_parts_mut(data, 1024) };
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }

        let queue_iid = guid(
            0x0ec870a6,
            0x5d7e,
            0x4c22,
            [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
        );
        let allocator_iid = guid(
            0x6102dee4,
            0xaf59,
            0x4b09,
            [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
        );
        let list_iid = guid(
            0x5b160d0f,
            0xac1b,
            0x4185,
            [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
        );
        let (mut queue, mut allocator, mut list) = (null, null, null);
        let queue_desc = [0u32; 4];
        assert_eq!(
            crate::queue::create(null, queue_desc.as_ptr().cast(), &queue_iid, &mut queue),
            S_OK
        );
        assert_eq!(
            crate::list::create_allocator(null, 0, &allocator_iid, &mut allocator),
            S_OK
        );
        assert_eq!(
            crate::list::create_list(null, 0, 0, allocator, null, &list_iid, &mut list),
            S_OK
        );

        #[repr(C)]
        struct Location {
            resource: This,
            kind: u32,
            footprint: PlacedFootprint,
        }
        let rows = PlacedFootprint {
            offset: 0,
            format: 28,
            width: 4,
            height: 4,
            depth: 1,
            row_pitch: 256,
        };
        let to = Location {
            resource: image,
            kind: 0,
            footprint: PlacedFootprint { offset: 0, ..rows },
        };
        let from = Location {
            resource: source,
            kind: 1,
            footprint: rows,
        };
        type Copy =
            extern "system" fn(This, *const Location, u32, u32, u32, *const Location, *const u8);
        slot::<Copy>(list, 0x80)(list, &to, 0, 0, 0, &from, null.cast());
        slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list);
        slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);

        // Read the host texture back into a readback buffer.
        let readback_heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let readback_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: 1024,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut readback: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a readback buffer.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &readback_heap,
                    D3D12_HEAP_FLAG_NONE,
                    &readback_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .unwrap();
        }
        let readback = readback.unwrap();
        let host_image = host_resource(image).unwrap();
        let back = host.new_list().unwrap();
        let (_, before) = host_transition(image, 0x800).unwrap();
        // The upload list holds the texture until it is reset.
        assert!(std::sync::Arc::strong_count(&host_copy(image).unwrap()) > 2);
        use windows::core::Interface;
        let out_to = Location {
            resource: readback.as_raw().cast_const().cast(),
            kind: 1,
            footprint: rows,
        };
        let out_from = Location {
            resource: host_image.as_raw().cast_const().cast(),
            kind: 0,
            footprint: to.footprint,
        };
        // SAFETY: an open list, live resources, locations in the host layout.
        unsafe {
            back.recording().ResourceBarrier(&[crate::host::transition(
                host_image,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            back.recording().CopyTextureRegion(
                (&raw const out_to).cast(),
                0,
                0,
                0,
                (&raw const out_from).cast(),
                None,
            );
        }
        back.close();
        host.execute(&[&back]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copy finished (execute waits); the buffer is 1024 bytes.
        let read = unsafe {
            readback
                .Map(
                    0,
                    Some(&D3D12_RANGE {
                        Begin: 0,
                        End: 1024,
                    }),
                    Some(&mut mapped),
                )
                .unwrap();
            std::slice::from_raw_parts(mapped.cast::<u8>(), 1024)
        };
        for row in 0..4 {
            let at = row * 256;
            assert_eq!(read[at..at + 16], bytes[at..at + 16], "row {row}");
        }
    }

    #[test]
    fn committed_and_placed_resources_map_at_their_gpu_address() {
        let null = std::ptr::null();
        let mut committed = null;
        let hr = create_committed(
            null,
            &PROPERTIES,
            0,
            &buffer(256),
            0,
            null.cast(),
            &RESOURCE_IIDS[0],
            &mut committed,
        );
        assert_eq!(hr, S_OK);
        let mut data = std::ptr::null_mut();
        assert_eq!(
            slot::<Map>(committed, 0x40)(committed, 0, null.cast(), &mut data),
            S_OK
        );
        assert_eq!(slot::<Address>(committed, 0x58)(committed), data as u64);
        // SAFETY: the buffer is at least 256 zeroed bytes.
        assert_eq!(unsafe { *data.add(255) }, 0);
        assert_eq!(slot::<Release>(committed, 0x10)(committed), 0);

        let heap = heap(4 * KIB_64);
        let place = |offset, desc: &ResourceDesc, out: &mut This| {
            create_placed(
                null,
                heap,
                offset,
                desc,
                0,
                null.cast(),
                &RESOURCE_IIDS[0],
                out,
            )
        };
        let mut placed = null;
        assert_eq!(place(KIB_64, &buffer(KIB_64), &mut placed), S_OK);
        // SAFETY: a heap we made.
        let base = unsafe { (*heap.cast::<Heap>()).memory.ptr.as_ptr() } as u64;
        assert_eq!(slot::<Address>(placed, 0x58)(placed), base + KIB_64);
        let mut other = null;
        assert_eq!(
            place(4 * KIB_64, &buffer(1), &mut other),
            E_INVALIDARG,
            "past the end"
        );
        assert_eq!(
            place(100, &buffer(1), &mut other),
            E_INVALIDARG,
            "misaligned"
        );
        // The resource keeps the heap alive.
        assert_eq!(slot::<Release>(heap, 0x10)(heap), 1);
        assert_eq!(slot::<Release>(placed, 0x10)(placed), 0);
    }

    #[cfg(windows)]
    #[test]
    fn msaa_textures_drop_unordered_access() {
        let msaa_rt = ResourceDesc {
            sample_count: 4,
            flags: 1 | UNORDERED,
            ..texture(64, 64, 28, 1)
        };
        let host = host_desc(&msaa_rt).expect("host texture description");
        assert_eq!(host.flags, 1, "MSAA render target drops UAV but keeps RT");
        assert_eq!(crate::host::legal_for(host.flags, 0x8), 0);

        let msaa_srgb = ResourceDesc {
            sample_count: 2,
            flags: 1 | UNORDERED,
            ..texture(64, 64, 29, 1)
        };
        let host_srgb = host_desc(&msaa_srgb).expect("host sRGB texture description");
        assert_eq!(host_srgb.flags, 1);
        assert_eq!(host_srgb.format, 29, "sRGB stays typed once UAV is dropped");

        let single = ResourceDesc {
            sample_count: 1,
            flags: 1 | UNORDERED,
            ..texture(64, 64, 28, 1)
        };
        assert_eq!(host_desc(&single).unwrap().flags, 1 | UNORDERED);
    }
}
