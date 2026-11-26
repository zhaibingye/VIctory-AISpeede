//! Descriptor heaps: zeroed memory, with handles that are plain addresses
//! (CPU and GPU alike, as on the console).
//!
//! Descriptor sizes: views 32 bytes and samplers 16, the size of RDNA 2's
//! image and sampler descriptors, so a title that copies descriptors itself
//! moves whole ones. Render target and depth views are ours alone; 128 bytes
//! leaves room for whatever stride a title assumes.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::object::{self, DESCRIPTOR_HEAP, E_INVALIDARG, HRESULT, Iid, Object, Table, This, guid};
use crate::resource::{Memory, PAGEABLE_IIDS};

#[cfg(windows)]
pub(crate) mod host;

const HEAP_IIDS: [Iid; 4] = [
    guid(
        0x8efb471d,
        0x616c,
        0x4f49,
        [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeapDesc {
    kind: u32,
    count: u32,
    flags: u32,
    nodes: u32,
}

/// Newer titles (Grand Theft Auto VI) pass the description followed by
/// memory of their own for the descriptors: an address and its size in
/// bytes, or a null address to have the heap allocate (inferred from the
/// title's call sites; older titles pass the plain description).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Placement {
    address: u64,
    size: u64,
}

#[repr(C)]
pub(crate) struct Heap {
    object: Object,
    desc: HeapDesc,
    /// The first descriptor: in `memory`, or in the title's own memory.
    pub(crate) start: u64,
    memory: Option<Memory>,
    /// Shader-visible view and sampler heaps get a host heap with the same
    /// descriptor order when a draw first uses them (decision 0007, step
    /// 3); slots written before wait in `pending`.
    #[cfg(windows)]
    host: std::sync::OnceLock<Option<host::HostHeap>>,
    #[cfg(windows)]
    pending: std::sync::Mutex<Vec<u32>>,
    /// A shader-visible heap in the title's memory, tracked: the title also
    /// copies descriptors into it itself, as bytes (`sync_heaps`).
    #[cfg(windows)]
    written: Option<crate::watch::Tracked>,
}

pub(crate) fn heap<'a>(this: This) -> &'a Heap {
    // SAFETY: called from descriptor heap methods only.
    unsafe { &*this.cast::<Heap>() }
}

impl Heap {
    /// Whether this is a sampler heap.
    pub(crate) fn samplers(&self) -> bool {
        self.desc.kind == SAMPLERS
    }

    /// The host heap, made on first use from what the title wrote so far.
    /// Made late because a host view heap the size of Grand Theft Auto VI's
    /// costs over 100 MB of commit, which its boot needs for its page pool.
    #[cfg(windows)]
    pub(crate) fn host(&self) -> Option<&host::HostHeap> {
        if let Some(h) = self.host.get() {
            return h.as_ref();
        }
        if self.desc.flags & SHADER_VISIBLE == 0 || !matches!(self.desc.kind, VIEWS_KIND | SAMPLERS)
        {
            return None;
        }
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        // A recorder can take the fast path as soon as the OnceLock is
        // initialized. Publish only after replaying the queued writes.
        self.host
            .get_or_init(|| {
                let step = u64::from(increment(self.desc.kind)?);
                let h = host::HostHeap::new(self.samplers(), self.desc.count)?;
                for i in pending.drain(..) {
                    h.mirror(i, self.start + u64::from(i) * step, false);
                }
                Some(h)
            })
            .as_ref()
    }

    /// The descriptor index of `handle` in this heap, if it lies in it.
    pub(crate) fn index(&self, handle: u64) -> Option<u32> {
        let step = u64::from(increment(self.desc.kind)?);
        let at = handle.checked_sub(self.start)?;
        let index = at / step;
        (index < u64::from(self.desc.count)).then_some(index as u32)
    }
}

const VIEWS_KIND: u32 = 0;
const SAMPLERS: u32 = 1;
const SHADER_VISIBLE: u32 = 1;

/// Shader-visible heaps by address range, for mirroring descriptor writes:
/// start, end, heap.
#[cfg(windows)]
static SHADER_HEAPS: std::sync::Mutex<Vec<(u64, u64, usize)>> = std::sync::Mutex::new(Vec::new());

/// Makes the host descriptor for what the title wrote at `handle`, if it
/// is in a shader-visible heap. Keeps `SHADER_HEAPS` locked for the duration
/// so `Heap::drop` cannot free the heap concurrently.
#[cfg(windows)]
fn mirror(handle: u64) {
    let heaps = SHADER_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(&(_, _, at)) = heaps
        .iter()
        .find(|&&(start, end, _)| (start..end).contains(&handle))
    else {
        return;
    };
    // SAFETY: registered heaps stay alive until `Heap::drop`, which takes
    // the `SHADER_HEAPS` lock we hold here.
    let heap = unsafe { &*(at as *const Heap) };
    let Some(index) = heap.index(handle) else {
        return;
    };
    let mut pending = heap.pending.lock().unwrap_or_else(|e| e.into_inner());
    match heap.host.get() {
        None => pending.push(index),
        Some(host) => {
            drop(pending);
            if let Some(host) = host {
                host.mirror(index, handle, false);
            }
        }
    }
}

/// Mirrors what the title wrote into its shader-visible heaps without a
/// call: Grand Theft Auto VI copies descriptors (our records) into its heap
/// as bytes, so only a change of its memory shows them. Each slot of a page
/// written since the last sync is made again if its record changed (in the
/// world 1,300-14,400 slots a frame, against a few hundred views made by
/// calls). Called before each submission: shaders take descriptors from the
/// heap at indices in their data, where a slot never mirrored reads nothing.
#[cfg(windows)]
pub(crate) fn sync_heaps() {
    // Held throughout: a heap is not dropped while its slots are made.
    let heaps = SHADER_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    for &(start, _, at) in heaps.iter() {
        // SAFETY: registered heaps are live (removed when dropped).
        let heap = unsafe { &*(at as *const Heap) };
        let (Some(Some(host)), Some(written), Some(step)) =
            (heap.host.get(), &heap.written, increment(heap.desc.kind))
        else {
            continue;
        };
        let step = u64::from(step);
        for (from, to) in written.take(start as usize, usize::MAX) {
            // By page: one with nothing mapped (the title unmaps heap pages
            // too) holds no records.
            let page_size = crate::watch::PAGE;
            for page in (from & !(page_size - 1)..to).step_by(page_size) {
                let (page, end) = (page.max(from), (page + page_size).min(to));
                if !crate::host::read_into(page as *const u8, &mut [0u8; 4]) {
                    continue;
                }
                let (first, last) = ((page as u64 - start) / step, (end as u64 - start).div_ceil(step));
                for index in first..last.min(u64::from(heap.desc.count)) {
                    host.mirror(index as u32, start + index * step, false);
                }
            }
        }
    }
}

/// Gives back dynamic slots of the host heap behind the title heap at
/// `heap`, if that heap still exists.
#[cfg(windows)]
pub(crate) fn release_dynamic(heap: usize, first: u32, n: u32) {
    let heaps = SHADER_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    if heaps.iter().any(|&(_, _, h)| h == heap)
        // SAFETY: registered heaps are live (removed when dropped).
        && let Some(Some(host)) = unsafe { &*(heap as *const Heap) }.host.get()
    {
        host.release(first, n);
    }
}

/// Runs `f` on the host heap behind the title heap at `heap` while that
/// heap cannot be released; None if it is gone or has no host heap.
#[cfg(windows)]
pub(crate) fn with_host<R>(heap: usize, f: impl FnOnce(&host::HostHeap) -> R) -> Option<R> {
    let heaps = SHADER_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    if !heaps.iter().any(|&(_, _, h)| h == heap) {
        return None;
    }
    // SAFETY: registered heaps are live (removed when dropped, which waits
    // for the lock we hold).
    let host = unsafe { &*(heap as *const Heap) }.host.get()?.as_ref()?;
    Some(f(host))
}

#[cfg(windows)]
impl Drop for Heap {
    fn drop(&mut self) {
        let at = std::ptr::from_ref::<Heap>(self) as usize;
        let mut heaps = SHADER_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
        heaps.retain(|&(_, _, h)| h != at);
    }
}

/// Bytes per descriptor, by heap type (views, samplers, render targets,
/// depth).
fn increment(kind: u32) -> Option<u32> {
    [32, 16, 128, 128].get(kind as usize).copied()
}

/// Device method.
pub(crate) extern "system" fn increment_size(_: This, kind: u32) -> u32 {
    increment(kind).unwrap_or(0)
}

/// What our render target views hold: the resource and the view
/// description, turned into a host view when a command uses them.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RtvRecord {
    tag: u32,
    described: u32,
    pub(crate) resource: This,
    desc: [u32; 6],
}
const RTV_TAG: u32 = u32::from_le_bytes(*b"vRTV");
const _: () = assert!(size_of::<RtvRecord>() <= 112);

impl RtvRecord {
    /// The view description in the host's type (the same layout).
    #[cfg(windows)]
    pub(crate) fn desc(
        &self,
    ) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_RENDER_TARGET_VIEW_DESC> {
        const TEXTURE_2D: u32 = 4;
        self.host_desc(TEXTURE_2D)
    }

    /// The depth view description in the host's type (the same layout).
    #[cfg(windows)]
    pub(crate) fn depth_desc(
        &self,
    ) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_DEPTH_STENCIL_VIEW_DESC> {
        use windows::Win32::Graphics::Direct3D12::{D3D12_DSV_DIMENSION_TEXTURE2D, D3D12_DSV_DIMENSION_TEXTURE2DMS};
        let dimension = if self.described == 0 && crate::resource::host_resource(self.resource)
            .is_some_and(|resource| unsafe { resource.GetDesc() }.SampleDesc.Count > 1)
        {
            D3D12_DSV_DIMENSION_TEXTURE2DMS
        } else {
            D3D12_DSV_DIMENSION_TEXTURE2D
        };
        self.host_desc(dimension.0 as u32)
    }

    /// The description words as a host type, the format as the host has it.
    /// Without one, a typeless host copy gets a 2D view (`dimension`) in
    /// the title's format, which the host needs spelled out.
    #[cfg(windows)]
    fn host_desc<T>(&self, dimension: u32) -> Option<T> {
        assert!(size_of::<T>() <= size_of::<[u32; 6]>());
        let mut words = self.host_words(dimension, || crate::resource::view_format(self.resource))?;
        words[0] = crate::resource::pc_view_format(self.resource, words[0]);
        // SAFETY: both host view descriptions are 24 bytes in this layout.
        Some(unsafe { (&raw const words).cast::<T>().read_unaligned() })
    }

    /// [`RtvRecord::host_desc`]'s words; `own` is the format a view of the
    /// resource must name if its host copy is typeless (none if typed).
    #[cfg(windows)]
    fn host_words(&self, dimension: u32, own: impl FnOnce() -> Option<u32>) -> Option<[u32; 6]> {
        let mut words = if self.described == 0 {
            [own()?, dimension, 0, 0, 0, 0]
        } else {
            let mut words = self.desc;
            // "The resource's format" (unknown) names none to a typeless copy.
            if words[0] == 0 {
                words[0] = own().unwrap_or(0);
            }
            words
        };
        words[0] = crate::resource::pc_format(words[0]);
        Some(words)
    }
}

/// Device method: records the view at `handle` (descriptor memory).
pub(crate) extern "system" fn create_rtv(
    _: This,
    resource: This,
    desc: *const [u32; 6],
    handle: u64,
) {
    if handle == 0 {
        return;
    }
    // SAFETY: the title passes a description or null.
    let described = unsafe { desc.as_ref() };
    let record = RtvRecord {
        tag: RTV_TAG,
        described: u32::from(described.is_some()),
        resource,
        desc: described.copied().unwrap_or_default(),
    };
    // SAFETY: the handle is a descriptor slot, at least 112 bytes.
    unsafe { (handle as *mut RtvRecord).write_unaligned(record) };
}

/// Device method: a depth view, recorded as render target views are.
pub(crate) extern "system" fn create_dsv(
    this: This,
    resource: This,
    desc: *const [u32; 6],
    handle: u64,
) {
    create_rtv(this, resource, desc, handle);
    if handle != 0 {
        // SAFETY: just written by `create_rtv`.
        unsafe { (handle as *mut u32).write_unaligned(DSV_TAG) };
    }
}
const DSV_TAG: u32 = u32::from_le_bytes(*b"vDSV");

/// A shader view or sampler as the title described it (descriptions in the
/// host's layout, kept as words), for the host to build its own from.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum View {
    /// Constant buffer: GPU address and size.
    Constant(u64, u32),
    /// A null constant-buffer descriptor.
    NullConstant,
    /// Shader resource: resource, description (none: the resource's own).
    Shader(usize, Option<[u32; 10]>),
    /// Unordered access: resource, counter resource, description.
    Unordered(usize, usize, Option<[u32; 10]>),
    Sampler([u32; 13]),
    /// The console's own sampler description: flags, filters per axis,
    /// addressing, LOD and anisotropy settings, border color, then
    /// console-only tuning words.
    ConsoleSampler([u32; 27]),
    /// A view of memory without a resource object: whether it is unordered
    /// access, the resource description, then the view's (the plain one
    /// followed by the memory address and console fields).
    Placed(bool, [u32; 14], [u32; 16]),
}

/// Every distinct view in use; a descriptor holds a tag, an index into it
/// and the entry's generation, which survive titles copying descriptors as
/// bytes. A view of a resource lives as long as the resource: when the
/// title releases it, its entries are freed for reuse under a new
/// generation, so descriptors (and host slots made from them) naming the
/// old one go stale instead of naming whatever is made at the same address
/// next.
/// shortcut: views without a resource object (constant buffers, memory
/// views, samplers) are kept for the run; they are bounded by the title's
/// memory and sampler states.
struct Views {
    /// Each entry's view and generation.
    list: Vec<(View, u32)>,
    index: std::collections::BTreeMap<View, u32>,
    /// Entries naming each resource.
    by_resource: std::collections::BTreeMap<usize, Vec<u32>>,
    free: Vec<u32>,
}

static VIEWS: std::sync::Mutex<Views> = std::sync::Mutex::new(Views {
    list: Vec::new(),
    index: std::collections::BTreeMap::new(),
    by_resource: std::collections::BTreeMap::new(),
    free: Vec::new(),
});
const VIEW_TAG: u32 = u32::from_le_bytes(*b"vVIE");

/// 仅保留能决定如何修复的阶段，不输出资源地址。
#[cfg(windows)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TextureSrvReject {
    ViewKind,
    ViewShape,
    ViewRange,
    AddressRange,
    NativeAllocation,
    WriterAlias,
    SurrogateFormat,
    ScopedAuthority,
    NativeOrBytes { external_alias: bool, host_format: i32 },
}

/// 地址式 SRV 的布局签名。没有地址、指针、资源内容或用户路径。
#[cfg(windows)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlacedSrvShape {
    dimension: u32,
    width: u64,
    height: u32,
    slices: u16,
    mips: u16,
    layout: u32,
    flags: u32,
    mapping: u32,
    first_view_mip: u32,
    view_mips: u32,
}

impl View {
    /// A canonical 2D view with exact native color or depth format.
    /// A multi-mip view requires native authority for every named mip.
    #[cfg(windows)]
    pub(crate) fn paged_texture_srv(&self, array: bool)
        -> Result<crate::page_texture::TextureInput, TextureSrvReject> {
        use TextureSrvReject::*;
        use crate::page_texture::{MipSelection, TextureInput, TextureSelection};
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
        let (resource, desc) = match *self {
            Self::Shader(resource, desc) => (resource as This, desc),
            #[cfg(feature = "paged-title")]
            Self::Placed(false, _, _) => {
                let (resource, words) = host::paged_placed_texture(self, array)?;
                (resource, Some(words))
            }
            _ => return Err(ViewKind),
        };
        let copy = crate::resource::host_copy(resource).ok_or(NativeAllocation)?;
        // SAFETY: the retained native texture is live.
        let actual = unsafe { copy.texture.GetDesc() };
        let (format, mut selection) = if let Some(words) = desc {
            host::paged_2d_srv(&words).ok_or(ViewShape)?
        } else {
            let slices = (actual.DepthOrArraySize > 1).then_some(MipSelection {
                first: 0, count: u32::from(actual.DepthOrArraySize),
            });
            (DXGI_FORMAT(crate::resource::title_format(resource).ok_or(ViewKind)? as i32), TextureSelection {
                mips: MipSelection { first: 0, count: u32::MAX }, slices,
            })
        };
        if array != selection.slices.is_some() { return Err(ViewShape); }
        selection.mips = MipSelection::resolve(
            selection.mips.first, selection.mips.count, u32::from(actual.MipLevels),
        ).ok_or(ViewRange)?;
        if desc.is_some() && !array && selection.mips.first == 0 && selection.mips.count == 1
            && let Some(depth) = TextureInput::native_depth(copy.clone(), format)
        {
            return Ok(depth);
        }
        if crate::resource::stands_in(resource) {
            return Err(SurrogateFormat);
        }
        if actual.MipLevels > 1 || array {
            return TextureInput::scoped_color(copy, format, selection).ok_or(ScopedAuthority);
        }
        let plain_linear = crate::resource::paged_linear_texture(resource).filter(|input| input.format == format);
        #[cfg(feature = "paged-title")]
        let compatible = copy.exact_committed_bytes();
        #[cfg(not(feature = "paged-title"))]
        let compatible = None;
        // Pick between the canonical native image and the original linear
        // bytes at execution, after earlier resolves/uploads are ordered.
        let external_alias = copy.uses_unproven_alias_backing();
        TextureInput::native(copy.clone(), format, compatible.or(plain_linear)).or_else(|| {
            plain_linear.map(|input| TextureInput::Linear { copy, input })
        }).ok_or(NativeOrBytes { external_alias, host_format: actual.Format.0 })
    }

    /// 仅游戏显式绑定的带类型空 SRV 才可生成宿主空视图，绝不替代真实纹理。
    #[cfg(windows)]
    pub(crate) fn paged_null_texture_srv(&self, array: bool) -> Option<(
        windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
        crate::page_texture::TextureSelection,
    )> {
        use windows::Win32::Graphics::Direct3D12::{D3D12_REQ_MIP_LEVELS, D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION};
        let Self::Shader(0, Some(words)) = self else { return None; };
        let (format, selection) = host::paged_2d_srv(words)?;
        let within = |s: crate::page_texture::MipSelection, limit| {
            s.count > 0 && s.first.checked_add(s.count).is_some_and(|end| end <= limit)
        };
        if !crate::page_texture::sample_color(format)
            || array != selection.slices.is_some()
            || !within(selection.mips, D3D12_REQ_MIP_LEVELS)
            || selection.slices.is_some_and(|s| !within(s, D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION))
        {
            return None;
        }
        Some((format, selection))
    }

    /// 只记录可用于区分空、放置式、直接资源及其他类型的格式信息，省略地址和内容。
    #[cfg(windows)]
    pub(crate) fn texture_srv_metadata(&self) -> (&'static str, Option<u32>, Option<u32>, Option<u32>, Option<PlacedSrvShape>) {
        match self {
            Self::Shader(resource, desc) => (
                if *resource == 0 { "null-srv" } else { "resource-srv" },
                crate::resource::title_format(*resource as This),
                desc.as_ref().map(|words| words[0]),
                desc.as_ref().map(|words| words[1]),
                None,
            ),
            Self::Placed(false, resource, desc) => (
                "placed-srv", Some(resource[8]), Some(desc[0]), Some(desc[1]),
                Some(PlacedSrvShape {
                    dimension: resource[0],
                    width: u64::from(resource[4]) | (u64::from(resource[5]) << 32),
                    height: resource[6],
                    slices: resource[7] as u16,
                    mips: (resource[7] >> 16) as u16,
                    layout: resource[11], flags: resource[12], mapping: desc[2],
                    first_view_mip: desc[4], view_mips: desc[5],
                }),
            ),
            Self::Placed(true, resource, desc) => (
                "placed-uav", Some(resource[8]), Some(desc[0]), Some(desc[1]), None,
            ),
            Self::Unordered(_, _, _) => ("uav", None, None, None, None),
            Self::Constant(..) | Self::NullConstant => ("cbv", None, None, None, None),
            Self::Sampler(_) | Self::ConsoleSampler(_) => ("sampler", None, None, None, None),
        }
    }

    /// The resource objects the view names.
    fn resources(&self) -> [usize; 2] {
        match *self {
            View::Shader(r, _) => [r, 0],
            View::Unordered(r, counter, _) => [r, counter],
            _ => [0, 0],
        }
    }
}

fn record(handle: u64, view: View) {
    if handle == 0 {
        return;
    }
    let mut views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    let i = match views.index.get(&view) {
        Some(&i) => i,
        None => {
            let i = match views.free.pop() {
                Some(i) => {
                    views.list[i as usize].0 = view.clone();
                    i
                }
                None => {
                    views.list.push((view.clone(), 1));
                    views.list.len() as u32 - 1
                }
            };
            for r in view.resources().into_iter().filter(|&r| r != 0) {
                views.by_resource.entry(r).or_default().push(i);
            }
            views.index.insert(view, i);
            i
        }
    };
    let generation = views.list[i as usize].1;
    // SAFETY: the handle is a descriptor slot, at least 16 bytes.
    unsafe { (handle as *mut [u32; 3]).write_unaligned([VIEW_TAG, i, generation]) };
    drop(views);
    #[cfg(windows)]
    mirror(handle);
}

/// The title released `resource`: the views naming it go stale.
pub(crate) fn forget(resource: usize) {
    let mut views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(entries) = views.by_resource.remove(&resource) else {
        return;
    };
    for i in entries {
        let (view, generation) = views.list[i as usize].clone();
        // Entries freed through a view's other resource may hold another
        // view by now.
        if views.index.get(&view) != Some(&i) || !view.resources().contains(&resource) {
            continue;
        }
        views.index.remove(&view);
        views.list[i as usize].1 = generation.wrapping_add(1).max(1);
        // The view's other resource (a counter) keeps a stale entry, which
        // `forget` of that resource skips.
        views.free.push(i);
    }
}

/// The view of entry `i` if `generation` is still its own.
#[cfg(windows)]
fn view_at(i: u32, generation: u32) -> Option<View> {
    let views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    let (view, current) = views.list.get(i as usize)?;
    (*current == generation).then(|| view.clone())
}

/// Compact table snapshots reuse encoded view identities, not title
/// addresses. Generation zero is reserved for a null/stale slot.
#[cfg(windows)]
pub(crate) type TableRecords = Box<[[u32; 2]]>;

#[cfg(windows)]
impl Heap {
    /// VNME 仅捕获着色器已证明会访问的槽，避免分配和扫描此前的所有槽。
    pub(crate) fn proven_table_view(&self, first: u64, slot: u32) -> Option<View> {
        if self.desc.kind != VIEWS_KIND || first.checked_sub(self.start)? % 32 != 0 {
            return None;
        }
        let at = first.checked_add(u64::from(slot) * 32)?;
        self.index(at)?;
        let views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: checked aligned slot within this live 32-byte descriptor heap.
        let [tag, index, generation] = unsafe { (at as *const [u32; 3]).read_unaligned() };
        if tag != VIEW_TAG || generation == 0 {
            return None;
        }
        let (view, current) = views.list.get(index as usize)?;
        (*current == generation).then(|| view.clone())
    }

    pub(crate) fn table_records(&self, first: u64, count: u32) -> Option<TableRecords> {
        if self.desc.kind != VIEWS_KIND || count == 0 {
            return None;
        }
        let offset = first.checked_sub(self.start)?;
        if offset % 32 != 0 {
            return None;
        }
        self.index(first.checked_add(u64::from(count - 1) * 32)?)?;
        let views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
        Some((0..count).map(|i| {
            let at = first + u64::from(i) * 32;
            // SAFETY: The entire interval is inside the bound title heap; the record may be unaligned.
            let [tag, index, generation] = unsafe { (at as *const [u32; 3]).read_unaligned() };
            if tag == VIEW_TAG && generation != 0
                && views.list.get(index as usize).is_some_and(|v| v.1 == generation)
            {
                [index, generation]
            } else {
                [0, 0]
            }
        }).collect())
    }
}

#[cfg(windows)]
pub(crate) fn table_view([index, generation]: [u32; 2]) -> Option<View> {
    (generation != 0).then(|| view_at(index, generation)).flatten()
}

/// The view recorded at `handle`, if one is (and its resource is alive).
pub(crate) fn view(handle: u64) -> Option<View> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
    let views = VIEWS.lock().unwrap_or_else(|e| e.into_inner());
    let (view, current) = views.list.get(i as usize)?;
    (tag == VIEW_TAG && *current == generation).then(|| view.clone())
}

/// A description of `N` words, if the title passes one.
fn words<const N: usize>(desc: *const [u32; N]) -> Option<[u32; N]> {
    // SAFETY: the title passes a description or null.
    (!desc.is_null()).then(|| unsafe { desc.read_unaligned() })
}

/// Device method.
pub(crate) extern "system" fn create_cbv(_: This, desc: *const [u32; 4], handle: u64) {
    if let Some([lo, hi, size, _]) = words(desc) {
        let address = u64::from(lo) | u64::from(hi) << 32;
        record(handle, View::Constant(address, size));
    } else {
        record(handle, View::NullConstant);
    }
}

/// Device method.
pub(crate) extern "system" fn create_srv(
    _: This,
    resource: This,
    desc: *const [u32; 10],
    handle: u64,
) {
    record(handle, View::Shader(resource as usize, words(desc)));
}

/// Device method.
pub(crate) extern "system" fn create_uav(
    _: This,
    resource: This,
    counter: This,
    desc: *const [u32; 10],
    handle: u64,
) {
    let view = View::Unordered(resource as usize, counter as usize, words(desc));
    record(handle, view);
}

/// Device methods 0x230 and 0x238: shader resource and unordered access
/// views of memory the title placed itself, `(resource desc, view desc,
/// handle)`.
pub(crate) extern "system" fn create_placed_srv(
    _: This,
    resource: *const [u32; 14],
    desc: *const [u32; 16],
    handle: u64,
) {
    if let (Some(resource), Some(desc)) = (words(resource), words(desc)) {
        record(handle, View::Placed(false, resource, desc));
    }
}

pub(crate) extern "system" fn create_placed_uav(
    _: This,
    resource: *const [u32; 14],
    desc: *const [u32; 16],
    handle: u64,
) {
    if let (Some(resource), Some(desc)) = (words(resource), words(desc)) {
        record(handle, View::Placed(true, resource, desc));
    }
}

/// Device method.
pub(crate) extern "system" fn create_sampler(_: This, desc: *const [u32; 13], handle: u64) {
    if let Some(desc) = words(desc) {
        record(handle, View::Sampler(desc));
    }
}

/// Device method 0x1e8.
pub(crate) extern "system" fn create_sampler_x(_: This, desc: *const [u32; 27], handle: u64) {
    if let Some(desc) = words(desc) {
        record(handle, View::ConsoleSampler(desc));
    }
}

/// Device method: `count` descriptors of `kind` from `src` to `dst`.
pub(crate) extern "system" fn copy_simple(_: This, count: u32, dst: u64, src: u64, kind: u32) {
    let Some(step) = increment(kind).map(u64::from) else {
        return;
    };
    let len = u64::from(count) * step;
    if dst != 0 && src != 0 {
        // SAFETY: both ranges are descriptor slots the title owns.
        unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, len as usize) };
        #[cfg(windows)]
        for i in 0..u64::from(count) {
            mirror(dst + i * step);
        }
    }
}

/// Device method: ranges of descriptors (start and count each) gathered
/// from the sources and scattered to the destinations, in order.
#[allow(clippy::too_many_arguments)] // the method's own arguments
pub(crate) extern "system" fn copy(
    _: This,
    dst_ranges: u32,
    dst_starts: *const u64,
    dst_sizes: *const u32,
    src_ranges: u32,
    src_starts: *const u64,
    src_sizes: *const u32,
    kind: u32,
) {
    let Some(step) = increment(kind) else {
        return;
    };
    // Each range as (start, count); a null size array means one descriptor
    // per range.
    let ranges = |n: u32, starts: *const u64, sizes: *const u32| -> Vec<(u64, u32)> {
        (0..n as usize)
            .map(|i| {
                // SAFETY: the title passes `n` starts and sizes (or null sizes).
                unsafe {
                    (
                        *starts.add(i),
                        if sizes.is_null() { 1 } else { *sizes.add(i) },
                    )
                }
            })
            .collect()
    };
    if dst_starts.is_null() || src_starts.is_null() {
        return;
    }
    let src = ranges(src_ranges, src_starts, src_sizes);
    let mut dst = ranges(dst_ranges, dst_starts, dst_sizes).into_iter();
    let (mut to, mut room) = dst.next().unwrap_or((0, 0));
    for (mut from, mut left) in src {
        while left > 0 {
            while room == 0 {
                let Some(next) = dst.next() else { return };
                (to, room) = next;
            }
            let n = left.min(room);
            copy_simple(std::ptr::null(), n, to, from, kind);
            let bytes = u64::from(n) * u64::from(step);
            (to, from, room, left) = (to + bytes, from + bytes, room - n, left - n);
        }
    }
}

/// The view recorded at `handle`, if one is.
pub(crate) fn rtv_record(handle: u64) -> Option<RtvRecord> {
    tagged_record(handle, RTV_TAG)
}

/// The depth view recorded at `handle`, if one is.
pub(crate) fn dsv_record(handle: u64) -> Option<RtvRecord> {
    tagged_record(handle, DSV_TAG)
}

fn tagged_record(handle: u64, tag: u32) -> Option<RtvRecord> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let record = unsafe { (handle as *const RtvRecord).read_unaligned() };
    (record.tag == tag).then_some(record)
}

/// Device method.
pub(crate) extern "system" fn create_heap(
    _: This,
    desc_ptr: *const HeapDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc_ptr.as_ref() }) else {
        return E_INVALIDARG;
    };
    let Some(size) = increment(desc.kind).map(|i| u64::from(i) * u64::from(desc.count.max(1)))
    else {
        return E_INVALIDARG;
    };
    // SAFETY: read only as a hint: a title that passes the plain
    // description has its own stack or data after it.
    let placement = unsafe { desc_ptr.add(1).cast::<Placement>().read_unaligned() };
    // shortcut: a placement is recognised by its size matching the count,
    // since the plain and extended descriptions share the slot and the
    // interface version; find the flag that tells them apart if a title
    // trips over this.
    let placed = placement.address != 0 && placement.size == size;
    tracing::debug!(?desc, placed, "描述符堆");
    let (start, memory) = if placed {
        if placement.address.checked_add(size).is_none() {
            return E_INVALIDARG;
        }
        (placement.address, None)
    } else {
        match Memory::zeroed(size, 64) {
            Ok(memory) => (memory.address(), Some(memory)),
            Err(hr) => return hr,
        }
    };
    let visible = desc.flags & SHADER_VISIBLE != 0 && matches!(desc.kind, VIEWS_KIND | SAMPLERS);
    let heap = Box::new(Heap {
        object: Object::new(&FUNCTIONS),
        desc,
        start,
        memory,
        #[cfg(windows)]
        host: std::sync::OnceLock::new(),
        #[cfg(windows)]
        pending: Default::default(),
        #[cfg(windows)]
        written: (placed && visible)
            .then(|| crate::watch::track(start as usize, size as usize, true))
            .flatten(),
    });
    #[cfg(windows)]
    if visible {
        let at = std::ptr::from_ref::<Heap>(&heap) as usize;
        let mut heaps = SHADER_HEAPS.lock().unwrap_or_else(|e| e.into_inner());
        heaps.push((start, start + size, at));
    }
    #[cfg(not(windows))]
    let _ = visible;
    object::hand_out(DESCRIPTOR_HEAP, heap, iid, &HEAP_IIDS, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &HEAP_IIDS).then_some(this);
    object::answer(DESCRIPTOR_HEAP, iid, found, out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}
extern "system" fn desc(this: This, out: *mut HeapDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(heap(this).desc) };
    }
}
/// The same address for the CPU and the GPU start.
extern "system" fn start(this: This) -> u64 {
    heap(this).start
}

/// IUnknown 3, object 4, device child 1, heap 3.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<DESCRIPTOR_HEAP, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = desc as *const (); // 0x40
    f[9] = start as *const (); // 0x48 CPU
    f[10] = start as *const (); // 0x50 GPU
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{S_OK, slot};

    /// What newer titles pass; tests use it throughout, so that reading the
    /// placement stays inside the value.
    #[repr(C)]
    struct Extended(HeapDesc, Placement);

    fn create(desc: HeapDesc, placement: Placement, out: &mut This) -> HRESULT {
        let extended = Extended(desc, placement);
        create_heap(std::ptr::null(), &extended.0, &HEAP_IIDS[0], out)
    }

    /// A view description with the unknown format on a texture the host
    /// made typeless (a depth texture shaders read) takes the title's.
    #[cfg(windows)]
    #[test]
    fn an_unknown_format_in_a_view_description_is_the_resources() {
        let record = |format| RtvRecord {
            tag: RTV_TAG,
            described: 1,
            resource: std::ptr::null(),
            desc: [format, 3, 1, 0, 0, 0],
        };
        let d32 = 40;
        assert_eq!(record(0).host_words(3, || Some(d32)).unwrap()[..2], [d32, 3], "typeless host copy");
        assert_eq!(record(0).host_words(3, || None).unwrap()[0], 0, "typed host copy");
        assert_eq!(record(45).host_words(3, || Some(d32)).unwrap()[0], 45, "a format named is kept");
        assert_eq!(record(0).host_words(3, || Some(118)).unwrap()[0], 45, "as the host has it");
    }

    #[test]
    fn heaps_hold_count_descriptors_from_their_start() {
        let null = std::ptr::null();
        let desc = HeapDesc {
            kind: 0,
            count: 1000,
            flags: 1,
            nodes: 0,
        };
        let none = Placement {
            address: 0,
            size: 0,
        };
        let mut h = null;
        assert_eq!(create(desc, none, &mut h), S_OK);
        let cpu = slot::<extern "system" fn(This) -> u64>(h, 0x48)(h);
        assert_eq!(cpu % 64, 0);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x50)(h), cpu);
        assert_eq!(increment_size(null, 0), 32);
        // SAFETY: the heap holds 1000 descriptors of 32 bytes.
        unsafe { (cpu as *mut u8).add(999 * 32 + 31).write(1) };
        assert_eq!(increment_size(null, 9), 0);
        let bad = HeapDesc { kind: 9, ..desc };
        assert_eq!(create(bad, none, &mut h), E_INVALIDARG);
    }

    #[test]
    fn views_survive_copies() {
        let null = std::ptr::null();
        let mut heap = vec![0u64; 5 * 8]; // five 32-byte slots
        let at = |i: usize| heap.as_ptr() as u64 + 32 * i as u64;
        let (s0, s1, s2, s3, s4) = (at(0), at(1), at(2), at(3), at(4));
        create_cbv(null, &[0x1000, 1, 256, 0], s0);
        assert_eq!(view(s0), Some(View::Constant(0x1_0000_1000, 256)));
        create_cbv(null, &[0x2000, 1, 256, 0], s4);
        create_cbv(null, std::ptr::null(), s4);
        assert_eq!(view(s4), Some(View::NullConstant));
        create_srv(null, null, std::ptr::null(), s1);
        assert_eq!(view(s1), Some(View::Shader(0, None)));
        // The same view twice is one entry.
        create_srv(null, null, std::ptr::null(), s2);
        // SAFETY: both slots are in `heap`.
        assert_eq!(unsafe { *(s1 as *const [u32; 3]) }, unsafe {
            *(s2 as *const [u32; 3])
        });
        // Two ranges of one into one range of two.
        let (dst, dst_n, src, src_n) = ([s2], [2u32], [s0, s1], [1u32, 1]);
        copy(
            null,
            1,
            dst.as_ptr(),
            dst_n.as_ptr(),
            2,
            src.as_ptr(),
            src_n.as_ptr(),
            0,
        );
        assert_eq!(view(s2), view(s0));
        assert_eq!(view(s3), view(s1));
        assert_eq!(view(at(0) + 8), None);
        heap[0] = 0;
    }

    /// A released resource's views go stale; the same view of a new
    /// resource at the same address is a new record.
    #[test]
    fn views_die_with_their_resource() {
        let null = std::ptr::null();
        let slots = [0u64; 2 * 4];
        let (a, b) = (slots.as_ptr() as u64, slots.as_ptr() as u64 + 32);
        let resource = 0x7777_0000usize as This;
        create_srv(null, resource, std::ptr::null(), a);
        assert_eq!(view(a), Some(View::Shader(0x7777_0000, None)));
        forget(0x7777_0000);
        assert_eq!(view(a), None, "stale after release");
        create_srv(null, resource, std::ptr::null(), b);
        assert_eq!(view(b), Some(View::Shader(0x7777_0000, None)));
        // SAFETY: both slots are in `slots`.
        let (old, new) = unsafe { (*(a as *const [u32; 3]), *(b as *const [u32; 3])) };
        assert_ne!(old, new, "a new generation");
    }

    #[cfg(windows)]
    #[test]
    fn table_snapshot_tracks_copies_range_and_stale_views() {
        let null = std::ptr::null();
        let desc = HeapDesc { kind: VIEWS_KIND, count: 6, flags: 0, nodes: 0 };
        let mut h = null;
        assert_eq!(create(desc, Placement { address: 0, size: 0 }, &mut h), S_OK);
        let heap = heap(h);
        let at = |i: u32| heap.start + u64::from(i) * 32;
        let r = 0x7b77_0000usize as This;
        create_srv(null, r, null.cast(), at(0));
        create_cbv(null, null.cast(), at(1));
        // SAFETY: two non-overlapping table intervals in the allocated heap.
        unsafe { std::ptr::copy_nonoverlapping(at(0) as *const u8, at(3) as *mut u8, 64) };
        let before = heap.table_records(at(0), 3).unwrap();
        assert_eq!(heap.table_records(at(3), 3).unwrap(), before);
        assert_eq!(table_view(before[1]), Some(View::NullConstant));
        assert_eq!(before[2], [0, 0]);
        assert!(heap.table_records(at(5), 2).is_none());
        assert!(heap.table_records(at(0) + 4, 2).is_none());
        assert_eq!(heap.proven_table_view(at(3), 0), Some(View::Shader(r as usize, None)));
        assert_eq!(heap.proven_table_view(at(3), 1), Some(View::NullConstant));
        assert_eq!(heap.proven_table_view(at(3), 2), None);
        assert_eq!(heap.proven_table_view(at(5), 1), None);
        assert_eq!(heap.proven_table_view(at(0) + 4, 1), None);
        assert_eq!(heap.proven_table_view(u64::MAX & !31, 1), None);
        forget(r as usize);
        assert_eq!(heap.proven_table_view(at(3), 0), None);
        assert_eq!(heap.proven_table_view(at(3), 1), Some(View::NullConstant));
        assert_eq!(heap.table_records(at(3), 3).unwrap()[0], [0, 0]);
        assert!(table_view(before[0]).is_none());
        slot::<extern "system" fn(This) -> u32>(h, 0x10)(h);
    }

    /// A record the title copies into its shader-visible heap as bytes (no
    /// call) is made on the host at the next sync, and only then.
    #[cfg(windows)]
    #[test]
    fn records_copied_as_bytes_are_made_at_the_next_sync() {
        let memory = crate::watch::tests::pool(1);
        let desc = HeapDesc {
            kind: SAMPLERS,
            count: 64,
            flags: SHADER_VISIBLE,
            nodes: 1,
        };
        let mut h = std::ptr::null();
        let placement = Placement {
            address: memory as u64,
            size: 64 * 16,
        };
        assert_eq!(create(desc, placement, &mut h), S_OK);
        if let Some(host) = heap(h).host() {
            // Our record, written where no heap sees it, then copied in.
            let mut elsewhere = [0u64; 2];
            let at = elsewhere.as_mut_ptr() as u64;
            create_sampler(std::ptr::null(), &[21, 3, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0], at);
            let slot = memory as u64 + 5 * 16;
            // SAFETY: 16 bytes of each, ours; the heap's page is guarded
            // (our handler opens it).
            unsafe { std::ptr::copy_nonoverlapping(at as *const u8, slot as *mut u8, 16) };
            assert!(!host.made(5));
            sync_heaps();
            assert!(host.made(5), "made at the sync");
        }
        slot::<extern "system" fn(This) -> u32>(h, 0x10)(h);
        crate::watch::tests::free(memory);
    }

    /// Grand Theft Auto VI places its shader-visible heap in its own memory
    /// and checks that the heap starts there.
    #[test]
    fn placed_heaps_start_at_the_title_memory() {
        let mut memory = vec![0u8; 64 * 32];
        let address = memory.as_mut_ptr() as u64;
        let desc = HeapDesc {
            kind: 0,
            count: 64,
            flags: 1,
            nodes: 1,
        };
        let mut h = std::ptr::null();
        assert_eq!(
            create(
                desc,
                Placement {
                    address,
                    size: 64 * 32
                },
                &mut h
            ),
            S_OK
        );
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x48)(h), address);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x50)(h), address);
        // A size that does not match the count is not a placement.
        assert_eq!(create(desc, Placement { address, size: 7 }, &mut h), S_OK);
        assert_ne!(slot::<extern "system" fn(This) -> u64>(h, 0x48)(h), address);
    }
}
