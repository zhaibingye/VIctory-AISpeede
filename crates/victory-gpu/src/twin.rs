//! Depth textures the title also writes through unordered access (decision
//! 0007, "One host texture per texture memory"). The console allows both
//! uses of one texture (Grand Theft Auto VI's shadow-map copies and its
//! material depth, written by compute and then tested as depth); the host
//! does not: a depth-stencil resource takes no unordered access, in any
//! format. Such a texture gets a writable twin, a colour texture per plane
//! (depth, stencil) that its unordered access views write, and the newer
//! contents move between the two by copies where the use changes: within
//! a list where it records the change (`HostList::transition`), and at
//! submission for each list's first use, from the side the lists before it
//! left newer. A 16-bit depth plane in the host's 24-bit one (the
//! console's 16-bit depth with stencil) is converted on the way, so a
//! value written as 16-bit depth compares equal to the same depth
//! rasterised.
//!
//! shortcut: uses the lists do not name (shaders indexing the heap
//! themselves) move nothing; a whole twin is copied for any use of either
//! plane, also when the next pass overwrites all of it; console 24-bit
//! depth gets no twin. Upgrade when a frame trace shows such a use, or the
//! copies (counted on the frame-rate line) cost.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering::Relaxed};

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::s;

use crate::host::{HostList, host, transition};
use crate::object::This;
use crate::resource::HostTexture;

/// Which side holds the newer contents.
pub(crate) const BOTH: u8 = 0;
pub(crate) const DEPTH: u8 = 1;
pub(crate) const TWIN: u8 = 2;

const RENDER_TARGET: u32 = 0x4;
const UNORDERED_ACCESS: u32 = 0x8;
const DEPTH_WRITE: u32 = 0x10;
const COPY_DEST: u32 = 0x400;
const COPY_SOURCE: u32 = 0x800;
const RESOLVE_DEST: u32 = 0x1000;

/// Copies between a depth texture and its twin since the last frame-rate
/// line (either way, within lists and at submission).
pub(crate) static COPIES: AtomicU64 = AtomicU64::new(0);

pub(crate) struct Twin {
    /// The colour texture of each plane the depth texture has (depth, then
    /// stencil), shader-writable.
    pub(crate) planes: Vec<Arc<HostTexture>>,
    /// Which side is newer as of the lists submitted so far.
    pub(crate) newer: AtomicU8,
    /// Per plane and subresource: how its bytes go through `scratch`.
    routes: Vec<Route>,
    /// A shader-writable buffer the copies go through, in the common state
    /// between them.
    scratch: ID3D12Resource,
}

// SAFETY: D3D12 objects are free-threaded; the rest is atomics and data.
unsafe impl Send for Twin {}
// SAFETY: as above.
unsafe impl Sync for Twin {}

/// One subresource of one plane.
struct Route {
    plane: usize,
    /// Its index in the twin's plane texture and in the depth texture.
    twin_sub: u32,
    depth_sub: u32,
    /// Its place in the scratch buffer as the twin's texture has it.
    twin: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    /// As the depth texture has it, where that differs: a 16-bit depth
    /// plane the host keeps in 24 bits (32 per texel), converted.
    depth: Option<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
}

/// A depth texture's use in one list: the side its first use needs (made
/// newer at submission, `at_submission`) and the side the list left newer
/// (`None`: unchanged from where it started).
pub(crate) struct Use {
    first: u8,
    last: Option<u8>,
    depth: Arc<HostTexture>,
}

/// The twin's plane formats for a console depth format (resource or
/// view format): 16- or 32-bit depth, then 8-bit stencil if it has one.
/// `None` for 24-bit depth, whose host plane takes no unordered view.
fn plane_formats(console: u32) -> Option<&'static [u32]> {
    const R16: u32 = 53;
    const R32: u32 = 39;
    const R8: u32 = 60;
    match console {
        53 | 55 => Some(&[R16]),
        39..=41 => Some(&[R32]),
        118..=120 => Some(&[R16, R8]),
        19..=22 => Some(&[R32, R8]),
        _ => None,
    }
}

/// The twin of the depth texture `depth`, whose title resource has format
/// `console`; `None` if the host makes none.
fn make(console: u32, depth: &ID3D12Resource) -> Option<Twin> {
    let host = host()?;
    let Some(formats) = plane_formats(console) else {
        once(console, 0, 0, "a depth texture of this format gets no writable twin: its unordered writes are dropped");
        return None;
    };
    // SAFETY: a live resource.
    let dd = unsafe { depth.GetDesc() };
    let subs = u32::from(dd.MipLevels) * u32::from(dd.DepthOrArraySize).max(1);
    let mut planes = Vec::new();
    for &f in formats {
        let desc = D3D12_RESOURCE_DESC {
            Format: DXGI_FORMAT(f as i32),
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Alignment: 0,
            ..dd
        };
        let texture = managed_common(host, &desc).inspect_err(|e| {
            tracing::warn!(error = %e, ?desc, "no writable twin for a depth texture");
        }).ok()?;
        crate::host::name(&texture, || format!("depth twin {}x{} f{f}", dd.Width, dd.Height));
        planes.push(Arc::new(HostTexture::new(texture, 0, Vec::new())));
    }
    // The depth texture's footprints, every plane, and the twin's.
    let n = subs * planes.len() as u32;
    let mut df = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n as usize];
    let mut drows = vec![0u32; n as usize];
    // SAFETY: outputs sized for `n` subresources.
    unsafe { host.device.GetCopyableFootprints(&dd, 0, n, 0, Some(df.as_mut_ptr()), Some(drows.as_mut_ptr()), None, None) };
    let mut routes = Vec::new();
    let mut at = 0u64;
    let mut place = |mut f: D3D12_PLACED_SUBRESOURCE_FOOTPRINT, rows: u32| {
        f.Offset = at;
        let bytes = u64::from(f.Footprint.RowPitch) * u64::from(rows) * u64::from(f.Footprint.Depth);
        at = (at + bytes).next_multiple_of(512);
        f
    };
    for (p, plane) in planes.iter().enumerate() {
        // SAFETY: a live resource.
        let td = unsafe { plane.texture.GetDesc() };
        let mut tf = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); subs as usize];
        let mut trows = vec![0u32; subs as usize];
        // SAFETY: outputs sized for `subs` subresources.
        unsafe { host.device.GetCopyableFootprints(&td, 0, subs, 0, Some(tf.as_mut_ptr()), Some(trows.as_mut_ptr()), None, None) };
        for s in 0..subs {
            let d = (p as u32 * subs + s) as usize;
            let same = df[d].Footprint.Format == td.Format;
            // The same bytes: both sides copy through the depth texture's
            // footprint; else each its own, converted between.
            let route = if same {
                Route { plane: p, twin_sub: s, depth_sub: d as u32, twin: place(df[d], drows[d]), depth: None }
            } else {
                let twin = place(tf[s as usize], trows[s as usize]);
                Route { plane: p, twin_sub: s, depth_sub: d as u32, twin, depth: Some(place(df[d], drows[d])) }
            };
            if route.depth.is_some() && (td.Format != DXGI_FORMAT_R16_TYPELESS || df[d].Footprint.Format != DXGI_FORMAT_R32_TYPELESS) {
                once(console, td.Format.0 as u32, df[d].Footprint.Format.0 as u32, "a depth twin plane with no conversion to the host's: not made");
                return None;
            }
            routes.push(route);
        }
    }
    let scratch = scratch_buffer(at)?;
    crate::host::name(&scratch, || format!("depth twin copies {at:#x}"));
    tracing::debug!(console, width = dd.Width, height = dd.Height, planes = planes.len(), "a writable twin for a depth texture");
    Some(Twin { planes, newer: AtomicU8::new(DEPTH), routes, scratch })
}

/// A default-heap buffer of `size` bytes shaders may write, in the common
/// state.
fn scratch_buffer(size: u64) -> Option<ID3D12Resource> {
    let host = host()?;
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(256),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    managed_common(host, &desc).ok()
}

fn managed_common(host: &crate::host::Host, desc: &D3D12_RESOURCE_DESC) -> windows::core::Result<ID3D12Resource> {
    // SAFETY: query one initialized description.
    let bytes = unsafe { host.device.GetResourceAllocationInfo(0, &[*desc]) }.SizeInBytes;
    host.managed_gpu(bytes, |heap| {
        let mut resource = None;
        // SAFETY: initialized descriptions and our out slot.
        unsafe {
            host.device.CreateCommittedResource(
                heap, D3D12_HEAP_FLAG_NONE, desc, D3D12_RESOURCE_STATE_COMMON, None, &mut resource,
            )?;
        }
        resource.ok_or_else(windows::core::Error::empty)
    })
}

/// Logs once per (console format, view format, plane) what happened.
fn once(console: u32, view: u32, plane: u32, what: &str) {
    static SEEN: std::sync::Mutex<Vec<(u32, u32, u32, String)>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let key = (console, view, plane, what.to_owned());
    if seen.len() < 64 && !seen.contains(&key) {
        tracing::info!(console, view, plane, "{what}");
        seen.push(key);
    }
}

/// An unordered access view at `at` of `resource`, a depth texture whose
/// host copy is `copy`, made on its twin's plane the title's view names
/// (`words`: the view description, or `None` for the resource's default
/// mip-0 plane-0 view), in a UAV-capable format of that plane's family (the
/// title's own if it is one, else the plane's integer or normalised/float
/// one, as the title's format is). The twin's plane texture, which the slot
/// then views.
pub(crate) fn uav(
    resource: This,
    copy: &Arc<HostTexture>,
    words: Option<&[u32; 10]>,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<ID3D12Resource> {
    let console = crate::resource::title_format(resource)?;
    let twin = copy.twin.get_or_init(|| make(console, &copy.texture)).as_ref()?;
    let mut words = match words {
        Some(&w) => w,
        // SAFETY: a live resource.
        None => default_uav_words(console, &unsafe { copy.texture.GetDesc() })?,
    };
    const TEXTURE_1D: u32 = 2;
    const TEXTURE_1D_ARRAY: u32 = 3;
    const TEXTURE_2D: u32 = 4;
    const TEXTURE_2D_ARRAY: u32 = 5;
    // The plane: last of the 2D view's fields (mip, plane; mip, first
    // slice, slices, plane); 1D views have only plane 0.
    let plane_word = match words[1] {
        TEXTURE_1D | TEXTURE_1D_ARRAY => None,
        TEXTURE_2D => Some(3),
        TEXTURE_2D_ARRAY => Some(5),
        other => {
            once(console, words[0], other, "an unordered view of a depth texture of this dimension: not made");
            return None;
        }
    };
    let plane = plane_word.map_or(0, |w| words[w] as usize);
    let Some(texture) = twin.planes.get(plane) else {
        once(console, words[0], plane as u32, "an unordered view of a depth plane the twin does not have: not made");
        return None;
    };
    // SAFETY: a live resource.
    let own = unsafe { texture.texture.GetDesc() }.Format.0 as u32;
    let requested = words[0];
    let format = uav_format(own, requested, console);
    // Like ordinary texture UAVs, a twin needs device support for the
    // resolved concrete format, not merely a match in its DXGI family.
    if !crate::descriptor::host::typed_store(format) {
        once(console, format, plane as u32, "an unordered view of a depth twin in a format the host cannot store");
        return None;
    }
    once(console, requested, plane as u32, "an unordered view of a depth texture, made on its writable twin");
    words[0] = format;
    if let Some(w) = plane_word {
        words[w] = 0;
    }
    // SAFETY: 40 bytes in the host's layout.
    let desc = unsafe { std::ptr::from_ref(&words).cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>().read_unaligned() };
    // SAFETY: a live texture and a slot of our heap.
    unsafe { host()?.device.CreateUnorderedAccessView(&texture.texture, None, Some(&desc), at) };
    Some(texture.texture.clone())
}

/// Default `D3D12_UNORDERED_ACCESS_VIEW_DESC` words when the caller passes a
/// null description: mip 0, plane 0, and every slice of an array texture.
fn default_uav_words(console: u32, desc: &D3D12_RESOURCE_DESC) -> Option<[u32; 10]> {
    let slices = u32::from(desc.DepthOrArraySize);
    let (dim, array) = match desc.Dimension {
        D3D12_RESOURCE_DIMENSION_TEXTURE1D if slices > 1 => (3, slices),
        D3D12_RESOURCE_DIMENSION_TEXTURE1D => (2, 0),
        D3D12_RESOURCE_DIMENSION_TEXTURE2D if slices > 1 => (5, slices),
        D3D12_RESOURCE_DIMENSION_TEXTURE2D => (4, 0),
        _ => {
            once(console, 0, desc.Dimension.0 as u32, "a default unordered view of a depth texture of this dimension: not made");
            return None;
        }
    };
    Some([console, dim, 0, 0, array, 0, 0, 0, 0, 0])
}

/// Picks a concrete UAV-capable host format in `own`'s family (`R16_TYPELESS`,
/// `R32_TYPELESS`, or `R8_TYPELESS`). Typeless (`39`, `53`, `60`) and depth
/// (`40` `D32_FLOAT`, `55` `D16_UNORM`) members share `own`'s family but are
/// rejected by D3D12 `CreateUnorderedAccessView` (removing the device), so
/// they fall back to the plane's colour/integer format alongside cross-family
/// and `DXGI_FORMAT_UNKNOWN` (`0`) views.
fn uav_format(own: u32, raw_view: u32, console: u32) -> u32 {
    let raw = if raw_view == 0 { console } else { raw_view };
    let view = crate::resource::pc_format(raw);
    if crate::resource::family(view) == own && !matches!(view, 39 | 40 | 53 | 55 | 60) {
        view
    } else {
        // R16_TYPELESS: unorm or uint; R32_TYPELESS: float or uint; R8: uint.
        match (own, integer(view)) {
            (53, false) => 56,
            (53, true) => 57,
            (39, false) => 41,
            (39, true) => 42,
            _ => 62,
        }
    }
}

/// Whether a format holds integers (`_UINT`, `_SINT`, or planar stencil).
fn integer(format: u32) -> bool {
    matches!(
        format,
        3 | 4 | 7 | 8 | 12 | 14 | 17 | 18 | 22 | 25 | 30 | 32 | 36 | 38 | 42 | 43 | 47 | 50 | 52 | 57 | 59 | 62 | 64
    )
}

/// Before a use in `list` of the depth texture `depth` (with `twin`) in
/// host state `wanted`: the side the use needs is brought up to date (a
/// copy recorded now, or by the list's prelude for its first use). For
/// unordered access the twin's planes are moved there and true returned:
/// the depth texture itself is not used.
pub(crate) fn route(list: &HostList, depth: &Arc<HostTexture>, twin: &Twin, wanted: u32) -> bool {
    let side = if wanted & UNORDERED_ACCESS != 0 { TWIN } else { DEPTH };
    let writes = side == TWIN || wanted & (DEPTH_WRITE | RENDER_TARGET | COPY_DEST | RESOLVE_DEST) != 0;
    if let Some(to_depth) = list.twin_use(depth, side, writes)
        && twin.copy(&list.list, depth, to_depth, &mut |t, w| list.move_texture(t, w))
    {
        list.need_rebind();
    }
    if side == DEPTH {
        return false;
    }
    let barriers: Vec<_> = twin.planes.iter().filter_map(|p| list.move_texture(p, UNORDERED_ACCESS)).collect();
    if !barriers.is_empty() {
        // SAFETY: an open list and live textures.
        unsafe { list.list.ResourceBarrier(&barriers) };
    }
    true
}

impl Use {
    pub(crate) fn new(depth: &Arc<HostTexture>, side: u8, writes: bool) -> Self {
        Use { first: side, last: writes.then_some(side), depth: depth.clone() }
    }

    /// A later use in the list needing `side`: whether a copy must be
    /// recorded first (true: into the depth texture).
    pub(crate) fn again(&mut self, side: u8, writes: bool) -> Option<bool> {
        again(self.first, &mut self.last, side, writes)
    }
}

/// [`Use::again`] on its first side and the side it left newer.
fn again(first: u8, last: &mut Option<u8>, side: u8, writes: bool) -> Option<bool> {
    // Unchanged since the list's start: only the first use's side is
    // known to be current there.
    let current = match *last {
        None => first == side,
        Some(newer) => newer == BOTH || newer == side,
    };
    let copy = (!current).then_some(side == DEPTH);
    if copy.is_some() {
        *last = Some(BOTH);
    }
    if writes {
        *last = Some(side);
    }
    copy
}

/// Where a list's uses leave a twin's newer side, from `newer` before it:
/// the copy its prelude makes first (true: into the depth texture), and the
/// newer side after the list.
fn submitted(first: u8, last: Option<u8>, newer: u8) -> (Option<bool>, u8) {
    let copy = match (first, newer) {
        (DEPTH, TWIN) => Some(true),
        (TWIN, DEPTH) => Some(false),
        _ => None,
    };
    let start = if copy.is_some() { BOTH } else { newer };
    (copy, last.unwrap_or(start))
}

/// At submission, for a list's twin uses in order: the copies its prelude
/// records first (the depth texture and whether into it), the twins'
/// newer sides moved on to where the list leaves them.
pub(crate) fn at_submission<'a>(uses: impl Iterator<Item = &'a Use>) -> Vec<(Arc<HostTexture>, bool)> {
    let mut copies = Vec::new();
    for u in uses {
        let Some(Some(twin)) = u.depth.twin.get() else { continue };
        let (to_depth, after) = submitted(u.first, u.last, twin.newer.load(Relaxed));
        twin.newer.store(after, Relaxed);
        if let Some(to_depth) = to_depth {
            copies.push((u.depth.clone(), to_depth));
        }
    }
    copies
}

/// Moves a texture to a state for a copy: the barrier, if any.
pub(crate) type Mover<'a> = dyn FnMut(&Arc<HostTexture>, u32) -> Option<D3D12_RESOURCE_BARRIER> + 'a;

impl Twin {
    /// Records into `l` the copy of the newer side into the other: from
    /// the twin into `depth` (`to_depth`) or back. `mv` moves textures to
    /// the copy states (tracked by the list, or from their real states in
    /// a prelude). True if it ran our conversion, which replaces the
    /// list's compute root signature and pipeline.
    pub(crate) fn copy(&self, l: &ID3D12GraphicsCommandList, depth: &Arc<HostTexture>, to_depth: bool, mv: &mut Mover) -> bool {
        let (src, dst) = if to_depth { (COPY_SOURCE, COPY_DEST) } else { (COPY_DEST, COPY_SOURCE) };
        let mut barriers: Vec<_> = self.planes.iter().filter_map(|p| mv(p, src)).collect();
        barriers.extend(mv(depth, dst));
        let at = |r: &ID3D12Resource, sub: u32| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the copy does not release.
            pResource: unsafe { std::mem::transmute_copy(r) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: sub },
        };
        let placed = |f: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&self.scratch) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: f },
        };
        let converts = self.routes.iter().any(|r| r.depth.is_some());
        let state = |from, to| transition(&self.scratch, D3D12_RESOURCE_STATES(from), D3D12_RESOURCE_STATES(to));
        let mut converted = false;
        // SAFETY: an open list; live resources in the states named (the
        // scratch buffer starts common: its first copy promotes it).
        unsafe {
            if !barriers.is_empty() {
                l.ResourceBarrier(&barriers);
            }
            for r in &self.routes {
                let plane = &self.planes[r.plane].texture;
                if to_depth {
                    l.CopyTextureRegion(&placed(r.twin), 0, 0, 0, &at(plane, r.twin_sub), None);
                } else {
                    l.CopyTextureRegion(&placed(r.depth.unwrap_or(r.twin)), 0, 0, 0, &at(&depth.texture, r.depth_sub), None);
                }
            }
            if converts && let Some(c) = converter() {
                l.ResourceBarrier(&[state(COPY_DEST as i32, UNORDERED_ACCESS as i32)]);
                l.SetComputeRootSignature(&c.root);
                l.SetPipelineState(&c.pso);
                l.SetComputeRootUnorderedAccessView(1, self.scratch.GetGPUVirtualAddress());
                for r in &self.routes {
                    let Some(d) = r.depth else { continue };
                    let (t, d) = (r.twin.Footprint, d);
                    let constants = [t.Width, t.Height, t.RowPitch, r.twin.Offset as u32, d.Footprint.RowPitch, d.Offset as u32, u32::from(to_depth), 0];
                    l.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
                    l.Dispatch(t.Width.div_ceil(2).div_ceil(64), t.Height, 1);
                }
                l.ResourceBarrier(&[uav_barrier(), state(UNORDERED_ACCESS as i32, COPY_SOURCE as i32)]);
                converted = true;
            } else {
                l.ResourceBarrier(&[state(COPY_DEST as i32, COPY_SOURCE as i32)]);
            }
            for r in &self.routes {
                let plane = &self.planes[r.plane].texture;
                if to_depth {
                    l.CopyTextureRegion(&at(&depth.texture, r.depth_sub), 0, 0, 0, &placed(r.depth.unwrap_or(r.twin)), None);
                } else {
                    l.CopyTextureRegion(&at(plane, r.twin_sub), 0, 0, 0, &placed(r.twin), None);
                }
            }
            l.ResourceBarrier(&[state(COPY_SOURCE as i32, 0)]);
        }
        COPIES.fetch_add(1, Relaxed);
        converted
    }
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
        },
    }
}

/// 16-bit depth (unorm, two per dword) to the host's 24-bit depth plane (in
/// the low 24 bits of 32) and back, in the scratch buffer: `u * (2^24 - 1)
/// / 65535` rounded, exactly, and back to the 16-bit value whose 24-bit one
/// is nearest.
const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=8, b0), UAV(u0)"
cbuffer C : register(b0) { uint width; uint height; uint pitch16; uint at16; uint pitch32; uint at32; uint toDepth; uint unused; };
RWByteAddressBuffer buf : register(u0);
uint d24(uint u) { return u * 256 + (u * 255 + 32767) / 65535; }
uint u16(uint d) {
    d &= 0xffffff;
    uint u = min(d >> 8, 65535);
    uint best = u;
    int err = abs((int)d24(u) - (int)d);
    if (u > 0 && abs((int)d24(u - 1) - (int)d) < err) { best = u - 1; err = abs((int)d24(u - 1) - (int)d); }
    if (u < 65535 && abs((int)d24(u + 1) - (int)d) < err) { best = u + 1; }
    return best;
}
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint x = id.x * 2;
    uint y = id.y;
    if (x >= width || y >= height) return;
    uint a = at16 + y * pitch16 + x * 2;
    uint b = at32 + y * pitch32 + x * 4;
    bool two = x + 1 < width;
    if (toDepth != 0) {
        uint pair = buf.Load(a);
        buf.Store(b, d24(pair & 0xffff));
        if (two) buf.Store(b + 4, d24(pair >> 16));
    } else {
        uint lo = u16(buf.Load(b));
        uint hi = two ? u16(buf.Load(b + 4)) : 0;
        buf.Store(a, lo | hi << 16);
    }
}
"#;

struct Converter {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Converter {}
// SAFETY: as above.
unsafe impl Sync for Converter {}

fn converter() -> Option<&'static Converter> {
    static MADE: std::sync::OnceLock<Option<Converter>> = std::sync::OnceLock::new();
    MADE.get_or_init(|| match compile() {
        Ok(c) => Some(c),
        Err(e) => {
            tracing::warn!(error = %e, "no depth conversion: 16-bit depth written through unordered views does not reach the depth texture");
            None
        }
    })
    .as_ref()
}

fn compile() -> windows::core::Result<Converter> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(SOURCE.as_ptr().cast(), SOURCE.len(), s!("depth"), None, None, s!("main"), s!("cs_5_1"), 0, 0, &mut code, Some(&mut errors))
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe { std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize()) };
            tracing::warn!("depth conversion: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() },
            ..Default::default()
        })?;
        Ok(Converter { root, pso })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The material-depth frame of Grand Theft Auto VI: written through
    /// unordered views, read as a shader resource in the same list, written
    /// again; read in the next list; tested as depth in a later one; the
    /// next frame writes it through unordered views again.
    #[test]
    fn copies_follow_the_side_each_use_needs() {
        // List 1: unordered write first (the prelude copies the depth side
        // over, newer since the last frame), a read (copy back now), a
        // write again.
        let mut last = Some(TWIN);
        assert_eq!(again(TWIN, &mut last, DEPTH, false), Some(true));
        assert_eq!(last, Some(BOTH));
        assert_eq!(again(TWIN, &mut last, TWIN, true), None);
        assert_eq!(submitted(TWIN, last, DEPTH), (Some(false), TWIN));
        // List 2 only reads: its prelude copies into the depth texture.
        assert_eq!(submitted(DEPTH, None, TWIN), (Some(true), BOTH));
        // List 3 renders to it: nothing to copy, the depth side is newer.
        assert_eq!(submitted(DEPTH, Some(DEPTH), BOTH), (None, DEPTH));
        // A list that starts reading the depth side, unchanged since, does
        // not know the twin is current: a use of the twin copies.
        let mut last = None;
        assert_eq!(again(DEPTH, &mut last, DEPTH, false), None);
        assert_eq!(again(DEPTH, &mut last, TWIN, true), Some(false));
        assert_eq!(last, Some(TWIN));
    }

    #[test]
    fn uav_formats_and_default_words_stay_uav_capable() {
        // Depth and typeless members of the twin's own family map to concrete
        // UAV-capable formats instead of reaching CreateUnorderedAccessView.
        assert_eq!(uav_format(39, 40, 40), 41); // D32_FLOAT -> R32_FLOAT
        assert_eq!(uav_format(39, 39, 39), 41); // R32_TYPELESS -> R32_FLOAT
        assert_eq!(uav_format(39, 0, 40), 41);  // UNKNOWN on D32_FLOAT -> R32_FLOAT
        assert_eq!(uav_format(39, 42, 40), 42); // R32_UINT preserved
        assert_eq!(uav_format(39, 43, 40), 43); // R32_SINT preserved
        assert_eq!(uav_format(53, 55, 55), 56); // D16_UNORM -> R16_UNORM
        assert_eq!(uav_format(53, 53, 53), 56); // R16_TYPELESS -> R16_UNORM
        assert_eq!(uav_format(53, 0, 118), 56); // UNKNOWN on console D16S8 -> R16_UNORM
        assert_eq!(uav_format(53, 54, 55), 54); // R16_FLOAT preserved
        assert_eq!(uav_format(53, 57, 55), 57); // R16_UINT preserved
        assert_eq!(uav_format(60, 60, 20), 62); // R8_TYPELESS -> R8_UINT
        assert_eq!(uav_format(60, 22, 20), 62); // X32_TYPELESS_G8X24_UINT -> R8_UINT
        assert_eq!(uav_format(60, 120, 118), 62); // console X16_TYPELESS_G8_UINT -> R8_UINT
        assert_eq!(uav_format(60, 61, 20), 61); // R8_UNORM preserved

        let d = |dimension, depth_or_array_size| D3D12_RESOURCE_DESC {
            Dimension: dimension,
            DepthOrArraySize: depth_or_array_size,
            ..Default::default()
        };
        assert_eq!(
            default_uav_words(55, &d(D3D12_RESOURCE_DIMENSION_TEXTURE1D, 1)),
            Some([55, 2, 0, 0, 0, 0, 0, 0, 0, 0])
        );
        assert_eq!(
            default_uav_words(55, &d(D3D12_RESOURCE_DIMENSION_TEXTURE1D, 4)),
            Some([55, 3, 0, 0, 4, 0, 0, 0, 0, 0])
        );
        assert_eq!(
            default_uav_words(40, &d(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 1)),
            Some([40, 4, 0, 0, 0, 0, 0, 0, 0, 0])
        );
        assert_eq!(
            default_uav_words(40, &d(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 3)),
            Some([40, 5, 0, 0, 3, 0, 0, 0, 0, 0])
        );
        assert_eq!(
            default_uav_words(40, &d(D3D12_RESOURCE_DIMENSION_TEXTURE3D, 2)),
            None
        );
    }

    #[test]
    fn depth_typed_and_default_uavs_write_through_the_twin() {
        let Some(host) = host() else { return };
        let iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        #[repr(C, align(8))]
        struct Aligned([u32; 14]);
        let make_depth = |dim: u32, w: u32, h: u32, slices: u16, format: u32| -> This {
            let layout = std::alloc::Layout::from_size_align(1 << 20, 65536).unwrap();
            // SAFETY: non-zero aligned allocation; leaked as title memory in tests.
            let address = unsafe { std::alloc::alloc_zeroed(layout) as u64 };
            let words = Aligned([
                dim,
                0,
                0,
                0,
                w,
                0,
                h,
                u32::from(slices) | 1 << 16,
                format,
                1,
                0,
                0,
                2,
                0,
            ]);
            let mut out = std::ptr::null();
            let hr = crate::resource::create_at_address(
                std::ptr::null(),
                address,
                words.0.as_ptr().cast(),
                0,
                std::ptr::null(),
                &iid,
                &mut out,
            );
            assert_eq!(hr, crate::object::S_OK);
            out
        };

        let d32 = make_depth(3, 8, 8, 1, 40);
        let d16_arr = make_depth(3, 8, 8, 2, 55);
        let d16_1d = make_depth(2, 8, 1, 1, 55);
        let copy32 = crate::resource::host_copy(d32).unwrap();
        let copy16 = crate::resource::host_copy(d16_arr).unwrap();
        let copy1d = crate::resource::host_copy(d16_1d).unwrap();

        // SAFETY: plain shader-visible heap creation and handle queries.
        let (heap, step) = unsafe {
            let heap: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 5,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .unwrap();
            let step = host
                .device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            (heap, step)
        };
        let slot = |i: u32| {
            // SAFETY: a live heap.
            let mut h = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
            h.ptr += (i * step) as usize;
            h
        };

        // Explicit D32_FLOAT (40), R32_TYPELESS (39), DXGI_FORMAT_UNKNOWN (0),
        // a null-pDesc view of a 2-slice D16_UNORM array, and a 1D D16_UNORM view.
        assert!(uav(d32, &copy32, Some(&[40, 4, 0, 0, 0, 0, 0, 0, 0, 0]), slot(0)).is_some());
        assert!(uav(d16_arr, &copy16, None, slot(1)).is_some());
        assert!(uav(d32, &copy32, Some(&[39, 4, 0, 0, 0, 0, 0, 0, 0, 0]), slot(2)).is_some());
        assert!(uav(d32, &copy32, Some(&[0, 4, 0, 0, 0, 0, 0, 0, 0, 0]), slot(3)).is_some());
        assert!(uav(d16_1d, &copy1d, Some(&[55, 2, 0, 0, 0, 0, 0, 0, 0, 0]), slot(4)).is_some());
        // SAFETY: plain device health query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());

        let cs = r#"
#define RS "DescriptorTable(UAV(u0, numDescriptors=2))"
RWTexture2D<float> out32 : register(u0);
RWTexture2DArray<float> out16 : register(u1);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    out32[id.xy] = 0.25 + 0.01 * id.x;
    out16[uint3(id.xy, 0)] = 0.25;
    out16[uint3(id.xy, 1)] = 0.75;
}
"#;
        let (mut code, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
        // SAFETY: source and entry names outlive D3DCompile.
        unsafe {
            D3DCompile(
                cs.as_ptr().cast(),
                cs.len(),
                s!("twin_uav_test"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                Some(&mut errors),
            )
            .unwrap();
        }
        let code = code.unwrap();
        // SAFETY: compiled shader blob holds a serialized root signature.
        let (root, pso) = unsafe {
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            (root, pso)
        };

        let list = host.new_list().unwrap();
        assert!(route(&list, &copy32, copy32.twin.get().unwrap().as_ref().unwrap(), UNORDERED_ACCESS));
        assert!(route(&list, &copy16, copy16.twin.get().unwrap().as_ref().unwrap(), UNORDERED_ACCESS));
        // SAFETY: open list, live heap, root signature, and pipeline.
        unsafe {
            list.list.SetDescriptorHeaps(&[Some(heap.clone())]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pso);
            list.list
                .SetComputeRootDescriptorTable(0, heap.GetGPUDescriptorHandleForHeapStart());
            list.list.Dispatch(1, 1, 1);
        }
        // Transition both depth textures to COPY_SOURCE: `route` copies each
        // twin's unordered writes into its depth texture first.
        crate::list::ensure_state(&list, d32, COPY_SOURCE);
        crate::list::ensure_state(&list, d16_arr, COPY_SOURCE);

        let mut fp32 = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut fp16 = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 2];
        let (mut end32, mut span16) = (0u64, 0u64);
        // SAFETY: live resources and sized footprint outputs.
        let base16 = unsafe {
            let d = copy32.texture.GetDesc();
            host.device
                .GetCopyableFootprints(&d, 0, 1, 0, Some(&mut fp32), None, None, Some(&mut end32));
            let base16 = end32.next_multiple_of(512);
            let d = copy16.texture.GetDesc();
            host.device.GetCopyableFootprints(
                &d,
                0,
                2,
                base16,
                Some(fp16.as_mut_ptr()),
                None,
                None,
                Some(&mut span16),
            );
            base16
        };
        let (readback, ptr) = host.shared_buffer(base16 + span16).unwrap();
        let sub = |r: &ID3D12Resource, i: u32| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed pointer for CopyTextureRegion.
            pResource: unsafe { std::mem::transmute_copy(r) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i },
        };
        let placed = |fp: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed pointer for CopyTextureRegion.
            pResource: unsafe { std::mem::transmute_copy(&readback) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp },
        };
        // SAFETY: open list; depth textures in COPY_SOURCE, readback buffer writable.
        unsafe {
            list.list
                .CopyTextureRegion(&placed(fp32), 0, 0, 0, &sub(&copy32.texture, 0), None);
            list.list
                .CopyTextureRegion(&placed(fp16[0]), 0, 0, 0, &sub(&copy16.texture, 0), None);
            list.list
                .CopyTextureRegion(&placed(fp16[1]), 0, 0, 0, &sub(&copy16.texture, 1), None);
        }
        list.close();
        host.execute(&[&list]);

        // SAFETY: GPU work waited on in `execute`; offsets lie inside `readback`.
        unsafe {
            for x in [0usize, 3, 7] {
                let got = (ptr.add(fp32.Offset as usize + x * 4) as *const f32).read();
                let want = 0.25 + 0.01 * x as f32;
                assert!((got - want).abs() < 1e-6, "d32[{x}] = {got}, want {want}");
            }
            let s0 = (ptr.add(fp16[0].Offset as usize) as *const u16).read();
            let s1 = (ptr.add(fp16[1].Offset as usize) as *const u16).read();
            assert_eq!(s0, 16384);
            assert_eq!(s1, 49151);
            assert!(host.device.GetDeviceRemovedReason().is_ok());
        }
    }
}
