//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory changed since the last detiling
//! ([`crate::watch`], which follows writes through every mapping of the
//! same physical pages) in the runtime's memory; elsewhere, a content hash
//! of the whole memory at every submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Textures read through descriptor tables are brought up to date, and for
//! shaders that index the heap themselves every texture with a resource
//! object the heap views (`crate::bindless`); textures the title placed in
//! memory itself are not made for the heap (see `descriptor::host`). Plane
//! 0 is uploaded (every mip and array slice), depth and stencil planes are
//! not.

use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;
use windows::core::Interface;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    sparse_host: std::sync::Weak<crate::resource::HostTexture>,
    /// Bytes needed during a copy.
    size: u64,
    #[cfg(feature = "diagnostics")]
    pub(crate) upload_payload: u64,
    stable: Mutex<std::sync::Weak<Upload>>,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    memory: *const u8,
    /// The memory's pages changed since detiled (none: not the runtime's
    /// memory, hashed instead).
    tracked: Option<crate::watch::Tracked>,
    last: Mutex<Last>,
}

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not tracked).
#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
    sparse_first: Option<u32>,
    retry: bool,
}

pub(crate) struct Upload {
    buffer: ID3D12Resource,
    data: usize,
    ready: std::sync::atomic::AtomicBool,
    #[cfg(feature = "diagnostics")]
    _upload_bytes: UploadBytes<'static>,
}

/// Shared scratch buffer for reading tracked title-texture memory before
/// detiling, so each `Staging` does not retain a full-size `Vec<u8>`.
static READ_SCRATCH: Mutex<Vec<u8>> = Mutex::new(Vec::new());

pub(crate) fn trim_idle_scratch() {
    let mut scratch = READ_SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
    if scratch.capacity() > crate::working_set::limit(crate::working_set::Cache::TextureRead) {
        *scratch = Vec::new();
    }
}

#[cfg(feature = "diagnostics")]
static LIVE_UPLOAD_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Keep the accounting with the staging object, including any references held
/// by command lists. No global list of staging objects is needed.
#[cfg(feature = "diagnostics")]
struct UploadBytes<'a> {
    total: &'a std::sync::atomic::AtomicU64,
    size: u64,
}

#[cfg(feature = "diagnostics")]
impl<'a> UploadBytes<'a> {
    fn new(total: &'a std::sync::atomic::AtomicU64, size: u64) -> Self {
        total.fetch_add(size, std::sync::atomic::Ordering::Relaxed);
        Self { total, size }
    }
}

#[cfg(feature = "diagnostics")]
impl Drop for UploadBytes<'_> {
    fn drop(&mut self) {
        self.total.fetch_sub(self.size, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Logical upload-buffer bytes and shared CPU scratch capacity, not OS commit.
#[cfg(feature = "diagnostics")]
pub(crate) fn retained_copy_bytes() -> (u64, Option<usize>) {
    let upload = LIVE_UPLOAD_BYTES.load(std::sync::atomic::Ordering::Relaxed);
    let scratch = READ_SCRATCH.try_lock().ok().map(|s| s.capacity());
    (upload, scratch)
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

/// A host copy or resolve writes `resource`: its host texture's contents
/// are the host's, not title memory's, which stays as it was.
pub(crate) fn host_written(resource: This) {
    if let Some(copy) = crate::resource::host_copy(resource) {
        copy.host_written.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

fn is_host_written(resource: This) -> bool {
    crate::resource::host_copy(resource)
        .is_some_and(|c| c.host_written.load(std::sync::atomic::Ordering::Relaxed))
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Copies [`alias_copy`] recorded since the last frame-rate line.
pub(crate) static ALIAS_COPIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Records, once per list, the copy of `from`'s host texture into `to`'s:
/// two title textures at one memory that cannot share a host texture
/// (`resource::alias_source`: block-compressed against plain formats of
/// the same bytes, or a cast the host refused), `to` (a placed view's)
/// reading what the GPU wrote through `from`. Through a buffer, each
/// subresource with its own format, as many as have the same bytes per
/// row and rows in both; `to` no longer takes its contents from title
/// memory.
pub(crate) fn alias_copy(list: &HostList, from: This, to: This) {
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    let (Some(a), Some(b), Some(host)) = (
        crate::resource::host_resource(from),
        crate::resource::host_resource(to),
        host(),
    ) else {
        return;
    };
    if !crate::resource::copy_formats_match(to, from) {
        return;
    }
    // An odd key: textures' own first uses are keyed by the host texture.
    if !list.first_use(to as usize | 1) {
        return;
    }
    // SAFETY: live resources.
    let (da, mut db) = unsafe { (a.GetDesc(), b.GetDesc()) };
    let destination = crate::resource::host_copy(to);
    let sparse = destination.as_ref().and_then(|h| h.sparse.as_ref());
    if sparse.is_some() {
        db.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    }
    let count = |d: &D3D12_RESOURCE_DESC| {
        if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
            u32::from(d.MipLevels)
        } else {
            u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize).max(1)
        }
    };
    let footprints = |d: &D3D12_RESOURCE_DESC, n: u32| {
        let mut f = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n as usize];
        let mut rows = vec![0u32; n as usize];
        let mut total = 0;
        // SAFETY: outputs sized for `n` subresources.
        unsafe {
            host.device
                .GetCopyableFootprints(d, 0, n, 0, Some(f.as_mut_ptr()), Some(rows.as_mut_ptr()), None, Some(&mut total));
        }
        (f, rows, total)
    };
    let (fa, ra, total) = footprints(&da, count(&da));
    let (fb, rb, _) = footprints(&db, count(&db));
    // Subresources in order while both have the same bytes there (mips of
    // a block-compressed texture past its last whole block do not).
    let n = (0..fa.len().min(fb.len()))
        .take_while(|&i| {
            let (x, y) = (fa[i], fb[i]);
            x.Offset == y.Offset && x.Footprint.RowPitch == y.Footprint.RowPitch && ra[i] == rb[i] && x.Footprint.Depth == y.Footprint.Depth
        })
        .count() as u32;
    if n == 0 {
        once!(?da, ?db, "an aliased texture not copied: its layouts differ on the host");
        return;
    }
    let Some(buffer) = host.default_buffer(total) else { return };
    let first = sparse.map_or(0, |s| {
        // GPU-produced aliases cannot be detected by refresh() at all.
        // Map their detailed mips before the alias copy runs, in the same
        // direct queue, or keep serving only the already-resident suffix.
        s.promote(b, 0);
        s.first()
    });
    if first >= n {
        return;
    }
    list.keep(&buffer);
    crate::texture::host_written(to);
    crate::list::ensure_state(list, from, COPY_SOURCE);
    crate::list::ensure_state(list, to, COPY_DEST);
    let place = |r: &ID3D12Resource, f: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the copy does not release.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: f },
    };
    let sub = |r: &ID3D12Resource, i: u32| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i },
    };
    let l = &list.list;
    // SAFETY: an open list; live resources in the states named.
    unsafe {
        for i in first..n {
            l.CopyTextureRegion(&place(&buffer, fa[i as usize]), 0, 0, 0, &sub(a, i), None);
        }
        l.ResourceBarrier(&[crate::host::transition(
            &buffer,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        for i in first..n {
            l.CopyTextureRegion(&sub(b, i), 0, 0, 0, &place(&buffer, fb[i as usize]), None);
        }
    }
    ALIAS_COPIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    static LOGGED: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
    let mut logged = LOGGED.lock().unwrap_or_else(|e| e.into_inner());
    if logged.len() < 64 && !logged.contains(&(from as usize, to as usize)) {
        logged.push((from as usize, to as usize));
        tracing::info!(from = ?from, to = ?to, a = da.Format.0, b = db.Format.0, w = da.Width, h = da.Height, subresources = n, "aliased texture copied");
    }
}

/// Records, once per list, the copy of `resource`'s staging buffer into
/// its host copy, and the detiling that refreshes the staging buffer at
/// submission if the memory changed. Leaves the texture in the copy
/// destination state.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    // Once per list for each host texture, however many title textures of
    // its memory the list samples.
    if is_host_written(resource) || !list.first_use(texture.as_raw() as usize) {
        return;
    }
    let Some(staging) = staging(resource, texture) else {
        return;
    };
    let Some(upload) = staging.for_recording() else { return };
    record(list, resource, texture, &staging, &upload);
    // The recorded copy can be replayed: pin its address until list reset.
    list.defer_texture(staging, upload);
}

/// For textures shaders reach through the heap itself, at submission
/// (`crate::bindless::uploads`): brings the staging buffer up to date and
/// records its copy into `list` only if the memory changed since (or was
/// never uploaded). True if it recorded; the texture is then in the copy
/// destination state.
pub(crate) fn upload_changed(list: &HostList, resource: This, texture: &ID3D12Resource) -> bool {
    if is_host_written(resource) {
        return false;
    }
    let Some(staging) = staging(resource, texture) else {
        return false;
    };
    let stable = staging.stable.lock().unwrap_or_else(|e| e.into_inner()).upgrade();
    let Some(upload) = staging.refresh(stable.as_ref()) else { return false };
    record(list, resource, texture, &staging, &upload);
    list.keep_upload_until_fence(upload);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::TextureQueuedBindless, staging.upload_payload);
    true
}

/// Records the copy of every subresource of `staging` into `texture`,
/// moving it to the copy destination state first.
fn record(list: &HostList, resource: This, texture: &ID3D12Resource, staging: &Staging, upload: &Upload) {
    const COPY_DEST: u32 = 0x400;
    crate::list::ensure_state(list, resource, COPY_DEST);
    for (i, footprint) in staging.footprints.iter().enumerate() {
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (upload.buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = u64::from(f.RowPitch) * u64::from(staging.rows[i]) * u64::from(f.Depth);
            list.note_read(
                || {
                    format!(
                        "texture upload {resource:?} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.size,
                        footprint.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + footprint.Offset, len),
            );
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&upload.buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
}

impl Staging {
    fn allocate(&self) -> Option<Arc<Upload>> {
        let (buffer, data) = host()?.upload_buffer(self.size)?;
        #[cfg(feature = "diagnostics")]
        crate::host::name(&buffer, || format!("texture upload {:#x} bytes", self.size));
        Some(Arc::new(Upload {
            buffer, data: data as usize, ready: false.into(),
            #[cfg(feature = "diagnostics")]
            _upload_bytes: UploadBytes::new(&LIVE_UPLOAD_BYTES, self.size),
        }))
    }

    fn for_recording(&self) -> Option<Arc<Upload>> {
        let mut stable = self.stable.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(upload) = stable.upgrade() {
            return Some(upload);
        }
        let upload = self.allocate()?;
        *stable = Arc::downgrade(&upload);
        Some(upload)
    }

    /// Refresh a recorded list's backing in place, or make a transient
    /// copy on demand. With no dirty bytes, no transient heap is created.
    pub(crate) fn refresh(&self, existing: Option<&Arc<Upload>>) -> Option<Arc<Upload>> {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let (read, changed) = self.refresh_now(existing);
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
        changed
    }

    fn refresh_now(&self, existing: Option<&Arc<Upload>>) -> (usize, Option<Arc<Upload>>) {
        use std::sync::atomic::Ordering::Relaxed;
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let sparse = self.sparse_host.upgrade();
        let first = sparse.as_ref().and_then(|h| h.sparse.as_ref()).map(|s| s.first());
        let must_fill = last.retry || first != last.sparse_first
            || existing.is_some_and(|upload| !upload.ready.load(Relaxed));
        let prior_hash = last.hash;
        let mut fill = |tiled: &[u8], hash: Option<blake3::Hash>| {
            let upload = existing.cloned().or_else(|| self.allocate());
            let upload = upload.filter(|upload| self.detile(tiled, upload));
            last.retry = upload.is_none();
            if let Some(upload) = &upload {
                upload.ready.store(true, Relaxed);
                last.hash = hash;
                last.sparse_first = sparse.as_ref().and_then(|h| h.sparse.as_ref()).map(|s| s.first());
            }
            upload
        };
        if let Some(t) = &self.tracked {
            if t.take(start, start + size).is_empty() && !must_fill {
                return (0, None);
            }
            let mut scratch = READ_SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
            scratch.resize(size, 0);
            crate::host::copy_mapped(scratch.as_mut_ptr(), self.memory, size, false);
            let upload = fill(&scratch, None);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
            return (size, upload);
        }
        if first.is_some() {
            let mut scratch = READ_SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
            scratch.resize(size, 0);
            crate::host::copy_mapped(scratch.as_mut_ptr(), self.memory, size, false);
            let hash = blake3::hash(&scratch);
            return (size, if must_fill || prior_hash != Some(hash) { fill(&scratch, Some(hash)) } else { None });
        }
        if !committed(self.memory, size) {
            return (0, None);
        }
        // SAFETY: committed memory of the texture's layout size.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        let hash = blake3::hash(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if !must_fill && prior_hash == Some(hash) {
            return (size, None);
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        (size, fill(tiled, Some(hash)))
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer.
    fn detile(&self, tiled: &[u8], upload: &Upload) -> bool {
        if let Some(h) = self.sparse_host.upgrade()
            && let Some(s) = &h.sparse
        {
            let mut end = self.layout.slice_size as usize;
            for (i, mip) in self.layout.mips.iter().take(s.first() as usize).enumerate() {
                let at = mip.offset as usize;
                if tiled.get(at..end).is_none_or(|span| span.iter().any(|&b| b != 0)) {
                    // A recorded list will copy every level. Map the data
                    // before the list executes even if its view hides it.
                    s.promote(&h.texture, i as u32);
                    break;
                }
                end = at;
            }
        }
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = (i as u32 % self.mips, i as u32 / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = subresource_len(footprint.Offset, pitch, rows, f.Depth, self.size);
            // SAFETY: `subresource_len` clamps `len` to `self.size - footprint.Offset`,
            // keeping the slice strictly inside the mapped staging buffer even when
            // GetCopyableFootprints does not pad the final row of the last subresource.
            let linear = unsafe {
                std::slice::from_raw_parts_mut((upload.data as *mut u8).add(footprint.Offset as usize), len)
            };
            if let Err(e) =
                self.layout
                    .detile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
            {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return false;
            }
        }
        true
    }
}

/// Bytes available in the staging buffer for a subresource at `offset`.
/// D3D12's `GetCopyableFootprints` omits trailing row-pitch padding on the
/// final row of the final subresource, so `pitch * rows * depth` can exceed
/// `total - offset`.
fn subresource_len(offset: u64, pitch: usize, rows: u32, depth: u32, total: u64) -> usize {
    let padded = pitch
        .saturating_mul(rows as usize)
        .saturating_mul(depth as usize);
    let remaining = usize::try_from(total.saturating_sub(offset)).unwrap_or(usize::MAX);
    padded.min(remaining)
}

/// Whether `[at, at + len)` is committed, readable memory (streamed textures
/// may be unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    let mut p = at as usize;
    let Some(end) = p.checked_add(len) else {
        return false;
    };
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        let unreadable = info.Protect == 0
            || info.Protect & (PAGE_NOACCESS | PAGE_GUARD) != 0;
        if n == 0 || info.State != MEM_COMMIT || unreadable {
            return false;
        }
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= p {
            return false;
        }
        p = next;
    }
    true
}

/// The staging state of a title texture's host texture, made on first use
/// (one for all title textures of its memory).
fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    let copy = crate::resource::host_copy(resource)?;
    let mut held = copy.staging.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = held.as_ref() {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let mut d = unsafe { texture.GetDesc() };
    if copy.sparse.is_some() {
        // A staging footprint describes the linear copy, not the reserved
        // destination's hardware tile swizzle.
        d.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    }
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    #[cfg(feature = "diagnostics")]
    let mut row_sizes = vec![0u64; count as usize];
    #[cfg(feature = "diagnostics")]
    let row_sizes_out = Some(row_sizes.as_mut_ptr());
    #[cfg(not(feature = "diagnostics"))]
    let row_sizes_out = None;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            row_sizes_out,
            Some(&mut total),
        );
    }
    #[cfg(feature = "diagnostics")]
    let upload_payload = logical_upload_bytes(&footprints, &rows, &row_sizes);
    let layout_size = layout.size as usize;
    let s = Arc::new(Staging {
        sparse_host: if copy.sparse.is_some() { Arc::downgrade(&copy) } else { std::sync::Weak::new() },
        size: total,
        stable: Mutex::default(),
        #[cfg(feature = "diagnostics")]
        upload_payload,
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        memory,
        tracked: crate::watch::track(memory as usize, layout_size, true),
        last: Mutex::default(),
    });
    *held = Some(s.clone());
    Some(s)
}

/// Logical texture-copy payload, without staging pitch or alignment padding.
#[cfg(feature = "diagnostics")]
fn logical_upload_bytes(
    footprints: &[D3D12_PLACED_SUBRESOURCE_FOOTPRINT],
    rows: &[u32],
    row_sizes: &[u64],
) -> u64 {
    footprints.iter().zip(rows).zip(row_sizes).fold(0u64, |n, ((f, &rows), &row)| {
        n.saturating_add(row.saturating_mul(u64::from(rows)).saturating_mul(u64::from(f.Footprint.Depth)))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "diagnostics")]
    #[test]
    fn logical_upload_does_not_count_pitch_padding() {
        let footprint = |depth| D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT { RowPitch: 256, Depth: depth, ..Default::default() },
            ..Default::default()
        };
        assert_eq!(logical_upload_bytes(&[footprint(1), footprint(2)], &[6, 3], &[40, 20]), 360);
    }

    #[cfg(feature = "diagnostics")]
    #[test]
    fn upload_bytes_follow_the_allocation_not_its_arc_references() {
        use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

        let total = AtomicU64::new(0);
        let upload = Arc::new(UploadBytes::new(&total, 73));
        let in_flight = upload.clone();
        let other = UploadBytes::new(&total, 51);
        assert_eq!(total.load(Relaxed), 124);
        drop(upload);
        assert_eq!(total.load(Relaxed), 124);
        drop(other);
        assert_eq!(total.load(Relaxed), 73);
        drop(in_flight);
        assert_eq!(total.load(Relaxed), 0);
    }

    #[test]
    fn subresource_len_stays_inside_staging_total() {
        // A 4x4 RGBA8 texture has RowPitch=256, rows=4, last_row=16 -> total=784.
        // Padded size (1024) must be clamped to 784 so from_raw_parts_mut stays in bounds.
        assert_eq!(subresource_len(0, 256, 4, 1, 784), 784);
        assert_eq!(subresource_len(512, 256, 4, 1, 784), 272);
        assert_eq!(subresource_len(0, 256, 4, 1, 2048), 1024);
    }

    #[test]
    fn committed_rejects_overflow_and_noaccess_pages() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        assert!(!committed(usize::MAX as *const u8, 16));
        // SAFETY: reserving and committing one private page for the test, freed below.
        let noaccess =
            unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS) };
        assert!(!noaccess.is_null());
        assert!(!committed(noaccess.cast::<u8>(), 4096));
        // SAFETY: freeing the allocation above.
        unsafe { VirtualFree(noaccess, 0, MEM_RELEASE) };

        // Negative control: a readable committed page still passes.
        // SAFETY: reserving and committing one private page for the test, freed below.
        let rw =
            unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        assert!(!rw.is_null());
        assert!(committed(rw.cast::<u8>(), 4096));
        // SAFETY: freeing the allocation above.
        unsafe { VirtualFree(rw, 0, MEM_RELEASE) };
    }
}
