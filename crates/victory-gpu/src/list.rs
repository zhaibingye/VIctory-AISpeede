//! Command allocators and command lists.
//!
//! The title's inlined code writes packets into a list only while its put
//! pointer is below a limit, and calls the list's method otherwise. Our
//! lists keep the pointer and every limit at zero, so every command reaches
//! the table as a call.
//!
//! Each list records onto a host command list as its methods are called
//! (decision 0007); draw state lives in `crate::draw`. Methods not
//! written yet log once with their caller, so a run lists what to do next.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::mem::offset_of;

use crate::object::{
    self, ALLOCATOR, E_INVALIDARG, HRESULT, Iid, LIST, Object, S_OK, Table, This, guid,
};
#[cfg(not(all(windows, feature = "paged-title")))]
use crate::queue::Written;
use crate::resource::PAGEABLE_IIDS;

const ALLOCATOR_IIDS: [Iid; 4] = [
    guid(
        0x6102dee4,
        0xaf59,
        0x4b09,
        [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

/// The list interfaces up to `ID3D12GraphicsCommandList6` (the table has
/// all their slots): those whose ids are the same as on PC, the console's
/// own List5 and List6 (Grand Theft Auto VI asks for List6), then the
/// device child's.
const LIST_IIDS: [Iid; 10] = [
    guid(
        0x7116d91c,
        0xe7e4,
        0x47ce,
        [0xb8, 0xc6, 0xec, 0x81, 0x68, 0xf4, 0x37, 0xe5],
    ),
    guid(
        0x5b160d0f,
        0xac1b,
        0x4185,
        [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
    ),
    guid(
        0x553103fb,
        0x1fe7,
        0x4557,
        [0xbb, 0x38, 0x94, 0x6d, 0x7d, 0x0e, 0x7c, 0xa7],
    ),
    guid(
        0x38c3e585,
        0xff17,
        0x412c,
        [0x91, 0x50, 0x4f, 0xc6, 0xf9, 0xd7, 0x2a, 0x28],
    ),
    guid(
        0x6fda83a7,
        0xb84c,
        0x4e38,
        [0x9a, 0xc8, 0xc7, 0xbd, 0x22, 0x01, 0x6b, 0x3d],
    ),
    guid(
        0x8754318e,
        0xd3a9,
        0x4541,
        [0x98, 0xcf, 0x64, 0x5b, 0x50, 0xdc, 0x48, 0x74],
    ),
    guid(
        0x55050859,
        0x4024,
        0x474c,
        [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74],
    ),
    guid(
        0xc3827890,
        0x4024,
        0x474c,
        [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74],
    ),
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
struct Allocator {
    object: Object,
    kind: u32,
}

/// The layout the title's inlined code reads.
#[repr(C)]
struct List {
    object: Object,
    kind: u32,
    /// The put pointer, then the plain, draw, API and unbiased API limits:
    /// all zero, so every inline check fails.
    put: [u64; 5],
    reserved: [u32; 64],
    /// State the inline code keeps for building packets; unused by us.
    graphics_root_headers: [u32; 128],
    compute_root_headers: [u32; 128],
    pipeline_shadow: [u32; 8],
    /// Newer titles' lists are larger: Grand Theft Auto VI's state flush
    /// (title+0x32786) copies a pipeline's 32 bytes to +0x544..+0x564 and
    /// tests +0x548, so its layout runs past 0x560. Its size is unknown;
    /// this keeps our fields clear of whatever the title writes there.
    newer: [u32; 256],
    /// Ours, after the console layout: memory handed out by
    /// [`allocate_upload`], kept until the list is reset; the host list the
    /// calls record into (decision 0007), and the draw state the title set.
    uploads: std::sync::Mutex<Vec<std::sync::Arc<Upload>>>,
    #[cfg(windows)]
    host: Option<crate::host::HostList>,
    #[cfg(windows)]
    state: std::sync::Mutex<crate::draw::State>,
}
const _: () = assert!(offset_of!(List, put) == 0x18);
const _: () = assert!(offset_of!(List, pipeline_shadow) + 32 == 0x560);

/// Runs `f` with the host list and draw state of a title list, if there
/// is a host GPU.
#[cfg(windows)]
pub(crate) fn with_state(this: This, f: impl FnOnce(&crate::host::HostList, &mut crate::draw::State)) {
    let Some(list) = host_list(this) else { return };
    // SAFETY: the title passes one of our lists.
    let state = unsafe { &(*this.cast::<List>()).state };
    f(list, &mut state.lock().unwrap_or_else(|e| e.into_inner()));
}

/// The host list behind a title list, if there is a host GPU.
#[cfg(windows)]
pub(crate) fn host_list<'a>(this: This) -> Option<&'a crate::host::HostList> {
    if this.is_null() {
        return None;
    }
    // SAFETY: the title passes one of our lists.
    unsafe { (*this.cast::<List>()).host.as_ref() }
}

pub(crate) extern "system" fn create_allocator(
    _: This,
    kind: u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    let allocator = Box::new(Allocator {
        object: Object::new(&ALLOCATOR_FUNCTIONS),
        kind,
    });
    object::hand_out(ALLOCATOR, allocator, iid, &ALLOCATOR_IIDS, out)
}

/// Lists start open, as in D3D12.
pub(crate) extern "system" fn create_list(
    _: This,
    _nodes: u32,
    kind: u32,
    allocator: This,
    pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    new_list(kind, allocator, pipeline, iid, out)
}

/// The console's list creation: type, flags, node mask and barrier batch
/// size in a description.
#[repr(C)]
#[derive(Debug)]
pub(crate) struct ListDesc {
    kind: u32,
    flags: u32,
    nodes: u32,
    barrier_batch: u32,
}

pub(crate) extern "system" fn create_list_x(
    _: This,
    desc: *const ListDesc,
    allocator: This,
    pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "游戏主机命令列表");
    new_list(desc.kind, allocator, pipeline, iid, out)
}

fn new_list(
    kind: u32,
    allocator: This,
    pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    if allocator.is_null() {
        return E_INVALIDARG;
    }
    tracing::debug!(kind, "命令列表");
    let list = Box::new(List {
        object: if crate::device::newer() {
            Object::new(&NEWER_LIST_FUNCTIONS)
        } else {
            Object::new(&LIST_FUNCTIONS)
        },
        kind,
        put: [0; 5],
        reserved: [0; 64],
        graphics_root_headers: [0; 128],
        compute_root_headers: [0; 128],
        pipeline_shadow: [0; 8],
        newer: [0; 256],
        uploads: Default::default(),
        #[cfg(windows)]
        host: crate::host::host().and_then(|h| match h.new_list() {
            Ok(list) => Some(list),
            Err(e) => {
                tracing::warn!(error = %e, "无法创建宿主命令列表");
                None
            }
        }),
        #[cfg(windows)]
        state: {
            let mut state = crate::draw::State::default();
            state.set_pipeline(pipeline);
            std::sync::Mutex::new(state)
        },
    });
    let hr = object::hand_out(LIST, list, iid, &LIST_IIDS, out);
    if hr == S_OK {
        crate::pipeline::bound(pipeline);
    }
    hr
}

extern "system" fn allocator_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &ALLOCATOR_IIDS).then_some(this);
    object::answer(ALLOCATOR, iid, found, out)
}
extern "system" fn allocator_release(this: This) -> u32 {
    object::release_boxed::<Allocator>(this)
}
/// Nothing is recorded, so there is nothing to free.
extern "system" fn allocator_reset(_: This) -> HRESULT {
    S_OK
}

extern "system" fn list_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(LIST, iid, object::is(iid, &LIST_IIDS).then_some(this), out)
}
extern "system" fn list_release(this: This) -> u32 {
    object::release_boxed::<List>(this)
}
extern "system" fn list_type(this: This) -> u32 {
    // SAFETY: `this` is one of our lists.
    unsafe { (*this.cast::<List>()).kind }
}
extern "system" fn close(this: This) -> HRESULT {
    #[cfg(windows)]
    if let Some(host) = host_list(this) {
        host.close();
    }
    S_OK
}
extern "system" fn reset(this: This, allocator: This, pipeline: This) -> HRESULT {
    if allocator.is_null() {
        return E_INVALIDARG;
    }
    // Every submission of the list has run (submissions are waited for).
    // SAFETY: the title passes one of our lists.
    let uploads = unsafe { &(*this.cast::<List>()).uploads };
    uploads.lock().unwrap_or_else(|e| e.into_inner()).clear();
    #[cfg(windows)]
    with_state(this, |host, state| {
        host.reset();
        state.reset(pipeline);
    });
    crate::pipeline::bound(pipeline);
    #[cfg(not(windows))]
    let _ = this;
    S_OK
}

/// A resource barrier as titles pass it: type, flags, then a resource and
/// either a subresource and two states (transition) or a second resource
/// (aliasing). The same layout as the host's.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Barrier {
    kind: u32,
    flags: u32,
    first: This,
    second: u64,
    last: u64,
}

/// The barrier on host resources, or `None` if it names a resource the host
/// GPU does not have (buffers so far).
#[cfg(windows)]
fn host_barrier(list: &crate::host::HostList, b: &Barrier) -> Option<Barrier> {
    use windows::core::Interface;
    let host = |r: This| crate::resource::host_resource(r).map(|h| h.as_raw().cast_const().cast());
    match b.kind {
        0 => {
            // A split barrier's first half: its second half transitions.
            const BEGIN_ONLY: u32 = 1;
            if b.flags & BEGIN_ONLY != 0 {
                return None;
            }
            // shortcut: whole resources, from the state the list tracked
            // (`HostList::transition`); subresources are not told apart.
            let wanted = crate::host::texture_state(b.last as u32);
            let (texture, before) = list.transition(b.first, wanted)?;
            Some(Barrier {
                kind: 0,
                flags: 0,
                first: texture.as_raw().cast_const().cast(),
                second: 0xffff_ffff | (u64::from(before) << 32),
                last: u64::from(crate::host::legal_state(texture, wanted)),
            })
        }
        1 => Some(Barrier {
            first: if b.first.is_null() {
                b.first
            } else {
                host(b.first)?
            },
            second: if b.second == 0 {
                0
            } else {
                host(b.second as This)? as u64
            },
            ..*b
        }),
        2 => Some(Barrier {
            first: if b.first.is_null() {
                b.first
            } else {
                host(b.first)?
            },
            ..*b
        }),
        _ => None,
    }
}

/// An unordered access barrier on every resource: earlier shader writes
/// finish before later work reads or writes.
const UAV_BARRIER: Barrier = Barrier {
    kind: 2,
    flags: 0,
    first: std::ptr::null(),
    second: 0,
    last: 0,
};

/// Returns the console's barrier "actions" (none).
extern "system" fn barrier(this: This, count: u32, barriers: *const Barrier) -> u32 {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && !barriers.is_null()
    {
        // SAFETY: the title passes `count` barriers.
        let barriers = unsafe { std::slice::from_raw_parts(barriers, count as usize) };
        if cfg!(feature = "paged-title") {
            use crate::page_resource::PageBarrier;
            for b in barriers {
                let typed = match (b.kind, b.flags) {
                    (2, 0) => Some(if b.first.is_null() || crate::resource::stands_in(b.first) {
                        PageBarrier::Uav
                    } else {
                        crate::resource::host_copy(b.first).map_or(PageBarrier::Uav, PageBarrier::NativeUav)
                    }),
                    (0, 0) if b.second as u32 == u32::MAX && !b.first.is_null() => {
                        if let Some(texture) = crate::resource::host_copy(b.first) {
                            Some(PageBarrier::Texture(texture, crate::host::texture_state(b.last as u32)))
                        } else if crate::resource::buffer_bytes(b.first).is_some() {
                            Some(PageBarrier::Uav)
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                if let Some(typed) = typed {
                    list.record_page_barrier(typed);
                } else {
                    list.mark_unlowered();
                    break;
                }
            }
            return 0;
        }
        let mut host: Vec<Barrier> = barriers
            .iter()
            .filter_map(|b| host_barrier(list, b))
            .collect();
        // Buffers have host copies, not resources the barrier can name:
        // a barrier on one waits for every shader write before it.
        if barriers.iter().any(|b| {
            b.kind != 1 && !b.first.is_null() && crate::resource::buffer_bytes(b.first).is_some()
        }) {
            host.push(UAV_BARRIER);
        }
        if !host.is_empty() {
            use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER;
            const _: () = assert!(size_of::<Barrier>() == size_of::<D3D12_RESOURCE_BARRIER>());
            // SAFETY: the same layout; borrowed resource pointers, never
            // released through this slice.
            let host = unsafe {
                std::slice::from_raw_parts(
                    host.as_ptr().cast::<D3D12_RESOURCE_BARRIER>(),
                    host.len(),
                )
            };
            // SAFETY: an open list and live resources.
            unsafe { list.recording().ResourceBarrier(host) };
        }
    }
    0
}

/// Moves a title texture's host copy to `state` in the list, if needed.
#[cfg(windows)]
pub(crate) fn ensure_state(list: &crate::host::HostList, resource: This, state: u32) {
    use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES as State;
    if let Some((texture, before)) = list.transition(resource, state) {
        let state = crate::host::legal_state(texture, state);
        let barrier = crate::host::transition(texture, State(before as i32), State(state as i32));
        // SAFETY: an open list and a live texture.
        unsafe { list.recording().ResourceBarrier(&[barrier]) };
    }
}

#[cfg(windows)]
const COPY_DEST: u32 = 0x400;
#[cfg(windows)]
const COPY_SOURCE: u32 = 0x800;

/// `len` bytes at `offset` of a title buffer, if they are inside it.
#[cfg(windows)]
fn buffer_range(buffer: This, offset: u64, len: u64) -> Option<*mut u8> {
    let (data, size) = crate::resource::buffer_bytes(buffer)?;
    let end = offset.checked_add(len)?;
    if end > size {
        tracing::warn!(offset, len, size, "复制范围超出缓冲区");
        return None;
    }
    // SAFETY: inside the buffer, checked above.
    Some(unsafe { data.add(offset as usize) })
}

/// Buffer to buffer: both are host memory, copied at submission.
extern "system" fn copy_buffer_region(
    this: This,
    dst: This,
    dst_offset: u64,
    src: This,
    src_offset: u64,
    len: u64,
) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && let Some(to) = buffer_range(dst, dst_offset, len)
        && let Some(from) = buffer_range(src, src_offset, len)
    {
        list.defer_copy(to, from, len as usize);
    }
    #[cfg(not(windows))]
    let _ = (this, dst, dst_offset, src, src_offset, len);
}

extern "system" fn copy_resource(this: This, dst: This, src: This) {
    #[cfg(windows)]
    {
        let Some(list) = host_list(this) else {
            return;
        };
        if cfg!(feature = "paged-title") {
            if let (Some(to), Some(from)) = (
                crate::resource::buffer_bytes(dst), crate::resource::buffer_bytes(src),
            ) {
                if to.1 != from.1 || (to.1 != 0 && (to.0 == from.0
                    || crate::copy_plan::GuestCopy::new(to.0 as usize, from.0 as usize, to.1 as usize).is_none()))
                {
                    list.mark_unlowered();
                } else {
                    list.defer_copy(to.0, from.0, to.1 as usize);
                }
            } else if let Some(copy) = crate::resource::host_copy(src)
                .zip(crate::resource::host_copy(dst))
                .and_then(|(from, to)| crate::page_resource::TextureCopy::capture(from, to))
            {
                list.record_page_texture_copy(copy);
            } else {
                list.mark_unlowered();
            }
            return;
        }
        if let (Some(to), Some(from)) = (
            crate::resource::host_resource(dst),
            crate::resource::host_resource(src),
        ) {
            if !crate::resource::copy_formats_match(dst, src) {
                return;
            }
            if crate::texture::title_written(from) {
                crate::texture::upload(list, src, from);
            }
            ensure_state(list, dst, COPY_DEST);
            ensure_state(list, src, COPY_SOURCE);
            crate::texture::host_written(dst);
            #[cfg(feature = "diagnostics")]
            list.note_op(|| format!("copy resource {src:?} to {dst:?}"));
            // SAFETY: an open list and live textures.
            unsafe { list.recording().CopyResource(to, from) };
        } else if let (Some((to, a)), Some((from, b))) = (
            crate::resource::buffer_bytes(dst),
            crate::resource::buffer_bytes(src),
        ) {
            list.defer_copy(to, from, a.min(b) as usize);
        }
    }
    #[cfg(not(windows))]
    let _ = (this, dst, src);
}

extern "system" fn copy_resource_x(this: This, dst: This, src: This, _flags: u32) {
    #[cfg(windows)]
    if cfg!(feature = "paged-title") && _flags != 0 {
        if let Some(list) = host_list(this) {
            list.mark_unlowered();
        }
        return;
    }
    copy_resource(this, dst, src);
}

/// A copy location as titles pass it, laid out as the host's: a resource,
/// then a subresource index (type 0) or a placed footprint (type 1).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CopyLocation {
    resource: This,
    kind: u32,
    footprint: crate::resource::PlacedFootprint,
}

/// Texture to texture on the GPU; buffer to texture through an upload
/// buffer filled at submission. Texture to buffer (read back) is not done
/// yet.
extern "system" fn copy_texture_region(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    source_box: *const [u32; 6],
) {
    #[cfg(windows)]
    {
        use windows::Win32::Graphics::Direct3D12::{D3D12_BOX, D3D12_TEXTURE_COPY_LOCATION};
        use windows::core::Interface;
        const FOOTPRINT: u32 = 1;
        let Some(list) = host_list(this) else { return; };
        if cfg!(feature = "paged-title") {
            // In subresource form the ABI initializes only the 32-bit
            // union member, not the rest of the placed-footprint storage.
            let subresource = |location: *const CopyLocation| {
                if location.is_null() {
                    return None;
                }
                // SAFETY: the title supplies a readable copy location.
                unsafe {
                    ((*location).kind == 0).then(|| (
                        (*location).resource, (&raw const (*location).footprint).cast::<u32>().read(),
                    ))
                }
            };
            let native = |location| subresource(location).and_then(|(resource, index)| {
                crate::resource::host_copy(resource).map(|copy| (copy, index))
            });
            // SAFETY: the ABI supplies either a readable six-field box or null.
            let bounds = unsafe { source_box.as_ref() }.copied();
            // SAFETY: the discriminator is initialized for every
            // readable location; read the footprint only in that variant.
            if !src.is_null() && unsafe { (*src).kind } == FOOTPRINT {
                let target = native(dst);
                let upload = if z == 0 {
                    // SAFETY: the initialized footprint variant.
                    unsafe { (*src).footprint.linear_page_view((*src).resource) }
                        .zip(target)
                        .and_then(|(from, (to, index))| crate::page_resource::TextureUpload::capture(from, to, index, (x, y), bounds))
                } else {
                    None
                };
                if let Some(upload) = upload {
                    list.record_page_texture_upload(upload);
                } else {
                    list.mark_unlowered();
                }
                return;
            }
            // SAFETY: inspect the destination footprint only after checking
            // its initialized discriminator.
            if !dst.is_null() && unsafe { (*dst).kind } == FOOTPRINT {
                let readback = if x == 0 && y == 0 && z == 0 && bounds.is_none() {
                    // SAFETY: the initialized footprint variant.
                    let target = unsafe { (*dst).footprint.linear_page_view((*dst).resource) };
                    native(src).zip(target)
                        .and_then(|((from, index), to)| crate::page_resource::TextureReadback::capture_subresource(from, index, to))
                } else {
                    None
                };
                if let Some(readback) = readback {
                    list.record_page_texture_readback(readback);
                } else {
                    list.mark_unlowered();
                }
                return;
            }
            if z != 0 {
                list.mark_unlowered();
                return;
            }
            if let Some(copy) = native(src).zip(native(dst))
                .and_then(|((from, src_index), (to, dst_index))| {
                    crate::page_resource::TextureCopy::capture_region(from, src_index, to, dst_index, x, y, bounds)
                })
            {
                list.record_page_texture_copy(copy);
            } else {
                list.mark_unlowered();
            }
            return;
        }
        // SAFETY: the title passes both locations (null is tolerated).
        let (Some(&dst), Some(&src)) = (unsafe { dst.as_ref() }, unsafe { src.as_ref() }) else {
            return;
        };
        let Some(to) = crate::resource::host_resource(dst.resource) else {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!("尚未实现从纹理复制到缓冲区"));
            return;
        };
        let mut from = src;
        let _upload; // keeps the source alive while this call records it
        #[cfg(feature = "diagnostics")]
        let mut read = None;
        if src.kind == FOOTPRINT {
            if !crate::resource::copy_formats_match(dst.resource, std::ptr::null()) {
                return;
            }
            let (Some(host), Some(bytes)) = (crate::host::host(), src.footprint.bytes()) else {
                return;
            };
            let Some(data) = buffer_range(src.resource, src.footprint.offset, bytes) else {
                return;
            };
            let Some((buffer, staging)) = list.upload(host, bytes) else {
                return;
            };
            // The list owns this mapped D3D12 upload buffer. Wine can report
            // its address as MEM_FREE, so use the host-upload path.
            list.defer_fill(staging, data, bytes as usize);
            from.resource = buffer.as_raw().cast_const().cast();
            from.footprint.offset = 0;
            #[cfg(feature = "diagnostics")]
            {
                // SAFETY: a live buffer.
                read = Some((unsafe { buffer.GetGPUVirtualAddress() }, bytes));
            }
            _upload = buffer;
        } else {
            let Some(texture) = crate::resource::host_resource(src.resource) else {
                return;
            };
            if !crate::resource::copy_formats_match(dst.resource, src.resource) {
                return;
            }
            if crate::texture::title_written(texture) {
                crate::texture::upload(list, src.resource, texture);
            }
            ensure_state(list, src.resource, COPY_SOURCE);
            from.resource = texture.as_raw().cast_const().cast();
        }
        ensure_state(list, dst.resource, COPY_DEST);
        crate::texture::host_written(dst.resource);
        let mut to_location = dst;
        to_location.resource = to.as_raw().cast_const().cast();
        let host_box = (!source_box.is_null()).then(|| source_box.cast::<D3D12_BOX>());
        #[cfg(feature = "diagnostics")]
        {
            let text = || {
                // SAFETY: a live texture; the box is the title's or null.
                let (d, b) = unsafe { (to.GetDesc(), host_box.map(|b| *b)) };
                format!(
                    "copy to {:?} {}x{} f{} sub/kind {} at {x},{y},{z} from {:?} kind {} footprint {:?} box {b:?}",
                    dst.resource,
                    d.Width,
                    d.Height,
                    d.Format.0,
                    dst.kind,
                    src.resource,
                    src.kind,
                    (src.kind == FOOTPRINT).then_some(src.footprint)
                )
            };
            match read {
                Some(range) => list.note_read(text, range),
                None => list.note_op(text),
            }
        }
        const _: () =
            assert!(size_of::<CopyLocation>() == size_of::<D3D12_TEXTURE_COPY_LOCATION>());
        // SAFETY: an open list; both locations in the host's layout, naming
        // live resources; the box is the title's (same layout).
        unsafe {
            list.recording().CopyTextureRegion(
                (&raw const to_location).cast(),
                x,
                y,
                z,
                (&raw const from).cast(),
                host_box,
            )
        };
    }
    #[cfg(not(windows))]
    let _ = (this, dst, x, y, z, src, source_box);
}

#[allow(clippy::too_many_arguments)] // the method's own arguments
extern "system" fn copy_texture_region_x(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    source_box: *const [u32; 6],
    _flags: u32,
) {
    #[cfg(windows)]
    if cfg!(feature = "paged-title") && _flags != 0 {
        if let Some(list) = host_list(this) {
            list.mark_unlowered();
        }
        return;
    }
    copy_texture_region(this, dst, x, y, z, src, source_box);
}

extern "system" fn resolve(
    this: This,
    dst: This,
    dst_sub: u32,
    src: This,
    src_sub: u32,
    format: u32,
) {
    #[cfg(all(windows, feature = "paged-title"))]
    {
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
        let Some(list) = host_list(this) else { return; };
        if let (Some(to), Some(from)) = (crate::resource::host_copy(dst), crate::resource::host_copy(src))
            && !crate::resource::stands_in(dst) && !crate::resource::stands_in(src)
            && let Some(op) = crate::page_resource::TextureResolve::capture(from, src_sub, to, dst_sub,
                DXGI_FORMAT(crate::resource::pc_view_format(dst, format) as i32))
        {
            list.record_page_texture_resolve(op);
        } else {
            list.mark_unlowered();
        }
    }
    #[cfg(all(windows, not(feature = "paged-title")))]
    if let Some(list) = host_list(this)
        && let (Some(to), Some(from)) = (
            crate::resource::host_resource(dst),
            crate::resource::host_resource(src),
        )
    {
        use windows::Win32::Foundation::RECT;
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_RESOLVE_MODE_AVERAGE, ID3D12GraphicsCommandList1,
        };
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
        use windows::core::Interface;
        const RESOLVE_DEST: u32 = 0x1000;
        const RESOLVE_SOURCE: u32 = 0x2000;
        if !crate::resource::copy_formats_match(dst, src) {
            return;
        }
        let format = crate::resource::pc_view_format(dst, format);
        ensure_state(list, dst, RESOLVE_DEST);
        ensure_state(list, src, RESOLVE_SOURCE);
        crate::texture::host_written(dst);
        #[cfg(feature = "diagnostics")]
        list.note_op(|| format!("resolve {src:?} to {dst:?} format {format}"));
        // SAFETY: an open list and live textures; the region is inside both.
        unsafe {
            let (a, b) = (to.GetDesc(), from.GetDesc());
            let format = DXGI_FORMAT(format as i32);
            if (a.Width, a.Height) == (b.Width, b.Height) {
                list.recording()
                    .ResolveSubresource(to, dst_sub, from, src_sub, format);
            } else if let Ok(list1) = list.recording().cast::<ID3D12GraphicsCommandList1>() {
                // The console clips a larger source to the destination.
                let rect = RECT {
                    left: 0,
                    top: 0,
                    right: a.Width.min(b.Width) as i32,
                    bottom: a.Height.min(b.Height) as i32,
                };
                list1.ResolveSubresourceRegion(
                    to,
                    dst_sub,
                    0,
                    0,
                    from,
                    src_sub,
                    Some(&rect),
                    format,
                    D3D12_RESOLVE_MODE_AVERAGE,
                );
            }
        }
    }
    #[cfg(not(windows))]
    let _ = (this, dst, dst_sub, src, src_sub, format);
}

#[repr(C)]
pub(crate) struct Rect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

extern "system" fn clear_rtv(
    this: This,
    handle: u64,
    color: *const [f32; 4],
    count: u32,
    rects: *const Rect,
) {
    #[cfg(windows)]
    {
        let Some(list) = host_list(this) else { return };
        let Some(record) = crate::descriptor::rtv_record(handle) else {
            if cfg!(feature = "paged-title") { list.mark_unlowered(); }
            return;
        };
        let (Some(texture), Some(host)) = (
            crate::resource::host_resource(record.resource),
            crate::host::host(),
        ) else {
            if cfg!(feature = "paged-title") { list.mark_unlowered(); }
            return;
        };
        // SAFETY: the title passes four floats (null is tolerated).
        let Some(color) = (unsafe { color.as_ref() }) else {
            if cfg!(feature = "paged-title") { list.mark_unlowered(); }
            return;
        };
        if cfg!(feature = "paged-title") && count > 0 && rects.is_null() {
            list.mark_unlowered();
            return;
        }
        let rects = (count > 0 && !rects.is_null()).then(|| {
            // SAFETY: the title passes `count` rectangles, laid out as the host's.
            unsafe { std::slice::from_raw_parts(rects.cast(), count as usize) }
        });
        if cfg!(feature = "paged-title") {
            let Some(target) = crate::resource::host_copy(record.resource)
                .and_then(|copy| crate::page_draw::ColorTarget::capture_candidate(copy, record.desc()))
            else {
                list.mark_unlowered();
                return;
            };
            list.record_page_clear(crate::page_draw::ColorClear {
                target, color: *color, rects: rects.unwrap_or_default().to_vec(),
            });
            return;
        }
        const RENDER_TARGET: u32 = 0x4;
        let rtv = host.rtv(texture, record.desc().as_ref());
        // SAFETY: an open list, a fresh view of a live texture.
        unsafe {
            if let Some((_, before)) = list.transition(record.resource, RENDER_TARGET) {
                use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES as State;
                list.recording().ResourceBarrier(&[crate::host::transition(
                    texture,
                    State(before as i32),
                    State(RENDER_TARGET as i32),
                )]);
            }
            list.recording().ClearRenderTargetView(rtv, color, rects);
        }
    }
    #[cfg(not(windows))]
    let _ = (this, handle, color, count, rects);
}

/// `ClearDepthStencilView(handle, flags, depth, stencil, count, rects)`;
/// flags: 1 depth, 2 stencil.
extern "system" fn clear_dsv(
    this: This,
    handle: u64,
    flags: u32,
    depth: f32,
    stencil: u8,
    count: u32,
    rects: *const Rect,
) {
    #[cfg(windows)]
    {
        use windows::Win32::Graphics::Direct3D12::D3D12_CLEAR_FLAGS;
        let Some(list) = host_list(this) else { return };
        let Some(record) = crate::descriptor::dsv_record(handle) else {
            if cfg!(feature = "paged-title") { list.mark_unlowered(); }
            return;
        };
        let (Some(texture), Some(host)) = (
            crate::resource::host_resource(record.resource),
            crate::host::host(),
        ) else {
            if cfg!(feature = "paged-title") { list.mark_unlowered(); }
            return;
        };
        if cfg!(feature = "paged-title") && count > 0 && rects.is_null() {
            list.mark_unlowered();
            return;
        }
        let rects = (count > 0 && !rects.is_null()).then(|| {
            // SAFETY: the title passes `count` rectangles, laid out as the host's.
            unsafe { std::slice::from_raw_parts(rects.cast(), count as usize) }
        });
        if cfg!(feature = "paged-title") {
            let Some(target) = crate::resource::host_copy(record.resource)
                .and_then(|copy| crate::page_draw::DepthTarget::capture(copy, record.depth_desc()))
            else {
                list.mark_unlowered();
                return;
            };
            list.record_page_clear_depth(crate::page_draw::DepthClear {
                target, flags, depth, stencil, rects: rects.unwrap_or_default().to_vec(),
            });
            return;
        }
        const DEPTH_WRITE: u32 = 0x10;
        ensure_state(list, record.resource, DEPTH_WRITE);
        let dsv = host.dsv(texture, record.depth_desc().as_ref());
        // SAFETY: an open list, a fresh view of a live texture.
        unsafe {
            list.recording()
                .ClearDepthStencilView(dsv, D3D12_CLEAR_FLAGS(flags as i32 & 3), depth, stencil, rects)
        };
    }
    #[cfg(not(windows))]
    let _ = (this, handle, flags, depth, stencil, count, rects);
}

/// `CopyMemoryToMemoryX(destination, source, size, flags)`: GPU addresses
/// are title memory, copied at submission like buffer copies (if mapped).
extern "system" fn copy_memory(this: This, dst: u64, src: u64, len: u64, _flags: u32) {
    #[cfg(windows)]
    if let Some(list) = host_list(this) {
        if cfg!(feature = "paged-title") {
            if _flags != 0 || (len != 0 && (dst == 0 || src == 0)) {
                list.mark_unlowered();
            } else if len != 0 {
                if crate::copy_plan::GuestCopy::new(dst as usize, src as usize, len as usize).is_some() {
                    list.defer_copy(dst as *mut u8, src as *const u8, len as usize);
                } else {
                    list.mark_unlowered();
                }
            }
            return;
        }
        if dst == 0 || src == 0 { return; }
        let len = len as usize;
        list.defer(move || {
            if crate::texture::committed(src as *const u8, len)
                && crate::texture::committed(dst as *const u8, len)
            {
                // SAFETY: both ranges are committed title memory.
                unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, len) };
            }
        });
    }
    #[cfg(not(windows))]
    let _ = (this, dst, src, len);
}

/// `DriverHintX(feature, value)`: no hints for the host.
extern "system" fn driver_hint(_: This, _feature: u64, _value: u64) -> u64 {
    0
}

/// `DispatchX(x, y, z, flags)`: a dispatch with console flags, which the
/// host has no use for (the ray-tracing builder dispatches this way).
extern "system" fn dispatch_x(this: This, x: u32, y: u32, z: u32, _flags: u32) {
    #[cfg(windows)]
    state::dispatch(this, x, y, z);
    #[cfg(not(windows))]
    {
        let _ = (this, x, y, z);
        count_dispatch();
    }
}

extern "system" fn begin_query(this: This, _heap: This, _kind: u32, _index: u32) {
    #[cfg(all(windows, feature = "paged-title"))]
    record_query_boundary(this, _heap, _kind, _index, true);
    let _ = this;
}

extern "system" fn end_query(this: This, _heap: This, _kind: u32, _index: u32) {
    #[cfg(all(windows, feature = "paged-title"))]
    record_query_boundary(this, _heap, _kind, _index, false);
    let _ = this;
}

#[cfg(all(windows, feature = "paged-title"))]
fn record_query_boundary(this: This, query_heap: This, kind: u32, index: u32, begin: bool) {
    use crate::page_queue::QueryOp;
    use windows::Win32::Graphics::Direct3D12::D3D12_QUERY_TYPE_TIMESTAMP;
    let Some(list) = host_list(this) else { return; };
    let query = crate::query::native_heap(query_heap, kind)
        .filter(|(heap, kind)| index < heap.count && (!begin || *kind != D3D12_QUERY_TYPE_TIMESTAMP));
    if let Some((heap, kind)) = query {
        list.record_page_query(if begin { QueryOp::Begin { heap, kind, index } }
            else { QueryOp::End { heap, kind, index } });
    } else {
        list.mark_unlowered();
    }
}

/// `ResolveQueryData(heap, type, first, count, buffer, offset)`: writes
/// each query's result into the buffer once the list has run. Timestamps
/// (type 2) are the GPU clock then; occlusion counts (0) say a million
/// samples passed and binary occlusion (1) visible, so nothing is culled
/// by a query.
/// The legacy path below approximates results; VNME never uses it.
extern "system" fn resolve_queries(
    this: This,
    _heap: This,
    kind: u32,
    _first: u32,
    count: u32,
    buffer: This,
    offset: u64,
) {
    #[cfg(all(windows, feature = "paged-title"))]
    {
        if let Some(list) = host_list(this) {
            if count == 0 { return; }
            let op = crate::query::native_heap(_heap, kind)
                .zip(crate::resource::buffer_bytes(buffer))
                .and_then(|((heap, kind), (data, size))| crate::page_queue::QueryOp::resolve(
                    heap, kind, _first, count, data as u64, size, offset,
                ));
            if let Some(op) = op { list.record_page_query(op); }
            else { list.mark_unlowered(); }
        }
        return;
    }
    #[cfg(not(all(windows, feature = "paged-title")))]
    {
        const OCCLUSION: u32 = 0;
        const BINARY: u32 = 1;
        const TIMESTAMP: u32 = 2;
        const PIPELINE_STATISTICS: u32 = 3;
        let Some((data, size)) = crate::resource::buffer_bytes(buffer) else {
            return;
        };
        let (what, bytes) = match kind {
            OCCLUSION => (Written::Value(1 << 20), 8),
            BINARY => (Written::Value(1), 8),
            TIMESTAMP => (Written::GpuTimestamp, 8),
            PIPELINE_STATISTICS => (Written::Value(0), 88),
            _ => (Written::Value(0), 16),
        };
        let Some(end) = offset.checked_add(u64::from(count) * bytes) else {
            return;
        };
        if end > size {
            return;
        }
        for i in 0..u64::from(count) {
            // Wider results are written as 8-byte pieces.
            for piece in 0..bytes / 8 {
                // SAFETY: inside the buffer, checked above.
                let at = unsafe { data.add((offset + i * bytes + piece * 8) as usize) };
                let what = if piece == 0 { what } else { Written::Value(0) };
                defer_write(this, at as u64, what, 8);
            }
        }
    }
}

/// IUnknown 3, object 4, device child 1, allocator 2.
static ALLOCATOR_FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<ALLOCATOR, 24>());
    f[0] = allocator_query as *const ();
    f[2] = allocator_release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = allocator_reset as *const (); // 0x40
    f[9] = object::ignore as *const (); // 0x48 allocation callback
    Table(f)
};

extern "system" fn set_pipeline(this: This, pipeline: This) {
    crate::pipeline::bound(pipeline);
    #[cfg(windows)]
    with_state(this, |_, state| state.set_pipeline(pipeline));
    #[cfg(not(windows))]
    let _ = this;
}

/// 全局 UAV 屏障：让后续 GPU 命令看到先前的着色器写入。
fn wait_for_shader_writes(this: This) {
    #[cfg(windows)]
    if let Some(list) = host_list(this) {
        if cfg!(feature = "paged-title") {
            // 屏障必须和其前后的命令一起进入 VNME，不能写进旧命令列表。
            list.record_page_barrier(crate::page_resource::PageBarrier::Uav);
            return;
        }
        use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER;
        // SAFETY: Barrier 与 D3D12 的布局一致，空资源指针表示全局 UAV 屏障。
        let barrier =
            unsafe { std::mem::transmute::<Barrier, D3D12_RESOURCE_BARRIER>(UAV_BARRIER) };
        // SAFETY: 此处仍在录制宿主命令列表。
        unsafe { list.recording().ResourceBarrier(&[barrier]) };
    }
    #[cfg(not(windows))]
    let _ = this;
}

/// 管线两端的同步点沿用全局 UAV 顺序，不宣称复现 Xbox 专用缓存控制。
extern "system" fn sync(this: This, _flags: u32, _control: u32, _address: u64, _size: u64) {
    wait_for_shader_writes(this);
}

/// Reset with console flags `(allocator, pipeline, flags)`; the flags only
/// tune the console's command memory.
extern "system" fn reset_x(this: This, allocator: This, pipeline: This, _flags: u32) -> HRESULT {
    reset(this, allocator, pipeline)
}

/// Newer lists' 0x4a0 `(size, alignment) -> pointer`: memory the list
/// owns until it is reset, which the CPU writes and the GPU reads at the
/// same address; null on failure. Grand Theft Auto VI's BVH builder puts
/// its instance descriptions there (and wrote through our old
/// `E_NOTIMPL` as a pointer).
extern "system" fn allocate_upload(this: This, size: u64, alignment: u64) -> *mut u8 {
    let align = usize::try_from(alignment).unwrap_or(0).max(16);
    let Some(len) = usize::try_from(size).ok().and_then(|s| s.checked_add(align)) else {
        return std::ptr::null_mut();
    };
    if this.is_null() || !align.is_power_of_two() {
        return std::ptr::null_mut();
    }
    let Some(block) = Upload::new(len) else {
        return std::ptr::null_mut();
    };
    let block = std::sync::Arc::new(block);
    let start = block.start();
    // SAFETY: the offset is below `align`, so `size` bytes fit after it.
    let at = unsafe { start.add(start.align_offset(align)) };
    // SAFETY: the title passes one of our lists.
    let uploads = unsafe { &(*this.cast::<List>()).uploads };
    #[cfg(all(windows, feature = "paged-title"))]
    if let Some(list) = host_list(this) {
        list.record_upload_lease(block.clone());
    }
    uploads.lock().unwrap_or_else(|e| e.into_inner()).push(block);
    at
}

/// Memory [`allocate_upload`] hands out, zeroed, freed when the list is
/// reset. On Windows it lies in [4 GiB, 2^40), where translated shaders
/// find title memory by address (`crate::memory`; the console's GPU
/// addresses have 40 bits), and is a buffer of its own to the host copies
/// (`crate::shadow`), so its copy is exactly it.
pub(crate) struct Upload {
    #[cfg(windows)]
    at: usize,
    #[cfg(windows)]
    len: usize,
    #[cfg(all(windows, feature = "paged-title"))]
    free: fn(usize),
    #[cfg(not(windows))]
    block: Vec<u8>,
}

impl Upload {
    #[cfg(all(windows, feature = "paged-title"))]
    fn new(len: usize) -> Option<Upload> {
        let len = len.checked_next_multiple_of(victory_memory::PAGE_BYTES as usize)?;
        let (at, free) = crate::title_backing::allocate_upload(len)?;
        Some(Upload { at, len, free })
    }

    #[cfg(all(windows, not(feature = "paged-title")))]
    fn new(len: usize) -> Option<Upload> {
        use windows_sys::Win32::System::Memory::{
            MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER, MEM_RESERVE,
            MemExtendedParameterAddressRequirements, PAGE_READWRITE, VirtualAlloc2,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let mut requirements = MEM_ADDRESS_REQUIREMENTS {
            LowestStartingAddress: (1usize << 32) as *mut _,
            HighestEndingAddress: ((1usize << 40) - 1) as *mut _,
            Alignment: 0,
        };
        let mut parameter = MEM_EXTENDED_PARAMETER::default();
        parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
        parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
        // Watched like the title's pools: copies take only written pages.
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
        if at == 0 {
            return None;
        }
        crate::shadow::register(at, at as u64, len as u64);
        Some(Upload { at, len })
    }

    #[cfg(not(windows))]
    fn new(len: usize) -> Option<Upload> {
        Some(Upload {
            block: vec![0; len],
        })
    }

    fn start(&self) -> *mut u8 {
        #[cfg(windows)]
        return self.at as *mut u8;
        #[cfg(not(windows))]
        return self.block.as_ptr().cast_mut();
    }
}

#[cfg(all(windows, feature = "paged-title"))]
impl Drop for Upload {
    fn drop(&mut self) {
        crate::watch::title_memory_remapped(self.at, self.len);
        (self.free)(self.at);
    }
}

#[cfg(all(windows, not(feature = "paged-title")))]
impl Drop for Upload {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
        crate::shadow::unregister(self.at, self.at as u64);
        // Not guarded any more: memory allocated here next is new.
        crate::watch::title_memory_remapped(self.at, self.len);
        // SAFETY: our allocation; lists that still use its copy hold that.
        unsafe { VirtualFree(self.at as *mut _, 0, MEM_RELEASE) };
    }
}

// SAFETY: plain memory, handed out once.
#[cfg(windows)]
unsafe impl Send for Upload {}

/// Draw state the host has no use for (root signatures: ours come with
/// the pipeline; depth bounds, depth order and shading rate hints): any
/// arguments, nothing returned.
extern "system" fn ignored_state(_: This) {}

/// 只为定位渲染目标首次使用记录被忽略的丢弃；不访问区域指针。
extern "system" fn discard_resource(this: This, resource: This, region: *const core::ffi::c_void) {
    #[cfg(all(windows, feature = "paged-title", feature = "diagnostics"))]
    if crate::page_inventory::enabled()
        && let Some(list) = host_list(this)
        && let Some(copy) = crate::resource::host_copy(resource)
    {
        list.note_page_discard(copy, region.is_null());
    }
    let _ = (this, resource, region);
}

/// The title's draw state (decision 0007, step 3), kept for its draws or
/// set on the host list at once. Without a host GPU these do nothing.
#[cfg(windows)]
mod state {
    use super::{This, with_state};
    use crate::draw::{IndexView, VertexView};
    use windows::Win32::Foundation::RECT;
    use windows::Win32::Graphics::Direct3D12::D3D12_VIEWPORT;

    /// `count` items at `items`, or none.
    fn items<'a, T>(items: *const T, count: u32) -> &'a [T] {
        if items.is_null() || count == 0 {
            return &[];
        }
        // SAFETY: the title passes `count` items.
        unsafe { std::slice::from_raw_parts(items, count as usize) }
    }

    pub(super) extern "system" fn topology(this: This, topology: u32) {
        with_state(this, |list, s| s.set_topology(list, topology));
    }
    pub(super) extern "system" fn viewports(
        this: This,
        count: u32,
        viewports: *const D3D12_VIEWPORT,
    ) {
        with_state(this, |list, s| s.set_viewports(list, items(viewports, count)));
    }
    pub(super) extern "system" fn scissors(this: This, count: u32, rects: *const RECT) {
        with_state(this, |list, s| s.set_scissors(list, items(rects, count)));
    }
    pub(super) extern "system" fn blend_factor(this: This, factor: *const [f32; 4]) {
        // SAFETY: the title passes four floats or null.
        with_state(this, |list, s| s.set_blend_factor(list, unsafe { factor.as_ref() }));
    }
    pub(super) extern "system" fn stencil_ref(this: This, reference: u32) {
        with_state(this, |list, state| state.set_stencil_ref(list, reference));
    }
    pub(super) extern "system" fn heaps(this: This, count: u32, heaps: *const This) {
        with_state(this, |_, s| s.set_heaps(items(heaps, count)));
    }
    /// `ClearUnorderedAccessViewUint(GPU handle, CPU handle, resource,
    /// values, rectangle count, rectangles)`.
    pub(super) extern "system" fn clear_unordered_uint(
        this: This,
        _gpu: u64,
        cpu: u64,
        _resource: This,
        values: *const [u32; 4],
        count: u32,
        rects: *const RECT,
    ) {
        clear_unordered(this, cpu, values, false, items(rects, count));
    }
    /// `ClearUnorderedAccessViewFloat`, the same with four floats.
    pub(super) extern "system" fn clear_unordered_float(
        this: This,
        _gpu: u64,
        cpu: u64,
        _resource: This,
        values: *const [u32; 4],
        count: u32,
        rects: *const RECT,
    ) {
        clear_unordered(this, cpu, values, true, items(rects, count));
    }
    fn clear_unordered(this: This, cpu: u64, values: *const [u32; 4], float: bool, rects: &[RECT]) {
        // SAFETY: the title passes four values (null is tolerated).
        let Some(&values) = (unsafe { values.as_ref() }) else {
            return;
        };
        with_state(this, |list, s| {
            s.clear_unordered(list, cpu, values, float, rects);
        });
    }
    pub(super) extern "system" fn compute_table(this: This, slot: u32, handle: u64) {
        with_state(this, |_, s| s.set_table(true, slot, handle));
    }
    pub(super) extern "system" fn graphics_table(this: This, slot: u32, handle: u64) {
        with_state(this, |_, s| s.set_table(false, slot, handle));
    }
    pub(super) extern "system" fn compute_constant(this: This, slot: u32, value: u32, offset: u32) {
        with_state(this, |_, s| s.set_constants(true, slot, &[value], offset));
    }
    pub(super) extern "system" fn graphics_constant(
        this: This,
        slot: u32,
        value: u32,
        offset: u32,
    ) {
        with_state(this, |_, s| s.set_constants(false, slot, &[value], offset));
    }
    pub(super) extern "system" fn compute_constants(
        this: This,
        slot: u32,
        count: u32,
        values: *const u32,
        offset: u32,
    ) {
        with_state(this, |_, s| {
            s.set_constants(true, slot, items(values, count), offset)
        });
    }
    pub(super) extern "system" fn graphics_constants(
        this: This,
        slot: u32,
        count: u32,
        values: *const u32,
        offset: u32,
    ) {
        with_state(this, |_, s| {
            s.set_constants(false, slot, items(values, count), offset)
        });
    }
    /// Root constant buffer, shader resource and unordered access views.
    pub(super) extern "system" fn compute_view(this: This, slot: u32, address: u64) {
        with_state(this, |_, s| s.set_view(true, slot, address));
    }
    pub(super) extern "system" fn graphics_view(this: This, slot: u32, address: u64) {
        with_state(this, |_, s| s.set_view(false, slot, address));
    }
    pub(super) extern "system" fn index_buffer(this: This, view: *const IndexView) {
        // SAFETY: the title passes a view or null.
        with_state(this, |_, s| s.set_index_buffer(unsafe { view.as_ref() }));
    }
    pub(super) extern "system" fn vertex_buffers(
        this: This,
        start: u32,
        count: u32,
        views: *const VertexView,
    ) {
        with_state(this, |_, s| {
            if views.is_null() {
                s.clear_vertex_buffers(start, count);
            } else {
                s.set_vertex_buffers(start, items(views, count));
            }
        });
    }
    /// `(count, handles, one range, depth)`: with one range, the handles
    /// follow the first at the render target view size (128 bytes).
    pub(super) extern "system" fn targets(
        this: This,
        count: u32,
        handles: *const u64,
        one_range: i32,
        depth: *const u64,
    ) {
        let handles: Vec<u64> = if one_range != 0 && !handles.is_null() {
            // SAFETY: the title passes the first handle.
            let first = unsafe { *handles };
            (0..u64::from(count)).map(|i| first + 128 * i).collect()
        } else {
            items(handles, count).to_vec()
        };
        // SAFETY: the title passes a handle or null.
        let depth = unsafe { depth.as_ref() }.copied().unwrap_or(0);
        with_state(this, |list, s| s.set_targets(list, &handles, depth));
    }
    pub(super) extern "system" fn draw(
        this: This,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        super::count_draw();
        with_state(this, |list, s| {
            s.draw(list, vertices, instances, first, first_instance)
        });
    }
    pub(super) extern "system" fn dispatch(this: This, x: u32, y: u32, z: u32) {
        super::count_dispatch();
        with_state(this, |list, s| s.dispatch(list, x, y, z));
    }
    /// The groups a mesh dispatch of `x` × `y` × `z` groups runs: the
    /// title's mesh programs read only the x group id, so `x` (y and z
    /// other than 1 are logged once); none if any dimension is empty.
    pub(super) fn mesh_groups(x: u32, y: u32, z: u32) -> Option<u32> {
        if x == 0 || y == 0 || z == 0 {
            return None;
        }
        if y != 1 || z != 1 {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(x, y, z, "网格任务在 Y 或 Z 维度上有多个线程组"));
        }
        Some(x)
    }
    /// `DispatchMesh(x, y, z)` (the PC method of List6).
    pub(super) extern "system" fn dispatch_mesh(this: This, x: u32, y: u32, z: u32) {
        super::count_draw();
        if let Some(x) = mesh_groups(x, y, z) {
            with_state(this, |list, s| s.dispatch_mesh(list, x));
        }
    }
    /// `DispatchMeshX(x, flags, indirect arguments, payload)`.
    /// shortcut: dispatches whose group count is in GPU memory (indirect)
    /// are dropped; make them ExecuteIndirect when the title uses them.
    pub(super) extern "system" fn dispatch_mesh_x(
        this: This,
        x: u32,
        _flags: u64,
        indirect: u64,
        _payload: u64,
    ) {
        super::count_draw();
        if indirect != 0 {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!("已丢弃间接网格任务"));
            return;
        }
        with_state(this, |list, s| s.dispatch_mesh(list, x));
    }
    pub(super) extern "system" fn draw_indexed(
        this: This,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        super::count_draw();
        with_state(this, |list, s| {
            s.draw_indexed(list, indices, instances, first, base, first_instance)
        });
    }
}

/// Without a host GPU every draw state call is ignored.
#[cfg(not(windows))]
mod state {
    pub(super) use super::dispatch;
    pub(super) use super::draw;
    pub(super) use super::draw as draw_indexed;
    pub(super) use super::draw as dispatch_mesh;
    pub(super) use super::draw as dispatch_mesh_x;
    pub(super) use super::ignored_state as blend_factor;
    pub(super) use super::ignored_state as clear_unordered_float;
    pub(super) use super::ignored_state as clear_unordered_uint;
    pub(super) use super::ignored_state as compute_constant;
    pub(super) use super::ignored_state as compute_constants;
    pub(super) use super::ignored_state as compute_table;
    pub(super) use super::ignored_state as compute_view;
    pub(super) use super::ignored_state as graphics_constant;
    pub(super) use super::ignored_state as graphics_constants;
    pub(super) use super::ignored_state as graphics_table;
    pub(super) use super::ignored_state as graphics_view;
    pub(super) use super::ignored_state as heaps;
    pub(super) use super::ignored_state as index_buffer;
    pub(super) use super::ignored_state as scissors;
    pub(super) use super::ignored_state as stencil_ref;
    pub(super) use super::ignored_state as targets;
    pub(super) use super::ignored_state as topology;
    pub(super) use super::ignored_state as vertex_buffers;
    pub(super) use super::ignored_state as viewports;
}

fn count_draw() {
    DRAWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Draws of the kinds the host does not run yet (indirect, mesh), and
/// dispatches, counted per frame for the log.
/// shortcut: dropped until their shaders translate (decision 0007).
#[cfg(not(windows))]
extern "system" fn draw(_: This) {
    count_draw();
}

/// `ExecuteIndirect(signature, max count, arguments, offset, count buffer,
/// offset)`. Grand Theft Auto VI's signatures (title+0x2479466, 2026-11-24)
/// hold one argument each: draw (stride 16), indexed draw (20), dispatch
/// (12), mesh dispatch (12), and the console's single-pass dispatch, type
/// 105 (stride 12: the three group counts; its flags live in the
/// signature, the title passes none), run as a plain dispatch. They run
/// through the host's indirect execution (`State::execute_indirect`).
/// shortcut: signatures that change root arguments are dropped (logged
/// once per signature).
extern "system" fn execute_indirect(
    this: This,
    signature: This,
    max: u32,
    arguments: This,
    offset: u64,
    count: This,
    count_offset: u64,
) {
    count_draw();
    let Some((stride, args)) = crate::signature::layout(signature) else {
        return;
    };
    let address = crate::resource::buffer_bytes(arguments).map(|(p, _)| p as u64 + offset);
    let counter = crate::resource::buffer_bytes(count).map(|(p, _)| p as u64 + count_offset);
    let mut ran = false;
    #[cfg(windows)]
    if let ([[kind, ..]], Some(address)) = (args, address)
        && (count.is_null() || counter.is_some())
    {
        const DISPATCH: u32 = 2;
        const DISPATCH_SINGLE: u32 = 105;
        let kind = if *kind == DISPATCH_SINGLE { DISPATCH } else { *kind };
        with_state(this, |list, s| {
            ran = s.execute_indirect(list, (kind, stride, max), address, counter);
        });
    } else {
        #[cfg(all(windows, feature = "diagnostics"))]
        crate::pipeline::note_why(format!(
            "indirect signature {args:?}: arguments in a buffer {}, count in a buffer {}",
            address.is_some(),
            count.is_null() || counter.is_some()
        ));
    }
    if ran {
        return;
    }
    INDIRECT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    static SEEN: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&(signature as usize)) {
        seen.push(signature as usize);
        tracing::info!(
            stride,
            ?args,
            max,
            address = ?address.map(|a| format!("{a:#x}")),
            counted = !count.is_null(),
            "已丢弃间接执行（首次遇到此命令签名）"
        );
    }
}

/// Indirect executions dropped since the last call.
pub(crate) fn take_indirect() -> u64 {
    INDIRECT.swap(0, std::sync::atomic::Ordering::Relaxed)
}
static INDIRECT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
fn count_dispatch() {
    DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}
#[cfg(not(windows))]
extern "system" fn dispatch(_: This) {
    count_dispatch();
}
static DRAWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DISPATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Draws (any kind, mesh and indirect included) and compute dispatches
/// recorded since the last call.
pub(crate) fn take_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (DRAWS.swap(0, Relaxed), DISPATCHES.swap(0, Relaxed))
}

/// A GPU wait for a memory value `(address, comparison, reference, ...)`.
/// Lists run one after another and to completion, so a value another queue
/// or the CPU has to write first cannot be waited for; nothing to do.
extern "system" fn wait_value(_: This, _address: u64, _comparison: u32, _reference: u64) {}

/// The enhanced barriers `(group count, groups)`: buffer, texture and
/// global barriers with layouts instead of states. Every host use of a
/// title texture (clears, copies) moves its host copy to the state it needs
/// first, so only their ordering matters: one barrier on every shader
/// write (the title's buffer barriers order compute passes).
extern "system" fn barrier_groups(this: This, count: u32, _groups: *const u8) {
    if count > 0 {
        wait_for_shader_writes(this);
    }
}

/// The bottom-of-pipe value writes `(address, value, flush, flags)`, 32 and
/// 64 bits: done when the list has run, as the GPU would. Flag 1 writes the
/// GPU timestamp instead (64-bit form), flag 2 the system timestamp.
extern "system" fn write32(this: This, address: u64, value: u32, _flush: u32, _flags: u32) {
    #[cfg(all(windows, feature = "paged-title"))]
    {
        record_pipe_write(this, address, (_flush == 0 && _flags == 0)
            .then_some(crate::page_queue::PipeWriteValue::Word(value)));
        return;
    }
    #[cfg(not(all(windows, feature = "paged-title")))]
    defer_write(this, address, Written::Value(u64::from(value)), 4);
}
extern "system" fn write64(this: This, address: u64, value: u64, _flush: u32, flags: u32) {
    const GPU_TIMESTAMP: u32 = 1;
    #[cfg(all(windows, feature = "paged-title"))]
    {
        let _ = value;
        record_pipe_write(this, address, (_flush == 0 && flags == GPU_TIMESTAMP)
            .then_some(crate::page_queue::PipeWriteValue::GpuTimestamp));
    }
    #[cfg(not(all(windows, feature = "paged-title")))]
    {
        const SYSTEM_TIMESTAMP: u32 = 2;
        let what = match flags {
            GPU_TIMESTAMP => Written::GpuTimestamp,
            SYSTEM_TIMESTAMP => Written::SystemTimestamp,
            _ => Written::Value(value),
        };
        defer_write(this, address, what, 8);
    }
}

#[cfg(all(windows, feature = "paged-title"))]
fn record_pipe_write(this: This, address: u64, value: Option<crate::page_queue::PipeWriteValue>) {
    if let Some(list) = host_list(this) {
        if let Some(op) = value.and_then(|value| crate::page_queue::PipeWrite::capture(address, value)) {
            list.record_pipe_write(op);
        } else {
            list.mark_unlowered();
        }
    }
}

#[cfg(not(all(windows, feature = "paged-title")))]
fn defer_write(this: This, address: u64, what: Written, bytes: usize) {
    if address == 0 {
        return;
    }
    #[cfg(windows)]
    if let Some(list) = host_list(this) {
        list.defer_write(address as *mut u8, what, bytes);
        return;
    }
    let _ = this;
    // Without a host GPU nothing runs later: write now.
    what.write(address as *mut u8, bytes);
}

/// The list methods by their slot in older titles' lists (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, list 1, graphics list 96
/// (59 as on PC, then console ones), then versions 1-6 (6 + 1 + 1 + 11 + 3
/// + 13): 140 slots.
const METHODS: [(usize, *const ()); 52] = [
    (0, list_query as *const ()),
    (2, list_release as *const ()),
    (7, object::parent_device as *const ()),      // 0x38
    (8, list_type as *const ()),                  // 0x40
    (9, close as *const ()),                      // 0x48
    (10, reset as *const ()),                     // 0x50
    (12, state::draw as *const ()),               // 0x60 instanced
    (13, state::draw_indexed as *const ()),       // 0x68 indexed
    (14, state::dispatch as *const ()),           // 0x70
    (15, copy_buffer_region as *const ()),        // 0x78
    (16, copy_texture_region as *const ()),       // 0x80
    (17, copy_resource as *const ()),             // 0x88
    (19, resolve as *const ()),                   // 0x98
    (20, state::topology as *const ()),           // 0xa0
    (21, state::viewports as *const ()),          // 0xa8
    (22, state::scissors as *const ()),           // 0xb0
    (23, state::blend_factor as *const ()),       // 0xb8
    (24, state::stencil_ref as *const ()),        // 0xc0
    (25, set_pipeline as *const ()),              // 0xc8
    (26, barrier as *const ()),                   // 0xd0
    (32, state::heaps as *const ()),              // 0x100
    (33, ignored_state as *const ()),             // 0x108 compute root signature
    (34, ignored_state as *const ()),             // 0x110 graphics root signature
    (35, state::compute_table as *const ()),      // 0x118
    (36, state::graphics_table as *const ()),     // 0x120
    (37, state::compute_constant as *const ()),   // 0x128
    (38, state::graphics_constant as *const ()),  // 0x130
    (39, state::compute_constants as *const ()),  // 0x138
    (40, state::graphics_constants as *const ()), // 0x140
    (41, state::compute_view as *const ()),       // 0x148 CBV
    (42, state::graphics_view as *const ()),      // 0x150 CBV
    (43, state::compute_view as *const ()),       // 0x158 SRV
    (44, state::graphics_view as *const ()),      // 0x160 SRV
    (45, state::compute_view as *const ()),       // 0x168 UAV
    (46, state::graphics_view as *const ()),      // 0x170 UAV
    (47, state::index_buffer as *const ()),       // 0x178
    (48, state::vertex_buffers as *const ()),     // 0x180
    (50, state::targets as *const ()),            // 0x190
    (51, clear_dsv as *const ()),                 // 0x198
    (52, clear_rtv as *const ()),                 // 0x1a0
    (59, execute_indirect as *const ()),          // 0x1d8 indirect
    (60, copy_memory as *const ()),               // 0x1e0
    (68, ignored_state as *const ()),             // 0x220 depth bounds
    (69, write32 as *const ()),                   // 0x228
    (70, write64 as *const ()),                   // 0x230
    (71, wait_value as *const ()),                // 0x238, 32 bits
    (72, wait_value as *const ()),                // 0x240, 64 bits
    (73, ignored_state as *const ()),             // 0x248 pixel shader depth order
    (83, copy_texture_region_x as *const ()),     // 0x298
    (84, copy_resource_x as *const ()),           // 0x2a0
    (124, ignored_state as *const ()),            // 0x3e0 shading rate
    (127, state::dispatch_mesh as *const ()),     // 0x3f8 mesh dispatch
];
/// The rest of [`METHODS`], kept apart only to stay readable.
const MORE_METHODS: [(usize, *const ()); 12] = [
    (53, state::clear_unordered_uint as *const ()), // 0x1a8
    (54, state::clear_unordered_float as *const ()), // 0x1b0
    (55, discard_resource as *const ()),            // 0x1b8 DiscardResource
    (28, begin_query as *const ()),                  // 0xe0 BeginQuery
    (29, end_query as *const ()),                  // 0xe8 EndQuery
    (30, resolve_queries as *const ()),             // 0xf0
    (78, dispatch_x as *const ()),                  // 0x270
    (98, state::dispatch_mesh_x as *const ()),      // 0x310
    (99, driver_hint as *const ()),                 // 0x318
    // Pipeline syncs at the top and bottom of the pipe, 0x408 and 0x410.
    (129, sync as *const ()),
    (130, sync as *const ()),
    (137, reset_x as *const ()), // 0x448
];

/// Newer titles' lists have two more methods in the console part of the
/// base list, after the clears (0x198/0x1a0 as before) and before
/// ExecuteIndirect, which moved from 0x1d8 to 0x1e8: every slot from 0x1d8
/// on is two later (taken here as starting exactly at 0x1d8). Inferred from
/// Grand Theft Auto VI's call shapes: ExecuteIndirect 0x1e8, bottom-of-pipe
/// writes 0x238/0x240, waits 0x248/0x250, depth bounds 0x230, shading rate
/// 0x3f0/0x400, syncs 0x418/0x420, ResetX 0x458 (asserts on failure).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 59 { slot + 2 } else { slot }
}

/// Methods only newer titles' lists have. 0x490: the enhanced barrier
/// `(group count, groups)`, with 16-byte groups `(type, count, barriers)`
/// and 40-byte buffer barriers (title+0x95e0 builds them). 0x470 `(u64)`,
/// called from Grand Theft Auto VI's state flush with its result unused.
/// shortcut: 0x470's purpose is unknown; ignored until a title shows it
/// matters.
const NEWER_METHODS: [(usize, *const ()); 3] = [
    (142, ignored_state as *const ()),
    (146, barrier_groups as *const ()),
    (148, allocate_upload as *const ()),
];

const fn list_table(newer: bool) -> [*const (); 160] {
    let mut f = object::d3d12_object(object::missing_table::<LIST, 160>());
    let mut i = 0;
    while i < METHODS.len() + MORE_METHODS.len() {
        let (slot, method) = if i < METHODS.len() {
            METHODS[i]
        } else {
            MORE_METHODS[i - METHODS.len()]
        };
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    let mut i = 0;
    while newer && i < NEWER_METHODS.len() {
        f[NEWER_METHODS[i].0] = NEWER_METHODS[i].1;
        i += 1;
    }
    f
}

/// Every slot up to the limit exists and logs, for the methods newer
/// titles add.
static LIST_FUNCTIONS: Table<160> = Table(list_table(false));
static NEWER_LIST_FUNCTIONS: Table<160> = Table(list_table(true));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use tracing::{Event, Metadata, Subscriber, span};

    struct PipelineBinds(Arc<AtomicUsize>);

    impl Subscriber for PipelineBinds {
        fn enabled(&self, metadata: &Metadata<'_>) -> bool {
            metadata.is_event()
                && metadata.target() == "victory_gpu::pipeline"
                && metadata.fields().field("pipeline").is_some()
                && metadata.fields().field("frame").is_some()
        }

        fn new_span(&self, _: &span::Attributes<'_>) -> span::Id {
            unreachable!()
        }

        fn record(&self, _: &span::Id, _: &span::Record<'_>) {}
        fn record_follows_from(&self, _: &span::Id, _: &span::Id) {}
        fn enter(&self, _: &span::Id) {}
        fn exit(&self, _: &span::Id) {}

        fn event(&self, event: &Event<'_>) {
            if self.enabled(event.metadata()) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    #[test]
    fn lists_bind_the_initial_pipeline_on_creation() {
        use crate::pipeline::{ComputeHead, create_compute};

        // A synthetic compute description matching the public ABI.
        #[repr(C)]
        struct ComputeDesc {
            root: This,
            code: *const u8,
            size: usize,
        }

        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let desc = ListDesc {
            kind: 0,
            flags: 0,
            nodes: 0,
            barrier_batch: 0,
        };
        let bytes = [0xa5, 0xb6, 0xc7];
        let compute = ComputeDesc {
            root: null,
            code: bytes.as_ptr(),
            size: bytes.len(),
        };
        let compute = std::ptr::from_ref(&compute).cast::<ComputeHead>();
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        let binds = Arc::new(AtomicUsize::new(0));
        let subscriber = tracing::Dispatch::new(PipelineBinds(binds.clone()));

        tracing::dispatcher::with_default(&subscriber, || {
            for console in [false, true] {
                let create = |pipeline, out| {
                    if console {
                        create_list_x(null, &desc, allocator, pipeline, &LIST_IIDS[1], out)
                    } else {
                        create_list(null, 0, 0, allocator, pipeline, &LIST_IIDS[1], out)
                    }
                };
                let mut pipeline = null;
                let mut list = null;
                assert_eq!(
                    create_compute(null, compute, &PAGEABLE_IIDS[0], &mut pipeline),
                    S_OK
                );
                let before = binds.load(Ordering::Relaxed);

                assert_eq!(create(null, &mut list), S_OK);
                assert_eq!(binds.load(Ordering::Relaxed), before);
                assert_eq!(release(list), 0);

                assert_eq!(create(pipeline, &mut list), S_OK);
                assert_eq!(binds.load(Ordering::Relaxed), before + 1);
                assert_eq!(release(list), 0);
                assert_eq!(release(pipeline), 0);
            }
        });
        assert_eq!(release(allocator), 0);
    }

    /// ClearUnorderedAccessViewUint/Float and DiscardResource come before
    /// the methods newer lists insert: the AMD upscaler calls them there.
    #[test]
    fn unordered_access_clears_and_discard_have_the_same_slots_in_both_layouts() {
        for table in [&LIST_FUNCTIONS, &NEWER_LIST_FUNCTIONS] {
            assert_eq!(table.0[0x1a8 / 8], state::clear_unordered_uint as *const ());
            assert_eq!(
                table.0[0x1b0 / 8],
                state::clear_unordered_float as *const ()
            );
            assert_eq!(table.0[0x1b8 / 8], discard_resource as *const ());
        }
    }

    /// Mesh programs read only the x group id, so a mesh dispatch runs its
    /// x groups; one with no groups in any dimension runs nothing.
    #[cfg(windows)]
    #[test]
    fn mesh_dispatches_run_x_groups_unless_empty() {
        use state::mesh_groups;
        assert_eq!(mesh_groups(5, 1, 1), Some(5));
        assert_eq!(mesh_groups(0, 1, 1), None);
        assert_eq!(mesh_groups(5, 0, 1), None);
        assert_eq!(mesh_groups(5, 1, 0), None);
        assert_eq!(mesh_groups(5, 2, 3), Some(5));
        assert_eq!(mesh_groups(5, u32::MAX, u32::MAX), Some(5));
    }

    #[test]
    fn every_inline_command_falls_back_to_the_table() {
        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let mut list = null;
        let hr = create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list);
        assert_eq!(hr, S_OK);
        // What the title's inline draw checks: put < draw limit.
        // SAFETY: the list is 0x560 bytes.
        let (put, draw_limit, api_limit) = unsafe {
            let at = |o: usize| list.cast::<u8>().add(o).cast::<u64>().read();
            (at(0x18), at(0x28), at(0x30))
        };
        assert!(put >= draw_limit && put >= api_limit);
        // The console's ID3D12GraphicsCommandList6, as Grand Theft Auto VI asks.
        let mut same = null;
        assert_eq!(list_query(list, &LIST_IIDS[7], &mut same), S_OK);
        assert_eq!(same, list);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(list, 0x10)(list), 1);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(list, 0x40)(list), 0);
        assert_eq!(
            slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list),
            S_OK
        );
        // List-owned upload memory: aligned, writable, below 2^40 (where
        // shaders find title memory), gone at reset.
        #[cfg(not(all(windows, feature = "paged-title")))]
        {
            let p = allocate_upload(list, 40, 0x100);
            assert!(!p.is_null() && (p as usize).is_multiple_of(0x100));
            #[cfg(windows)]
            assert!((1 << 32..1 << 40).contains(&(p as usize)), "{p:p}");
            // SAFETY: 40 bytes were handed out.
            unsafe { p.write_bytes(0xab, 40) };
        }
        assert!(allocate_upload(list, 8, 0x30).is_null(), "not a power of two");
        assert_eq!(reset(list, allocator, null), S_OK);
        // SAFETY: the list is ours.
        assert!(unsafe { &(*list.cast::<List>()).uploads }.lock().unwrap().is_empty());
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(list), 0);
        assert_eq!(release(allocator), 0);
    }

    /// An offset near `u64::MAX` must not overflow and write outside the
    /// query destination buffer.
    #[test]
    fn resolve_queries_rejects_overflowing_offset() {
        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let mut list = null;
        assert_eq!(
            create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list),
            S_OK
        );
        let buffer = crate::resource::test_buffer(64);
        // Used to overflow `offset + count * 8` (panic in debug, wrap in release).
        resolve_queries(list, null, 0, 0, 2, buffer, u64::MAX - 8);
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        for object in [buffer, list, allocator] {
            assert_eq!(release(object), 0);
        }
    }

    fn non_host_list(newer: bool) -> This {
        let table = if newer {
            &NEWER_LIST_FUNCTIONS
        } else {
            &LIST_FUNCTIONS
        };
        Box::into_raw(Box::new(List {
            object: Object::new(table),
            kind: 0,
            put: [0; 5],
            reserved: [0; 64],
            graphics_root_headers: [0; 128],
            compute_root_headers: [0; 128],
            pipeline_shadow: [0; 8],
            newer: [0; 256],
            uploads: Default::default(),
            #[cfg(windows)]
            host: None,
            #[cfg(windows)]
            state: Default::default(),
        }))
        .cast()
    }

    #[test]
    fn sync_is_safe_on_null_and_non_host_lists() {
        type SyncFn = extern "system" fn(This, u32, u32, u64, u64);
        type BarrierGroupsFn = extern "system" fn(This, u32, *const u8);

        let base = list_table(false);
        assert_eq!(base[129], sync as *const ());
        assert_eq!(base[130], sync as *const ());
        let newer = list_table(true);
        assert_eq!(newer[131], sync as *const ());
        assert_eq!(newer[132], sync as *const ());
        assert_eq!(newer[146], barrier_groups as *const ());

        sync(std::ptr::null(), 1, 2, 0x1000, 64);
        barrier_groups(std::ptr::null(), 1, std::ptr::null());

        for (is_newer, slots) in [(false, [129usize, 130]), (true, [131usize, 132])] {
            let list = non_host_list(is_newer);
            for slot_index in slots {
                let offset = slot_index * size_of::<*const ()>();
                slot::<SyncFn>(list, offset)(list, 0, 0, 0, 0);
                slot::<SyncFn>(list, offset)(list, 0x3, 0x1, 0xdead_beef, 256);
            }
            if is_newer {
                let offset = 146 * size_of::<*const ()>();
                slot::<BarrierGroupsFn>(list, offset)(list, 0, std::ptr::null());
                slot::<BarrierGroupsFn>(list, offset)(list, 2, std::ptr::null());
            }
            assert_eq!(slot::<extern "system" fn(This) -> u32>(list, 0x10)(list), 0);
        }
    }

    #[cfg(windows)]
    #[test]
    fn sync_orders_dependent_compute_uav_dispatches() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::Win32::Graphics::Direct3D12::{
            D3D_ROOT_SIGNATURE_VERSION_1_0, D3D12_COMPUTE_PIPELINE_STATE_DESC,
            D3D12_ROOT_CONSTANTS, D3D12_ROOT_DESCRIPTOR, D3D12_ROOT_PARAMETER,
            D3D12_ROOT_PARAMETER_0, D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            D3D12_ROOT_PARAMETER_TYPE_UAV, D3D12_ROOT_SIGNATURE_DESC,
            D3D12_ROOT_SIGNATURE_FLAG_NONE, D3D12_SHADER_BYTECODE, D3D12_SHADER_VISIBILITY_ALL,
            D3D12SerializeRootSignature, ID3D12PipelineState, ID3D12RootSignature,
        };
        use windows::core::s;

        let Some(host) = crate::host::host() else {
            return;
        };

        const SOURCE: &str = r#"
cbuffer Params : register(b0) { uint count; uint stage; };
RWStructuredBuffer<uint> buf_a : register(u0);
RWStructuredBuffer<uint> buf_b : register(u1);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= count) return;
    if (stage == 0u) {
        uint v = id.x ^ 0xA5A50001u;
        for (uint k = 0u; k < 32u; ++k) {
            v = v * 1664525u + 1013904223u;
        }
        buf_a[id.x] = v;
    } else if (stage == 1u) {
        uint rev = count - 1u - id.x;
        uint v = buf_a[rev];
        buf_b[id.x] = v + 1u;
    } else if (stage == 2u) {
        uint rev = count - 1u - id.x;
        uint v = buf_b[rev];
        buf_a[id.x] = v * 3u;
    } else {
        uint rev = count - 1u - id.x;
        uint v = buf_a[rev];
        buf_b[id.x] = v ^ 0x55AA55AAu;
    }
}
"#;

        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: valid static HLSL source and entry-point/target names.
        let compiled = unsafe {
            D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                s!("sync_ordering_test"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                Some(&mut errors),
            )
        };
        if let Err(e) = compiled {
            let msg = errors
                .map(|b| {
                    // SAFETY: the compiler error blob owns its bytes.
                    let slice = unsafe {
                        std::slice::from_raw_parts(
                            b.GetBufferPointer().cast::<u8>(),
                            b.GetBufferSize(),
                        )
                    };
                    String::from_utf8_lossy(slice).into_owned()
                })
                .unwrap_or_default();
            panic!("D3DCompile failed ({e}): {msg}");
        }
        let code = code.expect("compiled shader blob");

        let params = [
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Num32BitValues: 2,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR {
                        ShaderRegister: 1,
                        RegisterSpace: 0,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
        ];
        let rs_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: params.len() as u32,
            pParameters: params.as_ptr(),
            NumStaticSamplers: 0,
            pStaticSamplers: std::ptr::null(),
            Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
        };
        let mut rs_blob: Option<ID3DBlob> = None;
        let mut rs_errors: Option<ID3DBlob> = None;
        // SAFETY: `rs_desc` points to valid `params` array for the duration of the call.
        unsafe {
            D3D12SerializeRootSignature(
                &rs_desc,
                D3D_ROOT_SIGNATURE_VERSION_1_0,
                &mut rs_blob,
                Some(&mut rs_errors),
            )
            .expect("serialize root signature");
        }
        let rs_blob = rs_blob.expect("root signature blob");

        // SAFETY: `rs_blob` and `code` own valid serialized root signature and DXBC bytes.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            let rs_bytes = std::slice::from_raw_parts(
                rs_blob.GetBufferPointer().cast::<u8>(),
                rs_blob.GetBufferSize(),
            );
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, rs_bytes).unwrap();
            let pso = host
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

        const COUNT: u32 = 65_536;
        const GROUPS: u32 = COUNT / 64;
        let bytes = u64::from(COUNT) * 4;
        let (buf_a, ptr_a) = host.shared_buffer(bytes).expect("shared buffer A");
        let (buf_b, ptr_b) = host.shared_buffer(bytes).expect("shared buffer B");
        // SAFETY: `ptr_a` and `ptr_b` point to `bytes` mapped bytes each.
        unsafe {
            ptr_a.write_bytes(0, bytes as usize);
            ptr_b.write_bytes(0, bytes as usize);
        }
        // SAFETY: live D3D12 buffer resources.
        let (gpu_a, gpu_b) =
            unsafe { (buf_a.GetGPUVirtualAddress(), buf_b.GetGPUVirtualAddress()) };

        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let mut list = null;
        assert_eq!(
            create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list),
            S_OK
        );
        let hl = host_list(list).expect("host list present when host GPU exists");

        // `device::tests::newer_device_levels_are_the_same_device` may flip
        // `device::newer()` in parallel, so inspect this list's own vtable.
        // SAFETY: `list` starts with `Object` whose first pointer is its vtable.
        let is_newer = unsafe {
            std::ptr::eq(
                *list.cast::<*const *const ()>(),
                NEWER_LIST_FUNCTIONS.0.as_ptr(),
            )
        };
        let (top_slot, bottom_slot) = if is_newer {
            (newer_slot(129), newer_slot(130))
        } else {
            (129, 130)
        };

        type SyncFn = extern "system" fn(This, u32, u32, u64, u64);
        let sync_top = slot::<SyncFn>(list, top_slot * size_of::<*const ()>());
        let sync_bottom = slot::<SyncFn>(list, bottom_slot * size_of::<*const ()>());

        // SAFETY: `hl.recording()` is an open list and `root`, `pso`, `buf_a`, `buf_b` outlive execution.
        unsafe {
            hl.recording().SetComputeRootSignature(&root);
            hl.recording().SetPipelineState(&pso);
            hl.recording().SetComputeRootUnorderedAccessView(1, gpu_a);
            hl.recording().SetComputeRootUnorderedAccessView(2, gpu_b);

            let p0 = [COUNT, 0u32];
            hl.recording()
                .SetComputeRoot32BitConstants(0, 2, p0.as_ptr().cast(), 0);
            hl.recording().Dispatch(GROUPS, 1, 1);

            sync_top(list, 0, 0, 0, 0);

            let p1 = [COUNT, 1u32];
            hl.recording()
                .SetComputeRoot32BitConstants(0, 2, p1.as_ptr().cast(), 0);
            hl.recording().Dispatch(GROUPS, 1, 1);

            sync_bottom(list, 0x1, 0x2, 0x1000, 64);

            let p2 = [COUNT, 2u32];
            hl.recording()
                .SetComputeRoot32BitConstants(0, 2, p2.as_ptr().cast(), 0);
            hl.recording().Dispatch(GROUPS, 1, 1);

            // `count == 0` is a no-op; `count > 0` orders pass 3 after pass 2.
            barrier_groups(list, 0, std::ptr::null());
            barrier_groups(list, 1, std::ptr::null());

            let p3 = [COUNT, 3u32];
            hl.recording()
                .SetComputeRoot32BitConstants(0, 2, p3.as_ptr().cast(), 0);
            hl.recording().Dispatch(GROUPS, 1, 1);
        }

        assert_eq!(
            slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list),
            S_OK
        );
        host.execute(&[hl]);

        // SAFETY: `host.execute` waited for GPU completion; both mappings cover `COUNT` u32s.
        let (out_a, out_b) = unsafe {
            (
                std::slice::from_raw_parts(ptr_a.cast::<u32>(), COUNT as usize),
                std::slice::from_raw_parts(ptr_b.cast::<u32>(), COUNT as usize),
            )
        };
        for i in 0..COUNT {
            let mut v = i ^ 0xA5A5_0001;
            for _ in 0..32 {
                v = v.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            }
            let expected_a = v.wrapping_add(1).wrapping_mul(3);
            assert_eq!(
                out_a[i as usize], expected_a,
                "ordering mismatch in buf_a at {i}"
            );

            let rev = COUNT - 1 - i;
            let mut v_rev = rev ^ 0xA5A5_0001;
            for _ in 0..32 {
                v_rev = v_rev.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            }
            let expected_b = v_rev.wrapping_add(1).wrapping_mul(3) ^ 0x55AA_55AA;
            assert_eq!(
                out_b[i as usize], expected_b,
                "ordering mismatch in buf_b at {i}"
            );
        }

        assert_eq!(reset(list, allocator, null), S_OK);
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(list), 0);
        assert_eq!(release(allocator), 0);
    }
}
