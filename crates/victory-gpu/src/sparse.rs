//! Private on-demand backing for sampled-only, single-slice BC1 textures.
//! Existing views keep the original resource and subresource numbering.
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::{Acquire, Relaxed, Release}};
use std::sync::{Mutex, OnceLock};
use windows::Win32::Graphics::Direct3D12::*;

/// Reserved backing for eligible hidden detailed mips; opt out for A/B.
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SPARSE_BC1").is_none_or(|v| v != "0"))
}

pub(crate) fn wholly_absent(
    mut at: usize,
    end: usize,
    mut query: impl FnMut(usize) -> Option<(usize, bool)>,
) -> bool {
    if at >= end {
        return false;
    }
    while at < end {
        let Some((next, true)) = query(at) else { return false };
        if next <= at {
            return false;
        }
        at = next;
    }
    true
}

#[cfg(test)]
#[test]
fn sparse_extent_requires_all_pages_to_be_absent() {
    let query = |at| match at {
        0..4096 => Some((4096, true)),
        4096..8192 => Some((8192, true)),
        8192..12288 => Some((12288, false)),
        _ => None,
    };
    assert!(wholly_absent(4095, 8192, query));
    assert!(!wholly_absent(4095, 8193, query));
    assert!(!wholly_absent(12288, 12289, query));
    assert!(!wholly_absent(8192, 8192, query));
    assert!(!wholly_absent(5, 8, |at| Some((at, true))));
}

// Physical backing owned by living sparse HostTextures. The driver's
// residency budget remains authoritative; these counts describe our heaps.
static MAPPED: AtomicU64 = AtomicU64::new(0);
static USED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn counts() -> (u64, u64) {
    (MAPPED.load(Relaxed), USED.load(Relaxed))
}

pub(crate) struct Sparse {
    // Overall starting tile for each standard subresource, then the packed tail and/or end.
    starts: Vec<u32>,
    first: AtomicU32,
    refused: AtomicBool,
    used: AtomicBool,
    change: Mutex<()>,
}
impl Sparse {
    /// On an unpublished resource or the host's direct queue, add mappings.
    /// Newly added tiles are never reused or removed while the resource lives.
    fn map(texture: &ID3D12Resource, first: u32, count: u32) -> windows::core::Result<()> {
        let host = crate::host::host().ok_or(windows::core::Error::empty())?;
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: u64::from(count) * 65536,
            Properties: D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                CreationNodeMask: 1,
                VisibleNodeMask: 1,
                ..Default::default()
            },
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
            ..Default::default()
        };
        let mut heap: Option<ID3D12Heap> = None;
        // SAFETY: The heap description is initialized; the COM out slot and texture are live.
        unsafe {
            host.device.CreateHeap(&desc, &mut heap)?;
            let heap = heap.ok_or(windows::core::Error::empty())?;
            // The resource's COM reference count (including descriptor heaps)
            // governs these backing heaps' lifetime, even beyond the title handle.
            let key = windows::core::GUID::from_u128(
                0x0128e665_6b89_412d_a55e_12d744300000 + u128::from(first),
            );
            texture.SetPrivateDataInterface(&key, &heap)?;
            host.map_sparse(texture, first, &heap, count);
        }
        Ok(())
    }

    pub(crate) fn try_new(
        mut desc: D3D12_RESOURCE_DESC,
        state: D3D12_RESOURCE_STATES,
        limit: u32,
    ) -> Option<(ID3D12Resource, Sparse)> {
        if limit == 0 {
            return None;
        }
        let host = crate::host::host()?;
        static SUPPORTED: OnceLock<bool> = OnceLock::new();
        if !*SUPPORTED.get_or_init(|| {
            let mut opt = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
            // SAFETY: The feature ID and byte count match the writable options structure.
            unsafe {
                host.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS,
                        (&raw mut opt).cast(),
                        size_of_val(&opt) as u32,
                    )
                    .is_ok()
                    && opt.TiledResourcesTier.0 >= 2
            }
        }) {
            return None;
        }
        // SAFETY: The one initialized description is borrowed for this allocation query.
        let full_committed_bytes = unsafe {
            host.device
                .GetResourceAllocationInfo(0, &[desc])
                .SizeInBytes
        };
        if full_committed_bytes == 0 || full_committed_bytes == u64::MAX {
            return None;
        }
        desc.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
        let mut texture = None;
        // SAFETY: The descriptor and writable COM out slot are live; no clear value is supplied.
        unsafe {
            host.device
                .CreateReservedResource(&desc, state, None, &mut texture)
                .ok()?;
        }
        let texture: ID3D12Resource = texture?;
        let mut total = 0;
        let mut packed = D3D12_PACKED_MIP_INFO::default();
        let mut num = u32::from(desc.MipLevels);
        let mut locations = vec![D3D12_SUBRESOURCE_TILING::default(); num as usize];
        // SAFETY: The texture is live and num is initialized to the capacity of locations.
        unsafe {
            host.device.GetResourceTiling(
                &texture,
                Some(&mut total),
                Some(&mut packed),
                None,
                Some(&mut num),
                0,
                locations.as_mut_ptr(),
            );
        }
        let standard = usize::from(packed.NumStandardMips);
        if standard > locations.len()
            || standard > num as usize
            || total == 0
            || standard + usize::from(packed.NumPackedMips) != usize::from(desc.MipLevels)
        {
            return None;
        }
        let mut starts: Vec<_> = locations[..standard]
            .iter()
            .map(|t| t.StartTileIndexInOverallResource)
            .collect();
        if packed.NumPackedMips > 0 {
            if packed.NumTilesForPackedMips == 0 || packed
                .StartTileIndexInOverallResource
                .checked_add(packed.NumTilesForPackedMips)
                != Some(total)
            {
                return None;
            }
            starts.push(packed.StartTileIndexInOverallResource);
        }
        starts.push(total);
        if starts.first() != Some(&0) || starts.windows(2).any(|s| s[0] >= s[1]) {
            return None;
        }
        // We issue one linear mapping per mip prefix. Require each
        // standard subresource to cover exactly its reported interval.
        for (i, t) in locations[..standard].iter().enumerate() {
            let tiles = t.WidthInTiles
                .checked_mul(u32::from(t.HeightInTiles))?
                .checked_mul(u32::from(t.DepthInTiles))?;
            if tiles != starts[i + 1] - starts[i] {
                return None;
            }
        }
        let first = limit
            .min(u32::from(packed.NumStandardMips))
            .min(u32::from(desc.MipLevels) - 1);
        if first == 0 {
            return None;
        }
        let skipped = starts[first as usize];
        // Never select a device/shape with larger capacity after full promotion,
        // even when its initial tail would be cheap.
        if u64::from(total) * 65536 > full_committed_bytes {
            return None;
        }
        Self::map(&texture, first, total - skipped).ok()?;
        MAPPED.fetch_add(u64::from(total - skipped) * 65536, Relaxed);
        Some((
            texture,
            Self {
                starts,
                first: AtomicU32::new(first),
                refused: AtomicBool::new(false),
                used: AtomicBool::new(false),
                change: Mutex::new(()),
            },
        ))
    }

    pub(crate) fn first(&self) -> u32 {
        self.first.load(Acquire)
    }

    fn mapped_bytes(&self) -> u64 {
        u64::from(self.starts.last().unwrap() - self.starts[self.first() as usize]) * 65536
    }

    pub(crate) fn note_used(&self) {
        if !self.used.load(Relaxed) {
            let _guard = self.change.lock().unwrap_or_else(|e| e.into_inner());
            if !self.used.swap(true, Relaxed) {
                USED.fetch_add(self.mapped_bytes(), Relaxed);
            }
        }
    }

    pub(crate) fn promote(&self, texture: &ID3D12Resource, first: u32) -> bool {
        if first >= self.first() {
            return true;
        }
        let _guard = self.change.lock().unwrap_or_else(|e| e.into_inner());
        let previous = self.first();
        if first >= previous {
            return true;
        }
        if self.refused.load(Relaxed) {
            return false;
        }
        let count = self.starts[previous as usize] - self.starts[first as usize];
        #[cfg(test)]
        let result = if FAIL_PROMOTION.with(|fail| fail.replace(false)) {
            Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_OUTOFMEMORY,
            ))
        } else {
            Self::map(texture, first, count)
        };
        #[cfg(not(test))]
        let result = Self::map(texture, first, count);
        if let Err(e) = result {
            if let Some(host) = crate::host::host()
                // SAFETY: This only queries the live host device for its removal reason.
                && unsafe { host.device.GetDeviceRemovedReason() }.is_err()
            {
                crate::host::fail(format_args!(
                    "device lost during sparse texture promotion: {e}"
                ));
            }
            // Like a refused committed texture, do not retry an allocation
            // every draw. This generation can still serve all resident mips.
            // A replacement after normal retirement uses its widest known view.
            self.refused.store(true, Relaxed);
            tracing::warn!(error=%e, "cannot grow sparse texture; keeping its resident mip range");
            return false;
        }
        let bytes = u64::from(count) * 65536;
        MAPPED.fetch_add(bytes, Relaxed);
        if self.used.load(Relaxed) {
            USED.fetch_add(bytes, Relaxed);
        }
        self.first.store(first, Release);
        true
    }
}
impl Drop for Sparse {
    fn drop(&mut self) {
        let bytes = self.mapped_bytes();
        MAPPED.fetch_sub(bytes, Relaxed);
        if *self.used.get_mut() {
            USED.fetch_sub(bytes, Relaxed);
        }
    }
}

/// A placed 2D texture has one slice; a later view may use the one-slice
/// array spelling. Both views must select mapped mips before publication.
pub(crate) fn resident_srv(
    sparse: &Sparse,
    texture: &ID3D12Resource,
    desc: &mut D3D12_SHADER_RESOURCE_VIEW_DESC,
    mips: u32,
) -> Option<()> {
    // SAFETY: each union member is selected only for that view dimension.
    let (first, count, lod, plane) = unsafe {
        match desc.ViewDimension {
            D3D12_SRV_DIMENSION_TEXTURE2D => {
                let d = &mut desc.Anonymous.Texture2D;
                (&mut d.MostDetailedMip, &mut d.MipLevels, &mut d.ResourceMinLODClamp, d.PlaneSlice)
            }
            D3D12_SRV_DIMENSION_TEXTURE2DARRAY => {
                let d = &mut desc.Anonymous.Texture2DArray;
                if d.FirstArraySlice != 0 || d.ArraySize != 1 {
                    return None;
                }
                (&mut d.MostDetailedMip, &mut d.MipLevels, &mut d.ResourceMinLODClamp, d.PlaneSlice)
            }
            _ => return None,
        }
    };
    let end = if *count == u32::MAX { mips } else { first.checked_add(*count)? };
    if *first >= end || end > mips || plane != 0 {
        return None;
    }
    if !sparse.promote(texture, *first) {
        let (new_first, new_count, new_lod) = resident_window(*first, end, sparse.first(), *lod)?;
        (*first, *count, *lod) = (new_first, new_count, new_lod);
    }
    Some(())
}

fn resident_window(first: u32, end: u32, resident: u32, lod: f32) -> Option<(u32, u32, f32)> {
    let kept = first.max(resident);
    if kept >= end {
        return None;
    }
    // The clamp is relative to the view's most detailed mip. Keep its
    // effective resource LOD where it was rather than applying the shift twice.
    Some((kept, end - kept, (lod - (kept - first) as f32).max(0.0)))
}

#[cfg(test)]
#[test]
fn sparse_window_never_exposes_unmapped_mips_or_extends_the_view() {
    assert_eq!(resident_window(0, 8, 2, 3.5), Some((2, 6, 1.5)));
    assert_eq!(resident_window(1, 6, 3, 0.5), Some((3, 3, 0.0)));
    assert_eq!(resident_window(3, 6, 2, 1.25), Some((3, 3, 1.25)));
    assert_eq!(resident_window(0, 2, 2, 0.0), None);
}

#[cfg(test)]
thread_local! { pub(crate) static FAIL_PROMOTION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }

#[cfg(all(test, feature = "diagnostics"))]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_BC1_UNORM, DXGI_SAMPLE_DESC};

    /// No game or hardware adapter: make a reserved BC1 texture and inject
    /// one failed allocation before publishing a wider view.
    #[test]
    #[ignore = "run alone with VICTORY_D3D12_WARP=1"]
    fn sparse_bc1_promotion_keeps_resource_identity_and_last_good_range() {
        assert!(crate::switch::on("VICTORY_D3D12_WARP"), "use the WARP software adapter");
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 1024,
            Height: 1024,
            DepthOrArraySize: 1,
            MipLevels: 8,
            Format: DXGI_FORMAT_BC1_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            ..Default::default()
        };
        let Some((prior, range)) = Sparse::try_new(desc, D3D12_RESOURCE_STATE_COMMON, 3) else {
            panic!("WARP did not expose a supported reserved BC1 layout");
        };
        use windows::core::Interface;
        let id = prior.as_raw();
        let first = range.first();
        assert!(first > 0);
        FAIL_PROMOTION.with(|fail| fail.set(true));
        assert!(!range.promote(&prior, 0));
        assert_eq!(range.first(), first);
        assert!(range.promote(&prior, first), "the previously resident range is still valid");
        assert_eq!(prior.as_raw(), id);
    }
    #[test]
    #[ignore = "run alone with VICTORY_D3D12_WARP=1 and sparse BC1 enabled"]
    fn placed_bc1_uses_sparse_backing_but_normal_resources_do_not_share_it() {
        use crate::descriptor::{View, host::placed_texture};
        use crate::object::This;
        use windows_sys::Win32::System::Memory::*;

        assert!(crate::switch::on("VICTORY_D3D12_WARP") && enabled());
        // The placed-resource map keeps this reservation until test exit.
        let base = unsafe {
            let base = VirtualAlloc(std::ptr::null_mut(), 8 << 20, MEM_RESERVE, PAGE_NOACCESS);
            assert!(!base.is_null());
            assert_eq!(VirtualAlloc(base, 4096, MEM_COMMIT, PAGE_READWRITE), base);
            base as usize
        };
        let desc = [3u32, 0, 0, 0, 2048, 0, 2048, 0x3_0001, 71, 1, 0, 0, 0, 0];
        let mut view = [0u32; 16];
        (view[0], view[1], view[2], view[4], view[5], view[10], view[11]) = (
            71, 4, 5768, 2, 1, base as u32, (base as u64 >> 32) as u32,
        );
        let placed = || {
            let Some(View::Shader(object, _)) = placed_texture(&View::Placed(false, desc, view)) else {
                panic!("missing placed BC1 view");
            };
            crate::resource::host_copy(object as This).expect("host copy")
        };
        let old = placed();
        assert!(old.sparse.as_ref().unwrap().first() > 0);

        // Creating a normal resource at the same address gets conventional
        // backing. It must not accidentally reuse the sparse allocation.
        let mut title: This = std::ptr::null();
        let iid = crate::object::IUNKNOWN;
        // SAFETY: the descriptor record has the console repr(C) layout.
        let aligned = unsafe { desc.as_ptr().cast::<crate::resource::ResourceDesc>().read_unaligned() };
        let hr = crate::resource::create_at_address(
            std::ptr::null_mut(),
            base as u64,
            &aligned,
            0,
            std::ptr::null(),
            &iid,
            &mut title,
        );
        assert_eq!(hr, 0);
        let normal = crate::resource::host_copy(title).expect("ordinary host copy");
        assert!(normal.sparse.is_none());
        assert_ne!(old.texture, normal.texture);
        assert_eq!(placed().texture, old.texture, "the cached placed view keeps its host identity");
    }

}
