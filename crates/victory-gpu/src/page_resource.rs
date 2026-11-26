//! Submission-local attachment authority and resource transitions.
//! The global resource ledger is changed only after successful execution.
use std::collections::BTreeMap;
use std::sync::{Arc, atomic::Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use victory_memory::{authority::{Representation, Version}, coherence::Fence};

use crate::host::{HostList, legal_state, transition};
use crate::page_slabs::BatchError;
use crate::resource::HostTexture;

/// Latest submitted values on the direct queue. Only its current
/// bounded window sees these; guest-visible completion is independent.
#[derive(Clone, Default)]
pub(crate) struct SubmittedTail {
    versions: BTreeMap<(usize, Option<u32>), Version>,
    states: BTreeMap<usize, D3D12_RESOURCE_STATES>,
}

#[derive(Default)]
pub(crate) struct ResourcePlan {
    inherited: SubmittedTail,
    #[cfg(feature = "paged-title")]
    refreshed: BTreeMap<usize, Arc<HostTexture>>,
    #[cfg(feature = "paged-title")]
    snapshot: Option<crate::page_slabs::SnapshotDigests>,
    pending: BTreeMap<(usize, Option<u32>), (Arc<HostTexture>, Version)>,
    state: BTreeMap<usize, (Arc<HostTexture>, D3D12_RESOURCE_STATES)>,
}

impl ResourcePlan {
    pub(crate) fn after(tail: &SubmittedTail) -> Self {
        Self { inherited: tail.clone(), ..Self::default() }
    }

    pub(crate) fn advance_submitted_tail(&self, tail: &mut SubmittedTail) {
        tail.versions.extend(self.pending.iter().map(|(&key, (_, version))| (key, *version)));
        tail.states.extend(self.state.iter().map(|(&key, (_, state))| (key, *state)));
    }

    fn key(copy: &Arc<HostTexture>) -> usize {
        Arc::as_ptr(copy) as usize
    }

    fn version_at(&self, copy: &Arc<HostTexture>, subresource: Option<u32>) -> Option<Version> {
        let key = (Self::key(copy), subresource);
        self.pending.get(&key).map(|(_, version)| *version)
            .or_else(|| self.inherited.versions.get(&key).copied())
            .or_else(|| copy.native_authority.lock().unwrap_or_else(|e| e.into_inner())
                .get(&subresource).map(|complete| complete.version))
    }

    fn version(&self, copy: &Arc<HostTexture>) -> Option<Version> {
        self.version_at(copy, None)
    }

    pub(crate) fn authoritative(&self, copy: &Arc<HostTexture>) -> bool {
        #[cfg(feature = "paged-title")]
        if self.refreshed.contains_key(&Self::key(copy)) {
            return true;
        }
        self.version(copy).map_or_else(
            || copy.host_written.load(Ordering::Acquire),
            |version| version.representation == Representation::NativeGpu,
        )
    }

    /// Convert the same logical version on demand. This copies current
    /// pinned bytes without inventing a new content version.
    pub(crate) fn require_native(
        &mut self, host: &crate::host::Host, list: &HostList,
        batch: &crate::page_slabs::PinnedBatch, copy: &Arc<HostTexture>,
    ) -> Result<(), BatchError> {
        #[cfg(feature = "paged-title")]
        {
            let key = Self::key(copy);
            // A persistent guest pointer can modify compatible backing
            // without Unmap. Before its first GPU consumer in this batch,
            // read current bytes. Never replace writes produced earlier in
            // the same ordered native stream with the old CPU snapshot.
            if copy.has_pointer_exposure()
                && !self.pending.contains_key(&(key, None))
                && !self.inherited.versions.contains_key(&(key, None))
                && !self.refreshed.contains_key(&key)
            {
                let source = copy.visible_linear_writeback()?.ok_or(BatchError::UnsupportedTextureMap)?;
                let unchanged = self.snapshot.as_ref().is_some_and(|snapshot| {
                    copy.native_matches_snapshot(batch, snapshot)
                });
                if !unchanged {
                    self.transition_texture(list, copy, D3D12_RESOURCE_STATE_COPY_DEST.0 as u32)?;
                    source.copy_to(host, list, batch, &copy.texture, (0, 0))?;
                }
                self.refreshed.insert(key, copy.clone());
                return Ok(());
            }
        }
        if self.authoritative(copy) {
            return Ok(());
        }
        #[cfg(feature = "paged-title")]
        if let Some(mut version) = self.version(copy)
            && version.representation == Representation::CompatibleBytes
        {
            let source = copy.compatible_texture_source().ok_or(BatchError::UnsupportedTextureMap)?;
            self.transition_texture(list, copy, D3D12_RESOURCE_STATE_COPY_DEST.0 as u32)?;
            source.copy_to(host, list, batch, &copy.texture, (0, 0))?;
            version.representation = Representation::NativeGpu;
            self.pending.insert((Self::key(copy), None), (copy.clone(), version));
            return Ok(());
        }
        let _ = (host, list, batch);
        Err(BatchError::TargetNotInitialized)
    }

    pub(crate) fn require_color_subresource(
        &mut self, host: &crate::host::Host, list: &HostList, batch: &crate::page_slabs::PinnedBatch,
        copy: &Arc<HostTexture>, sub: ColorSubresource,
    ) -> Result<(), BatchError> {
        self.require_texture_authority(host, list, batch, copy, sub.authority())
    }

    fn require_texture_authority(
        &mut self, host: &crate::host::Host, list: &HostList, batch: &crate::page_slabs::PinnedBatch,
        copy: &Arc<HostTexture>, subresource: Option<u32>,
    ) -> Result<(), BatchError> {
        if subresource.is_none() {
            return self.require_native(host, list, batch, copy);
        }
        if self.version_at(copy, subresource).is_some_and(|v| v.representation == Representation::NativeGpu) {
            Ok(())
        } else {
            Err(BatchError::TargetNotInitialized)
        }
    }

    pub(crate) fn record_write(&mut self, copy: &Arc<HostTexture>) -> Result<(), BatchError> {
        self.record_write_at(copy, None)
    }

    pub(crate) fn record_color_write(&mut self, copy: &Arc<HostTexture>, sub: ColorSubresource) -> Result<(), BatchError> {
        self.record_write_at(copy, sub.authority())
    }

    fn record_write_at(&mut self, copy: &Arc<HostTexture>, subresource: Option<u32>) -> Result<(), BatchError> {
        let version = Version::successor(
            self.version_at(copy, subresource), Representation::NativeGpu, copy.byte_observation(),
        ).ok_or_else(|| BatchError::Native(windows::Win32::Foundation::E_FAIL.into()))?;
        self.pending.insert((Self::key(copy), subresource), (copy.clone(), version));
        Ok(())
    }

    pub(crate) fn state(&self, copy: &Arc<HostTexture>) -> D3D12_RESOURCE_STATES {
        let key = Self::key(copy);
        self.state.get(&key).map(|(_, state)| *state)
            .or_else(|| self.inherited.states.get(&key).copied())
            .unwrap_or_else(|| D3D12_RESOURCE_STATES(copy.state.load(Ordering::Acquire) as i32))
    }

    pub(crate) fn transition_texture(
        &mut self, list: &HostList, copy: &Arc<HostTexture>, wanted: u32,
    ) -> Result<(), BatchError> {
        let writable = 0x4 | 0x8 | 0x10 | 0x400 | 0x1000;
        let write = wanted & writable;
        if copy.sparse.is_some() || copy.twin.get().is_some_and(Option::is_some)
            || wanted & !(writable | 0x20 | 0x40 | 0x80 | 0x800 | 0x2000) != 0
            || write != 0 && (wanted != write || !write.is_power_of_two())
            || legal_state(&copy.texture, wanted) != wanted
        {
            return Err(BatchError::UnsupportedBarrier);
        }
        let before = self.state(copy);
        let after = D3D12_RESOURCE_STATES(wanted as i32);
        if before != after {
            list.keep(&copy.texture);
            // SAFETY: live attachment, ordered submission-local state,
            // and a checked whole-resource transition.
            unsafe { list.recording().ResourceBarrier(&[transition(&copy.texture, before, after)]) };
            self.state.insert(ResourcePlan::key(copy), (copy.clone(), after));
        }
        Ok(())
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn set_snapshot(&mut self, snapshot: crate::page_slabs::SnapshotDigests) {
        self.snapshot = Some(snapshot);
    }

    pub(crate) fn commit(self, submitted: Fence, observed: Fence, batch: &crate::page_slabs::PinnedBatch) {
        assert!(observed.0 >= submitted.0, "resource authority requires a completed fence");
        #[cfg(feature = "paged-title")]
        if let Some(snapshot) = &self.snapshot {
            for (key, copy) in &self.refreshed {
                if !self.pending.contains_key(&(*key, None)) {
                    copy.record_native_snapshot(batch, snapshot);
                }
            }
        }
        let _ = batch;
        for (copy, state) in self.state.into_values() {
            copy.state.store(state.0 as u32, Ordering::Release);
        }
        for ((_, subresource), (copy, mut version)) in self.pending {
            version.observation = copy.byte_observation();
            let complete = version.complete(submitted, observed).expect("fence observed above");
            copy.native_authority.lock().unwrap_or_else(|e| e.into_inner()).insert(subresource, complete);
            if subresource.is_none() {
                // Never advertise a partial native texture as whole.
                copy.host_written.store(true, Ordering::Release);
            }
        }
    }
}

/// Include possible compatible ranges before page admission. An earlier
/// full overwrite may supply native authority before this operation.
pub(crate) fn collect_compatible_source(
    copy: &Arc<HostTexture>, ranges: &mut Vec<(u64, u64)>,
) -> Result<(), BatchError> {
    #[cfg(feature = "paged-title")]
    if !copy.has_pointer_exposure() && let Some(source) = copy.compatible_texture_source() {
        ranges.extend(source.rows()?);
    }
    let _ = (copy, ranges);
    Ok(())
}

pub(crate) enum PageBarrier {
    Texture(Arc<HostTexture>, u32),
    Uav,
}

impl PageBarrier {
    pub(crate) fn encode(
        &self, list: &HostList, resources: &mut ResourcePlan,
    ) -> Result<(), BatchError> {
        match self {
            Self::Texture(copy, wanted) => {
                resources.transition_texture(list, copy, *wanted)?;
            }
            Self::Uav => {
                let barrier = D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                            pResource: std::mem::ManuallyDrop::new(None),
                        }),
                    },
                };
                // SAFETY: a global ordered UAV barrier uses no pointer.
                unsafe { list.recording().ResourceBarrier(&[barrier]) };
            }
        }
        Ok(())
    }
}

/// Plane zero of an exact 2D color texture. State transitions remain
/// whole-resource for now, but byte authority is scoped to the copy.
#[derive(Clone, Copy)]
pub(crate) struct ColorSubresource {
    pub(crate) index: u32,
    pub(crate) width: u64,
    pub(crate) height: u32,
    scoped: bool,
}

impl ColorSubresource {
    pub(crate) fn from_mip_slice(desc: D3D12_RESOURCE_DESC, mip: u32, slice: u32) -> Option<Self> {
        let mips = u32::from(desc.MipLevels);
        if mip >= mips || slice >= u32::from(desc.DepthOrArraySize) { return None; }
        Self::capture(desc, slice.checked_mul(mips)?.checked_add(mip)?)
    }

    pub(crate) fn is_scoped(self) -> bool { self.scoped }

    pub(crate) fn capture(desc: D3D12_RESOURCE_DESC, index: u32) -> Option<Self> {
        if desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1
            || desc.SampleDesc.Quality != 0 || desc.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
        {
            return None;
        }
        let mips = u32::from(desc.MipLevels);
        let total = mips.checked_mul(u32::from(desc.DepthOrArraySize))?;
        if index >= total { return None; }
        let scoped = total != 1;
        if scoped {
            crate::page_texture::linear_texel_bytes(desc.Format)?;
        }
        let mip = index % mips;
        Some(Self {
            index, width: desc.Width.checked_shr(mip)?.max(1),
            height: desc.Height.checked_shr(mip)?.max(1), scoped,
        })
    }

    fn authority(self) -> Option<u32> { self.scoped.then_some(self.index) }

    pub(crate) fn allows(self, copy: &HostTexture) -> bool {
        !self.scoped || (!copy.uses_unproven_alias_backing() && !copy.has_pointer_exposure())
    }
}

/// An exact region from a placed linear source into a native target.
/// Partial uploads consume existing target authority; full ones establish it.
pub(crate) struct TextureUpload {
    source: crate::page_texture::LinearTexture,
    target: Arc<HostTexture>,
    offset: (u32, u32),
    sub: ColorSubresource,
    fills_target: bool,
    exact_backing: bool,
}

impl TextureUpload {
    #[cfg(feature = "paged-title")]
    pub(crate) fn preserves_color(&self) -> Option<&Arc<HostTexture>> {
        (!self.fills_target).then_some(&self.target)
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn target(&self) -> &Arc<HostTexture> { &self.target }

    pub(crate) fn capture(
        source: crate::page_texture::LinearTexture, target: Arc<HostTexture>, index: u32,
        (x, y): (u32, u32), bounds: Option<[u32; 6]>,
    ) -> Option<Self> {
        let source = source.crop(bounds)?;
        let end_x = u64::from(x.checked_add(source.width)?);
        let end_y = y.checked_add(source.height)?;
        // SAFETY: the retained resource is live.
        let desc = unsafe { target.texture.GetDesc() };
        let sub = ColorSubresource::capture(desc, index)?;
        let exact_backing = !desc.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        if (!sub.scoped && desc.Flags.0 & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0) == 0)
            || exact_backing && target.uses_unproven_alias_backing()
            || !sub.allows(&target)
            || target.sparse.is_some()
            || target.twin.get().is_some_and(Option::is_some)
            || end_x > sub.width || end_y > sub.height
            || desc.Format != source.format
        {
            return None;
        }
        let fills_target = x == 0 && y == 0 && end_x == sub.width && end_y == sub.height;
        Some(Self { source, target, offset: (x, y), sub, fills_target, exact_backing })
    }

    pub(crate) fn ranges(&self, ranges: &mut Vec<(u64, u64)>) -> Result<(), BatchError> {
        ranges.extend(self.source.rows()?);
        Ok(())
    }

    pub(crate) fn encode(&self, host: &crate::host::Host, list: &HostList,
        batch: &crate::page_slabs::PinnedBatch, plan: &mut ResourcePlan,
    ) -> Result<(), BatchError> {
        if !self.sub.allows(&self.target) || self.exact_backing && self.target.uses_unproven_alias_backing() {
            return Err(BatchError::UnsupportedTextureMap);
        }
        if !self.fills_target {
            plan.require_color_subresource(host, list, batch, &self.target, self.sub)?;
        }
        plan.transition_texture(list, &self.target, D3D12_RESOURCE_STATE_COPY_DEST.0 as u32)?;
        self.source.copy_to_subresource(host, list, batch, &self.target.texture, self.sub.index, self.offset)?;
        plan.record_write_at(&self.target, self.sub.authority())?;
        Ok(())
    }
}

/// A full texture-to-buffer copy ending in pinned writable pages.
pub(crate) struct TextureReadback {
    source: Arc<HostTexture>,
    target: crate::page_texture::LinearTexture,
    sub: ColorSubresource,
}

impl TextureReadback {
    pub(crate) fn capture(source: Arc<HostTexture>, target: crate::page_texture::LinearTexture) -> Option<Self> {
        Self::capture_subresource(source, 0, target).filter(|readback| !readback.sub.scoped)
    }

    pub(crate) fn capture_subresource(
        source: Arc<HostTexture>, index: u32, target: crate::page_texture::LinearTexture,
    ) -> Option<Self> {
        // SAFETY: the retained resource is live.
        let desc = unsafe { source.texture.GetDesc() };
        let sub = ColorSubresource::capture(desc, index)?;
        if !sub.allows(&source)
            || (!sub.scoped && desc.Flags.0 & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0) == 0)
            || source.sparse.is_some()
            || source.twin.get().is_some_and(Option::is_some)
            || sub.width != u64::from(target.width) || sub.height != target.height
            || desc.Format != target.format
        {
            return None;
        }
        Some(Self { source, target, sub })
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn prepare_mirror(
        &self, ranges: &[(u64, u64)], pins: &victory_memory::physical::BackingPins,
    ) -> Result<crate::resource::PreparedByteMirror, BatchError> {
        self.source.prepare_byte_mirror(ranges, pins)
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn publish_mirror(
        &self, prepared: crate::resource::PreparedByteMirror, slabs: &crate::page_slabs::PageSlabs,
    ) {
        self.source.publish_byte_mirror(prepared, slabs);
    }

    pub(crate) fn ranges(&self, ranges: &mut Vec<(u64, u64)>) -> Result<(), BatchError> {
        ranges.extend(self.target.rows()?);
        Ok(())
    }

    pub(crate) fn encode(
        &self, host: &crate::host::Host, list: &HostList, batch: &crate::page_slabs::PinnedBatch,
        plan: &mut ResourcePlan,
        dirty: &mut std::collections::BTreeSet<victory_memory::physical::PhysicalPage>,
    ) -> Result<(), BatchError> {
        if !self.sub.allows(&self.source) {
            return Err(BatchError::UnsupportedTextureMap);
        }
        if self.sub.scoped {
            plan.require_color_subresource(host, list, batch, &self.source, self.sub)?;
        } else if !plan.authoritative(&self.source) {
            return Err(BatchError::TargetNotInitialized);
        }
        let before = plan.state(&self.source);
        plan.transition_texture(list, &self.source, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32)?;
        self.target.copy_from_subresource(host, list, batch, &self.source.texture, self.sub.index, dirty)?;
        plan.transition_texture(list, &self.source, before.0 as u32)
    }
}

/// An ordered copy between exact, native-authoritative color surfaces.
/// Title-backed uploads, conversion, sparse and twin surfaces are not
/// interchangeable with this operation.
pub(crate) struct TextureCopy {
    src: Arc<HostTexture>,
    dst: Arc<HostTexture>,
    area: Option<CopyArea>,
    exact_backing: bool,
    source: ColorSubresource,
    destination: ColorSubresource,
    subresources: u32,
}

struct CopyArea {
    x: u32,
    y: u32,
    source: D3D12_BOX,
    fills_destination: bool,
}

impl TextureCopy {
    #[cfg(feature = "paged-title")]
    pub(crate) fn preserves_color(&self) -> Option<&Arc<HostTexture>> {
        self.area.as_ref().filter(|area| !area.fills_destination).map(|_| &self.dst)
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn pointer_source(&self) -> Option<&Arc<HostTexture>> {
        self.src.has_pointer_exposure().then_some(&self.src)
    }

    pub(crate) fn target(&self) -> &Arc<HostTexture> { &self.dst }

    pub(crate) fn capture(src: Arc<HostTexture>, dst: Arc<HostTexture>) -> Option<Self> {
        Self::capture_copy(src, 0, dst, 0, None)
    }

    pub(crate) fn capture_region(
        src: Arc<HostTexture>, src_index: u32, dst: Arc<HostTexture>, dst_index: u32,
        x: u32, y: u32, bounds: Option<[u32; 6]>,
    ) -> Option<Self> {
        Self::capture_copy(src, src_index, dst, dst_index, Some((x, y, bounds)))
    }

    fn capture_copy(
        src: Arc<HostTexture>, src_index: u32, dst: Arc<HostTexture>, dst_index: u32,
        region: Option<(u32, u32, Option<[u32; 6]>)>,
    ) -> Option<Self> {
        // SAFETY: the references retain both native resources.
        let (a, b) = unsafe { (src.texture.GetDesc(), dst.texture.GetDesc()) };
        let source = ColorSubresource::capture(a, src_index)?;
        let destination = ColorSubresource::capture(b, dst_index)?;
        let scoped = source.scoped || destination.scoped;
        let writable = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0;
        let exact_backing = scoped || !a.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)
            || !b.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        if Arc::ptr_eq(&src, &dst) || src.texture == dst.texture
            || !source.allows(&src) || !destination.allows(&dst)
            || (!scoped && (a.Flags.0 & writable == 0 || b.Flags.0 & writable == 0))
            || exact_backing && (src.uses_unproven_alias_backing() || dst.uses_unproven_alias_backing())
            || a.Format != b.Format
            || src.sparse.is_some() || dst.sparse.is_some()
            || src.twin.get().is_some_and(Option::is_some)
            || dst.twin.get().is_some_and(Option::is_some)
        {
            return None;
        }
        let (area, subresources) = if let Some((x, y, bounds)) = region {
            // Partial copies use texel bounds only on exact pixel formats.
            crate::page_texture::linear_texel_bytes(a.Format)?;
            let width = u32::try_from(source.width).ok()?;
            let [left, top, front, right, bottom, back] = bounds.unwrap_or([0, 0, 0, width, source.height, 1]);
            if front != 0 || back != 1 || left >= right || top >= bottom
                || right > width || bottom > source.height
            {
                return None;
            }
            let end_x = x.checked_add(right - left)?;
            let end_y = y.checked_add(bottom - top)?;
            if u64::from(end_x) > destination.width || end_y > destination.height {
                return None;
            }
            (Some(CopyArea {
                x, y,
                source: D3D12_BOX { left, top, front, right, bottom, back },
                fills_destination: x == 0 && y == 0 && u64::from(end_x) == destination.width && end_y == destination.height,
            }), 1)
        } else {
            if a.Width != b.Width || a.Height != b.Height
                || a.MipLevels != b.MipLevels || a.DepthOrArraySize != b.DepthOrArraySize
            {
                return None;
            }
            (None, u32::from(a.MipLevels).checked_mul(u32::from(a.DepthOrArraySize))?)
        };
        Some(Self { src, dst, area, exact_backing, source, destination, subresources })
    }

    /// A region carries one pair; a whole copy carries each mip/slice
    /// in native subresource order. Authority stays keyed per subresource.
    fn authorities(&self, sub: ColorSubresource) -> impl Iterator<Item = Option<u32>> {
        (0..self.subresources).map(move |offset| sub.authority().map(|first| first + offset))
    }

    pub(crate) fn ranges(&self, ranges: &mut Vec<(u64, u64)>) -> Result<(), BatchError> {
        collect_compatible_source(&self.src, ranges)
    }

    pub(crate) fn encode(
        &self, host: &crate::host::Host, list: &HostList,
        batch: &crate::page_slabs::PinnedBatch, plan: &mut ResourcePlan,
    ) -> Result<(), BatchError> {
        if !self.source.allows(&self.src) || !self.destination.allows(&self.dst)
            || self.exact_backing && (self.src.uses_unproven_alias_backing() || self.dst.uses_unproven_alias_backing()) {
            return Err(BatchError::UnsupportedTextureMap);
        }
        for authority in self.authorities(self.source) {
            plan.require_texture_authority(host, list, batch, &self.src, authority)?;
        }
        if self.area.as_ref().is_some_and(|area| !area.fills_destination) {
            plan.require_color_subresource(host, list, batch, &self.dst, self.destination)?;
        }
        plan.transition_texture(list, &self.src, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32)?;
        plan.transition_texture(list, &self.dst, D3D12_RESOURCE_STATE_COPY_DEST.0 as u32)?;
        list.keep(&self.src.texture);
        list.keep(&self.dst.texture);
        // SAFETY: both resources stay alive in COPY_SOURCE/DEST. A region
        // uses checked texel bounds and borrows the retained resources.
        // A whole copy uses matching texture shapes and proven source contents.
        unsafe {
            let cmd = list.recording();
            if let Some(area) = &self.area {
                let location = |texture: &ID3D12Resource, index| D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(texture),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: index },
                };
                cmd.CopyTextureRegion(&location(&self.dst.texture, self.destination.index), area.x, area.y, 0,
                    &location(&self.src.texture, self.source.index), Some(&area.source));
            } else {
                cmd.CopyResource(&self.dst.texture, &self.src.texture);
            }
        }
        for authority in self.authorities(self.destination) {
            plan.record_write_at(&self.dst, authority)?;
        }
        Ok(())
    }
}
