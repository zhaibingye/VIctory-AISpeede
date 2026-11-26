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
    versions: BTreeMap<usize, Version>,
    states: BTreeMap<usize, D3D12_RESOURCE_STATES>,
}

#[derive(Default)]
pub(crate) struct ResourcePlan {
    inherited: SubmittedTail,
    pending: BTreeMap<usize, (Arc<HostTexture>, Version)>,
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

    fn version(&self, copy: &Arc<HostTexture>) -> Option<Version> {
        let key = Self::key(copy);
        self.pending.get(&key).map(|(_, version)| *version)
            .or_else(|| self.inherited.versions.get(&key).copied())
            .or_else(|| copy.native_authority.lock().unwrap_or_else(|e| e.into_inner())
                .as_ref().map(|complete| complete.version))
    }

    pub(crate) fn authoritative(&self, copy: &Arc<HostTexture>) -> bool {
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
            self.pending.insert(Self::key(copy), (copy.clone(), version));
            return Ok(());
        }
        let _ = (host, list, batch);
        Err(BatchError::TargetNotInitialized)
    }

    pub(crate) fn record_write(&mut self, copy: &Arc<HostTexture>) -> Result<(), BatchError> {
        let version = Version::successor(
            self.version(copy), Representation::NativeGpu, copy.byte_observation(),
        ).ok_or_else(|| BatchError::Native(windows::Win32::Foundation::E_FAIL.into()))?;
        self.pending.insert(Self::key(copy), (copy.clone(), version));
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

    pub(crate) fn commit(self, submitted: Fence, observed: Fence) {
        assert!(observed.0 >= submitted.0, "resource authority requires a completed fence");
        for (copy, state) in self.state.into_values() {
            copy.state.store(state.0 as u32, Ordering::Release);
        }
        for (copy, mut version) in self.pending.into_values() {
            // An alias or pointer can escape after recording the write.
            version.observation = copy.byte_observation();
            *copy.native_authority.lock().unwrap_or_else(|e| e.into_inner()) =
                version.complete(submitted, observed);
            // Existing consumers still use the native-content flag.
            copy.host_written.store(true, Ordering::Release);
        }
    }
}

/// Include possible compatible ranges before page admission. An earlier
/// full overwrite may supply native authority before this operation.
pub(crate) fn collect_compatible_source(
    copy: &Arc<HostTexture>, ranges: &mut Vec<(u64, u64)>,
) -> Result<(), BatchError> {
    #[cfg(feature = "paged-title")]
    if let Some(source) = copy.compatible_texture_source() {
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

fn single_color(d: D3D12_RESOURCE_DESC) -> bool {
    d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        && d.MipLevels == 1 && d.DepthOrArraySize == 1
        && d.SampleDesc.Count == 1 && d.SampleDesc.Quality == 0
        && !d.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
}

/// An exact region from a placed linear source into a native target.
/// Partial uploads consume existing target authority; full ones establish it.
pub(crate) struct TextureUpload {
    source: crate::page_texture::LinearTexture,
    target: Arc<HostTexture>,
    offset: (u32, u32),
    fills_target: bool,
}

impl TextureUpload {
    #[cfg(feature = "paged-title")]
    pub(crate) fn preserves_color(&self) -> Option<&Arc<HostTexture>> {
        (!self.fills_target).then_some(&self.target)
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn target(&self) -> &Arc<HostTexture> { &self.target }

    pub(crate) fn capture(
        source: crate::page_texture::LinearTexture, target: Arc<HostTexture>,
        (x, y): (u32, u32), bounds: Option<[u32; 6]>,
    ) -> Option<Self> {
        let source = source.crop(bounds)?;
        let end_x = u64::from(x.checked_add(source.width)?);
        let end_y = y.checked_add(source.height)?;
        // SAFETY: the retained resource is live.
        let desc = unsafe { target.texture.GetDesc() };
        if !single_color(desc) || !desc.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)
            || target.sparse.is_some()
            || target.twin.get().is_some_and(Option::is_some)
            || end_x > desc.Width || end_y > desc.Height
            || desc.Format != source.format
        {
            return None;
        }
        let fills_target = x == 0 && y == 0 && end_x == desc.Width && end_y == desc.Height;
        Some(Self { source, target, offset: (x, y), fills_target })
    }

    pub(crate) fn ranges(&self, ranges: &mut Vec<(u64, u64)>) -> Result<(), BatchError> {
        ranges.extend(self.source.rows()?);
        Ok(())
    }

    pub(crate) fn encode(&self, host: &crate::host::Host, list: &HostList,
        batch: &crate::page_slabs::PinnedBatch, plan: &mut ResourcePlan,
    ) -> Result<(), BatchError> {
        if !self.fills_target {
            plan.require_native(host, list, batch, &self.target)?;
        }
        plan.transition_texture(list, &self.target, D3D12_RESOURCE_STATE_COPY_DEST.0 as u32)?;
        self.source.copy_to(host, list, batch, &self.target.texture, self.offset)?;
        plan.record_write(&self.target)?;
        Ok(())
    }
}

/// A full texture-to-buffer copy ending in pinned writable pages.
pub(crate) struct TextureReadback {
    source: Arc<HostTexture>,
    target: crate::page_texture::LinearTexture,
}

impl TextureReadback {
    pub(crate) fn capture(source: Arc<HostTexture>, target: crate::page_texture::LinearTexture) -> Option<Self> {
        // SAFETY: the retained resource is live.
        let desc = unsafe { source.texture.GetDesc() };
        if !single_color(desc)
            || desc.Flags.0 & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0) == 0
            || source.sparse.is_some()
            || source.twin.get().is_some_and(Option::is_some)
            || desc.Width != u64::from(target.width) || desc.Height != target.height
            || desc.Format != target.format
        {
            return None;
        }
        Some(Self { source, target })
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn publish_mirror(&self, ranges: &[(u64, u64)], backing: &crate::title_backing::SubmissionBacking) {
        self.source.publish_byte_mirror(ranges, backing.pins());
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
        if !plan.authoritative(&self.source) {
            return Err(BatchError::TargetNotInitialized);
        }
        let before = plan.state(&self.source);
        plan.transition_texture(list, &self.source, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32)?;
        self.target.copy_from(host, list, batch, &self.source.texture, dirty)?;
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
    pub(crate) fn target(&self) -> &Arc<HostTexture> { &self.dst }

    pub(crate) fn capture(src: Arc<HostTexture>, dst: Arc<HostTexture>) -> Option<Self> {
        Self::capture_copy(src, dst, None)
    }

    pub(crate) fn capture_region(
        src: Arc<HostTexture>, dst: Arc<HostTexture>, x: u32, y: u32, bounds: Option<[u32; 6]>,
    ) -> Option<Self> {
        Self::capture_copy(src, dst, Some((x, y, bounds)))
    }

    fn capture_copy(
        src: Arc<HostTexture>, dst: Arc<HostTexture>, region: Option<(u32, u32, Option<[u32; 6]>)>,
    ) -> Option<Self> {
        // SAFETY: the references retain both native resources.
        let (a, b) = unsafe { (src.texture.GetDesc(), dst.texture.GetDesc()) };
        if Arc::ptr_eq(&src, &dst) || src.texture == dst.texture
            || !single_color(a) || !single_color(b)
            || !a.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)
            || !b.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)
            || a.Format != b.Format
            || src.sparse.is_some() || dst.sparse.is_some()
            || src.twin.get().is_some_and(Option::is_some)
            || dst.twin.get().is_some_and(Option::is_some)
        {
            return None;
        }
        let area = if let Some((x, y, bounds)) = region {
            // Single-plane ordinary color formats only: block and video
            // layout rules do not follow these texel bounds.
            use windows::Win32::Graphics::Dxgi::Common::*;
            if !matches!(a.Format, DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
                | DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB
                | DXGI_FORMAT_R16G16B16A16_FLOAT | DXGI_FORMAT_R10G10B10A2_UNORM)
            {
                return None;
            }
            let width = u32::try_from(a.Width).ok()?;
            let [left, top, front, right, bottom, back] = bounds.unwrap_or([0, 0, 0, width, a.Height, 1]);
            if front != 0 || back != 1 || left >= right || top >= bottom
                || right > width || bottom > a.Height
            {
                return None;
            }
            let end_x = x.checked_add(right - left)?;
            let end_y = y.checked_add(bottom - top)?;
            if u64::from(end_x) > b.Width || end_y > b.Height {
                return None;
            }
            Some(CopyArea {
                x, y,
                source: D3D12_BOX { left, top, front, right, bottom, back },
                fills_destination: x == 0 && y == 0 && u64::from(end_x) == b.Width && end_y == b.Height,
            })
        } else {
            if a.Width != b.Width || a.Height != b.Height {
                return None;
            }
            None
        };
        Some(Self { src, dst, area })
    }

    pub(crate) fn ranges(&self, ranges: &mut Vec<(u64, u64)>) -> Result<(), BatchError> {
        collect_compatible_source(&self.src, ranges)
    }

    pub(crate) fn encode(
        &self, host: &crate::host::Host, list: &HostList,
        batch: &crate::page_slabs::PinnedBatch, plan: &mut ResourcePlan,
    ) -> Result<(), BatchError> {
        plan.require_native(host, list, batch, &self.src)?;
        if self.area.as_ref().is_some_and(|area| !area.fills_destination) {
            plan.require_native(host, list, batch, &self.dst)?;
        }
        plan.transition_texture(list, &self.src, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32)?;
        plan.transition_texture(list, &self.dst, D3D12_RESOURCE_STATE_COPY_DEST.0 as u32)?;
        list.keep(&self.src.texture);
        list.keep(&self.dst.texture);
        // SAFETY: both resources stay alive in COPY_SOURCE/DEST. A region
        // uses two borrowed subresource zero locations and checked texel bounds.
        unsafe {
            let cmd = list.recording();
            if let Some(area) = &self.area {
                let location = |texture: &ID3D12Resource| D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(texture),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                };
                cmd.CopyTextureRegion(&location(&self.dst.texture), area.x, area.y, 0,
                    &location(&self.src.texture), Some(&area.source));
            } else {
                cmd.CopyResource(&self.dst.texture, &self.src.texture);
            }
        }
        plan.record_write(&self.dst)?;
        Ok(())
    }
}
