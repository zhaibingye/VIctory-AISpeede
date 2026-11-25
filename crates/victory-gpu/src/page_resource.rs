//! Submission-local attachment authority and resource transitions.
//! The global resource ledger is changed only after successful execution.
use std::collections::BTreeMap;
use std::sync::{Arc, atomic::Ordering};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, legal_state, transition};
use crate::page_slabs::BatchError;
use crate::resource::HostTexture;

#[derive(Default)]
pub(crate) struct ResourcePlan {
    initialized: BTreeMap<usize, Arc<HostTexture>>,
    state: BTreeMap<usize, (Arc<HostTexture>, D3D12_RESOURCE_STATES)>,
}

impl ResourcePlan {
    fn key(copy: &Arc<HostTexture>) -> usize {
        Arc::as_ptr(copy) as usize
    }

    pub(crate) fn authoritative(&self, copy: &Arc<HostTexture>) -> bool {
        copy.host_written.load(Ordering::Acquire) || self.initialized.contains_key(&Self::key(copy))
    }

    pub(crate) fn initialize(&mut self, copy: &Arc<HostTexture>) {
        self.initialized.insert(Self::key(copy), copy.clone());
    }

    pub(crate) fn state(&self, copy: &Arc<HostTexture>) -> D3D12_RESOURCE_STATES {
        self.state.get(&Self::key(copy)).map_or_else(
            || D3D12_RESOURCE_STATES(copy.state.load(Ordering::Acquire) as i32),
            |(_, state)| *state,
        )
    }

    fn transition_texture(
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

    pub(crate) fn commit(self) {
        for (copy, state) in self.state.into_values() {
            copy.state.store(state.0 as u32, Ordering::Release);
        }
        for copy in self.initialized.into_values() {
            copy.host_written.store(true, Ordering::Release);
        }
    }
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
        && d.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)
        && !d.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
}

/// Full replacement from a placed linear view into a native title
/// texture. It shares the queue's authority and transition ledger.
pub(crate) struct TextureUpload {
    source: crate::page_texture::LinearTexture,
    target: Arc<HostTexture>,
}

impl TextureUpload {
    pub(crate) fn capture(source: crate::page_texture::LinearTexture, target: Arc<HostTexture>) -> Option<Self> {
        // SAFETY: the retained resource is live.
        let desc = unsafe { target.texture.GetDesc() };
        if !single_color(desc) || target.sparse.is_some()
            || target.twin.get().is_some_and(Option::is_some)
            || desc.Width != u64::from(source.width) || desc.Height != source.height
            || desc.Format != source.format
        {
            return None;
        }
        Some(Self { source, target })
    }

    pub(crate) fn ranges(&self, ranges: &mut Vec<(u64, u64)>) -> Result<(), BatchError> {
        ranges.extend(self.source.rows()?);
        Ok(())
    }

    pub(crate) fn encode(&self, host: &crate::host::Host, list: &HostList,
        batch: &crate::page_slabs::PinnedBatch, plan: &mut ResourcePlan,
    ) -> Result<(), BatchError> {
        plan.transition_texture(list, &self.target, D3D12_RESOURCE_STATE_COPY_DEST.0 as u32)?;
        self.source.copy_to(host, list, batch, &self.target.texture)?;
        plan.initialize(&self.target);
        Ok(())
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

    pub(crate) fn encode(&self, list: &HostList, plan: &mut ResourcePlan) -> Result<(), BatchError> {
        if !plan.authoritative(&self.src)
            || self.area.as_ref().is_some_and(|area| !area.fills_destination && !plan.authoritative(&self.dst))
        {
            return Err(BatchError::TargetNotInitialized);
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
        plan.initialize(&self.dst);
        Ok(())
    }
}
