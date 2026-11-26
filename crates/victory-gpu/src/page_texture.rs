//! Read-only derived native textures, built from current captured GPU
//! pages rather than an independent CPU shadow. Only explicit linear
//! layouts are accepted; swizzled/native layouts require their own decode.
use victory_memory::gpu_table::ADDRESS_LIMIT;
use windows::Win32::Foundation::{E_FAIL, E_OUTOFMEMORY};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Error;

use crate::host::{Host, HostList, transition};
use crate::page_slabs::{BatchError, PinnedBatch};

/// One admitted shader view. A native-capable view can use exact byte
/// backing before its first GPU write; selection follows ordered authority.
pub(crate) enum TextureInput {
    Linear {
        copy: std::sync::Arc<crate::resource::HostTexture>,
        input: LinearTexture,
    },
    Native {
        copy: std::sync::Arc<crate::resource::HostTexture>,
        format: DXGI_FORMAT,
        exact_backing: bool,
        compatible_input: Option<LinearTexture>,
    },
}

const EXTENDED_COLOR: [DXGI_FORMAT; 6] = [
    DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R11G11B10_FLOAT,
    DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
];

fn extended_color_support(host: &Host, format: DXGI_FORMAT) -> bool {
    static SUPPORT: [std::sync::OnceLock<bool>; EXTENDED_COLOR.len()] =
        [const { std::sync::OnceLock::new() }; EXTENDED_COLOR.len()];
    let Some(index) = EXTENDED_COLOR.iter().position(|&f| f == format) else { return true; };
    // There is one process host adapter. Avoid querying the same native
    // format for each draw while still checking optional sampling support.
    *SUPPORT[index].get_or_init(|| {
        let mut caps = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: format, ..Default::default() };
        // SAFETY: the feature query fills this local structure.
        unsafe { host.device.CheckFeatureSupport(
            D3D12_FEATURE_FORMAT_SUPPORT, (&raw mut caps).cast(), size_of_val(&caps) as u32,
        ) }.is_ok()
            && caps.Support1.contains(D3D12_FORMAT_SUPPORT1_SHADER_LOAD | D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE)
    })
}

impl TextureInput {
    pub(crate) fn native(
        copy: std::sync::Arc<crate::resource::HostTexture>, format: DXGI_FORMAT,
        compatible_input: Option<LinearTexture>,
    ) -> Option<Self> {
        // SAFETY: the resource is retained by the view.
        let desc = unsafe { copy.texture.GetDesc() };
        let basic = matches!(format, DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM);
        let extended = EXTENDED_COLOR.contains(&format);
        if !(basic || extended)
            || extended && copy.uses_unproven_alias_backing()
            || desc.Format != format || desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
            || desc.DepthOrArraySize != 1 || desc.MipLevels != 1
            || desc.SampleDesc.Count != 1 || desc.SampleDesc.Quality != 0
            || desc.Flags.0 & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0) == 0
            || desc.Flags.0 & (D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0) != 0
            || copy.sparse.is_some() || copy.twin.get().is_some_and(Option::is_some)
        {
            return None;
        }
        let compatible_input = compatible_input.filter(|input| {
            input.format == format && u64::from(input.width) == desc.Width
                && input.height == desc.Height && !copy.uses_unproven_alias_backing()
        });
        let exact_backing = extended || compatible_input.is_some();
        Some(Self::Native { copy, format, exact_backing, compatible_input })
    }

    pub(crate) fn native_depth(
        copy: std::sync::Arc<crate::resource::HostTexture>, format: DXGI_FORMAT,
    ) -> Option<Self> {
        // SAFETY: retained native resource, so the description is live.
        let desc = unsafe { copy.texture.GetDesc() };
        if !matches!((desc.Format, format),
            (DXGI_FORMAT_R16_TYPELESS, DXGI_FORMAT_R16_UNORM)
            | (DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32_FLOAT)
            | (DXGI_FORMAT_R24G8_TYPELESS, DXGI_FORMAT_R24_UNORM_X8_TYPELESS)
            | (DXGI_FORMAT_R32G8X24_TYPELESS, DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS))
            || desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
            || desc.DepthOrArraySize != 1 || desc.MipLevels != 1
            || desc.SampleDesc.Count != 1 || desc.SampleDesc.Quality != 0
            || !desc.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
            || desc.Flags.contains(D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)
            || copy.sparse.is_some() || copy.twin.get().is_some_and(Option::is_some)
            || copy.uses_unproven_alias_backing()
        {
            return None;
        }
        Some(Self::Native { copy, format, exact_backing: true, compatible_input: None })
    }

    pub(crate) fn ranges(&self, ranges: &mut Vec<(u64, u64)>) -> Result<(), BatchError> {
        match self {
            Self::Linear { input, .. } => ranges.extend(input.rows()?),
            Self::Native { copy, compatible_input: Some(input), .. } => {
                // A queued earlier operation can establish native
                // authority, but it cannot create a newer CPU version.
                if !crate::page_resource::ResourcePlan::default().authoritative(copy) {
                    ranges.extend(input.rows()?);
                }
            }
            Self::Native { copy, .. } => {
                crate::page_resource::collect_compatible_source(copy, ranges)?;
            }
        }
        Ok(())
    }

    pub(crate) fn aliases_target(&self, target: &crate::resource::HostTexture) -> bool {
        let copy = match self {
            Self::Linear { copy, .. } | Self::Native { copy, .. } => copy,
        };
        copy.texture == target.texture
    }

    pub(crate) fn encode(
        &self, host: &Host, list: &HostList, batch: &PinnedBatch,
        plan: &mut crate::page_resource::ResourcePlan,
    ) -> Result<(ID3D12Resource, DXGI_FORMAT), BatchError> {
        match self {
            Self::Linear { copy, input } => {
                if plan.authoritative(copy) {
                    return Err(BatchError::UnsupportedTextureMap);
                }
                Ok((input.encode(host, list, batch)?, input.format))
            }
            Self::Native { copy, format, exact_backing, compatible_input } => {
                if *exact_backing && (copy.uses_unproven_alias_backing()
                    || copy.twin.get().is_some_and(Option::is_some))
                {
                    return Err(BatchError::UnsupportedTextureMap);
                }
                if let Some(input) = compatible_input
                    && !plan.authoritative(copy)
                {
                    return Ok((input.encode(host, list, batch)?, input.format));
                }
                if !extended_color_support(host, *format) {
                    return Err(BatchError::UnsupportedTextureMap);
                }
                plan.require_native(host, list, batch, copy)?;
                let state = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                plan.transition_texture(list, copy, state.0 as u32)?;
                list.keep(&copy.texture);
                Ok((copy.texture.clone(), *format))
            }
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct LinearTexture {
    pub(crate) address: u64,
    pub(crate) row_pitch: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) format: DXGI_FORMAT,
}

impl LinearTexture {
    pub(crate) fn rows(self) -> Result<impl Iterator<Item = (u64, u64)>, BatchError> {
        let bytes = u64::from(self.width) * 4;
        let pitch = u64::from(self.row_pitch);
        let end = self.address.saturating_add(pitch * u64::from(self.height.saturating_sub(1)))
            .saturating_add(bytes);
        if !matches!(self.format, DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM)
            || self.address == 0 || self.width == 0 || self.height == 0
            || self.width > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION
            || self.height > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION
            || bytes > pitch || end > ADDRESS_LIMIT
        {
            return Err(BatchError::Range(self.address, end));
        }
        Ok((0..u64::from(self.height)).map(move |row| {
            let start = self.address + row * pitch;
            (start, start + bytes)
        }))
    }

    pub(crate) fn crop(self, bounds: Option<[u32; 6]>) -> Option<Self> {
        let [left, top, front, right, bottom, back] =
            bounds.unwrap_or([0, 0, 0, self.width, self.height, 1]);
        if front != 0 || back != 1 || left >= right || top >= bottom
            || right > self.width || bottom > self.height
        {
            return None;
        }
        let address = self.address.checked_add(u64::from(top) * u64::from(self.row_pitch))?
            .checked_add(u64::from(left) * 4)?;
        Some(Self { address, width: right - left, height: bottom - top, ..self })
    }

    fn native_desc(self) -> D3D12_RESOURCE_DESC {
        D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(self.width),
            Height: self.height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: self.format,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        }
    }

    pub(crate) fn encode(
        self, host: &Host, list: &HostList, batch: &PinnedBatch,
    ) -> Result<ID3D12Resource, BatchError> {
        let _ = self.rows()?;
        let desc = self.native_desc();
        // SAFETY: one validated 2D texture description.
        let alloc = unsafe { host.device.GetResourceAllocationInfo(0, &[desc]) }.SizeInBytes;
        let dest = D3D12_RESOURCE_STATE_COPY_DEST;
        let texture: ID3D12Resource = host.managed_gpu(alloc, |heap| {
            let mut resource = None;
            // SAFETY: checked texture description and live output slot.
            unsafe {
                host.device.CreateCommittedResource(heap, D3D12_HEAP_FLAG_NONE, &desc, dest, None,
                    &mut resource)?;
            }
            resource.ok_or_else(|| Error::from(E_FAIL))
        }).map_err(BatchError::Native)?;
        self.copy_to(host, list, batch, &texture, (0, 0))?;
        // SAFETY: copy_to leaves the retained texture in COPY_DEST.
        unsafe {
            list.recording().ResourceBarrier(&[transition(&texture, dest,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                    | D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        }
        Ok(texture)
    }

    fn native_footprint(self, host: &Host) -> Result<(D3D12_PLACED_SUBRESOURCE_FOOTPRINT, u64), BatchError> {
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let (mut row_count, mut row_bytes, mut total) = (0, 0, 0);
        // SAFETY: staging describes the cropped source, not the target,
        // with one subresource and correctly sized output locations.
        unsafe {
            host.device.GetCopyableFootprints(&self.native_desc(), 0, 1, 0,
                Some(&mut footprint), Some(&mut row_count), Some(&mut row_bytes), Some(&mut total));
        }
        if row_count != self.height || row_bytes != u64::from(self.width) * 4
            || total == u64::MAX || total == 0 || footprint.Footprint.Depth != 1
        {
            return Err(BatchError::Native(E_FAIL.into()));
        }
        Ok((footprint, total))
    }

    /// Keep a native texture-to-linear copy on the GPU so subsequent
    /// commands can observe it before the submission's single CPU merge.
    pub(crate) fn copy_from(
        self, host: &Host, list: &HostList, batch: &PinnedBatch, texture: &ID3D12Resource,
        dirty: &mut std::collections::BTreeSet<victory_memory::physical::PhysicalPage>,
    ) -> Result<(), BatchError> {
        let rows = self.rows()?;
        let (footprint, total) = self.native_footprint(host)?;
        let staging = host.default_buffer(total)
            .ok_or_else(|| BatchError::Native(E_OUTOFMEMORY.into()))?;
        list.keep(texture);
        list.keep(&staging);
        // SAFETY: caller supplies a retained exact native subresource
        // in COPY_SOURCE. The separate budgeted staging buffer is in
        // COPY_DEST and sized for the native footprint.
        unsafe {
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&staging),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
            };
            list.recording().CopyTextureRegion(&to, 0, 0, 0, &from, None);
        }
        let pitch = u64::from(footprint.Footprint.RowPitch);
        crate::page_view::scatter_regions(list, batch, rows.enumerate().map(|(i, range)| {
            (range, footprint.Offset + i as u64 * pitch)
        }), &staging, dirty)
    }

    /// Gather current GPU bytes at the exact point of use. The caller
    /// supplies a matching native subresource zero in COPY_DEST.
    pub(crate) fn copy_to(
        self, host: &Host, list: &HostList, batch: &PinnedBatch, texture: &ID3D12Resource,
        (x, y): (u32, u32),
    ) -> Result<(), BatchError> {
        let rows = self.rows()?;
        // SAFETY: live native resource. Only an exact single subresource
        // is consumed here, regardless of the guest's pitch.
        let desc = unsafe { texture.GetDesc() };
        if x.checked_add(self.width).is_none_or(|end| u64::from(end) > desc.Width)
            || y.checked_add(self.height).is_none_or(|end| end > desc.Height)
            || desc.Format != self.format || desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
            || desc.SampleDesc.Count != 1 || desc.SampleDesc.Quality != 0
            || desc.MipLevels != 1 || desc.DepthOrArraySize != 1
        {
            return Err(BatchError::Native(E_FAIL.into()));
        }
        let (footprint, total) = self.native_footprint(host)?;
        let dest = D3D12_RESOURCE_STATE_COPY_DEST;
        let read = D3D12_RESOURCE_STATE_COPY_SOURCE;
        let staging = host.default_buffer(total)
            .ok_or_else(|| BatchError::Native(E_OUTOFMEMORY.into()))?;
        let pitch = u64::from(footprint.Footprint.RowPitch);
        crate::page_view::gather_regions(list, batch, rows.enumerate().map(|(i, range)| {
            (range, footprint.Offset + i as u64 * pitch)
        }), &staging, dest, read)?;
        list.keep(texture);
        // SAFETY: both resources stay alive in the named states. The
        // placed buffer footprint describes the validated source extent.
        unsafe {
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&staging),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
            };
            let cmd = list.recording();
            cmd.CopyTextureRegion(&to, x, y, 0, &from, None);
        }
        Ok(())
    }
}

#[cfg(test)]
#[test]
fn paged_linear_texture_rejects_unknown_or_short_layouts() {
    let mut tex = LinearTexture { address: 0x10000, row_pitch: 16, width: 2, height: 2,
        format: DXGI_FORMAT_R8G8B8A8_UNORM };
    assert_eq!(tex.rows().unwrap().collect::<Vec<_>>(), [(0x10000, 0x10008), (0x10010, 0x10018)]);
    tex.row_pitch = 7;
    assert!(tex.rows().is_err());
    tex.row_pitch = 16;
    tex.format = DXGI_FORMAT_NV12;
    assert!(tex.rows().is_err());
}

/// One explicit SRV table parameter, not a title bindless heap. The
/// trusted recorder must prove that the root signature matches the table
/// and that the shader uses no other descriptor heap.
pub(crate) struct SampledTexture {
    pub(crate) input: LinearTexture,
    pub(crate) root: u32,
}

pub(crate) struct PreparedSample {
    slot: u32,
    root: u32,
}

impl SampledTexture {
    pub(crate) fn prepare(
        &self, host: &Host, list: &HostList, batch: &PinnedBatch,
        descriptors: &mut crate::page_queue::PageDescriptors<'_>,
    ) -> Result<PreparedSample, BatchError> {
        let texture = self.input.encode(host, list, batch)?;
        let slot = descriptors.reserve_texture(&texture, self.input.format)
            .map_err(BatchError::Native)?;
        Ok(PreparedSample { slot, root: self.root })
    }
}

impl PreparedSample {
    pub(crate) fn bind(
        &self, cmd: &ID3D12GraphicsCommandList, descriptors: &crate::page_queue::PageDescriptors<'_>,
    ) {
        // SAFETY: the batch retains its heap and the texture descriptor;
        // the graphics root signature names a texture table at this root.
        unsafe {
            cmd.SetDescriptorHeaps(&[Some(descriptors.heap.heap.clone())]);
            cmd.SetGraphicsRootDescriptorTable(self.root, descriptors.heap.gpu(self.slot));
        }
    }
}
