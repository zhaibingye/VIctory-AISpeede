//! Read-only derived native textures, built from current captured GPU
//! pages rather than an independent CPU shadow. Only explicit linear
//! layouts are accepted; swizzled/native layouts require their own decode.
use victory_memory::gpu_table::ADDRESS_LIMIT;
use windows::Win32::Foundation::{E_FAIL, E_OUTOFMEMORY};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{Error, Interface};

use crate::host::{Host, HostList, transition};
use crate::page_slabs::{BatchError, PinnedBatch};

/// Exact mip interval expressed by a canonical 2D shader view.
#[derive(Clone, Copy)]
pub(crate) struct MipSelection {
    pub(crate) first: u32,
    pub(crate) count: u32,
}

impl MipSelection {
    pub(crate) const SINGLE: Self = Self { first: 0, count: 1 };

    pub(crate) fn resolve(first: u32, count: u32, total: u32) -> Option<Self> {
        let available = total.checked_sub(first)?;
        let count = if count == u32::MAX { available } else { count };
        (count != 0 && count <= available).then_some(Self { first, count })
    }
}

/// Shader dimension and plane-zero subresource selection. A one-slice
/// array is still an array view and must match the shader operand.
#[derive(Clone, Copy)]
pub(crate) struct TextureSelection {
    pub(crate) mips: MipSelection,
    pub(crate) slices: Option<MipSelection>,
}

impl TextureSelection {
    pub(crate) const SINGLE: Self = Self { mips: MipSelection::SINGLE, slices: None };
}

const SHADER_READ: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATES(
    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0 | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0,
);

/// Temporary per-command states for disjoint native subresource accesses.
/// Duplicate views of a subresource share one transition pair.
#[derive(Default)]
pub(crate) struct ScopedTextureAccess(std::collections::BTreeMap<
    (usize, u32), (ID3D12Resource, crate::page_resource::ColorSubresource, D3D12_RESOURCE_STATES, D3D12_RESOURCE_STATES),
>);

impl ScopedTextureAccess {
    pub(crate) fn insert(&mut self, texture: &ID3D12Resource, sub: crate::page_resource::ColorSubresource,
        before: D3D12_RESOURCE_STATES, wanted: D3D12_RESOURCE_STATES,
    ) -> Result<(), BatchError> {
        if !sub.is_scoped() || crate::host::legal_state(texture, wanted.0 as u32) != wanted.0 as u32 {
            return Err(BatchError::UnsupportedBarrier);
        }
        let (_, _, earlier, planned) = self.0.entry((texture.as_raw() as usize, sub.index))
            .or_insert_with(|| (texture.clone(), sub, before, wanted));
        if *earlier != before || *planned != wanted {
            return Err(BatchError::UnsupportedBarrier);
        }
        Ok(())
    }

    pub(crate) fn transition(&self, list: &ID3D12GraphicsCommandList, restore: bool) {
        if self.0.is_empty() { return; }
        let barriers = self.0.values().filter_map(|(texture, sub, before, wanted)| {
            let (from, to) = if restore { (*wanted, *before) } else { (*before, *wanted) };
            (from != to).then(|| sub.temporary_transition(texture, from, to))
        }).collect::<Vec<_>>();
        // SAFETY: each live subresource has one admitted state for the
        // operation, and the same pair restores its baseline afterward.
        if !barriers.is_empty() {
            unsafe { list.ResourceBarrier(&barriers) };
        }
    }
}

/// One admitted shader view, consumed from ordered native authority or
/// from an exact compatible byte representation.
pub(crate) enum TextureInput {
    Linear {
        copy: std::sync::Arc<crate::resource::HostTexture>,
        input: LinearTexture,
    },
    Scoped {
        copy: std::sync::Arc<crate::resource::HostTexture>,
        format: DXGI_FORMAT,
        selection: TextureSelection,
    },
    Native {
        copy: std::sync::Arc<crate::resource::HostTexture>,
        format: DXGI_FORMAT,
        exact_backing: bool,
        depth_plane: bool,
        compatible_input: Option<LinearTexture>,
    },
}

const EXTENDED_COLOR: [DXGI_FORMAT; 8] = [
    DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R11G11B10_FLOAT,
    DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
    DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8G8_UNORM,
];

/// 只接受字节布局确切的单平面格式，不从大小推测格式。
/// R8/RG8 是独立纹理；不将 NV12 等多平面资源当作这些格式。
pub(crate) fn linear_texel_bytes(format: DXGI_FORMAT) -> Option<u32> {
    Some(match format {
        DXGI_FORMAT_R8_UNORM => 1,
        DXGI_FORMAT_R8G8_UNORM => 2,
        DXGI_FORMAT_R16G16B16A16_FLOAT => 8,
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM
        | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB
        | DXGI_FORMAT_R10G10B10A2_UNORM | DXGI_FORMAT_R11G11B10_FLOAT
        | DXGI_FORMAT_R32_FLOAT | DXGI_FORMAT_R32_UINT | DXGI_FORMAT_R8G8B8A8_UINT => 4,
        _ => return None,
    })
}

pub(crate) fn sample_color(format: DXGI_FORMAT) -> bool {
    matches!(format, DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM)
        || EXTENDED_COLOR.contains(&format)
}

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

#[cfg(feature = "paged-title")]
pub(crate) fn resolve_support(host: &Host, format: DXGI_FORMAT) -> bool {
    static SUPPORT: std::sync::Mutex<std::collections::BTreeMap<i32, bool>> =
        std::sync::Mutex::new(std::collections::BTreeMap::new());
    let mut formats = SUPPORT.lock().unwrap_or_else(|e| e.into_inner());
    *formats.entry(format.0).or_insert_with(|| {
        let mut caps = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: format, ..Default::default() };
        // SAFETY: this structure matches the feature query.
        unsafe { host.device.CheckFeatureSupport(
            D3D12_FEATURE_FORMAT_SUPPORT, (&raw mut caps).cast(), size_of_val(&caps) as u32,
        ) }.is_ok() && caps.Support1.contains(D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RESOLVE)
    })
}

impl TextureInput {
    pub(crate) fn native(
        copy: std::sync::Arc<crate::resource::HostTexture>, format: DXGI_FORMAT,
        compatible_input: Option<LinearTexture>,
    ) -> Option<Self> {
        // SAFETY: the resource is retained by the view.
        let desc = unsafe { copy.texture.GetDesc() };
        let extended = EXTENDED_COLOR.contains(&format);
        let plain = desc.Flags.0 & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0
            | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0) == 0;
        if !sample_color(format)
            || (extended || plain) && copy.uses_unproven_alias_backing()
            || desc.Format != format || desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
            || desc.DepthOrArraySize != 1 || desc.MipLevels != 1
            || desc.SampleDesc.Count != 1 || desc.SampleDesc.Quality != 0
            || desc.Flags.0 & (D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0) != 0
            || copy.sparse.is_some() || copy.twin.get().is_some_and(Option::is_some)
        {
            return None;
        }
        let compatible_input = compatible_input.filter(|input| {
            input.format == format && u64::from(input.width) == desc.Width
                && input.height == desc.Height && !copy.uses_unproven_alias_backing()
        });
        let exact_backing = extended || plain || compatible_input.is_some();
        Some(Self::Native { copy, format, exact_backing, depth_plane: false, compatible_input })
    }

    pub(crate) fn scoped_color(
        copy: std::sync::Arc<crate::resource::HostTexture>, format: DXGI_FORMAT, selection: TextureSelection,
    ) -> Option<Self> {
        // SAFETY: retained native resource.
        let desc = unsafe { copy.texture.GetDesc() };
        if !sample_color(format) || desc.Format != format
            || (desc.DepthOrArraySize > 1) != selection.slices.is_some()
            || (desc.DepthOrArraySize == 1 && desc.MipLevels <= 1)
            || desc.Flags.contains(D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)
            || copy.sparse.is_some() || copy.twin.get().is_some_and(Option::is_some)
        {
            return None;
        }
        let mips = MipSelection::resolve(selection.mips.first, selection.mips.count, u32::from(desc.MipLevels))?;
        let slices = match selection.slices {
            Some(slices) => Some(MipSelection::resolve(slices.first, slices.count, u32::from(desc.DepthOrArraySize))?),
            None => None,
        };
        let sub = crate::page_resource::ColorSubresource::from_mip_slice(desc, mips.first, slices.map_or(0, |s| s.first))?;
        if !sub.allows(&copy) { return None; }
        Some(Self::Scoped { copy, format, selection: TextureSelection { mips, slices } })
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn has_exact_guest_bytes(&self) -> bool { matches!(self, Self::Linear { .. }) }

    pub(crate) fn selection(&self) -> TextureSelection {
        match self { Self::Scoped { selection, .. } => *selection, _ => TextureSelection::SINGLE }
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
        Some(Self::Native { copy, format, exact_backing: true, depth_plane: true, compatible_input: None })
    }

    pub(crate) fn ranges(&self, ranges: &mut Vec<(u64, u64)>) -> Result<(), BatchError> {
        #[cfg(feature = "paged-title")]
        if self.pointer_source().is_some() {
            // The closed queue includes these pages under the resource
            // owner, so its own later writeback may share those pages.
            return Ok(());
        }
        match self {
            Self::Scoped { .. } => {},
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

    #[cfg(feature = "paged-title")]
    pub(crate) fn pointer_source(&self) -> Option<&std::sync::Arc<crate::resource::HostTexture>> {
        match self {
            Self::Native { copy, .. } if copy.has_pointer_exposure() => Some(copy),
            _ => None,
        }
    }

    pub(crate) fn aliases_target(&self, target: &crate::resource::HostTexture) -> bool {
        let copy = match self {
            Self::Linear { copy, .. } | Self::Native { copy, .. } | Self::Scoped { copy, .. } => copy,
        };
        copy.texture == target.texture
    }

    pub(crate) fn is_native_depth_read_of(
        &self, target: &std::sync::Arc<crate::resource::HostTexture>,
    ) -> bool {
        matches!(self, Self::Native { copy, depth_plane: true, .. }
            if std::sync::Arc::ptr_eq(copy, target)
                && !copy.has_pointer_exposure() && !copy.uses_unproven_alias_backing())
    }

    pub(crate) fn conflicts_color(
        &self, target: &std::sync::Arc<crate::resource::HostTexture>,
        sub: crate::page_resource::ColorSubresource,
    ) -> bool {
        if !self.aliases_target(target) { return false; }
        let Self::Scoped { copy, selection, .. } = self else { return true; };
        if !sub.is_scoped() || !std::sync::Arc::ptr_eq(copy, target) { return true; }
        // SAFETY: the exact resource and its descriptor belong to the view.
        let mips = u32::from(unsafe { copy.texture.GetDesc() }.MipLevels);
        if mips == 0 { return true; }
        let (mip, slice) = (sub.index % mips, sub.index / mips);
        let slices = selection.slices.unwrap_or(MipSelection::SINGLE);
        mip >= selection.mips.first && mip - selection.mips.first < selection.mips.count
            && slice >= slices.first && slice - slices.first < slices.count
    }

    pub(crate) fn encode(
        &self, host: &Host, list: &HostList, batch: &PinnedBatch,
        plan: &mut crate::page_resource::ResourcePlan, mut shared: Option<&mut ScopedTextureAccess>,
    ) -> Result<(ID3D12Resource, DXGI_FORMAT), BatchError> {
        match self {
            Self::Scoped { copy, format, selection } => {
                if !extended_color_support(host, *format) {
                    return Err(BatchError::UnsupportedTextureMap);
                }
                // SAFETY: live resource retained by this view.
                let desc = unsafe { copy.texture.GetDesc() };
                let mips = selection.mips;
                let slices = selection.slices.unwrap_or(MipSelection::SINGLE);
                for slice in slices.first..slices.first + slices.count {
                    for mip in mips.first..mips.first + mips.count {
                        let sub = crate::page_resource::ColorSubresource::from_mip_slice(desc, mip, slice)
                            .ok_or(BatchError::UnsupportedTextureMap)?;
                        if !sub.allows(copy) { return Err(BatchError::UnsupportedTextureMap); }
                        plan.require_color_subresource(host, list, batch, copy, sub)?;
                        if let Some(reads) = shared.as_deref_mut() {
                            if copy.twin.get().is_some_and(Option::is_some) {
                                return Err(BatchError::UnsupportedBarrier);
                            }
                            reads.insert(&copy.texture, sub, plan.state(copy), SHADER_READ)?;
                        }
                    }
                }
                if shared.is_none() {
                    plan.transition_texture(list, copy, SHADER_READ.0 as u32)?;
                }
                list.keep(&copy.texture);
                Ok((copy.texture.clone(), *format))
            }
            Self::Linear { copy, input } => {
                if shared.is_some() || plan.authoritative(copy) {
                    return Err(BatchError::UnsupportedTextureMap);
                }
                if !extended_color_support(host, input.format) {
                    return Err(BatchError::UnsupportedTextureMap);
                }
                Ok((input.encode(host, list, batch)?, input.format))
            }
            Self::Native { copy, format, exact_backing, depth_plane, compatible_input } => {
                if shared.is_some() || *exact_backing && (copy.uses_unproven_alias_backing()
                    || copy.twin.get().is_some_and(Option::is_some))
                {
                    return Err(BatchError::UnsupportedTextureMap);
                }
                if !extended_color_support(host, *format) {
                    return Err(BatchError::UnsupportedTextureMap);
                }
                #[cfg(feature = "paged-title")]
                let pointer_backed = copy.has_pointer_exposure();
                #[cfg(not(feature = "paged-title"))]
                let pointer_backed = false;
                if let Some(input) = compatible_input
                    && !pointer_backed && !plan.authoritative(copy)
                {
                    return Ok((input.encode(host, list, batch)?, input.format));
                }
                if *depth_plane {
                    if pointer_backed { return Err(BatchError::UnsupportedTextureMap); }
                    plan.require_depth_planes(copy, 1)?;
                } else {
                    plan.require_native(host, list, batch, copy)?;
                }
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

/// 给当前队列核对原生传输足迹；来宾和宿主的 padding 彼此独立。
fn checked_texture_footprint(host: &Host, desc: D3D12_RESOURCE_DESC)
    -> Result<(D3D12_PLACED_SUBRESOURCE_FOOTPRINT, u64), BatchError>
{
    let depth = if desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(desc.DepthOrArraySize)
    } else {
        1
    };
    let texel = linear_texel_bytes(desc.Format).ok_or(BatchError::UnsupportedTextureMap)?;
    let width = desc.Width.checked_mul(u64::from(texel)).ok_or(BatchError::UnsupportedTextureMap)?;
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let (mut rows, mut row_bytes, mut total) = (0, 0, 0);
    // 安全：两种调用方都已验证要复制的 subresource 0 与尺寸。
    unsafe {
        host.device.GetCopyableFootprints(&desc, 0, 1, 0,
            Some(&mut footprint), Some(&mut rows), Some(&mut row_bytes), Some(&mut total));
    }
    let pitch = u64::from(footprint.Footprint.RowPitch);
    let last = u64::from(depth).checked_mul(u64::from(rows))
        .and_then(|n| n.checked_sub(1))
        .and_then(|n| n.checked_mul(pitch))
        .and_then(|n| n.checked_add(width))
        .and_then(|n| n.checked_add(footprint.Offset));
    if rows != desc.Height || row_bytes != width || width > pitch || total == u64::MAX || total == 0
        || footprint.Footprint.Depth != depth || last.is_none_or(|end| end > total)
    {
        return Err(BatchError::UnsupportedTextureMap);
    }
    Ok((footprint, total))
}

/// 地址式三维资源的线性字节表示。只有每行和每层都已证明后才可构造。
/// GPU 保留顺序：从当前物理页复制，随后将需要发布的写入散回相同的页。
#[cfg(feature = "paged-title")]
#[derive(Clone, Copy)]
pub(crate) struct LinearVolume {
    pub(crate) plane: LinearTexture,
    pub(crate) depth: u16,
    pub(crate) slice_pitch: u64,
}

#[cfg(feature = "paged-title")]
impl LinearVolume {
    pub(crate) fn rows(self) -> Result<impl DoubleEndedIterator<Item = (u64, u64)>, BatchError> {
        const LIMIT: u32 = D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION;
        let invalid = || BatchError::Range(self.plane.address, self.plane.address);
        let texel = linear_texel_bytes(self.plane.format).ok_or_else(invalid)?;
        let width = u64::from(self.plane.width) * u64::from(texel);
        let pitch = u64::from(self.plane.row_pitch);
        let depth = u64::from(self.depth);
        let height = u64::from(self.plane.height);
        if self.plane.address == 0 || depth == 0 || height == 0 || width == 0
            || u32::from(self.depth) > LIMIT || self.plane.height > LIMIT || self.plane.width > LIMIT
            || width > pitch
        {
            return Err(invalid());
        }
        let slice_bytes = ((height - 1) * pitch).checked_add(width).ok_or_else(invalid)?;
        let end = self.slice_pitch.checked_mul(depth - 1)
            .and_then(|n| n.checked_add(slice_bytes))
            .and_then(|n| self.plane.address.checked_add(n)).ok_or_else(invalid)?;
        if (depth > 1 && self.slice_pitch < slice_bytes) || end > ADDRESS_LIMIT {
            return Err(BatchError::Range(self.plane.address, end));
        }
        Ok((0..depth * height).map(move |index| {
            let at = self.plane.address + index / height * self.slice_pitch + index % height * pitch;
            (at, at + width)
        }))
    }

    fn footprint(self, host: &Host, texture: &ID3D12Resource)
        -> Result<(D3D12_PLACED_SUBRESOURCE_FOOTPRINT, u64), BatchError>
    {
        // 安全：查询保留的宿主资源，不读取其像素。
        let desc = unsafe { texture.GetDesc() };
        if desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D || desc.MipLevels != 1
            || desc.Width != u64::from(self.plane.width) || desc.Height != self.plane.height
            || desc.DepthOrArraySize != self.depth || desc.Format != self.plane.format
            || desc.SampleDesc.Count != 1 || desc.SampleDesc.Quality != 0
        {
            return Err(BatchError::UnsupportedTextureMap);
        }
        checked_texture_footprint(host, desc)
    }

    /// 调用方保证上传/读回所需状态；只交换有效行，不覆盖来宾填充字节。
    pub(crate) fn transfer(
        self, host: &Host, list: &HostList, batch: &PinnedBatch, texture: &ID3D12Resource,
        written_pages: Option<&mut std::collections::BTreeSet<victory_memory::physical::PhysicalPage>>,
    ) -> Result<(), BatchError> {
        let rows = self.rows()?;
        let (footprint, total) = self.footprint(host, texture)?;
        let staging = host.default_buffer(total).ok_or_else(|| BatchError::Native(E_OUTOFMEMORY.into()))?;
        let pitch = u64::from(footprint.Footprint.RowPitch);
        let regions = rows.enumerate().map(|(row, range)| (range, footprint.Offset + row as u64 * pitch));
        let native = D3D12_TEXTURE_COPY_LOCATION {
            // 安全：对象由调用方持有，并在本命令列表保留到 Fence。
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        let bytes = D3D12_TEXTURE_COPY_LOCATION {
            // 安全：暂存缓冲区按原生足迹分配，指针不取得独立所有权。
            pResource: unsafe { std::mem::transmute_copy(&staging) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
        };
        list.keep(texture);
        list.keep(&staging);
        if let Some(dirty) = written_pages {
            // 安全：调用方将纹理放在 COPY_SOURCE，暂存从 COPY_DEST 开始。
            unsafe { list.recording().CopyTextureRegion(&bytes, 0, 0, 0, &native, None); }
            crate::page_view::scatter_regions(list, batch, regions, &staging, dirty, None)
        } else {
            crate::page_view::gather_regions(list, batch, regions, &staging,
                D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)?;
            // 安全：调用方将纹理放在 COPY_DEST；gather 已转换暂存状态。
            unsafe { list.recording().CopyTextureRegion(&native, 0, 0, 0, &bytes, None); }
            Ok(())
        }
    }
}

#[cfg(all(test, feature = "paged-title"))]
#[test]
fn linear_volume_excludes_all_padding_and_rejects_overlapping_slices() {
    let volume = LinearVolume {
        plane: LinearTexture {
            address: 0x10000, row_pitch: 32, width: 2, height: 2,
            format: DXGI_FORMAT_R16G16B16A16_FLOAT,
        },
        depth: 2,
        slice_pitch: 128,
    };
    assert_eq!(volume.rows().unwrap().collect::<Vec<_>>(),
        [(0x10000, 0x10010), (0x10020, 0x10030), (0x10080, 0x10090), (0x100a0, 0x100b0)]);
    assert!((LinearVolume { slice_pitch: 47, ..volume }).rows().is_err());
    assert!((LinearVolume { slice_pitch: u64::MAX, ..volume }).rows().is_err());
}

impl LinearTexture {
    pub(crate) fn rows(self) -> Result<impl Iterator<Item = (u64, u64)>, BatchError> {
        let texel = linear_texel_bytes(self.format).ok_or(BatchError::Range(self.address, self.address))?;
        let bytes = u64::from(self.width) * u64::from(texel);
        let pitch = u64::from(self.row_pitch);
        let end = self.address.saturating_add(pitch * u64::from(self.height.saturating_sub(1)))
            .saturating_add(bytes);
        if self.address == 0 || self.width == 0 || self.height == 0
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
            .checked_add(u64::from(left) * u64::from(linear_texel_bytes(self.format)?))?;
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
        checked_texture_footprint(host, self.native_desc())
    }

    /// Keep a native texture-to-linear copy on the GPU so subsequent
    /// commands can observe it before the submission's single CPU merge.
    pub(crate) fn copy_from_subresource(
        self, host: &Host, list: &HostList, batch: &PinnedBatch, texture: &ID3D12Resource, index: u32,
        dirty: &mut std::collections::BTreeSet<victory_memory::physical::PhysicalPage>,
        exact: Option<&mut crate::page_resource::ResourcePlan>,
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
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: index },
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
        }), &staging, dirty, exact)
    }

    /// Gather current GPU bytes at the exact point of use. The caller
    /// supplies a matching native subresource zero in COPY_DEST.
    pub(crate) fn copy_to(
        self, host: &Host, list: &HostList, batch: &PinnedBatch, texture: &ID3D12Resource,
        origin: (u32, u32),
    ) -> Result<(), BatchError> {
        self.copy_to_subresource(host, list, batch, texture, 0, origin)
    }

    pub(crate) fn copy_to_subresource(
        self, host: &Host, list: &HostList, batch: &PinnedBatch, texture: &ID3D12Resource,
        index: u32, (x, y): (u32, u32),
    ) -> Result<(), BatchError> {
        let rows = self.rows()?;
        // SAFETY: the native resource is live and retained by the caller.
        let desc = unsafe { texture.GetDesc() };
        let sub = crate::page_resource::ColorSubresource::capture(desc, index)
            .ok_or_else(|| BatchError::Native(E_FAIL.into()))?;
        if x.checked_add(self.width).is_none_or(|end| u64::from(end) > sub.width)
            || y.checked_add(self.height).is_none_or(|end| end > sub.height)
            || desc.Format != self.format
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
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: index },
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
fn paged_linear_video_plane_rows_keep_padding_outside_the_copy() {
    let span = LinearTexture {
        address: 0x10000, row_pitch: 256, width: 5, height: 2, format: DXGI_FORMAT_R8_UNORM,
    };
    assert_eq!(span.rows().unwrap().collect::<Vec<_>>(), [(0x10000, 0x10005), (0x10100, 0x10105)]);
    let chroma = LinearTexture { format: DXGI_FORMAT_R8G8_UNORM, ..span };
    assert_eq!(chroma.rows().unwrap().collect::<Vec<_>>(), [(0x10000, 0x1000a), (0x10100, 0x1010a)]);
    let crop = chroma.crop(Some([1, 1, 0, 4, 2, 1])).unwrap();
    assert_eq!(crop.rows().unwrap().collect::<Vec<_>>(), [(0x10102, 0x10108)]);
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
