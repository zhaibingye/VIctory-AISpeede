//! 从固定物理页构造宿主纹理，不维持独立的 CPU 影子副本。
//! 共享体纹理只支持可证明的线性布局或有界的 4 KiB 分块布局。
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

/// 地址式三维资源的精确字节后备。宿主原生纹理只是单次派发的工作表示。
#[cfg(feature = "paged-title")]
pub(crate) struct VolumeBytes {
    pub(crate) format: DXGI_FORMAT,
    width: u32,
    height: u32,
    depth: u16,
    layout: VolumeLayout,
}

#[cfg(feature = "paged-title")]
enum VolumeLayout {
    Linear { base: u64, row_pitch: u64, slice_pitch: u64 },
    Blocked { regions: Vec<VolumeRegion>, spans: Vec<(u64, u64)> },
}

#[cfg(feature = "paged-title")]
#[derive(Clone, Copy)]
struct VolumeRegion {
    start: u64,
    len: u64,
    row: u64,
    offset: u64,
}

#[cfg(feature = "paged-title")]
impl VolumeBytes {
    pub(crate) fn new(base: u64, format: DXGI_FORMAT, layout: &victory_formats::texture::Layout)
        -> Option<Self>
    {
        use victory_formats::texture::{Dimension, Swizzle};
        const LIMIT: u32 = D3D12_REQ_TEXTURE3D_U_V_OR_W_DIMENSION;
        let [mip] = layout.mips.as_slice() else { return None; };
        let texel = u64::from(linear_texel_bytes(format)?);
        let (width, height, depth) = (mip.width, mip.height, mip.depth);
        let end = base.checked_add(layout.size)?;
        if base == 0 || end > ADDRESS_LIMIT || !base.is_multiple_of(layout.alignment)
            || layout.desc.dimension != Dimension::Tex3d || layout.desc.samples != 1
            || layout.desc.block != 1 || u64::from(layout.desc.element_bytes) != texel
            || !layout.addressable() || mip.tail.is_some()
            || [width, height, depth].into_iter().any(|n| n == 0 || n > LIMIT)
        {
            return None;
        }
        let layout = match layout.desc.swizzle {
            Swizzle::LINEAR if !layout.thick && layout.slices == depth => {
                let start = base.checked_add(layout.offset(0, 0, 0, 0).ok()?)?;
                let row_pitch = u64::from(mip.pitch) * texel;
                let row_bytes = u64::from(width) * texel;
                let slice_pitch = layout.slice_size;
                let slice_bytes = u64::from(height - 1) * row_pitch + row_bytes;
                let last = slice_pitch.checked_mul(u64::from(depth - 1))?
                    .checked_add(slice_bytes)?.checked_add(start)?;
                if row_bytes > row_pitch || depth > 1 && slice_pitch < slice_bytes || last > end {
                    return None;
                }
                VolumeLayout::Linear { base: start, row_pitch, slice_pitch }
            }
            Swizzle::SW_4KB_S if layout.thick => {
                // 只为有界小体纹理展开一次，合并来宾和宿主都连续的字节。
                // 更大的工作集需要 GPU 重排，不能用无上限的 CopyBufferRegion。
                const MAX_TEXELS: u64 = 262_144;
                const MAX_REGIONS: usize = 32_768;
                if u64::from(width) * u64::from(height) * u64::from(depth) > MAX_TEXELS {
                    return None;
                }
                let mut regions: Vec<VolumeRegion> = Vec::new();
                for z in 0..depth {
                    for y in 0..height {
                        let row = u64::from(z) * u64::from(height) + u64::from(y);
                        for x in 0..width {
                            let start = base.checked_add(layout.offset(0, x, y, z).ok()?)?;
                            let offset = u64::from(x) * texel;
                            if start.checked_add(texel)? > end { return None; }
                            if let Some(last) = regions.last_mut()
                                && last.row == row && last.start + last.len == start
                                && last.offset + last.len == offset
                            {
                                last.len += texel;
                            } else {
                                if regions.len() == MAX_REGIONS { return None; }
                                regions.push(VolumeRegion { start, len: texel, row, offset });
                            }
                        }
                    }
                }
                let mut spans: Vec<_> = regions.iter().map(|r| (r.start, r.start + r.len)).collect();
                spans.sort_unstable();
                let mut merged = Vec::<(u64, u64)>::new();
                for (start, end) in spans {
                    if let Some((_, last)) = merged.last_mut() {
                        if start < *last { return None; }
                        if start == *last {
                            *last = end;
                            continue;
                        }
                    }
                    merged.push((start, end));
                }
                VolumeLayout::Blocked { regions, spans: merged }
            }
            _ => return None,
        };
        Some(Self { format, width, height, depth: u16::try_from(depth).ok()?, layout })
    }

    fn regions(&self) -> impl Iterator<Item = VolumeRegion> + '_ {
        let rows = u64::from(self.height);
        let row_bytes = u64::from(self.width) * u64::from(linear_texel_bytes(self.format).unwrap());
        let (linear, blocked) = match &self.layout {
            VolumeLayout::Linear { base, row_pitch, slice_pitch } => {
                (Some((*base, *row_pitch, *slice_pitch)), &[][..])
            }
            VolumeLayout::Blocked { regions, .. } => (None, regions.as_slice()),
        };
        linear.into_iter().flat_map(move |(base, row_pitch, slice_pitch)| {
            (0..rows * u64::from(self.depth)).map(move |row| VolumeRegion {
                start: base + row / rows * slice_pitch + row % rows * row_pitch,
                len: row_bytes, row, offset: 0,
            })
        }).chain(blocked.iter().copied())
    }

    pub(crate) fn ranges(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        let blocked = match &self.layout {
            VolumeLayout::Linear { .. } => None,
            VolumeLayout::Blocked { spans, .. } => Some(spans.as_slice()),
        };
        blocked.into_iter().flatten().copied().chain(
            self.regions().take(if blocked.is_some() { 0 } else { usize::MAX })
                .map(|r| (r.start, r.start + r.len)),
        )
    }

    fn footprint(&self, host: &Host, texture: &ID3D12Resource)
        -> Result<(D3D12_PLACED_SUBRESOURCE_FOOTPRINT, u64), BatchError>
    {
        // 安全：查询保留的宿主资源，不读取其像素。
        let desc = unsafe { texture.GetDesc() };
        if desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D || desc.MipLevels != 1
            || desc.Width != u64::from(self.width) || desc.Height != self.height
            || desc.DepthOrArraySize != self.depth || desc.Format != self.format
            || desc.SampleDesc.Count != 1 || desc.SampleDesc.Quality != 0
        {
            return Err(BatchError::UnsupportedTextureMap);
        }
        checked_texture_footprint(host, desc)
    }

    /// 调用方保证上传/读回所需状态；只交换有效行，不覆盖来宾填充字节。
    pub(crate) fn transfer(
        &self, host: &Host, list: &HostList, batch: &PinnedBatch, texture: &ID3D12Resource,
        written_pages: Option<&mut std::collections::BTreeSet<victory_memory::physical::PhysicalPage>>,
    ) -> Result<(), BatchError> {
        let (footprint, total) = self.footprint(host, texture)?;
        let staging = host.default_buffer(total).ok_or_else(|| BatchError::Native(E_OUTOFMEMORY.into()))?;
        let pitch = u64::from(footprint.Footprint.RowPitch);
        let regions = self.regions().map(|r| {
            ((r.start, r.start + r.len), footprint.Offset + r.row * pitch + r.offset)
        });
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
fn shared_volume_maps_only_texels_with_matching_host_rows() {
    use victory_formats::texture::{Desc, Dimension, Layout, Swizzle};
    for swizzle in [Swizzle::LINEAR, Swizzle::SW_4KB_S] {
        // 非对齐尺寸跨分块边界，同时覆盖最后一个不完整深度块。
        let layout = Layout::new(&Desc {
            dimension: Dimension::Tex3d, width: 19, height: 11, depth: 9,
            mips: 1, element_bytes: 8, block: 1, samples: 1, swizzle,
        }).unwrap();
        let base = 0x10000;
        let plan = VolumeBytes::new(base, DXGI_FORMAT_R16G16B16A16_FLOAT, &layout).unwrap();
        let mut covered = std::collections::BTreeSet::new();
        let mut mapped = Vec::new();
        for r in plan.regions() {
            assert!(r.offset + r.len <= 19 * 8);
            assert_eq!(r.len % 8, 0);
            for step in (0..r.len).step_by(8) {
                assert!(covered.insert(r.start + step));
                mapped.push((r.row, r.offset + step, r.start + step));
            }
        }
        let mut expected = Vec::new();
        for z in 0..9u32 {
            for y in 0..11u32 {
                for x in 0..19u32 {
                    expected.push((u64::from(z * 11 + y), u64::from(x * 8),
                        base + layout.offset(0, x, y, z).unwrap()));
                }
            }
        }
        assert_eq!(mapped, expected);
        let mut pinned = std::collections::BTreeSet::new();
        for (start, end) in plan.ranges() {
            for byte in (start..end).step_by(8) {
                assert!(pinned.insert(byte));
            }
        }
        assert_eq!(covered, pinned);
    }
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

/// 验证宿主 D3D12 有类型三维纹理视图的边缘写入，不依赖游戏材料或 VNME 准入规则。
#[cfg(test)]
#[test]
fn native_texture3d_typed_store_preserves_every_voxel_after_edge_writes() {
    let host = crate::host::host().expect("D3D12 宿主设备");
    let (root, pso) = crate::memory::tests::pipeline(r#"
#define RS "DescriptorTable(UAV(u0)), RootConstants(num32BitConstants=1, b0)"
RWTexture3D<float4> volume : register(u0);
cbuffer Args : register(b0) { uint phase; }
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 pos : SV_DispatchThreadID) {
    bool inside = all(pos < 33u);
    if (phase == 0u) {
        if (inside) volume[pos] = float4(float3(pos + 1u), 1.0);
    } else {
        if (!inside) volume[pos] = float4(-1.0, -2.0, -3.0, -4.0);
    }
}
"#);
    const SIDE: u32 = 33;
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE3D,
        Width: u64::from(SIDE), Height: SIDE, DepthOrArraySize: SIDE as u16, MipLevels: 1,
        Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let unordered = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    let copy_source = D3D12_RESOURCE_STATE_COPY_SOURCE;
    let make = |kind, desc: &D3D12_RESOURCE_DESC, state| -> ID3D12Resource {
        let mut out = None;
        // 安全：测试创建单个有效的小型资源，持有对象直到命令队列完成。
        unsafe {
            host.device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() },
                D3D12_HEAP_FLAG_NONE, desc, state, None, &mut out,
            )
        }.unwrap();
        out.unwrap()
    };
    let texture = make(D3D12_HEAP_TYPE_DEFAULT, &desc, unordered);
    let (footprint, total) = checked_texture_footprint(host, desc).unwrap();
    let readback = make(D3D12_HEAP_TYPE_READBACK, &D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: total, Height: 1, DepthOrArraySize: 1, MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    }, D3D12_RESOURCE_STATE_COPY_DEST);
    let list = host.new_list().unwrap();
    let cmd = list.recording();
    // 安全：独占测试堆中的一个描述符，绑定一个与原格式/全深度一致的三维资源。
    let heap: ID3D12DescriptorHeap = unsafe {
        let heap: ID3D12DescriptorHeap = host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: 1,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0,
        }).unwrap();
        let view = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: desc.Format, ViewDimension: D3D12_UAV_DIMENSION_TEXTURE3D,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Texture3D: D3D12_TEX3D_UAV { MipSlice: 0, FirstWSlice: 0, WSize: SIDE },
            },
        };
        host.device.CreateUnorderedAccessView(&texture, None, Some(&view), heap.GetCPUDescriptorHandleForHeapStart());
        cmd.SetDescriptorHeaps(&[Some(heap.clone())]);
        cmd.SetComputeRootSignature(&root);
        cmd.SetComputeRootDescriptorTable(0, heap.GetGPUDescriptorHandleForHeapStart());
        cmd.SetPipelineState(&pso);
        cmd.SetComputeRoot32BitConstant(1, 0, 0);
        cmd.Dispatch(5, 5, SIDE);
        // 两次有序派发：即使发生钳位，第一个写入的坐标值也必然被不同的标记覆盖。
        cmd.ResourceBarrier(&[transition(&texture, unordered, copy_source)]);
        cmd.ResourceBarrier(&[transition(&texture, copy_source, unordered)]);
        cmd.SetComputeRoot32BitConstant(1, 1, 0);
        cmd.Dispatch(5, 5, SIDE);
        cmd.ResourceBarrier(&[transition(&texture, unordered, copy_source)]);
        let src = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&texture),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        let dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&readback),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
        };
        cmd.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        heap
    };
    list.close();
    {
        let mut signal = host.direct_queue();
        host.run_native_group(&mut signal, &[Some(cmd.cast().unwrap())]).unwrap();
    }
    let half = |positive: u32| {
        let exponent = positive.ilog2();
        (((15 + exponent) << 10) | ((positive - (1 << exponent)) << (10 - exponent))) as u16
    };
    let pitch = footprint.Footprint.RowPitch as usize;
    let mut pointer = std::ptr::null_mut();
    // 安全：只在真实 Fence 后映射，查询过的 copy 足迹完全落在读回资源内。
    let bytes = unsafe {
        readback.Map(0, None, Some(&mut pointer)).unwrap();
        std::slice::from_raw_parts(pointer.cast::<u8>(), total as usize)
    };
    for z in 0..SIDE {
        for y in 0..SIDE {
            for x in 0..SIDE {
                let at = footprint.Offset as usize + (z * SIDE + y) as usize * pitch + x as usize * 8;
                let got = std::array::from_fn(|channel| {
                    let i = at + 2 * channel;
                    u16::from_le_bytes([bytes[i], bytes[i + 1]])
                });
                assert_eq!(got, [half(x + 1), half(y + 1), half(z + 1), half(1)],
                    "体素 ({x},{y},{z}) 被意外覆盖");
            }
        }
    }
    // 安全：本测试结束映射，不再持有或读取映射内容。
    unsafe { readback.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 })) };
    drop(heap);
}
