//! First closed draw form for the page renderer: one native color target
//! and fixed-function inputs. No descriptor or shader-side guest accesses.
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;

use std::sync::{Arc, atomic::Ordering};

use crate::host::{Host, HostList, transition_subresource};
#[cfg(all(test, feature = "diagnostics"))]
use crate::host::transition;
use crate::page_resource::{ColorSubresource, ResourcePlan};
use crate::page_shader::{ShaderBindings, UnloweredShader};
#[cfg(all(test, feature = "diagnostics"))]
use crate::page_shader::RootWords;
use crate::page_slabs::{BatchError, PinnedBatch};
use crate::page_view::InputView;
use crate::resource::HostTexture;

pub(crate) enum DrawCall {
    Direct { vertices: u32, instances: u32, first: u32, first_instance: u32 },
    Indexed { indices: u32, instances: u32, first: u32, base: i32, first_instance: u32 },
    Indirect(crate::page_indirect::IndirectArgs),
}

impl DrawCall {
    pub(crate) fn indexed(&self) -> bool {
        match self {
            Self::Direct { .. } => false,
            Self::Indexed { .. } => true,
            Self::Indirect(draw) => draw.kind == crate::page_indirect::IndirectKind::Indexed,
        }
    }
}

#[derive(Clone)]
pub(crate) struct ColorTarget {
    copy: Arc<HostTexture>,
    desc: Option<D3D12_RENDER_TARGET_VIEW_DESC>,
    sub: ColorSubresource,
    samples: u32,
    format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
}

impl ColorTarget {
    #[cfg(all(feature = "paged-title", feature = "diagnostics"))]
    pub(crate) fn format_id(&self) -> i32 { self.format.0 }

    /// Only an already-authoritative host render target is admitted here.
    /// Title-memory texture uploads, aliasing and sparse placement must
    /// be lowered before they can participate in this form of draw.
    pub(crate) fn capture(
        copy: Arc<HostTexture>,
        desc: Option<D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> Option<Self> {
        if !copy.host_written.load(Ordering::Acquire) {
            return None;
        }
        Self::capture_candidate(copy, desc)
    }

    pub(crate) fn capture_candidate(
        copy: Arc<HostTexture>,
        desc: Option<D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> Option<Self> {
        // SAFETY: the retained COM object is live.
        let tex = unsafe { copy.texture.GetDesc() };
        if copy.sparse.is_some() || copy.twin.get().is_some_and(Option::is_some)
            || !tex.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)
        {
            return None;
        }
        if tex.SampleDesc.Count > 1 {
            use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN;
            let sub = ColorSubresource::multisample(tex)?;
            if desc.as_ref().is_some_and(|d| d.ViewDimension != D3D12_RTV_DIMENSION_TEXTURE2DMS
                || d.Format != tex.Format && d.Format != DXGI_FORMAT_UNKNOWN)
                || copy.uses_unproven_alias_backing() || copy.has_pointer_exposure()
            {
                return None;
            }
            let desc = desc.map(|d| D3D12_RENDER_TARGET_VIEW_DESC { Format: tex.Format, ..d });
            return Some(Self { copy, desc, sub, samples: tex.SampleDesc.Count, format: tex.Format });
        }
        let (mip, slice) = match desc.as_ref() {
            None if tex.DepthOrArraySize == 1 => (0, 0),
            Some(d) if d.ViewDimension == D3D12_RTV_DIMENSION_TEXTURE2D
                && tex.DepthOrArraySize == 1 =>
            {
                // SAFETY: the view dimension selects Texture2D.
                let view = unsafe { d.Anonymous.Texture2D };
                if view.PlaneSlice != 0 { return None; }
                (view.MipSlice, 0)
            }
            Some(d) if d.ViewDimension == D3D12_RTV_DIMENSION_TEXTURE2DARRAY => {
                // SAFETY: the view dimension selects Texture2DArray.
                let view = unsafe { d.Anonymous.Texture2DArray };
                if view.PlaneSlice != 0 || view.ArraySize != 1 { return None; }
                (view.MipSlice, view.FirstArraySlice)
            }
            _ => return None,
        };
        let sub = ColorSubresource::from_mip_slice(tex, mip, slice)?;
        let format = desc.as_ref().map_or(tex.Format, |d| d.Format);
        if !sub.allows(&copy) || sub.is_scoped() && format != tex.Format {
            return None;
        }
        Some(Self { copy, desc, sub, samples: 1, format })
    }

    fn extent(&self) -> (u64, u32) { (self.sub.width, self.sub.height) }

    fn validate_native(&self) -> Result<(), BatchError> {
        if self.sub.allows(&self.copy) && !self.copy.twin.get().is_some_and(Option::is_some)
            && (self.samples == 1 || !self.copy.uses_unproven_alias_backing() && !self.copy.has_pointer_exposure())
        {
            Ok(())
        } else {
            Err(BatchError::unsupported_texture_map("无法确认纹理映射"))
        }
    }

}

pub(crate) struct ColorClear {
    pub(crate) target: ColorTarget,
    pub(crate) color: [f32; 4],
    pub(crate) rects: Vec<RECT>,
}

impl ColorClear {
    #[cfg(feature = "paged-title")]
    pub(crate) fn preserves_color(&self) -> Option<&Arc<HostTexture>> {
        (!self.replaces_all()).then_some(&self.target.copy)
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn written_targets<'a>(&'a self, out: &mut Vec<&'a Arc<HostTexture>>) {
        out.push(&self.target.copy);
    }

    fn replaces_all(&self) -> bool {
        let (width, height) = self.target.extent();
        self.rects.is_empty() || self.rects.iter().any(|rect| {
            rect.left == 0 && rect.top == 0
                && u64::try_from(rect.right).ok() == Some(width)
                && u32::try_from(rect.bottom).ok() == Some(height)
        })
    }

    pub(crate) fn encode(
        &self, host: &Host, list: &HostList, batch: &PinnedBatch, initialized: &mut ResourcePlan,
    ) -> Result<(), BatchError> {
        self.target.validate_native()?;
        let (width, height) = self.target.extent();
        if self.target.sub.is_scoped() && self.rects.iter().any(|rect| {
            rect.left < 0 || rect.top < 0 || rect.right <= rect.left || rect.bottom <= rect.top
                || rect.right as u64 > width || rect.bottom as u32 > height
        }) {
            return Err(BatchError::unsupported_texture_map("无法确认纹理映射"));
        }
        if !self.replaces_all() {
            initialized.require_color_subresource(host, list, batch, &self.target.copy, self.target.sub)?;
        }
        let texture = &self.target.copy.texture;
        let before = initialized.state(&self.target.copy);
        let after = D3D12_RESOURCE_STATE_RENDER_TARGET;
        let rtv = host.rtv(texture, self.target.desc.as_ref());
        list.keep(texture);
        // SAFETY: target and native RTV are validated and retained. We
        // scope transitions to the clear under the submission lock.
        unsafe {
            let cmd = list.recording();
            if before != after {
                cmd.ResourceBarrier(&[self.target.sub.temporary_transition(texture, before, after)]);
            }
            let rects = (!self.rects.is_empty()).then_some(self.rects.as_slice());
            cmd.ClearRenderTargetView(rtv, &self.color, rects);
            if before != after {
                cmd.ResourceBarrier(&[self.target.sub.temporary_transition(texture, after, before)]);
            }
        }
        initialized.record_color_write(&self.target.copy, self.target.sub)?;
        Ok(())
    }
}

#[derive(Clone)]
pub(crate) struct DepthTarget {
    copy: Arc<HostTexture>,
    desc: Option<D3D12_DEPTH_STENCIL_VIEW_DESC>,
    format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    planes: u32,
    read_only_planes: u32,
    samples: u32,
}

impl DepthTarget {
    #[cfg(all(feature = "paged-title", feature = "diagnostics"))]
    pub(crate) fn format_id(&self) -> i32 { self.format.0 }

    pub(crate) fn capture(
        copy: Arc<HostTexture>, desc: Option<D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> Option<Self> {
        use windows::Win32::Graphics::Dxgi::Common::*;
        // SAFETY: retained host resource.
        let tex = unsafe { copy.texture.GetDesc() };
        if copy.sparse.is_some() || copy.twin.get().is_some_and(Option::is_some)
            || tex.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
            || !crate::page_resource::supported_samples(tex.SampleDesc)
            || tex.SampleDesc.Count > 1 && (copy.uses_unproven_alias_backing() || copy.has_pointer_exposure())
            || tex.DepthOrArraySize != 1 || tex.MipLevels != 1
            || tex.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 == 0
        {
            return None;
        }
        let format = if let Some(d) = &desc {
            if tex.SampleDesc.Count == 1 {
                if d.ViewDimension != D3D12_DSV_DIMENSION_TEXTURE2D { return None; }
                // SAFETY: dimension selects Texture2D.
                if unsafe { d.Anonymous.Texture2D }.MipSlice != 0 { return None; }
            } else if d.ViewDimension != D3D12_DSV_DIMENSION_TEXTURE2DMS {
                return None;
            }
            if d.Format == DXGI_FORMAT_UNKNOWN { tex.Format } else { d.Format }
        } else {
            tex.Format
        };
        let (planes, typeless) = match format {
            DXGI_FORMAT_D16_UNORM => (1, DXGI_FORMAT_R16_TYPELESS),
            DXGI_FORMAT_D32_FLOAT => (1, DXGI_FORMAT_R32_TYPELESS),
            DXGI_FORMAT_D24_UNORM_S8_UINT => (3, DXGI_FORMAT_R24G8_TYPELESS),
            DXGI_FORMAT_D32_FLOAT_S8X24_UINT => (3, DXGI_FORMAT_R32G8X24_TYPELESS),
            _ => return None,
        };
        if tex.Format != format && tex.Format != typeless {
            return None;
        }
        let read_only_planes = desc.as_ref().map_or(0, |d| d.Flags.0 as u32);
        if read_only_planes & !planes != 0 {
            return None;
        }
        Some(Self { copy, desc, format, planes, read_only_planes, samples: tex.SampleDesc.Count })
    }

    fn validate_backing(&self) -> Result<(), BatchError> {
        if self.copy.twin.get().is_some_and(Option::is_some)
            || (self.samples > 1 || self.read_only_planes != 0)
                && (self.copy.uses_unproven_alias_backing() || self.copy.has_pointer_exposure())
        {
            return Err(BatchError::UnsupportedDepth);
        }
        Ok(())
    }

    fn begin(
        &self, host: &Host, list: &HostList, initialized: &ResourcePlan, sampled: bool,
    ) -> (D3D12_CPU_DESCRIPTOR_HANDLE, DepthScope) {
        let state_for = |plane_mask| {
            if self.read_only_planes & plane_mask != 0 {
                D3D12_RESOURCE_STATE_DEPTH_READ
            } else {
                D3D12_RESOURCE_STATE_DEPTH_WRITE
            }
        };
        let mut depth = state_for(1);
        if sampled {
            depth |= D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        }
        let scope = DepthScope {
            before: initialized.state(&self.copy),
            depth,
            stencil: (self.planes & 2 != 0).then(|| state_for(2)),
        };
        let texture = &self.copy.texture;
        list.keep(texture);
        scope.transition(list.recording(), texture, false);
        (host.dsv(texture, self.desc.as_ref()), scope)
    }
}

/// Only the command body sees mixed states. Restore the uniform baseline
/// before another operation or the submitted tail can observe this image.
struct DepthScope {
    before: D3D12_RESOURCE_STATES,
    depth: D3D12_RESOURCE_STATES,
    stencil: Option<D3D12_RESOURCE_STATES>,
}

impl DepthScope {
    fn transition(&self, cmd: &ID3D12GraphicsCommandList, texture: &ID3D12Resource, restore: bool) {
        let one = |index, desired| {
            let (old, new) = if restore { (desired, self.before) } else { (self.before, desired) };
            (old != new).then(|| transition_subresource(texture, index, old, new))
        };
        let states = if let Some(stencil) = self.stencil.filter(|&state| state != self.depth) {
            // The captured DSV has one mip and slice. D3D12 plane indices
            // therefore equal native subresource indices for this layout.
            [one(0, self.depth), one(1, stencil)]
        } else {
            [one(D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, self.depth), None]
        };
        // SAFETY: the exact texture and both states are retained; the
        // scope is paired within this command under the queue lock.
        unsafe {
            match states {
                [Some(depth), Some(stencil)] => cmd.ResourceBarrier(&[depth, stencil]),
                [Some(barrier), None] | [None, Some(barrier)] => cmd.ResourceBarrier(&[barrier]),
                [None, None] => {}
            }
        }
    }
}

pub(crate) struct DepthClear {
    pub(crate) target: DepthTarget,
    pub(crate) flags: u32,
    pub(crate) depth: f32,
    pub(crate) stencil: u8,
    pub(crate) rects: Vec<RECT>,
}

impl DepthClear {
    #[cfg(feature = "paged-title")]
    pub(crate) fn written_targets<'a>(&'a self, out: &mut Vec<&'a Arc<HostTexture>>) {
        out.push(&self.target.copy);
    }

    pub(crate) fn encode(
        &self, host: &Host, list: &HostList, initialized: &mut ResourcePlan,
    ) -> Result<(), BatchError> {
        self.target.validate_backing()?;
        let planes = self.target.planes;
        if self.flags == 0 || self.flags & !planes != 0 || self.flags & self.target.read_only_planes != 0
            || self.flags & 1 != 0 && !(0.0..=1.0).contains(&self.depth)
        {
            return Err(BatchError::UnsupportedDepth);
        }
        // SAFETY: the retained texture owns the checked single-mip DSV.
        let tex = unsafe { self.target.copy.texture.GetDesc() };
        let replaces_planes = self.rects.is_empty() || self.rects.iter().any(|rect| {
            rect.left == 0 && rect.top == 0 && rect.right > 0 && rect.bottom > 0
                && rect.right as u64 == tex.Width && rect.bottom as u32 == tex.Height
        });
        if !replaces_planes {
            initialized.require_depth_planes(&self.target.copy, self.flags)?;
        }
        let (handle, scope) = self.target.begin(host, list, initialized, false);
        // SAFETY: retained matching DSV in the scoped depth state.
        unsafe {
            let rects = (!self.rects.is_empty()).then_some(self.rects.as_slice());
            list.recording().ClearDepthStencilView(handle, D3D12_CLEAR_FLAGS(self.flags as i32),
                self.depth, self.stencil, rects);
        }
        scope.transition(list.recording(), &self.target.copy.texture, true);
        initialized.record_depth_writes(&self.target.copy, self.flags)?;
        Ok(())
    }
}

pub(crate) struct ClosedDraw {
    pipeline: ID3D12PipelineState,
    root: ID3D12RootSignature,
    targets: Vec<ColorTarget>,
    depth: Option<DepthTarget>,
    depth_tests: u32,
    depth_writes: u32,
    stencil_ref: u32,
    topology: D3D_PRIMITIVE_TOPOLOGY,
    viewport: D3D12_VIEWPORT,
    scissor: RECT,
    inputs: Vec<InputView>,
    call: DrawCall,
    shader: ShaderBindings,
    blend_factor: [f32; 4],
    sample: Option<crate::page_texture::SampledTexture>,
}

fn host_draw_topology(guest: u32, rectangle_gs: bool) -> Result<D3D_PRIMITIVE_TOPOLOGY, crate::page_shader::DrawState> {
    use crate::draw::RECTANGLES;
    use crate::page_shader::DrawState;
    use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    match (guest, rectangle_gs) {
        (RECTANGLES, true) => Ok(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST),
        (1..=5, false) => Ok(D3D_PRIMITIVE_TOPOLOGY(guest as i32)),
        (RECTANGLES, false) | (1..=5, true) => Err(DrawState::RectanglePipeline {
            topology: guest, geometry_shader: rectangle_gs,
        }),
        _ => Err(DrawState::Topology(guest)),
    }
}

#[cfg(test)]
#[test]
fn closed_rectangle_topology_requires_a_geometry_shader() {
    use crate::draw::RECTANGLES;
    use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    assert_eq!(host_draw_topology(RECTANGLES, true), Ok(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST));
    assert!(host_draw_topology(RECTANGLES, false).is_err());
    assert_eq!(host_draw_topology(4, false), Ok(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST));
    assert!(host_draw_topology(4, true).is_err());
}

impl ClosedDraw {
    #[cfg(feature = "paged-title")]
    pub(crate) fn preserved_colors<'a>(&'a self, out: &mut Vec<&'a Arc<HostTexture>>) {
        out.extend(self.targets.iter().map(|target| &target.copy));
        self.shader.pointer_sources(out);
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn written_targets<'a>(&'a self, out: &mut Vec<&'a Arc<HostTexture>>) {
        out.extend(self.targets.iter().map(|target| &target.copy));
        if let Some(depth) = &self.depth && self.depth_writes != 0 { out.push(&depth.copy); }
    }

    // The trusted producer must prove that the pipeline has no shader
    // memory dependencies, that all its input slots are represented,
    // and that the target view is compatible with the pipeline.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_proven_native(
        pipeline: ID3D12PipelineState,
        root: ID3D12RootSignature,
        target: ColorTarget,
        topology: D3D_PRIMITIVE_TOPOLOGY,
        viewport: D3D12_VIEWPORT,
        scissor: RECT,
        inputs: Vec<InputView>,
        call: DrawCall,
    ) -> Self {
        Self {
            pipeline, root, targets: vec![target], depth: None, depth_tests: 0, depth_writes: 0, stencil_ref: 0, topology, viewport, scissor, inputs, call,
            shader: ShaderBindings::default(),
            blend_factor: [1.0; 4],
            sample: None,
        }
    }

    /// 捕获已转换的游戏管线，以及它对应的资源、输入和输出状态。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_translated(
        pipeline: &crate::pipeline::host::HostPipeline,
        values: impl Fn(u8, u32) -> Option<u32>,
        view: impl Fn(u8, u32) -> Option<crate::descriptor::View>,
        address: impl Fn(u8) -> Option<u64>,
        targets: Vec<ColorTarget>,
        depth: Option<DepthTarget>,
        stencil_ref: u32,
        guest_topology: u32,
        viewport: D3D12_VIEWPORT,
        scissor: RECT,
        inputs: Vec<InputView>,
        call: DrawCall,
    ) -> Result<Self, UnloweredShader> {
        if pipeline.compute || pipeline.mesh || pipeline.marks.is_some() {
            return Err(UnloweredShader::UnknownUserData);
        }
        use crate::page_shader::{DrawState as Reason, UnloweredShader::DrawState};
        // 主机没有矩形列表。仅在管线确实含补角几何着色器时才把
        // 每组三个顶点送入三角形列表；着色器补出第四角和第二个三角形。
        let topology = host_draw_topology(guest_topology, pipeline.rectangles).map_err(DrawState)?;
        let formats = pipeline.closed_color_formats.as_ref().ok_or(DrawState(Reason::PipelineAttachments))?;
        let samples = pipeline.closed_samples.ok_or(DrawState(Reason::PipelineAttachments))?.Count;
        if formats.len() != targets.len() {
            return Err(DrawState(Reason::AttachmentCount { expected: formats.len(), actual: targets.len() }));
        }
        if targets.is_empty() && depth.is_none() {
            return Err(DrawState(Reason::NoAttachments));
        }
        let actual_depth = depth.as_ref().map(|d| d.format);
        if pipeline.closed_depth_format != actual_depth {
            return Err(DrawState(Reason::DepthFormat {
                expected: pipeline.closed_depth_format.map(|format| format.0),
                actual: actual_depth.map(|format| format.0),
            }));
        }
        if let Some(actual) = depth.iter().map(|d| d.samples).chain(targets.iter().map(|t| t.samples))
            .find(|&actual| actual != samples)
        {
            return Err(DrawState(Reason::SampleCount { expected: samples, actual }));
        }
        let shader = ShaderBindings::capture(pipeline, values, view, address, None)?;
        if depth.as_ref().is_some_and(|d| {
            pipeline.closed_depth_tests & !d.planes != 0
                || pipeline.closed_depth_writes & d.read_only_planes != 0
                || shader.samples_target(&d.copy)
                    && (d.read_only_planes & 1 == 0 || !shader.depth_reads_are_exact(&d.copy))
        }) {
            return Err(DrawState(Reason::DepthPlaneConflict));
        }
        let extent = |desc: D3D12_RESOURCE_DESC| (desc.Width, desc.Height);
        let mut first_extent = None;
        for (index, (target, &format)) in targets.iter().zip(formats).enumerate() {
            let view_format = target.format;
            if format == windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN
                || view_format != format
            {
                return Err(DrawState(Reason::ColorFormat { slot: index, expected: format.0, actual: view_format.0 }));
            }
            if shader.conflicts_color(&target.copy, target.sub)
                || depth.as_ref().is_some_and(|d| d.copy.texture == target.copy.texture)
            {
                return Err(DrawState(Reason::AttachmentAlias { slot: index }));
            }
            if targets.len() > 1 || target.sub.is_scoped() || samples > 1 {
                if target.copy.uses_unproven_alias_backing()
                    || targets[..index].iter().any(|other| other.copy.texture == target.copy.texture)
                    || *first_extent.get_or_insert_with(|| target.extent()) != target.extent()
                    || depth.as_ref().is_some_and(|d| d.copy.uses_unproven_alias_backing()
                        || extent(unsafe { d.copy.texture.GetDesc() }) != target.extent())
                {
                    return Err(DrawState(Reason::AttachmentBackingOrExtent { slot: index }));
                }
            }
        }
        Ok(Self {
            pipeline: pipeline.pso.clone(), root: pipeline.root.clone(),
            targets, depth, depth_tests: pipeline.closed_depth_tests, depth_writes: pipeline.closed_depth_writes,
            stencil_ref, topology, viewport, scissor, inputs, call, shader,
            blend_factor: [1.0; 4],
            sample: None,
        })
    }

    pub(crate) fn set_blend_factor(&mut self, factor: [f32; 4]) {
        self.blend_factor = factor;
    }

    pub(crate) fn ranges(&self, ranges: &mut Vec<(u64, u64)>) -> Result<(), BatchError> {
        for input in &self.inputs {
            ranges.push(input.range()?);
        }
        if let DrawCall::Indirect(draw) = self.call {
            draw.ranges(ranges)?;
        }
        if let Some(sample) = &self.sample {
            ranges.extend(sample.input.rows()?);
        }
        self.shader.ranges(ranges)?;
        Ok(())
    }

    pub(crate) fn encode(
        &self,
        host: &Host,
        list: &HostList,
        batch: &PinnedBatch,
        budget: &mut Option<ID3D12Resource>,
        descriptors: &mut crate::page_queue::PageDescriptors<'_>,
        initialized: &mut ResourcePlan,
        dirty: &mut std::collections::BTreeSet<victory_memory::physical::PhysicalPage>,
    ) -> Result<(), BatchError> {
        if let Some(depth) = &self.depth {
            depth.validate_backing()?;
            initialized.require_depth_planes(&depth.copy, self.depth_tests)?;
        }
        let sampled_depth = self.depth.as_ref().is_some_and(|d| self.shader.samples_target(&d.copy));
        if let Some(depth) = &self.depth && sampled_depth && !self.shader.depth_reads_are_exact(&depth.copy) {
            return Err(BatchError::UnsupportedDepth);
        }
        // New guest views of a native allocation can be created after
        // list capture. Recheck backing restrictions under the queue lock.
        if self.targets.len() > 1
            && (self.targets.iter().any(|t| t.copy.uses_unproven_alias_backing())
                || self.depth.as_ref().is_some_and(|d| d.copy.uses_unproven_alias_backing()))
        {
            return Err(BatchError::unsupported_texture_map("无法确认纹理映射"));
        }
        for target in &self.targets {
            target.validate_native()?;
            initialized.require_color_subresource(host, list, batch, &target.copy, target.sub)?;
        }
        let indirect = match self.call {
            DrawCall::Indirect(draw) => Some(draw.prepare(host, list, batch, budget)?),
            _ => None,
        };
        let sample = self.sample.as_ref()
            .map(|input| input.prepare(host, list, batch, descriptors)).transpose()?;
        let color_targets: Vec<_> = self.targets.iter().map(|target| target.copy.as_ref()).collect();
        let mut slab_reads = crate::page_view::SlabReads::default();
        let words = self.shader.prepare(host, list, batch, descriptors, initialized, dirty,
            &color_targets, &mut slab_reads, false)?;
        let uses_uav = self.shader.uses_uav();
        for &input in &self.inputs {
            if uses_uav {
                input.encode(host, list, batch)?;
            } else {
                input.encode_scoped(host, list, batch, &mut slab_reads)?;
            }
        }
        list.keep(&self.pipeline);
        list.keep(&self.root);
        let render_target = D3D12_RESOURCE_STATE_RENDER_TARGET;
        let mut rtvs = [D3D12_CPU_DESCRIPTOR_HANDLE::default(); 8];
        let mut states = [render_target; 8];
        for (slot, target) in self.targets.iter().enumerate() {
            let texture = &target.copy.texture;
            states[slot] = initialized.state(&target.copy);
            list.keep(texture);
            rtvs[slot] = host.rtv(texture, target.desc.as_ref());
        }
        let dsv = self.depth.as_ref().map(|d| d.begin(host, list, initialized, sampled_depth));
        // SAFETY: the proven draw owns its pipeline, root and target. The
        // inputs are bound from admitted pages or retained copies of them.
        unsafe {
            let cmd = list.recording();
            // This encoder runs under the direct queue submission lock.
            // Restore the prior state in this list without speculative
            // changes to global state if a later operation rejects it.
            for (target, before) in self.targets.iter().zip(states) {
                if before != render_target {
                    cmd.ResourceBarrier(&[target.sub.temporary_transition(&target.copy.texture, before, render_target)]);
                }
            }
            words.shared_access.transition(cmd, false);
            slab_reads.transition(cmd, false);
            slab_reads.bind(list);
            cmd.SetDescriptorHeaps(&[Some(descriptors.heap.heap.clone())]);
            cmd.SetGraphicsRootSignature(&self.root);
            cmd.SetPipelineState(&self.pipeline);
            if let Some(sample) = &sample {
                sample.bind(cmd, descriptors);
            }
            words.bind(cmd, false);
            cmd.IASetPrimitiveTopology(self.topology);
            cmd.RSSetViewports(&[self.viewport]);
            cmd.RSSetScissorRects(&[self.scissor]);
            cmd.OMSetBlendFactor(Some(&self.blend_factor));
            cmd.OMSetStencilRef(self.stencil_ref);
            cmd.OMSetRenderTargets(
                self.targets.len() as u32,
                (!self.targets.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(|(h, _)| std::ptr::from_ref(h)),
            );
            match self.call {
                DrawCall::Direct { vertices, instances, first, first_instance } => {
                    cmd.DrawInstanced(vertices, instances, first, first_instance);
                }
                DrawCall::Indexed { indices, instances, first, base, first_instance } => {
                    cmd.DrawIndexedInstanced(indices, instances, first, base, first_instance);
                }
                DrawCall::Indirect(_) => indirect.as_ref().expect("indirect arguments prepared").execute(cmd),
            }
            slab_reads.transition(cmd, true);
            words.shared_access.transition(cmd, true);
            for (target, before) in self.targets.iter().zip(states) {
                if before != render_target {
                    cmd.ResourceBarrier(&[target.sub.temporary_transition(&target.copy.texture, render_target, before)]);
                }
            }
        }
        if let (Some(depth), Some((_, scope))) = (&self.depth, dsv) {
            scope.transition(list.recording(), &depth.copy.texture, true);
            initialized.record_depth_writes(&depth.copy, self.depth_writes)?;
        }
        for target in &self.targets {
            initialized.record_color_write(&target.copy, target.sub)?;
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "diagnostics"))]
#[test]
fn paged_title_draw_reads_gpu_copied_vertices_and_renders_pixels() {
    check_paged_title_pixels(false, false, false);
}

#[cfg(all(test, feature = "diagnostics"))]
#[test]
fn paged_title_indirect_draw_reads_gpu_generated_records_and_count() {
    check_paged_title_pixels(true, false, false);
}

#[cfg(all(test, feature = "diagnostics"))]
#[test]
fn paged_title_draw_samples_texture_written_by_preceding_gpu_copy() {
    check_paged_title_pixels(false, true, false);
}

#[cfg(all(test, feature = "diagnostics"))]
#[test]
fn paged_title_draw_uses_relocated_texture_table_byte_base() {
    check_paged_title_pixels(false, true, true);
}

#[cfg(all(test, feature = "diagnostics"))]
fn compile_dxc_stage(source: &str, entry: &str, profile: &str) -> Vec<u8> {
    let dxc = std::env::var_os("VICTORY_PC_DXC").map(std::path::PathBuf::from)
        .or_else(|| {
            let mut paths: Vec<_> = std::fs::read_dir("C:/Program Files (x86)/Windows Kits/10/bin")
                .ok()?.filter_map(Result::ok)
                .map(|p| p.path().join("x64/dxc.exe"))
                .filter(|p| p.is_file()).collect();
            paths.sort();
            paths.pop()
        })
        .expect("Windows SDK dxc.exe is required for the paged direct-index test");
    let temp = tempfile::tempdir().unwrap();
    let src = temp.path().join("shader.hlsl");
    let blob = temp.path().join("shader.dxil");
    std::fs::write(&src, source).unwrap();
    let output = std::process::Command::new(dxc).args(["-T", profile, "-E", entry, "-Fo"])
        .arg(&blob).arg(&src).output().unwrap();
    assert!(output.status.success(), "DXC: {}", String::from_utf8_lossy(&output.stderr));
    std::fs::read(blob).unwrap()
}

#[cfg(all(test, feature = "diagnostics"))]
fn check_paged_title_pixels(indirect: bool, sampled: bool, relocated: bool) {
    use victory_memory::{PAGE_BYTES, physical::PageNumbers};
    use windows::Win32::Graphics::Direct3D::{
        D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, ID3DBlob, Fxc::D3DCompile,
    };
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::core::s;

    use crate::page_view::VertexView;

    let host = crate::host::host().expect("D3D12 renderer host (CI selects WARP)");
    let heap = crate::memory::heap().unwrap();
    let device = &host.device;
    let source = if sampled {
        r#"
float4 vertex(float2 p : POSITION) : SV_Position { return float4(p, 0, 1); }
Texture2D<float4> colors : register(t0);
SamplerState nearest : register(s0);
float4 pixel() : SV_Target0 { return colors.SampleLevel(nearest, float2(0.5, 0.5), 0); }
"#
    } else {
        r#"
float4 vertex(float2 p : POSITION) : SV_Position { return float4(p, 0, 1); }
cbuffer Color : register(b0) { float4 green; }
float4 pixel() : SV_Target0 { return green; }
"#
    };
    let compile = |entry, profile| {
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: HLSL and entry/profile literals are live during compile.
        unsafe {
            D3DCompile(source.as_ptr().cast(), source.len(), s!("page_draw"), None, None,
                entry, profile, 0, 0, &mut code, None).unwrap();
        }
        code.unwrap()
    };
    let fxc_vs = (!relocated).then(|| compile(s!("vertex"), s!("vs_5_0")));
    // Keep both stages in the same shader generation. DXBC and DXIL
    // stage signatures cannot be linked in one graphics PSO.
    let dxc_vs = relocated.then(|| compile_dxc_stage(
        "float4 vertex(float2 p : POSITION) : SV_Position { return float4(p, 0, 1); }",
        "vertex", "vs_6_6",
    ));
    let fxc_ps = (!relocated).then(|| compile(s!("pixel"), s!("ps_5_0")));
    let dxc_ps = relocated.then(|| compile_dxc_stage(r#"
cbuffer Table : register(b0) { uint table_bytes; }
SamplerState nearest : register(s0);
float4 pixel() : SV_Target0 {
    Texture2D<float4> colors = ResourceDescriptorHeap[(table_bytes >> 5) + 3];
    return colors.SampleLevel(nearest, float2(0.5, 0.5), 0);
}
"#, "pixel", "ps_6_6"));
    const FORMAT: DXGI_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;
    // SAFETY: creation descriptors and shader blobs live through calls.
    let (root, pso, target): (ID3D12RootSignature, ID3D12PipelineState, ID3D12Resource) = unsafe {
        let mut serialized: Option<ID3DBlob> = None;
        let constants = D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 4 },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        };
        let range = D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV, NumDescriptors: 1,
            ..Default::default()
        };
        let view = D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1, pDescriptorRanges: &range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        };
        let sampler = D3D12_STATIC_SAMPLER_DESC {
            Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            MaxAnisotropy: 1,
            ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
            MaxLOD: f32::MAX,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            ..Default::default()
        };
        if relocated {
            let parameter = D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER1_0 { Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 4,
                } },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            };
            D3D12SerializeVersionedRootSignature(
                &D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
                    Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
                    Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                        Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                            NumParameters: 1, pParameters: &parameter,
                            NumStaticSamplers: 1, pStaticSamplers: &sampler,
                            Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
                                | D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED,
                        },
                    },
                },
                &mut serialized,
                None,
            ).unwrap();
        } else {
            D3D12SerializeRootSignature(
                &D3D12_ROOT_SIGNATURE_DESC {
                    NumParameters: 1,
                    pParameters: if sampled { &view } else { &constants },
                    NumStaticSamplers: u32::from(sampled),
                    pStaticSamplers: if sampled { &sampler } else { std::ptr::null() },
                    Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
                },
                D3D_ROOT_SIGNATURE_VERSION_1,
                &mut serialized,
                None,
            ).unwrap();
        }
        let serialized = serialized.unwrap();
        let bytes = std::slice::from_raw_parts(
            serialized.GetBufferPointer().cast::<u8>(), serialized.GetBufferSize(),
        );
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes).unwrap();
        let input = D3D12_INPUT_ELEMENT_DESC {
            SemanticName: s!("POSITION"),
            Format: DXGI_FORMAT_R32G32_FLOAT,
            ..Default::default()
        };
        let mut blend = D3D12_BLEND_DESC::default();
        blend.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(root.clone())),
            VS: if let Some(vs) = &dxc_vs {
                D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr().cast(), BytecodeLength: vs.len() }
            } else {
                let vs = fxc_vs.as_ref().unwrap();
                D3D12_SHADER_BYTECODE { pShaderBytecode: vs.GetBufferPointer(), BytecodeLength: vs.GetBufferSize() }
            },
            PS: if let Some(ps) = &dxc_ps {
                D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr().cast(), BytecodeLength: ps.len() }
            } else {
                let ps = fxc_ps.as_ref().unwrap();
                D3D12_SHADER_BYTECODE { pShaderBytecode: ps.GetBufferPointer(), BytecodeLength: ps.GetBufferSize() }
            },
            BlendState: blend,
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: &input, NumElements: 1,
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            RTVFormats: [FORMAT, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN, DXGI_FORMAT_UNKNOWN],
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let pso = device.CreateGraphicsPipelineState(&desc);
        let _ = std::mem::ManuallyDrop::into_inner(desc.pRootSignature);
        let pso = pso.unwrap();
        let mut target = None;
        device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 4, Height: 4, DepthOrArraySize: 1, MipLevels: 1,
                Format: FORMAT, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            None,
            &mut target,
        ).unwrap();
        (root, pso, target.unwrap())
    };
    let clear = host.new_list().unwrap();
    // SAFETY: the target starts in render-target state.
    unsafe {
        clear.recording().ClearRenderTargetView(host.rtv(&target, None), &[1., 0., 0., 1.], None);
        clear.recording().ResourceBarrier(&[transition(
            &target, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
    }
    clear.close();
    host.execute(&[&clear]);

    const VA: usize = PAGE_BYTES as usize;
    const ALIAS: usize = 0x85_0000_0000;
    let id = PageNumbers::default().allocate(6).unwrap().offset(5);
    let (mut backing, section) = crate::title_backing::test_capture(id, &[VA, ALIAS]);
    {
        let mut bytes = section.lock().unwrap();
        bytes[..24].fill(0); // Degenerate triangle before the GPU copy.
        bytes[64..100].fill(0); // Zero count and args before GPU copies.
        bytes[160..164].copy_from_slice(&[0, 0, 255, 255]); // old blue texel
        bytes[176..180].copy_from_slice(&[0, 255, 0, 255]); // GPU will copy green

        for (slot, value) in bytes[112..132].chunks_exact_mut(4).zip([3u32, 1, 0, 0, 1]) {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        for (slot, value) in bytes[32..56].chunks_exact_mut(4)
            .zip([-1.0f32, -1., -1., 3., 3., -1.])
        {
            slot.copy_from_slice(&value.to_le_bytes());
        }
    }
    let target_state = Arc::new(HostTexture::new(
        target.clone(), D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32, Vec::new(),
    ));
    assert!(ColorTarget::capture(target_state.clone(), None).is_none());
    target_state.host_written.store(true, Ordering::Release);
    let color = ColorTarget::capture(target_state.clone(), None).unwrap();
    let title = host.new_list().unwrap();
    title.defer_copy(VA as *mut u8, (ALIAS + 32) as *const u8, 24);
    let call = if indirect {
        title.defer_copy((VA + 64) as *mut u8, (ALIAS + 112) as *const u8, 16);
        title.defer_copy((VA + 96) as *mut u8, (ALIAS + 128) as *const u8, 4);
        DrawCall::Indirect(crate::page_indirect::IndirectArgs {
            kind: crate::page_indirect::IndirectKind::Draw, stride: 16, max: 2,
            arguments: (VA + 64) as u64, count: Some((VA + 96) as u64),
        })
    } else {
        DrawCall::Direct { vertices: 3, instances: 1, first: 0, first_instance: 0 }
    };
    let mut draw = ClosedDraw::from_proven_native(
        pso, root, color, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
        D3D12_VIEWPORT { Width: 4.0, Height: 4.0, MaxDepth: 1.0, ..Default::default() },
        RECT { left: 0, top: 0, right: 4, bottom: 4 },
        vec![InputView::Vertex(0, VertexView { address: VA as u64, size: 24, stride: 8 })],
        call,
    );
    if sampled {
        use crate::page_texture::{LinearTexture, SampledTexture};
        title.defer_copy((VA + 160) as *mut u8, (ALIAS + 176) as *const u8, 4);
        let input = LinearTexture {
            address: (VA + 160) as u64, row_pitch: 4, width: 1, height: 1,
            format: FORMAT,
        };
        if relocated {
            let mut native = None;
            // SAFETY: 测试创建独立、未初始化的一像素纹理作为线性源的身份。
            unsafe {
                device.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                        Width: 1, Height: 1, DepthOrArraySize: 1, MipLevels: 1,
                        Format: FORMAT, SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COMMON, None, &mut native,
                ).unwrap();
            }
            let copy = Arc::new(HostTexture::new(native.unwrap(), 0, Vec::new()));
            draw.shader.record_test_texture(7, 3, crate::page_texture::TextureInput::Linear { copy, input });
            // 未重定位的表会读取错误的描述符槽位。
            draw.shader.constants.push(RootWords::test_words(0, vec![0; 4], vec![(0, 7)]));
        } else {
            draw.sample = Some(SampledTexture { root: 0, input });
        }
    } else {
        draw.shader.constants.push(RootWords::test_words(
            0, [0.0f32, 1.0, 0.0, 1.0].into_iter().map(f32::to_bits).collect(), Vec::new(),
        ));
    }
    title.record_page_draw(draw);
    title.close();
    let stats = host.execute_paged_title(heap, &mut backing, &[&title]).unwrap();
    assert_eq!((stats.uploaded, stats.downloaded), (1, 1));
    assert_eq!(target_state.state.load(Ordering::Acquire), D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);

    // The completed draw must have restored the recorded copy-source state.
    let read = host.new_list().unwrap();
    let (readback, footprint) = crate::capture::read_back(device, read.recording(), &target).unwrap();
    read.close();
    host.execute(&[&read]);
    // SAFETY: the readback has completed and the footprint bounds the pixel.
    unsafe {
        let mut ptr = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut ptr)).unwrap();
        let row = ptr.cast::<u8>().add(footprint.Offset as usize + 2 * footprint.Footprint.RowPitch as usize);
        assert_eq!(std::slice::from_raw_parts(row.add(8), 4), &[0, 255, 0, 255]);
        readback.Unmap(0, None);
    }
}
