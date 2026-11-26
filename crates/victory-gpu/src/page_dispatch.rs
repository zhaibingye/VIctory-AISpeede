//! Typed dispatches run in the same ordered queue as draws and copies.
use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{Host, HostList};
use crate::page_indirect::{IndirectArgs, IndirectKind};
use crate::page_queue::PageDescriptors;
use crate::page_shader::{ShaderBindings, UnloweredShader};
use crate::page_slabs::{BatchError, PinnedBatch};
use crate::descriptor::View;
use crate::pipeline::host::HostPipeline;

pub(crate) enum DispatchCall {
    Direct(u32, u32, u32),
    Indirect(IndirectArgs),
}

pub(crate) struct ClosedDispatch {
    pipeline: ID3D12PipelineState,
    root: ID3D12RootSignature,
    shader: ShaderBindings,
    call: DispatchCall,
}

impl ClosedDispatch {
    pub(crate) fn capture(
        pipeline: &HostPipeline,
        constants: impl Fn(u8, u32) -> Option<u32>,
        view: impl Fn(u8, u32) -> Option<View>,
        address: impl Fn(u8) -> Option<u64>,
        call: DispatchCall,
    ) -> Result<Self, UnloweredShader> {
        if !pipeline.compute || pipeline.mesh {
            return Err(UnloweredShader::GraphicsState);
        }
        if let DispatchCall::Direct(x, y, z) = call
            && [x, y, z].into_iter().any(|n| n > 65535)
        {
            return Err(UnloweredShader::GraphicsState);
        }
        if let DispatchCall::Indirect(args) = call
            && args.kind != IndirectKind::Dispatch
        {
            return Err(UnloweredShader::GraphicsState);
        }
        Ok(Self {
            pipeline: pipeline.pso.clone(),
            root: pipeline.root.clone(),
            shader: ShaderBindings::capture(pipeline, constants, view, address)?,
            call,
        })
    }

    pub(crate) fn ranges(&self, ranges: &mut Vec<(u64, u64)>) -> Result<(), BatchError> {
        self.shader.ranges(ranges)?;
        if let DispatchCall::Indirect(args) = self.call {
            args.ranges(ranges)?;
        }
        Ok(())
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn reads_guest_only(&self) -> bool {
        !self.shader.uses_guest_uav()
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn preserved_colors<'a>(&'a self, out: &mut Vec<&'a std::sync::Arc<crate::resource::HostTexture>>) {
        self.shader.pointer_sources(out);
        self.written_targets(out);
    }

    pub(crate) fn written_targets<'a>(&'a self, out: &mut Vec<&'a std::sync::Arc<crate::resource::HostTexture>>) {
        self.shader.written_targets(out);
    }

    pub(crate) fn encode(
        &self, host: &Host, list: &HostList, batch: &PinnedBatch,
        budget: &mut Option<ID3D12Resource>, descriptors: &mut PageDescriptors<'_>,
        plan: &mut crate::page_resource::ResourcePlan,
        dirty: &mut std::collections::BTreeSet<victory_memory::physical::PhysicalPage>,
    ) -> Result<(), BatchError> {
        let args = if let DispatchCall::Indirect(args) = self.call {
            Some(args.prepare(host, list, batch, budget)?)
        } else {
            None
        };
        let words = self.shader.prepare(host, list, batch, descriptors, plan, dirty)?;
        if self.shader.uses_uav() {
            crate::page_resource::PageBarrier::Uav.encode(list, plan)?;
        }
        list.keep(&self.pipeline);
        list.keep(&self.root);
        // SAFETY: the root and pipeline match the captured shader, with
        // validated constants and a retained heap populated before bind.
        unsafe {
            let cmd = list.recording();
            cmd.SetDescriptorHeaps(&[Some(descriptors.heap.heap.clone())]);
            cmd.SetComputeRootSignature(&self.root);
            cmd.SetPipelineState(&self.pipeline);
            words.bind(cmd, true);
            match self.call {
                DispatchCall::Direct(x, y, z) => cmd.Dispatch(x, y, z),
                DispatchCall::Indirect(_) => args.as_ref().expect("indirect args prepared").execute(cmd),
            }
        }
        self.shader.finish_texture_writes(list, plan)
    }
}
