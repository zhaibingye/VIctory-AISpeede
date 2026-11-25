//! First closed draw form for the page renderer: one native color target
//! and fixed-function inputs. No descriptor or shader-side guest accesses.
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;

use std::sync::{Arc, atomic::Ordering};

use crate::host::{Host, HostList, transition};
use crate::page_shader::{RootWords, UnloweredShader};
use crate::page_slabs::{BatchError, PinnedBatch};
use crate::page_view::InputView;
use crate::resource::HostTexture;

pub(crate) enum DrawCall {
    Direct { vertices: u32, instances: u32, first: u32, first_instance: u32 },
    Indexed { indices: u32, instances: u32, first: u32, base: i32, first_instance: u32 },
}

#[derive(Clone)]
pub(crate) struct ColorTarget {
    copy: Arc<HostTexture>,
    desc: Option<D3D12_RENDER_TARGET_VIEW_DESC>,
}

impl ColorTarget {
    /// Only an already-authoritative host render target is admitted here.
    /// Title-memory texture uploads, aliasing and sparse placement must
    /// be lowered before they can participate in this form of draw.
    pub(crate) fn capture(
        copy: Arc<HostTexture>,
        desc: Option<D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> Option<Self> {
        // SAFETY: the retained COM object is live.
        let tex = unsafe { copy.texture.GetDesc() };
        if copy.sparse.is_some()
            || !copy.host_written.load(Ordering::Acquire)
            || tex.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
            || tex.SampleDesc.Count != 1
            || tex.DepthOrArraySize != 1
            || tex.MipLevels != 1
            || tex.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 == 0
            || desc.as_ref().is_some_and(|d| d.ViewDimension != D3D12_RTV_DIMENSION_TEXTURE2D)
        {
            return None;
        }
        Some(Self { copy, desc })
    }
}

pub(crate) struct ClosedDraw {
    pipeline: ID3D12PipelineState,
    root: ID3D12RootSignature,
    target: ColorTarget,
    topology: D3D_PRIMITIVE_TOPOLOGY,
    viewport: D3D12_VIEWPORT,
    scissor: RECT,
    inputs: Vec<InputView>,
    call: DrawCall,
    constants: Vec<RootWords>,
    blend_factor: [f32; 4],
}

impl ClosedDraw {
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
            pipeline, root, target, topology, viewport, scissor, inputs, call,
            constants: Vec::new(),
            blend_factor: [1.0; 4],
        }
    }

    /// Admit a real translated title PSO only when every stage gets an
    /// unambiguous captured constant block and has no guest memory views.
    /// The producer is still responsible for complete IA and target state.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_translated(
        pipeline: &crate::pipeline::host::HostPipeline,
        values: impl Fn(u8, u32) -> Option<u32>,
        target: ColorTarget,
        topology: D3D_PRIMITIVE_TOPOLOGY,
        viewport: D3D12_VIEWPORT,
        scissor: RECT,
        inputs: Vec<InputView>,
        call: DrawCall,
    ) -> Result<Self, UnloweredShader> {
        if pipeline.compute || pipeline.mesh || pipeline.rectangles || pipeline.marks.is_some() {
            return Err(UnloweredShader::UnknownUserData);
        }
        // SAFETY: query the retained native target description.
        let format = target.desc.as_ref().map_or_else(
            || unsafe { target.copy.texture.GetDesc() }.Format, |d| d.Format,
        );
        if pipeline.closed_color_format != Some(format) {
            return Err(UnloweredShader::GraphicsState);
        }
        let mut constants = Vec::new();
        for stage in &pipeline.stages {
            let t = stage.translated();
            if let Some(words) = crate::page_shader::constants_only(
                &t.host, &t.used, stage.user_data, &values,
            )? {
                constants.push(words);
            }
        }
        Ok(Self {
            pipeline: pipeline.pso.clone(), root: pipeline.root.clone(),
            target, topology, viewport, scissor, inputs, call, constants,
            blend_factor: [1.0; 4],
        })
    }

    pub(crate) fn set_blend_factor(&mut self, factor: [f32; 4]) {
        self.blend_factor = factor;
    }

    pub(crate) fn ranges(&self, ranges: &mut Vec<(u64, u64)>) -> Result<(), BatchError> {
        for input in &self.inputs {
            ranges.push(input.range()?);
        }
        Ok(())
    }

    pub(crate) fn encode(
        &self,
        host: &Host,
        list: &HostList,
        batch: &PinnedBatch,
    ) -> Result<(), BatchError> {
        for &input in &self.inputs {
            input.encode(host, list, batch)?;
        }
        list.keep(&self.pipeline);
        list.keep(&self.root);
        let texture = &self.target.copy.texture;
        let before = D3D12_RESOURCE_STATES(self.target.copy.state.load(Ordering::Acquire) as i32);
        let render_target = D3D12_RESOURCE_STATE_RENDER_TARGET;
        list.keep(texture);
        let rtv = host.rtv(texture, self.target.desc.as_ref());
        // SAFETY: the proven draw owns its pipeline, root and target. The
        // inputs were gathered in this same list from admitted pages.
        unsafe {
            let cmd = list.recording();
            // This encoder runs under the direct queue submission lock.
            // Restore the prior state in this list without speculative
            // changes to global state if a later operation rejects it.
            if before != render_target {
                cmd.ResourceBarrier(&[transition(texture, before, render_target)]);
            }
            cmd.SetGraphicsRootSignature(&self.root);
            cmd.SetPipelineState(&self.pipeline);
            for root in &self.constants {
                cmd.SetGraphicsRoot32BitConstants(
                    root.parameter, root.words.len() as u32, root.words.as_ptr().cast(), 0,
                );
            }
            cmd.IASetPrimitiveTopology(self.topology);
            cmd.RSSetViewports(&[self.viewport]);
            cmd.RSSetScissorRects(&[self.scissor]);
            cmd.OMSetBlendFactor(Some(&self.blend_factor));
            cmd.OMSetRenderTargets(1, Some(&rtv), false, None);
            match self.call {
                DrawCall::Direct { vertices, instances, first, first_instance } => {
                    cmd.DrawInstanced(vertices, instances, first, first_instance);
                }
                DrawCall::Indexed { indices, instances, first, base, first_instance } => {
                    cmd.DrawIndexedInstanced(indices, instances, first, base, first_instance);
                }
            }
            if before != render_target {
                cmd.ResourceBarrier(&[transition(texture, render_target, before)]);
            }
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "diagnostics"))]
#[test]
fn paged_title_draw_reads_gpu_copied_vertices_and_renders_pixels() {
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
    let source = r#"
float4 vertex(float2 p : POSITION) : SV_Position { return float4(p, 0, 1); }
cbuffer Color : register(b0) { float4 green; }
float4 pixel() : SV_Target0 { return green; }
"#;
    let compile = |entry, profile| {
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: HLSL and entry/profile literals are live during compile.
        unsafe {
            D3DCompile(source.as_ptr().cast(), source.len(), s!("page_draw"), None, None,
                entry, profile, 0, 0, &mut code, None).unwrap();
        }
        code.unwrap()
    };
    let vs = compile(s!("vertex"), s!("vs_5_0"));
    let ps = compile(s!("pixel"), s!("ps_5_0"));
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
        D3D12SerializeRootSignature(
            &D3D12_ROOT_SIGNATURE_DESC {
                NumParameters: 1,
                pParameters: &constants,
                Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
                ..Default::default()
            },
            D3D_ROOT_SIGNATURE_VERSION_1,
            &mut serialized,
            None,
        ).unwrap();
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
            VS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: vs.GetBufferPointer(), BytecodeLength: vs.GetBufferSize(),
            },
            PS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: ps.GetBufferPointer(), BytecodeLength: ps.GetBufferSize(),
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
    let mut draw = ClosedDraw::from_proven_native(
        pso, root, color, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
        D3D12_VIEWPORT { Width: 4.0, Height: 4.0, MaxDepth: 1.0, ..Default::default() },
        RECT { left: 0, top: 0, right: 4, bottom: 4 },
        vec![InputView::Vertex(0, VertexView { address: VA as u64, size: 24, stride: 8 })],
        DrawCall::Direct { vertices: 3, instances: 1, first: 0, first_instance: 0 },
    );
    draw.constants.push(RootWords {
        parameter: 0,
        words: [0.0f32, 1.0, 0.0, 1.0].into_iter().map(f32::to_bits).collect(),
    });
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
