//! First closed draw form for the page renderer: one native color target
//! and fixed-function inputs. No descriptor or shader-side guest accesses.
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;

use std::collections::BTreeMap;
use std::sync::{Arc, atomic::Ordering};

use crate::host::{Host, HostList, transition};
use crate::page_shader::{RootWords, UnloweredShader};
use crate::page_slabs::{BatchError, PinnedBatch};
use crate::page_view::InputView;
use crate::resource::HostTexture;

pub(crate) enum DrawCall {
    Direct { vertices: u32, instances: u32, first: u32, first_instance: u32 },
    Indexed { indices: u32, instances: u32, first: u32, base: i32, first_instance: u32 },
    Indirect(crate::page_indirect::IndirectDraw),
}

impl DrawCall {
    pub(crate) fn indexed(&self) -> bool {
        match self {
            Self::Direct { .. } => false,
            Self::Indexed { .. } => true,
            Self::Indirect(draw) => draw.indexed,
        }
    }
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
    sample: Option<crate::page_texture::SampledTexture>,
    texture_tables: BTreeMap<u8, BTreeMap<u32, crate::page_texture::LinearTexture>>,
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
            sample: None,
            texture_tables: BTreeMap::new(),
        }
    }

    /// Admit a real translated title PSO only when every stage gets an
    /// unambiguous captured constant block and has no guest memory views.
    /// The producer is still responsible for complete IA and target state.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_translated(
        pipeline: &crate::pipeline::host::HostPipeline,
        values: impl Fn(u8, u32) -> Option<u32>,
        texture: impl Fn(u8, u32) -> Option<crate::page_texture::LinearTexture>,
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
        let mut texture_tables = BTreeMap::<u8, BTreeMap<u32, _>>::new();
        for stage in &pipeline.stages {
            let t = stage.translated();
            let needs_table = t.host.user_data.iter().any(|u| u.root == victory_shader::RootType::Table);
            let slots = if needs_table {
                stage.shader.static_texture_slots().ok_or(UnloweredShader::DescriptorTable)?
            } else {
                &[]
            };
            if let Some(words) = crate::page_shader::stage_words(
                &t.host, &t.used, stage.user_data, &values,
                |root| slots.iter().any(|s| s.root == root),
            )? {
                for slot in slots {
                    if !words.tables.iter().any(|&(_, root)| root == slot.root) {
                        return Err(UnloweredShader::DescriptorTable);
                    }
                    let table = texture_tables.entry(slot.root).or_default();
                    if let std::collections::btree_map::Entry::Vacant(e) = table.entry(slot.index) {
                        e.insert(texture(slot.root, slot.index).ok_or(UnloweredShader::DescriptorTable)?);
                    }
                }
                constants.push(words);
            } else if !slots.is_empty() {
                return Err(UnloweredShader::DescriptorTable);
            }
        }
        Ok(Self {
            pipeline: pipeline.pso.clone(), root: pipeline.root.clone(),
            target, topology, viewport, scissor, inputs, call, constants,
            blend_factor: [1.0; 4],
            sample: None,
            texture_tables,
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
        for table in self.texture_tables.values() {
            for texture in table.values() {
                ranges.extend(texture.rows()?);
            }
        }
        Ok(())
    }

    pub(crate) fn encode(
        &self,
        host: &Host,
        list: &HostList,
        batch: &PinnedBatch,
        budget: &mut Option<ID3D12Resource>,
        descriptors: &mut crate::page_queue::PageDescriptors<'_>,
    ) -> Result<(), BatchError> {
        let indirect = match self.call {
            DrawCall::Indirect(draw) => Some(draw.prepare(host, list, batch, budget)?),
            _ => None,
        };
        let sample = self.sample.as_ref()
            .map(|input| input.prepare(host, list, batch, descriptors)).transpose()?;
        let mut bases = BTreeMap::new();
        for (&root, table) in &self.texture_tables {
            let size = table.last_key_value().expect("captured slots exist").0 + 1;
            let mut views = vec![None; size as usize];
            for (&index, &texture) in table {
                views[index as usize] = Some((texture.encode(host, list, batch)?, texture.format));
            }
            let first = descriptors.reserve_texture_run(&views).map_err(BatchError::Native)?;
            bases.insert(root, first);
        }
        let words: Vec<_> = self.constants.iter().map(|root| {
            root.relocated(&bases).map(|words| (root.parameter, words))
                .ok_or_else(|| BatchError::Native(windows::Win32::Foundation::E_FAIL.into()))
        }).collect::<Result<_, _>>()?;
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
            if !bases.is_empty() {
                // Directly indexed root signatures require their descriptor
                // heap to be set before the root signature itself.
                cmd.SetDescriptorHeaps(&[Some(descriptors.heap.heap.clone())]);
            }
            cmd.SetGraphicsRootSignature(&self.root);
            cmd.SetPipelineState(&self.pipeline);
            if let Some(sample) = &sample {
                sample.bind(cmd, descriptors);
            }
            for (param, words) in &words {
                cmd.SetGraphicsRoot32BitConstants(
                    *param, words.len() as u32, words.as_ptr().cast(), 0,
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
                DrawCall::Indirect(_) => indirect.as_ref().expect("indirect arguments prepared").draw(cmd),
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
fn compile_relocated_pixel(source: &str) -> Vec<u8> {
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
    let output = std::process::Command::new(dxc).args(["-T", "ps_6_6", "-E", "pixel", "-Fo"])
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
    let vs = compile(s!("vertex"), s!("vs_5_0"));
    let fxc_ps = (!relocated).then(|| compile(s!("pixel"), s!("ps_5_0")));
    let dxc_ps = relocated.then(|| compile_relocated_pixel(r#"
cbuffer Table : register(b0) { uint table_bytes; }
SamplerState nearest : register(s0);
float4 pixel() : SV_Target0 {
    Texture2D<float4> colors = ResourceDescriptorHeap[(table_bytes >> 5) + 3];
    return colors.SampleLevel(nearest, float2(0.5, 0.5), 0);
}
"#));
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
            VS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: vs.GetBufferPointer(), BytecodeLength: vs.GetBufferSize(),
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
        DrawCall::Indirect(crate::page_indirect::IndirectDraw {
            indexed: false, stride: 16, max: 2,
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
            draw.texture_tables.entry(7).or_default().insert(3, input);
            // An unrelocated table would read from the wrong heap slot.
            draw.constants.push(RootWords {
                parameter: 0, words: vec![0; 4], tables: vec![(0, 7)],
            });
        } else {
            draw.sample = Some(SampledTexture { root: 0, input });
        }
    } else {
        draw.constants.push(RootWords {
            parameter: 0,
            words: [0.0f32, 1.0, 0.0, 1.0].into_iter().map(f32::to_bits).collect(),
            tables: Vec::new(),
        });
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
