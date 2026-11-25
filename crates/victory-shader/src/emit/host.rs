//! Host D3D12 checks for emitted shaders: pipeline creation and, for
//! tests, drawing a full-screen triangle and reading the target back.

use std::ffi::CString;

use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::core::{Interface, PCSTR};

use super::HostShader;
use super::container::SigElement;
use crate::dxbc::{Container, FourCC};

pub struct Host {
    pub device: ID3D12Device,
}

/// What the stage declares that a pipeline must match.
fn sig(dxil: &[u8], fourcc: FourCC) -> Result<Vec<SigElement>, String> {
    let c = Container::parse(dxil).map_err(|e| e.to_string())?;
    match c.part(fourcc) {
        Some(p) => super::signature(p.data).map_err(|e| e.to_string()),
        None => Ok(Vec::new()),
    }
}

fn format(e: &SigElement) -> DXGI_FORMAT {
    let i = usize::from(e.cols.clamp(1, 4)) - 1;
    match e.comp {
        9 => [
            DXGI_FORMAT_R32_FLOAT,
            DXGI_FORMAT_R32G32_FLOAT,
            DXGI_FORMAT_R32G32B32_FLOAT,
            DXGI_FORMAT_R32G32B32A32_FLOAT,
        ][i],
        4 => [
            DXGI_FORMAT_R32_SINT,
            DXGI_FORMAT_R32G32_SINT,
            DXGI_FORMAT_R32G32B32_SINT,
            DXGI_FORMAT_R32G32B32A32_SINT,
        ][i],
        _ => [
            DXGI_FORMAT_R32_UINT,
            DXGI_FORMAT_R32G32_UINT,
            DXGI_FORMAT_R32G32B32_UINT,
            DXGI_FORMAT_R32G32B32A32_UINT,
        ][i],
    }
}

fn target_format(e: &SigElement) -> DXGI_FORMAT {
    match e.comp {
        9 => DXGI_FORMAT_R32G32B32A32_FLOAT,
        4 => DXGI_FORMAT_R32G32B32A32_SINT,
        _ => DXGI_FORMAT_R32G32B32A32_UINT,
    }
}

fn err(e: windows::core::Error) -> String {
    e.message().to_string()
}

/// The first adapter that is a GPU: not flagged software, and with video
/// memory of its own. A machine without one (GitHub's Windows runner has only
/// `Microsoft Basic Render Driver`) has nothing to run the hardware-only
/// checks on, and CI proved that the flag of adapter 0 alone is not enough:
/// the runner's default adapter is not flagged software and its device still
/// refuses a wave of 32
/// (`permutes_read_the_lane_named_and_zero_from_lanes_left_out`). WARP and the
/// basic render driver report no dedicated video memory, which is what this
/// asks for as well.
fn hardware_adapter(factory: &IDXGIFactory4) -> windows::core::Result<IDXGIAdapter1> {
    // SAFETY: plain DXGI adapter enumeration.
    unsafe {
        for index in 0.. {
            let Ok(adapter) = factory.EnumAdapters1(index) else {
                break;
            };
            let desc = adapter.GetDesc1()?;
            let software = desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0;
            if !software && desc.DedicatedVideoMemory > 0 {
                return Ok(adapter);
            }
        }
    }
    Err(windows::core::Error::new(
        windows::Win32::Foundation::E_FAIL,
        "no hardware adapter",
    ))
}

/// The shader-visible view heap size every host takes (resource binding
/// tiers 1 and 2), as `victory_gpu`'s host has it.
const PORTABLE_VIEWS: u32 = 1_000_000;

impl Host {
    /// The default adapter, or WARP with `warp` (or `VICTORY_WARP=1`).
    /// A software default adapter (a machine without a GPU, as CI) is
    /// refused, so hardware-only checks never run on WARP.
    pub fn new(warp: bool) -> Result<Self, String> {
        let warp = warp || std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1");
        // SAFETY: plain D3D12/DXGI calls with valid out pointers.
        unsafe {
            let mut device: Option<ID3D12Device> = None;
            let factory: IDXGIFactory4 =
                CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).map_err(err)?;
            let adapter: IDXGIAdapter1 = if warp {
                factory.EnumWarpAdapter().map_err(err)?
            } else {
                hardware_adapter(&factory).map_err(err)?
            };
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).map_err(err)?;
            let device = device.ok_or("no device")?;
            let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            device
                .CheckFeatureSupport(
                    D3D12_FEATURE_SHADER_MODEL,
                    (&raw mut sm).cast(),
                    size_of_val(&sm) as u32,
                )
                .map_err(err)?;
            if sm.HighestShaderModel.0 < D3D_SHADER_MODEL_6_6.0 {
                return Err(format!(
                    "the device supports SM {:#x} only",
                    sm.HighestShaderModel.0
                ));
            }
            // A device that caps its view heaps below `HEAP_VIEWS` (NVIDIA:
            // 1,000,000) lowers the limit translated shaders clamp to, as
            // the runtime host does (`victory_gpu`'s `probe_view_limit`), so
            // heaps made here and shaders translated after agree.
            // shortcut: shaders translated before the first `Host::new`
            // keep the limit they were made with.
            let heap = |count: u32| {
                device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: count,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
            };
            if heap(super::HEAP_VIEWS).is_err() && heap(PORTABLE_VIEWS).is_ok() {
                super::set_view_heap_limit(PORTABLE_VIEWS);
            }
            Ok(Host { device })
        }
    }

    /// The root signature translated shaders expect: user data per stage
    /// at `b0`, immediate samplers per stage at `s<n>`, heaps directly
    /// indexed.
    pub fn root_signature(
        &self,
        vs: &HostShader,
        ps: &HostShader,
    ) -> Result<ID3D12RootSignature, String> {
        self.root_signature_for(&[
            (vs, D3D12_SHADER_VISIBILITY_VERTEX),
            (ps, D3D12_SHADER_VISIBILITY_PIXEL),
        ])
    }

    /// The root signature for these stages: user data of stage k is root
    /// parameter k, root views follow (by root parameter, CBV before SRV
    /// before UAV), then the title-memory page table and written-pages
    /// bitmap if a stage reads or writes title memory.
    pub fn root_signature_for(
        &self,
        stages: &[(&HostShader, D3D12_SHADER_VISIBILITY)],
    ) -> Result<ID3D12RootSignature, String> {
        let mut params: Vec<D3D12_ROOT_PARAMETER1> = stages
            .iter()
            .map(|&(s, vis)| D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: super::USER_DATA_SPACE,
                        Num32BitValues: s.user_data_dwords().max(1),
                    },
                },
                ShaderVisibility: vis,
            })
            .collect();
        // (slot, kind): 0 CBV, 1 SRV, 2 UAV.
        let mut views: Vec<(u8, u8)> = stages
            .iter()
            .flat_map(|(s, _)| {
                let cbvs = s.root_cbvs.iter().map(|&r| (r, 0));
                let srvs = s.root_srvs.iter().map(|&r| (r, 1));
                cbvs.chain(srvs).chain(s.root_uavs.iter().map(|&r| (r, 2)))
            })
            .collect();
        views.sort_unstable();
        views.dedup();
        let mut views: Vec<(u32, u32, u8)> = views
            .into_iter()
            .map(|(slot, kind)| (u32::from(slot), super::ROOT_CBV_SPACE, kind))
            .collect();
        if stages.iter().any(|(s, _)| s.memory) {
            views.push((0, super::MEMORY_SPACE, 1));
        }
        if stages.iter().any(|(s, _)| s.memory_writes) {
            views.push((0, super::MEMORY_SPACE, 2));
        }
        if stages.iter().any(|(s, _)| s.view_addresses) {
            views.push((1, super::MEMORY_SPACE, 1));
        }
        if stages.iter().any(|(s, _)| s.view_marks) {
            views.push((super::VIEW_MARKS, super::MEMORY_SPACE, 1));
            views.push((super::VIEW_WRITES, super::MEMORY_SPACE, 2));
        }
        for (register, space, kind) in views {
            params.push(D3D12_ROOT_PARAMETER1 {
                ParameterType: match kind {
                    0 => D3D12_ROOT_PARAMETER_TYPE_CBV,
                    1 => D3D12_ROOT_PARAMETER_TYPE_SRV,
                    _ => D3D12_ROOT_PARAMETER_TYPE_UAV,
                },
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                        ShaderRegister: register,
                        RegisterSpace: space,
                        Flags: D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            });
        }
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS19::default();
        // SAFETY: the feature query writes the structure of the given size.
        // Older runtimes may not recognize OPTIONS19; that means no support.
        let aniso_point_mip_supported = unsafe {
            self.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS19,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
        }
        .is_ok()
            && options.AnisoFilterWithPointMipSupported.as_bool();
        let mut samplers = Vec::new();
        for &(s, vis) in stages {
            for (n, d) in s.static_samplers.iter().enumerate() {
                let [u, v, w] = d.address.map(|a| D3D12_TEXTURE_ADDRESS_MODE(a as i32));
                samplers.push(D3D12_STATIC_SAMPLER_DESC {
                    Filter: D3D12_FILTER(d.filter_for_host(aniso_point_mip_supported) as i32),
                    AddressU: u,
                    AddressV: v,
                    AddressW: w,
                    MipLODBias: d.mip_lod_bias,
                    MaxAnisotropy: d.max_anisotropy,
                    ComparisonFunc: D3D12_COMPARISON_FUNC(d.comparison as i32),
                    BorderColor: D3D12_STATIC_BORDER_COLOR(d.border as i32),
                    MinLOD: d.min_lod,
                    MaxLOD: d.max_lod,
                    ShaderRegister: n as u32,
                    RegisterSpace: super::SAMPLER_SPACE,
                    ShaderVisibility: vis,
                });
            }
        }
        let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: params.len() as u32,
                    pParameters: params.as_ptr(),
                    NumStaticSamplers: samplers.len() as u32,
                    pStaticSamplers: samplers.as_ptr(),
                    Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
                        | D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
                        | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED,
                },
            },
        };
        // SAFETY: the description points at live arrays.
        unsafe {
            let mut blob = None;
            let mut error = None;
            D3D12SerializeVersionedRootSignature(&desc, &mut blob, Some(&mut error))
                .map_err(err)?;
            let blob: ID3DBlob = blob.ok_or("no root signature blob")?;
            let bytes = std::slice::from_raw_parts(
                blob.GetBufferPointer().cast::<u8>(),
                blob.GetBufferSize(),
            );
            self.device.CreateRootSignature(0, bytes).map_err(err)
        }
    }

    /// A compute pipeline; user data is root parameter 0.
    pub fn compute_pipeline(
        &self,
        cs: &HostShader,
    ) -> Result<(ID3D12RootSignature, ID3D12PipelineState), String> {
        let root = self.root_signature_for(&[(cs, D3D12_SHADER_VISIBILITY_ALL)])?;
        let pso = self.compute_pipeline_with(&root, &cs.dxil)?;
        Ok((root, pso))
    }

    pub fn compute_pipeline_with(
        &self,
        root: &ID3D12RootSignature,
        cs: &[u8],
    ) -> Result<ID3D12PipelineState, String> {
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            // SAFETY: a borrowed pointer the descriptor never releases.
            pRootSignature: unsafe { std::mem::transmute_copy(root) },
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: cs.as_ptr().cast(),
                BytecodeLength: cs.len(),
            },
            ..Default::default()
        };
        // SAFETY: the description points at live data.
        unsafe { self.device.CreateComputePipelineState(&desc) }.map_err(err)
    }

    /// A graphics pipeline from a vertex and a pixel shader, with float or
    /// integer 32-bit targets as the pixel shader writes them.
    pub fn pipeline(
        &self,
        vs: &HostShader,
        ps: &HostShader,
    ) -> Result<(ID3D12RootSignature, ID3D12PipelineState), String> {
        let root = self.root_signature(vs, ps)?;
        let pso = self.pipeline_with(&root, &vs.dxil, &ps.dxil)?;
        Ok((root, pso))
    }

    /// A graphics pipeline with the given root signature; vertex inputs
    /// packed in register order.
    pub fn pipeline_with(
        &self,
        root: &ID3D12RootSignature,
        vs: &[u8],
        ps: &[u8],
    ) -> Result<ID3D12PipelineState, String> {
        self.pipeline_gs(root, vs, &[], ps)
    }

    /// As [`Host::pipeline_with`], with a geometry shader (none if empty).
    pub fn pipeline_gs(
        &self,
        root: &ID3D12RootSignature,
        vs: &[u8],
        gs: &[u8],
        ps: &[u8],
    ) -> Result<ID3D12PipelineState, String> {
        self.graphics_pipeline(root, vs, gs, ps, None)
    }

    /// As [`Host::pipeline_with`], testing depth against a D32 buffer with
    /// `func` (no depth writes), for [`Host::draw_behind_depth`].
    #[cfg(test)]
    pub fn depth_tested_pipeline(
        &self,
        root: &ID3D12RootSignature,
        vs: &[u8],
        ps: &[u8],
        func: D3D12_COMPARISON_FUNC,
    ) -> Result<ID3D12PipelineState, String> {
        self.graphics_pipeline(root, vs, &[], ps, Some(func))
    }

    fn graphics_pipeline(
        &self,
        root: &ID3D12RootSignature,
        vs: &[u8],
        gs: &[u8],
        ps: &[u8],
        depth_test: Option<D3D12_COMPARISON_FUNC>,
    ) -> Result<ID3D12PipelineState, String> {
        let inputs = sig(vs, FourCC::ISG1)?;
        let names: Vec<CString> = inputs
            .iter()
            .map(|e| CString::new(e.name.clone()).unwrap())
            .collect();
        let layout: Vec<D3D12_INPUT_ELEMENT_DESC> = inputs
            .iter()
            .zip(&names)
            .filter(|(e, _)| e.kind == 0)
            .map(|(e, n)| D3D12_INPUT_ELEMENT_DESC {
                SemanticName: PCSTR(n.as_ptr().cast()),
                SemanticIndex: e.index,
                Format: format(e),
                InputSlot: 0,
                AlignedByteOffset: D3D12_APPEND_ALIGNED_ELEMENT,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                InstanceDataStepRate: 0,
            })
            .collect();
        let outputs = sig(ps, FourCC::OSG1)?;
        let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: a borrowed pointer the descriptor never releases.
            pRootSignature: unsafe { std::mem::transmute_copy(root) },
            VS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: vs.as_ptr().cast(),
                BytecodeLength: vs.len(),
            },
            PS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: ps.as_ptr().cast(),
                BytecodeLength: ps.len(),
            },
            GS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: if gs.is_empty() { std::ptr::null() } else { gs.as_ptr().cast() },
                BytecodeLength: gs.len(),
            },
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: layout.as_ptr(),
                NumElements: layout.len() as u32,
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        for rt in &mut desc.BlendState.RenderTarget {
            rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        }
        for o in outputs.iter().filter(|o| o.kind == 16) {
            let n = o.row.unwrap_or(0) as usize;
            desc.RTVFormats[n] = target_format(o);
            desc.NumRenderTargets = desc.NumRenderTargets.max(n as u32 + 1);
        }
        if outputs.iter().any(|o| matches!(o.kind, 17..=19)) {
            desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
            desc.DepthStencilState.DepthEnable = true.into();
            desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
            desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        }
        if outputs.iter().any(|o| o.kind == 20) {
            desc.DSVFormat = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
            desc.DepthStencilState.StencilEnable = true.into();
        }
        if let Some(func) = depth_test {
            desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
            desc.DepthStencilState.DepthEnable = true.into();
            desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
            desc.DepthStencilState.DepthFunc = func;
        }
        // SAFETY: the description points at live data.
        let pso = unsafe { self.device.CreateGraphicsPipelineState(&desc) };
        pso.map_err(err)
    }

    /// A mesh pipeline from a mesh and a pixel shader (user data of the mesh
    /// stage is root parameter 0, the pixel stage's 1).
    pub fn mesh_pipeline(
        &self,
        ms: &HostShader,
        ps: &HostShader,
    ) -> Result<(ID3D12RootSignature, ID3D12PipelineState), String> {
        let root = self.root_signature_for(&[
            (ms, D3D12_SHADER_VISIBILITY_MESH),
            (ps, D3D12_SHADER_VISIBILITY_PIXEL),
        ])?;
        let pso = self.mesh_pipeline_with(&root, &ms.dxil, &ps.dxil)?;
        Ok((root, pso))
    }

    /// A mesh pipeline with the given root signature: the pipeline-state
    /// stream form, with the targets [`Host::pipeline_with`] would use.
    pub fn mesh_pipeline_with(
        &self,
        root: &ID3D12RootSignature,
        ms: &[u8],
        ps: &[u8],
    ) -> Result<ID3D12PipelineState, String> {
        /// A stream subobject: its type, then its description, pointer-aligned.
        #[repr(C, align(8))]
        struct Sub<T>(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, T);
        #[repr(C)]
        struct Stream {
            root: Sub<*mut std::ffi::c_void>,
            ms: Sub<D3D12_SHADER_BYTECODE>,
            ps: Sub<D3D12_SHADER_BYTECODE>,
            blend: Sub<D3D12_BLEND_DESC>,
            mask: Sub<u32>,
            raster: Sub<D3D12_RASTERIZER_DESC>,
            depth: Sub<D3D12_DEPTH_STENCIL_DESC>,
            targets: Sub<D3D12_RT_FORMAT_ARRAY>,
            dsv: Sub<DXGI_FORMAT>,
            samples: Sub<DXGI_SAMPLE_DESC>,
        }
        let outputs = sig(ps, FourCC::OSG1)?;
        let mut blend = D3D12_BLEND_DESC::default();
        for rt in &mut blend.RenderTarget {
            rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        }
        let mut targets = D3D12_RT_FORMAT_ARRAY::default();
        for o in outputs.iter().filter(|o| o.kind == 16) {
            let n = o.row.unwrap_or(0) as usize;
            targets.RTFormats[n] = target_format(o);
            targets.NumRenderTargets = targets.NumRenderTargets.max(n as u32 + 1);
        }
        let mut depth = D3D12_DEPTH_STENCIL_DESC::default();
        let mut dsv = DXGI_FORMAT_UNKNOWN;
        if outputs.iter().any(|o| matches!(o.kind, 17..=19)) {
            dsv = DXGI_FORMAT_D32_FLOAT;
            depth.DepthEnable = true.into();
            depth.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
            depth.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        }
        if outputs.iter().any(|o| o.kind == 20) {
            dsv = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
            depth.StencilEnable = true.into();
        }
        let code = |b: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: b.as_ptr().cast(),
            BytecodeLength: b.len(),
        };
        let mut stream = Stream {
            root: Sub(
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE,
                root.as_raw(),
            ),
            ms: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, code(ms)),
            ps: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, code(ps)),
            blend: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, blend),
            mask: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, u32::MAX),
            raster: Sub(
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER,
                D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
            ),
            depth: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, depth),
            targets: Sub(
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS,
                targets,
            ),
            dsv: Sub(
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT,
                dsv,
            ),
            samples: Sub(
                D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC,
                DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
            ),
        };
        let desc = D3D12_PIPELINE_STATE_STREAM_DESC {
            SizeInBytes: size_of::<Stream>(),
            pPipelineStateSubobjectStream: (&raw mut stream).cast(),
        };
        let device: ID3D12Device2 = self.device.cast().map_err(err)?;
        // SAFETY: the stream points at live data and outlives the call.
        unsafe { device.CreatePipelineState(&desc) }.map_err(err)
    }

    /// Whether this device fuses DXIL `FMad`, which the lowering emits for
    /// the console's fused `v_fma_f32`/`v_fmac_f32`/`v_fma_mix*`: where it
    /// does not, their results can differ in kind (NaN where an overflowing
    /// product meets an infinity, 0 where the fused result is tiny). Runs
    /// the lowered `v_fma_f32` of (1 - 2^-24, 1 + 2^-23, -1) once: 2^-24 -
    /// 2^-47 fused, 0 not.
    pub fn fuses_fmad(&self) -> Result<bool, String> {
        let shader = fmad_probe().map_err(|e| e.to_string())?;
        let (root, pso) = self.compute_pipeline(&shader)?;
        let operands = [0x3F7F_FFFF, 0x3F80_0001, 0xBF80_0000];
        let mut user_data = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            if u.root == crate::native::RootType::Constants {
                user_data[u.index as usize] = operands[u.dword as usize % 3];
            }
        }
        let dev = &self.device;
        // SAFETY: D3D12 calls with descriptions that outlive them; the
        // resources stay alive until the fence says the GPU is done.
        unsafe {
            let queue: ID3D12CommandQueue = dev
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .map_err(err)?;
            let alloc: ID3D12CommandAllocator = dev
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .map_err(err)?;
            let list: ID3D12GraphicsCommandList = dev
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso)
                .map_err(err)?;
            let buffer = |heap, flags| -> Result<ID3D12Resource, String> {
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: heap,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: 256,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        Flags: flags,
                        ..Default::default()
                    },
                    if heap == D3D12_HEAP_TYPE_READBACK {
                        D3D12_RESOURCE_STATE_COPY_DEST
                    } else {
                        D3D12_RESOURCE_STATE_COMMON
                    },
                    None,
                    &mut r,
                )
                .map_err(err)?;
                r.ok_or_else(|| "no buffer".to_string())
            };
            let out = buffer(
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            )?;
            let readback = buffer(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE)?;
            let views: ID3D12DescriptorHeap = dev
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    ..Default::default()
                })
                .map_err(err)?;
            dev.CreateUnorderedAccessView(
                &out,
                None,
                Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_UAV {
                            NumElements: 64,
                            Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                            ..Default::default()
                        },
                    },
                }),
                views.GetCPUDescriptorHandleForHeapStart(),
            );
            list.SetDescriptorHeaps(&[Some(views.clone())]);
            list.SetComputeRootSignature(&root);
            list.SetComputeRoot32BitConstants(
                0,
                user_data.len() as u32,
                user_data.as_ptr().cast(),
                0,
            );
            list.Dispatch(1, 1, 1);
            list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: std::mem::transmute_copy(&out),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                    }),
                },
            }]);
            list.CopyBufferRegion(&readback, 0, &out, 0, 4);
            list.Close().map_err(err)?;
            queue.ExecuteCommandLists(&[Some(list.cast().map_err(err)?)]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(err)?;
            queue.Signal(&fence, 1).map_err(err)?;
            while fence.GetCompletedValue() < 1 {
                if let Err(e) = dev.GetDeviceRemovedReason() {
                    return Err(format!("device removed: {}", err(e)));
                }
                std::thread::yield_now();
            }
            let mut p = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut p)).map_err(err)?;
            let bits = p.cast::<u32>().read();
            readback.Unmap(0, None);
            Ok(bits != 0)
        }
    }
}

/// `v_fma_f32 v0, s0, s1, s2` (through VGPRs) stored to dword 0 of the raw
/// UAV a table at s4 names, lowered as a title's program is.
fn fmad_probe() -> Result<HostShader, crate::ShaderError> {
    use crate::native::{
        ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
    };
    let words: [u32; 11] = [
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xFA00_0000,
        0x7E02_0200, // v_mov_b32 v1, s0
        0x7E04_0201, // v_mov_b32 v2, s1
        0x7E06_0202, // v_mov_b32 v3, s2
        0xD54B_0000, // v_fma_f32 v0, v1, v2, v3
        0x040E_0501,
        0xE070_0000, // buffer_store_dword v0, off, s[8:11], 0
        0x8002_0000,
        0xBF81_0000, // s_endpgm
    ];
    let param = |root, user_data, sgprs, slot| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs,
        base_register: 0,
        user_data,
        slot,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            param(RootType::Constants, 0, 4, 0),
            param(RootType::Table, 4, 1, 1),
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = crate::ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = crate::structure::structure(&p)?;
    let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = crate::io::raise(&p, &native)?;
    let lowered = super::lower::lower(&super::lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        view_marks: false,
    })?;
    Ok(HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        memory: lowered.memory,
        memory_writes: lowered.memory_writes,
        view_addresses: lowered.view_addresses,
        heap: lowered.heap,
        heap_writes: lowered.heap_writes,
        view_marks: lowered.view_marks,
    })
}

/// A vertex shader that feeds `ps`'s inputs (zeros) and a position.
pub fn stub_vs(ps: &HostShader) -> Result<HostShader, String> {
    let inputs = sig(&ps.dxil, FourCC::ISG1)?;
    let mut outputs: Vec<SigElement> = inputs
        .into_iter()
        .filter(|e| matches!(e.kind, 0 | 3 | 6 | 7))
        .collect();
    if !outputs.iter().any(|e| e.kind == 3) {
        let row = outputs
            .iter()
            .filter_map(|e| e.row)
            .map(|r| r + 1)
            .max()
            .unwrap_or(0);
        outputs.push(SigElement {
            name: "SV_Position".into(),
            index: 0,
            kind: 3,
            comp: 9,
            interp: 4,
            row: Some(row),
            col: 0,
            cols: 4,
            usage: 0,
            half: false,
        });
    }
    Ok(stub(false, outputs))
}

/// A vertex shader whose position is the vertex's (`POSITION`, four floats
/// at offset 0).
#[cfg(test)]
pub fn position_vs() -> HostShader {
    use super::bitcode::{Attr, Term};
    let mut s = super::shader::Shader::new(false);
    let b = s.entry;
    for col in 0..4 {
        let undef = s.m.undef(s.t.i32);
        let args = [s.m.i32(0), s.m.i32(0), s.m.i8(col), undef];
        let v = s.op(b, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &args);
        let args = [s.m.i32(0), s.m.i32(0), s.m.i8(col), v];
        s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
    }
    let element = |name: &str, kind| SigElement {
        name: name.into(),
        index: 0,
        kind,
        comp: 9,
        interp: if kind == 3 { 4 } else { 0 },
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0xf,
        half: false,
    };
    s.inputs = vec![element("POSITION", 0)];
    s.outputs = vec![element("SV_Position", 3)];
    s.m.terminate(b, Term::Ret);
    HostShader {
        stage: crate::native::HwStage::Vs,
        dxil: s.finish(),
        user_data: Vec::new(),
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        memory: false,
        memory_writes: false,
        view_addresses: false,
        heap: false,
        heap_writes: false,
        view_marks: false,
    }
}

/// A pixel shader that reads nothing and writes target 0.
pub fn stub_ps() -> HostShader {
    let target = SigElement {
        name: "SV_Target".into(),
        index: 0,
        kind: 16,
        comp: 9,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0,
        half: false,
    };
    stub(true, vec![target])
}

fn stub(pixel: bool, outputs: Vec<SigElement>) -> HostShader {
    use super::bitcode::{Attr, Term};
    let mut s = super::shader::Shader::new(pixel);
    let b = s.entry;
    for (id, e) in outputs.iter().enumerate() {
        for col in 0..e.cols {
            let (suffix, zero) = if e.comp == 9 {
                (".f32", s.m.f32(0))
            } else {
                (".i32", s.m.i32(0))
            };
            let args = [s.m.i32(id as u32), s.m.i32(0), s.m.i8(col), zero];
            s.op(
                b,
                &format!("storeOutput{suffix}"),
                Attr::NoUnwind,
                s.t.void,
                5,
                &args,
            );
        }
    }
    s.outputs = outputs
        .into_iter()
        .map(|mut e| {
            e.usage = (1 << e.cols) - 1;
            e
        })
        .collect();
    s.m.terminate(b, Term::Ret);
    HostShader {
        stage: if pixel {
            crate::native::HwStage::Ps
        } else {
            crate::native::HwStage::Vs
        },
        dxil: s.finish(),
        user_data: Vec::new(),
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        memory: false,
        memory_writes: false,
        view_addresses: false,
        heap: false,
        heap_writes: false,
        view_marks: false,
    }
}

/// A root argument for [`Host::draw`].
#[cfg(test)]
pub enum Root {
    Constants(Vec<u32>),
    /// A table starting at this heap index.
    Table {
        sampler: bool,
        index: usize,
    },
    /// A root CBV (or with `srv`, a root SRV) over a buffer of these
    /// dwords ([`Host::dispatch`] only).
    View {
        srv: bool,
        data: Vec<u32>,
    },
    /// A root raw UAV of this many dwords, zeroed, read back after the
    /// heap's UAVs ([`Host::dispatch`] only).
    Uav(u32),
}

/// Bindings for [`Host::draw`]: root arguments and the heap contents.
#[cfg(test)]
pub struct Draw<'a> {
    pub roots: Vec<(u32, Root)>,
    /// RGBA32F textures (width, height, texels) at view-heap indices 0..
    pub textures: &'a [(u32, u32, &'a [[f32; 4]])],
    /// Samplers at sampler-heap indices 0..
    pub samplers: &'a [D3D12_SAMPLER_DESC],
    /// Three vertices of stride bytes (none for position-free layouts).
    pub vertices: &'a [f32],
    pub stride: u32,
    pub size: (u32, u32),
    /// Mesh thread groups to dispatch instead of drawing (0: draw).
    pub mesh_groups: u32,
}

#[cfg(test)]
impl Host {
    /// Draws one triangle into an RGBA32F target and returns its texels.
    pub fn draw(
        &self,
        root: &ID3D12RootSignature,
        pso: &ID3D12PipelineState,
        d: &Draw,
    ) -> Result<Vec<[f32; 4]>, String> {
        let dev = &self.device;
        let (w, h) = d.size;
        // SAFETY: D3D12 calls with descriptions that outlive them; resources
        // are kept alive until the fence says the GPU is done.
        unsafe {
            let queue: ID3D12CommandQueue = dev
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .map_err(err)?;
            let alloc: ID3D12CommandAllocator = dev
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .map_err(err)?;
            let list: ID3D12GraphicsCommandList = dev
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, pso)
                .map_err(err)?;
            let tex_desc = |w: u32, h: u32, flags| D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: u64::from(w),
                Height: h,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Flags: flags,
                ..Default::default()
            };
            let buf_desc = |size: u64| D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let heap = |t| D3D12_HEAP_PROPERTIES {
                Type: t,
                ..Default::default()
            };
            let make = |props: D3D12_HEAP_TYPE,
                        desc: &D3D12_RESOURCE_DESC,
                        state,
                        clear: Option<&D3D12_CLEAR_VALUE>| {
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(
                    &heap(props),
                    D3D12_HEAP_FLAG_NONE,
                    desc,
                    state,
                    clear.map(|c| c as *const _),
                    &mut r,
                )
                .map_err(err)?;
                r.ok_or_else(|| "no resource".to_string())
            };
            let target = make(
                D3D12_HEAP_TYPE_DEFAULT,
                &tex_desc(w, h, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                Some(&D3D12_CLEAR_VALUE {
                    Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                    Anonymous: D3D12_CLEAR_VALUE_0 { Color: [0.0; 4] },
                }),
            )?;
            let rtv_heap: ID3D12DescriptorHeap = dev
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: 1,
                    ..Default::default()
                })
                .map_err(err)?;
            let rtv = rtv_heap.GetCPUDescriptorHandleForHeapStart();
            dev.CreateRenderTargetView(&target, None, rtv);
            let views: ID3D12DescriptorHeap = dev
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: d.textures.len().max(1) as u32,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    ..Default::default()
                })
                .map_err(err)?;
            let sampler_heap: ID3D12DescriptorHeap = dev
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                    NumDescriptors: d.samplers.len().max(1) as u32,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    ..Default::default()
                })
                .map_err(err)?;
            let view_size = dev
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                as usize;
            let sampler_size =
                dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) as usize;
            let mut keep = Vec::new();
            for (i, &(tw, th, texels)) in d.textures.iter().enumerate() {
                let tex = make(
                    D3D12_HEAP_TYPE_DEFAULT,
                    &tex_desc(tw, th, D3D12_RESOURCE_FLAG_NONE),
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                )?;
                let pitch = (tw as usize * 16).next_multiple_of(256);
                let upload = make(
                    D3D12_HEAP_TYPE_UPLOAD,
                    &buf_desc((pitch * th as usize) as u64),
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                )?;
                let mut p = std::ptr::null_mut();
                upload.Map(0, None, Some(&mut p)).map_err(err)?;
                for y in 0..th as usize {
                    let row = &texels[y * tw as usize..(y + 1) * tw as usize];
                    std::ptr::copy_nonoverlapping(
                        row.as_ptr().cast::<u8>(),
                        p.cast::<u8>().add(y * pitch),
                        row.len() * 16,
                    );
                }
                upload.Unmap(0, None);
                let dst = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&tex),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        SubresourceIndex: 0,
                    },
                };
                let src = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&upload),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: 0,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                                Width: tw,
                                Height: th,
                                Depth: 1,
                                RowPitch: pitch as u32,
                            },
                        },
                    },
                };
                list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                list.ResourceBarrier(&[transition(
                    &tex,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
                        | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                )]);
                let mut at = views.GetCPUDescriptorHandleForHeapStart();
                at.ptr += i * view_size;
                dev.CreateShaderResourceView(&tex, None, at);
                keep.push(tex);
                keep.push(upload);
            }
            for (i, s) in d.samplers.iter().enumerate() {
                let mut at = sampler_heap.GetCPUDescriptorHandleForHeapStart();
                at.ptr += i * sampler_size;
                dev.CreateSampler(s, at);
            }
            let vb = if d.vertices.is_empty() {
                None
            } else {
                let bytes = d.vertices.len() * 4;
                let vb = make(
                    D3D12_HEAP_TYPE_UPLOAD,
                    &buf_desc(bytes as u64),
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                )?;
                let mut p = std::ptr::null_mut();
                vb.Map(0, None, Some(&mut p)).map_err(err)?;
                std::ptr::copy_nonoverlapping(
                    d.vertices.as_ptr().cast::<u8>(),
                    p.cast::<u8>(),
                    bytes,
                );
                vb.Unmap(0, None);
                Some(vb)
            };
            list.SetDescriptorHeaps(&[Some(views.clone()), Some(sampler_heap.clone())]);
            list.SetGraphicsRootSignature(root);
            for (param, arg) in &d.roots {
                match arg {
                    Root::Constants(data) => list.SetGraphicsRoot32BitConstants(
                        *param,
                        data.len() as u32,
                        data.as_ptr().cast(),
                        0,
                    ),
                    Root::Table { sampler, index } => {
                        let (heap, size) = if *sampler {
                            (&sampler_heap, sampler_size)
                        } else {
                            (&views, view_size)
                        };
                        let mut at = heap.GetGPUDescriptorHandleForHeapStart();
                        at.ptr += (index * size) as u64;
                        list.SetGraphicsRootDescriptorTable(*param, at);
                    }
                    Root::View { .. } | Root::Uav(_) => return Err("root views in a draw".into()),
                }
            }
            list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.RSSetViewports(&[D3D12_VIEWPORT {
                Width: w as f32,
                Height: h as f32,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            list.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                right: w as i32,
                bottom: h as i32,
                ..Default::default()
            }]);
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            if let Some(vb) = &vb {
                list.IASetVertexBuffers(
                    0,
                    Some(&[D3D12_VERTEX_BUFFER_VIEW {
                        BufferLocation: vb.GetGPUVirtualAddress(),
                        SizeInBytes: (d.vertices.len() * 4) as u32,
                        StrideInBytes: d.stride,
                    }]),
                );
            }
            if d.mesh_groups > 0 {
                let list6: ID3D12GraphicsCommandList6 = list.cast().map_err(err)?;
                list6.DispatchMesh(d.mesh_groups, 1, 1);
            } else {
                list.DrawInstanced(3, 1, 0, 0);
            }
            list.ResourceBarrier(&[transition(
                &target,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            let pitch = (w as usize * 16).next_multiple_of(256);
            let readback = make(
                D3D12_HEAP_TYPE_READBACK,
                &buf_desc((pitch * h as usize) as u64),
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
            )?;
            let dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&readback),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                            Width: w,
                            Height: h,
                            Depth: 1,
                            RowPitch: pitch as u32,
                        },
                    },
                },
            };
            let src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&target),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            list.Close().map_err(err)?;
            queue.ExecuteCommandLists(&[Some(list.cast().map_err(err)?)]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(err)?;
            queue.Signal(&fence, 1).map_err(err)?;
            while fence.GetCompletedValue() < 1 {
                if dev.GetDeviceRemovedReason().is_err() {
                    return Err(format!(
                        "device removed: {}",
                        err(dev.GetDeviceRemovedReason().unwrap_err())
                    ));
                }
                std::thread::yield_now();
            }
            if let Err(e) = dev.GetDeviceRemovedReason() {
                return Err(format!("device removed: {}", err(e)));
            }
            let mut p = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut p)).map_err(err)?;
            let mut out = Vec::with_capacity((w * h) as usize);
            for y in 0..h as usize {
                let row = std::slice::from_raw_parts(
                    p.cast::<u8>().add(y * pitch).cast::<[f32; 4]>(),
                    w as usize,
                );
                out.extend_from_slice(row);
            }
            readback.Unmap(0, None);
            drop(keep);
            Ok(out)
        }
    }
}

#[cfg(test)]
impl Host {
    /// Draws one triangle (`vertices`: three of four floats) with `pso`
    /// ([`Host::depth_tested_pipeline`]) over a `size` D32 depth buffer
    /// cleared to `clear`, with no render target and an RGBA32F texture UAV
    /// of `size` (zeroed) at view-heap index 0; returns the UAV's texels.
    /// Cleared to 0 with a LESS test, every pixel is hidden: what the UAV
    /// holds afterwards the pixel program wrote for hidden pixels.
    pub fn draw_behind_depth(
        &self,
        root: &ID3D12RootSignature,
        pso: &ID3D12PipelineState,
        roots: &[(u32, Root)],
        vertices: &[f32; 12],
        size: (u32, u32),
        clear: f32,
    ) -> Result<Vec<[f32; 4]>, String> {
        let dev = &self.device;
        let (w, h) = size;
        let sample = DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        };
        let texture = |format, flags| D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(w),
            Height: h,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: sample,
            Flags: flags,
            ..Default::default()
        };
        let buffer = |bytes: u64| D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: sample,
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let make = |kind, desc: &D3D12_RESOURCE_DESC, state, clear: Option<&D3D12_CLEAR_VALUE>| {
            let mut r: Option<ID3D12Resource> = None;
            // SAFETY: descriptions that outlive the call; our out pointer.
            unsafe {
                dev.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: kind,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    desc,
                    state,
                    clear.map(|c| c as *const _),
                    &mut r,
                )
            }
            .map_err(err)?;
            r.ok_or_else(|| "no resource".to_string())
        };
        let uav = make(
            D3D12_HEAP_TYPE_DEFAULT,
            &texture(
                DXGI_FORMAT_R32G32B32A32_FLOAT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            None,
        )?;
        let depth = make(
            D3D12_HEAP_TYPE_DEFAULT,
            &texture(
                DXGI_FORMAT_D32_FLOAT,
                D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
            ),
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            Some(&D3D12_CLEAR_VALUE {
                Format: DXGI_FORMAT_D32_FLOAT,
                Anonymous: D3D12_CLEAR_VALUE_0 {
                    DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                        Depth: clear,
                        Stencil: 0,
                    },
                },
            }),
        )?;
        let vb = make(
            D3D12_HEAP_TYPE_UPLOAD,
            &buffer(48),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            None,
        )?;
        let pitch = (w as usize * 16).next_multiple_of(256);
        let readback = make(
            D3D12_HEAP_TYPE_READBACK,
            &buffer((pitch * h as usize) as u64),
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
        )?;
        // SAFETY: D3D12 calls with descriptions that outlive them; resources
        // are kept alive until the fence says the GPU is done.
        unsafe {
            let queue: ID3D12CommandQueue = dev
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .map_err(err)?;
            let alloc: ID3D12CommandAllocator = dev
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .map_err(err)?;
            let list: ID3D12GraphicsCommandList = dev
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, pso)
                .map_err(err)?;
            let views: ID3D12DescriptorHeap = dev
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    ..Default::default()
                })
                .map_err(err)?;
            dev.CreateUnorderedAccessView(
                &uav,
                None,
                None,
                views.GetCPUDescriptorHandleForHeapStart(),
            );
            let dsv_heap: ID3D12DescriptorHeap = dev
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                    NumDescriptors: 1,
                    ..Default::default()
                })
                .map_err(err)?;
            let dsv = dsv_heap.GetCPUDescriptorHandleForHeapStart();
            dev.CreateDepthStencilView(&depth, None, dsv);
            let mut p = std::ptr::null_mut();
            vb.Map(0, None, Some(&mut p)).map_err(err)?;
            std::ptr::copy_nonoverlapping(vertices.as_ptr().cast::<u8>(), p.cast::<u8>(), 48);
            vb.Unmap(0, None);
            list.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, clear, 0, None);
            list.SetDescriptorHeaps(&[Some(views.clone())]);
            list.SetGraphicsRootSignature(root);
            for (param, arg) in roots {
                match arg {
                    Root::Constants(data) => list.SetGraphicsRoot32BitConstants(
                        *param,
                        data.len() as u32,
                        data.as_ptr().cast(),
                        0,
                    ),
                    Root::Table {
                        sampler: false,
                        index: 0,
                    } => list.SetGraphicsRootDescriptorTable(
                        *param,
                        views.GetGPUDescriptorHandleForHeapStart(),
                    ),
                    _ => return Err("only constants and the UAV's table here".into()),
                }
            }
            list.OMSetRenderTargets(0, None, false, Some(&dsv));
            list.RSSetViewports(&[D3D12_VIEWPORT {
                Width: w as f32,
                Height: h as f32,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            list.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                right: w as i32,
                bottom: h as i32,
                ..Default::default()
            }]);
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list.IASetVertexBuffers(
                0,
                Some(&[D3D12_VERTEX_BUFFER_VIEW {
                    BufferLocation: vb.GetGPUVirtualAddress(),
                    SizeInBytes: 48,
                    StrideInBytes: 16,
                }]),
            );
            list.DrawInstanced(3, 1, 0, 0);
            list.ResourceBarrier(&[transition(
                &uav,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            let dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&readback),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                            Width: w,
                            Height: h,
                            Depth: 1,
                            RowPitch: pitch as u32,
                        },
                    },
                },
            };
            let src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&uav),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            list.Close().map_err(err)?;
            queue.ExecuteCommandLists(&[Some(list.cast().map_err(err)?)]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(err)?;
            queue.Signal(&fence, 1).map_err(err)?;
            while fence.GetCompletedValue() < 1 {
                if let Err(e) = dev.GetDeviceRemovedReason() {
                    return Err(format!("device removed: {}", err(e)));
                }
                std::thread::yield_now();
            }
            let mut p = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut p)).map_err(err)?;
            let mut out = Vec::with_capacity((w * h) as usize);
            for y in 0..h as usize {
                let row = std::slice::from_raw_parts(
                    p.cast::<u8>().add(y * pitch).cast::<[f32; 4]>(),
                    w as usize,
                );
                out.extend_from_slice(row);
            }
            readback.Unmap(0, None);
            Ok(out)
        }
    }
}

#[cfg(test)]
fn transition(
    r: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(r) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// A view in the heap for [`Host::dispatch`].
#[cfg(test)]
pub enum View<'a> {
    /// An RGBA32F texture SRV (width, height, texels).
    Texture(u32, u32, &'a [[f32; 4]]),
    /// An RGBA32F texture SRV with mips (width, height, levels, texels of
    /// each level in turn).
    Mipped(u32, u32, u32, &'a [[f32; 4]]),
    /// An RGBA32F cube SRV (face size, texels of faces +x -x +y -y +z -z).
    Cube(u32, &'a [[f32; 4]]),
    /// An RGBA32F cube SRV with mips (face size, levels, texels of each
    /// face's levels in turn).
    MippedCube(u32, u32, &'a [[f32; 4]]),
    /// An RGBA32F square texture SRV (size, levels) whose texels in level
    /// `n` are all `n`.
    Levels(u32, u32),
    /// A raw buffer UAV of this many dwords, zeroed.
    RawUav(u32),
    /// An RGBA32F texture UAV (width, height), zeroed.
    TextureUav(u32, u32),
    /// An R32_UINT texture UAV (width, height), zeroed (atomics).
    UintTextureUav(u32, u32),
    /// An R32G32_UINT texture UAV (width, height), zeroed (64-bit atomics).
    Uint2TextureUav(u32, u32),
    /// A typed RGBA16F buffer UAV of this many elements, zeroed.
    HalfUav(u32),
    /// A constant buffer of these dwords.
    Cbv(&'a [u32]),
    /// A buffer SRV over these dwords.
    Buffer { data: &'a [u32], stride: Stride },
    /// Title memory: a buffer holding these dwords viewed raw, as an SRV at
    /// its index and a UAV at the next (whose entry is [`View::Pair`]),
    /// read back like the UAVs.
    Memory(&'a [u32]),
    /// As Memory, but the first descriptor is null to prove that raw
    /// page reads use only the UAV slot.
    MemoryUavOnly(&'a [u32]),
    /// Null raw SRV and UAV at this index and the next ([`View::Pair`]).
    NullPair,
    /// The second slot of a [`View::Memory`] or [`View::NullPair`].
    Pair,
    /// Last in the heap, which then has [`super::view_heap_limit`] slots: the
    /// host's null raw buffer SRV at [`super::null_buffer_view`] and UAV
    /// below it.
    NullBuffers,
}

/// How [`View::Buffer`] is viewed.
#[cfg(test)]
#[derive(Clone, Copy)]
pub enum Stride {
    Raw,
    /// RGBA32F elements.
    Typed,
    /// Structured, elements of this many bytes.
    Structured(u32),
    /// Typed, elements in this format of this many bytes.
    Format(DXGI_FORMAT, u32),
}

/// Bindings for [`Host::dispatch`]: root arguments, the view heap in order,
/// the sampler heap, and the group counts.
#[cfg(test)]
pub struct Dispatch<'a> {
    pub roots: Vec<(u32, Root)>,
    pub views: &'a [View<'a>],
    pub samplers: &'a [D3D12_SAMPLER_DESC],
    pub groups: [u32; 3],
}

#[cfg(test)]
impl Host {
    /// Runs a compute pipeline and returns each UAV's contents in heap
    /// order (texels as float bits, row by row).
    pub fn dispatch(
        &self,
        root: &ID3D12RootSignature,
        pso: &ID3D12PipelineState,
        d: &Dispatch,
    ) -> Result<Vec<Vec<u32>>, String> {
        let dev = &self.device;
        // SAFETY: D3D12 calls with descriptions that outlive them; resources
        // stay alive until the fence says the GPU is done.
        unsafe {
            let queue: ID3D12CommandQueue = dev
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .map_err(err)?;
            let alloc: ID3D12CommandAllocator = dev
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .map_err(err)?;
            let list: ID3D12GraphicsCommandList = dev
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, pso)
                .map_err(err)?;
            let make = |heap: D3D12_HEAP_TYPE, desc: &D3D12_RESOURCE_DESC, state| {
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: heap,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    desc,
                    state,
                    None,
                    &mut r,
                )
                .map_err(err)?;
                r.ok_or_else(|| "no resource".to_string())
            };
            let sample = DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            };
            let texture_as = |w: u32, h: u32, flags, format| D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: u64::from(w),
                Height: h,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: format,
                SampleDesc: sample,
                Flags: flags,
                ..Default::default()
            };
            let texture = |w, h, flags| texture_as(w, h, flags, DXGI_FORMAT_R32G32B32A32_FLOAT);
            let buffer = |size: u64, flags| D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: sample,
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
                ..Default::default()
            };
            // Texel words: 4 for RGBA32F, 1 for R32_UINT, 2 for R32G32_UINT.
            let footprint_as =
                |r: &ID3D12Resource, w: u32, h: u32, words: u32| D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(r),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: 0,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: match words {
                                    1 => DXGI_FORMAT_R32_UINT,
                                    2 => DXGI_FORMAT_R32G32_UINT,
                                    _ => DXGI_FORMAT_R32G32B32A32_FLOAT,
                                },
                                Width: w,
                                Height: h,
                                Depth: 1,
                                RowPitch: (w * 4 * words).next_multiple_of(256),
                            },
                        },
                    },
                };
            let footprint = |r: &ID3D12Resource, w: u32, h: u32| footprint_as(r, w, h, 4);
            let subresource = |r: &ID3D12Resource| D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(r),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            let heap = |ty, n: usize| -> Result<ID3D12DescriptorHeap, String> {
                dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: ty,
                    NumDescriptors: n.max(1) as u32,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    ..Default::default()
                })
                .map_err(err)
            };
            // An upload buffer holding `data` (sized for a constant buffer).
            let upload = |data: &[u32]| -> Result<ID3D12Resource, String> {
                let size = (data.len() * 4).next_multiple_of(256);
                let b = make(
                    D3D12_HEAP_TYPE_UPLOAD,
                    &buffer(size as u64, D3D12_RESOURCE_FLAG_NONE),
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                )?;
                let mut p = std::ptr::null_mut();
                b.Map(0, None, Some(&mut p)).map_err(err)?;
                std::ptr::copy_nonoverlapping(data.as_ptr(), p.cast::<u32>(), data.len());
                b.Unmap(0, None);
                Ok(b)
            };
            let slots = match d.views.last() {
                Some(View::NullBuffers) => super::view_heap_limit() as usize,
                _ => d.views.len(),
            };
            let views = heap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, slots)?;
            let samplers = heap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, d.samplers.len())?;
            let view_size = dev
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                as usize;
            let sampler_size =
                dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) as usize;
            let mut keep = Vec::new();
            // UAVs: (resource, dwords or texture size, readback buffer).
            let mut uavs = Vec::new();
            for (i, v) in d.views.iter().enumerate() {
                let mut at = views.GetCPUDescriptorHandleForHeapStart();
                at.ptr += i * view_size;
                match *v {
                    View::Texture(..)
                    | View::Cube(..)
                    | View::Mipped(..)
                    | View::MippedCube(..) => {
                        let (w0, h0, mut texels, faces, levels) = match *v {
                            View::Texture(w, h, t) => (w, h, t, 1, 1),
                            View::Mipped(w, h, l, t) => (w, h, t, 1, l),
                            View::Cube(n, t) => (n, n, t, 6, 1),
                            View::MippedCube(n, l, t) => (n, n, t, 6, l),
                            _ => unreachable!(),
                        };
                        let mut desc = texture(w0, h0, D3D12_RESOURCE_FLAG_NONE);
                        desc.DepthOrArraySize = faces as u16;
                        desc.MipLevels = levels as u16;
                        let tex = make(
                            D3D12_HEAP_TYPE_DEFAULT,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                        )?;
                        // Subresources face by face, each level in turn.
                        for face in 0..(faces * levels) as usize {
                            let level = face as u32 % levels;
                            let (w, h) = ((w0 >> level).max(1), (h0 >> level).max(1));
                            let pitch = (w as usize * 16).next_multiple_of(256);
                            let upload = make(
                                D3D12_HEAP_TYPE_UPLOAD,
                                &buffer((pitch * h as usize) as u64, D3D12_RESOURCE_FLAG_NONE),
                                D3D12_RESOURCE_STATE_GENERIC_READ,
                            )?;
                            let mut p = std::ptr::null_mut();
                            upload.Map(0, None, Some(&mut p)).map_err(err)?;
                            for y in 0..h as usize {
                                let row = &texels[y * w as usize..(y + 1) * w as usize];
                                std::ptr::copy_nonoverlapping(
                                    row.as_ptr().cast::<u8>(),
                                    p.cast::<u8>().add(y * pitch),
                                    row.len() * 16,
                                );
                            }
                            texels = &texels[(w * h) as usize..];
                            upload.Unmap(0, None);
                            let mut at_face = subresource(&tex);
                            at_face.Anonymous.SubresourceIndex = face as u32;
                            list.CopyTextureRegion(
                                &at_face,
                                0,
                                0,
                                0,
                                &footprint(&upload, w, h),
                                None,
                            );
                            keep.push(upload);
                        }
                        list.ResourceBarrier(&[transition(
                            &tex,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        )]);
                        let cube = D3D12_SHADER_RESOURCE_VIEW_DESC {
                            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                            ViewDimension: D3D12_SRV_DIMENSION_TEXTURECUBE,
                            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                                TextureCube: D3D12_TEXCUBE_SRV {
                                    MipLevels: levels,
                                    ..Default::default()
                                },
                            },
                        };
                        dev.CreateShaderResourceView(
                            &tex,
                            (faces == 6).then_some(&cube as *const _),
                            at,
                        );
                        keep.push(tex);
                    }
                    View::Levels(w, levels) => {
                        let mut desc = texture(w, w, D3D12_RESOURCE_FLAG_NONE);
                        desc.MipLevels = levels as u16;
                        let tex = make(
                            D3D12_HEAP_TYPE_DEFAULT,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                        )?;
                        for level in 0..levels {
                            let n = (w >> level).max(1);
                            let pitch = (n as usize * 16).next_multiple_of(256);
                            let upload = make(
                                D3D12_HEAP_TYPE_UPLOAD,
                                &buffer((pitch * n as usize) as u64, D3D12_RESOURCE_FLAG_NONE),
                                D3D12_RESOURCE_STATE_GENERIC_READ,
                            )?;
                            let mut p = std::ptr::null_mut();
                            upload.Map(0, None, Some(&mut p)).map_err(err)?;
                            let row = vec![[level as f32; 4]; n as usize];
                            for y in 0..n as usize {
                                std::ptr::copy_nonoverlapping(
                                    row.as_ptr().cast::<u8>(),
                                    p.cast::<u8>().add(y * pitch),
                                    row.len() * 16,
                                );
                            }
                            upload.Unmap(0, None);
                            let mut at_level = subresource(&tex);
                            at_level.Anonymous.SubresourceIndex = level;
                            list.CopyTextureRegion(
                                &at_level,
                                0,
                                0,
                                0,
                                &footprint(&upload, n, n),
                                None,
                            );
                            keep.push(upload);
                        }
                        list.ResourceBarrier(&[transition(
                            &tex,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        )]);
                        dev.CreateShaderResourceView(&tex, None, at);
                        keep.push(tex);
                    }
                    View::RawUav(_) | View::HalfUav(_) => {
                        // (dwords, format, elements, flags)
                        let (n, format, elements, flags) = match *v {
                            View::HalfUav(e) => (
                                2 * e,
                                DXGI_FORMAT_R16G16B16A16_FLOAT,
                                e,
                                D3D12_BUFFER_UAV_FLAG_NONE,
                            ),
                            View::RawUav(n) => {
                                (n, DXGI_FORMAT_R32_TYPELESS, n, D3D12_BUFFER_UAV_FLAG_RAW)
                            }
                            _ => unreachable!(),
                        };
                        let bytes = u64::from(n) * 4;
                        let b = make(
                            D3D12_HEAP_TYPE_DEFAULT,
                            &buffer(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                            D3D12_RESOURCE_STATE_COMMON,
                        )?;
                        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                            Format: format,
                            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                                Buffer: D3D12_BUFFER_UAV {
                                    NumElements: elements,
                                    Flags: flags,
                                    ..Default::default()
                                },
                            },
                        };
                        dev.CreateUnorderedAccessView(&b, None, Some(&desc), at);
                        let rb = make(
                            D3D12_HEAP_TYPE_READBACK,
                            &buffer(bytes, D3D12_RESOURCE_FLAG_NONE),
                            D3D12_RESOURCE_STATE_COPY_DEST,
                        )?;
                        uavs.push((b, None, n, rb));
                    }
                    View::TextureUav(w, h) => {
                        let t = make(
                            D3D12_HEAP_TYPE_DEFAULT,
                            &texture(w, h, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        )?;
                        dev.CreateUnorderedAccessView(&t, None, None, at);
                        let pitch = (w * 16).next_multiple_of(256);
                        let rb = make(
                            D3D12_HEAP_TYPE_READBACK,
                            &buffer(u64::from(pitch * h), D3D12_RESOURCE_FLAG_NONE),
                            D3D12_RESOURCE_STATE_COPY_DEST,
                        )?;
                        uavs.push((t, Some((w, h)), 4, rb));
                    }
                    View::UintTextureUav(w, h) | View::Uint2TextureUav(w, h) => {
                        let (format, words) = match *v {
                            View::UintTextureUav(..) => (DXGI_FORMAT_R32_UINT, 1),
                            _ => (DXGI_FORMAT_R32G32_UINT, 2),
                        };
                        let t = make(
                            D3D12_HEAP_TYPE_DEFAULT,
                            &texture_as(w, h, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, format),
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        )?;
                        dev.CreateUnorderedAccessView(&t, None, None, at);
                        let pitch = (w * 4 * words).next_multiple_of(256);
                        let rb = make(
                            D3D12_HEAP_TYPE_READBACK,
                            &buffer(u64::from(pitch * h), D3D12_RESOURCE_FLAG_NONE),
                            D3D12_RESOURCE_STATE_COPY_DEST,
                        )?;
                        uavs.push((t, Some((w, h)), words, rb));
                    }
                    View::Cbv(data) => {
                        let b = upload(data)?;
                        dev.CreateConstantBufferView(
                            Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC {
                                BufferLocation: b.GetGPUVirtualAddress(),
                                SizeInBytes: (data.len() * 4).next_multiple_of(256) as u32,
                            }),
                            at,
                        );
                        keep.push(b);
                    }
                    View::Buffer { data, stride } => {
                        let b = upload(data)?;
                        let (format, elements, stride, flags) = match stride {
                            Stride::Raw => (
                                DXGI_FORMAT_R32_TYPELESS,
                                data.len(),
                                0,
                                D3D12_BUFFER_SRV_FLAG_RAW,
                            ),
                            Stride::Typed => (
                                DXGI_FORMAT_R32G32B32A32_FLOAT,
                                data.len() / 4,
                                0,
                                D3D12_BUFFER_SRV_FLAG_NONE,
                            ),
                            Stride::Structured(s) => (
                                DXGI_FORMAT_UNKNOWN,
                                data.len() * 4 / s as usize,
                                s,
                                D3D12_BUFFER_SRV_FLAG_NONE,
                            ),
                            Stride::Format(f, bytes) => (
                                f,
                                data.len() * 4 / bytes as usize,
                                0,
                                D3D12_BUFFER_SRV_FLAG_NONE,
                            ),
                        };
                        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                            Format: format,
                            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                                Buffer: D3D12_BUFFER_SRV {
                                    FirstElement: 0,
                                    NumElements: elements as u32,
                                    StructureByteStride: stride,
                                    Flags: flags,
                                },
                            },
                        };
                        dev.CreateShaderResourceView(&b, Some(&desc), at);
                        keep.push(b);
                    }
                    View::Memory(_) | View::MemoryUavOnly(_) | View::NullPair => {
                        let n = match *v {
                            View::Memory(data) | View::MemoryUavOnly(data) => data.len() as u32,
                            _ => 0,
                        };
                        let b = if let View::Memory(data) | View::MemoryUavOnly(data) = *v {
                            let bytes = u64::from(n) * 4;
                            let b = make(
                                D3D12_HEAP_TYPE_DEFAULT,
                                &buffer(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                                D3D12_RESOURCE_STATE_COPY_DEST,
                            )?;
                            let up = upload(data)?;
                            list.CopyBufferRegion(&b, 0, &up, 0, bytes);
                            list.ResourceBarrier(&[transition(
                                &b,
                                D3D12_RESOURCE_STATE_COPY_DEST,
                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            )]);
                            keep.push(up);
                            Some(b)
                        } else {
                            None
                        };
                        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
                            Format: DXGI_FORMAT_R32_TYPELESS,
                            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                                Buffer: D3D12_BUFFER_SRV {
                                    NumElements: n,
                                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                                    ..Default::default()
                                },
                            },
                        };
                        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                            Format: DXGI_FORMAT_R32_TYPELESS,
                            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                                Buffer: D3D12_BUFFER_UAV {
                                    NumElements: n,
                                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                                    ..Default::default()
                                },
                            },
                        };
                        let mut next = at;
                        next.ptr += view_size;
                        let srv_source = match v {
                            View::MemoryUavOnly(_) => None,
                            _ => b.as_ref(),
                        };
                        dev.CreateShaderResourceView(srv_source, Some(&srv), at);
                        dev.CreateUnorderedAccessView(b.as_ref(), None, Some(&uav), next);
                        if let Some(b) = b {
                            let rb = make(
                                D3D12_HEAP_TYPE_READBACK,
                                &buffer(u64::from(n) * 4, D3D12_RESOURCE_FLAG_NONE),
                                D3D12_RESOURCE_STATE_COPY_DEST,
                            )?;
                            uavs.push((b, None, n, rb));
                        }
                    }
                    View::Pair => {}
                    View::NullBuffers => {
                        let null = super::null_buffer_view() as usize;
                        let (mut srv_at, mut uav_at) = (at, at);
                        srv_at.ptr += (null - i) * view_size;
                        uav_at.ptr += (null - 1 - i) * view_size;
                        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
                            Format: DXGI_FORMAT_R32_TYPELESS,
                            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                                Buffer: D3D12_BUFFER_SRV {
                                    NumElements: 1,
                                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                                    ..Default::default()
                                },
                            },
                        };
                        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                            Format: DXGI_FORMAT_R32_TYPELESS,
                            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                                Buffer: D3D12_BUFFER_UAV {
                                    NumElements: 1,
                                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                                    ..Default::default()
                                },
                            },
                        };
                        dev.CreateShaderResourceView(None, Some(&srv), srv_at);
                        dev.CreateUnorderedAccessView(None, None, Some(&uav), uav_at);
                    }
                }
            }
            for (i, s) in d.samplers.iter().enumerate() {
                let mut at = samplers.GetCPUDescriptorHandleForHeapStart();
                at.ptr += i * sampler_size;
                dev.CreateSampler(s, at);
            }
            list.SetDescriptorHeaps(&[Some(views.clone()), Some(samplers.clone())]);
            list.SetComputeRootSignature(root);
            for (param, arg) in &d.roots {
                match arg {
                    Root::Constants(data) => list.SetComputeRoot32BitConstants(
                        *param,
                        data.len() as u32,
                        data.as_ptr().cast(),
                        0,
                    ),
                    Root::Table { sampler, index } => {
                        let (h, size) = if *sampler {
                            (&samplers, sampler_size)
                        } else {
                            (&views, view_size)
                        };
                        let mut at = h.GetGPUDescriptorHandleForHeapStart();
                        at.ptr += (index * size) as u64;
                        list.SetComputeRootDescriptorTable(*param, at);
                    }
                    Root::View { srv, data } => {
                        let b = upload(data)?;
                        let va = b.GetGPUVirtualAddress();
                        if *srv {
                            list.SetComputeRootShaderResourceView(*param, va);
                        } else {
                            list.SetComputeRootConstantBufferView(*param, va);
                        }
                        keep.push(b);
                    }
                    &Root::Uav(n) => {
                        let bytes = u64::from(n) * 4;
                        let b = make(
                            D3D12_HEAP_TYPE_DEFAULT,
                            &buffer(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                            D3D12_RESOURCE_STATE_COMMON,
                        )?;
                        list.SetComputeRootUnorderedAccessView(*param, b.GetGPUVirtualAddress());
                        let rb = make(
                            D3D12_HEAP_TYPE_READBACK,
                            &buffer(bytes, D3D12_RESOURCE_FLAG_NONE),
                            D3D12_RESOURCE_STATE_COPY_DEST,
                        )?;
                        uavs.push((b, None, n, rb));
                    }
                }
            }
            let [x, y, z] = d.groups;
            list.Dispatch(x, y, z);
            for (r, size, n, rb) in &uavs {
                // (Buffers were promoted from the common state by the dispatch.)
                list.ResourceBarrier(&[transition(
                    r,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                match size {
                    Some((w, h)) => list.CopyTextureRegion(
                        &footprint_as(rb, *w, *h, *n),
                        0,
                        0,
                        0,
                        &subresource(r),
                        None,
                    ),
                    None => list.CopyBufferRegion(rb, 0, r, 0, u64::from(*n) * 4),
                }
            }
            list.Close().map_err(err)?;
            queue.ExecuteCommandLists(&[Some(list.cast().map_err(err)?)]);
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(err)?;
            queue.Signal(&fence, 1).map_err(err)?;
            while fence.GetCompletedValue() < 1 {
                if let Err(e) = dev.GetDeviceRemovedReason() {
                    return Err(format!("device removed: {}", err(e)));
                }
                std::thread::yield_now();
            }
            let mut out = Vec::new();
            for (_, size, n, rb) in &uavs {
                let mut p = std::ptr::null_mut();
                rb.Map(0, None, Some(&mut p)).map_err(err)?;
                let words = match size {
                    Some((w, h)) => {
                        let words = *n as usize;
                        let pitch = (*w as usize * 4 * words).next_multiple_of(256);
                        (0..*h as usize)
                            .flat_map(|y| {
                                std::slice::from_raw_parts(
                                    p.cast::<u8>().add(y * pitch).cast::<u32>(),
                                    *w as usize * words,
                                )
                                .to_vec()
                            })
                            .collect()
                    }
                    None => std::slice::from_raw_parts(p.cast::<u32>(), *n as usize).to_vec(),
                };
                rb.Unmap(0, None);
                out.push(words);
            }
            drop(keep);
            Ok(out)
        }
    }
}
