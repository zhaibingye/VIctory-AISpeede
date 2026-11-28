//! Synthetic guest draw for the native PresentX fixture.

use super::{EDGE, Guest};
use crate::object::{HRESULT, IUNKNOWN, Iid, This, slot};
use std::ptr::null;
use windows::Win32::{
    Foundation::RECT,
    Graphics::{
        Direct3D::{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, ID3DBlob},
        Direct3D12::*,
        Dxgi::Common::{DXGI_FORMAT, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC},
    },
};

const FORMAT: DXGI_FORMAT = DXGI_FORMAT_R8G8B8A8_UNORM;

// Unlike the native-PSO tests in page_draw, guest PSOs need console programs,
// not FXC/PC bytecode. These are the guest equivalents of the fixture HLSL:
//   float4 vertex(uint id : SV_VertexID) : SV_Position {
//       return float4(float(id & 1) * 4 - 1, float(id & 2) * 2 - 1, 0.5, 1);
//   }
//   float4 pixel() : SV_Target0 { return float4(1, 0, 0, 0.5); }
const VS: &[u32] = &[
    0x3604_0081, // v_and_b32 v2, 1, v0 (vertex ID)
    0x3606_0082, // v_and_b32 v3, 2, v0
    0x7E04_0D02, // v_cvt_f32_u32 v2, v2
    0x7E06_0D03, // v_cvt_f32_u32 v3, v3
    0x1008_04F6, // v_mul_f32 v4, 4.0, v2
    0x0608_08F3, // v_add_f32 v4, -1.0, v4
    0x100A_06F4, // v_mul_f32 v5, 2.0, v3
    0x060A_0AF3, // v_add_f32 v5, -1.0, v5
    0x7E0C_02F0, // v_mov_b32 v6, 0.5
    0x7E0E_02F2, // v_mov_b32 v7, 1.0
    0xF800_08CF,
    0x0706_0504, // exp pos0 v4, v5, v6, v7 done
    0xBF81_0000, // s_endpgm
];
const PS: &[u32] = &[
    0x7E00_02F2, // v_mov_b32 v0, 1.0 (red)
    0x7E02_0280, // v_mov_b32 v1, 0 (green, blue)
    0x7E04_02F0, // v_mov_b32 v2, 0.5 (alpha)
    0xF800_180F,
    0x0201_0100, // exp mrt0 v0, v1, v1, v2 done vm
    0xBF81_0000, // s_endpgm
];

/// Wrap just one program and one float4 output in the supported guest
/// container layout (`victory-shader/src/native.rs`). No shader assets.
fn shader_container(pixel: bool) -> Vec<u8> {
    let (slot, code, vgprs, sysval, semantic) = if pixel {
        (0x30, PS, 3u32, 64u32, c"SV_Target")
    } else {
        (0x28, VS, 8u32, 1u32, c"SV_Position")
    };
    // OSG1 header plus one stream-0, register-0, 32-bit float4 output.
    let mut signature: Vec<u8> = [1u32, 8, 0, 40, 0, sysval, 3, 0, 0xf, 0]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    signature.extend_from_slice(semantic.to_bytes_with_nul());
    signature.resize(signature.len().next_multiple_of(4), 0);

    const RECORD: usize = 0x68;
    const STAGE_DATA: usize = RECORD + 0x54;
    const PATCHES: usize = 0x14c;
    const PATCH_STRIDE: usize = 4 + 10 * 10 * 4;
    let code_at = STAGE_DATA
        + if pixel {
            PATCHES + 8 * PATCH_STRIDE
        } else {
            0x18c
        };
    let mut payload = vec![0; code_at];
    payload[..8].copy_from_slice(&6u64.to_le_bytes());
    for at in [0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x48, 0x60] {
        payload[at..at + 8].copy_from_slice(&0xDEAD_BEEF_DEAD_BEEFu64.to_le_bytes());
    }
    payload[slot..slot + 8].copy_from_slice(&(RECORD as u64).to_le_bytes());
    payload[RECORD..RECORD + 8].copy_from_slice(&(code_at as u64).to_le_bytes());
    payload[RECORD + 8..RECORD + 16].copy_from_slice(&(size_of_val(code) as u64).to_le_bytes());
    payload[RECORD + 0x14..RECORD + 0x18].copy_from_slice(&(1u32 << 5).to_le_bytes()); // wave32
    payload[RECORD + 0x50..STAGE_DATA].copy_from_slice(&vgprs.to_le_bytes());
    if pixel {
        for i in 0..8 {
            let at = STAGE_DATA + PATCHES + i * PATCH_STRIDE;
            payload[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        }
    }
    payload.extend(code.iter().flat_map(|w| w.to_le_bytes()));

    let mut fssm = b"FSSM".to_vec();
    fssm.extend_from_slice(&0x101u32.to_le_bytes());
    fssm.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    fssm.extend_from_slice(&payload);
    let compressed = miniz_oxide::deflate::compress_to_vec_zlib(&fssm, 1);
    let mut dxil = vec![0; 0x18];
    dxil.extend_from_slice(b"XDAT");
    dxil.extend(
        [1u32, 0x24, 0x0d, 0x30, (0x18 + compressed.len()) as u32]
            .into_iter()
            .flat_map(u32::to_le_bytes),
    );
    dxil.extend_from_slice(b"XBCH");
    dxil.extend_from_slice(&3u32.to_le_bytes());
    dxil.extend_from_slice(&(compressed.len() as u64).to_le_bytes());
    dxil.extend_from_slice(&(fssm.len() as u64).to_le_bytes());
    dxil.extend_from_slice(&compressed);
    dxil.resize(dxil.len().next_multiple_of(4), 0);

    const FIRST: usize = 0x28;
    let second = FIRST + 8 + dxil.len();
    let total = second + 8 + signature.len();
    let mut container = b"DXBC".to_vec();
    container.extend_from_slice(&[0; 16]);
    container.extend(
        [1, total as u32, 2, FIRST as u32, second as u32]
            .into_iter()
            .flat_map(u32::to_le_bytes),
    );
    for (tag, bytes) in [(b"DXIL", dxil), (b"OSG1", signature)] {
        container.extend_from_slice(tag);
        container.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        container.extend_from_slice(&bytes);
    }
    container
}

#[must_use = "keep the guest pipeline and root signature alive until the draw retires"]
pub(super) struct OverlayDraw {
    pipeline: Guest,
    _root: Guest,
}

#[repr(C, align(8))]
struct Sub<T>(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, T);

#[repr(C)]
struct GraphicsStream {
    root: Sub<This>,
    vs: Sub<D3D12_SHADER_BYTECODE>,
    ps: Sub<D3D12_SHADER_BYTECODE>,
    blend: Sub<D3D12_BLEND_DESC>,
    raster: Sub<D3D12_RASTERIZER_DESC>,
    depth: Sub<D3D12_DEPTH_STENCIL_DESC>,
    topology: Sub<D3D12_PRIMITIVE_TOPOLOGY_TYPE>,
    formats: Sub<D3D12_RT_FORMAT_ARRAY>,
    samples: Sub<DXGI_SAMPLE_DESC>,
}

fn overlay_pipeline() -> OverlayDraw {
    let device: This = (&raw const crate::device::DEVICE).cast();
    let mut blob: Option<ID3DBlob> = None;
    let root_desc = D3D12_ROOT_SIGNATURE_DESC {
        Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
        ..Default::default()
    };
    // SAFETY: the empty signature has no borrowed parameters or samplers.
    unsafe {
        D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None)
    }
    .expect("serialize empty overlay root signature");
    let blob = blob.unwrap();
    let create_root = slot::<
        extern "system" fn(This, u32, *const u8, usize, *const Iid, *mut This) -> HRESULT,
    >(device, 0x80);
    let root = Guest::new(|out| {
        // SAFETY: the blob remains live for the guest call, which copies it.
        unsafe {
            create_root(
                device,
                0,
                blob.GetBufferPointer().cast(),
                blob.GetBufferSize(),
                &IUNKNOWN,
                out,
            )
        }
    });
    let vs = shader_container(false);
    let ps = shader_container(true);
    let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
        pShaderBytecode: bytes.as_ptr().cast(),
        BytecodeLength: bytes.len(),
    };
    let target = D3D12_RENDER_TARGET_BLEND_DESC {
        BlendEnable: true.into(),
        SrcBlend: D3D12_BLEND_SRC_ALPHA,
        DestBlend: D3D12_BLEND_INV_SRC_ALPHA,
        BlendOp: D3D12_BLEND_OP_ADD,
        SrcBlendAlpha: D3D12_BLEND_ONE,
        DestBlendAlpha: D3D12_BLEND_ZERO,
        BlendOpAlpha: D3D12_BLEND_OP_ADD,
        LogicOp: D3D12_LOGIC_OP_NOOP,
        RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
        ..Default::default()
    };
    let mut blend = D3D12_BLEND_DESC::default();
    blend.RenderTarget[0] = target;
    let mut formats = D3D12_RT_FORMAT_ARRAY {
        NumRenderTargets: 1,
        ..Default::default()
    };
    formats.RTFormats[0] = FORMAT;
    let mut stream = GraphicsStream {
        root: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, root.0),
        vs: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, code(&vs)),
        ps: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, code(&ps)),
        blend: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, blend),
        raster: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER,
            D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
        ),
        depth: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL,
            D3D12_DEPTH_STENCIL_DESC {
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                ..Default::default()
            },
        ),
        topology: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY,
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        ),
        formats: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS,
            formats,
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
        SizeInBytes: size_of_val(&stream),
        pPipelineStateSubobjectStream: (&raw mut stream).cast(),
    };
    let create_pso = slot::<
        extern "system" fn(
            This,
            *const crate::pipeline::StreamDesc,
            *const Iid,
            *mut This,
        ) -> HRESULT,
    >(device, 0x2b0);
    let pipeline = Guest::new(|out| create_pso(device, (&raw const desc).cast(), &IUNKNOWN, out));
    let lowered = crate::pipeline::require_host_pipeline(pipeline.0)
        .expect("lower synthetic guest overlay pipeline");
    assert_eq!(lowered.closed_color_formats.as_deref(), Some(&[FORMAT][..]));
    assert_eq!(lowered.closed_color_writes, 1);
    assert_eq!(lowered.closed_color_destination_reads, 1);
    OverlayDraw {
        pipeline,
        _root: root,
    }
}

/// Record a fullscreen triangle clipped to the left half of the native overlay.
/// RGB consumes the previous RTV color; alpha comes from the pixel shader.
///
/// # Safety
/// `list` is a live, open guest direct list, and `rtv` names a live
/// single-sample RGBA8 UNORM RTV of the fixture size. Retain the overlay
/// resource and returned objects until the queue has fenced this draw.
pub(super) unsafe fn record_overlay_draw(list: This, rtv: u64) -> OverlayDraw {
    let draw = overlay_pipeline();
    let viewport = D3D12_VIEWPORT {
        Width: EDGE as f32,
        Height: EDGE as f32,
        MaxDepth: 1.,
        ..Default::default()
    };
    let scissor = RECT {
        left: 0,
        top: 0,
        right: EDGE as i32 / 2,
        bottom: EDGE as i32,
    };
    slot::<extern "system" fn(This, u32, *const u64, i32, *const u64)>(list, 0x190)(
        list,
        1,
        &rtv,
        0,
        null(),
    );
    slot::<extern "system" fn(This, This)>(list, 0xc8)(list, draw.pipeline.0);
    // The guest SetGraphicsRootSignature slot is ignored; the lowered PSO
    // supplies its native empty root signature when the draw is encoded.
    slot::<extern "system" fn(This, u32)>(list, 0xa0)(
        list,
        D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST.0 as u32,
    );
    slot::<extern "system" fn(This, u32, *const D3D12_VIEWPORT)>(list, 0xa8)(list, 1, &viewport);
    slot::<extern "system" fn(This, u32, *const RECT)>(list, 0xb0)(list, 1, &scissor);
    slot::<extern "system" fn(This, u32, u32, u32, u32)>(list, 0x60)(list, 3, 1, 0, 0);
    draw
}
