//! Series S spatial fallback when the console's FidelityFX compute passes
//! cannot run. Writes the output the title supplied, without changing the
//! console mode, render sizes, or the presentation planes.
use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::{Error, s};

use crate::object::This;

const SOURCE: &str = r#"
cbuffer Sizes : register(b0) { uint2 rendered; uint2 output_size; };
Texture2D<float4> colour : register(t0);
RWTexture2D<float4> output_image : register(u0);
SamplerState bilinear : register(s0);

[numthreads(8, 8, 1)]
void stretch(uint3 id : SV_DispatchThreadID) {
    if (any(id.xy >= output_size)) return;
    uint w, h;
    colour.GetDimensions(w, h);
    float2 dims = float2(w, h);
    float2 src = (float2(id.xy) + 0.5) * float2(rendered) / float2(output_size);
    // Keep the filter inside the rendered rectangle, even if source has
    // padding or a larger allocation left over from dynamic resolution.
    src = clamp(src, float2(0.5, 0.5), float2(rendered) - 0.5);
    output_image[id.xy] = colour.SampleLevel(bilinear, src / dims, 0);
}
"#;

struct Stretch {
    root: ID3D12RootSignature,
    pipeline: ID3D12PipelineState,
}

// SAFETY: the root signature and pipeline are free-threaded D3D12 objects.
unsafe impl Send for Stretch {}

static PASS: Mutex<Option<Stretch>> = Mutex::new(None);

fn make(device: &ID3D12Device) -> windows::core::Result<Stretch> {
    let ranges = [
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            OffsetInDescriptorsFromTableStart: 0,
            ..Default::default()
        },
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
            NumDescriptors: 1,
            OffsetInDescriptorsFromTableStart: 1,
            ..Default::default()
        },
    ];
    let parameters = [
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    Num32BitValues: 4,
                    ..Default::default()
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: ranges.len() as u32,
                    pDescriptorRanges: ranges.as_ptr(),
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
    ];
    let sampler = D3D12_STATIC_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        MaxLOD: f32::MAX,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        ..Default::default()
    };
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: parameters.len() as u32,
        pParameters: parameters.as_ptr(),
        NumStaticSamplers: 1,
        pStaticSamplers: &sampler,
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
    };
    // SAFETY: descriptions and all their arrays live through the calls.
    unsafe {
        let mut root = None;
        D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut root, None)?;
        let root = root.ok_or_else(Error::empty)?;
        let bytes = |b: &ID3DBlob| {
            std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize())
        };
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes(&root))?;
        let mut code = None;
        D3DCompile(
            SOURCE.as_ptr().cast(), SOURCE.len(), s!("series_s_stretch"),
            None, None, s!("stretch"), s!("cs_5_1"), 0, 0, &mut code, None,
        )?;
        let code = code.ok_or_else(Error::empty)?;
        let code = bytes(&code);
        let pipeline = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: code.as_ptr().cast(),
                BytecodeLength: code.len(),
            },
            ..Default::default()
        })?;
        Ok(Stretch { root, pipeline })
    }
}

fn extent(wanted: [u32; 2], actual: [u32; 2]) -> Option<[u32; 2]> {
    let dim = |i: usize| if wanted[i] == 0 { actual[i] } else { wanted[i].min(actual[i]) };
    let size = [dim(0), dim(1)];
    (size[0] != 0 && size[1] != 0).then_some(size)
}

fn format(format: DXGI_FORMAT) -> DXGI_FORMAT {
    DXGI_FORMAT(match format.0 {
        1 => 2,
        9 => 10,
        23 => 24,
        27 => 28,
        other => other,
    })
}

fn supports(device: &ID3D12Device, format: DXGI_FORMAT, write: bool) -> bool {
    let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: format, ..Default::default() };
    // SAFETY: the query writes a plain local structure of its named size.
    let ok = unsafe {
        device.CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, (&raw mut support).cast(), size_of_val(&support) as u32)
    };
    if write {
        ok.is_ok() && support.Support1.0 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW.0 != 0
            && support.Support2.0 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE.0 != 0
    } else {
        ok.is_ok() && support.Support1.0 & D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE.0 != 0
    }
}

pub(crate) fn record(
    this: This,
    colour: This,
    output: This,
    render: [u32; 2],
    upscale: [u32; 2],
) -> bool {
    if this.is_null() || colour.is_null() || output.is_null() {
        return false;
    }
    let mut done = false;
    crate::list::with_state(this, |list, state| {
        let (Some(src), Some(dst), Some(host)) = (
            crate::resource::host_resource(colour),
            crate::resource::host_resource(output),
            crate::host::host(),
        ) else { return };
        if src == dst {
            return;
        }
        // SAFETY: both resources live through the borrowed list.
        let (src_desc, dst_desc) = unsafe { (src.GetDesc(), dst.GetDesc()) };
        let valid = |d: &D3D12_RESOURCE_DESC| {
            d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
                && d.DepthOrArraySize == 1 && d.SampleDesc.Count == 1 && d.MipLevels != 0
        };
        if !valid(&src_desc) || !valid(&dst_desc)
            || src_desc.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 != 0
            || dst_desc.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0
        {
            return;
        }
        let (Ok(sw), Ok(dw)) = (u32::try_from(src_desc.Width), u32::try_from(dst_desc.Width)) else { return };
        let (Some(render), Some(upscale)) = (extent(render, [sw, src_desc.Height]), extent(upscale, [dw, dst_desc.Height])) else { return };
        let (input, target) = (format(src_desc.Format), format(dst_desc.Format));
        if !supports(&host.device, input, false) || !supports(&host.device, target, true) {
            return;
        }
        let (root, pipeline) = {
            let mut pass = PASS.lock().unwrap_or_else(|e| e.into_inner());
            if pass.is_none() {
                *pass = match make(&host.device) {
                    Ok(p) => Some(p),
                    Err(e) => {
                        static ONCE: std::sync::Once = std::sync::Once::new();
                        ONCE.call_once(|| tracing::warn!(error = %e, "Series S fallback pipeline not created"));
                        return;
                    }
                };
            }
            let p = pass.as_ref().expect("created above");
            (p.root.clone(), p.pipeline.clone())
        };
        // Own the descriptors for this recording, not a rotating heap that
        // reuses a slot while a title command list can still reference it.
        // SAFETY: The live host device receives a description for a two-entry shader-visible heap.
        let heap: ID3D12DescriptorHeap = match unsafe {
            host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 2,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })
        } {
            Ok(h) => h,
            Err(_) => return,
        };
        crate::list::ensure_state(list, colour, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0 as u32);
        crate::list::ensure_state(list, output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32);
        list.keep(&heap);
        // SAFETY: the list is open, resources held by ensure_state, and
        // newly allocated descriptor heap has exactly these two slots.
        unsafe {
            let cpu = heap.GetCPUDescriptorHandleForHeapStart();
            let step = host.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            host.device.CreateShaderResourceView(src, Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: input,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() },
                },
            }), cpu);
            host.device.CreateUnorderedAccessView(dst, None, Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: target,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                ..Default::default()
            }), D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu.ptr + step as usize });
            list.list.SetDescriptorHeaps(&[Some(heap.clone())]);
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pipeline);
            let sizes = [render[0], render[1], upscale[0], upscale[1]];
            list.list.SetComputeRoot32BitConstants(0, 4, sizes.as_ptr().cast(), 0);
            list.list.SetComputeRootDescriptorTable(1, heap.GetGPUDescriptorHandleForHeapStart());
            list.list.Dispatch(upscale[0].div_ceil(8), upscale[1].div_ceil(8), 1);
        }
        crate::texture::host_written(output);
        state.host_state_lost();
        done = true;
    });
    done
}

#[cfg(test)]
#[test]
fn extent_stays_inside_the_rendered_rectangle() {
    assert_eq!(extent([3, 2], [4, 3]), Some([3, 2]));
    assert_eq!(extent([999, 2], [4, 3]), Some([4, 2]));
    assert_eq!(extent([0, 0], [4, 3]), Some([4, 3]));
    assert_eq!(extent([0, 0], [0, 3]), None);
}

#[cfg(all(test, feature = "diagnostics"))]
#[test]
#[ignore = "run alone with VICTORY_D3D12_WARP=1"]
fn spatial_pipeline_compiles_with_the_software_adapter() {
    assert!(crate::switch::on("VICTORY_D3D12_WARP"));
    let host = crate::host::host().expect("D3D12 software adapter");
    make(&host.device).expect("compile shader, serialize root signature and create compute pipeline");
}

#[cfg(all(test, feature = "diagnostics"))]
#[test]
#[ignore = "run alone with VICTORY_D3D12_WARP=1"]
fn padded_render_is_stretched_without_reading_past_its_edges() {
    use crate::host::host;
    use crate::object::IUNKNOWN;
    use windows::Win32::Foundation::RECT;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM as RGBA8;

    assert!(crate::switch::on("VICTORY_D3D12_WARP"));
    let host = host().expect("WARP software adapter");
    let mut allocator: This = std::ptr::null();
    let mut title_list: This = std::ptr::null();
    assert_eq!(crate::list::create_allocator(std::ptr::null(), 0, &IUNKNOWN, &mut allocator), 0);
    assert_eq!(crate::list::create_list(
        std::ptr::null(), 0, 0, allocator, std::ptr::null(), &IUNKNOWN, &mut title_list,
    ), 0);
    let commands = crate::list::host_list(title_list).expect("host command list");
    let list = &commands.list;

    let mut source_va = Box::new([0u8; 65536]);
    let mut target_va = Box::new([0u8; 65536]);
    let texture = |address: *mut u8, width, height, flags| {
        let mut object: This = std::ptr::null();
        let words = [3u32, 0, 0, 0, width, 0, height, 0x1_0001, RGBA8.0 as u32, 1, 0, 0, flags, 0];
        // SAFETY: these 56 bytes are the ResourceDesc's known title layout.
        let desc = unsafe { words.as_ptr().cast::<crate::resource::ResourceDesc>().read_unaligned() };
        assert_eq!(crate::resource::create_at_address(
            std::ptr::null(), address as u64, &desc, 0, std::ptr::null(), &IUNKNOWN, &mut object,
        ), 0);
        object
    };
    let source = texture(source_va.as_mut_ptr(), 4, 3, 1);
    let output = texture(target_va.as_mut_ptr(), 5, 3, 4);
    let input = crate::resource::host_resource(source).expect("source");
    let dst = crate::resource::host_resource(output).expect("destination");

    crate::list::ensure_state(commands, source, D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32);
    // SAFETY: the targets and view live through recording.
    unsafe {
        let rtv = host.rtv(input, None);
        list.ClearRenderTargetView(rtv, &[1.0, 0.0, 0.0, 1.0], None);
        list.ClearRenderTargetView(
            rtv, &[0.0, 0.0, 1.0, 1.0],
            Some(&[RECT { left: 0, top: 0, right: 3, bottom: 2 }]),
        );
    }
    assert!(record(title_list, source, output, [3, 2], [5, 3]));
    crate::list::ensure_state(commands, output, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
    let (readback, footprint) = crate::capture::read_back(&host.device, list, dst).expect("read back the written image");
    commands.close();
    host.execute(&[commands]);
    let mut bytes = std::ptr::null_mut();
    // SAFETY: execute waited; each mapped row has twenty bytes of texels.
    unsafe {
        host.device.GetDeviceRemovedReason().unwrap();
        readback.Map(0, None, Some(&mut bytes)).unwrap();
        for y in 0..3 {
            let row = std::slice::from_raw_parts(bytes.cast::<u8>().add(footprint.Offset as usize + y * footprint.Footprint.RowPitch as usize), 20);
            assert_eq!(row, [[0u8, 0, 255, 255]; 5].as_flattened());
        }
        readback.Unmap(0, None);
    }
    // Objects are released through their existing COM ABI.
    for object in [title_list, allocator, output, source] {
        let release: extern "system" fn(This) -> u32 = crate::object::slot(object, 0x10);
        release(object);
    }
}
