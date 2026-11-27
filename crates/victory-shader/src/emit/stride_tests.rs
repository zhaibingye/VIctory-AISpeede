//! End-to-end tests of descriptors whose emission needs more than one
//! binding: a buffer read by element through the resource heap (no
//! used-resource entry, so no stride; the view-stride table gives it), and
//! one of two immediate samplers chosen by uniform control flow. Synthetic
//! GFX10 programs lowered to DXIL, validated by `dxil.dll` when present,
//! and run on the host GPU and WARP.

use windows::Win32::Graphics::Direct3D12::D3D12_SAMPLER_DESC;

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, StaticSampler, VIEW_STRIDES, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// User data: s0 the heap index of a structured buffer (a root constant),
/// s1 the resource heap, s2 a table with a raw UAV. Each lane stores dword
/// 1 of element `lane` to dword `lane` of the UAV.
const HEAP_ELEMENT_WAVE32: &[u32] = &[
    0x8F04_8500, // s_lshl_b32 s4, s0, 5
    0xBE86_0301, // s_mov_b32 s6, s1
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0203,
    0x0800_0000, // s_load_dwordx4 s[8:11], s[6:7], s4
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
    0x3402_0082, // v_lshlrev_b32 v1, 2, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE030_2004,
    0x8002_0200, // buffer_load_dword v2, v0, s[8:11], 0 idxen offset:4
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE070_1000,
    0x8004_0201, // buffer_store_dword v2, v1, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

/// User data: s0 a table with a 2x1 texture, s1 a root constant, s2 a table
/// with a raw UAV. The root constant picks the immediate sampler by a
/// branch, clamping (non-zero) or wrapping (zero); the program samples the
/// texture at u = 1.25 and stores red to dword 0 of the UAV.
const CHOSEN_SAMPLER_WAVE32: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF40C_0202,
    0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0x0
    0xBE86_0302, // s_mov_b32 s6, s2
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0403,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[6:7], 0x0
    0xBF06_8001, // s_cmp_eq_u32 s1, 0
    0xBF85_0006, // s_cbranch_scc1 -> 0x40
    0xBE94_03FF,
    0x0000_0092, // s_mov_b32 s20, 0x92 (clamp x, y, z)
    0xBE95_0380, // s_mov_b32 s21, 0
    0xBE96_0380, // s_mov_b32 s22, 0
    0xBE97_0380, // s_mov_b32 s23, 0
    0xBF82_0004, // s_branch -> 0x50
    0xBE94_0380, // 0x40: s_mov_b32 s20, 0 (wrap)
    0xBE95_0380, // s_mov_b32 s21, 0
    0xBE96_0380, // s_mov_b32 s22, 0
    0xBE97_0380, // s_mov_b32 s23, 0
    0x7E00_02FF,
    0x3FA0_0000, // 0x50: v_mov_b32 v0, 1.25
    0x7E02_02FF,
    0x3F00_0000, // v_mov_b32 v1, 0.5
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xF09C_0108,
    0x00A2_0200, // image_sample_lz v2, v[0:1], s[8:15], s[20:23] dmask:0x1 dim:2D
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x7E06_0280, // v_mov_b32 v3, 0
    0xE070_1000,
    0x8004_0203, // buffer_store_dword v2, v3, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

/// User data: s0 the heap index of a constant buffer (a root constant), s1
/// the resource heap, s2 a table with a raw UAV. Each lane stores dword 2
/// of the constant buffer (a scalar buffer load) to dword `lane` of the UAV.
const HEAP_CONSTANT_WAVE32: &[u32] = &[
    0x8F04_8500, // s_lshl_b32 s4, s0, 5
    0xBE86_0301, // s_mov_b32 s6, s1
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0203,
    0x0800_0000, // s_load_dwordx4 s[8:11], s[6:7], s4
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xF420_0504,
    0xFA00_0008, // s_buffer_load_dword s20, s[8:11], 0x8
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x7E02_0214, // v_mov_b32 v1, s20
    0x3404_0082, // v_lshlrev_b32 v2, 2, v0
    0xE070_1000,
    0x8004_0102, // buffer_store_dword v1, v2, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

/// User data: s0 a table with a raw SRV, s2 a table with a raw UAV. Each
/// lane loads the half at byte `2 × lane` and stores it to dword `lane`.
const HALF_LOAD_WAVE32: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0202,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
    0x3402_0081, // v_lshlrev_b32 v1, 1, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE028_1000,
    0x8002_0201, // buffer_load_ushort v2, v1, s[8:11], 0 offen
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x3406_0082, // v_lshlrev_b32 v3, 2, v0
    0xE070_1000,
    0x8004_0203, // buffer_store_dword v2, v3, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

/// User data: s2 a table with a raw UAV. Each lane stores its index as the
/// half at byte `2 × lane` (two lanes share each dword).
const HALF_STORE_WAVE32: &[u32] = &[
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
    0x3402_0081, // v_lshlrev_b32 v1, 1, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE068_1000,
    0x8004_0001, // buffer_store_short v0, v1, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

fn param(root: RootType, user_data: u8) -> RootParam {
    RootParam {
        root,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot: user_data,
    }
}

/// An entry at offset 0 of the table in user data `user_data`.
fn entry(kind: ResourceType, dwords: u8, user_data: u8) -> UsedResource {
    UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data,
        slot: user_data,
    }
}

fn build(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
    threads: u32,
) -> HostShader {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            rsrc2: 3 << 1,
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
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
    })
    .expect("lower");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        static_samplers: lowered
            .samplers
            .iter()
            .map(|&w| StaticSampler::from_words(w))
            .collect(),
        samplers: lowered.samplers,
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        memory: lowered.memory,
        memory_writes: lowered.memory_writes,
        view_addresses: lowered.view_addresses,
        heap: lowered.heap,
        heap_writes: lowered.heap_writes,
        view_marks: lowered.view_marks,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

#[test]
fn a_heap_buffer_read_by_element_takes_the_views_stride() {
    let shader = build(
        HEAP_ELEMENT_WAVE32,
        vec![
            param(RootType::Constants, 0),
            param(RootType::ResourceHeap, 1),
            param(RootType::Table, 2),
        ],
        vec![entry(ResourceType::Uav, 4, 2)],
        32,
    );
    assert_eq!(shader.root_srvs, [VIEW_STRIDES]);
    // 32 elements of 12 bytes: (100 + e, 200 + e, 300 + e).
    let data: Vec<u32> = (0..32).flat_map(|e| [100 + e, 200 + e, 300 + e]).collect();
    let views = [
        View::RawUav(32),
        View::Buffer {
            data: &data,
            stride: Stride::Structured(12),
        },
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match u.root {
                // The buffer is at heap index 1, the UAV's table at 0.
                RootType::Constants => 1,
                RootType::Table => 0,
                other => panic!("unexpected user data {other:?}"),
            };
        }
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    // User data, then the stride table: heap index 1 holds
                    // 12-byte elements.
                    roots: vec![
                        (0, Root::Constants(ud)),
                        (
                            1,
                            Root::View {
                                srv: true,
                                data: vec![4, 12],
                            },
                        ),
                    ],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        let want: Vec<u32> = (0..32).map(|l| 200 + l).collect();
        assert_eq!(words[0], want, "{label}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// A buffer read at a heap index from data whose slot holds no buffer view
/// (the view-stride table holds 0 there; a texture here) reads the host's
/// null buffer view instead (`emit`, "Buffer views at computed heap
/// indices"): a texture descriptor read as a buffer one is a wild address.
#[test]
fn a_heap_buffer_read_where_no_buffer_view_is_reads_zeros() {
    let shader = build(
        HEAP_ELEMENT_WAVE32,
        vec![
            param(RootType::Constants, 0),
            param(RootType::ResourceHeap, 1),
            param(RootType::Table, 2),
        ],
        vec![entry(ResourceType::Uav, 4, 2)],
        32,
    );
    let texel = [[1.0f32, 2.0, 3.0, 4.0]];
    let views = [View::RawUav(32), View::Texture(1, 1, &texel), View::NullBuffers];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            // The texture at heap index 1, the UAV's table at 0.
            ud[u.index as usize] = u32::from(u.root == RootType::Constants);
        }
        let dispatch = Dispatch {
            // Heap index 1 holds no buffer view: stride 0.
            roots: vec![(0, Root::Constants(ud)), (1, Root::View { srv: true, data: vec![4, 0] })],
            views: &views,
            samplers: &[],
            groups: [1, 1, 1],
        };
        let words = host
            .dispatch(&root, &pso, &dispatch)
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(words[0], [0; 32], "{label}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// Constant data through the resource heap itself is read from a raw view
/// (the host's constant buffer views need an alignment the title's lack).
#[test]
fn a_constant_buffer_through_the_heap_reads_as_a_raw_buffer() {
    let shader = build(
        HEAP_CONSTANT_WAVE32,
        vec![
            param(RootType::Constants, 0),
            param(RootType::ResourceHeap, 1),
            param(RootType::Table, 2),
        ],
        vec![entry(ResourceType::Uav, 4, 2)],
        32,
    );
    assert!(shader.heap && !shader.heap_writes, "reads the heap itself, writes through a table");
    assert_eq!(shader.root_srvs, [VIEW_STRIDES], "the guard's stride table");
    let data: Vec<u32> = vec![7, 8, 9, 10];
    let views = [
        View::RawUav(32),
        View::Buffer {
            data: &data,
            stride: Stride::Raw,
        },
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match u.root {
                // The constants are at heap index 1, the UAV's table at 0.
                RootType::Constants => 1,
                RootType::Table => 0,
                other => panic!("unexpected user data {other:?}"),
            };
        }
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    // User data, then the stride table: heap index 1 holds a
                    // buffer view.
                    roots: vec![(0, Root::Constants(ud)), (1, Root::View { srv: true, data: vec![4, 4] })],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(words[0], vec![9; 32], "{label}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// `buffer_load_ushort`: the half at its byte address, zero-extended (the
/// BVH builder reads 16-bit fields this way).
#[test]
fn a_half_buffer_load_reads_the_half_at_its_address() {
    let shader = build(
        HALF_LOAD_WAVE32,
        vec![param(RootType::Table, 0), param(RootType::Table, 2)],
        vec![entry(ResourceType::Srv, 4, 0), entry(ResourceType::Uav, 4, 2)],
        32,
    );
    // 32 halves 0x8000 + 3 × lane, two per dword (low half first).
    let half = |l: u32| 0x8000 + 3 * l;
    let data: Vec<u32> = (0..16).map(|k| half(2 * k) | half(2 * k + 1) << 16).collect();
    let views = [
        View::Buffer {
            data: &data,
            stride: Stride::Raw,
        },
        View::RawUav(32),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            // The SRV's table at heap index 0, the UAV's at 1.
            ud[u.index as usize] = if u.slot == 0 { 0 } else { 32 };
        }
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        let want: Vec<u32> = (0..32).map(half).collect();
        assert_eq!(words[0], want, "{label}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// `buffer_store_short`: each lane's half lands at its byte address and
/// leaves the other half of the dword alone.
#[test]
fn half_buffer_stores_leave_the_other_half_alone() {
    let shader = build(
        HALF_STORE_WAVE32,
        vec![param(RootType::Table, 2)],
        vec![entry(ResourceType::Uav, 4, 2)],
        32,
    );
    let views = [View::RawUav(16)];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        let want: Vec<u32> = (0..16).map(|k| (2 * k) | ((2 * k + 1) << 16)).collect();
        assert_eq!(words[0], want, "{label}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

#[test]
fn a_sampler_chosen_by_a_uniform_branch_samples_as_chosen() {
    let shader = build(
        CHOSEN_SAMPLER_WAVE32,
        vec![
            param(RootType::Table, 0),
            param(RootType::Constants, 1),
            param(RootType::Table, 2),
        ],
        vec![
            entry(ResourceType::Srv, 8, 0),
            entry(ResourceType::Uav, 4, 2),
        ],
        1,
    );
    assert_eq!(shader.samplers.len(), 2);
    // Red 1 at u < 0.5, 2 from there: u = 1.25 wraps to 0.25, clamps to 1.
    let texels = [[1.0f32, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]];
    let views = [View::Texture(2, 1, &texels), View::RawUav(1)];
    let samplers: [D3D12_SAMPLER_DESC; 0] = [];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (clamp, want) in [(1, 2.0f32), (0, 1.0)] {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Table, 0) => 0,
                    (RootType::Constants, 1) => clamp,
                    // The UAV at heap index 1: 32 bytes in.
                    (RootType::Table, 2) => 32,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let words = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &views,
                        samplers: &samplers,
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            assert_eq!(f32::from_bits(words[0][0]), want, "{label}, clamp {clamp}");
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}
// s0：常量表；s1：动态常量行；s2：输出表。同行的逐分量结果与静态
// 读取的标记一起写回，确保越界只影响动态读取，不会意外读取第零行。
const TABLE_CONSTANT_ROW_WAVE32: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0202,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0
    0x7E02_0201, // v_mov_b32 v1, s1
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xF420_0504,
    0xFA00_0000, // s_buffer_load_dword s20, s[8:11], 0
    0xE038_2000,
    0x8002_0201, // buffer_load_dwordx4 v[2:5], v1, s[8:11], 0 idxen
    0xBF8C_0070, // 等待向量和标量内存读取
    0x7E0C_0214, // v_mov_b32 v6, s20
    0xE078_0000,
    0x8004_0200, // buffer_store_dwordx4 v[2:5], s[16:19], 0
    0xE070_0010,
    0x8004_0600, // buffer_store_dword v6, s[16:19], 0 offset:16
    0xBF81_0000, // s_endpgm
];

#[test]
fn table_constant_dynamic_vector_does_not_wrap_or_read_past_the_view() {
    let shader = build(
        TABLE_CONSTANT_ROW_WAVE32,
        vec![
            param(RootType::Table, 0),
            param(RootType::Constants, 1),
            param(RootType::Table, 2),
        ],
        vec![entry(ResourceType::Cbv, 4, 0), entry(ResourceType::Uav, 4, 2)],
        1,
    );
    let first = [11, 12, 13, 14];
    let last = [21, 22, 23, 24];
    let mut full = vec![0u32; 16_384];
    full[..4].copy_from_slice(&first);
    full[16_380..].copy_from_slice(&last);
    let short = [11, 12, 13, 14, 25, 26];
    let neighbor = [0xCAFEu32; 4];
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let host = Host::new(false).or_else(|_| Host::new(true)).expect("需要 SM 6.6 实机或 WARP");
    let (root, pso) = host.compute_pipeline(&shader).expect("创建常量边界测试 PSO");
    for (data, index, expected) in [
        (full.as_slice(), 0, first),
        (&full, 4095, last),
        (&full, 4096, [0; 4]),
        (&full, 0x1000_0000, [0; 4]),
        (&full, u32::MAX, [0; 4]),
        (&short, 1, [25, 26, 0, 0]),
    ] {
        let mut ud = vec![0; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match (u.root, u.index) {
                (RootType::Table, 0) => 0,
                (RootType::Constants, 1) => index,
                (RootType::Table, 2) => 2 * 32, // 表根持有堆字节地址，不是槽号。
                other => panic!("未知绑定 {other:?}"),
            };
        }
        let views = [
            View::Buffer { data, stride: Stride::Raw },
            View::Buffer { data: &neighbor, stride: Stride::Raw },
            View::RawUav(5),
        ];
        let words = host.dispatch(
            &root, &pso, &Dispatch {
                roots: vec![(0, Root::Constants(ud))], views: &views, samplers: &[], groups: [1, 1, 1],
            },
        ).expect("运行不含游戏数据的常量边界测试");
        assert_eq!(&words[0][..4], &expected, "索引 {index:#x}");
        assert_eq!(words[0][4], first[0], "静态读取必须使用同一常量表");
    }
}

#[test]
fn scalar_structured_table_load_uses_byte_offsets_across_records() {
    // 两个表分别绑定输入和输出。结构化输入每记录 64 字节，
    // 标量读从第 60 字节开始，四个 dword 跨过记录边界。
    const CODE: &[u32] = &[
        0xBE84_0300, 0xBE85_0380, 0xF408_0202, 0xFA00_0000,
        0xBE8C_0302, 0xBE8D_0380, 0xF408_0406, 0xFA00_0000,
        0xBF8C_C07F,
        0xF428_0504, 0xFA00_003C, // s_buffer_load_dwordx4 s[20:23], s[8:11], 60
        0xBF8C_C07F,
        0x7E04_0214, 0x7E06_0215, 0x7E08_0216, 0x7E0A_0217,
        0xE078_0000, 0x8004_0200, // buffer_store_dwordx4 v[2:5], s[16:19], 0
        0xBF81_0000,
    ];
    let shader = build(
        CODE,
        vec![param(RootType::Table, 0), param(RootType::Table, 2)],
        vec![
            UsedResource { stride: 64, ..entry(ResourceType::Srv, 4, 0) },
            entry(ResourceType::Uav, 4, 2),
        ],
        1,
    );
    // 对照槽与输入使用不同数据，防止错误重定位得到同样的结果。
    let neighbor = [0xDEADu32; 32];
    let data: Vec<u32> = (100..132).collect();
    let views = [
        View::Buffer { data: &neighbor, stride: Stride::Raw },
        View::Buffer { data: &data, stride: Stride::Raw },
        View::RawUav(4),
    ];
    let mut words = vec![0; shader.user_data_dwords() as usize];
    for u in &shader.user_data {
        words[u.index as usize] = match (u.root, u.index) {
            (RootType::Table, 0) => 32,
            (RootType::Table, 2) => 64,
            other => panic!("未知绑定 {other:?}"),
        };
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let host = Host::new(false).or_else(|_| Host::new(true)).expect("需要 SM 6.6 实机或 WARP");
    let (root, pso) = host.compute_pipeline(&shader).expect("创建标量缓冲读取测试 PSO");
    let actual = host.dispatch(
        &root, &pso, &Dispatch {
            roots: vec![(0, Root::Constants(words))], views: &views, samplers: &[], groups: [1, 1, 1],
        },
    ).expect("运行标量缓冲读取测试");
    assert_eq!(actual[0], [115, 116, 117, 118]);
}
