//! Single console instructions as the lowering computes them, against what
//! the console GPU computes.
//!
//! Each probe is a hand-assembled compute program around the instruction
//! under test (its encoding and modifiers), lowered as a title's program
//! is, run on the hardware adapter and on WARP over edge inputs, and
//! compared with a CPU model of the hardware. A probe fails when a result
//! differs from the model in kind (NaN, infinity, zero and its sign,
//! integers, decisions) or an ordinary float result lies outside the
//! probe's tolerance in ULPs, and when the two adapters disagree: the
//! translation must not lean on the host's choices.
//!
//! The model is RDNA2 in the console's float mode: IEEE mode off, DX10
//! clamp on, round to nearest even, 32-bit denormals flushed on input and
//! output, 16- and 64-bit denormals kept. Its sources are named per family:
//! AMD's RDNA2 ISA reference guide (30 November 2020) and measurements on
//! RDNA2 hardware published by others. Probes whose result no public source
//! fixes (`model: None`) only check that the two adapters agree.

use super::host::{Dispatch, Host, Root, Stride, View};
use super::valu_tests::ONE_AT_A_TIME;
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{gfx10, io, ir, resource, structure};

// ---- encodings ----

const S_ENDPGM: u32 = 0xBF81_0000;

const fn v(n: u8) -> u16 {
    256 + n as u16
}

const fn s(n: u8) -> u16 {
    n as u16
}

/// An inline integer constant 0..=64.
const fn int(n: u8) -> u16 {
    128 + n as u16
}

const VCC: u16 = 106;
const EXEC_LO: u16 = 126;
const SRC_SCC: u16 = 253;

/// VOP3 modifiers: `abs`/`neg` per source (bit k), `clamp`, the output
/// modifier (1 `mul:2`, 2 `mul:4`, 3 `div:2`) and `op_sel`.
#[derive(Clone, Copy, Default)]
struct M {
    abs: u8,
    neg: u8,
    clamp: bool,
    omod: u8,
    opsel: u8,
}

const NONE: M = M {
    abs: 0,
    neg: 0,
    clamp: false,
    omod: 0,
    opsel: 0,
};

const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

const fn vop2(op: u16, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOPC: the lane mask to VCC (`v_cmpx`: to EXEC).
const fn vopc(op: u16, src0: u16, vsrc1: u8) -> u32 {
    0x7C00_0000 | (op as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3A. VOP2 opcodes are `0x100 + op` here, VOP1 `0x180 + op`, VOPC as is
/// (`vdst` then names the SGPR of the mask).
fn vop3(op: u16, vdst: u8, src: [u16; 3], m: M) -> [u32; 2] {
    [
        0xD400_0000
            | u32::from(op) << 16
            | u32::from(m.clamp) << 15
            | u32::from(m.opsel & 0xF) << 11
            | u32::from(m.abs & 7) << 8
            | u32::from(vdst),
        u32::from(m.neg & 7) << 29
            | u32::from(m.omod & 3) << 27
            | u32::from(src[2] & 0x1FF) << 18
            | u32::from(src[1] & 0x1FF) << 9
            | u32::from(src[0] & 0x1FF),
    ]
}

/// VOP3B: a carry (or other lane mask) out to `sdst`.
fn vop3b(op: u16, vdst: u8, sdst: u8, src: [u16; 3], clamp: bool) -> [u32; 2] {
    [
        0xD400_0000
            | u32::from(op) << 16
            | u32::from(clamp) << 15
            | u32::from(sdst) << 8
            | u32::from(vdst),
        u32::from(src[2] & 0x1FF) << 18
            | u32::from(src[1] & 0x1FF) << 9
            | u32::from(src[0] & 0x1FF),
    ]
}

/// VOP3P: `op_sel` and `op_sel_hi` per source (bit k), `neg_lo`/`neg_hi`.
fn vop3p(op: u16, vdst: u8, src: [u16; 3], opsel: u8, opsel_hi: u8, neg: [u8; 2]) -> [u32; 2] {
    [
        0xCC00_0000
            | u32::from(op) << 16
            | u32::from(opsel_hi >> 2 & 1) << 14
            | u32::from(opsel & 7) << 11
            | u32::from(neg[1] & 7) << 8
            | u32::from(vdst),
        u32::from(neg[0] & 7) << 29
            | u32::from(opsel_hi & 3) << 27
            | u32::from(src[2] & 0x1FF) << 18
            | u32::from(src[1] & 0x1FF) << 9
            | u32::from(src[0] & 0x1FF),
    ]
}

/// SDWA: the VOP1/VOP2 word (its src0 field replaced) and the SDWA dword.
/// `sel` 0-3 a byte, 4-5 a word, 6 the dword; `unused` 0 pad, 1 sign
/// extend, 2 preserve.
#[derive(Clone, Copy)]
struct Sdwa {
    dst_sel: u8,
    unused: u8,
    sel: [u8; 2],
    sext: [bool; 2],
    neg: [bool; 2],
    abs: [bool; 2],
    clamp: bool,
}

const SDWA: Sdwa = Sdwa {
    dst_sel: 6,
    unused: 0,
    sel: [6, 6],
    sext: [false, false],
    neg: [false, false],
    abs: [false, false],
    clamp: false,
};

fn sdwa(word: u32, src0: u8, x: Sdwa) -> [u32; 2] {
    [
        word & !0x1FF | 0xF9,
        u32::from(src0)
            | u32::from(x.dst_sel) << 8
            | u32::from(x.unused) << 11
            | u32::from(x.clamp) << 13
            | u32::from(x.sel[0]) << 16
            | u32::from(x.sext[0]) << 19
            | u32::from(x.neg[0]) << 20
            | u32::from(x.abs[0]) << 21
            | u32::from(x.sel[1]) << 24
            | u32::from(x.sext[1]) << 27
            | u32::from(x.neg[1]) << 28
            | u32::from(x.abs[1]) << 29,
    ]
}

const fn sop1(op: u16, sdst: u8, src0: u16) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | (op as u32) << 8 | (src0 as u32 & 0xFF)
}

const fn sop2(op: u16, sdst: u8, src0: u16, src1: u16) -> u32 {
    0x8000_0000
        | (op as u32) << 23
        | (sdst as u32) << 16
        | (src1 as u32 & 0xFF) << 8
        | (src0 as u32 & 0xFF)
}

const fn sopc(op: u16, src0: u16, src1: u16) -> u32 {
    0xBF00_0000 | (op as u32) << 16 | (src1 as u32 & 0xFF) << 8 | (src0 as u32 & 0xFF)
}

/// `s_load_dwordx4 s[dst:dst+3], s[base:base+1], offset`.
const fn s_load_dwordx4(dst: u8, base: u8, offset: u32) -> [u32; 2] {
    [
        0xF408_0000 | (dst as u32) << 6 | (base as u32) >> 1,
        125 << 25 | offset,
    ]
}

/// MUBUF `op vdata, vaddr, s[srsrc:srsrc+3], 0 offen`.
const fn mubuf(op: u32, vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
    [
        0xE000_0000 | op << 18 | 1 << 12,
        0x80 << 24 | ((srsrc as u32) >> 2) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

/// `v_cndmask_b32 v5, 0, 1, vcc`: a lane mask in VCC as a number.
fn vcc_to_v5() -> [u32; 2] {
    vop3(0x101, 5, [int(0), int(1), VCC], NONE)
}

/// A compare into VCC, then [`vcc_to_v5`].
fn with_vcc(word: u32) -> Vec<u32> {
    let mut w = vec![word];
    w.extend(vcc_to_v5());
    w
}

/// The program around a probe. Inputs a..d arrive in v1..v4 (and, for
/// scalar probes, a wave-uniform copy in s16..s19); results go out from
/// v5..v8, which start as d. Per-lane probes give thread `t` input `t`;
/// scalar probes give every thread of group `g` input `g`.
fn program(body: &[u32], scalar: bool) -> Vec<u32> {
    let mut w = vec![sop1(0x03, 5, int(0))];
    w.extend(s_load_dwordx4(8, 4, 0));
    w.extend(s_load_dwordx4(12, 4, 0x20));
    // v10 = (group * 64 + thread) * 16, this thread's output.
    w.extend(vop3(0x346, 10, [s(6), int(6), v(0)], NONE));
    w.push(vop2(0x1A, 10, int(4), 10));
    if scalar {
        w.extend(vop3(0x11A, 9, [int(4), s(6), 0], NONE));
    } else {
        w.push(vop1(0x01, 9, v(10)));
    }
    w.extend(mubuf(0x0E, 1, 9, 8));
    for k in 5..9 {
        w.push(vop1(0x01, k, v(4)));
    }
    if scalar {
        for k in 0..4 {
            w.push(vop1(0x02, 16 + k, v(1 + k)));
        }
    }
    w.extend_from_slice(body);
    w.extend(mubuf(0x1E, 5, 10, 12));
    w.push(S_ENDPGM);
    w
}

/// Lowers a probe's program: a 64-thread group of two waves of 32, 512
/// bytes of LDS; one view table at s4 (the input buffer, then the output
/// UAV), the group id in s6.
fn lower_program(words: &[u32]) -> Result<HostShader, String> {
    let used = |kind, table_offset_dwords| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 4,
        slot: 0,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 4,
            slot: 0,
        }],
        used: vec![used(ResourceType::Srv, 0), used(ResourceType::Uav, 8)],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 6 << 1 | 1 << 7 | 1 << 15,
            threads: [64, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let e = |e: crate::ShaderError| e.to_string();
    let p = ir::build(&native.code, native.wave64, native.vgprs).map_err(e)?;
    let st = structure::structure(&p).map_err(e)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).map_err(e)?;
    let io = io::raise(&p, &native).map_err(e)?;
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
    .map_err(e)?;
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

/// The body's instructions as the decoder prints them.
fn disassembly(body: &[u32], scalar: bool) -> String {
    let words = program(body, scalar);
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let insts = gfx10::decode(&code).expect("the probe decodes");
    let prologue = program(&[], scalar).len() - 3;
    let (lo, hi) = (4 * prologue, 4 * (prologue + body.len()));
    insts
        .iter()
        .filter(|i| (lo..hi).contains(&i.offset))
        .map(|i| i.text(false))
        .collect::<Vec<_>>()
        .join("; ")
}

/// Runs a lowered probe on `host` over `inputs`: each input's v5..v8.
fn run(
    host: &Host,
    shader: &HostShader,
    inputs: &[[u32; 4]],
    scalar: bool,
) -> Result<Vec<[u32; 4]>, String> {
    let (root, pso) = host.compute_pipeline(shader)?;
    let groups = if scalar {
        inputs.len()
    } else {
        inputs.len().div_ceil(64)
    };
    let mut data: Vec<u32> = inputs.iter().flatten().copied().collect();
    data.resize(
        data.len()
            .max(4 * if scalar { groups } else { 64 * groups }),
        0,
    );
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        assert_eq!((u.root, u.slot), (RootType::Table, 0), "user data {u:?}");
        ud[u.index as usize] = 0;
    }
    let views = [
        View::Buffer {
            data: &data,
            stride: Stride::Raw,
        },
        View::RawUav(groups as u32 * 64 * 4),
    ];
    let mut roots = vec![(0, Root::Constants(ud))];
    if shader.view_addresses {
        // The first slot has a buffer address. The second has only a native
        // base word. An SGPR reading word zero sees both at the same stride.
        roots.push((
            1,
            Root::View {
                srv: true,
                data: vec![0x1234_5600, 12, 0x89AB_CD00, super::VIEW_BASE_ONLY],
            },
        ));
    }
    let out = host.dispatch(
        &root,
        &pso,
        &Dispatch {
            roots,
            views: &views,
            samplers: &[],
            groups: [groups as u32, 1, 1],
        },
    )?;
    let lane = |t: usize| -> [u32; 4] { out[0][4 * t..4 * t + 4].try_into().expect("4 dwords") };
    Ok((0..inputs.len())
        .map(|i| {
            if scalar {
                // Both waves of the group must agree.
                let (a, b) = (lane(64 * i), lane(64 * i + 33));
                if a == b { a } else { [0xBAD0_BAD0; 4] }
            } else {
                lane(i)
            }
        })
        .collect())
}

fn check_wave32(
    name: &str,
    body: &[u32],
    inputs: &[[u32; 4]],
    expected: impl Fn(usize) -> [u32; 4],
) {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_FEATURE_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS1,
    };

    let shader =
        lower_program(&program(body, false)).unwrap_or_else(|e| panic!("lower {name}: {e}"));
    if let Ok(mut val) = validator::Validator::load(validator::Validator::default_path().as_deref())
    {
        val.validate(&mut shader.dxil.clone())
            .expect("validate the shader's DXIL");
        eprintln!("{name} DXIL validated");
    }

    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 device for {name}");
        return;
    };
    let mut waves = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    // SAFETY: the feature query writes the matching options structure.
    let support = unsafe {
        host.device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS1,
            (&raw mut waves).cast(),
            size_of_val(&waves) as u32,
        )
    };
    if support.is_err()
        || !waves.WaveOps.as_bool()
        || !(waves.WaveLaneCountMin..=waves.WaveLaneCountMax).contains(&32)
    {
        eprintln!("Wave32 unavailable for the loop test");
        return;
    }

    let out = run(&host, &shader, inputs, false).expect("dispatch the shader");
    assert_eq!(out.len(), inputs.len());
    for (t, actual) in out.into_iter().enumerate() {
        assert_eq!(actual, expected(t), "thread {t}");
    }
    eprintln!("{name} readback matched on {} threads", inputs.len());
}

/// A single loop entry supplies a Wave32 mask, and the latch replaces it
/// with integer data. Each iteration reads the dword and the lane's bit.
/// A wave-uniform branch then joins a masked dword with an integer.
#[test]
fn wave32_mask_data_at_loop_entry() {
    let mut body = Vec::from(vop3(0x0C5, 22, [int(0), v(1), 0], NONE)); // s22 = v1 != 0
    body.extend([
        sop1(0x03, 20, s(22)),  // s_mov_b32 s20, s22
        sop1(0x03, 24, int(0)), // s_mov_b32 s24, 0
        // The loop starts here. Save the previous and current values.
        sop1(0x03, 22, s(20)), // s_mov_b32 s22, s20
        vop1(0x01, 5, v(6)),   // v_mov_b32 v5, v6
        vop1(0x01, 6, s(22)),  // v_mov_b32 v6, s22
        vop1(0x01, 7, v(8)),   // v_mov_b32 v7, v8
    ]);
    body.extend(vop3(0x101, 8, [int(0), int(1), s(20)], NONE)); // v_cndmask_b32 v8, 0, 1, s20
    body.extend([
        0xB014_5319,                   // s_movk_i32 s20, 0x5319
        sop2(0x00, 24, s(24), int(1)), // s_add_u32 s24, s24, 1
        sopc(0x0A, s(24), int(2)),     // s_cmp_lt_u32 s24, 2
        0xBF85_FFF6,                   // s_cbranch_scc1 -10
        // After the loop, one wave takes each arm of a numeric join.
        vop1(0x02, 26, v(2)),           // v_readfirstlane_b32 s26, v2
        sopc(0x06, s(26), int(0)),      // s_cmp_eq_u32 s26, 0
        0xBF85_0002,                    // s_cbranch_scc1 2
        sop2(0x0E, 28, s(22), int(63)), // s_and_b32 s28, s22, 63
        0xBF82_0001,                    // s_branch 1
        sop1(0x03, 28, int(42)),        // s_mov_b32 s28, 42
        vop1(0x01, 9, s(28)),           // v_mov_b32 v9, s28
    ]);
    body.extend(vop3(0x346, 8, [v(9), int(1), v(8)], NONE)); // pack the dword and bit
    let masks: [u32; 2] = [0x9123_4321, 0x8321_1234];
    let inputs: [[u32; 4]; 64] =
        std::array::from_fn(|t| [(masks[t / 32] >> (t % 32)) & 1, (t / 32) as u32, 0, 0]);
    check_wave32("Wave32 loop", &body, &inputs, |t| {
        let lane = t % 32;
        let first = masks[t / 32];
        let second = 0x5319;
        let joined = if t < 32 { 42 } else { second & 63 };
        [
            first,
            second,
            (first >> lane) & 1,
            (joined << 1) | ((second >> lane) & 1),
        ]
    });
}

/// A compare mask and a scalar phi supply the lane bits on different
/// edges of a forward join. The scalar phi selects two computed words.
#[test]
fn wave32_forward_mask_join_uses_scalar_input() {
    let mut body = vec![
        vop1(0x02, 26, v(2)),          // v_readfirstlane_b32 s26, v2
        vop1(0x02, 27, v(3)),          // v_readfirstlane_b32 s27, v3
        sopc(0x06, s(26), int(0)),     // s_cmp_eq_u32 s26, 0
        0xBF85_0007,                   // s_cbranch_scc1 compare
        sopc(0x06, s(26), int(1)),     // s_cmp_eq_u32 s26, 1
        0xBF85_0002,                   // s_cbranch_scc1 add_one
        sop2(0x00, 20, s(27), int(2)), // s_add_u32 s20, s27, 2
        0xBF82_0001,                   // s_branch scalar_join
        sop2(0x00, 20, s(27), int(1)), // add_one: s_add_u32 s20, s27, 1
        vop1(0x01, 7, s(20)),          // scalar_join: v_mov_b32 v7, s20
        0xBF82_0003,                   // s_branch mask_join
    ];
    body.extend(vop3(0x0C5, 20, [int(0), v(1), 0], NONE)); // compare: s20 = v1 != 0
    body.push(vop1(0x01, 7, int(0)));
    body.extend(vop3(0x101, 5, [int(0), int(1), s(20)], NONE)); // mask_join: this lane's bit
    body.extend(vop3(0x365, 6, [193, int(0), 0], NONE)); // v_mbcnt_lo_u32_b32 v6, -1, 0

    let compare_mask: u32 = 0x8AA5_1243;
    let seeds = [0, 0xA622_B270, 0xF123_4500, 0x62AA_1230];
    let inputs: [[u32; 4]; 128] = std::array::from_fn(|t| {
        [
            (compare_mask >> (t % 32)) & 1,
            (t / 32) as u32,
            seeds[t / 32],
            0,
        ]
    });
    check_wave32("Wave32 forward mask join", &body, &inputs, |t| {
        let wave = t / 32;
        let lane = (t % 32) as u32;
        let value = seeds[wave] + if wave == 1 { 1 } else { 2 };
        let (mask, word) = if wave == 0 {
            (compare_mask, 0)
        } else {
            (value, value)
        };
        [(mask >> lane) & 1, lane, word, 0]
    });
}

/// One incoming scalar word passes through an earlier join. A branch
/// guarantees that the descriptor word on its other edge is never read.
#[test]
fn wave32_mask_join_reads_only_the_live_scalar_word() {
    let mut body = vec![
        vop1(0x02, 26, v(2)),          // v_readfirstlane_b32 s26, v2
        vop1(0x02, 27, v(3)),          // v_readfirstlane_b32 s27, v3
        sopc(0x06, s(26), int(0)),     // s_cmp_eq_u32 s26, 0
        0xBF85_0003,                   // s_cbranch_scc1 descriptor
        sop2(0x00, 20, s(27), int(1)), // s_add_u32 s20, s27, 1
        sop1(0x03, 22, int(1)),        // s_mov_b32 s22, 1
        0xBF82_0002,                   // s_branch test_valid
        sop1(0x03, 20, s(8)),          // descriptor: s_mov_b32 s20, s8
        sop1(0x03, 22, int(0)),        // s_mov_b32 s22, 0
        sopc(0x06, s(22), int(0)),     // test_valid: s_cmp_eq_u32 s22, 0
        0xBF85_0002,                   // s_cbranch_scc1 compare
        vop1(0x01, 7, int(1)),         // v_mov_b32 v7, 1
        0xBF82_0003,                   // s_branch mask_join
    ];
    body.extend(vop3(0x0C5, 20, [int(0), v(1), 0], NONE)); // compare: s20 = v1 != 0
    body.push(vop1(0x01, 7, int(0)));
    body.extend(vop3(0x101, 5, [int(0), int(1), s(20)], NONE)); // mask_join: this lane's bit
    body.extend(vop3(0x365, 6, [193, int(0), 0], NONE)); // v_mbcnt_lo_u32_b32 v6, -1, 0

    let compare_mask: u32 = 0x8421_35A7;
    let scalar_seed = 0x9122_53E0;
    let inputs: [[u32; 4]; 64] = std::array::from_fn(|t| {
        [
            (compare_mask >> (t % 32)) & 1,
            (t / 32) as u32,
            scalar_seed,
            0,
        ]
    });
    check_wave32("Wave32 live scalar join", &body, &inputs, |t| {
        let lane = (t % 32) as u32;
        let (mask, branch) = if t < 32 {
            (compare_mask, 0)
        } else {
            (scalar_seed + 1, 1)
        };
        [(mask >> lane) & 1, lane, branch, 0]
    });
}

/// Both EXEC arms can run in one wave. Scalar writes take effect in program
/// order, including the unconditional write following an empty EXEC update.
#[test]
fn wave32_scalar_write_order_across_exec_branches() {
    let mut body = Vec::from(vop3(0x0C5, 20, [int(0), v(1), 0], NONE));
    body.extend(vop3(0x0C5, 30, [int(0), v(2), 0], NONE));
    body.extend([
        sop1(0x3C, 24, s(30)),         // save EXEC, then keep the selected lanes
        0xBF88_0002,                   // skip the first arm if EXEC is empty
        0xB014_0531,                   // s_movk_i32 s20, 0x0531
        vop1(0x01, 7, s(20)),          // keep the first arm's per-lane write
        sop2(0x14, 126, s(24), s(30)), // EXEC = saved & ~selected
        sop1(0x03, 29, int(37)),       // runs even if the second arm is empty
        0xBF88_0001,
        0xB014_7528,            // s_movk_i32 s20, 0x7528
        sop1(0x03, 126, s(24)), // restore EXEC
        vop1(0x01, 5, s(20)),
        vop1(0x01, 8, s(29)),
    ]);
    body.extend(vop3(0x101, 6, [int(0), int(1), s(20)], NONE));
    let selected = [0u32, u32::MAX, 0x8000_0001, 0xAAAA_5555];
    let inputs: [[u32; 4]; 128] = std::array::from_fn(|t| {
        [
            u32::from(t % 3 == 0),
            (selected[t / 32] >> (t % 32)) & 1,
            0,
            0,
        ]
    });
    check_wave32("Wave32 scalar write order", &body, &inputs, |t| {
        let mask = selected[t / 32];
        let lane = t % 32;
        let last = if mask == u32::MAX { 0x0531 } else { 0x7528 };
        let first = if mask >> lane & 1 != 0 { 0x0531 } else { 0 };
        [last, last >> lane & 1, first, 37]
    });
}

/// A word taken under a narrowed EXEC still contains bits outside that
/// subset. If EXEC is empty the branch skips the scalar complement entirely.
#[test]
fn wave32_complete_mask_after_a_conditional_complement() {
    let mut body = Vec::from(vop3(0x0C5, 20, [int(0), v(1), 0], NONE));
    body.extend(vop3(0x0C5, 30, [int(0), v(2), 0], NONE));
    body.extend([
        sop1(0x3C, 24, s(30)),
        0xBF88_0002,
        sop1(0x07, 20, s(20)), // s_not_b32 s20, s20
        vop1(0x01, 7, s(20)),  // observed inside the narrowed EXEC
        sop1(0x03, 126, s(24)),
        vop1(0x01, 5, s(20)), // all lanes read the same word
    ]);
    body.extend(vop3(0x101, 6, [int(0), int(1), s(20)], NONE));
    let original = [0x9142_2881u32, 0x5A11_44A2, 0x8080_1010, 0x8081_1200];
    let selected = [0u32, u32::MAX, 1 << 31, 0x5555_AAAA];
    let inputs: [[u32; 4]; 128] = std::array::from_fn(|t| {
        let (w, l) = (t / 32, t % 32);
        [original[w] >> l & 1, selected[w] >> l & 1, 0, 0]
    });
    check_wave32("Wave32 conditional complement", &body, &inputs, |t| {
        let (w, l) = (t / 32, t % 32);
        let word = if selected[w] == 0 {
            original[w]
        } else {
            !original[w]
        };
        let inside = if selected[w] >> l & 1 == 0 { 0 } else { word };
        [word, word >> l & 1, inside, 0]
    });
}

/// A scalar mask survives its native branch. The same dispatch reads
/// descriptor word zero from two different heap slots.
#[test]
fn wave32_descriptor_words_after_a_conditional_mask_write() {
    let mut body = Vec::from(vop3(0x0C5, 20, [int(0), v(1), 0], NONE));
    body.extend(vop3(0x0C5, 30, [int(0), v(2), 0], NONE));
    body.extend([
        sop1(0x3C, 24, s(30)),
        0xBF88_0002,
        sop1(0x07, 20, s(20)),
        vop1(0x01, 7, s(20)),
        sop1(0x03, 126, s(24)),
        vop1(0x01, 5, s(20)),
        vop1(0x01, 6, s(8)),
        vop1(0x01, 8, s(12)),
    ]);
    let inputs: [[u32; 4]; 64] =
        std::array::from_fn(|t| [u32::from(t % 2 == 0), u32::from(t == 63), 0, 0]);
    check_wave32("Wave32 descriptor words", &body, &inputs, |t| {
        let word = if t < 32 { 0x5555_5555 } else { 0xAAAA_AAAA };
        [
            word,
            0x1234_5600,
            if t == 63 { word } else { 0 },
            0x89AB_CD00,
        ]
    });
}

/// Read the native first active lane (lane zero under empty EXEC), then
/// permute across the two 16-lane rows with both inactive-source policies.
#[test]
fn wave32_lane_reads_use_the_native_exec_word() {
    let mut body = Vec::from(vop3(0x0C5, 20, [int(0), v(1), 0], NONE));
    body.extend(vop3(0x0C5, 30, [int(0), v(2), 0], NONE));
    body.extend([
        sop1(0x03, 25, 255),
        0x7654_3210,
        sop1(0x03, 26, 255),
        0xFEDC_BA98,
        sop1(0x3C, 24, s(30)),
        0xBF88_0001,
        sop1(0x07, 20, s(20)),
        vop1(0x02, 21, v(3)),
    ]);
    body.extend(vop3(0x378, 7, [v(3), s(25), s(26)], NONE));
    body.extend(vop3(0x378, 8, [v(3), s(25), s(26)], M { opsel: 2, ..NONE }));
    body.extend([
        sop1(0x03, 126, s(24)),
        vop1(0x01, 5, s(20)),
        vop1(0x01, 6, s(21)),
    ]);
    let selected = [0u32, u32::MAX, 0x8000_8001, 0xAAAA_5555];
    let inputs: [[u32; 4]; 128] = std::array::from_fn(|t| {
        [
            u32::from(t % 2 == 0),
            selected[t / 32] >> (t % 32) & 1,
            t as u32 + 100,
            57,
        ]
    });
    check_wave32("Wave32 lane reads", &body, &inputs, |t| {
        let mask = selected[t / 32];
        let lane = t % 32;
        let first = if mask == 0 {
            0
        } else {
            mask.trailing_zeros() as usize
        };
        let on = mask >> lane & 1 != 0;
        let other = lane ^ 16;
        let source = if mask >> other & 1 != 0 {
            Some((t / 32 * 32 + other) as u32 + 100)
        } else {
            None
        };
        [
            if mask == 0 { 0x5555_5555 } else { 0xAAAA_AAAA },
            (t / 32 * 32 + first) as u32 + 100,
            if on { source.unwrap_or(57) } else { 57 },
            if on { source.unwrap_or(0) } else { 57 },
        ]
    });
}

// ---- comparing ----

/// How a result dword is compared with the model's.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// Bit for bit.
    Int,
    /// An f32: NaN matches any NaN; zeros, infinities exactly (a zero of
    /// either sign with `any_zero`); finite values within `ulps`.
    F32 { ulps: u32, any_zero: bool },
    /// An f16 in the low half (as `F32`), the high half bit for bit.
    F16 { ulps: u32, any_zero: bool },
    /// Two f16s, each as `F32`.
    Halves { ulps: u32 },
    /// An f32 as `F32`, finite values within `abs` of each other (functions
    /// whose host error is bounded absolutely, as sine and log near 1).
    Abs { abs: f32 },
}

const INT: Kind = Kind::Int;
const F32: Kind = Kind::F32 {
    ulps: 0,
    any_zero: false,
};
const F16: Kind = Kind::F16 {
    ulps: 0,
    any_zero: false,
};

fn close(sign: u32, exp_mask: u32, a: u32, b: u32, ulps: u32, any_zero: bool) -> bool {
    let mag = sign - 1;
    let nan = |x: u32| x & mag > exp_mask;
    let zero = |x: u32| x & mag == 0;
    if nan(a) || nan(b) {
        return nan(a) && nan(b);
    }
    // A result rounded up to the smallest normal from below it: whether the
    // hardware flushes by the value before or after rounding is not public.
    let tiny = |x: u32, y: u32| x & mag == 0x0080_0000 && y == x & sign;
    if zero(a) || zero(b) {
        return a == b || any_zero && zero(a) && zero(b) || sign == 0x8000_0000 && tiny(a, b);
    }
    if a & mag == exp_mask || b & mag == exp_mask {
        return a == b;
    }
    a & sign == b & sign && (a & mag).abs_diff(b & mag) <= ulps
}

fn same(kind: Kind, want: u32, got: u32) -> bool {
    match kind {
        Kind::Int => want == got,
        Kind::F32 { ulps, any_zero } => close(0x8000_0000, 0x7F80_0000, want, got, ulps, any_zero),
        Kind::F16 { ulps, any_zero } => {
            want >> 16 == got >> 16
                && close(0x8000, 0x7C00, want & 0xFFFF, got & 0xFFFF, ulps, any_zero)
        }
        Kind::Abs { abs } => {
            let inf = |x: u32| x & 0x7FFF_FFFF == 0x7F80_0000;
            if nan(want) || nan(got) || inf(want) || inf(got) {
                return close(0x8000_0000, 0x7F80_0000, want, got, 0, false);
            }
            (fl(want) - fl(got)).abs() <= abs || close(0x8000_0000, 0x7F80_0000, want, got, 4, true)
        }
        Kind::Halves { ulps } => {
            close(0x8000, 0x7C00, want & 0xFFFF, got & 0xFFFF, ulps, false)
                && close(0x8000, 0x7C00, want >> 16, got >> 16, ulps, false)
        }
    }
}

/// One instruction (or a short sequence) under test.
struct Probe {
    body: Vec<u32>,
    scalar: bool,
    inputs: Vec<[u32; 4]>,
    /// The hardware's v5.. for an input; `None`: no source fixes it.
    model: Option<Model>,
    kind: Kind,
    /// Known defects of the lowering, each with the inputs it shows on: the
    /// passing tests leave those inputs out, each defect's ignored test
    /// runs only them.
    defects: Vec<(&'static str, Pred)>,
}

type Pred = Box<dyn Fn([u32; 4]) -> bool>;
type Model = Box<dyn Fn([u32; 4]) -> Vec<u32>>;

fn probe(
    body: impl Into<Vec<u32>>,
    inputs: Vec<[u32; 4]>,
    kind: Kind,
    model: impl Fn([u32; 4]) -> Vec<u32> + 'static,
) -> Probe {
    Probe {
        body: body.into(),
        scalar: false,
        inputs,
        model: Some(Box::new(model)),
        kind,
        defects: Vec::new(),
    }
}

impl Probe {
    /// Marks the inputs `on` holds for as showing `defect`.
    fn shows(mut self, defect: &'static str, on: impl Fn([u32; 4]) -> bool + 'static) -> Self {
        self.defects.push((defect, Box::new(on)));
        self
    }

    /// Marks the inputs where the hardware's result and `ours` (what the
    /// lowering computes instead) differ by the probe's comparison.
    fn shows_where(
        self,
        defect: &'static str,
        hw: impl Fn([u32; 4]) -> Vec<u32> + 'static,
        ours: impl Fn([u32; 4]) -> Vec<u32> + 'static,
    ) -> Self {
        let kind = self.kind;
        self.shows(defect, move |i| {
            !hw(i).iter().zip(ours(i)).all(|(&a, b)| same(kind, a, b))
        })
    }
}

/// Tags whose inputs the lowering still gets wrong: the open defect, and a
/// last-bit difference the tests leave out (see [`F16_DOUBLE`]).
const OPEN: &[&str] = &[FMA, F16_DOUBLE];

/// The probes without the inputs of open defects.
fn passing(probes: Vec<Probe>) -> Vec<Probe> {
    probes
        .into_iter()
        .filter_map(|mut p| {
            let defects = std::mem::take(&mut p.defects);
            p.inputs
                .retain(|&i| !defects.iter().any(|(d, on)| OPEN.contains(d) && on(i)));
            (!p.inputs.is_empty()).then_some(p)
        })
        .collect()
}

/// The probes with only the inputs that show `defect` (and no other open
/// one).
fn defective(probes: Vec<Probe>, defect: &str) -> Vec<Probe> {
    probes
        .into_iter()
        .filter_map(|mut p| {
            let defects = std::mem::take(&mut p.defects);
            let on = defects.iter().find(|(d, _)| *d == defect)?;
            p.inputs.retain(|&i| {
                (on.1)(i)
                    && !defects
                        .iter()
                        .any(|(d, o)| *d != defect && OPEN.contains(d) && o(i))
            });
            (!p.inputs.is_empty()).then_some(p)
        })
        .collect()
}

/// Runs probes on the hardware adapter and WARP; returns the failures
/// (each with its first differing inputs).
fn check(probes: &[Probe]) -> Vec<String> {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut hosts: Vec<(&str, bool, Host)> = [("hardware", false), ("warp", true)]
        .into_iter()
        .filter_map(|(label, warp)| Host::new(warp).ok().map(|h| (label, warp, h)))
        .collect();
    assert!(!hosts.is_empty(), "no SM 6.6 device");
    let report = std::env::var("VICTORY_SEMANTICS_REPORT").ok();
    let mut validator = validator::Validator::load(validator::Validator::default_path().as_deref());
    let mut failures = Vec::new();
    for p in probes {
        let what = disassembly(&p.body, p.scalar);
        let shader = match lower_program(&program(&p.body, p.scalar)) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{what}: not lowered: {e}");
                failures.push(format!("{what}: not lowered: {e}"));
                continue;
            }
        };
        if let Ok(val) = validator.as_mut()
            && let Err(e) = val.validate(&mut shader.dxil.clone())
        {
            eprintln!("{what}: invalid DXIL: {e}");
            failures.push(format!("{what}: invalid DXIL: {e}"));
            continue;
        }
        let mut results: Vec<(&str, Vec<[u32; 4]>)> = Vec::new();
        for (label, warp, h) in &mut hosts {
            match run(h, &shader, &p.inputs, p.scalar) {
                Ok(r) => results.push((label, r)),
                Err(e) => {
                    // A removed device takes the rest with it: make a new one.
                    eprintln!("{what}: {label}: {e}");
                    failures.push(format!("{what}: {label}: {e}"));
                    *h = Host::new(*warp).expect("the device again");
                }
            }
        }
        if results.len() < hosts.len() {
            continue;
        }
        let mut bad = Vec::new();
        let example = |label: &str, i: usize, want: &[u32], got: &[u32; 4]| {
            format!(
                "  {label}: in {:08x?} want {:08x?} got {:08x?}",
                p.inputs[i],
                want,
                &got[..want.len()]
            )
        };
        let (mut wrong, mut apart) = (vec![0usize; results.len()], 0usize);
        let mut all = Vec::new();
        for (i, &input) in p.inputs.iter().enumerate() {
            if let Some(model) = &p.model {
                let want = model(input);
                for (k, (label, r)) in results.iter().enumerate() {
                    if !want.iter().zip(&r[i]).all(|(&w, &g)| same(p.kind, w, g)) {
                        wrong[k] += 1;
                        if wrong[k] <= 6 {
                            bad.push(example(label, i, &want, &r[i]));
                        }
                        all.push(example(label, i, &want, &r[i]));
                    }
                }
            }
            let agree =
                |x: &[u32; 4], y: &[u32; 4]| x.iter().zip(y).all(|(&a, &b)| same(p.kind, a, b));
            if results.len() == 2 && !agree(&results[0].1[i], &results[1].1[i]) {
                apart += 1;
                if apart <= 4 {
                    bad.push(format!(
                        "  hardware and warp differ: in {:08x?}: {:08x?} vs {:08x?}",
                        p.inputs[i], results[0].1[i], results[1].1[i]
                    ));
                }
            }
        }
        let summary = format!(
            "{what}: {} inputs, model mismatches {:?}, hardware/warp differ on {apart}",
            p.inputs.len(),
            results
                .iter()
                .zip(&wrong)
                .map(|((l, _), n)| format!("{l} {n}"))
                .collect::<Vec<_>>()
        );
        eprintln!("{summary}");
        if let Some(path) = &report {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .expect("report");
            writeln!(f, "{summary}").expect("report");
            for b in all
                .iter()
                .chain(bad.iter().filter(|b| b.contains("differ")))
            {
                writeln!(f, "{b}").expect("report");
            }
        }
        if wrong.iter().any(|&n| n > 0) || apart > 0 {
            failures.push(format!("{summary}\n{}", bad.join("\n")));
        }
    }
    failures
}

fn assert_clean(failures: Vec<String>) {
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

// ---- inputs ----

const F32_EDGES: &[u32] = &[
    0x0000_0000,
    0x8000_0000, // ±0
    0x0000_0001,
    0x8000_0001, // ±smallest denormal
    0x007F_FFFF,
    0x807F_FFFF, // ±largest denormal
    0x0080_0000,
    0x8080_0000, // ±smallest normal
    0x3F80_0000,
    0xBF80_0000, // ±1
    0x3F00_0000,
    0xBF00_0000, // ±0.5
    0x3FC0_0000,
    0xC020_0000, // 1.5, -2.5
    0x3F7F_FFFF,
    0x3F80_0001, // 1 ∓ 1 ulp
    0x3EFF_FFFF, // 0.49999997
    0x4B00_0000,
    0x4B00_0001,
    0x4AFF_FFFF, // 2^23, 2^23 + 1, 2^23 - 0.5
    0x4F00_0000,
    0xCF00_0000,
    0x4EFF_FFFF,
    0xCF00_0001, // around ±2^31
    0x4F80_0000,
    0x4F7F_FFFF, // around 2^32
    0x7F7F_FFFF,
    0xFF7F_FFFF, // ±largest
    0x7F80_0000,
    0xFF80_0000, // ±infinity
    0x7FC0_0000,
    0xFFC0_0000,
    0x7F80_0001, // quiet, negative quiet, signalling NaN
    0x1E3C_E508, // 1e-20 (squares to a denormal)
    0x5F80_0000, // 2^64 (squares past the largest)
    0x3DCC_CCCD, // 0.1
    0x4049_0FDB, // pi
    0xC2C8_0000, // -100
];

/// Third operands for three-source float probes.
const F32_THIRD: &[u32] = &[
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x3F80_0000,
    0xBF80_0000,
    0x7F80_0000,
    0xFF80_0000,
    0x7FC0_0000,
    0xFF7F_FFFF,
    0x2E3C_E508,
];

const U32_EDGES: &[u32] = &[
    0,
    1,
    2,
    3,
    7,
    0x1F,
    0x20,
    0x21,
    0x3F,
    0x40,
    0x7F,
    0x80,
    0xFF,
    0x100,
    0x7FFF,
    0x8000,
    0xFFFF,
    0x1_0000,
    0x7F_FFFF,
    0x80_0000,
    0xFF_FFFF,
    0x100_0000,
    0x7FFF_FFFF,
    0x8000_0000,
    0x8000_0001,
    0xFFFF_FFFF,
    0xFFFF_FFFE,
    0xFFFF_FFE0,
    0x1234_5678,
    0xDEAD_BEEF,
    0x5555_5555,
    0xAAAA_AAAA,
    0xFF00_0001,
    0x00FF_FF00,
];

/// The old value of masked or untouched results: d.
const OLD: u32 = 0x0BAD_F00D;

fn unary(xs: &[u32]) -> Vec<[u32; 4]> {
    xs.iter().map(|&a| [a, 0, 0, OLD]).collect()
}

fn pairs(xs: &[u32], ys: &[u32]) -> Vec<[u32; 4]> {
    xs.iter()
        .flat_map(|&a| ys.iter().map(move |&b| [a, b, 0, OLD]))
        .collect()
}

fn triples(xs: &[u32], ys: &[u32], zs: &[u32]) -> Vec<[u32; 4]> {
    xs.iter()
        .flat_map(|&a| {
            ys.iter()
                .flat_map(move |&b| zs.iter().map(move |&c| [a, b, c, OLD]))
        })
        .collect()
}

// ---- the hardware model ----

fn fl(x: u32) -> f32 {
    f32::from_bits(x)
}

fn nan(x: u32) -> bool {
    x & 0x7FFF_FFFF > 0x7F80_0000
}

/// A 32-bit denormal as the console's mode reads and writes it: a zero of
/// its sign (RDNA2 ISA, MODE `FP_DENORM`; the mode the console compiles
/// for flushes both).
fn ftz(x: u32) -> u32 {
    if x & 0x7F80_0000 == 0 {
        x & 0x8000_0000
    } else {
        x
    }
}

fn out(r: f32) -> u32 {
    ftz(r.to_bits())
}

fn mods_in(x: u32, k: usize, m: M) -> u32 {
    let mut x = x;
    if m.abs >> k & 1 != 0 {
        x &= 0x7FFF_FFFF;
    }
    if m.neg >> k & 1 != 0 {
        x ^= 0x8000_0000;
    }
    x
}

/// Output modifiers of a float result: the multiplier, which turns -0
/// into +0, then the clamp to [0, 1], which gives +0 for NaN (DX10 clamp)
/// and for -0. Measured on RDNA2 by the Leonida project (gfx1031 and
/// gfx1036 for the multiplier with IEEE off and denormals flushed; the
/// clamp's -0 and NaN on gfx1036 for f16).
fn mods_out(r: u32, m: M) -> u32 {
    let mut r = r;
    if m.omod != 0 && !nan(r) {
        r = out(fl(r) * [1.0, 2.0, 4.0, 0.5][usize::from(m.omod)]);
        if r == 0x8000_0000 {
            r = 0;
        }
    }
    if m.clamp {
        r = if nan(r) || fl(r) <= 0.0 {
            0
        } else if fl(r) >= 1.0 {
            0x3F80_0000
        } else {
            r
        };
    }
    r
}

/// An exact (or once-rounded) result as the hardware writes it: below the
/// smallest normal a zero of its sign. Whether the hardware tests that
/// before or after rounding is not public; before is what both host
/// adapters do.
fn out64(r: f64) -> u32 {
    if r.abs() < f64::from(f32::MIN_POSITIVE) {
        (r.to_bits() >> 32) as u32 & 0x8000_0000
    } else {
        (r as f32).to_bits()
    }
}

fn d(x: u32) -> f64 {
    f64::from(fl(ftz(x)))
}

fn add(a: u32, b: u32) -> u32 {
    out64(d(a) + d(b))
}

fn mul(a: u32, b: u32) -> u32 {
    out64(d(a) * d(b))
}

/// DX9's multiply: a zero operand (a denormal read as one) gives +0 against
/// anything, infinity and NaN included (RDNA2 ISA `V_MUL_LEGACY_F32`; the
/// +0 and the denormal operands measured on RDNA2 by the Leonida project).
fn mul_legacy(a: u32, b: u32) -> u32 {
    let (a, b) = (ftz(a), ftz(b));
    if a & 0x7FFF_FFFF == 0 || b & 0x7FFF_FFFF == 0 {
        0
    } else {
        mul(a, b)
    }
}

fn fma(a: u32, b: u32, c: u32) -> u32 {
    out64(d(a).mul_add(d(b), d(c)))
}

/// `v_mad_f32`: the product rounded (and flushed) before the add.
fn mad(a: u32, b: u32, c: u32) -> u32 {
    add(mul(a, b), c)
}

fn mad_legacy(a: u32, b: u32, c: u32) -> u32 {
    add(mul_legacy(a, b), c)
}

/// Minimum or maximum: one NaN operand gives the other, two give the
/// first; -0 orders below +0 (measured on RDNA2, RX 6800, by the Leonida
/// project with the AnyPS5 hardware oracle).
fn min_max(max: bool, a: u32, b: u32) -> u32 {
    let (a, b) = (ftz(a), ftz(b));
    match (nan(a), nan(b)) {
        (true, true) | (false, true) => a,
        (true, false) => b,
        _ if fl(a) == fl(b) => {
            if max {
                a & b
            } else {
                a | b
            }
        }
        _ => {
            if (fl(a) > fl(b)) == max {
                a
            } else {
                b
            }
        }
    }
}

/// The median; with a NaN among the operands their minimum (same source).
fn med3(a: u32, b: u32, c: u32) -> u32 {
    if nan(a) || nan(b) || nan(c) {
        return min_max(false, min_max(false, a, b), c);
    }
    let lo = min_max(false, a, b);
    let hi = min_max(true, a, b);
    min_max(true, lo, min_max(false, hi, c))
}

/// Float to integer: toward zero, saturating, NaN 0 (RDNA2 ISA
/// `V_CVT_I32_F32`, `V_CVT_U32_F32`), denormals read as zero.
fn to_i32(x: u32) -> u32 {
    let x = ftz(x);
    if nan(x) {
        return 0;
    }
    let f = f64::from(fl(x)).trunc();
    f.clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32 as u32
}

fn to_u32(x: u32) -> u32 {
    let x = ftz(x);
    if nan(x) {
        return 0;
    }
    f64::from(fl(x)).trunc().clamp(0.0, f64::from(u32::MAX)) as u32
}

/// `v_cvt_flr_i32_f32` and `v_cvt_rpi_i32_f32`: floor of x (of x + 0.5
/// taken exactly), saturating; a NaN saturates by its sign; a denormal reads
/// as zero (measured on RDNA2, gfx1035 and gfx1036, by the Leonida project).
fn to_i32_floor(x: u32, half: bool) -> u32 {
    let x = ftz(x);
    if nan(x) {
        return if x >> 31 != 0 {
            0x8000_0000
        } else {
            0x7FFF_FFFF
        };
    }
    let f = (f64::from(fl(x)) + if half { 0.5 } else { 0.0 }).floor();
    f.clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32 as u32
}

/// A value to f16, to nearest even (or toward zero), denormals made.
fn to_f16(x: f64, toward_zero: bool) -> u32 {
    let b = x.to_bits();
    let sign = (b >> 48) as u32 & 0x8000;
    if x.is_nan() {
        return sign | 0x7E00;
    }
    let a = x.abs();
    if a == f64::INFINITY {
        return sign | 0x7C00;
    }
    // Units of the smallest f16 denormal (2^-24), rounded.
    let scaled = a * f64::from(1u32 << 24);
    let (e, m) = if a < 2f64.powi(-14) {
        (0, scaled)
    } else {
        let e = ((a.to_bits() >> 52) & 0x7FF) as i32 - 1023;
        (e + 15, a / 2f64.powi(e - 10))
    };
    let r = if toward_zero {
        m.trunc()
    } else {
        round_even(m)
    };
    let bits = if e == 0 {
        r as u32
    } else {
        ((e as u32) << 10) + (r as u32 - 0x400)
    };
    if bits >= 0x7C00 {
        return sign | if toward_zero { 0x7BFF } else { 0x7C00 };
    }
    sign | bits
}

fn round_even(m: f64) -> f64 {
    let r = m.round();
    if (r - m).abs() == 0.5 && r % 2.0 != 0.0 {
        r - m.signum()
    } else {
        r
    }
}

fn from_f16(h: u32) -> f32 {
    let (sign, e, m) = (h >> 15 & 1, h >> 10 & 0x1F, h & 0x3FF);
    let v = match e {
        0 => f64::from(m) * 2f64.powi(-24),
        0x1F if m == 0 => f64::INFINITY,
        0x1F => return f32::from_bits(sign << 31 | 0x7FC0_0000 | m << 13),
        _ => f64::from(m | 0x400) * 2f64.powi(e as i32 - 25),
    };
    (if sign != 0 { -v } else { v }) as f32
}

// ---- probes ----

/// Source modifiers worth trying on float probes: none, and neg, abs and
/// -|x| on each source.
fn source_mods() -> Vec<M> {
    let mut out = vec![NONE];
    for k in 0..3 {
        out.push(M {
            neg: 1 << k,
            ..NONE
        });
        out.push(M {
            abs: 1 << k,
            ..NONE
        });
        out.push(M {
            abs: 1 << k,
            neg: 1 << k,
            ..NONE
        });
    }
    out
}

/// Output modifiers: each multiplier, the clamp, and both.
fn output_mods() -> Vec<M> {
    let mut out = Vec::new();
    for omod in 0..4 {
        for clamp in [false, true] {
            if omod != 0 || clamp {
                out.push(M {
                    omod,
                    clamp,
                    ..NONE
                });
            }
        }
    }
    out
}

type F2 = fn(u32, u32) -> u32;
type F3 = fn(u32, u32, u32) -> u32;

/// Defects the probes show, by name (each has an ignored test below).
const FMA: &str = "multiply-adds are fused only where the host fuses";
const OMOD: &str = "output multipliers keep a -0 result";
const DENORMAL: &str = "denormal operands of min, max and float selects are the host's";
const F16_ROUND: &str = "f16 results round toward zero";
const F16_TWICE: &str = "f16 arithmetic rounds in f32 first, as the host chooses";
const SCC_MIN_MAX: &str = "s_min and s_max set SCC from the result";
const FLOOR_CVT: &str = "v_cvt_flr and v_cvt_rpi: NaN and the half";
const MED3_ZERO: &str = "v_med3_f32 orders -0 and +0 as the host does";
const CONSTANT_CMP: &str = "a constant compare removes the WARP device";
/// A fused f16 result (`v_fma_f16`, `v_pk_fma_f16`, `v_fma_mix{lo,hi}_f16`)
/// whose single rounding the lowering's f32 intermediate changes: the f32
/// value lands on an f16 tie the exact one is not on (1 f16 ULP; the
/// exact result needs more than f32 between the multiply-add and the
/// narrowing).
const F16_DOUBLE: &str = "a fused f16 result rounded in f32 first";

/// Whether rounding `x` to f32 first changes its rounding to f16.
fn twice(x: f64) -> bool {
    to_f16(f64::from(x as f32), false) != to_f16(x, false)
}

/// Output modifiers as the lowering applies them: the multiplier keeps a
/// -0, the clamp is DXIL's saturate.
fn mods_out_kept(r: u32, m: M) -> u32 {
    let mut r = r;
    if m.omod != 0 && !nan(r) {
        r = out(fl(r) * [1.0, 2.0, 4.0, 0.5][usize::from(m.omod)]);
    }
    if m.clamp {
        r = if nan(r) || fl(r) < 0.0 {
            0
        } else if fl(r) >= 1.0 {
            0x3F80_0000
        } else {
            r
        };
    }
    r
}

fn denormal(x: u32) -> bool {
    x & 0x7F80_0000 == 0 && x & 0x7FFF_FFFF != 0
}

/// A two-source VOP3 float probe with modifiers, the output multiplier's
/// -0 marked.
fn vop3_f2(op: u16, m: M, f: F2, inputs: Vec<[u32; 4]>, kind: Kind) -> Probe {
    let x = move |i: [u32; 4]| f(mods_in(i[0], 0, m), mods_in(i[1], 1, m));
    probe(vop3(op, 5, [v(1), v(2), 0], m), inputs, kind, move |i| {
        vec![mods_out(x(i), m)]
    })
    .shows_where(
        OMOD,
        move |i| vec![mods_out(x(i), m)],
        move |i| vec![mods_out_kept(x(i), m)],
    )
}

/// A three-source VOP3 float probe with modifiers; `ours` is what the
/// lowering computes in place of `f` (the multiply-adds: unfused).
fn vop3_f3(op: u16, m: M, f: F3, ours: F3, inputs: Vec<[u32; 4]>, kind: Kind) -> Probe {
    let x = move |i: [u32; 4], f: F3| {
        let s = |k| mods_in(i[k], k, m);
        f(s(0), s(1), s(2))
    };
    probe(vop3(op, 5, [v(1), v(2), v(3)], m), inputs, kind, move |i| {
        vec![mods_out(x(i, f), m)]
    })
    .shows_where(
        FMA,
        move |i| vec![mods_out(x(i, f), m)],
        move |i| vec![mods_out(x(i, ours), m)],
    )
    .shows_where(
        OMOD,
        move |i| vec![mods_out(x(i, f), m)],
        move |i| vec![mods_out_kept(x(i, f), m)],
    )
}

/// Results within `ulps`; under a clamp a -0 may come back as either zero
/// (the f32 clamp of -0 is measured nowhere public).
fn tolerant(ulps: u32, m: M) -> Kind {
    Kind::F32 {
        ulps,
        any_zero: m.clamp,
    }
}

/// The legacy multiply (the title's most used instruction), the plain one,
/// the adds and the multiply-adds, with zeros, infinities, NaNs and
/// denormals among the operands, with the title's modifiers. Fused results
/// within 1 ULP: whether the host fuses is a last-bit question on ordinary
/// values; differences in kind are not (they are the `FMA` defect).
fn arithmetic_probes() -> Vec<Probe> {
    let ab = pairs(F32_EDGES, F32_EDGES);
    let abc = triples(F32_EDGES, F32_EDGES, F32_THIRD);
    let fused = tolerant(1, NONE);
    let accumulate = |op: u16, f: F3, ours: F3| {
        probe(
            [vop1(0x01, 5, v(3)), vop2(op, 5, v(1), 2)],
            abc.clone(),
            fused,
            move |i| vec![f(i[0], i[1], i[2])],
        )
        .shows_where(
            FMA,
            move |i| vec![f(i[0], i[1], i[2])],
            move |i| vec![ours(i[0], i[1], i[2])],
        )
    };
    let mut p = vec![
        probe([vop2(0x07, 5, v(1), 2)], ab.clone(), F32, |i| {
            vec![mul_legacy(i[0], i[1])]
        }),
        probe([vop2(0x08, 5, v(1), 2)], ab.clone(), F32, |i| {
            vec![mul(i[0], i[1])]
        }),
        probe([vop2(0x03, 5, v(1), 2)], ab.clone(), F32, |i| {
            vec![add(i[0], i[1])]
        }),
        probe([vop2(0x04, 5, v(1), 2)], ab.clone(), F32, |i| {
            vec![add(i[0], i[1] ^ 0x8000_0000)]
        }),
        probe([vop2(0x05, 5, v(1), 2)], ab, F32, |i| {
            vec![add(i[1], i[0] ^ 0x8000_0000)]
        }),
        // v_fmac_f32, v_mac_f32 and v_mac_legacy_f32 accumulate into v5 = c.
        accumulate(0x2B, fma, mad),
        accumulate(0x1F, mad, mad),
        accumulate(0x06, mad_legacy, mad_legacy),
        vop3_f3(0x14B, NONE, fma, mad, abc.clone(), fused),
        vop3_f3(0x141, NONE, mad, mad, abc.clone(), fused),
        vop3_f3(0x140, NONE, mad_legacy, mad_legacy, abc, fused),
    ];
    let small = pairs(F32_EDGES, &F32_EDGES[..16]);
    let abc_small = triples(&F32_EDGES[..16], &F32_EDGES[..16], F32_THIRD);
    for m in source_mods().into_iter().chain(output_mods()) {
        if m.abs >> 2 != 0 || m.neg >> 2 != 0 {
            continue;
        }
        p.push(vop3_f2(0x107, m, mul_legacy, small.clone(), tolerant(0, m)));
        p.push(vop3_f2(0x103, m, add, small.clone(), tolerant(0, m)));
        let subrev: F2 = |a, b| add(b, a ^ 0x8000_0000);
        p.push(vop3_f2(0x105, m, subrev, small.clone(), tolerant(0, m)));
        if m.abs | m.neg != 0 {
            let sub: F2 = |a, b| add(a, b ^ 0x8000_0000);
            p.push(vop3_f2(0x104, m, sub, small.clone(), tolerant(0, m)));
        }
        // v_fmac_f32 in VOP3: the accumulator v5 = c.
        let x = move |i: [u32; 4], f: F3| f(mods_in(i[0], 0, m), mods_in(i[1], 1, m), i[2]);
        let mut body = vec![vop1(0x01, 5, v(3))];
        body.extend(vop3(0x12B, 5, [v(1), v(2), 0], m));
        p.push(
            probe(body, abc_small.clone(), tolerant(1, m), move |i| {
                vec![mods_out(x(i, fma), m)]
            })
            .shows_where(
                FMA,
                move |i| vec![mods_out(x(i, fma), m)],
                move |i| vec![mods_out(x(i, mad), m)],
            )
            .shows_where(
                OMOD,
                move |i| vec![mods_out(x(i, fma), m)],
                move |i| vec![mods_out_kept(x(i, fma), m)],
            ),
        );
    }
    for m in source_mods().into_iter().chain(output_mods()) {
        p.push(vop3_f3(
            0x14B,
            m,
            fma,
            mad,
            abc_small.clone(),
            tolerant(1, m),
        ));
    }
    for m in output_mods() {
        p.push(vop3_f3(
            0x141,
            m,
            mad,
            mad,
            abc_small.clone(),
            tolerant(1, m),
        ));
        p.push(vop3_f3(
            0x140,
            m,
            mad_legacy,
            mad_legacy,
            abc_small.clone(),
            tolerant(1, m),
        ));
        p.push(vop3_f2(0x108, m, mul, small.clone(), tolerant(0, m)));
        p.push(vop3_f2(
            0x104,
            m,
            |a, b| add(a, b ^ 0x8000_0000),
            small.clone(),
            tolerant(0, m),
        ));
    }
    p
}

#[test]
fn arithmetic_matches_the_hardware() {
    assert_clean(check(&passing(arithmetic_probes())));
}

/// Minimum, maximum, median, and `v_max_f32 x, x clamp` (the title's
/// saturate) with NaN, -0 and denormals.
fn min_max_probes() -> Vec<Probe> {
    let ab = pairs(F32_EDGES, F32_EDGES);
    let abc = triples(F32_EDGES, F32_EDGES, F32_THIRD);
    let max3: F3 = |a, b, c| min_max(true, min_max(true, a, b), c);
    let min3: F3 = |a, b, c| min_max(false, min_max(false, a, b), c);
    let any_denormal = |i: [u32; 4]| i[..3].iter().any(|&x| denormal(x));
    let mut p = vec![
        probe([vop2(0x10, 5, v(1), 2)], ab.clone(), F32, |i| {
            vec![min_max(true, i[0], i[1])]
        }),
        probe([vop2(0x0F, 5, v(1), 2)], ab, F32, |i| {
            vec![min_max(false, i[0], i[1])]
        }),
        vop3_f3(0x157, NONE, med3, med3, abc.clone(), F32).shows(MED3_ZERO, |i| {
            i[..3].contains(&0) && i[..3].contains(&0x8000_0000)
        }),
        vop3_f3(0x154, NONE, max3, max3, abc.clone(), F32),
        vop3_f3(0x151, NONE, min3, min3, abc, F32),
    ];
    let clamp = M {
        clamp: true,
        ..NONE
    };
    p.push(probe(
        vop3(0x110, 5, [v(1), v(1), 0], clamp),
        unary(F32_EDGES),
        tolerant(0, clamp),
        move |i| vec![mods_out(min_max(true, i[0], i[0]), clamp)],
    ));
    let max: F2 = |a, b| min_max(true, a, b);
    for m in output_mods() {
        p.push(vop3_f2(
            0x110,
            m,
            max,
            pairs(F32_EDGES, &F32_EDGES[..16]),
            tolerant(0, m),
        ));
    }
    p.into_iter()
        .map(|p| p.shows(DENORMAL, any_denormal))
        .collect()
}

#[test]
fn min_max_and_clamp_match_the_hardware() {
    assert_clean(check(&passing(min_max_probes())));
}

/// The 16 float predicates of `v_cmp_*_f32` (RDNA2 ISA: `o` ordered, `u`
/// unordered, `n*` true when unordered).
fn compare(op: u16, a: u32, b: u32) -> bool {
    use std::cmp::Ordering::{Equal, Greater, Less};
    // The ordered relation (None: unordered); `n*` predicates negate theirs.
    let o = fl(a).partial_cmp(&fl(b));
    let ordered = |want: &[std::cmp::Ordering]| o.is_some_and(|o| want.contains(&o));
    match op & 0xF {
        0x0 => false,
        0x1 => ordered(&[Less]),
        0x2 => ordered(&[Equal]),
        0x3 => ordered(&[Less, Equal]),
        0x4 => ordered(&[Greater]),
        0x5 => ordered(&[Less, Greater]),
        0x6 => ordered(&[Greater, Equal]),
        0x7 => o.is_some(),
        0x8 => o.is_none(),
        0x9 => !ordered(&[Greater, Equal]),
        0xA => !ordered(&[Less, Greater]),
        0xB => !ordered(&[Greater]),
        0xC => !ordered(&[Less, Equal]),
        0xD => !ordered(&[Equal]),
        0xE => !ordered(&[Less]),
        _ => true,
    }
}

/// Compares as VCC read by `v_cndmask_b32`, the e64 forms with source
/// modifiers into an SGPR mask, and `v_cmpx` narrowing the lanes that
/// write (the others keep their register).
fn compare_probes() -> Vec<Probe> {
    let normal: Vec<u32> = F32_EDGES.iter().copied().filter(|&x| ftz(x) == x).collect();
    let ab = pairs(&normal, &normal);
    let mut p = Vec::new();
    for op in 1..15u16 {
        p.push(probe(
            with_vcc(vopc(op, v(1), 2)),
            ab.clone(),
            INT,
            move |i| vec![u32::from(compare(op, i[0], i[1]))],
        ));
    }
    for m in [
        M { neg: 1, ..NONE },
        M { abs: 2, ..NONE },
        M {
            abs: 3,
            neg: 3,
            ..NONE
        },
    ] {
        for op in [0x1u16, 0x4, 0x9, 0xD] {
            let mut body = vop3(op, 20, [v(1), v(2), 0], m).to_vec();
            body.extend(vop3(0x101, 5, [int(0), int(1), s(20)], NONE));
            p.push(probe(body, ab.clone(), INT, move |i| {
                vec![u32::from(compare(
                    op,
                    mods_in(i[0], 0, m),
                    mods_in(i[1], 1, m),
                ))]
            }));
        }
    }
    for op in [0x11u16, 0x14, 0x1B, 0x1D] {
        let body = [
            sop1(0x03, 20, EXEC_LO),
            vopc(op, v(1), 2),
            vop1(0x01, 5, int(1)),
            sop1(0x03, 126, s(20)),
        ];
        p.push(probe(body, ab.clone(), INT, move |i| {
            vec![if compare(op, i[0], i[1]) { 1 } else { OLD }]
        }));
    }
    p
}

#[test]
fn compares_match_the_hardware() {
    assert_clean(check(&compare_probes()));
}

/// Compares of denormal operands: whether the hardware flushes them first
/// is fixed by no public source; the two adapters must at least agree.
#[test]
fn compares_of_denormals_agree_across_adapters() {
    let denormals = [0x0000_0001, 0x8000_0001, 0x007F_FFFF, 0x807F_FFFF];
    let others = [
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x0080_0000,
        0x3F80_0000,
    ];
    let mut inputs = pairs(&denormals, &others);
    inputs.extend(pairs(&others, &denormals));
    let p: Vec<Probe> = [0x1u16, 0x2, 0x3, 0xD]
        .into_iter()
        .map(|op| Probe {
            body: with_vcc(vopc(op, v(1), 2)),
            scalar: false,
            inputs: inputs.clone(),
            model: None,
            kind: INT,
            defects: Vec::new(),
        })
        .collect();
    assert_clean(check(&p));
}

/// `v_cmp_class_f32`: the class of the raw operand, denormals never flushed
/// (RDNA2 ISA; measured on RDNA2 in both denormal modes by the Leonida
/// project).
fn class(x: u32) -> u32 {
    let (neg, mag) = (x >> 31 != 0, x & 0x7FFF_FFFF);
    let bit = match mag {
        0x7F80_0001..=0x7FBF_FFFF => 0,
        0x7FC0_0000.. => 1,
        0x7F80_0000 => 9,
        0x0080_0000.. => 8,
        1.. => 7,
        0 => 6,
    };
    match (neg, bit) {
        (_, 0 | 1) => bit,
        (true, b) => 11 - b,
        (false, b) => b,
    }
}

#[test]
fn class_compares_match_the_hardware() {
    let masks: Vec<u32> = (0..10).map(|k| 1 << k).chain([0x3FF, 0x3, 0x204]).collect();
    let inputs = pairs(F32_EDGES, &masks);
    assert_clean(check(&[probe(
        with_vcc(vopc(0x88, v(1), 2)),
        inputs,
        INT,
        |i| vec![i[1] >> class(i[0]) & 1],
    )]));
}

/// Conversions, whose results become indices, addresses and counts.
fn conversion_probes() -> Vec<Probe> {
    let xs = unary(F32_EDGES);
    let ints = unary(U32_EDGES);
    let mut p = vec![
        probe([vop1(0x08, 5, v(1))], xs.clone(), INT, |i| {
            vec![to_i32(i[0])]
        }),
        probe([vop1(0x07, 5, v(1))], xs.clone(), INT, |i| {
            vec![to_u32(i[0])]
        }),
        // The lowering: NaN 0 (D3D's conversion), and x + 0.5 rounded first.
        probe([vop1(0x0D, 5, v(1))], xs.clone(), INT, |i| {
            vec![to_i32_floor(i[0], false)]
        })
        .shows(FLOOR_CVT, |i| nan(i[0])),
        probe([vop1(0x0C, 5, v(1))], xs.clone(), INT, |i| {
            vec![to_i32_floor(i[0], true)]
        })
        .shows_where(
            FLOOR_CVT,
            |i| vec![to_i32_floor(i[0], true)],
            |i| {
                vec![if nan(i[0]) {
                    0
                } else {
                    to_i32_floor(out(fl(ftz(i[0])) + 0.5), false)
                }]
            },
        ),
        probe([vop1(0x05, 5, v(1))], ints.clone(), F32, |i| {
            vec![(i[0] as i32 as f32).to_bits()]
        }),
        probe([vop1(0x06, 5, v(1))], ints.clone(), F32, |i| {
            vec![(i[0] as f32).to_bits()]
        }),
        // An f16 result keeps the destination's high half (v5 = d).
        probe([vop1(0x0A, 5, v(1))], xs.clone(), F16, |i| {
            vec![i[3] & 0xFFFF_0000 | to_f16(f64::from(fl(ftz(i[0]))), false)]
        })
        .shows_where(
            F16_ROUND,
            |i| vec![to_f16(d(i[0]), false)],
            |i| vec![to_f16(d(i[0]), true)],
        ),
        probe(
            [vop2(0x2F, 5, v(1), 2)],
            pairs(F32_EDGES, &F32_EDGES[..12]),
            Kind::Halves { ulps: 0 },
            |i| {
                vec![
                    to_f16(f64::from(fl(ftz(i[0]))), true)
                        | to_f16(f64::from(fl(ftz(i[1]))), true) << 16,
                ]
            },
        ),
    ];
    for n in 0..4u16 {
        p.push(probe(
            [vop1(0x11 + n, 5, v(1))],
            ints.clone(),
            F32,
            move |i| vec![((i[0] >> (8 * n) & 0xFF) as f32).to_bits()],
        ));
    }
    for m in output_mods() {
        p.push(probe(
            vop3(0x191, 5, [v(1), 0, 0], m),
            ints.clone(),
            tolerant(0, m),
            move |i| vec![mods_out(((i[0] & 0xFF) as f32).to_bits(), m)],
        ));
    }
    let halves: Vec<u32> = [
        0x0000u32, 0x8000, 0x0001, 0x83FF, 0x0400, 0x3C00, 0xBC00, 0x3BFF, 0x7BFF, 0xFBFF, 0x7C00,
        0xFC00, 0x7E00, 0xFE00, 0x7C01, 0x5640,
    ]
    .iter()
    .map(|&h| 0xABCD_0000 | h)
    .collect();
    p.push(probe([vop1(0x0B, 5, v(1))], unary(&halves), F32, |i| {
        vec![from_f16(i[0] & 0xFFFF).to_bits()]
    }));
    p
}

#[test]
fn conversions_match_the_hardware() {
    assert_clean(check(&passing(conversion_probes())));
}

/// `v_fract_f32`: never 1.0, NaN for infinities, a NaN passes, a denormal
/// reads as zero (measured on RDNA2 by the Leonida project).
fn fract(x: u32) -> u32 {
    let x = ftz(x);
    if nan(x) {
        return x;
    }
    if x & 0x7FFF_FFFF == 0x7F80_0000 {
        return 0xFFC0_0000;
    }
    let f = fl(x);
    let r = out(f - f.floor());
    if fl(r) >= 1.0 {
        0x3F7F_FFFF
    } else {
        r & 0x7FFF_FFFF
    }
}

/// Sine or cosine of `x` turns, reduced first (`x - round_even(x)`, exact):
/// infinities and NaN give NaN, half turns of sine and quarter turns of
/// cosine exact zeros, a zero passes (measured on RDNA2 by the Leonida
/// project).
fn turns(x: u32, cosine: bool) -> u32 {
    let x = ftz(x);
    if nan(x) || x & 0x7FFF_FFFF == 0x7F80_0000 {
        return 0xFFC0_0000;
    }
    if x & 0x7FFF_FFFF == 0 {
        return if cosine { 0x3F80_0000 } else { x };
    }
    let f = f64::from(fl(x));
    let part = f - f.round_ties_even();
    if part.abs() == if cosine { 0.25 } else { 0.5 } || !cosine && part == 0.0 {
        return 0;
    }
    let a = part * std::f64::consts::TAU;
    out((if cosine { a.cos() } else { a.sin() }) as f32)
}

/// Rounding to integers and the fraction at their boundaries.
fn rounding_probes() -> Vec<Probe> {
    let xs = unary(F32_EDGES);
    fn round(f: fn(f32) -> f32) -> impl Fn([u32; 4]) -> Vec<u32> {
        move |i| vec![out(f(fl(ftz(i[0]))))]
    }
    // The host's sine is only good to 2^-11 absolutely (as D3D allows).
    let trig = Kind::Abs { abs: 1.0 / 2048.0 };
    vec![
        probe([vop1(0x20, 5, v(1))], xs.clone(), F32, |i| {
            vec![fract(i[0])]
        }),
        probe([vop1(0x24, 5, v(1))], xs.clone(), F32, round(f32::floor)),
        probe([vop1(0x22, 5, v(1))], xs.clone(), F32, round(f32::ceil)),
        probe([vop1(0x21, 5, v(1))], xs.clone(), F32, round(f32::trunc)),
        probe(
            [vop1(0x23, 5, v(1))],
            xs.clone(),
            F32,
            round(f32::round_ties_even),
        ),
        probe([vop1(0x35, 5, v(1))], xs.clone(), trig, |i| {
            vec![turns(i[0], false)]
        }),
        probe([vop1(0x36, 5, v(1))], xs, trig, |i| vec![turns(i[0], true)]),
    ]
}

#[test]
fn rounding_matches_the_hardware() {
    assert_clean(check(&rounding_probes()));
}

/// The transcendental instructions' results: denormals in and out are
/// zeros of their sign whatever the mode, rcp(±0) = ±inf, rsq(-0) = -inf,
/// sqrt(-0) = -0, log(±0) = -inf, invalid operations NaN (measured on
/// RDNA2, RX 6800, by the Leonida project). Ordinary results within the
/// hardware's 1 ULP plus the host's own error.
fn transcendental(op: u16, x: u32) -> u32 {
    let x = ftz(x);
    let f = f64::from(fl(x));
    if nan(x) {
        return x | 0x0040_0000;
    }
    let r = match op {
        0x2A | 0x2B => 1.0 / f,
        0x2E if x == 0x8000_0000 => f64::NEG_INFINITY,
        0x2E => 1.0 / f.sqrt(),
        0x33 => f.sqrt(),
        0x27 if f == 0.0 => f64::NEG_INFINITY,
        0x27 => f.log2(),
        _ => f.exp2(),
    };
    out(r as f32)
}

fn transcendental_probes() -> Vec<Probe> {
    let xs = unary(F32_EDGES);
    let near = Kind::F32 {
        ulps: 4,
        any_zero: false,
    };
    let log_kind = Kind::Abs {
        abs: 1.0 / 2_097_152.0,
    };
    let mut p: Vec<Probe> = [0x2Au16, 0x2B, 0x2E, 0x33, 0x27, 0x25]
        .into_iter()
        .map(|op| {
            // log2 near 1: an absolute bound (D3D's 2^-21).
            let kind = if op == 0x27 { log_kind } else { near };
            probe([vop1(op, 5, v(1))], xs.clone(), kind, move |i| {
                vec![transcendental(op, i[0])]
            })
        })
        .collect();
    // The title's output modifiers on them: v_rcp, v_log with mul/div,
    // v_exp and v_sqrt with clamp.
    for (op, m) in [
        (0x2Au16, M { omod: 1, ..NONE }),
        (0x2A, M { omod: 3, ..NONE }),
        (0x27, M { omod: 3, ..NONE }),
        (0x27, M { omod: 1, ..NONE }),
        (
            0x25,
            M {
                clamp: true,
                ..NONE
            },
        ),
        (
            0x33,
            M {
                clamp: true,
                ..NONE
            },
        ),
        (0x33, M { abs: 1, ..NONE }),
    ] {
        let kind = if op == 0x27 { log_kind } else { tolerant(4, m) };
        let x = move |i: [u32; 4]| transcendental(op, mods_in(i[0], 0, m));
        p.push(
            probe(
                vop3(0x180 + op, 5, [v(1), 0, 0], m),
                xs.clone(),
                kind,
                move |i| vec![mods_out(x(i), m)],
            )
            .shows_where(
                OMOD,
                move |i| vec![mods_out(x(i), m)],
                move |i| vec![mods_out_kept(x(i), m)],
            ),
        );
    }
    p
}

#[test]
fn transcendental_specials_match_the_hardware() {
    assert_clean(check(&passing(transcendental_probes())));
}

// ---- integers ----

fn sext24(x: u32) -> i64 {
    i64::from((x << 8) as i32 >> 8)
}

/// A field of `a`: `off` and `width` read from their low five bits, the
/// shift arithmetic for a signed field (so a field past bit 31 extends from
/// bit 31; RDNA2 ISA `V_BFE_*`, measured on gfx1036 for issue #2).
fn bfe(a: u32, off: u32, width: u32, signed: bool) -> u32 {
    let (off, w) = (off & 31, width & 31);
    if w == 0 {
        return 0;
    }
    let shifted = if signed {
        ((a as i32) >> off) as u32
    } else {
        a >> off
    };
    let field = shifted & ((1u32 << w) - 1);
    if signed && field >> (w - 1) & 1 != 0 {
        field | !((1u32 << w) - 1)
    } else {
        field
    }
}

/// First set bit from the top (`v_ffbh_u32`), -1 for none.
fn ffbh(x: u32) -> u32 {
    if x == 0 { u32::MAX } else { x.leading_zeros() }
}

fn integer_probes() -> Vec<Probe> {
    let ab = pairs(U32_EDGES, U32_EDGES);
    let thirds = [0u32, 1, 0xFFFF_FFFF, 0x8000_0000, 0x1234_5678];
    let abc = triples(U32_EDGES, U32_EDGES, &thirds);
    let shifts: Vec<u32> = [
        0u32,
        1,
        7,
        15,
        16,
        31,
        32,
        33,
        63,
        64,
        0xFF,
        0xFFFF_FFE1,
        0x8000_0010,
    ]
    .to_vec();
    let fields: Vec<[u32; 4]> = U32_EDGES
        .iter()
        .flat_map(|&a| {
            let shifts = shifts.clone();
            shifts
                .clone()
                .into_iter()
                .flat_map(move |o| shifts.clone().into_iter().map(move |w| [a, o, w, OLD]))
        })
        .collect();
    let vop2_int = |op: u16, f: F2| {
        probe([vop2(op, 5, v(1), 2)], ab.clone(), INT, move |i| {
            vec![f(i[0], i[1])]
        })
    };
    let vop3_int = |op: u16, f: F3, inputs: &Vec<[u32; 4]>| {
        probe(
            vop3(op, 5, [v(1), v(2), v(3)], NONE),
            inputs.clone(),
            INT,
            move |i| vec![f(i[0], i[1], i[2])],
        )
    };
    let shift_inputs = pairs(U32_EDGES, &shifts);
    let shift = |op: u16, f: F2| {
        // `op v5, v2, v1`: v1 (a) shifted by v2 (b).
        probe(
            [vop2(op, 5, v(2), 1)],
            shift_inputs.clone(),
            INT,
            move |i| vec![f(i[1], i[0])],
        )
    };
    vec![
        vop3_int(
            0x143,
            |a, b, c| (a & 0xFF_FFFF).wrapping_mul(b & 0xFF_FFFF).wrapping_add(c),
            &abc,
        ),
        vop3_int(
            0x142,
            |a, b, c| ((sext24(a) * sext24(b)) as u32).wrapping_add(c),
            &abc,
        ),
        vop2_int(0x0B, |a, b| (a & 0xFF_FFFF).wrapping_mul(b & 0xFF_FFFF)),
        vop2_int(0x09, |a, b| (sext24(a) * sext24(b)) as u32),
        vop3_int(0x169, |a, b, _| a.wrapping_mul(b), &ab),
        vop3_int(
            0x16A,
            |a, b, _| ((u64::from(a) * u64::from(b)) >> 32) as u32,
            &ab,
        ),
        vop3_int(
            0x16C,
            |a, b, _| ((i64::from(a as i32) * i64::from(b as i32)) >> 32) as u32,
            &ab,
        ),
        // Shifts read the count's low five bits.
        shift(0x1A, |n, x| x << (n & 31)),
        shift(0x16, |n, x| x >> (n & 31)),
        shift(0x18, |n, x| ((x as i32) >> (n & 31)) as u32),
        vop3_int(
            0x346,
            |a, b, c| (a << (b & 31)).wrapping_add(c),
            &triples(U32_EDGES, &shifts, &thirds),
        ),
        vop3_int(
            0x347,
            |a, b, c| a.wrapping_add(b) << (c & 31),
            &triples(U32_EDGES, &thirds, &shifts),
        ),
        vop3_int(
            0x36F,
            |a, b, c| (a << (b & 31)) | c,
            &triples(U32_EDGES, &shifts, &thirds),
        ),
        vop3_int(
            0x14E,
            |a, b, c| ((u64::from(a) << 32 | u64::from(b)) >> (c & 31)) as u32,
            &triples(U32_EDGES, &thirds, &shifts),
        ),
        vop3_int(
            0x14F,
            |a, b, c| ((u64::from(a) << 32 | u64::from(b)) >> (8 * (c & 3))) as u32,
            &triples(U32_EDGES, &thirds, &shifts),
        ),
        vop3_int(0x148, |a, o, w| bfe(a, o, w, false), &fields),
        vop3_int(0x149, |a, o, w| bfe(a, o, w, true), &fields),
        vop3_int(0x14A, |a, b, c| (a & b) | (!a & c), &abc),
        vop3_int(
            0x363,
            |w, o, _| ((1u64 << (w & 31)) as u32).wrapping_sub(1) << (o & 31),
            &pairs(&shifts, &shifts),
        ),
        vop3_int(0x364, |a, b, _| a.count_ones().wrapping_add(b), &ab),
        probe([vop1(0x39, 5, v(1))], unary(U32_EDGES), INT, |i| {
            vec![ffbh(i[0])]
        }),
        probe([vop1(0x3A, 5, v(1))], unary(U32_EDGES), INT, |i| {
            vec![if i[0] == 0 {
                u32::MAX
            } else {
                i[0].trailing_zeros()
            }]
        }),
        // First bit that differs from the sign, from the top; -1 for 0 and -1.
        probe([vop1(0x3B, 5, v(1))], unary(U32_EDGES), INT, |i| {
            let x = if i[0] >> 31 != 0 { !i[0] } else { i[0] };
            vec![ffbh(x)]
        }),
        probe([vop1(0x38, 5, v(1))], unary(U32_EDGES), INT, |i| {
            vec![i[0].reverse_bits()]
        }),
        vop2_int(0x13, |a, b| a.min(b)),
        vop2_int(0x14, |a, b| a.max(b)),
        vop2_int(0x11, |a, b| (a as i32).min(b as i32) as u32),
        vop2_int(0x12, |a, b| (a as i32).max(b as i32) as u32),
        vop3_int(
            0x158,
            |a, b, c| {
                let (a, b, c) = (a as i32, b as i32, c as i32);
                a.min(b).max(a.max(b).min(c)) as u32
            },
            &abc,
        ),
        vop3_int(0x159, |a, b, c| a.min(b).max(a.max(b).min(c)), &abc),
        vop3_int(0x36D, |a, b, c| a.wrapping_add(b).wrapping_add(c), &abc),
        vop3_int(0x345, |a, b, c| (a ^ b).wrapping_add(c), &abc),
        vop3_int(0x371, |a, b, c| (a & b) | c, &abc),
        vop3_int(0x372, |a, b, c| a | b | c, &abc),
        vop3_int(0x178, |a, b, c| a ^ b ^ c, &abc),
        vop2_int(0x25, |a, b| a.wrapping_add(b)),
        vop2_int(0x26, |a, b| a.wrapping_sub(b)),
        vop2_int(0x27, |a, b| b.wrapping_sub(a)),
    ]
}

#[test]
fn integers_match_the_hardware() {
    assert_clean(check(&integer_probes()));
}

/// Carries as lane masks: `v_add_co_u32` into VCC, `v_add_co_ci_u32`
/// taking and giving VCC, the last carry read back by `v_cndmask_b32`.
#[test]
fn carries_match_the_hardware() {
    let inputs = triples(U32_EDGES, U32_EDGES, &[0, 1, 0xFFFF_FFFF]);
    let mut add = vop3b(0x30F, 5, 106, [v(1), v(2), 0], false).to_vec();
    add.push(vop2(0x28, 6, v(3), 4));
    add.extend(vop3(0x101, 7, [int(0), int(1), VCC], NONE));
    let mut sub = vop3b(0x310, 5, 106, [v(1), v(2), 0], false).to_vec();
    sub.push(vop2(0x29, 6, v(3), 4));
    sub.extend(vop3(0x101, 7, [int(0), int(1), VCC], NONE));
    assert_clean(check(&[
        probe(add, inputs.clone(), INT, |i| {
            let (s, c1) = i[0].overflowing_add(i[1]);
            let t = u64::from(i[2]) + u64::from(i[3]) + u64::from(c1);
            vec![s, t as u32, (t >> 32) as u32]
        }),
        probe(sub, inputs, INT, |i| {
            let (s, b1) = i[0].overflowing_sub(i[1]);
            let t = i64::from(i[2]) - i64::from(i[3]) - i64::from(b1);
            vec![s, t as u32, u32::from(t < 0)]
        }),
    ]));
}

/// `v_cndmask_b32` on a compare, plain and with float source modifiers (a
/// sign-bit change of the chosen source; the title selects negated floats
/// this way).
#[test]
fn selects_match_the_hardware() {
    assert_clean(check(&passing(select_probes())));
}

fn select_probes() -> Vec<Probe> {
    let inputs: Vec<[u32; 4]> = pairs(U32_EDGES, U32_EDGES)
        .into_iter()
        .zip(
            F32_EDGES
                .iter()
                .cycle()
                .zip(F32_EDGES.iter().cycle().skip(7)),
        )
        .map(|(i, (&c, &d))| [i[0], i[1], c, d])
        .collect();
    let select = |m: M| {
        let mut w = vec![vopc(0xC4, v(1), 2)];
        w.extend(vop3(0x101, 5, [v(3), v(4), VCC], m));
        probe(w, inputs.clone(), F32, move |i| {
            let pick = if i[0] > i[1] { 1 } else { 0 };
            vec![mods_in(i[2 + pick], pick, m)]
        })
        .shows(DENORMAL, move |i| {
            (m.abs | m.neg) != 0 && denormal(i[2 + usize::from(i[0] > i[1])])
        })
    };
    vec![
        select(NONE),
        select(M { neg: 1, ..NONE }),
        select(M { neg: 2, ..NONE }),
        select(M { abs: 3, ..NONE }),
        select(M {
            abs: 1,
            neg: 1,
            ..NONE
        }),
    ]
}

// ---- 16 bits ----

const F16_EDGES: &[u32] = &[
    0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3C00, 0xBC00, 0x3800, 0x3E00, 0xC100, 0x3BFF,
    0x3C01, 0x7BFF, 0xFBFF, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01, 0x5640, 0x2E66, 0x1400,
];

fn h(x: u32) -> f64 {
    f64::from(from_f16(x & 0xFFFF))
}

/// Halves in both 16-bit parts of each operand (a junk high half for the
/// low-half instructions).
fn half_pairs() -> Vec<[u32; 4]> {
    let hi = [0xABCD, 0x3C00, 0x7E00, 0x8000];
    pairs(F16_EDGES, F16_EDGES)
        .into_iter()
        .enumerate()
        .map(|(n, i)| {
            [
                i[0] | hi[n % 4] << 16,
                i[1] | F16_EDGES[n % F16_EDGES.len()] << 16,
                (n as u32 * 0x9E37) & 0xFFFF | F16_EDGES[(n / 3) % F16_EDGES.len()] << 16,
                0x5A5A_3C00 ^ (n as u32 & 0xFF),
            ]
        })
        .collect()
}

// An f16 op works on the low halves, NaN in → NaN out, denormals kept (the
// console's mode keeps 16-bit denormals), the result rounded once to
// nearest even into the low half; the high half kept (RDNA2 ISA, 16-bit
// results; nearest even measured on RDNA2 by the Leonida project).

fn f16_min_max(max: bool, a: u32, b: u32) -> u32 {
    let (a, b) = (a & 0xFFFF, b & 0xFFFF);
    let nan = |x: u32| x & 0x7FFF > 0x7C00;
    match (nan(a), nan(b)) {
        (true, true) | (false, true) => a,
        (true, false) => b,
        _ if h(a) == h(b) => {
            if max {
                a & b
            } else {
                a | b
            }
        }
        _ => {
            if (h(a) > h(b)) == max {
                a
            } else {
                b
            }
        }
    }
}

fn f16_denormal(x: u32) -> bool {
    x & 0x7C00 == 0 && x & 0x3FF != 0
}

/// Tags an f16 probe whose model `f(i, toward_zero)` rounds either way and
/// whose f16 operands are `read(i)`.
fn f16_tags(
    p: Probe,
    f: impl Fn([u32; 4], bool) -> Vec<u32> + Clone + 'static,
    read: impl Fn([u32; 4]) -> Vec<u32> + 'static,
) -> Probe {
    let g = f.clone();
    p.shows(F16_TWICE, move |i| {
        read(i).into_iter().any(|x| f16_denormal(x & 0xFFFF))
    })
    .shows_where(F16_ROUND, move |i| f(i, false), move |i| g(i, true))
}

fn sixteen_bit_probes() -> Vec<Probe> {
    let ins = half_pairs();
    let f16 = F16;
    let two = |op: u16, f: fn(f64, f64) -> f64| {
        let m = move |i: [u32; 4], rtz: bool| {
            vec![i[3] & 0xFFFF_0000 | to_f16(f(h(i[0]), h(i[1])), rtz)]
        };
        f16_tags(
            probe([vop2(op, 5, v(1), 2)], ins.clone(), f16, move |i| {
                m(i, false)
            }),
            m,
            |i| vec![i[0], i[1]],
        )
    };
    let fma3 = move |i: [u32; 4], d: u32, rtz: bool| {
        vec![d & 0xFFFF_0000 | to_f16(h(i[0]).mul_add(h(i[1]), h(i[2])), rtz)]
    };
    let mut p = vec![
        two(0x32, |a, b| a + b),
        two(0x33, |a, b| a - b),
        two(0x35, |a, b| a * b),
        probe([vop2(0x39, 5, v(1), 2)], ins.clone(), f16, |i| {
            vec![i[3] & 0xFFFF_0000 | f16_min_max(true, i[0], i[1])]
        }),
        probe([vop2(0x3A, 5, v(1), 2)], ins.clone(), f16, |i| {
            vec![i[3] & 0xFFFF_0000 | f16_min_max(false, i[0], i[1])]
        }),
        f16_tags(
            probe(
                vop3(0x34B, 5, [v(1), v(2), v(3)], NONE),
                ins.clone(),
                f16,
                move |i| fma3(i, i[3], false),
            ),
            move |i, r| fma3(i, i[3], r),
            |i| vec![i[0], i[1], i[2]],
        )
        .shows(F16_DOUBLE, |i| twice(h(i[0]).mul_add(h(i[1]), h(i[2])))),
        // v_fmac_f16 accumulates into v5's low half (v5 = c).
        f16_tags(
            probe(
                [vop1(0x01, 5, v(3)), vop2(0x36, 5, v(1), 2)],
                ins.clone(),
                f16,
                move |i| fma3(i, i[2], false),
            ),
            move |i, r| fma3(i, i[2], r),
            |i| vec![i[0], i[1], i[2]],
        )
        .shows(F16_DOUBLE, |i| twice(h(i[0]).mul_add(h(i[1]), h(i[2])))),
        // v_pack_b32_f16 and v_cvt_f16_f32 with an f32 operand.
        probe(
            vop3(0x311, 5, [v(1), v(2), 0], NONE),
            ins.clone(),
            INT,
            |i| vec![i[0] & 0xFFFF | i[1] << 16],
        ),
    ];
    // Unary f16: reciprocal, square root, its reciprocal, exp2 and log2.
    for (op, f) in [
        (0x54u16, (|x: f64| 1.0 / x) as fn(f64) -> f64),
        (0x55, f64::sqrt),
        (0x56, |x: f64| 1.0 / x.sqrt()),
        (0x58, f64::exp2),
        (0x57, |x: f64| {
            if x == 0.0 {
                f64::NEG_INFINITY
            } else {
                x.log2()
            }
        }),
    ] {
        let m = move |i: [u32; 4], rtz: bool| vec![i[3] & 0xFFFF_0000 | to_f16(f(h(i[0])), rtz)];
        p.push(f16_tags(
            probe(
                [vop1(op, 5, v(1))],
                ins.clone(),
                Kind::F16 {
                    ulps: 1,
                    any_zero: false,
                },
                move |i| m(i, false),
            ),
            m,
            |i| vec![i[0]],
        ));
    }
    // Packed: each half from its own half (op_sel 0, op_sel_hi 1), then
    // with src0's halves swapped and src1's low half read by both.
    for (opsel, opsel_hi) in [(0u8, 7u8), (1, 6), (2, 5)] {
        let pick = move |x: u32, k: u32, high: bool| {
            let bit = if high { opsel_hi } else { opsel } >> k & 1;
            if bit != 0 { x >> 16 } else { x & 0xFFFF }
        };
        let pk = move |op: u16, f: fn(f64, f64, f64) -> f64| {
            let m = move |i: [u32; 4], rtz: bool| {
                let r = |high: bool| {
                    to_f16(
                        f(
                            h(pick(i[0], 0, high)),
                            h(pick(i[1], 1, high)),
                            h(pick(i[2], 2, high)),
                        ),
                        rtz,
                    )
                };
                vec![r(false) | r(true) << 16]
            };
            f16_tags(
                probe(
                    vop3p(op, 5, [v(1), v(2), v(3)], opsel, opsel_hi, [0, 0]),
                    half_pairs(),
                    Kind::Halves { ulps: 0 },
                    move |i| m(i, false),
                ),
                m,
                |i| i[..3].iter().flat_map(|&x| [x, x >> 16]).collect(),
            )
        };
        p.push(pk(0x0F, |a, b, _| a + b));
        p.push(pk(0x10, |a, b, _| a * b));
        p.push(
            pk(0x0E, |a, b, c| a.mul_add(b, c)).shows(F16_DOUBLE, move |i| {
                [false, true].iter().any(|&hi| {
                    twice(h(pick(i[0], 0, hi)).mul_add(h(pick(i[1], 1, hi)), h(pick(i[2], 2, hi))))
                })
            }),
        );
    }
    // v_pk_fmac_f16: both halves accumulate into v5 = c.
    let m = |i: [u32; 4], rtz: bool| {
        let r = |s: u32| to_f16(h(i[0] >> s).mul_add(h(i[1] >> s), h(i[2] >> s)), rtz);
        vec![r(0) | r(16) << 16]
    };
    p.push(
        f16_tags(
            probe(
                [vop1(0x01, 5, v(3)), vop2(0x3C, 5, v(1), 2)],
                half_pairs(),
                Kind::Halves { ulps: 0 },
                move |i| m(i, false),
            ),
            m,
            |i| i[..3].iter().flat_map(|&x| [x, x >> 16]).collect(),
        )
        .shows(F16_DOUBLE, |i| {
            [0, 16]
                .iter()
                .any(|&s| twice(h(i[0] >> s).mul_add(h(i[1] >> s), h(i[2] >> s))))
        }),
    );
    p
}

#[test]
fn sixteen_bit_floats_match_the_hardware() {
    assert_clean(check(&passing(sixteen_bit_probes())));
}

/// `v_fma_mix*`: each source an f32 (a denormal read as zero, measured on
/// RDNA2 by the Leonida project) or, with `op_sel_hi`, the f16 half
/// `op_sel` names; `neg_hi` is the absolute value; fused (RDNA2 ISA).
/// `mixlo`/`mixhi` round the f32 result to nearest even into one half.
fn mix_source(x: u32, k: u32, opsel: u8, opsel_hi: u8, neg: [u8; 2]) -> f64 {
    let mut v = if opsel_hi >> k & 1 != 0 {
        h(if opsel >> k & 1 != 0 { x >> 16 } else { x })
    } else {
        f64::from(fl(ftz(x)))
    };
    if neg[1] >> k & 1 != 0 {
        v = v.abs();
    }
    if neg[0] >> k & 1 != 0 {
        v = -v;
    }
    v
}

fn mix_probes() -> Vec<Probe> {
    let mut inputs = triples(&F32_EDGES[..16], &F32_EDGES[..16], F32_THIRD);
    inputs.extend(half_pairs());
    let mut p = Vec::new();
    for (opsel, opsel_hi, neg) in [
        (0u8, 0u8, [0u8, 0u8]),
        (0, 2, [0, 0]),
        (2, 2, [0, 0]),
        (1, 1, [4, 0]),
        (0, 7, [0, 0]),
        (5, 7, [1, 2]),
    ] {
        // Fused (the hardware), or the product rounded first (the lowering's
        // multiply-add where the host does not fuse).
        let f = move |i: [u32; 4], fused: bool| -> f64 {
            let s = |k: usize| mix_source(i[k], k as u32, opsel, opsel_hi, neg);
            if fused {
                s(0).mul_add(s(1), s(2))
            } else {
                f64::from(fl(out64(s(0) * s(1)))) + s(2)
            }
        };
        let f16_read = move |i: [u32; 4]| -> Vec<u32> {
            (0..3)
                .filter(|&k| opsel_hi >> k & 1 != 0)
                .map(|k| {
                    if opsel >> k & 1 != 0 {
                        i[k] >> 16
                    } else {
                        i[k]
                    }
                })
                .collect()
        };
        let words = |op| vop3p(op, 5, [v(1), v(2), v(3)], opsel, opsel_hi, neg);
        p.push(
            probe(words(0x20), inputs.clone(), tolerant(1, NONE), move |i| {
                vec![out64(f(i, true))]
            })
            .shows_where(
                FMA,
                move |i| vec![out64(f(i, true))],
                move |i| vec![out64(f(i, false))],
            )
            .shows(F16_TWICE, move |i| {
                f16_read(i).into_iter().any(|x| f16_denormal(x & 0xFFFF))
            }),
        );
        // Toward zero: of the value nudged one f64 step toward zero, as the
        // exact fused value may lie past what f64 rounded it to.
        let nudge = |x: f64, rtz: bool| {
            if rtz && x.is_finite() && x != 0.0 {
                f64::from_bits(x.to_bits() - 1)
            } else {
                x
            }
        };
        let lo = move |i: [u32; 4], fused: bool, rtz: bool| {
            vec![i[3] & 0xFFFF_0000 | to_f16(nudge(f(i, fused), rtz), rtz)]
        };
        let hi = move |i: [u32; 4], fused: bool, rtz: bool| {
            vec![i[3] & 0xFFFF | to_f16(nudge(f(i, fused), rtz), rtz) << 16]
        };
        p.push(
            f16_tags(
                probe(words(0x21), inputs.clone(), F16, move |i| {
                    lo(i, true, false)
                }),
                move |i, r| lo(i, true, r),
                f16_read,
            )
            .shows_where(
                FMA,
                move |i| lo(i, true, false),
                move |i| lo(i, false, false),
            )
            .shows(F16_DOUBLE, move |i| twice(f(i, true))),
        );
        p.push(
            f16_tags(
                probe(
                    words(0x22),
                    inputs.clone(),
                    Kind::Halves { ulps: 0 },
                    move |i| hi(i, true, false),
                ),
                move |i, r| hi(i, true, r),
                f16_read,
            )
            .shows_where(
                FMA,
                move |i| hi(i, true, false),
                move |i| hi(i, false, false),
            )
            .shows(F16_DOUBLE, move |i| twice(f(i, true))),
        );
    }
    p
}

#[test]
fn mixed_precision_matches_the_hardware() {
    assert_clean(check(&passing(mix_probes())));
}

/// `v_dot4_i32_i8` and `v_dot4c_i32_i8`: four signed byte products and the
/// accumulator (RDNA2 ISA), with `op_sel_hi` as assemblers write it (all
/// set) and clear.
#[test]
fn byte_dot_products_match_the_hardware() {
    let bytes = [
        0u32,
        0x0102_0304,
        0x7F7F_7F7F,
        0x8080_8080,
        0xFF01_80FF,
        0x1234_5678,
        0xFFFF_FFFF,
    ];
    let inputs = triples(
        &bytes,
        &bytes,
        &[0, 1, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FF00],
    );
    let dot = |a: u32, b: u32, c: u32| {
        (0..4).fold(c as i32, |s, k| {
            s.wrapping_add(i32::from((a >> (8 * k)) as i8) * i32::from((b >> (8 * k)) as i8))
        }) as u32
    };
    assert_clean(check(&[
        probe(
            vop3p(0x16, 5, [v(1), v(2), v(3)], 0, 7, [0, 0]),
            inputs.clone(),
            INT,
            move |i| vec![dot(i[0], i[1], i[2])],
        ),
        // op_sel_hi clear: how the hardware reads the multiplicands then is
        // not public; the adapters must agree.
        Probe {
            body: vop3p(0x16, 5, [v(1), v(2), v(3)], 0, 0, [0, 0]).to_vec(),
            scalar: false,
            inputs: inputs.clone(),
            model: None,
            kind: INT,
            defects: Vec::new(),
        },
        probe(
            [vop1(0x01, 5, v(3)), vop2(0x0D, 5, v(1), 2)],
            inputs,
            INT,
            move |i| vec![dot(i[0], i[1], i[2])],
        ),
    ]));
}

// ---- SDWA ----

/// SDWA operand selection and destination placement in the forms the title
/// uses: words and bytes of a source, zero- or sign-extended, and a 16-bit
/// result into the high word, the low one kept or zeroed (RDNA2 ISA,
/// "Sub-Dword Addressing").
#[test]
fn sub_dword_operands_match_the_hardware() {
    let ints = pairs(U32_EDGES, &U32_EDGES[..12]);
    let floats = unary(F32_EDGES);
    let halves: Vec<[u32; 4]> = half_pairs();
    let sel = |x: u32, s: u8, sext: bool| -> u32 {
        let (w, at) = if s < 4 {
            (8, 8 * u32::from(s))
        } else {
            (16, 16 * u32::from(s - 4))
        };
        let f = x >> at & ((1 << w) - 1);
        if sext && f >> (w - 1) != 0 {
            f | !((1 << w) - 1)
        } else {
            f
        }
    };
    let src = |s0: u8, sext: bool| Sdwa {
        sel: [s0, 6],
        sext: [sext, false],
        ..SDWA
    };
    let mut p = vec![
        probe(
            sdwa(vop1(0x06, 5, 0), 1, src(5, false)),
            unary(U32_EDGES),
            F32,
            move |i| vec![(sel(i[0], 5, false) as f32).to_bits()],
        ),
        probe(
            sdwa(vop1(0x06, 5, 0), 1, src(4, false)),
            unary(U32_EDGES),
            F32,
            move |i| vec![(sel(i[0], 4, false) as f32).to_bits()],
        ),
        probe(
            sdwa(vop1(0x0B, 5, 0), 1, src(5, false)),
            halves.clone(),
            F32,
            move |i| vec![from_f16(i[0] >> 16).to_bits()],
        ),
        probe(
            sdwa(
                vop1(0x0A, 5, 0),
                1,
                Sdwa {
                    dst_sel: 5,
                    unused: 2,
                    ..SDWA
                },
            ),
            floats.clone(),
            Kind::Halves { ulps: 0 },
            |i| vec![i[3] & 0xFFFF | to_f16(f64::from(fl(ftz(i[0]))), false) << 16],
        )
        .shows_where(
            F16_ROUND,
            |i| vec![to_f16(d(i[0]), false)],
            |i| vec![to_f16(d(i[0]), true)],
        ),
        probe(
            sdwa(
                vop1(0x0A, 5, 0),
                1,
                Sdwa {
                    dst_sel: 5,
                    unused: 0,
                    ..SDWA
                },
            ),
            floats,
            Kind::Halves { ulps: 0 },
            |i| vec![to_f16(f64::from(fl(ftz(i[0]))), false) << 16],
        )
        .shows_where(
            F16_ROUND,
            |i| vec![to_f16(d(i[0]), false)],
            |i| vec![to_f16(d(i[0]), true)],
        ),
    ];
    for (op, s0, sext, f) in [
        (0x1Bu16, 4u8, false, (|a: u32, b: u32| a & b) as F2),
        (0x25, 5, false, |a, b| a.wrapping_add(b)),
        (0x26, 1, false, |a, b| a.wrapping_sub(b)),
        (0x25, 0, true, |a, b| a.wrapping_add(b)),
        (0x1D, 5, false, |a, b| a ^ b),
    ] {
        p.push(probe(
            sdwa(vop2(op, 5, 0, 2), 1, src(s0, sext)),
            ints.clone(),
            INT,
            move |i| vec![f(sel(i[0], s0, sext), i[1])],
        ));
    }
    assert_clean(check(&passing(p)));
}

// ---- scalar ----

fn scalar(
    body: impl Into<Vec<u32>>,
    inputs: Vec<[u32; 4]>,
    model: impl Fn([u32; 4]) -> Vec<u32> + 'static,
) -> Probe {
    Probe {
        body: body.into(),
        scalar: true,
        inputs,
        model: Some(Box::new(model)),
        kind: INT,
        defects: Vec::new(),
    }
}

/// The result out as `v5 = s20` and, for instructions that write SCC, SCC
/// as `v6 = scc ? 1 : 0` (`s_cselect_b32`).
fn scalar_out(
    op_word: u32,
    inputs: Vec<[u32; 4]>,
    f: impl Fn([u32; 4]) -> (u32, Option<bool>) + 'static,
) -> Probe {
    let writes_scc = f([0; 4]).1.is_some();
    let mut body = vec![op_word];
    if writes_scc {
        body.push(sop2(0x0A, 21, int(1), int(0)));
        body.push(vop1(0x01, 6, s(21)));
    }
    body.push(vop1(0x01, 5, s(20)));
    scalar(body, inputs, move |i| {
        let (r, scc) = f(i);
        match scc {
            Some(c) => vec![r, u32::from(c)],
            None => vec![r],
        }
    })
}

/// `op s20, s16, s17`.
fn sop2_probe(op: u16, inputs: Vec<[u32; 4]>, f: fn(u32, u32) -> (u32, Option<bool>)) -> Probe {
    scalar_out(sop2(op, 20, s(16), s(17)), inputs, move |i| f(i[0], i[1]))
}

/// `op s20, s16`.
fn sop1_probe(op: u16, inputs: Vec<[u32; 4]>, f: fn(u32) -> (u32, Option<bool>)) -> Probe {
    scalar_out(sop1(op, 20, s(16)), inputs, move |i| f(i[0]))
}

/// Scalar ALU results and SCC (RDNA2 ISA, SOP1/SOP2/SOPC): shift counts
/// read five bits, `s_bfe` takes its offset from bits 4:0 and its width from
/// bits 22:16 of the second operand.
fn scalar_probes() -> Vec<Probe> {
    let ab = pairs(&U32_EDGES[..24], &U32_EDGES[..24]);
    let shifts = pairs(&U32_EDGES[..24], &[0, 1, 16, 31, 32, 33, 63, 0xFFFF_FFE1]);
    let fields: Vec<[u32; 4]> = [0u32, 0x8000_0000, 0xFFFF_FFFF, 0x1234_5678, 0xD2F9_0159]
        .iter()
        .flat_map(|&a| {
            [0u32, 4, 16, 28, 31, 32].iter().flat_map(move |&o| {
                [0u32, 1, 4, 16, 31, 32, 33, 0x7F].map(move |w| [a, o | w << 16, 0, OLD])
            })
        })
        .collect();
    let sbfe = |a: u32, b: u32, signed: bool| -> u32 {
        let (o, w) = (b & 31, b >> 16 & 0x7F);
        let x = if signed {
            ((a as i32) >> o) as u32
        } else {
            a >> o
        };
        if w == 0 {
            return 0;
        }
        if w >= 32 {
            return x;
        }
        let f = x & ((1 << w) - 1);
        if signed && f >> (w - 1) & 1 != 0 {
            f | !((1 << w) - 1)
        } else {
            f
        }
    };
    let mut p = vec![
        // (SCC of the adds, their carry or overflow, the lowering refuses.)
        sop2_probe(0x02, ab.clone(), |a, b| (a.wrapping_add(b), None)),
        sop2_probe(0x00, ab.clone(), |a, b| (a.wrapping_add(b), None)),
        sop2_probe(0x0E, ab.clone(), |a, b| (a & b, Some(a & b != 0))),
        sop2_probe(0x10, ab.clone(), |a, b| (a | b, Some(a | b != 0))),
        sop2_probe(0x26, ab.clone(), |a, b| (a.wrapping_mul(b), None)),
        sop2_probe(0x35, ab.clone(), |a, b| {
            (((u64::from(a) * u64::from(b)) >> 32) as u32, None)
        }),
        // SCC: whether the first operand was chosen (RDNA2 ISA S_MIN/S_MAX).
        sop2_probe(0x07, ab.clone(), |a, b| (a.min(b), Some(a < b)))
            .shows(SCC_MIN_MAX, |i| (i[0] < i[1]) != (i[0].min(i[1]) != 0)),
        sop2_probe(0x09, ab.clone(), |a, b| (a.max(b), Some(a > b)))
            .shows(SCC_MIN_MAX, |i| (i[0] > i[1]) != (i[0].max(i[1]) != 0)),
        sop2_probe(0x06, ab.clone(), |a, b| {
            (
                (a as i32).min(b as i32) as u32,
                Some((a as i32) < (b as i32)),
            )
        })
        .shows(SCC_MIN_MAX, |i| {
            ((i[0] as i32) < (i[1] as i32)) != ((i[0] as i32).min(i[1] as i32) != 0)
        }),
        sop2_probe(0x08, ab.clone(), |a, b| {
            (
                (a as i32).max(b as i32) as u32,
                Some((a as i32) > (b as i32)),
            )
        })
        .shows(SCC_MIN_MAX, |i| {
            ((i[0] as i32) > (i[1] as i32)) != ((i[0] as i32).max(i[1] as i32) != 0)
        }),
        sop2_probe(0x1E, shifts.clone(), |a, n| {
            (a << (n & 31), Some(a << (n & 31) != 0))
        }),
        sop2_probe(0x20, shifts.clone(), |a, n| {
            (a >> (n & 31), Some(a >> (n & 31) != 0))
        }),
        sop2_probe(0x22, shifts.clone(), |a, n| {
            let r = ((a as i32) >> (n & 31)) as u32;
            (r, Some(r != 0))
        }),
        sop2_probe(
            0x24,
            pairs(&[0, 1, 4, 31, 32, 33], &[0, 1, 16, 31, 32]),
            |w, o| {
                (
                    ((1u64 << (w & 31)) as u32).wrapping_sub(1) << (o & 31),
                    None,
                )
            },
        ),
        sop1_probe(0x13, unary(U32_EDGES), |a| {
            (if a == 0 { u32::MAX } else { a.trailing_zeros() }, None)
        }),
        sop1_probe(0x15, unary(U32_EDGES), |a| (ffbh(a), None)),
        sop1_probe(0x0F, unary(U32_EDGES), |a| (a.count_ones(), Some(a != 0))),
        sop1_probe(0x0B, unary(U32_EDGES), |a| (a.reverse_bits(), None)),
        sop1_probe(0x34, unary(U32_EDGES), |a| {
            let r = (a as i32).wrapping_abs() as u32;
            (r, Some(r != 0))
        }),
    ];
    let bfe_u: fn(u32, u32) -> (u32, Option<bool>) = |a, b| {
        let (o, w) = (b & 31, b >> 16 & 0x7F);
        let x = a >> o;
        let r = if w == 0 {
            0
        } else if w >= 32 {
            x
        } else {
            x & ((1 << w) - 1)
        };
        (r, Some(r != 0))
    };
    p.push(sop2_probe(0x27, fields.clone(), bfe_u));
    p.push(scalar(
        [sop2(0x28, 20, s(16), s(17)), vop1(0x01, 5, s(20))],
        fields,
        move |i| vec![sbfe(i[0], i[1], true)],
    ));
    // Compares into SCC, read by s_cselect.
    for (op, f) in [
        (0x06u16, (|a: u32, b: u32| a == b) as fn(u32, u32) -> bool),
        (0x07, |a, b| a != b),
        (0x0A, |a, b| a < b),
        (0x08, |a, b| a > b),
        (0x04, |a, b| (a as i32) < (b as i32)),
        (0x02, |a, b| (a as i32) > (b as i32)),
    ] {
        p.push(scalar(
            [
                sopc(op, s(16), s(17)),
                sop2(0x0A, 20, s(18), s(19)),
                vop1(0x01, 5, s(20)),
            ],
            ab.clone(),
            move |i| vec![if f(i[0], i[1]) { i[2] } else { i[3] }],
        ));
    }
    // SCC as a vector operand (`src_scc`).
    p.push(scalar(
        [sopc(0x0A, s(16), s(17)), vop1(0x01, 5, SRC_SCC)],
        ab,
        |i| vec![u32::from(i[0] < i[1])],
    ));
    p
}

#[test]
fn scalar_alu_matches_the_hardware() {
    assert_clean(check(&passing(scalar_probes())));
}

// ---- lanes ----

/// Masked lanes keep their registers: `s_and_saveexec_b32` narrowing to the
/// lanes of a compare, writes there, then EXEC restored (no branch, as the
/// 2020 compiler never leaves one out but other code may).
#[test]
fn masked_lanes_keep_their_registers() {
    let inputs = pairs(U32_EDGES, &U32_EDGES[..8]);
    let body = [
        vopc(0xC4, v(1), 2), // v_cmp_gt_u32 vcc, v1, v2
        sop1(0x3C, 20, VCC), // s_and_saveexec_b32 s20, vcc
        vop2(0x25, 5, v(1), 2),
        vop1(0x01, 6, int(7)),
        sop1(0x03, 126, s(20)),
        vop2(0x26, 7, v(1), 2),
    ];
    let mut cmpx = vec![sop1(0x03, 20, EXEC_LO), vopc(0xD4, v(1), 2)]; // v_cmpx_gt_u32
    cmpx.extend(vop3(0x16A, 5, [v(1), v(2), 0], NONE));
    cmpx.push(sop1(0x03, 126, s(20)));
    assert_clean(check(&[
        probe(body, inputs.clone(), INT, |i| {
            let on = i[0] > i[1];
            vec![
                if on { i[0].wrapping_add(i[1]) } else { OLD },
                if on { 7 } else { OLD },
                i[0].wrapping_sub(i[1]),
            ]
        }),
        probe(cmpx, inputs, INT, |i| {
            vec![if i[0] > i[1] {
                ((u64::from(i[0]) * u64::from(i[1])) >> 32) as u32
            } else {
                OLD
            }]
        }),
    ]));
}

// ---- defects ----
//
// Each runs the inputs a defect of the lowering showed on. The fixed ones
// run as tests; the open one is ignored until it is fixed.

fn defect(probes: Vec<Probe>, which: &str) {
    let probes = defective(probes, which);
    assert!(!probes.is_empty(), "no input shows {which}");
    assert_clean(check(&probes));
}

/// Open: `v_fma_f32`, `v_fmac_f32` and `v_fma_mix*` are fused on the
/// hardware (RDNA2 ISA; LLVM's `getNewFMAInst` maps `V_FMAC_F32` to
/// `V_FMA_F32`); the lowering emits DXIL `FMad`, which the host may leave
/// unfused, and WARP does: NaN for `fma(1.5, MAX, -inf)` (the product
/// overflows first), +inf for `fma(1.5, MAX, -MAX)`, 0 for
/// `fma(1 - ulp, 1 + ulp, -1)`, +0 for a tiny negative product plus +0.
/// The RX 6700 XT fuses, and `victory-gpu` warns once at device creation
/// on a host that does not. DXIL has no f32 fused multiply-add, so fixing
/// it means computing the 6.6 M instances in f64 (`Fma` on doubles, where
/// the host has them: exact product, one rounding to f64, then to f32) or
/// with an error-free product in f32.
#[test]
#[ignore = "defect: multiply-adds are fused only where the host fuses"]
fn defect_multiply_adds_fused_only_where_the_host_fuses() {
    let mut p = arithmetic_probes();
    p.extend(mix_probes());
    defect(p, FMA);
}

/// The VOP3 output multipliers flush the result and turn a zero into +0
/// with IEEE mode off (measured on RDNA2, gfx1031 and gfx1036, by the
/// Leonida project). Was a defect: the host's multiply kept -0.
#[test]
fn output_multipliers_give_positive_zero() {
    let mut p = arithmetic_probes();
    p.extend(min_max_probes());
    p.extend(transcendental_probes());
    defect(p, OMOD);
}

/// Denormal operands of `v_min/max/med3/min3/max3_f32` read as zeros of
/// their sign, and a float `v_cndmask_b32` with `abs`/`neg` moves only the
/// sign bit. Was a defect: the host decided (WARP flushed, the RX 6700 XT
/// did not, and a flushed denormal met +0 as an equal).
#[test]
fn min_max_and_selects_take_denormals_on_the_bits() {
    let mut p = min_max_probes();
    p.extend(select_probes());
    defect(p, DENORMAL);
}

/// f16 results round to nearest even, the console's default mode (toward
/// zero only where the program sets that, which the title does around its
/// 16-bit interpolation). Was a defect: DXIL `legacyF32ToF16` rounds toward
/// zero on both adapters here (65504 where the hardware gives infinity).
#[test]
fn f16_results_round_to_nearest_even() {
    let mut p = conversion_probes();
    p.extend(sixteen_bit_probes());
    p.extend(mix_probes());
    defect(p, F16_ROUND);
}

/// f16 arithmetic gives the same bits on every host: the f32 intermediate
/// of an add or multiply of halves is exact or rounds innocuously, and the
/// narrowing is on the bits. Was a defect: the hosts' own narrowing kept
/// extra precision on one adapter and not the other (`1.0 * 1.5 + 0x8001`
/// in `v_pk_fma_f16` gave 0x3e00 on WARP, 0x3dff on the RX 6700 XT).
#[test]
fn f16_arithmetic_rounds_once() {
    let mut p = sixteen_bit_probes();
    p.extend(mix_probes());
    defect(p, F16_TWICE);
}

/// `s_min_*`/`s_max_*` set SCC to whether the first operand was chosen
/// (RDNA2 ISA). Was a defect: SCC was `result != 0`.
#[test]
fn scalar_min_max_set_scc_from_the_compare() {
    defect(scalar_probes(), SCC_MIN_MAX);
}

/// `v_cvt_flr_i32_f32`/`v_cvt_rpi_i32_f32` saturate a NaN by its sign and
/// add the half exactly (measured on RDNA2, gfx1035/gfx1036, by the
/// Leonida project). Was a defect: NaN gave 0 and x + 0.5 was rounded
/// first (0x3effffff gave 1, the hardware 0). The title uses neither.
#[test]
fn floor_conversions_match_the_hardware() {
    defect(conversion_probes(), FLOOR_CVT);
}

/// `v_med3_f32` orders -0 below +0, as its min and max do. Was a defect:
/// plain DXIL `FMin`/`FMax` gave either zero (the adapters differed).
#[test]
fn med3_orders_signed_zeros() {
    defect(min_max_probes(), MED3_ZERO);
}

/// `v_cmp_f_f32` and `v_cmp_tru_f32` (always false, always true; the title
/// uses neither). Was a defect: `fcmp false`/`fcmp true` removed the WARP
/// device.
#[test]
fn constant_compares_run() {
    let ab = pairs(&F32_EDGES[..8], &F32_EDGES[..8]);
    let p = [0x0u16, 0xF]
        .into_iter()
        .map(|op| {
            probe(with_vcc(vopc(op, v(1), 2)), ab.clone(), INT, move |_| {
                vec![u32::from(op == 0xF)]
            })
            .shows(CONSTANT_CMP, |_| true)
        })
        .collect();
    defect(p, CONSTANT_CMP);
}

/// The 16-bit round mode a program sets around its 16-bit results (the
/// title sets toward zero around `v_interp_p2_f16`, then nearest even
/// again): `s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2)` (RDNA2 ISA, MODE
/// bits 3:2).
#[test]
fn f16_results_round_as_the_mode_says() {
    let setreg = |mode: u32| [0xBA80_0881u32, mode];
    let mut body = setreg(3).to_vec();
    body.push(vop1(0x0A, 5, v(1)));
    body.extend(setreg(0));
    body.push(vop1(0x0A, 6, v(1)));
    let narrow = |x: u32, rtz: bool| to_f16(f64::from(fl(ftz(x))), rtz);
    assert_clean(check(&[probe(body, unary(F32_EDGES), F16, move |i| {
        vec![
            i[3] & 0xFFFF_0000 | narrow(i[0], true),
            i[3] & 0xFFFF_0000 | narrow(i[0], false),
        ]
    })]));
}

/// The check `victory-gpu` runs at device creation: the RX 6700 XT fuses
/// the lowering's multiply-add, WARP does not.
#[test]
fn hosts_that_do_not_fuse_are_named() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    if let Ok(warp) = Host::new(true) {
        assert_eq!(warp.fuses_fmad(), Ok(false), "WARP");
    }
    if let Ok(hardware) = Host::new(false) {
        let fuses = hardware.fuses_fmad().expect("the check runs");
        eprintln!("hardware adapter fuses FMad: {fuses}");
    }
}

/// A result that does not depend on the inputs (the control for the
/// constant compares above).
#[test]
fn constant_results_run() {
    assert_clean(check(&[probe(
        [vop1(0x01, 5, int(1))],
        unary(&F32_EDGES[..8]),
        INT,
        |_| vec![1],
    )]));
}

// ---- more of the title's forms ----

/// 64-bit shifts read six bits of the count (RDNA2 ISA `V_LSHLREV_B64`).
#[test]
fn wide_shifts_match_the_hardware() {
    let counts = [0u32, 1, 31, 32, 33, 63, 64, 65, 0xFFFF_FFC1];
    let inputs: Vec<[u32; 4]> = counts
        .iter()
        .flat_map(|&n| {
            U32_EDGES
                .iter()
                .map(move |&x| [n, x, x.rotate_left(13) ^ 0x8000_0001, OLD])
        })
        .collect();
    let wide = |i: [u32; 4]| u64::from(i[1]) | u64::from(i[2]) << 32;
    let split = |x: u64| vec![x as u32, (x >> 32) as u32];
    let shift = |op: u16, f: fn(u64, u32) -> u64| {
        probe(
            vop3(op, 5, [v(1), v(2), 0], NONE),
            inputs.clone(),
            INT,
            move |i| split(f(wide(i), i[0] & 63)),
        )
    };
    assert_clean(check(&[
        shift(0x2FF, |x, n| x << n),
        shift(0x300, |x, n| x >> n),
        shift(0x301, |x, n| ((x as i64) >> n) as u64),
    ]));
}

/// `v_cvt_pk_u8_f32`: a float to a byte (NaN 0, saturated; ties to even,
/// which no public source states) into byte `S1[1:0]` of `S2` (RDNA2 ISA).
#[test]
fn byte_packing_matches_the_hardware() {
    let mut xs = F32_EDGES.to_vec();
    xs.extend(
        [
            0.5f32, 1.5, 2.5, 127.5, 128.5, 254.5, 255.49, 255.5, 256.0, -0.5, 3.75,
        ]
        .map(f32::to_bits),
    );
    let inputs = triples(&xs, &[0, 1, 2, 3, 7], &[0xAABB_CCDD]);
    assert_clean(check(&[probe(
        vop3(0x15E, 5, [v(1), v(2), v(3)], NONE),
        inputs,
        INT,
        |i| {
            let x = fl(ftz(i[0]));
            let byte = if x.is_nan() {
                0
            } else {
                x.round_ties_even().clamp(0.0, 255.0) as u32
            };
            let at = 8 * (i[1] & 3);
            vec![i[2] & !(0xFF << at) | byte << at]
        },
    )]));
}

/// The other half of a branch: `s_andn2_wrexec_b32` (EXEC = S0 & ~EXEC,
/// the title's else) after a `v_cmpx`, each side writing its own lanes.
#[test]
fn else_lanes_keep_their_registers() {
    let inputs = pairs(U32_EDGES, &U32_EDGES[..8]);
    let body = [
        sop1(0x03, 20, EXEC_LO),
        vopc(0xD4, v(1), 2),
        vop1(0x01, 5, int(1)),
        sop1(0x47, 21, s(20)),
        vop1(0x01, 6, int(2)),
        sop1(0x03, 126, s(20)),
    ];
    assert_clean(check(&[probe(body, inputs, INT, |i| {
        let on = i[0] > i[1];
        vec![if on { 1 } else { OLD }, if on { OLD } else { 2 }]
    })]));
}

/// DS `op` at `addr + offset`: data from `data`, result to `vdst`.
fn ds(op: u32, vdst: u8, addr: u8, data: u8, offset: u8) -> [u32; 2] {
    [
        0xD800_0000 | op << 18 | u32::from(offset),
        u32::from(vdst) << 24 | u32::from(data) << 8 | u32::from(addr),
    ]
}

/// LDS bytes and halves: zero and sign extension on reads, partial writes
/// keeping the other bytes (RDNA2 ISA DS). Each thread uses its own dword.
#[test]
fn lds_bytes_and_halves_match_the_hardware() {
    let inputs = pairs(U32_EDGES, &U32_EDGES[..10]);
    let mut reads = vec![vop2(0x1A, 11, int(2), 0)];
    reads.extend(ds(0x0D, 0, 11, 1, 0));
    reads.extend(ds(0x3A, 5, 11, 0, 1));
    reads.extend(ds(0x39, 6, 11, 0, 3));
    reads.extend(ds(0x3C, 7, 11, 0, 2));
    reads.extend(ds(0x3B, 8, 11, 0, 2));
    let mut writes = vec![vop2(0x1A, 11, int(2), 0)];
    writes.extend(ds(0x0D, 0, 11, 4, 0));
    writes.extend(ds(0x1E, 0, 11, 1, 1));
    writes.extend(ds(0x1F, 0, 11, 2, 2));
    writes.extend(ds(0x36, 5, 11, 0, 0));
    let byte = |x: u32, k: u32| x >> (8 * k) & 0xFF;
    assert_clean(check(&[
        probe(reads, inputs.clone(), INT, move |i| {
            let a = i[0];
            vec![
                byte(a, 1),
                byte(a, 3) as u8 as i8 as i32 as u32,
                a >> 16,
                (a >> 16) as u16 as i16 as i32 as u32,
            ]
        }),
        probe(writes, inputs, INT, move |i| {
            vec![i[3] & 0xFF | byte(i[0], 0) << 8 | (i[1] & 0xFFFF) << 16]
        }),
    ]));
}
