//! A conservative proof of exact descriptor slots and admitted accesses.
//! UsedResource records describe ranges, not an upper bound on runtime
//! indices. Trace the actual descriptor operands instead of trusting them.
use std::collections::{BTreeMap, BTreeSet};

use crate::gfx10::{Fields, NULL, Operand};
use crate::native::{HwStage, NativeProgram, ResourceType, RootType, StageIo};
use crate::ir::Reg;
use crate::resource::{Address, Role, Source};
use crate::{Container, FourCC, ShaderError};

/// Offsets are native-size SRV slots relative to the table value bound by
/// the title, not absolute indices in an existing heap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct StaticTextureSlot {
    pub root: u8,
    pub index: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StaticViewAccess {
    TextureRead { array: bool },
    TextureWrite {
        x: u32, y: u32, z: Option<u32>,
        sources: [&'static str; 3], exec_guard: bool,
        dispatch: Option<([u32; 3], [u32; 3])>,
    },
    /// A statically bounded aligned byte window on a raw UAV, as used by DXIL.
    RawUav { offset: u32, bytes: u32, write: bool },
    RawSrv { offset: u32, bytes: u32 },
    /// 游戏表内 CBV 由宿主翻译为原始只读缓冲区。
    ConstantTable { offset: u32, bytes: u32 },
    RootSrv { bytes: u32 },
    StructuredSrv { stride: u32, elements: u32 },
    /// 元素索引由完整的只读结构化描述符限制，绑定时不可裁剪视图。
    StructuredSrvDescriptor { stride: u32 },
    StructuredUav { stride: u32, elements: u32, write: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct StaticViewSlot {
    pub slot: StaticTextureSlot,
    pub access: StaticViewAccess,
}

pub fn static_texture_slots(container: &[u8]) -> Result<Option<Vec<StaticTextureSlot>>, ShaderError> {
    Ok(static_view_slots(container)?.and_then(|views| {
        views.into_iter().map(|view| {
            matches!(view.access, StaticViewAccess::TextureRead { array: false }).then_some(view.slot)
        }).collect()
    }))
}

/// 地址来源与静态上界的类别，不保留寄存器编号或计算出来的地址。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferOperandEvidence {
    pub producer: &'static str,
    pub bounded: bool,
    pub bounded_at_access: bool,
    pub aligned_dword_bound: bool,
}

/// 静态证据不完整时的原因。不输出着色器字节、资源内容或来宾地址。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaticViewFailure {
    ParseOrAnalysis,
    NoNativeStage,
    DynamicSampler { operation: &'static str },
    UnsupportedAccess { operation: &'static str, image_dimension: Option<u8> },
    UnsupportedBufferAccess {
        operation: &'static str,
        descriptor: &'static str,
        indexed: bool,
        vector_offset: bool,
        immediate_aligned: bool,
        scalar_offset: &'static str,
        scalar_evidence: Option<BufferOperandEvidence>,
        index_evidence: Option<BufferOperandEvidence>,
        offset_evidence: Option<BufferOperandEvidence>,
        typed: bool,
        lds: bool,
        tfe: bool,
    },
    UnprovenRootView { operation: &'static str },
    UntracedSource { operation: &'static str },
    DivergentSource { operation: &'static str },
    MissingTableAddress { operation: &'static str },
    UnprovenTableSlot { operation: &'static str, reason: TableSlotFailure },
}

/// 不记录当前地址、表偏移原值或寄存器数据，只标明哪条证明失败。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableSlotFailure {
    UnknownRoot,
    RootKind(RootType),
    DynamicOffset,
    OffsetOverflow,
    Misaligned,
    OutsideWindow,
    NoDeclaration,
    RootSlotMismatch,
    Declaration {
        kind: ResourceType,
        dwords: u8,
        stride: u32,
        unique: bool,
        field_fits_stride: Option<bool>,
        statically_bounded_index: Option<bool>,
    },
}

pub enum StaticViewReport {
    Proven(Vec<StaticViewSlot>),
    Rejected(StaticViewFailure),
}

/// Prove descriptor operands and raw access byte windows from native
/// instructions, not inferred from resource declarations. Excludes
/// arbitrary addresses and does not by itself admit a host shader.
pub fn static_view_slots(container: &[u8]) -> Result<Option<Vec<StaticViewSlot>>, ShaderError> {
    Ok(match static_view_report(container)? {
        StaticViewReport::Proven(slots) => Some(slots),
        StaticViewReport::Rejected(_) => None,
    })
}

pub fn static_view_report(container: &[u8]) -> Result<StaticViewReport, ShaderError> {
    view_report(container, None)
}

/// 使用当前直接派发的组数；结果不能复用于组数不同或未知的派发。
pub fn direct_dispatch_view_report(container: &[u8], groups: [u32; 3]) -> Result<StaticViewReport, ShaderError> {
    let direct = groups.iter().all(|&n| (1..=65535).contains(&n)).then_some(groups);
    view_report(container, direct)
}

fn view_report(container: &[u8], direct: Option<[u32; 3]>) -> Result<StaticViewReport, ShaderError> {
    use StaticViewFailure as Reason;
    let rejected = |reason| Ok(StaticViewReport::Rejected(reason));
    let c = Container::parse(container)?;
    let dxil = c.part(FourCC::DXIL).or(c.part(FourCC::ACCC))
        .ok_or(ShaderError::Malformed("no DXIL part"))?;
    let Some(native) = crate::native_programs(dxil.data)?.into_iter()
        .find(|p| matches!(p.stage, HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms))
    else {
        // The translated host blob cannot stand in for a guest proof.
        return rejected(Reason::NoNativeStage);
    };
    let (p, st) = crate::range::build(&native.patched_code(), native.wave64, native.vgprs)?;
    let st = st?;
    let resources = crate::resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let mut bounds = crate::range::Ranges::new(&p, &st);
    if let (HwStage::Cs, StageIo::Cs(cs)) = (native.stage, &native.io) {
        for (axis, threads) in cs.threads.into_iter().take(usize::from(cs.thread_id_vgprs())).enumerate() {
            if let Some(upper) = threads.checked_sub(1) {
                bounds.bound_input(Reg::V(axis as u8), upper);
            }
        }
        if let Some(groups) = direct {
            for (register, count) in cs.system_sgprs().0.into_iter().zip(groups) {
                if let Some(register) = register {
                    bounds.bound_input(Reg::S(register), count - 1);
                }
            }
        }
    }
    let dispatch = match (&native.io, direct) {
        (StageIo::Cs(cs), Some(groups)) => Some((groups, cs.threads)),
        _ => None,
    };
    let addresses: BTreeMap<_, _> = resources.scalar.iter().copied().collect();
    let mut slots = BTreeSet::new();
    for descriptor in &resources.descriptors {
        let (b, i) = descriptor.site;
        let inst = &p.blocks[b].insts[i];
        let name = inst.name();
        if descriptor.role == Role::Sampler {
            // Immediate samplers use the host PSO's static samplers;
            // do not admit a second, dynamically indexed sampler heap.
            if !matches!(descriptor.source, Source::Constant(_)) {
                return rejected(Reason::DynamicSampler { operation: name });
            }
            continue;
        }
        let access = if let Some(access) = bounded_texture_write(inst, &mut bounds, (b, i), dispatch) {
            AccessProof::Native(access)
        } else if let Fields::Image { dim, .. } = inst.fields
            && let Some(access) = closed_image_read(name, dim)
        {
            AccessProof::Native(access)
        } else if let Some(access) = bounded_scalar_buffer_read(inst, &mut bounds) {
            access
        } else if let Some(access) = bounded_typed_load(inst, &mut bounds) {
            access
        } else if let Some(access) = bounded_structured_access(inst, &mut bounds) {
            access
        } else if let Some(access) = bounded_raw_access(inst, &mut bounds) {
            AccessProof::Native(access)
        } else {
            if let Some(reason) = buffer_access_evidence(inst, &mut bounds, (b, i), &descriptor.source) {
                return rejected(reason);
            }
            let image_dimension = match inst.fields {
                Fields::Image { dim, .. } => Some(dim),
                _ => None,
            };
            return rejected(Reason::UnsupportedAccess { operation: name, image_dimension });
        };
        if let Source::Built { address, bits, words } = descriptor.source {
            let Some(root) = proven_root_srv(&native, address, bits, words, access) else {
                return rejected(Reason::UnprovenRootView { operation: name });
            };
            slots.insert(root);
            continue;
        }
        let Source::Loaded { load, first } = descriptor.source else {
            return rejected(Reason::UntracedSource { operation: name });
        };
        if resources.waterfall.loads.contains(&load) {
            return rejected(Reason::DivergentSource { operation: name });
        }
        let Some(&address) = addresses.get(&load) else {
            return rejected(Reason::MissingTableAddress { operation: name });
        };
        match proven_slot(&native, address, first, access) {
            Ok(view) => { slots.insert(view); }
            Err(reason) => return rejected(Reason::UnprovenTableSlot { operation: name, reason }),
        }
    }
    Ok(StaticViewReport::Proven(slots.into_iter().collect()))
}

fn buffer_access_evidence(
    inst: &crate::ir::Inst, bounds: &mut crate::range::Ranges<'_>, site: (usize, usize), source: &Source,
) -> Option<StaticViewFailure> {
    let Fields::Buffer { addr, offset, soffset, idxen, offen, format, lds, tfe, .. } = &inst.fields
        else { return None; };
    let mut prove = |reg| {
        let value = inst.read(reg)?;
        Some(BufferOperandEvidence {
            producer: bounds.value_source_at(value, site).0,
            bounded: bounds.unsigned_upper_bound(value).is_some(),
            bounded_at_access: bounds.unsigned_upper_bound_at(value, site).is_some(),
            aligned_dword_bound: bounds.dword_offset_hi(value).is_some(),
        })
    };
    let (index_evidence, offset_evidence) = if let Operand::Vector { reg, .. } = *addr {
        (
            idxen.then(|| prove(Reg::V(reg))).flatten(),
            offen.then(|| reg.checked_add(u8::from(*idxen)).and_then(|r| prove(Reg::V(r)))).flatten(),
        )
    } else {
        (None, None)
    };
    let (scalar_offset, scalar_evidence) = match *soffset {
        Operand::Scalar { code: NULL, .. } | Operand::Const(128) => ("zero", None),
        Operand::Const(code @ 129..=192) if (code - 128).is_multiple_of(4) => ("aligned_constant", None),
        Operand::Const(_) => ("other_constant", None),
        Operand::Scalar { code: code @ 0..=105, dwords: 1 } => ("scalar", prove(Reg::S(code))),
        _ => ("other", None),
    };
    let descriptor = match source {
        Source::Loaded { .. } => "scalar_load",
        Source::Built { .. } => "built",
        Source::Constant(_) => "constant",
        Source::Fetched { .. } => "vector_load",
        _ => "other",
    };
    Some(StaticViewFailure::UnsupportedBufferAccess {
        operation: inst.name(), descriptor, indexed: *idxen, vector_offset: *offen,
        immediate_aligned: offset.is_multiple_of(4), scalar_offset, scalar_evidence,
        index_evidence, offset_evidence, typed: format.is_some(), lds: *lds, tfe: *tfe,
    })
}

fn closed_image_read(operation: &str, dimension: u8) -> Option<StaticViewAccess> {
    // getDimensions 只读取视图信息，不写资源。仍要求绑定对应的原生 SRV，
    // 以便返回真实维度和 mip。下面的描述符证明并不会因此接受 UAV。
    // 暂不接纳数组尺寸查询：现有翻译还不能保证数组层数通道的语义。
    if operation == "image_get_resinfo" && dimension != 1 {
        return None;
    }
    if !operation.starts_with("image_sample")
        && !matches!(operation, "image_load" | "image_load_mip" | "image_get_resinfo")
    {
        return None;
    }
    match dimension {
        1 => Some(StaticViewAccess::TextureRead { array: false }),
        4 => Some(StaticViewAccess::TextureRead { array: true }),
        _ => None,
    }
}

#[cfg(test)]
#[test]
fn page_image_size_query_uses_the_exact_2d_shader_view() {
    assert_eq!(closed_image_read("image_get_resinfo", 1), Some(StaticViewAccess::TextureRead { array: false }));
    assert_eq!(closed_image_read("image_get_resinfo", 4), None);
    assert_eq!(closed_image_read("image_get_resinfo", 3), None);
    assert_eq!(closed_image_read("image_store", 1), None);
}

/// One instruction set for raw atomic admission and DXIL lowering.
/// Compare-exchange is separately encoded; the others use atomicBinOp.
pub(super) fn raw_atomic_opcode(name: &str) -> Option<u32> {
    Some(match name.strip_prefix("buffer_atomic_")? {
        "add" => 0,
        "and" => 1,
        "or" => 2,
        "xor" => 3,
        "smin" => 4,
        "smax" => 5,
        "umin" => 6,
        "umax" => 7,
        "swap" => 8,
        "cmpswap" => u32::MAX,
        _ => return None,
    })
}

fn bounded_raw_access(inst: &crate::ir::Inst, bounds: &mut crate::range::Ranges<'_>) -> Option<StaticViewAccess> {
    let name = inst.name();
    let (write, bytes) = if raw_atomic_opcode(name).is_some() {
        // Even compare-exchange's two input operands modify one dword.
        (true, 4)
    } else {
        let (operation, width) = name.strip_prefix("buffer_")?.split_once('_')?;
        let write = match operation {
            "store" => true,
            "load" => false,
            _ => return None,
        };
        let bytes = match width {
            "dword" => 4,
            "dwordx2" => 8,
            "dwordx3" => 12,
            "dwordx4" => 16,
            _ => return None,
        };
        (write, bytes)
    };
    let Fields::Buffer {
        addr, offset, soffset, offen, idxen: false, format: None, lds: false, tfe: false, ..
    } = &inst.fields else { return None };
    if !offset.is_multiple_of(4) {
        return None;
    }
    let offset = u32::from(*offset);
    let vector_hi = if *offen {
        let Operand::Vector { reg, .. } = *addr else { return None; };
        bounds.dword_offset_hi(inst.read(crate::ir::Reg::V(reg))?)?
    } else {
        0
    };
    let (offset, scalar_hi) = match *soffset {
        Operand::Scalar { code: NULL, .. } => (offset, 0),
        Operand::Const(n @ 128..=192) => {
            let constant = u32::from(n - 128);
            if !constant.is_multiple_of(4) { return None; }
            (offset.checked_add(constant)?, 0)
        }
        Operand::Scalar { code: code @ 0..=105, dwords: 1 } =>
            (offset, bounds.dword_offset_hi(inst.read(crate::ir::Reg::S(code))?)?),
        _ => return None,
    };
    let span = vector_hi.checked_add(scalar_hi)?.checked_add(bytes)?;
    // The renderer may treat this entire conservative interval as
    // dirty, but never bind outside the proven descriptor footprint.
    (offset.checked_add(span)? <= 65536)
        .then_some(StaticViewAccess::RawUav { offset, bytes: span, write })
}

fn bounded_texture_write(
    inst: &crate::ir::Inst, bounds: &mut crate::range::Ranges<'_>, site: (usize, usize),
    dispatch: Option<([u32; 3], [u32; 3])>,
) -> Option<StaticViewAccess> {
    let Fields::Image {
        addr, dim, dmask: 15, a16: false, d16: false, tfe: false, lwe: false, ..
    } = &inst.fields else { return None; };
    if inst.name() != "image_store" || !matches!(*dim, 1 | 2) { return None; }
    let mut regs = addr.iter().flat_map(|operand| match *operand {
        Operand::Vector { reg, dwords } => reg..reg.saturating_add(dwords),
        _ => 0..0,
    });
    let a = inst.read(crate::ir::Reg::V(regs.next()?))?;
    let b = inst.read(crate::ir::Reg::V(regs.next()?))?;
    let (x, y) = (bounds.unsigned_upper_bound_at(a, site)?, bounds.unsigned_upper_bound_at(b, site)?);
    let (source_x, exec_guard) = bounds.value_source_at(a, site);
    let (source_y, _) = bounds.value_source_at(b, site);
    let (z, source_z) = if *dim == 2 {
        let c = inst.read(crate::ir::Reg::V(regs.next()?))?;
        (Some(bounds.unsigned_upper_bound_at(c, site)?), bounds.value_source_at(c, site).0)
    } else {
        (None, "-")
    };
    Some(StaticViewAccess::TextureWrite { x, y, z,
        sources: [source_x, source_y, source_z], exec_guard, dispatch,
    })
}

enum AccessProof {
    Native(StaticViewAccess),
    ScalarBufferRead { offset: u32, bytes: u32 },
    TypedRead { immediate: u32, vector_hi: u32, footprint: super::lower::TypedLoadFootprint },
    StructuredAccess { elements: Option<u32>, last_byte: u32, write: bool },
}

fn scalar_buffer_read_window(name: &str, offset: i32, scalar_hi: u32) -> Option<(u32, u32)> {
    let read_bytes = match name.strip_prefix("s_buffer_load_dword")? {
        "" => 4u32,
        "x2" => 8,
        "x4" => 16,
        "x8" => 32,
        "x16" => 64,
        _ => return None,
    };
    let offset = u32::try_from(offset).ok()?;
    if !offset.is_multiple_of(4) || !scalar_hi.is_multiple_of(4) {
        return None;
    }
    let bytes = scalar_hi.checked_add(read_bytes)?;
    (offset.checked_add(bytes)? <= 65536).then_some((offset, bytes))
}

fn bounded_scalar_buffer_read(
    inst: &crate::ir::Inst,
    bounds: &mut crate::range::Ranges<'_>,
) -> Option<AccessProof> {
    let Fields::Smem { offset, soffset, .. } = inst.fields else { return None; };
    let scalar_hi = match soffset {
        Operand::Scalar { code: NULL, .. } => 0,
        Operand::Const(n @ 128..=192) => u32::from(n - 128),
        Operand::Scalar { code: code @ 0..=105, dwords: 1 } =>
            bounds.dword_offset_hi(inst.read(crate::ir::Reg::S(code))?)?,
        _ => return None,
    };
    // 宿主使用字节偏移逐个读取 dword；不允许负偏移、回绕和未知对齐。
    let (offset, bytes) = scalar_buffer_read_window(inst.name(), offset, scalar_hi)?;
    Some(AccessProof::ScalarBufferRead { offset, bytes })
}

fn bounded_typed_load(
    inst: &crate::ir::Inst, bounds: &mut crate::range::Ranges<'_>,
) -> Option<AccessProof> {
    let name = inst.name();
    if !name.starts_with("tbuffer_load_format_") {
        return None;
    }
    let Fields::Buffer { addr, offset, soffset, offen, idxen: false,
        format: Some(format), lds: false, tfe: false, .. } = &inst.fields else { return None; };
    let footprint = super::lower::typed_load_footprint(*format)?;
    let scalar = match *soffset {
        Operand::Scalar { code: NULL, .. } => 0,
        Operand::Const(n @ 128..=192) => u32::from(n - 128),
        _ => return None,
    };
    let vector_hi = if *offen {
        let Operand::Vector { reg, .. } = *addr else { return None; };
        // Low bits must be known, since packed elements cannot cross
        // the containing dword the host converter actually loads.
        bounds.dword_offset_hi(inst.read(crate::ir::Reg::V(reg))?)?
    } else {
        0
    };
    Some(AccessProof::TypedRead {
        immediate: u32::from(*offset).checked_add(scalar)?,
        vector_hi, footprint,
    })
}

fn bounded_structured_access(
    inst: &crate::ir::Inst, bounds: &mut crate::range::Ranges<'_>,
) -> Option<AccessProof> {
    let (op, size) = inst.name().strip_prefix("buffer_")?.split_once('_')?;
    let write = match op { "load" => false, "store" => true, _ => return None };
    let bytes = match size {
        "dword" => 4,
        "dwordx2" => 8,
        "dwordx3" => 12,
        "dwordx4" => 16,
        _ => return None,
    };
    let Fields::Buffer {
        addr: Operand::Vector { reg, .. }, offset, soffset,
        idxen: true, offen: false, format: None, lds: false, tfe: false, ..
    } = &inst.fields else { return None; };
    if !offset.is_multiple_of(4) || !matches!(soffset, Operand::Scalar { code: NULL, .. } | Operand::Const(128)) {
        return None;
    }
    let index = inst.read(Reg::V(*reg))?;
    let elements = bounds.unsigned_upper_bound(index)
        .and_then(|max_index| max_index.checked_add(1));
    // 不为动态写入推测范围。只读候选仍必须通过描述符类型和字段范围证明。
    if write && elements.is_none() { return None; }
    Some(AccessProof::StructuredAccess {
        elements, last_byte: u32::from(*offset).checked_add(bytes)?, write,
    })
}

fn table_slot(address: Address, first: u8) -> Result<u32, TableSlotFailure> {
    if address.dynamic.is_some() {
        return Err(TableSlotFailure::DynamicOffset);
    }
    let bytes = address.offset.checked_add(u32::from(first) * 4)
        .ok_or(TableSlotFailure::OffsetOverflow)?;
    if !bytes.is_multiple_of(32) {
        return Err(TableSlotFailure::Misaligned);
    }
    let slot = bytes / 32;
    // 只为闭合路径能精确填充的有限表区间建立证明。
    (slot < 256).then_some(slot).ok_or(TableSlotFailure::OutsideWindow)
}

fn proven_root_srv(
    native: &NativeProgram, address: Address, bits: u32, words: [u32; 2], proof: AccessProof,
) -> Option<StaticViewSlot> {
    if address.dynamic.is_some() || bits >> 30 != 0 || words[0] != u32::MAX {
        return None;
    }
    let bytes = match proof {
        AccessProof::Native(StaticViewAccess::RawUav { offset, bytes, write: false })
        | AccessProof::ScalarBufferRead { offset, bytes }
            if address.offset.is_multiple_of(4) =>
        {
            address.offset.checked_add(offset)?.checked_add(bytes)?
        }
        AccessProof::TypedRead { immediate, vector_hi, footprint } => {
            let start = address.offset.checked_add(immediate)?;
            let end = start.checked_add(vector_hi)?;
            let raw_start = if footprint.element_bits < 32 {
                if (start & 3) * 8 + footprint.element_bits > 32 {
                    return None;
                }
                end & !3
            } else {
                if !start.is_multiple_of(4) { return None; }
                end
            };
            raw_start.checked_add(footprint.dwords.checked_mul(4)?)?
        }
        _ => return None,
    };
    let root = native.layout.get(address.param)?;
    if root.root != RootType::Descriptor { return None; }
    // Match the translator's used-kind selection without assuming a later
    // matching declaration is the one the DXIL actually binds.
    let mut declarations = native.used.iter().filter(|u| u.user_data == root.user_data);
    let used = declarations.next()?;
    if declarations.next().is_some() || used.root != RootType::Descriptor
        || used.kind != ResourceType::Srv || used.slot != root.slot
    {
        return None;
    }
    (bytes <= 65536).then_some(StaticViewSlot {
        slot: StaticTextureSlot { root: root.slot, index: 0 },
        access: StaticViewAccess::RootSrv { bytes },
    })
}

fn proven_slot(
    native: &NativeProgram, address: Address, first: u8, proof: AccessProof,
) -> Result<StaticViewSlot, TableSlotFailure> {
    let param = native.layout.get(address.param).ok_or(TableSlotFailure::UnknownRoot)?;
    let index = table_slot(address, first)?;
    if param.root != RootType::Table {
        return Err(TableSlotFailure::RootKind(param.root));
    }
    // Match the translator's used-kind selection first; skipping a
    // mismatched declaration here can otherwise prove the wrong DXIL view.
    let relevant = native.used.iter().filter(|u| {
        u.root == RootType::Table && u.user_data == param.user_data
            && u.table_offset_dwords <= index * 8
    });
    let used = relevant.clone().max_by_key(|u| u.table_offset_dwords).ok_or(TableSlotFailure::NoDeclaration)?;
    if used.slot != param.slot {
        return Err(TableSlotFailure::RootSlotMismatch);
    }
    let unique = relevant.filter(|u| u.table_offset_dwords == used.table_offset_dwords).count() == 1;
    let (field_fits_stride, statically_bounded_index) = match &proof {
        AccessProof::StructuredAccess { elements, last_byte, .. } =>
            (Some(*last_byte <= used.stride), Some(elements.is_some())),
        _ => (None, None),
    };
    let reject = TableSlotFailure::Declaration {
        kind: used.kind, dwords: used.dwords, stride: used.stride, unique,
        field_fits_stride, statically_bounded_index,
    };
    let access = match proof {
        AccessProof::StructuredAccess { elements, last_byte, write }
            if unique && used.dwords == 4
                && (4..=2048).contains(&used.stride) && used.stride.is_multiple_of(4)
                && last_byte <= used.stride =>
        {
            match (used.kind, elements) {
                (ResourceType::Srv, Some(elements)) if !write && elements.checked_mul(used.stride).is_some_and(|bytes| bytes <= 65536) =>
                    StaticViewAccess::StructuredSrv { stride: used.stride, elements },
                // 只开放已证明内层字段不会跨界的顶点只读访问。堆中的完整 SRV
                // 使用真实元素数做越界检查，与不带长度的根描述符不同。
                (ResourceType::Srv, None) if !write && native.stage == HwStage::Vs =>
                    StaticViewAccess::StructuredSrvDescriptor { stride: used.stride },
                (ResourceType::Uav, Some(elements)) if native.stage == HwStage::Cs
                    && elements.checked_mul(used.stride).is_some_and(|bytes| bytes <= 65536) =>
                    StaticViewAccess::StructuredUav { stride: used.stride, elements, write },
                _ => return Err(reject),
            }
        }
        AccessProof::ScalarBufferRead { offset, bytes } if unique && used.dwords == 4 => {
            match used.kind {
                ResourceType::Cbv => StaticViewAccess::ConstantTable { offset, bytes },
                ResourceType::Srv if used.stride == 0 => StaticViewAccess::RawSrv { offset, bytes },
                _ => return Err(reject),
            }
        }
        AccessProof::Native(access @ StaticViewAccess::TextureWrite { .. })
            if native.stage == HwStage::Cs && unique && used.kind == ResourceType::Uav && used.dwords == 8 => access,
        AccessProof::Native(access @ StaticViewAccess::TextureRead { .. }) if used.kind == ResourceType::Srv && used.dwords == 8 => access,
        AccessProof::Native(StaticViewAccess::RawUav { offset, bytes, write: false })
            if used.kind == ResourceType::Srv && used.dwords == 4 && used.stride == 0 && unique =>
        {
            StaticViewAccess::RawSrv { offset, bytes }
        }
        AccessProof::Native(access @ StaticViewAccess::RawUav { .. })
            if used.kind == ResourceType::Uav && used.dwords == 4 && used.stride == 0 && unique => access,
        _ => return Err(reject),
    };
    Ok(StaticViewSlot { slot: StaticTextureSlot { root: param.slot, index }, access })
}

#[cfg(test)]
#[test]
fn page_texture_table_proof_requires_static_aligned_offsets() {
    let a = Address { param: 0, offset: 48, dynamic: None };
    assert_eq!(table_slot(a, 4), Ok(2));
    assert_eq!(table_slot(a, 0), Err(TableSlotFailure::Misaligned));
    assert_eq!(table_slot(Address { dynamic: Some(crate::ir::Value(0)), ..a }, 4), Err(TableSlotFailure::DynamicOffset));
    assert_eq!(table_slot(Address { offset: u32::MAX - 3, ..a }, 1), Err(TableSlotFailure::OffsetOverflow));
    assert_eq!(table_slot(Address { offset: 256 * 32, ..a }, 0), Err(TableSlotFailure::OutsideWindow));
}


#[cfg(test)]
#[test]
fn scalar_buffer_proof_bounds_all_four_dwords_and_rejects_wrap() {
    let name = "s_buffer_load_dwordx4";
    assert_eq!(scalar_buffer_read_window(name, 16, 0), Some((16, 16)));
    assert_eq!(scalar_buffer_read_window(name, 16, 12), Some((16, 28)));
    assert_eq!(scalar_buffer_read_window(name, 65520, 0), Some((65520, 16)));
    assert_eq!(scalar_buffer_read_window(name, 65524, 0), None);
    assert_eq!(scalar_buffer_read_window(name, -4, 16), None);
    assert_eq!(scalar_buffer_read_window(name, 16, 2), None);
    assert_eq!(scalar_buffer_read_window(name, 16, u32::MAX - 3), None);
    assert_eq!(scalar_buffer_read_window("s_buffer_store_dwordx4", 0, 0), None);
}
