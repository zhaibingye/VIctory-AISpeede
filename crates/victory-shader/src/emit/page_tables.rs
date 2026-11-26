//! A conservative proof of exact descriptor slots and admitted accesses.
//! UsedResource records describe ranges, not an upper bound on runtime
//! indices. Trace the actual descriptor operands instead of trusting them.
use std::collections::{BTreeMap, BTreeSet};

use crate::gfx10::{Fields, NULL, Operand};
use crate::native::{HwStage, NativeProgram, ResourceType, RootType};
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
    TextureRead,
    TextureWrite { x: u32, y: u32 },
    /// A statically bounded aligned byte window on a raw UAV, as used by DXIL.
    RawUav { offset: u32, bytes: u32, write: bool },
    RawSrv { offset: u32, bytes: u32 },
    RootSrv { bytes: u32 },
    StructuredSrv { stride: u32, elements: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct StaticViewSlot {
    pub slot: StaticTextureSlot,
    pub access: StaticViewAccess,
}

pub fn static_texture_slots(container: &[u8]) -> Result<Option<Vec<StaticTextureSlot>>, ShaderError> {
    Ok(static_view_slots(container)?.and_then(|views| {
        views.into_iter().map(|view| {
            (view.access == StaticViewAccess::TextureRead).then_some(view.slot)
        }).collect()
    }))
}

/// Prove descriptor operands and raw access byte windows from native
/// instructions, not inferred from resource declarations. Excludes
/// arbitrary addresses and does not by itself admit a host shader.
pub fn static_view_slots(container: &[u8]) -> Result<Option<Vec<StaticViewSlot>>, ShaderError> {
    let c = Container::parse(container)?;
    let dxil = c.part(FourCC::DXIL).or(c.part(FourCC::ACCC))
        .ok_or(ShaderError::Malformed("no DXIL part"))?;
    let Some(native) = crate::native_programs(dxil.data)?.into_iter()
        .find(|p| matches!(p.stage, HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms))
    else {
        // The translated host blob cannot stand in for a guest proof.
        return Ok(None);
    };
    let (p, st) = crate::range::build(&native.patched_code(), native.wave64, native.vgprs)?;
    let st = st?;
    let resources = crate::resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let mut bounds = crate::range::Ranges::new(&p, &st);
    let addresses: BTreeMap<_, _> = resources.scalar.iter().copied().collect();
    let mut slots = BTreeSet::new();
    for descriptor in &resources.descriptors {
        let (b, i) = descriptor.site;
        let inst = &p.blocks[b].insts[i];
        if descriptor.role == Role::Sampler {
            // Immediate samplers use the host PSO's static samplers;
            // do not admit a second, dynamically indexed sampler heap.
            if !matches!(descriptor.source, Source::Constant(_)) {
                return Ok(None);
            }
            continue;
        }
        let name = inst.name();
        let access = if let Some(access) = bounded_texture_write(inst, &mut bounds) {
            AccessProof::Native(access)
        } else if matches!(inst.fields, Fields::Image { dim: 1, .. })
            && (name.starts_with("image_sample") || matches!(name, "image_load" | "image_load_mip"))
        {
            AccessProof::Native(StaticViewAccess::TextureRead)
        } else if let Some(access) = bounded_structured_read(inst, &mut bounds) {
            access
        } else if let Some(access) = bounded_raw_access(inst, &mut bounds) {
            AccessProof::Native(access)
        } else {
            return Ok(None);
        };
        if let Source::Built { address, bits, words } = descriptor.source {
            let Some(root) = proven_root_srv(&native, address, bits, words, access) else {
                return Ok(None);
            };
            slots.insert(root);
            continue;
        }
        let Source::Loaded { load, first } = descriptor.source else {
            return Ok(None);
        };
        if resources.waterfall.loads.contains(&load) {
            return Ok(None);
        }
        let Some(view) = addresses.get(&load).and_then(|a| proven_slot(&native, *a, first, access)) else {
            return Ok(None);
        };
        slots.insert(view);
    }
    Ok(Some(slots.into_iter().collect()))
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
    inst: &crate::ir::Inst, bounds: &mut crate::range::Ranges<'_>,
) -> Option<StaticViewAccess> {
    let Fields::Image {
        addr, dim: 1, dmask: 15, a16: false, d16: false, tfe: false, lwe: false, ..
    } = &inst.fields else { return None; };
    if inst.name() != "image_store" { return None; }
    let mut regs = addr.iter().flat_map(|operand| match *operand {
        Operand::Vector { reg, dwords } => reg..reg.saturating_add(dwords),
        _ => 0..0,
    });
    let x = bounds.unsigned_upper_bound(inst.read(crate::ir::Reg::V(regs.next()?))?)?;
    let y = bounds.unsigned_upper_bound(inst.read(crate::ir::Reg::V(regs.next()?))?)?;
    Some(StaticViewAccess::TextureWrite { x, y })
}

enum AccessProof {
    Native(StaticViewAccess),
    StructuredRead { elements: u32, last_byte: u32 },
}

fn bounded_structured_read(
    inst: &crate::ir::Inst, bounds: &mut crate::range::Ranges<'_>,
) -> Option<AccessProof> {
    let bytes = match inst.name() {
        "buffer_load_dword" => 4,
        "buffer_load_dwordx2" => 8,
        "buffer_load_dwordx3" => 12,
        "buffer_load_dwordx4" => 16,
        _ => return None,
    };
    let Fields::Buffer {
        addr: Operand::Vector { reg, .. }, offset, soffset,
        idxen: true, offen: false, format: None, lds: false, tfe: false, ..
    } = &inst.fields else { return None; };
    if !offset.is_multiple_of(4) || !matches!(soffset, Operand::Scalar { code: NULL, .. } | Operand::Const(128)) {
        return None;
    }
    let max_index = bounds.unsigned_upper_bound(inst.read(crate::ir::Reg::V(*reg))?)?;
    Some(AccessProof::StructuredRead {
        elements: max_index.checked_add(1)?,
        last_byte: u32::from(*offset).checked_add(bytes)?,
    })
}

fn table_slot(address: Address, first: u8) -> Option<u32> {
    if address.dynamic.is_some() {
        return None;
    }
    let bytes = address.offset.checked_add(u32::from(first) * 4)?;
    if !bytes.is_multiple_of(32) {
        return None;
    }
    let slot = bytes / 32;
    // A proof for explicitly bounded small table runs. A later encoder
    // must populate exactly these slots, not read outside captured views.
    (slot < 256).then_some(slot)
}

fn proven_root_srv(
    native: &NativeProgram, address: Address, bits: u32, words: [u32; 2], proof: AccessProof,
) -> Option<StaticViewSlot> {
    let AccessProof::Native(StaticViewAccess::RawUav { offset, bytes, write: false }) = proof else {
        return None;
    };
    if address.dynamic.is_some() || bits >> 30 != 0 || words[0] != u32::MAX
        || !address.offset.is_multiple_of(4)
    {
        return None;
    }
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
    let bytes = address.offset.checked_add(offset)?.checked_add(bytes)?;
    (bytes <= 65536).then_some(StaticViewSlot {
        slot: StaticTextureSlot { root: root.slot, index: 0 },
        access: StaticViewAccess::RootSrv { bytes },
    })
}

fn proven_slot(
    native: &NativeProgram, address: Address, first: u8, proof: AccessProof,
) -> Option<StaticViewSlot> {
    let param = native.layout.get(address.param)?;
    let index = table_slot(address, first)?;
    if param.root != RootType::Table {
        return None;
    }
    // Match the translator's used-kind selection first; skipping a
    // mismatched declaration here can otherwise prove the wrong DXIL view.
    let relevant = native.used.iter().filter(|u| {
        u.root == RootType::Table && u.user_data == param.user_data
            && u.table_offset_dwords <= index * 8
    });
    let used = relevant.clone().max_by_key(|u| u.table_offset_dwords)?;
    if used.slot != param.slot {
        return None;
    }
    let unique = || relevant.clone()
        .filter(|u| u.table_offset_dwords == used.table_offset_dwords).count() == 1;
    let access = match proof {
        AccessProof::StructuredRead { elements, last_byte }
            if unique() && used.kind == ResourceType::Srv && used.dwords == 4
                && (4..=2048).contains(&used.stride) && used.stride.is_multiple_of(4)
                && last_byte <= used.stride && elements.checked_mul(used.stride)? <= 65536 =>
        {
            StaticViewAccess::StructuredSrv { stride: used.stride, elements }
        }
        AccessProof::Native(access @ StaticViewAccess::TextureWrite { .. })
            if native.stage == HwStage::Cs && unique() && used.kind == ResourceType::Uav && used.dwords == 8 => access,
        AccessProof::Native(access @ StaticViewAccess::TextureRead) if used.kind == ResourceType::Srv && used.dwords == 8 => access,
        AccessProof::Native(StaticViewAccess::RawUav { offset, bytes, write: false })
            if used.kind == ResourceType::Srv && used.dwords == 4 && used.stride == 0 && unique() =>
        {
            StaticViewAccess::RawSrv { offset, bytes }
        }
        AccessProof::Native(access @ StaticViewAccess::RawUav { .. })
            if used.kind == ResourceType::Uav && used.dwords == 4 && used.stride == 0 && unique() => access,
        _ => return None,
    };
    Some(StaticViewSlot { slot: StaticTextureSlot { root: param.slot, index }, access })
}

#[cfg(test)]
#[test]
fn page_texture_table_proof_requires_static_aligned_offsets() {
    let a = Address { param: 0, offset: 48, dynamic: None };
    assert_eq!(table_slot(a, 4), Some(2));
    assert_eq!(table_slot(a, 0), None);
    assert_eq!(table_slot(Address { dynamic: Some(crate::ir::Value(0)), ..a }, 4), None);
    assert_eq!(table_slot(Address { offset: u32::MAX - 3, ..a }, 1), None);
    assert_eq!(table_slot(Address { offset: 256 * 32, ..a }, 0), None);
}
