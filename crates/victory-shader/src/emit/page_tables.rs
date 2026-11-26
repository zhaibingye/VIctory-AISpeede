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
    /// An immediate aligned byte window on a raw UAV, as used by DXIL.
    RawUav { offset: u32, bytes: u32, write: bool },
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
    let resources = crate::resource::track(&p, &native.layout, native.stage, Some(&st?))?;
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
        let access = if matches!(inst.fields, Fields::Image { dim: 1, .. })
            && (name.starts_with("image_sample") || matches!(name, "image_load" | "image_load_mip"))
        {
            StaticViewAccess::TextureRead
        } else if let Some(access) = immediate_raw_access(name, &inst.fields) {
            access
        } else {
            return Ok(None);
        };
        let Source::Loaded { load, first } = descriptor.source else {
            return Ok(None);
        };
        if resources.waterfall.loads.contains(&load) {
            return Ok(None);
        }
        let Some(slot) = addresses.get(&load).and_then(|a| proven_slot(&native, *a, first, access)) else {
            return Ok(None);
        };
        slots.insert(StaticViewSlot { slot, access });
    }
    Ok(Some(slots.into_iter().collect()))
}

fn immediate_raw_access(name: &str, fields: &Fields) -> Option<StaticViewAccess> {
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
    let Fields::Buffer {
        offset, soffset, offen: false, idxen: false, format: None, lds: false, tfe: false, ..
    } = fields else { return None };
    if !matches!(soffset, Operand::Scalar { code: NULL, .. } | Operand::Const(128))
        || !offset.is_multiple_of(4)
    {
        return None;
    }
    Some(StaticViewAccess::RawUav { offset: u32::from(*offset), bytes, write })
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

fn proven_slot(
    native: &NativeProgram, address: Address, first: u8, access: StaticViewAccess,
) -> Option<StaticTextureSlot> {
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
    let admitted = match access {
        StaticViewAccess::TextureRead => used.kind == ResourceType::Srv && used.dwords == 8,
        StaticViewAccess::RawUav { .. } => {
            used.kind == ResourceType::Uav && used.dwords == 4 && used.stride == 0
                && relevant.filter(|u| u.table_offset_dwords == used.table_offset_dwords).count() == 1
        }
    };
    admitted.then_some(StaticTextureSlot { root: param.slot, index })
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
