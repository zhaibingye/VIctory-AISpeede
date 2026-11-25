//! A conservative proof of the exact texture slots a title shader reads.
//! UsedResource records describe ranges, not an upper bound on runtime
//! indices. Trace the actual descriptor operands instead of trusting them.
use std::collections::{BTreeMap, BTreeSet};

use crate::gfx10::Fields;
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

/// Prove the descriptor operand footprint for one native shader. A
/// successful proof of descriptors alone does not admit the program:
/// raw memory, root views and writes must be checked by the renderer too.
pub fn static_texture_slots(container: &[u8]) -> Result<Option<Vec<StaticTextureSlot>>, ShaderError> {
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
        if !matches!(inst.fields, Fields::Image { dim: 1, .. })
            || !(name.starts_with("image_sample") || matches!(name, "image_load" | "image_load_mip"))
        {
            return Ok(None);
        }
        let Source::Loaded { load, first } = descriptor.source else {
            return Ok(None);
        };
        if resources.waterfall.loads.contains(&load) {
            return Ok(None);
        }
        let Some(slot) = addresses.get(&load).and_then(|a| proven_slot(&native, *a, first)) else {
            return Ok(None);
        };
        slots.insert(slot);
    }
    Ok(Some(slots.into_iter().collect()))
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

fn proven_slot(native: &NativeProgram, address: Address, first: u8) -> Option<StaticTextureSlot> {
    let param = native.layout.get(address.param)?;
    let index = table_slot(address, first)?;
    if param.root != RootType::Table {
        return None;
    }
    let used = native.used.iter().filter(|u| {
        u.root == RootType::Table && u.user_data == param.user_data
            && u.slot == param.slot && u.table_offset_dwords <= index * 8
    }).max_by_key(|u| u.table_offset_dwords)?;
    (used.kind == ResourceType::Srv && used.dwords == 8)
        .then_some(StaticTextureSlot { root: param.slot, index })
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
