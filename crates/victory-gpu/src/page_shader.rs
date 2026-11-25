//! Shader dependency admission for the closed page renderer. An access
//! with no lowering is rejected, not filled with the legacy root views.
use std::collections::BTreeMap;

use crate::host::{Host, HostList};
use crate::page_queue::PageDescriptors;
use crate::page_slabs::{BatchError, PinnedBatch};
use crate::page_texture::LinearTexture;
use crate::pipeline::host::HostPipeline;

use victory_shader::emit::{HostShader, UserData};
use victory_shader::{RootType, UsedResource};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UnloweredShader {
    GraphicsState,
    DynamicAddress,
    DescriptorHeap,
    RootView,
    DescriptorTable,
    UnknownUserData,
    MissingConstant { slot: u8, dword: u32 },
    ConstantLayout,
}

pub(crate) struct RootWords {
    pub(crate) parameter: u32,
    pub(crate) words: Vec<u32>,
    pub(crate) tables: Vec<(usize, u8)>,
}

pub(crate) fn constants_only(
    shader: &HostShader,
    used: &[UsedResource],
    block: Option<(u32, u32)>,
    values: impl Fn(u8, u32) -> Option<u32>,
) -> Result<Option<RootWords>, UnloweredShader> {
    stage_words(shader, used, block, values, |_| false)
}

pub(crate) fn stage_words(
    shader: &HostShader,
    used: &[UsedResource],
    block: Option<(u32, u32)>,
    values: impl Fn(u8, u32) -> Option<u32>,
    tables: impl Fn(u8) -> bool,
) -> Result<Option<RootWords>, UnloweredShader> {
    if shader.memory || shader.memory_writes || shader.view_addresses {
        return Err(UnloweredShader::DynamicAddress);
    }
    if shader.heap || shader.heap_writes || shader.view_marks {
        return Err(UnloweredShader::DescriptorHeap);
    }
    if !shader.root_cbvs.is_empty() || !shader.root_srvs.is_empty() || !shader.root_uavs.is_empty()
        || used.iter().any(|u| u.root == RootType::Descriptor)
    {
        return Err(UnloweredShader::RootView);
    }
    for u in used {
        if u.root != RootType::Table || u.kind != victory_shader::ResourceType::Srv
            || !tables(u.slot)
        {
            return Err(UnloweredShader::DescriptorTable);
        }
    }
    pack_words(&shader.user_data, shader.user_data_dwords(), block, values, tables)
}

fn pack_constants(
    metadata: &[UserData],
    size: u32,
    block: Option<(u32, u32)>,
    values: impl Fn(u8, u32) -> Option<u32>,
) -> Result<Option<RootWords>, UnloweredShader> {
    pack_words(metadata, size, block, values, |_| false)
}

fn pack_words(
    metadata: &[UserData],
    size: u32,
    block: Option<(u32, u32)>,
    values: impl Fn(u8, u32) -> Option<u32>,
    supported_table: impl Fn(u8) -> bool,
) -> Result<Option<RootWords>, UnloweredShader> {
    if metadata.is_empty() && block.is_none() {
        return Ok(None);
    }
    let Some((parameter, count)) = block else {
        return Err(UnloweredShader::ConstantLayout);
    };
    if count != size || !(1..=64).contains(&count) {
        return Err(UnloweredShader::ConstantLayout);
    }
    let mut words = vec![0; count as usize];
    let mut occupied = 0u64;
    let mut tables = Vec::new();
    for u in metadata {
        let w = words.get_mut(u.index as usize).ok_or(UnloweredShader::ConstantLayout)?;
        let mask = 1u64 << u.index;
        if occupied & mask != 0 {
            return Err(UnloweredShader::ConstantLayout);
        }
        occupied |= mask;
        match u.root {
            RootType::Constants => {
                *w = values(u.slot, u.dword)
                    .ok_or(UnloweredShader::MissingConstant { slot: u.slot, dword: u.dword })?;
            }
            RootType::Table if u.dword == 0 && supported_table(u.slot) => {
                tables.push((u.index as usize, u.slot));
            }
            _ => return Err(UnloweredShader::UnknownUserData),
        }
    }
    Ok(Some(RootWords { parameter, words, tables }))
}

#[cfg(test)]
#[test]
fn closed_draw_constant_admission_never_invents_shader_values() {
    let u = UserData { index: 2, slot: 1, root: RootType::Constants, dword: 3 };
    let result = pack_constants(&[u], 4, Some((5, 4)), |slot, dword| {
        (slot == 1 && dword == 3).then_some(42)
    }).unwrap().unwrap();
    assert_eq!((result.parameter, result.words), (5, vec![0, 0, 42, 0]));
    assert!(matches!(
        pack_constants(&[u], 4, Some((5, 4)), |_, _| None),
        Err(UnloweredShader::MissingConstant { slot: 1, dword: 3 })
    ));
    assert!(matches!(
        pack_constants(&[UserData { root: RootType::Table, ..u }], 4, Some((5, 4)), |_, _| Some(0)),
        Err(UnloweredShader::UnknownUserData)
    ));
    assert!(matches!(
        pack_constants(&[u, u], 4, Some((5, 4)), |_, _| Some(0)),
        Err(UnloweredShader::ConstantLayout)
    ));
}

#[cfg(test)]
#[test]
fn closed_draw_rejects_unlowered_shader_accesses() {
    let shader = HostShader {
        stage: victory_shader::HwStage::Ps,
        dxil: Vec::new(),
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
    };
    assert!(matches!(constants_only(&shader, &[], None, |_, _| None), Ok(None)));
    for (variant, expected) in [
        (HostShader { memory: true, ..shader.clone() }, UnloweredShader::DynamicAddress),
        (HostShader { memory_writes: true, ..shader.clone() }, UnloweredShader::DynamicAddress),
        (HostShader { view_addresses: true, ..shader.clone() }, UnloweredShader::DynamicAddress),
        (HostShader { heap: true, ..shader.clone() }, UnloweredShader::DescriptorHeap),
        (HostShader { view_marks: true, ..shader.clone() }, UnloweredShader::DescriptorHeap),
        (HostShader { root_uavs: vec![3], ..shader.clone() }, UnloweredShader::RootView),
    ] {
        assert_eq!(constants_only(&variant, &[], None, |_, _| None).err(), Some(expected));
    }
    let used = UsedResource {
        kind: victory_shader::ResourceType::Srv, root: RootType::Table,
        table_offset_dwords: 0, lower_bound: 0, stride: 8, dwords: 8, user_data: 0, slot: 0,
    };
    assert_eq!(
        constants_only(&shader, &[used], None, |_, _| None).err(),
        Some(UnloweredShader::DescriptorTable),
    );
}


impl RootWords {
    pub(crate) fn relocated(&self, bases: &std::collections::BTreeMap<u8, u32>) -> Option<Vec<u32>> {
        let mut words = self.words.clone();
        for &(word, slot) in &self.tables {
            words[word] = bases.get(&slot)?.checked_mul(32)?;
        }
        Some(words)
    }
}

#[cfg(test)]
#[test]
fn closed_draw_relocates_table_byte_offsets_without_changing_constants() {
    let table = UserData { index: 0, slot: 4, root: RootType::Table, dword: 0 };
    let constant = UserData { index: 2, slot: 1, root: RootType::Constants, dword: 3 };
    let words = pack_words(&[table, constant], 4, Some((5, 4)), |_, _| Some(0x700), |s| s == 4)
        .unwrap().unwrap();
    assert_eq!(words.relocated(&[(4, 32)].into()), Some(vec![1024, 0, 0x700, 0]));
    assert_eq!(words.relocated(&std::collections::BTreeMap::new()), None);
}


/// One validated shader binding snapshot for either graphics or compute.
#[derive(Default)]
pub(crate) struct ShaderBindings {
    pub(crate) constants: Vec<RootWords>,
    pub(crate) texture_tables: BTreeMap<u8, BTreeMap<u32, LinearTexture>>,
}

impl ShaderBindings {
    pub(crate) fn capture(
        pipeline: &HostPipeline,
        values: impl Fn(u8, u32) -> Option<u32>,
        texture: impl Fn(u8, u32) -> Option<LinearTexture>,
    ) -> Result<Self, UnloweredShader> {
        if pipeline.marks.is_some() {
            return Err(UnloweredShader::DescriptorHeap);
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
            if let Some(words) = stage_words(
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
        Ok(Self { constants, texture_tables })
    }

    pub(crate) fn ranges(&self, ranges: &mut Vec<(u64, u64)>) -> Result<(), BatchError> {
        for table in self.texture_tables.values() {
            for texture in table.values() {
                ranges.extend(texture.rows()?);
            }
        }
        Ok(())
    }

    pub(crate) fn prepare(
        &self, host: &Host, list: &HostList, batch: &PinnedBatch,
        descriptors: &mut PageDescriptors<'_>,
    ) -> Result<Vec<(u32, Vec<u32>)>, BatchError> {
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
        Ok(words)
    }
}
