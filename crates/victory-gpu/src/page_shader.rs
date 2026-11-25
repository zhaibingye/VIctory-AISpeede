//! Shader dependency admission for the closed page renderer. An access
//! with no lowering is rejected, not filled with the legacy root views.
use std::collections::{BTreeMap, BTreeSet};

use crate::host::{Host, HostList};
use crate::page_queue::PageDescriptors;
use crate::page_slabs::{BatchError, PinnedBatch};
use crate::descriptor::View;
use crate::page_texture::TextureInput;
use crate::page_uav::BoundedRawUav;
use crate::page_queue::TableDescriptor;
use victory_shader::emit::StaticViewAccess;
use victory_memory::{PAGE_BYTES, physical::PhysicalPage};
use crate::pipeline::host::HostPipeline;

use windows::Win32::Graphics::Direct3D12::*;

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
    stage_words(shader, used, block, values, |_| false, |_| false, |_| false)
}

pub(crate) fn stage_words(
    shader: &HostShader,
    used: &[UsedResource],
    block: Option<(u32, u32)>,
    values: impl Fn(u8, u32) -> Option<u32>,
    tables: impl Fn(u8) -> bool,
    uavs: impl Fn(u8) -> bool,
    cbvs: impl Fn(u8) -> bool,
) -> Result<Option<RootWords>, UnloweredShader> {
    if shader.memory || shader.memory_writes || shader.view_addresses {
        return Err(UnloweredShader::DynamicAddress);
    }
    if shader.heap || shader.heap_writes || shader.view_marks {
        return Err(UnloweredShader::DescriptorHeap);
    }
    if !shader.root_srvs.is_empty() || !shader.root_uavs.is_empty()
        || shader.root_cbvs.iter().any(|&slot| !cbvs(slot))
    {
        return Err(UnloweredShader::RootView);
    }
    for u in used {
        if u.root == RootType::Descriptor && u.kind == victory_shader::ResourceType::Cbv && cbvs(u.slot) {
            continue;
        }
        if u.root != RootType::Table || !tables(u.slot)
            || !(u.kind == victory_shader::ResourceType::Srv
                || u.kind == victory_shader::ResourceType::Uav && uavs(u.slot))
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
    tables: BTreeMap<u8, BTreeMap<u32, ClosedView>>,
    cbvs: Vec<(u32, ConstantSpan)>,
}

enum ClosedView {
    Texture(TextureInput),
    Raw(BoundedRawUav),
}

#[derive(Clone, Copy)]
struct ConstantSpan(u64);

impl ConstantSpan {
    /// Matches the size annotated on the translated root-CBV handle.
    const BYTES: u64 = 65536;

    fn capture(address: u64) -> Option<Self> {
        // The console root address is 16-byte aligned. D3D12 alignment
        // applies to the separate native buffer, not the gathered source.
        (address != 0 && address.is_multiple_of(16)
            && address.checked_add(Self::BYTES)? <= victory_memory::gpu_table::ADDRESS_LIMIT)
            .then_some(Self(address))
    }

    fn range(self) -> (u64, u64) {
        (self.0, self.0 + Self::BYTES)
    }
}

pub(crate) struct PreparedShader {
    words: Vec<(u32, Vec<u32>)>,
    cbvs: Vec<(u32, u64)>,
}

impl PreparedShader {
    pub(crate) fn bind(&self, cmd: &ID3D12GraphicsCommandList, compute: bool) {
        // SAFETY: the caller binds the matching prevalidated root
        // signature; every address points to retained read-only memory.
        unsafe {
            for (param, words) in &self.words {
                if compute {
                    cmd.SetComputeRoot32BitConstants(*param, words.len() as u32, words.as_ptr().cast(), 0);
                } else {
                    cmd.SetGraphicsRoot32BitConstants(*param, words.len() as u32, words.as_ptr().cast(), 0);
                }
            }
            for &(param, gpu) in &self.cbvs {
                if compute {
                    cmd.SetComputeRootConstantBufferView(param, gpu);
                } else {
                    cmd.SetGraphicsRootConstantBufferView(param, gpu);
                }
            }
        }
    }
}

impl ShaderBindings {
    pub(crate) fn capture(
        pipeline: &HostPipeline,
        values: impl Fn(u8, u32) -> Option<u32>,
        view: impl Fn(u8, u32) -> Option<View>,
        address: impl Fn(u8) -> Option<u64>,
    ) -> Result<Self, UnloweredShader> {
        if pipeline.marks.is_some() {
            return Err(UnloweredShader::DescriptorHeap);
        }
        let mut constants = Vec::new();
        let mut cbvs = Vec::new();
        let mut tables = BTreeMap::<u8, BTreeMap<u32, _>>::new();
        for stage in &pipeline.stages {
            let t = stage.translated();
            for &(slot, param) in &stage.root_cbvs {
                let span = address(slot).and_then(ConstantSpan::capture)
                    .ok_or(UnloweredShader::RootView)?;
                cbvs.push((param, span));
            }
            let needs_table = t.host.user_data.iter().any(|u| u.root == victory_shader::RootType::Table);
            let slots = if needs_table {
                stage.shader.static_view_slots().ok_or(UnloweredShader::DescriptorTable)?
            } else {
                &[]
            };
            if let Some(words) = stage_words(
                &t.host, &t.used, stage.user_data, &values,
                |root| slots.iter().any(|s| s.slot.root == root),
                |root| pipeline.compute && slots.iter().any(|s| {
                    s.slot.root == root && matches!(s.access, StaticViewAccess::RawWrite { .. })
                }),
                |slot| stage.root_cbvs.iter().any(|&(bound, _)| bound == slot),
            )? {
                for proof in slots {
                    let slot = proof.slot;
                    let reject = UnloweredShader::DescriptorTable;
                    if !words.tables.iter().any(|&(_, root)| root == slot.root) {
                        return Err(reject);
                    }
                    let table = tables.entry(slot.root).or_default();
                    if let std::collections::btree_map::Entry::Vacant(e) = table.entry(slot.index) {
                        let actual = view(slot.root, slot.index).ok_or(UnloweredShader::DescriptorTable)?;
                        let bound = match proof.access {
                            StaticViewAccess::TextureRead => ClosedView::Texture(
                                actual.paged_texture_srv().ok_or(UnloweredShader::DescriptorTable)?),
                            StaticViewAccess::RawWrite { .. } if pipeline.compute => {
                                let (start, end) = crate::descriptor::host::paged_raw_range(&actual)
                                    .ok_or(UnloweredShader::DescriptorTable)?;
                                ClosedView::Raw(BoundedRawUav::new(start, end)
                                    .ok_or(UnloweredShader::DescriptorTable)?)
                            }
                            _ => return Err(reject),
                        };
                        e.insert(bound);
                    }
                    match (table.get_mut(&slot.index), proof.access) {
                        (Some(ClosedView::Texture(_)), StaticViewAccess::TextureRead) => {},
                        (Some(ClosedView::Raw(raw)), StaticViewAccess::RawWrite { offset, bytes }) => {
                            raw.allow_write(offset, bytes).ok_or(UnloweredShader::DescriptorTable)?;
                        }
                        _ => return Err(reject),
                    }
                }
                constants.push(words);
            } else if !slots.is_empty() {
                return Err(UnloweredShader::DescriptorTable);
            }
        }
        Ok(Self { constants, tables, cbvs })
    }

    pub(crate) fn ranges(&self, ranges: &mut Vec<(u64, u64)>) -> Result<(), BatchError> {
        ranges.extend(self.cbvs.iter().map(|(_, span)| span.range()));
        for table in self.tables.values() {
            for view in table.values() {
                match view {
                    ClosedView::Texture(texture) => texture.ranges(ranges)?,
                    ClosedView::Raw(raw) => ranges.push(raw.range()),
                }
            }
        }
        Ok(())
    }

    pub(crate) fn samples_target(&self, target: &crate::resource::HostTexture) -> bool {
        self.tables.values().flat_map(|table| table.values())
            .any(|view| matches!(view, ClosedView::Texture(tex) if tex.aliases_target(target)))
    }

    pub(crate) fn writes_raw(&self) -> bool {
        self.tables.values().flat_map(|table| table.values())
            .any(|view| matches!(view, ClosedView::Raw(_)))
    }

    pub(crate) fn prepare(
        &self, host: &Host, list: &HostList, batch: &PinnedBatch,
        descriptors: &mut PageDescriptors<'_>,
        plan: &mut crate::page_resource::ResourcePlan,
        dirty: &mut BTreeSet<PhysicalPage>,
    ) -> Result<PreparedShader, BatchError> {
        let mut bases = BTreeMap::new();
        let mut shader_dirty = BTreeSet::new();
        for (&root, table) in &self.tables {
            let size = table.last_key_value().expect("captured slots exist").0 + 1;
            let mut views: Vec<_> = (0..size).map(|_| None).collect();
            for (&index, view) in table {
                views[index as usize] = Some(match view {
                    ClosedView::Texture(texture) => {
                        let (resource, format) = texture.encode(host, list, batch, plan)?;
                        TableDescriptor::Texture(resource, format)
                    }
                    ClosedView::Raw(raw) => raw.prepare(batch, &mut shader_dirty)?,
                });
            }
            let first = descriptors.reserve_table_run(&views).map_err(BatchError::Native)?;
            bases.insert(root, first);
        }
        if !shader_dirty.is_empty() {
            // Staged readers are snapshots. A shader reading and writing
            // the same physical pages needs a single directly shared view.
            let mut reads: Vec<_> = self.cbvs.iter().map(|(_, span)| span.range()).collect();
            for view in self.tables.values().flat_map(|table| table.values()) {
                if let ClosedView::Texture(tex) = view {
                    tex.ranges(&mut reads)?;
                }
            }
            for (start, end) in reads {
                let mut at = start & !(PAGE_BYTES - 1);
                while at < end {
                    let page = batch.pinned_at(at).ok_or(BatchError::NotCaptured(at))?;
                    if shader_dirty.contains(&page.physical()) {
                        return Err(BatchError::Range(start, end));
                    }
                    at += PAGE_BYTES;
                }
            }
            dirty.extend(shader_dirty);
        }
        let words: Vec<_> = self.constants.iter().map(|root| {
            root.relocated(&bases).map(|words| (root.parameter, words))
                .ok_or_else(|| BatchError::Native(windows::Win32::Foundation::E_FAIL.into()))
        }).collect::<Result<_, _>>()?;
        let mut gathered = BTreeMap::new();
        let mut cbvs = Vec::with_capacity(self.cbvs.len());
        for &(param, span) in &self.cbvs {
            let gpu = match gathered.entry(span.0) {
                std::collections::btree_map::Entry::Occupied(e) => *e.get(),
                std::collections::btree_map::Entry::Vacant(e) => {
                    let buffer = host.default_buffer(ConstantSpan::BYTES).ok_or_else(|| {
                        BatchError::Native(windows::Win32::Foundation::E_OUTOFMEMORY.into())
                    })?;
                    crate::page_view::gather_span(list, batch, span.range(), &buffer,
                        D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER)?;
                    // SAFETY: gather retains the resource. Its native base,
                    // bound at offset zero, has D3D12 buffer alignment.
                    *e.insert(unsafe { buffer.GetGPUVirtualAddress() })
                }
            };
            cbvs.push((param, gpu));
        }
        Ok(PreparedShader { words, cbvs })
    }
}
