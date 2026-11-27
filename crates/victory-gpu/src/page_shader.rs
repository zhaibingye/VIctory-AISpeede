//! Shader dependency admission for the closed page renderer. An access
//! with no lowering is rejected, not filled with the legacy root views.
use std::collections::{BTreeMap, BTreeSet};

use crate::host::{Host, HostList};
use crate::page_queue::PageDescriptors;
use crate::page_slabs::{BatchError, PinnedBatch};
use crate::descriptor::View;
use crate::page_texture::TextureInput;
use crate::page_uav::BoundedBufferView;
use crate::page_queue::TableDescriptor;
use victory_shader::emit::StaticViewAccess;
use victory_memory::physical::PhysicalPage;
use crate::pipeline::host::HostPipeline;

use windows::Win32::Graphics::Direct3D12::*;

use victory_shader::emit::{HostShader, UserData};
use victory_shader::{RootType, UsedResource};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UnloweredShader {
    GraphicsState,
    DrawState(DrawState),
    Pipeline(crate::pipeline::RequiredPipeline),
    DynamicAddress,
    DescriptorHeap,
    RootView,
    DescriptorTable,
    DescriptorProof { stage: &'static str, reason: victory_shader::emit::StaticViewFailure },
    DescriptorDeclaration { root: u8, kind: victory_shader::ResourceType, source: RootType },
    DescriptorSlot { stage: &'static str, root: u8, index: u32, reason: DescriptorSlotReason },
    UnknownUserData,
    MissingConstant { slot: u8, dword: u32 },
    ConstantLayout,
}

/// 区分绑定、原生视图和访问范围问题，不输出实际地址或资源字节。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DescriptorSlotReason {
    RootWord,
    UnboundOrStale,
    UnsupportedTexture {
        array: bool, volume: bool, kind: &'static str, title_format: Option<u32>,
        view_format: Option<u32>, view_dimension: Option<u32>,
        texture_signature: Option<crate::resource::TextureSrvShape>,
        phase: crate::descriptor::TextureSrvReject,
    },
    UnsupportedBuffer,
    ScalarBufferView {
        expected_stride: u32,
        actual: crate::descriptor::host::BufferViewShape,
    },
    UnsupportedTextureWrite {
        title_format: Option<u32>, view_format: Option<i32>, view_dimension: Option<i32>,
        phase: crate::page_uav::TextureUavReject,
    },
    TextureWriteRange {
        host_format: i32, xyz_max: (u32, u32, Option<u32>),
        view_width_height_w_window: (u64, u32, Option<(u32, u32)>),
        coordinate_sources: [&'static str; 3], exec_guard: bool,
        groups_and_threads: Option<([u32; 3], [u32; 3])>,
    },
    UnsupportedAccess,
    ConstantTableDescriptorBudget,
    OutsideRangeOrConflictingView,
}

/// 无法闭合的绘制状态。字段为稳定诊断键，不输出资源地址。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DrawState {
    Opaque,
    Topology(u32),
    RectanglePipeline { topology: u32, geometry_shader: bool },
    TargetsNotCaptured,
    ViewportNotCaptured,
    ScissorNotCaptured,
    VertexInput { slot: u32 },
    IndexInput,
    PipelineAttachments,
    AttachmentCount { expected: usize, actual: usize },
    NoAttachments,
    DepthFormat { expected: Option<i32>, actual: Option<i32> },
    SampleCount { expected: u32, actual: u32 },
    DepthPlaneConflict,
    ColorFormat { slot: usize, expected: i32, actual: i32 },
    AttachmentAlias { slot: usize },
    AttachmentBackingOrExtent { slot: usize },
}

pub(crate) struct RootWords {
    pub(crate) parameter: u32,
    pub(crate) words: Vec<u32>,
    pub(crate) tables: Vec<(usize, u8)>,
    root_srvs: Vec<(usize, u8)>,
}

pub(crate) fn constants_only(
    shader: &HostShader,
    used: &[UsedResource],
    block: Option<(u32, u32)>,
    values: impl Fn(u8, u32) -> Option<u32>,
) -> Result<Option<RootWords>, UnloweredShader> {
    stage_words(shader, used, block, values, |_| false, |_| false, |_| false, |_| false, |_| false)
}

pub(crate) fn stage_words(
    shader: &HostShader,
    used: &[UsedResource],
    block: Option<(u32, u32)>,
    values: impl Fn(u8, u32) -> Option<u32>,
    tables: impl Fn(u8) -> bool,
    uavs: impl Fn(u8) -> bool,
    table_cbvs: impl Fn(u8) -> bool,
    cbvs: impl Fn(u8) -> bool,
    root_srvs: impl Fn(u8) -> bool,
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
        // 立即采样器已由该着色器生成静态绑定，并由宿主 PSO 写入根签名，
        // 不占用运行时描述符表。动态采样器表仍不属于此闭合路径。
        if u.root == RootType::ImmediateSampler
            && u.kind == victory_shader::ResourceType::Sampler
            && !shader.static_samplers.is_empty()
            && shader.static_samplers.len() == shader.samplers.len()
        {
            continue;
        }
        if u.root == RootType::Descriptor
            && (u.kind == victory_shader::ResourceType::Cbv && cbvs(u.slot)
                || u.kind == victory_shader::ResourceType::Srv && root_srvs(u.slot))
        {
            continue;
        }
        if u.root != RootType::Table || !tables(u.slot)
            || !(u.kind == victory_shader::ResourceType::Srv
                || u.kind == victory_shader::ResourceType::Uav && uavs(u.slot)
                || u.kind == victory_shader::ResourceType::Cbv && table_cbvs(u.slot))
        {
            return Err(UnloweredShader::DescriptorDeclaration {
                root: u.slot, kind: u.kind, source: u.root,
            });
        }
    }
    pack_words(&shader.user_data, shader.user_data_dwords(), block, values, tables, root_srvs)
}

fn pack_constants(
    metadata: &[UserData],
    size: u32,
    block: Option<(u32, u32)>,
    values: impl Fn(u8, u32) -> Option<u32>,
) -> Result<Option<RootWords>, UnloweredShader> {
    pack_words(metadata, size, block, values, |_| false, |_| false)
}

fn pack_words(
    metadata: &[UserData],
    size: u32,
    block: Option<(u32, u32)>,
    values: impl Fn(u8, u32) -> Option<u32>,
    supported_table: impl Fn(u8) -> bool,
    supported_root_srv: impl Fn(u8) -> bool,
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
    let mut root_srvs = Vec::new();
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
            RootType::Descriptor if supported_root_srv(u.slot) => match u.dword {
                victory_shader::emit::ROOT_SRV_VIEW => root_srvs.push((u.index as usize, u.slot)),
                victory_shader::emit::ROOT_VIEW_OFFSET => *w = 0,
                _ => return Err(UnloweredShader::UnknownUserData),
            },
            _ => return Err(UnloweredShader::UnknownUserData),
        }
    }
    Ok(Some(RootWords { parameter, words, tables, root_srvs }))
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
        Some(UnloweredShader::DescriptorDeclaration { root: 0, kind: used.kind, source: used.root }),
    );
    let sampler = [0; 4];
    let baked = HostShader {
        samplers: vec![sampler],
        static_samplers: vec![victory_shader::emit::StaticSampler::from_words(sampler)],
        ..shader.clone()
    };
    let declared = UsedResource {
        root: RootType::ImmediateSampler, kind: victory_shader::ResourceType::Sampler,
        slot: 7, dwords: 4, stride: 0, ..used
    };
    assert!(matches!(constants_only(&baked, &[declared], None, |_, _| None), Ok(None)));
    assert!(constants_only(&shader, &[declared], None, |_, _| None).is_err());
    assert!(constants_only(&baked, &[UsedResource { root: RootType::Table, ..declared }], None,
        |_, _| None).is_err());
}


impl RootWords {
    pub(crate) fn relocated(&self, bases: &BTreeMap<u8, u32>, srvs: &BTreeMap<u8, u32>) -> Option<Vec<u32>> {
        let mut words = self.words.clone();
        for &(word, slot) in &self.tables {
            words[word] = bases.get(&slot)?.checked_mul(32)?;
        }
        for &(word, slot) in &self.root_srvs {
            words[word] = *srvs.get(&slot)?;
        }
        Some(words)
    }
}

#[cfg(test)]
#[test]
fn closed_draw_relocates_table_byte_offsets_without_changing_constants() {
    let table = UserData { index: 0, slot: 4, root: RootType::Table, dword: 0 };
    let constant = UserData { index: 2, slot: 1, root: RootType::Constants, dword: 3 };
    let words = pack_words(&[table, constant], 4, Some((5, 4)), |_, _| Some(0x700), |s| s == 4, |_| false)
        .unwrap().unwrap();
    assert_eq!(words.relocated(&[(4, 32)].into(), &BTreeMap::new()), Some(vec![1024, 0, 0x700, 0]));
    assert_eq!(words.relocated(&BTreeMap::new(), &BTreeMap::new()), None);
}


/// One validated shader binding snapshot for either graphics or compute.
#[derive(Default)]
pub(crate) struct ShaderBindings {
    pub(crate) constants: Vec<RootWords>,
    tables: BTreeMap<u8, BTreeMap<u32, ClosedView>>,
    cbvs: Vec<(u32, ConstantSpan)>,
    root_srvs: BTreeMap<u8, ByteSpan>,
    #[cfg(feature = "paged-title")]
    has_ordered_color_reads: bool,
}

enum ClosedView {
    Texture(TextureInput),
    NullTexture(windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT, crate::page_texture::TextureSelection),
    TextureUav(crate::page_uav::ShaderTextureUav),
    Raw(BoundedBufferView),
    ScalarStructured(BoundedBufferView, u32),
    TableConstant(BoundedBufferView),
    Structured(BoundedBufferView, u32),
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
    pub(crate) shared_access: crate::page_texture::ScopedTextureAccess,
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

type ByteSpan = (u64, u64);
type ReadSpan = (ByteSpan, u32); // zero stride denotes a raw view

/// Bind bounded reads to canonical slabs when possible; pack the remaining
/// spans into a single fence-owned buffer. Root CBVs retain separate storage.
struct BoundedReads {
    views: BTreeMap<ReadSpan, (ID3D12Resource, u64)>,
}

impl BoundedReads {
    fn prepare(
        host: &Host, list: &HostList, batch: &PinnedBatch,
        spans: impl IntoIterator<Item = ReadSpan>,
        mut direct: Option<&mut crate::page_view::SlabReads>,
        state: D3D12_RESOURCE_STATES,
    ) -> Result<Self, BatchError> {
        let spans: BTreeSet<_> = spans.into_iter().collect();
        let mut views = BTreeMap::new();
        let mut size = 0u64;
        let mut offsets = BTreeMap::new();
        for key @ ((start, end), stride) in spans {
            let invalid = || BatchError::Range(start, end);
            let length = end.checked_sub(start).filter(|&len| len != 0).ok_or_else(invalid)?;
            let alignment = if stride == 0 {
                u64::from(D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT)
            } else {
                // Least common multiple of element stride and 16-byte alignment.
                u64::from(stride) << (4 - stride.trailing_zeros().min(4))
            };
            if let Some(direct) = direct.as_deref_mut() {
                if let Some(view) = direct.direct(list, batch, (start, end), alignment, state)? {
                    views.insert(key, view);
                    continue;
                }
            }
            let offset = size.checked_next_multiple_of(alignment).ok_or_else(invalid)?;
            size = offset.checked_add(length).ok_or_else(invalid)?;
            offsets.insert(key, offset);
        }
        if !offsets.is_empty() {
            let buffer = host.default_buffer(size).ok_or_else(|| {
                BatchError::Native(windows::Win32::Foundation::E_OUTOFMEMORY.into())
            })?;
            crate::page_view::gather_regions(list, batch, offsets.iter().map(|(&(span, _), &at)| (span, at)),
                &buffer, D3D12_RESOURCE_STATE_COPY_DEST, state)?;
            views.extend(offsets.into_iter().map(|(key, at)| (key, (buffer.clone(), at))));
        }
        Ok(Self { views })
    }

    fn slot(&self, span: ByteSpan, stride: u32) -> Result<(&ID3D12Resource, u64), BatchError> {
        self.views.get(&(span, stride)).map(|(buffer, at)| (buffer, *at))
            .ok_or(BatchError::Range(span.0, span.1))
    }
}

#[cfg(all(test, feature = "diagnostics"))]
impl ShaderBindings {
    pub(crate) fn record_test_texture(&mut self, root: u8, slot: u32, input: TextureInput) {
        self.tables.entry(root).or_default().insert(slot, ClosedView::Texture(input));
    }
}

#[cfg(all(test, feature = "diagnostics"))]
impl RootWords {
    pub(crate) fn test_words(parameter: u32, words: Vec<u32>, tables: Vec<(usize, u8)>) -> Self {
        Self { parameter, words, tables, root_srvs: Vec::new() }
    }
}

impl ShaderBindings {
    pub(crate) fn capture(
        pipeline: &HostPipeline,
        values: impl Fn(u8, u32) -> Option<u32>,
        view: impl Fn(u8, u32) -> Option<View>,
        address: impl Fn(u8) -> Option<u64>,
        direct_groups: Option<[u32; 3]>,
    ) -> Result<Self, UnloweredShader> {
        if pipeline.marks.is_some() {
            return Err(UnloweredShader::DescriptorHeap);
        }
        let mut constants = Vec::new();
        let mut cbvs = Vec::new();
        let mut root_srvs = BTreeMap::<u8, ByteSpan>::new();
        let mut tables = BTreeMap::<u8, BTreeMap<u32, _>>::new();
        #[cfg(feature = "paged-title")]
        let mut has_ordered_color_reads = false;
        for stage in &pipeline.stages {
            let t = stage.translated();
            for &(slot, param) in &stage.root_cbvs {
                let span = address(slot).and_then(ConstantSpan::capture)
                    .ok_or(UnloweredShader::RootView)?;
                cbvs.push((param, span));
            }
            let needs_table_or_root = t.host.user_data.iter().any(|u| {
                u.root == RootType::Table || u.root == RootType::Descriptor && u.dword == victory_shader::emit::ROOT_SRV_VIEW
            });
            let direct_report;
            let slots = if needs_table_or_root {
                use victory_shader::emit::{StaticViewFailure, StaticViewReport};
                let reject = |reason| UnloweredShader::DescriptorProof { stage: stage.shader.stage, reason };
                match stage.shader.static_view_slots() {
                    Ok(slots) => slots,
                    Err(failure @ StaticViewFailure::UnsupportedAccess { operation: "image_store", .. })
                        if pipeline.compute =>
                    {
                        let groups = direct_groups.filter(|groups| groups.iter().all(|&n| n > 0))
                            .ok_or_else(|| reject(failure))?;
                        direct_report = stage.shader.direct_view_report(groups);
                        match &*direct_report {
                            StaticViewReport::Proven(slots) => slots,
                            StaticViewReport::Rejected(reason) => return Err(reject(*reason)),
                        }
                    }
                    Err(reason) => return Err(reject(reason)),
                }
            } else {
                &[]
            };
            if let Some(words) = stage_words(
                &t.host, &t.used, stage.user_data, &values,
                |root| slots.iter().any(|s| s.slot.root == root && !matches!(s.access, StaticViewAccess::RootSrv { .. })),
                |root| pipeline.compute && slots.iter().any(|s| {
                    s.slot.root == root && matches!(s.access, StaticViewAccess::RawUav { .. } | StaticViewAccess::StructuredUav { .. } | StaticViewAccess::TextureWrite { .. })
                }),
                |root| slots.iter().any(|s| s.slot.root == root && matches!(s.access, StaticViewAccess::ConstantTable { .. } | StaticViewAccess::ConstantTableDescriptor)),
                |slot| stage.root_cbvs.iter().any(|&(bound, _)| bound == slot),
                |root| slots.iter().any(|s| s.slot.root == root && matches!(s.access, StaticViewAccess::RootSrv { .. })),
            )? {
                for proof in slots {
                    let slot = proof.slot;
                    if let StaticViewAccess::RootSrv { bytes } = proof.access {
                        if !words.root_srvs.iter().any(|&(_, root)| root == slot.root) {
                            return Err(UnloweredShader::RootView);
                        }
                        let start = address(slot.root).filter(|&va| va != 0 && va.is_multiple_of(4))
                            .ok_or(UnloweredShader::RootView)?;
                        let end = start.checked_add(u64::from(bytes))
                            .filter(|&end| end <= victory_memory::gpu_table::ADDRESS_LIMIT)
                            .ok_or(UnloweredShader::RootView)?;
                        root_srvs.entry(slot.root).and_modify(|span| span.1 = span.1.max(end))
                            .or_insert((start, end));
                        continue;
                    }
                    let reason = |reason| UnloweredShader::DescriptorSlot {
                        stage: stage.shader.stage, root: slot.root, index: slot.index, reason,
                    };
                    let reject = reason(DescriptorSlotReason::OutsideRangeOrConflictingView);
                    let buffer = reason(DescriptorSlotReason::UnsupportedBuffer);
                    if !words.tables.iter().any(|&(_, root)| root == slot.root) {
                        return Err(reason(DescriptorSlotReason::RootWord));
                    }
                    let table = tables.entry(slot.root).or_default();
                    if let std::collections::btree_map::Entry::Vacant(e) = table.entry(slot.index) {
                        let actual = view(slot.root, slot.index).ok_or(reason(DescriptorSlotReason::UnboundOrStale))?;
                        let bound = match proof.access {
                            StaticViewAccess::TextureRead { array } => {
                                if let Some((format, selection)) = actual.paged_null_texture_srv(array) {
                                    ClosedView::NullTexture(format, selection)
                                } else {
                                    ClosedView::Texture(actual.paged_texture_srv(array).map_err(|phase| {
                                        let (kind, title_format, view_format, view_dimension, texture_signature) =
                                            actual.texture_srv_metadata();
                                        reason(DescriptorSlotReason::UnsupportedTexture {
                                            array, volume: false, kind, title_format, view_format, view_dimension, texture_signature, phase,
                                        })
                                    })?)
                                }
                            }
                            #[cfg(feature = "paged-title")]
                            StaticViewAccess::VolumeTextureRead => {
                                ClosedView::Texture(actual.paged_volume_srv().map_err(|phase| {
                                    let (kind, title_format, view_format, view_dimension, texture_signature) =
                                        actual.texture_srv_metadata();
                                    reason(DescriptorSlotReason::UnsupportedTexture {
                                        array: false, volume: true, kind, title_format, view_format, view_dimension,
                                        texture_signature, phase,
                                    })
                                })?)
                            }
                            StaticViewAccess::TextureWrite { .. } if pipeline.compute => {
                                ClosedView::TextureUav(crate::descriptor::host::paged_shader_texture_uav(&actual).map_err(|phase| {
                                    let (title_format, view_format, view_dimension) = crate::descriptor::host::paged_uav_format(&actual);
                                    reason(DescriptorSlotReason::UnsupportedTextureWrite { title_format, view_format, view_dimension, phase })
                                })?)
                            }
                            StaticViewAccess::ConstantTable { .. } | StaticViewAccess::ConstantTableDescriptor => {
                                let (start, end) = crate::descriptor::host::paged_table_constant_range(&actual)
                                    .ok_or(buffer)?;
                                ClosedView::TableConstant(BoundedBufferView::new(start, end, false)
                                    .ok_or(buffer)?)
                            }
                            StaticViewAccess::RawSrv { .. } => {
                                let (start, end) = crate::descriptor::host::paged_raw_srv_range(&actual)
                                    .ok_or(buffer)?;
                                ClosedView::Raw(BoundedBufferView::new(start, end, false)
                                    .ok_or(buffer)?)
                            }
                            StaticViewAccess::ScalarStructuredSrv { stride, .. } => {
                                let unsupported = || reason(DescriptorSlotReason::ScalarBufferView {
                                    expected_stride: stride,
                                    actual: crate::descriptor::host::buffer_view_shape(&actual),
                                });
                                let (start, end) = crate::descriptor::host::paged_scalar_srv_range(&actual, stride)
                                    .ok_or_else(unsupported)?;
                                // SMEM 以字节读取。只为已验证的范围暴露原始视图，
                                // 不与同槽的向量结构化访问混用。
                                ClosedView::ScalarStructured(
                                    BoundedBufferView::new(start, end, false).ok_or_else(unsupported)?, stride,
                                )
                            }
                            StaticViewAccess::StructuredSrv { stride, .. } => {
                                let (start, end) = crate::descriptor::host::paged_structured_srv_range(&actual, stride)
                                    .ok_or(buffer)?;
                                ClosedView::Structured(BoundedBufferView::new(start, end, false).ok_or(buffer)?, stride)
                            }
                            StaticViewAccess::StructuredUav { stride, .. } if pipeline.compute => {
                                let (start, end) = crate::descriptor::host::paged_structured_uav_range(&actual, stride)
                                    .ok_or(buffer)?;
                                ClosedView::Structured(BoundedBufferView::new(start, end, true).ok_or(buffer)?, stride)
                            }
                            StaticViewAccess::RawUav { .. } if pipeline.compute => {
                                let (start, end) = crate::descriptor::host::paged_raw_range(&actual)
                                    .ok_or(buffer)?;
                                ClosedView::Raw(BoundedBufferView::new(start, end, true)
                                    .ok_or(buffer)?)
                            }
                            _ => return Err(reason(DescriptorSlotReason::UnsupportedAccess)),
                        };
                        #[cfg(feature = "paged-title")]
                        { has_ordered_color_reads |= matches!(&bound, ClosedView::Texture(TextureInput::OrderedRender(_))); }
                        e.insert(bound);
                    }
                    match (table.get_mut(&slot.index), proof.access) {
                        (Some(ClosedView::Texture(texture)), StaticViewAccess::TextureRead { array })
                            if texture.selection().matches_2d(array) => {},
                        (Some(ClosedView::Texture(texture)), StaticViewAccess::VolumeTextureRead)
                            if matches!(texture.selection().kind, crate::page_texture::TextureViewKind::Volume) => {},
                        (Some(ClosedView::NullTexture(_, selection)), StaticViewAccess::TextureRead { array })
                            if selection.matches_2d(array) => {},
                        (Some(ClosedView::TextureUav(view)), StaticViewAccess::TextureWrite { x, y, z, sources, exec_guard, dispatch }) => {
                            if !view.allows(x, y, z) && !view.allows_dispatch_edge(x, y, z, dispatch) {
                                return Err(reason(DescriptorSlotReason::TextureWriteRange {
                                    host_format: view.format_id(), xyz_max: (x, y, z),
                                    view_width_height_w_window: view.view_extent(),
                                    coordinate_sources: sources, exec_guard, groups_and_threads: dispatch,
                                }));
                            }
                        }
                        (Some(ClosedView::Raw(raw)), StaticViewAccess::RawSrv { offset, bytes })
                        | (Some(ClosedView::TableConstant(raw)), StaticViewAccess::ConstantTable { offset, bytes })
                            if !raw.unordered() =>
                        {
                            raw.allow_access(offset, bytes, false).ok_or(reject)?;
                        }
                        (Some(ClosedView::Structured(view, actual)), StaticViewAccess::StructuredSrv { stride, elements })
                            if *actual == stride && !view.unordered() =>
                        {
                            view.allow_access(0, elements.checked_mul(stride).ok_or(reject)?, false).ok_or(reject)?;
                        }
                        (Some(ClosedView::ScalarStructured(view, actual)), StaticViewAccess::ScalarStructuredSrv { stride, offset, bytes })
                            if *actual == stride =>
                        {
                            view.allow_access(offset, bytes, false).ok_or(reject)?;
                        }
                        (Some(ClosedView::TableConstant(view)), StaticViewAccess::ConstantTableDescriptor)
                            if !view.unordered() =>
                        {
                            view.allow_complete_constant_read()
                                .ok_or(reason(DescriptorSlotReason::ConstantTableDescriptorBudget))?;
                        }
                        (Some(ClosedView::Structured(view, actual)), StaticViewAccess::StructuredUav { stride, elements, write })
                            if *actual == stride && view.unordered() =>
                        {
                            view.allow_access(0, elements.checked_mul(stride).ok_or(reject)?, write).ok_or(reject)?;
                        }
                        (Some(ClosedView::Raw(raw)), StaticViewAccess::RawUav { offset, bytes, write }) if raw.unordered() => {
                            raw.allow_access(offset, bytes, write).ok_or(reject)?;
                        }
                        _ => return Err(reject),
                    }
                }
                constants.push(words);
            } else if !slots.is_empty() {
                return Err(UnloweredShader::DescriptorTable);
            }
        }
        let mut destinations = Vec::<&crate::page_uav::ShaderTextureUav>::new();
        for view in tables.values().flat_map(|table| table.values()) {
            if let ClosedView::TextureUav(uav) = view {
                let target = uav.target().as_ref();
                if destinations.iter().any(|other| other.target().texture == target.texture) {
                    return Err(UnloweredShader::DescriptorTable);
                }
                destinations.push(uav);
            }
        }
        for view in tables.values().flat_map(|table| table.values()) {
            if let ClosedView::Texture(source) = view
                && destinations.iter().any(|output| source.conflicts_color(output.target(), output.subresource()))
            {
                return Err(UnloweredShader::DescriptorTable);
            }
        }
        Ok(Self {
            constants, tables, cbvs, root_srvs,
            #[cfg(feature = "paged-title")]
            has_ordered_color_reads,
        })
    }

    pub(crate) fn ranges(&self, ranges: &mut Vec<(u64, u64)>) -> Result<(), BatchError> {
        self.ordinary_ranges(ranges)?;
        #[cfg(feature = "paged-title")]
        for volume in self.texture_outputs().filter_map(|uav| uav.byte_backing()) {
            ranges.extend(volume.ranges());
        }
        Ok(())
    }

    fn ordinary_ranges(&self, ranges: &mut Vec<(u64, u64)>) -> Result<(), BatchError> {
        ranges.extend(self.cbvs.iter().map(|(_, span)| span.range()));
        ranges.extend(self.root_srvs.values().copied());
        for table in self.tables.values() {
            for view in table.values() {
                match view {
                    ClosedView::Texture(texture) => texture.ranges(ranges)?,
                    // The queue footprint records writable native textures separately.
                    ClosedView::TextureUav(_) | ClosedView::NullTexture(..) => {},
                    ClosedView::Raw(raw) | ClosedView::TableConstant(raw)
                    | ClosedView::Structured(raw, _) | ClosedView::ScalarStructured(raw, _) => ranges.push(raw.range()),
                }
            }
        }
        Ok(())
    }

    pub(crate) fn samples_target(&self, target: &crate::resource::HostTexture) -> bool {
        self.tables.values().flat_map(|table| table.values())
            .any(|view| matches!(view, ClosedView::Texture(tex) if tex.aliases_target(target)))
    }

    pub(crate) fn depth_reads_are_exact(&self, target: &std::sync::Arc<crate::resource::HostTexture>) -> bool {
        self.tables.values().flat_map(|table| table.values()).all(|view| match view {
            ClosedView::Texture(tex) if tex.aliases_target(target) => tex.is_native_depth_read_of(target),
            _ => true,
        })
    }

    pub(crate) fn conflicts_color(
        &self, target: &std::sync::Arc<crate::resource::HostTexture>,
        sub: crate::page_resource::ColorSubresource,
    ) -> bool {
        self.tables.values().flat_map(|table| table.values())
            .any(|view| matches!(view, ClosedView::Texture(tex) if tex.conflicts_color(target, sub)))
    }

    pub(crate) fn uses_uav(&self) -> bool {
        self.tables.values().flat_map(|table| table.values())
            .any(|view| matches!(view, ClosedView::TextureUav(_))
                || matches!(view, ClosedView::Raw(raw) | ClosedView::Structured(raw, _) if raw.unordered()))
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn verify_shared_volume(
        &self, batch: &PinnedBatch, indirect: bool,
    ) -> Result<(), BatchError> {
        let mut outputs = self.texture_outputs();
        let Some(volume) = outputs.find_map(|uav| uav.byte_backing()) else { return Ok(()); };
        // 暂只接纳单一体纹理写者和能证明字节跨度的读取。间接参数另有别名语义。
        if indirect {
            return Err(BatchError::unsupported_texture_map("共享体纹理不支持间接派发"));
        }
        if self.texture_outputs().count() != 1 {
            return Err(BatchError::unsupported_texture_map("共享体纹理派发不止一个纹理输出"));
        }
        if self.tables.values().flat_map(|table| table.values()).any(|view| {
            matches!(view, ClosedView::Texture(t) if !t.has_exact_guest_bytes())
        }) {
            return Err(BatchError::unsupported_texture_map("共享体纹理派发有读取视图无法证明来宾字节范围"));
        }
        let mut pages = BTreeSet::new();
        for span in volume.ranges() {
            batch.for_each_physical_span(span, |page, _| {
                pages.insert(page);
                Ok(())
            })?;
        }
        let mut ordinary = Vec::new();
        self.ordinary_ranges(&mut ordinary)?;
        for span in ordinary {
            batch.for_each_physical_span(span, |page, _| {
                if pages.contains(&page) { return Err(BatchError::Range(span.0, span.1)); }
                Ok(())
            })?;
        }
        Ok(())
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn uses_guest_uav(&self) -> bool {
        self.texture_outputs().any(|uav| uav.byte_backing().is_some())
            || self.tables.values().flat_map(|table| table.values())
            .any(|view| matches!(view, ClosedView::Raw(raw) | ClosedView::Structured(raw, _) if raw.unordered()))
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn pointer_sources<'a>(&'a self, out: &mut Vec<&'a std::sync::Arc<crate::resource::HostTexture>>) {
        out.extend(self.tables.values().flat_map(|table| table.values()).filter_map(|view| {
            match view {
                ClosedView::Texture(input) => input.pointer_source(),
                _ => None,
            }
        }));
    }

    fn texture_outputs(&self) -> impl Iterator<Item = &crate::page_uav::ShaderTextureUav> {
        self.tables.values().flat_map(|table| table.values()).filter_map(|view| match view {
            ClosedView::TextureUav(uav) => Some(uav),
            _ => None,
        })
    }

    pub(crate) fn finish_texture_writes(
        &self, host: &Host, list: &HostList, batch: &PinnedBatch,
        plan: &mut crate::page_resource::ResourcePlan, dirty: &mut BTreeSet<PhysicalPage>,
    ) -> Result<(), BatchError> {
        let mut outputs = self.texture_outputs().peekable();
        if outputs.peek().is_some() {
            crate::page_resource::PageBarrier::Uav.encode(list, plan)?;
            for output in outputs {
                output.finish_write(host, list, batch, plan, dirty)?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn ordered_color_reads<'a>(&'a self, out: &mut Vec<&'a crate::page_texture::OrderedRenderRead>) {
        if !self.has_ordered_color_reads { return; }
        out.extend(self.tables.values().flat_map(|table| table.values()).filter_map(|view| match view {
            ClosedView::Texture(TextureInput::OrderedRender(read)) => Some(read),
            _ => None,
        }));
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn written_targets<'a>(&'a self, out: &mut Vec<&'a std::sync::Arc<crate::resource::HostTexture>>) {
        out.extend(self.texture_outputs().map(crate::page_uav::ShaderTextureUav::target));
    }

    pub(crate) fn prepare(
        &self, host: &Host, list: &HostList, batch: &PinnedBatch,
        descriptors: &mut PageDescriptors<'_>,
        plan: &mut crate::page_resource::ResourcePlan,
        dirty: &mut BTreeSet<PhysicalPage>,
        color_targets: &[&crate::resource::HostTexture],
        slab_reads: &mut crate::page_view::SlabReads,
        compute: bool,
    ) -> Result<PreparedShader, BatchError> {
        let spans = self.tables.values().flat_map(|table| table.values()).filter_map(|view| {
            match view {
                ClosedView::Raw(raw) | ClosedView::TableConstant(raw) if !raw.unordered() => Some((raw.range(), 0)),
                ClosedView::ScalarStructured(view, _) => Some((view.range(), 0)),
                ClosedView::Structured(view, stride) if !view.unordered() => Some((view.range(), *stride)),
                _ => None,
            }
        });
        let mut state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        if !compute { state |= D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE; }
        let bounded_reads = BoundedReads::prepare(host, list, batch,
            spans.chain(self.root_srvs.values().map(|&span| (span, 0))),
            (!self.uses_uav()).then_some(slab_reads), state)?;
        let mut root_srv_slots = BTreeMap::new();
        for (&root, &span) in &self.root_srvs {
            let (buffer, at) = bounded_reads.slot(span, 0)?;
            let view = TableDescriptor::Raw(buffer.clone(), at, span.1 - span.0, false);
            let slot = descriptors.reserve_table_run(&[Some(view)]).map_err(BatchError::Native)?;
            root_srv_slots.insert(root, slot);
        }
        let mut bases = BTreeMap::new();
        let mut shader_dirty = BTreeSet::new();
        let mut shared_access = crate::page_texture::ScopedTextureAccess::default();
        let texture_reads: Vec<_> = self.tables.values().flat_map(|table| table.values())
            .filter_map(|view| match view {
                ClosedView::Texture(input) => Some(input),
                _ => None,
            }).collect();
        let shared_uavs: Vec<_> = self.texture_outputs().map(|output| output.target().as_ref())
            .filter(|output| texture_reads.iter().any(|input| input.aliases_target(output)))
            .collect();
        for (&root, table) in &self.tables {
            let size = table.last_key_value().expect("captured slots exist").0 + 1;
            let mut views: Vec<_> = (0..size).map(|_| None).collect();
            for (&index, view) in table {
                views[index as usize] = Some(match view {
                    ClosedView::Texture(texture) => {
                        let shared = color_targets.iter().chain(&shared_uavs).any(|target| texture.aliases_target(target));
                        let (resource, format) = texture.encode(host, list, batch, plan,
                            shared.then_some(&mut shared_access))
                            .map_err(|e| e.at_texture(crate::page_slabs::TextureContentUse::ShaderRead { root, index },
                                texture.host_texture()))?;
                        TableDescriptor::Texture(resource, format, texture.selection())
                    }
                    ClosedView::NullTexture(format, selection) => TableDescriptor::NullTexture(*format, *selection),
                    ClosedView::TextureUav(view) => {
                        let shared = shared_uavs.iter().any(|target| target.texture == view.target().texture);
                        view.prepare(host, list, batch, plan, shared.then_some(&mut shared_access))
                            .map_err(|e| e.at_texture(crate::page_slabs::TextureContentUse::ShaderWrite { root, index },
                                &view.target().texture))?
                    },
                    ClosedView::Raw(raw) if raw.unordered() => raw.prepare_uav(batch, &mut shader_dirty, 0)?,
                    ClosedView::Raw(raw) | ClosedView::TableConstant(raw) | ClosedView::ScalarStructured(raw, _) => {
                        let span = raw.range();
                        let (buffer, at) = bounded_reads.slot(span, 0)?;
                        TableDescriptor::Raw(buffer.clone(), at, span.1 - span.0, false)
                    }
                    ClosedView::Structured(view, stride) if view.unordered() =>
                        view.prepare_uav(batch, &mut shader_dirty, *stride)?,
                    ClosedView::Structured(view, stride) => {
                        let span = view.range();
                        let (buffer, at) = bounded_reads.slot(span, *stride)?;
                        let elements = ((span.1 - span.0) / u64::from(*stride)) as u32;
                        TableDescriptor::Structured(buffer.clone(), at, elements, *stride, false)
                    }
                });
            }
            let first = descriptors.reserve_table_run(&views).map_err(BatchError::Native)?;
            bases.insert(root, first);
        }
        if !shader_dirty.is_empty() {
            // Staged readers are snapshots. A shader reading and writing
            // the same physical pages needs a single directly shared view.
            let mut reads: Vec<_> = self.cbvs.iter().map(|(_, span)| span.range()).collect();
            reads.extend(self.root_srvs.values().copied());
            for view in self.tables.values().flat_map(|table| table.values()) {
                match view {
                    ClosedView::Texture(tex) => tex.ranges(&mut reads)?,
                    ClosedView::Raw(raw) | ClosedView::TableConstant(raw) if !raw.unordered() => reads.push(raw.range()),
                    ClosedView::ScalarStructured(view, _) => reads.push(view.range()),
                    ClosedView::Structured(view, _) if !view.unordered() => reads.push(view.range()),
                    _ => {},
                }
            }
            for span in reads {
                batch.for_each_physical_span(span, |page, _| {
                    if shader_dirty.contains(&page) { return Err(BatchError::Range(span.0, span.1)); }
                    Ok(())
                })?;
            }
            dirty.extend(shader_dirty);
        }
        let words: Vec<_> = self.constants.iter().map(|root| {
            root.relocated(&bases, &root_srv_slots).map(|words| (root.parameter, words))
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
        Ok(PreparedShader { words, cbvs, shared_access })
    }
}
