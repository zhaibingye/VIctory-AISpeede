//! Submission-local attachment authority and resource transitions.
//! The global resource ledger is changed only after successful execution.
use std::collections::BTreeMap;
use std::sync::{Arc, atomic::Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use victory_memory::{PAGE_BYTES, authority::{Representation, Version}, coherence::Fence, physical::PhysicalPage};

use crate::host::{HostList, legal_state, transition};
use crate::page_slabs::BatchError;
use crate::resource::HostTexture;

/// Latest submitted values on the direct queue. Only its current
/// bounded window sees these; guest-visible completion is independent.
#[derive(Clone, Default)]
pub(crate) struct SubmittedTail {
    versions: BTreeMap<(usize, Option<u32>), Version>,
    states: BTreeMap<usize, D3D12_RESOURCE_STATES>,
}

/// 仅用于诊断已完成的 GPU 写入。记录的是可能与共享字节相交的页，
/// 不是最新版本；写回、CPU 再写和未完成的批次都不由这张表裁决。
#[cfg(all(feature = "paged-title", feature = "diagnostics"))]
#[derive(Default)]
struct NativeWriteTrace {
    pages: std::collections::BTreeSet<PhysicalPage>,
    without_layout: usize,
    without_mapping: usize,
    known_private_heap: usize,
    truncated: bool,
}

#[cfg(all(feature = "paged-title", feature = "diagnostics"))]
static NATIVE_WRITES: std::sync::LazyLock<std::sync::Mutex<NativeWriteTrace>> = std::sync::LazyLock::new(Default::default);

#[cfg(all(feature = "paged-title", feature = "diagnostics"))]
impl NativeWriteTrace {
    const MAX_PAGES: usize = 32_768;

    fn capture(spans: impl Iterator<Item = Option<(u64, u64)>>) -> Self {
        let mut backing = crate::title_backing::SubmissionBacking::default();
        let mut result = Self::default();
        let mut pinned_spans = Vec::new();
        for span in spans {
            let Some((start, end)) = span else {
                result.without_layout += 1;
                continue;
            };
            let (start, end) = (start as usize, end as usize);
            if crate::title_backing::is_private_heap_span(start, end) {
                result.known_private_heap += 1;
            } else {
                backing.include(start, end);
                pinned_spans.push((start, end));
            }
        }
        backing.flush();
        for (start, end) in pinned_spans {
            let Some(runs) = backing.pins().physical_runs(start, end) else {
                result.without_mapping += 1;
                continue;
            };
            for (first, count) in runs {
                for i in 0..count {
                    if result.pages.len() == Self::MAX_PAGES {
                        result.truncated = true;
                        return result;
                    }
                    result.pages.insert(first.offset(i));
                }
            }
        }
        result
    }

    fn incomplete(&self) -> bool {
        self.without_layout != 0 || self.without_mapping != 0 || self.truncated
    }

    fn merge(&mut self, other: Self) {
        self.without_layout = self.without_layout.saturating_add(other.without_layout);
        self.without_mapping = self.without_mapping.saturating_add(other.without_mapping);
        self.known_private_heap = self.known_private_heap.saturating_add(other.known_private_heap);
        self.truncated |= other.truncated;
        for id in other.pages {
            if self.pages.len() == Self::MAX_PAGES && !self.pages.contains(&id) {
                self.truncated = true;
                break;
            }
            self.pages.insert(id);
        }
    }
}

/// 只在拒绝位置记录无地址、无内容的证据；未发现重叠不承诺字节有效。
#[cfg(all(feature = "paged-title", feature = "diagnostics"))]
pub(crate) fn trace_rejected_shared_read(span: Option<(u64, u64)>) {
    if !tracing::enabled!(tracing::Level::TRACE) { return; }
    let read = NativeWriteTrace::capture(std::iter::once(span));
    let writers = NATIVE_WRITES.lock().unwrap_or_else(|e| e.into_inner());
    tracing::trace!(
        known_overlap = !writers.pages.is_disjoint(&read.pages),
        read_pages = read.pages.len(),
        completed_write_pages = writers.pages.len(),
        incomplete_read = read.incomplete() || read.known_private_heap != 0,
        incomplete_history = writers.incomplete(),
        writers_without_layout = writers.without_layout,
        writers_without_mapping = writers.without_mapping,
        writers_on_default_private_heap = writers.known_private_heap,
        history_truncated = writers.truncated,
        "共享纹理拒绝：已完成原生写入与读取物理范围对比"
    );
}

#[derive(Default)]
pub(crate) struct ResourcePlan {
    inherited: SubmittedTail,
    #[cfg(feature = "paged-title")]
    refreshed: BTreeMap<usize, Arc<HostTexture>>,
    #[cfg(feature = "paged-title")]
    snapshot: Option<crate::page_slabs::SnapshotDigests>,
    pending: BTreeMap<(usize, Option<u32>), (Arc<HostTexture>, Version)>,
    #[cfg(feature = "paged-title")]
    pub(crate) section_texture_read: bool,
    #[cfg(feature = "paged-title")]
    pub(crate) ordered_color_backing: std::collections::BTreeSet<usize>,
    #[cfg(feature = "paged-title")]
    ordered_color_initialized: std::collections::BTreeSet<usize>,
    // 已编码而未提交的批次计数；提交成功前不得发布。
    pub(crate) encoded_draws: u64,
    pub(crate) encoded_dispatches: u64,
    #[cfg(feature = "paged-title")]
    pub(crate) unclassified_native_write: bool,
    #[cfg(all(feature = "paged-title", feature = "diagnostics"))]
    trace_writers: BTreeMap<usize, Option<(u64, u64)>>,
    #[cfg(all(feature = "paged-title", feature = "diagnostics"))]
    trace_capture: Option<NativeWriteTrace>,
    guest_overwrites: BTreeMap<PhysicalPage, Vec<std::ops::Range<usize>>>,
    state: BTreeMap<usize, (Arc<HostTexture>, D3D12_RESOURCE_STATES)>,
    #[cfg(feature = "paged-title")]
    query_writes: BTreeMap<usize, (Arc<crate::query::NativeQueryHeap>, BTreeMap<u32, Option<D3D12_QUERY_TYPE>>)>,
}

impl ResourcePlan {
    pub(crate) fn after(tail: &SubmittedTail) -> Self {
        Self { inherited: tail.clone(), ..Self::default() }
    }

    pub(crate) fn advance_submitted_tail(&self, tail: &mut SubmittedTail) {
        tail.versions.extend(self.pending.iter().map(|(&key, (_, version))| (key, *version)));
        tail.states.extend(self.state.iter().map(|(&key, (_, state))| (key, *state)));
    }

    fn key(copy: &Arc<HostTexture>) -> usize {
        Arc::as_ptr(copy) as usize
    }

    fn version_at(&self, copy: &Arc<HostTexture>, subresource: Option<u32>) -> Option<Version> {
        let key = (Self::key(copy), subresource);
        self.pending.get(&key).map(|(_, version)| *version)
            .or_else(|| self.inherited.versions.get(&key).copied())
            .or_else(|| copy.native_authority.lock().unwrap_or_else(|e| e.into_inner())
                .get(&subresource).map(|complete| complete.version))
    }

    fn version(&self, copy: &Arc<HostTexture>) -> Option<Version> {
        self.version_at(copy, None)
    }

    pub(crate) fn authoritative(&self, copy: &Arc<HostTexture>) -> bool {
        #[cfg(feature = "paged-title")]
        if self.refreshed.contains_key(&Self::key(copy)) {
            return true;
        }
        self.version(copy).map_or_else(
            || copy.host_written.load(Ordering::Acquire),
            |version| version.representation == Representation::NativeGpu,
        )
    }

    /// Convert the same logical version on demand. This copies current
    /// pinned bytes without inventing a new content version.
    pub(crate) fn require_native(
        &mut self, host: &crate::host::Host, list: &HostList,
        batch: &crate::page_slabs::PinnedBatch, copy: &Arc<HostTexture>,
    ) -> Result<(), BatchError> {
        #[cfg(feature = "paged-title")]
        {
            let key = Self::key(copy);
            // A persistent guest pointer can modify compatible backing
            // without Unmap. Before its first GPU consumer in this batch,
            // read current bytes. Never replace writes produced earlier in
            // the same ordered native stream with the old CPU snapshot.
            if copy.has_pointer_exposure()
                && !self.pending.contains_key(&(key, None))
                && !self.inherited.versions.contains_key(&(key, None))
                && !self.refreshed.contains_key(&key)
            {
                let source = copy.visible_linear_writeback()?.ok_or(BatchError::unsupported_texture_map("无法确认纹理映射"))?;
                let unchanged = self.snapshot.as_ref().is_some_and(|snapshot| {
                    copy.native_matches_snapshot(batch, snapshot)
                });
                if !unchanged {
                    self.transition_texture(list, copy, D3D12_RESOURCE_STATE_COPY_DEST.0 as u32)?;
                    source.copy_to(host, list, batch, &copy.texture, (0, 0))?;
                }
                self.refreshed.insert(key, copy.clone());
                return Ok(());
            }
        }
        if self.authoritative(copy) {
            return Ok(());
        }
        #[cfg(feature = "paged-title")]
        if let Some(mut version) = self.version(copy)
            && version.representation == Representation::CompatibleBytes
        {
            let source = copy.compatible_texture_source().ok_or(BatchError::unsupported_texture_map("无法确认纹理映射"))?;
            self.transition_texture(list, copy, D3D12_RESOURCE_STATE_COPY_DEST.0 as u32)?;
            source.copy_to(host, list, batch, &copy.texture, (0, 0))?;
            version.representation = Representation::NativeGpu;
            self.pending.insert((Self::key(copy), None), (copy.clone(), version));
            return Ok(());
        }
        let _ = (host, list, batch);
        Err(BatchError::TargetNotInitialized)
    }

    pub(crate) fn require_color_subresource(
        &mut self, host: &crate::host::Host, list: &HostList, batch: &crate::page_slabs::PinnedBatch,
        copy: &Arc<HostTexture>, sub: ColorSubresource,
    ) -> Result<(), BatchError> {
        self.require_texture_authority(host, list, batch, copy, sub.authority())
    }

    fn require_texture_authority(
        &mut self, host: &crate::host::Host, list: &HostList, batch: &crate::page_slabs::PinnedBatch,
        copy: &Arc<HostTexture>, subresource: Option<u32>,
    ) -> Result<(), BatchError> {
        if subresource.is_none() {
            return self.require_native(host, list, batch, copy);
        }
        if self.version_at(copy, subresource).is_some_and(|v| v.representation == Representation::NativeGpu) {
            Ok(())
        } else {
            Err(BatchError::TargetNotInitialized)
        }
    }

    /// DSV formats in the closed renderer have one mip and slice.
    /// Track depth (plane 0) and stencil (plane 1) independently.
    pub(crate) fn require_depth_planes(&self, copy: &Arc<HostTexture>, mask: u32) -> Result<(), BatchError> {
        for plane in 0..2 {
            if mask & (1 << plane) != 0
                && !self.version_at(copy, Some(plane)).is_some_and(|v| v.representation == Representation::NativeGpu)
            {
                return Err(BatchError::TargetNotInitialized);
            }
        }
        Ok(())
    }

    pub(crate) fn record_depth_writes(&mut self, copy: &Arc<HostTexture>, mask: u32) -> Result<(), BatchError> {
        for plane in 0..2 {
            if mask & (1 << plane) != 0 {
                self.record_write_at(copy, Some(plane))?;
            }
        }
        Ok(())
    }

    pub(crate) fn record_color_write(&mut self, copy: &Arc<HostTexture>, sub: ColorSubresource) -> Result<(), BatchError> {
        self.record_write_at_with(copy, sub.authority(), false)
    }

    /// 明确弃旧可用于写入门槛，不等价于初始化。只支持无可观察字节后备的整资源。
    #[cfg(feature = "paged-title")]
    pub(crate) fn record_discarded_color(&mut self, copy: &Arc<HostTexture>, sub: ColorSubresource) -> Result<(), BatchError> {
        let key = Self::key(copy);
        if !can_discard_color(copy, sub) {
            return Err(BatchError::unsupported_texture_map("无法证明弃旧范围和内容后备"));
        }
        let version = Version::successor(self.version(copy), Representation::Discarded, copy.byte_observation())
            .ok_or_else(|| BatchError::Native(windows::Win32::Foundation::E_FAIL.into()))?;
        self.pending.insert((key, None), (copy.clone(), version));
        self.refreshed.remove(&key);
        self.ordered_color_initialized.remove(&key);
        Ok(())
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn color_is_discarded(&self, copy: &Arc<HostTexture>, sub: ColorSubresource) -> bool {
        !sub.is_scoped() && self.version(copy).is_some_and(|v| v.representation == Representation::Discarded)
    }

    /// 只承认这个批次内已编码的整张写入，不依赖上一个 Fence 或旧标记。
    pub(crate) fn record_full_color_write(&mut self, copy: &Arc<HostTexture>, sub: ColorSubresource) -> Result<(), BatchError> {
        self.record_write_at(copy, sub.authority())?;
        #[cfg(feature = "paged-title")]
        if !sub.is_scoped() && self.ordered_color_backing.contains(&Self::key(copy)) {
            self.ordered_color_initialized.insert(Self::key(copy));
        }
        Ok(())
    }

    #[cfg(feature = "paged-title")]
    fn has_ordered_color_write(&self, copy: &Arc<HostTexture>) -> bool {
        self.ordered_color_initialized.contains(&Self::key(copy))
            && self.pending.get(&(Self::key(copy), None))
                .is_some_and(|(_, v)| v.representation == Representation::NativeGpu)
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn require_ordered_color(&self, copy: &Arc<HostTexture>) -> Result<(), BatchError> {
        if self.has_ordered_color_write(copy) && copy.current_native_generation()
        {
            Ok(())
        } else {
            Err(BatchError::unsupported_texture_map("本批次没有该颜色纹理的已验证整张原生写入"))
        }
    }

    #[cfg(all(feature = "paged-title", feature = "diagnostics"))]
    pub(crate) fn capture_trace_writers(&mut self) {
        if !self.trace_writers.is_empty() {
            self.trace_capture = Some(NativeWriteTrace::capture(self.trace_writers.values().copied()));
        }
    }

    fn record_write_at(&mut self, copy: &Arc<HostTexture>, subresource: Option<u32>) -> Result<(), BatchError> {
        self.record_write_at_with(copy, subresource, true)
    }

    fn record_write_at_with(
        &mut self, copy: &Arc<HostTexture>, subresource: Option<u32>, full_if_discarded: bool,
    ) -> Result<(), BatchError> {
        let old = self.version_at(copy, subresource);
        // 局部（或覆盖范围未获证明的）绘制不能发布整面内容。
        if !full_if_discarded && old.is_some_and(|v| v.representation == Representation::Discarded) {
            return Ok(());
        }
        #[cfg(feature = "paged-title")]
        if !self.unclassified_native_write {
            // 默认系统堆块不能与固定的 XMem 节或独立虚拟内存保留区共享同一个后备。
            self.unclassified_native_write = !copy.began_on_private_heap
                || copy.candidate_guest_span.is_none_or(|(start, end)|
                    !crate::title_backing::is_private_heap_span(start as usize, end as usize));
        }

        #[cfg(all(feature = "paged-title", feature = "diagnostics"))]
        if tracing::enabled!(tracing::Level::TRACE) {
            self.trace_writers.insert(Self::key(copy), copy.candidate_guest_span);
        }
        let version = Version::successor(
            old, Representation::NativeGpu, copy.byte_observation(),
        ).ok_or_else(|| BatchError::Native(windows::Win32::Foundation::E_FAIL.into()))?;
        self.pending.insert((Self::key(copy), subresource), (copy.clone(), version));
        Ok(())
    }

    pub(crate) fn state(&self, copy: &Arc<HostTexture>) -> D3D12_RESOURCE_STATES {
        let key = Self::key(copy);
        self.state.get(&key).map(|(_, state)| *state)
            .or_else(|| self.inherited.states.get(&key).copied())
            .unwrap_or_else(|| D3D12_RESOURCE_STATES(copy.state.load(Ordering::Acquire) as i32))
    }

    pub(crate) fn transition_texture(
        &mut self, list: &HostList, copy: &Arc<HostTexture>, wanted: u32,
    ) -> Result<(), BatchError> {
        let writable = 0x4 | 0x8 | 0x10 | 0x400 | 0x1000;
        let write = wanted & writable;
        if copy.sparse.is_some() || copy.twin.get().is_some_and(Option::is_some)
            || wanted & !(writable | 0x20 | 0x40 | 0x80 | 0x800 | 0x2000) != 0
            || write != 0 && (wanted != write || !write.is_power_of_two())
            || legal_state(&copy.texture, wanted) != wanted
        {
            return Err(BatchError::UnsupportedBarrier);
        }
        let before = self.state(copy);
        let after = D3D12_RESOURCE_STATES(wanted as i32);
        if before != after {
            list.keep(&copy.texture);
            // SAFETY: live attachment, ordered submission-local state,
            // and a checked whole-resource transition.
            unsafe { list.recording().ResourceBarrier(&[transition(&copy.texture, before, after)]) };
            self.state.insert(ResourcePlan::key(copy), (copy.clone(), after));
        }
        Ok(())
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn set_snapshot(&mut self, snapshot: crate::page_slabs::SnapshotDigests) {
        self.snapshot = Some(snapshot);
    }

    /// Unconditional writes own these exact bytes even if their new
    /// value equals the pre-submit snapshot. Conditional shader writes
    /// deliberately do not use this proof.
    pub(crate) fn record_guest_overwrite(&mut self, page: PhysicalPage, start: usize, count: usize) {
        assert!(count != 0 && start + count <= PAGE_BYTES as usize);
        let spans = self.guest_overwrites.entry(page).or_default();
        let mut merged = start..start + count;
        let first = spans.partition_point(|span| span.end < merged.start);
        let mut last = first;
        while let Some(span) = spans.get(last).filter(|span| span.start <= merged.end) {
            merged.start = merged.start.min(span.start);
            merged.end = merged.end.max(span.end);
            last += 1;
        }
        spans.splice(first..last, [merged]);
    }

    pub(crate) fn guest_overwrites(&self, page: PhysicalPage) -> &[std::ops::Range<usize>] {
        self.guest_overwrites.get(&page).map_or(&[], Vec::as_slice)
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn record_query(&mut self, heap: &Arc<crate::query::NativeQueryHeap>, index: u32, state: Option<D3D12_QUERY_TYPE>) {
        self.query_writes.entry(Arc::as_ptr(heap) as usize)
            .or_insert_with(|| (heap.clone(), Default::default())).1.insert(index, state);
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn queries_initialized(&self, heap: &Arc<crate::query::NativeQueryHeap>, indices: std::ops::Range<u32>, kind: D3D12_QUERY_TYPE) -> bool {
        let ordered = self.query_writes.get(&(Arc::as_ptr(heap) as usize));
        let completed = heap.initialized.lock().unwrap_or_else(|e| e.into_inner());
        indices.into_iter().all(|index| match ordered.and_then(|(_, slots)| slots.get(&index)) {
            Some(state) => *state == Some(kind),
            None => completed.get(&index) == Some(&kind),
        })
    }

    pub(crate) fn commit(self, submitted: Fence, observed: Fence, batch: &crate::page_slabs::PinnedBatch) {
        assert!(observed.0 >= submitted.0, "resource authority requires a completed fence");
        #[cfg(all(feature = "paged-title", feature = "diagnostics"))]
        if let Some(observed_writes) = self.trace_capture {
            NATIVE_WRITES.lock().unwrap_or_else(|e| e.into_inner()).merge(observed_writes);
        }

        #[cfg(feature = "paged-title")]
        if let Some(snapshot) = &self.snapshot {
            for (key, copy) in &self.refreshed {
                if !self.pending.contains_key(&(*key, None)) {
                    copy.record_native_snapshot(batch, snapshot);
                }
            }
        }
        let _ = batch;
        #[cfg(feature = "paged-title")]
        for (heap, slots) in self.query_writes.into_values() {
            heap.initialized.lock().unwrap_or_else(|e| e.into_inner()).extend(slots.into_iter().filter_map(|(index, state)| state.map(|kind| (index, kind))));
        }
        for (copy, state) in self.state.into_values() {
            copy.state.store(state.0 as u32, Ordering::Release);
        }
        for ((_, subresource), (copy, mut version)) in self.pending {
            version.observation = copy.byte_observation();
            let complete = version.complete(submitted, observed).expect("fence observed above");
            copy.native_authority.lock().unwrap_or_else(|e| e.into_inner()).insert(subresource, complete);
            if subresource.is_none() {
                // Never advertise a partial native texture as whole.
                copy.host_written.store(version.representation == Representation::NativeGpu, Ordering::Release);
            }
        }
    }
}

#[cfg(all(test, feature = "paged-title"))]
#[test]
fn ordered_color_requires_full_write_to_same_native_object_in_current_batch() {
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_SAMPLE_DESC};
    let host = crate::host::host().expect("D3D12 宿主");
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 4, Height: 4, DepthOrArraySize: 1, MipLevels: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        ..Default::default()
    };
    let make = || {
        let mut texture = None::<ID3D12Resource>;
        // 安全：结构与目标类型匹配；此测试只检查计划，不提交命令。
        unsafe { host.device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut texture,
        ) }.unwrap();
        Arc::new(HostTexture::new(texture.unwrap(), 0, Vec::new()))
    };
    let target = make();
    let other = make();
    let sub = ColorSubresource::capture(desc, 0).unwrap();
    let mut plan = ResourcePlan::default();
    plan.ordered_color_backing.extend([ResourcePlan::key(&target), ResourcePlan::key(&other)]);
    assert!(!plan.has_ordered_color_write(&target));
    plan.record_color_write(&target, sub).unwrap();
    assert!(!plan.has_ordered_color_write(&target));
    plan.record_full_color_write(&other, sub).unwrap();
    assert!(!plan.has_ordered_color_write(&target));
    plan.record_full_color_write(&target, sub).unwrap();
    assert!(plan.has_ordered_color_write(&target));
    let mut tail = SubmittedTail::default();
    plan.advance_submitted_tail(&mut tail);
    assert!(!ResourcePlan::after(&tail).has_ordered_color_write(&target));
    plan.record_discarded_color(&target, sub).unwrap();
    assert!(plan.color_is_discarded(&target, sub));
    assert!(!plan.authoritative(&target) && !plan.has_ordered_color_write(&target));
    assert!(plan.authoritative(&other) && plan.has_ordered_color_write(&other));
    plan.record_color_write(&target, sub).unwrap();
    assert!(!plan.authoritative(&target));
    plan.advance_submitted_tail(&mut tail);
    let after = ResourcePlan::after(&tail);
    assert!(after.color_is_discarded(&target, sub) && !after.authoritative(&target));
    plan.record_full_color_write(&target, sub).unwrap();
    assert!(plan.authoritative(&target) && plan.has_ordered_color_write(&target));
}

/// Include possible compatible ranges before page admission. An earlier
/// full overwrite may supply native authority before this operation.
pub(crate) fn collect_compatible_source(
    copy: &Arc<HostTexture>, ranges: &mut Vec<(u64, u64)>,
) -> Result<(), BatchError> {
    #[cfg(feature = "paged-title")]
    if !copy.has_pointer_exposure() && let Some(source) = copy.compatible_texture_source() {
        ranges.extend(source.rows()?);
    }
    let _ = (copy, ranges);
    Ok(())
}

pub(crate) enum PageBarrier {
    Texture(Arc<HostTexture>, u32),
    NativeUav(Arc<HostTexture>),
    Uav,
    #[cfg(feature = "paged-title")]
    DiscardColor(Arc<HostTexture>, ColorSubresource),
    #[cfg(all(feature = "paged-title", feature = "diagnostics"))]
    EnhancedUav {
        counts_by_kind: [u64; 4], headers_skipped: u32,
        textures: Vec<(std::sync::Weak<HostTexture>, crate::list::CandidateTextureBarrier)>,
    },
}

#[cfg(feature = "paged-title")]
fn can_discard_color(copy: &HostTexture, sub: ColorSubresource) -> bool {
    !sub.is_scoped() && !copy.has_pointer_exposure() && !copy.uses_unproven_alias_backing()
        && copy.exact_committed_bytes().is_none()
        && copy.sparse.is_none() && !copy.twin.get().is_some_and(Option::is_some)
}

impl PageBarrier {
    #[cfg(feature = "paged-title")]
    pub(crate) fn whole_color_discard(copy: Arc<HostTexture>) -> Option<Self> {
        // 安全：copy 保留在整个录制及执行期间。
        let desc = unsafe { copy.texture.GetDesc() };
        if !desc.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) { return None; }
        let sub = ColorSubresource::capture(desc, 0)?;
        can_discard_color(&copy, sub).then_some(Self::DiscardColor(copy, sub))
    }

    pub(crate) fn native_uav(list: &HostList, copy: &HostTexture) {
        // A title texture may still have accesses through canonical byte
        // pages or another native representation. A named host barrier is
        // sufficient only when the resource identity is unambiguous.
        let exact = !copy.has_pointer_exposure() && !copy.uses_unproven_alias_backing()
            && copy.sparse.is_none() && !copy.twin.get().is_some_and(Option::is_some)
            && legal_state(&copy.texture, D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32) != 0;
        Self::emit_uav(list, exact.then_some(&copy.texture));
    }

    fn emit_uav(list: &HostList, texture: Option<&ID3D12Resource>) {
        if let Some(texture) = texture {
            list.keep(texture);
        }
        let barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    // SAFETY: borrow the interface representation; the
                    // barrier must not release this non-owning copy.
                    pResource: std::mem::ManuallyDrop::new(texture.map(|resource| unsafe { std::ptr::read(resource) })),
                }),
            },
        };
        // SAFETY: the optional resource permits UAV and is retained across
        // the real Fence. Otherwise this barrier orders any UAV resource.
        unsafe { list.recording().ResourceBarrier(&[barrier]) };
    }

    pub(crate) fn encode(
        &self, list: &HostList, resources: &mut ResourcePlan,
    ) -> Result<(), BatchError> {
        match self {
            Self::Texture(copy, wanted) => resources.transition_texture(list, copy, *wanted),
            Self::NativeUav(copy) => { Self::native_uav(list, copy); Ok(()) }
            Self::Uav => { Self::emit_uav(list, None); Ok(()) }
            #[cfg(feature = "paged-title")]
            Self::DiscardColor(copy, sub) => {
                resources.record_discarded_color(copy, *sub)?;
                resources.transition_texture(list, copy, D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32)?;
                // 安全：精确整资源已转至 RENDER_TARGET，列表保留该资源。
                list.keep(&copy.texture);
                unsafe { list.recording().DiscardResource(&copy.texture, None) };
                Ok(())
            }
            #[cfg(all(feature = "paged-title", feature = "diagnostics"))]
            Self::EnhancedUav { .. } => { Self::emit_uav(list, None); Ok(()) }
        }
    }
}

/// Plane zero of an exact 2D color texture. Byte authority is scoped;
/// an operation may also borrow the exact native subresource state.
#[derive(Clone, Copy)]
pub(crate) struct ColorSubresource {
    pub(crate) index: u32,
    pub(crate) width: u64,
    pub(crate) height: u32,
    scoped: bool,
}

impl ColorSubresource {
    /// 体纹理的深度分片不是 D3D12 子资源。此路径只接纳单 mip；
    /// 所有 W 分片遵守同一份权威版本和完整资源屏障。
    pub(crate) fn single_volume(desc: D3D12_RESOURCE_DESC, mip: u32) -> Option<Self> {
        if desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D || mip != 0
            || desc.MipLevels != 1 || desc.DepthOrArraySize == 0
            || desc.Width == 0 || desc.Height == 0
            || desc.SampleDesc.Count != 1 || desc.SampleDesc.Quality != 0
            || desc.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
        {
            return None;
        }
        crate::page_texture::linear_texel_bytes(desc.Format)?;
        Some(Self { index: 0, width: desc.Width, height: desc.Height, scoped: false })
    }

    /// Multisample color attachments are whole, single-plane 2D resources.
    /// Keep this separate from byte-footprint and region-copy admission.
    pub(crate) fn multisample(desc: D3D12_RESOURCE_DESC) -> Option<Self> {
        if desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
            || desc.DepthOrArraySize != 1 || desc.MipLevels != 1
            || !supported_samples(desc.SampleDesc) || desc.SampleDesc.Count == 1
            || !desc.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)
            || desc.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
        {
            return None;
        }
        crate::page_texture::linear_texel_bytes(desc.Format)?;
        Some(Self { index: 0, width: desc.Width, height: desc.Height, scoped: false })
    }

    pub(crate) fn from_mip_slice(desc: D3D12_RESOURCE_DESC, mip: u32, slice: u32) -> Option<Self> {
        let mips = u32::from(desc.MipLevels);
        if mip >= mips || slice >= u32::from(desc.DepthOrArraySize) { return None; }
        Self::capture(desc, slice.checked_mul(mips)?.checked_add(mip)?)
    }

    pub(crate) fn is_scoped(self) -> bool { self.scoped }

    pub(crate) fn capture(desc: D3D12_RESOURCE_DESC, index: u32) -> Option<Self> {
        if desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count != 1
            || desc.SampleDesc.Quality != 0 || desc.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
        {
            return None;
        }
        let mips = u32::from(desc.MipLevels);
        let total = mips.checked_mul(u32::from(desc.DepthOrArraySize))?;
        if index >= total { return None; }
        let scoped = total != 1;
        if scoped {
            crate::page_texture::linear_texel_bytes(desc.Format)?;
        }
        let mip = index % mips;
        Some(Self {
            index, width: desc.Width.checked_shr(mip)?.max(1),
            height: desc.Height.checked_shr(mip)?.max(1), scoped,
        })
    }

    fn authority(self) -> Option<u32> { self.scoped.then_some(self.index) }

    /// Paired barriers only: the planner's persistent baseline is still
    /// whole-resource, so callers must restore it in the same recording.
    pub(crate) fn temporary_transition(
        self, resource: &ID3D12Resource,
        before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES,
    ) -> D3D12_RESOURCE_BARRIER {
        let index = self.authority().unwrap_or(D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES);
        crate::host::transition_subresource(resource, index, before, after)
    }

    pub(crate) fn allows(self, copy: &HostTexture) -> bool {
        !self.scoped || (!copy.uses_unproven_alias_backing() && !copy.has_pointer_exposure())
    }
}

#[cfg(test)]
#[test]
fn volume_depth_does_not_create_independent_array_authorities() {
    let mut desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE3D,
        Width: 64, Height: 32, DepthOrArraySize: 8, MipLevels: 1,
        Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    };
    let one = ColorSubresource::single_volume(desc, 0).unwrap();
    assert_eq!((one.index, one.width, one.height, one.authority()), (0, 64, 32, None));
    assert!(ColorSubresource::from_mip_slice(desc, 0, 1).is_none());
    assert!(ColorSubresource::single_volume(desc, 1).is_none());
    desc.MipLevels = 2;
    assert!(ColorSubresource::single_volume(desc, 0).is_none());
}

pub(crate) fn supported_samples(samples: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC) -> bool {
    samples.Quality == 0 && matches!(samples.Count, 1 | 2 | 4 | 8 | 16)
}

#[cfg(feature = "paged-title")]
pub(crate) struct TextureResolve {
    src: Arc<HostTexture>,
    dst: Arc<HostTexture>,
    format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
}

#[cfg(feature = "paged-title")]
impl TextureResolve {
    pub(crate) fn capture(
        src: Arc<HostTexture>, src_index: u32, dst: Arc<HostTexture>, dst_index: u32,
        format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    ) -> Option<Self> {
        // SAFETY: the retained resources own these descriptions.
        let (a, b) = unsafe { (src.texture.GetDesc(), dst.texture.GetDesc()) };
        ColorSubresource::multisample(a)?;
        let dest = ColorSubresource::capture(b, dst_index)?;
        if src_index != 0 || dest.is_scoped() || (a.Width, a.Height) != (b.Width, b.Height)
            || a.Format != format || b.Format != format
        {
            return None;
        }
        let resolve = Self { src, dst, format };
        resolve.exact_backing().then_some(resolve)
    }

    fn exact_backing(&self) -> bool {
        self.src.texture != self.dst.texture && !self.src.has_pointer_exposure()
            && [&self.src, &self.dst].into_iter().all(|copy| {
                copy.sparse.is_none() && !copy.uses_unproven_alias_backing()
                    && !copy.twin.get().is_some_and(Option::is_some)
            })
    }

    pub(crate) fn target(&self) -> &Arc<HostTexture> { &self.dst }

    pub(crate) fn encode(&self, host: &crate::host::Host, list: &HostList, plan: &mut ResourcePlan)
        -> Result<(), BatchError>
    {
        if !self.exact_backing() { return Err(BatchError::unsupported_texture_map("无法确认纹理映射")); }
        if !plan.authoritative(&self.src) { return Err(BatchError::TargetNotInitialized); }
        // Resource allocation alone does not establish resolve support for
        // a format. Verify the adapter's flag before issuing the command.
        if !crate::page_texture::resolve_support(host, self.format) {
            return Err(BatchError::unsupported_texture_map("无法确认纹理映射"));
        }
        plan.transition_texture(list, &self.src, D3D12_RESOURCE_STATE_RESOLVE_SOURCE.0 as u32)?;
        plan.transition_texture(list, &self.dst, D3D12_RESOURCE_STATE_RESOLVE_DEST.0 as u32)?;
        list.keep(&self.src.texture);
        list.keep(&self.dst.texture);
        // SAFETY: typed format, resource extents, samples, source authority
        // and native resolve states are proven; subresource zero is the whole target.
        unsafe { list.recording().ResolveSubresource(&self.dst.texture, 0, &self.src.texture, 0, self.format) };
        plan.record_write_at(&self.dst, None)
    }
}

/// An exact region from a placed linear source into a native target.
/// Partial uploads consume existing target authority; full ones establish it.
pub(crate) struct TextureUpload {
    source: crate::page_texture::LinearTexture,
    target: Arc<HostTexture>,
    offset: (u32, u32),
    sub: ColorSubresource,
    fills_target: bool,
    exact_backing: bool,
}

impl TextureUpload {
    #[cfg(all(feature = "paged-title", feature = "diagnostics"))]
    pub(crate) fn color_write_coverage(&self, subresource: u32) -> &'static str {
        if self.sub.index != subresource { "同对象其他子资源上传" }
        else if self.fills_target { "同子资源整面上传" }
        else { "同子资源局部上传" }
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn preserves_color(&self) -> Option<&Arc<HostTexture>> {
        (!self.fills_target).then_some(&self.target)
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn target(&self) -> &Arc<HostTexture> { &self.target }

    pub(crate) fn capture(
        source: crate::page_texture::LinearTexture, target: Arc<HostTexture>, index: u32,
        (x, y): (u32, u32), bounds: Option<[u32; 6]>,
    ) -> Option<Self> {
        let source = source.crop(bounds)?;
        let end_x = u64::from(x.checked_add(source.width)?);
        let end_y = y.checked_add(source.height)?;
        // SAFETY: the retained resource is live.
        let desc = unsafe { target.texture.GetDesc() };
        let sub = ColorSubresource::capture(desc, index)?;
        let exact_backing = !desc.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        if exact_backing && target.uses_unproven_alias_backing()
            || !sub.allows(&target)
            || target.sparse.is_some()
            || target.twin.get().is_some_and(Option::is_some)
            || end_x > sub.width || end_y > sub.height
            || desc.Format != source.format
        {
            return None;
        }
        let fills_target = x == 0 && y == 0 && end_x == sub.width && end_y == sub.height;
        Some(Self { source, target, offset: (x, y), sub, fills_target, exact_backing })
    }

    pub(crate) fn ranges(&self, ranges: &mut Vec<(u64, u64)>) -> Result<(), BatchError> {
        ranges.extend(self.source.rows()?);
        Ok(())
    }

    pub(crate) fn encode(&self, host: &crate::host::Host, list: &HostList,
        batch: &crate::page_slabs::PinnedBatch, plan: &mut ResourcePlan,
    ) -> Result<(), BatchError> {
        if !self.sub.allows(&self.target) || self.exact_backing && self.target.uses_unproven_alias_backing() {
            return Err(BatchError::unsupported_texture_map("无法确认纹理映射"));
        }
        if !self.fills_target {
            plan.require_color_subresource(host, list, batch, &self.target, self.sub)?;
        }
        plan.transition_texture(list, &self.target, D3D12_RESOURCE_STATE_COPY_DEST.0 as u32)?;
        self.source.copy_to_subresource(host, list, batch, &self.target.texture, self.sub.index, self.offset)?;
        if self.fills_target {
            plan.record_full_color_write(&self.target, self.sub)?;
        } else {
            plan.record_color_write(&self.target, self.sub)?;
        }
        Ok(())
    }
}

/// A full texture-to-buffer copy ending in pinned writable pages.
pub(crate) struct TextureReadback {
    source: Arc<HostTexture>,
    target: crate::page_texture::LinearTexture,
    sub: ColorSubresource,
    plain: bool,
}

impl TextureReadback {
    pub(crate) fn capture(source: Arc<HostTexture>, target: crate::page_texture::LinearTexture) -> Option<Self> {
        Self::capture_subresource(source, 0, target).filter(|readback| !readback.sub.scoped)
    }

    pub(crate) fn capture_subresource(
        source: Arc<HostTexture>, index: u32, target: crate::page_texture::LinearTexture,
    ) -> Option<Self> {
        // SAFETY: the retained resource is live.
        let desc = unsafe { source.texture.GetDesc() };
        let sub = ColorSubresource::capture(desc, index)?;
        let plain = desc.Flags.0 & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0
            | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0) == 0;
        if !sub.allows(&source)
            || plain && source.uses_unproven_alias_backing()
            || source.sparse.is_some()
            || source.twin.get().is_some_and(Option::is_some)
            || sub.width != u64::from(target.width) || sub.height != target.height
            || desc.Format != target.format
        {
            return None;
        }
        Some(Self { source, target, sub, plain })
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn prepare_mirror(
        &self, ranges: &[(u64, u64)], pins: &victory_memory::physical::BackingPins,
    ) -> Result<crate::resource::PreparedByteMirror, BatchError> {
        self.source.prepare_byte_mirror(ranges, pins)
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn publish_mirror(
        &self, prepared: crate::resource::PreparedByteMirror, slabs: &crate::page_slabs::PageSlabs,
    ) {
        self.source.publish_byte_mirror(prepared, slabs);
    }

    pub(crate) fn ranges(&self, ranges: &mut Vec<(u64, u64)>) -> Result<(), BatchError> {
        ranges.extend(self.target.rows()?);
        Ok(())
    }

    pub(crate) fn explicit_ranges(&self, ranges: &mut Vec<(u64, u64)>) -> Result<(), BatchError> {
        self.ranges(ranges)?;
        collect_compatible_source(&self.source, ranges)
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn pointer_source(&self) -> Option<&Arc<HostTexture>> {
        self.source.has_pointer_exposure().then_some(&self.source)
    }

    pub(crate) fn encode(
        &self, host: &crate::host::Host, list: &HostList, batch: &crate::page_slabs::PinnedBatch,
        plan: &mut ResourcePlan,
        dirty: &mut std::collections::BTreeSet<victory_memory::physical::PhysicalPage>,
        guest_copy: bool,
    ) -> Result<(), BatchError> {
        if !self.sub.allows(&self.source) || self.plain && self.source.uses_unproven_alias_backing() {
            return Err(BatchError::unsupported_texture_map("无法确认纹理映射"));
        }
        // An explicit guest command consumes the latest ordered resource,
        // including certified escaped CPU bytes if no native write superseded
        // them in this batch. Internal mirrors instead publish the existing
        // native version; never re-upload bytes into the version they expose.
        if guest_copy || self.sub.scoped {
            plan.require_color_subresource(host, list, batch, &self.source, self.sub)?;
        } else if !plan.authoritative(&self.source) {
            return Err(BatchError::TargetNotInitialized);
        }
        let before = plan.state(&self.source);
        let needed = D3D12_RESOURCE_STATE_COPY_SOURCE;
        let texture = &self.source.texture;
        if !self.sub.scoped {
            plan.transition_texture(list, &self.source, needed.0 as u32)?;
            self.target.copy_from_subresource(host, list, batch, texture, self.sub.index, dirty, guest_copy.then_some(&mut *plan))?;
            return plan.transition_texture(list, &self.source, before.0 as u32);
        }
        if self.source.twin.get().is_some_and(Option::is_some)
            || legal_state(texture, needed.0 as u32) != needed.0 as u32
        {
            return Err(BatchError::UnsupportedBarrier);
        }
        list.keep(texture);
        if before != needed {
            // SAFETY: the exact source is retained and the ordered plan
            // supplies its state before this temporary scope.
            unsafe { list.recording().ResourceBarrier(&[self.sub.temporary_transition(texture, before, needed)]) };
        }
        self.target.copy_from_subresource(host, list, batch, texture, self.sub.index, dirty, guest_copy.then_some(&mut *plan))?;
        if before != needed {
            // SAFETY: the copy leaves the selected subresource in COPY_SOURCE.
            unsafe { list.recording().ResourceBarrier(&[self.sub.temporary_transition(texture, needed, before)]) };
        }
        Ok(())
    }
}

/// An ordered copy between exact, native-authoritative color surfaces.
/// Title-backed uploads, conversion, sparse and twin surfaces are not
/// interchangeable with this operation.
pub(crate) struct TextureCopy {
    src: Arc<HostTexture>,
    dst: Arc<HostTexture>,
    area: Option<CopyArea>,
    exact_backing: bool,
    source: ColorSubresource,
    destination: ColorSubresource,
    subresources: u32,
}

struct CopyArea {
    x: u32,
    y: u32,
    source: D3D12_BOX,
    fills_destination: bool,
}

impl TextureCopy {
    #[cfg(all(feature = "paged-title", feature = "diagnostics"))]
    pub(crate) fn color_write_coverage(&self, subresource: u32) -> &'static str {
        match &self.area {
            None => "同对象整资源复制",
            Some(_) if self.destination.index != subresource => "同对象其他子资源复制",
            Some(area) if area.fills_destination => "同子资源整面复制",
            Some(_) => "同子资源局部复制",
        }
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn preserves_color(&self) -> Option<&Arc<HostTexture>> {
        self.area.as_ref().filter(|area| !area.fills_destination).map(|_| &self.dst)
    }

    #[cfg(feature = "paged-title")]
    pub(crate) fn pointer_source(&self) -> Option<&Arc<HostTexture>> {
        self.src.has_pointer_exposure().then_some(&self.src)
    }

    pub(crate) fn target(&self) -> &Arc<HostTexture> { &self.dst }

    pub(crate) fn capture(src: Arc<HostTexture>, dst: Arc<HostTexture>) -> Option<Self> {
        Self::capture_copy(src, 0, dst, 0, None)
    }

    pub(crate) fn capture_region(
        src: Arc<HostTexture>, src_index: u32, dst: Arc<HostTexture>, dst_index: u32,
        x: u32, y: u32, bounds: Option<[u32; 6]>,
    ) -> Option<Self> {
        Self::capture_copy(src, src_index, dst, dst_index, Some((x, y, bounds)))
    }

    fn capture_copy(
        src: Arc<HostTexture>, src_index: u32, dst: Arc<HostTexture>, dst_index: u32,
        region: Option<(u32, u32, Option<[u32; 6]>)>,
    ) -> Option<Self> {
        // SAFETY: the references retain both native resources.
        let (a, b) = unsafe { (src.texture.GetDesc(), dst.texture.GetDesc()) };
        let source = ColorSubresource::capture(a, src_index)?;
        let destination = ColorSubresource::capture(b, dst_index)?;
        let scoped = source.scoped || destination.scoped;
        let writable = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0;
        let exact_backing = scoped || !a.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)
            || !b.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        let same_allocation = src.texture == dst.texture;
        let separate_subresources = region.is_some() && Arc::ptr_eq(&src, &dst)
            && source.scoped && destination.scoped && source.index != destination.index;
        if same_allocation && !separate_subresources
            || !source.allows(&src) || !destination.allows(&dst)
            || (a.Flags.0 & writable == 0 || b.Flags.0 & writable == 0)
                && crate::page_texture::linear_texel_bytes(a.Format).is_none()
            || exact_backing && (src.uses_unproven_alias_backing() || dst.uses_unproven_alias_backing())
            || a.Format != b.Format
            || src.sparse.is_some() || dst.sparse.is_some()
            || src.twin.get().is_some_and(Option::is_some)
            || dst.twin.get().is_some_and(Option::is_some)
        {
            return None;
        }
        let (area, subresources) = if let Some((x, y, bounds)) = region {
            // Partial copies use texel bounds only on exact pixel formats.
            crate::page_texture::linear_texel_bytes(a.Format)?;
            let width = u32::try_from(source.width).ok()?;
            let [left, top, front, right, bottom, back] = bounds.unwrap_or([0, 0, 0, width, source.height, 1]);
            if front != 0 || back != 1 || left >= right || top >= bottom
                || right > width || bottom > source.height
            {
                return None;
            }
            let end_x = x.checked_add(right - left)?;
            let end_y = y.checked_add(bottom - top)?;
            if u64::from(end_x) > destination.width || end_y > destination.height {
                return None;
            }
            (Some(CopyArea {
                x, y,
                source: D3D12_BOX { left, top, front, right, bottom, back },
                fills_destination: x == 0 && y == 0 && u64::from(end_x) == destination.width && end_y == destination.height,
            }), 1)
        } else {
            if a.Width != b.Width || a.Height != b.Height
                || a.MipLevels != b.MipLevels || a.DepthOrArraySize != b.DepthOrArraySize
            {
                return None;
            }
            (None, u32::from(a.MipLevels).checked_mul(u32::from(a.DepthOrArraySize))?)
        };
        Some(Self { src, dst, area, exact_backing, source, destination, subresources })
    }

    /// A region carries one pair; a whole copy carries each mip/slice
    /// in native subresource order. Authority stays keyed per subresource.
    fn authorities(&self, sub: ColorSubresource) -> impl Iterator<Item = Option<u32>> {
        (0..self.subresources).map(move |offset| sub.authority().map(|first| first + offset))
    }

    pub(crate) fn ranges(&self, ranges: &mut Vec<(u64, u64)>) -> Result<(), BatchError> {
        collect_compatible_source(&self.src, ranges)
    }

    pub(crate) fn encode(
        &self, host: &crate::host::Host, list: &HostList,
        batch: &crate::page_slabs::PinnedBatch, plan: &mut ResourcePlan,
    ) -> Result<(), BatchError> {
        if !self.source.allows(&self.src) || !self.destination.allows(&self.dst)
            || self.exact_backing && (self.src.uses_unproven_alias_backing() || self.dst.uses_unproven_alias_backing()) {
            return Err(BatchError::unsupported_texture_map("无法确认纹理映射"));
        }
        for authority in self.authorities(self.source) {
            plan.require_texture_authority(host, list, batch, &self.src, authority)?;
        }
        if self.area.as_ref().is_some_and(|area| !area.fills_destination) {
            plan.require_color_subresource(host, list, batch, &self.dst, self.destination)?;
        }
        let source_state = D3D12_RESOURCE_STATE_COPY_SOURCE;
        let destination_state = D3D12_RESOURCE_STATE_COPY_DEST;
        let same_allocation = Arc::ptr_eq(&self.src, &self.dst);
        let mut scoped = crate::page_texture::ScopedTextureAccess::default();
        if same_allocation {
            if self.src.twin.get().is_some_and(Option::is_some) {
                return Err(BatchError::UnsupportedBarrier);
            }
            let before = plan.state(&self.src);
            scoped.insert(&self.src.texture, self.source, before, source_state)?;
            scoped.insert(&self.dst.texture, self.destination, before, destination_state)?;
        } else {
            plan.transition_texture(list, &self.src, source_state.0 as u32)?;
            plan.transition_texture(list, &self.dst, destination_state.0 as u32)?;
        }
        list.keep(&self.src.texture);
        list.keep(&self.dst.texture);
        let cmd = list.recording();
        if same_allocation { scoped.transition(cmd, false); }
        // SAFETY: the retained source and destination subresources are
        // distinct and in COPY_SOURCE/DEST. A region uses checked bounds.
        // A whole copy uses matching texture shapes and proven source contents.
        unsafe {
            if let Some(area) = &self.area {
                let location = |texture: &ID3D12Resource, index| D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(texture),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: index },
                };
                cmd.CopyTextureRegion(&location(&self.dst.texture, self.destination.index), area.x, area.y, 0,
                    &location(&self.src.texture, self.source.index), Some(&area.source));
            } else {
                cmd.CopyResource(&self.dst.texture, &self.src.texture);
            }
        }
        if same_allocation { scoped.transition(cmd, true); }
        if !self.destination.is_scoped() && self.area.as_ref().is_none_or(|area| area.fills_destination) {
            plan.record_full_color_write(&self.dst, self.destination)?;
        } else {
            for authority in self.authorities(self.destination) {
                plan.record_write_at(&self.dst, authority)?;
            }
        }
        Ok(())
    }
}
