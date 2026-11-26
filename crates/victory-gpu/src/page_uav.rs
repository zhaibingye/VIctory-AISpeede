//! Raw UAV clears write directly to captured slabs, with no separate
//! authoritative copy. Join aliases before creating native range views.
use std::collections::{BTreeMap, BTreeSet};

use victory_memory::{PAGE_BYTES, gpu_table::ADDRESS_LIMIT, physical::PhysicalPage};

use crate::host::{Host, HostList};
use crate::page_queue::PageDescriptors;
use crate::page_resource::{ColorSubresource, PageBarrier, ResourcePlan};
use crate::page_slabs::{BatchError, PinnedBatch, PinnedPage};

pub(crate) struct RawClear {
    pub(crate) start: u64,
    pub(crate) end: u64,
    pub(crate) values: [u32; 4],
}

impl RawClear {
    pub(crate) fn range(&self) -> Result<(u64, u64), BatchError> {
        if self.start == 0 || self.start >= self.end || self.end > ADDRESS_LIMIT
            || !self.start.is_multiple_of(16) || !self.end.is_multiple_of(4)
        {
            return Err(BatchError::Range(self.start, self.end));
        }
        Ok((self.start, self.end))
    }

    pub(crate) fn encode(
        &self,
        host: &Host,
        list: &HostList,
        pages: &PinnedBatch,
        descriptors: &mut PageDescriptors<'_>,
        resources: &mut ResourcePlan,
        dirty: &mut BTreeSet<PhysicalPage>,
    ) -> Result<(), BatchError> {
        let (mut va, end) = self.range()?;
        let mut spans: BTreeMap<_, (&PinnedPage, Vec<(u64, u64)>)> = BTreeMap::new();
        while va < end {
            let page = pages.pinned_at(va).ok_or(BatchError::NotCaptured(va))?;
            let offset = va % PAGE_BYTES;
            let len = (end - va).min(PAGE_BYTES - offset);
            spans.entry(page.physical()).or_insert_with(|| (page, Vec::new()))
                .1.push((offset, offset + len));
            va += len;
        }
        PageBarrier::Uav.encode(list, resources)?;
        let cmd = list.recording();
        // SAFETY: this heap stays bound for the clear and owned by the batch.
        unsafe { cmd.SetDescriptorHeaps(&[Some(descriptors.heap.heap.clone())]) };
        for (physical, (page, mut ranges)) in spans {
            ranges.sort_unstable();
            let mut ranges = ranges.into_iter().peekable();
            while let Some((start, mut end)) = ranges.next() {
                while let Some(&(next, last)) = ranges.peek() {
                    if next > end { break; }
                    ranges.next();
                    end = end.max(last);
                }
                let offset = page.offset() + start;
                let bytes = end - start;
                let gpu_slot = descriptors.reserve_raw_uav(page.buffer(), offset, bytes)
                    .map_err(BatchError::Native)?;
                let cpu = host.cpu_view();
                let view = crate::descriptor::host::raw_uav_desc(offset / 4, (bytes / 4) as u32);
                // SAFETY: page is pinned in UNORDERED_ACCESS, GPU/CPU
                // descriptors describe the exact same validated slot span.
                unsafe {
                    host.device.CreateUnorderedAccessView(page.buffer(), None, Some(&view), cpu);
                    cmd.ClearUnorderedAccessViewUint(descriptors.heap.gpu(gpu_slot), cpu,
                        page.buffer(), &self.values, &[]);
                }
            }
            dirty.insert(physical);
        }
        PageBarrier::Uav.encode(list, resources)
    }
}

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::*};

/// Native descriptor identity, bounds and byte authority shared by clears
/// and shader stores. Only a non-array 2D UAV can use the shader proof.
struct NativeTextureUav {
    target: std::sync::Arc<crate::resource::HostTexture>,
    format: DXGI_FORMAT,
    sub: ColorSubresource,
}

impl NativeTextureUav {
    fn capture(target: std::sync::Arc<crate::resource::HostTexture>, format: DXGI_FORMAT, mip: Option<u32>) -> Option<Self> {
        // SAFETY: the retained native allocation is live.
        let desc = unsafe { target.texture.GetDesc() };
        if desc.Format != format || desc.DepthOrArraySize != 1
            || (mip.is_none() && desc.MipLevels != 1)
            || !desc.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
            || target.sparse.is_some() || target.twin.get().is_some_and(Option::is_some)
            || target.uses_unproven_alias_backing()
        {
            return None;
        }
        let sub = ColorSubresource::from_mip_slice(desc, mip.unwrap_or(0), 0)?;
        sub.allows(&target).then_some(Self { target, format, sub })
    }

    fn check_native(&self, host: &Host) -> Result<(), BatchError> {
        if !self.sub.allows(&self.target) || self.target.uses_unproven_alias_backing()
            || self.target.twin.get().is_some_and(Option::is_some)
        {
            return Err(BatchError::UnsupportedTextureMap);
        }
        let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT { Format: self.format, ..Default::default() };
        // SAFETY: the feature query writes this exact structure.
        unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_FORMAT_SUPPORT, (&raw mut support).cast(), size_of_val(&support) as u32,
            )
        }.map_err(BatchError::Native)?;
        if !support.Support1.contains(D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW)
            || !support.Support2.contains(D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE)
        {
            return Err(BatchError::UnsupportedTextureMap);
        }
        Ok(())
    }

}

/// Exact shader UAV that names an ordered native version. It conserves
/// existing contents: a bounded shader store is not a full overwrite proof.
pub(crate) struct ShaderTextureUav(NativeTextureUav);

impl ShaderTextureUav {
    pub(crate) fn capture(target: std::sync::Arc<crate::resource::HostTexture>, format: DXGI_FORMAT, mip: Option<u32>) -> Option<Self> {
        if !matches!(format, DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_R16G16B16A16_FLOAT) {
            return None;
        }
        NativeTextureUav::capture(target, format, mip).map(Self)
    }

    pub(crate) fn target(&self) -> &std::sync::Arc<crate::resource::HostTexture> {
        &self.0.target
    }

    pub(crate) fn allows(&self, x: u32, y: u32) -> bool {
        u64::from(x) < self.0.sub.width && y < self.0.sub.height
    }

    pub(crate) fn record_write(&self, resources: &mut ResourcePlan) -> Result<(), BatchError> {
        resources.record_color_write(self.target(), self.0.sub)
    }

    pub(crate) fn prepare(
        &self, host: &Host, list: &HostList, pages: &PinnedBatch, resources: &mut ResourcePlan,
    ) -> Result<crate::page_queue::TableDescriptor, BatchError> {
        self.0.check_native(host)?;
        resources.require_color_subresource(host, list, pages, self.target(), self.0.sub)?;
        resources.transition_texture(list, self.target(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32)?;
        let texture = &self.target().texture;
        list.keep(texture);
        Ok(crate::page_queue::TableDescriptor::TextureUav(texture.clone(), self.0.format, self.0.sub.index))
    }
}

/// A clear of a single typed native texture subresource.
pub(crate) struct TextureClear {
    view: NativeTextureUav,
    values: [u32; 4],
    float: bool,
    rects: Vec<RECT>,
    replaces_all: bool,
}

impl TextureClear {
    pub(crate) fn capture(
        target: std::sync::Arc<crate::resource::HostTexture>,
        format: DXGI_FORMAT, mip: Option<u32>, values: [u32; 4], float: bool, rects: &[RECT],
    ) -> Option<Self> {
        let typed = if float {
            matches!(format, DXGI_FORMAT_R32_FLOAT | DXGI_FORMAT_R16G16B16A16_FLOAT
                | DXGI_FORMAT_R8G8B8A8_UNORM)
        } else {
            matches!(format, DXGI_FORMAT_R32_UINT | DXGI_FORMAT_R8G8B8A8_UINT)
        };
        if !typed { return None; }
        let view = NativeTextureUav::capture(target, format, mip)?;
        let (w, h) = (i32::try_from(view.sub.width).ok()?, i32::try_from(view.sub.height).ok()?);
        if rects.iter().any(|r| r.left < 0 || r.top < 0 || r.left >= r.right || r.top >= r.bottom
            || r.right > w || r.bottom > h)
        {
            return None;
        }
        let replaces_all = rects.is_empty() || rects.iter().any(|r| {
            r.left == 0 && r.top == 0 && r.right == w && r.bottom == h
        });
        Some(Self { view, values, float, rects: rects.to_vec(), replaces_all })
    }

    pub(crate) fn target(&self) -> &std::sync::Arc<crate::resource::HostTexture> {
        &self.view.target
    }

    pub(crate) fn preserves_color(&self) -> Option<&std::sync::Arc<crate::resource::HostTexture>> {
        (!self.replaces_all).then_some(self.target())
    }

    pub(crate) fn encode(
        &self, host: &Host, list: &HostList, pages: &PinnedBatch,
        descriptors: &mut PageDescriptors<'_>, resources: &mut ResourcePlan,
    ) -> Result<(), BatchError> {
        self.view.check_native(host)?;
        if !self.replaces_all {
            resources.require_color_subresource(host, list, pages, self.target(), self.view.sub)?;
        }
        resources.transition_texture(list, self.target(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32)?;
        PageBarrier::Uav.encode(list, resources)?;
        let texture = &self.target().texture;
        list.keep(texture);
        let slot = descriptors.reserve_texture_uav(texture, self.view.format, self.view.sub.index).map_err(BatchError::Native)?;
        let cpu = host.cpu_view();
        let view = crate::descriptor::host::paged_texture_uav_desc(self.view.format, self.view.sub.index);
        // SAFETY: both descriptors describe the same retained texture
        // in UNORDERED_ACCESS and every rectangle fits its subresource.
        unsafe {
            host.device.CreateUnorderedAccessView(texture, None, Some(&view), cpu);
            let cmd = list.recording();
            cmd.SetDescriptorHeaps(&[Some(descriptors.heap.heap.clone())]);
            let gpu = descriptors.heap.gpu(slot);
            if self.float {
                cmd.ClearUnorderedAccessViewFloat(gpu, cpu, texture, &self.values.map(f32::from_bits), &self.rects);
            } else {
                cmd.ClearUnorderedAccessViewUint(gpu, cpu, texture, &self.values, &self.rects);
            }
        }
        PageBarrier::Uav.encode(list, resources)?;
        resources.record_color_write(self.target(), self.view.sub)
    }
}

/// A captured descriptor prefix and its independently proven access windows.
/// Keeping the prefix preserves byte offsets in the translated shader.
pub(crate) struct BoundedBufferView {
    unordered: bool,
    start: u64,
    limit: u64,
    end: u64,
    writes: Vec<(u64, u64)>,
}

impl BoundedBufferView {
    pub(crate) fn new(start: u64, limit: u64, unordered: bool) -> Option<Self> {
        let alignment = if unordered { 16 } else { 4 };
        (start != 0 && start.is_multiple_of(alignment) && start < limit && limit <= ADDRESS_LIMIT)
            .then_some(Self { unordered, start, limit, end: start, writes: Vec::new() })
    }

    pub(crate) fn unordered(&self) -> bool { self.unordered }

    pub(crate) fn allow_access(&mut self, offset: u32, bytes: u32, write: bool) -> Option<()> {
        let start = self.start.checked_add(u64::from(offset))?;
        let end = start.checked_add(u64::from(bytes))?;
        if bytes == 0 || !offset.is_multiple_of(4) || !bytes.is_multiple_of(4)
            || end > self.limit || write && !self.unordered {
            return None;
        }
        self.end = self.end.max(end);
        if write {
            self.writes.push((start, end));
        }
        Some(())
    }

    pub(crate) fn range(&self) -> (u64, u64) { (self.start, self.end) }

    pub(crate) fn prepare_uav(
        &self, batch: &PinnedBatch, dirty: &mut BTreeSet<PhysicalPage>, stride: u32,
    ) -> Result<crate::page_queue::TableDescriptor, BatchError> {
        if !self.unordered {
            return Err(BatchError::Range(self.start, self.end));
        }
        let first = batch.pinned_at(self.start).ok_or(BatchError::NotCaptured(self.start))?;
        let page_base = self.start & !(PAGE_BYTES - 1);
        let mut at = page_base;
        while at < self.end {
            let page = batch.pinned_at(at).ok_or(BatchError::NotCaptured(at))?;
            // Native UAVs address one contiguous resource. Reject gaps
            // and repeated/aliased slots inside a single native view.
            if page.slab() != first.slab() || page.offset() != first.offset() + at - page_base {
                return Err(BatchError::Range(self.start, self.end));
            }
            at += PAGE_BYTES;
        }
        let offset = first.offset() + self.start % PAGE_BYTES;
        let bytes = self.end - self.start;
        if stride != 0 && (!offset.is_multiple_of(u64::from(stride)) || !bytes.is_multiple_of(u64::from(stride))) {
            return Err(BatchError::Range(self.start, self.end));
        }
        for &(start, end) in &self.writes {
            let mut at = start & !(PAGE_BYTES - 1);
            while at < end {
                dirty.insert(batch.pinned_at(at).ok_or(BatchError::NotCaptured(at))?.physical());
                at += PAGE_BYTES;
            }
        }
        Ok(if stride == 0 {
            crate::page_queue::TableDescriptor::Raw(first.buffer().clone(), offset, bytes, true)
        } else {
            crate::page_queue::TableDescriptor::Structured(
                first.buffer().clone(), offset, (bytes / u64::from(stride)) as u32, stride, true,
            )
        })
    }
}
