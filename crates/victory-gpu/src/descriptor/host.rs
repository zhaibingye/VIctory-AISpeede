//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed.

use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{VIEW_TAG, View, view_at};
use crate::host::host;

pub(crate) struct HostHeap {
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    /// Slots mirroring the title's; view heaps have `dynamic` more.
    count: u32,
    /// Dynamic slots past `count` (none in sampler heaps): [`DYNAMIC`]
    /// where the host takes them, else [`MIN_DYNAMIC`].
    dynamic: u32,
    sampler: bool,
    /// The guest descriptor in each title slot, including deferred textures.
    made: Mutex<Mirrored>,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    /// With it the view's own copy of title memory, if it has one.
    kept: Mutex<Vec<Kept>>,
    /// Where the search for free dynamic slots starts, and which are taken
    /// (by lists not yet reset).
    next: Mutex<(u32, Vec<bool>)>,
    /// Slot pairs kept for host copies' views (`crate::memory`), past the
    /// dynamic slots and below the null buffer views: the next never
    /// taken, those given back, and where they end.
    pairs: Mutex<(u32, Vec<u32>)>,
    pairs_end: u32,
    /// View heaps: the view-stride table translated shaders read
    /// (`victory_shader::emit::VIEW_STRIDES`), a dword per title slot: the
    /// byte stride of the buffer view there, 0 for none. Mapped at the
    /// address kept beside it.
    strides: Option<(ID3D12Resource, usize)>,
    /// View heaps: the view-mark table translated shaders read
    /// (`victory_shader::emit`, "Write marks"), two dwords per slot as the
    /// stride table has one: the mark bit and phase of the unordered buffer
    /// view there, `VIEW_MARK_NONE` for none. Mapped at the address kept
    /// beside it.
    marks: Option<(ID3D12Resource, usize)>,
    /// Per-slot guest buffer addresses and native descriptor base words.
    /// Buffer, mapping and host GPU address, allocated on first use.
    addresses: std::sync::OnceLock<Option<(ID3D12Resource, usize, u64)>>,
    /// What the title slots view that title memory feeds (`crate::bindless`).
    pub(crate) viewed: std::sync::Arc<crate::bindless::Viewed>,
}

struct Mirrored {
    keys: Vec<u64>,
    /// Placed textures whose host descriptor will be made at the next draw.
    deferred: std::collections::HashSet<u32>,
    /// Guest addresses of tables copied to per-list slots.
    table_addresses: std::collections::HashMap<u32, [u32; 2]>,
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!("{:#x} past its end (slot {})", va - start - bytes, (va - start) / step)
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
const PORTABLE_VIEWS: u32 = 1_000_000;

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run) or released.
/// 2^16 ran out seconds into the world (every world run on an RX 9070 XT,
/// each a device removal shortly after; #59).
const DYNAMIC: u32 = 1 << 18;

/// The dynamic slots of a heap whose host would not take [`DYNAMIC`] more
/// (a capped heap, NVIDIA): the title's slots keep their room first.
const MIN_DYNAMIC: u32 = 1 << 16;

/// The last slot and the null raw SRV/UAV below it must never be used
/// for a title view, transient table or a pair.
const NULL_SLOTS: u32 = 3;

fn capped_layout(limit: u32, requested: u32) -> (u32, u32) {
    // Preserve the previous capped heap's title range. Take the small
    // reservation out of transient slots at the cap, not title slots.
    let title = requested.min(limit - MIN_DYNAMIC);
    let dynamic = (limit - NULL_SLOTS - title).min(MIN_DYNAMIC);
    (title, dynamic)
}

/// Asks once, before the first shader is translated, whether the host takes
/// a view heap of `HEAP_VIEWS` slots, and if not lowers the view limit
/// translated shaders clamp their indices below to its cap (NVIDIA:
/// 1,000,000): an index past the heap's end faults the device. The title
/// makes pipelines before its heap, so the heap cannot tell first.
pub(crate) fn probe_view_limit() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let Some(host) = host() else { return };
        let views = victory_shader::emit::HEAP_VIEWS;
        let make = |count: u32| {
            // SAFETY: plain heap creation; the heap is dropped at once.
            unsafe {
                host.device.CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: count,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
            }
        };
        if let Err(e) = make(views) {
            if make(PORTABLE_VIEWS).is_ok() {
                tracing::warn!(error = %e, "host view heaps capped at {PORTABLE_VIEWS}: shaders clamp view indices below it");
                victory_shader::emit::set_view_heap_limit(PORTABLE_VIEWS);
            } else {
                tracing::warn!(error = %e, "no host view heap of {PORTABLE_VIEWS} either");
            }
        }
    });
}

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        let host = host()?;
        // The same limit as the shaders, whichever comes first.
        probe_view_limit();
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        // Every slot a shader can index (title slots, also past a capped
        // heap, and dynamic ones): shaders read the table as a root view,
        // which has no bounds, at indices clamped below `HEAP_VIEWS`.
        let strides = (!sampler)
            .then(|| host.upload_buffer(u64::from(count.max(victory_shader::emit::HEAP_VIEWS)) * 4))
            .flatten()
            .map(|(buffer, data)| (buffer, data as usize));
        let marks = (!sampler)
            .then(|| {
                let bytes = u64::from(count.max(victory_shader::emit::HEAP_VIEWS)) * 8;
                let (buffer, data) = host.upload_buffer(bytes)?;
                crate::host::name(&buffer, || "view marks".into());
                // Every slot unmarked (all ones) until a view is made there.
                // SAFETY: our mapped buffer of `bytes` bytes.
                unsafe { std::ptr::write_bytes(data, 0xFF, bytes as usize) };
                Some((buffer, data as usize))
            })
            .flatten();
        // The larger dynamic range only where the host takes it on top of
        // every title slot; else the smaller one, and the cap after that.
        // Translated shaders clamp their indices below the view limit or
        // `HEAP_SAMPLERS`: slots up to there exist and stay null.
        let (views, samplers) = (victory_shader::emit::view_heap_limit(), victory_shader::emit::HEAP_SAMPLERS);
        let (heap, count, extra) = if sampler {
            match make(count.max(samplers)) {
                Ok(heap) => (heap, count, 0),
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        } else if views < victory_shader::emit::HEAP_VIEWS {
            // Match the clamp embedded in the shaders and keep the
            // previous capped heap's title range. Reserve typed nulls
            // before allowing transient allocations.
            let heap = match make(views) {
                Ok(heap) => heap,
                Err(e) => {
                    tracing::warn!(error = %e, views, "no host view heap at its cap");
                    return None;
                }
            };
            let (title, dynamic) = capped_layout(views, count);
            if title < count {
                tracing::warn!(count, kept = title, "host view heap capped at {views}: the title's top slots are left out");
            }
            (heap, title, dynamic)
        } else if let Ok(heap) = make((count + DYNAMIC).max(views)) {
            (heap, count, DYNAMIC)
        } else {
            // Even when an oversized request fails, do not expose a heap
            // smaller than the clamp used by already-translated shaders.
            match make((count + MIN_DYNAMIC).max(views)) {
                Ok(heap) => (heap, count, MIN_DYNAMIC),
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        };
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let (gpu, total) = unsafe {
            (heap.GetGPUDescriptorHandleForHeapStart().ptr, heap.GetDesc().NumDescriptors)
        };
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(total) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        let h = HostHeap {
            heap,
            cpu,
            step,
            count,
            dynamic: extra,
            sampler,
            made: Mutex::new(Mirrored {
                keys: vec![0; count as usize],
                deferred: Default::default(),
                table_addresses: Default::default(),
            }),
            kept: Mutex::new(vec![None; (count + extra) as usize]),
            next: Mutex::new((0, vec![false; extra as usize])),
            pairs: Mutex::new((count + extra, Vec::new())),
            pairs_end: match victory_shader::emit::null_buffer_view() {
                null if !sampler && null > count + extra && null < total => null - 1,
                _ => total,
            },
            strides,
            marks,
            addresses: std::sync::OnceLock::new(),
            viewed: Default::default(),
        };
        if !sampler {
            crate::bindless::register(&h.viewed);
        }
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, total) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
        }
        tracing::debug!(ms = started.elapsed().as_millis(), "host descriptor heap made null");
        // What shaders read instead of a slot without a buffer view where
        // they take one at a computed index (`victory_shader::emit`, "Buffer
        // views at computed heap indices"): a null raw SRV, and UAV below it.
        let null = victory_shader::emit::null_buffer_view();
        if !sampler && null > count + extra && null < total {
            h.raw_view_at(null, None, 0, 0, false);
            h.raw_view_at(null - 1, None, 0, 0, true);
        } else if !sampler {
            tracing::warn!(count, total, "no room for the null buffer views: computed indices of slots without a buffer view read what is there");
        }
        Some(h)
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        const BLOCK: u32 = 1 << 16;
        let Some(host) = host() else { return Ok(()) };
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: BLOCK,
                    ..Default::default()
                })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < BLOCK {
                let n = filled.min(BLOCK - filled);
                host.device.CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = BLOCK.min(total - done);
                host.device.CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

impl HostHeap {
    pub(crate) fn dynamic_capacity(&self) -> u32 {
        self.dynamic
    }

    /// `n` consecutive free dynamic slots (view heaps only), taken until
    /// [`HostHeap::release`]: the first's index.
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        if self.sampler || n == 0 || n > self.dynamic {
            return None;
        }
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let (next, taken) = &mut *guard;
        let n = n as usize;
        // Search from `next` on, then from the start (extending past cursor by
        // n - 1 so contiguous runs crossing the cursor are not missed).
        let cursor = *next as usize;
        let end = self.dynamic as usize;
        let second_end = cursor.saturating_add(n.saturating_sub(1)).min(end);
        let starts = (cursor..end).chain(0..second_end);
        let mut run = 0;
        let mut found = None;
        let mut last = usize::MAX;
        for at in starts {
            // A run cannot wrap past the end.
            if at != last.wrapping_add(1) {
                run = 0;
            }
            last = at;
            run = if taken[at] { 0 } else { run + 1 };
            if run == n {
                found = Some(at + 1 - n);
                break;
            }
        }
        let Some(first) = found else {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                let free = taken.iter().filter(|slot| !**slot).count();
                tracing::warn!(
                    requested = n, free, dynamic = self.dynamic, mirrored = self.count,
                    "no contiguous dynamic descriptor range: a table is not made"
                );
            });
            return None;
        };
        taken[first..first + n].fill(true);
        *next = ((first + n) % self.dynamic as usize) as u32;
        Some(self.count + first as u32)
    }

    /// Two consecutive slots (view heaps only) kept until
    /// [`HostHeap::free_pair`]: past the dynamic slots where the heap has
    /// room (a host copy's views, `crate::memory`, for as long as it
    /// lives), else dynamic ones.
    pub(crate) fn pair(&self) -> Option<u32> {
        if self.sampler {
            return None;
        }
        {
            let mut pairs = self.pairs.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(index) = pairs.1.pop() {
                return Some(index);
            }
            if pairs.0 + 2 <= self.pairs_end {
                pairs.0 += 2;
                return Some(pairs.0 - 2);
            }
        }
        self.dynamic(2)
    }

    /// Gives back a pair [`HostHeap::pair`] returned.
    pub(crate) fn free_pair(&self, index: u32) {
        if index >= self.count + self.dynamic {
            self.pairs.lock().unwrap_or_else(|e| e.into_inner()).1.push(index);
        } else {
            self.release(index, 2);
        }
    }

    /// The title slots this heap has host slots for; fewer than the title's
    /// when the host caps its size.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned).
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, first: u32, n: u32) {
        let Some(at) = first.checked_sub(self.count) else {
            return;
        };
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let end = (at + n).min(self.dynamic) as usize;
        if let Some(slots) = guard.1.get_mut(at as usize..end) {
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            let views = &mut kept[first as usize..self.count as usize + end];
            for (offset, view) in views.iter_mut().enumerate() {
                self.replace_kept(first + offset as u32, view, None);
            }
            slots.fill(false);
        }
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot.
    pub(crate) fn make_at(&self, index: u32, view: &View) {
        let (kept, mark) = make_marked(view, self.cpu(index));
        if kept.is_none() {
            null_for(view, self.cpu(index));
        }
        self.keep(index, kept.flatten());
        self.set_stride(index, Some(view));
        self.set_mark(index, mark);
        self.note_table_address(index, Some(view));
    }

    /// Makes slot `index` (a dynamic one) a raw view, shader resource or
    /// unordered access (`uav`), of `bytes` of `buffer` from byte `first`
    /// (both multiples of 4), which it holds; a null one for no buffer, or
    /// for unordered access to a buffer shaders cannot write.
    pub(crate) fn raw_view_at(&self, index: u32, buffer: Option<&ID3D12Resource>, first: u64, bytes: u64, uav: bool) {
        let Some(host) = host() else { return };
        let writable = |b: &ID3D12Resource| {
            // SAFETY: a live buffer.
            unsafe { b.GetDesc() }.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
        };
        let buffer = buffer.filter(|&b| !uav || writable(b));
        // shortcut: a buffer view holds at most 2^27 elements (512 MiB raw).
        let (first, elements) = match buffer {
            Some(_) => (first / 4, (bytes / 4).clamp(1, 1 << 27) as u32),
            // A null view still names its size.
            None => (0, 1),
        };
        // SAFETY: a slot of our heap; the buffer is live (or none: a null view).
        unsafe {
            if uav {
                let desc = raw_uav_desc(first, elements);
                host.device
                    .CreateUnorderedAccessView(buffer, None, Some(&desc), self.cpu(index));
            } else {
                let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_SRV {
                            FirstElement: first,
                            NumElements: elements,
                            StructureByteStride: 0,
                            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        },
                    },
                };
                host.device
                    .CreateShaderResourceView(buffer, Some(&desc), self.cpu(index));
            }
        }
        self.keep(index, buffer.map(|b| (b.clone(), None)));
        self.write_stride(index, 4);
        self.set_mark(index, None);
        self.note_table_address(index, None);
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        self.keep(index, None);
        self.set_stride(index, None);
        self.set_mark(index, None);
        if index < self.count {
            self.viewed.set(index, None, false);
        } else {
            self.note_table_address(index, None);
        }
    }

    /// Enters the byte stride of `view` (0 for none or not a buffer view)
    /// for slot `index` in the view-stride table.
    fn set_stride(&self, index: u32, view: Option<&View>) {
        self.write_stride(index, view.map_or(0, view_stride));
    }

    fn write_stride(&self, index: u32, stride: u32) {
        let Some((_, data)) = &self.strides else { return };
        if index >= self.count.max(victory_shader::emit::HEAP_VIEWS) {
            return;
        }
        // SAFETY: a dword of the mapped table, which has one per slot below
        // the larger of the title's count and `HEAP_VIEWS`.
        unsafe { (*data as *mut u32).add(index as usize).write_volatile(stride) };
    }

    /// The view-stride table, for root SRV `VIEW_STRIDES`.
    pub(crate) fn strides(&self) -> Option<&ID3D12Resource> {
        self.strides.as_ref().map(|(buffer, _)| buffer)
    }

    /// Enters slot `index`'s view-mark entry: the mark bit and phase of the
    /// unordered buffer view there ([`crate::shadow::Shadow::mark_entry`]),
    /// none for anything else.
    pub(crate) fn set_mark(&self, index: u32, entry: Option<(u32, u32)>) {
        let Some((_, data)) = &self.marks else { return };
        if index >= self.count.max(victory_shader::emit::HEAP_VIEWS) {
            return;
        }
        let entry = entry.unwrap_or((victory_shader::emit::VIEW_MARK_NONE, 0));
        // SAFETY: two dwords of the mapped table, which has two per slot
        // below the larger of the title's count and `HEAP_VIEWS`.
        unsafe {
            let at = (*data as *mut u32).add(2 * index as usize);
            at.write_volatile(entry.0);
            at.add(1).write_volatile(entry.1);
        }
    }

    /// The view-mark table, for the root SRV of shaders marking their writes.
    pub(crate) fn marks(&self) -> Option<&ID3D12Resource> {
        self.marks.as_ref().map(|(buffer, _)| buffer)
    }

    /// Slot `index` now views `resource` (or nothing of the host's).
    fn keep(&self, index: u32, resource: Kept) {
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = kept.get_mut(index as usize) {
            self.replace_kept(index, slot, resource);
        }
    }

    fn replace_kept(&self, _index: u32, slot: &mut Kept, resource: Kept) {
        let old = std::mem::replace(slot, resource);
        #[cfg(feature = "diagnostics")]
        if let Some((old, _)) = &old
            && !self.sampler
            && slot.as_ref().map(|s| windows::core::Interface::as_raw(&s.0)) != Some(windows::core::Interface::as_raw(old))
            // SAFETY: a live resource.
            && unsafe { old.GetDesc() }.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
        {
            use windows::core::Interface;
            let left = crate::resource::life::holders(old);
            crate::resource::life::note(old.as_raw() as usize, crate::resource::life::SLOT, u64::from(_index), left);
        }
        // Title threads rewrite slots while the GPU runs a submission;
        // one of its shaders may hold the old descriptor (or read the
        // slot at an index from wrong data): what it views is freed
        // only once that work has run.
        if let Some((old, _)) = old {
            crate::host::retire(old);
        }
    }

    /// The host address of the slots' view addresses (see the field), made
    /// on first use from the slots made so far.
    pub(crate) fn view_addresses(&self) -> Option<u64> {
        if let Some(made) = self.addresses.get() {
            return made.as_ref().map(|m| m.2);
        }
        // Mirror updates hold the same lock. Publish the initial snapshot
        // before a writer can update a slot and miss the new mapping.
        let slots = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let made = self.addresses.get_or_init(|| {
            use victory_shader::emit::{VIEW_ADDRESS_BYTES, VIEW_ADDRESS_PREFIX_RECORDS};
            let count = self.count + self.dynamic;
            let records = u64::from(count) + u64::from(VIEW_ADDRESS_PREFIX_RECORDS);
            let (buffer, data) = host()?.upload_buffer(records * u64::from(VIEW_ADDRESS_BYTES))?;
            // SAFETY: the first two records are the header and zero fallback.
            unsafe {
                data.cast::<[u32; 2]>().write_unaligned([count, 0]);
                data.cast::<[u32; 2]>().add(1).write_unaligned([0, 0]);
            }
            crate::host::name(&buffer, || "view addresses".into());
            let data = data as usize;
            for (i, &key) in slots.keys.iter().enumerate() {
                let view = (key != 0)
                    .then(|| view_at(key as u32, (key >> 32) as u32))
                    .flatten();
                write_address(data, i, address_words(view.as_ref()));
            }
            for index in self.count..count {
                write_address(
                    data,
                    index as usize,
                    slots.table_addresses.get(&index).copied().unwrap_or([0, 0]),
                );
            }
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some((buffer, data, gpu))
        });
        made.as_ref().map(|m| m.2)
    }

    /// Slot `index` now holds `view` (or no view): its view address.
    fn note_address(&self, index: u32, view: Option<&View>) {
        if let Some(Some((_, data, _))) = self.addresses.get() {
            write_address(*data, index as usize, address_words(view));
        }
    }

    /// Record address metadata for a remapped table entry, even before a
    /// shader first asks for the mapped address buffer. A recycled slot is
    /// overwritten by make_at/make_null before the new list can read it.
    fn note_table_address(&self, index: u32, view: Option<&View>) {
        if self.sampler || !(self.count..self.count + self.dynamic).contains(&index) {
            return;
        }
        let words = address_words(view);
        let mut slots = self.made.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(Some((_, data, _))) = self.addresses.get() {
            write_address(*data, index as usize, words);
        } else if words != [0, 0] {
            slots.table_addresses.insert(index, words);
        } else {
            slots.table_addresses.remove(&index);
        }
    }

    /// Installs a batch-owned explicit 2D SRV in a reserved dynamic slot.
    pub(crate) fn paged_texture_at(
        &self, index: u32, texture: Option<&ID3D12Resource>, format: DXGI_FORMAT,
        view: crate::page_texture::TextureSelection,
    ) {
        let Some(host) = host() else { return };
        let mips = view.mips;
        let (dimension, fields) = match view.slices {
            Some(slices) => (D3D12_SRV_DIMENSION_TEXTURE2DARRAY, D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                    MostDetailedMip: mips.first, MipLevels: mips.count,
                    FirstArraySlice: slices.first, ArraySize: slices.count,
                    ..Default::default()
                },
            }),
            None => (D3D12_SRV_DIMENSION_TEXTURE2D, D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV { MostDetailedMip: mips.first, MipLevels: mips.count, ..Default::default() },
            }),
        };
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: format, ViewDimension: dimension,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: fields,
        };
        // SAFETY: a reserved view slot and a live exact texture, or a
        // typed null descriptor on release.
        unsafe { host.device.CreateShaderResourceView(texture, Some(&desc), self.cpu(index)) };
        self.keep(index, texture.map(|t| (t.clone(), None)));
    }

    pub(crate) fn paged_texture_uav_at(
        &self, index: u32, texture: Option<&ID3D12Resource>, format: DXGI_FORMAT, selection: crate::page_uav::TextureUavSelection,
    ) {
        let Some(host) = host() else { return; };
        let desc = paged_texture_uav_desc(format, selection);
        // SAFETY: this slot is owned by the reservation and retains the texture.
        unsafe { host.device.CreateUnorderedAccessView(texture, None, Some(&desc), self.cpu(index)) };
        self.keep(index, texture.map(|resource| (resource.clone(), None)));
    }

    /// Installs a bounded, counter-free structured buffer view.
    pub(crate) fn paged_structured_at(
        &self, index: u32, resource: Option<&ID3D12Resource>, offset: u64, elements: u32, stride: u32,
        unordered: bool,
    ) {
        let Some(host) = host() else { return };
        if unordered {
            let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_UNKNOWN,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        FirstElement: offset / u64::from(stride),
                        NumElements: elements,
                        StructureByteStride: stride,
                        ..Default::default()
                    },
                },
            };
            // SAFETY: reserved slot and a retained structured buffer, or a null descriptor.
            unsafe { host.device.CreateUnorderedAccessView(resource, None, Some(&desc), self.cpu(index)) };
            self.keep(index, resource.map(|buffer| (buffer.clone(), None)));
            return;
        }
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: offset / u64::from(stride),
                    NumElements: elements,
                    StructureByteStride: stride,
                    Flags: D3D12_BUFFER_SRV_FLAG_NONE,
                },
            },
        };
        // SAFETY: reserved slot and a retained buffer or typed null view.
        unsafe { host.device.CreateShaderResourceView(resource, Some(&desc), self.cpu(index)) };
        self.keep(index, resource.map(|buffer| (buffer.clone(), None)));
    }

    /// Makes slots `index` and `index + 1` (dynamic ones) a raw shader
    /// resource view and a raw unordered access view of the first `bytes`
    /// (a multiple of 4) of `buffer`, which they hold; null views for no
    /// buffer.
    pub(crate) fn raw_pair_at(&self, index: u32, buffer: Option<&ID3D12Resource>, bytes: u64) {
        let Some(host) = host() else { return };
        // shortcut: a buffer view holds at most 2^27 elements (512 MiB raw);
        // a longer copy's rest reads 0 through it.
        let elements = (bytes / 4).min(1 << 27) as u32;
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: 0,
                    NumElements: elements,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    NumElements: elements,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                    ..Default::default()
                },
            },
        };
        // SAFETY: slots of our heap; the buffer is live (or none: null views).
        unsafe {
            host.device
                .CreateShaderResourceView(buffer, Some(&srv), self.cpu(index));
            host.device
                .CreateUnorderedAccessView(buffer, None, Some(&uav), self.cpu(index + 1));
        }
        self.keep(index, buffer.map(|b| (b.clone(), None)));
        self.keep(index + 1, buffer.map(|b| (b.clone(), None)));
        self.write_stride(index, 4);
        self.write_stride(index + 1, 4);
        // Writes through these are title memory's, marked in its page table.
        self.set_mark(index, None);
        self.set_mark(index + 1, None);
    }

    /// The shader-visible handle of slot `index`.
    pub(crate) fn gpu(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: a live shader-visible heap.
        let start = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart() };
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: start.ptr + u64::from(index) * u64::from(self.step),
        }
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// Makes host slot `index` from the record the title has at `handle`
    /// if it changed since, and returns the view recorded there. Buffer
    /// views are made only for draws (`for_draw`), which also copy their
    /// data: a host copy of a buffer is made on first use.
    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<View> {
        if index >= self.count {
            // Diagnostic (NVIDIA heap cap): the highest title slot past the
            // host heap seen so far.
            static HIGHEST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let old = HIGHEST.fetch_max(index, std::sync::atomic::Ordering::Relaxed);
            if old == 0 || index / 65536 > old / 65536 {
                tracing::warn!(
                    index,
                    count = self.count,
                    for_draw,
                    "title slot past the host heap"
                );
            }
            return None;
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        // Generations start at 1, so a made slot's key is never 0.
        let key = u64::from(generation) << 32 | u64::from(i);
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read.
            if made.keys[index as usize] != 0 {
                made.keys[index as usize] = 0;
                made.deferred.remove(&index);
                self.make_null(index);
                self.note_address(index, None);
            }
            return None;
        };
        if made.keys[index as usize] != key || (for_draw && made.deferred.contains(&index)) {
            self.set_stride(index, Some(&view));
            self.note_address(index, Some(&view));
            // Shaders index the heap itself (`crate::bindless`): buffer views
            // are made when written.
            // shortcut: views of textures the title placed itself are made
            // when a draw reads them through a table: a host texture (and a
            // staging copy) for each of the ~20,000 the title has views of
            // ran the host out of memory (run b2).
            let later = !for_draw && is_placed_texture(&view);
            let (kept, mark) = if later {
                (None, None)
            } else {
                make_marked(&view, self.cpu(index))
            };
            // A transient placed-allocation failure leaves a null slot
            // for this draw, not a supposedly completed descriptor forever.
            let pending = kept.is_none() && placed_pending(&view);
            made.keys[index as usize] = key;
            if later || pending {
                made.deferred.insert(index);
            } else {
                made.deferred.remove(&index);
            }
            self.set_mark(index, mark);
            if kept.is_none() && !self.sampler {
                null_for(&view, self.cpu(index));
            }
            if !self.sampler {
                self.viewed.set(index, Some(&view), kept.is_some());
            }
            self.keep(index, kept.flatten());
        }
        Some(view)
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let key = *self.made.lock().unwrap_or_else(|e| e.into_inner()).keys.get(index as usize)?;
        // A view the host could not make leaves a null view, its record
        // noted all the same; it keeps no resource.
        let kept = self.kept.lock().unwrap_or_else(|e| e.into_inner()).get(index as usize).is_some_and(Option::is_some);
        if key == 0 || (!kept && !self.sampler) {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }

    /// Whether slot `index` was made from a record (tests).
    #[cfg(test)]
    pub(crate) fn made(&self, index: u32) -> bool {
        let made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        made.keys[index as usize] != 0 && !made.deferred.contains(&index)
    }
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A view description from its words, with the format as the host has it.
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

/// Exact plane-zero 2D view whose native layout can be re-created
/// without reinterpreting swizzles, clamp, or subresource selection.
pub(super) fn paged_2d_srv(words: &[u32; 10]) -> Option<(DXGI_FORMAT, crate::page_texture::TextureSelection)> {
    use crate::page_texture::{MipSelection, TextureSelection};
    let desc = srv_desc(words);
    if desc.Shader4ComponentMapping != D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING {
        return None;
    }
    let (mips, slices) = match desc.ViewDimension {
        D3D12_SRV_DIMENSION_TEXTURE2D => {
            // SAFETY: the dimension selects Texture2D.
            let v = unsafe { desc.Anonymous.Texture2D };
            if v.PlaneSlice != 0 || v.ResourceMinLODClamp != 0.0 { return None; }
            (MipSelection { first: v.MostDetailedMip, count: v.MipLevels }, None)
        }
        D3D12_SRV_DIMENSION_TEXTURE2DARRAY => {
            // SAFETY: the dimension selects Texture2DArray.
            let v = unsafe { desc.Anonymous.Texture2DArray };
            if v.PlaneSlice != 0 || v.ResourceMinLODClamp != 0.0 || v.ArraySize == u32::MAX { return None; }
            (MipSelection { first: v.MostDetailedMip, count: v.MipLevels },
                Some(MipSelection { first: v.FirstArraySlice, count: v.ArraySize }))
        }
        _ => return None,
    };
    Some((desc.Format, TextureSelection { mips, slices }))
}

pub(crate) fn paged_texture_uav_desc(
    format: DXGI_FORMAT, selection: crate::page_uav::TextureUavSelection,
) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let (dimension, fields) = if let Some(depth) = selection.volume {
        (D3D12_UAV_DIMENSION_TEXTURE3D, D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Texture3D: D3D12_TEX3D_UAV {
                MipSlice: selection.mip, FirstWSlice: depth.first, WSize: depth.count,
            },
        })
    } else if let Some(slices) = selection.slices {
        (D3D12_UAV_DIMENSION_TEXTURE2DARRAY, D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Texture2DArray: D3D12_TEX2D_ARRAY_UAV {
                MipSlice: selection.mip, FirstArraySlice: slices.first, ArraySize: slices.count, PlaneSlice: 0,
            },
        })
    } else {
        (D3D12_UAV_DIMENSION_TEXTURE2D, D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_UAV { MipSlice: selection.mip, PlaneSlice: 0 },
        })
    };
    D3D12_UNORDERED_ACCESS_VIEW_DESC { Format: format, ViewDimension: dimension, Anonymous: fields }
}

fn exact_texture_uav(view: &View) -> Option<(
    std::sync::Arc<crate::resource::HostTexture>, DXGI_FORMAT, Option<crate::page_uav::TextureUavSelection>,
)> {
    use crate::page_uav::TextureUavSelection;
    let View::Unordered(resource, 0, words) = view else { return None; };
    let resource = *resource as crate::object::This;
    if crate::resource::stands_in(resource) {
        return None;
    }
    let (format, selection) = if let Some(words) = words {
        let desc = uav_desc(words);
        let selection = match desc.ViewDimension {
            D3D12_UAV_DIMENSION_TEXTURE2D => {
                // SAFETY: the dimension selects Texture2D.
                let tex = unsafe { desc.Anonymous.Texture2D };
                if tex.PlaneSlice != 0 { return None; }
                TextureUavSelection { mip: tex.MipSlice, slices: None, volume: None }
            }
            D3D12_UAV_DIMENSION_TEXTURE2DARRAY => {
                // SAFETY: the dimension selects Texture2DArray.
                let tex = unsafe { desc.Anonymous.Texture2DArray };
                if tex.PlaneSlice != 0 || tex.ArraySize == 0 || tex.ArraySize == u32::MAX { return None; }
                TextureUavSelection {
                    mip: tex.MipSlice,
                    slices: Some(crate::page_texture::MipSelection { first: tex.FirstArraySlice, count: tex.ArraySize }),
                    volume: None,
                }
            }
            D3D12_UAV_DIMENSION_TEXTURE3D => {
                // SAFETY: the dimension selects Texture3D.
                let tex = unsafe { desc.Anonymous.Texture3D };
                TextureUavSelection {
                    mip: tex.MipSlice,
                    slices: None,
                    volume: Some(crate::page_texture::MipSelection { first: tex.FirstWSlice, count: tex.WSize }),
                }
            }
            _ => return None,
        };
        (desc.Format, Some(selection))
    } else {
        (DXGI_FORMAT(crate::resource::title_format(resource)? as i32), None)
    };
    Some((crate::resource::host_copy(resource)?, format, selection))
}

pub(crate) fn paged_shader_texture_uav(view: &View) -> Option<crate::page_uav::ShaderTextureUav> {
    let (target, format, selection) = exact_texture_uav(view)?;
    crate::page_uav::ShaderTextureUav::capture(target, format, selection)
}

pub(crate) fn paged_texture_clear(
    view: &View, values: [u32; 4], float: bool, rects: &[windows::Win32::Foundation::RECT],
) -> Option<crate::page_uav::TextureClear> {
    let (target, format, selection) = exact_texture_uav(view)?;
    crate::page_uav::TextureClear::capture(target, format, selection, values, float, rects)
}

pub(crate) fn raw_uav_desc(first: u64, elements: u32) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_UAV {
                FirstElement: first,
                NumElements: elements,
                Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                ..Default::default()
            },
        },
    }
}

/// 表内常量缓冲区按原始字节绑定；整个可访问前缀由页批次捕获。
pub(crate) fn paged_table_constant_range(view: &View) -> Option<(u64, u64)> {
    let &View::Constant(address, size) = view else { return None; };
    if address == 0 || !address.is_multiple_of(16) || size == 0 || !size.is_multiple_of(4) {
        return None;
    }
    let end = address.checked_add(u64::from(size))?;
    (end <= victory_memory::gpu_table::ADDRESS_LIMIT).then_some((address, end))
}

pub(crate) fn paged_raw_srv_range(view: &View) -> Option<(u64, u64)> {
    let View::Shader(resource, Some(words)) = view else { return None; };
    let desc = srv_desc(words);
    if desc.Format != DXGI_FORMAT_R32_TYPELESS || desc.ViewDimension != D3D12_SRV_DIMENSION_BUFFER
        || desc.Shader4ComponentMapping != D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING
    {
        return None;
    }
    // SAFETY: the dimension selects Buffer.
    let raw = unsafe { desc.Anonymous.Buffer };
    if raw.Flags != D3D12_BUFFER_SRV_FLAG_RAW || raw.StructureByteStride != 0 {
        return None;
    }
    checked_buffer_view(*resource, raw.FirstElement, raw.NumElements, 4)
}

pub(crate) fn paged_raw_range(view: &View) -> Option<(u64, u64)> {
    let View::Unordered(resource, 0, Some(words)) = view else { return None };
    let desc = uav_desc(words);
    if desc.Format != DXGI_FORMAT_R32_TYPELESS || desc.ViewDimension != D3D12_UAV_DIMENSION_BUFFER {
        return None;
    }
    // SAFETY: the dimension selects Buffer.
    let raw = unsafe { desc.Anonymous.Buffer };
    if raw.Flags != D3D12_BUFFER_UAV_FLAG_RAW || raw.StructureByteStride != 0 || raw.CounterOffsetInBytes != 0 {
        return None;
    }
    checked_buffer_view(*resource, raw.FirstElement, raw.NumElements, 4)
}

pub(crate) fn paged_structured_srv_range(view: &View, stride: u32) -> Option<(u64, u64)> {
    let View::Shader(resource, Some(words)) = view else { return None; };
    let desc = srv_desc(words);
    if desc.Format != DXGI_FORMAT_UNKNOWN || desc.ViewDimension != D3D12_SRV_DIMENSION_BUFFER
        || desc.Shader4ComponentMapping != D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING
    {
        return None;
    }
    // SAFETY: the checked view discriminator selects Buffer.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride != stride || buffer.Flags != D3D12_BUFFER_SRV_FLAG_NONE {
        return None;
    }
    checked_buffer_view(*resource, buffer.FirstElement, buffer.NumElements, stride)
}

pub(crate) fn paged_structured_uav_range(view: &View, stride: u32) -> Option<(u64, u64)> {
    let View::Unordered(resource, 0, Some(words)) = view else { return None; };
    let desc = uav_desc(words);
    if desc.Format != DXGI_FORMAT_UNKNOWN || desc.ViewDimension != D3D12_UAV_DIMENSION_BUFFER {
        return None;
    }
    // SAFETY: the checked view dimension selects Buffer.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride != stride
        || buffer.Flags != D3D12_BUFFER_UAV_FLAG_NONE || buffer.CounterOffsetInBytes != 0
    {
        return None;
    }
    checked_buffer_view(*resource, buffer.FirstElement, buffer.NumElements, stride)
}

fn checked_buffer_view(resource: usize, first: u64, elements: u32, stride: u32) -> Option<(u64, u64)> {
    let (base, size) = crate::resource::buffer_bytes(resource as crate::object::This)?;
    let offset = first.checked_mul(u64::from(stride))?;
    let len = u64::from(elements) * u64::from(stride);
    let stop = offset.checked_add(len)?;
    if len == 0 || stop > size || base.is_null() {
        return None;
    }
    Some(((base as u64).checked_add(offset)?, (base as u64).checked_add(stop)?))
}

pub(crate) fn paged_raw_clear(view: &View, values: [u32; 4]) -> Option<crate::page_uav::RawClear> {
    let (start, end) = paged_raw_range(view)?;
    let clear = crate::page_uav::RawClear { start, end, values };
    clear.range().ok()?;
    Some(clear)
}

fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// The byte stride of a buffer view (its element size, at least 1), 0 for
/// other views: what the view-stride table holds for it.
fn view_stride(view: &View) -> u32 {
    let (unordered, words): (bool, &[u32; 10]) = match view {
        // Read as raw buffers (`make`).
        View::Constant(..) | View::NullConstant => return 4,
        View::Shader(_, Some(desc)) => (false, desc),
        View::Unordered(_, _, Some(desc)) => (true, desc),
        View::Placed(unordered, resource, desc) if resource[0] == 1 => {
            match desc[..10].try_into() {
                Ok(words) => (*unordered, words),
                Err(_) => return 0,
            }
        }
        _ => return 0,
    };
    // The view dimension: buffer (1) for shader and unordered views alike.
    if words[1] != 1 {
        return 0;
    }
    let element = if unordered {
        let desc = uav_desc(words);
        // SAFETY: a buffer view's union member.
        let buffer = unsafe { desc.Anonymous.Buffer };
        if buffer.StructureByteStride > 0 {
            u64::from(buffer.StructureByteStride)
        } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
            4
        } else {
            crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
        }
    } else {
        element_size(&srv_desc(words))
    };
    // Nonzero: shaders taking views at computed indices read the null buffer
    // view where the table holds 0 (`victory_shader::emit`).
    element.max(1) as u32
}

/// A buffer view of title memory at `address` on its host copy: the first
/// element moves by the copy's offset.
/// Bytes per element of a buffer view.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// Two dwords per title slot. RDNA 2 ISA §§8.1.8 and 8.2.6 define a
/// buffer's byte address and an image's base in 256-byte units.
fn address_words(view: Option<&View>) -> [u32; 2] {
    if let Some((address, _)) = view.and_then(buffer_range) {
        return [address as u32, (address >> 32) as u32 & 0xFFFF];
    }
    let base = match view {
        Some(View::Constant(address, _)) => *address as u32,
        Some(View::Shader(r, _) | View::Unordered(r, _, _)) => {
            let resource = *r as crate::object::This;
            if let Some((address, _)) = crate::resource::texture_memory(resource) {
                ((address as u64) >> 8) as u32
            } else {
                crate::resource::buffer_bytes(resource)
                    .map_or(0, |(address, _)| address as usize as u32)
            }
        }
        Some(View::Placed(_, resource, desc)) if resource[0] != 1 => {
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            (address >> 8) as u32
        }
        _ => 0,
    };
    if base == 0 { [0, 0] } else { [base, victory_shader::emit::VIEW_BASE_ONLY] }
}

fn write_address(data: usize, index: usize, words: [u32; 2]) {
    const { assert!(size_of::<[u32; 2]>() == victory_shader::emit::VIEW_ADDRESS_BYTES as usize) };
    // SAFETY: the mapped table holds one two-dword record per title slot
    // after the fixed prefix; callers supply a live title-slot index.
    unsafe {
        (data as *mut [u32; 2])
            .add(victory_shader::emit::VIEW_ADDRESS_PREFIX_RECORDS as usize + index)
            .write_unaligned(words)
    };
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => (
            crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The host copy a buffer view's bytes are in (the one its host view
/// reads, [`crate::shadow::for_view`]) and the view's title range.
pub(crate) fn view_copy(view: &View) -> Option<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)> {
    let (address, len) = buffer_range(view)?;
    let (shadow, _) = crate::shadow::for_view(address, len, u64::from(view_stride(view)))?;
    Some((shadow, address, len))
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32)?.0
    };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The host resource a slot made by [`make`] views, if any, and the view's
/// own copy of title memory ([`crate::shadow::for_view`]): the slot holds
/// it, so that draws reading the view find and refresh that copy.
type Kept = Option<(ID3D12Resource, Option<std::sync::Arc<crate::shadow::Shadow>>)>;

/// Buffer views refused so far (diagnostics: the first are logged).
#[cfg(feature = "diagnostics")]
static REFUSED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// A structured view's bytes (`bytes` from byte `start` of its host copy)
/// as a raw view: its first dword and its dwords, if they start on 16 bytes
/// (a raw view's alignment), are whole dwords and not too many for one view.
///
/// Translated shaders read untyped buffers as raw bytes (scalar loads at a
/// byte address, raw loads at `element * stride + offset`). Through a
/// structured host view such a read gives zeros on NVIDIA (an RTX 5090:
/// byte 20 of a 16-byte-stride view read 0), while structured reads
/// through a raw view are right; so the host's view is raw where it can be.
/// The view-stride table keeps the title's stride. Ported from
/// fxceless/victory-gta6 (GPL-3.0), commit 578d459c, `raw_window`.
fn raw_window(start: u64, bytes: u64) -> Option<(u64, u32)> {
    if !start.is_multiple_of(16) || !bytes.is_multiple_of(4) {
        return None;
    }
    let dwords = u32::try_from(bytes / 4).ok().filter(|&d| d <= 1 << 27)?;
    Some((start / 4, dwords))
}

fn buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    if element == 0 {
        once!(len, "buffer view of elements without a size");
        return None;
    }
    // The copy its own bytes are in, its first element counted from there.
    let start = address + buffer.FirstElement * element;
    let Some((shadow, own)) =
        crate::shadow::for_view(start, u64::from(buffer.NumElements) * element, element)
    else {
        #[cfg(feature = "diagnostics")]
        if REFUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
            tracing::info!(start = format_args!("{start:#x}"), len, element, count = buffer.NumElements, "refused buffer view: no host copy");
        }
        return None;
    };
    buffer.FirstElement = shadow.offset(start) / element;
    let (first, count) = (buffer.FirstElement, buffer.NumElements);
    if !in_buffer(&shadow.buffer, first, count, element) {
        #[cfg(feature = "diagnostics")]
        if REFUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
            tracing::info!(start = format_args!("{start:#x}"), first, count, element, "refused buffer view: empty or past its copy");
        }
        once!(first, count, element, "buffer view empty or past its host copy");
        return None;
    }
    if desc.Format == DXGI_FORMAT_UNKNOWN
        && buffer.StructureByteStride > 0
        && let Some((first, count)) = raw_window(first * element, u64::from(count) * element)
    {
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
        buffer.FirstElement = first;
        buffer.NumElements = count;
        buffer.StructureByteStride = 0;
        buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&shadow.buffer, Some(&desc), at)
    };
    Some(Some((shadow.buffer.clone(), own.then_some(shadow))))
}

/// Whether `count` elements of `element` bytes from element `first` lie in
/// `buffer`. The console takes a view that is empty or runs past its
/// memory; the host removes the device for one.
fn in_buffer(buffer: &ID3D12Resource, first: u64, count: u32, element: u64) -> bool {
    // SAFETY: a live buffer.
    within(unsafe { buffer.GetDesc() }.Width, first, count, element)
}

fn within(width: u64, first: u64, count: u32, element: u64) -> bool {
    count > 0 && first.saturating_add(u64::from(count)).saturating_mul(element) <= width
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    if element == 0 {
        once!(len, "unordered buffer view of elements without a size");
        return None;
    }
    let start = address + buffer.FirstElement * element;
    let (shadow, own) = crate::shadow::for_view(start, u64::from(buffer.NumElements) * element, element)?;
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    buffer.FirstElement = shadow.offset(start) / element;
    buffer.CounterOffsetInBytes = 0;
    let (first, count) = (buffer.FirstElement, buffer.NumElements);
    if !in_buffer(&shadow.buffer, first, count, element) {
        once!(first, count, element, "unordered buffer view empty or past its host copy");
        return None;
    }
    if desc.Format == DXGI_FORMAT_UNKNOWN
        && buffer.StructureByteStride > 0
        && let Some((first, count)) = raw_window(first * element, u64::from(count) * element)
    {
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
        buffer.FirstElement = first;
        buffer.NumElements = count;
        buffer.StructureByteStride = 0;
        buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateUnorderedAccessView(&shadow.buffer, None::<&ID3D12Resource>, Some(&desc), at)
    };
    MARK.with(|m| m.set(shadow.mark_entry(start)));
    Some(Some((shadow.buffer.clone(), own.then_some(shadow))))
}

thread_local! {
    /// The view-mark entry of the unordered buffer view [`make`] last made
    /// on this thread ([`make_marked`]).
    static MARK: std::cell::Cell<Option<(u32, u32)>> = const { std::cell::Cell::new(None) };
}

/// [`make`], and the view-mark entry of the view if it is an unordered
/// buffer view whose writes shaders mark (`victory_shader::emit`, "Write
/// marks").
fn make_marked(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> (Option<Kept>, Option<(u32, u32)>) {
    MARK.with(|m| m.set(None));
    let kept = make(view, at);
    (kept, MARK.with(std::cell::Cell::take))
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in a format neither of the
/// resource's family nor among the formats it was made castable to
/// (`castable`, `resource::castable`). A view of a typed texture in another
/// format of its family takes the texture's format unless it is castable.
/// shortcut: such a view (sRGB of UNORM, say, on a texture the host made
/// without castable formats) reads without its conversion.
fn fits(texture: &ID3D12Resource, castable: &[u32], mut format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    use crate::resource::{DEPTH_FORMATS, family};
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    // Every family has more than one format, its typeless one first.
    let typeless = family(own) == own && family(own + 1) == own;
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    if view == 0 {
        return !typeless && (!unordered || typed_store(own));
    }
    let cast = castable.contains(&view);
    if family(view) != family(own) && !cast {
        once!(view, own, "view format of another family");
        return false;
    }
    let target = if !typeless && !cast { own } else { view };
    if unordered && !typed_store(target) {
        once!(target, own, "an unordered view in a format the host cannot store");
        return false;
    }
    if let Some(format) = format
        && !typeless
        && !cast
    {
        *format = d.Format;
    }
    true
}

/// Whether a view without a description of `resource` (a title texture)
/// may take its host texture's defaults: not when the host texture has
/// another format than the title's (typeless, or shared with a title
/// texture of another format of the same memory).
/// shortcut: such views are null; spell the description out (dimension,
/// mips) if the title makes them.
fn default_view(resource: crate::object::This) -> bool {
    let other = crate::resource::view_format(resource);
    if other.is_some() {
        once!(format = other, "a view without a description of a texture whose host texture has another format");
    }
    other.is_none()
}

/// Whether the host stores to typed unordered views of `format`.
pub(crate) fn typed_store(format: u32) -> bool {
    static KNOWN: Mutex<Vec<(u32, bool)>> = Mutex::new(Vec::new());
    let mut known = KNOWN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&(_, ok)) = known.iter().find(|(f, _)| *f == format) {
        return ok;
    }
    let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: DXGI_FORMAT(format as i32),
        ..Default::default()
    };
    let ok = host().is_some_and(|h| {
        // SAFETY: a plain query into a struct of its size.
        unsafe {
            h.device.CheckFeatureSupport(
                D3D12_FEATURE_FORMAT_SUPPORT,
                (&raw mut support).cast(),
                size_of_val(&support) as u32,
            )
        }
        .is_ok()
            && support.Support1.0 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW.0 != 0
    });
    known.push((format, ok));
    ok
}

/// Makes the host descriptor for `view` at `at` (in any host view heap):
/// the host resource it views, if it can.
pub(crate) fn make_view(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<ID3D12Resource> {
    make(view, at).flatten().map(|k| k.0)
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    match view {
        &View::Constant(address, size) => {
            // Constant buffers in the title's slots, through the heap itself
            // or a table, are read as raw buffers by translated shaders
            // (`victory_shader::emit`, "Constant buffers in the heap"): the
            // console needs 16-byte alignment, a host constant buffer view
            // 256, which most of the title's do not have on the host.
            let shadow = crate::shadow::find(address, u64::from(size))?;
            let bytes = u64::from(size).min(shadow.left(address));
            let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        NumElements: (bytes / 4) as u32,
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        ..Default::default()
                    },
                },
            };
            buffer_srv(address, bytes, desc, at)
        }
        View::NullConstant => {
            // A buffer read through a heap index takes a raw SRV on the host
            // even when the title called it a constant buffer, so the null
            // that stands in for it is the same null buffer view.
            null_buffer_view(at);
            Some(None)
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let Some(copy) = crate::resource::host_copy(resource) {
                let texture = &copy.texture;
                let mut desc = desc.as_ref().map(srv_desc);
                if let Some(d) = &mut desc {
                    d.Format.0 = crate::resource::pc_view_format(resource, d.Format.0 as u32) as i32;
                }
                if desc.is_none() && !default_view(resource) {
                    return None;
                }
                if let Some(sparse) = &copy.sparse {
                    let srv = desc.as_mut()?;
                    // SAFETY: The COM resource is live for the duration of this description query.
                    let mips = u32::from(unsafe { texture.GetDesc() }.MipLevels);
                    crate::sparse::resident_srv(sparse, texture, srv, mips)?;
                }
                if !fits(texture, &copy.castable, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Some(Some((texture.clone(), None)))
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at)
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some(copy) = crate::resource::host_copy(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at)
                    }
                    _ => None,
                };
            };
            let texture = &copy.texture;
            // A depth texture's unordered views write its twin.
            // SAFETY: a live resource.
            if unsafe { texture.GetDesc() }.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 != 0 {
                return crate::twin::uav(resource, &copy, desc.as_ref(), at)
                    .map(|twin| Some((twin, None)));
            }
            let mut desc = desc.as_ref().map(uav_desc);
            if let Some(d) = &mut desc {
                d.Format.0 = crate::resource::pc_view_format(resource, d.Format.0 as u32) as i32;
            }
            if desc.is_none() && !default_view(resource) {
                return None;
            }
            if !fits(texture, &copy.castable, desc.as_mut().map(|d| &mut d.Format), true) {
                return None;
            }
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Some(Some((texture.clone(), None)))
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words);
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make(&view, at);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                // shortcut: unordered access to a texture the title placed
                // itself is not mirrored; give it a resource as shader
                // views of such textures get when a dispatch writes one.
                once!(dimension = resource[0], "placed unordered texture view not mirrored");
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at)
            } else {
                buffer_srv(address, len, srv_desc(words), at)
            }
        }
    }
}

/// Whether `view` is a view of a texture the title placed in memory itself.
fn is_placed_texture(view: &View) -> bool {
    matches!(view, View::Placed(_, resource, _) if resource[0] != 1)
}

fn placed_pending(view: &View) -> bool {
    let View::Placed(false, resource, desc) = view else {
        return false;
    };
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    crate::resource::placement_pending(address, resource)
}

/// A shader view of a texture the title placed in memory itself, as a view
/// of the resource made for that memory (on first use).
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    const BUFFER: u32 = 1;
    let View::Placed(false, resource, desc) = view else {
        return None;
    };
    if resource[0] == BUFFER {
        return None;
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let words: [u32; 10] = desc[..10].try_into().ok()?;
    let texture = crate::resource::texture_at(address, resource, &srv_desc(&words))?;
    Some(View::Shader(texture as usize, Some(words)))
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Unordered views: whether a buffer, and the title's view dimension.
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it).
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return null_buffer_view(at);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => return null_buffer_view(at),
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        View::Shader(_, Some(desc)) => return null_like(desc, at),
        View::Placed(false, _, desc) => {
            return null_like(desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        // Read as raw buffers, as made (`make`).
        View::Constant(..) | View::NullConstant => return null_buffer_view(at),
        _ => return null_view(at),
    };
    let desc = if buffer {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        }
    } else {
        null_uav_desc(dimension)
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D), 2D if it names none: a 2D null view where the shader writes a 3D
/// texture is a mismatch GPU-based validation reports (the world's wind
/// field, CS_CopyWindFieldBeforeUpdateWithSmooth, 2026-11-24).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capped_heap_keeps_title_slots_and_separates_the_null_views() {
        let (title, dynamic) = capped_layout(PORTABLE_VIEWS, 1_108_259);
        assert_eq!(title, 934_464);
        assert_eq!(dynamic, 65_533);
        assert_eq!(title + dynamic, PORTABLE_VIEWS - 3);
        assert_eq!(capped_layout(PORTABLE_VIEWS, 128), (128, MIN_DYNAMIC));
    }

    #[test]
    #[ignore = "run alone with VICTORY_D3D12_WARP=1"]
    fn placed_texture_failure_defers_new_slots_without_blocking_an_existing_view() {
        use crate::resource::{allow_placement_retry, REFUSE_NEXT_TEXTURE};
        assert!(crate::switch::on("VICTORY_D3D12_WARP"));
        allow_placement_retry();
        let heap = HostHeap::new(false, 8).expect("WARP descriptor heap");
        // The placed map retains descriptions in this dedicated test
        // process. These addresses are stable and never read by the test.
        let handle = || {
            let address = Box::leak(Box::new([0u8; 65536])).as_mut_ptr() as u64;
            let resource = [3, 0, 0, 0, 64, 0, 64, 0x1_0001, 28, 1, 0, 0, 0, 0];
            let mut words = [0; 16];
            (words[0], words[1], words[2], words[5], words[10], words[11]) = (
                28, 4, 5768, 1, address as u32, (address >> 32) as u32,
            );
            let mut slot = Box::new([0u32; 8]);
            super::super::record(slot.as_mut_ptr() as u64, View::Placed(false, resource, words));
            slot
        };
        let live = handle();
        let failed = handle();
        let new = handle();
        let at = |h: &[u32; 8]| h.as_ptr() as u64;
        assert!(heap.mirror(0, at(&live), true).is_some() && heap.made(0));

        REFUSE_NEXT_TEXTURE.with(|refuse| refuse.set(true));
        assert!(heap.mirror(1, at(&failed), true).is_some() && !heap.made(1));
        crate::resource::hold_placement_pressure_for_test();
        assert!(heap.mirror(2, at(&new), true).is_some() && !heap.made(2));
        // A new view of an existing backing must still be made normally.
        assert!(heap.mirror(3, at(&live), true).is_some() && heap.made(3));

        allow_placement_retry();
        assert!(heap.mirror(1, at(&failed), true).is_some() && heap.made(1));
        assert!(heap.mirror(2, at(&new), true).is_some() && heap.made(2));
    }


#[test]
    fn descriptor_base_word_tracks_the_guest_view_address() {
        let address = 0x12_3456_7800u64;
        let mut resource = [0; 14];
        let mut desc = [0; 16];
        desc[10] = address as u32;
        desc[11] = (address >> 32) as u32;
        resource[0] = 3; // 2D texture
        let image = View::Placed(false, resource, desc);
        let image_words = [0x1234_5678, victory_shader::emit::VIEW_BASE_ONLY];
        assert_eq!(address_words(Some(&image)), image_words);
        resource[0] = 1; // Buffer SRV with a nonzero FirstElement
        desc[0] = DXGI_FORMAT_R32_TYPELESS.0 as u32;
        desc[1] = D3D12_SRV_DIMENSION_BUFFER.0 as u32;
        desc[4] = 5;
        desc[6] = 16;
        desc[8] = D3D12_BUFFER_SRV_FLAG_RAW.0 as u32;
        let buffer = View::Placed(false, resource, desc);
        assert_eq!(address_words(Some(&buffer)), [0x3456_7814, 0x12]);
        assert_eq!(address_words(None), [0, 0]);
    }

    #[test]
    fn a_reused_title_slot_publishes_the_native_base_word() {
        use crate::descriptor as d;
        use victory_shader::emit::VIEW_BASE_ONLY;

        let Some(device) = host() else { return };
        let submit = || {
            let list = device.new_list().expect("command list");
            list.close();
            device.execute(&[&list]);
        };
        let desc = [0u32, 2, 1, 0, 0, 0, 0, 0];
        let mut object = std::ptr::null();
        let null = std::ptr::null();
        assert_eq!(
            d::create_heap(null, desc.as_ptr().cast(), &d::HEAP_IIDS[0], &mut object),
            crate::object::S_OK
        );
        let title = d::heap(object);
        let heap = title.host().expect("host view heap");
        let slot = title.start + u64::from(d::increment(d::VIEWS_KIND).unwrap());
        let pool = crate::watch::tests::pool(1);
        let address = pool as u64;
        crate::shadow::register(pool, address, crate::watch::PAGE as u64);
        let neighbour = address + 256;
        d::create_cbv(null, &[neighbour as u32, (neighbour >> 32) as u32, 256, 0], title.start);
        let mut image = [0; 14];
        image[0] = 3;
        let image_address = address + 512;
        let mut desc = [0; 16];
        desc[0] = DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32;
        desc[1] = D3D12_SRV_DIMENSION_TEXTURE2D.0 as u32;
        desc[2] = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        desc[5] = 1;
        desc[10] = image_address as u32;
        desc[11] = (image_address >> 32) as u32;
        d::create_placed_srv(null, &image, &desc, slot);
        submit();
        let dynamic = heap.dynamic(1).expect("one table slot");
        heap.make_at(dynamic, &View::Constant(neighbour, 256));
        heap.view_addresses().expect("mapped view addresses");
        let data = heap.addresses.get().unwrap().as_ref().unwrap().1 as *const [u32; 2];
        let check = |want| {
            // SAFETY: a header, a fallback record and two live title entries.
            unsafe {
                assert_eq!(data.read_volatile(), [heap.count + heap.dynamic, 0]);
                assert_eq!(data.add(1).read_volatile(), [0, 0]);
                assert_eq!(data.add(2).read_volatile(), [neighbour as u32, VIEW_BASE_ONLY]);
                assert_eq!(data.add(3).read_volatile(), want);
            }
        };
        check([(image_address >> 8) as u32, VIEW_BASE_ONLY]);
        let address_of_dynamic = || {
            // SAFETY: the allocated table slot is covered by the header.
            unsafe { data.add(2 + dynamic as usize).read_volatile() }
        };
        assert_eq!(address_of_dynamic(), [neighbour as u32, VIEW_BASE_ONLY]);
        heap.make_null(dynamic);
        assert_eq!(address_of_dynamic(), [0, 0]);
        heap.release(dynamic, 1);

        // Clear a deferred image before any draw materializes its texture.
        // SAFETY: slot 1 of the live title heap.
        unsafe { (slot as *mut [u32; 3]).write_unaligned([0; 3]) };
        heap.mirror(1, slot, false);
        submit();
        check([0, 0]);

        let mut buffer = [0; 14];
        buffer[0] = 1;
        buffer[4] = 256;
        desc[10] = address as u32;
        desc[11] = (address >> 32) as u32;
        desc[0] = DXGI_FORMAT_R32_TYPELESS.0 as u32;
        desc[1] = D3D12_SRV_DIMENSION_BUFFER.0 as u32;
        desc[4] = 4; // Raw D3D12 SRVs start on a 16-byte boundary.
        desc[5] = 0;
        desc[6] = 16;
        desc[8] = D3D12_BUFFER_SRV_FLAG_RAW.0 as u32;
        d::create_placed_srv(null, &buffer, &desc, slot);
        submit();
        let first = address + 4 * 4;
        check([first as u32, (first >> 32) as u32 & 0xFFFF]);
        d::release(object);
        crate::shadow::unregister(pool, address);
        crate::watch::tests::free(pool);
    }

    /// The world's 16-byte view of 0 elements removed the device.
    #[test]
    fn empty_and_overlong_buffer_views_are_refused() {
        assert!(within(64, 0, 4, 16));
        assert!(within(64, 3, 1, 16));
        assert!(!within(64, 0, 0, 16));
        assert!(!within(64, 3, 2, 16));
        assert!(!within(64, u64::MAX, 1, 16));
    }

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// A null description clears the host CBV slot without removing the device.
    #[test]
    fn null_constant_buffer_view_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        assert!(make(&View::NullConstant, heap.cpu(0)).is_some());
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        // Every index a translated shader can reach exists (clamped below
        // `HEAP_VIEWS`, `HEAP_SAMPLERS`).
        // SAFETY: plain queries of live heaps.
        unsafe {
            assert!(heap.heap.GetDesc().NumDescriptors >= victory_shader::emit::view_heap_limit());
            let samplers = HostHeap::new(true, 128).unwrap();
            assert!(samplers.heap.GetDesc().NumDescriptors >= victory_shader::emit::HEAP_SAMPLERS);
        }
        // A slot that stops viewing a resource leaves it alive until the GPU
        // work in flight has run (a read of a freed host texture otherwise).
        // SAFETY: a plain committed buffer; one reference taken and given back.
        unsafe {
            let host = host().unwrap();
            let mut buffer: Option<ID3D12Resource> = None;
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: 256,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .unwrap();
            let buffer = buffer.unwrap();
            let references = |b: &ID3D12Resource| {
                let raw = windows::core::Interface::as_raw(b);
                let unknown = &*(raw as *const *const windows::core::IUnknown_Vtbl);
                ((**unknown).AddRef)(raw);
                ((**unknown).Release)(raw)
            };
            let slot = heap.count + 2;
            heap.raw_view_at(slot, Some(&buffer), 0, 256, false);
            assert_eq!(references(&buffer), 2, "ours and the slot's");
            heap.make_null(slot);
            assert!(references(&buffer) >= 2, "the slot's, retired");
            crate::host::release_retired();
            assert_eq!(references(&buffer), 1, "ours");
        }
        let a = heap.dynamic(heap.dynamic - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + heap.dynamic - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        heap.release(a, 50);
        assert_eq!(heap.dynamic(50), Some(4));
        assert_eq!(heap.dynamic(40), Some(4 + heap.dynamic - 40));
    }

    #[test]
    fn unsupported_typed_uav_format_is_refused() {
        let Some(host) = host() else {
            return;
        };
        let mut tex: Option<ID3D12Resource> = None;
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 4,
            Height: 4,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R16_TYPELESS,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        // SAFETY: plain committed typeless texture on the default heap.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    None,
                    &mut tex,
                )
                .unwrap();
        }
        let tex = tex.unwrap();
        const B5G5R5A1_UNORM: u32 = 86;
        let mut fmt = DXGI_FORMAT(B5G5R5A1_UNORM as i32);
        // Unsupported typed UAV format must be refused even when castable.
        assert_eq!(
            super::fits(&tex, &[B5G5R5A1_UNORM], Some(&mut fmt), true),
            super::typed_store(B5G5R5A1_UNORM)
        );
        // SRV with the same castable format remains allowed.
        let mut srv_fmt = DXGI_FORMAT(B5G5R5A1_UNORM as i32);
        assert!(super::fits(&tex, &[B5G5R5A1_UNORM], Some(&mut srv_fmt), false));
    }

    #[test]
    fn supported_typed_uav_formats_are_accepted() {
        let Some(host) = host() else {
            return;
        };
        let make_tex = |format: DXGI_FORMAT| {
            let mut tex: Option<ID3D12Resource> = None;
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 4,
                Height: 4,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: format,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            // SAFETY: plain committed texture on the default heap.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES {
                            Type: D3D12_HEAP_TYPE_DEFAULT,
                            ..Default::default()
                        },
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        None,
                        &mut tex,
                    )
                    .unwrap();
            }
            tex.unwrap()
        };

        let typeless = make_tex(DXGI_FORMAT_R8G8B8A8_TYPELESS);
        assert!(!super::fits(&typeless, &[], None, true));
        for format in [DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UINT] {
            let mut fmt = format;
            assert!(super::fits(&typeless, &[], Some(&mut fmt), true));
            assert_eq!(fmt, format);
        }
        let mut srgb = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        assert!(super::fits(&typeless, &[], Some(&mut srgb), true));
        assert_eq!(srgb, DXGI_FORMAT_R8G8B8A8_UNORM);

        for format in [DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R16G16B16A16_FLOAT] {
            let typed = make_tex(format);
            assert!(super::fits(&typed, &[], None, true));
            let mut fmt = format;
            assert!(super::fits(&typed, &[], Some(&mut fmt), true));
        }
    }

    #[test]
    fn dynamic_search_can_find_a_run_across_the_cursor() {
        let Some(heap) = HostHeap::new(false, 4) else {
            eprintln!("no host descriptor heap; skipped");
            return;
        };
        let first = heap.dynamic(1).unwrap();
        let left = heap.dynamic(2).unwrap();
        let right = heap.dynamic(2).unwrap();
        let rest = heap.dynamic(heap.dynamic - 5).unwrap();
        assert_eq!((first, left, right, rest), (4, 5, 7, 9));

        // Reallocating the left half puts the cursor back in the middle.
        heap.release(left, 2);
        assert_eq!(heap.dynamic(2), Some(left));
        heap.release(left, 2);
        heap.release(right, 2);
        assert_eq!(heap.dynamic(4), Some(left));
        assert_eq!(heap.dynamic(1), None);

        // A free slot at each physical end does not form a two-slot range.
        heap.release(first, 1);
        let last = first + heap.dynamic - 1;
        heap.release(last, 1);
        assert_eq!(heap.dynamic(2), None);
    }
}
