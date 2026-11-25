//! Draws on the host GPU (decision 0007, step 3). A list keeps the state
//! the title sets for its draws (pipeline, descriptor heaps, root
//! arguments, vertex and index buffers, targets); each draw then brings the
//! host list up to date: the host pipeline and our root signature, the
//! user data the translated shaders read, the host copies of the buffers
//! and descriptors they read, and the states of the textures involved.
//! State the host keeps across draws (viewports, scissors, blend factor,
//! stencil reference, topology, targets) is set on the host list as the
//! title sets it.

use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::{ResourceType, RootType, UsedResource};

use crate::descriptor::{TableRecords, View, table_view};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use crate::host::HostList;
use crate::object::This;
use crate::pipeline::host::HostPipeline;

/// A root argument as the title set it.
#[derive(Debug, Clone, Default)]
enum Root {
    #[default]
    None,
    /// A descriptor table's first descriptor (an address in a heap).
    Table(u64),
    Constants(Vec<Option<u32>>),
    /// A root view's GPU address.
    View(u64),
}

type FreshTables = std::collections::HashMap<usize, std::collections::HashMap<TableRecords, u32>>;

#[cfg(test)]
#[test]
fn fresh_tables_share_by_view_generation_length_and_heap() {
    let mut tables = FreshTables::new();
    let original: TableRecords = Box::new([[4, 1], [0, 0], [7, 1]]);
    tables.entry(42).or_default().insert(original, 300);
    let moved = [[4, 1], [0, 0], [7, 1]];
    assert_eq!(tables[&42].get(moved.as_slice()), Some(&300));
    assert_eq!(tables[&42].get(&moved[..2]), None);
    assert_eq!(tables[&42].get([[4, 2], [0, 0], [7, 1]].as_slice()), None);
    assert_eq!(tables[&42].get([[0, 0], [0, 0], [7, 1]].as_slice()), None);
    assert!(!tables.contains_key(&43));
}

pub(crate) use crate::page_view::{IndexView, VertexView};
use crate::page_view::InputView;

/// What one title list has set, and what its host list was last given.
#[derive(Default)]
pub(crate) struct State {
    pipeline: usize,
    /// View heap, sampler heap.
    heaps: [usize; 2],
    graphics: Vec<Root>,
    compute: Vec<Root>,
    vertex: [VertexView; 32],
    index: IndexView,
    /// Bound render target and depth resources, for their host states.
    targets: Vec<usize>,
    depth: usize,
    page_target: Option<crate::page_draw::ColorTarget>,
    page_depth: Option<crate::page_draw::DepthTarget>,
    page_stencil: u32,
    page_viewport: Option<D3D12_VIEWPORT>,
    page_scissor: Option<RECT>,
    page_blend: Option<[f32; 4]>,
    page_state_opaque: bool,
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// What the host list has: pipeline, root signature, heaps.
    host_pipeline: usize,
    /// Graphics, compute.
    host_roots: [usize; 2],
    host_heaps: [usize; 2],
    /// The same immutable table can be referenced at different title
    /// addresses. Reuse only a full, equal snapshot in the same bound heap.
    fresh_tables: FreshTables,
    /// The dynamic slots the list holds: title heap, first, count; given
    /// back when the list is reset or released.
    dynamic: Vec<(usize, u32, u32)>,
    /// A fresh table of the stage being bound could not be made.
    table_failed: bool,
    /// The title-memory page table made for this list (`crate::memory`)
    /// and its view heap.
    memory: Option<(usize, crate::memory::Map)>,
    /// Views made for root SRVs and UAVs in this list ([`State::root_view`]):
    /// (view heap, host copy, unordered access) → dynamic slot.
    root_slots: std::collections::HashMap<(usize, usize, bool), u32>,
    /// The run of dynamic slots root views take from (view heap, next
    /// slot, end): taken [`ROOT_VIEW_RUN`] at a time, so that single slots
    /// do not break up the runs fresh tables need.
    root_run: (usize, u32, u32),
}

impl Drop for State {
    fn drop(&mut self) {
        for &(heap, first, n) in &self.dynamic {
            match (heap, crate::memory::heap()) {
                (MEMORY_HEAP, Some(ours)) => ours.release(first, n),
                _ => crate::descriptor::release_dynamic(heap, first, n),
            }
        }
    }
}

/// The key of our own view heap (`crate::memory::heap`) among the title's.
const MEMORY_HEAP: usize = crate::memory::OURS;

/// Dynamic slots taken at a time for root views.
const ROOT_VIEW_RUN: u32 = 256;

/// Dynamic slots taken for root views ([`State::root_view`]) since the last
/// call.
pub(crate) fn take_root_view_slots() -> u64 {
    ROOT_VIEW_SLOTS.swap(0, Ordering::Relaxed)
}
static ROOT_VIEW_SLOTS: AtomicU64 = AtomicU64::new(0);

/// Draws recorded on the host GPU since the last frame line.
static DRAWN: AtomicU64 = AtomicU64::new(0);

/// Draws run on the host so far.
pub(crate) fn drawn() -> u64 {
    DRAWN.load(Ordering::Relaxed)
}

/// The draws traced since the last present (diagnostics builds, with
/// debug logging for this module): logged when the frame comes out black.
#[cfg(feature = "diagnostics")]
static FRAME: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The draws traced since the last call.
#[cfg(feature = "diagnostics")]
pub(crate) fn take_frame() -> Vec<String> {
    std::mem::take(&mut *FRAME.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Draws, mesh dispatches and dispatches recorded but not run on the host
/// (no host pipeline) since the last call.
pub(crate) fn take_dropped() -> u64 {
    DROPPED.swap(0, Ordering::Relaxed)
}
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Marks the last traced draw as not run on the host, counts it by why
/// (the reason [`not_run`] gave, if its pipeline has a host one) and notes
/// it in the list (diagnostics).
#[cfg(feature = "diagnostics")]
fn dropped(list: &HostList, pipeline: usize, kind: &str) {
    let why = NOT_RUN.with(|w| w.replace(""));
    if why.is_empty() || crate::pipeline::host_pipeline(pipeline as This).is_none() {
        crate::pipeline::note_drop(pipeline as This);
    } else {
        crate::pipeline::note_why(format!("{kind}: {why}"));
    }
    list.note_op(|| format!("DROPPED {kind} ({why}) {}", crate::pipeline::shaders_of(pipeline as This)));
    if let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
        last.push_str(" DROPPED");
    }
}

#[cfg(feature = "diagnostics")]
thread_local! {
    /// Why the draw or dispatch this thread last prepared cannot run.
    static NOT_RUN: std::cell::Cell<&'static str> = const { std::cell::Cell::new("") };
}

/// Notes why the draw or dispatch being prepared cannot run (diagnostics).
fn not_run(why: &'static str) {
    #[cfg(feature = "diagnostics")]
    NOT_RUN.with(|w| w.set(why));
    #[cfg(not(feature = "diagnostics"))]
    let _ = why;
}

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

/// The host address of a root view: its buffer's host copy, or for a
/// constant buffer not 256-byte aligned there (the console needs 16) a
/// copy in upload memory. What shaders write through a root UAV goes back
/// to title memory after the submission.
/// shortcut: a root view has no size; constant buffers are taken as 64 KiB,
/// other views as 1 MiB (both cut at their buffer's end).
fn root_view(list: &HostList, address: u64, srv: bool, uav: bool) -> Option<u64> {
    let shadow = crate::shadow::find(address, 1)?;
    let len = shadow
        .left(address)
        .min(if srv || uav { 1 << 20 } else { 1 << 16 });
    let gpu = shadow.gpu(address);
    if srv || uav || gpu % 256 == 0 {
        shadow.refresh(list, address, len);
        if uav {
            // A host root descriptor: no heap view, so its writes are not
            // marked.
            shadow.gpu_writes(list, address, len, false);
        }
        return Some(gpu);
    }
    let (gpu, to) = list.scratch(len)?;
    list.defer_fill(to, address as *const u8, len as usize);
    Some(gpu)
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

fn set(roots: &mut Vec<Root>, slot: u32, root: Root) {
    let slot = slot as usize;
    if roots.len() <= slot {
        roots.resize(slot + 1, Root::None);
    }
    roots[slot] = root;
}

impl State {
    pub(crate) fn reset(&mut self, pipeline: This) {
        *self = State::default();
        self.pipeline = pipeline as usize;
    }

    /// The title's topology; console-only ones become host lists (see
    /// [`crate::primitive`]).
    pub(crate) fn set_topology(&mut self, list: &HostList, topology: u32) {
        self.topology = topology;
        if cfg!(feature = "paged-title") {
            return;
        }
        let host = match topology {
            RECTANGLES => 4,
            TRIANGLE_FAN | LINE_LOOP | QUADS | QUAD_STRIP => {
                once!(topology, "console topology drawn as a triangle list");
                4
            }
            t => t,
        };
        // SAFETY: an open list.
        unsafe {
            list.recording()
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host as i32))
        };
    }

    pub(crate) fn set_pipeline(&mut self, pipeline: This) {
        self.pipeline = pipeline as usize;
    }

    /// Others recorded into the host list (`crate::external`): its heaps,
    /// root signatures and pipeline are set again before the next draw or
    /// dispatch.
    pub(crate) fn host_state_lost(&mut self) {
        // No title heaps (0) differ from what the host list has, too.
        (self.host_pipeline, self.host_roots, self.host_heaps) = (0, [0; 2], [usize::MAX; 2]);
        self.page_state_opaque = true;
    }

    pub(crate) fn set_heaps(&mut self, heaps: &[This]) {
        self.heaps = [0; 2];
        for &h in heaps {
            if !h.is_null() {
                let samplers = crate::descriptor::heap(h).samplers();
                self.heaps[usize::from(samplers)] = h as usize;
            }
        }
    }

    fn roots(&mut self, compute: bool) -> &mut Vec<Root> {
        if compute {
            &mut self.compute
        } else {
            &mut self.graphics
        }
    }

    pub(crate) fn set_table(&mut self, compute: bool, slot: u32, handle: u64) {
        set(self.roots(compute), slot, Root::Table(handle));
    }

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, address: u64) {
        set(self.roots(compute), slot, Root::View(address));
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
        let roots = self.roots(compute);
        let s = slot as usize;
        if !matches!(roots.get(s), Some(Root::Constants(_))) {
            set(roots, slot, Root::Constants(Vec::new()));
        }
        if let Root::Constants(c) = &mut roots[s] {
            let end = offset as usize + values.len();
            if c.len() < end {
                c.resize(end, None);
            }
            for (word, &value) in c[offset as usize..end].iter_mut().zip(values) {
                *word = Some(value);
            }
        }
    }

    pub(crate) fn set_vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        for (i, v) in views.iter().enumerate() {
            if let Some(slot) = self.vertex.get_mut(start as usize + i) {
                *slot = *v;
            }
        }
    }

    pub(crate) fn clear_vertex_buffers(&mut self, start: u32, count: u32) {
        for slot in self.vertex.iter_mut().skip(start as usize).take(count as usize) {
            *slot = VertexView::default();
        }
    }

    pub(crate) fn set_index_buffer(&mut self, view: Option<&IndexView>) {
        self.index = view.copied().unwrap_or_default();
    }

    /// Render targets and depth, set on the host list at once.
    pub(crate) fn set_targets(&mut self, list: &HostList, handles: &[u64], depth: u64) {
        let Some(host) = crate::host::host() else {
            return;
        };
        self.targets.clear();
        self.page_target = None;
        self.page_depth = None;
        let mut rtvs = Vec::new();
        for &h in handles {
            let Some(record) = crate::descriptor::rtv_record(h) else {
                once!(
                    handle = format_args!("{h:#x}"),
                    "render target without a recorded view"
                );
                continue;
            };
            let Some(texture) = crate::resource::host_resource(record.resource) else {
                continue;
            };
            self.targets.push(record.resource as usize);
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(list, record.resource);
            let desc = record.desc();
            if handles.len() == 1 {
                self.page_target = crate::resource::host_copy(record.resource)
                    .and_then(|copy| crate::page_draw::ColorTarget::capture_candidate(copy, desc));
            }
            if !cfg!(feature = "paged-title") {
                rtvs.push(host.rtv(texture, desc.as_ref()));
            }
        }
        self.depth = 0;
        if cfg!(feature = "paged-title") {
            if depth != 0 {
                let depth_record = crate::descriptor::dsv_record(depth);
                self.page_depth = depth_record.and_then(|record| {
                    crate::resource::host_copy(record.resource)
                        .and_then(|copy| crate::page_draw::DepthTarget::capture(copy, record.depth_desc()))
                });
                if self.page_depth.is_none() {
                    list.mark_unlowered();
                }
            }
            return;
        }
        let dsv = crate::descriptor::dsv_record(depth).and_then(|record| {
            let texture = crate::resource::host_resource(record.resource)?;
            self.depth = record.resource as usize;
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(list, record.resource);
            Some(host.dsv(texture, record.depth_desc().as_ref()))
        });
        // SAFETY: an open list; the views were just made.
        unsafe {
            list.recording().OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
    }

    fn record_strict_draw(&self, list: &HostList, call: crate::page_draw::DrawCall) -> bool {
        if let Some(draw) = self.capture_page_draw(call) {
            list.record_page_draw(draw);
            true
        } else {
            list.mark_unlowered();
            false
        }
    }

    fn capture_page_draw(
        &self,
        call: crate::page_draw::DrawCall,
    ) -> Option<crate::page_draw::ClosedDraw> {
        if self.page_state_opaque || !matches!(self.topology, 1..=5) {
            return None;
        }
        let target = self.page_target.clone()?;
        let viewport = self.page_viewport?;
        let scissor = self.page_scissor?;
        let pipeline = crate::pipeline::host_pipeline(self.pipeline as This)?;
        let mut inputs = Vec::new();
        for (slot, &view) in self.vertex.iter().enumerate() {
            if pipeline.slots & (1 << slot) != 0 {
                let input = InputView::Vertex(slot as u32, view);
                input.range().ok()?;
                inputs.push(input);
            }
        }
        if call.indexed() {
            let input = InputView::Index(self.index);
            input.range().ok()?;
            inputs.push(input);
        }
        let mut draw = crate::page_draw::ClosedDraw::from_translated(
            pipeline,
            |slot, dword| self.page_word(false, slot, dword),
            |root, index| self.page_view(false, root, index),
            |slot| self.page_address(false, slot),
            target,
            self.page_depth.clone(),
            self.page_stencil,
            D3D_PRIMITIVE_TOPOLOGY(self.topology as i32),
            viewport,
            scissor,
            inputs,
            call,
        ).ok()?;
        draw.set_blend_factor(self.page_blend.unwrap_or([1.0; 4]));
        Some(draw)
    }

    /// The user-data value of a root argument, as translated shaders read
    /// it: a table's byte offset in its heap, a constant's value, a root
    /// view's address (title memory).
    fn user_value(&self, compute: bool, slot: u8, root: RootType, dword: u32) -> u32 {
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            (Some(Root::View(address)), RootType::Descriptor) if dword < 2 => {
                (address >> (32 * dword)) as u32
            }
            (Some(Root::Table(handle)), RootType::Table) => match self.heap_of(*handle) {
                Some(heap) => (handle - heap.start) as u32,
                None => {
                    once!(
                        handle = format_args!("{handle:#x}"),
                        "descriptor table outside the bound heaps"
                    );
                    0
                }
            },
            (Some(Root::Constants(c)), RootType::Constants) => {
                c.get(dword as usize).copied().flatten().unwrap_or(0)
            }
            _ => {
                once!(
                    slot,
                    ?root,
                    "a shader reads a root argument the title did not set"
                );
                0
            }
        }
    }

    /// The host addresses of the bound view heap's title-memory page table
    /// and the list's page bitmap for it, set up on first use (the table
    /// names every host copy at each submission).
    fn memory_map(&mut self, list: &HostList) -> Option<(u64, u64)> {
        let heap = match self.heaps[0] {
            0 => MEMORY_HEAP,
            h => h,
        };
        if let Some((h, map)) = &self.memory
            && *h == heap
        {
            return Some((map.table, map.dirty));
        }
        let map = crate::memory::bind(list, heap)?;
        let out = (map.table, map.dirty);
        self.memory = Some((heap, map));
        Some(out)
    }

    /// The bound heap holding `handle`.
    fn heap_of(&self, handle: u64) -> Option<&'static crate::descriptor::Heap> {
        self.heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| h.index(handle).is_some())
    }

    /// Brings the host list up to date for a draw; false if the draw cannot
    /// run on the host.
    fn prepare(&mut self, list: &HostList, indexed: bool) -> bool {
        self.prepare_for(list, indexed, false)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches.
    fn prepare_for(&mut self, list: &HostList, indexed: bool, mesh: bool) -> bool {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return false;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        if p.compute || p.mesh != mesh {
            once!(mesh, "a draw or mesh dispatch with another kind of pipeline");
            not_run("another kind of pipeline");
            return false;
        }
        if !self.bind(list, p, false) {
            return false;
        }
        let l = list.recording();
        // Vertex buffers the input layout reads.
        let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
        let mut count = 0;
        for (slot, (view, &v)) in views.iter_mut().zip(&self.vertex).enumerate() {
            if p.slots & 1 << slot == 0 {
                continue;
            }
            count = slot + 1;
            if v.address == 0 || v.size == 0 {
                continue;
            }
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                continue;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            list.record_page_input(InputView::Vertex(slot as u32, v));
            *view = v.native(shadow.gpu(v.address));
        }
        // SAFETY: an open list; the views name live host buffers.
        unsafe {
            if count > 0 {
                l.IASetVertexBuffers(0, Some(&views[..count]));
            }
        }
        if indexed {
            let v = self.index;
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                not_run("index buffer outside title buffers");
                return false;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            list.record_page_input(InputView::Index(v));
            let view = v.native(shadow.gpu(v.address));
            // SAFETY: an open list; the view names a live host buffer.
            unsafe { l.IASetIndexBuffer(Some(&view)) };
        }
        const RENDER_TARGET: u32 = 0x4;
        const DEPTH_WRITE: u32 = 0x10;
        for &t in &self.targets {
            crate::list::ensure_state(list, t as This, RENDER_TARGET);
        }
        if self.depth != 0 {
            crate::list::ensure_state(list, self.depth as This, DEPTH_WRITE);
        }
        // A depth copy of ours (`crate::twin`) ran a compute pipeline.
        if list.take_rebind() {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            (self.host_pipeline, self.host_roots[1]) = (p.pso.as_raw() as usize, 0);
        }
        true
    }

    /// A descriptor a shader reads through a table: its host slot made if
    /// it changed, constant data copied, textures made readable.
    /// What draws and dispatches share: heaps, our root signature, the
    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold.
    /// False if the draw or dispatch must not run: a table it reads could
    /// not be made.
    fn bind(&mut self, list: &HostList, p: &HostPipeline, compute: bool) -> bool {
        let l = list.recording();
        // A depth copy of ours (`crate::twin`) since the last one ran a
        // compute pipeline.
        if list.take_rebind() {
            (self.host_pipeline, self.host_roots[1]) = (0, 0);
        }
        // Heaps first: a root signature with directly indexed heaps must be
        // set after them. Title memory's views need a view heap: ours where
        // the title binds none (the BVH builder's dispatches may not).
        let memory_heap = self.heaps[0] == 0
            && p.stages.iter().any(|s| s.memory.is_some() || has_root_views(&s.translated().host));
        self.bind_heaps(list, memory_heap);
        let root = p.root.as_raw() as usize;
        if self.host_roots[usize::from(compute)] != root {
            // SAFETY: an open list and a live root signature.
            unsafe {
                if compute {
                    l.SetComputeRootSignature(&p.root);
                } else {
                    l.SetGraphicsRootSignature(&p.root);
                }
            }
            self.host_roots[usize::from(compute)] = root;
        }
        // By host object: a title pipeline released and another made at its
        // address is another host pipeline (the list keeps the old one).
        if self.host_pipeline != p.pso.as_raw() as usize {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            self.host_pipeline = p.pso.as_raw() as usize;
            list.used_pipeline(&p.name);
            // The title may release the pipeline before it submits the list.
            list.keep(&p.pso);
            list.keep(&p.root);
        }
        #[cfg(feature = "diagnostics")]
        crate::capture::binding(&p.name);
        // The bound view heap's mark table and the bitmap shaders mark
        // their writes through views in (`crate::shadow::copy_back_marks`).
        if let Some((table_param, bitmap_param)) = p.marks {
            let heap = match self.heaps[0] {
                0 => crate::memory::heap(),
                h => crate::descriptor::heap(h as This).host(),
            };
            let (Some(table), Some(bitmap)) = (heap.and_then(|h| h.marks()), crate::shadow::view_writes()) else {
                once!("a shader marking its writes without a view heap: dropped");
                not_run("write marks without a view heap");
                return false;
            };
            list.keep(table);
            // SAFETY: an open list; the table lives while the list holds it,
            // the bitmap for good.
            unsafe {
                let table = table.GetGPUVirtualAddress();
                if compute {
                    l.SetComputeRootShaderResourceView(table_param, table);
                    l.SetComputeRootUnorderedAccessView(bitmap_param, bitmap);
                } else {
                    l.SetGraphicsRootShaderResourceView(table_param, table);
                    l.SetGraphicsRootUnorderedAccessView(bitmap_param, bitmap);
                }
            }
        }
        // Shaders taking views from the heap itself read (or write) views no
        // table names: everything the bound heap's views reach is brought up
        // to date at submission (`crate::bindless`).
        if p.stages.iter().any(|s| s.translated().host.heap) {
            let writes = p.stages.iter().any(|s| s.translated().host.heap_writes);
            if let Some(host) = (self.heaps[0] != 0)
                .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
                .flatten()
            {
                list.defer_heap(&host.viewed);
            }
            crate::bindless::note_draw(writes);
        }
        for stage in &p.stages {
            let t = stage.translated();
            self.table_failed = false;
            let made = self.tables(list, &t.used, compute);
            // A depth copy of ours made readable a texture the tables hold
            // (`crate::twin`), with a compute pipeline: this one's again
            // (root arguments are set below).
            if list.take_rebind() {
                // SAFETY: an open list, a live root signature and pipeline.
                unsafe {
                    if compute {
                        l.SetComputeRootSignature(&p.root);
                    }
                    l.SetPipelineState(&p.pso);
                }
                self.host_pipeline = p.pso.as_raw() as usize;
                self.host_roots[1] = if compute { root } else { 0 };
            }
            if self.table_failed {
                static DROPPED_TABLES: AtomicU64 = AtomicU64::new(0);
                let n = DROPPED_TABLES.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    tracing::warn!(
                        dropped = n,
                        "a draw or dispatch dropped: no dynamic slots for its tables"
                    );
                }
                not_run("no dynamic slots for its tables");
                return false;
            }
            // Root views a shader reads as title memory: the copies holding
            // their addresses go into the page table.
            let mut roots = Vec::new();
            if let Some((param, dwords)) = stage.user_data {
                let mut values = vec![0u32; dwords as usize];
                for u in &t.host.user_data {
                    if u.root == RootType::Descriptor
                        && u.dword == 0
                        && let Some(Root::View(address)) =
                            self.roots_ref(compute).get(usize::from(u.slot))
                        && let Some(shadow) = crate::shadow::find(*address, 1)
                    {
                        let uav = t.used.iter().any(|r| {
                            r.slot == u.slot
                                && r.root == RootType::Descriptor
                                && r.kind == ResourceType::Uav
                        });
                        roots.push((*address, uav.then_some(shadow)));
                    }
                    let fresh = made
                        .iter()
                        .find(|m| m.0 == u.slot && u.root == RootType::Table);
                    values[u.index as usize] = match fresh {
                        Some(&(_, offset)) => offset,
                        None if u.dword == victory_shader::emit::ROOT_VIEW_OFFSET => {
                            // Where the root address lies in the copy its view shows.
                            let offset = match self.roots_ref(compute).get(usize::from(u.slot)) {
                                Some(&Root::View(address)) => crate::shadow::find(address, 1)
                                    .and_then(|s| u32::try_from(s.offset(address)).ok()),
                                _ => None,
                            };
                            match offset {
                                Some(offset) => offset,
                                None => return false,
                            }
                        }
                        None if is_root_view(u) => {
                            let uav = u.dword == victory_shader::emit::ROOT_UAV_VIEW;
                            match self.root_view(list, compute, u.slot, uav) {
                                Some((slot, _)) => slot,
                                None => return false,
                            }
                        }
                        None => self.user_value(compute, u.slot, u.root, u.dword),
                    };
                }
                // SAFETY: an open list; `values` covers the parameter.
                unsafe {
                    if compute {
                        l.SetComputeRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    } else {
                        l.SetGraphicsRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    }
                }
            }
            for (views, kind) in [(&stage.root_cbvs, 0), (&stage.root_srvs, 1), (&stage.root_uavs, 2)] {
                for &(slot, param) in views {
                    if kind == 1 && slot == victory_shader::emit::VIEW_STRIDES {
                        // The bound view heap's stride table; without one
                        // the shader would read an unbound root view.
                        let heap = (self.heaps[0] != 0)
                            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
                            .flatten();
                        let Some(table) = heap.and_then(|h| h.strides()) else {
                            once!("a shader reads view strides without a view heap: dropped");
                            not_run("view strides without a view heap");
                            return false;
                        };
                        list.keep(table);
                        // SAFETY: an open list; the table lives while the list
                        // holds it.
                        unsafe {
                            let gpu = table.GetGPUVirtualAddress();
                            if compute {
                                l.SetComputeRootShaderResourceView(param, gpu);
                            } else {
                                l.SetGraphicsRootShaderResourceView(param, gpu);
                            }
                        }
                        continue;
                    }
                    let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot))
                    else {
                        once!(slot, "a shader reads a root view the title did not set");
                        continue;
                    };
                    let Some(gpu) = root_view(list, *address, kind == 1, kind == 2) else {
                        continue;
                    };
                    // SAFETY: an open list; the address is in live memory.
                    unsafe {
                        match (compute, kind) {
                            (false, 0) => l.SetGraphicsRootConstantBufferView(param, gpu),
                            (false, 1) => l.SetGraphicsRootShaderResourceView(param, gpu),
                            (false, _) => l.SetGraphicsRootUnorderedAccessView(param, gpu),
                            (true, 0) => l.SetComputeRootConstantBufferView(param, gpu),
                            (true, 1) => l.SetComputeRootShaderResourceView(param, gpu),
                            (true, _) => l.SetComputeRootUnorderedAccessView(param, gpu),
                        }
                    }
                }
            }
            if let Some(param) = stage.view_addresses {
                let gpu = (self.heaps[0] != 0)
                    .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
                    .flatten()
                    .and_then(|h| h.view_addresses());
                let Some(gpu) = gpu else {
                    once!("a shader reading a view's address dropped: no view heap");
                    not_run("view addresses without a view heap");
                    return false;
                };
                // SAFETY: an open list; the heap's buffer lives with the heap.
                unsafe {
                    if compute {
                        l.SetComputeRootShaderResourceView(param, gpu);
                    } else {
                        l.SetGraphicsRootShaderResourceView(param, gpu);
                    }
                }
            }
            if let Some(param) = stage.memory {
                let Some((table, dirty)) = self.memory_map(list) else {
                    once!("a shader reading title memory dropped: no view heap or page table");
                    not_run("title memory without a view heap or page table");
                    return false;
                };
                crate::memory::note_use(&p.name);
                // What shaders write through a root UAV goes back by its
                // bytes, as through root views ([`root_view`]).
                for (address, shadow) in &roots {
                    if let (Some(shadow), Some((_, map))) = (shadow, &self.memory) {
                        let len = shadow.left(*address).min(1 << 20);
                        shadow.refresh(list, *address, len);
                        shadow.gpu_writes(list, *address, len, false);
                        map.precise(*address, len);
                    }
                }
                // SAFETY: an open list; list memory alive until it is reset.
                unsafe {
                    if compute {
                        l.SetComputeRootShaderResourceView(param, table);
                    } else {
                        l.SetGraphicsRootShaderResourceView(param, table);
                    }
                    if let Some(param) = stage.memory_dirty {
                        if compute {
                            l.SetComputeRootUnorderedAccessView(param, dirty);
                        } else {
                            l.SetGraphicsRootUnorderedAccessView(param, dirty);
                        }
                    }
                }
            }
        }
        #[cfg(feature = "diagnostics")]
        crate::capture::binding("");
        true
    }

    /// The heap index of a raw view (`uav`: unordered access) of the host
    /// copy holding the root view the title set at root parameter `slot`,
    /// and the root address's byte offset in it: a dynamic slot of the view
    /// heap the shader indexes, made once per list and copy; the range from
    /// the root address brought up to date at submission
    /// (`victory_shader::emit`, "Root SRVs and root UAVs"). None if the
    /// draw must be dropped: no root view set, or no dynamic slot.
    /// The copy is brought up to date from the root address to its end: a
    /// shader reaches wherever data its root view leads to lie (a vertex
    /// fetch's streams sit at offsets in the root view's own header).
    /// shortcut: what shaders write through a root UAV goes back to title
    /// memory for 1 MiB from its address, as before the views.
    fn root_view(&mut self, list: &HostList, compute: bool, slot: u8, uav: bool) -> Option<(u32, u32)> {
        let Some(&Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) else {
            once!(slot, "a shader reads a root view the title did not set: dropped");
            return None;
        };
        let shadow = crate::shadow::find(address, 1)?;
        let offset = u32::try_from(shadow.offset(address)).ok()?;
        shadow.refresh(list, address, shadow.left(address));
        if uav {
            // Writes through the view are marked wherever they land in the
            // copy; else declared for 1 MiB.
            let len = if shadow.marked() { shadow.left(address) } else { shadow.left(address).min(1 << 20) };
            shadow.gpu_writes(list, address, len, true);
        }
        let (key, host) = match self.heaps[0] {
            0 => (MEMORY_HEAP, crate::memory::heap()?),
            h => (
                std::ptr::from_ref(crate::descriptor::heap(h as This)) as usize,
                crate::descriptor::heap(h as This).host()?,
            ),
        };
        let copy = std::sync::Arc::as_ptr(&shadow) as usize;
        if let Some(&index) = self.root_slots.get(&(key, copy, uav)) {
            return Some((index, offset));
        }
        let (run_heap, next, end) = self.root_run;
        if run_heap != key || next >= end {
            let Some(first) = host.dynamic(ROOT_VIEW_RUN) else {
                once!("no dynamic slots for root views: dropped");
                return None;
            };
            self.dynamic.push((key, first, ROOT_VIEW_RUN));
            self.root_run = (key, first, first + ROOT_VIEW_RUN);
        }
        let index = self.root_run.1;
        self.root_run.1 += 1;
        ROOT_VIEW_SLOTS.fetch_add(1, Ordering::Relaxed);
        host.raw_view_at(index, Some(&shadow.buffer), 0, shadow.range().1 & !3, uav);
        if uav {
            host.set_mark(index, shadow.mark_entry(shadow.range().0));
        }
        self.root_slots.insert((key, copy, uav), index);
        Some((index, offset))
    }

    /// Gives the host list the title's heaps if it has others.
    fn bind_heaps(&mut self, list: &HostList, memory_heap: bool) {
        let want = if memory_heap {
            [MEMORY_HEAP, self.heaps[1]]
        } else {
            self.heaps
        };
        if self.host_heaps != want {
            let mut heaps: Vec<Option<ID3D12DescriptorHeap>> = self
                .heaps
                .iter()
                .filter(|&&h| h != 0)
                .filter_map(|&h| crate::descriptor::heap(h as This).host())
                .map(|h| Some(h.heap.clone()))
                .collect();
            if memory_heap && let Some(h) = crate::memory::heap() {
                heaps.insert(0, Some(h.heap.clone()));
            }
            heaps.iter().flatten().for_each(|h| list.keep(h));
            // SAFETY: an open list and shader-visible heaps.
            unsafe { list.recording().SetDescriptorHeaps(&heaps) };
            self.host_heaps = want;
            self.host_roots = [0; 2];
        }
    }

    /// `ClearUnorderedAccessViewUint` or `...Float` (`float`): the view the
    /// title made at `cpu` (its CPU handle; the GPU one names the same view
    /// in the bound heap) cleared to `values` on the host, through a
    /// dynamic slot of the bound view heap that the list holds until reset.
    pub(crate) fn clear_unordered(
        &mut self,
        list: &HostList,
        cpu: u64,
        values: [u32; 4],
        float: bool,
        rects: &[RECT],
    ) {
        if cfg!(feature = "paged-title") {
            let clear = (!float && rects.is_empty()).then(|| crate::descriptor::view(cpu)).flatten()
                .and_then(|view| crate::descriptor::host::paged_raw_clear(&view, values));
            if let Some(clear) = clear {
                list.record_page_raw_clear(clear);
            } else {
                list.mark_unlowered();
            }
            return;
        }
        let Some(view) = crate::descriptor::view(cpu) else {
            once!("an unordered access clear of a slot without a view");
            return;
        };
        let heap = (self.heaps[0] != 0).then(|| crate::descriptor::heap(self.heaps[0] as This));
        let (Some(heap), Some(host)) = (heap, crate::host::host()) else {
            once!("an unordered access clear without a bound view heap");
            return;
        };
        let Some(visible) = heap.host() else { return };
        let Some(slot) = visible.dynamic(1) else {
            return;
        };
        self.dynamic
            .push((std::ptr::from_ref(heap) as usize, slot, 1));
        visible.make_at(slot, &view);
        let at = host.cpu_view();
        let Some(resource) = crate::descriptor::host::make_view(&view, at) else {
            once!("an unordered access clear of a view the host cannot make");
            return;
        };
        let written = UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        };
        readable(list, &written, &view, false);
        self.bind_heaps(list, false);
        list.keep(&resource);
        let gpu = visible.gpu(slot);
        // SAFETY: an open list; both slots view the live resource.
        unsafe {
            if float {
                let values = values.map(f32::from_bits);
                list.recording()
                    .ClearUnorderedAccessViewFloat(gpu, at, &resource, &values, rects);
            } else {
                list.recording()
                    .ClearUnorderedAccessViewUint(gpu, at, &resource, &values, rects);
            }
        }
    }

    fn record_strict_dispatch(&self, list: &HostList, call: crate::page_dispatch::DispatchCall) -> bool {
        let ready = (!self.page_state_opaque)
            .then(|| crate::pipeline::host_pipeline(self.pipeline as This)).flatten()
            .and_then(|pipeline| crate::page_dispatch::ClosedDispatch::capture(
                pipeline,
                |slot, dword| self.page_word(true, slot, dword),
                |root, index| self.page_view(true, root, index),
                |slot| self.page_address(true, slot),
                call,
            ).ok());
        if let Some(dispatch) = ready {
            list.record_page_dispatch(dispatch);
            true
        } else {
            list.mark_unlowered();
            false
        }
    }

    fn page_address(&self, compute: bool, slot: u8) -> Option<u64> {
        match *self.roots_ref(compute).get(slot as usize)? {
            Root::View(address) => Some(address),
            _ => None,
        }
    }

    fn page_word(&self, compute: bool, slot: u8, dword: u32) -> Option<u32> {
        match self.roots_ref(compute).get(slot as usize)? {
            Root::Constants(words) => words.get(dword as usize).copied().flatten(),
            _ => None,
        }
    }

    fn page_view(&self, compute: bool, root: u8, index: u32) -> Option<crate::descriptor::View> {
        let Root::Table(first) = *self.roots_ref(compute).get(root as usize)? else { return None };
        let heap = self.heap_of(first)?;
        let records = heap.table_records(first, index.checked_add(1)?)?;
        table_view(*records.get(index as usize)?)
    }

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32) {
        if cfg!(feature = "paged-title") {
            if x != 0 && y != 0 && z != 0 {
                self.record_strict_dispatch(list, crate::page_dispatch::DispatchCall::Direct(x, y, z));
            }
            return;
        }
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            dropped(list, self.pipeline, "dispatch");
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(list, self.pipeline, "dispatch");
            return;
        };
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            return;
        }
        if !self.bind(list, p, true) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(list, self.pipeline, "dispatch");
            return;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("dispatch {x}x{y}x{z}"), true));
        #[cfg(feature = "diagnostics")]
        self.probe_small(list, true);
        // SAFETY: an open list with everything set.
        unsafe { list.recording().Dispatch(x, y, z) };
        #[cfg(feature = "diagnostics")]
        self.probe_written(list);
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
    }

    /// What a draw or dispatch runs, for the device-removed report
    /// (diagnostics): the pipeline, then per stage the user data it gets
    /// (with the view at that heap index, if one is), the host addresses
    /// of its root views and the views its tables name.
    #[cfg(feature = "diagnostics")]
    fn op_text(&self, kind: String, compute: bool) -> String {
        use std::fmt::Write;
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return kind;
        };
        let mut text = format!("{kind} {}", p.name);
        let views = (self.heaps[0] != 0)
            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
            .flatten();
        for stage in &p.stages {
            let t = stage.translated();
            let _ = write!(text, " | {}:", stage.shader.hash);
            for u in t.host.user_data.iter().filter(|u| !is_root_view(u)) {
                let v = self.user_value(compute, u.slot, u.root, u.dword);
                let _ = write!(text, " u{}={v:#x}", u.index);
                let index = if u.root == RootType::Table { v / 32 } else { v };
                if let Some(slot) = views.and_then(|h| h.describe_slot(index)) {
                    let _ = write!(text, " [heap {index}: {slot}]");
                    // A slot the host never made: what the title's memory
                    // holds there (our record, zeros, or words it wrote itself).
                    if slot == "null" && self.heaps[0] != 0 {
                        let at = crate::descriptor::heap(self.heaps[0] as This).start + u64::from(index) * 32;
                        let mut words = [0u8; 32];
                        if crate::host::read_into(at as *const u8, &mut words) {
                            let w: Vec<u32> =
                                words.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                            let _ = write!(text, " (title words {w:x?}");
                            if let Some(view) = crate::descriptor::view(at) {
                                let _ = write!(text, ": {}", describe(Some(&view)));
                            }
                            let _ = write!(text, ")");
                        }
                    }
                }
            }
            let viewed = t.host.user_data.iter().filter(|u| is_root_view(u)).map(|u| u.slot);
            let slots: std::collections::BTreeSet<u8> =
                stage.root_cbvs.iter().chain(&stage.root_srvs).map(|r| r.0).chain(viewed).collect();
            for slot in slots {
                if let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) {
                    let host = crate::shadow::find(*address, 1)
                        .map(|s| (s.gpu(*address), s.left(*address)));
                    // Its first dwords (a vertex stream header, say).
                    let mut head = [0u8; 16];
                    let head = crate::host::read_into(*address as *const u8, &mut head)
                        .then(|| head.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<_>>());
                    let _ = write!(text, " root{slot}={address:#x} host {host:x?} head {head:x?}");
                }
            }
        }
        let _ = write!(text, " tables {:?}", self.reads(compute));
        if !compute {
            let mem = |r: usize| crate::resource::texture_memory(r as This).map_or(0, |(p, _)| p as u64);
            let targets: Vec<String> = self.targets.iter().map(|&t| format!("{t:#x} mem {:#x}", mem(t))).collect();
            let _ = write!(text, " targets {targets:?} depth {:#x} mem {:#x}", self.depth, mem(self.depth));
        }
        text
    }


    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, x: u32) {
        #[cfg(feature = "diagnostics")]
        self.trace("mesh", x, 0);
        let ran = self.prepare_for(list, false, true)
            && match list.recording().cast::<ID3D12GraphicsCommandList6>() {
                Ok(l6) => {
                    #[cfg(feature = "diagnostics")]
                    list.note_op(|| self.op_text(format!("mesh {x}"), false));
                    // SAFETY: an open list with everything set.
                    unsafe { l6.DispatchMesh(x, 1, 1) };
                    DRAWN.fetch_add(1, Ordering::Relaxed);
                    true
                }
                Err(_) => false,
            };
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(list, self.pipeline, "mesh");
        }
    }

    fn roots_ref(&self, compute: bool) -> &Vec<Root> {
        if compute {
            &self.compute
        } else {
            &self.graphics
        }
    }

    fn table_entry(&self, list: &HostList, u: &UsedResource, compute: bool) {
        let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
            return;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        if let Some(view) = host.mirror(index, handle, true) {
            readable(list, u, &view, true);
        }
    }

    /// The descriptors a shader reads through tables, made from the
    /// title's slots. Constant buffers there are raw views of their data
    /// (translated shaders read them so: the host wants constant buffer
    /// views 256-byte aligned, the console only 16), brought up to date at
    /// submission like other buffer views. Returns the byte offsets of
    /// tables made afresh in dynamic slots, by root parameter.
    ///
    /// Within a list, the same table holding the same descriptors is made
    /// once.
    fn tables(&mut self, list: &HostList, used: &[UsedResource], compute: bool) -> Vec<(u8, u32)> {
        let tables = |u: &&UsedResource| u.root == RootType::Table;
        let mut fresh: Vec<u8> = Vec::new();
        // Tables reaching past a host heap smaller than the title's (the
        // host's limit, 1,000,000 views on NVIDIA) are made afresh too: the
        // title's slots there have no host slot of their own.
        for u in used.iter().filter(tables) {
            if fresh.contains(&u.slot) {
                continue;
            }
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
                continue;
            };
            let end = table + u64::from(u.table_offset_dwords) * 4;
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let (Some(host), Some(index)) = (heap.host(), heap.index(end)) else {
                continue;
            };
            if index >= host.count() {
                fresh.push(u.slot);
            }
        }
        fresh.sort_unstable();
        fresh.dedup();
        for u in used.iter().filter(tables) {
            if !fresh.contains(&u.slot) {
                self.table_entry(list, u, compute);
            }
        }
        let mut made = Vec::new();
        for slot in fresh {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let entries: Vec<&UsedResource> = used
                .iter()
                .filter(tables)
                .filter(|u| u.slot == slot)
                .collect();
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let Some(host) = heap.host() else {
                continue;
            };
            let n = entries
                .iter()
                .map(|u| u.table_offset_dwords / 8 + 1)
                .max()
                .unwrap_or(1);
            let heap_id = std::ptr::from_ref(heap) as usize;
            let Some(records) = (n <= host.dynamic_capacity())
                .then(|| heap.table_records(table, n))
                .flatten()
            else {
                self.table_failed = true;
                continue;
            };
            let tables = self.fresh_tables.entry(heap_id).or_default();
            let mut allocated = false;
            let first = if let Some(&first) = tables.get(records.as_ref()) {
                first
            } else {
                let Some(first) = host.dynamic(n) else {
                    // Never borrow another live list's slots or submit a
                    // table whose descriptors were not initialized.
                    self.table_failed = true;
                    continue;
                };
                self.dynamic.push((heap_id, first, n));
                allocated = true;
                first
            };
            if allocated {
                // Keep the full interval, including intervening array slots.
                for (k, &record) in records.iter().enumerate() {
                    let index = first + k as u32;
                    match table_view(record) {
                        Some(view) => host.make_at(index, &view),
                        None => host.make_null(index),
                    }
                }
            }
            // Stages can use different entries of an equal table. The slot
            // range can be shared; resource preparation cannot be skipped.
            for u in entries {
                let k = (u.table_offset_dwords / 8) as usize;
                if let Some(view) = table_view(records[k]) {
                    readable(list, u, &view, true);
                }
            }
            if allocated {
                tables.insert(records, first);
            }
            made.push((slot, first * 32));
        }
        made
    }

    /// The first draws, at debug level: what they bind and read
    /// (diagnostics builds).
    #[cfg(feature = "diagnostics")]
    fn trace(&self, kind: &str, count: u32, first: u32) {
        static TRACED: AtomicU64 = AtomicU64::new(0);
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let listed = crate::capture::listed(crate::queue::presented() + 1);
        let logged = listed || TRACED.fetch_add(1, Ordering::Relaxed) < 64;
        let targets: Vec<String> = self
            .targets
            .iter()
            .filter_map(|&t| crate::resource::host_resource(t as This))
            .map(|t| {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                format!("{}x{} f{}", d.Width, d.Height, d.Format.0)
            })
            .collect();
        let v = self.vertex[0];
        let line = format!(
            "{kind} {count} from {first} topology {} pipeline {:#x} targets {targets:?} vb {:#x}+{} stride {} reads {:?}",
            self.topology,
            self.pipeline,
            v.address,
            v.size,
            v.stride,
            self.reads(false)
        );
        if logged {
            tracing::debug!("draw {line}");
        }
        let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if frame.len() < 256 {
            frame.push(line);
        }
    }

    /// In a listed frame: each small buffer (64 bytes at most) the draw's
    /// tables name, as title memory now and as its host copy when the list
    /// is submitted (before the GPU runs it) and after it has run.
    #[cfg(feature = "diagnostics")]
    fn probe_small(&self, list: &HostList, compute: bool) {
        if !crate::capture::listed(crate::queue::presented() + 1) {
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else { return };
        for stage in &p.stages {
            for u in &stage.translated().used {
                let (RootType::Table, Some(Root::Table(table))) =
                    (u.root, self.roots_ref(compute).get(usize::from(u.slot)))
                else {
                    continue;
                };
                let Some(view) = crate::descriptor::view(table + u64::from(u.table_offset_dwords) * 4) else {
                    continue;
                };
                let range = match view {
                    View::Constant(address, size) => Some((address, u64::from(size))),
                    ref v => crate::descriptor::host::buffer_range(v),
                };
                let Some((at, len)) = range.filter(|r| r.1 <= 64) else {
                    continue;
                };
                let Some(shadow) = crate::shadow::find(at, len) else { continue };
                let who = format!("{} {:?} s{}+{} {at:#x}+{len}", stage.shader.hash, u.kind, u.slot, u.table_offset_dwords);
                let (s, w) = (shadow.clone(), who.clone());
                list.defer_host(move || tracing::info!(target: "victory_gpu::trace", "small {w} at submission: {}", s.peek(at, len)));
                list.defer_after(move || tracing::info!(target: "victory_gpu::trace", "small {who} after the list: {}", shadow.peek(at, len)));
            }
        }
    }

    /// In a listed frame, right after a dispatch: each 1D or 3D texture it
    /// writes through a table, read back in the list itself (what the
    /// dispatch left there, before anything later in the frame).
    #[cfg(feature = "diagnostics")]
    fn probe_written(&self, list: &HostList) {
        if !crate::capture::listed(crate::queue::presented() + 1) {
            return;
        }
        let (Some(p), Some(host)) = (crate::pipeline::host_pipeline(self.pipeline as This), crate::host::host()) else {
            return;
        };
        for stage in &p.stages {
            for u in stage.translated().used.iter().filter(|u| u.kind == ResourceType::Uav) {
                let (RootType::Table, Some(Root::Table(table))) = (u.root, self.compute.get(usize::from(u.slot))) else {
                    continue;
                };
                let Some(View::Unordered(r, ..)) = crate::descriptor::view(table + u64::from(u.table_offset_dwords) * 4) else {
                    continue;
                };
                let Some(t) = crate::resource::host_resource(r as This) else { continue };
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE1D && d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE3D {
                    continue;
                }
                const COPY_SOURCE: u32 = 0x800;
                const UNORDERED: u32 = 0x8;
                // All UAV writes done before the copy.
                let all = D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
                    },
                };
                // SAFETY: an open list.
                unsafe { list.recording().ResourceBarrier(&[all]) };
                crate::list::ensure_state(list, r as This, COPY_SOURCE);
                let read = crate::capture::read_back(&host.device, list.recording(), t);
                crate::list::ensure_state(list, r as This, UNORDERED);
                let Some((buffer, f)) = read else { continue };
                list.keep(&buffer);
                struct Readback(ID3D12Resource);
                // SAFETY: D3D12 objects are free-threaded.
                unsafe impl Send for Readback {}
                // SAFETY: as above.
                unsafe impl Sync for Readback {}
                let buffer = Readback(buffer);
                let who = format!("{} s{}+{} {r:#x} {}x{}x{} f{}", stage.shader.hash, u.slot, u.table_offset_dwords, d.Width, d.Height, d.DepthOrArraySize, d.Format.0);
                list.defer_after(move || {
                    let mut data = std::ptr::null_mut();
                    // SAFETY: the list has run; the mapping spans the first slice.
                    unsafe {
                        if buffer.0.Map(0, None, Some(&mut data)).is_err() {
                            return;
                        }
                        let n = (f.Footprint.RowPitch * f.Footprint.Height) as usize / 4;
                        let w = std::slice::from_raw_parts(data.cast::<u32>(), n);
                        let set = w.iter().filter(|&&x| x != 0).count();
                        let first: Vec<f32> = w.iter().take(8).map(|&x| f32::from_bits(x)).collect();
                        tracing::info!(target: "victory_gpu::trace", "written {who} right after the dispatch: {set} of {n} dwords set, first {first:?}");
                        buffer.0.Unmap(0, None);
                    }
                });
            }
        }
    }

    /// What a draw's shaders read, one line per used resource (diagnostics
    /// builds).
    #[cfg(feature = "diagnostics")]
    fn reads(&self, compute: bool) -> Vec<String> {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return vec!["no host pipeline".into()];
        };
        let mut out = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            for u in &t.used {
                let what = match (u.root, self.roots_ref(compute).get(usize::from(u.slot))) {
                    (RootType::Table, Some(Root::Table(table))) => {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        let view = crate::descriptor::view(handle);
                        let mut text = describe(view.as_ref());
                        // The title's view words, and whether the host slot
                        // the shader reads was made from a view.
                        if let Some(View::Shader(_, Some(w)) | View::Unordered(_, _, Some(w))) = &view {
                            text.push_str(&format!(" words {w:?}"));
                        }
                        let slot = (self.heaps[0] != 0)
                            .then(|| crate::descriptor::heap(self.heaps[0] as This))
                            .and_then(|h| {
                                let i = handle.checked_sub(h.start)? / 32;
                                h.host()?.describe_slot(u32::try_from(i).ok()?)
                            });
                        let current: String = view.as_ref().map(|v| format!("{v:?}")).unwrap_or_default().chars().take(120).collect();
                        text.push_str(match slot.as_deref() {
                            Some("null") => " slot null",
                            Some(s) if s.starts_with(&current) => " slot made",
                            Some(_) => " slot stale",
                            None => " slot unknown",
                        });
                        text
                    }
                    (root, _) => format!("{root:?}"),
                };
                out.push(format!(
                    "{} {:?} s{}+{}: {what}",
                    stage.shader.hash, u.kind, u.slot, u.table_offset_dwords
                ));
            }
        }
        out
    }

    pub(crate) fn draw(
        &mut self,
        list: &HostList,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        if cfg!(feature = "paged-title") {
            self.record_strict_draw(list, crate::page_draw::DrawCall::Direct {
                vertices, instances, first, first_instance,
            });
            return;
        }
        #[cfg(feature = "diagnostics")]
        self.trace("draw", vertices, first);
        // Rectangles: completed by the pipeline's geometry shader, else
        // expanded into vertex buffers of their own.
        let rects = self.topology == RECTANGLES && !self.completes_rectangles();
        let ran = self.prepare(list, false)
            && (!rects || self.rectangles(list, vertices, first) || {
                not_run("rectangle vertices outside their buffer");
                false
            });
        if ran {
            // Rectangles were expanded into their own vertex buffers.
            let (count, from) = if rects {
                (vertices / 3 * 6, 0)
            } else {
                (vertices, first)
            };
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false));
            #[cfg(feature = "diagnostics")]
            self.probe_small(list, false);
            // SAFETY: an open list with everything set.
            unsafe {
                list.recording()
                    .DrawInstanced(count, instances, from, first_instance)
            };
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(list, self.pipeline, "draw");
        }
    }

    /// Whether the bound pipeline's geometry shader completes rectangles.
    fn completes_rectangles(&self) -> bool {
        crate::pipeline::host_pipeline(self.pipeline as This).is_some_and(|p| p.rectangles)
    }

    /// Binds vertex buffers holding the rectangles of vertices `first..first
    /// + count` as triangles, made when the list is submitted.
    fn rectangles(&self, list: &HostList, count: u32, first: u32) -> bool {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        let rects = u64::from(count / 3);
        for slot in 0..32 {
            let elements: Vec<(u32, u32)> = p
                .inputs
                .iter()
                .filter(|e| e.slot == slot && !e.per_instance)
                .map(|e| (e.offset, e.format))
                .collect();
            if elements.is_empty() {
                continue;
            }
            let v = self.vertex[slot as usize];
            let stride = u64::from(v.stride);
            let from = v.address + u64::from(first) * stride;
            if v.address == 0
                || stride == 0
                || u64::from(first + count) * stride > u64::from(v.size)
                || crate::shadow::find(from, rects * 3 * stride).is_none()
            {
                once!(slot, "rectangle vertices outside their buffer");
                return false;
            }
            let Some((gpu, to)) = list.scratch(rects * 6 * stride) else {
                return false;
            };
            let (from, to, stride) = (from as usize, to as usize, stride as usize);
            let n = rects as usize;
            list.defer_host(move || {
                // SAFETY: the vertices lie in a title buffer (checked when
                // recorded) and the scratch memory is the list's.
                let (src, dst) = unsafe {
                    (
                        std::slice::from_raw_parts(from as *const u8, n * 3 * stride),
                        std::slice::from_raw_parts_mut(to as *mut u8, n * 6 * stride),
                    )
                };
                crate::primitive::expand_rects(src, dst, stride, &elements);
            });
            let view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: (rects * 6 * stride as u64) as u32,
                StrideInBytes: v.stride,
            };
            // SAFETY: an open list; the view names live scratch memory.
            unsafe { list.recording().IASetVertexBuffers(slot, Some(&[view])) };
        }
        true
    }

    pub(crate) fn draw_indexed(
        &mut self,
        list: &HostList,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        if cfg!(feature = "paged-title") {
            self.record_strict_draw(list, crate::page_draw::DrawCall::Indexed {
                indices, instances, first, base, first_instance,
            });
            return;
        }
        #[cfg(feature = "diagnostics")]
        self.trace("indexed", indices, first);
        let rects = self.topology == RECTANGLES && !self.completes_rectangles();
        if rects {
            once!("indexed rectangles not drawn");
            not_run("indexed rectangles");
        }
        let ran = !rects && self.prepare(list, true);
        if ran {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.recording()
                    .DrawIndexedInstanced(indices, instances, first, base, first_instance)
            };
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(list, self.pipeline, "indexed");
        }
    }
}

impl State {
    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them. False if it cannot run on the host.
    pub(crate) fn execute_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
    ) -> bool {
        if cfg!(feature = "paged-title") {
            if max == 0 {
                return true;
            }
            use crate::page_indirect::{IndirectArgs, IndirectKind};
            let kind = match kind {
                0 => IndirectKind::Draw,
                1 => IndirectKind::Indexed,
                2 => IndirectKind::Dispatch,
                _ => {
                    list.mark_unlowered();
                    return false;
                }
            };
            let args = IndirectArgs { kind, stride, max, arguments, count };
            return if kind == IndirectKind::Dispatch {
                self.record_strict_dispatch(list, crate::page_dispatch::DispatchCall::Indirect(args))
            } else {
                self.record_strict_draw(list, crate::page_draw::DrawCall::Indirect(args))
            };
        }
        // Skip before the argument clamp can mutate the indirect buffer.
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            crate::pipeline::note_drop(self.pipeline as This);
            list.note_op(|| format!("DROPPED indirect kind {kind} (skipped) {}", crate::pipeline::shaders_of(self.pipeline as This)));
            return false;
        }
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        #[cfg(feature = "diagnostics")]
        let pipeline = self.pipeline as This;
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| {
            crate::pipeline::note_why(format!("indirect kind {kind}: {what}"));
            list.note_op(|| format!("DROPPED indirect kind {kind} max {max} ({what}) {}", crate::pipeline::shaders_of(pipeline)));
        };
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        // A call with no host pipeline cannot run, and that is the common
        // case while the world builds its pipelines: #400 measures 1 400-1 750
        // indirect calls a frame dropped this way. Drop it here, before the
        // argument refresh and the clamp dispatch, which would otherwise copy
        // the arguments and dispatch a compute shader for records nothing
        // reads. Same order as the `is_skipped` check above, for the same
        // reason.
        if crate::pipeline::host_pipeline(self.pipeline as This).is_none() {
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            why("no host pipeline");
            return false;
        }
        let len = u64::from(stride) * u64::from(max.max(1));
        let Some(args) = crate::shadow::find(arguments, len) else {
            why("arguments outside title buffers");
            return false;
        };
        let counter = match count {
            Some(at) => match crate::shadow::find(at, 4) {
                Some(s) => Some((s, at)),
                None => {
                    why("count outside title buffers");
                    return false;
                }
            },
            None => None,
        };
        let Some(signature) = matches!(kind, DRAW | INDEXED | DISPATCH | MESH)
            .then(|| host_signature(kind, stride))
            .flatten()
        else {
            why("no host signature");
            return false;
        };
        args.refresh(list, arguments, len);
        if let Some((s, at)) = &counter {
            s.refresh(list, *at, 4);
        }
        // The clamp writes the records where it reads them: for a buffer in
        // place that is title memory, so they are clamped in a copy.
        let (records, offset, gpu) = if args.in_place() {
            match crate::indirect::copied(list, &args.buffer, args.offset(arguments), len) {
                Some((copy, gpu)) => (copy, 0, gpu),
                None => {
                    why("no copy of in-place arguments");
                    return false;
                }
            }
        } else {
            (args.buffer.clone(), args.offset(arguments), args.gpu(arguments))
        };
        // Bounded first: the clamp binds a compute pipeline of its own.
        if !crate::indirect::bound(list, &records, gpu, (kind, stride, max)) {
            why("not bounded (no clamp, or arguments not 4-byte aligned)");
            return false;
        }
        self.host_pipeline = 0;
        self.host_roots[1] = 0;
        let ran = match kind {
            DRAW | INDEXED => {
                (self.topology != RECTANGLES || self.completes_rectangles() || {
                    not_run("rectangles");
                    false
                }) && self.prepare(list, kind == INDEXED)
            }
            MESH => self.prepare_for(list, false, true),
            DISPATCH => match crate::pipeline::host_pipeline(self.pipeline as This) {
                Some(p) if p.compute => self.bind(list, p, true),
                _ => false,
            },
            _ => false,
        };
        if !ran {
            crate::indirect::done(list, &records);
            #[cfg(feature = "diagnostics")]
            match crate::pipeline::host_pipeline(self.pipeline as This) {
                None => {
                    crate::pipeline::note_drop(self.pipeline as This);
                    list.note_op(|| format!("DROPPED indirect kind {kind} (no host pipeline) {}", crate::pipeline::shaders_of(pipeline)));
                }
                Some(p) => why(&format!(
                    "prepare failed (compute {}, mesh {}): {}",
                    p.compute,
                    p.mesh,
                    NOT_RUN.with(|w| w.replace(""))
                )),
            }
            return false;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(format!("indirect kind {kind} max {max} at {arguments:#x}"), kind == DISPATCH)
        });
        // SAFETY: an open list with everything set; the buffers are host
        // copies the list keeps alive (refresh), the signature is cached.
        unsafe {
            list.recording().ExecuteIndirect(
                &signature,
                max,
                &records,
                offset,
                counter.as_ref().map(|(s, _)| &s.buffer),
                counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
            )
        };
        // A listed frame: what the records held (after the clamp).
        #[cfg(feature = "diagnostics")]
        if crate::capture::dump_on() {
            let count = counter.as_ref().map(|(s, at)| (&s.buffer, s.offset(*at)));
            let name = crate::pipeline::shaders_of(pipeline);
            crate::capture::read_indirect(list, (&records, offset), count, (kind, stride, max), name);
        }
        crate::indirect::done(list, &records);
        if kind == DISPATCH {
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
        } else {
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        true
    }
}

/// The host command signature for one argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once.
pub(crate) fn host_signature(kind: u32, stride: u32) -> Option<ID3D12CommandSignature> {
    static MADE: std::sync::Mutex<Vec<((u32, u32), ID3D12CommandSignature)>> =
        std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        return Some(s.clone());
    }
    let argument = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of one argument; no root arguments.
    let result = unsafe {
        crate::host::host()?.device.CreateCommandSignature(
            &desc,
            None::<&ID3D12RootSignature>,
            &mut signature,
        )
    };
    if let Err(e) = result {
        once!(error = %e, kind, stride, "host command signature refused");
        return None;
    }
    let signature = signature?;
    made.push(((kind, stride), signature.clone()));
    Some(signature)
}

/// Gets what a shader reads or writes through a view ready: buffer data
/// copied, textures moved to the state they need. `marked`: a shader's
/// writes through the view mark their pages (every translated shader's do);
/// the host's own (clears) do not.
fn readable(list: &HostList, u: &UsedResource, view: &View, marked: bool) {
    const SHADER_RESOURCE: u32 = 0x40 | 0x80;
    const UNORDERED_ACCESS: u32 = 0x8;
    let placed = crate::descriptor::host::placed_texture(view);
    // A placed view of memory the GPU writes through another title texture
    // reads that one's host copy.
    if let Some(View::Shader(b, _)) = &placed
        && let Some(a) = crate::resource::alias_source(*b as This)
    {
        crate::texture::alias_copy(list, a, *b as This);
    }
    let view = placed.as_ref().unwrap_or(view);
    // Buffer views read title memory: their range is copied at submission;
    // what shaders write through unordered access goes back to it after.
    // Constant buffers (raw views) to their copy's end at most; others in
    // the copy their host view reads.
    let range = match *view {
        View::Constant(address, size) => crate::shadow::find(address, 1).and_then(|s| {
            let len = u64::from(size).min(s.left(address));
            crate::shadow::find(address, len).map(|s| (s, address, len))
        }),
        _ => crate::descriptor::host::view_copy(view),
    };
    if let Some((shadow, address, len)) = range
        && len > 0
    {
        shadow.refresh(list, address, len);
        if u.kind == ResourceType::Uav {
            shadow.gpu_writes(list, address, len, marked);
        }
    }
    match (view, u.kind) {
        (View::Shader(resource, _), ResourceType::Srv) => {
            if let Some(t) = crate::resource::host_resource(*resource as This)
                && crate::texture::title_written(t)
            {
                crate::texture::upload(list, *resource as This, t);
            }
            crate::list::ensure_state(list, *resource as This, SHADER_RESOURCE);
            #[cfg(feature = "diagnostics")]
            if let Some(t) = crate::resource::host_resource(*resource as This) {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Flags.0 & 7 == 0 {
                    tracing::debug!(
                        resource = format_args!("{resource:#x}"),
                        ?d,
                        "a shader reads a texture the title fills"
                    );
                }
            }
        }
        (View::Unordered(resource, _, _), ResourceType::Uav) => {
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(list, *resource as This);
            crate::list::ensure_state(list, *resource as This, UNORDERED_ACCESS);
        }
        _ => {}
    }
}

/// A view in a few words (diagnostics builds).
#[cfg(feature = "diagnostics")]
fn describe(view: Option<&View>) -> String {
    let texture = |r: This| {
        crate::resource::host_resource(r).map(|t| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            // Its console memory: another view of it (a placed one) is
            // another host texture.
            let at = crate::resource::texture_memory(r).map_or(0, |(p, _)| p as u64);
            format!("{}x{} f{} flags {:#x} mem {at:#x}", d.Width, d.Height, d.Format.0, d.Flags.0)
        })
    };
    // Small buffers (exposure, counters): their title memory now, as floats.
    let small = |range: Option<(u64, u64)>| match range {
        Some((at, len)) if len <= 64 && crate::texture::committed(at as *const u8, len as usize) => {
            // SAFETY: `len` committed bytes of title memory.
            let b = unsafe { std::slice::from_raw_parts(at as *const u8, len as usize) };
            let f: Vec<f32> = b.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect();
            format!(" = {f:?}")
        }
        _ => String::new(),
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This).unwrap_or_else(|| {
            let range = crate::descriptor::host::buffer_range(v);
            format!("buffer {range:?}{}", small(range))
        }),
        // A buffer has no host texture: its title range, as for a shader view.
        Some(v @ View::Unordered(r, ..)) => texture(*r as This).map_or_else(
            || {
                let range = crate::descriptor::host::buffer_range(v);
                format!("uav buffer {range:?}{}", small(range))
            },
            |t| format!("uav {t}"),
        ),
        Some(View::Placed(u, r, d)) => format!(
            "placed uav={u} dim {} {}x{} f{} layout {:#x} at {:#x}",
            r[0],
            u64::from(r[4]) | u64::from(r[5]) << 32,
            r[6],
            r[8],
            r[11],
            u64::from(d[10]) | u64::from(d[11]) << 32
        ),
        // The first four values, as a colour rectangle's constants are, read
        // through the kernel (asking `VirtualQuery` first cost milliseconds
        // per draw in the title's pools, with every draw noted).
        Some(View::Constant(address, size)) if *size >= 16 => {
            let mut bytes = [0u8; 16];
            if crate::host::read_into(*address as *const u8, &mut bytes) {
                let v: [f32; 4] = std::array::from_fn(|i| f32::from_le_bytes([0, 1, 2, 3].map(|k| bytes[4 * i + k])));
                format!("Constant({address:#x}, {size}) {v:?}")
            } else {
                format!("Constant({address:#x}, {size})")
            }
        }
        Some(v) => format!("{v:?}").chars().take(60).collect(),
    }
}

/// A user-data entry that takes the heap index of a root view's view.
fn is_root_view(u: &victory_shader::emit::UserData) -> bool {
    u.root == RootType::Descriptor && u.dword >= victory_shader::emit::ROOT_SRV_VIEW
}

/// Whether a shader reads root views through views the host makes.
fn has_root_views(t: &victory_shader::emit::HostShader) -> bool {
    t.user_data.iter().any(is_root_view)
}

/// The console's own primitive topologies.
const TRIANGLE_FAN: u32 = 6;
const RECTANGLES: u32 = 17;
const LINE_LOOP: u32 = 18;
const QUADS: u32 = 19;
const QUAD_STRIP: u32 = 20;

impl State {
    pub(crate) fn set_viewports(&mut self, list: &HostList, viewports: &[D3D12_VIEWPORT]) {
        self.page_viewport = match viewports { [v] => Some(*v), _ => None };
        if cfg!(feature = "paged-title") { return; }
        // SAFETY: an open list.
        unsafe { list.recording().RSSetViewports(viewports) };
    }

    pub(crate) fn set_scissors(&mut self, list: &HostList, rects: &[RECT]) {
        self.page_scissor = match rects { [r] => Some(*r), _ => None };
        if cfg!(feature = "paged-title") { return; }
        // SAFETY: an open list.
        unsafe { list.recording().RSSetScissorRects(rects) };
    }

    pub(crate) fn set_blend_factor(&mut self, list: &HostList, factor: Option<&[f32; 4]>) {
        self.page_blend = factor.copied();
        if cfg!(feature = "paged-title") { return; }
        // SAFETY: an open list.
        unsafe { list.recording().OMSetBlendFactor(factor) };
    }
}

impl State {
    pub(crate) fn set_stencil_ref(&mut self, list: &HostList, reference: u32) {
        self.page_stencil = reference;
        if !cfg!(feature = "paged-title") {
            // SAFETY: an open list.
            unsafe { list.recording().OMSetStencilRef(reference) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{HRESULT, S_OK};
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::core::s;

    /// Reads 16 dwords 64 bytes apart from `offset` past the root view
    /// (heap index in the table) as the translated shaders do: the root
    /// address's offset in the view plus their own.
    const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0)), DescriptorTable(UAV(u0)), RootConstants(num32BitConstants=2, b0)"
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
cbuffer c : register(b0) { uint root; uint offset; }
[RootSignature(RS)]
[numthreads(16, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    dst.Store(id.x * 4, src.Load(root + offset + id.x * 64));
}
"#;

    fn compile() -> (ID3D12RootSignature, ID3D12PipelineState) {
        compile_cs(SOURCE)
    }

    /// A compute shader with its root signature inside (`RS`).
    fn compile_cs(source: &str) -> (ID3D12RootSignature, ID3D12PipelineState) {
        let host = crate::host::host().unwrap();
        let (mut code, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
        // SAFETY: the source and names outlive the call.
        unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("root"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                Some(&mut errors),
            )
            .unwrap();
        }
        let code = code.unwrap();
        // SAFETY: the blob's own bytes.
        let bytes = unsafe {
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
        };
        // SAFETY: a compiled shader with its root signature inside.
        unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            };
            let pso = host.device.CreateComputePipelineState(&desc).unwrap();
            (root, pso)
        }
    }

    /// A title buffer of `bytes` in memory of its own.
    fn buffer(bytes: u64) -> This {
        let desc: [u32; 14] = [1, 0, 0, 0, bytes as u32, (bytes >> 32) as u32, 1, 1 | 1 << 16, 0, 1, 0, 1, 0, 0];
        let properties = [1u32, 0, 0, 0, 0];
        let mut out = std::ptr::null();
        let iid = crate::object::guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
        let made: HRESULT = crate::resource::create_committed(
            std::ptr::null(),
            properties.as_ptr().cast(),
            0,
            desc.as_ptr().cast(),
            0,
            std::ptr::null(),
            &iid,
            &mut out,
        );
        assert_eq!(made, S_OK);
        out
    }

    /// Root views (`victory_shader::emit`, "Root SRVs and root UAVs") through
    /// our device: a raw view of the host copy holding the root address,
    /// brought up to date from there to the copy's end, so data 1.5 MiB past
    /// the root address (past the 1 MiB the old root views brought up to
    /// date) reads right; reads past the copy's end read 0 (a host root
    /// descriptor read whatever lay there). Two root addresses in one copy
    /// share its view.
    #[test]
    fn root_views_reach_their_copy_and_read_zeros_past_it() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        const MIB: u64 = 1 << 20;
        let (src, dst) = (buffer(4 * MIB), buffer(256));
        let (src_bytes, _) = crate::resource::buffer_bytes(src).unwrap();
        let (dst_bytes, _) = crate::resource::buffer_bytes(dst).unwrap();
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), S_OK);
        let title_heap = crate::descriptor::heap(heap);
        let uav: [u32; 10] = [39, 1, 0, 0, 64, 0, 0, 0, 1, 0];
        crate::descriptor::create_uav(null, dst, null, &uav, title_heap.start + 32);
        let heap_host = title_heap.host().expect("a host view heap");
        let (root, pso) = compile();
        let list = host.new_list().unwrap();
        let out = crate::shadow::find(dst_bytes as u64, 256).expect("the output's host copy");
        let src = src_bytes as u64;
        let cases = [
            // Data 1.5 MiB past the root address.
            (src + 0x100, 0x18_0000u32, (0..16).map(|k| 0x5000 + k).collect::<Vec<u32>>()),
            // The copy's last 256 bytes: four lanes inside, the rest past it.
            (src + 4 * MIB - 256, 0, (0..16).map(|k| if k < 4 { 0x7000 + k } else { 0 }).collect()),
        ];
        for (round, (address, offset, want)) in cases.into_iter().enumerate() {
            if round > 0 {
                list.reset();
            }
            // The title writes after recording, before submitting.
            let mut state = State::default();
            state.set_heaps(&[heap]);
            state.set_view(true, 0, address);
            let (index, at) = state.root_view(&list, true, 0, false).expect("a root view");
            assert_eq!(u64::from(at), address - src, "the root address's offset in the copy");
            state.set_view(true, 1, address - 0x40);
            let other = state.root_view(&list, true, 1, false).expect("a root view");
            assert_eq!(other, (index, at - 0x40), "one view per copy");
            let constants = [at, offset];
            // SAFETY: an open list, live objects of ours.
            unsafe {
                list.recording().SetDescriptorHeaps(&[Some(heap_host.heap.clone())]);
                list.recording().SetComputeRootSignature(&root);
                list.recording().SetComputeRootDescriptorTable(0, heap_host.gpu(index));
                list.recording().SetComputeRootDescriptorTable(1, heap_host.gpu(1));
                list.recording().SetComputeRoot32BitConstants(2, 2, constants.as_ptr().cast(), 0);
                list.recording().SetPipelineState(&pso);
                list.recording().Dispatch(1, 1, 1);
            }
            list.close();
            for k in 0..16u64 {
                let at = address + u64::from(offset) + 64 * k;
                if at + 4 <= src + 4 * MIB {
                    let value = if round == 0 { 0x5000 + k as u32 } else { 0x7000 + k as u32 };
                    // SAFETY: inside the title buffer.
                    unsafe { (at as *mut u32).write(value) };
                }
            }
            host.execute(&[&list]);
            assert_eq!(out.read_host(dst_bytes as u64, 16), want, "round {round}");
        }
    }

    /// Title memory of `bytes` (64 KiB aligned) for textures at an address.
    fn title_memory(bytes: usize) -> u64 {
        let layout = std::alloc::Layout::from_size_align(bytes, 1 << 16).unwrap();
        // SAFETY: a non-zero size; leaked, as title memory outlives a test.
        unsafe { std::alloc::alloc_zeroed(layout) as u64 }
    }

    /// A 2D resource description as words: size, format, layout, flags.
    fn texture_words(width: u32, height: u32, format: u32, layout: u32, flags: u32) -> [u32; 14] {
        [3, 0, 0, 0, width, 0, height, 1 | 1 << 16, format, 1, 0, layout, flags, 0]
    }

    /// A title texture (resource object) at `address`.
    fn texture_at_address(address: u64, words: &[u32; 14]) -> This {
        let iid = crate::object::guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
        let mut out = std::ptr::null();
        let made = crate::resource::create_at_address(std::ptr::null(), address, words.as_ptr().cast(), 0, std::ptr::null(), &iid, &mut out);
        assert_eq!(made, S_OK);
        out
    }

    /// A placed shader view (device 0x230) of a 2D texture at `address`.
    fn placed_view(address: u64, words: [u32; 14]) -> View {
        let mut desc = [0u32; 16];
        desc[..6].copy_from_slice(&[words[8], 4, 0x1688, 0, 0, 1]);
        (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
        View::Placed(false, words, desc)
    }

    fn used(kind: ResourceType) -> UsedResource {
        UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        }
    }

    /// Heap and table CBVs use raw SRVs in translated shaders. Clearing the
    /// title CBV must leave a null raw SRV of the same kind in the host slot.
    #[test]
    fn cleared_constant_view_reads_zero_as_a_raw_buffer() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let shader = compile_cs(
            r#"
#define RS "DescriptorTable(SRV(t0)), UAV(u0)"
ByteAddressBuffer cb : register(t0);
RWByteAddressBuffer out_value : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() { out_value.Store(0, cb.Load(0)); }
"#,
        );
        let slots = Slots::new(1);
        // SAFETY: a live, shader-visible view heap.
        let at = unsafe { slots.0.GetCPUDescriptorHandleForHeapStart() };
        let (out, readback) = host.shared_buffer(4).unwrap();
        let buffer = crate::resource::test_buffer(16);
        let (memory, _) = crate::resource::buffer_bytes(buffer).unwrap();
        let address = memory as u64;
        let value = 0x1122_3344u32;
        // SAFETY: the aligned start of the title's 16-byte buffer.
        unsafe { memory.cast::<u32>().write(value) };
        let cbv = [address as u32, (address >> 32) as u32, 16, 0];
        let mut storage = [0u64; 4];
        let handle = storage.as_mut_ptr() as u64;
        let mut got = Vec::new();
        for desc in [&cbv as *const [u32; 4], std::ptr::null()] {
            crate::descriptor::create_cbv(std::ptr::null(), desc, handle);
            let view = crate::descriptor::view(handle).unwrap();
            let _resource = crate::descriptor::host::make_view(&view, at);
            let list = host.new_list().unwrap();
            readable(&list, &used(ResourceType::Cbv), &view, false);
            // SAFETY: the shader outputs to this shared buffer.
            unsafe { readback.cast::<u32>().write(0xdead_beef) };
            run(&list, &slots, &shader, 0, Some(&out), (1, 1));
            list.close();
            host.execute(&[&list]);
            // SAFETY: the dispatch finished and the four bytes are mapped.
            got.push(unsafe { readback.cast::<u32>().read() });
        }
        let release = crate::object::slot::<extern "system" fn(This) -> u32>(buffer, 0x10);
        assert_eq!(release(buffer), 0);
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        assert_eq!(
            got,
            [value, 0],
            "raw constant view before and after clearing"
        );
    }

    /// Slots of a shader-visible view heap of ours.
    struct Slots(ID3D12DescriptorHeap, u32);

    impl Slots {
        fn new(n: u32) -> Self {
            let host = crate::host::host().unwrap();
            let desc = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: n,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            };
            // SAFETY: plain creation.
            let heap = unsafe { host.device.CreateDescriptorHeap(&desc) }.unwrap();
            // SAFETY: a plain query.
            let step = unsafe { host.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) };
            Slots(heap, step)
        }

        fn make(&self, i: u32, view: &View) {
            // SAFETY: a live heap.
            let mut at = unsafe { self.0.GetCPUDescriptorHandleForHeapStart() };
            at.ptr += (i * self.1) as usize;
            assert!(crate::descriptor::host::make_view(view, at).is_some(), "slot {i} made");
        }

        fn gpu(&self, i: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
            // SAFETY: a live shader-visible heap.
            let mut at = unsafe { self.0.GetGPUDescriptorHandleForHeapStart() };
            at.ptr += u64::from(i * self.1);
            at
        }
    }

    /// Records a dispatch of `pso` over `groups` with the table at slot
    /// `table` as parameter 0 and, if given, root UAV `out` as parameter 1.
    fn run(
        list: &HostList,
        slots: &Slots,
        (root, pso): &(ID3D12RootSignature, ID3D12PipelineState),
        table: u32,
        out: Option<&ID3D12Resource>,
        groups: (u32, u32),
    ) {
        // SAFETY: an open list and live objects.
        unsafe {
            list.recording().SetDescriptorHeaps(&[Some(slots.0.clone())]);
            list.recording().SetComputeRootSignature(root);
            list.recording().SetPipelineState(pso);
            list.recording().SetComputeRootDescriptorTable(0, slots.gpu(table));
            if let Some(out) = out {
                list.recording().SetComputeRootUnorderedAccessView(1, out.GetGPUVirtualAddress());
            }
            list.recording().Dispatch(groups.0, groups.1, 1);
        }
    }

    const READ: &str = r#"
#define RS "DescriptorTable(SRV(t0)), UAV(u1)"
Texture2D<float4> t : register(t0);
RWByteAddressBuffer o : register(u1);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    o.Store4((id.y * 64 + id.x) * 16, asuint(t.Load(int3(id.xy, 0))));
}
"#;

    /// One texture memory written through an unordered view of a resource
    /// object in one format and read through a placed view in another: a
    /// castable pair shares one host texture (no copy), a pair of other
    /// shapes over the same bytes (block-compressed against plain) is two,
    /// bridged by a copy at the read.
    #[test]
    fn a_texture_written_in_one_format_reads_in_another_through_a_placed_view() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(4);
        let read = compile_cs(READ);
        let (out, data) = host.shared_buffer(64 * 64 * 16).unwrap();
        // Format written and size, what the shader writes, format read and
        // size, the texel expected, whether one host texture.
        type Case = (u32, u32, &'static str, u32, u32, [f32; 4], bool);
        let cases: [Case; 2] = [
            // R32_UINT holding R9G9B9E5 (1, 0.5, 0.25): the sky's tables.
            (
                42,
                64,
                "RWTexture2D<uint> o : register(u0); void put(uint2 p) { o[p] = 256u | 128u << 9 | 64u << 18 | 16u << 27; }",
                67,
                64,
                [1.0, 0.5, 0.25, 1.0],
                true,
            ),
            // R32G32_UINT blocks read as BC1: solid red blocks.
            (
                17,
                16,
                "RWTexture2D<uint2> o : register(u0); void put(uint2 p) { o[p] = uint2(0xF800, 0); }",
                71,
                64,
                [1.0, 0.0, 0.0, 1.0],
                false,
            ),
        ];
        for (round, (written, size, put, format, read_size, want, shared)) in cases.into_iter().enumerate() {
            let address = title_memory(1 << 20);
            let object = texture_at_address(address, &texture_words(size, size, written, 0x105, 4));
            let view = placed_view(address, texture_words(read_size, read_size, format, 0x105, 0));
            let Some(View::Shader(placed, _)) = crate::descriptor::host::placed_texture(&view) else {
                panic!("round {round}: a placed texture");
            };
            let a = crate::resource::host_copy(object).unwrap();
            let b = crate::resource::host_copy(placed as This).unwrap();
            assert_eq!(std::sync::Arc::ptr_eq(&a, &b), shared, "round {round}: one host texture");
            let unordered = View::Unordered(object as usize, 0, Some([written, 4, 0, 0, 0, 0, 0, 0, 0, 0]));
            slots.make(0, &unordered);
            slots.make(1, &view);
            let write = compile_cs(&format!(
                "#define RS \"DescriptorTable(UAV(u0))\"\n{put}\n[RootSignature(RS)]\n[numthreads(8, 8, 1)]\nvoid main(uint3 id : SV_DispatchThreadID) {{ put(id.xy); }}"
            ));
            let list = host.new_list().unwrap();
            let copies = crate::texture::ALIAS_COPIES.load(Ordering::Relaxed);
            readable(&list, &used(ResourceType::Uav), &unordered, false);
            run(&list, &slots, &write, 0, None, (size / 8, size / 8));
            readable(&list, &used(ResourceType::Srv), &view, false);
            run(&list, &slots, &read, 1, Some(&out), (read_size / 8, read_size / 8));
            list.close();
            host.execute(&[&list]);
            let copied = crate::texture::ALIAS_COPIES.load(Ordering::Relaxed) > copies;
            assert_eq!(copied, !shared, "round {round}: a copy only where not shared");
            let n = read_size as usize;
            for i in [0usize, 7, n * 9 + 33, n * n - 1] {
                let (x, y) = (i % n, i / n);
                // SAFETY: inside the mapped output.
                let got = unsafe { (data.add((y * 64 + x) * 16) as *const [f32; 4]).read() };
                assert_eq!(got, want, "round {round} texel {x},{y}");
            }
        }
    }

    /// A depth texture with stencil (the console's 16-bit depth, the
    /// host's 24-bit) written through unordered views of its planes, then
    /// read as depth and stencil: the 16-bit depth arrives in 24 bits as
    /// the host converts the same value, the stencil as written. Rendered
    /// as depth (a clear), then used through the unordered view: back in
    /// 16 bits.
    #[test]
    fn a_depth_texture_written_through_unordered_views_reads_as_depth() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(4);
        let address = title_memory(1 << 22);
        let depth = texture_at_address(address, &texture_words(64, 64, 118, 0, 2));
        let copy = crate::resource::host_copy(depth).expect("a host depth texture");
        let planes = [
            View::Unordered(depth as usize, 0, Some([56, 4, 0, 0, 0, 0, 0, 0, 0, 0])),
            View::Unordered(depth as usize, 0, Some([62, 4, 0, 1, 0, 0, 0, 0, 0, 0])),
        ];
        slots.make(0, &planes[0]);
        slots.make(1, &planes[1]);
        let reads = [
            View::Shader(depth as usize, Some([119, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0])),
            View::Shader(depth as usize, Some([120, 4, 0x1688, 0, 0, 1, 1, 0, 0, 0])),
        ];
        slots.make(2, &reads[0]);
        slots.make(3, &reads[1]);
        let write = compile_cs(
            r#"
#define RS "DescriptorTable(UAV(u0, numDescriptors=2))"
RWTexture2D<float> d : register(u0);
RWTexture2D<uint> s : register(u1);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    d[id.xy] = (1 + id.x + 64 * id.y) * (1.0 / 65535.0);
    s[id.xy] = (id.x ^ id.y) & 0xff;
}
"#,
        );
        let read = compile_cs(
            r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors=2)), UAV(u1)"
Texture2D<float> d : register(t0);
Texture2D<uint2> s : register(t1);
RWByteAddressBuffer o : register(u1);
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    o.Store2((id.y * 64 + id.x) * 8, uint2(asuint(d.Load(int3(id.xy, 0))), s.Load(int3(id.xy, 0)).g));
}
"#,
        );
        let (out, data) = host.shared_buffer(64 * 64 * 8).unwrap();
        // The depth texture's planes as bytes, read back after.
        // SAFETY: a live resource.
        let dd = unsafe { copy.texture.GetDesc() };
        let mut f = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 2];
        let mut total = 0;
        // SAFETY: outputs for two subresources.
        unsafe { host.device.GetCopyableFootprints(&dd, 0, 2, 0, Some(f.as_mut_ptr()), None, None, Some(&mut total)) };
        let (raw, raw_data) = host.shared_buffer(total).unwrap();
        let list = host.new_list().unwrap();
        for v in &planes {
            readable(&list, &used(ResourceType::Uav), v, false);
        }
        run(&list, &slots, &write, 0, None, (8, 8));
        for v in &reads {
            readable(&list, &used(ResourceType::Srv), v, false);
        }
        assert!(list.take_rebind(), "the conversion ran in the list");
        run(&list, &slots, &read, 2, Some(&out), (8, 8));
        crate::list::ensure_state(&list, depth, 0x800);
        let at = |r: &ID3D12Resource, sub: u32| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the copy does not release.
            pResource: unsafe { std::mem::transmute_copy(r) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: sub },
        };
        let to = |fp: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&raw) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp },
        };
        // SAFETY: an open list, the texture in the copy source state.
        unsafe {
            list.recording().CopyTextureRegion(&to(f[0]), 0, 0, 0, &at(&copy.texture, 0), None);
            list.recording().CopyTextureRegion(&to(f[1]), 0, 0, 0, &at(&copy.texture, 1), None);
        }
        list.close();
        host.execute(&[&list]);
        let d24 = |u: u32| u * 256 + (u * 255 + 32767) / 65535;
        for (x, y) in [(0usize, 0usize), (5, 0), (63, 2), (17, 40), (63, 63)] {
            let k = (1 + x + 64 * y) as u32;
            // SAFETY: inside the mapped buffers.
            let [depth_bits, stencil] = unsafe { (data.add((y * 64 + x) * 8) as *const [u32; 2]).read() };
            let got = f32::from_bits(depth_bits);
            assert!((got - k as f32 / 65535.0).abs() < 1e-7, "depth at {x},{y}: {got}");
            assert_eq!(stencil, ((x ^ y) & 0xff) as u32, "stencil at {x},{y}");
            // SAFETY: inside the mapped buffer, by the footprints.
            let (word, byte) = unsafe {
                (
                    (raw_data.add(f[0].Offset as usize + y * f[0].Footprint.RowPitch as usize + x * 4) as *const u32).read(),
                    raw_data.add(f[1].Offset as usize + y * f[1].Footprint.RowPitch as usize + x).read(),
                )
            };
            assert_eq!(word & 0xff_ffff, d24(k), "24-bit depth at {x},{y}");
            assert_eq!(u32::from(byte), ((x ^ y) & 0xff) as u32);
        }
        // Rendered as depth (a clear of both planes), then used through
        // the unordered view: copied to the twin, converted back.
        list.reset();
        crate::list::ensure_state(&list, depth, 0x10);
        let dsv = D3D12_DEPTH_STENCIL_VIEW_DESC {
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_D24_UNORM_S8_UINT,
            ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        let handle = host.dsv(&copy.texture, Some(&dsv));
        // SAFETY: an open list, the texture in the depth write state.
        unsafe {
            list.recording().ClearDepthStencilView(handle, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.5, 0x11, None)
        };
        readable(&list, &used(ResourceType::Uav), &planes[0], false);
        let twin = copy.twin.get().and_then(Option::as_ref).expect("a twin");
        let mut g = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 2];
        for (p, plane) in twin.planes.iter().enumerate() {
            // SAFETY: a live resource.
            let pd = unsafe { plane.texture.GetDesc() };
            // SAFETY: an output for one subresource.
            unsafe { host.device.GetCopyableFootprints(&pd, 0, 1, f[p].Offset, Some(&mut g[p]), None, None, None) };
            if let Some(b) = list.move_texture(plane, 0x800) {
                // SAFETY: an open list.
                unsafe { list.recording().ResourceBarrier(&[b]) };
            }
        }
        // SAFETY: an open list, the twin's planes in the copy source state.
        unsafe {
            list.recording().CopyTextureRegion(&to(g[0]), 0, 0, 0, &at(&twin.planes[0].texture, 0), None);
            list.recording().CopyTextureRegion(&to(g[1]), 0, 0, 0, &at(&twin.planes[1].texture, 0), None);
        }
        list.close();
        host.execute(&[&list]);
        for (x, y) in [(0usize, 0usize), (63, 63)] {
            // SAFETY: inside the mapped buffer, by the footprints.
            let (half, byte) = unsafe {
                (
                    (raw_data.add(g[0].Offset as usize + y * g[0].Footprint.RowPitch as usize + x * 2) as *const u16).read(),
                    raw_data.add(g[1].Offset as usize + y * g[1].Footprint.RowPitch as usize + x).read(),
                )
            };
            assert_eq!(half, 32768, "16-bit depth at {x},{y}");
            assert_eq!(byte, 0x11, "stencil at {x},{y}");
        }
    }


    /// Unordered-access-only 1D and 3D title textures (the colour grading
    /// bakes: a 4096 R32_FLOAT curve, a 64^3 RGBA16F table) keep what a
    /// dispatch writes through their unordered views, whole (a 3D view of
    /// every depth slice, -1), and a shader view reads it.
    #[test]
    fn unordered_1d_and_3d_textures_keep_their_writes() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let slots = Slots::new(2);
        let (out, data) = host.shared_buffer(64 * 64 * 16).unwrap();
        // Description words (dimension, width, height, depth | mips << 16,
        // format), the unordered view's and shader view's words, the
        // shaders' bodies, groups, a texel to check and its value.
        type Case = ([u32; 5], [u32; 10], [u32; 10], &'static str, &'static str, (u32, u32), [f32; 4]);
        let cases: [Case; 2] = [
            (
                [2, 4096, 1, 1 | 1 << 16, 41],
                [41, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                [41, 2, 0x1688, 0, 0, 1, 0, 0, 0, 0],
                "RWTexture1D<float> o : register(u0); void put(uint3 p) { if (p.y == 0 && p.z == 0) o[p.x] = p.x * 0.25; }",
                "Texture1D<float4> t : register(t0); float4 get(uint3 p) { return t.Load(int2(p.y * 64 + p.x, 0)); }",
                (512, 1),
                [(2 * 64 + 7) as f32 * 0.25, 0.0, 0.0, 1.0],
            ),
            (
                [4, 64, 64, 64 | 1 << 16, 10],
                [10, 8, 0, 0, u32::MAX, 0, 0, 0, 0, 0],
                [10, 8, 0x1688, 0, 0, 1, 0, 0, 0, 0],
                "RWTexture3D<float4> o : register(u0); void put(uint3 p) { o[p] = float4(p, 1) / 64; }",
                "Texture3D<float4> t : register(t0); float4 get(uint3 p) { return t.Load(int4(p.x, p.y, 40, 0)); }",
                (8, 8),
                [7.0 / 64.0, 2.0 / 64.0, 40.0 / 64.0, 1.0 / 64.0],
            ),
        ];
        for (round, (d, uav, srv, put, get, groups, want)) in cases.into_iter().enumerate() {
            let address = title_memory(1 << 22);
            let words = [d[0], 0, 0, 0, d[1], 0, d[2], d[3], d[4], 1, 0, 0, 4, 0];
            let texture = texture_at_address(address, &words);
            let unordered = View::Unordered(texture as usize, 0, Some(uav));
            let shader = View::Shader(texture as usize, Some(srv));
            slots.make(0, &unordered);
            slots.make(1, &shader);
            let write = compile_cs(&format!(
                "#define RS \"DescriptorTable(UAV(u0))\"\n{put}\n[RootSignature(RS)]\n[numthreads(8, 8, 1)]\nvoid main(uint3 id : SV_DispatchThreadID) {{ for (uint z = 0; z < 64; z++) put(uint3(id.xy, z)); }}"
            ));
            let read = compile_cs(&format!(
                "#define RS \"DescriptorTable(SRV(t0)), UAV(u1)\"\n{get}\nRWByteAddressBuffer o : register(u1);\n[RootSignature(RS)]\n[numthreads(8, 8, 1)]\nvoid main(uint3 id : SV_DispatchThreadID) {{ o.Store4((id.y * 64 + id.x) * 16, asuint(get(id))); }}"
            ));
            let list = host.new_list().unwrap();
            readable(&list, &used(ResourceType::Uav), &unordered, false);
            run(&list, &slots, &write, 0, None, groups);
            readable(&list, &used(ResourceType::Srv), &shader, false);
            run(&list, &slots, &read, 1, Some(&out), (8, 8));
            list.close();
            host.execute(&[&list]);
            // SAFETY: inside the mapped output: texel (7, 2) of the read.
            let got = unsafe { (data.add((2 * 64 + 7) * 16) as *const [f32; 4]).read() };
            assert_eq!(got, want, "round {round}");
        }
    }

    /// A structured view the title places off its buffer's elements (the
    /// culling constants at a ring position) is made over a copy of its own
    /// bytes; one on an element views the buffer's copy.
    #[test]
    fn structured_views_off_their_buffers_elements_get_their_own_copy() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let src = buffer(1 << 16);
        let (bytes, _) = crate::resource::buffer_bytes(src).unwrap();
        let start = bytes as u64;
        // SAFETY: a plain CPU-side heap of one view.
        let cpu: ID3D12DescriptorHeap = unsafe {
            host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                ..Default::default()
            })
        }
        .unwrap();
        // SAFETY: a live heap.
        let at = unsafe { cpu.GetCPUDescriptorHandleForHeapStart() };
        // Four 456-byte elements placed at `address` (resource and view
        // descriptions as the title passes them to device 0x230).
        let view = |address: u64| {
            let resource: [u32; 14] = [1, 0, 0, 0, 4 * 456, 0, 1, 1 | 1 << 16, 0, 1, 0, 1, 0, 0];
            let mut desc = [0u32; 16];
            desc[..10].copy_from_slice(&[0, 1, 5768, 0, 0, 0, 4, 456, 0, 0]);
            (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
            View::Placed(false, resource, desc)
        };
        let width = |address: u64| {
            let made = crate::descriptor::host::make_view(&view(address), at).expect("a view");
            // SAFETY: a live buffer.
            unsafe { made.GetDesc() }.Width
        };
        assert_eq!(width(start + 456), 1 << 16, "on an element: the buffer's copy");
        assert_eq!(width(start + 100), (4 * 456u64).next_multiple_of(256), "off: its own bytes");
        let (copy, address, len) = crate::descriptor::host::view_copy(&view(start + 100)).unwrap();
        assert_eq!((copy.range(), address, len), ((start + 100, 4 * 456), start + 100, 4 * 456));
        drop(copy);
        // A slot made from the view holds its copy: after more than a
        // thousand other views' copies, draws still find (and refresh) the
        // copy the slot shows; rewritten, the slot lets it go.
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), S_OK);
        let slots = crate::descriptor::heap(heap).host().expect("a host view heap");
        slots.make_at(0, &view(start + 100));
        let shown = std::sync::Arc::downgrade(&crate::descriptor::host::view_copy(&view(start + 100)).unwrap().0);
        for k in 1..1200 {
            if (100 + 4 * k) % 456 != 0 {
                crate::descriptor::host::view_copy(&view(start + 100 + 4 * k)).unwrap();
            }
        }
        let found = crate::descriptor::host::view_copy(&view(start + 100)).unwrap().0;
        assert!(shown.upgrade().is_some_and(|s| std::sync::Arc::ptr_eq(&s, &found)), "the slot's copy");
        drop(found);
        slots.make_null(0);
        assert!(shown.upgrade().is_none(), "released with the slot");

        // Dynamic tables retain their copy while their list owns the slot.
        let first = slots.dynamic(1).unwrap();
        let next = slots.dynamic(1).unwrap();
        let hold = |index, address| {
            let view = view(address);
            slots.make_at(index, &view);
            let copy = crate::descriptor::host::view_copy(&view).unwrap().0;
            std::sync::Arc::downgrade(&copy)
        };
        let released = hold(first, start + 100);
        let kept = hold(next, start + 104);
        let mut state = State::default();
        state.dynamic.push((heap as usize, first, 1));
        assert!(released.upgrade().is_some());
        drop(state);
        assert!(released.upgrade().is_none(), "the released table still owns its copy");
        assert!(kept.upgrade().is_some(), "another table still owns its copy");
        slots.release(next, 1);
        assert!(kept.upgrade().is_none());
    }
}
