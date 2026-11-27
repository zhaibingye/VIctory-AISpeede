//! DXIL emission (decision 0003, step 3): a console vertex, pixel, compute
//! or mesh shader as a PC DXIL container (SM 6.6) that host D3D12 accepts.
//! Compute shaders keep the console's thread-group size; one that depends
//! on the lane layout (cross-lane operations) declares the console's wave
//! size (`WaveSize`), so the host device must support it.
//!
//! What the host binds for a translated shader ([`HostShader`]):
//!
//! * **User data** at `b0, space 0` (32-bit root constants or a constant
//!   buffer, per stage): dword `index` holds a root parameter's value, as
//!   [`UserData`] lists. A root-constant dword is the constant. A
//!   descriptor table is the byte offset of its first descriptor in its
//!   heap: the descriptor index times 32 for CBV/SRV/UAV tables, times 16
//!   for sampler tables (the console's descriptor sizes; the code adds
//!   offsets in those units). The heap bases of dynamic resources are 0
//!   and not listed.
//! * **Descriptors** through SM 6.6 `ResourceDescriptorHeap[]` and
//!   `SamplerDescriptorHeap[]`, at the index the console address gives, so
//!   the host heaps keep the title's descriptor order. The root signature
//!   needs `CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED` and
//!   `SAMPLER_HEAP_DIRECTLY_INDEXED`.
//! * **Immediate samplers** as static samplers `s<n>, space 0`, `n` the
//!   position in [`HostShader::samplers`] (the console sampler words) and
//!   [`HostShader::static_samplers`] (the same decoded into a D3D12 static
//!   sampler description). Comparison sampling (added 2026-11-23): an
//!   immediate sampler it uses has a comparison filter there (bit 0x80);
//!   a heap sampler it uses must be a comparison sampler on the host. One
//!   of two immediate samplers chosen by uniform control flow takes two
//!   entries; the sampling runs with both and the result is chosen.
//! * **Root CBVs** as root descriptors `b<i>` in [`ROOT_CBV_SPACE`], `i`
//!   the title's root parameter index ([`HostShader::root_cbvs`]); the
//!   address the title sets is the buffer's.
//! * **Root SRVs and root UAVs** (raw buffers; changed 2026-11-24) are
//!   resource-heap views the host makes: a raw SRV (UAV) of the whole host
//!   copy holding the root address, at the heap index the host writes into
//!   the [`HostShader::user_data`] entry marked [`ROOT_SRV_VIEW`]
//!   ([`ROOT_UAV_VIEW`]) (dword `index` = the root descriptor's own first
//!   user-data dword, which such a program never reads as data), and the
//!   root address's byte offset in that copy into the entry marked
//!   [`ROOT_VIEW_OFFSET`] (its second); the shader adds that offset to
//!   every offset it reads at. Host root descriptors have no bounds, so a
//!   shader reading past the copy read other memory or faulted; through
//!   the view it reads 0 and its writes there are dropped. One view per
//!   host copy (and kind) serves every draw of a list. In programs that
//!   use title memory (below) root SRVs and UAVs are title memory instead;
//!   elsewhere [`HostShader::root_srvs`] and [`HostShader::root_uavs`] now
//!   hold only the host's own tables (`VIEW_STRIDES`).
//! * **Vertex fetch** (added 2026-11-23). Grand Theft Auto VI's mesh and
//!   vertex programs fetch vertices themselves with typed-format loads
//!   (`tbuffer_load_format_*`): the format (8_8_8_8 unorm, 16_16 snorm,
//!   2_10_10_10 unorm, 8_8_8_8 and 16_16_16_16 uint...) is in the
//!   instruction, so the translation reads raw bytes and converts them
//!   itself. Their descriptors are built around a root SRV (stride 1, no
//!   record limit): the shader reads it as the root SRV view above. The
//!   vertex index, the stride and the attribute offset are the shader's
//!   own arithmetic on data in that buffer (per-stream headers), not
//!   descriptor fields.
//! * **Buffer views in the heap** are declared by how the shader reads
//!   them: raw (ByteAddressBuffer) for dword loads and stores, atomics and
//!   typed-format loads without an index; structured with the used-resource
//!   entry's stride when indexed; typed with four 32-bit float channels for
//!   `buffer_load_format_*` and `buffer_store_format_*`. The host's view at
//!   that index must be of that kind and start at the console descriptor's
//!   base address (raw: `R32_TYPELESS` with the raw flag), whatever kind the
//!   title's view description says. Typed-format loads from heap
//!   descriptors (1,024 world compute programs, ray-tracing material
//!   resolves) read raw views this way. Buffers read by element through
//!   the resource heap (no used-resource entry, so no stride) are raw too,
//!   addressed at element × the view's stride + offset, the stride from
//!   the view-stride table: root SRV `t`[`VIEW_STRIDES`] in
//!   [`ROOT_CBV_SPACE`], listed in [`HostShader::root_srvs`], whose dword
//!   `i` the host keeps at the byte stride of the buffer view at heap
//!   index `i`.
//! * **Constant buffers in the heap** (added 2026-11-24): scalar buffer
//!   loads through a descriptor taken from the resource heap itself (root
//!   type 5, so no used-resource entry) read a raw buffer view, not a
//!   constant buffer view: the title's constant buffers are 16-byte
//!   aligned, the host's views 256. The host makes a raw view at the
//!   constant buffer's address for each constant buffer view in the heap.
//!   Constant buffers in descriptor tables are read the same way (scalar
//!   loads and per-lane rows), so tables are never copied per draw.
//! * **Rectangle lists** ([`rectangle_gs`]): a geometry shader made from a
//!   vertex shader's outputs completes each rectangle with its fourth
//!   corner `v0 + v2 - v1`, as the console's hardware does.
//! * **Heap indices** (added 2026-11-24) are clamped to [`view_heap_limit`]
//!   − 1 (samplers [`HEAP_SAMPLERS`] − 1): the host's heaps have at least
//!   that many slots, those past its own null, so an index past the heap
//!   (data still wrong; the console would read whatever memory follows its
//!   heap) reads nothing instead of faulting the device. The limit is
//!   [`HEAP_VIEWS`], or the cap of a host whose view heaps are capped lower
//!   (NVIDIA: 1,000,000), set before translating ([`set_view_heap_limit`]).
//! * **Buffer views at computed heap indices** (added 2026-11-24): a
//!   buffer read through a descriptor the shader takes from the resource
//!   heap itself first looks up the view-stride table (root SRV
//!   `t`[`VIEW_STRIDES`], which then covers every index below
//!   the view limit, dynamic slots too); where it holds 0 (no buffer view
//!   there) the shader reads heap index [`null_buffer_view`] instead (for
//!   unordered access the index below), which the host keeps a null raw
//!   buffer SRV (UAV). The index comes from data, and a texture descriptor
//!   read as a buffer one is a wild address (the texture's shifted 8 bits
//!   down: GPU faults at low addresses).
//! * **Heap access** ([`HostShader::heap`], [`HostShader::heap_writes`],
//!   added 2026-11-24): whether the shader takes descriptors from the
//!   resource heap itself, and unordered access views among them; the
//!   host then keeps everything the heap's views reach current, and
//!   copies back what such views may have written.
//!
//! * **Texture atomics** (added 2026-11-23): a texture UAV the shader uses
//!   atomically is declared a 32-bit unsigned single-channel view; the
//!   host's view at that index must be an R32 integer view. Global loads
//!   from a root SRV read it raw. 64-bit texture atomics (two dwords per
//!   value; added 2026-11-24) are SM 6.6's on typed heap resources: the
//!   host's view must be `R32G32_UINT` (low dword in R) and the device must
//!   report 64-bit atomics on typed resources (options 9) and on descriptor
//!   heap resources (options 11), else the pipeline is refused.
//!
//! * **Root parameters past the user SGPRs** (added 2026-11-24): a
//!   compute program whose root signature does not fit its 16 user SGPRs
//!   loads the rest from memory whose 40-bit address is in user data 0-1;
//!   the table starts with user-data dword 16. They are user data as any
//!   other: the host fills dword `index` as above.
//! * What shaders write through a root UAV is the title's to see.
//! * **Title memory** (added 2026-11-24, decision 0005's unified-memory
//!   item): accesses at addresses the program computes (pointers in root
//!   constants or loaded from memory, BVH nodes) go through a page table,
//!   the root SRV `t0` in [`MEMORY_SPACE`] ([`HostShader::memory`]), dwords:
//!   4096 level-1 entries for address bits 28-39 (a level-2 block number,
//!   0 for none), then blocks of 4096 entries for bits 16-27 (64 KiB pages),
//!   two dwords each: the resource-heap index of a raw SRV of the host
//!   memory holding the page, whose raw UAV is the next index, and the low
//!   dword of the title address that memory starts at (byte offset =
//!   address low dword - it). Block 0 names null views (its entry 0 is
//!   how the shader tells a miss); addresses at or above 2^40 use it. The
//!   root UAV `u0` in [`MEMORY_SPACE`] ([`HostShader::memory_writes`]) takes
//!   bit `block × 4096 + page` for each page a shader writes (pages of an
//!   access's first and last byte: what the host copies back) and for each
//!   page with no copy an access reached, or for an address whose 256 MiB
//!   region has no block, bit `level-1 index` (block 0's bits). Reads there
//!   give 0, writes are dropped. In such a program (and one with root
//!   parameters past its user SGPRs) root SRVs and UAVs are title memory
//!   too: no root views; the shader reads their addresses as user data
//!   (root type [`RootType::Descriptor`] in [`HostShader::user_data`],
//!   dword 0 the address's low dword, 1 the high), and the host must have
//!   a copy holding each such address.
//! * **View addresses** ([`HostShader::view_addresses`]): a buffer
//!   descriptor's dwords 0-1 read as numbers (the BVH SRV's base address)
//!   come from the root SRV `t1` in [`MEMORY_SPACE`]: per resource-heap
//!   index, two dwords hold the low and high parts of a buffer view's
//!   48-bit title address. A view with only a native descriptor base word
//!   stores that word first and [`VIEW_BASE_ONLY`] second. A buffer-pointer
//!   read of that entry returns zero. The empty entry is two zero dwords.
//! * **Write marks** ([`HostShader::view_marks`], with
//!   [`Options::view_marks`]; added 2026-11-25): every store or atomic
//!   through an unordered buffer view from the heap (tables, the heap
//!   itself, root UAVs as their heap views) also sets bits in the root UAV
//!   `u`[`VIEW_WRITES`] in [`MEMORY_SPACE`], a bitmap of
//!   [`VIEW_WRITE_BITS`] bits, so the host copies back only what shaders
//!   wrote. The root SRV `t`[`VIEW_MARKS`] there holds two dwords per heap
//!   index: a base bit and a phase (0-65535); a write of byte `x` of the
//!   view (its offset in the view, an element times the stride plus the
//!   offset) sets bit `base + ((phase + x) >> 16)`, for the access's first
//!   and last byte, at most [`VIEW_WRITE_BITS`] − 1. Base
//!   [`VIEW_MARK_NONE`]: nothing is marked. The host gives each copy of
//!   title memory a run of bits, one per 64 KiB title page it touches, and
//!   a view the bit of its first byte's page as base with that byte's
//!   offset in its page as phase. Writes at computed addresses mark the
//!   page-table bitmap instead ("Title memory").
//!
//! Entry points: [`translate`] (container bytes as the title passes them
//! at pipeline creation → [`HostShader`]), [`emit`] (the same with the
//! earlier stages already run). The root constants of each stage must
//! cover [`HostShader::user_data_dwords`]. Stability: these types and the
//! binding scheme are what pipeline creation builds on; changes are
//! announced.
//! * **Signatures** as the console container has them (same semantics,
//!   registers and masks), so the input layout and stage linking are the
//!   title's; pixel outputs are plain 32-bit values.
//! * **Mesh shaders** ([`HwStage::Ms`], `ms_6_6`): user data and
//!   descriptors as above (the root constants visible to the mesh stage);
//!   the pipeline is a mesh pipeline (pipeline-state stream, no input
//!   layout) with the title's pixel shader. The thread group is the
//!   console's in lanes (`[numthreads]` in the DXIL), the vertex and
//!   primitive maxima its own, triangles only, no wave size required. The
//!   console numbers groups in x only: the draw path passes the title's
//!   count as `DispatchMesh(x, 1, 1)`. Per-primitive outputs (`PSG1`) are
//!   the console's, without `SV_PrimitiveID` and `SV_CullPrimitive` (the
//!   console never exports them).

mod bitcode;
#[cfg(all(test, windows))]
mod bvh_tests;
mod container;
#[cfg(all(test, windows))]
mod diff_tests;
#[cfg(all(test, windows))]
mod flow_tests;
#[cfg(all(windows, any(test, feature = "host")))]
pub mod host;
mod lower;
mod rect;
pub use rect::rectangle_gs;
#[cfg(all(test, windows))]
mod memory_tests;
mod sampler;
#[cfg(all(test, windows))]
mod semantics_tests;
mod shader;
#[cfg(all(test, windows))]
mod stride_tests;
#[cfg(all(test, windows))]
mod synthetic_tests;
#[cfg(windows)]
pub mod validator;
#[cfg(all(test, windows))]
mod valu_tests;
#[cfg(all(test, windows))]
mod waterfall_tests;
mod wave;

mod page_tables;
pub use page_tables::{CONSTANT_TABLE_MAX_BYTES, BufferOperandEvidence, StaticTextureSlot, StaticViewAccess, StaticViewSlot, StaticViewFailure, TableSlotFailure, StaticViewReport, direct_dispatch_view_report, static_view_report, static_texture_slots, static_view_slots};

use crate::dxbc::{Container, FourCC};
use crate::native::{HwStage, NativeProgram, RootType};
use crate::{ShaderError, io, ir, resource, structure};

pub use sampler::StaticSampler;
pub use shader::{
    HEAP_SAMPLERS, HEAP_VIEWS, MEMORY_SPACE, ROOT_CBV_SPACE, SAMPLER_SPACE, null_buffer_view,
    set_view_heap_limit, view_heap_limit,
    USER_DATA_SPACE, VIEW_MARK_NONE, VIEW_MARKS, VIEW_STRIDES, VIEW_WRITE_BITS, VIEW_WRITES,
};

/// Byte stride of the host's per-slot view-address record at t1 in memory space.
pub const VIEW_ADDRESS_BYTES: u32 = 8;
/// Before the per-view records: the title slot count, then a zero record
/// used by a translated shader for a slot outside that range.
pub const VIEW_ADDRESS_PREFIX_RECORDS: u32 = 2;
/// Word 1 of a view-address entry carrying only native descriptor word 0.
/// A buffer address uses at most the low 16 bits of word 1.
pub const VIEW_BASE_ONLY: u32 = 1 << 16;

/// The loop limit without an override: 2^20.
pub const DEFAULT_LOOP_LIMIT: u32 = lower::LOOP_LIMIT;

/// The iteration count after which a translated loop is left, per lane and
/// per entry into the loop: [`DEFAULT_LOOP_LIMIT`].
///
/// With the `diagnostics` feature, `VICTORY_SHADER_LOOP_LIMIT` (decimal or
/// `0x` hex, at least 1) sets it instead. A loop that hits the limit in
/// every lane of a full-screen pass still runs for seconds (2^20 times a
/// 4K screen is about 10^13 iterations); a much lower limit shows whether
/// such loops are what makes a frame slow. The value is part of the
/// translation, so the shader cache must key on it (`victory-gpu`'s does).
pub fn loop_limit() -> u32 {
    static LIMIT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        if cfg!(feature = "diagnostics") {
            parse_loop_limit(std::env::var("VICTORY_SHADER_LOOP_LIMIT").ok().as_deref())
        } else {
            lower::LOOP_LIMIT
        }
    })
}

/// `VICTORY_SHADER_LOOP_LIMIT`'s value; the default when it is unset,
/// unreadable or 0. Pure, for testing.
fn parse_loop_limit(value: Option<&str>) -> u32 {
    let parsed = value
        .map(str::trim)
        .and_then(|v| match v.strip_prefix("0x") {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => v.parse().ok(),
        });
    match parsed {
        Some(n) if n > 0 => n,
        _ => lower::LOOP_LIMIT,
    }
}

/// A translated shader and what the host binds for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostShader {
    pub stage: HwStage,
    /// The DXIL container, hashed as D3D12 expects.
    pub dxil: Vec<u8>,
    /// The user-data dwords the shader reads.
    pub user_data: Vec<UserData>,
    /// Immediate samplers: the four sampler words of `s<n>`.
    pub samplers: Vec<[u32; 4]>,
    /// The same samplers decoded, `s<n>` at position n: the static sampler
    /// descriptions the host's root signature declares.
    pub static_samplers: Vec<StaticSampler>,
    /// Root CBVs the shader reads: root parameter indices, each bound as
    /// `b<index>` in [`ROOT_CBV_SPACE`].
    pub root_cbvs: Vec<u8>,
    /// Root SRVs (raw buffers) the host binds as root descriptors: `t<index>`
    /// in [`ROOT_CBV_SPACE`] (the host's own tables, and title root SRVs in
    /// programs using title memory; others are views, module notes).
    pub root_srvs: Vec<u8>,
    /// Root UAVs (raw buffers) the host binds as root descriptors:
    /// `u<index>` in [`ROOT_CBV_SPACE`] (likewise).
    pub root_uavs: Vec<u8>,
    /// Whether the shader reads title memory at addresses it computes: the
    /// host binds the page table `t0` in [`MEMORY_SPACE`] (module notes,
    /// "Title memory").
    pub memory: bool,
    /// Whether the host binds the page bitmap `u0` in [`MEMORY_SPACE`]
    /// (pages written and pages missed): whenever [`HostShader::memory`].
    pub memory_writes: bool,
    /// Whether it reads buffer views' title addresses from their
    /// descriptors: the host binds the view addresses `t1` in
    /// [`MEMORY_SPACE`] (module notes, "Title memory").
    pub view_addresses: bool,
    /// Whether it takes descriptors from the resource heap at indices it
    /// computes (root type 5, not through a table): the host cannot tell
    /// which views it reads, so every view in the heap must be current.
    pub heap: bool,
    /// Whether some of those are unordered access views: it may write
    /// through any of the heap's.
    pub heap_writes: bool,
    /// Whether it marks the pages it writes through buffer views: the host
    /// binds the view-mark table `t`[`VIEW_MARKS`] and the bitmap
    /// `u`[`VIEW_WRITES`] in [`MEMORY_SPACE`] (module notes, "Write
    /// marks").
    pub view_marks: bool,
}

impl HostShader {
    /// Size of the user-data buffer the shader declares, in dwords (a
    /// multiple of 4; 0 when it reads none). Root constants must cover it.
    pub fn user_data_dwords(&self) -> u32 {
        let n = self
            .user_data
            .iter()
            .map(|u| u.index + 1)
            .max()
            .unwrap_or(0);
        n.next_multiple_of(4)
    }
}

/// [`UserData::dword`] of a root descriptor's entry that takes the heap
/// index of the host's raw SRV of it (module notes, "Root SRVs and root
/// UAVs"); dwords 0 and 1 are its address (programs using title memory).
pub const ROOT_SRV_VIEW: u32 = 2;
/// As [`ROOT_SRV_VIEW`], for the raw UAV of a root UAV.
pub const ROOT_UAV_VIEW: u32 = 3;
/// [`UserData::dword`] of a root descriptor's entry that takes its
/// address's byte offset in the view [`ROOT_SRV_VIEW`] or [`ROOT_UAV_VIEW`]
/// names.
pub const ROOT_VIEW_OFFSET: u32 = 4;

/// One dword of user data and the root parameter that fills it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserData {
    /// Dword in the user-data buffer (the console's user-data index).
    pub index: u32,
    /// Root parameter index in the title's root signature.
    pub slot: u8,
    /// `Constants`, `Table` or `Descriptor` (see the module notes for the
    /// value).
    pub root: RootType,
    /// Dword within the parameter; for a root descriptor
    /// [`ROOT_SRV_VIEW`], [`ROOT_UAV_VIEW`] or [`ROOT_VIEW_OFFSET`] name
    /// its view's heap index or its offset in that view instead.
    pub dword: u32,
}

/// How a translation is made.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Options {
    /// Writes through buffer views mark the pages they write (module notes,
    /// "Write marks"); the host must then bind the view-mark table and the
    /// bitmap ([`HostShader::view_marks`]).
    pub view_marks: bool,
    /// Bind raw title-memory loads to each page's UAV descriptor. A paged
    /// host must keep the corresponding slabs in UAV state during dispatch.
    /// Does not change dirty/miss tracking or ordinary descriptor loads.
    pub raw_page_uav_reads: bool,
}

/// Translates the vertex, pixel, compute or mesh program of a console
/// shader container, without write marks.
pub fn translate(container: &[u8]) -> Result<HostShader, ShaderError> {
    translate_with(container, Options::default())
}

/// [`translate`] made as `options` say.
pub fn translate_with(container: &[u8], options: Options) -> Result<HostShader, ShaderError> {
    let c = Container::parse(container)?;
    let dxil = c
        .part(FourCC::DXIL)
        .or(c.part(FourCC::ACCC))
        .ok_or(ShaderError::Malformed("no DXIL part"))?;
    let native = crate::native_programs(dxil.data)?
        .into_iter()
        .find(|p| {
            matches!(
                p.stage,
                HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms
            )
        })
        .ok_or(ShaderError::Unsupported {
            offset: 0,
            what: "a container without a vertex, pixel, compute or mesh program",
        })?;
    // Structurisation first: tracing per-lane descriptors needs it (its
    // failure is reported after the earlier stages').
    let (p, st) = crate::range::build(&native.patched_code(), native.wave64, native.vgprs)?;
    let res = resource::track(&p, &native.layout, native.stage, st.as_ref().ok())?;
    let io = io::raise(&p, &native)?;
    let st = st?;
    emit_with(&c, &native, &p, &st, &io, &res, options)
}

/// Emits a program whose earlier stages already ran, without write marks.
pub fn emit(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
) -> Result<HostShader, ShaderError> {
    emit_with(c, native, p, st, io, res, Options::default())
}

/// [`emit`] made as `options` say.
pub fn emit_with(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
    options: Options,
) -> Result<HostShader, ShaderError> {
    let sig = |fourcc| -> Result<Vec<container::SigElement>, ShaderError> {
        match c.part(fourcc) {
            Some(part) => signature(part.data),
            None => Ok(Vec::new()),
        }
    };
    let pixel = native.stage == HwStage::Ps;
    let mut inputs = sig(FourCC::ISG1)?;
    let mut outputs = sig(FourCC::OSG1)?;
    // Interpolation as DXC declares it by default; pixel inputs that are
    // read get the mode their barycentrics say.
    let interpolated = if pixel { &mut inputs } else { &mut outputs };
    for e in interpolated {
        e.interp = match (e.kind, e.comp) {
            (3, _) => 4,
            // Constant, as DXC declares it (the AMD compiler crashed on a
            // front face without a mode).
            (13, _) => 1,
            (0 | 6 | 7, 9) => 2,
            (0, _) => 1,
            _ => 0,
        };
    }
    // With a conservative depth output (SV_DepthLessEqual, GreaterEqual) DXIL
    // wants the position centroid (or per sample), as DXC declares it then.
    if pixel && outputs.iter().any(|e| matches!(e.kind, 18 | 19)) {
        for e in inputs.iter_mut().filter(|e| e.kind == 3) {
            e.interp = 5;
        }
    }
    // Per-primitive outputs are constant. The console never exports a
    // primitive id or cull flag a mesh shader writes (it culls with the
    // primitive export), so those rows are left out.
    let mut prims = sig(FourCC::PSG1)?;
    prims.retain(|e| !matches!(e.kind, 10 | 30));
    for e in &mut prims {
        e.interp = 1;
    }
    // A mesh group's id arrives multiplied by max(threads, vertices,
    // primitives) (the code divides by that); the thread count is in PSV0
    // from runtime info version 2 on.
    // shortcut: older containers fall back to the group's lane count, right
    // when the HLSL group fills its waves (Grand Theft Auto VI's do).
    let mesh_stride = match &native.io {
        crate::native::StageIo::Ms(ms) => {
            let psv = c.part(FourCC::PSV0).map(|p| p.data);
            let threads = psv
                .filter(|d| crate::dxbc::u32_at(d, 0).is_ok_and(|n| n >= 48))
                .and_then(|d| crate::dxbc::u32_at(d, 4 + 36).ok());
            threads.map_or(ms.threads, |t| {
                t.max(ms.max_vertices).max(ms.max_primitives)
            })
        }
        _ => 0,
    };
    let parts = |st, res| lower::Parts {
        native,
        p,
        st,
        io,
        res,
        inputs: inputs.clone(),
        outputs: outputs.clone(),
        prims: prims.clone(),
        mesh_stride,
        view_marks: options.view_marks,
    };
    let l = match lower::lower_page_access(&parts(st, res), options.raw_page_uav_reads) {
        // Discarded lanes kept on as helpers can read what no surviving
        // lane does (a register only killed lanes skipped writing); such a
        // program kills them as the console does (`structure::killing`).
        Err(_) if !st.discards.is_empty() => {
            let st = structure::killing(p)?;
            let res = resource::track(p, &native.layout, native.stage, Some(&st))?;
            lower::lower_page_access(&parts(&st, &res), options.raw_page_uav_reads)?
        }
        l => l?,
    };
    Ok(HostShader {
        stage: native.stage,
        dxil: l.dxil,
        user_data: l.user_data,
        static_samplers: l
            .samplers
            .iter()
            .enumerate()
            .map(|(n, &w)| {
                let mut s = StaticSampler::from_words(w);
                if l.cmp_samplers >> n & 1 != 0 {
                    s.filter |= 0x80;
                }
                s
            })
            .collect(),
        samplers: l.samplers,
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        memory: l.memory,
        memory_writes: l.memory_writes,
        view_addresses: l.view_addresses,
        heap: l.heap,
        heap_writes: l.heap_writes,
        view_marks: l.view_marks,
    })
}

/// The rows of an `ISG1`/`OSG1` part.
fn signature(part: &[u8]) -> Result<Vec<container::SigElement>, ShaderError> {
    use crate::dxbc::u32_at;
    let count = u32_at(part, 0)? as usize;
    let base = u32_at(part, 4)? as usize;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let at = base + 32 * i;
        let name_at = u32_at(part, at + 4)? as usize;
        let name = part
            .get(name_at..)
            .and_then(|s| s.split(|&b| b == 0).next())
            .ok_or(ShaderError::Malformed("signature name out of bounds"))?;
        let mask = *part
            .get(at + 24)
            .ok_or(ShaderError::Malformed("truncated"))?;
        if u32_at(part, at + 28)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a minimum-precision signature element",
            });
        }
        if u32_at(part, at)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a signature element of another stream",
            });
        }
        let kind = match u32_at(part, at + 12)? {
            0 => 0,
            1 => 3,
            2 => 6,
            3 => 7,
            4 => 4,
            5 => 5,
            6 => 1,
            7 => 10,
            8 => 2,
            9 => 13,
            10 => 12,
            64 => 16,
            65 => 17,
            66 => 14,
            67 => 19,
            68 => 18,
            69 => 20,
            70 => 15,
            25 => 30, // cull primitive
            _ => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: "a system value in the signature",
                });
            }
        };
        // 16-bit elements are declared 32-bit; reading or writing one
        // converts (16-bit values travel packed on the console).
        let raw = u32_at(part, at + 16)?;
        let comp = match (kind, raw) {
            // DXIL declares the front face a uint (the validator insists).
            (13, _) => 5,
            (_, 1 | 4) => 5,
            (_, 2 | 5) => 4,
            (_, 3 | 6) => 9,
            (_, c) => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: match c {
                        0 => "a signature element of unknown type",
                        _ => "a 64-bit signature element",
                    },
                });
            }
        };
        let register = u32_at(part, at + 20)?;
        out.push(container::SigElement {
            name: String::from_utf8_lossy(name).into_owned(),
            index: u32_at(part, at + 8)?,
            kind,
            comp,
            interp: 0,
            row: (register != u32::MAX).then_some(register),
            col: mask.trailing_zeros().min(3) as u8,
            cols: mask.count_ones().max(1) as u8,
            usage: 0,
            half: (4..=6).contains(&raw),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::bitcode::Attr;
    use super::lower::LOOP_LIMIT;
    use super::parse_loop_limit;

    #[test]
    fn the_loop_limit_switch_parses_or_falls_back() {
        assert_eq!(parse_loop_limit(None), LOOP_LIMIT);
        assert_eq!(parse_loop_limit(Some("4096")), 4096);
        assert_eq!(parse_loop_limit(Some(" 0x1000 ")), 4096);
        assert_eq!(parse_loop_limit(Some("1")), 1);
        for bad in ["", "0", "-5", "lots", "0x", "4294967296"] {
            assert_eq!(parse_loop_limit(Some(bad)), LOOP_LIMIT, "{bad:?}");
        }
    }
    use super::container::SigElement;
    use super::shader::Shader;
    use super::*;
    use crate::native::{ComputeIo, StageIo};

    /// A pixel shader writing SV_StencilRef (channel y of `exp mrtz`).
    /// DXC sets ShaderFlags.StencilRef (bit 11) for a pixel shader whose
    /// output signature has SV_StencilRef, and the feature bit
    /// D3D_SHADER_FEATURE_STENCIL_REF (0x200) in SFI0; the validator checks
    /// both against the module.
    #[test]
    fn a_stencil_export_declares_the_stencil_ref_feature() {
        let words = [
            0x7E00_02F2u32, // v_mov_b32 v0, 1.0
            0x7E02_0285,    // v_mov_b32 v1, 5
            0xF800_1883,
            0x0000_0100, // exp mrtz v0, v1, off, off done vm
            0xBF81_0000,
        ];
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let el = |name: &str, kind, comp| SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp,
            interp: 0,
            row: None,
            col: 0,
            cols: 1,
            usage: 1,
            half: false,
        };
        let osg1 =
            container::signature(&[el("SV_Depth", 17, 9), el("SV_StencilRef", 20, 5)], false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let out = emit(&c, &native, &p, &st, &io, &res).unwrap();
        // dxil.dll, where there is one, checks the flags against the module.
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut out.dxil.clone()).unwrap();
        }
        let host = Container::parse(&out.dxil).unwrap();
        let sfi0 = host.part(FourCC::SFI0).unwrap().data;
        let features = u64::from_le_bytes(sfi0.try_into().unwrap());
        assert_ne!(features & 0x200, 0, "SFI0 {features:#x}: no StencilRef");
    }

    /// Reads the `ShaderFlags` word (entry property tag 0) from `dx.entryPoints`
    /// in a DXIL container emitted by [`bitcode::Module`].
    fn module_shader_flags(dxil_container: &[u8]) -> u64 {
        let host = Container::parse(dxil_container).unwrap();
        let bitcode = &host.part(FourCC::DXIL).unwrap().data[24..];
        assert_eq!(&bitcode[..4], b"BC\xc0\xde");
        let words: Vec<u32> = bitcode[4..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| u32::from_le_bytes(c))
            .collect();
        struct Reader<'a> {
            words: &'a [u32],
            pos: usize,
        }
        impl Reader<'_> {
            fn read(&mut self, width: u32) -> u64 {
                let mut out = 0u64;
                for i in 0..width {
                    let word = self.words[self.pos / 32];
                    let bit = (word >> (self.pos % 32)) & 1;
                    out |= u64::from(bit) << i;
                    self.pos += 1;
                }
                out
            }
            fn vbr(&mut self, width: u32) -> u64 {
                let mut out = 0u64;
                let mut shift = 0u32;
                let mask = (1u64 << (width - 1)) - 1;
                loop {
                    let chunk = self.read(width);
                    out |= (chunk & mask) << shift;
                    if chunk >> (width - 1) == 0 {
                        return out;
                    }
                    shift += width - 1;
                }
            }
            fn align32(&mut self) {
                self.pos = (self.pos + 31) & !31;
            }
        }
        enum Md {
            Val(u64),
            Node(Vec<u64>),
            Other,
        }
        let mut r = Reader {
            words: &words,
            pos: 0,
        };
        let mut stack: Vec<(u64, u32)> = Vec::new();
        let mut num_values = 0u64;
        let mut consts = std::collections::HashMap::<u64, u64>::new();
        let mut mds: Vec<Md> = Vec::new();
        let mut last_name: Vec<u8> = Vec::new();
        let mut entry_node = None;
        let total_bits = words.len() * 32;
        while r.pos < total_bits {
            let width = stack.last().map_or(2, |b| b.1);
            let id = r.read(width);
            match id {
                0 => {
                    r.align32();
                    stack.pop();
                    if stack.is_empty() {
                        break;
                    }
                }
                1 => {
                    let block = r.vbr(8);
                    let new_width = r.vbr(4) as u32;
                    r.align32();
                    let len_words = r.read(32) as usize;
                    if matches!(block, 8 | 11 | 15) {
                        stack.push((block, new_width));
                    } else {
                        r.pos += len_words * 32;
                    }
                }
                3 => {
                    let code = r.vbr(6);
                    let nops = r.vbr(6) as usize;
                    let ops: Vec<u64> = (0..nops).map(|_| r.vbr(6)).collect();
                    match stack.last().map(|b| b.0) {
                        Some(8) if matches!(code, 7 | 8) => num_values += 1,
                        Some(11) if code != 1 => {
                            if code == 4 {
                                let v = ops[0];
                                let decoded = if v & 1 != 0 {
                                    (v >> 1).wrapping_neg()
                                } else {
                                    v >> 1
                                };
                                consts.insert(num_values, decoded);
                            }
                            num_values += 1;
                        }
                        Some(15) => match code {
                            1 => mds.push(Md::Other),
                            2 => mds.push(Md::Val(ops[1])),
                            3 => mds.push(Md::Node(ops)),
                            4 => last_name = ops.iter().map(|&c| c as u8).collect(),
                            10 if last_name == b"dx.entryPoints" => entry_node = Some(ops[0]),
                            _ => {}
                        },
                        _ => {}
                    }
                }
                other => panic!("unexpected bitstream entry id {other}"),
            }
        }
        let Md::Node(entry) = &mds[entry_node.expect("dx.entryPoints") as usize] else {
            panic!("dx.entryPoints[0] is not a node");
        };
        let props_id = entry[4];
        if props_id == 0 {
            return 0;
        }
        let Md::Node(props) = &mds[(props_id - 1) as usize] else {
            panic!("entry properties is not a node");
        };
        for &[tag_md, val_md] in props.as_chunks::<2>().0 {
            let Md::Val(tag_id) = mds[(tag_md - 1) as usize] else {
                continue;
            };
            if consts.get(&tag_id) == Some(&0) {
                let Md::Val(flags_id) = mds[(val_md - 1) as usize] else {
                    panic!("shader flags property is not a value");
                };
                return consts[&flags_id];
            }
        }
        0
    }

    /// A vertex (or mesh) program that declares a UAV — such as the
    /// written-pages bitmap `cd94d35` emits when `Options::view_marks` is set
    /// for a store through an unordered view — must declare
    /// `UAVsAtEveryStage` (ShaderFlags bit 16) and the matching SFI0 feature
    /// bit `0x4`. Compute and pixel programs must not set it.
    #[test]
    fn vertex_uav_writes_declare_uavs_at_every_stage() {
        let store_uav = |stage: HwStage, tail: &[u32]| {
            let base = stage.user_sgpr(0);
            let mut w = vec![
                0xBE80_0380u32 | (u32::from(base + 1) << 16), // s_mov_b32 s[base+1], 0
                0xF408_0300 | (u32::from(base) >> 1),
                0xFA00_0000, // s_load_dwordx4 s[12:15], s[base:base+1], null
                0xBF8C_0000, // s_waitcnt all
                0x7E00_02F2, // v_mov_b32 v0, 1.0
                0xE070_0000,
                0x8003_0000, // buffer_store_dword v0, off, s[12:15], 0
            ];
            w.extend_from_slice(tail);
            w.push(0xBF81_0000); // s_endpgm
            w
        };
        let translate_stage = |stage: HwStage, options: Options| {
            let (words, io, osg1) = match stage {
                HwStage::Vs => {
                    let pos = SigElement {
                        name: "SV_Position".into(),
                        index: 0,
                        kind: 3,
                        comp: 9,
                        interp: 0,
                        row: Some(0),
                        col: 0,
                        cols: 4,
                        usage: 0xF,
                        half: false,
                    };
                    (
                        store_uav(stage, &[0xF800_08CF, 0x0000_0000]), // exp pos0 v0..v0 done
                        StageIo::Vs {
                            inputs: vec![],
                            outputs: vec![],
                        },
                        Some(container::signature(&[pos], false)),
                    )
                }
                HwStage::Ps => {
                    let target = SigElement {
                        name: "SV_Target".into(),
                        index: 0,
                        kind: 16,
                        comp: 9,
                        interp: 0,
                        row: Some(0),
                        col: 0,
                        cols: 4,
                        usage: 0xF,
                        half: false,
                    };
                    (
                        store_uav(stage, &[0xF800_180F, 0x0000_0000]), // exp mrt0 v0..v0 done vm
                        StageIo::Ps {
                            inputs: vec![],
                            input_addr: 0,
                            db_shader_control: 0,
                            patches: vec![],
                        },
                        Some(container::signature(&[target], false)),
                    )
                }
                HwStage::Cs => (
                    store_uav(stage, &[]),
                    StageIo::Cs(ComputeIo {
                        rsrc2: 1 << 1,
                        threads: [1, 1, 1],
                    }),
                    None,
                ),
                HwStage::Ms => {
                    let pos = SigElement {
                        name: "SV_Position".into(),
                        index: 0,
                        kind: 3,
                        comp: 9,
                        interp: 0,
                        row: Some(0),
                        col: 0,
                        cols: 4,
                        usage: 0xF,
                        half: false,
                    };
                    (
                        store_uav(
                            stage,
                            &[
                                0x7E02_02FF,
                                TRIANGLE, // v_mov_b32 v1, TRIANGLE
                                0xF800_0941,
                                0x0000_0001, // exp prim v1, off, off, off done
                                0xF800_08CF,
                                0x0000_0000, // exp pos0 v0..v0 done
                            ],
                        ),
                        StageIo::Ms(crate::native::MeshIo {
                            outputs: vec![],
                            clip_cntl: 0,
                            threads: 32,
                            max_vertices: 32,
                            max_primitives: 32,
                            topology: 2,
                            lds_bytes: 0,
                        }),
                        Some(container::signature(&[pos], false)),
                    )
                }
                _ => unreachable!(),
            };
            let native = NativeProgram {
                stage,
                wave64: false,
                vgprs: 4,
                layout: vec![crate::native::RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                }],
                used: vec![crate::native::UsedResource {
                    kind: crate::native::ResourceType::Uav,
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    lower_bound: 0,
                    stride: 0,
                    dwords: 4,
                    user_data: 0,
                    slot: 0,
                }],
                io,
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let parts: Vec<_> = osg1
                .as_deref()
                .map(|o| (FourCC::OSG1, o))
                .into_iter()
                .collect();
            let bytes = crate::dxbc::build(&parts);
            let c = Container::parse(&bytes).unwrap();
            let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
            let st = structure::structure(&p).unwrap();
            let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = io::raise(&p, &native).unwrap();
            let out = emit_with(&c, &native, &p, &st, &io, &res, options).unwrap();
            #[cfg(windows)]
            if let Ok(mut v) = super::validator::Validator::load(
                super::validator::Validator::default_path().as_deref(),
            ) {
                v.validate(&mut out.dxil.clone()).unwrap();
            }
            let host = Container::parse(&out.dxil).unwrap();
            let sfi0 = host.part(FourCC::SFI0).unwrap().data;
            let features = u64::from_le_bytes(sfi0.try_into().unwrap());
            let flags = module_shader_flags(&out.dxil);
            (out, features, flags)
        };

        let marks = Options { view_marks: true, ..Default::default() };
        for stage in [HwStage::Vs, HwStage::Ms] {
            let (out, sfi0, flags) = translate_stage(stage, marks);
            assert!(
                out.view_marks,
                "{stage:?} store must emit the view-mark UAV"
            );
            assert_ne!(
                sfi0 & 0x4,
                0,
                "{stage:?} SFI0 {sfi0:#x}: missing UAVsAtEveryStage"
            );
            assert_ne!(
                flags & shader::flags::UAVS_AT_EVERY_STAGE,
                0,
                "{stage:?} module flags {flags:#x}: missing UAVsAtEveryStage"
            );
        }

        let (vs_no_marks, vs_no_marks_sfi0, vs_no_marks_flags) =
            translate_stage(HwStage::Vs, Options::default());
        assert!(!vs_no_marks.view_marks);
        assert_eq!(
            vs_no_marks_sfi0 & 0x4,
            0,
            "VS without declared UAVs must not set UAVsAtEveryStage in SFI0"
        );
        assert_eq!(
            vs_no_marks_flags & shader::flags::UAVS_AT_EVERY_STAGE,
            0,
            "VS without declared UAVs must not set UAVsAtEveryStage in module flags"
        );

        for stage in [HwStage::Ps, HwStage::Cs] {
            let (out, sfi0, flags) = translate_stage(stage, marks);
            assert!(
                out.view_marks,
                "{stage:?} store must emit the view-mark UAV"
            );
            assert_eq!(
                sfi0 & 0x4,
                0,
                "{stage:?} SFI0 {sfi0:#x}: must not set UAVsAtEveryStage"
            );
            assert_eq!(
                flags & shader::flags::UAVS_AT_EVERY_STAGE,
                0,
                "{stage:?} module flags {flags:#x}: must not set UAVsAtEveryStage"
            );
        }
    }

    #[test]
    fn immediate_sampler_slots_distinguish_comparison_uses() {
        let words = [
            0xBE81_0380u32, // s_mov_b32 s1, 0
            0xF40C_0000,
            0xFA00_0000, // s_load_dwordx8 s[0:7], s[0:1], null
            0xBE88_0380, // s_mov_b32 s8, 0
            0xBE89_0380, // s_mov_b32 s9, 0
            0xBE8A_0380, // s_mov_b32 s10, 0
            0xBE8B_0380, // s_mov_b32 s11, 0
            0x7E00_02F0, // v_mov_b32 v0, 0.5 (comparison reference)
            0x7E02_02F0, // v_mov_b32 v1, 0.5 (u)
            0x7E04_02F0, // v_mov_b32 v2, 0.5 (v)
            0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
            0xF080_0108,
            0x0040_0301, // image_sample v3, v[1:2], s[0:7], s[8:11]
            0xF0A0_0108,
            0x0040_0400, // image_sample_c v4, v[0:2], s[0:7], s[8:11]
            0xF0BC_0108,
            0x0040_0500, // image_sample_c_lz v5, v[0:2], s[0:7], s[8:11]
            0xF080_0108,
            0x0040_0601, // image_sample v6, v[1:2], s[0:7], s[8:11]
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0xF800_180F,
            0x0605_0403, // exp mrt0, v3, v4, v5, v6 done vm
            0xBF81_0000,
        ];
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 7,
            layout: vec![crate::native::RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![crate::native::UsedResource {
                kind: crate::native::ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 8,
                user_data: 0,
                slot: 0,
            }],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let output = SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        };
        let osg1 = container::signature(&[output], false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let out = emit(&c, &native, &p, &st, &io, &res).unwrap();

        let sampler = StaticSampler::from_words([0; 4]);
        let comparison = StaticSampler {
            filter: sampler.filter | 0x80,
            ..sampler
        };
        assert_eq!(out.samplers, [[0; 4]; 2]);
        assert_eq!(out.static_samplers, [sampler, comparison]);
        #[cfg(windows)]
        if let Some(path) = super::validator::Validator::default_path() {
            let mut v = super::validator::Validator::load(Some(&path)).unwrap();
            v.validate(&mut out.dxil.clone()).unwrap();
        }
    }

    /// A wave64 compute program that writes only EXEC's high dword
    /// (`s_mov_b32 exec_hi, 0`) before a vector instruction. The new mask is
    /// opaque and keyed by the unchanged low dword, so building its lane bit
    /// needed the mask's own value: lowering recursed until the stack
    /// overflowed, which takes the whole process down (found by
    /// random-input testing). Now the instruction fails like any other that
    /// cannot be expressed: its result is missing, and only a side effect
    /// that uses it would stop the translation.
    #[test]
    fn an_exec_high_write_does_not_overflow_the_stack() {
        // s_mov_b32 exec_hi, 0; v_mov_b32 v1, v0; s_endpgm
        let words = [0xBEFF_0380u32, 0x7E02_0300, 0xBF81_0000];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [64, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let container = crate::dxbc::build(&[]);
        let c = Container::parse(&container).unwrap();
        if let Err(e) = emit(&c, &native, &p, &st, &io, &res) {
            assert!(e.to_string().contains("its own value"), "{e}");
        }
    }

    /// A pixel shader writing a constant colour.
    #[cfg(windows)]
    fn constant_color() -> Vec<u8> {
        let mut s = Shader::new(true);
        s.outputs.push(SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        });
        let b = s.entry;
        for (c, v) in [1.0f32, 0.5, 0.25, 1.0].into_iter().enumerate() {
            let args = [
                s.m.i32(0),
                s.m.i32(0),
                s.m.i8(c as u8),
                s.m.f32(v.to_bits()),
            ];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
        }
        s.m.terminate(b, super::bitcode::Term::Ret);
        s.finish()
    }

    #[test]
    #[cfg(windows)]
    fn stub_pipeline_is_created() {
        let host = match super::host::Host::new(false) {
            Ok(host) => host,
            Err(e) => {
                eprintln!("no SM 6.6 device ({e}); skipped");
                return;
            }
        };
        let ps = super::HostShader {
            stage: crate::HwStage::Ps,
            dxil: constant_color(),
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
        let file = |var: &str, stage| {
            std::env::var(var).ok().map(|p| super::HostShader {
                stage,
                dxil: std::fs::read(p).unwrap(),
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
            })
        };
        let ps = file("VICTORY_REF_PS", crate::HwStage::Ps).unwrap_or(ps);
        let vs = file("VICTORY_REF_VS", crate::HwStage::Vs)
            .unwrap_or_else(|| super::host::stub_vs(&ps).unwrap());
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut vs.dxil.clone()).unwrap();
        }
        host.pipeline(&vs, &ps).unwrap();
        // The runtime checks the container hash: no hash or a wrong one is
        // refused. (Recent runtimes also take sixteen 0x01 bytes, a bypass
        // value; ours works with older ones too.)
        let mut wrong = ps.dxil[4..20].to_vec();
        wrong[0] ^= 1;
        for digest in [&[0u8; 16][..], &wrong] {
            let mut other = ps.clone();
            other.dxil[4..20].copy_from_slice(digest);
            assert!(
                host.pipeline(&vs, &other).is_err(),
                "digest {digest:02x?} accepted"
            );
        }
    }

    #[test]
    #[cfg(windows)]
    fn constant_color_validates() {
        let mut c = constant_color();
        let path = super::validator::Validator::default_path();
        let Ok(mut v) = super::validator::Validator::load(path.as_deref()) else {
            eprintln!("no dxil.dll; skipped");
            return;
        };
        let ours = c[4..20].to_vec();
        v.validate(&mut c).unwrap();
        assert_eq!(c[4..20], ours[..], "our hash differs from the validator's");
    }

    /// Triangle (0, 1, 2), each corner's 9-bit index in its 10-bit field.
    const TRIANGLE: u32 = 1 << 10 | 2 << 20;
    /// The edge flags of the three corners (bits 9, 19 and 29).
    const EDGE_FLAGS: u32 = 1 << 9 | 1 << 19 | 1 << 29;
    /// The culling bit.
    const CULL: u32 = 1 << 31;

    /// A one-wave (32-lane) triangle mesh program whose lanes all export
    /// primitive `prim` and, as vertex `lane`, the position
    /// (lane & 1 ? 3 : -1, lane & 2 ? 3 : -1, 0.5, 1): triangle (0, 1, 2)
    /// covers the whole target.
    fn triangle_mesh(prim: u32) -> Result<HostShader, ShaderError> {
        let words = [
            0xD765_0001u32, // v_mbcnt_lo_u32_b32 v1, -1, 0
            0x0001_00C1,
            0x3604_0281, // v_and_b32 v2, 1, v1
            0x3606_0282, // v_and_b32 v3, 2, v1
            0x7E04_0D02, // v_cvt_f32_u32 v2, v2
            0x7E06_0D03, // v_cvt_f32_u32 v3, v3
            0x1008_04F6, // v_mul_f32 v4, 4.0, v2
            0x0608_08F3, // v_add_f32 v4, -1.0, v4
            0x100A_06F4, // v_mul_f32 v5, 2.0, v3
            0x060A_0AF3, // v_add_f32 v5, -1.0, v5
            0x7E0C_02F0, // v_mov_b32 v6, 0.5
            0x7E0E_02F2, // v_mov_b32 v7, 1.0
            0x7E12_02FF, // v_mov_b32 v9, prim
            prim,
            0xF800_0941, // exp prim v9, off, off, off done
            0x0000_0009,
            0xF800_08CF, // exp pos0 v4, v5, v6, v7 done
            0x0706_0504,
            0xBF81_0000, // s_endpgm
        ];
        let native = NativeProgram {
            stage: HwStage::Ms,
            wave64: false,
            vgprs: 10,
            layout: vec![],
            used: vec![],
            io: StageIo::Ms(crate::native::MeshIo {
                outputs: vec![],
                clip_cntl: 0,
                threads: 32,
                max_vertices: 32,
                max_primitives: 32,
                topology: 2,
                lds_bytes: 0,
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let position = SigElement {
            name: "SV_Position".into(),
            index: 0,
            kind: 3,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0xF,
            half: false,
        };
        let osg1 = container::signature(&[position], false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes)?;
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
        let st = structure::structure(&p)?;
        let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
        let io = io::raise(&p, &native)?;
        emit(&c, &native, &p, &st, &io, &res)
    }

    /// A primitive export's corners are 9-bit indices: the edge flag above
    /// each (bits 9, 19, 29) is no part of it. Masking ten bits made a
    /// flagged corner index 512 or more, past any mesh program's vertices
    /// (the draw test below shows the triangle). Flagged, unflagged and
    /// culled exports translate, and dxil.dll, where there is one, takes
    /// each module.
    #[test]
    fn primitive_export_indices_drop_the_edge_flags() {
        #[cfg(windows)]
        let mut validator = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        )
        .ok();
        for prim in [
            TRIANGLE,
            TRIANGLE | EDGE_FLAGS,
            TRIANGLE | EDGE_FLAGS | CULL,
        ] {
            let ms = triangle_mesh(prim).unwrap_or_else(|e| panic!("{prim:#x}: {e}"));
            #[cfg(windows)]
            if let Some(v) = validator.as_mut() {
                v.validate(&mut ms.dxil.clone())
                    .unwrap_or_else(|e| panic!("{prim:#x}: {e}"));
            }
        }
    }

    /// A triangle mesh program on the host GPU and on WARP (where they have
    /// mesh shaders): triangle (0, 1, 2) covers the target with its corners'
    /// edge flags set as without them; with the culling bit, nothing is lit.
    #[test]
    #[cfg(windows)]
    fn edge_flagged_triangles_draw_as_unflagged_ones() {
        use super::host::{Draw, Host, Root};
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_FEATURE_D3D12_OPTIONS7, D3D12_FEATURE_DATA_D3D12_OPTIONS7,
            D3D12_MESH_SHADER_TIER_NOT_SUPPORTED, D3D12_SHADER_VISIBILITY_MESH,
        };
        let _one = super::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let ps = constant_color();
        let mut ran = 0;
        for (label, warp) in [("hardware", false), ("warp", true)] {
            let Ok(host) = Host::new(warp) else {
                eprintln!("no SM 6.6 {label} device; skipped");
                continue;
            };
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            // SAFETY: the query writes the structure of the given size.
            let queried = unsafe {
                host.device.CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS7,
                    (&raw mut options).cast(),
                    size_of_val(&options) as u32,
                )
            };
            if queried.is_err() || options.MeshShaderTier == D3D12_MESH_SHADER_TIER_NOT_SUPPORTED {
                eprintln!("no mesh shaders on the {label} device; skipped");
                continue;
            }
            for (prim, culled) in [
                (TRIANGLE, false),
                (TRIANGLE | EDGE_FLAGS, false),
                (TRIANGLE | EDGE_FLAGS | CULL, true),
            ] {
                let ms = triangle_mesh(prim).unwrap();
                let root = host
                    .root_signature_for(&[(&ms, D3D12_SHADER_VISIBILITY_MESH)])
                    .unwrap();
                let pso = host
                    .mesh_pipeline_with(&root, &ms.dxil, &ps)
                    .unwrap_or_else(|e| panic!("{label}: mesh pipeline: {e}"));
                let texels = host
                    .draw(
                        &root,
                        &pso,
                        &Draw {
                            roots: vec![(0, Root::Constants(vec![0]))],
                            textures: &[],
                            samplers: &[],
                            vertices: &[],
                            stride: 0,
                            size: (16, 16),
                            mesh_groups: 1,
                        },
                    )
                    .unwrap_or_else(|e| panic!("{label}: draw: {e}"));
                let lit = texels.iter().filter(|t| t[3] != 0.0).count();
                if culled {
                    assert_eq!(lit, 0, "{label}, {prim:#x}: a culled triangle lit texels");
                    continue;
                }
                assert!(
                    texels.iter().all(|t| *t == [1.0, 0.5, 0.25, 1.0]),
                    "{label}, {prim:#x}: {lit} of 256 texels lit"
                );
            }
            eprintln!("{label}: edge-flagged triangles drew");
            ran += 1;
        }
        if ran == 0 {
            eprintln!("no device with mesh shaders; skipped");
        }
    }

    /// A pixel shader interpolating at sample positions (PERSP_SAMPLE
    /// barycentrics, `SPI_PS_INPUT_ADDR` bit 0) runs at sample frequency:
    /// DXC's PSV0 says so in PSInfo.SampleFrequency (runtime info byte 1)
    /// whenever an input is interpolated per sample.
    #[test]
    fn sample_interpolated_inputs_set_psv_sample_frequency() {
        let words = [
            0xC808_0000u32, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001,    // v_interp_p2_f32 v2, v1, attr0.x
            0xF800_180F,
            0x0202_0202, // exp mrt0, v2, v2, v2, v2 done vm
            0xBF81_0000,
        ];
        let psv_of = |input_addr: u32| {
            let native = super::NativeProgram {
                stage: super::HwStage::Ps,
                wave64: false,
                vgprs: 4,
                layout: vec![],
                used: vec![],
                io: crate::native::StageIo::Ps {
                    inputs: vec![crate::native::PsInput {
                        usage: crate::native::Usage::Generic,
                        row: 0,
                        mask: 0b1111,
                        attr: 0,
                        flat: false,
                        fp16: false,
                        pack2: false,
                        row2: 0,
                    }],
                    input_addr,
                    db_shader_control: 0,
                    patches: vec![],
                },
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let el = |name: &str, kind, usage| SigElement {
                name: name.into(),
                index: 0,
                kind,
                comp: 9,
                interp: 0,
                row: Some(0),
                col: 0,
                cols: 4,
                usage,
                half: false,
            };
            let isg1 = super::container::signature(&[el("TEXCOORD", 0, 0)], true);
            let osg1 = super::container::signature(&[el("SV_Target", 16, 15)], false);
            let bytes =
                crate::dxbc::build(&[(super::FourCC::ISG1, &isg1), (super::FourCC::OSG1, &osg1)]);
            let c = super::Container::parse(&bytes).unwrap();
            let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
            let st = crate::structure::structure(&p).unwrap();
            let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = crate::io::raise(&p, &native).unwrap();
            let out = super::emit(&c, &native, &p, &st, &io, &res).unwrap();
            // dxil.dll, where there is one, checks PSV0 against the module.
            #[cfg(windows)]
            if let Ok(mut v) = super::validator::Validator::load(
                super::validator::Validator::default_path().as_deref(),
            ) {
                v.validate(&mut out.dxil.clone()).unwrap();
            }
            let host = super::Container::parse(&out.dxil).unwrap();
            host.part(super::FourCC::PSV0).unwrap().data.to_vec()
        };
        // Runtime info starts after its size; PSInfo is {DepthOutput, SampleFrequency}.
        let center = psv_of(1 << 1);
        assert_eq!(center[4 + 1], 0, "center interpolation is per pixel");
        let sample = psv_of(1 << 0);
        assert_eq!(sample[4 + 1], 1, "PSInfo.SampleFrequency");
    }

    /// Emits a pixel program of `words` with these inputs and signatures.
    fn emit_pixel(
        words: &[u32],
        inputs: Vec<crate::native::PsInput>,
        input_addr: u32,
        isg1: &[u8],
        osg1: &[u8],
    ) -> Result<HostShader, ShaderError> {
        let native = super::NativeProgram {
            stage: super::HwStage::Ps,
            wave64: false,
            vgprs: 8,
            layout: vec![],
            used: vec![],
            io: crate::native::StageIo::Ps {
                inputs,
                input_addr,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let bytes = crate::dxbc::build(&[(FourCC::ISG1, isg1), (FourCC::OSG1, osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native)?;
        let out = emit(&c, &native, &p, &st, &io, &res)?;
        // dxil.dll, where there is one, checks the module.
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut out.dxil.clone()).unwrap();
        }
        Ok(out)
    }

    /// A float4 element per row (16-bit ones declared so in the part).
    fn rows(elements: &[(&str, u8, bool)], input: bool) -> Vec<u8> {
        let els: Vec<SigElement> = elements
            .iter()
            .enumerate()
            .map(|(row, &(name, kind, _))| SigElement {
                name: name.into(),
                index: if kind == 3 { 0 } else { row as u32 },
                kind,
                comp: 9,
                interp: 0,
                row: Some(row as u32),
                col: 0,
                cols: 4,
                usage: 0,
                half: false,
            })
            .collect();
        let mut part = container::signature(&els, input);
        for (row, &(.., half)) in elements.iter().enumerate() {
            if half {
                part[8 + 32 * row + 16] = 6;
            }
        }
        part
    }

    /// Attribute 0 in row 0, 16-bit (one value per dword) or not.
    fn texcoord(fp16: bool) -> crate::native::PsInput {
        crate::native::PsInput {
            usage: crate::native::Usage::Generic,
            row: 0,
            mask: 1,
            attr: 0,
            flat: false,
            fp16,
            pack2: false,
            row2: 0,
        }
    }

    /// A 16-bit result written over a first interpolation half keeps that
    /// half's high 16 bits, which have no host form: what reads only the low
    /// half translates, a read of the whole dword does not.
    #[test]
    fn a_half_over_a_first_interpolation_half_reads_as_a_half() {
        let program = |read: u32| {
            let words = [
                0xD742_0002u32,
                0x0002_0000, // v_interp_p1ll_f16 v2, v0, attr0.x
                0xD75A_0003,
                0x040A_0200, // v_interp_p2_f16 v3, v1, attr0.x, v2
                0xCC21_0002,
                0x0A01_E503, // v_fma_mixlo_f16 v2, sel_lo(v3), 1.0, 0
                read,        // v4 from v2
                0xF800_180F,
                0x0404_0404, // exp mrt0, v4, v4, v4, v4 done vm
                0xBF81_0000,
            ];
            let isg1 = rows(&[("TEXCOORD", 0, true)], true);
            let osg1 = rows(&[("SV_Target", 16, false)], false);
            emit_pixel(&words, vec![texcoord(true)], 1 << 1, &isg1, &osg1)
        };
        program(0x7E08_1702).unwrap(); // v_cvt_f32_f16 v4, v2
        program(0x3408_0490).unwrap(); // v_lshlrev_b32 v4, 16, v2
        let whole = program(0x7E08_0302).unwrap_err(); // v_mov_b32 v4, v2
        assert!(whole.to_string().contains("first half"), "{whole}");
        let part = program(0x3408_0488).unwrap_err(); // v_lshlrev_b32 v4, 8, v2
        assert!(part.to_string().contains("first half"), "{part}");
    }

    /// A pixel export before the `done` one, both under a lane guard (EXEC
    /// narrowed without a branch): the lanes outside it are discarded at the
    /// `done` export, so the first one stores in every lane. With EXEC
    /// restored before the `done` export those lanes would write only the
    /// second target, which stays refused.
    #[test]
    fn an_export_before_a_discarding_done_export_stores_unguarded() {
        let program = |restore: &[u32]| {
            let mut words = vec![
                0xBE80_037Eu32, // s_mov_b32 s0, exec_lo
                0x7C22_0080,    // v_cmpx_lt_f32 exec, 0, v0
                0xF800_000F,
                0x0000_0000, // exp mrt0, v0, v0, v0, v0
            ];
            words.extend(restore);
            words.extend([
                0xF800_181F,
                0x0000_0000, // exp mrt1, v0, v0, v0, v0 done vm
                0xBF81_0000,
            ]);
            let isg1 = rows(&[("SV_Position", 3, false)], true);
            let osg1 = rows(&[("SV_Target", 16, false), ("SV_Target", 16, false)], false);
            emit_pixel(&words, vec![], 1 << 8, &isg1, &osg1)
        };
        program(&[]).unwrap();
        let e = program(&[0xBEFE_0300]).unwrap_err(); // s_mov_b32 exec_lo, s0
        assert!(e.to_string().contains("a conditional export"), "{e}");
    }

    /// The compiler reuses the j register (`v_mov` under a kill's EXEC) and
    /// interpolates in the other lanes: those still hold j, which
    /// structurisation's lane conditions prove. Interpolating in the lanes
    /// the copy ran in stays refused.
    #[test]
    fn an_interpolation_after_j_is_reused_in_other_lanes_translates() {
        let program = |other_lanes: &[u32]| {
            let mut words = vec![
                0xBE80_037Eu32, // s_mov_b32 s0, exec_lo
                0x7C22_0480,    // v_cmpx_lt_f32 exec, 0, v2
                0x7E02_0280,    // v_mov_b32 v1, 0
            ];
            words.extend(other_lanes);
            words.extend([
                0xC80C_0000, // v_interp_p1_f32 v3, v0, attr0.x
                0xC80D_0001, // v_interp_p2_f32 v3, v1, attr0.x
                0xBEFE_0300, // s_mov_b32 exec_lo, s0
                0xF800_180F,
                0x0303_0303, // exp mrt0, v3, v3, v3, v3 done vm
                0xBF81_0000,
            ]);
            let isg1 = rows(&[("TEXCOORD", 0, false), ("SV_Position", 3, false)], true);
            let osg1 = rows(&[("SV_Target", 16, false)], false);
            emit_pixel(&words, vec![texcoord(false)], 1 << 1 | 1 << 8, &isg1, &osg1)
        };
        program(&[0x8A7E_7E00]).unwrap(); // s_andn2_b32 exec_lo, s0, exec_lo
        let e = program(&[]).unwrap_err();
        assert!(e.to_string().contains("computed barycentrics"), "{e}");
    }

    /// Two elements sharing a row, the one at the higher channels listed
    /// first (as console signatures have them): the signature part keeps
    /// them in element order within the row, as the validator does.
    #[test]
    fn elements_sharing_a_row_keep_their_order() {
        let el = |index, col, cols| SigElement {
            name: "TEXCOORD".into(),
            index,
            kind: 0,
            comp: 9,
            interp: 0,
            row: Some(0),
            col,
            cols,
            usage: 0,
            half: false,
        };
        let isg1 = container::signature(&[el(1, 1, 2), el(0, 0, 1)], true);
        let osg1 = rows(&[("SV_Target", 16, false)], false);
        let words = [
            0xC808_0000u32, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001,    // v_interp_p2_f32 v2, v1, attr0.x
            0xF800_180F,
            0x0202_0202, // exp mrt0, v2, v2, v2, v2 done vm
            0xBF81_0000,
        ];
        emit_pixel(&words, vec![texcoord(false)], 1 << 1, &isg1, &osg1).unwrap();
    }

    /// An unread float input packed beside the front face (as console
    /// signatures have it when the float was `nointerpolation`) takes the
    /// row's constant mode: DXIL wants one mode per row.
    #[test]
    fn a_row_keeps_one_interpolation_mode() {
        let el = |name: &str, kind, comp, row, col| SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp,
            interp: 0,
            row: Some(row),
            col,
            cols: 1,
            usage: 0,
            half: false,
        };
        let isg1 = container::signature(
            &[
                el("TEXCOORD", 0, 9, 0, 0),
                el("SV_IsFrontFace", 13, 5, 0, 1),
                el("COLOR", 0, 9, 1, 0),
            ],
            true,
        );
        let osg1 = rows(&[("SV_Target", 16, false)], false);
        let words = [
            0xC808_0000u32, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001,    // v_interp_p2_f32 v2, v1, attr0.x
            0xF800_180F,
            0x0202_0202, // exp mrt0, v2, v2, v2, v2 done vm
            0xBF81_0000,
        ];
        let color = crate::native::PsInput {
            row: 1,
            ..texcoord(false)
        };
        emit_pixel(&words, vec![color], 1 << 1, &isg1, &osg1).unwrap();
    }

    /// The LLVM text `dxc -dumpbin` prints for a DXIL container, when dxc
    /// is installed.
    #[cfg(windows)]
    fn dump_dxil(container: &[u8]) -> Option<String> {
        let dxc = match std::env::var_os("VICTORY_PC_DXC") {
            Some(p) => std::path::PathBuf::from(p),
            None => super::validator::Validator::default_path()?.with_file_name("dxc.exe"),
        };
        if !dxc.exists() {
            return None;
        }
        let file = std::env::temp_dir().join(format!("victory-interp-{}.dxil", std::process::id()));
        std::fs::write(&file, container).ok()?;
        let out = std::process::Command::new(dxc)
            .arg("-dumpbin")
            .arg(&file)
            .output()
            .ok()?;
        let _ = std::fs::remove_file(&file);
        assert!(
            out.status.success(),
            "dxc -dumpbin failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        Some(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    #[cfg(windows)]
    #[test]
    fn interpolated_f16_writes_the_op_sel_half_with_modifiers() {
        // p1_f32 v2, v0, attr0.x; then a p2_f16 completing it into v2.
        let p2 = |dst_high: bool, omod: u32, clamp: bool| {
            let vgpr = |n: u32| 256 + n;
            let mut w0 = 0x35u32 << 26 | 0x35au32 << 16 | 2;
            if dst_high {
                w0 |= 1 << 14; // op_sel[3]: the high destination half.
            }
            if clamp {
                w0 |= 1 << 15;
            }
            let w1 = vgpr(1) << 9 | vgpr(2) << 18;
            [w0, w1 | omod << 27]
        };
        let dump = |dst_high: bool, omod: u32, clamp: bool| {
            let [w0, w1] = p2(dst_high, omod, clamp);
            let words = [
                0xC808_0000u32, // v_interp_p1_f32 v2, v0, attr0.x
                w0,
                w1, // v_interp_p2_f16 v2, attr0, v1, v0
                0xF800_180F,
                0x0202_0202, // exp mrt0, v2, v2, v2, v2 done vm
                0xBF81_0000,
            ];
            let native = super::NativeProgram {
                stage: super::HwStage::Ps,
                wave64: false,
                vgprs: 4,
                layout: vec![],
                used: vec![],
                io: crate::native::StageIo::Ps {
                    inputs: vec![crate::native::PsInput {
                        usage: crate::native::Usage::Generic,
                        row: 0,
                        mask: 0b1111,
                        attr: 0,
                        flat: false,
                        fp16: true,
                        pack2: false,
                        row2: 0,
                    }],
                    input_addr: 2,
                    db_shader_control: 0,
                    patches: vec![],
                },
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let el = |name: &str, kind, usage, half| SigElement {
                name: name.into(),
                index: 0,
                kind,
                comp: 9,
                interp: 0,
                row: Some(0),
                col: 0,
                cols: 4,
                usage,
                half,
            };
            let isg1 = super::container::signature(&[el("TEXCOORD", 0, 0, true)], true);
            let osg1 = super::container::signature(&[el("SV_Target", 16, 15, false)], false);
            let bytes =
                crate::dxbc::build(&[(super::FourCC::ISG1, &isg1), (super::FourCC::OSG1, &osg1)]);
            let c = super::Container::parse(&bytes).unwrap();
            let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
            let st = crate::structure::structure(&p).unwrap();
            let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = crate::io::raise(&p, &native).unwrap();
            super::emit(&c, &native, &p, &st, &io, &res).unwrap().dxil
        };
        let (Some(low), Some(high)) = (
            dump_dxil(&dump(false, 0, false)),
            dump_dxil(&dump(true, 1, true)),
        ) else {
            eprintln!("no dxc.exe; DXIL check skipped");
            return;
        };
        // The low-half write keeps the destination's high half.
        assert!(low.contains("-65536"), "low write keeps the high half");
        // The high-half write shifts the result up by 16.
        assert!(high.contains("shl i32"), "op_sel[3] writes the high half");
        // omod mul:2 and clamp (dx.op.unary.f32, opcode 7) are applied.
        assert!(high.contains("2.000000e+00"), "omod multiplies by two");
        assert!(
            high.contains("dx.op.unary.f32(i32 7"),
            "clamp saturates the result"
        );
        assert!(!low.contains("dx.op.unary.f32(i32 7"));
    }

    /// A pixel shader reading channel z of an input whose element starts at
    /// w. Looking the channel up subtracted the element's first channel
    /// before checking the channel lies in it, so in a build with overflow
    /// checks (tests, debug) any read below an element that starts at z or
    /// w panicked, whichever row it was in (found by random-input testing).
    #[test]
    fn reading_a_channel_below_an_input_element_does_not_underflow() {
        // v_interp_mov_f32 v3, p0, attr0.z; s_endpgm
        let words = [0xC80E_0202u32, 0xBF81_0000];
        let native = super::NativeProgram {
            stage: super::HwStage::Ps,
            wave64: false,
            vgprs: 4,
            layout: vec![],
            used: vec![],
            io: crate::native::StageIo::Ps {
                inputs: vec![crate::native::PsInput {
                    usage: crate::native::Usage::Generic,
                    row: 0,
                    mask: 0b1000,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let w_only = SigElement {
            name: "TEXCOORD".into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 3,
            cols: 1,
            usage: 0,
            half: false,
        };
        let isg1 = super::container::signature(&[w_only], true);
        let bytes = crate::dxbc::build(&[(super::FourCC::ISG1, &isg1)]);
        let c = super::Container::parse(&bytes).unwrap();
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        // Translated or refused with a reason; it must not panic.
        let _ = super::emit(&c, &native, &p, &st, &io, &res);
    }
}
