//! Title memory at raw GPU addresses (decision 0005, the unified-memory
//! item; the contract is `victory_shader::emit`'s "Title memory").
//! Translated shaders that read or write memory at addresses they compute
//! look each address up in a page table: per 64 KiB page below 2^40, the
//! heap index of a pair of raw views (SRV, UAV) of the host copy holding
//! the page (`crate::shadow`) and the low dword of that copy's title
//! address. The copies are the ones draws and dispatches use through views,
//! so what one shader writes through a view another reads at its address,
//! within a submission too.
//!
//! Each view heap has one page table, which every list using title memory
//! with that heap bound reads, and each host copy has a pair of views in
//! slots of the heap kept for it while it lives ([`HostHeap::pair`], past
//! the per-draw dynamic slots). At each submission with such a list the
//! table is rewritten if copies came or went since, and every copy it names
//! is brought up to date (page by page, as every refresh; once per
//! submission until a step writes title memory). Shaders mark the pages
//! they write in their list's bitmap; after the submission each marked page
//! goes back into title memory ([`crate::shadow::Shadow::copy_back_page`])
//! and the bitmap is cleared.
//!
//! Cost per submission with such a list: one change check of every host
//! copy (the refresh), copies of the pages the title wrote, a compare of
//! each 64 KiB page shaders wrote, and the table's rewrite when copies came
//! or went (two views per new copy); per list a bitmap of 256 KiB.
//!
//! Memory no copy holds reads 0 and takes no writes (null views,
//! bounds-checked); shaders mark such pages (or 256 MiB regions with no
//! block) in the bitmap and [`log_frame`] counts them.
//! shortcut: a page two copies share without either holding all of it is
//! the larger one's.
//! shortcut: a table names copies in at most [`CAPACITY`] 256 MiB regions;
//! past that, the smallest copies read as missing (logged once).

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::descriptor::host::HostHeap;
use crate::host::HostList;
use crate::shadow::Shadow;

/// Pages per level-2 block: 4096 pages of 64 KiB, 256 MiB.
const BLOCK: usize = 4096;
/// Level-1 entries: 2^40 bytes in 256 MiB blocks.
const LEVEL1: usize = 4096;
/// Level-2 blocks a table has room for besides the null one: copies in 512
/// distinct 256 MiB regions (the world's ~4,100 copies use a few dozen).
const CAPACITY: usize = 512;
/// The key of our own view heap ([`heap`]) among the title's.
pub(crate) const OURS: usize = usize::MAX;

/// A list's use of title memory with a view heap bound.
pub(crate) struct Map {
    /// The heap's page table's host GPU address (the root SRV).
    pub(crate) table: u64,
    /// The list's page bitmap's host GPU address (the root UAV).
    pub(crate) dirty: u64,
    /// Ranges that go back byte for byte instead (root UAVs, see
    /// [`Map::precise`]).
    precise: Arc<Mutex<Vec<(u64, u64)>>>,
}

impl Map {
    /// Shaders write `[address, address + len)` through a root UAV, which
    /// goes back by its bytes before and after the run
    /// ([`Shadow::gpu_writes`]): the marked pages it touches are left to
    /// that. A whole page from the copy would put back what it held at
    /// the submission over what the title (or DirectStorage, through
    /// physical pages no watch sees) wrote there since.
    /// shortcut: a marked page partly in such a range loses what shaders
    /// wrote in its other part.
    pub(crate) fn precise(&self, address: u64, len: u64) {
        self.precise
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((address, len));
    }
}

/// What a heap's page table names: the copies, the level-1 index of each
/// block (block 0 is the null one), and the copy each entry names.
#[derive(Default)]
struct Layout {
    copies: Vec<Arc<Shadow>>,
    blocks: Vec<usize>,
    owner: Vec<Option<usize>>,
}

/// A view heap's page table, shared by every list using title memory with
/// the heap bound, and the view pairs of the copies it names.
struct Shared {
    /// The host heap it was made for (a heap made later at a released one's
    /// address is another).
    host: usize,
    table: ID3D12Resource,
    data: usize,
    gpu: u64,
    /// The pair of null views entries without a copy name.
    null: u32,
    /// Each copy's view pair (its first slot), by copy.
    pairs: HashMap<usize, u32>,
    /// The copies' generation it names ([`crate::shadow::generation`]).
    generation: u64,
    layout: Arc<Layout>,
}

// SAFETY: `data` is the persistent mapping of `table`, written only under
// the lock; D3D12 resources are free-threaded.
unsafe impl Send for Shared {}

/// Page tables by view heap (the title heap's address, or [`OURS`]).
static SHARED: Mutex<Vec<(usize, Shared)>> = Mutex::new(Vec::new());

fn shared() -> std::sync::MutexGuard<'static, Vec<(usize, Shared)>> {
    SHARED.lock().unwrap_or_else(|e| e.into_inner())
}

/// Runs `f` on the host heap behind view heap `key` while it cannot go
/// away; None if it is gone or has no host heap.
fn with_heap<R>(key: usize, f: impl FnOnce(&HostHeap) -> R) -> Option<R> {
    if key == OURS {
        return heap().map(f);
    }
    crate::descriptor::with_host(key, f)
}

/// Since the last frame line: draws and dispatches with shaders using title
/// memory (by pipeline), page tables written, pages copied back, pages and
/// 256 MiB regions missed (no copy), and the first missed addresses.
#[derive(Default)]
struct Stats {
    uses: std::collections::BTreeMap<String, u64>,
    tables: u64,
    written: u64,
    /// Pages shaders wrote that were not copied back: their list read
    /// memory with no copy, so what they computed is not the console's.
    withheld: u64,
    /// The same for ranges written through views (bytes).
    withheld_bytes: u64,
    missed_pages: u64,
    missed_regions: u64,
    first_missed: Vec<u64>,
}

static STATS: std::sync::Mutex<Option<Stats>> = std::sync::Mutex::new(None);

fn stats<R>(f: impl FnOnce(&mut Stats) -> R) -> R {
    f(STATS.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_default())
}

/// `bytes` of a range shaders wrote through a view did not go back into
/// title memory: their list read memory with no copy.
pub(crate) fn note_withheld(bytes: u64) {
    stats(|s| s.withheld_bytes += bytes);
}

/// A draw or dispatch of `pipeline` uses title memory.
pub(crate) fn note_use(pipeline: &str) {
    stats(|s| *s.uses.entry(pipeline.to_owned()).or_default() += 1);
}

/// Logs what title memory did since the last call, if anything: a line per
/// presented frame while shaders use it.
pub(crate) fn log_frame() {
    let s = stats(std::mem::take);
    if s.uses.is_empty() && s.written + s.withheld + s.withheld_bytes + s.missed_pages + s.missed_regions == 0 {
        return;
    }
    let mut uses: Vec<(&String, &u64)> = s.uses.iter().collect();
    uses.sort_by(|a, b| b.1.cmp(a.1));
    let uses: Vec<String> = uses.iter().take(32).map(|(p, n)| format!("{n} {p}")).collect();
    let missed: Vec<String> = s.first_missed.iter().map(|a| format!("{a:#x}")).collect();
    tracing::info!(
        "title memory this frame: {} draws/dispatches ({}), {} page tables, {} pages \
         copied back, {} withheld and {:.1} MB of views kept (their list read memory with no copy), missed {} pages \
         and {} 256 MiB regions with no copy (first: {})",
        s.uses.values().sum::<u64>(),
        uses.join(", "),
        s.tables,
        s.written,
        s.withheld,
        s.withheld_bytes as f64 / 1e6,
        s.missed_pages,
        s.missed_regions,
        missed.join(" ")
    );
}

/// Our view heap for lists that bind none of the title's (dynamic slots
/// only).
pub(crate) fn heap() -> Option<&'static crate::descriptor::host::HostHeap> {
    static HEAP: std::sync::OnceLock<Option<crate::descriptor::host::HostHeap>> =
        std::sync::OnceLock::new();
    HEAP.get_or_init(|| crate::descriptor::host::HostHeap::new(false, 0))
        .as_ref()
}

/// The page table for host copies of title memory `copies` (start,
/// bytes), copy k's view pair at heap index `pairs[k]`, `null` the null
/// pair: the table's dwords, the level-1 index of each block, and the copy
/// each entry names. At most [`CAPACITY`] blocks besides the null one.
fn table(copies: &[(u64, u64)], pairs: &[u32], null: u32) -> (Vec<u32>, Vec<usize>, Vec<Option<usize>>) {
    let mut order: Vec<usize> = (0..copies.len()).collect();
    order.sort_by_key(|&k| std::cmp::Reverse(copies[k].1));
    let mut level1 = vec![0u32; LEVEL1];
    // Block 0 is the null block (no copy).
    let mut blocks = vec![usize::MAX];
    let mut owner: Vec<Option<(usize, bool)>> = vec![None; BLOCK];
    let mut full = false;
    for &k in &order {
        let (start, len) = copies[k];
        if len == 0 || start + len > 1 << 40 {
            continue;
        }
        for page in start >> 16..=(start + len - 1) >> 16 {
            let l1 = (page >> 12) as usize;
            if level1[l1] == 0 {
                if blocks.len() > CAPACITY {
                    full = true;
                    continue;
                }
                level1[l1] = blocks.len() as u32;
                blocks.push(l1);
                owner.resize(owner.len() + BLOCK, None);
            }
            let entry = level1[l1] as usize * BLOCK + (page & 0xFFF) as usize;
            // Larger copies come first: a copy holding the whole page takes
            // it from them; one holding part of it only takes it if free.
            let whole = page << 16 >= start && (page + 1) << 16 <= start + len;
            if whole || owner[entry].is_none() {
                owner[entry] = Some((k, whole));
            }
        }
    }
    if full {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!(CAPACITY, "title memory copies in more 256 MiB regions than a page table has room for: the rest read as missing"));
    }
    let mut dwords = level1;
    for o in &owner {
        let entry = match *o {
            Some((k, _)) => [pairs[k], copies[k].0 as u32],
            None => [null, 0],
        };
        dwords.extend(entry);
    }
    (dwords, blocks, owner.into_iter().map(|o| o.map(|o| o.0)).collect())
}

impl Shared {
    /// A page table for `heap`, named nothing yet (written at the first
    /// submission that uses it).
    fn new(heap: &HostHeap) -> Option<Shared> {
        let bytes = (LEVEL1 + (CAPACITY + 1) * BLOCK * 2) as u64 * 4;
        let (table, data) = crate::host::host()?.upload_buffer(bytes)?;
        crate::host::name(&table, || "title memory page table".into());
        let null = heap.pair()?;
        heap.raw_pair_at(null, None, 0);
        // SAFETY: a live buffer.
        let gpu = unsafe { table.GetGPUVirtualAddress() };
        Some(Shared {
            host: std::ptr::from_ref(heap) as usize,
            table,
            data: data as usize,
            gpu,
            null,
            pairs: HashMap::new(),
            generation: u64::MAX,
            layout: Arc::default(),
        })
    }

    /// Names every host copy there is now (`now`: their generation): view
    /// pairs made for new copies, those of copies gone made null and given
    /// back, the table rewritten. Only while no submission runs (the GPU
    /// reads the table and the views).
    fn rewrite(&mut self, heap: &HostHeap, now: u64) {
        let copies = crate::shadow::all();
        let live: HashSet<usize> = copies.iter().map(|c| Arc::as_ptr(c) as usize).collect();
        self.pairs.retain(|copy, &mut slot| {
            let keep = live.contains(copy);
            if !keep {
                heap.raw_pair_at(slot, None, 0);
                heap.free_pair(slot);
            }
            keep
        });
        let (mut named, mut slots) = (Vec::with_capacity(copies.len()), Vec::with_capacity(copies.len()));
        for c in copies {
            let key = Arc::as_ptr(&c) as usize;
            let slot = match self.pairs.get(&key) {
                Some(&slot) => slot,
                None => {
                    let Some(slot) = heap.pair() else {
                        static ONCE: std::sync::Once = std::sync::Once::new();
                        ONCE.call_once(|| tracing::warn!("no descriptor slots for a title memory copy's views: it reads as missing"));
                        continue;
                    };
                    heap.raw_pair_at(slot, Some(&c.buffer), c.range().1.next_multiple_of(4));
                    self.pairs.insert(key, slot);
                    slot
                }
            };
            named.push(c);
            slots.push(slot);
        }
        let ranges: Vec<(u64, u64)> = named.iter().map(|c| c.range()).collect();
        let (dwords, blocks, owner) = table(&ranges, &slots, self.null);
        // SAFETY: the mapping has room for `CAPACITY` blocks, which `table`
        // keeps to; no submission runs.
        unsafe { std::ptr::copy_nonoverlapping(dwords.as_ptr(), self.data as *mut u32, dwords.len()) };
        tracing::debug!(copies = ranges.len(), blocks = blocks.len() - 1, "title memory page table");
        self.layout = Arc::new(Layout {
            copies: named,
            blocks,
            owner,
        });
        self.generation = now;
        stats(|s| s.tables += 1);
    }
}

/// The page table of view heap `key`, as the last submission using it
/// left it.
fn layout(key: usize) -> Arc<Layout> {
    shared()
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, s)| s.layout.clone())
        .unwrap_or_default()
}

/// At submission, for a list using title memory with view heap `key`
/// bound: the heap's page table names every host copy there is, each
/// brought up to date (`add` takes a copy's range, once per submission
/// until a step writes title memory).
pub(crate) fn prepare(key: usize, mut add: impl FnMut(&Arc<Shadow>, usize, usize)) {
    let layout = with_heap(key, |heap| {
        let mut shared = shared();
        let host = std::ptr::from_ref(heap) as usize;
        let (_, s) = shared.iter_mut().find(|(k, s)| *k == key && s.host == host)?;
        let now = crate::shadow::generation();
        if s.generation != now {
            s.rewrite(heap, now);
        }
        Some(s.layout.clone())
    })
    .flatten();
    for c in layout.iter().flat_map(|l| &l.copies) {
        let (start, len) = c.range();
        add(c, start as usize, len as usize);
    }
}

/// `list` uses title memory with view heap `key` bound: the heap's page
/// table (made on first use) and a page bitmap of the list's, the table
/// brought up to date at each submission of the list, its written pages
/// copied back after.
pub(crate) fn bind(list: &HostList, key: usize) -> Option<Map> {
    let table = with_heap(key, |heap| {
        let mut shared = shared();
        let host = std::ptr::from_ref(heap) as usize;
        if let Some((_, s)) = shared.iter().find(|(k, s)| *k == key && s.host == host) {
            list.keep(&s.table);
            return Some(s.gpu);
        }
        // A released heap's table goes with it (lists that name it hold it).
        shared.retain(|(k, _)| *k != key);
        let s = Shared::new(heap)?;
        list.keep(&s.table);
        let gpu = s.gpu;
        shared.push((key, s));
        Some(gpu)
    })
    .flatten()?;
    let bits = (CAPACITY + 1) * BLOCK;
    let (bitmap, data) = crate::host::host()?.shared_buffer((bits / 8).next_multiple_of(256) as u64)?;
    crate::host::name(&bitmap, || "title memory written pages".into());
    let words = bits / 32;
    // SAFETY: our mapped buffer of at least `words` dwords.
    unsafe { std::ptr::write_bytes(data.cast::<u32>(), 0, words) };
    // SAFETY: a live buffer.
    let dirty = unsafe { bitmap.GetGPUVirtualAddress() };
    list.keep(&bitmap);
    list.defer_memory(key);
    let precise: Arc<Mutex<Vec<(u64, u64)>>> = Arc::default();
    let byte_for_byte = precise.clone();
    let data = data as usize;
    list.defer_missed(move || {
        let l = layout(key);
        // SAFETY: the bitmap is mapped and the list's run is over.
        let marks = unsafe { std::slice::from_raw_parts(data as *const u32, words) };
        marks.iter().enumerate().any(|(w, &mark)| {
            (0..32).filter(|b| mark >> b & 1 != 0).any(|b| {
                let entry = w * 32 + b;
                entry / BLOCK == 0 || l.blocks.get(entry / BLOCK).is_none() || l.owner.get(entry).is_none_or(|o| o.is_none())
            })
        })
    });
    list.defer_after(move || {
        let Layout { copies, blocks, owner } = &*layout(key);
        let precise = byte_for_byte.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the bitmap is mapped and the list's run is over.
        let marks = unsafe { std::slice::from_raw_parts_mut(data as *mut u32, words) };
        let (mut pages, mut regions, mut missed) = (0, 0, Vec::new());
        let mut writes = Vec::new();
        for (w, mark) in marks.iter_mut().enumerate() {
            let mut bits = std::mem::take(mark);
            while bits != 0 {
                let entry = w * 32 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let (block, page) = (entry / BLOCK, entry % BLOCK);
                // Block 0's bits are 256 MiB regions with no block.
                let Some(&l1) = blocks.get(block).filter(|_| block != 0) else {
                    regions += 1;
                    missed.push((entry as u64) << 28);
                    continue;
                };
                let at = (l1 << 28 | page << 16) as u64;
                match owner.get(entry) {
                    Some(_) if precise.iter().any(|&(a, n)| a < at + (1 << 16) && at < a + n) => {}
                    Some(Some(k)) => writes.push((*k, at)),
                    _ => {
                        pages += 1;
                        missed.push(at);
                    }
                }
            }
        }
        // What shaders computed from memory with no copy (read as zeros) is
        // not the console's: it does not go into title memory, where it broke
        // the title's own data (garbage pointers in title threads, run c1).
        // shortcut: root UAV writes of such a list still go back (their
        // bytes, `Shadow::gpu_writes`).
        let withheld = if pages + regions > 0 { writes.len() as u64 } else { 0 };
        let mut written = 0;
        // Buffers in place hold the shaders' writes whatever is withheld.
        for &(k, at) in writes.iter().filter(|_| withheld > 0) {
            copies[k].landed(at as usize, 1 << 16);
        }
        for &(k, at) in writes.iter().take_while(|_| withheld == 0) {
            let n = copies[k].copy_back_page(at as usize);
            tracing::debug!(
                page = format_args!("{at:#x}"),
                copy = format_args!("{:#x?}", copies[k].range()),
                bytes = n,
                "title memory page written back"
            );
            written += 1;
        }
        stats(|s| {
            s.written += written;
            s.withheld += withheld;
            s.missed_pages += pages;
            s.missed_regions += regions;
            let room = 4usize.saturating_sub(s.first_missed.len());
            s.first_missed.extend(missed.into_iter().take(room));
        });
    });
    Some(Map {
        table,
        dirty,
        precise,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shader's lookup (`victory_shader` lowering) over a table's
    /// dwords: the view index and the byte offset in that view.
    fn look_up(dwords: &[u32], address: u64) -> (u32, u32) {
        let (lo, hi) = (address as u32, (address >> 32) as u32);
        let inside = hi < 256;
        let l1 = if inside { hi << 4 | lo >> 28 } else { 0 };
        let block = if inside { dwords[l1 as usize] } else { 0 };
        let entry = block << 12 | (lo >> 16 & 0xFFF);
        let at = 4096 + 2 * entry as usize;
        (dwords[at], lo.wrapping_sub(dwords[at + 1]))
    }

    #[test]
    fn addresses_find_the_copy_holding_them() {
        // A 4 MiB window, a buffer inside it from 0x40_0000+0x1_8000
        // (pages in part), and a buffer above 4 GiB.
        let copies = [
            (0x40_0000, 0x40_0000),
            (0x41_8000, 0x3_0000),
            (0xd_1234_0000, 0x2_0000),
        ];
        let (dwords, blocks, owner) = table(&copies, &[102, 104, 106], 100);
        // Window, whole pages of the inner buffer, the high buffer.
        assert_eq!(look_up(&dwords, 0x40_0010), (102, 0x10));
        assert_eq!(look_up(&dwords, 0x42_0004), (104, 0x8004));
        assert_eq!(look_up(&dwords, 0xd_1235_fffc), (106, 0x1_fffc));
        // Pages the inner buffer holds in part stay the window's.
        assert_eq!(look_up(&dwords, 0x41_8000), (102, 0x1_8000));
        assert_eq!(look_up(&dwords, 0x44_0000), (102, 0x4_0000));
        // No copy, and above 2^40: the null pair, which reads nothing.
        assert_eq!(look_up(&dwords, 0x9000_0000).0, 100);
        assert_eq!(look_up(&dwords, 0x200_0000_0000).0, 100);
        // Entries map back to their copy and page.
        let entry = |a: u64| blocks.iter().position(|&l| l == (a >> 28) as usize).unwrap() * BLOCK
            + (a >> 16 & 0xFFF) as usize;
        assert_eq!(owner[entry(0x42_0000)], Some(1));
        assert_eq!(owner[entry(0xd_1234_0000)], Some(2));
        assert_eq!(blocks.len(), 3, "null, low 256 MiB, the high buffer's");
    }

    /// Copies in more regions than a table has room for: the largest are
    /// named, the rest read as missing; the table fits its buffer.
    #[test]
    fn a_full_table_names_the_largest_copies() {
        let copies: Vec<(u64, u64)> = (0..CAPACITY as u64 + 2).map(|k| (k << 28, 0x1_0000 + k)).collect();
        let pairs: Vec<u32> = (0..copies.len() as u32).map(|k| 10 + 2 * k).collect();
        let (dwords, blocks, _) = table(&copies, &pairs, 4);
        assert_eq!(blocks.len(), CAPACITY + 1);
        assert_eq!(dwords.len(), LEVEL1 + (CAPACITY + 1) * BLOCK * 2);
        assert_eq!(look_up(&dwords, (CAPACITY as u64 + 1) << 28).0, pairs[CAPACITY + 1]);
        assert_eq!(look_up(&dwords, 0).0, 4, "the smallest copy is left out");
    }

    /// Reads 4 dwords of title memory at (`lo`, `hi`) through the page
    /// table and the heap's views, as translated shaders do, into `dst`
    /// from byte `at`.
    const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0, numDescriptors=unbounded)), SRV(t0, space=2), UAV(u0), RootConstants(num32BitConstants=3, b0)"
ByteAddressBuffer views[] : register(t0);
ByteAddressBuffer table : register(t0, space2);
RWByteAddressBuffer dst : register(u0);
cbuffer c : register(b0) { uint lo; uint hi; uint at; }
[RootSignature(RS)]
[numthreads(4, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint a = lo + id.x * 4;
    bool inside = hi < 256;
    uint l1 = inside ? (hi << 4 | a >> 28) : 0;
    uint block = inside ? table.Load(l1 * 4) : 0;
    uint entry = block << 12 | (a >> 16 & 0xFFF);
    uint2 e = table.Load2((4096 + 2 * entry) * 4);
    dst.Store(at + id.x * 4, views[NonUniformResourceIndex(e.x)].Load(a - e.y));
}
"#;

    fn pipeline() -> (
        windows::Win32::Graphics::Direct3D12::ID3D12RootSignature,
        windows::Win32::Graphics::Direct3D12::ID3D12PipelineState,
    ) {
        use windows::Win32::Graphics::Direct3D::Fxc::{D3DCOMPILE_ENABLE_UNBOUNDED_DESCRIPTOR_TABLES, D3DCompile};
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::core::s;
        let host = crate::host::host().unwrap();
        let (mut code, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
        // SAFETY: the source and names outlive the call; the errors are text.
        unsafe {
            let made = D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                s!("memory"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                D3DCOMPILE_ENABLE_UNBOUNDED_DESCRIPTOR_TABLES,
                0,
                &mut code,
                Some(&mut errors),
            );
            let text = errors.map(|e| String::from_utf8_lossy(std::slice::from_raw_parts(e.GetBufferPointer().cast::<u8>(), e.GetBufferSize())).into_owned());
            assert!(made.is_ok(), "{text:?}");
        }
        let code = code.unwrap();
        // SAFETY: the blob's own bytes; a compiled shader with its root
        // signature inside.
        unsafe {
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
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

    /// A copy's view pair, in view heap `key`'s table.
    fn pair_of(key: usize, copy: &Arc<Shadow>) -> Option<u32> {
        shared()
            .iter()
            .find(|(k, _)| *k == key)?
            .1
            .pairs
            .get(&(Arc::as_ptr(copy) as usize))
            .copied()
    }

    /// Two lists of a submission read title memory through one page table;
    /// the copies' views stay where they are for later lists and
    /// submissions, also when another copy comes; none takes a per-draw
    /// dynamic slot.
    #[test]
    fn lists_share_a_table_and_copies_keep_their_views() {
        use crate::watch::PAGE;
        let (Some(host), Some(views)) = (crate::host::host(), heap()) else {
            return; // no host GPU here
        };
        let (root, pso) = pipeline();
        let (out, at) = host.shared_buffer(256).unwrap();
        // SAFETY: a live buffer.
        let out_gpu = unsafe { out.GetGPUVirtualAddress() };
        // Title memory (below 2^40, as the runtime places it): 1, 2 and
        // (from the second submission on) 1 page, write-watched as its pools.
        let pool = |pages: usize| -> usize {
            use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
            (0..256u64)
                .map(|k| 0x40_0000_0000 + (k << 28))
                // SAFETY: a fresh allocation where nothing is mapped, or none.
                .map(|hint| unsafe { VirtualAlloc(hint as *const _, pages * PAGE, MEM_RESERVE | MEM_COMMIT | crate::watch::tests::MEM_WRITE_WATCH, PAGE_READWRITE) } as usize)
                .find(|&p| p != 0)
                .expect("memory below 2^40")
        };
        let memory = [pool(1), pool(2), pool(1)];
        let copy = |p: usize, pages: usize| {
            crate::shadow::register(p, p as u64, (pages * PAGE) as u64);
            crate::shadow::find(p as u64, (pages * PAGE) as u64).unwrap()
        };
        let (a, b) = (copy(memory[0], 1), copy(memory[1], 2));
        let lists = [host.new_list().unwrap(), host.new_list().unwrap()];
        // Each list reads 4 dwords at its address into its 16 bytes of `out`.
        let record = |reads: [usize; 2]| -> Vec<Map> {
            let maps: Vec<Map> = lists.iter().map(|l| bind(l, OURS).expect("a page table")).collect();
            for (k, (list, &address)) in lists.iter().zip(&reads).enumerate() {
                let constants = [address as u32, (address >> 32) as u32, 16 * k as u32];
                // SAFETY: an open list, live objects of ours.
                unsafe {
                    list.list.SetDescriptorHeaps(&[Some(views.heap.clone())]);
                    list.list.SetComputeRootSignature(&root);
                    list.list.SetComputeRootDescriptorTable(0, views.gpu(0));
                    list.list.SetComputeRootShaderResourceView(1, maps[k].table);
                    list.list.SetComputeRootUnorderedAccessView(2, out_gpu);
                    list.list.SetComputeRoot32BitConstants(3, 3, constants.as_ptr().cast(), 0);
                    list.list.SetPipelineState(&pso);
                    list.list.Dispatch(1, 1, 1);
                }
                list.close();
            }
            maps
        };
        // The title writes after recording, before submitting.
        let fill = |p: usize, base: u32| {
            for k in 0..4 {
                // SAFETY: our pool (a guarded page faults into the handler).
                unsafe { (p as *mut u32).add(k).write_volatile(base + k as u32) };
            }
        };
        let read = || -> Vec<u32> {
            // SAFETY: the mapped output, the GPU done with it.
            unsafe { std::slice::from_raw_parts(at.cast::<u32>(), 8).to_vec() }
        };
        let maps = record([memory[0] + 16, memory[1] + PAGE]);
        assert_eq!(maps[0].table, maps[1].table, "one table for the heap");
        assert_ne!(maps[0].dirty, maps[1].dirty, "a bitmap per list");
        fill(memory[0] + 16, 0x100);
        fill(memory[1] + PAGE, 0x200);
        host.execute(&[&lists[0], &lists[1]]);
        assert_eq!(read(), [0x100, 0x101, 0x102, 0x103, 0x200, 0x201, 0x202, 0x203]);
        let pairs = (pair_of(OURS, &a).unwrap(), pair_of(OURS, &b).unwrap());
        assert!(pairs.0 >= 1 << 18 && pairs.1 >= 1 << 18, "past the dynamic slots");
        // A second submission, after a third copy came.
        let c = copy(memory[2], 1);
        for list in &lists {
            list.reset();
        }
        record([memory[2] + 32, memory[0] + 16]);
        fill(memory[2] + 32, 0x300);
        fill(memory[0] + 16, 0x400);
        host.execute(&[&lists[0], &lists[1]]);
        assert_eq!(read(), [0x300, 0x301, 0x302, 0x303, 0x400, 0x401, 0x402, 0x403]);
        assert_eq!((pair_of(OURS, &a).unwrap(), pair_of(OURS, &b).unwrap()), pairs, "views kept");
        assert!(pair_of(OURS, &c).is_some());
        for (&p, copy) in memory.iter().zip([a, b, c]) {
            crate::shadow::unregister(p, p as u64);
            drop(copy);
        }
    }
}
