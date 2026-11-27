//! XMem 标题视图的诊断历史：只记录物理节身份和 VA 关系，不保活、不读取后备。

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::ops::{Bound::Excluded, Range, RangeInclusive};
use std::sync::{LazyLock, Mutex, OnceLock};

use victory_memory::{PAGE_BYTES, physical::{PageNumbers, PhysicalPage}};

const PAGE: usize = PAGE_BYTES as usize;
const LIVE_LIMIT: usize = 131_072;
const UNMAP_LIMIT: usize = 32_768;
const REBIND_PAIR_LIMIT: usize = 65_536;
static RECORDING: OnceLock<bool> = OnceLock::new();
static HISTORY: LazyLock<Mutex<SectionHistory>> = LazyLock::new(Default::default);

#[derive(Clone, Copy)]
struct Binding {
    base: usize,
    end: usize,
    section: PhysicalPage,
    first: PhysicalPage,
}

impl Binding {
    fn new(section: PhysicalPage, offset: usize, base: usize, len: usize) -> Option<Self> {
        if section.is_private() || base == 0 || len == 0 || !base.is_multiple_of(PAGE)
            || !len.is_multiple_of(PAGE) || offset.checked_add(len / PAGE)? > PageNumbers::MAX_SECTION_PAGES
        {
            return None;
        }
        Some(Self { base, end: base.checked_add(len)?, section, first: section.offset(offset) })
    }

    fn page_at(self, at: usize) -> PhysicalPage {
        self.first.offset((at - self.base) / PAGE)
    }

    fn overlap(self, span: &Range<usize>) -> Option<Range<usize>> {
        let (start, end) = (self.base.max(span.start), self.end.min(span.end));
        (start < end).then_some(start..end)
    }

    fn page_ids(self, span: Range<usize>) -> RangeInclusive<PhysicalPage> {
        self.page_at(span.start)..=self.page_at(span.end - 1)
    }
}

#[derive(Clone, Copy)]
struct Event {
    run: Binding,
    sequence: u64,
}

#[derive(Clone, Copy, Default)]
struct Stats {
    maps: u64,
    unmaps: u64,
    evicted_live: u64,
    evicted_unmaps: u64,
}

#[derive(Default)]
struct SectionHistory {
    // 长期活跃的绑定不受解绑历史轮转影响；超过独立上限时淘汰最早的视图。
    live: BTreeMap<usize, Event>,
    bind_order: BTreeSet<(u64, usize)>,
    unmaps: VecDeque<Event>,
    stats: Stats,
}

impl SectionHistory {
    fn remove_live(&mut self, base: usize) {
        if let Some(event) = self.live.remove(&base) {
            self.bind_order.remove(&(event.sequence, base));
        }
    }

    fn observe(&mut self, run: Binding, mapped: bool, live_limit: usize, unmap_limit: usize) {
        self.remove_live(run.base);
        let count = if mapped { &mut self.stats.maps } else { &mut self.stats.unmaps };
        *count = count.saturating_add(1);
        let event = Event { run, sequence: self.stats.maps.saturating_add(self.stats.unmaps) };
        if mapped {
            self.live.insert(run.base, event);
            self.bind_order.insert((event.sequence, run.base));
            if self.live.len() > live_limit {
                let (_, base) = self.bind_order.pop_first().expect("indexed XMem binding");
                self.live.remove(&base);
                self.stats.evicted_live = self.stats.evicted_live.saturating_add(1);
            }
        } else {
            // 解绑点提供真实 Slice，即使其原绑定记录已淘汰也不能漏掉这次成功解绑。
            self.unmaps.push_back(event);
            if self.unmaps.len() > unmap_limit {
                self.unmaps.pop_front();
                self.stats.evicted_unmaps = self.stats.evicted_unmaps.saturating_add(1);
            }
        }
    }

    fn snapshot(&self, span: Option<&Range<usize>>) -> Snapshot {
        let live = span.map_or_else(Vec::new, |span| {
            let first = self.live.range(..=span.start).next_back()
                .filter(|(_, event)| event.run.end > span.start);
            first.into_iter().chain(self.live.range((Excluded(span.start), Excluded(span.end))))
                .map(|(_, event)| *event).collect()
        });
        Snapshot { live, unmaps: self.unmaps.clone(), stats: self.stats, retained_live: self.live.len() }
    }
}

/// 在首次发布可映射的节身份前、映射锁之外缓存诊断开关。
pub fn prepare_xmem_section_history() {
    let _ = RECORDING.get_or_init(crate::page_inventory::enabled);
}

/// 只在 XMem 成功发布或解绑实际标题视图后调用。`section` 和 `offset` 来自真实 Slice；
/// `base`/`len` 是该次完成的视图范围，不是原始请求的范围。此处只获取单向诊断锁，
/// 不查询映射表或日志，不保活 section；不把映射/提交当作字节写入。
pub fn note_xmem_section_mapping(section: PhysicalPage, offset: usize, base: usize, len: usize, mapped: bool) {
    if RECORDING.get() == Some(&true) && let Some(run) = Binding::new(section, offset, base, len) {
        HISTORY.lock().unwrap_or_else(|e| e.into_inner()).observe(run, mapped, LIVE_LIMIT, UNMAP_LIMIT);
    }
}

struct Snapshot {
    live: Vec<Event>,
    unmaps: VecDeque<Event>,
    stats: Stats,
    retained_live: usize,
}

#[derive(Default)]
struct TargetHistory {
    matched_pages: usize,
    matched_views: usize,
    matched_sections: usize,
    bind_first: Option<u64>,
    bind_last: Option<u64>,
    unmaps_at_va: usize,
    unmaps_with_page: usize,
    same_page_rebinds: usize,
    same_section_rebinds_with_other_page: usize,
    different_section_rebinds: usize,
    incomplete_rebind_events: usize,
}

impl Snapshot {
    fn compare(&self, span: Option<&Range<usize>>, pages: &BTreeSet<PhysicalPage>, mut pairs: usize) -> TargetHistory {
        let mut result = TargetHistory::default();
        let mut live_targets = Vec::new();
        let mut matched_pages = BTreeSet::new();
        let mut matched_sections = BTreeSet::new();
        if let Some(span) = span {
            for &event in &self.live {
                let Some(hit) = event.run.overlap(span) else { continue; };
                let mut page_hits = pages.range(event.run.page_ids(hit)).peekable();
                if page_hits.peek().is_none() { continue; }
                matched_pages.extend(page_hits.copied());
                matched_sections.insert(event.run.section);
                live_targets.push(event);
                result.bind_first = Some(result.bind_first.map_or(event.sequence, |n| n.min(event.sequence)));
                result.bind_last = Some(result.bind_last.map_or(event.sequence, |n| n.max(event.sequence)));
            }
        }
        result.matched_pages = matched_pages.len();
        result.matched_sections = matched_sections.len();
        result.matched_views = live_targets.len();

        // 近因优先：到达比较上限后仍完整统计有界历史中的 VA/物理身份单独命中。
        for old in self.unmaps.iter().rev() {
            let run = old.run;
            result.unmaps_with_page += usize::from(pages.range(run.page_ids(run.base..run.end)).next().is_some());
            let Some(at_va) = span.and_then(|span| run.overlap(span)) else { continue; };
            result.unmaps_at_va += 1;
            let start = live_targets.partition_point(|event| event.run.end <= at_va.start);
            let (mut same_page, mut other_page, mut other_section) = (false, false, false);
            for event in live_targets[start..].iter().take_while(|event| event.run.base < at_va.end) {
                if pairs == 0 {
                    result.incomplete_rebind_events += 1;
                    break;
                }
                pairs -= 1;
                let Some(hit) = event.run.overlap(&at_va) else { continue; };
                // 要在同一 VA 重叠段且新绑定的页身份命中目标，不能拼接两种单独的命中。
                if old.sequence >= event.sequence || pages.range(event.run.page_ids(hit.clone())).next().is_none() {
                    continue;
                }
                if event.run.section != run.section {
                    other_section = true;
                } else if event.run.page_at(hit.start) != run.page_at(hit.start) {
                    other_page = true;
                } else {
                    same_page = true;
                }
            }
            result.same_page_rebinds += usize::from(same_page);
            result.same_section_rebinds_with_other_page += usize::from(other_page);
            result.different_section_rebinds += usize::from(other_section);
        }
        result
    }
}

pub(crate) fn trace_color_section_history(
    index: usize, slot: usize, subresource: u32, span: Option<(u64, u64)>, pages: &BTreeSet<PhysicalPage>,
) {
    if !crate::page_inventory::enabled() || !tracing::enabled!(tracing::Level::INFO) { return; }
    let span = span.and_then(|(start, end)| Some(usize::try_from(start).ok()?..usize::try_from(end).ok()?))
        .filter(|span| span.start < span.end);
    // 只复制有界诊断记录，随后在锁外比较。目标的 VA 估算和页集合已捕获，不能在此回查。
    let snapshot = HISTORY.lock().unwrap_or_else(|e| e.into_inner()).snapshot(span.as_ref());
    let hits = snapshot.compare(span.as_ref(), pages, REBIND_PAIR_LIMIT);
    let target_section_pages = pages.iter().filter(|page| !page.is_private()).count();
    let Stats { maps, unmaps, evicted_live, evicted_unmaps } = snapshot.stats;
    tracing::info!(
        index, slot, subresource, target_section_pages, candidate_span_known = span.is_some(),
        target_live_match_pages = hits.matched_pages,
        target_live_unmatched_section_pages = target_section_pages - hits.matched_pages,
        target_live_match_views = hits.matched_views, target_live_match_sections = hits.matched_sections,
        bind_first = ?hits.bind_first, bind_last = ?hits.bind_last, event_sequence = maps.saturating_add(unmaps),
        maps, unmaps, retained_live = snapshot.retained_live, evicted_live, live_limit = LIVE_LIMIT,
        retained_unmaps = snapshot.unmaps.len(), evicted_unmaps, unmap_limit = UNMAP_LIMIT,
        first_retained_unmap = ?snapshot.unmaps.front().map(|event| event.sequence),
        target_va_unmaps = hits.unmaps_at_va, target_page_unmaps = hits.unmaps_with_page,
        same_page_rebind_unmaps = hits.same_page_rebinds,
        same_section_other_page_rebind_unmaps = hits.same_section_rebinds_with_other_page,
        different_section_rebind_unmaps = hits.different_section_rebinds,
        incomplete_rebind_events = hits.incomplete_rebind_events, rebind_pair_limit = REBIND_PAIR_LIMIT,
        "[VNME_XMEM_SECTION_HISTORY] 颜色目标溯源：XMem 已发布视图的进程内关系序号，非 Fence。目标为整资源估算；VA 与捕获页集合并非同次采样。重绑按已解绑视图计数，类别可重叠，不判定直接前代；同页包含切分恢复。未匹配不排除历史；不证明物理节释放、字节写入或像素有效"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    const VA: usize = 16 * PAGE;

    fn binding(section: PhysicalPage, offset: usize, base: usize, pages: usize) -> Binding {
        Binding::new(section, offset, base, pages * PAGE).unwrap()
    }

    #[test]
    fn old_live_view_survives_unmap_history_rotation() {
        let mut numbers = PageNumbers::default();
        let target = numbers.allocate(8).unwrap();
        let other = numbers.allocate(1).unwrap();
        let mut history = SectionHistory::default();
        history.observe(binding(target, 2, VA, 4), true, 2, 1);
        let noise = binding(other, 0, VA + 4 * PAGE, 1);
        for _ in 0..4 {
            history.observe(noise, true, 2, 1);
            history.observe(noise, false, 2, 1);
        }
        let span = VA + PAGE + 3..VA + 3 * PAGE;
        let pages = [target.offset(2), target.offset(3), target.offset(4), target.offset(5), other].into();
        let snapshot = history.snapshot(Some(&span));
        let hits = snapshot.compare(Some(&span), &pages, 64);
        assert_eq!((hits.matched_pages, hits.matched_views, hits.matched_sections), (2, 1, 1));
        assert_eq!((hits.bind_first, hits.bind_last), (Some(1), Some(1)));
        assert_eq!((hits.unmaps_at_va, hits.unmaps_with_page), (0, 1));
        assert_eq!((snapshot.stats.evicted_live, snapshot.stats.evicted_unmaps), (0, 3));
    }

    #[test]
    fn rebind_distinguishes_restored_slice_other_slice_and_other_section() {
        let mut numbers = PageNumbers::default();
        let original = numbers.allocate(8).unwrap();
        let replacement = numbers.allocate(2).unwrap();
        let mut history = SectionHistory::default();
        let prior = binding(original, 0, VA, 4);
        history.observe(prior, true, 8, 8);
        history.observe(prior, false, 8, 8);
        history.observe(binding(original, 0, VA, 1), true, 8, 8);
        // 中间一页没有重新发布（例如宿主提交失败）。解绑仍然是既成事实。
        history.observe(binding(original, 5, VA + 2 * PAGE, 1), true, 8, 8);
        history.observe(binding(replacement, 1, VA + 3 * PAGE, 1), true, 8, 8);
        let span = VA..VA + 4 * PAGE;
        let pages = [original, original.offset(5), replacement.offset(1)].into();
        let snapshot = history.snapshot(Some(&span));
        let hits = snapshot.compare(Some(&span), &pages, 64);
        assert_eq!((hits.matched_pages, hits.matched_views, hits.matched_sections), (3, 3, 2));
        assert_eq!((hits.bind_first, hits.bind_last), (Some(3), Some(5)));
        assert_eq!((hits.unmaps_at_va, hits.unmaps_with_page), (1, 1));
        assert_eq!((hits.same_page_rebinds, hits.same_section_rebinds_with_other_page, hits.different_section_rebinds), (1, 1, 1));
        let failed = VA + PAGE..VA + 2 * PAGE;
        let hits = history.snapshot(Some(&failed)).compare(Some(&failed), &[original.offset(1)].into(), 64);
        assert_eq!((hits.unmaps_at_va, hits.matched_pages, hits.same_page_rebinds), (1, 0, 0));
    }

    #[test]
    fn target_page_must_match_inside_the_same_va_overlap() {
        let mut numbers = PageNumbers::default();
        let first = numbers.allocate(1).unwrap();
        let replacement = numbers.allocate(2).unwrap();
        let mut history = SectionHistory::default();
        let original = binding(first, 0, VA, 1);
        history.observe(original, true, 4, 4);
        history.observe(original, false, 4, 4);
        history.observe(binding(replacement, 0, VA, 2), true, 4, 4);
        let span = VA..VA + 2 * PAGE;
        let hits = history.snapshot(Some(&span)).compare(Some(&span), &[replacement.offset(1)].into(), 64);
        assert_eq!((hits.unmaps_at_va, hits.matched_pages, hits.different_section_rebinds), (1, 1, 0));
    }

    #[test]
    fn eviction_and_comparison_limit_do_not_invent_rebinds() {
        let mut numbers = PageNumbers::default();
        let first = numbers.allocate(2).unwrap();
        let replacement = numbers.allocate(2).unwrap();
        let mut history = SectionHistory::default();
        let original = binding(first, 0, VA, 2);
        let unrelated = binding(first, 0, VA + 3 * PAGE, 2);
        history.observe(original, true, 1, 1);
        history.observe(unrelated, true, 1, 1);
        history.observe(original, false, 1, 1); // 已淘汰的视图仍可登记真实解绑。
        history.observe(binding(replacement, 0, VA + PAGE, 2), true, 2, 1);
        let span = VA..VA + 2 * PAGE;
        let snapshot = history.snapshot(Some(&span));
        let hits = snapshot.compare(Some(&span), &[replacement].into(), 0);
        assert_eq!((hits.unmaps_at_va, hits.unmaps_with_page, hits.different_section_rebinds), (1, 0, 0));
        assert_eq!((hits.matched_pages, hits.incomplete_rebind_events), (1, 1));
        assert_eq!(snapshot.stats.evicted_live, 1);
        let hits = snapshot.compare(Some(&span), &[replacement.offset(1)].into(), 64);
        assert_eq!((hits.matched_pages, hits.different_section_rebinds), (0, 0));
    }
}
