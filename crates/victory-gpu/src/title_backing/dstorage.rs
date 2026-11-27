//! DirectStorage CPU 写入的诊断摘要；不保活后备、不读字节、不参与权威判断。

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{LazyLock, Mutex};

use victory_memory::{PAGE_BYTES, physical::{BackingPins, PhysicalPage}};

use super::{BACKING, BackingPinner};

const PAGE: usize = PAGE_BYTES as usize;
const MAX_PAGES: usize = 32_768;
const MAX_RUN_PAGES: usize = 8_192;
const MAX_VA_PROBE_PAGES: usize = 256;
const MAX_VA_CANDIDATE_PAGES: usize = 32_768;
const MAX_VA_RANGES: usize = 16_384;
static HISTORY: LazyLock<Mutex<WriteHistory>> = LazyLock::new(Default::default);

#[derive(Default)]
struct PageWrites {
    confirmed: bool,
    possible: bool,
    confirmed_whole: bool,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct TargetHits {
    any: usize,
    confirmed: usize,
    possible: usize,
    confirmed_whole: usize,
}

#[derive(Default)]
struct WriteHistory {
    pages: BTreeMap<PhysicalPage, (u64, PageWrites)>,
    by_recency: BTreeSet<(u64, PhysicalPage)>,
    confirmed_runs: u64,
    possible_runs: u64,
    skipped_invalid_runs: u64,
    skipped_page_events: usize,
    evicted_page_entries: u64,
    va: VaWriteHistory,
}

impl WriteHistory {
    fn observe_run(
        &mut self, first: PhysicalPage, offset: usize, len: usize, confirmed: bool,
        capacity: usize, run_limit: usize,
    ) {
        if len == 0 { return; }
        let Some(end) = offset.checked_add(len).filter(|_| offset < PAGE && !first.is_private()) else {
            self.skipped_invalid_runs = self.skipped_invalid_runs.saturating_add(1);
            return;
        };
        let runs = if confirmed { &mut self.confirmed_runs } else { &mut self.possible_runs };
        *runs = runs.saturating_add(1);
        let now = self.confirmed_runs.saturating_add(self.possible_runs);
        let pages = end.div_ceil(PAGE);
        let retained = pages.min(run_limit);
        self.skipped_page_events = self.skipped_page_events.saturating_add(pages - retained);
        for i in 0..retained {
            let page = first.offset(i);
            let (last, writes) = self.pages.entry(page).or_default();
            self.by_recency.remove(&(*last, page));
            *last = now;
            self.by_recency.insert((now, page));
            if confirmed {
                writes.confirmed = true;
                let start = i * PAGE;
                // 必须由这一次确认写覆盖整页；不能拼接多次部分写。
                writes.confirmed_whole |= offset <= start && end - start >= PAGE;
            } else {
                writes.possible = true;
            }
            if self.pages.len() > capacity {
                let (_, evicted) = self.by_recency.pop_first().expect("indexed DirectStorage page");
                self.pages.remove(&evicted);
                self.evicted_page_entries = self.evicted_page_entries.saturating_add(1);
            }
        }
    }

    fn compare(&self, target_pages: &BTreeSet<PhysicalPage>) -> TargetHits {
        let mut hits = TargetHits::default();
        for page in target_pages {
            if let Some((_, writes)) = self.pages.get(page) {
                hits.any += 1;
                hits.confirmed += usize::from(writes.confirmed);
                hits.possible += usize::from(writes.possible);
                hits.confirmed_whole += usize::from(writes.confirmed_whole);
            }
        }
        hits
    }
}

/// 记录单一连续 XMem section run 的覆盖。`first` 为写侧实际保活的首个物理页；
/// `offset` 为首个页内偏移。调用者保证 `len` 在此 run 内。成功只报实际写入的前缀；
/// `confirmed = false` 只表示失败时可能修改此范围。
pub fn note_dstorage_section_write(first: PhysicalPage, offset: usize, len: usize, confirmed: bool) {
    if crate::page_inventory::enabled() {
        // 诊断锁内只处理页身份和摘要，不进入映射、资源或日志代码。
        HISTORY.lock().unwrap_or_else(|e| e.into_inner())
            .observe_run(first, offset, len, confirmed, MAX_PAGES, MAX_RUN_PAGES);
    }
}

#[derive(Clone, Copy)]
struct VaSpan {
    first: usize,
    pages: usize,
}

impl VaSpan {
    fn new(start: usize, len: usize) -> Option<Self> {
        let end = start.checked_add(len)?.checked_next_multiple_of(PAGE)?;
        let first = start - start % PAGE;
        Some(Self { first, pages: (end - first) / PAGE })
    }

    fn sample(self, pin: Option<BackingPinner>) -> Vec<Option<PhysicalPage>> {
        let Some(pin) = pin else { return Vec::new(); };
        let end = self.first + self.pages.min(MAX_VA_PROBE_PAGES) * PAGE;
        // 仅借用现有映射捕获入口。所有临时 lease 都在采样返回前释放；
        // 调用者持有的 handle 只有身份和范围，不保活 section 或 title 映射。
        let mut pins = BackingPins::default();
        pin(&[(self.first, end)], &mut pins);
        (self.first..end).step_by(PAGE).map(|at| pins.expected(at)).collect()
    }
}

/// 一次裸 VA fill 的诊断 handle；不持锁或保活后备。
#[must_use = "finish the probe immediately after the raw VA fill"]
pub struct DstorageVaWrite {
    start: usize,
    requested: usize,
    before: Vec<Option<PhysicalPage>>,
}

fn current_pinner() -> Option<BackingPinner> {
    BACKING.get().map(|&(pin, _, _)| pin)
}

/// fill 前捕获 XMem 页身份候选；只采样区间前缀，不读取目标字节。
/// 两次观察即使相同也不证明 fill 期间的映射稳定（可能有中途重映射）。
pub fn probe_dstorage_va_write(start: usize, len: usize) -> Option<DstorageVaWrite> {
    (len != 0 && crate::page_inventory::enabled())
        .then(|| DstorageVaWrite::begin(start, len, current_pinner()))
}

impl DstorageVaWrite {
    fn begin(start: usize, requested: usize, pin: Option<BackingPinner>) -> Self {
        let before = VaSpan::new(start, requested).map_or_else(Vec::new, |span| span.sample(pin));
        Self { start, requested, before }
    }

    /// fill 后立即采样。`Some(n)` 仅限定日志候选的范围，`None` 表示失败且
    /// 可能改写整个请求范围；都不登记成直接物理写入的确认或可能页。
    pub fn finish(self, produced: Option<usize>) {
        if let Some(sample) = self.complete(produced, current_pinner()) {
            HISTORY.lock().unwrap_or_else(|e| e.into_inner()).va.observe(sample, MAX_VA_CANDIDATE_PAGES);
        }
    }

    fn complete(self, produced: Option<usize>, pin: Option<BackingPinner>) -> Option<VaObservation> {
        let len = produced.unwrap_or(self.requested);
        if len == 0 { return None; }
        let span = (len <= self.requested).then(|| VaSpan::new(self.start, len)).flatten();
        Some(VaObservation {
            before: self.before,
            after: span.map_or_else(Vec::new, |span| span.sample(pin)),
            range: span.map(|span| (self.start..self.start + len, span.pages)),
            failed: produced.is_none(),
        })
    }
}

struct VaObservation {
    before: Vec<Option<PhysicalPage>>,
    after: Vec<Option<PhysicalPage>>,
    range: Option<(std::ops::Range<usize>, usize)>,
    failed: bool,
}

#[derive(Clone, Copy, Default)]
struct VaStats {
    writes: u64,
    failed_writes: u64,
    unattributed_writes: u64,
    incomplete_writes: u64,
    invalid_writes: u64,
    over_limit_writes: u64,
    skipped_page_events: usize,
    unknown_endpoint_page_events: usize,
    different_identity_page_events: usize,
    evicted_page_entries: u64,
    evicted_ranges: u64,
}

#[derive(Default)]
struct VaWriteHistory {
    candidates: BTreeMap<PhysicalPage, u64>,
    by_recency: BTreeSet<(u64, PhysicalPage)>,
    ranges: VecDeque<(std::ops::Range<usize>, bool)>,
    stats: VaStats,
}

impl VaWriteHistory {
    fn observe(&mut self, sample: VaObservation, capacity: usize) {
        self.stats.writes = self.stats.writes.saturating_add(1);
        self.stats.failed_writes = self.stats.failed_writes.saturating_add(u64::from(sample.failed));
        let Some((range, pages)) = sample.range else {
            self.stats.invalid_writes = self.stats.invalid_writes.saturating_add(1);
            self.stats.incomplete_writes = self.stats.incomplete_writes.saturating_add(1);
            self.stats.unattributed_writes = self.stats.unattributed_writes.saturating_add(1);
            return;
        };
        if self.ranges.len() == MAX_VA_RANGES {
            self.ranges.pop_front();
            self.stats.evicted_ranges = self.stats.evicted_ranges.saturating_add(1);
        }
        self.ranges.push_back((range, sample.failed));
        let sampled = pages.min(MAX_VA_PROBE_PAGES);
        let limited = sampled != pages;
        self.stats.over_limit_writes = self.stats.over_limit_writes.saturating_add(u64::from(limited));
        self.stats.skipped_page_events = self.stats.skipped_page_events.saturating_add(pages - sampled);
        let mut incomplete = limited;
        let mut attributed = false;
        for i in 0..sampled {
            // 只用实际写入范围内的前缀；缺页、未安装 pinner 等均不推断身份。
            let before = sample.before.get(i).copied().flatten();
            let after = sample.after.get(i).copied().flatten();
            if before.is_none() || after.is_none() {
                incomplete = true;
                self.stats.unknown_endpoint_page_events = self.stats.unknown_endpoint_page_events.saturating_add(1);
            } else if before != after {
                self.stats.different_identity_page_events = self.stats.different_identity_page_events.saturating_add(1);
            }
            for page in before.into_iter().chain(after.filter(|&page| Some(page) != before)) {
                if !page.is_private() {
                    attributed = true;
                    self.retain_candidate(page, capacity);
                }
            }
        }
        self.stats.incomplete_writes = self.stats.incomplete_writes.saturating_add(u64::from(incomplete));
        self.stats.unattributed_writes = self.stats.unattributed_writes.saturating_add(u64::from(!attributed));
    }

    fn retain_candidate(&mut self, page: PhysicalPage, capacity: usize) {
        let now = self.stats.writes;
        if let Some(last) = self.candidates.insert(page, now) {
            self.by_recency.remove(&(last, page));
        }
        self.by_recency.insert((now, page));
        if self.candidates.len() > capacity {
            let (_, evicted) = self.by_recency.pop_first().expect("indexed VA candidate page");
            self.candidates.remove(&evicted);
            self.stats.evicted_page_entries = self.stats.evicted_page_entries.saturating_add(1);
        }
    }

    fn compare(&self, target_pages: &BTreeSet<PhysicalPage>) -> usize {
        target_pages.iter().filter(|&page| self.candidates.contains_key(page)).count()
    }

    fn compare_ranges(&self, target: (usize, usize)) -> (usize, usize) {
        let (start, end) = target;
        self.ranges.iter().filter(|(range, _)| range.start < end && start < range.end)
            .fold((0, 0), |(success, failure), (_, failed)| {
                (success + usize::from(!*failed), failure + usize::from(*failed))
            })
    }
}

pub(crate) fn trace_dstorage_write_history(
    index: usize, slot: usize, subresource: u32, target_pages: &BTreeSet<PhysicalPage>,
    candidate_span: Option<(u64, u64)>,
) {
    // 调用者已有原生历史的诊断捕获；不复查目标映射，也不回查历史 VA。
    let (hits, retained, confirmed_runs, possible_runs, invalid, skipped, evicted, va_hits, va_retained, va_overlaps, va) = {
        let history = HISTORY.lock().unwrap_or_else(|e| e.into_inner());
        (
            history.compare(target_pages), history.pages.len(), history.confirmed_runs, history.possible_runs,
            history.skipped_invalid_runs, history.skipped_page_events, history.evicted_page_entries,
            history.va.compare(target_pages), history.va.candidates.len(),
            candidate_span.map(|(start, end)| history.va.compare_ranges((start as usize, end as usize))),
            history.va.stats,
        )
    };
    tracing::info!(
        index, slot, subresource,
        target_mapped_page_ids = target_pages.len(), target_matched_page_ids = hits.any,
        target_unmatched_mapped_page_ids = target_pages.len() - hits.any,
        matched_confirmed_pages = hits.confirmed, matched_possible_pages = hits.possible,
        matched_pages_with_confirmed_whole_write = hits.confirmed_whole,
        retained_page_ids = retained, confirmed_section_runs = confirmed_runs, possible_section_runs = possible_runs,
        skipped_invalid_section_runs = invalid, skipped_page_events = skipped, evicted_page_entries = evicted,
        retained_page_limit = MAX_PAGES, per_run_page_event_limit = MAX_RUN_PAGES,
        "[VNME_DSTORAGE_WRITE_HISTORY] 颜色目标溯源：DirectStorage 直接物理写入候选（确认和可能页可重叠；整页仅指单次确认写覆盖页字节；目标为整资源估算，不证明提交顺序、目标字节相交、像素覆盖或当前内容；未匹配不排除来源）"
    );
    tracing::info!(
        index, slot, subresource,
        target_mapped_page_ids = target_pages.len(),
        target_matched_va_candidate_section_pages = va_hits,
        retained_va_candidate_section_pages = va_retained,
        va_writes = va.writes, failed_va_writes = va.failed_writes,
        unattributed_va_writes = va.unattributed_writes, incomplete_va_writes = va.incomplete_writes,
        invalid_va_writes = va.invalid_writes, va_writes_over_probe_limit = va.over_limit_writes,
        skipped_va_probe_page_events = va.skipped_page_events,
        va_probe_page_events_with_unknown_endpoint = va.unknown_endpoint_page_events,
        va_probe_page_events_with_different_identities = va.different_identity_page_events,
        evicted_va_candidate_page_entries = va.evicted_page_entries,
        va_only_overlap_success_failed = ?va_overlaps, evicted_va_ranges = va.evicted_ranges, va_range_limit = MAX_VA_RANGES,
        va_probe_page_limit = MAX_VA_PROBE_PAGES, va_candidate_page_limit = MAX_VA_CANDIDATE_PAGES,
        "[VNME_DSTORAGE_VA_WRITE_HISTORY] 颜色目标溯源：裸 VA fill 前后 XMem section 页身份候选（只取受限前缀；未归属表示两端均未见 XMem 页；同身份不证明全程稳定或实际写到该页；未匹配不排除来源）"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_memory::physical::PageNumbers;

    #[test]
    fn dstorage_history_matches_captured_identity_and_exact_prefix() {
        let mut numbers = PageNumbers::default();
        let first = numbers.allocate(4).unwrap();
        let replacement = numbers.allocate(1).unwrap();
        let mut history = WriteHistory::default();
        history.observe_run(first, PAGE - 2, PAGE + 4, true, 8, 8);
        assert_eq!(history.compare(&[first, first.offset(1), first.offset(2), first.offset(3)].into()),
            TargetHits { any: 3, confirmed: 3, possible: 0, confirmed_whole: 1 });
        assert_eq!(history.compare(&[first.offset(1)].into()).confirmed_whole, 1);

        let captured = |page| {
            let mut pins = BackingPins::default();
            pins.remember(PAGE, 2 * PAGE, Some(page));
            BTreeSet::from([pins.expected(PAGE).unwrap()])
        };
        assert_eq!(history.compare(&captured(first)).confirmed, 1);
        assert_eq!(history.compare(&captured(replacement)), TargetHits::default());
    }

    #[test]
    fn dstorage_history_does_not_promote_failures_or_combine_partial_writes() {
        let first = PageNumbers::default().allocate(2).unwrap();
        let next = first.offset(1);
        let mut history = WriteHistory::default();
        history.observe_run(first, 0, PAGE / 2, true, 8, 8);
        history.observe_run(first, PAGE / 2, PAGE / 2, true, 8, 8);
        history.observe_run(first, 0, 2 * PAGE, false, 8, 8);
        assert_eq!(history.compare(&[first, next].into()),
            TargetHits { any: 2, confirmed: 1, possible: 2, confirmed_whole: 0 });
        assert_eq!(history.compare(&[next].into()),
            TargetHits { any: 1, confirmed: 0, possible: 1, confirmed_whole: 0 });
    }

    #[test]
    fn dstorage_history_reports_skips_and_evictions() {
        let first = PageNumbers::default().allocate(3).unwrap();
        let mut history = WriteHistory::default();
        history.observe_run(first, 0, 3 * PAGE, true, 2, 2);
        history.observe_run(first, 0, 1, false, 2, 2); // 刷新首个页的最近记录。
        history.observe_run(first.offset(2), 0, PAGE, true, 2, 2);
        history.observe_run(first, 1, usize::MAX, true, 2, 2);
        assert_eq!(history.pages.len(), 2);
        assert_eq!(history.confirmed_runs, 2);
        assert_eq!(history.possible_runs, 1);
        assert_eq!(history.skipped_page_events, 1);
        assert_eq!(history.skipped_invalid_runs, 1);
        assert_eq!(history.evicted_page_entries, 1);
        assert_eq!(history.compare(&[first.offset(1)].into()), TargetHits::default());
        assert_eq!(history.compare(&[first, first.offset(2)].into()),
            TargetHits { any: 2, confirmed: 2, possible: 1, confirmed_whole: 2 });
    }

    const VA: usize = 16 * PAGE;

    fn synthetic_mapping<const OFFSET: usize>(ranges: &[(usize, usize)], pins: &mut BackingPins) {
        let first = PageNumbers::default().allocate(MAX_VA_PROBE_PAGES + OFFSET).unwrap();
        let [(start, end)] = ranges else { panic!("one bounded prefix per sample") };
        assert_eq!(*start, VA);
        assert!(*end <= VA + MAX_VA_PROBE_PAGES * PAGE);
        pins.remember(*start, *end, Some(first.offset(OFFSET)));
    }

    #[test]
    fn raw_va_remap_yields_both_endpoints_without_physical_write_confirmation() {
        let first = PageNumbers::default().allocate(2).unwrap();
        let replacement = first.offset(1);
        let mut history = WriteHistory::default();
        let handle = DstorageVaWrite::begin(VA, PAGE, Some(synthetic_mapping::<0>));
        let sample = handle.complete(Some(PAGE), Some(synthetic_mapping::<1>)).unwrap();
        history.va.observe(sample, MAX_VA_CANDIDATE_PAGES);

        assert_eq!(history.va.compare(&[first, replacement].into()), 2);
        assert_eq!(history.va.stats.different_identity_page_events, 1);
        assert_eq!(history.va.stats.incomplete_writes, 0);
        assert_eq!(history.compare(&[first, replacement].into()), TargetHits::default());
        assert_eq!((history.confirmed_runs, history.possible_runs), (0, 0));
    }

    #[test]
    fn raw_va_short_success_uses_only_produced_prefix_even_with_equal_endpoints() {
        let first = PageNumbers::default().allocate(3).unwrap();
        let mut history = WriteHistory::default();
        let handle = DstorageVaWrite::begin(VA + PAGE - 2, 2 * PAGE, Some(synthetic_mapping::<0>));
        assert_eq!(handle.before.len(), 3);
        let sample = handle.complete(Some(2), Some(synthetic_mapping::<0>)).unwrap();
        assert_eq!(sample.after, [Some(first)]);
        history.va.observe(sample, MAX_VA_CANDIDATE_PAGES);

        assert_eq!(history.va.compare(&[first].into()), 1);
        assert_eq!(history.va.compare(&[first.offset(1), first.offset(2)].into()), 0);
        assert_eq!(history.va.compare_ranges((VA + PAGE - 2, VA + PAGE)), (1, 0));
        assert_eq!(history.va.compare_ranges((VA + PAGE, VA + 2 * PAGE)), (0, 0));
        assert_eq!(history.va.stats.different_identity_page_events, 0);
        assert_eq!(history.compare(&[first].into()), TargetHits::default());
    }

    #[test]
    fn raw_va_failure_reports_unobserved_prefix_limits_and_candidate_eviction() {
        let first = PageNumbers::default().allocate(MAX_VA_PROBE_PAGES).unwrap();
        let mut history = VaWriteHistory::default();
        let handle = DstorageVaWrite::begin(VA, (MAX_VA_PROBE_PAGES + 1) * PAGE, None);
        let sample = handle.complete(None, Some(synthetic_mapping::<0>)).unwrap();
        assert_eq!(sample.after.len(), MAX_VA_PROBE_PAGES);
        history.observe(sample, 2);

        assert_eq!(history.stats.writes, 1);
        assert_eq!(history.stats.failed_writes, 1);
        assert_eq!(history.stats.incomplete_writes, 1);
        assert_eq!(history.stats.over_limit_writes, 1);
        assert_eq!(history.stats.skipped_page_events, 1);
        assert_eq!(history.stats.unknown_endpoint_page_events, MAX_VA_PROBE_PAGES);
        assert_eq!(history.stats.evicted_page_entries, (MAX_VA_PROBE_PAGES - 2) as u64);
        assert_eq!(history.stats.unattributed_writes, 0);
        assert_eq!(history.compare(&[first, first.offset(MAX_VA_PROBE_PAGES - 1)].into()), 1);

        let unmapped = DstorageVaWrite::begin(VA, PAGE, None).complete(Some(1), None).unwrap();
        history.observe(unmapped, 2);
        assert_eq!(history.stats.unattributed_writes, 1);
        assert_eq!(history.stats.incomplete_writes, 2);
    }

    #[test]
    fn raw_va_zero_and_invalid_results_do_not_retain_before_candidates() {
        let mut history = VaWriteHistory::default();
        let begin = || DstorageVaWrite::begin(VA, 1, Some(synthetic_mapping::<0>));
        assert!(begin().complete(Some(0), None).is_none());
        history.observe(begin().complete(Some(2), None).unwrap(), 2);
        history.observe(DstorageVaWrite::begin(usize::MAX, 2, None).complete(None, None).unwrap(), 2);
        assert_eq!(history.stats.invalid_writes, 2);
        assert_eq!(history.stats.incomplete_writes, 2);
        assert_eq!(history.stats.unattributed_writes, 2);
        assert!(history.candidates.is_empty());
    }
}
