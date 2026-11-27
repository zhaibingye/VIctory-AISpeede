//! DirectStorage CPU 写入的诊断摘要；不保活后备、不读字节、不参与权威判断。

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{LazyLock, Mutex};

use victory_memory::{PAGE_BYTES, physical::PhysicalPage};

const PAGE: usize = PAGE_BYTES as usize;
const MAX_PAGES: usize = 32_768;
const MAX_RUN_PAGES: usize = 8_192;
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
    unattributed_va_writes: u64,
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

    fn observe_va(&mut self) {
        self.unattributed_va_writes = self.unattributed_va_writes.saturating_add(1);
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

/// 计一次 VA 写入路径；不推断物理页或成功写入范围。
pub fn note_dstorage_va_write() {
    if crate::page_inventory::enabled() {
        HISTORY.lock().unwrap_or_else(|e| e.into_inner()).observe_va();
    }
}

pub(crate) fn trace_dstorage_write_history(
    index: usize, slot: usize, subresource: u32, target_pages: &BTreeSet<PhysicalPage>,
) {
    // 调用者已有原生历史的诊断捕获；不复查目标映射。
    let (hits, retained, confirmed_runs, possible_runs, invalid, skipped, evicted, va) = {
        let history = HISTORY.lock().unwrap_or_else(|e| e.into_inner());
        (
            history.compare(target_pages), history.pages.len(), history.confirmed_runs, history.possible_runs,
            history.skipped_invalid_runs, history.skipped_page_events, history.evicted_page_entries,
            history.unattributed_va_writes,
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
        unattributed_va_writes = va, retained_page_limit = MAX_PAGES, per_run_page_event_limit = MAX_RUN_PAGES,
        "[VNME_DSTORAGE_WRITE_HISTORY] 颜色目标溯源：DirectStorage CPU 写入候选（确认和可能页可重叠；整页仅指单次确认写覆盖页字节；目标为整资源估算，不证明提交顺序、目标字节相交、像素覆盖或当前内容；未匹配不排除来源）"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_memory::physical::{BackingPins, PageNumbers};

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
    fn dstorage_history_reports_skips_evictions_and_unattributed_writes() {
        let first = PageNumbers::default().allocate(3).unwrap();
        let mut history = WriteHistory::default();
        history.observe_run(first, 0, 3 * PAGE, true, 2, 2);
        history.observe_run(first, 0, 1, false, 2, 2); // 刷新首个页的最近记录。
        history.observe_run(first.offset(2), 0, PAGE, true, 2, 2);
        history.observe_run(first, 1, usize::MAX, true, 2, 2);
        history.observe_va();
        assert_eq!(history.pages.len(), 2);
        assert_eq!(history.confirmed_runs, 2);
        assert_eq!(history.possible_runs, 1);
        assert_eq!(history.skipped_page_events, 1);
        assert_eq!(history.skipped_invalid_runs, 1);
        assert_eq!(history.evicted_page_entries, 1);
        assert_eq!(history.unattributed_va_writes, 1);
        assert_eq!(history.compare(&[first.offset(1)].into()), TargetHits::default());
        assert_eq!(history.compare(&[first, first.offset(2)].into()),
            TargetHits { any: 2, confirmed: 2, possible: 1, confirmed_whole: 2 });
    }
}
