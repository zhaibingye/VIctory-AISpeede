//! 一次性失败诊断：只观察已匹配的 XMem 节页，不参与颜色内容准入。

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::{
    OnceLock,
    atomic::{AtomicBool, Ordering},
};

use victory_memory::{
    PAGE_BYTES,
    physical::{BackingPins, PhysicalPage},
};

use super::SubmissionBacking;

const PAGE: usize = PAGE_BYTES as usize;
const PAGE_LIMIT: usize = 512;

struct CandidateSpan {
    bytes: Range<usize>,
    aligned: Range<usize>,
}

impl CandidateSpan {
    fn bounded(span: Option<(u64, u64)>, expected_ids: usize) -> Result<Self, &'static str> {
        if expected_ids > PAGE_LIMIT {
            return Err("over-limit");
        }
        let (start, end) = span.ok_or("unknown-span")?;
        if start == 0 || start >= end {
            return Err("invalid-span");
        }
        let start = usize::try_from(start).map_err(|_| "invalid-span")?;
        let end = usize::try_from(end).map_err(|_| "invalid-span")?;
        let aligned_end = end.checked_next_multiple_of(PAGE).ok_or("invalid-span")?;
        let aligned = (start - start % PAGE)..aligned_end;
        if aligned.len() / PAGE > PAGE_LIMIT {
            return Err("over-limit");
        }
        Ok(Self { bytes: start..end, aligned })
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
struct Counts {
    candidate_pages: usize,
    unmapped_pages: usize,
    private_pages: usize,
    unexpected_section_pages: usize,
    missing_expected_page_ids: usize,
    read_failed_pages: usize,
    zero_pages: usize,
    nonzero_pages: usize,
}

impl Counts {
    fn readable_pages(&self) -> usize {
        self.zero_pages + self.nonzero_pages
    }

    fn complete(&self) -> bool {
        self.candidate_pages != 0
            && self.readable_pages() == self.candidate_pages
            && self.missing_expected_page_ids == 0
    }
}

fn summarize(
    span: &CandidateSpan,
    expected_pages: &BTreeSet<PhysicalPage>,
    pins: &BackingPins,
    scratch: &mut [u8],
    mut stage: impl FnMut(usize, PhysicalPage, &mut [u8]) -> bool,
) -> Counts {
    let mut counts = Counts {
        candidate_pages: span.aligned.len() / PAGE,
        ..Counts::default()
    };
    let mut matched_ids = BTreeSet::new();
    for va in span.aligned.clone().step_by(PAGE) {
        let Some(page) = pins.expected(va) else {
            counts.unmapped_pages += 1;
            continue;
        };
        if page.is_private() {
            counts.private_pages += 1;
            continue;
        }
        if !expected_pages.contains(&page) {
            counts.unexpected_section_pages += 1;
            continue;
        }
        matched_ids.insert(page);
        if !stage(va, page, scratch) {
            // 失败的读取可能只填了一部分；绝不检查残留值，也不补零推断。
            counts.read_failed_pages += 1;
            continue;
        }
        let start = span.bytes.start.saturating_sub(va);
        let end = (span.bytes.end - va).min(PAGE);
        if scratch[start..end].iter().any(|&byte| byte != 0) {
            counts.nonzero_pages += 1;
        } else {
            counts.zero_pages += 1;
        }
    }
    // 页身份去重比较；其余计数按 VA 候选页（别名分开计）。不证明逐 VA 映射未变。
    counts.missing_expected_page_ids = expected_pages.len() - matched_ids.len();
    counts
}

fn observe(
    span: Option<(u64, u64)>,
    expected_pages: &BTreeSet<PhysicalPage>,
) -> Result<Counts, &'static str> {
    // 在捕获前同时限制候选 VA 页数和预期身份集合大小。超限不采样前缀。
    let span = CandidateSpan::bounded(span, expected_pages.len())?;
    let mut backing = SubmissionBacking::default();
    if backing.api().is_none() {
        return Err("backing-unavailable");
    }
    let mut scratch = Vec::new();
    scratch.try_reserve_exact(PAGE).map_err(|_| "scratch-unavailable")?;
    scratch.resize(PAGE, 0);
    backing.include(span.aligned.start, span.aligned.end);
    backing.flush();
    // stage_page 只读取已发布、已捕获的节页；若 VA 换绑，复用现有的临时节视图，
    // 不读替代页。不保留捕获和单页缓冲，不访问 GPU 或更新权威、覆盖、版本。
    Ok(summarize(
        &span,
        expected_pages,
        backing.pins(),
        &mut scratch,
        |va, page, scratch| backing.stage_page(va, page, scratch),
    ))
}

pub(crate) fn trace_color_backing_bytes(
    index: usize,
    slot: usize,
    subresource: u32,
    span: Option<(u64, u64)>,
    expected_pages: &BTreeSet<PhysicalPage>,
) {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    static ATTEMPTED: AtomicBool = AtomicBool::new(false);
    if !crate::page_inventory::enabled()
        || !tracing::enabled!(tracing::Level::INFO)
        || !*ENABLED.get_or_init(|| {
            std::env::var_os("VICTORY_VNME_PROBE_COLOR_BYTES").is_some_and(|v| v == "1")
        })
        || ATTEMPTED.swap(true, Ordering::Relaxed)
    {
        return;
    }
    // 进程内只试一次，包括范围未知、超限和不可读，不在后续失败重试。
    let (status, counts) = match observe(span, expected_pages) {
        Ok(counts) if counts.complete() => ("complete", counts),
        Ok(counts) => ("partial", counts),
        Err(status) => (status, Counts::default()),
    };
    tracing::info!(
        index, slot, subresource, status, complete = counts.complete(), page_limit = PAGE_LIMIT,
        candidate_pages = counts.candidate_pages, expected_page_ids = expected_pages.len(),
        unmapped_pages = counts.unmapped_pages, private_pages = counts.private_pages,
        unexpected_section_pages = counts.unexpected_section_pages,
        missing_expected_page_ids = counts.missing_expected_page_ids,
        read_failed_pages = counts.read_failed_pages, readable_pages = counts.readable_pages(),
        unobserved_pages = counts.candidate_pages - counts.readable_pages(),
        zero_pages = counts.zero_pages, nonzero_pages = counts.nonzero_pages,
        "[VNME_COLOR_BACKING_BYTES] 失败时的 XMem CPU 逐页观测：整资源候选区间（仅分类页内相交字节），两次捕获只比较页身份集合。非原子快照，不证明写入来源、有效旧像素或 GPU 权威"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_memory::{physical::PageNumbers, virtual_pages::VirtualPages};

    const VA: usize = 8 * PAGE;

    #[test]
    fn bounds_mapping_work_before_capture_and_allows_510_pages() {
        let span = |start, end| Some((start as u64, end as u64));
        let allowed = CandidateSpan::bounded(span(VA + 1, VA + 510 * PAGE), 510).unwrap();
        assert_eq!(allowed.aligned.len() / PAGE, 510);
        assert!(CandidateSpan::bounded(span(VA, VA + PAGE_LIMIT * PAGE), PAGE_LIMIT).is_ok());
        assert_eq!(
            CandidateSpan::bounded(span(VA, VA + PAGE_LIMIT * PAGE + 1), 1).err(),
            Some("over-limit")
        );
        assert_eq!(
            CandidateSpan::bounded(span(VA, VA + PAGE), PAGE_LIMIT + 1).err(),
            Some("over-limit")
        );
        assert_eq!(CandidateSpan::bounded(None, 0).err(), Some("unknown-span"));
        assert_eq!(CandidateSpan::bounded(span(VA, VA), 0).err(), Some("invalid-span"));
        assert_eq!(
            CandidateSpan::bounded(Some((VA as u64, u64::MAX)), 0).err(),
            Some("invalid-span")
        );
    }

    #[test]
    fn classifies_only_in_span_bytes_and_counts_aliased_va_pages() {
        let first = PageNumbers::default().allocate(2).unwrap();
        let last = first.offset(1);
        let range = ((VA + 1) as u64, (VA + 3 * PAGE - 1) as u64);
        let span = CandidateSpan::bounded(Some(range), 2).unwrap();
        let mut pins = BackingPins::default();
        pins.remember(VA, VA + PAGE, Some(first));
        pins.remember(VA + PAGE, VA + 3 * PAGE, Some(first));
        let mut scratch = vec![0; PAGE];
        let counts = summarize(&span, &[first, last].into(), &pins, &mut scratch, |_, page, bytes| {
            bytes.fill(0);
            bytes[if page == first { 0 } else { PAGE - 1 }] = 1;
            true
        });
        assert!(counts.complete());
        // Only the second VA's range includes its nonzero byte, despite aliasing the first.
        assert_eq!(
            counts,
            Counts { candidate_pages: 3, zero_pages: 2, nonzero_pages: 1, ..Counts::default() }
        );
    }

    #[test]
    fn rejects_unknown_bindings_and_does_not_classify_failed_reads() {
        let first = PageNumbers::default().allocate(4).unwrap();
        let (unreadable, missing, unexpected) = (first.offset(1), first.offset(2), first.offset(3));
        let expected = [first, unreadable, missing].into();
        let span = CandidateSpan::bounded(Some((VA as u64, (VA + 5 * PAGE) as u64)), 3).unwrap();
        let mut pins = BackingPins::default();
        pins.remember(VA, VA + 2 * PAGE, Some(first));
        pins.remember(VA + 3 * PAGE, VA + 4 * PAGE, Some(unexpected));
        let private_va = VA + 4 * PAGE;
        let mut private = VirtualPages::default();
        private.committed(private_va, private_va, PAGE);
        pins.remember(private_va, private_va + PAGE, Some(private.page(private_va).unwrap()));
        let mut scratch = vec![0; PAGE];
        let counts = summarize(&span, &expected, &pins, &mut scratch, |_, page, bytes| {
            assert!(page == first || page == unreadable);
            bytes.fill(1);
            page != unreadable
        });
        assert!(!counts.complete());
        assert_eq!(
            counts,
            Counts {
                candidate_pages: 5,
                unmapped_pages: 1,
                private_pages: 1,
                unexpected_section_pages: 1,
                missing_expected_page_ids: 1,
                read_failed_pages: 1,
                nonzero_pages: 1,
                ..Counts::default()
            }
        );
    }

    #[test]
    fn a_missing_expected_id_prevents_completeness_even_if_all_va_pages_read() {
        let first = PageNumbers::default().allocate(2).unwrap();
        let span = CandidateSpan::bounded(Some((VA as u64, (VA + PAGE) as u64)), 2).unwrap();
        let mut pins = BackingPins::default();
        pins.remember(VA, VA + PAGE, Some(first));
        let mut scratch = vec![0; PAGE];
        let counts = summarize(&span, &[first, first.offset(1)].into(), &pins, &mut scratch, |_, _, bytes| {
            bytes.fill(0);
            true
        });
        assert_eq!(counts.readable_pages(), counts.candidate_pages);
        assert_eq!(counts.missing_expected_page_ids, 1);
        assert!(!counts.complete());
    }
}
