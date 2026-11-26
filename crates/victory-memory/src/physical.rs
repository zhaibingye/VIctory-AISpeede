//! Physical allocations and alias identity, independent of host address.

use std::any::Any;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use crate::{
    PAGE_BYTES,
    budget::{Budget, BudgetProfile, Charge, Headroom, HostLimits, HostPressure, NoSpace, Pool, PoolLimits},
};

/// Thread-safe ledger for the host's managed physical allocations. XMem and
/// graphics use the same process ledger; isolated fixtures can own another.
#[derive(Clone, Default)]
pub struct HostBudget(Arc<Mutex<Budget>>);

impl HostBudget {
    pub fn process() -> &'static Self {
        static HOST: std::sync::LazyLock<HostBudget> =
            std::sync::LazyLock::new(HostBudget::default);
        &HOST
    }

    pub fn reserve_ram(&self, bytes: u64) -> Result<Reservation, NoSpace> {
        let charge = self.lock().charge(Pool::Ram, bytes)?;
        Ok(self.reservation(charge))
    }

    /// DXGI reports our process's total local use, including managed
    /// resources, so count excess over the ledger rather than adding both.
    pub fn reserve_local(
        &self, bytes: u64, observed: u64, dxgi_budget: u64, headroom: u64,
    ) -> Result<Reservation, NoSpace> {
        let mut budget = self.lock();
        let outside = observed.saturating_sub(budget.used(Pool::Local));
        budget.observe_local(dxgi_budget, outside.saturating_add(headroom));
        let charge = budget.charge(Pool::Local, bytes)?;
        Ok(self.reservation(charge))
    }

    /// 在分配锁内读取宿主压力并发布两个池的上限。不完整的采样不改旧值；
    /// 采样回调只能读 OS 数据，不得再次访问本账本，也不能获取或释放租约。
    pub fn refresh_pressure(
        &self, reserve: Headroom, profile: BudgetProfile,
        sample: impl FnOnce() -> Option<HostPressure>,
    ) -> Option<HostLimits> {
        let mut budget = self.lock();
        let mut sample = sample()?;
        sample.managed_ram = budget.used(Pool::Ram);
        sample.managed_local = budget.used(Pool::Local);
        let limits = HostLimits::plan(sample, reserve, profile);
        budget.apply_host_limits(limits);
        Some(limits)
    }

    /// 在同一时刻读取实际入账容量和软硬线；超过上限的既有资源仍须真实回收。
    pub fn pool_state(&self, pool: Pool) -> (u64, PoolLimits) {
        let budget = self.lock();
        (budget.used(pool), budget.limits(pool))
    }

    pub fn used(&self, pool: Pool) -> u64 {
        self.lock().used(pool)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Budget> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn reservation(&self, charge: Charge) -> Reservation {
        Reservation {
            budget: self.clone(),
            charge: Some(charge),
        }
    }
}

/// Retain this lease with the host allocation, not with individual views.
pub struct Reservation {
    budget: HostBudget,
    charge: Option<Charge>,
}

impl Reservation {
    pub fn resize(&mut self, bytes: u64) -> Result<(), NoSpace> {
        self.budget.lock().resize(self.charge.as_mut().expect("one charge per reservation"), bytes)
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.budget.lock().release(self.charge.take().expect("one charge per reservation"));
    }
}

/// 给惰性 Windows 物理节记账。键是物理节内的页号，不是某一个映射地址。
/// 已提交的页直到物理节销毁才释放额度；取消某个别名不改变账本。
pub struct DeferredRam {
    lease: Reservation,
    pages: Box<[bool]>,
    count: u64,
}

impl DeferredRam {
    pub fn new(budget: &HostBudget, pages: usize) -> Result<Self, NoSpace> {
        Ok(Self { lease: budget.reserve_ram(0)?, pages: vec![false; pages].into_boxed_slice(), count: 0 })
    }

    /// 先预扣新页并执行真实宿主提交。拒绝不调用宿主，提交失败撤回预扣。
    /// 调用方串行化物理节内各次提交，且宿主提交在失败时不新增页。
    pub fn commit(&mut self, start: usize, count: usize, apply: impl FnOnce() -> bool)
        -> Result<bool, NoSpace>
    {
        let Some(pages) = start.checked_add(count).and_then(|end| self.pages.get_mut(start..end)) else {
            return Ok(false);
        };
        let fresh = pages.iter().filter(|&&committed| !committed).count() as u64;
        if fresh == 0 { return Ok(true); }
        let next = self.count + fresh;
        self.lease.resize(next * PAGE_BYTES)?;
        if !apply() {
            self.lease.resize(self.count * PAGE_BYTES).expect("回退仅退还 RAM 额度");
            return Ok(false);
        }
        pages.fill(true);
        self.count = next;
        Ok(true)
    }
}

#[test]
fn deferred_section_counts_unique_committed_pages_and_undoes_failure() {
    let budget = HostBudget::default();
    let mut backing = DeferredRam::new(&budget, 8).unwrap();
    assert_eq!(budget.used(Pool::Ram), 0);
    assert!(backing.commit(1, 3, || true).unwrap());
    assert_eq!(budget.used(Pool::Ram), 3 * PAGE_BYTES);
    assert!(backing.commit(2, 2, || panic!("已提交的别名无需重新提交")).unwrap());
    assert!(!backing.commit(2, 3, || false).unwrap());
    assert_eq!(budget.used(Pool::Ram), 3 * PAGE_BYTES);
    assert!(backing.commit(2, 3, || true).unwrap());
    assert_eq!(budget.used(Pool::Ram), 4 * PAGE_BYTES);
    budget.lock().apply_host_limits(HostLimits {
        ram: PoolLimits { soft: 3 * PAGE_BYTES, hard: 4 * PAGE_BYTES },
        local: PoolLimits { soft: 0, hard: 0 },
    });
    assert!(backing.commit(1, 4, || panic!("既有页可在限额下降后继续映射")).unwrap());
    assert!(backing.commit(5, 1, || panic!("超限不能调用宿主")).is_err());
    assert!(!backing.commit(7, 2, || panic!("非法区间不能调用宿主")).unwrap());
    drop(backing);
    assert_eq!(budget.used(Pool::Ram), 0);
}

/// Stable backing identity. XMem numbers below the reserved private bit
/// keep their title ABI encoding (21-bit index of 64 KiB pages, expressed
/// in 4 KiB frames); private mappings use a disjoint internal namespace.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PhysicalPage(usize);

const FRAMES: usize = PAGE_BYTES as usize / 4096;
const FIRST: usize = 0x10_0000;
const END: usize = (1 << 21) * FRAMES;

impl PhysicalPage {
    const PRIVATE: usize = 1usize << (usize::BITS - 1);

    pub(crate) fn private(generation: usize) -> Option<Self> {
        (generation < Self::PRIVATE / FRAMES).then(|| Self(Self::PRIVATE | generation * FRAMES))
    }

    pub fn is_private(self) -> bool {
        self.0 & Self::PRIVATE != 0
    }

    pub fn number(self) -> usize {
        self.0
    }

    pub fn offset_from(self, section: Self) -> Option<usize> {
        if self.is_private() != section.is_private() { return None; }
        self.0.checked_sub(section.0).map(|distance| distance / FRAMES)
    }

    /// Caller keeps the index within the allocated extent.
    pub fn offset(self, index: usize) -> Self {
        Self(self.0 + index * FRAMES)
    }
}

pub struct PageNumbers {
    next: usize,
}

impl Default for PageNumbers {
    fn default() -> Self {
        Self { next: FIRST }
    }
}

impl PageNumbers {
    /// 一个物理节最多能使用的来宾页号；预留元数据前可先拒绝不可能的规模。
    pub const MAX_SECTION_PAGES: usize = (END - FIRST) / FRAMES;

    pub fn allocate(&mut self, count: usize) -> Option<PhysicalPage> {
        let span = count.checked_mul(FRAMES)?;
        let end = self.next.checked_add(span)?;
        if count == 0 || end > END {
            return None;
        }
        let first = self.next;
        self.next = end;
        Some(PhysicalPage(first))
    }
}

/// Reverse mapping for the OS-backed views. Only successful host maps are
/// recorded here; splitting a view does not create a new physical identity.
#[derive(Default)]
pub struct Aliases(HashMap<PhysicalPage, Vec<usize>>);

impl Aliases {
    pub fn addresses(&self, page: PhysicalPage) -> &[usize] {
        self.0.get(&page).map_or(&[], Vec::as_slice)
    }

    pub fn mapped(&mut self, page: PhysicalPage, address: usize) {
        self.0.entry(page).or_default().push(address);
    }

    pub fn unmapped(&mut self, page: PhysicalPage, address: usize) {
        if let std::collections::hash_map::Entry::Occupied(mut entry) = self.0.entry(page) {
            entry.get_mut().retain(|&a| a != address);
            if entry.get().is_empty() {
                entry.remove();
            }
        }
    }
}

/// Retains real host backing across a GPU submission. Aliases and repeated
/// ranges can name one allocation; there is only one retained reference to it.
#[derive(Default)]
pub struct BackingPins {
    sections: BTreeMap<PhysicalPage, Arc<dyn Any + Send + Sync>>,
    // Aligned half-open VA runs from the first observation in this
    // submission. None describes a run without XMem backing.
    mappings: BTreeMap<usize, (usize, Option<PhysicalPage>)>,
}

impl BackingPins {
    /// `section` identifies the first physical page of this allocation.
    pub fn keep<T: Send + Sync + 'static>(&mut self, section: PhysicalPage, backing: &Arc<T>) {
        self.sections.entry(section).or_insert_with(|| backing.clone());
    }

    /// Recover the pinned section for an identity previously captured by
    /// this submission. The caller checks the offset against the extent.
    pub fn section<T: Any + Send + Sync>(&self, page: PhysicalPage) -> Option<(Arc<T>, usize)> {
        if page.is_private() { return None; }
        let (&first, section) = self.sections.range(..=page).next_back()?;
        Some((section.clone().downcast::<T>().ok()?, page.offset_from(first)?))
    }

    /// Capture only previously unseen addresses. If a queue chunk sees a
    /// remap, it must not retarget pending writebacks from an earlier chunk.
    pub fn remember(&mut self, start: usize, end: usize, first: Option<PhysicalPage>) {
        const PAGE: usize = PAGE_BYTES as usize;
        debug_assert!(start.is_multiple_of(PAGE) && end.is_multiple_of(PAGE));
        let mut at = start;
        while at < end {
            if let Some((_, &(previous_end, _))) = self.mappings.range(..=at).next_back()
                && at < previous_end
            {
                at = previous_end;
                continue;
            }
            let to = self.mappings.range(at..).next()
                .map_or(end, |(&next, _)| end.min(next));
            let page = first.map(|p| p.offset((at - start) / PAGE));
            self.mappings.insert(at, (to, page));
            at = to;
        }
    }

    /// End of the current run (clipped to `end`) and the first page at
    /// `address`. Holes are also returned so a caller can make progress.
    pub fn run_at(&self, address: usize, end: usize) -> (usize, Option<PhysicalPage>) {
        if let Some((&start, &(limit, first))) = self.mappings.range(..=address).next_back()
            && address < limit
        {
            return (limit.min(end), first.map(|p| p.offset((address - start) / PAGE_BYTES as usize)));
        }
        let limit = self.mappings.range(address..).next().map_or(end, |(&next, _)| end.min(next));
        (limit, None)
    }

    /// 以固定的物理页覆盖半开区间。任何空洞都会使证明失败；不退回虚拟地址。
    /// 返回连续物理区间的起点和页数，不分配每页一个条目。
    pub fn physical_runs(&self, start: usize, end: usize) -> Option<Vec<(PhysicalPage, usize)>> {
        const PAGE: usize = PAGE_BYTES as usize;
        if start >= end { return None; }
        let mut at = start & !(PAGE - 1);
        let end = end.checked_next_multiple_of(PAGE)?;
        let mut runs = Vec::<(PhysicalPage, usize)>::new();
        while at < end {
            let (to, first) = self.run_at(at, end);
            let first = first?;
            let count = (to - at) / PAGE;
            if count == 0 { return None; }
            if let Some((previous, len)) = runs.last_mut()
                && first.offset_from(*previous) == Some(*len)
            {
                *len += count;
            } else {
                runs.push((first, count));
            }
            at = to;
        }
        Some(runs)
    }

    pub fn expected(&self, address: usize) -> Option<PhysicalPage> {
        self.run_at(address, usize::MAX).1
    }
}


#[cfg(test)]
#[test]
fn physical_runs_keep_first_backing_and_never_fill_a_hole() {
    let page = PAGE_BYTES as usize;
    let first = PageNumbers::default().allocate(12).unwrap();
    let mut pins = BackingPins::default();
    pins.remember(page, page * 2, Some(first));
    pins.remember(page * 2, page * 3, Some(first.offset(1)));
    pins.remember(page * 3, page * 4, Some(first.offset(8)));
    pins.remember(page * 4, page * 5, Some(first.offset(1)));
    pins.remember(page * 2, page * 3, Some(first.offset(10)));
    assert_eq!(pins.physical_runs(page + 7, page * 2 + 8), Some(vec![(first, 2)]));
    assert_eq!(pins.physical_runs(page * 2 + 1, page * 4 + 7), Some(vec![
        (first.offset(1), 1), (first.offset(8), 1), (first.offset(1), 1),
    ]));
    assert_eq!(pins.physical_runs(page * 4, page * 5 + 1), None);
    assert_eq!(pins.physical_runs(page, page), None);
}
