//! Host-side policy for disposable caches. Guest allocations, dirty data
//! and memory still in use by a command list are not cache.

const MIB: usize = 1 << 20;
const GIB: usize = 1 << 30;
const RAM_RESERVE: u64 = 2 * GIB as u64;

#[derive(Clone, Copy)]
pub(crate) enum Cache {
    Snapshot,
    TextureRead,
    ListUpload,
}

impl Cache {
    const fn ceiling(self) -> usize {
        match self {
            Self::Snapshot => 64 * MIB,
            Self::TextureRead => 64 * MIB,
            Self::ListUpload => 128 * MIB,
        }
    }

    fn allowance(self, available: u64) -> usize {
        // Never reserve the full 5 GiB. Start with conservative sub-budgets,
        // and make room for Windows and the game before retaining scratch.
        match available {
            ..=RAM_RESERVE => 0,
            n if n <= 3 * GIB as u64 => self.ceiling() / 4,
            _ => self.ceiling(),
        }
    }
}

#[cfg(windows)]
mod platform {
    use super::{Cache, MIB, RAM_RESERVE};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
    use std::time::{Duration, Instant};
    use windows::Win32::Graphics::Dxgi::{DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO, IDXGIAdapter3};
    #[cfg(feature = "paged-title")]
    use victory_memory::{
        budget::{BudgetProfile, Headroom, HostPressure, Pool},
        physical::HostBudget,
    };
    #[cfg(feature = "paged-title")]
    use windows_sys::Win32::System::ProcessStatus::{K32GetPerformanceInfo, PERFORMANCE_INFORMATION};
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    static PRESSURE: AtomicBool = AtomicBool::new(false);
    static AVAILABLE: AtomicU64 = AtomicU64::new(0);

    // Mappings are not disposable cache: the GPU can still name them after
    // the title has reallocated the range. Bound actual live leases.
    const MAPPING_LIMIT: u64 = 512 * MIB as u64;
    const MAX_MAPPING: u64 = 16 * MIB as u64;
    static MAPPED: AtomicU64 = AtomicU64::new(0);

    pub(crate) struct MappingLease(u64);

    impl MappingLease {
        pub(crate) fn acquire(bytes: u64) -> Option<Self> {
            if bytes == 0 || bytes > MAX_MAPPING {
                return None;
            }
            MAPPED.fetch_update(Relaxed, Relaxed, |used| {
                used.checked_add(bytes).filter(|&next| next <= MAPPING_LIMIT)
            }).ok()?;
            Some(Self(bytes))
        }
    }

    impl Drop for MappingLease {
        fn drop(&mut self) {
            MAPPED.fetch_sub(self.0, Relaxed);
        }
    }

    pub(crate) fn mapping_headroom() -> bool {
        AVAILABLE.load(Relaxed) > RAM_RESERVE && !PRESSURE.load(Relaxed)
    }

    #[derive(Default)]
    struct Samples {
        baseline_ram_used: Option<u64>,
        last: Option<Instant>,
        peak_sum: u64,
        last_log: Option<Instant>,
    }

    static SAMPLES: Mutex<Samples> = Mutex::new(Samples {
        baseline_ram_used: None, last: None, peak_sum: 0, last_log: None,
    });

    fn memory() -> Option<MEMORYSTATUSEX> {
        let mut status = MEMORYSTATUSEX {
            dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
            ..Default::default()
        };
        // SAFETY: the caller provides a full, initialized status structure.
        (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(status)
    }

    #[cfg(feature = "paged-title")]
    fn commit_available() -> u64 {
        let mut info = PERFORMANCE_INFORMATION {
            cb: std::mem::size_of::<PERFORMANCE_INFORMATION>() as u32,
            ..Default::default()
        };
        // 安全：传入完整结构及其大小；查询失败时不发放新提交额度。
        if unsafe { K32GetPerformanceInfo(&mut info, info.cb) } == 0 {
            return 0;
        }
        (info.CommitLimit.saturating_sub(info.CommitTotal) as u64)
            .saturating_mul(info.PageSize as u64)
    }

    #[cfg(feature = "paged-title")]
    fn adaptive_sample(adapter: Option<&IDXGIAdapter3>)
        -> (Option<MEMORYSTATUSEX>, Option<DXGI_QUERY_VIDEO_MEMORY_INFO>)
    {
        use windows::Win32::Graphics::Dxgi::IDXGIAdapter1;
        use windows::core::Interface;

        let mut ram = None;
        let mut local = None;
        HostBudget::process().refresh_pressure(
            Headroom {
                physical_ram: RAM_RESERVE,
                commit: 512 * MIB as u64,
                local: crate::residency::LOCAL_RESERVE,
            },
            BudgetProfile::Auto,
            || {
                ram = memory();
                local = adapter.and_then(query_local);
                // 安全：通过设备的适配器读取硬件描述；DXGI 实时预算不是显存容量。
                let capacity = adapter.and_then(|a| a.cast::<IDXGIAdapter1>().ok())
                    .and_then(|a| unsafe { a.GetDesc1().ok() })
                    .map_or(0, |d| d.DedicatedVideoMemory as u64);
                // 没有有效的 RAM 或独显采样时冻结相应的新增准入，不释放现有租约。
                Some(HostPressure {
                    physical_ram: ram.as_ref().map_or(0, |s| s.ullTotalPhys),
                    available_ram: ram.as_ref().map_or(0, |s| s.ullAvailPhys),
                    available_commit: commit_available(),
                    local_capacity: capacity,
                    local_budget: local.as_ref().map_or(0, |v| v.Budget),
                    local_observed_use: local.as_ref().map_or(0, |v| v.CurrentUsage),
                    managed_ram: 0,
                    managed_local: 0,
                })
            },
        );
        (ram, local)
    }

    pub fn begin() {
        #[cfg(feature = "paged-title")]
        let (initial, _) = adaptive_sample(None);
        #[cfg(not(feature = "paged-title"))]
        let initial = memory();
        if let Some(s) = initial {
            AVAILABLE.store(s.ullAvailPhys, Relaxed);
            SAMPLES.lock().unwrap_or_else(|e| e.into_inner()).baseline_ram_used =
                Some(s.ullTotalPhys.saturating_sub(s.ullAvailPhys));
        }
    }

    pub(crate) fn limit(cache: Cache) -> usize {
        if PRESSURE.load(Relaxed) { 0 } else { cache.allowance(AVAILABLE.load(Relaxed)) }
    }

    pub(crate) fn query_local(adapter: &IDXGIAdapter3) -> Option<DXGI_QUERY_VIDEO_MEMORY_INFO> {
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: live adapter, node zero, and initialized output storage.
        unsafe { adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info) }
            .ok().map(|()| info)
    }

    pub(crate) fn sample(adapter: Option<&IDXGIAdapter3>) {
        let Ok(mut state) = SAMPLES.try_lock() else { return };
        let now = Instant::now();
        if state.last.is_some_and(|last| now.duration_since(last) < Duration::from_millis(500)) {
            return;
        }
        state.last = Some(now);
        #[cfg(feature = "paged-title")]
        let (ram, local) = adaptive_sample(adapter);
        #[cfg(not(feature = "paged-title"))]
        let (ram, local) = (memory(), adapter.and_then(query_local));
        AVAILABLE.store(ram.as_ref().map_or(0, |s| s.ullAvailPhys), Relaxed);
        let Some(s) = ram else {
            PRESSURE.store(true, Relaxed);
            return;
        };
        let used = s.ullTotalPhys.saturating_sub(s.ullAvailPhys);
        let ram_delta = used.saturating_sub(*state.baseline_ram_used.get_or_insert(used));
        let sum = local.map(|info| ram_delta.saturating_add(info.CurrentUsage));
        if let Some(sum) = sum {
            state.peak_sum = state.peak_sum.max(sum);
        }
        // 回收使用软线，准入使用硬线；回暖时保留滞回区，避免来回扩缩。
        let margin = if PRESSURE.load(Relaxed) { 128 * MIB as u64 } else { 0 };
        #[cfg(feature = "paged-title")]
        let (over_ram, over_vram, over_target) = {
            let ledger = HostBudget::process();
            let (ram, ram_limits) = ledger.pool_state(Pool::Ram);
            let (vram, local_limits) = ledger.pool_state(Pool::Local);
            (
                s.ullAvailPhys <= RAM_RESERVE || ram >= ram_limits.soft.saturating_sub(margin),
                vram > 0 && vram >= local_limits.soft.saturating_sub(margin),
                ram > ram_limits.hard || vram > local_limits.hard,
            )
        };
        #[cfg(not(feature = "paged-title"))]
        let (over_ram, over_vram, over_target) = (
            ram_delta >= (crate::console::UNIFIED_MEMORY as u64).saturating_sub(margin),
            local.is_some_and(|v| {
                v.CurrentUsage >= crate::residency::LOCAL_LIMIT.min(v.Budget).saturating_sub(margin)
            }),
            sum.is_some_and(|n| n > (crate::console::UNIFIED_MEMORY + crate::console::MAX_HOST_CACHE) as u64),
        );
        PRESSURE.store(over_ram || over_vram, Relaxed);
        crate::shadow::trim_idle();
        crate::texture::trim_idle_scratch();
        let idle_upload = crate::host::trim_idle_upload();
        if state.last_log.is_none_or(|last| now.duration_since(last) >= Duration::from_secs(5)) {
            state.last_log = Some(now);
            let (managed_local, managed_ram) = crate::residency::counters();
            #[cfg(feature = "paged-title")]
            {
                let ledger = HostBudget::process();
                let (_, ram) = ledger.pool_state(Pool::Ram);
                let (_, local) = ledger.pool_state(Pool::Local);
                tracing::info!(
                    ram_soft_mib = ram.soft / MIB as u64,
                    ram_hard_mib = ram.hard / MIB as u64,
                    local_soft_mib = local.soft / MIB as u64,
                    local_hard_mib = local.hard / MIB as u64,
                    "VNME 动态宿主分配预算"
                );
            }
            tracing::info!(
                ram_delta_mib = ram_delta / MIB as u64,
                ram_available_mib = s.ullAvailPhys / MIB as u64,
                local_usage_mib = ?local.map(|info| info.CurrentUsage / MIB as u64),
                local_budget_mib = ?local.map(|info| info.Budget / MIB as u64),
                combined_sample_mib = ?sum.map(|n| n / MIB as u64),
                managed_local_mib = managed_local / MIB as u64,
                managed_ram_reserved_mib = managed_ram / MIB as u64,
                combined_peak_mib = ?sum.map(|_| state.peak_sum / MIB as u64),
                idle_upload_mib = idle_upload / MIB,
                mapped_title_mib = MAPPED.load(Relaxed) / MIB as u64,
                over_target,
                ram_pressure = s.ullAvailPhys <= RAM_RESERVE,
                "Series S 宿主内存（RAM 为系统占用变化，不重复计入 GPU 共享内存）"
            );
        }
    }
}

#[cfg(windows)]
pub use platform::begin;
#[cfg(windows)]
pub(crate) use platform::{MappingLease, limit, mapping_headroom, query_local, sample};

#[test]
fn idle_cache_leaves_headroom_and_never_reserves_the_remainder() {
    let caps = [Cache::Snapshot, Cache::TextureRead, Cache::ListUpload];
    let allowance = |available| caps.iter().map(|cache| cache.allowance(available)).sum::<usize>();
    assert_eq!(allowance(2 * GIB as u64), 0);
    assert_eq!(allowance(2 * GIB as u64 + 1), 64 * MIB);
    assert_eq!(allowance(3 * GIB as u64 + 1), 256 * MIB);
    assert!(allowance(u64::MAX) <= crate::console::MAX_HOST_CACHE);
}
