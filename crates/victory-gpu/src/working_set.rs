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

    pub fn begin() {
        if let Some(s) = memory() {
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
        let Some(s) = memory() else { return };
        AVAILABLE.store(s.ullAvailPhys, Relaxed);
        let used = s.ullTotalPhys.saturating_sub(s.ullAvailPhys);
        let ram_delta = used.saturating_sub(*state.baseline_ram_used.get_or_insert(used));
        let local = adapter.and_then(query_local);
        let sum = local.map(|info| ram_delta.saturating_add(info.CurrentUsage));
        let ram_target = crate::console::UNIFIED_MEMORY as u64;
        let vram_target = crate::residency::LOCAL_LIMIT;
        let target = (crate::console::UNIFIED_MEMORY + crate::console::MAX_HOST_CACHE) as u64;
        if let Some(sum) = sum {
            state.peak_sum = state.peak_sum.max(sum);
        }
        // Small hysteresis so crossing either cap doesn't churn the pool.
        let margin = if PRESSURE.load(Relaxed) { 128 * MIB as u64 } else { 0 };
        let over_ram = ram_delta >= ram_target.saturating_sub(margin);
        let over_vram = local.is_some_and(|v| {
            v.CurrentUsage >= vram_target.min(v.Budget).saturating_sub(margin)
        });
        PRESSURE.store(over_ram || over_vram, Relaxed);
        crate::shadow::trim_idle();
        crate::texture::trim_idle_scratch();
        let idle_upload = crate::host::trim_idle_upload();
        if state.last_log.is_none_or(|last| now.duration_since(last) >= Duration::from_secs(5)) {
            state.last_log = Some(now);
            let (managed_local, managed_ram) = crate::residency::counters();
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
                over_target = sum.is_some_and(|bytes| bytes > target),
                ram_pressure = s.ullAvailPhys <= RAM_RESERVE,
                "Series S host memory (RAM delta is system-wide; GPU shared RAM not added)"
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
