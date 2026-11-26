//! 宿主预算策略。只计算新分配能用多少，不改变 Series S 的逻辑容量。

/// 本地内存与系统内存分开采样；非本地 GPU 分配在 RAM 账本统计。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostPressure {
    pub physical_ram: u64,
    pub available_ram: u64,
    pub available_commit: u64,
    pub managed_ram: u64,
    pub local_capacity: u64,
    pub local_budget: u64,
    pub local_observed_use: u64,
    pub managed_local: u64,
}

/// 每项留给操作系统或驱动的安全余量，调用方按设备与压力决定。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Headroom {
    pub physical_ram: u64,
    pub commit: u64,
    pub local: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BudgetProfile {
    #[default]
    Auto,
    Performance,
    Manual { ram: u64, local: u64 },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolLimits {
    pub soft: u64,
    pub hard: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostLimits {
    pub ram: PoolLimits,
    pub local: PoolLimits,
}

impl HostLimits {
    pub fn plan(sample: HostPressure, reserve: Headroom, profile: BudgetProfile) -> Self {
        // 可用量已经剔除了当前占用。因此只加回本账本，再按物理总量封顶。
        // 分页文件增加提交额度时，仍不能突破可用物理内存这条线。
        let free_ram = sample.available_ram.saturating_sub(reserve.physical_ram);
        let free_commit = sample.available_commit.saturating_sub(reserve.commit);
        let ram = sample.managed_ram.saturating_add(free_ram.min(free_commit))
            .min(sample.physical_ram.saturating_sub(reserve.physical_ram));
        // DXGI 当前使用可能包含已入账部分，先扣掉其中不在本账本的量。
        let outside_local = sample.local_observed_use.saturating_sub(sample.managed_local);
        let local = sample.local_capacity.min(sample.local_budget)
            .saturating_sub(outside_local.saturating_add(reserve.local));
        let (ram, local) = match profile {
            BudgetProfile::Manual { ram: cap_ram, local: cap_local } =>
                (ram.min(cap_ram), local.min(cap_local)),
            _ => (ram, local),
        };
        let band = |hard| PoolLimits {
            // 性能模式只提高软线，不能提高硬上限。
            soft: hard - hard / if profile == BudgetProfile::Performance { 16 } else { 8 },
            hard,
        };
        Self { ram: band(ram), local: band(local) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_budget_preserves_headroom_without_double_counting() {
        let sample = HostPressure {
            physical_ram: 32, available_ram: 20, available_commit: 25, managed_ram: 6,
            local_capacity: 16, local_budget: 14, local_observed_use: 7, managed_local: 5,
        };
        let reserve = Headroom { physical_ram: 4, commit: 2, local: 1 };
        let auto = HostLimits::plan(sample, reserve, BudgetProfile::Auto);
        assert_eq!((auto.ram.hard, auto.local.hard), (22, 11));
        let performance = HostLimits::plan(sample, reserve, BudgetProfile::Performance);
        assert_eq!(performance.ram.hard, auto.ram.hard);
        assert!(performance.ram.soft > auto.ram.soft);
        let manual = HostLimits::plan(sample, reserve, BudgetProfile::Manual { ram: 15, local: 99 });
        assert_eq!((manual.ram.hard, manual.local.hard), (15, 11));
        let pressure = HostLimits::plan(HostPressure {
            available_ram: 9, available_commit: 0, local_budget: 1, ..sample
        }, reserve, BudgetProfile::Auto);
        assert_eq!((pressure.ram.hard, pressure.local.hard), (6, 0));
    }
}
