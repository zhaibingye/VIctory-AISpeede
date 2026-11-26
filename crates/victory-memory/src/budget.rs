mod adaptive;
pub use adaptive::{BudgetProfile, Headroom, HostLimits, HostPressure, PoolLimits};

pub const RAM_LIMIT: u64 = 10 << 30;
pub const LOCAL_LIMIT: u64 = 5 << 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pool {
    Ram,
    Local,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NoSpace {
    pub pool: Pool,
    pub shortfall: u64,
}

/// Serialized physical allocation ledger. Initially no DXGI budget is known.
#[derive(Default, Debug)]
pub struct Budget {
    ram: u64,
    local: u64,
    local_cap: u64,
    adaptive: Option<HostLimits>,
}

#[must_use = "retain this charge with the physical allocation"]
#[derive(Debug)]
pub struct Charge {
    pool: Pool,
    bytes: u64,
}

impl Budget {
    pub fn observe_local(&mut self, dxgi: u64, outside_ledger: u64) -> u64 {
        self.local_cap = match self.adaptive {
            Some(limits) => limits.local.hard.min(dxgi.saturating_sub(outside_ledger)),
            None => LOCAL_LIMIT.min(dxgi).saturating_sub(outside_ledger),
        };
        self.local.saturating_sub(self.local_cap)
    }

    pub(crate) fn apply_host_limits(&mut self, limits: HostLimits) {
        self.local_cap = limits.local.hard;
        self.adaptive = Some(limits);
    }

    /// 上限收紧时保留真实占用，不取消已签发的分配租约。
    pub fn limits(&self, pool: Pool) -> PoolLimits {
        match (self.adaptive, pool) {
            (Some(limits), Pool::Ram) => limits.ram,
            (Some(limits), Pool::Local) => PoolLimits {
                soft: limits.local.soft.min(self.local_cap), hard: self.local_cap,
            },
            (None, Pool::Ram) => PoolLimits { soft: RAM_LIMIT, hard: RAM_LIMIT },
            (None, Pool::Local) => PoolLimits { soft: self.local_cap, hard: self.local_cap },
        }
    }

    #[cfg(windows)]
    pub(crate) fn local_limit(&self) -> u64 {
        self.local_cap
    }

    pub fn used(&self, pool: Pool) -> u64 {
        match pool {
            Pool::Ram => self.ram,
            Pool::Local => self.local,
        }
    }

    pub fn charge(&mut self, pool: Pool, bytes: u64) -> Result<Charge, NoSpace> {
        let cap = self.limits(pool).hard;
        let used = match pool {
            Pool::Ram => &mut self.ram,
            Pool::Local => &mut self.local,
        };
        let free = cap.saturating_sub(*used);
        if bytes > free {
            return Err(NoSpace { pool, shortfall: bytes - free });
        }
        *used += bytes;
        Ok(Charge { pool, bytes })
    }

    pub(crate) fn resize(&mut self, charge: &mut Charge, bytes: u64) -> Result<(), NoSpace> {
        if bytes > charge.bytes {
            let _increase = self.charge(charge.pool, bytes - charge.bytes)?;
        } else if bytes < charge.bytes {
            self.release(Charge { pool: charge.pool, bytes: charge.bytes - bytes });
        }
        charge.bytes = bytes;
        Ok(())
    }

    pub fn release(&mut self, charge: Charge) {
        let used = match charge.pool {
            Pool::Ram => &mut self.ram,
            Pool::Local => &mut self.local,
        };
        *used = used.checked_sub(charge.bytes).expect("charge belongs to its ledger");
    }
}

#[cfg(test)]
#[test]
fn adaptive_pressure_blocks_new_charges_without_forgetting_live_bytes() {
    let mut budget = Budget::default();
    let old = budget.charge(Pool::Ram, 80).unwrap();
    budget.apply_host_limits(HostLimits {
        ram: PoolLimits { soft: 32, hard: 64 },
        local: PoolLimits { soft: 7 << 30, hard: 8 << 30 },
    });
    assert_eq!(budget.used(Pool::Ram), 80);
    assert_eq!(budget.charge(Pool::Ram, 1).unwrap_err(), NoSpace { pool: Pool::Ram, shortfall: 1 });
    assert_eq!(budget.observe_local(9 << 30, 0), 0);
    let gpu = budget.charge(Pool::Local, 7 << 30).unwrap();
    assert_eq!(budget.observe_local(6 << 30, 0), 1 << 30);
    assert_eq!(budget.limits(Pool::Local), PoolLimits { soft: 6 << 30, hard: 6 << 30 });
    budget.release(gpu);
    budget.release(old);
    assert!(budget.charge(Pool::Ram, 64).is_ok());
}
