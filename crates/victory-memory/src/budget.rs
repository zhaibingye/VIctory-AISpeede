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
}

#[must_use = "retain this charge with the physical allocation"]
#[derive(Debug)]
pub struct Charge {
    pool: Pool,
    bytes: u64,
}

impl Budget {
    pub fn observe_local(&mut self, dxgi: u64, outside_ledger: u64) -> u64 {
        self.local_cap = LOCAL_LIMIT.min(dxgi).saturating_sub(outside_ledger);
        self.local.saturating_sub(self.local_cap)
    }

    pub fn used(&self, pool: Pool) -> u64 {
        match pool {
            Pool::Ram => self.ram,
            Pool::Local => self.local,
        }
    }

    pub fn charge(&mut self, pool: Pool, bytes: u64) -> Result<Charge, NoSpace> {
        let (used, cap) = match pool {
            Pool::Ram => (&mut self.ram, RAM_LIMIT),
            Pool::Local => (&mut self.local, self.local_cap),
        };
        let free = cap.saturating_sub(*used);
        if bytes > free {
            return Err(NoSpace { pool, shortfall: bytes - free });
        }
        *used += bytes;
        Ok(Charge { pool, bytes })
    }

    pub fn release(&mut self, charge: Charge) {
        let used = match charge.pool {
            Pool::Ram => &mut self.ram,
            Pool::Local => &mut self.local,
        };
        *used = used.checked_sub(charge.bytes).expect("charge belongs to its ledger");
    }
}
