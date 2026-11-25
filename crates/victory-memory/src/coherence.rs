//! Version/authority for one physical page on a single ordered queue.

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Fence(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    Read,
    Modify,
    FullOverwrite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Need {
    Ready,
    Wait(Fence),
    Upload,
    Download,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Invalid {
    NotReady(Need),
    PastFence,
    VersionExhausted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Authority { Ram, Both, Gpu }

#[derive(Clone, Copy, Debug)]
enum Pending { Use, Transfer }
#[derive(Debug)]
pub struct Page {
    authority: Authority,
    version: u64,
    until: Option<(Fence, Pending)>,
}

impl Default for Page {
    fn default() -> Self {
        Self { authority: Authority::Ram, version: 0, until: None }
    }
}

impl Page {
    pub fn version(&self) -> u64 { self.version }

    pub fn settle(&mut self, completed: Fence) {
        if let Some((fence, pending)) = self.until
            && fence <= completed
        {
            self.until = None;
            if matches!(pending, Pending::Transfer) { self.authority = Authority::Both; }
        }
    }

    pub fn cpu_need(&mut self, access: Access, completed: Fence) -> Need {
        self.settle(completed);
        if let Some((fence, _)) = self.until { return Need::Wait(fence); }
        if self.authority == Authority::Gpu && access != Access::FullOverwrite {
            return Need::Download;
        }
        Need::Ready
    }

    pub fn gpu_need(&mut self, access: Access, completed: Fence) -> Need {
        self.settle(completed);
        if let Some((fence, _)) = self.until { return Need::Wait(fence); }
        if self.authority == Authority::Ram && access != Access::FullOverwrite {
            return Need::Upload;
        }
        Need::Ready
    }

    /// Call under CPU-exclusive access, after the bytes have been written.
    pub fn cpu_written(&mut self, access: Access, completed: Fence) -> Result<(), Invalid> {
        check(self.cpu_need(if access == Access::FullOverwrite { access } else { Access::Modify }, completed))?;
        self.bump()?;
        self.authority = Authority::Ram;
        Ok(())
    }

    /// Call after recording work and reserving its queue signal value. No
    /// version becomes CPU-visible before that value is completed.
    pub fn gpu_submitted(
        &mut self, access: Access, completed: Fence, signal: Fence,
    ) -> Result<(), Invalid> {
        if signal <= completed { return Err(Invalid::PastFence); }
        check(self.gpu_need(access, completed))?;
        if access != Access::Read {
            self.bump()?;
            self.authority = Authority::Gpu;
        }
        self.until = Some((signal, Pending::Use));
        Ok(())
    }

    pub fn upload_submitted(&mut self, completed: Fence, signal: Fence) -> Result<(), Invalid> {
        let need = self.gpu_need(Access::Read, completed);
        self.transfer(need, Need::Upload, completed, signal)
    }

    pub fn download_submitted(&mut self, completed: Fence, signal: Fence) -> Result<(), Invalid> {
        let need = self.cpu_need(Access::Read, completed);
        self.transfer(need, Need::Download, completed, signal)
    }

    pub fn discard_gpu(&mut self, completed: Fence) -> Result<(), Invalid> {
        check(self.cpu_need(Access::Read, completed))?;
        self.authority = Authority::Ram;
        Ok(())
    }

    fn transfer(
        &mut self, need: Need, expected: Need, completed: Fence, signal: Fence,
    ) -> Result<(), Invalid> {
        if signal <= completed { return Err(Invalid::PastFence); }
        if need != expected { return Err(Invalid::NotReady(need)); }
        self.until = Some((signal, Pending::Transfer));
        Ok(())
    }

    fn bump(&mut self) -> Result<(), Invalid> {
        self.version = self.version.checked_add(1).ok_or(Invalid::VersionExhausted)?;
        Ok(())
    }
}

fn check(need: Need) -> Result<(), Invalid> {
    if need == Need::Ready { Ok(()) } else { Err(Invalid::NotReady(need)) }
}
