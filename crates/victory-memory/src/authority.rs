//! VNME representation authority. GPU completion is not CPU visibility.
use crate::coherence::Fence;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Representation {
    CompatibleBytes,
    NativeGpu,
    /// 游戏明确允许丢弃旧内容，但尚未证明任何可读取的新像素。
    Discarded,
}

/// Monotonic evidence about bytes that escape the native representation.
/// A missing observation never proves an address is private.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ByteExposure {
    TitleAllocation = 1,
    SharedHeap = 2,
    ExternalAddress = 4,
    MappedPointer = 8,
    GuestGpuAddress = 16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Observation {
    /// No positive proof either way; retain byte-compatible access.
    Unproven,
    /// Title bytes are observable; bits record the known exposure routes.
    GuestBytes(u8),
    /// Established by a resource contract, never by format alone.
    GpuOnly,
}

impl Observation {
    pub fn from_exposures(bits: u8) -> Self {
        if bits == 0 { Self::Unproven } else { Self::GuestBytes(bits) }
    }

    pub fn requires_compatible_bytes(self) -> bool {
        !matches!(self, Self::GpuOnly)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Version {
    pub number: u64,
    pub representation: Representation,
    pub observation: Observation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompletedVersion {
    pub version: Version,
    pub fence: Fence,
}

impl Version {
    pub fn successor(
        prior: Option<Self>, representation: Representation, observation: Observation,
    ) -> Option<Self> {
        let number = prior.map_or(Some(1), |prior| prior.number.checked_add(1))?;
        Some(Self { number, representation, observation })
    }

    pub fn complete(self, submitted: Fence, observed: Fence) -> Option<CompletedVersion> {
        (observed.0 >= submitted.0).then_some(CompletedVersion { version: self, fence: submitted })
    }
}
