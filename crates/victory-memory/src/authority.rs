//! VNME representation authority. GPU completion is not CPU visibility.
use crate::coherence::Fence;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Representation {
    CompatibleBytes,
    NativeGpu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Observation {
    /// The title can observe the bytes or that possibility is unproven.
    PreserveCompatibleBytes,
    /// Established by a resource contract, never by format alone.
    GpuOnly,
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
