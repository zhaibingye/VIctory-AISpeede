//! Submission-local attachment authority and resource transitions.
//! The global resource ledger is changed only after successful execution.
use std::collections::BTreeMap;
use std::sync::{Arc, atomic::Ordering};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, legal_state, transition};
use crate::page_slabs::BatchError;
use crate::resource::HostTexture;

#[derive(Default)]
pub(crate) struct ResourcePlan {
    initialized: BTreeMap<usize, Arc<HostTexture>>,
    state: BTreeMap<usize, (Arc<HostTexture>, D3D12_RESOURCE_STATES)>,
}

impl ResourcePlan {
    fn key(copy: &Arc<HostTexture>) -> usize {
        Arc::as_ptr(copy) as usize
    }

    pub(crate) fn authoritative(&self, copy: &Arc<HostTexture>) -> bool {
        copy.host_written.load(Ordering::Acquire) || self.initialized.contains_key(&Self::key(copy))
    }

    pub(crate) fn initialize(&mut self, copy: &Arc<HostTexture>) {
        self.initialized.insert(Self::key(copy), copy.clone());
    }

    pub(crate) fn state(&self, copy: &Arc<HostTexture>) -> D3D12_RESOURCE_STATES {
        self.state.get(&Self::key(copy)).map_or_else(
            || D3D12_RESOURCE_STATES(copy.state.load(Ordering::Acquire) as i32),
            |(_, state)| *state,
        )
    }

    pub(crate) fn commit(self) {
        for (copy, state) in self.state.into_values() {
            copy.state.store(state.0 as u32, Ordering::Release);
        }
        for copy in self.initialized.into_values() {
            copy.host_written.store(true, Ordering::Release);
        }
    }
}

pub(crate) enum PageBarrier {
    Texture(Arc<HostTexture>, u32),
    Uav,
}

impl PageBarrier {
    pub(crate) fn encode(
        &self, list: &HostList, resources: &mut ResourcePlan,
    ) -> Result<(), BatchError> {
        match self {
            Self::Texture(copy, wanted) => {
                let writable = 0x4 | 0x8 | 0x10 | 0x400 | 0x1000;
                let write = wanted & writable;
                if copy.sparse.is_some() || copy.twin.get().is_some_and(Option::is_some)
                    || wanted & !(writable | 0x20 | 0x40 | 0x80 | 0x800 | 0x2000) != 0
                    || write != 0 && (*wanted != write || !write.is_power_of_two())
                    || legal_state(&copy.texture, *wanted) != *wanted
                {
                    return Err(BatchError::UnsupportedBarrier);
                }
                let before = resources.state(copy);
                let after = D3D12_RESOURCE_STATES(*wanted as i32);
                if before != after {
                    list.keep(&copy.texture);
                    // SAFETY: live attachment, ordered submission-local state,
                    // and a checked whole-resource transition.
                    unsafe { list.recording().ResourceBarrier(&[transition(&copy.texture, before, after)]) };
                    resources.state.insert(ResourcePlan::key(copy), (copy.clone(), after));
                }
            }
            Self::Uav => {
                let barrier = D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                            pResource: std::mem::ManuallyDrop::new(None),
                        }),
                    },
                };
                // SAFETY: a global ordered UAV barrier uses no pointer.
                unsafe { list.recording().ResourceBarrier(&[barrier]) };
            }
        }
        Ok(())
    }
}
