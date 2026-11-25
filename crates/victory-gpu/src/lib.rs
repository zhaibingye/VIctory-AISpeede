//! D3D12.X on host D3D12.
//!
//! D3D12.X records commands from code inlined into the title, which writes
//! packets into the command list while its put pointer is below a limit and
//! calls the list's method otherwise. Our lists keep the limits at zero, so
//! every command arrives here as a call, to be run on host D3D12. The
//! title's GPU addresses are CPU addresses (console memory is unified).
//!
//! [`device`] is the device object titles create, [`dxgi`] its adapter side,
//! [`resource`] heaps and resources, [`fence`] fences, [`queue`] command
//! queues, [`list`] command allocators and lists, [`descriptor`]
//! descriptor heaps, [`root`] root signatures, [`query`] query heaps,
//! [`signature`] command signatures, [`pipeline`] pipeline states,
//! [`frame`] frame pacing, [`object`] what all console graphics objects
//! share, [`console`] which console model the title sees, `external` host
//! libraries recording into title lists. Planned:
//! `present` (swap chain).

#[cfg(windows)]
mod bindless;
#[cfg(all(windows, feature = "diagnostics"))]
mod capture;
pub mod console;
#[cfg(windows)]
#[allow(dead_code, reason = "native copy binding is enabled at unified renderer cutover")]
mod copy_plan;
#[cfg(all(windows, feature = "diagnostics"))]
mod copystat;
pub mod descriptor;
pub mod device;
#[cfg(windows)]
mod draw;
pub mod dxgi;
#[cfg(windows)]
pub mod external;
pub mod fence;
pub mod frame;
#[cfg(windows)]
mod host;
#[cfg(windows)]
mod indirect;
pub mod list;
#[cfg(windows)]
mod memory;
#[cfg(windows)]
#[allow(dead_code, reason = "will replace title dispatch after direct and texture views use physical pages")]
mod page_queue;
#[cfg(windows)]
#[allow(dead_code, reason = "closed native draws are admitted at renderer cutover")]
mod page_draw;
#[cfg(windows)]
#[allow(dead_code, reason = "wired into title submissions together with direct/texture coherence at cutover")]
mod page_slabs;
#[cfg(windows)]
#[allow(dead_code, reason = "physical input views are enabled at unified renderer cutover")]
mod page_view;
pub mod object;
pub mod pipeline;
#[cfg(windows)]
mod primitive;
pub mod query;
pub mod queue;
pub mod resource;
#[cfg(windows)]
mod residency;
pub mod root;
#[cfg(windows)]
mod sampler;
#[cfg(windows)]
pub mod shader_cache;
#[cfg(windows)]
mod shadow;
#[cfg(all(windows, test))]
mod shared_memory_probe;
pub mod signature;
mod switch;
#[cfg(windows)]
mod tdr;
#[cfg(windows)]
pub mod title_backing;
#[cfg(windows)]
mod sparse;
#[cfg(windows)]
mod texture;
#[cfg(windows)]
mod twin;
#[cfg(windows)]
mod watch;
#[cfg(any(windows, test))]
mod working_set;
#[cfg(windows)]
pub use working_set::begin as begin_memory_observation;

#[cfg(windows)]
pub use host::on_failure;
#[cfg(windows)]
pub use watch::{
    Writing as KernelWrite, title_memory_aliased, title_memory_remapped,
    writing as kernel_will_write,
};

#[cfg(test)]
mod bc1_detile_experiment;

#[cfg(windows)]
mod upscale;
