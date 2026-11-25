//! The host GPU: one D3D12 device and direct queue (decision 0007), and a
//! window to present on.
//!
//! shortcut: every submission is waited for, so CPU and GPU never overlap;
//! overlap them with per-submission fences once more than clears run.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, RECT, WAIT_OBJECT_0, WPARAM};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::{
    D3D_FEATURE_LEVEL_12_0, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, ID3DBlob,
};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{Interface, PCSTR, PCWSTR, s, w};

/// Resource states without the console's own bits (compression hints from
/// bit 25 up), which the host rejects.
pub(crate) fn pc_state(state: u32) -> u32 {
    state & 0x01ff_ffff
}

/// A console state as a host texture state: PC bits only, and none of the
/// buffer-only ones the console allows on textures (vertex, constant and
/// index buffer, stream out, indirect arguments, as in "generic read").
pub(crate) fn texture_state(state: u32) -> u32 {
    const BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x100 | 0x200;
    pc_state(state) & !BUFFER_ONLY
}

/// Render target views live only until their command is recorded, so a
/// ring of host descriptors is enough.
const RTV_RING: u32 = 1024;

pub(crate) struct Host {
    pub(crate) device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    /// The last value signalled on `fence`, behind the submission lock.
    submitted: Mutex<u64>,
    pub(crate) paged: Mutex<crate::page_slabs::PageSlabs>,
    event: HANDLE,
    rtvs: ID3D12DescriptorHeap,
    rtv_size: u32,
    /// A ring of depth views, used as `rtvs`.
    dsvs: ID3D12DescriptorHeap,
    dsv_size: u32,
    next_dsv: AtomicU32,
    /// A ring of CPU-only views, for the CPU handles of unordered access
    /// clears, used as `rtvs`.
    cpu_views: ID3D12DescriptorHeap,
    cpu_view_size: u32,
    next_cpu_view: AtomicU32,
    /// With `VICTORY_D3D12_DEBUG` set (diagnostics builds): the debug
    /// layer's messages.
    messages: Option<ID3D12InfoQueue>,
    next_rtv: AtomicU32,
    presenter: Mutex<Presentation>,
    memory_adapter: OnceLock<Option<IDXGIAdapter3>>,
}

// SAFETY: D3D12 devices, queues, fences and descriptor heaps are
// free-threaded; the event handle is only waited on under `submitted`.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

/// Time spent in submissions since the last frame-rate line, in
/// microseconds: preparing them on the CPU (title memory copies, texture
/// detiling), then waiting for the GPU.
static SUBMISSION_US: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// After submissions: shader-written ranges compared and copied back into
/// title memory (microseconds, bytes that differed), since the last
/// frame-rate line.
static COPY_BACK: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Of the preparing: title memory copies (microseconds, bytes) and texture
/// refreshes (microseconds, bytes hashed), since the last frame-rate line.
pub(crate) static PREPARING: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// What the runtime does when the host GPU fails for good: report it like a
/// crash and end the run. The console has no such state, and carrying on
/// draws nothing at seconds per frame.
static ON_FAILURE: OnceLock<fn(&str) -> !> = OnceLock::new();

/// Sets what a host GPU failure does (the runtime's crash report).
pub fn on_failure(report: fn(&str) -> !) {
    let _ = ON_FAILURE.set(report);
}

/// Ends the run on a host GPU failure: what failed, the removal reason and
/// the debug layer's first error, if any (`VICTORY_D3D12_DEBUG=1` in a
/// diagnostics build names the command).
pub(crate) fn fail(what: std::fmt::Arguments) -> ! {
    let mut text = what.to_string();
    if let Some(host) = host() {
        // SAFETY: a plain query.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            text.push_str(&format!("; device removed: {e}"));
        }
        match host.log_messages() {
            Some(first) => text.push_str(&format!("; debug layer: {first}")),
            None if host.messages.is_none() => text.push_str(
                "; a diagnostics build with VICTORY_D3D12_DEBUG=1 names the command",
            ),
            None => {}
        }
    }
    tracing::error!("{text}");
    match ON_FAILURE.get() {
        Some(report) => report(&text),
        None => std::process::abort(),
    }
}

/// The host GPU, created on first use; `None` without one (a run then
/// behaves as before: nothing executes).
pub(crate) fn host() -> Option<&'static Host> {
    static HOST: OnceLock<Option<Host>> = OnceLock::new();
    HOST.get_or_init(|| match create() {
        Ok(host) => Some(host),
        Err(e) => {
            tracing::warn!(error = %e, "no host D3D12 device; command lists will not run");
            None
        }
    })
    .as_ref()
}

fn create() -> windows::core::Result<Host> {
    // SAFETY: plain D3D12 object creation with valid descriptions.
    unsafe {
        // The debug layer only in builds with diagnostics.
        let debug = cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_DEBUG");
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => layer.iter().for_each(|l| l.EnableDebugLayer()),
                Err(e) => tracing::warn!(error = %e, "no D3D12 debug layer"),
            }
            // GPU-based validation (`VICTORY_D3D12_GBV`): out-of-range
            // descriptor heap indices and the like become messages naming
            // the command instead of page faults. Without state tracking
            // (directly indexed textures are not transitioned).
            if crate::switch::on("VICTORY_D3D12_GBV")
                && let Some(l) = &layer
            {
                if let Ok(d1) = l.cast::<ID3D12Debug1>() {
                    d1.SetEnableGPUBasedValidation(true);
                }
                if let Ok(d2) = l.cast::<ID3D12Debug2>() {
                    d2.SetGPUBasedValidationFlags(
                        D3D12_GPU_BASED_VALIDATION_FLAGS_DISABLE_STATE_TRACKING,
                    );
                }
                tracing::info!("GPU-based validation on");
            }
        }
        // Diagnostics: device-removed data (the command that faulted, the
        // faulting address), read when the device is removed.
        if dred_on() {
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(dred) = dred
            {
                dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
        }
        let mut device: Option<ID3D12Device> = None;
        // Diagnostics: the software adapter, on which a shader's wild access
        // is an access violation in this process (a crash report with the
        // address and instruction) instead of a GPU fault and driver reset.
        if cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_WARP") {
            let adapter = warp_adapter()?;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            tracing::info!("host device on the software adapter (WARP)");
        } else {
            device = Some(create_hardware_device()?);
            // Whether a GPU hang ends the run or freezes the PC (#1054).
            crate::tdr::log();
        }
        let device = device.ok_or_else(windows::core::Error::empty)?;
        let messages = debug
            .then(|| device.cast::<ID3D12InfoQueue>().ok())
            .flatten();
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = CreateEventW(None, false, false, None)?;
        let rtvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let dsvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let dsv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        let cpu_views: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let cpu_view_size =
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        if let Ok(desc) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            .and_then(|factory| factory.EnumAdapterByLuid::<IDXGIAdapter1>(device.GetAdapterLuid()))
            .and_then(|adapter| adapter.GetDesc1())
        {
            tracing::info!(
                adapter = %adapter_name(&desc),
                vendor = format_args!("{:#06x}", desc.VendorId),
                device = format_args!("{:#06x}", desc.DeviceId),
                vram_mib = desc.DedicatedVideoMemory >> 20,
                "host D3D12 adapter"
            );
        }
        tracing::info!("host D3D12 device ready");
        // The console fuses its multiply-adds (`v_fma_f32`, `v_fmac_f32`,
        // `v_fma_mix*`); the lowering's DXIL `FMad` is fused only where the
        // host chooses (the RX 6700 XT does, WARP does not).
        match (victory_shader::emit::host::Host {
            device: device.clone(),
        })
        .fuses_fmad()
        {
            Ok(true) => {}
            Ok(false) => tracing::warn!(
                "the host GPU does not fuse multiply-adds: results of the console's fused v_fma and v_fmac can differ in kind here"
            ),
            Err(e) => tracing::debug!(error = %e, "multiply-add fusing not checked"),
        }
        Ok(Host {
            device,
            queue,
            fence,
            submitted: Mutex::new(0),
            paged: Mutex::default(),
            event,
            rtvs,
            rtv_size,
            dsvs,
            dsv_size,
            next_dsv: AtomicU32::new(0),
            cpu_views,
            cpu_view_size,
            next_cpu_view: AtomicU32::new(0),
            messages,
            next_rtv: AtomicU32::new(0),
            memory_adapter: OnceLock::new(),
            presenter: Mutex::new(Presentation::default()),
        })
    }
}

/// DXGI may return hardware for `EnumWarpAdapter` (notably under Proton).
/// Reject it before creating a device in diagnostics' software-only mode.
fn warp_adapter() -> windows::core::Result<IDXGIAdapter> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter()?;
        let desc = adapter.cast::<IDXGIAdapter1>()?.GetDesc1()?;
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
            tracing::warn!(
                vendor = format_args!("{:#x}", desc.VendorId),
                flags = format_args!("{:#x}", desc.Flags),
                "DXGI did not identify the WARP adapter as software; refusing device"
            );
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_FAIL,
            ));
        }
        Ok(adapter)
    }
}

fn adapter_name(desc: &DXGI_ADAPTER_DESC1) -> String {
    let len = desc
        .Description
        .iter()
        .position(|&c| c == 0)
        .unwrap_or(desc.Description.len());
    String::from_utf16_lossy(&desc.Description[..len])
}

fn is_software_adapter(desc: &DXGI_ADAPTER_DESC1) -> bool {
    desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0
        || (desc.VendorId == 0x1414 && desc.DeviceId == 0x8c)
}

/// Resolves `VICTORY_D3D12_ADAPTER` against DXGI adapter names: a 0-based
/// index or a case-insensitive substring of the adapter description.
fn match_adapter(spec: &str, names: &[String]) -> Option<usize> {
    let spec = spec.trim();
    if spec.is_empty() {
        return None;
    }
    if let Ok(index) = spec.parse::<usize>() {
        return (index < names.len()).then_some(index);
    }
    let needle = spec.to_ascii_lowercase();
    names
        .iter()
        .position(|name| name.to_ascii_lowercase().contains(&needle))
}

/// Picks the host hardware D3D12 device: `VICTORY_D3D12_ADAPTER` (0-based DXGI
/// index or case-insensitive name substring) when set, else the first hardware
/// adapter returned by `DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE`, falling back to
/// DXGI's default adapter (`None`).
fn create_hardware_device() -> windows::core::Result<ID3D12Device> {
    let spec = std::env::var("VICTORY_D3D12_ADAPTER")
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty());
    // SAFETY: plain DXGI adapter enumeration and D3D12CreateDevice calls.
    unsafe {
        if let Some(spec) = spec {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
            let mut adapters = Vec::new();
            let mut names = Vec::new();
            for i in 0.. {
                let Ok(adapter) = factory.EnumAdapters1(i) else {
                    break;
                };
                let name = adapter
                    .GetDesc1()
                    .map(|d| adapter_name(&d))
                    .unwrap_or_default();
                adapters.push(adapter);
                names.push(name);
            }
            let Some(chosen) = match_adapter(&spec, &names) else {
                tracing::warn!(
                    spec,
                    available = ?names,
                    "VICTORY_D3D12_ADAPTER matched no DXGI adapter"
                );
                return Err(windows::core::Error::from_hresult(DXGI_ERROR_NOT_FOUND));
            };
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapters[chosen], D3D_FEATURE_LEVEL_12_0, &mut device)?;
            return device.ok_or_else(windows::core::Error::empty);
        }
        if let Ok(factory6) = CreateDXGIFactory2::<IDXGIFactory6>(DXGI_CREATE_FACTORY_FLAGS(0)) {
            for i in 0.. {
                let Ok(adapter) = factory6.EnumAdapterByGpuPreference::<IDXGIAdapter1>(
                    i,
                    DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                ) else {
                    break;
                };
                if adapter
                    .GetDesc1()
                    .is_ok_and(|desc| is_software_adapter(&desc))
                {
                    continue;
                }
                let mut device: Option<ID3D12Device> = None;
                if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).is_ok()
                    && let Some(device) = device
                {
                    return Ok(device);
                }
            }
        }
        let mut device: Option<ID3D12Device> = None;
        D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?;
        device.ok_or_else(windows::core::Error::empty)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn swap_chain_failures_retry_once_per_second_and_success_clears_the_wait() {
        use super::{Duration, Instant, SwapChainRetry};

        let start = Instant::now();
        let mut retry = SwapChainRetry::default();
        let mut attempts = 0;
        // Two seconds at 60 Hz, with a failure on every eligible attempt.
        for frame in 0..120 {
            retry.attempt(start + Duration::from_millis(frame * 1000 / 60), || {
                attempts += 1;
                Err::<(), _>("desktop unavailable")
            });
        }
        assert_eq!(attempts, 2);
        let recovered = start + Duration::from_secs(2);
        assert_eq!(retry.attempt(recovered, || Ok::<_, ()>(42)), Some(Ok(42)));
        assert_eq!(retry.attempt(recovered, || Ok::<_, ()>(43)), Some(Ok(43)));
    }

    use super::*;

    #[test]
    fn adapter_selector_matches_index_or_substring() {
        let names = vec![
            "Intel(R) UHD Graphics".to_owned(),
            "NVIDIA GeForce RTX 3050 6GB Laptop GPU".to_owned(),
            "Microsoft Basic Render Driver".to_owned(),
        ];
        assert_eq!(super::match_adapter("0", &names), Some(0));
        assert_eq!(super::match_adapter(" 1 ", &names), Some(1));
        assert_eq!(super::match_adapter("3", &names), None);
        assert_eq!(super::match_adapter("nvidia", &names), Some(1));
        assert_eq!(super::match_adapter("RTX 3050", &names), Some(1));
        assert_eq!(super::match_adapter("uhd", &names), Some(0));
        assert_eq!(super::match_adapter("radeon", &names), None);
        assert_eq!(super::match_adapter("   ", &names), None);
    }

    #[test]
    fn list_scratch_grows_to_fit_and_reuses_fenced_chunks() {
        let Some(host) = host() else { return };
        let list = host.new_list().unwrap();
        let first = list.scratch(1).unwrap();
        let second = list.scratch(1).unwrap();
        assert_eq!(second.0 - first.0, 256);
        assert_eq!(second.1 as usize - first.1 as usize, 256);
        assert_eq!(list.scratch.lock().unwrap().chunks[0].capacity, FIRST_SCRATCH_CHUNK);
        list.scratch(96 << 10).unwrap();
        assert_eq!(list.scratch.lock().unwrap().chunks[1].capacity, 128 << 10);
        assert!(list.scratch(u64::MAX).is_none());
        assert!(list.scratch(SCRATCH_CHUNK + 1).is_some());
        list.close();
        host.execute(&[&list]);
        list.reset();
        assert!(list.scratch.lock().unwrap().chunks.is_empty());
        assert!(list.scratch(96 << 10).is_some());
        assert_eq!(list.scratch.lock().unwrap().chunks.len(), 1);
        list.close();
        host.execute(&[&list]);
        list.reset();
        assert!(list.scratch.lock().unwrap().chunks.is_empty());
        assert!(list.scratch(512).is_some());
        list.close();
    }

    #[test]
    fn present_compositor_builds_both_pipeline_states() {
        let Some(host) = host() else {
            eprintln!("skipping compositor initialization: no D3D12 device");
            return;
        };
        Compositor::new(&host.device, DXGI_FORMAT_R8G8B8A8_UNORM)
            .expect("create compositor root signature and both pipeline states");
        eprintln!("PresentX compositor initialized");
    }

    #[test]
    fn warp_adapter_accepts_only_software() {
        // This probes DXGI without creating a D3D12 device or running a title.
        // SAFETY: plain DXGI adapter queries.
        let raw = unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            factory.EnumWarpAdapter::<IDXGIAdapter>()
        };
        let Ok(raw) = raw else { return };
        // SAFETY: EnumWarpAdapter returns a DXGI adapter, which implements
        // IDXGIAdapter1; `raw` owns that interface.
        let desc = unsafe { raw.cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap() };
        eprintln!(
            "EnumWarpAdapter: vendor={:#x}, flags={:#x}",
            desc.VendorId, desc.Flags
        );
        assert_eq!(
            warp_adapter().is_ok(),
            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        );
    }

    /// A node of the device-removed data is a list by pointer or by name.
    /// The name is what still matches when the runtime records another
    /// object than the one we hold (the debug layer wraps ours), and a
    /// different list or an unnamed node never matches.
    #[test]
    fn a_breadcrumb_node_is_found_by_pointer_or_by_name() {
        let ours = [0x1000, 0x1008];
        let name = list_name(7);
        let is = |pointer, text| node_is_list(pointer, text, &ours, &name);
        assert_eq!(name, "victory host list 7");
        assert!(is(Some(0x1000), None));
        assert!(is(Some(0x1008), Some("other")));
        assert!(is(Some(0x9999), Some("victory host list 7")));
        assert!(is(None, Some("victory host list 7")));
        assert!(!is(Some(0x9999), Some("victory host list 70")));
        assert!(!is(Some(0x9999), None));
        assert!(!is(None, None));
        // The debug layer appends its own note to the name we gave.
        assert!(is(None, Some("victory host list 7 (debug layer indirect)")));
        assert!(is(None, Some("victory host list 7 (x)")));
        assert!(!is(
            None,
            Some("victory host list 70 (debug layer indirect)")
        ));
        assert!(!is(None, Some("victory host list 7 debug layer")));
        assert!(!is(None, Some("victory host list 7(")));
        assert!(is(None, Some("victory host list 7")));
        assert_eq!(list_name(70), "victory host list 70");
    }

    /// The pointers of one list: the interface pointers it answers with
    /// are all there, and another list's are not.
    #[test]
    fn a_list_knows_every_pointer_it_answers_with() {
        // SAFETY: plain device and list creation on the software adapter.
        unsafe {
            let Ok(adapter) = warp_adapter() else { return };
            let mut device: Option<ID3D12Device> = None;
            if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).is_err() {
                return;
            }
            let device = device.unwrap();
            // An open list holds its allocator: one each.
            let make = || -> ID3D12GraphicsCommandList {
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .unwrap();
                device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .unwrap()
            };
            let (a, b) = (make(), make());
            let pointers = identities(&a);
            assert!(pointers.contains(&(a.as_raw() as usize)));
            let unknown = a.cast::<windows::core::IUnknown>().unwrap();
            assert!(pointers.contains(&(unknown.as_raw() as usize)));
            let command_list = a.cast::<ID3D12CommandList>().unwrap();
            assert!(pointers.contains(&(command_list.as_raw() as usize)));
            let other = identities(&b);
            assert!(pointers.iter().all(|p| !other.contains(p)));
            // As a node: the same object by any of its pointers; a node
            // naming the other list by pointer and our name still matches.
            let name = list_name(3);
            let is = |pointer, text| node_is_list(pointer, text, &pointers, &name);
            assert!(is(Some(command_list.as_raw() as usize), None));
            assert!(!is(Some(b.as_raw() as usize), None));
            assert!(is(Some(b.as_raw() as usize), Some(&name)));
        }
    }

    /// A node as the runtime gives it, built by hand: `HostList::is_node`
    /// reads the pointer and the copied name, and the name alone finds a
    /// list the runtime records under another object.
    #[test]
    fn is_node_reads_the_name_the_runtime_copied() {
        let Some(host) = host() else { return };
        let Ok(list) = host.new_list() else { return };
        let mut wide: Vec<u16> = list_name(list.id).encode_utf16().collect();
        wide.push(0);
        let mut node = D3D12_AUTO_BREADCRUMB_NODE::default();
        assert!(!list.is_node(&node), "an empty node is no list");
        node.pCommandListDebugNameW = windows::core::PCWSTR(wide.as_ptr());
        assert!(list.is_node(&node), "found by its name alone");
        let mut other: Vec<u16> = list_name(list.id + 1).encode_utf16().collect();
        other.push(0);
        node.pCommandListDebugNameW = windows::core::PCWSTR(other.as_ptr());
        assert!(!list.is_node(&node), "another list's name is no match");
        // With the debug layer: our name and the layer's note, in either
        // name field; and a list whose number only starts with ours is not it.
        let note = std::ffi::CString::new(format!("{} (debug layer indirect)", list_name(list.id)))
            .unwrap();
        node.pCommandListDebugNameW = windows::core::PCWSTR::null();
        node.pCommandListDebugNameA = note.as_ptr().cast();
        assert!(
            list.is_node(&node),
            "found by its ANSI name with the layer's note"
        );
        let longer =
            std::ffi::CString::new(format!("{}0 (debug layer indirect)", list_name(list.id)))
                .unwrap();
        node.pCommandListDebugNameA = longer.as_ptr().cast();
        assert!(!list.is_node(&node), "a number that only starts with ours is no match");
    }

    /// Submitting the same `HostList` twice in one `Host::execute` call must
    /// split into separate `ExecuteCommandLists` batches separated by a fence
    /// wait so neither the `prelude` allocator is clobbered nor the same direct
    /// command list is submitted twice in a single `ExecuteCommandLists` call.
    #[test]
    fn duplicate_list_in_execute_uses_distinct_preludes() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let null = std::ptr::null();
        let res_iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        // 16x16 RGBA8 render target starting in COMMON (0).
        let tex_desc: [u32; 14] = [
            3, 0, // dimension=2D, pad
            0, 0, // alignment=0
            16, 0, // width=16
            16,    // height=16
            (1 << 16) | 1,
            28, 1, 0, 0, 1, // RGBA8_UNORM, flags=ALLOW_RENDER_TARGET
            0,
        ];
        let mut tex = null;
        assert_eq!(
            crate::resource::create_at_address(
                null,
                0x3_0000,
                tex_desc.as_ptr().cast(),
                0,
                null.cast(),
                &res_iid,
                &mut tex,
            ),
            crate::object::S_OK
        );
        let list = host.new_list().unwrap();
        // Record RENDER_TARGET (0x4) -> COPY_SOURCE (0x800) inside `list`.
        assert!(list.transition(tex, 0x4).is_none());
        let (host_tex, before) = list.transition(tex, 0x800).unwrap();
        assert_eq!(before, 0x4);
        // SAFETY: open list and live texture.
        unsafe {
            list.recording().ResourceBarrier(&[transition(
                host_tex,
                D3D12_RESOURCE_STATES(0x4),
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
        }
        list.close();

        // Execute the SAME `HostList` twice in one submission:
        //   Chunk 1 prelude: COMMON (0) -> RENDER_TARGET (0x4), list ends at COPY_SOURCE (0x800), fence wait.
        //   Chunk 2 prelude: COPY_SOURCE (0x800) -> RENDER_TARGET (0x4), list ends at COPY_SOURCE (0x800), fence wait.
        host.execute(&[&list, &list]);
        assert_eq!(
            crate::resource::host_copy(tex)
                .unwrap()
                .state
                .load(Ordering::Relaxed),
            0x800
        );

        // Distinct-list batching control: two distinct lists stay in one chunk.
        let list2 = host.new_list().unwrap();
        list2.close();
        assert_eq!(unique_run_chunks(&[&list, &list2]).len(), 1);
        assert_eq!(unique_run_chunks(&[&list, &list]).len(), 2);
        host.execute(&[&list, &list2]);
        // SAFETY: plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());

        type Release = extern "system" fn(crate::object::This) -> u32;
        crate::object::slot::<Release>(tex, 0x10)(tex);
    }

    #[test]
    fn unique_run_chunks_preserves_distinct_batching_and_splits_repeats() {
        let (a, b, c) = (1u8, 2u8, 3u8);
        assert_eq!(unique_run_chunks::<u8>(&[]).len(), 0);
        assert_eq!(unique_run_chunks(&[&a, &b, &c]).len(), 1);
        let dup_in = [&a, &a];
        let dup = unique_run_chunks(&dup_in);
        assert_eq!(dup.len(), 2);
        assert_eq!((dup[0].len(), dup[1].len()), (1, 1));
        let mixed_in = [&a, &b, &a, &c, &b];
        let mixed = unique_run_chunks(&mixed_in);
        assert_eq!(mixed.len(), 2);
        assert_eq!((mixed[0].len(), mixed[1].len()), (2, 3));
    }
}

/// Partitions `runs` into maximal contiguous chunks in which no list pointer
/// repeats. Distinct lists remain batched in one `ExecuteCommandLists` call,
/// while a repeated list waits on the GPU fence before its prelude allocator is
/// reset and its body is resubmitted.
fn unique_run_chunks<'a, T>(runs: &'a [&'a T]) -> Vec<&'a [&'a T]> {
    let mut chunks = Vec::new();
    let mut start = 0;
    let mut seen: Vec<usize> = Vec::with_capacity(runs.len());
    for (i, item) in runs.iter().enumerate() {
        let id = std::ptr::from_ref(*item) as usize;
        if seen.contains(&id) {
            chunks.push(&runs[start..i]);
            start = i;
            seen.clear();
        }
        seen.push(id);
    }
    if start < runs.len() {
        chunks.push(&runs[start..]);
    }
    chunks
}

/// A host command list and its allocator, always used together.
pub(crate) struct HostList {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    /// Closed state and whether a caller could have recorded native work.
    recording_flags: std::sync::atomic::AtomicU8,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices),
    /// reused after a reset.
    scratch: Mutex<Scratch>,
    /// Host texture states the list assumes, by host copy: at its start and
    /// at its end. Lists are recorded on many threads and run in submission
    /// order, so the transitions from a texture's real state are made at
    /// submission, in `prelude`. Holding the copies keeps every texture the
    /// list names alive until it is reset, even if the title releases the
    /// resource first.
    states: Mutex<States>,
    /// Depth textures with a writable twin the list uses (`crate::twin`),
    /// by host copy: the side its first use needs and the side it leaves
    /// newer.
    twins: Mutex<std::collections::HashMap<usize, crate::twin::Use>>,
    /// A copy of ours replaced the list's compute root signature and
    /// pipeline: draws and dispatches set theirs again.
    rebind: std::sync::atomic::AtomicBool,
    /// The list that runs those transitions just before this one.
    prelude: (ID3D12CommandAllocator, ID3D12GraphicsCommandList),
    /// The pipelines the list binds, for reports of slow submissions.
    pipelines: Mutex<std::collections::BTreeSet<String>>,
    /// Which list this is, for the name that tells it apart in device-removed
    /// data (see [`HostList::is_node`]).
    id: u64,
    /// The work its indirect executions may still do (`indirect`): a
    /// shader-writable word, zeroed when the list is submitted.
    budget: OnceLock<Option<(ID3D12Resource, usize)>>,
    /// Where its indirect executions clamp records copied out of title
    /// memory (`crate::indirect::copied`), its size and whether it was
    /// used yet (between uses it is in the unordered-access state).
    arguments: Mutex<Option<(ID3D12Resource, u64, bool)>>,
    /// The texts of the markers `note_op` set, in order (for device-removed
    /// data without marker texts), with the host GPU range a copy reads.
    #[cfg(feature = "diagnostics")]
    notes: Mutex<Vec<Note>>,
    /// In a frame `crate::capture` lists: the textures the list renders to
    /// or writes, and the pipeline that first wrote each (diagnostics).
    #[cfg(feature = "diagnostics")]
    targets: Mutex<Vec<(usize, String)>>,
    /// With `VICTORY_GPU_TIMING`: a GPU timestamp before each noted command
    /// and at the end, and where they are read back.
    #[cfg(feature = "diagnostics")]
    stamps: OnceLock<Option<(ID3D12QueryHeap, ID3D12Resource)>>,
}

/// Timestamps per list (diagnostics); commands past them are not timed.
#[cfg(feature = "diagnostics")]
const STAMPS: u32 = 4096;

/// A marker's text and the host range its command reads (diagnostics).
#[cfg(feature = "diagnostics")]
type Note = (String, Option<(u64, u64)>);

type States = std::collections::HashMap<
    usize,
    (u32, u32, std::sync::Arc<crate::resource::HostTexture>),
>;

/// Whether device-removed data is on (`VICTORY_D3D12_DRED`, diagnostics
/// builds).
pub(crate) fn dred_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_DRED"))
}

/// Whether each list of a submission runs and is timed on its own, slow
/// ones logged with their commands (`VICTORY_GPU_TIMING`, diagnostics
/// builds; the commands are noted with `VICTORY_D3D12_DRED` on).
#[cfg(feature = "diagnostics")]
fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_GPU_TIMING"))
}

/// Names a host object for the device-removed report, which names the
/// allocations a page fault hit (only with device-removed data on).
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    if dred_on()
        && let Ok(object) = object.cast::<ID3D12Object>()
    {
        // SAFETY: a plain call on a live object.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text())) };
    }
}

/// The id of the next host list (see [`HostList::is_node`]).
static NEXT_LIST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// The name a host list gets with device-removed data on.
fn list_name(id: u64) -> String {
    format!("victory host list {id}")
}

/// The pointers one of our list objects answers to: its own and the one each
/// interface a runtime may record it by gives (COM identity is the
/// `IUnknown` one). Only our live object is queried: a node of the
/// device-removed data may name a list that is long released, so its pointer
/// is compared and never dereferenced.
fn identities(list: &impl Interface) -> Vec<usize> {
    let mut found = vec![list.as_raw() as usize];
    let mut add = |pointer: Option<*mut core::ffi::c_void>| {
        if let Some(pointer) = pointer {
            found.push(pointer as usize);
        }
    };
    add(list
        .cast::<windows::core::IUnknown>()
        .ok()
        .map(|i| i.as_raw()));
    add(list.cast::<ID3D12Object>().ok().map(|i| i.as_raw()));
    add(list.cast::<ID3D12DeviceChild>().ok().map(|i| i.as_raw()));
    add(list.cast::<ID3D12CommandList>().ok().map(|i| i.as_raw()));
    found.sort_unstable();
    found.dedup();
    found
}

/// Whether a node of the device-removed data is a list: by pointer, or by the
/// name the runtime copied when the list was submitted. With the debug layer
/// on, the object we hold is its wrapper and the runtime records another
/// one, so no pointer of ours matches, and it appends its own note to our
/// name ("victory host list 158 (debug layer indirect)"). The name matches
/// exactly or followed by " (": never as the start of another number.
fn node_is_list(
    node_pointer: Option<usize>,
    node_name: Option<&str>,
    pointers: &[usize],
    name: &str,
) -> bool {
    node_pointer.is_some_and(|p| pointers.contains(&p))
        || node_name.is_some_and(|n| {
            n == name
                || n.strip_prefix(name)
                    .is_some_and(|rest| rest.starts_with(" ("))
        })
}

/// Upload chunks handed out front to back: the chunks, the one in use and
/// the bytes used in it.
#[derive(Default)]
struct Scratch {
    chunks: Vec<ScratchChunk>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

const SCRATCH_CHUNK: u64 = 4 << 20;
const FIRST_SCRATCH_CHUNK: u64 = 64 << 10;

static IDLE_UPLOAD: Mutex<Vec<ScratchChunk>> = Mutex::new(Vec::new());

fn limit_upload(pool: &mut Vec<ScratchChunk>) -> usize {
    let limit = crate::working_set::limit(crate::working_set::Cache::ListUpload);
    let mut bytes = pool.iter().map(|chunk| chunk.capacity as usize).sum::<usize>();
    while bytes > limit {
        bytes -= pool.pop().expect("nonempty when over limit").capacity as usize;
    }
    bytes
}

pub(crate) fn trim_idle_upload() -> usize {
    limit_upload(&mut IDLE_UPLOAD.lock().unwrap_or_else(|e| e.into_inner()))
}

struct ScratchChunk {
    _buffer: ID3D12Resource,
    mapped: usize,
    gpu: u64,
    capacity: u64,
}

/// One piece of submission-time work, in recorded order.
enum Step {
    /// A copy into title memory (destination, source, bytes): the title's
    /// buffer copies, which later refreshes must see.
    Copy(crate::copy_plan::GuestCopy),
    /// An operation lowered into the page renderer's ordered command stream.
    Page(crate::page_queue::PageOp),
    /// A copy of title memory into the list's own upload memory.
    Fill(usize, usize, usize),
    /// Brings a range (address, bytes) of a buffer's host copy up to date:
    /// once per submission, whichever lists ask, until a `Copy` or `Run`
    /// intervenes.
    Refresh(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Work that may write title memory.
    Run(Box<dyn Fn() + Send>),
    /// Work that writes host memory only.
    Host(Box<dyn Fn() + Send>),
    /// Shaders write a range (address, bytes) of a buffer's host copy;
    /// whether they mark the pages they write (no snapshot then).
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize, bool),
    /// Brings a texture's staging buffer up to date: once per submission,
    /// whichever lists sample it, until a `Copy` or `Run` intervenes.
    Texture(std::sync::Arc<crate::texture::Staging>, std::sync::Arc<crate::texture::Upload>),
    /// Brings every buffer range a view heap's views reach up to date
    /// (`crate::bindless`).
    Heap(std::sync::Arc<crate::bindless::Viewed>),
    /// Shaders read title memory through a view heap's page table (the
    /// heap's key, `crate::memory`): the table names every copy, each
    /// brought up to date.
    Memory(usize),
}

type AfterSubmission = Box<dyn Fn(&crate::title_backing::SubmissionBacking) + Send>;

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload buffers its GPU commands read, kept until the list is reset.
#[derive(Default)]
struct Deferred {
    /// Destination, source, bytes.
    copies: Vec<Step>,
    uploads: Vec<ID3D12Resource>,
    transient_texture_uploads: Vec<std::sync::Arc<crate::texture::Upload>>,
    /// Host objects the list's commands name (pipelines, root signatures,
    /// descriptor heaps): the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
    /// In-place resources need the backing and its budget lease until reset.
    mapped_shadows: std::collections::HashMap<usize, std::sync::Arc<crate::shadow::Shadow>>,
    /// Buffer ranges refreshed since the last title copy (host copy,
    /// address, bytes): refreshing them again would find the same bytes.
    refreshed: std::collections::HashSet<(usize, usize, usize)>,
    /// Textures whose contents the list brings up to date.
    textures: std::collections::HashSet<usize>,
    /// Bottom-of-pipe writes, done once the list has run: address, what,
    /// bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// Whether a step zeroes the indirect-work budget.
    budget_zeroed: bool,
    /// Work done once the list has run, before the bottom-of-pipe writes
    /// (title memory shaders wrote at addresses they computed goes back).
    after: Vec<AfterSubmission>,
    /// Asked once the list has run, before anything goes back: whether its
    /// shaders read title memory with no copy (`crate::memory`); then what
    /// they wrote through views does not go back either.
    missed: Vec<Box<dyn Fn() -> bool + Send>>,
}

impl Host {
    // Identity-preserving, monotonic additions to a reserved BC texture's
    // tile map. Recording/refresh ordering requires the same direct queue;
    // a refreshing submission already holds the submission mutex.
    pub(crate) fn map_sparse(
        &self,
        texture: &ID3D12Resource,
        first: u32,
        heap: &ID3D12Heap,
        count: u32,
    ) {
        let start = D3D12_TILED_RESOURCE_COORDINATE {
            Subresource: first,
            ..Default::default()
        };
        let region = D3D12_TILE_REGION_SIZE {
            NumTiles: count,
            ..Default::default()
        };
        // SAFETY: Map one region: all parameter arrays have one entry, and the caller supplies
        // the live reserved texture and a backing heap sized for this tile range.
        unsafe {
            self.queue.UpdateTileMappings(
                texture,
                1,
                Some(&start),
                Some(&region),
                heap,
                1,
                Some(&D3D12_TILE_RANGE_FLAG_NONE),
                Some(&0),
                Some(&count),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
    }
    /// Whether the host runs mesh shaders at shader model 6.6 (translated
    /// mesh shaders need both); logged once.
    pub(crate) fn mesh_shaders(&self) -> bool {
        static MESH: OnceLock<bool> = OnceLock::new();
        *MESH.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS7,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_SHADER_MODEL,
                            (&raw mut model).cast(),
                            size_of_val(&model) as u32,
                        )
                        .is_ok()
            };
            let mesh = ok
                && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
                && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
            tracing::info!(mesh, "host mesh shader support");
            mesh
        })
    }

    /// Whether the host runs 64-bit atomics on typed resources in the
    /// descriptor heap (translated 64-bit image atomics); logged once.
    pub(crate) fn heap_atomics64(&self) -> bool {
        static ATOMICS: OnceLock<bool> = OnceLock::new();
        *ATOMICS.get_or_init(|| {
            let mut typed = D3D12_FEATURE_DATA_D3D12_OPTIONS9::default();
            let mut heap = D3D12_FEATURE_DATA_D3D12_OPTIONS11::default();
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS9,
                        (&raw mut typed).cast(),
                        size_of_val(&typed) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_D3D12_OPTIONS11,
                            (&raw mut heap).cast(),
                            size_of_val(&heap) as u32,
                        )
                        .is_ok()
            };
            let atomics = ok
                && typed.AtomicInt64OnTypedResourceSupported.as_bool()
                && heap.AtomicInt64OnDescriptorHeapResourceSupported.as_bool();
            tracing::info!(atomics, "host 64-bit atomics on typed heap resources");
            atomics
        })
    }

    /// A new list, open for recording.
    pub(crate) fn new_list(&self) -> windows::core::Result<HostList> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            let prelude_allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let prelude: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &prelude_allocator,
                None,
            )?;
            prelude.Close()?;
            let id = NEXT_LIST.fetch_add(1, Ordering::Relaxed);
            name(&list, || list_name(id));
            Ok(HostList {
                allocator,
                list,
                recording_flags: 0.into(),
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                states: Mutex::default(),
                twins: Mutex::default(),
                rebind: false.into(),
                prelude: (prelude_allocator, prelude),
                pipelines: Mutex::default(),
                id,
                budget: OnceLock::new(),
                arguments: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                notes: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                targets: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
            })
        }
    }

    fn memory_adapter(&self) -> Option<&IDXGIAdapter3> {
        self.memory_adapter.get_or_init(|| {
            // SAFETY: query the adapter that owns this live device.
            unsafe {
                // Don't add a local GPU segment to physical RAM on UMA.
                let mut arch = D3D12_FEATURE_DATA_ARCHITECTURE::default();
                self.device.CheckFeatureSupport(
                    D3D12_FEATURE_ARCHITECTURE,
                    (&raw mut arch).cast(),
                    size_of_val(&arch) as u32,
                ).ok()?;
                if arch.UMA.as_bool() {
                    return None;
                }
                CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                    .and_then(|f| f.EnumAdapterByLuid(self.device.GetAdapterLuid()))
                    .ok()
            }
        }).as_ref()
    }

    pub(crate) fn local_memory_for_admission(&self) -> Option<(u64, u64)> {
        let info = crate::working_set::query_local(self.memory_adapter()?)?;
        Some((info.CurrentUsage, info.Budget))
    }

    pub(crate) fn completed_fence(&self) -> Option<victory_memory::coherence::Fence> {
        // SAFETY: a plain query on the direct queue's live fence.
        let completed = unsafe { self.fence.GetCompletedValue() };
        (completed != u64::MAX).then_some(victory_memory::coherence::Fence(completed))
    }

    /// Signal once for an entire native list group. The caller holds the
    /// submission lock across preparation and physical writeback.
    pub(crate) fn run_native_group(
        &self, submitted: &mut u64, group: &[Option<ID3D12CommandList>],
    ) -> windows::core::Result<victory_memory::coherence::Fence> {
        *submitted = submitted.checked_add(1).filter(|&v| v != u64::MAX)
            .ok_or_else(|| windows::core::Error::from(windows::Win32::Foundation::E_FAIL))?;
        // SAFETY: callers supply closed lists and serialize the device event
        // and queue with the submission lock.
        unsafe {
            self.queue.ExecuteCommandLists(group);
            self.queue.Signal(&self.fence, *submitted)?;
            self.fence.SetEventOnCompletion(*submitted, self.event)?;
            if WaitForSingleObject(self.event, INFINITE) != WAIT_OBJECT_0 {
                return Err(windows::core::Error::from_thread());
            }
        }
        Ok(victory_memory::coherence::Fence(*submitted))
    }

    pub(crate) fn direct_queue(&self) -> std::sync::MutexGuard<'_, u64> {
        self.submitted.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Runs closed lists and waits for them.
    pub(crate) fn execute(&self, runs: &[&HostList]) {
        crate::working_set::sample(self.memory_adapter());
        // One submission at a time from preparation to copy-back: snapshots
        // and the ranges shaders write are per host copy, not per submission.
        let mut submitted = self.direct_queue();
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let mut faults = crate::copystat::page_faults();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        // Descriptors the title copied into its heaps itself, before the
        // heaps' views are brought up to date below.
        crate::descriptor::sync_heaps();
        // Textures heap-indexing shaders may read, uploaded first by a list
        // of ours where their title memory changed.
        let heaps: Vec<std::sync::Arc<crate::bindless::Viewed>> = runs
            .iter()
            .flat_map(|l| {
                l.deferred()
                    .copies
                    .iter()
                    .filter_map(|s| match s {
                        Step::Heap(v) => Some(v.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        #[cfg(feature = "diagnostics")]
        let uploading = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::UploadsUs);
        let uploads = (!heaps.is_empty())
            .then(|| crate::bindless::uploads(&heaps))
            .flatten();
        #[cfg(feature = "diagnostics")]
        drop(uploading);
        let runs: Vec<&HostList> = uploads
            .as_ref()
            .map(|u| u.list())
            .into_iter()
            .chain(runs.iter().copied())
            .collect();
        let runs = &runs[..];
        // A frame `crate::capture` lists: every command of every list, in
        // the order the GPU runs them (with `VICTORY_D3D12_DRED`, which
        // notes them).
        #[cfg(feature = "diagnostics")]
        let traced = crate::capture::listed(crate::queue::presented() + 1);
        #[cfg(feature = "diagnostics")]
        if traced {
            for (i, r) in runs.iter().enumerate() {
                let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                for (k, (note, _)) in notes.iter().enumerate() {
                    tracing::info!(
                        target: "victory_gpu::trace",
                        "s{} l{i}/{} #{k}: {note}",
                        *submitted + 1,
                        runs.len()
                    );
                }
            }
        }
        // Split only when a `HostList` repeats so distinct lists stay batched in
        // a single `ExecuteCommandLists` call, while a repeated direct list
        // waits on the GPU fence before its prelude allocator is reset and its
        // body is resubmitted (per D3D12 ExecuteCommandLists ordering rules).
        let chunks = unique_run_chunks(runs);
        let mut lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2 * runs.len());
        // Where each run's lists start in `lists`.
        let mut firsts = Vec::with_capacity(runs.len());
        let mut prep_us = started.elapsed().as_micros() as u64;
        let mut gpu = std::time::Duration::ZERO;
        let mut removed_in = None;
        let multiple_chunks = chunks.len() > 1;
        let mut chunk_base = 0usize;
        // Keep the physical backing through all fences AND writeback, even
        // across repeated-list chunks, without extending the list's life.
        let mut backing = crate::title_backing::SubmissionBacking::default();
        for chunk in chunks {
            let chunk_start = std::time::Instant::now();
            lists.clear();
            firsts.clear();
            let mut batch = Batch { backing, ..Default::default() };
            for list in chunk {
                list.run_deferred(&mut batch);
                firsts.push(lists.len());
                #[cfg(feature = "diagnostics")]
                let _timed =
                    crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::PreludeUs);
                if let Some(prelude) = list.prelude() {
                    lists.push(Some(prelude));
                }
                lists.push(list.list.cast().ok());
            }
            batch.flush();
            backing = batch.backing;
            let prepared = std::time::Instant::now();
            prep_us += (prepared - chunk_start).as_micros() as u64;
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(
                crate::copystat::C::FaultsPrep,
                crate::copystat::page_faults().saturating_sub(faults),
            );
            // Diagnostics: each list on its own, timed (`VICTORY_GPU_TIMING`).
            #[cfg(feature = "diagnostics")]
            let groups: Vec<&[Option<ID3D12CommandList>]> = if timing() {
                firsts
                    .iter()
                    .zip(firsts.iter().skip(1).chain([&lists.len()]))
                    .map(|(&a, &b)| &lists[a..b])
                    .collect()
            } else {
                vec![&lists[..]]
            };
            #[cfg(not(feature = "diagnostics"))]
            let groups = [&lists[..]];
            for (i, group) in groups.iter().enumerate() {
                let group_started = std::time::Instant::now();
                #[cfg(feature = "diagnostics")]
                crate::resource::life::note(0, crate::resource::life::SUBMITTED, *submitted + 1, 0);
                if let Err(e) = self.run_native_group(&mut submitted, group) {
                    fail(format_args!(
                        "host submission fence wait failed: {e}{}",
                        self.removal_data(runs)
                    ));
                }
                #[cfg(feature = "diagnostics")]
                crate::resource::life::note(0, crate::resource::life::RAN, *submitted, 0);
                #[cfg(feature = "diagnostics")]
                if groups.len() > 1 && group_started.elapsed() > std::time::Duration::from_millis(50)
                {
                    // SAFETY: a plain query.
                    let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                    let run_idx = chunk_base + i;
                    let commands = runs[run_idx]
                        .notes
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .len();
                    tracing::warn!(
                        ms = group_started.elapsed().as_millis(),
                        commands,
                        "slow host list"
                    );
                    for (ms, op) in runs[run_idx].slowest(frequency, 6) {
                        tracing::warn!("slow list command, {ms:.1} ms: {op}");
                    }
                }
                if groups.len() > 1 && removed_in.is_none() {
                    // SAFETY: a plain query.
                    if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                        removed_in = Some(chunk_base + i);
                    }
                }
                let _ = (i, group_started);
            }
            gpu += prepared.elapsed();
            chunk_base += chunk.len();
            if multiple_chunks && chunk_base < runs.len() {
                // Do not prepare another list on a removed device.
                // SAFETY: a plain query, after the fence wait.
                if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                    break;
                }
                #[cfg(feature = "diagnostics")]
                {
                    faults = crate::copystat::page_faults();
                }
            }
        }
        let _ = &firsts;
        SUBMISSION_US[0].fetch_add(prep_us, Ordering::Relaxed);
        SUBMISSION_US[1].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
        // The host resets a device whose submission runs for about 2 s.
        if gpu > std::time::Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            tracing::warn!(ms = gpu.as_millis(), lists = runs.len(), ?pipelines, "slow host submission");
        }
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            fail(format_args!(
                "host GPU device removed after a submission of {} lists{}{}{}",
                runs.len(),
                self.removal_data(runs),
                last_notes(runs),
                removed_during(runs, removed_in)
            ));
        }
        // What shaders wrote goes back before the bottom-of-pipe writes,
        // which tell the title the GPU is done; each range once per
        // submission, however many dispatches declared it (the world's
        // passes declare the same big ranges dozens of times).
        let copying = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let faults = crate::copystat::page_faults();
        // One list's lock at a time: a submission may name a list twice.
        let mut back: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, bool)> = Vec::new();
        let mut index: std::collections::HashMap<(usize, usize, usize), usize> = std::collections::HashMap::new();
        for list in runs {
            let deferred = list.deferred();
            // Shaders that read memory with no copy computed something else
            // than the console's: it stays out of title memory, where it broke
            // the title's own data (runs c1, c4).
            let missed = deferred.missed.iter().any(|check| check());
            for step in &deferred.copies {
                if let Step::GpuWrite(shadow, address, len, _) = step {
                    let key = (std::sync::Arc::as_ptr(shadow) as usize, *address, *len);
                    match index.get(&key) {
                        Some(&i) => back[i].3 &= !missed,
                        None => {
                            index.insert(key, back.len());
                            back.push((shadow.clone(), *address, *len, !missed));
                        }
                    }
                }
            }
        }
        // Pages shaders marked first: the copy-backs below end what the
        // copies keep for them.
        let withheld: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = back
            .iter()
            .filter(|b| !b.3)
            .map(|(s, a, n, _)| (s.clone(), *a, *n))
            .collect();
        crate::shadow::copy_back_marks(&withheld, &backing);
        for (shadow, address, len, goes_back) in &back {
            let (address, len) = (*address, *len);
            if *goes_back {
                let written = shadow.copy_back(address, len, &backing);
                COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            } else {
                shadow.keep_writes(address, len, &backing);
                crate::memory::note_withheld(len as u64);
            }
        }
        for list in runs {
            for work in &list.deferred().after {
                work(&backing);
            }
        }
        COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::FaultsBack, crate::copystat::page_faults().saturating_sub(faults));
        for list in runs {
            for &(at, what, bytes) in &list.deferred().writes {
                what.write(at as *mut u8, bytes);
            }
        }
        drop(backing);
        // The submission has run and the next waits for our lock.
        release_retired();
        self.log_messages();
        // A listed frame: what this submission's lists drew, as it is now
        // (the dumps submit lists of their own).
        #[cfg(feature = "diagnostics")]
        if traced {
            let sub = *submitted;
            let mut targets: Vec<(usize, String)> = Vec::new();
            for r in runs {
                for t in r.targets.lock().unwrap_or_else(|e| e.into_inner()).iter() {
                    if !targets.iter().any(|x| x.0 == t.0) {
                        targets.push(t.clone());
                    }
                }
            }
            // The capture can submit work that takes both locks.
            drop(uploads);
            drop(submitted);
            crate::capture::dump_submission(sub, &targets);
        }
    }

    /// With `VICTORY_D3D12_DRED`: the faulting address and the allocations
    /// it lies in (by the names we give them) or the title memory copy just
    /// below it; for each list that did not finish, the command it stopped
    /// at and what the draws and dispatches in flight there run (or, in
    /// lists that never started, the pipelines they bind).
    fn removal_data(&self, runs: &[&HostList]) -> String {
        let Ok(dred) = self.device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
            return String::new();
        };
        let mut text = String::new();
        let mut fault_va = None;
        // SAFETY: the outputs are the runtime's, valid while the device
        // lives; nodes are walked as linked, histories have their counts.
        unsafe {
            if let Ok(fault) = dred.GetPageFaultAllocationOutput1()
                && fault.PageFaultVA != 0
            {
                let va = fault.PageFaultVA;
                fault_va = Some(va);
                text.push_str(&format!("; page fault at {va:#x}"));
                for (what, head) in [
                    ("inside", fault.pHeadExistingAllocationNode),
                    ("inside freed", fault.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 8
                    {
                        let name = if n.ObjectNameW.is_null() {
                            "unnamed".to_owned()
                        } else {
                            n.ObjectNameW.to_string().unwrap_or_default()
                        };
                        names.push(format!("{name} (type {})", n.AllocationType.0));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        text.push_str(&format!("; {what} {names:?}"));
                    }
                    #[cfg(feature = "diagnostics")]
                    for name in names.iter().filter(|_| what == "inside freed") {
                        text.push_str(&crate::resource::life::of(name));
                    }
                }
                text.push_str(&crate::shadow::near(va));
                text.push_str(&crate::descriptor::host::near_heaps(va));
            }
            // The plain output: the extended one (with marker texts) came
            // back without stopped lists on this host.
            let crumbs = match dred.GetAutoBreadcrumbsOutput() {
                Ok(crumbs) => crumbs,
                Err(e) => {
                    text.push_str(&format!("; no breadcrumbs ({e})"));
                    return text;
                }
            };
            // The nodes, oldest list first, and which run each is.
            let mut nodes = Vec::new();
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = node.as_ref() {
                nodes.push(p);
                node = p.pNext;
            }
            let index: Vec<Option<usize>> = nodes
                .iter()
                .map(|p| runs.iter().position(|r| r.is_node(p)))
                .collect();
            if let Some(last) = nodes.last()
                && index.iter().all(Option::is_none)
            {
                let name = |w: windows::core::PCWSTR, a: *const u8| {
                    let w = (!w.is_null()).then(|| w.to_string().unwrap_or_default());
                    let a = (!a.is_null()).then(|| {
                        std::ffi::CStr::from_ptr(a.cast())
                            .to_string_lossy()
                            .into_owned()
                    });
                    format!("{w:?}/{a:?}")
                };
                tracing::warn!(
                    nodes = nodes.len(),
                    last_pointer = ?last.pCommandList.as_ref().map(|l| l.as_raw()),
                    last_names = %name(last.pCommandListDebugNameW, last.pCommandListDebugNameA),
                    "no breadcrumb node named one of our lists"
                );
            }
            // Every list's progress (done/count), in the runtime's order.
            let progress: Vec<String> = nodes
                .iter()
                .zip(&index)
                .map(|(p, index)| {
                    let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                    format!("{done}/{} {index:?}", p.BreadcrumbCount)
                })
                .collect();
            text.push_str(&format!("; breadcrumbs (done/count, list) {progress:?}"));
            for (p, index) in nodes.iter().zip(&index) {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                if done < p.BreadcrumbCount && !p.pCommandHistory.is_null() {
                    let history =
                        std::slice::from_raw_parts(p.pCommandHistory, p.BreadcrumbCount as usize);
                    text.push_str(&format!(
                        "; a list stopped at command {done} of {} (op {})",
                        p.BreadcrumbCount, history[done as usize].0
                    ));
                    let run = index.map(|i| runs[i]);
                    // The k-th marker command is the k-th note.
                    #[cfg(feature = "diagnostics")]
                    let markers: Vec<(u32, String)> = run.map_or_else(Vec::new, |r| {
                        let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                        history
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
                            .zip(notes.iter())
                            .map(|((at, _), (note, range))| {
                                let place = fault_va.zip(*range).map(|(va, r)| fault_in(va, r));
                                (at as u32, format!("{note}{}", place.unwrap_or_default()))
                            })
                            .collect()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    let markers: Vec<(u32, String)> = Vec::new();
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = fault_va;
                    if done > 0 {
                        text.push_str(&in_flight(history, &markers, done as usize));
                    } else if let Some(r) = run {
                        let pipelines = r.pipelines.lock().unwrap_or_else(|e| e.into_inner());
                        text.push_str(&format!(", its pipelines {pipelines:?}"));
                    }
                }
            }
        }
        text
    }

    /// Moves the debug layer's messages into our log; returns the first
    /// error among them.
    pub(crate) fn log_messages(&self) -> Option<String> {
        let Some(queue) = &self.messages else {
            return None;
        };
        let mut first_error = None;
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    let text = String::from_utf8_lossy(text);
                    let text = text.trim_end_matches('\0');
                    tracing::warn!(id = m.ID.0, severity = m.Severity.0, "D3D12: {text}");
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && first_error.is_none() {
                        first_error = Some(text.to_owned());
                    }
                }
            }
            queue.ClearStoredMessages();
        }
        first_error
    }

    /// A host render target view of `resource`, valid until it is recorded.
    pub(crate) fn rtv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.rtv_size) as usize;
            self.device
                .CreateRenderTargetView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }

    /// A slot of CPU-only views, valid until it is recorded.
    pub(crate) fn cpu_view(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_cpu_view.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a live heap.
        let mut handle = unsafe { self.cpu_views.GetCPUDescriptorHandleForHeapStart() };
        handle.ptr += (slot * self.cpu_view_size) as usize;
        handle
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.dsvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.dsv_size) as usize;
            self.device
                .CreateDepthStencilView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }
}

const NATIVE_RECORDING: u8 = 1;
const CLOSED_RECORDING: u8 = 2;

impl HostList {
    /// Borrow the writable native body. A list that exposes it can no
    /// longer prove that all its work is present in the title-copy stream.
    pub(crate) fn recording(&self) -> &ID3D12GraphicsCommandList {
        if self.recording_flags.load(Ordering::Relaxed) & NATIVE_RECORDING == 0 {
            self.recording_flags.fetch_or(NATIVE_RECORDING, Ordering::Relaxed);
        }
        &self.list
    }

    pub(crate) fn mark_unlowered(&self) {
        self.recording_flags.fetch_or(NATIVE_RECORDING, Ordering::Relaxed);
    }

    /// Admit the complete closed page list or none of it. Native work,
    /// deferred uploads/refreshes and fence callbacks need their own
    /// ordered lowering before this gate can accept them.
    pub(crate) fn append_closed_page_ops(
        &self,
        ops: &mut Vec<crate::page_queue::PageOp>,
    ) -> bool {
        if self.recording_flags.load(Ordering::Acquire) != CLOSED_RECORDING {
            return false;
        }
        if !self.states.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
            || !self.twins.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
        {
            return false;
        }
        let deferred = self.deferred();
        if !deferred.writes.is_empty()
            || !deferred.after.is_empty()
            || !deferred.missed.is_empty()
            || deferred.copies.iter().any(|step| !matches!(step, Step::Copy(_) | Step::Page(_)))
        {
            return false;
        }
        ops.extend(deferred.copies.iter().filter_map(|step| match step {
            Step::Copy(copy) => Some(crate::page_queue::PageOp::Copy(*copy)),
            Step::Page(op) => Some(op.clone()),
            _ => None,
        }));
        true
    }

    /// The timestamp heap and its read-back buffer (diagnostics, timing on).
    #[cfg(feature = "diagnostics")]
    fn stamps(&self) -> Option<&(ID3D12QueryHeap, ID3D12Resource)> {
        if !timing() {
            return None;
        }
        self.stamps
            .get_or_init(|| {
                let device = &host()?.device;
                let mut heap: Option<ID3D12QueryHeap> = None;
                let mut back: Option<ID3D12Resource> = None;
                let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(STAMPS) * 8,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                // SAFETY: plain object creation.
                unsafe {
                    device
                        .CreateQueryHeap(
                            &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: STAMPS, NodeMask: 0 },
                            &mut heap,
                        )
                        .ok()?;
                    device
                        .CreateCommittedResource(
                            &readback,
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            None,
                            &mut back,
                        )
                        .ok()?;
                }
                Some((heap?, back?))
            })
            .as_ref()
    }

    /// The noted commands that took longest on the GPU, longest first, as
    /// (milliseconds, text) (diagnostics, timing on, after the list ran).
    #[cfg(feature = "diagnostics")]
    fn slowest(&self, frequency: u64, n: usize) -> Vec<(f64, String)> {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (Some((_, back)), count) = (self.stamps(), notes.len().min(STAMPS as usize - 1)) else {
            return Vec::new();
        };
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: (count + 1) * 8 };
        // SAFETY: a read-back buffer the finished list resolved into.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return Vec::new();
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), count + 1).to_vec();
            back.Unmap(0, None);
            s
        };
        let mut times: Vec<(f64, String)> = (0..count)
            .map(|i| {
                let ticks = stamps[i + 1].saturating_sub(stamps[i]);
                (ticks as f64 * 1e3 / frequency.max(1) as f64, notes[i].0.chars().take(300).collect())
            })
            .collect();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        times.truncate(n);
        times
    }

    /// Whether `node` of the device-removed data (its breadcrumbs) is this
    /// list.
    fn is_node(&self, node: &D3D12_AUTO_BREADCRUMB_NODE) -> bool {
        // SAFETY: only the pointer and the name text are read: the pointer is
        // compared, never used, and the name is the runtime's copy.
        let (pointer, wide, ansi) = unsafe {
            (
                node.pCommandList.as_ref().map(|l| l.as_raw() as usize),
                (!node.pCommandListDebugNameW.is_null())
                    .then(|| node.pCommandListDebugNameW.to_string().unwrap_or_default()),
                (!node.pCommandListDebugNameA.is_null()).then(|| {
                    std::ffi::CStr::from_ptr(node.pCommandListDebugNameA.cast())
                        .to_string_lossy()
                        .into_owned()
                }),
            )
        };
        let (pointers, name) = (identities(&self.list), list_name(self.id));
        node_is_list(pointer, wide.as_deref(), &pointers, &name)
            || node_is_list(None, ansi.as_deref(), &pointers, &name)
    }

    pub(crate) fn close(&self) {
        #[cfg(feature = "diagnostics")]
        if let Some((heap, back)) = self.stamps() {
            let count = (self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32).min(STAMPS - 1);
            // SAFETY: an open list, our heap and read-back buffer.
            unsafe {
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, count);
                self.list.ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count + 1, back, 0);
            }
        }
        // SAFETY: our list.
        if let Err(e) = unsafe { self.list.Close() } {
            fail(format_args!(
                "a host command list did not close ({e}): a command recorded into it was invalid"
            ));
        }
        self.recording_flags.fetch_or(CLOSED_RECORDING, Ordering::Release);
    }

    /// Only after the list's last submission finished (every one is waited
    /// for).
    pub(crate) fn reset(&self) {
        self.recording_flags.store(0, Ordering::Relaxed);
        *self.deferred() = Deferred::default();
        self.states.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.twins.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        self.notes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        self.targets.lock().unwrap_or_else(|e| e.into_inner()).clear();
        // SAFETY: the list is not executing.
        let reset = unsafe {
            self.allocator
                .Reset()
                .and_then(|()| self.list.Reset(&self.allocator, None))
        };
        if let Err(e) = reset {
            fail(format_args!("a host command list did not reset ({e})"));
        }
        let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let mut pool = IDLE_UPLOAD.lock().unwrap_or_else(|e| e.into_inner());
        pool.append(&mut scratch.chunks);
        limit_upload(&mut pool);
        (scratch.current, scratch.used) = (0, 0);
    }

    /// Moves a title texture's host copy to `wanted` (a host state) in this
    /// list: the host texture and the state it leaves, if that is a change.
    /// The list's first use of a texture records nothing: its prelude brings
    /// the texture there at submission.
    /// A depth texture with a writable twin (`crate::twin`) has the side
    /// the use needs brought up to date first; unordered access moves the
    /// twin, not the depth texture (nothing returned).
    pub(crate) fn transition(
        &self,
        resource: crate::object::This,
        wanted: u32,
    ) -> Option<(&'static ID3D12Resource, u32)> {
        let texture = crate::resource::host_resource(resource)?;
        let copy = crate::resource::host_copy(resource)?;
        if let Some(Some(twin)) = copy.twin.get()
            && crate::twin::route(self, &copy, twin, wanted)
        {
            return None;
        }
        let wanted = legal_state(texture, wanted);
        self.track(&copy, wanted).map(|before| (texture, before))
    }

    /// The list's state for host texture `copy` becomes `wanted` (a legal
    /// one): the state it leaves, if that is a change. Nothing for the
    /// list's first use (its prelude moves the texture there).
    fn track(&self, copy: &std::sync::Arc<crate::resource::HostTexture>, wanted: u32) -> Option<u32> {
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.entry(std::sync::Arc::as_ptr(copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                copy.note_used();
                v.insert((wanted, wanted, copy.clone()));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some(before)
            }
        }
    }

    /// [`HostList::transition`] for a host texture of ours (a twin's
    /// plane), without twins: the barrier to record, if one is needed.
    pub(crate) fn move_texture(
        &self,
        copy: &std::sync::Arc<crate::resource::HostTexture>,
        wanted: u32,
    ) -> Option<D3D12_RESOURCE_BARRIER> {
        let wanted = legal_state(&copy.texture, wanted);
        let before = self.track(copy, wanted)?;
        Some(transition(
            &copy.texture,
            D3D12_RESOURCE_STATES(before as i32),
            D3D12_RESOURCE_STATES(wanted as i32),
        ))
    }

    /// A use of the depth texture `depth` (with a twin) needing `side`
    /// (`crate::twin::DEPTH` or `TWIN`): whether a copy must be recorded
    /// now (true: into the depth texture). The list's first use is the
    /// prelude's to prepare.
    pub(crate) fn twin_use(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        side: u8,
        writes: bool,
    ) -> Option<bool> {
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        match twins.entry(std::sync::Arc::as_ptr(depth) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(crate::twin::Use::new(depth, side, writes));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => o.get_mut().again(side, writes),
        }
    }

    /// A copy of ours replaced the compute root signature and pipeline.
    pub(crate) fn need_rebind(&self) {
        self.rebind.store(true, Ordering::Relaxed);
    }

    /// Whether the list's compute root signature and pipeline must be set
    /// again (once).
    pub(crate) fn take_rebind(&self) -> bool {
        self.rebind.swap(false, Ordering::Relaxed)
    }

    /// At submission: the list that moves each texture this list uses from
    /// its real state to the one the list starts with; the textures' real
    /// states become those the list ends with.
    fn prelude(&self) -> Option<ID3D12CommandList> {
        // Depth textures and their twins first, from their real states and
        // back: the side each first use needs, made newer.
        let twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let copies = crate::twin::at_submission(twins.values());
        drop(twins);
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let barriers: Vec<D3D12_RESOURCE_BARRIER> = states
            .values()
            .filter_map(|(first, last, copy)| {
                let real = copy.state.swap(*last, Ordering::Relaxed);
                (real != *first).then(|| {
                    transition(
                        &copy.texture,
                        D3D12_RESOURCE_STATES(real as i32),
                        D3D12_RESOURCE_STATES(*first as i32),
                    )
                })
            })
            .collect();
        if barriers.is_empty() && copies.is_empty() {
            return None;
        }
        let (allocator, list) = &self.prelude;
        // SAFETY: the prelude is not executing (every submission is waited
        // for); the textures are live.
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
                for (depth, to_depth) in &copies {
                    let Some(Some(twin)) = depth.twin.get() else { continue };
                    // From each texture's real state to the copy's and back.
                    let mut back = Vec::new();
                    twin.copy(list, depth, *to_depth, &mut |t, wanted| {
                        let real = t.state.load(Ordering::Relaxed);
                        let wanted = legal_state(&t.texture, wanted);
                        (real != wanted).then(|| {
                            back.push(transition(
                                &t.texture,
                                D3D12_RESOURCE_STATES(wanted as i32),
                                D3D12_RESOURCE_STATES(real as i32),
                            ));
                            transition(&t.texture, D3D12_RESOURCE_STATES(real as i32), D3D12_RESOURCE_STATES(wanted as i32))
                        })
                    });
                    if !back.is_empty() {
                        list.ResourceBarrier(&back);
                    }
                }
                if !barriers.is_empty() {
                    list.ResourceBarrier(&barriers);
                }
                list.Close()
            })
        };
        if let Err(e) = recorded {
            fail(format_args!("a host state prelude did not record ({e})"));
        }
        list.cast().ok()
    }

    /// Notes what the command about to be recorded does, as a marker
    /// command whose text the list keeps (diagnostics, with device-removed
    /// data on): the report pairs the breadcrumbs' marker commands with the
    /// texts in order and names the commands a fault stopped among.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_op(&self, what: impl FnOnce() -> String) {
        self.note(what, None);
    }

    /// As [`HostList::note_op`], for a copy that reads the host GPU range
    /// `(address, bytes)`: the report places the faulting address in it.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_read(&self, what: impl FnOnce() -> String, range: (u64, u64)) {
        self.note(what, Some(range));
    }

    #[cfg(feature = "diagnostics")]
    fn note(&self, what: impl FnOnce() -> String, range: Option<(u64, u64)>) {
        if dred_on() {
            let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
            if let Some((heap, _)) = self.stamps() && at < STAMPS - 1 {
                // SAFETY: an open list and our query heap.
                unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
            }
            let what = what();
            traced(&what);
            let text: Vec<u16> = what.encode_utf16().chain([0]).collect();
            const UNICODE: u32 = 0;
            // SAFETY: an open list; the marker is copied when recorded.
            unsafe {
                self.list.SetMarker(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
            };
            self.notes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((what, range));
        }
    }

    /// Notes a texture the list renders to or writes, in a listed frame.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_target(&self, resource: usize, by: &str) {
        let mut targets = self.targets.lock().unwrap_or_else(|e| e.into_inner());
        if !targets.iter().any(|t| t.0 == resource) {
            targets.push((resource, by.to_owned()));
        }
    }

    /// Notes a pipeline the list binds.
    pub(crate) fn used_pipeline(&self, name: &str) {
        let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if !pipelines.contains(name) {
            pipelines.insert(name.to_owned());
        }
    }

    fn deferred(&self) -> std::sync::MutexGuard<'_, Deferred> {
        self.deferred.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies `len` bytes from `src` to `dst` when the list is submitted.
    pub(crate) fn defer_copy(&self, dst: *mut u8, src: *const u8, len: usize) {
        if let Some(copy) = crate::copy_plan::GuestCopy::new(dst as usize, src as usize, len) {
            let mut deferred = self.deferred();
            deferred.copies.push(Step::Copy(copy));
            deferred.refreshed.clear();
        }
    }

    #[allow(dead_code, reason = "closed draws require paged queue submission")]
    pub(crate) fn record_page_draw(&self, draw: crate::page_draw::ClosedDraw) {
        self.deferred().copies.push(Step::Page(crate::page_queue::PageOp::Draw(std::sync::Arc::new(draw))));
    }

    pub(crate) fn record_page_dispatch(&self, dispatch: crate::page_dispatch::ClosedDispatch) {
        self.deferred().copies.push(Step::Page(crate::page_queue::PageOp::Dispatch(std::sync::Arc::new(dispatch))));
    }

    pub(crate) fn record_page_clear(&self, clear: crate::page_draw::ColorClear) {
        self.deferred().copies.push(Step::Page(crate::page_queue::PageOp::Clear(std::sync::Arc::new(clear))));
    }

    pub(crate) fn record_page_clear_depth(&self, clear: crate::page_draw::DepthClear) {
        self.deferred().copies.push(Step::Page(crate::page_queue::PageOp::ClearDepth(std::sync::Arc::new(clear))));
    }

    pub(crate) fn record_page_texture_readback(&self, readback: crate::page_resource::TextureReadback) {
        self.deferred().copies.push(Step::Page(crate::page_queue::PageOp::ReadbackTexture(std::sync::Arc::new(readback))));
    }

    pub(crate) fn record_page_texture_upload(&self, upload: crate::page_resource::TextureUpload) {
        self.deferred().copies.push(Step::Page(crate::page_queue::PageOp::UploadTexture(std::sync::Arc::new(upload))));
    }

    pub(crate) fn record_page_texture_copy(&self, copy: crate::page_resource::TextureCopy) {
        self.deferred().copies.push(Step::Page(crate::page_queue::PageOp::CopyTexture(std::sync::Arc::new(copy))));
    }

    pub(crate) fn record_page_barrier(&self, barrier: crate::page_resource::PageBarrier) {
        self.deferred().copies.push(Step::Page(crate::page_queue::PageOp::Barrier(std::sync::Arc::new(barrier))));
    }

    pub(crate) fn record_page_raw_clear(&self, clear: crate::page_uav::RawClear) {
        self.deferred().copies.push(Step::Page(crate::page_queue::PageOp::ClearRaw(std::sync::Arc::new(clear))));
    }

    pub(crate) fn record_page_input(&self, view: crate::page_view::InputView) {
        self.deferred().copies.push(Step::Page(crate::page_queue::PageOp::Input(view)));
    }

    /// Copies `len` bytes of title memory at `src` into the list's upload
    /// memory at `dst` when the list is submitted.
    pub(crate) fn defer_fill(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.deferred()
            .copies
            .push(Step::Fill(dst as usize, src as usize, len));
    }

    /// Brings `[address, address + len)` of a buffer's host copy up to date
    /// when the list is submitted (once per range until a title copy
    /// intervenes), keeping the copy alive until the list is reset.
    pub(crate) fn defer_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.refreshed.insert(key) {
            deferred
                .copies
                .push(Step::Refresh(shadow.clone(), address, len));
        }
    }

    /// Shaders of the list write `[address, address + len)` of a buffer's
    /// host copy: copied back into title memory once the list has run
    /// (`marked`: the pages they mark).
    pub(crate) fn defer_gpu_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
        marked: bool,
    ) {
        self.deferred()
            .copies
            .push(Step::GpuWrite(shadow.clone(), address, len, marked));
    }

    /// Shaders of the list take views from the heap itself
    /// (`crate::bindless`): what the heap's views reach is brought up to
    /// date at submission (once until a title copy intervenes).
    pub(crate) fn defer_heap(&self, viewed: &std::sync::Arc<crate::bindless::Viewed>) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(viewed) as usize, usize::MAX, 0);
        if deferred.refreshed.insert(key) {
            deferred.copies.push(Step::Heap(viewed.clone()));
        }
    }

    /// Shaders of the list read title memory through the page table of
    /// view heap `heap` (`crate::memory::prepare` at submission).
    pub(crate) fn defer_memory(&self, heap: usize) {
        self.deferred().copies.push(Step::Memory(heap));
    }

    /// After the list has run, before its writes go back: `check` says
    /// whether its shaders read title memory with no copy.
    pub(crate) fn defer_missed(&self, check: impl Fn() -> bool + Send + 'static) {
        self.deferred().missed.push(Box::new(check));
    }

    /// Runs `work` once the list has run (each submission), before its
    /// bottom-of-pipe writes.
    pub(crate) fn defer_after(&self, work: impl Fn() + Send + 'static) {
        self.defer_writeback(move |_| work());
    }

    pub(crate) fn defer_writeback(
        &self, work: impl Fn(&crate::title_backing::SubmissionBacking) + Send + 'static,
    ) {
        self.deferred().after.push(Box::new(work));
    }

    /// Stores `what` (`bytes` of it) at `dst` once the list has run.
    pub(crate) fn defer_write(&self, dst: *mut u8, what: crate::queue::Written, bytes: usize) {
        self.deferred().writes.push((dst as usize, what, bytes));
    }

    /// A mapped upload buffer of `size` bytes, alive until the list is
    /// reset.
    ///
    /// shortcut: one buffer per copy; pool them once uploads are frequent.
    pub(crate) fn upload(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let (buffer, data) = host.upload_buffer(size)?;
        name(&buffer, || format!("list upload {size:#x}"));
        self.deferred().uploads.push(buffer.clone());
        Some((buffer, data))
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        let host = host()?;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let len = len.max(1).checked_next_multiple_of(256)?;
        if len > SCRATCH_CHUNK {
            let (buffer, data) = host.upload_buffer(len)?;
            name(&buffer, || format!("list scratch {len:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            self.deferred().uploads.push(buffer);
            return Some((gpu, data));
        }
        while let Some(chunk) = s.chunks.get(s.current) {
            if len <= chunk.capacity - s.used {
                break;
            }
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let next = s.chunks.last().map_or(FIRST_SCRATCH_CHUNK, |c| 2 * c.capacity);
            let capacity = len.next_multiple_of(FIRST_SCRATCH_CHUNK).max(next).min(SCRATCH_CHUNK);
            let cached = {
                let mut pool = IDLE_UPLOAD.lock().unwrap_or_else(|e| e.into_inner());
                pool.iter().position(|c| c.capacity == capacity).map(|i| pool.swap_remove(i))
            };
            let chunk = match cached {
                Some(chunk) => chunk,
                None => {
                    let (buffer, data) = host.upload_buffer(capacity)?;
                    name(&buffer, || format!("list scratch {capacity:#x}"));
                    // SAFETY: a live buffer.
                    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
                    ScratchChunk { _buffer: buffer, mapped: data as usize, gpu, capacity }
                }
            };
            #[cfg(feature = "diagnostics")]
            tracing::debug!(target: "victory_gpu::scratch", list = ?std::ptr::from_ref(self),
                chunk = s.current, requested = len, capacity, "allocated list scratch");
            s.chunks.push(chunk);
            s.used = 0;
        }
        let chunk = &s.chunks[s.current];
        let at = s.used;
        let result = (chunk.gpu + at, (chunk.mapped + at as usize) as *mut u8);
        s.used += len;
        Some(result)
    }

    /// A shader-writable buffer of the list's of at least `len` bytes for
    /// indirect arguments (`crate::indirect::copied`), and whether this is
    /// its first use (it is still in the common state). One buffer, reused
    /// by each execution in turn; a bigger one replaces it when needed (the
    /// list keeps the old one).
    pub(crate) fn arguments(&self, len: u64) -> Option<(ID3D12Resource, bool)> {
        let mut arguments = self.arguments.lock().unwrap_or_else(|e| e.into_inner());
        if arguments.as_ref().is_none_or(|a| a.1 < len) {
            if let Some((old, _, _)) = arguments.take() {
                self.keep(&old);
            }
            let size = len.max(1).checked_next_multiple_of(64 << 10)?;
            let (buffer, _) = host()?.shared_buffer(size)?;
            name(&buffer, || format!("list indirect arguments {size:#x}"));
            *arguments = Some((buffer, size, false));
        }
        let (buffer, _, used) = arguments.as_mut()?;
        Some((buffer.clone(), !std::mem::replace(used, true)))
    }

    /// The host address of the list's indirect-work budget, zeroed at each
    /// submission of the list from the first use after a reset on.
    pub(crate) fn budget(&self) -> Option<u64> {
        let (buffer, data) = self
            .budget
            .get_or_init(|| {
                let (b, d) = host()?.shared_buffer(256)?;
                Some((b, d as usize))
            })
            .as_ref()?;
        let mut deferred = self.deferred();
        if !deferred.budget_zeroed {
            deferred.budget_zeroed = true;
            let at = *data;
            // SAFETY: the list's own mapped word, written before it runs.
            deferred
                .copies
                .push(Step::Host(Box::new(move || unsafe { (at as *mut u32).write_volatile(0) })));
        }
        // SAFETY: a live buffer.
        Some(unsafe { buffer.GetGPUVirtualAddress() })
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.deferred().kept.push(object);
        }
    }

    pub(crate) fn keep_mapped_shadow(&self, copy: &std::sync::Arc<crate::shadow::Shadow>) {
        self.deferred().mapped_shadows
            .entry(std::sync::Arc::as_ptr(copy) as usize)
            .or_insert_with(|| copy.clone());
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a texture to bring up to date).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key)
    }

    /// Runs `work` when the list is submitted, in order with its copies.
    /// `work` may write title memory (refreshes after it copy again).
    pub(crate) fn defer(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Run(Box::new(work)));
        self.deferred().refreshed.clear();
    }

    /// As [`HostList::defer`], for work that writes only host memory.
    pub(crate) fn defer_host(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Host(Box::new(work)));
    }

    /// Brings a texture's staging buffer up to date when the list is
    /// submitted, keeping it alive until the list is reset.
    pub(crate) fn defer_texture(
        &self,
        staging: std::sync::Arc<crate::texture::Staging>,
        upload: std::sync::Arc<crate::texture::Upload>,
    ) {
        self.deferred().copies.push(Step::Texture(staging, upload));
    }

    pub(crate) fn keep_upload_until_fence(&self, upload: std::sync::Arc<crate::texture::Upload>) {
        self.deferred().transient_texture_uploads.push(upload);
    }

    pub(crate) fn release_fenced_uploads(&self) {
        self.deferred().transient_texture_uploads.clear();
    }

    /// Runs the list's submission-time steps; refreshes go into `batch`
    /// (the title's big buffers, a 174 MB one among them, are read by most
    /// lists of a submission, through many ranges).
    fn run_deferred(&self, batch: &mut Batch) {
        #[cfg(feature = "diagnostics")]
        use crate::copystat::{C, add};
        for step in &self.deferred().copies {
            match step {
                Step::Page(crate::page_queue::PageOp::Input(_)) => continue,
                Step::Page(_) => fail(format_args!("paged operation reached legacy submission")),
                _ => {}
            }
            let started = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(
                started,
                match step {
                    Step::Refresh(..) => C::StepRefreshUs,
                    Step::Page(_) => unreachable!("handled before timing"),
                    Step::Copy(..) | Step::Fill(..) => C::StepCopyUs,
                    Step::Run(_) | Step::Host(_) => C::StepWorkUs,
                    Step::GpuWrite(..) => C::StepGpuWriteUs,
                    Step::Texture(..) => C::StepTextureUs,
                    Step::Heap(_) => C::StepHeapUs,
                    Step::Memory(_) => C::StepMemoryUs,
                },
            );
            let copied = match step {
                Step::Page(_) => unreachable!("handled before legacy execution"),
                &Step::Copy(crate::copy_plan::GuestCopy { dst, src, len }) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    batch.title_written();
                    copy_mapped(dst as *mut u8, src as *const u8, len, true);
                    len
                }
                &Step::Fill(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    copy_mapped(dst as *mut u8, src as *const u8, len, false);
                    len
                }
                Step::Refresh(shadow, address, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Refreshes, 1);
                        add(C::RefreshBytes, *len as u64);
                    }
                    batch.add(shadow, *address, *len);
                    continue;
                }
                Step::Run(work) | Step::Host(work) => {
                    if matches!(step, Step::Run(_)) {
                        batch.title_written();
                    }
                    work();
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, started.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::GpuWrite(shadow, address, len, marked) => {
                    batch.backing.include(*address, address.saturating_add(*len));
                    // The shaders start from the title's data, which a
                    // snapshot must hold (marked ranges keep none).
                    if !marked {
                        batch.flush();
                    }
                    shadow.writing(*address, *len, *marked);
                    continue;
                }
                Step::Texture(staging, upload) => {
                    let refreshed = batch.texture_due(staging) && staging.refresh(Some(upload)).is_some();
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::TextureQueuedTable, staging.upload_payload);
                        if !refreshed {
                            add(C::TextureQueuedWithoutRefresh, staging.upload_payload);
                        }
                    }
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = refreshed;
                    continue;
                }
                Step::Heap(viewed) => {
                    crate::bindless::refresh(viewed, |s, a, l| batch.add(s, a, l));
                    continue;
                }
                &Step::Memory(heap) => {
                    crate::memory::prepare(heap, |s, a, l| batch.add(s, a, l));
                    continue;
                }
            };
            PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
        }
    }
}

/// A submission's refreshes not done yet, merged per buffer copy into the
/// range union: done before a step that writes title memory or
/// gives a range to shaders, and before the lists run.
///
/// Also the textures brought up to date in the submission so far: most
/// lists of a world submission sample the same textures, and a texture in
/// physical-page memory is hashed whole each time it is checked. Checked
/// once, it is checked again only after a step that writes title memory.
///
/// A copy's range brought up to date is not copied again in the submission
/// until a step writes title memory: every list of a world submission using
/// title memory asks for every copy whole. What the title's threads write
/// meanwhile is for later submissions (on the console it would race the
/// GPU).
#[derive(Default)]
struct Batch {
    backing: crate::title_backing::SubmissionBacking,
    refreshes: Vec<ShadowRefresh>,
    /// Where each copy's refresh is in `refreshes`, by copy.
    pending: std::collections::HashMap<usize, usize>,
    /// Each copy's exact refreshed ranges since the last title write. The
    /// copy is kept so its address cannot be reused for a different copy.
    current: std::collections::HashMap<usize, ShadowRefresh>,
    /// Staging buffers (by address) refreshed since the last title write.
    textures: std::collections::HashSet<usize>,
}

type ShadowRefresh = (std::sync::Arc<crate::shadow::Shadow>, RefreshRanges);

/// Most copies only request one range. Gather disjoint requests without
/// allocating on the single-range path, then merge only touching ranges.
enum RefreshRanges {
    One((usize, usize)),
    Many(Vec<(usize, usize)>),
}

impl RefreshRanges {
    fn add(&mut self, range: (usize, usize)) {
        match self {
            Self::One((from, to)) if range.0 <= *to && *from <= range.1 => {
                *from = (*from).min(range.0);
                *to = (*to).max(range.1);
            }
            Self::One(previous) => *self = Self::Many(vec![*previous, range]),
            Self::Many(ranges) => ranges.push(range),
        }
    }

    fn merge(&mut self) {
        if let Self::Many(ranges) = self {
            ranges.sort_unstable();
            ranges.dedup_by(|next, previous| {
                if next.0 > previous.1 {
                    return false;
                }
                previous.1 = previous.1.max(next.1);
                true
            });
            if ranges.len() == 1 {
                *self = Self::One(ranges[0]);
            }
        }
    }

    fn slices(&self) -> &[(usize, usize)] {
        match self {
            Self::One(range) => std::slice::from_ref(range),
            Self::Many(ranges) => ranges,
        }
    }

    /// The ranges are merged before recording them as current.
    fn covers(&self, (from, to): (usize, usize)) -> bool {
        let ranges = self.slices();
        let end = ranges.partition_point(|&(start, _)| start <= from);
        end != 0 && to <= ranges[end - 1].1
    }
}

impl Batch {
    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        let range = (address, address + len);
        let key = std::sync::Arc::as_ptr(shadow) as usize;
        if len == 0 || self.current.get(&key).is_some_and(|(_, ranges)| ranges.covers(range)) {
            return;
        }
        match self.pending.get(&key) {
            Some(&i) => self.refreshes[i].1.add(range),
            None => {
                self.pending.insert(key, self.refreshes.len());
                self.refreshes.push((shadow.clone(), RefreshRanges::One(range)));
            }
        }
    }

    /// Whether `staging` is still to be refreshed in this submission (and
    /// notes it refreshed).
    fn texture_due<T>(&mut self, staging: &std::sync::Arc<T>) -> bool {
        self.textures.insert(std::sync::Arc::as_ptr(staging) as *const () as usize)
    }

    /// Before a step that writes title memory: pending refreshes are done
    /// first, and textures are checked again after it.
    fn title_written(&mut self) {
        self.flush();
        self.textures.clear();
        self.current.clear();
    }

    /// Does the refreshes; returns the bytes copied.
    fn flush(&mut self) -> usize {
        #[cfg(feature = "diagnostics")]
        let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::FlushUs);
        self.pending.clear();
        let mut total = 0;
        use std::collections::hash_map::Entry;
        for (_, ranges) in &mut self.refreshes {
            ranges.merge();
            for &(from, to) in ranges.slices() {
                self.backing.include(from, to);
            }
        }
        self.backing.flush();
        for (shadow, ranges) in self.refreshes.drain(..) {
            for &(from, to) in ranges.slices() {
                let started = std::time::Instant::now();
                let copied = shadow.refresh_from_backing(from, to - from, &self.backing);
                let us = started.elapsed().as_micros() as u64;
                #[cfg(feature = "diagnostics")]
                crate::copystat::range(from, to - from, us, 0);
                PREPARING[0].fetch_add(us, Ordering::Relaxed);
                PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
                total += copied;
            }
            let key = std::sync::Arc::as_ptr(&shadow) as usize;
            match self.current.entry(key) {
                Entry::Vacant(entry) => {
                    entry.insert((shadow, ranges));
                }
                Entry::Occupied(mut entry) => {
                    let current = &mut entry.get_mut().1;
                    for &range in ranges.slices() {
                        current.add(range);
                    }
                    current.merge();
                }
            }
        }
        total
    }
}

/// Copies title memory at submission. The ranges were checked against their
/// resources when recorded, but the title may since have unmapped pages of
/// them (tiled memory, page mappings), which read as zeros here; a title
/// destination (`title_dst`) that is not all mapped takes nothing from its
/// first unmapped page on.
///
/// The copy runs in the kernel (`ReadProcessMemory`), which reports an
/// unmapped page of the source, or of the destination it writes, instead of
/// faulting. Asking `VirtualQuery` first cost about 2 ms per copy in the
/// title's big pools (it walks the page tables to the region's end), which
/// made copies run at 100-200 MB/s; a title destination is asked only once
/// its copy has failed (asked before each, the world's 80 copies per frame
/// cost 60 ms).
pub(crate) fn copy_mapped(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) {
    #[cfg(feature = "diagnostics")]
    match crate::copystat::mode() {
        1 => {
            // SAFETY: timing only: the caller vouches for the source.
            unsafe { std::ptr::copy(src, dst, len) };
            return;
        }
        2 => return,
        _ => {}
    }
    if len == 0 {
        return;
    }
    // The kernel writes the destination: it must not be guarded meanwhile.
    let _writing = title_dst.then(|| crate::watch::writing(dst as usize, len));
    #[cfg(feature = "diagnostics")]
    let moved = std::time::Instant::now();
    let whole = read(src, dst, len);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::MoveUs, moved.elapsed().as_micros() as u64);
    if whole {
        return;
    }
    // The copy stopped at an unmapped page. One of a title destination ends
    // it: the retries below write the destination themselves (zeros for
    // what cannot be read), and would fault there.
    if title_dst && !crate::texture::committed(dst, len) {
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Fallbacks, 1);
    // Something is unmapped: page by page (64 KiB, then 4 KiB in a piece
    // that fails), zeros for what cannot be read.
    for_pieces(src as usize, len, 64 << 10, |at, n| {
        let to = dst.wrapping_add(at - src as usize);
        if !read(at as *const u8, to, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                let to = dst.wrapping_add(at - src as usize);
                if !read(at as *const u8, to, n) {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
                    // SAFETY: inside the destination, which is mapped.
                    unsafe { to.write_bytes(0, n) };
                }
            });
        }
    });
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable.
pub(crate) fn read_into(src: *const u8, dst: &mut [u8]) -> bool {
    read(src, dst.as_mut_ptr(), dst.len())
}

/// Copies `len` bytes from `src` to `dst` in the kernel: false if a source
/// page is not readable or a destination page not writable (the bytes
/// before that page may have been copied).
fn read(src: *const u8, dst: *mut u8, len: usize) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut done = 0;
    // SAFETY: the destination is `len` bytes the caller may write, host
    // memory or title memory; the kernel checks both ranges, and a page it
    // cannot read or write fails the call instead of faulting.
    (unsafe { ReadProcessMemory(GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done) != 0 }) && done == len
}

/// The paged engine must not mark a dirty page clean unless the entire
/// write succeeded. Unlike legacy writeback, this has no partial fallback.
pub(crate) fn write_mapped_exact(dst: *mut u8, src: &[u8]) -> bool {
    read(src.as_ptr(), dst, src.len())
}

/// Copies host memory into title memory after a submission (what shaders
/// wrote), leaving pages that are not mapped alone.
///
/// Through [`read`]: the kernel refuses an unmapped destination page as it
/// does a source page. `WriteProcessMemory` makes the same copy, but asks
/// for the destination's region first, at `VirtualQuery`'s cost (see
/// [`copy_mapped`]): 0.35 ms for each piece of the world's copy-back, 39 ms
/// per frame.
pub(crate) fn write_mapped(dst: *mut u8, src: *const u8, len: usize) {
    let write = |at: usize, n: usize| read(src.wrapping_add(at - dst as usize), at as *mut u8, n);
    if len == 0 || write(dst as usize, len) {
        return;
    }
    for_pieces(dst as usize, len, 64 << 10, |at, n| {
        if !write(at, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                write(at, n);
            });
        }
    });
}

/// Calls `f` with the pieces of `[start, start + len)` cut at multiples of
/// `step`.
fn for_pieces(start: usize, len: usize, step: usize, mut f: impl FnMut(usize, usize)) {
    let end = start + len;
    let mut at = start;
    while at < end {
        let next = ((at / step + 1) * step).min(end);
        f(at, next - at);
        at = next;
    }
}

#[cfg(test)]
mod title_copies {
    use super::{copy_mapped, write_mapped};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE,
        VirtualAlloc, VirtualFree, VirtualProtect,
    };

    const BLOCK: usize = 64 << 10;

    /// Three reserved blocks, the first and the last committed and filled
    /// with `fill`: title memory with an unmapped block in it.
    fn holed(fill: u8) -> *mut u8 {
        // SAFETY: a fresh reservation; the committed blocks are ours to fill.
        unsafe {
            let base =
                VirtualAlloc(std::ptr::null(), 3 * BLOCK, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            for block in [0, 2] {
                let at = base.add(block * BLOCK);
                assert!(!VirtualAlloc(at.cast(), BLOCK, MEM_COMMIT, PAGE_READWRITE).is_null());
                at.write_bytes(fill, BLOCK);
            }
            base
        }
    }

    fn block<'a>(base: *const u8, n: usize) -> &'a [u8] {
        // SAFETY: tests ask for committed blocks only.
        unsafe { std::slice::from_raw_parts(base.add(n * BLOCK), BLOCK) }
    }

    fn release(base: *mut u8) {
        // SAFETY: a reservation of `holed`.
        assert_ne!(unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) }, 0);
    }

    fn pattern() -> Vec<u8> {
        (0..3 * BLOCK).map(|i| (i % 251) as u8 + 1).collect()
    }

    #[test]
    fn title_copies_reach_mapped_destinations_and_stop_at_unmapped_pages() {
        let from = pattern();
        // All mapped: copied, through a destination that was never asked.
        let to = holed(0xee);
        copy_mapped(to, from.as_ptr(), BLOCK, true);
        assert_eq!(block(to, 0), &from[..BLOCK]);
        release(to);
        // An unmapped block in the destination: nothing faults, and nothing
        // arrives from that block on.
        let to = holed(0xee);
        copy_mapped(to, from.as_ptr(), 3 * BLOCK, true);
        assert!(block(to, 2).iter().all(|&b| b == 0xee));
        release(to);
        // An unmapped block in the source reads as zeros.
        let holes = holed(7);
        let mut to = vec![0xeeu8; 3 * BLOCK];
        copy_mapped(to.as_mut_ptr(), holes, 3 * BLOCK, true);
        assert!(to[..BLOCK].iter().all(|&b| b == 7));
        assert!(to[BLOCK..2 * BLOCK].iter().all(|&b| b == 0));
        assert!(to[2 * BLOCK..].iter().all(|&b| b == 7));
        release(holes);
    }

    #[test]
    fn copy_back_writes_mapped_pages_only() {
        let from = pattern();
        let to = holed(0xee);
        write_mapped(to, from.as_ptr(), 3 * BLOCK);
        assert_eq!(block(to, 0), &from[..BLOCK]);
        assert_eq!(block(to, 2), &from[2 * BLOCK..]);
        release(to);
        // Neither aligned nor whole: the bytes around the range stay.
        let to = holed(0xee);
        // SAFETY: inside the first block.
        write_mapped(unsafe { to.add(100) }, from.as_ptr(), BLOCK - 300);
        let first = block(to, 0);
        assert_eq!(first[100..BLOCK - 200], from[..BLOCK - 300]);
        let around = first[..100].iter().chain(&first[BLOCK - 200..]);
        assert!(around.into_iter().all(|&b| b == 0xee));
        // A page that is mapped but not writable is not written either
        // (guarded title blocks are opened before they are copied back).
        let mut old = 0;
        // SAFETY: a committed block of ours.
        let protected =
            unsafe { VirtualProtect(to.add(2 * BLOCK).cast(), BLOCK, PAGE_READONLY, &mut old) };
        assert_ne!(protected, 0);
        // SAFETY: the third block.
        write_mapped(unsafe { to.add(2 * BLOCK) }, from.as_ptr(), BLOCK);
        assert!(block(to, 2).iter().all(|&b| b == 0xee));
        release(to);
    }
}

impl Host {
    /// A buffer in the upload heap, mapped for writing.
    /// A GPU-only buffer of `size` bytes in the copy-destination state.
    pub(crate) fn default_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        // SAFETY: allocation info for one initialized description.
        let bytes = unsafe { self.device.GetResourceAllocationInfo(0, &[desc]) }.SizeInBytes;
        self.managed_gpu(bytes, |heap| {
            let mut buffer: Option<ID3D12Resource> = None;
            // SAFETY: live device, initialized descriptors and out slot.
            unsafe {
                self.device.CreateCommittedResource(
                    heap, D3D12_HEAP_FLAG_NONE, &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST, None, &mut buffer,
                )?;
            }
            buffer.ok_or_else(windows::core::Error::empty)
        }).ok()
    }

    pub(crate) fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        // SAFETY: allocation info for a single initialized description.
        let bytes = unsafe { self.device.GetResourceAllocationInfo(0, &[desc]) }.SizeInBytes;
        let mut data = std::ptr::null_mut();
        let made = self.managed_system(bytes, || {
            let mut buffer: Option<ID3D12Resource> = None;
            // SAFETY: live device, initialized description and output slot.
            unsafe {
                self.device.CreateCommittedResource(
                    &heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut buffer,
                )?;
            }
            buffer.ok_or_else(windows::core::Error::empty)
        }).and_then(|buffer| {
            // SAFETY: the committed buffer is CPU-visible and the output is ours.
            unsafe { buffer.Map(0, None, Some(&mut data))? };
            Ok(buffer)
        });
        match made {
            Ok(buffer) => Some((buffer, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no upload buffer");
                None
            }
        }
    }
}

/// The swap chain formats a presented texture can be copied into: its own
/// family's.
fn swap_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

const BUFFERS: u32 = 2;
const MAX_PLANES: u32 = 8;

/// Built-in fullscreen-triangle compositor for multi-plane `PresentX` calls
/// (e.g. Series S presenting a 1080p scene plane with a 4K UI overlay plane).
struct Compositor {
    format: DXGI_FORMAT,
    root: ID3D12RootSignature,
    opaque: ID3D12PipelineState,
    overlay: ID3D12PipelineState,
    rtv_heap: ID3D12DescriptorHeap,
    srv_heap: ID3D12DescriptorHeap,
    srv_stride: u32,
}

impl Compositor {
    fn new(device: &ID3D12Device, format: DXGI_FORMAT) -> windows::core::Result<Self> {
        const SOURCE: &str = r#"
            Texture2D tex : register(t0);
            SamplerState smp : register(s0);
            void vs(uint id : SV_VertexID, out float4 pos : SV_Position, out float2 uv : TEXCOORD) {
                uv = float2((id << 1) & 2, id & 2);
                pos = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
            }
            float4 ps(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target {
                return tex.Sample(smp, uv);
            }
        "#;
        let compile = |entry: PCSTR, target: PCSTR| -> windows::core::Result<ID3DBlob> {
            let mut code: Option<ID3DBlob> = None;
            let mut errors: Option<ID3DBlob> = None;
            // SAFETY: static source and entry/target strings; out pointers ours.
            let res = unsafe {
                D3DCompile(
                    SOURCE.as_ptr().cast(),
                    SOURCE.len(),
                    s!("present_composite"),
                    None,
                    None,
                    entry,
                    target,
                    0,
                    0,
                    &mut code,
                    Some(&mut errors),
                )
            };
            if let Err(e) = res {
                if let Some(errors) = errors {
                    // SAFETY: the blob's own bytes.
                    let text = unsafe {
                        std::slice::from_raw_parts(
                            errors.GetBufferPointer().cast::<u8>(),
                            errors.GetBufferSize(),
                        )
                    };
                    tracing::warn!("present compositor: {}", String::from_utf8_lossy(text));
                }
                return Err(e);
            }
            code.ok_or_else(windows::core::Error::empty)
        };
        let vs = compile(s!("vs"), s!("vs_5_1"))?;
        let ps = compile(s!("ps"), s!("ps_5_1"))?;
        // SAFETY: compiled shader blobs live through PSO creation.
        unsafe {
            let vs_bytes =
                std::slice::from_raw_parts(vs.GetBufferPointer().cast::<u8>(), vs.GetBufferSize());
            let ps_bytes =
                std::slice::from_raw_parts(ps.GetBufferPointer().cast::<u8>(), ps.GetBufferSize());
            // Wine's D3DCompile does not embed a root signature in the shader blobs.
            let range = D3D12_DESCRIPTOR_RANGE {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                NumDescriptors: 1,
                OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
                ..Default::default()
            };
            let parameter = D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: 1,
                        pDescriptorRanges: &range,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            };
            let sampler = D3D12_STATIC_SAMPLER_DESC {
                Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
                AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressW: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                MaxAnisotropy: 16,
                ComparisonFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
                BorderColor: D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
                MaxLOD: f32::MAX,
                ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
                ..Default::default()
            };
            let desc = D3D12_ROOT_SIGNATURE_DESC {
                NumParameters: 1,
                pParameters: &parameter,
                NumStaticSamplers: 1,
                pStaticSamplers: &sampler,
                ..Default::default()
            };
            let mut signature = None;
            D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut signature, None)?;
            let signature = signature.ok_or_else(windows::core::Error::empty)?;
            let signature = std::slice::from_raw_parts(
                signature.GetBufferPointer().cast::<u8>(),
                signature.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, signature)?;
            let make_pso = |blend: bool| -> windows::core::Result<ID3D12PipelineState> {
                let mut rt_blend = D3D12_RENDER_TARGET_BLEND_DESC {
                    RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                    ..Default::default()
                };
                if blend {
                    rt_blend.BlendEnable = true.into();
                    rt_blend.SrcBlend = D3D12_BLEND_ONE;
                    rt_blend.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
                    rt_blend.BlendOp = D3D12_BLEND_OP_ADD;
                    rt_blend.SrcBlendAlpha = D3D12_BLEND_ONE;
                    rt_blend.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
                    rt_blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
                }
                let mut blend_desc = D3D12_BLEND_DESC::default();
                blend_desc.RenderTarget[0] = rt_blend;
                let mut rtv_formats = [DXGI_FORMAT_UNKNOWN; 8];
                rtv_formats[0] = format;
                let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::ManuallyDrop::new(Some(root.clone())),
                    VS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: vs_bytes.as_ptr().cast(),
                        BytecodeLength: vs_bytes.len(),
                    },
                    PS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: ps_bytes.as_ptr().cast(),
                        BytecodeLength: ps_bytes.len(),
                    },
                    BlendState: blend_desc,
                    SampleMask: u32::MAX,
                    RasterizerState: D3D12_RASTERIZER_DESC {
                        FillMode: D3D12_FILL_MODE_SOLID,
                        CullMode: D3D12_CULL_MODE_NONE,
                        DepthClipEnable: false.into(),
                        ..Default::default()
                    },
                    PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                    NumRenderTargets: 1,
                    RTVFormats: rtv_formats,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    ..Default::default()
                };
                let pso = device.CreateGraphicsPipelineState(&desc);
                let _ = std::mem::ManuallyDrop::into_inner(desc.pRootSignature);
                pso
            };
            let opaque = make_pso(false)?;
            let overlay = make_pso(true)?;
            let rtv_heap: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                })?;
            let srv_heap: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: MAX_PLANES,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })?;
            let srv_stride =
                device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            Ok(Compositor {
                format,
                root,
                opaque,
                overlay,
                rtv_heap,
                srv_heap,
                srv_stride,
            })
        }
    }
}

#[derive(Default)]
struct Presentation {
    presenter: Option<Presenter>,
    retry: SwapChainRetry,
}

#[derive(Default)]
struct SwapChainRetry {
    next_attempt: Option<Instant>,
}

impl SwapChainRetry {
    /// A desktop can become presentable again. Keep retrying while limiting
    /// the DXGI calls and warnings during the outage.
    fn attempt<T, E>(
        &mut self,
        now: Instant,
        operation: impl FnOnce() -> Result<T, E>,
    ) -> Option<Result<T, E>> {
        if self.next_attempt.is_some_and(|next| now < next) {
            return None;
        }
        let result = operation();
        self.next_attempt = result.is_err().then(|| now + Duration::from_secs(1));
        Some(result)
    }
}

struct Presenter {
    swap_chain: IDXGISwapChain3,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    copy: HostList,
    hwnd: HWND,
    meter: Meter,
    compositor: Option<Compositor>,
}

/// The frame rate, shown in the window's caption every second and logged
/// every few seconds.
struct Meter {
    since: std::time::Instant,
    frames: u32,
    #[cfg(feature = "diagnostics")]
    diagnostic_frames: u64,
    drawn: u64,
    logged: std::time::Instant,
    /// The caption the window had, once read.
    caption: Option<Vec<u16>>,
}

impl Meter {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Meter {
            since: now,
            frames: 0,
            #[cfg(feature = "diagnostics")]
            diagnostic_frames: 0,
            drawn: crate::draw::drawn(),
            logged: now,
            caption: None,
        }
    }

    /// Counts a presented frame.
    fn frame(&mut self, hwnd: HWND) {
        #[cfg(feature = "diagnostics")]
        tracing::debug!(target: "victory_gpu::frame", token = crate::frame::current(), "present");
        self.frames += 1;
        crate::memory::log_frame();
        #[cfg(feature = "diagnostics")]
        crate::watch::frame_done();
        #[cfg(feature = "diagnostics")]
        {
            self.diagnostic_frames += 1;
        }
        let elapsed = self.since.elapsed();
        if elapsed < std::time::Duration::from_secs(1) {
            return;
        }
        let drawn = crate::draw::drawn();
        let frames = f64::from(self.frames);
        let fps = frames / elapsed.as_secs_f64();
        let ms = elapsed.as_secs_f64() * 1000.0 / frames;
        let draws = (drawn - self.drawn) as f64 / frames;
        (self.since, self.frames, self.drawn) = (std::time::Instant::now(), 0, drawn);
        // Of a frame, ms preparing submissions and ms waiting for the GPU.
        let [cpu, gpu] = [0, 1].map(|i| {
            SUBMISSION_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames
        });
        let [copy_ms, copy_mb, texture_ms, texture_mb] = [0, 1, 2, 3].map(|i| {
            let scale = if i % 2 == 0 { 1e3 } else { 1e6 };
            PREPARING[i].swap(0, Ordering::Relaxed) as f64 / scale / frames
        });
        let [back_ms, back_mb] = [0, 1].map(|i| {
            COPY_BACK[i].swap(0, Ordering::Relaxed) as f64 / [1e3, 1e6][i] / frames
        });
        let dropped = crate::draw::take_dropped() as f64 / frames;
        let indirect = crate::list::take_indirect() as f64 / frames;
        let root_views = crate::draw::take_root_view_slots() as f64 / frames;
        if self.logged.elapsed() >= std::time::Duration::from_secs(5) {
            self.logged = std::time::Instant::now();
            tracing::info!(
                "{fps:.1} fps, {ms:.1} ms/frame ({cpu:.1} preparing: copies {copy_ms:.1} ms \
                 {copy_mb:.1} MB, textures {texture_ms:.1} ms {texture_mb:.1} MB; {gpu:.1} on the \
                 GPU; {back_ms:.1} copying back {back_mb:.1} MB), {draws:.0} draws/frame, \
                 dropped {dropped:.0} without a host pipeline ({making} being made) and \
                 {indirect:.0} indirect; {root_views:.0} root-view slots/frame",
                making = crate::pipeline::making()
            );
            let (textures, bytes, used, shared, saved, replaced) = crate::resource::texture_counts();
            tracing::info!(
                "host textures: {textures} ({:.0} MB, {:.0} MB of them used by lists), title textures sharing one {shared} (saving {:.0} MB), replacing one {replaced}; \
                 copies between textures of one memory {:.1}/frame, depth twin copies {:.1}/frame",
                bytes as f64 / 1e6,
                used as f64 / 1e6,
                saved as f64 / 1e6,
                crate::texture::ALIAS_COPIES.swap(0, Ordering::Relaxed) as f64 / frames,
                crate::twin::COPIES.swap(0, Ordering::Relaxed) as f64 / frames,
            );
            #[cfg(feature = "diagnostics")]
            {
                let diagnostic_frames = std::mem::take(&mut self.diagnostic_frames) as f64;
                // Drain the reasons when reporting them. The window caption's
                // shorter updates must leave the earlier drops in this interval.
                let drops = crate::pipeline::take_drops();
                crate::copystat::log(diagnostic_frames);
                crate::bindless::log(diagnostic_frames);
                crate::shadow::log_views(diagnostic_frames);
                for (why, n) in drops.iter().take(40) {
                    tracing::info!(
                        "dropped {:.1}/frame without a host pipeline: {why}",
                        *n as f64 / diagnostic_frames
                    );
                }
            }
        }
        if self.caption.is_none() {
            // The window's own text, read without a message.
            let mut buffer = vec![0u16; 512];
            // SAFETY: a window handle and a buffer of that length.
            let n = unsafe { InternalGetWindowText(hwnd, &mut buffer) };
            buffer.truncate(n.max(0) as usize);
            self.caption = Some(buffer);
        }
        let Some(caption) = &self.caption else { return };
        let text = format!(" | VIctory {fps:.0} fps {ms:.1} ms/frame {draws:.0} draws/frame");
        let mut text: Vec<u16> = caption.iter().copied().chain(text.encode_utf16()).collect();
        text.push(0);
        // Setting the text waits for the window's thread (a title thread,
        // which may not pump for a while), so a thread of ours does it and
        // presenting never waits; it takes the newest text when free.
        static PENDING: Mutex<Option<(usize, Vec<u16>)>> = Mutex::new(None);
        static SETTER: std::sync::Once = std::sync::Once::new();
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((hwnd.0 as usize, text));
        SETTER.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("victory-caption".into())
                .spawn(|| {
                    loop {
                        let next = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
                        match next {
                            // SAFETY: a window handle and a NUL-terminated text.
                            Some((hwnd, text)) => unsafe {
                                let _ = SetWindowTextW(HWND(hwnd as *mut _), PCWSTR(text.as_ptr()));
                            },
                            None => std::thread::sleep(std::time::Duration::from_millis(100)),
                        }
                    }
                });
        });
    }
}

impl Host {
    /// Shows one or more `planes` (each a live host texture and its current
    /// host state, kept in that state), compositing planes 1.. on top of
    /// plane 0 when multiple planes are presented, as frame `frame` (saved if
    /// captures are on). `None` if nothing was shown; else, in diagnostics
    /// builds, whether the frame was black.
    pub(crate) fn present(&self, planes: &[(&ID3D12Resource, u32)], frame: u64) -> Option<bool> {
        self.present_with_compositor(
            planes,
            frame,
            Compositor::new,
            #[cfg(all(test, feature = "diagnostics"))]
            |_, _| {},
        )
    }

    fn present_with_compositor(
        &self,
        planes: &[(&ID3D12Resource, u32)],
        frame: u64,
        create_compositor: impl FnOnce(&ID3D12Device, DXGI_FORMAT) -> windows::core::Result<Compositor>,
        #[cfg(all(test, feature = "diagnostics"))] inspect: impl FnOnce(
            &ID3D12Resource,
            &D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
        ),
    ) -> Option<bool> {
        let &(primary, _) = planes.first()?;
        // SAFETY: live resources.
        let primary_desc = unsafe { primary.GetDesc() };
        let Some(format) = swap_format(primary_desc.Format) else {
            tracing::warn!(format = primary_desc.Format.0, "cannot present this format");
            return None;
        };
        let (mut width, mut height) = (primary_desc.Width as u32, primary_desc.Height);
        for &(tex, _) in &planes[1..] {
            // SAFETY: live resource.
            let d = unsafe { tex.GetDesc() };
            width = width.max(d.Width as u32);
            height = height.max(d.Height);
        }
        let mut presentation = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        let Presentation { presenter, retry } = &mut *presentation;
        if presenter.is_none() {
            *presenter = match retry
                .attempt(Instant::now(), || self.presenter(width, height, format))?
            {
                Ok(p) => Some(p),
                Err((step, e)) => {
                    tracing::warn!(error = %e, step, width, height, format = format.0, "no swap chain");
                    return None;
                }
            };
        }
        let p = presenter.as_mut()?;
        if (p.width, p.height, p.format) != (width, height, format) {
            let resized = retry.attempt(Instant::now(), || {
                // SAFETY: no back buffer is held; every submission was waited for.
                unsafe {
                    p.swap_chain.ResizeBuffers(
                        BUFFERS,
                        width,
                        height,
                        format,
                        DXGI_SWAP_CHAIN_FLAG(0),
                    )
                }
            })?;
            match resized {
                Ok(()) => (p.width, p.height, p.format) = (width, height, format),
                Err(e) => {
                    tracing::warn!(error = %e, width, height, format = format.0, "swap chain not resized");
                    return None;
                }
            }
        }
        // Leave the copy list closed if compositor initialization fails.
        let compositor = if planes.len() > 1 {
            if p.compositor.as_ref().is_none_or(|c| c.format != format) {
                match create_compositor(&self.device, format) {
                    Ok(c) => p.compositor = Some(c),
                    Err(e) => {
                        // The frame is still shown: its first plane alone, as
                        // a single-plane present would.
                        tracing::warn!(
                            error = %e,
                            "no present compositor; presenting the first plane"
                        );
                        p.compositor = None;
                    }
                }
            }
            p.compositor.as_ref()
        } else {
            None
        };
        // SAFETY: our swap chain, list and live source textures.
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let Ok(back) = p.swap_chain.GetBuffer::<ID3D12Resource>(index) else {
                return None;
            };
            p.copy.reset();
            let list = p.copy.recording();
            // One plane, or no compositor to combine several: copy the first.
            let (capture, probe) = if planes.len() == 1 || compositor.is_none() {
                let (texture, state) = planes[0];
                let state = D3D12_RESOURCE_STATES(state as i32);
                let mut before = Vec::with_capacity(2);
                if state != D3D12_RESOURCE_STATE_COPY_SOURCE {
                    before.push(transition(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE));
                }
                before.push(transition(
                    &back,
                    D3D12_RESOURCE_STATE_PRESENT,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ));
                list.ResourceBarrier(&before);
                list.CopyResource(&back, texture);
                #[cfg(feature = "diagnostics")]
                let capture = (cfg!(test) || crate::capture::wanted(frame))
                    .then(|| crate::capture::read_back(&self.device, list, texture))
                    .flatten();
                #[cfg(feature = "diagnostics")]
                let probe = crate::capture::probe(&self.device, list, texture);
                #[cfg(not(feature = "diagnostics"))]
                let (capture, probe) = {
                    let _ = frame;
                    (None::<()>, None::<()>)
                };
                let mut after = Vec::with_capacity(2);
                if state != D3D12_RESOURCE_STATE_COPY_SOURCE {
                    after.push(transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state));
                }
                after.push(transition(
                    &back,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_PRESENT,
                ));
                list.ResourceBarrier(&after);
                (capture, probe)
            } else {
                let comp = compositor?;
                let rtv = comp.rtv_heap.GetCPUDescriptorHandleForHeapStart();
                self.device.CreateRenderTargetView(
                    &back,
                    Some(&D3D12_RENDER_TARGET_VIEW_DESC {
                        Format: format,
                        ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
                        Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                            Texture2D: D3D12_TEX2D_RTV {
                                MipSlice: 0,
                                PlaneSlice: 0,
                            },
                        },
                    }),
                    rtv,
                );
                let srv_cpu_base = comp.srv_heap.GetCPUDescriptorHandleForHeapStart();
                let srv_gpu_base = comp.srv_heap.GetGPUDescriptorHandleForHeapStart();
                let count = planes.len().min(MAX_PLANES as usize);
                let mut before = Vec::with_capacity(count + 1);
                let mut after = Vec::with_capacity(count);
                for (i, &(tex, state)) in planes[..count].iter().enumerate() {
                    let d = tex.GetDesc();
                    let srv_fmt = swap_format(d.Format).unwrap_or(d.Format);
                    let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                        ptr: srv_cpu_base.ptr + i * comp.srv_stride as usize,
                    };
                    self.device.CreateShaderResourceView(
                        tex,
                        Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                            Format: srv_fmt,
                            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                                Texture2D: D3D12_TEX2D_SRV {
                                    MostDetailedMip: 0,
                                    MipLevels: 1,
                                    PlaneSlice: 0,
                                    ResourceMinLODClamp: 0.0,
                                },
                            },
                        }),
                        handle,
                    );
                    let state = D3D12_RESOURCE_STATES(state as i32);
                    if state != D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
                        && !planes[..i].iter().any(|(seen, _)| seen.as_raw() == tex.as_raw())
                    {
                        before.push(transition(tex, state, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));
                        after.push(transition(tex, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, state));
                    }
                }
                before.push(transition(
                    &back,
                    D3D12_RESOURCE_STATE_PRESENT,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                ));
                list.ResourceBarrier(&before);
                list.SetDescriptorHeaps(&[Some(comp.srv_heap.clone())]);
                list.SetGraphicsRootSignature(&comp.root);
                list.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: width as f32,
                    Height: height as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]);
                list.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: width as i32,
                    bottom: height as i32,
                }]);
                list.OMSetRenderTargets(1, Some(&rtv), false, None);
                list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                for i in 0..count {
                    list.SetPipelineState(if i == 0 { &comp.opaque } else { &comp.overlay });
                    list.SetGraphicsRootDescriptorTable(
                        0,
                        D3D12_GPU_DESCRIPTOR_HANDLE {
                            ptr: srv_gpu_base.ptr + (i as u64) * u64::from(comp.srv_stride),
                        },
                    );
                    list.DrawInstanced(3, 1, 0, 0);
                }
                #[cfg(feature = "diagnostics")]
                let (capture, probe) = {
                    list.ResourceBarrier(&[transition(
                        &back,
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    )]);
                    let capture = (cfg!(test) || crate::capture::wanted(frame))
                        .then(|| crate::capture::read_back(&self.device, list, &back))
                        .flatten();
                    let probe = crate::capture::probe(&self.device, list, &back);
                    list.ResourceBarrier(&[transition(
                        &back,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_PRESENT,
                    )]);
                    (capture, probe)
                };
                #[cfg(not(feature = "diagnostics"))]
                let (capture, probe) = {
                    let _ = frame;
                    list.ResourceBarrier(&[transition(
                        &back,
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                        D3D12_RESOURCE_STATE_PRESENT,
                    )]);
                    (None::<()>, None::<()>)
                };
                if !after.is_empty() {
                    list.ResourceBarrier(&after);
                }
                (capture, probe)
            };
            p.copy.close();
            self.execute(&[&p.copy]);
            #[cfg(feature = "diagnostics")]
            if let Some((buffer, footprint)) = capture {
                // Tests inspect the completed copy before flip-discard Present
                // can invalidate the back buffer, with the presenter still locked.
                #[cfg(test)]
                inspect(&buffer, &footprint);
                crate::capture::save_read_back(frame, &buffer, &footprint);
            }
            let _ = p.swap_chain.Present(0, DXGI_PRESENT(0));
            p.meter.frame(p.hwnd);
            #[cfg(feature = "diagnostics")]
            return Some(probe.is_some_and(|b| crate::capture::black(&b)));
            #[cfg(not(feature = "diagnostics"))]
            {
                let _ = (capture, probe);
                Some(false)
            }
        }
    }

    fn presenter(
        &self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> Result<Presenter, (&'static str, windows::core::Error)> {
        let hwnd = match title_window() {
            Some(hwnd) => hwnd,
            None => window(width, height).map_err(|e| ("window", e))?,
        };
        // SAFETY: plain object creation for our window.
        unsafe {
            let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))
                .map_err(|e| ("DXGI factory", e))?;
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: BUFFERS,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                ..Default::default()
            };
            let swap_chain = factory
                .CreateSwapChainForHwnd(&self.queue, hwnd, &desc, None, None)
                .and_then(|chain| chain.cast::<IDXGISwapChain3>())
                .map_err(|e| ("swap chain", e))?;
            let copy = self.new_list().map_err(|e| ("copy list", e))?;
            copy.close();
            tracing::info!(width, height, format = format.0, "presenting to a window");
            Ok(Presenter {
                swap_chain,
                width,
                height,
                format,
                copy,
                hwnd,
                meter: Meter::new(),
                compositor: None,
            })
        }
    }
}

/// A visible top-level window of this process: the one the title made, if
/// it made one.
fn title_window() -> Option<HWND> {
    extern "system" fn visit(hwnd: HWND, found: LPARAM) -> windows::core::BOOL {
        let mut process = 0;
        // SAFETY: a window handle from the enumeration; `found` is ours.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut process));
            if process == windows::Win32::System::Threading::GetCurrentProcessId()
                && IsWindowVisible(hwnd).as_bool()
            {
                *(found.0 as *mut Option<HWND>) = Some(hwnd);
                return false.into();
            }
        }
        true.into()
    }
    let mut found: Option<HWND> = None;
    // SAFETY: the callback only writes `found`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&raw mut found as isize)) };
    found
}

/// `state` without the bits the host texture's flags forbid: the console
/// lets the title name states its own resources allow (unordered access on
/// a depth target, whose host copy cannot have both).
pub(crate) fn legal_state(texture: &ID3D12Resource, state: u32) -> u32 {
    // SAFETY: a plain query.
    legal_for(unsafe { texture.GetDesc() }.Flags.0 as u32, state)
}

/// [`legal_state`] for a texture with resource flags `flags`.
pub(crate) fn legal_for(flags: u32, state: u32) -> u32 {
    // (flag, set or clear forbids, state bits): render target, unordered
    // access, depth, denied shader resource.
    const FORBIDDEN: [(u32, bool, u32); 4] = [
        (0x1, false, 0x4),
        (0x4, false, 0x8),
        (0x2, false, 0x10 | 0x20),
        (0x8, true, 0x40 | 0x80),
    ];
    FORBIDDEN
        .iter()
        .filter(|&&(flag, deny, _)| (flags & flag != 0) == deny)
        .fold(state, |s, &(_, _, bits)| s & !bits)
}

impl Host {
    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly).
    pub(crate) fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        // SAFETY: allocation info for a single initialized description.
        let bytes = unsafe { self.device.GetResourceAllocationInfo(0, &[desc]) }.SizeInBytes;
        let mut data = std::ptr::null_mut();
        let made = self.managed_system(bytes, || {
            let mut buffer: Option<ID3D12Resource> = None;
            // SAFETY: live device, initialized description and output slot.
            unsafe {
                self.device.CreateCommittedResource(
                    &heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buffer,
                )?;
            }
            buffer.ok_or_else(windows::core::Error::empty)
        }).and_then(|buffer| {
            // SAFETY: the committed buffer is CPU-visible and the output is ours.
            unsafe { buffer.Map(0, None, Some(&mut data))? };
            Ok(buffer)
        });
        match made {
            Ok(buffer) => Some((buffer, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no shared buffer");
                None
            }
        }
    }
}

/// Host resources that view heap slots stopped viewing while a submission
/// may have been running ([`retire`]).
static RETIRED: Mutex<Vec<ID3D12Resource>> = Mutex::new(Vec::new());

/// Keeps `resource` alive until the GPU work running now (or the next
/// submission's) has run: title threads rewrite view heap slots while the
/// GPU runs, and a shader that took the old descriptor still reads what it
/// views.
pub(crate) fn retire(resource: ID3D12Resource) {
    RETIRED.lock().unwrap_or_else(|e| e.into_inner()).push(resource);
}

/// After a submission has run, with none running: the retired resources go.
pub(crate) fn release_retired() {
    let retired = std::mem::take(&mut *RETIRED.lock().unwrap_or_else(|e| e.into_inner()));
    drop(retired);
}

/// Diagnostics: a noted command running a shader `VICTORY_TRACE_SHADERS`
/// names (hashes, comma-separated) is logged with what it binds, the first
/// 500.
#[cfg(feature = "diagnostics")]
fn traced(what: &str) {
    static HASHES: OnceLock<Vec<String>> = OnceLock::new();
    static LOGGED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let hashes = HASHES.get_or_init(|| {
        let v = std::env::var("VICTORY_TRACE_SHADERS").unwrap_or_default();
        v.split(',').map(str::trim).filter(|h| !h.is_empty()).map(str::to_owned).collect()
    });
    if hashes.iter().any(|h| what.contains(h.as_str())) && LOGGED.fetch_add(1, Ordering::Relaxed) < 500 {
        tracing::info!("traced command: {what}");
    }
}

/// Diagnostics, `VICTORY_GPU_TIMING`: every command of the list during
/// whose run (each list alone) the device was found removed.
fn removed_during(runs: &[&HostList], list: Option<usize>) -> String {
    #[cfg(feature = "diagnostics")]
    if let Some(i) = list {
        let notes = runs[i].notes.lock().unwrap_or_else(|e| e.into_inner());
        let mut text = format!("; removed while list {i} ran alone ({} commands):", notes.len());
        for (k, (n, _)) in notes.iter().enumerate() {
            text.push_str(&format!(" [{k}] {n}"));
        }
        return text;
    }
    let _ = (runs, list);
    String::new()
}

/// For adapters without device-removed data (the software one): the last
/// commands each list of the failed submission noted (diagnostics, with
/// `VICTORY_D3D12_DRED`).
fn last_notes(runs: &[&HostList]) -> String {
    #[cfg(feature = "diagnostics")]
    {
        let mut text = String::new();
        for (i, r) in runs.iter().enumerate() {
            let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
            for (n, _) in notes.iter().rev().take(6).rev() {
                text.push_str(&format!("; list {i} of {}, of its {}: {n}", runs.len(), notes.len()));
            }
        }
        text
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = runs;
        String::new()
    }
}

/// Where the faulting address `va` lies against a range `(start, bytes)` a
/// command reads (diagnostics).
#[cfg(feature = "diagnostics")]
fn fault_in(va: u64, (start, bytes): (u64, u64)) -> String {
    let end = start + bytes;
    if va < start {
        format!(" (fault {:#x} below its source)", start - va)
    } else if va < end {
        format!(" (fault at source +{:#x})", va - start)
    } else {
        format!(" (fault {:#x} past its source's end)", va - end)
    }
}

/// The commands that may have been running where a list stopped at
/// command `done`: those between the barriers around it (work between
/// barriers overlaps on the GPU), by kind, and what the markers just
/// before them say (`HostList::note_op`: diagnostics builds with
/// device-removed data on).
fn in_flight(history: &[D3D12_AUTO_BREADCRUMB_OP], markers: &[(u32, String)], done: usize) -> String {
    let barrier = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER;
    let from = history[..done].iter().rposition(barrier).map_or(0, |b| b + 1);
    let to = history[done..]
        .iter()
        .position(barrier)
        .map_or(history.len(), |b| (done + b).max(done + 1));
    let mut text = format!(", commands {from}..{to} between its barriers:");
    let mut shown = 0;
    for i in from..to {
        let op = history[i];
        if op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER || shown == 12 {
            continue;
        }
        shown += 1;
        let marker = i
            .checked_sub(1)
            .filter(|&m| history[m] == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
            .and_then(|m| markers.iter().find(|(at, _)| *at as usize == m))
            .map_or("no marker", |(_, s)| s.as_str());
        text.push_str(&format!("; command {i} (op {}): {marker}", op.0));
    }
    text
}

pub(crate) fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// A window of the given client size, owned by its own message thread.
fn window(width: u32, height: u32) -> windows::core::Result<HWND> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("victory-window".into())
        .spawn(move || {
            let made = make_window(width, height);
            let ok = made.is_ok();
            let _ = sender.send(made.map(|h| h.0 as usize));
            if ok {
                pump();
            }
        })
        .map_err(|_| windows::core::Error::empty())?;
    let hwnd = receiver
        .recv()
        .map_err(|_| windows::core::Error::empty())??;
    Ok(HWND(hwnd as *mut _))
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: the default handling for our window.
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

fn make_window(width: u32, height: u32) -> windows::core::Result<HWND> {
    const CLASS: PCWSTR = w!("VIctory");
    // SAFETY: registers our class once and creates a window of it.
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(window_proc),
            hInstance: instance.into(),
            lpszClassName: CLASS,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };
        RegisterClassExW(&class);
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false)?;
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            CLASS,
            w!("VIctory"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
}

fn pump() {
    let mut message = MSG::default();
    // SAFETY: the standard message loop of this thread's window.
    unsafe {
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

#[cfg(test)]
mod batch {   // Batch::texture_due lives here; `mod tests` is already taken at the top of this file
    use super::Batch;
    use std::sync::Arc;

    #[test]
    fn a_texture_is_checked_once_per_submission() {
        let mut batch = Batch::default();
        let (a, b) = (Arc::new([0u8; 16]), Arc::new([0u8; 16]));
        // The first list sampling each texture checks it ...
        assert!(batch.texture_due(&a));
        assert!(batch.texture_due(&b));
        // ... later lists of the same submission do not.
        assert!(!batch.texture_due(&a));
        assert!(!batch.texture_due(&a.clone()));
        assert!(!batch.texture_due(&b));
    }

    #[test]
    fn title_writes_make_textures_due_again() {
        let mut batch = Batch::default();
        let a = Arc::new(1u32);
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A copy into title memory or work that may write it may have
        // changed the texture's memory.
        batch.title_written();
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A fresh submission starts over.
        let mut next = Batch::default();
        assert!(next.texture_due(&a));
    }

    /// A host copy of tracked title memory: a submission whose lists all ask
    /// for it whole copies the one page the title wrote, once, and again
    /// only after a step that writes title memory.
    #[test]
    fn a_copy_is_refreshed_once_per_submission_by_the_pages_written() {
        use crate::watch::PAGE;
        let p = crate::watch::tests::pool(4);
        let len = 4 * PAGE;
        crate::shadow::register(p, p as u64, len as u64);
        let Some(copy) = crate::shadow::find(p as u64, len as u64) else {
            crate::shadow::unregister(p, p as u64);
            return; // no host GPU here
        };
        let mut batch = Batch::default();
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), len, "all of it at first");
        // SAFETY: our pool, guarded since the copy took it.
        unsafe { ((p + PAGE + 8) as *mut u8).write_volatile(7) };
        // The next submission: two lists ask for the copy.
        let mut batch = Batch::default();
        batch.add(&copy, p, len);
        batch.add(&copy, p + 100, 200);
        assert_eq!(batch.flush(), PAGE, "the page written");
        assert_eq!(copy.read_host((p + PAGE + 8) as u64, 1)[0], 7);
        // SAFETY: as above.
        unsafe { (p as *mut u8).write_volatile(1) };
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), 0, "nothing again in the submission");
        batch.title_written();
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(), PAGE, "after a title write, what changed");
        crate::shadow::unregister(p, p as u64);
        drop(copy);
        crate::watch::tests::free(p);
    }

    /// Two requested ranges leave a dirty interior page available to its
    /// own reader. The change journal must not consume it with the first two.
    #[test]
    fn disjoint_refreshes_leave_the_dirty_gap_for_its_reader() {
        use crate::watch::PAGE;

        let title = crate::watch::tests::pool(3);
        let len = 3 * PAGE;
        crate::shadow::register(title, title as u64, len as u64);
        let Some(copy) = crate::shadow::find(title as u64, len as u64) else {
            crate::shadow::unregister(title, title as u64);
            crate::watch::tests::free(title);
            assert!(std::env::var_os("VICTORY_TEST_SHADOW_DISJOINT_REQUIRE_GPU").is_none());
            return;
        };
        let mut warm = Batch::default();
        warm.add(&copy, title, len);
        let warmed = warm.flush();
        for (page, marker) in [11u8, 22, 33].into_iter().enumerate() {
            // SAFETY: each address belongs to the test's guarded allocation.
            unsafe { ((title + page * PAGE) as *mut u8).write_volatile(marker) };
        }

        let mut batch = Batch::default();
        batch.add(&copy, title + 2 * PAGE, PAGE);
        batch.add(&copy, title, PAGE);
        batch.add(&copy, title + PAGE / 2, PAGE / 2);
        let first_bytes = batch.flush();
        let markers = [0, 1, 2].map(|page| copy.read_host((title + page * PAGE) as u64, 1)[0]);
        // SAFETY: the test still owns the title allocation.
        unsafe { ((title + 2 * PAGE) as *mut u8).write_volatile(44) };
        batch.add(&copy, title + 2 * PAGE, PAGE);
        let repeated_bytes = batch.flush();
        batch.add(&copy, title + PAGE, PAGE);
        let late_bytes = batch.flush();
        let late_marker = copy.read_host((title + PAGE) as u64, 1)[0];
        batch.title_written();
        batch.add(&copy, title + 2 * PAGE, PAGE);
        let next_phase_bytes = batch.flush();
        let next_phase_marker = copy.read_host((title + 2 * PAGE) as u64, 1)[0];

        // The temporary read batches hold copies of the backing while live.
        drop((warm, batch));
        crate::shadow::unregister(title, title as u64);
        drop(copy);
        crate::watch::tests::free(title);

        eprintln!(
            "disjoint refresh copied {first_bytes} then {late_bytes} bytes, markers {markers:?} then {late_marker}"
        );
        assert_eq!(warmed, len);
        assert_eq!(first_bytes, 2 * PAGE);
        assert_eq!(markers, [11, 0, 33]);
        assert_eq!(repeated_bytes, 0);
        assert_eq!(late_bytes, PAGE);
        assert_eq!(late_marker, 22);
        assert_eq!(next_phase_bytes, PAGE);
        assert_eq!(next_phase_marker, 44);
    }

    #[test]
    fn disjoint_refreshes_see_alias_writes_on_repeated_submissions() {
        use crate::watch::{PAGE, tests::twin_views, title_memory_remapped};
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};

        if crate::host::host().is_none() {
            assert!(std::env::var_os("VICTORY_TEST_SHADOW_DISJOINT_REQUIRE_GPU").is_none());
            return;
        }
        let len = 3 * PAGE;
        let (title, alias) = twin_views(3);
        crate::shadow::register(title, title as u64, len as u64);
        crate::shadow::register(alias, alias as u64, len as u64);
        let copy = crate::shadow::find(title as u64, len as u64).unwrap();
        let other_copy = crate::shadow::find(alias as u64, len as u64).unwrap();
        assert!(!std::sync::Arc::ptr_eq(&copy, &other_copy));
        assert_eq!(copy.update(title, len), len);
        assert_eq!(other_copy.update(alias, len), len);

        let mut previous_gap = 0;
        for value in [11u32, 21, 31] {
            // SAFETY: both addresses are in our alias view, outside the gap.
            unsafe {
                (alias as *mut u32).write_volatile(value);
                ((alias + 2 * PAGE) as *mut u32).write_volatile(value + 2);
            }
            let mut batch = Batch::default();
            batch.add(&copy, title, PAGE);
            batch.add(&copy, title + 2 * PAGE, PAGE);
            let gate = std::sync::Barrier::new(2);
            std::thread::scope(|scope| {
                let writer = scope.spawn(|| {
                    gate.wait();
                    // SAFETY: the gap is outside both active refresh ranges.
                    unsafe { ((alias + PAGE) as *mut u32).write_volatile(value + 1) };
                });
                gate.wait();
                let bytes = batch.flush();
                writer.join().unwrap();
                assert_eq!(bytes, 2 * PAGE);
            });
            let markers = |s: &crate::shadow::Shadow, base| {
                [0, 1, 2].map(|page| s.read_host((base + page * PAGE) as u64, 1)[0])
            };
            assert_eq!(markers(&copy, title), [value, previous_gap, value + 2]);
            assert_eq!(other_copy.update(alias, len), len, "the other backing sees its own dirt");
            assert_eq!(markers(&other_copy, alias), [value, value + 1, value + 2]);
            batch.add(&copy, title + PAGE, PAGE);
            assert_eq!(batch.flush(), PAGE);
            assert_eq!(markers(&copy, title), [value, value + 1, value + 2]);
            previous_gap = value + 1;
        }

        crate::shadow::unregister(title, title as u64);
        crate::shadow::unregister(alias, alias as u64);
        drop((copy, other_copy));
        for address in [title, alias] {
            // SAFETY: both views were made by this test and are no longer read.
            assert_ne!(unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: address as *mut _ }) }, 0);
            title_memory_remapped(address, len);
        }
    }

    #[test]
    fn disjoint_refreshes_handle_mapping_invalidation_between_submissions() {
        use crate::watch::{PAGE, tests::{free, pool}, title_memory_remapped};
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_DECOMMIT, PAGE_READWRITE, VirtualAlloc, VirtualFree};

        if crate::host::host().is_none() {
            assert!(std::env::var_os("VICTORY_TEST_SHADOW_DISJOINT_REQUIRE_GPU").is_none());
            return;
        }
        let title = pool(3);
        let middle = title + PAGE;
        let len = 3 * PAGE;
        crate::shadow::register(title, title as u64, len as u64);
        let copy = crate::shadow::find(title as u64, len as u64).unwrap();
        // SAFETY: the committed middle page is owned by the test.
        unsafe { (middle as *mut u8).write_volatile(12) };
        assert_eq!(copy.update(title, len), len);
        assert_eq!(copy.read_host(middle as u64, 1)[0], 12);
        // SAFETY: the middle page is committed inside our reservation.
        assert_ne!(unsafe { VirtualFree(middle as *mut _, PAGE, MEM_DECOMMIT) }, 0);
        title_memory_remapped(middle, PAGE);

        let mut batch = Batch::default();
        batch.add(&copy, title, PAGE);
        batch.add(&copy, title + 2 * PAGE, PAGE);
        assert_eq!(batch.flush(), 0);
        assert_eq!(copy.read_host(middle as u64, 1)[0], 12);
        batch.add(&copy, middle, PAGE);
        assert_eq!(batch.flush(), PAGE);
        assert_eq!(copy.read_host(middle as u64, 1)[0], 0);
        drop(batch);

        // SAFETY: recommit only the missing region of our live reservation.
        assert_eq!(unsafe { VirtualAlloc(middle as *mut _, PAGE, MEM_COMMIT, PAGE_READWRITE) } as usize, middle);
        title_memory_remapped(middle, PAGE);
        // SAFETY: the recommitted page is writable and stays live below.
        unsafe { (middle as *mut u8).write_volatile(34) };
        let mut batch = Batch::default();
        batch.add(&copy, middle, PAGE);
        assert_eq!(batch.flush(), PAGE);
        assert_eq!(copy.read_host(middle as u64, 1)[0], 34);

        drop(batch);
        crate::shadow::unregister(title, title as u64);
        drop(copy);
        free(title);
    }

    #[test]
    fn disjoint_partial_refreshes_preserve_pending_gpu_writes() {
        use crate::watch::{PAGE, tests::{free, pool}};

        let Some(host) = crate::host::host() else {
            assert!(std::env::var_os("VICTORY_TEST_SHADOW_DISJOINT_REQUIRE_GPU").is_none());
            return;
        };
        let title = pool(1);
        crate::shadow::register(title, title as u64, PAGE as u64);
        let copy = crate::shadow::find(title as u64, PAGE as u64).unwrap();
        assert_eq!(copy.update(title, PAGE), PAGE);
        let busy = title + 32;
        copy.writing(busy, 4, false);
        for (offset, value) in [(4, 11u8), (16, 22), (32, 33), (40, 44)] {
            // SAFETY: every marker belongs to the test's live title page.
            unsafe { ((title + offset) as *mut u8).write_volatile(value) };
        }
        let mut batch = Batch::default();
        batch.add(&copy, title + 4, 4);
        batch.add(&copy, busy, 12);
        assert_eq!(batch.flush(), 12, "the pending dword and the gap stay untouched");
        let markers = || [4, 16, 32, 40].map(|offset| copy.read_host((title + offset) as u64, 1)[0]);
        assert_eq!(markers(), [11, 0, 0, 44]);

        let (upload, data) = host.upload_buffer(4).unwrap();
        let writer = host.new_list().unwrap();
        // SAFETY: both buffers live through the synchronous submission; the
        // copy targets the pending dword, outside the CPU refreshes above.
        unsafe {
            data.cast::<u32>().write(55);
            writer.list.CopyBufferRegion(&copy.buffer, 32, &upload, 0, 4);
        }
        writer.close();
        host.execute(&[&writer]);
        assert_eq!(copy.copy_back(busy, 4, &Default::default()), 4);
        // SAFETY: the copy-back completed and the test owns this page.
        assert_eq!(unsafe { (busy as *const u8).read_volatile() }, 55);

        batch.add(&copy, title + 16, 4);
        assert_eq!(batch.flush(), 4, "the unobserved partial page stays dirty");
        assert_eq!(markers(), [11, 22, 55, 44]);
        drop(batch);
        crate::shadow::unregister(title, title as u64);
        drop(copy);
        free(title);
    }

    /// The hashing a submission does for textures in physical-page memory,
    /// with every list checking the textures it samples (before) and with
    /// each checked once (after). Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture texture_checks`.
    #[test]
    #[ignore = "benchmark"]
    fn texture_checks_per_submission() {
        const TEXTURES: usize = 48;
        const SIZE: usize = 4 << 20;
        const LISTS: usize = 12;
        let textures: Vec<Arc<Vec<u8>>> = (0..TEXTURES)
            .map(|t| Arc::new((0..SIZE).map(|i| (i * 31 + t) as u8).collect()))
            .collect();
        let submit = |dedupe: bool| {
            let mut batch = Batch::default();
            let mut hashed = 0usize;
            let started = std::time::Instant::now();
            for _ in 0..LISTS {
                for t in &textures {
                    if !dedupe || batch.texture_due(t) {
                        std::hint::black_box(blake3::hash(t));
                        hashed += 1;
                    }
                }
            }
            (started.elapsed(), hashed)
        };
        submit(false); // warm up
        let (before, n_before) = submit(false);
        let (after, n_after) = submit(true);
        assert_eq!((n_before, n_after), (TEXTURES * LISTS, TEXTURES));
        println!(
            "{LISTS} lists sampling {TEXTURES} textures of {} MiB: \
             before {n_before} hashes in {before:?}, after {n_after} in {after:?} ({:.1}x)",
            SIZE >> 20,
            before.as_secs_f64() / after.as_secs_f64()
        );
    }
}

#[cfg(all(test, feature = "diagnostics"))]
mod present_plane_tests {
    use super::*;

    #[test]
    fn compositor_failure_preserves_single_plane_presentation() {
        let Some(host) = host() else {
            eprintln!("skipping PresentX recovery: no D3D12 device");
            return;
        };
        let list = host.new_list().unwrap();
        let make_plane = |width, height, color: &[f32; 4]| {
            let mut texture = None;
            // SAFETY: live device, valid render target description, and an open list.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES {
                            Type: D3D12_HEAP_TYPE_DEFAULT,
                            ..Default::default()
                        },
                        D3D12_HEAP_FLAG_NONE,
                        &D3D12_RESOURCE_DESC {
                            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                            Width: width,
                            Height: height,
                            DepthOrArraySize: 1,
                            MipLevels: 1,
                            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                            SampleDesc: DXGI_SAMPLE_DESC {
                                Count: 1,
                                Quality: 0,
                            },
                            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                            ..Default::default()
                        },
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                        None,
                        &mut texture,
                    )
                    .unwrap();
                let texture = texture.unwrap();
                list.recording()
                    .ClearRenderTargetView(host.rtv(&texture, None), color, None);
                texture
            }
        };
        let scene = make_plane(32, 32, &[1.0, 0.0, 0.0, 1.0]);
        let overlay = make_plane(64, 64, &[0.0, 0.0, 128.0 / 255.0, 128.0 / 255.0]);
        list.close();
        host.execute(&[&list]);
        let state = D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32;
        assert!(
            host.present(&[(&scene, state)], 1).is_some(),
            "single-plane control"
        );
        let mut attempted = false;
        // A failing compositor still presents the frame: its first plane, as
        // a single-plane present does, instead of dropping it.
        assert!(
            host.present_with_compositor(
                &[(&scene, state), (&overlay, state)],
                2,
                |_, _| {
                    attempted = true;
                    Err(windows::core::Error::from_hresult(
                        windows::Win32::Foundation::E_FAIL,
                    ))
                },
                |_, _| {},
            )
            .is_some(),
            "the frame is presented after a compositor failure"
        );
        assert!(attempted);
        assert!(
            host.present(&[(&scene, state)], 3).is_some(),
            "single-plane presentation after compositor initialization failed"
        );
        let mut pixel = None;
        assert!(
            host.present_with_compositor(
                &[(&scene, state), (&overlay, state)],
                4,
                Compositor::new,
                |buffer, footprint| {
                    let mut mapped = std::ptr::null_mut();
                    // SAFETY: execute waited for the pre-Present copy. Read one
                    // RGBA pixel from its footprint, before unmapping the buffer.
                    unsafe {
                        buffer.Map(0, None, Some(&mut mapped)).unwrap();
                        pixel = Some(
                            mapped
                                .cast::<u8>()
                                .add(footprint.Offset as usize)
                                .cast::<[u8; 4]>()
                                .read(),
                        );
                        buffer.Unmap(0, None);
                    }
                },
            )
            .is_some(),
            "compositor initialization can be retried after failure"
        );
        // Premultiplied half-blue over opaque red: neither an ignored overlay
        // (red) nor a replaced scene (blue) may pass this check.
        assert_eq!(pixel, Some([127, 0, 128, 255]));
        // A texture may appear as two planes but has only one resource
        // state. It must remain valid for the next (copy-only) present.
        assert!(host.present(&[(&scene, state), (&scene, state)], 5).is_some());
        assert!(host.present(&[(&scene, state)], 6).is_some());
    }
}
