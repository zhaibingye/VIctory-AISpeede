use super::*;
use crate::{
    descriptor, list,
    object::{IUNKNOWN, slot},
    resource, title_backing,
};
use std::{
    ffi::c_void,
    ptr::null,
    sync::{LazyLock, Mutex},
    time::{Duration, Instant},
};
use victory_memory::{
    PAGE_BYTES,
    authority::{CompletedVersion, Representation},
    gpu_table::ADDRESS_LIMIT,
    virtual_pages::VirtualPages,
};
use windows::Win32::Graphics::{
    Direct3D12::*,
    Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC},
};
use windows_sys::Win32::System::Memory::{
    MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER, MEM_RELEASE, MEM_RESERVE,
    MemExtendedParameterAddressRequirements, PAGE_READWRITE, VirtualAlloc2, VirtualFree,
};

#[path = "present_overlay_draw.rs"]
mod present_overlay_draw;

static BACKING: LazyLock<Mutex<VirtualPages>> =
    LazyLock::new(|| Mutex::new(VirtualPages::default()));
const PAGE: usize = PAGE_BYTES as usize;
const EDGE: usize = 32;
const TIMEOUT: Duration = Duration::from_secs(15);

extern "system" fn allocate(bytes: usize, _: u64) -> *mut c_void {
    let len = bytes.next_multiple_of(PAGE);
    // Match the guest GPU VA range and commit whole, independent 64 KiB pages.
    let mut bounds = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: (1usize << 32) as *mut _,
        HighestEndingAddress: (ADDRESS_LIMIT as usize - 1) as *mut _,
        Alignment: PAGE,
    };
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut bounds).cast();
    // SAFETY: requests a fresh reservation; `bounds` lives through the call.
    let at = unsafe {
        VirtualAlloc2(
            std::ptr::null_mut(),
            std::ptr::null(),
            len,
            MEM_RESERVE | MEM_COMMIT,
            PAGE_READWRITE,
            &mut parameter,
            1,
        )
    };
    if !at.is_null() {
        let mut pages = BACKING.lock().unwrap();
        pages.reserved(at as usize);
        pages.committed(at as usize, at as usize, len);
    }
    at
}

extern "system" fn free(at: *mut c_void, _: u64) {
    let mut pages = BACKING.lock().unwrap();
    pages.released(at as usize);
    // SAFETY: this is the base of an allocation from `allocate`.
    unsafe { VirtualFree(at, 0, MEM_RELEASE) };
}

fn install_backing() {
    resource::use_allocator(allocate, free);
    title_backing::use_backing(
        |ranges, pins| {
            let pages = BACKING.lock().unwrap();
            for &(start, end) in ranges {
                let mut at = start & !(PAGE - 1);
                let end = end.next_multiple_of(PAGE);
                while at < end {
                    let (next, id) = pages.run_at(at, end);
                    pins.remember_with_origin(at, next, id, pages.has_own_reservation(at));
                    at = next;
                }
            }
        },
        // This experiment has no CPU-visible GPU destination.
        |_, _, _, _, _| false,
        |mut at, end, pins, visit| {
            let pages = BACKING.lock().unwrap();
            while at < end {
                let (limit, saved) = pins.run_at(at, end);
                let (next, current) = pages.run_at(at, limit);
                let same = saved.is_some() && saved == current;
                visit(at, same.then_some(at), next - at, same);
                at = next;
            }
        },
    );
}

struct Guest(This);

impl Guest {
    fn new(create: impl FnOnce(*mut This) -> HRESULT) -> Self {
        let mut object = null();
        assert_eq!(create(&mut object), S_OK);
        assert!(!object.is_null());
        Self(object)
    }
}

impl Drop for Guest {
    fn drop(&mut self) {
        slot::<extern "system" fn(This) -> u32>(self.0, 0x10)(self.0);
    }
}

fn texture(cpu_visible: bool) -> Guest {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: if cpu_visible {
            D3D12_HEAP_TYPE_CUSTOM
        } else {
            D3D12_HEAP_TYPE_DEFAULT
        },
        CPUPageProperty: if cpu_visible {
            D3D12_CPU_PAGE_PROPERTY_WRITE_BACK
        } else {
            D3D12_CPU_PAGE_PROPERTY_UNKNOWN
        },
        MemoryPoolPreference: if cpu_visible {
            D3D12_MEMORY_POOL_L0
        } else {
            D3D12_MEMORY_POOL_UNKNOWN
        },
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: EDGE as u64,
        Height: EDGE as u32,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: if cpu_visible {
            D3D12_TEXTURE_LAYOUT_ROW_MAJOR
        } else {
            D3D12_TEXTURE_LAYOUT_UNKNOWN
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        ..Default::default()
    };
    Guest::new(|out| {
        resource::create_committed(
            null(),
            (&raw const heap).cast(),
            0,
            (&raw const desc).cast(),
            0,
            null(),
            &IUNKNOWN,
            out,
        )
    })
}

#[test]
#[ignore = "requires an interactive Windows desktop and D3D12 device"]
fn present_x_resolves_explicit_texture_write_before_copy_resource() {
    // Backing/allocator hooks and the presenter are process-global. Run this
    // single fixture in a child so it cannot intercept other tests' presents.
    const CHILD: &str = "VICTORY_PRESENT_X_CPU_WRITE_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let test = "queue::present_fixture::present_x_resolves_explicit_texture_write_before_copy_resource";
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test, "--ignored", "--nocapture"])
            .env(CHILD, "1")
            .status()
            .expect("start isolated PresentX fixture");
        assert!(status.success(), "isolated PresentX fixture: {status}");
        return;
    }
    install_backing();
    let host = crate::host::host().expect("D3D12 device for the PresentX fixture");
    let desc = QueueDescX {
        kind: 0,
        flags: 0,
        engine_or_pipe: 0,
        queue: 0,
    };
    let q = Guest::new(|out| create_x(null(), &desc, &QUEUE_IIDS[0], out));
    // SAFETY: q retains one of the two queue tables; their PresentX and
    // Signal offsets differ, independently of the current device setting.
    let newer = unsafe { q.0.cast::<*const *const ()>().read() } == NEWER_FUNCTIONS.0.as_ptr();
    let present_x = slot::<extern "system" fn(This, u32, *const Plane, *const u8) -> HRESULT>(
        q.0,
        if newer { 0x100 } else { 0xf0 },
    );
    let execute = slot::<extern "system" fn(This, u32, *const This)>(q.0, 0x50);
    let signal = slot::<extern "system" fn(This, This, u64) -> HRESULT>(
        q.0,
        if newer { 0x80 } else { 0x70 },
    );
    let fence = Guest::new(|out| crate::fence::create(null(), 0, 0, &IUNKNOWN, out));
    let completed = slot::<extern "system" fn(This) -> u64>(fence.0, 0x40);
    let wait_for_queue = |value| {
        assert_eq!(signal(q.0, fence.0, value), S_OK);
        let start = Instant::now();
        while completed(fence.0) < value {
            assert!(
                start.elapsed() < TIMEOUT,
                "guest queue did not reach fence {value}"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    };
    let allocator = Guest::new(|out| list::create_allocator(null(), 0, &IUNKNOWN, out));
    let new_list =
        || Guest::new(|out| list::create_list(null(), 0, 0, allocator.0, null(), &IUNKNOWN, out));
    let submit = |cmd: &Guest, fence| {
        assert_eq!(
            slot::<extern "system" fn(This) -> HRESULT>(cmd.0, 0x48)(cmd.0),
            S_OK
        );
        execute(q.0, 1, &cmd.0);
        wait_for_queue(fence);
    };

    let scene = texture(false);
    let overlay = texture(true);
    let destination = texture(false);
    let [scene_native, overlay_native, destination_native] = [&scene, &overlay, &destination]
        .map(|r| resource::host_copy(r.0).expect("each guest resource has a native texture"));
    assert!(overlay_native.exact_committed_bytes().is_some());
    assert!(!overlay_native.has_pointer_exposure());
    assert_ne!(scene_native.texture, overlay_native.texture);
    assert_ne!(overlay_native.texture, destination_native.texture);

    // Neither an optimization hint nor CPU initialization: use two full
    // guest clears and wait until their real native write is published.
    let prime = new_list();
    let clear = slot::<extern "system" fn(This, u64, *const [f32; 4], u32, *const list::Rect)>(
        prime.0, 0x1a0,
    );
    for (resource, color) in [
        (&scene, [1., 0., 0., 1.]),
        (&overlay, [0., 0., 128. / 255., 128. / 255.]),
    ] {
        let mut descriptor = [0u64; 14];
        let handle = descriptor.as_mut_ptr() as u64;
        descriptor::create_rtv(null(), resource.0, null(), handle);
        clear(prime.0, handle, &color, 0, null());
    }
    submit(&prime, 1);
    let version = |native: &resource::HostTexture| -> CompletedVersion {
        native.native_authority.lock().unwrap()[&None]
    };
    let assert_completed_native = |v: CompletedVersion| {
        assert_eq!(v.version.representation, Representation::NativeGpu);
        assert!(v.fence.0 > 0 && host.completed_fence().unwrap().0 >= v.fence.0);
    };
    let (scene_a, overlay_a) = (version(&scene_native), version(&overlay_native));
    assert_completed_native(scene_a);
    assert_completed_native(overlay_a);

    let planes = [&scene.0, &overlay.0].map(|resources| Plane {
        token: 0,
        count: 1,
        resources,
        _rest: [0; 6],
    });
    let present = |label: &str, fence| {
        let observed = host.observe_next_present();
        assert_eq!(
            present_x(q.0, planes.len() as u32, planes.as_ptr(), null()),
            S_OK
        );
        // Wait for the guest presentation job to finish, not just its DXGI call.
        wait_for_queue(fence);
        let observed = observed
            .recv_timeout(TIMEOUT)
            .expect("fenced readback from DXGI S_OK");
        eprintln!(
            "PresentX {label}: RGBA {:?}, composited={}",
            observed.rgba, observed.composited
        );
        assert!(observed.composited, "{label} used the fallback path");
        observed.rgba
    };
    let assert_blend = |actual: [u8; 4], expected: [u8; 4], label: &str| {
        assert!(
            actual
                .into_iter()
                .zip(expected)
                .all(|(a, b)| a.abs_diff(b) <= 1),
            "{label}: actual RGBA {actual:?}, expected {expected:?} (±1)"
        );
    };
    let a = present("A", 2);
    // A premultiplied half-blue overlay over opaque red requires both sources.
    assert_blend(a, [127, 0, 128, 255], "native A");
    assert_eq!(present("no write", 3), a);
    assert_eq!(version(&scene_native), scene_a);
    assert_eq!(version(&overlay_native), overlay_a);

    // Write the entire public subresource, without Map or a persistent address.
    const B: [u8; 4] = [0, 128, 0, 128];
    let pixels_b = [B; EDGE * EDGE];
    let write = slot::<
        extern "system" fn(This, u32, *const [u32; 6], *const u8, u32, u32) -> HRESULT,
    >(overlay.0, 0x60);
    assert_eq!(
        write(
            overlay.0,
            0,
            null(),
            pixels_b.as_ptr().cast(),
            (EDGE * 4) as u32,
            size_of_val(&pixels_b) as u32
        ),
        S_OK
    );
    let overlay_b = version(&overlay_native);
    assert_eq!(
        overlay_b.version.representation,
        Representation::CompatibleBytes
    );
    assert_eq!(overlay_b.version.number, overlay_a.version.number + 1);
    assert!(std::sync::Arc::ptr_eq(
        &overlay_native,
        &resource::host_copy(overlay.0).unwrap()
    ));
    let overlay_state = overlay_native.state.load(Ordering::Acquire);
    let shown_b = present("after WriteToSubresource, before CopyResource", 4);
    let presented_version = version(&overlay_native);
    assert_eq!(overlay_native.state.load(Ordering::Acquire), overlay_state);

    let copying = new_list();
    slot::<extern "system" fn(This, This, This)>(copying.0, 0x88)(
        copying.0,
        destination.0,
        overlay.0,
    );
    submit(&copying, 5);
    assert_completed_native(version(&destination_native));
    assert_eq!(
        version(&overlay_native).version.number,
        overlay_b.version.number
    );
    assert_eq!(version(&scene_native), scene_a);

    // Read the native destination (not either texture's guest backing) after
    // CopyResource. The additional list only reads it, restoring its state.
    let read = host.new_list().unwrap();
    let command = read.recording();
    let native = &destination_native.texture;
    let before = D3D12_RESOURCE_STATES(destination_native.state.load(Ordering::Acquire) as i32);
    let source = D3D12_RESOURCE_STATE_COPY_SOURCE;
    // SAFETY: the guest fence already retired the only writer, and the
    // command list is open. Keep the destination's published state intact.
    let (data, layout) = unsafe {
        if before != source {
            command.ResourceBarrier(&[crate::host::transition(native, before, source)]);
        }
        let result = crate::capture::read_back(&host.device, command, native).unwrap();
        if before != source {
            command.ResourceBarrier(&[crate::host::transition(native, source, before)]);
        }
        result
    };
    read.close();
    host.execute(&[&read]); // waits on the real readback fence
    // SAFETY: readback completed; RGBA8 rows are covered by this footprint.
    let (first, full_b) = unsafe {
        let mut mapped = std::ptr::null_mut();
        data.Map(0, None, Some(&mut mapped)).unwrap();
        let start = mapped.cast::<u8>().add(layout.Offset as usize);
        let first = start.cast::<[u8; 4]>().read();
        let full_b = (0..EDGE).all(|y| {
            let row = start.add(y * layout.Footprint.RowPitch as usize);
            std::slice::from_raw_parts(row.cast::<[u8; 4]>(), EDGE)
                .iter()
                .all(|&pixel| pixel == B)
        });
        data.Unmap(0, None);
        (first, full_b)
    };
    eprintln!("CopyResource from the same overlay: RGBA {first:?}, full texture B={full_b}");
    assert!(
        full_b,
        "native CopyResource did not read exact B: first RGBA {first:?}"
    );
    // Leave the regression assertion last, so stale PresentX cannot suppress
    // the same-source CopyResource control and its actual GPU evidence.
    assert_blend(
        shown_b,
        [127, 128, 0, 255],
        "PresentX must composite the explicit B write",
    );
    assert_completed_native(presented_version);
    assert_eq!(presented_version.version.number, overlay_b.version.number);
    assert!(presented_version.fence.0 > overlay_b.fence.0);
}

#[test]
#[ignore = "requires an interactive Windows desktop and hardware D3D12 device"]
fn present_x_preserves_native_overlay_around_partial_blend() {
    const CHILD: &str = "VICTORY_PRESENT_X_NATIVE_BLEND_CHILD";
    if std::env::var_os(CHILD).is_none() {
        // The backing hooks, presenter and capture settings are process-global.
        let captures = tempfile::tempdir().expect("temporary PresentX capture directory");
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "queue::present_fixture::present_x_preserves_native_overlay_around_partial_blend",
                "--ignored",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("VICTORY_CAPTURE", captures.path())
            .env_remove("VICTORY_CAPTURE_FRAMES")
            .env_remove("VICTORY_CAPTURE_WORLD")
            .env_remove("VICTORY_DUMP_TARGETS")
            .env_remove("VICTORY_D3D12_WARP")
            .status()
            .expect("start isolated native PresentX fixture");
        assert!(
            status.success(),
            "isolated native PresentX fixture: {status}"
        );
        return;
    }
    install_backing();
    crate::host::on_failure(|reason| {
        eprintln!("native PresentX fixture: {reason}");
        std::process::exit(1);
    });
    let host = crate::host::host().expect("hardware D3D12 device for the native PresentX fixture");
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1,
        IDXGIFactory4,
    };
    // SAFETY: plain DXGI queries for the adapter selected by the live device.
    let adapter = unsafe {
        CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            .and_then(|f| f.EnumAdapterByLuid::<IDXGIAdapter1>(host.device.GetAdapterLuid()))
            .and_then(|a| a.GetDesc1())
            .expect("DXGI adapter description")
    };
    assert_eq!(adapter.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32, 0);

    let q = Guest::new(|out| {
        let desc = QueueDescX {
            kind: 0,
            flags: 0,
            engine_or_pipe: 0,
            queue: 0,
        };
        create_x(null(), &desc, &QUEUE_IIDS[0], out)
    });
    // SAFETY: q retains its queue table, which selects the console slots.
    let newer = unsafe { q.0.cast::<*const *const ()>().read() } == NEWER_FUNCTIONS.0.as_ptr();
    let execute = slot::<extern "system" fn(This, u32, *const This, u64, u32)>(
        q.0,
        if newer { 0x118 } else { 0x108 },
    );
    let present_x = slot::<extern "system" fn(This, u32, *const Plane, *const u8) -> HRESULT>(
        q.0,
        if newer { 0x100 } else { 0xf0 },
    );
    let signal = slot::<extern "system" fn(This, This, u64) -> HRESULT>(
        q.0,
        if newer { 0x80 } else { 0x70 },
    );
    let fence = Guest::new(|out| crate::fence::create(null(), 0, 0, &IUNKNOWN, out));
    let completed = slot::<extern "system" fn(This) -> u64>(fence.0, 0x40);
    let allocator = Guest::new(|out| list::create_allocator(null(), 0, &IUNKNOWN, out));
    let new_list =
        || Guest::new(|out| list::create_list(null(), 0, 0, allocator.0, null(), &IUNKNOWN, out));
    let close = |cmd: &Guest| {
        assert_eq!(
            slot::<extern "system" fn(This) -> HRESULT>(cmd.0, 0x48)(cmd.0),
            S_OK
        );
    };

    let scene = texture(false);
    let overlay = texture(false);
    let [scene_native, overlay_native] = [&scene, &overlay]
        .map(|r| resource::host_copy(r.0).expect("guest resource has a native texture"));
    assert_ne!(scene_native.texture, overlay_native.texture);
    for native in [&scene_native, &overlay_native] {
        assert!(native.exact_committed_bytes().is_none());
        assert!(!native.has_pointer_exposure());
        assert!(native.native_authority.lock().unwrap().is_empty());
    }

    let [scene_clear, overlay_clear, draw] = [new_list(), new_list(), new_list()];
    let mut scene_descriptor = [0u64; 14];
    let mut overlay_descriptor = [0u64; 14];
    let scene_rtv = scene_descriptor.as_mut_ptr() as u64;
    let overlay_rtv = overlay_descriptor.as_mut_ptr() as u64;
    for (cmd, target, rtv, color) in [
        (&scene_clear, &scene, scene_rtv, [0., 0., 1., 1.]),
        (&overlay_clear, &overlay, overlay_rtv, [0., 0.5, 0., 0.5]),
    ] {
        descriptor::create_rtv(null(), target.0, null(), rtv);
        slot::<extern "system" fn(This, u64, *const [f32; 4], u32, *const list::Rect)>(
            cmd.0, 0x1a0,
        )(cmd.0, rtv, &color, 0, null());
        close(cmd);
    }
    // SAFETY: draw is an open guest list and the overlay RTV names a live
    // resource. Retain the returned draw objects until the queue retires.
    let draw_objects = unsafe { present_overlay_draw::record_overlay_draw(draw.0, overlay_rtv) };
    close(&draw);

    let planes = [&scene.0, &overlay.0].map(|resources| Plane {
        token: 0,
        count: 1,
        resources,
        _rest: [0; 6],
    });
    let observed = host.observe_next_present();
    execute(q.0, 1, &scene_clear.0, 0, 0);
    // Queue-only boundaries prevent the native window from combining the
    // two independent clears (or either clear with the draw). No CPU wait
    // or intermediate PresentX prepares the overlay for the draw.
    assert_eq!(signal(q.0, fence.0, 1), S_OK);
    execute(q.0, 1, &overlay_clear.0, 0, 0);
    assert_eq!(signal(q.0, fence.0, 2), S_OK);
    execute(q.0, 1, &draw.0, 0, 0);
    assert_eq!(
        present_x(q.0, planes.len() as u32, planes.as_ptr(), null()),
        S_OK
    );
    assert_eq!(signal(q.0, fence.0, 3), S_OK);
    let start = Instant::now();
    while completed(fence.0) < 3 {
        assert!(start.elapsed() < TIMEOUT, "native PresentX did not retire");
        std::thread::sleep(Duration::from_millis(1));
    }
    drop(draw_objects);

    // The guest return means enqueue succeeded. Only the host observer
    // confirms the real readback fence, dual-plane dispatch and DXGI S_OK.
    let observed = observed
        .recv_timeout(TIMEOUT)
        .expect("fenced readback from DXGI S_OK");
    assert!(
        observed.composited,
        "native PresentX used the single-plane fallback"
    );
    let version = |native: &resource::HostTexture| -> CompletedVersion {
        native.native_authority.lock().unwrap()[&None]
    };
    let (scene_version, overlay_version) = (version(&scene_native), version(&overlay_native));
    let present_fence = host.completed_fence().expect("real fence after PresentX");
    for v in [scene_version, overlay_version] {
        assert_eq!(v.version.representation, Representation::NativeGpu);
        assert!(v.fence.0 > 0 && v.fence.0 < present_fence.0);
    }
    assert_eq!(scene_version.version.number, 1, "primary clear only");
    assert_eq!(overlay_version.version.number, 2, "overlay clear then draw");
    assert!(scene_version.fence.0 < overlay_version.fence.0);

    // The left half gains red and halves its old green. The right half is
    // the clear-only control. Both halves show blue through the overlay.
    const DRAWN: [u8; 3] = [128, 64, 127];
    const RETAINED: [u8; 3] = [0, 128, 127];
    let matches_rgb = |actual: &[u8], expected: [u8; 3]| {
        actual.iter().zip(expected).all(|(a, b)| a.abs_diff(b) <= 1)
    };
    assert!(
        matches_rgb(&observed.rgba[..3], DRAWN) && observed.rgba[3].abs_diff(255) <= 1,
        "unscaled PresentX (0,0): RGBA {:?}, expected {DRAWN:?} + opaque alpha",
        observed.rgba
    );
    let rows = native_present_capture();
    assert_eq!(
        &rows[1..4],
        &observed.rgba[..3],
        "same back-buffer readback"
    );
    const CAPTURE_EDGE: usize = EDGE / 2;
    for (y, row) in rows.chunks_exact(1 + CAPTURE_EDGE * 3).enumerate() {
        assert_eq!(row[0], 0, "diagnostic capture uses unfiltered rows");
        for (x, actual) in row[1..].chunks_exact(3).enumerate() {
            let expected = if x < CAPTURE_EDGE / 2 {
                DRAWN
            } else {
                RETAINED
            };
            assert!(
                matches_rgb(actual, expected),
                "native PresentX ({},{}): RGB {actual:?}, expected {expected:?} (±1)",
                x * 2,
                y * 2
            );
        }
    }
    let retained = 1 + 3 * (CAPTURE_EDGE / 2);
    eprintln!(
        "native PresentX: drawn RGBA {:?}, retained RGB {:?}; verified all RGB samples of the {CAPTURE_EDGE}x{CAPTURE_EDGE} capture",
        observed.rgba,
        &rows[retained..retained + 3]
    );
}

fn native_present_capture() -> Vec<u8> {
    let folder = std::env::var_os("VICTORY_CAPTURE").expect("fixture capture directory");
    let png = std::fs::read(std::path::Path::new(&folder).join("frame-1.png"))
        .expect("diagnostic capture of the fenced back buffer");
    // The built-in capture writer emits IHDR, one IDAT and IEND. It takes
    // every second pixel of the real back buffer (no filtered resampling).
    assert!(png.starts_with(b"\x89PNG\r\n\x1a\n\0\0\0\x0dIHDR"));
    let word = |at| u32::from_be_bytes(png[at..at + 4].try_into().unwrap()) as usize;
    assert_eq!((word(16), word(20)), (EDGE / 2, EDGE / 2));
    assert_eq!(&png[24..29], &[8, 2, 0, 0, 0], "8-bit RGB PNG");
    assert_eq!(&png[37..41], b"IDAT");
    let len = (EDGE / 2) * (1 + (EDGE / 2) * 3);
    let rows =
        miniz_oxide::inflate::decompress_to_vec_zlib_with_limit(&png[41..41 + word(33)], len)
            .expect("inflate diagnostic RGB rows");
    assert_eq!(rows.len(), len);
    rows
}
