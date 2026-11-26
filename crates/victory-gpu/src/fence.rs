//! Guest fences: a CPU-visible value and its waiters. Native GPU
//! completion alone must not publish this value for work with writeback.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;
use std::sync::{Arc, Condvar, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::object::{self, E_NOTIMPL, FENCE, HRESULT, Iid, Object, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

pub(crate) const FENCE_IIDS: [Iid; 5] = [
    guid(
        0x0a753dcf,
        0xc4d8,
        0x4b91,
        [0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76],
    ),
    // ID3D12Fence1.
    guid(
        0x433685fe,
        0xe22b,
        0x4ca0,
        [0xa8, 0xdb, 0xb5, 0xb4, 0xf4, 0xdd, 0x0e, 0x4a],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
struct Fence {
    object: Object,
    flags: u32,
    state: Arc<FenceState>,
}

struct FenceState {
    // The address exposed to the title is the address of this atomic.
    value: AtomicU64,
    waiters: Mutex<Vec<(u64, PendingEvent)>>,
    published: Condvar,
    #[cfg(all(windows, feature = "paged-title"))]
    address_exposed: std::sync::atomic::AtomicBool,
}

struct PendingEvent(usize);

impl PendingEvent {
    fn retain(event: *mut c_void) -> Option<Self> {
        #[cfg(all(windows, feature = "paged-title"))]
        {
            use windows_sys::Win32::Foundation::{DUPLICATE_SAME_ACCESS, DuplicateHandle};
            use windows_sys::Win32::System::Threading::GetCurrentProcess;
            let mut duplicate = std::ptr::null_mut();
            // SAFETY: duplicate the caller's handle into our process;
            // ownership doesn't depend on its numeric handle staying open.
            let ok = unsafe {
                let process = GetCurrentProcess();
                DuplicateHandle(process, event, process, &mut duplicate, 0, 0, DUPLICATE_SAME_ACCESS)
            };
            (ok != 0).then_some(Self(duplicate as usize))
        }
        #[cfg(not(all(windows, feature = "paged-title")))]
        Some(Self(event as usize))
    }
}

#[cfg(all(windows, feature = "paged-title"))]
impl Drop for PendingEvent {
    fn drop(&mut self) {
        // SAFETY: this is our own duplicate, closed exactly once.
        unsafe { windows_sys::Win32::Foundation::CloseHandle(self.0 as *mut c_void) };
    }
}

impl FenceState {
    fn wake_events(waiters: &mut Vec<(u64, PendingEvent)>, value: u64) {
        waiters.retain(|(wanted, event)| {
            let reached = *wanted <= value;
            if reached { set_event(event.0); }
            !reached
        });
    }
}

#[cfg(all(windows, feature = "paged-title"))]
mod direct_events {
    use super::*;
    use std::sync::{OnceLock, Weak};
    use std::time::Duration;

    #[derive(Default)]
    struct Observer {
        fences: Mutex<Vec<Weak<FenceState>>>,
        changed: Condvar,
    }

    pub(super) fn watch(state: &Arc<FenceState>) {
        if !state.address_exposed.load(Ordering::Acquire) {
            return;
        }
        static OBSERVER: OnceLock<Arc<Observer>> = OnceLock::new();
        let observer = OBSERVER.get_or_init(|| {
            let observer = Arc::new(Observer::default());
            let worker = observer.clone();
            std::thread::Builder::new().name("vnme-fence-events".into())
                .spawn(move || worker.run())
                .unwrap_or_else(|e| crate::host::fail(format_args!("cannot start VNME fence observer: {e}")));
            observer
        });
        let mut fences = observer.fences.lock().unwrap_or_else(|e| e.into_inner());
        let shared = Arc::downgrade(state);
        if !fences.iter().any(|f| f.ptr_eq(&shared)) {
            fences.push(shared);
        }
        observer.changed.notify_one();
    }

    impl Observer {
        fn run(&self) -> ! {
            let mut fences = self.fences.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                fences.retain(|weak| {
                    let Some(state) = weak.upgrade() else { return false; };
                    let mut waiters = state.waiters.lock().unwrap_or_else(|e| e.into_inner());
                    let value = state.value.load(Ordering::Acquire);
                    FenceState::wake_events(&mut waiters, value);
                    !waiters.is_empty()
                });
                fences = if fences.is_empty() {
                    self.changed.wait(fences).unwrap_or_else(|e| e.into_inner())
                } else {
                    self.changed.wait_timeout(fences, Duration::from_millis(5))
                        .unwrap_or_else(|e| e.into_inner()).0
                };
            }
        }
    }
}

fn fence<'a>(this: This) -> &'a Fence {
    // SAFETY: called from fence methods only.
    unsafe { &*this.cast::<Fence>() }
}

pub(crate) extern "system" fn create(
    _: This,
    initial: u64,
    flags: u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::debug!(initial, flags, "fence");
    let fence = Box::new(Fence {
        object: Object::new(&FUNCTIONS),
        flags,
        state: Arc::new(FenceState {
            value: AtomicU64::new(initial),
            waiters: Mutex::new(Vec::new()),
            published: Condvar::new(),
            #[cfg(all(windows, feature = "paged-title"))]
            address_exposed: std::sync::atomic::AtomicBool::new(false),
        }),
    });
    object::hand_out(FENCE, fence, iid, &FENCE_IIDS, out)
}

/// A guest fence kept alive while its signal or dependency is queued.
#[cfg(all(windows, feature = "paged-title"))]
pub(crate) struct GuestFenceRef {
    object: usize,
    value: u64,
}

#[cfg(all(windows, feature = "paged-title"))]
impl GuestFenceRef {
    pub(crate) fn new(this: This, value: u64) -> Self {
        object::add_ref(this);
        Self { object: this as usize, value }
    }

    pub(crate) fn reached(&self) -> bool {
        completed(self.object as This) >= self.value
    }

    pub(crate) fn publish(self) {
        signal(self.object as This, self.value);
    }
}

#[cfg(all(windows, feature = "paged-title"))]
impl Drop for GuestFenceRef {
    fn drop(&mut self) { release(self.object as This); }
}

fn set_event(event: usize) {
    #[cfg(windows)]
    // SAFETY: the title passed an event handle.
    unsafe {
        windows_sys::Win32::System::Threading::SetEvent(event as *mut c_void)
    };
    #[cfg(not(windows))]
    let _ = event;
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        FENCE,
        iid,
        object::is(iid, &FENCE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Fence>(this)
}
extern "system" fn completed(this: This) -> u64 {
    fence(this).state.value.load(Ordering::Acquire)
}
extern "system" fn set_event_on_completion(this: This, value: u64, event: *mut c_void) -> HRESULT {
    let fence = &fence(this).state;
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    if fence.value.load(Ordering::Acquire) >= value {
        if !event.is_null() {
            set_event(event as usize);
        }
        return S_OK;
    }
    if event.is_null() {
        if !cfg!(feature = "paged-title") {
            return E_NOTIMPL;
        }
        // CPU wait only: release the fence lock while sleeping. Explicit
        // signals wake this immediately. Occasional rechecks observe
        // title writes through the exposed value address, which do not notify.
        while fence.value.load(Ordering::Acquire) < value {
            (waiters, _) = fence.published
                .wait_timeout(waiters, std::time::Duration::from_millis(10))
                .unwrap_or_else(|e| e.into_inner());
        }
        return S_OK;
    }
    let Some(event) = PendingEvent::retain(event) else { return crate::object::E_INVALIDARG; };
    waiters.push((value, event));
    drop(waiters);
    #[cfg(all(windows, feature = "paged-title"))]
    direct_events::watch(fence);
    S_OK
}
pub(crate) extern "system" fn signal(this: This, value: u64) -> HRESULT {
    let fence = &fence(this).state;
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    fence.value.store(value, Ordering::Release);
    FenceState::wake_events(&mut waiters, value);
    fence.published.notify_all();
    #[cfg(all(windows, feature = "paged-title"))]
    crate::title_timeline::fence_published();
    S_OK
}

extern "system" fn creation_flags(this: This) -> u32 {
    fence(this).flags
}
/// Where the value lives (GPU addresses are CPU addresses).
extern "system" fn gpu_address(this: This) -> u64 {
    let fence = &fence(this).state;
    #[cfg(all(windows, feature = "paged-title"))]
    {
        if !fence.address_exposed.swap(true, Ordering::AcqRel) {
            let pending = !fence.waiters.lock().unwrap_or_else(|e| e.into_inner()).is_empty();
            if pending { direct_events::watch(fence); }
        }
    }
    (&raw const fence.value) as u64
}

/// IUnknown 3, object 4, device child 1, fence 3, fence 1 (2, a console
/// one among them).
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<FENCE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = completed as *const (); // 0x40
    f[9] = set_event_on_completion as *const (); // 0x48
    f[10] = signal as *const (); // 0x50
    f[11] = creation_flags as *const (); // 0x58
    f[12] = gpu_address as *const (); // 0x60
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    #[cfg(windows)]
    #[test]
    fn signals_set_waiting_events() {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

        let mut f = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 1, 0, &FENCE_IIDS[0], &mut f), S_OK);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(f, 0x40)(f), 1);
        // SAFETY: plain auto-reset event.
        let event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
        assert!(!event.is_null());
        let wait = slot::<extern "system" fn(This, u64, *mut c_void) -> HRESULT>(f, 0x48);
        let signal = slot::<extern "system" fn(This, u64) -> HRESULT>(f, 0x50);
        assert_eq!(wait(f, 5, event), S_OK);
        // SAFETY: a valid event.
        assert_ne!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(signal(f, 5), S_OK);
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(wait(f, 5, event), S_OK, "already reached");
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        if cfg!(feature = "paged-title") {
            let handle = f as usize;
            std::thread::scope(|scope| {
                scope.spawn(move || { assert_eq!(signal(handle as This, 9), S_OK); });
                assert_eq!(wait(f, 9, std::ptr::null_mut()), S_OK);
            });
        } else {
            assert_eq!(wait(f, 9, std::ptr::null_mut()), E_NOTIMPL);
        }
        // Grand Theft Auto VI reads the value at the fence's address.
        let at = slot::<extern "system" fn(This) -> u64>(f, 0x60)(f);
        // SAFETY: the fence's value.
        assert_eq!(unsafe { *(at as *const u64) }, 5);
        // SAFETY: ours.
        unsafe { CloseHandle(event) };
        assert_eq!(slot::<extern "system" fn(This) -> u32>(f, 0x10)(f), 0);
    }

    #[test]
    fn fence1_interface_is_created_and_queried() {
        const FENCE1_IID: Iid = guid(
            0x433685fe,
            0xe22b,
            0x4ca0,
            [0xa8, 0xdb, 0xb5, 0xb4, 0xf4, 0xdd, 0x0e, 0x4a],
        );
        type Query = extern "system" fn(This, *const Iid, *mut This) -> HRESULT;
        type Flags = extern "system" fn(This) -> u32;
        type Release = extern "system" fn(This) -> u32;

        // Direct creation as ID3D12Fence1, and GetCreationFlags at slot 0x58.
        let mut f1 = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 3, 0x5, &FENCE1_IID, &mut f1), S_OK);
        assert!(!f1.is_null());
        assert_eq!(slot::<Flags>(f1, 0x58)(f1), 0x5);
        assert_eq!(slot::<Release>(f1, 0x10)(f1), 0);

        // QueryInterface from ID3D12Fence to ID3D12Fence1 returns the same object.
        let mut f = std::ptr::null();
        assert_eq!(
            create(std::ptr::null(), 0, 0x2, &FENCE_IIDS[0], &mut f),
            S_OK
        );
        let mut queried = std::ptr::null();
        assert_eq!(slot::<Query>(f, 0x00)(f, &FENCE1_IID, &mut queried), S_OK);
        assert_eq!(queried, f);
        assert_eq!(slot::<Flags>(queried, 0x58)(queried), 0x2);
        assert_eq!(slot::<Release>(queried, 0x10)(queried), 1);
        assert_eq!(slot::<Release>(f, 0x10)(f), 0);
    }
}
