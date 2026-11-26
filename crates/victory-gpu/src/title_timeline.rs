//! Schedule logical title work without putting blocked waits on the shared
//! native direct queue. The runner holds no scheduling locks during work.
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Duration;

use victory_memory::timeline::{Item, Timeline};

use crate::fence::GuestFenceRef;
use crate::host::fail;

type Job = Box<dyn FnOnce() + Send>;

pub(crate) struct Lane(Mutex<Timeline<GuestFenceRef, Job>>);

impl Lane {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self(Mutex::new(Timeline::new(256))))
    }

    pub(crate) fn wait(lane: &Arc<Self>, fence: GuestFenceRef) -> bool {
        enqueue(lane, Item::Dependency(fence))
    }

    pub(crate) fn work(lane: &Arc<Self>, work: impl FnOnce() + Send + 'static) -> bool {
        enqueue(lane, Item::Work(Box::new(work)))
    }
}

#[derive(Default)]
struct Scheduler {
    lanes: Mutex<VecDeque<Arc<Lane>>>,
    changed: Condvar,
}

static SCHEDULER: OnceLock<Arc<Scheduler>> = OnceLock::new();

fn enqueue(lane: &Arc<Lane>, item: Item<GuestFenceRef, Job>) -> bool {
    let scheduler = SCHEDULER.get_or_init(|| {
        let scheduler = Arc::new(Scheduler::default());
        let worker = scheduler.clone();
        std::thread::Builder::new().name("vnme-title-queue".into())
            .spawn(move || worker.run())
            .unwrap_or_else(|e| fail(format_args!("cannot start VNME title scheduler: {e}")));
        scheduler
    });
    let mut lanes = scheduler.lanes.lock().unwrap_or_else(|e| e.into_inner());
    let mut timeline = lane.0.lock().unwrap_or_else(|e| e.into_inner());
    if timeline.enqueue(item).is_err() {
        return false;
    }
    if !lanes.iter().any(|active| Arc::ptr_eq(active, lane)) {
        lanes.push_back(lane.clone());
    }
    scheduler.changed.notify_one();
    true
}

pub(crate) fn fence_published() {
    if let Some(scheduler) = SCHEDULER.get() {
        // A timeout also covers direct guest-visible address stores,
        // which bypass the fence API and cannot notify the runner.
        scheduler.changed.notify_one();
    }
}

impl Scheduler {
    fn run(&self) -> ! {
        loop {
            let mut lanes = self.lanes.lock().unwrap_or_else(|e| e.into_inner());
            let (lane, ready) = loop {
                let mut selected = None;
                for _ in 0..lanes.len() {
                    let lane = lanes.pop_front().expect("scheduled lane exists");
                    let mut timeline = lane.0.lock().unwrap_or_else(|e| e.into_inner());
                    let ready = timeline.take_ready(GuestFenceRef::reached);
                    let active = !timeline.is_idle();
                    drop(timeline);
                    if active {
                        lanes.push_back(lane.clone());
                    }
                    if let Some(ready) = ready {
                        selected = Some((lane, ready));
                        break;
                    }
                }
                if let Some(selected) = selected {
                    break selected;
                }
                lanes = if lanes.is_empty() {
                    self.changed.wait(lanes).unwrap_or_else(|e| e.into_inner())
                } else {
                    self.changed.wait_timeout(lanes, Duration::from_millis(5))
                        .unwrap_or_else(|e| e.into_inner()).0
                };
            };
            drop(lanes);
            (ready.work)();
            let mut timeline = lane.0.lock().unwrap_or_else(|e| e.into_inner());
            if !timeline.retire(ready.ticket) {
                fail(format_args!("VNME title queue retired the wrong operation"));
            }
        }
    }
}
