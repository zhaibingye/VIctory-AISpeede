//! Schedule logical title work without putting blocked waits on the shared
//! native direct queue. The runner holds no scheduling locks during work.
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Duration;

use victory_memory::timeline::{Item, Timeline};

use crate::fence::GuestFenceRef;
use crate::host::{Host, fail};
use crate::descriptor::host::HostHeap;
use crate::page_queue::ClosedTitleBatch;

enum Job {
    Execute { host: &'static Host, heap: &'static HostHeap, batches: Vec<ClosedTitleBatch> },
    Boundary(Box<dyn FnOnce() + Send>),
}

impl Job {
    fn run(self) {
        match self {
            Self::Execute { host, heap, batches } => {
                if let Err(error) = ClosedTitleBatch::execute_adjacent(host, heap, batches) {
                    fail(format_args!("paged title submission rejected: {error:?}"));
                }
            }
            Self::Boundary(work) => work(),
        }
    }

    fn collect_adjacent(&mut self, timeline: &mut Timeline<GuestFenceRef, Self>, ticket: victory_memory::timeline::Ticket) {
        let Self::Execute { host, heap, batches } = self else { return; };
        let mut commands = batches.iter().map(ClosedTitleBatch::len).sum::<usize>();
        // Bound both the number and command count; other logical lanes
        // can run after this group retires.
        for _ in 0..3 {
            let next = timeline.take_adjacent(ticket, |next| match next {
                Self::Execute { host: h, heap: d, batches: next } =>
                    std::ptr::eq(*host, *h) && std::ptr::eq(*heap, *d)
                    && commands.saturating_add(next[0].len()) <= 64
                    && next[0].page_free_candidate() && batches[0].page_free_candidate(),
                Self::Boundary(_) => false,
            });
            let Some(Self::Execute { batches: next, .. }) = next else { break; };
            commands += next[0].len();
            batches.extend(next);
        }
    }
}

pub(crate) struct Lane(Mutex<Timeline<GuestFenceRef, Job>>);

impl Lane {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self(Mutex::new(Timeline::new(256))))
    }

    pub(crate) fn wait(lane: &Arc<Self>, fence: GuestFenceRef) -> bool {
        enqueue(lane, Item::Dependency(fence))
    }

    pub(crate) fn execute(lane: &Arc<Self>, host: &'static Host, heap: &'static HostHeap, batch: ClosedTitleBatch) -> bool {
        enqueue(lane, Item::Work(Job::Execute { host, heap, batches: vec![batch] }))
    }

    pub(crate) fn work(lane: &Arc<Self>, work: impl FnOnce() + Send + 'static) -> bool {
        enqueue(lane, Item::Work(Job::Boundary(Box::new(work))))
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
                    let mut ready = timeline.take_ready(GuestFenceRef::reached);
                    if let Some(work) = &mut ready {
                        work.work.collect_adjacent(&mut timeline, work.ticket);
                    }
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
            ready.work.run();
            let mut timeline = lane.0.lock().unwrap_or_else(|e| e.into_inner());
            if !timeline.retire(ready.ticket) {
                fail(format_args!("VNME title queue retired the wrong operation"));
            }
        }
    }
}
