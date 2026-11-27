//! Exclusion between whole-batch snapshots and cooperating physical writers.
//! This tracks backing identities, not VAs or writes through unregistered paths.

use std::sync::{Arc, Condvar, Mutex};

use super::PhysicalPage;

// Inclusive endpoints, sorted by first page. Each Arc is also the unique
// registration token: different guards can cover exactly the same pages.
type Run = (PhysicalPage, PhysicalPage);
type Runs = Arc<[Run]>;

struct Active {
    snapshots: Vec<Runs>,
    writers: Vec<Runs>,
}

static ACTIVE: Mutex<Active> = Mutex::new(Active {
    snapshots: Vec::new(),
    writers: Vec::new(),
});
static RELEASED: Condvar = Condvar::new();

fn collect_runs(input: impl Iterator<Item = (PhysicalPage, usize)>) -> Runs {
    let mut runs: Vec<Run> = Vec::new();
    for (first, count) in input {
        if first.is_private() || count == 0 {
            continue;
        }
        let last = first.offset(count - 1);
        // A batch already in physical order needs only one entry per extent.
        if let Some((start, end)) = runs.last_mut()
            && *start <= first
            && first <= end.offset(1)
        {
            *end = (*end).max(last);
        } else {
            runs.push((first, last));
        }
    }
    runs.sort_unstable();
    runs.into()
}

fn conflict(mut left: &[Run], mut right: &[Run]) -> Option<PhysicalPage> {
    while let (Some(&(a, b)), Some(&(c, d))) = (left.first(), right.first()) {
        if b < c {
            left = &left[1..];
        } else if d < a {
            right = &right[1..];
        } else {
            return Some(a.max(c));
        }
    }
    None
}

fn unregister(active: &mut Vec<Runs>, token: &Runs) {
    let index = active
        .iter()
        .position(|runs| Arc::ptr_eq(runs, token))
        .expect("live guard");
    active.swap_remove(index);
}

#[must_use = "hold the snapshot guard across the entire batch copy"]
#[derive(Debug)]
pub struct Snapshot(Runs);

/// Register all non-private pages, or return a page claimed by an active or
/// waiting writer without registering any of the batch. Never waits for a writer.
pub fn try_snapshot(pages: impl Iterator<Item = PhysicalPage>) -> Result<Snapshot, PhysicalPage> {
    // The caller's iterator is never run under the registry lock.
    let runs = collect_runs(pages.map(|page| (page, 1)));
    if !runs.is_empty() {
        let mut active = ACTIVE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(page) = active
            .writers
            .iter()
            .find_map(|other| conflict(&runs, other))
        {
            return Err(page);
        }
        active.snapshots.push(runs.clone());
    }
    Ok(Snapshot(runs))
}

impl Drop for Snapshot {
    fn drop(&mut self) {
        if !self.0.is_empty() {
            unregister(
                &mut ACTIVE.lock().unwrap_or_else(|e| e.into_inner()).snapshots,
                &self.0,
            );
            RELEASED.notify_all();
        }
    }
}

#[must_use = "hold the writer guard until all writes to the runs are finished"]
#[derive(Debug)]
pub struct Writer(Runs);

/// Runs are (first backing page, number of pages within the extent). Writers
/// can overlap each other. This may wait for snapshots: do not hold an
/// overlapping snapshot or a lock needed to stage one (including the physical
/// and virtual-page maps). The raw VA fill's remap gate is not taken by staging.
pub fn writing(runs: &[(PhysicalPage, usize)]) -> Writer {
    let runs = collect_runs(runs.iter().copied());
    if !runs.is_empty() {
        let mut active = ACTIVE.lock().unwrap_or_else(|e| e.into_inner());
        // Reject new overlapping snapshots while older readers drain. Otherwise
        // consecutive submissions could keep a physical write waiting forever.
        active.writers.push(runs.clone());
        let _drained = RELEASED
            .wait_while(active, |active| {
                active
                    .snapshots
                    .iter()
                    .any(|other| conflict(&runs, other).is_some())
            })
            .unwrap_or_else(|e| e.into_inner());
    }
    Writer(runs)
}

impl Drop for Writer {
    fn drop(&mut self) {
        if !self.0.is_empty() {
            unregister(
                &mut ACTIVE.lock().unwrap_or_else(|e| e.into_inner()).writers,
                &self.0,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physical::PageNumbers;
    use std::{sync::mpsc, thread, time::Duration};

    #[test]
    fn writers_exclude_only_their_pages_until_last_drop() {
        let base = PageNumbers::default().allocate(16).unwrap();
        let private = PhysicalPage::private(0).unwrap();
        let a = writing(&[
            (base.offset(8), 1),
            (base, 3),
            (base.offset(6), 0),
            (private, 1),
        ]);
        let b = writing(&[(base.offset(2), 2)]);
        let reader = |at| try_snapshot([base.offset(at)].into_iter());
        assert_eq!(reader(2).unwrap_err(), base.offset(2));
        assert_eq!(reader(3).unwrap_err(), base.offset(3));
        assert_eq!(reader(8).unwrap_err(), base.offset(8));
        let snapshot =
            try_snapshot([base.offset(4), private, base.offset(5), base.offset(6)].into_iter())
                .unwrap();
        assert_eq!(&*snapshot.0, &[(base.offset(4), base.offset(6))]);
        drop(snapshot);
        // A rejected batch leaves no registration for its non-conflicting pages.
        assert_eq!(
            try_snapshot([base.offset(5), base].into_iter()).unwrap_err(),
            base
        );
        drop(writing(&[(base.offset(5), 1)]));
        drop(a);
        drop(reader(1).unwrap());
        assert_eq!(reader(2).unwrap_err(), base.offset(2));
        drop(b);
        drop(reader(2).unwrap());
    }

    #[test]
    fn writer_waits_for_all_overlapping_snapshots() {
        // Use identities disjoint from the other process-registry test.
        let base = PageNumbers::default().allocate(40).unwrap().offset(32);
        let a = try_snapshot([base.offset(5), base, base.offset(1)].into_iter()).unwrap();
        let b = try_snapshot([base.offset(1)].into_iter()).unwrap();
        let (start, started) = mpsc::channel();
        let (enter, entered) = mpsc::channel();
        let writer = thread::spawn(move || {
            // A hole between snapshot runs must be writable before either reader drops.
            drop(writing(&[(base.offset(3), 2)]));
            start.send(()).unwrap();
            enter.send(writing(&[(base.offset(1), 2)])).unwrap();
        });
        started.recv_timeout(Duration::from_secs(5)).unwrap();
        drop(a);
        // Wait for registration (not just thread startup), then check the
        // remaining snapshot prevents entry and new overlapping readers fail.
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while try_snapshot([base.offset(2)].into_iter()).is_ok() {
            assert!(
                std::time::Instant::now() < deadline,
                "writer did not claim its runs"
            );
            thread::sleep(Duration::from_millis(1));
        }
        assert!(matches!(entered.try_recv(), Err(mpsc::TryRecvError::Empty)));
        drop(b);
        let active = entered.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(
            try_snapshot([base.offset(2)].into_iter()).unwrap_err(),
            base.offset(2)
        );
        drop(active);
        drop(try_snapshot([base.offset(2)].into_iter()).unwrap());
        writer.join().unwrap();
    }
}
