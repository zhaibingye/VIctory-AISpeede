//! Logical submission order, independent of which queues share a GPU.
//! A blocked lane is never represented as a wait on the physical queue.
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

/// Queue-position identity, never a native or title-visible fence value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ticket { lane: u64, sequence: u64 }

pub enum Item<D, W> {
    Dependency(D),
    Work(W),
}

pub struct Ready<W> {
    pub ticket: Ticket,
    pub work: W,
}

/// Bounded FIFO with a single issued head. The dispatch owner retains the
/// issued item through native completion and any guest byte publication.
pub struct Timeline<D, W> {
    lane: u64,
    limit: usize,
    next: u64,
    pending: VecDeque<(Ticket, Item<D, W>)>,
    issued: Option<Ticket>,
}

impl<D, W> Timeline<D, W> {
    pub fn new(limit: usize) -> Self {
        static NEXT_LANE: AtomicU64 = AtomicU64::new(1);
        let lane = NEXT_LANE.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .expect("logical queue identity space exhausted");
        Self { lane, limit, next: 0, pending: VecDeque::new(), issued: None }
    }

    pub fn is_idle(&self) -> bool {
        self.issued.is_none() && self.pending.is_empty()
    }

    /// Return ownership to the caller on backpressure, without advancing
    /// the sequence or throwing out existing work.
    pub fn enqueue(&mut self, item: Item<D, W>) -> Result<Ticket, Item<D, W>> {
        if self.pending.len() + usize::from(self.issued.is_some()) >= self.limit {
            return Err(item);
        }
        let Some(next) = self.next.checked_add(1) else { return Err(item) };
        self.next = next;
        let ticket = Ticket { lane: self.lane, sequence: next };
        self.pending.push_back((ticket, item));
        Ok(ticket)
    }

    /// Select only a runnable head. The dependency callback must observe
    /// an already published value, not submit or block on another lane.
    pub fn take_ready(&mut self, mut reached: impl FnMut(&D) -> bool) -> Option<Ready<W>> {
        if self.issued.is_some() {
            return None;
        }
        loop {
            let (_, head) = self.pending.front()?;
            if let Item::Dependency(dep) = head
                && !reached(dep)
            {
                return None;
            }
            let (ticket, item) = self.pending.pop_front().expect("timeline head exists");
            if let Item::Work(work) = item {
                self.issued = Some(ticket);
                return Some(Ready { ticket, work });
            }
        }
    }

    /// Fold immediately adjacent work into the issued operation, without
    /// crossing even an already satisfied dependency marker.
    pub fn take_adjacent(&mut self, ticket: Ticket, fold: impl FnOnce(&W) -> bool) -> Option<W> {
        if self.issued != Some(ticket) {
            return None;
        }
        let (_, Item::Work(next)) = self.pending.front()? else {
            return None;
        };
        if !fold(next) {
            return None;
        }
        match self.pending.pop_front() {
            Some((_, Item::Work(next))) => Some(next),
            _ => unreachable!("validated adjacent work"),
        }
    }

    /// Call after the promised completion boundary, not just after enqueue.
    /// A stale or foreign ticket must never unblock the current work.
    pub fn retire(&mut self, ticket: Ticket) -> bool {
        if self.issued != Some(ticket) {
            return false;
        }
        self.issued = None;
        true
    }
}
