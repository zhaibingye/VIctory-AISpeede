use victory_memory::{
    PAGE_BYTES, budget::{Budget, LOCAL_LIMIT, Pool, RAM_LIMIT},
    coherence::{Access, Fence, Invalid, Need, Page}, guest::Guest, residency::Residency,
};

#[test]
fn aliases_share_identity_and_a_reused_address_never_reuses_the_id() {
    let mut guest = Guest::default();
    let id = guest.commit_page(PAGE_BYTES).unwrap();
    guest.alias_page(9 * PAGE_BYTES, id).unwrap();
    assert_eq!(guest.resolve(9 * PAGE_BYTES + 7), Some((id, 7)));
    assert_eq!(guest.unmap(PAGE_BYTES).unwrap(), None);
    assert_eq!(guest.unmap(9 * PAGE_BYTES).unwrap(), Some(id));
    assert_ne!(guest.commit_page(PAGE_BYTES).unwrap(), id);
    assert!(guest.alias_page(2 * PAGE_BYTES, id).is_err());
}

#[test]
fn closed_loop_has_no_old_cpu_write_or_duplicate_upload() {
    let mut p = Page::default();
    let f = Fence;
    let (mut ram, mut gpu) = (23, 0);
    p.cpu_written(Access::FullOverwrite, f(0)).unwrap();
    assert_eq!(p.gpu_need(Access::Read, f(0)), Need::Upload);
    p.upload_submitted(f(0), f(1)).unwrap();
    assert_eq!(p.gpu_need(Access::Read, f(0)), Need::Wait(f(1)));
    gpu += ram; // completion of the only upload
    p.gpu_submitted(Access::Read, f(1), f(2)).unwrap();
    assert_eq!(gpu, 23);
    assert_eq!(p.gpu_need(Access::Modify, f(2)), Need::Ready);
    p.gpu_submitted(Access::Modify, f(2), f(3)).unwrap();
    gpu += 7;
    assert_eq!(p.cpu_need(Access::Read, f(2)), Need::Wait(f(3)));
    assert_eq!(p.cpu_written(Access::Modify, f(3)), Err(Invalid::NotReady(Need::Download)));
    assert!(p.discard_gpu(f(3)).is_err());
    p.download_submitted(f(3), f(4)).unwrap();
    assert_eq!(p.cpu_need(Access::Read, f(3)), Need::Wait(f(4)));
    ram = gpu; // completion of the download
    assert_eq!(p.cpu_need(Access::Read, f(4)), Need::Ready);
    assert_eq!(ram, 30);
    assert_eq!(p.version(), 2);
    p.discard_gpu(f(4)).unwrap();
    assert_eq!(p.gpu_need(Access::Read, f(4)), Need::Upload);
}

#[test]
fn full_overwrite_skips_a_transfer_but_not_an_active_access() {
    let mut p = Page::default();
    p.gpu_submitted(Access::FullOverwrite, Fence(0), Fence(1)).unwrap();
    assert_eq!(p.cpu_need(Access::FullOverwrite, Fence(0)), Need::Wait(Fence(1)));
    p.cpu_written(Access::FullOverwrite, Fence(1)).unwrap();
    assert_eq!(p.cpu_need(Access::Read, Fence(1)), Need::Ready);
    assert_eq!(p.gpu_need(Access::Read, Fence(1)), Need::Upload);
}

#[test]
fn budgeting_charges_actual_capacity_and_freezes_on_a_shrink() {
    let mut b = Budget::default();
    assert!(b.charge(Pool::Local, 1).is_err());
    assert_eq!(b.observe_local(7 << 30, 0), 0);
    let vram = b.charge(Pool::Local, LOCAL_LIMIT).unwrap();
    assert!(b.charge(Pool::Local, PAGE_BYTES).is_err());
    assert_eq!(b.observe_local(4 << 30, 128 << 20), (1 << 30) + (128 << 20));
    assert!(b.charge(Pool::Local, 1).is_err());
    b.release(vram);
    let ram = b.charge(Pool::Ram, RAM_LIMIT).unwrap();
    assert!(b.charge(Pool::Ram, 1).is_err());
    b.release(ram);
    assert_eq!(b.used(Pool::Ram), 0);
}

#[test]
fn two_queue_ignores_in_flight_pages_and_protects_reused_pages() {
    let mut g = Guest::default();
    let a = g.commit_page(0).unwrap();
    let b = g.commit_page(PAGE_BYTES).unwrap();
    let c = g.commit_page(2 * PAGE_BYTES).unwrap();
    let mut r = Residency::default();
    r.access(a, 0);
    r.access(b, 0);
    r.access(c, 0);
    r.access(a, 0);
    assert_eq!(r.candidate(Fence(0)), Some(b));
    assert!(r.pin_recording(b));
    assert!(r.pin_recording(c));
    assert!(r.pin_recording(a));
    assert_eq!(r.candidate(Fence(u64::MAX)), None);
    assert!(r.end_recording(b, Some(Fence(7))));
    assert!(r.end_recording(c, None));
    assert!(!r.remove(b, Fence(6)));
    assert_eq!(r.candidate(Fence(6)), Some(c));
    assert!(r.remove(b, Fence(7)));
}

#[test]
fn six_gib_of_logical_pages_streams_through_five_gib_of_local_policy() {
    use victory_memory::engine::{Engine, Error};
    let mut engine = Engine::default();
    engine.set_local_budget(6 << 30, 0);
    let pages = (6 << 30) / PAGE_BYTES;
    let mut evicted = 0;
    for i in 0..pages {
        let id = engine.commit(i * PAGE_BYTES).unwrap();
        match engine.admit(id, 0) {
            Ok(()) => {}
            Err(Error::Space(_)) => {
                let victim = engine.candidate(Fence(0)).unwrap();
                engine.complete_eviction(victim, Fence(0)).unwrap();
                engine.admit(id, 0).unwrap();
                evicted += 1;
            }
            Err(e) => panic!("{e:?}"),
        }
        assert!(engine.used(Pool::Local) <= LOCAL_LIMIT);
    }
    assert_eq!(engine.used(Pool::Ram), 6 << 30);
    assert_eq!(evicted * PAGE_BYTES, 1 << 30);
    let first = engine.resolve(0).unwrap().0;
    engine.alias(9 << 30, first).unwrap();
    assert_eq!(engine.used(Pool::Ram), 6 << 30, "alias is not another physical page");
    assert_eq!(engine.unmap(0).unwrap(), None);
    assert_eq!(engine.unmap(9 << 30).unwrap(), Some(first));
    engine.complete_retirement(first, Fence(0)).unwrap();
    assert_eq!(engine.used(Pool::Ram), (6 << 30) - PAGE_BYTES);
}

#[test]
fn dirty_eviction_holds_backing_and_budget_until_copy_and_unmap() {
    use victory_memory::engine::{Engine, Error};
    let mut engine = Engine::default();
    engine.set_local_budget(2 * PAGE_BYTES, 0);
    let id = engine.commit(PAGE_BYTES).unwrap();
    engine.admit(id, 0).unwrap();
    engine.pin(id);
    engine.page(id).unwrap().gpu_submitted(Access::FullOverwrite, Fence(0), Fence(1)).unwrap();
    engine.end_pin(id, Some(Fence(1)));
    assert_eq!(engine.eviction_need(id, Fence(0)), Err(Error::NotEvictable));
    assert_eq!(engine.eviction_need(id, Fence(1)), Ok(Need::Download));
    engine.page(id).unwrap().download_submitted(Fence(1), Fence(2)).unwrap();
    assert_eq!(engine.complete_eviction(id, Fence(1)), Err(Error::TransferRequired(Need::Wait(Fence(2)))));
    assert_eq!(engine.used(Pool::Local), PAGE_BYTES);
    engine.complete_eviction(id, Fence(2)).unwrap();
    assert_eq!(engine.used(Pool::Local), 0);
}
