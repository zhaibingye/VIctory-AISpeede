use std::sync::Arc;

use victory_memory::{
    PAGE_BYTES,
    budget::{BudgetProfile, Headroom, HostPressure, NoSpace, Pool, RAM_LIMIT},
    physical::{Aliases, BackingPins, PageNumbers, HostBudget},
};

#[test]
fn lease_charges_a_section_once_until_the_last_view_releases_it() {
    let ram = HostBudget::default();
    let section = Arc::new(ram.reserve_ram(RAM_LIMIT - PAGE_BYTES).unwrap());
    let mapped_view = section.clone();
    let worker_view = section.clone();
    let other = ram.reserve_ram(PAGE_BYTES).unwrap();
    assert_eq!(ram.used(Pool::Ram), RAM_LIMIT);
    let error = ram.reserve_ram(PAGE_BYTES).err().unwrap();
    assert_eq!(error.pool, Pool::Ram);
    assert_eq!(error.shortfall, PAGE_BYTES);

    drop(section);
    drop(mapped_view);
    assert_eq!(ram.used(Pool::Ram), RAM_LIMIT);
    drop(worker_view);
    assert_eq!(ram.used(Pool::Ram), PAGE_BYTES);
    drop(other);
    assert_eq!(ram.used(Pool::Ram), 0);
}

#[test]
fn distinct_sections_never_alias_even_after_a_virtual_address_is_reused() {
    let mut ids = PageNumbers::default();
    let a = ids.allocate(2).unwrap().offset(1);
    let b = ids.allocate(1).unwrap();
    assert_ne!(a, b);
    assert_eq!(a.number() % 16, 0);
    assert_eq!(b.number(), a.number() + 16);
    let mut aliases = Aliases::default();
    aliases.mapped(a, 0x10000);
    aliases.mapped(a, 0x20000);
    aliases.unmapped(a, 0x10000);
    aliases.mapped(b, 0x10000);
    assert_eq!(aliases.addresses(a), [0x20000]);
    assert_eq!(aliases.addresses(b), [0x10000]);
    aliases.unmapped(a, 0x20000);
    assert!(aliases.addresses(a).is_empty());
}

#[test]
fn allocating_past_the_packed_abi_boundary_does_not_consume_ids() {
    let mut ids = PageNumbers::default();
    let first = ids.allocate(1).unwrap().number();
    let remaining = (1 << 21) - (first >> 4) - 1;
    assert!(ids.allocate(0).is_none());
    assert!(ids.allocate(remaining + 1).is_none());
    assert!(ids.allocate(usize::MAX).is_none());
    let last = ids.allocate(remaining).unwrap().offset(remaining - 1);
    assert_eq!(last.number() >> 4, (1 << 21) - 1);
    assert!(ids.allocate(1).is_none());
}

#[test]
fn submission_pins_charge_each_allocation_once_even_after_the_views_go_away() {
    use victory_memory::physical::BackingPins;

    let ram = HostBudget::default();
    let mut ids = PageNumbers::default();
    let a = ids.allocate(2).unwrap();
    let b = ids.allocate(1).unwrap();
    let backing_a = Arc::new(ram.reserve_ram(2 * PAGE_BYTES).unwrap());
    let backing_b = Arc::new(ram.reserve_ram(PAGE_BYTES).unwrap());
    let mut pins = BackingPins::default();
    pins.keep(a, &backing_a);
    pins.keep(a, &backing_a);
    pins.keep(b, &backing_b);
    assert_eq!(Arc::strong_count(&backing_a), 2);
    assert_eq!(Arc::strong_count(&backing_b), 2);

    drop(backing_a);
    drop(backing_b);
    assert_eq!(ram.used(Pool::Ram), 3 * PAGE_BYTES);
    drop(pins);
    assert_eq!(ram.used(Pool::Ram), 0);
}

#[test]
fn graphics_ram_and_guest_backing_compete_for_one_pool() {
    let budget = HostBudget::default();
    let guest = budget.reserve_ram(RAM_LIMIT - PAGE_BYTES).unwrap();
    let gpu = budget.clone().reserve_ram(PAGE_BYTES).unwrap();
    assert_eq!(budget.used(Pool::Ram), RAM_LIMIT);
    assert!(budget.reserve_ram(1).is_err());
    drop(gpu);
    assert_eq!(budget.used(Pool::Ram), RAM_LIMIT - PAGE_BYTES);
    drop(guest);
    assert_eq!(budget.used(Pool::Ram), 0);
}

#[test]
fn host_pressure_reduces_admission_without_releasing_in_flight_backing() {
    const GIB: u64 = 1 << 30;
    let budget = HostBudget::default();
    let reserve = Headroom { physical_ram: 2 * GIB, commit: GIB / 2, local: GIB / 4 };
    let initial = HostPressure {
        physical_ram: 16 * GIB,
        available_ram: 13 * GIB,
        available_commit: 16 * GIB,
        managed_ram: 0,
        local_capacity: 6 * GIB,
        local_budget: 6 * GIB,
        local_observed_use: 0,
        managed_local: 0,
    };
    let limits = budget.refresh_pressure(reserve, BudgetProfile::Auto, || Some(initial)).unwrap();
    assert_eq!(limits.ram.hard, 11 * GIB);
    assert_eq!(limits.local.hard, 6 * GIB - GIB / 4);
    let ram = Arc::new(budget.reserve_ram(5 * GIB).unwrap());
    let local = Arc::new(budget.reserve_local(4 * GIB, 0, 6 * GIB, reserve.local).unwrap());
    // 来宾视图释放后，在途命令仍持有两份后备。
    let mut ids = PageNumbers::default();
    let mut in_flight = BackingPins::default();
    in_flight.keep(ids.allocate(1).unwrap(), &ram);
    in_flight.keep(ids.allocate(1).unwrap(), &local);
    drop((ram, local));

    let pressured = HostPressure {
        available_ram: 2 * GIB,
        available_commit: GIB / 4,
        local_budget: 3 * GIB + 3 * GIB / 4,
        local_observed_use: 4 * GIB + GIB / 2,
        ..initial
    };
    let tight = budget.refresh_pressure(reserve, BudgetProfile::Auto, || Some(pressured)).unwrap();
    assert_eq!(tight.ram.hard, 5 * GIB);
    assert_eq!(tight.local.hard, 3 * GIB);
    assert!(budget.refresh_pressure(reserve, BudgetProfile::Auto, || None).is_none());
    assert_eq!(budget.pool_state(Pool::Ram), (5 * GIB, tight.ram));
    assert_eq!(budget.pool_state(Pool::Local), (4 * GIB, tight.local));
    assert_eq!(budget.reserve_ram(PAGE_BYTES).err(), Some(NoSpace {
        pool: Pool::Ram, shortfall: PAGE_BYTES,
    }));
    let next_local = budget.reserve_local(
        PAGE_BYTES, pressured.local_observed_use, pressured.local_budget, reserve.local,
    );
    assert_eq!(next_local.err(), Some(NoSpace { pool: Pool::Local, shortfall: PAGE_BYTES }));

    // 只有对应提交完成、最后一份租约释放后才回收；回暖后重新开放准入。
    drop(in_flight);
    assert_eq!(budget.used(Pool::Ram), 0);
    assert_eq!(budget.used(Pool::Local), 0);
    let recovered = HostPressure { available_ram: 7 * GIB, ..initial };
    let limits = budget.refresh_pressure(reserve, BudgetProfile::Auto, || Some(recovered)).unwrap();
    assert_eq!(limits.ram.hard, 5 * GIB);
    assert!(budget.reserve_ram(5 * GIB).is_ok());
    assert!(budget.reserve_local(4 * GIB, 0, 6 * GIB, reserve.local).is_ok());
}

#[test]
fn local_allocations_obey_the_smaller_dxgi_cap_without_double_counting() {
    const GIB: u64 = 1 << 30;
    let budget = HostBudget::default();
    let a = budget.reserve_local(2 * GIB, 0, 6 * GIB, GIB / 2).unwrap();
    let b = budget.reserve_local(GIB, 3 * GIB, 6 * GIB, GIB / 2).unwrap();
    // DXGI already includes the 3 GiB of managed allocations. Only the
    // extra 1 GiB and 0.5 GiB reserve reduce the 5 GiB cap.
    assert!(budget.reserve_local(GIB, 4 * GIB, 6 * GIB, GIB / 2).is_err());
    assert!(budget.reserve_local(1, 0, 2 * GIB, GIB / 2).is_err());
    assert_eq!(budget.used(Pool::Local), 3 * GIB); // Existing users keep their leases.
    drop(a);
    let c = budget.reserve_local(GIB / 2, GIB, 2 * GIB, GIB / 2).unwrap();
    assert_eq!(budget.used(Pool::Local), 3 * GIB / 2);
    drop((b, c));
    assert_eq!(budget.used(Pool::Local), 0);
    assert_eq!(budget.used(Pool::Ram), 0);
}

#[test]
fn writeback_snapshot_keeps_original_identity_across_partial_overlaps() {
    use victory_memory::physical::BackingPins;
    const PAGE: usize = PAGE_BYTES as usize;
    let mut numbers = PageNumbers::default();
    let old = numbers.allocate(3).unwrap();
    let later = numbers.allocate(4).unwrap();
    let mut pins = BackingPins::default();
    pins.remember(PAGE, 3 * PAGE, Some(old));
    pins.remember(3 * PAGE, 4 * PAGE, None);
    pins.remember(0, 4 * PAGE, Some(later));

    assert_eq!(pins.expected(0), Some(later));
    assert_eq!(pins.expected(PAGE + 1), Some(old));
    assert_eq!(pins.expected(2 * PAGE + PAGE / 2), Some(old.offset(1)));
    assert_eq!(pins.expected(3 * PAGE), None);
    assert_eq!(pins.expected(4 * PAGE), None);
    assert_eq!(pins.run_at(PAGE + 15, 2 * PAGE), (2 * PAGE, Some(old)));
    assert_eq!(pins.run_at(2 * PAGE + 7, 4 * PAGE), (3 * PAGE, Some(old.offset(1))));
    assert_eq!(pins.run_at(3 * PAGE, 5 * PAGE), (4 * PAGE, None));
    assert_eq!(pins.run_at(4 * PAGE, 5 * PAGE), (5 * PAGE, None));
}
