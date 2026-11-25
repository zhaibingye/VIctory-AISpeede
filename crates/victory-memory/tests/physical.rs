use std::sync::Arc;

use victory_memory::{
    PAGE_BYTES,
    budget::{Pool, RAM_LIMIT},
    physical::{Aliases, PageNumbers, RamPool},
};

#[test]
fn lease_charges_a_section_once_until_the_last_view_releases_it() {
    let ram = RamPool::default();
    let section = Arc::new(ram.reserve(RAM_LIMIT - PAGE_BYTES).unwrap());
    let mapped_view = section.clone();
    let worker_view = section.clone();
    let other = ram.reserve(PAGE_BYTES).unwrap();
    assert_eq!(ram.used(), RAM_LIMIT);
    let error = ram.reserve(PAGE_BYTES).err().unwrap();
    assert_eq!(error.pool, Pool::Ram);
    assert_eq!(error.shortfall, PAGE_BYTES);

    drop(section);
    drop(mapped_view);
    assert_eq!(ram.used(), RAM_LIMIT);
    drop(worker_view);
    assert_eq!(ram.used(), PAGE_BYTES);
    drop(other);
    assert_eq!(ram.used(), 0);
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

    let ram = RamPool::default();
    let mut ids = PageNumbers::default();
    let a = ids.allocate(2).unwrap();
    let b = ids.allocate(1).unwrap();
    let backing_a = Arc::new(ram.reserve(2 * PAGE_BYTES).unwrap());
    let backing_b = Arc::new(ram.reserve(PAGE_BYTES).unwrap());
    let mut pins = BackingPins::default();
    pins.keep(a, &backing_a);
    pins.keep(a, &backing_a);
    pins.keep(b, &backing_b);
    assert_eq!(Arc::strong_count(&backing_a), 2);
    assert_eq!(Arc::strong_count(&backing_b), 2);

    drop(backing_a);
    drop(backing_b);
    assert_eq!(ram.used(), 3 * PAGE_BYTES);
    drop(pins);
    assert_eq!(ram.used(), 0);
}
