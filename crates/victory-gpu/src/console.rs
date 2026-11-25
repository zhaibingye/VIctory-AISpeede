//! The Series S guest hardware profile. Nominal RAM includes the guest OS;
//! title APIs expose the smaller retail title budget, not all 10 GiB.

pub const NAME: &str = "Xbox Series S";
pub const DEVICE_TYPE: u32 = 0x60;
pub const COMPUTE_UNITS: u32 = 20;
pub const UNIFIED_MEMORY: usize = 10 << 30;
pub const TITLE_MEMORY: usize = 7_776 << 20;
pub const MAX_HOST_CACHE: usize = 5 << 30;

#[test]
fn title_budget_excludes_system_and_host_cache() {
    assert!(TITLE_MEMORY < UNIFIED_MEMORY);
    assert_eq!(UNIFIED_MEMORY + MAX_HOST_CACHE, 15 << 30);
}
