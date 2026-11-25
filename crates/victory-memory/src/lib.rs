//! Portable policy for the replacement Series S memory engine.
#![cfg_attr(not(windows), forbid(unsafe_code))]

pub mod budget;
pub mod coherence;
pub mod engine;
pub mod gpu_table;
pub mod guest;
pub mod physical;
pub mod residency;
pub mod virtual_pages;

pub const PAGE_BYTES: u64 = 64 << 10;

#[cfg(windows)]
pub mod d3d12;
