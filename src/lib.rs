//! tensorpeek — inspect safetensors, GGUF, npy and npz headers as JSON.
//!
//! The crate is a set of pure, individually testable modules glued together
//! by a thin CLI. Every parser reads only the header region of a file (plus,
//! for `.npz`, the ZIP central directory), so multi-gigabyte checkpoints are
//! inspected in milliseconds. There are no runtime dependencies: JSON
//! parsing/serialization, ZIP reading and even DEFLATE decompression for
//! compressed `.npz` members are implemented here on top of `std` alone.

pub mod builder;
pub mod cli;
pub mod gguf;
pub mod inflate;
pub mod json;
pub mod npy;
pub mod npz;
pub mod render;
pub mod report;
pub mod safetensors;
pub mod sniff;

/// The crate version, single-sourced from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
