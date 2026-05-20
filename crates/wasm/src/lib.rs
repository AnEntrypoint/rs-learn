#![cfg(target_arch = "wasm32")]

pub mod errors;
pub mod wasm_host;
pub mod wasm_learn;

pub use errors::{LlmError, Result};
pub use wasm_learn::{Learn, RecallHit};
