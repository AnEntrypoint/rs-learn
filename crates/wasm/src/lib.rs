pub mod errors;
pub mod embeddings;

#[cfg(target_arch = "wasm32")]
pub mod wasm_host;

#[cfg(target_arch = "wasm32")]
pub mod wasm_learn;

#[cfg(target_arch = "wasm32")]
pub mod pipeline;

pub use errors::{LlmError, Result};
pub use embeddings::EMBED_DIM;

#[cfg(target_arch = "wasm32")]
pub use wasm_learn::{Learn, RecallHit};
