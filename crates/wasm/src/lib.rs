pub mod errors;
pub mod embeddings;
pub mod learn;
pub mod graph;

#[cfg(target_arch = "wasm32")]
pub mod wasm_host;

#[cfg(target_arch = "wasm32")]
pub mod wasm_learn;

#[cfg(target_arch = "wasm32")]
pub mod pipeline;

pub use errors::{LlmError, Result};
pub use embeddings::EMBED_DIM;
pub use learn::{InstantCore, FeedbackPayload, EwcState};
pub use graph::{TemporalGraph, KvBackend, InvalidationOutcome, EdgeRow, NodeRow, EpisodeRow};

#[cfg(target_arch = "wasm32")]
pub use wasm_learn::{Learn, RecallHit};
