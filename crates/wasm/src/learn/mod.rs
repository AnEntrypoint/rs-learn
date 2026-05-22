pub mod instant_core;
pub mod deep_core;

#[cfg(target_arch = "wasm32")]
pub mod instant;

#[cfg(target_arch = "wasm32")]
pub mod persist;

pub use instant_core::{InstantCore, FeedbackPayload, EwcState, RANK, LR0, DECAY, MAX_ADAPTER_NORM};
pub use deep_core::{DeepCore, FISHER_DECAY, DEFAULT_LAMBDA, BOUNDARY_Z};
