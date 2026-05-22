pub mod instant_core;

#[cfg(target_arch = "wasm32")]
pub mod instant;

pub use instant_core::{InstantCore, FeedbackPayload, EwcState, RANK, LR0, DECAY, MAX_ADAPTER_NORM};
