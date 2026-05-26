pub mod time;
pub mod types;
pub mod temporal_core;
pub mod attention;

#[cfg(target_arch = "wasm32")]
pub mod host_kv_backend;

#[cfg(target_arch = "wasm32")]
pub mod attention_persist;

pub use temporal_core::{TemporalGraph, KvBackend, InvalidationOutcome};
pub use types::{EdgeRow, NodeRow, EpisodeRow};
pub use attention::{Attention, Subgraph, SubgraphNode, SubgraphEdge, Context, RELATION_VOCAB};
