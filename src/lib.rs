#[cfg(not(target_arch = "wasm32"))]
pub mod store;
#[cfg(not(target_arch = "wasm32"))]
pub mod embeddings;
#[cfg(not(target_arch = "wasm32"))]
pub mod memory;
#[cfg(not(target_arch = "wasm32"))]
pub mod attention;
#[cfg(not(target_arch = "wasm32"))]
pub mod router;
#[cfg(not(target_arch = "wasm32"))]
pub mod acp;
#[cfg(not(target_arch = "wasm32"))]
pub mod backend;
pub mod errors;
#[cfg(not(target_arch = "wasm32"))]
pub mod learn;
#[cfg(not(target_arch = "wasm32"))]
pub mod orchestrator;
#[cfg(not(target_arch = "wasm32"))]
pub mod observability;
#[cfg(not(target_arch = "wasm32"))]
pub mod rs_search_bridge;
#[cfg(not(target_arch = "wasm32"))]
pub mod graph;
#[cfg(not(target_arch = "wasm32"))]
pub mod simd;
#[cfg(not(target_arch = "wasm32"))]
pub mod cache;
#[cfg(not(target_arch = "wasm32"))]
pub mod spine;
pub mod db_path;
#[cfg(not(target_arch = "wasm32"))]
pub mod llm_gate;
pub use db_path::{resolve_db_path, resolve_db_path_for};

#[cfg(not(target_arch = "wasm32"))]
pub use acp::AcpClient;
#[cfg(not(target_arch = "wasm32"))]
pub use backend::{AgentBackend, ClaudeCliClient};
#[cfg(not(target_arch = "wasm32"))]
pub use attention::Attention;
#[cfg(not(target_arch = "wasm32"))]
pub use embeddings::Embedder;
pub use errors::{LlmError, Result};
#[cfg(not(target_arch = "wasm32"))]
pub use memory::Memory;
#[cfg(not(target_arch = "wasm32"))]
pub use orchestrator::{Orchestrator, QueryResult};
#[cfg(not(target_arch = "wasm32"))]
pub use router::{Route, Router};
#[cfg(not(target_arch = "wasm32"))]
pub use store::Store;
