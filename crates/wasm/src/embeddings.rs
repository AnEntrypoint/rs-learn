use crate::errors::{LlmError, Result};

pub const EMBED_DIM: usize = 384;

pub struct Embedder;

impl Embedder {
    pub fn new() -> Self {
        Self
    }

    pub fn embed(&self, _text: &str) -> Result<Vec<f32>> {
        Err(LlmError::Process(
            "rs-learn::embeddings::Embedder is retired. Embeddings now run pure-wasm inside rs-plugkit (crate::embed::embed_text via baked Nomic Q4_K_M GGUF). Call the orchestrator's memorize/recall verbs instead of invoking Embedder directly.".into()
        ))
    }
}

impl Default for Embedder {
    fn default() -> Self {
        Self::new()
    }
}

pub fn stats_snapshot() -> (u64, u64, u64) {
    (0, 0, 0)
}
