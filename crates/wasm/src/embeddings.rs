use crate::errors::{LlmError, Result};

pub const EMBED_DIM: usize = 768;

pub struct Embedder;

impl Embedder {
    pub fn new() -> Self {
        Self
    }

    #[cfg(target_arch = "wasm32")]
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        extern "C" {
            fn host_vec_embed(text_ptr: *const u8, text_len: u32) -> u64;
        }
        let packed = unsafe { host_vec_embed(text.as_ptr(), text.len() as u32) };
        let ptr = (packed & 0xFFFF_FFFF) as u32;
        let len = (packed >> 32) as u32;
        if ptr == 0 || len == 0 {
            return Err(LlmError::Process("host_vec_embed returned empty".into()));
        }
        let bytes = unsafe { core::slice::from_raw_parts(ptr as *const u8, len as usize) };
        let parsed: Vec<f32> = serde_json::from_slice(bytes)
            .map_err(|e| LlmError::Process(format!("vec_embed json parse: {}", e)))?;
        if parsed.len() != EMBED_DIM {
            return Err(LlmError::Process(format!(
                "embedding length {} != EMBED_DIM {}", parsed.len(), EMBED_DIM
            )));
        }
        Ok(parsed)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn embed(&self, _text: &str) -> Result<Vec<f32>> {
        Err(LlmError::Process("Embedder::embed unavailable outside wasm32 target".into()))
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
