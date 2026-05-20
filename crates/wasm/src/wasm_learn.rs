#![cfg(target_arch = "wasm32")]

use crate::errors::Result;
use crate::wasm_host;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallHit {
    pub id: String,
    pub text: String,
    pub score: f32,
    #[serde(default)]
    pub namespace: Option<String>,
}

pub struct Learn;

impl Learn {
    pub fn new() -> Self {
        Self
    }

    pub fn recall(&self, query: &str, limit: usize) -> Result<Vec<RecallHit>> {
        let bytes = wasm_host::vec_search(query, limit as u32)?;
        if bytes.is_empty() {
            return Ok(Vec::new());
        }
        let parsed: Vec<RecallHit> = serde_json::from_slice(&bytes)?;
        Ok(parsed)
    }

    pub fn memorize(&self, text: &str, namespace: &str) -> Result<()> {
        let now = wasm_host::now_ms();
        let key = format!("{}-{}", now, blake_short(text));
        wasm_host::kv_put(namespace, &key, text.as_bytes())?;
        wasm_host::kv_put("pending_index", &key, namespace.as_bytes())?;
        wasm_host::log(&format!("memorize ns={} key={}", namespace, key));
        Ok(())
    }
}

impl Default for Learn {
    fn default() -> Self {
        Self::new()
    }
}

fn blake_short(text: &str) -> String {
    let mut h: u64 = 1469598103934665603;
    for b in text.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    format!("{:016x}", h)
}
