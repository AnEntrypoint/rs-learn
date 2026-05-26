#![cfg(target_arch = "wasm32")]

use crate::errors::Result;
use crate::graph::attention::Attention;
use crate::wasm_host;
use serde::{Deserialize, Serialize};

const NS: &str = "rs-learn/graph/attention";

#[derive(Serialize, Deserialize)]
struct AttentionBlob {
    dim: usize,
    heads: usize,
    head_dim: usize,
    seed: u32,
    we: Vec<f32>,
}

pub fn save_attention(a: &Attention, seed: u32, key: &str) -> Result<()> {
    let blob = AttentionBlob {
        dim: a.dim,
        heads: a.heads,
        head_dim: a.head_dim,
        seed,
        we: a.we.clone(),
    };
    let json = serde_json::to_vec(&blob)?;
    wasm_host::kv_put(NS, key, &json)
}

pub fn load_attention(key: &str) -> Result<Option<(Attention, u32)>> {
    let bytes = wasm_host::kv_get(NS, key)?;
    if bytes.is_empty() { return Ok(None); }
    let blob: AttentionBlob = serde_json::from_slice(&bytes)?;
    let mut a = Attention::new(blob.dim, blob.heads, blob.head_dim, blob.seed);
    if blob.we.len() == a.we.len() { a.we = blob.we; }
    Ok(Some((a, blob.seed)))
}
