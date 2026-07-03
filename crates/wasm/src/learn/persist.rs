#![cfg(target_arch = "wasm32")]

use crate::errors::Result;
use crate::learn::deep_core::DeepCore;
use crate::learn::instant_core::InstantCore;
use crate::wasm_host;
use serde::{Deserialize, Serialize};

const NS_ADAPTER: &str = "rs-learn/learn/adapter";
const NS_FISHER: &str = "rs-learn/learn/fisher";
const NS_SNAPSHOT: &str = "rs-learn/learn/snapshot";

#[derive(Serialize, Deserialize)]
struct AdapterBlob {
    adapter_a: Vec<f32>,
    adapter_b: Vec<f32>,
    targets: Vec<String>,
    lr: f32,
    feedback_count: u64,
    resets: u64,
    #[serde(default)]
    version: u64,
}

pub fn save_adapter(core: &InstantCore, key: &str) -> Result<()> {
    let blob = AdapterBlob {
        adapter_a: core.adapter_a.clone(),
        adapter_b: core.adapter_b.clone(),
        targets: core.targets.clone(),
        lr: core.lr,
        feedback_count: core.feedback_count,
        resets: core.resets_performed,
        version: core.persist_version,
    };
    let json = serde_json::to_vec(&blob)?;
    wasm_host::kv_put(NS_ADAPTER, key, &json)
}

pub fn load_adapter(key: &str) -> Result<Option<InstantCore>> {
    let bytes = wasm_host::kv_get(NS_ADAPTER, key)?;
    if bytes.is_empty() { return Ok(None); }
    let blob: AdapterBlob = serde_json::from_slice(&bytes)?;
    let mut core = InstantCore::new(blob.targets);
    if blob.adapter_a.len() == core.adapter_a.len() { core.adapter_a = blob.adapter_a; }
    if blob.adapter_b.len() == core.adapter_b.len() { core.adapter_b = blob.adapter_b; }
    core.lr = blob.lr;
    core.feedback_count = blob.feedback_count;
    core.resets_performed = blob.resets;
    core.persist_version = blob.version;
    Ok(Some(core))
}

pub fn peek_adapter_version(key: &str) -> Result<Option<u64>> {
    let bytes = wasm_host::kv_get(NS_ADAPTER, key)?;
    if bytes.is_empty() { return Ok(None); }
    let blob: AdapterBlob = serde_json::from_slice(&bytes)?;
    Ok(Some(blob.version))
}

pub fn save_fisher(deep: &DeepCore, param_id: &str) -> Result<()> {
    if let Some(f) = deep.fisher.get(param_id) {
        let json = serde_json::to_vec(f)?;
        wasm_host::kv_put(NS_FISHER, param_id, &json)?;
    }
    if let Some(s) = deep.params_snapshot.get(param_id) {
        let json = serde_json::to_vec(s)?;
        wasm_host::kv_put(NS_SNAPSHOT, param_id, &json)?;
    }
    Ok(())
}

pub fn load_fisher_into(deep: &mut DeepCore, param_id: &str) -> Result<bool> {
    let fbytes = wasm_host::kv_get(NS_FISHER, param_id)?;
    let sbytes = wasm_host::kv_get(NS_SNAPSHOT, param_id)?;
    if fbytes.is_empty() || sbytes.is_empty() { return Ok(false); }
    let fisher: Vec<f32> = serde_json::from_slice(&fbytes)?;
    let snapshot: Vec<f32> = serde_json::from_slice(&sbytes)?;
    if fisher.len() != snapshot.len() { return Ok(false); }
    deep.fisher.insert(param_id.to_string(), fisher);
    deep.params_snapshot.insert(param_id.to_string(), snapshot);
    Ok(true)
}
