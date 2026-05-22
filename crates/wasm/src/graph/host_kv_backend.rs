#![cfg(target_arch = "wasm32")]

use crate::graph::temporal_core::KvBackend;
use crate::wasm_host;
use serde::Deserialize;

#[derive(Deserialize)]
struct KvEntry {
    key: String,
    #[serde(default)]
    value: Option<String>,
}

pub struct HostKv;

impl KvBackend for HostKv {
    fn get(&self, namespace: &str, key: &str) -> Option<Vec<u8>> {
        let bytes = wasm_host::kv_get(namespace, key).ok()?;
        if bytes.is_empty() { None } else { Some(bytes) }
    }

    fn put(&mut self, namespace: &str, key: &str, val: &[u8]) {
        let _ = wasm_host::kv_put(namespace, key, val);
    }

    fn list_prefix(&self, namespace: &str, prefix: &str) -> Vec<String> {
        let raw = match wasm_host::kv_query(namespace, prefix) {
            Ok(b) if !b.is_empty() => b,
            _ => return Vec::new(),
        };
        let parsed: Vec<KvEntry> = match serde_json::from_slice(&raw) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        parsed.into_iter()
            .filter(|e| e.key.starts_with(prefix))
            .map(|e| e.key)
            .collect()
    }
}
