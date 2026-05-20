#![cfg(target_arch = "wasm32")]

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::wasm_host;

const TTL_MS: u64 = 120_000;
const HMAC_KEY_DEFAULT: &str = "dev-only-not-secret-rotate-in-prod";
const SUMMARIZE_THRESHOLD: usize = 2048;
const KV_NS: &str = "rs-learn/pipeline";

extern "C" {
    fn host_env_get(key_ptr: *const u8, key_len: u32) -> u64;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingStep {
    pub kind: String,
    pub id: String,
    pub payload: Value,
    pub prompt_template: String,
    pub max_result_bytes: usize,
    pub result_schema: Value,
}

fn hmac_key() -> String {
    let k = "RS_LEARN_PIPELINE_HMAC_KEY";
    let packed = unsafe { host_env_get(k.as_ptr(), k.len() as u32) };
    let ptr = (packed & 0xFFFF_FFFF) as u32;
    let len = (packed >> 32) as u32;
    if ptr != 0 && len != 0 {
        let s = unsafe { core::slice::from_raw_parts(ptr as *const u8, len as usize) };
        if !s.is_empty() {
            return String::from_utf8_lossy(s).into_owned();
        }
    }
    HMAC_KEY_DEFAULT.to_string()
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 1469598103934665603;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

fn keyed_hash(key: &str, data: &str) -> String {
    let inner = fnv1a64(format!("{}|{}", key, data).as_bytes());
    let outer = fnv1a64(format!("{}|{:016x}", key, inner).as_bytes());
    format!("{:016x}{:016x}", outer, inner)
}

fn mint_step_id() -> String {
    let now = wasm_host::now_ms() as u64;
    let r = fnv1a64(format!("{}|{}", now, now.wrapping_mul(2654435761)).as_bytes());
    format!("stp_{:016x}", now ^ r)
}

pub fn mint_token(step_id: &str, kv_key: &str, deadline_ms: u64) -> String {
    let payload = format!("{}|{}|{}", step_id, kv_key, deadline_ms);
    format!("tkn_{}.{}", step_id, keyed_hash(&hmac_key(), &payload))
}

pub fn verify_token(token: &str, step_id: &str, kv_key: &str, deadline_ms: u64) -> bool {
    let expected = mint_token(step_id, kv_key, deadline_ms);
    if token.len() != expected.len() { return false; }
    let mut diff: u8 = 0;
    for (a, b) in token.bytes().zip(expected.bytes()) { diff |= a ^ b; }
    diff == 0
}

pub fn needs_summarize(text: &str) -> bool {
    text.len() > SUMMARIZE_THRESHOLD
}

pub fn evict_expired() {
    let now = wasm_host::now_ms() as u64;
    let raw = wasm_host::kv_query(KV_NS, "").unwrap_or_default();
    let entries: Vec<Value> = serde_json::from_slice(&raw).unwrap_or_default();
    for e in entries {
        let key = match e.get("key").and_then(|v| v.as_str()) { Some(k) => k.to_string(), None => continue };
        let val_str = e.get("value").and_then(|v| v.as_str()).unwrap_or("");
        let parsed: Value = serde_json::from_str(val_str).unwrap_or(Value::Null);
        let deadline = parsed.get("deadline_ms").and_then(|v| v.as_u64()).unwrap_or(0);
        if deadline < now {
            let _ = wasm_host::kv_put(KV_NS, &key, b"");
        }
    }
}

pub fn build_summarize_pending(text: &str, namespace: &str) -> Value {
    let step_id = mint_step_id();
    let now = wasm_host::now_ms() as u64;
    let deadline_ms = now + TTL_MS;
    let kv_key = format!("rs-learn/pipeline/{}", step_id);
    let bounded_input: String = text.chars().take(8192).collect();
    let result_schema = json!({
        "type": "object",
        "required": ["summary"],
        "properties": { "summary": { "type": "string", "maxLength": 800 } }
    });

    let state = json!({
        "flow_id": format!("flw_{:016x}", fnv1a64(format!("flow|{}", now).as_bytes())),
        "verb": "memorize",
        "original_body": { "text": text, "namespace": namespace },
        "pipeline": [
            { "step": "summarize", "status": "pending", "id": step_id },
            { "step": "embed", "status": "queued" },
            { "step": "persist", "status": "queued" }
        ],
        "cursor": 0,
        "results_so_far": {},
        "created_ms": now,
        "deadline_ms": deadline_ms,
        "attempts_used": 0,
        "result_schema": result_schema,
        "kind": "summarize",
        "kv_key": kv_key
    });
    let _ = wasm_host::kv_put(KV_NS, &step_id, state.to_string().as_bytes());

    json!({
        "ok": true,
        "pending_step": {
            "kind": "summarize",
            "id": step_id,
            "payload": {
                "input": bounded_input,
                "target_chars": 400,
                "preserve": ["entities", "numbers", "ids"]
            },
            "prompt_template": "Summarize the following text into <=400 chars, preserving entities and any numeric facts. Return JSON {\"summary\": string}. Input:\n{{input}}",
            "max_result_bytes": 4096,
            "result_schema": result_schema
        },
        "token": mint_token(&step_id, &kv_key, deadline_ms),
        "state_kv_key": kv_key,
        "deadline_ms": deadline_ms,
        "attempts_remaining": 2
    })
}
