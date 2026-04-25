use crate::backend::AgentBackend;
use crate::errors::{LlmError, Result};
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

pub struct LlmJson {
    backend: Arc<dyn AgentBackend>,
    max_attempts: u32,
    timeout_ms: u64,
    backoff_cap_ms: u64,
}

impl LlmJson {
    pub fn new(backend: Arc<dyn AgentBackend>) -> Self {
        Self {
            backend,
            max_attempts: env_u32("BUNGRAPH_LLM_MAX_ATTEMPTS", 2),
            timeout_ms: env_u64("BUNGRAPH_LLM_TIMEOUT_MS", 60_000),
            backoff_cap_ms: env_u64("BUNGRAPH_LLM_BACKOFF_CAP_MS", 20_000),
        }
    }

    pub fn with_limits(backend: Arc<dyn AgentBackend>, max_attempts: u32, timeout_ms: u64, backoff_cap_ms: u64) -> Self {
        Self { backend, max_attempts, timeout_ms, backoff_cap_ms }
    }

    pub async fn call<F>(&self, system: &str, user: &str, validate: F) -> Result<Value>
    where
        F: Fn(&Value) -> std::result::Result<(), String>,
    {
        let m = super::metrics::metrics();
        super::metrics::incr(&m.llm_calls, 1);
        let started = std::time::Instant::now();
        let mut last_err: Option<LlmError> = None;
        let mut out: Option<Value> = None;
        for attempt in 1..=self.max_attempts {
            match self.backend.generate(system, user, self.timeout_ms).await {
                Ok(v) => {
                    let v = coerce_json(v);
                    match validate(&v) {
                        Ok(()) => { out = Some(v); break; }
                        Err(msg) => {
                            tracing::warn!(attempt, "LLM JSON schema failed: {msg}");
                            last_err = Some(LlmError::Validation(format!("schema: {msg}")));
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(attempt, error=%e, "LLM call failed");
                    last_err = Some(e);
                }
            }
            if attempt < self.max_attempts {
                let backoff = backoff_ms(attempt, self.backoff_cap_ms);
                sleep(Duration::from_millis(backoff)).await;
            }
        }
        super::metrics::incr(&m.llm_total_ms, started.elapsed().as_millis() as u64);
        match out {
            Some(v) => Ok(v),
            None => Err(last_err.unwrap_or_else(|| LlmError::Validation("llm: no attempts".into()))),
        }
    }
}

fn coerce_json(v: Value) -> Value {
    if let Value::String(s) = &v {
        if let Some(parsed) = extract_json_block(s) {
            return parsed;
        }
        if let Ok(p) = serde_json::from_str::<Value>(s) {
            return p;
        }
    }
    v
}

fn extract_json_block(s: &str) -> Option<Value> {
    let trimmed = s.trim();
    if let Some(rest) = trimmed.strip_prefix("```json") {
        let body = rest.trim_start_matches('\n').trim_end_matches("```").trim();
        if let Ok(v) = serde_json::from_str::<Value>(body) { return Some(v); }
    }
    if let Some(rest) = trimmed.strip_prefix("```") {
        let body = rest.trim_start_matches('\n').trim_end_matches("```").trim();
        if let Ok(v) = serde_json::from_str::<Value>(body) { return Some(v); }
    }
    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    if end <= start { return None; }
    serde_json::from_str::<Value>(&trimmed[start..=end]).ok()
}

fn backoff_ms(attempt: u32, cap_ms: u64) -> u64 {
    let base: u64 = 500;
    let exp = base.saturating_mul(1u64 << attempt.min(10));
    let jitter = (rand::random::<u64>() % 200).saturating_add(1);
    exp.min(cap_ms).saturating_add(jitter)
}

pub fn require_object_field(v: &Value, field: &str) -> std::result::Result<(), String> {
    match v.get(field) {
        Some(_) => Ok(()),
        None => Err(format!("missing field '{field}'")),
    }
}

pub fn require_array_field(v: &Value, field: &str) -> std::result::Result<(), String> {
    match v.get(field) {
        Some(Value::Array(_)) => Ok(()),
        Some(other) => Err(format!("field '{field}' must be array, got {}", other_type(other))),
        None => Err(format!("missing field '{field}'")),
    }
}

fn other_type(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_fenced_json() {
        let s = "```json\n{\"x\":1}\n```";
        let v = extract_json_block(s).unwrap();
        assert_eq!(v["x"], 1);
    }

    #[test]
    fn extract_bare_json() {
        let s = "some preamble {\"y\":2} trailing";
        let v = extract_json_block(s).unwrap();
        assert_eq!(v["y"], 2);
    }

    #[test]
    fn validate_array_field() {
        let v = serde_json::json!({"edges":[]});
        assert!(require_array_field(&v, "edges").is_ok());
        assert!(require_array_field(&v, "nodes").is_err());
    }
}
