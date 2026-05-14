use crate::acp::parse_json_payload;
use crate::errors::{LlmError, Result};
use serde_json::{json, Value};
use std::time::Duration;
use tokio::time::timeout;

const DEFAULT_TIMEOUT_MS: u64 = 120_000;
const DEFAULT_MODEL: &str = "claude-3-5-haiku";

pub struct OpenAiCompatClient {
    base_url: String,
    model: String,
    api_key: Option<String>,
    http_client: reqwest::Client,
}

impl OpenAiCompatClient {
    pub fn new(base_url: String, model: Option<String>, api_key: Option<String>) -> Self {
        Self {
            base_url,
            model: model.unwrap_or_else(|| DEFAULT_MODEL.to_string()),
            api_key: api_key.or_else(|| std::env::var("OPENAI_API_KEY").ok()),
            http_client: reqwest::Client::new(),
        }
    }

    pub fn from_env() -> Result<Self> {
        let base_url = std::env::var("RS_LEARN_LLM_ENDPOINT")
            .or_else(|_| std::env::var("OPENAI_BASE_URL"))
            .unwrap_or_else(|_| "http://127.0.0.1:4800".to_string());
        let model = std::env::var("OPENAI_MODEL").ok();
        let api_key = std::env::var("OPENAI_API_KEY").ok();
        Ok(Self::new(base_url, model, api_key))
    }

    pub async fn generate(&self, system: &str, user: &str, timeout_ms: u64) -> Result<Value> {
        let _gate = crate::llm_gate::acquire().await
            .map_err(|e| LlmError::Process(format!("llm semaphore closed: {e}")))?;

        let prompt = if system.is_empty() {
            format!("{}\n\nRespond with ONLY a JSON object. No preamble. No code fence.", user)
        } else {
            format!("{}\n\n---\n\n{}\n\nRespond with ONLY a JSON object. No preamble. No code fence.", system, user)
        };

        let tmo = if timeout_ms == 0 { DEFAULT_TIMEOUT_MS } else { timeout_ms };

        let messages = vec![
            json!({"role": "user", "content": prompt}),
        ];

        let mut req = self.http_client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(&json!({
                "model": self.model,
                "messages": messages,
                "temperature": 1.0,
            }));

        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let response = timeout(
            Duration::from_millis(tmo),
            req.send(),
        )
        .await
        .map_err(|_| LlmError::Timeout(format!("OpenAI-compatible request timed out after {}ms", tmo)))?
        .map_err(|e| LlmError::Process(format!("OpenAI-compatible request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_else(|_| "unknown".to_string());
            return Err(LlmError::Process(format!(
                "OpenAI-compatible server returned {}: {}",
                status, body.chars().take(200).collect::<String>()
            )));
        }

        let resp_json: Value = response.json().await
            .map_err(|e| LlmError::Process(format!("Failed to parse OpenAI-compatible response: {}", e)))?;

        let content = resp_json
            .get("choices")
            .and_then(|choices| choices.get(0))
            .and_then(|first| first.get("message"))
            .and_then(|msg| msg.get("content"))
            .and_then(|c| c.as_str())
            .ok_or_else(|| LlmError::Validation(format!("Invalid OpenAI-compatible response format: {}", resp_json)))?;

        parse_json_payload(content).ok_or_else(|| {
            LlmError::Validation(format!("OpenAI-compatible response content not JSON (len={})", content.len()))
        })
    }

    pub async fn is_reachable(&self) -> bool {
        let health_url = format!("{}/health", self.base_url);
        let timeout_duration = Duration::from_secs(2);

        timeout(timeout_duration, self.http_client.get(&health_url).send())
            .await
            .ok()
            .and_then(|r| r.ok())
            .is_some()
    }
}
