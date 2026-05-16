use crate::acp::AcpClient;
use crate::backend::openai_compat::OpenAiCompatClient;
use crate::backend::AgentBackend;
use crate::errors::{LlmError, Result};
use async_trait::async_trait;
use serde_json::Value;
use std::path::PathBuf;

/// SWE-bench scores for model ranking. Embedded subset from acptoapi's swe-bench-scores.js.
/// Higher score = better SWE-Bench Verified performance.
const SWE_BENCH_SCORES: &[(&str, f32)] = &[
    ("claude/mythos-preview", 93.9),
    ("claude/sonnet-5", 92.4),
    ("gpt/5.5", 88.7),
    ("claude/opus-4.7", 87.6),
    ("gpt/5.3-codex", 85.0),
    ("claude/opus-4.5", 80.9),
    ("claude/opus-4.6", 80.8),
    ("deepseek/v4-pro-max", 80.6),
    ("kimi/k2.6", 80.2),
    ("gpt/5.2", 80.0),
    ("claude/sonnet-4.6", 79.6),
    ("deepseek/v4-pro-high", 79.4),
    ("deepseek/v4-flash-max", 79.0),
    ("qwen/3.6-plus", 78.8),
    ("deepseek/v4-flash-high", 78.6),
    ("mimo/v2-pro", 78.0),
    ("glm/5", 77.8),
    ("mistral/medium-3.5-128b", 77.6),
    ("muse/spark", 77.4),
    ("qwen/3.6-27b", 77.2),
    ("claude/sonnet-4.5", 77.2),
    ("kimi/k2.5-reasoning", 76.8),
    ("kimi/k2.5", 76.8),
    ("grok/4.20", 76.7),
    ("qwen/3.5-397b", 76.2),
    ("mimo/v2-omni", 74.8),
    ("claude/4.1-opus", 74.5),
    ("hy/3-preview", 74.4),
    ("glm/4.7", 73.8),
    ("deepseek/v4-flash", 73.7),
    ("deepseek/v4-pro", 73.6),
    ("qwen/3.6-35b-a3b", 73.4),
    ("mimo/v2-flash", 73.4),
    ("claude/haiku-4.5", 73.3),
    ("claude/4-sonnet", 72.7),
    ("laguna/m.1", 72.5),
    ("qwen/3.5-27b", 72.4),
    ("qwen/3.5-122b-a10b", 72.0),
    ("grok/code-fast-1", 70.8),
    ("qwen/3.5-35b-a3b", 69.2),
    ("laguna/xs.2", 68.2),
    ("gemini/2.5-pro", 63.8),
    ("gpt/4.1", 54.6),
    ("zaya/1-74b-preview", 53.2),
    ("o3/mini", 49.3),
    ("claude/3.5-sonnet", 49.0),
    ("deepseek/v3", 42.0),
    ("gpt/4.1-mini", 23.6),
];

/// Look up SWE-bench score for a model identifier. Returns None if not found.
pub fn get_swe_bench_score(model_id: &str) -> Option<f32> {
    let normalized = model_id.to_lowercase();
    // Exact match first
    if let Some((_, score)) = SWE_BENCH_SCORES.iter().find(|(k, _)| *k == normalized) {
        return Some(*score);
    }
    // Partial match: check if model_id contains a known key's family name
    for (key, score) in SWE_BENCH_SCORES {
        let family = key.split('/').nth(1).unwrap_or("");
        if !family.is_empty() && normalized.contains(family) {
            return Some(*score);
        }
    }
    None
}

/// Sort a list of model identifiers by SWE-bench score (highest first).
/// Models without scores are placed at the end.
pub fn sort_by_swe_bench(models: &mut [String]) {
    models.sort_by(|a, b| {
        let score_a = get_swe_bench_score(a).unwrap_or(0.0);
        let score_b = get_swe_bench_score(b).unwrap_or(0.0);
        score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Reads AGENTS.md content for fallback memory preservation.
fn read_agents_md(path: Option<&PathBuf>) -> Option<String> {
    if let Some(p) = path {
        return std::fs::read_to_string(p).ok();
    }
    // Default search paths
    for candidate in &["AGENTS.md", ".gm/AGENTS.md", "../AGENTS.md"] {
        if let Ok(content) = std::fs::read_to_string(candidate) {
            return Some(content);
        }
    }
    None
}

/// Fallback backend chain: tries HTTP (acptoapi), then ACP subprocess, then AGENTS.md.
pub struct FallbackBackend {
    http: OpenAiCompatClient,
    acp: Option<AcpClient>,
    agents_md_path: Option<PathBuf>,
}

impl FallbackBackend {
    pub fn new(http: OpenAiCompatClient, acp: Option<AcpClient>, agents_md_path: Option<String>) -> Self {
        Self {
            http,
            acp,
            agents_md_path: agents_md_path.map(PathBuf::from),
        }
    }

    async fn try_http(&self, system: &str, user: &str, timeout_ms: u64) -> Result<Value> {
        self.http.generate(system, user, timeout_ms).await
    }

    async fn try_acp(&self, system: &str, user: &str, timeout_ms: u64) -> Result<Value> {
        if let Some(acp) = &self.acp {
            acp.generate(system, user, timeout_ms).await
        } else {
            Err(LlmError::Process("ACP client not configured".into()))
        }
    }

    fn try_agents_md(&self, system: &str, user: &str) -> Result<Value> {
        let content = read_agents_md(self.agents_md_path.as_ref());
        match content {
            Some(md) => {
                eprintln!(
                    "[rs-learn] LLM backend unavailable — falling back to AGENTS.md context ({} bytes)",
                    md.len()
                );
                Ok(Value::String(format!(
                    "AGENTS.md fallback ({} bytes):\n\n{}",
                    md.len(),
                    md.chars().take(4000).collect::<String>()
                )))
            }
            None => {
                let msg = "LLM backend unavailable and AGENTS.md not found";
                eprintln!("[rs-learn] {}", msg);
                Err(LlmError::Process(msg.into()))
            }
        }
    }
}

#[async_trait]
impl AgentBackend for FallbackBackend {
    async fn generate(&self, system: &str, user: &str, timeout_ms: u64) -> Result<Value> {
        // Primary: acptoapi via HTTP
        match self.try_http(system, user, timeout_ms).await {
            Ok(v) => return Ok(v),
            Err(e) => {
                eprintln!("[rs-learn] acptoapi HTTP failed: {} — trying ACP subprocess", e);
            }
        }

        // Secondary: ACP subprocess
        match self.try_acp(system, user, timeout_ms).await {
            Ok(v) => return Ok(v),
            Err(e) => {
                eprintln!("[rs-learn] ACP subprocess failed: {} — falling back to AGENTS.md", e);
            }
        }

        // Tertiary: AGENTS.md content
        self.try_agents_md(system, user)
    }

    fn name(&self) -> &'static str {
        "fallback(acptoapi+acp+agents_md)"
    }
}
