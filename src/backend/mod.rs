mod claude_cli;

use crate::acp::AcpClient;
use crate::errors::{LlmError, Result};
use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;

pub use claude_cli::ClaudeCliClient;

#[async_trait]
pub trait AgentBackend: Send + Sync {
    async fn generate(&self, system: &str, user: &str, timeout_ms: u64) -> Result<Value>;
    fn name(&self) -> &'static str;
}

#[async_trait]
impl AgentBackend for AcpClient {
    async fn generate(&self, system: &str, user: &str, timeout_ms: u64) -> Result<Value> {
        AcpClient::generate(self, system, user, timeout_ms).await
    }
    fn name(&self) -> &'static str { "acp" }
}

#[async_trait]
impl AgentBackend for ClaudeCliClient {
    async fn generate(&self, system: &str, user: &str, timeout_ms: u64) -> Result<Value> {
        ClaudeCliClient::generate(self, system, user, timeout_ms).await
    }
    fn name(&self) -> &'static str { "claude-cli" }
}

pub fn from_env() -> Result<Arc<dyn AgentBackend>> {
    let explicit = std::env::var("RS_LEARN_BACKEND").ok();
    let has_acp = std::env::var("RS_LEARN_ACP_COMMAND").is_ok();
    let selection = match explicit.as_deref() {
        Some("claude-cli") => "claude-cli",
        Some("acp") => "acp",
        Some(other) => return Err(LlmError::Validation(
            format!("RS_LEARN_BACKEND='{other}' unknown; expected 'acp' or 'claude-cli'")
        )),
        None if has_acp => "acp",
        None => "claude-cli",
    };
    match selection {
        "acp" => Ok(Arc::new(AcpClient::from_env()?) as Arc<dyn AgentBackend>),
        "claude-cli" => Ok(Arc::new(ClaudeCliClient::from_env()?) as Arc<dyn AgentBackend>),
        _ => unreachable!(),
    }
}
