mod claude_cli;
mod openai_compat;

use crate::acp::AcpClient;
use crate::errors::{LlmError, Result};
use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;

pub use claude_cli::ClaudeCliClient;
pub use openai_compat::OpenAiCompatClient;

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

#[async_trait]
impl AgentBackend for OpenAiCompatClient {
    async fn generate(&self, system: &str, user: &str, timeout_ms: u64) -> Result<Value> {
        OpenAiCompatClient::generate(self, system, user, timeout_ms).await
    }
    fn name(&self) -> &'static str { "openai-compat" }
}

pub fn from_env() -> Result<Arc<dyn AgentBackend>> {
    let explicit = std::env::var("RS_LEARN_BACKEND").ok();

    match explicit.as_deref() {
        Some("openai-compat") => {
            return Ok(Arc::new(OpenAiCompatClient::from_env()?) as Arc<dyn AgentBackend>);
        }
        Some("claude-cli") => {
            return Ok(Arc::new(ClaudeCliClient::from_env()?) as Arc<dyn AgentBackend>);
        }
        Some("acp") => {
            return Ok(Arc::new(AcpClient::from_env()?) as Arc<dyn AgentBackend>);
        }
        Some(other) => {
            return Err(LlmError::Validation(
                format!("RS_LEARN_BACKEND='{other}' unknown; expected 'openai-compat', 'acp', or 'claude-cli'")
            ));
        }
        None => {}
    }

    if is_port_reachable("127.0.0.1", 4800) {
        tracing::info!("acptoapi detected on 127.0.0.1:4800, using as primary provider");
        return Ok(Arc::new(OpenAiCompatClient::from_env()?) as Arc<dyn AgentBackend>);
    }

    let has_acp = std::env::var("RS_LEARN_ACP_COMMAND").is_ok();
    let selection = if has_acp { "acp" } else { "claude-cli" };

    match selection {
        "acp" => Ok(Arc::new(AcpClient::from_env()?) as Arc<dyn AgentBackend>),
        "claude-cli" => Ok(Arc::new(ClaudeCliClient::from_env()?) as Arc<dyn AgentBackend>),
        _ => unreachable!(),
    }
}

fn is_port_reachable(host: &str, port: u16) -> bool {
    use std::net::TcpStream;
    use std::time::Duration;

    let addr = format!("{}:{}", host, port);
    TcpStream::connect_timeout(
        &addr.parse().unwrap_or_else(|_| {
            ([127, 0, 0, 1], port).into()
        }),
        Duration::from_millis(500),
    )
    .is_ok()
}
