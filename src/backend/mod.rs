mod openai_compat;
mod fallback;

use crate::errors::{LlmError, Result};
use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;

pub use fallback::FallbackBackend;
pub use openai_compat::OpenAiCompatClient;

#[async_trait]
pub trait AgentBackend: Send + Sync {
    async fn generate(&self, system: &str, user: &str, timeout_ms: u64) -> Result<Value>;
    fn name(&self) -> &'static str;
}

#[async_trait]
impl AgentBackend for OpenAiCompatClient {
    async fn generate(&self, system: &str, user: &str, timeout_ms: u64) -> Result<Value> {
        OpenAiCompatClient::generate(self, system, user, timeout_ms).await
    }
    fn name(&self) -> &'static str { "acptoapi" }
}

pub fn from_env() -> Result<Arc<dyn AgentBackend>> {
    let http = OpenAiCompatClient::from_env()?;
    let acp = crate::acp::AcpClient::from_env().ok();
    let agents_md_path = std::env::var("RS_LEARN_AGENTS_MD")
        .ok()
        .or_else(|| std::env::var("AGENTS_MD_PATH").ok());
    tracing::info!(
        "backend chain: acptoapi(http) + acp(subprocess) + agents_md(fallback)"
    );
    Ok(Arc::new(FallbackBackend::new(http, acp, agents_md_path)) as Arc<dyn AgentBackend>)
}
