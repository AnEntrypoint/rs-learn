use thiserror::Error;

pub type Result<T> = std::result::Result<T, LlmError>;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("acp transient: {0}")] Transient(String),
    #[error("acp timeout: {0}")] Timeout(String),
    #[error("acp process: {0}")] Process(String),
    #[error("acp validation: {0}")] Validation(String),
    #[error("acp aborted: {0}")] Aborted(String),
    #[error(transparent)] Io(#[from] std::io::Error),
    #[error(transparent)] Json(#[from] serde_json::Error),
    #[error(transparent)] Other(#[from] anyhow::Error),
}

impl LlmError {
    pub fn is_transient(&self) -> bool { matches!(self, LlmError::Transient(_) | LlmError::Timeout(_)) }
}
