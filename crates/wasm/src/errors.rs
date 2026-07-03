use thiserror::Error;

pub type Result<T> = std::result::Result<T, LlmError>;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("transient: {0}")] Transient(String),
    #[error("timeout: {0}")] Timeout(String),
    #[error("process: {0}")] Process(String),
    #[error("validation: {0}")] Validation(String),
    #[error("aborted: {0}")] Aborted(String),
    #[error(transparent)] Io(#[from] std::io::Error),
    #[error(transparent)] Json(#[from] serde_json::Error),
    #[error(transparent)] Other(#[from] anyhow::Error),
}

impl LlmError {
    pub fn is_transient(&self) -> bool { matches!(self, LlmError::Transient(_) | LlmError::Timeout(_)) }
}
