use std::sync::Arc;
use std::sync::OnceLock;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

static LLM_GATE: OnceLock<Arc<Semaphore>> = OnceLock::new();

pub fn llm_gate() -> Arc<Semaphore> {
    LLM_GATE.get_or_init(|| {
        let raw = std::env::var("RS_LEARN_LLM_MAX_PARALLEL")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(1);
        let permits = raw.clamp(1, 3) as usize;
        Arc::new(Semaphore::new(permits))
    }).clone()
}

pub async fn acquire() -> std::result::Result<OwnedSemaphorePermit, tokio::sync::AcquireError> {
    llm_gate().acquire_owned().await
}
