pub mod kmeans;
mod run;

pub use kmeans::{kmeans_plus_plus, kmeans_centroids, ClusterAssignment};

use crate::backend::AgentBackend;
use crate::learn::instant::InstantLoop;
use crate::learn::reasoning_bank::ReasoningBank;
use crate::observability;
use crate::router::{Router, TrainSample, IN};
use crate::store::types::{PatternRow, ReasoningRow};
use crate::store::{now_ms, Store};
use anyhow::Result;
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use uuid::Uuid;

const DEFAULT_K: usize = 100;
const DEFAULT_LIMIT: usize = 1000;
const DEFAULT_SEED: u32 = 42;

fn env_limit() -> usize {
    std::env::var("RS_LEARN_BG_LIMIT").ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n: &usize| n >= 100 && n <= 100_000)
        .unwrap_or(DEFAULT_LIMIT)
}

#[derive(Debug, Clone, Default)]
pub struct RunStats {
    pub clusters: usize,
    pub patterns_written: usize,
    pub reasoning_written: usize,
    pub trained_on: usize,
    pub duration_ms: u128,
}

pub struct BackgroundLoop {
    pub store: Arc<Store>,
    pub router: Arc<Mutex<Router>>,
    pub acp: Option<Arc<dyn AgentBackend>>,
    pub reasoning: Arc<ReasoningBank>,
    pub instant: Option<Arc<Mutex<InstantLoop>>>,
    pub k: usize,
    pub limit: usize,
    pub seed: u32,
    run_count: Arc<AtomicU64>,
    last_k: Arc<AtomicU64>,
    last_duration_ms: Arc<AtomicU64>,
    last_evicted_reasoning: Arc<AtomicU64>,
}

impl BackgroundLoop {
    pub fn new(store: Arc<Store>, router: Arc<Mutex<Router>>, acp: Option<Arc<dyn AgentBackend>>, reasoning: Arc<ReasoningBank>, instant: Option<Arc<Mutex<InstantLoop>>>) -> Arc<Self> {
        let run_count = Arc::new(AtomicU64::new(0));
        let last_k = Arc::new(AtomicU64::new(0));
        let last_duration_ms = Arc::new(AtomicU64::new(0));
        let last_evicted_reasoning = Arc::new(AtomicU64::new(0));
        let this = Arc::new(Self {
            store, router, acp, reasoning, instant,
            k: DEFAULT_K, limit: env_limit(), seed: DEFAULT_SEED,
            run_count: run_count.clone(), last_k: last_k.clone(), last_duration_ms: last_duration_ms.clone(),
            last_evicted_reasoning: last_evicted_reasoning.clone(),
        });
        observability::register("background", move || {
            json!({
                "run_count": run_count.load(Ordering::Relaxed),
                "last_k": last_k.load(Ordering::Relaxed),
                "last_duration_ms": last_duration_ms.load(Ordering::Relaxed),
                "last_evicted_reasoning": last_evicted_reasoning.load(Ordering::Relaxed),
            })
        });
        this
    }

    pub async fn schedule(self: Arc<Self>, interval: Duration) {
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            if let Err(e) = self.run_once().await {
                tracing::error!(target: "learn-bg", error = %e, "run_once failed");
            }
        }
    }
}

impl Drop for BackgroundLoop {
    fn drop(&mut self) { observability::unregister("background"); }
}

#[cfg(test)]
#[path = "bg_tests.rs"]
mod tests;
