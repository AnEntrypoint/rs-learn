pub mod kmeans;

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
const QUALITY_THRESHOLD: f32 = 0.7;

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
}

impl BackgroundLoop {
    pub fn new(store: Arc<Store>, router: Arc<Mutex<Router>>, acp: Option<Arc<dyn AgentBackend>>, reasoning: Arc<ReasoningBank>, instant: Option<Arc<Mutex<InstantLoop>>>) -> Arc<Self> {
        let run_count = Arc::new(AtomicU64::new(0));
        let last_k = Arc::new(AtomicU64::new(0));
        let last_duration_ms = Arc::new(AtomicU64::new(0));
        let this = Arc::new(Self {
            store, router, acp, reasoning, instant,
            k: DEFAULT_K, limit: DEFAULT_LIMIT, seed: DEFAULT_SEED,
            run_count: run_count.clone(), last_k: last_k.clone(), last_duration_ms: last_duration_ms.clone(),
        });
        observability::register("background", move || {
            json!({
                "run_count": run_count.load(Ordering::Relaxed),
                "last_k": last_k.load(Ordering::Relaxed),
                "last_duration_ms": last_duration_ms.load(Ordering::Relaxed),
            })
        });
        this
    }

    pub async fn run_once(&self) -> Result<RunStats> {
        let t0 = Instant::now();
        let rows = self.store.list_recent_trajectories_with_embeddings(self.limit).await?;
        let mut vectors: Vec<Vec<f32>> = Vec::new();
        let mut meta: Vec<(String, f64, String, Option<String>)> = Vec::new();
        for r in &rows {
            let Some(emb) = &r.query_embedding else { continue };
            if emb.len() != IN { continue; }
            vectors.push(emb.clone());
            meta.push((r.id.clone(), r.quality.unwrap_or(0.0), r.router_decision.clone().unwrap_or_default(), r.query.clone()));
        }
        if vectors.len() < 2 {
            self.run_count.fetch_add(1, Ordering::Relaxed);
            self.last_duration_ms.store(t0.elapsed().as_millis() as u64, Ordering::Relaxed);
            return Ok(RunStats { duration_ms: t0.elapsed().as_millis(), ..Default::default() });
        }
        let adaptive_k = self.k.min((vectors.len() / 4).max(2));
        let assigns = kmeans_plus_plus(&vectors, adaptive_k, self.seed);
        let kk = assigns.iter().map(|a| a.cluster).max().unwrap_or(0) + 1;
        let centroids = kmeans_centroids(&vectors, &assigns, kk);
        let mut groups: Vec<Vec<usize>> = (0..kk).map(|_| Vec::new()).collect();
        for a in &assigns { groups[a.cluster].push(a.index); }

        let mut patterns_written = 0usize;
        let mut reasoning_written = 0usize;
        for (j, members) in groups.iter().enumerate() {
            if members.is_empty() { continue; }
            let q_sum: f64 = members.iter().map(|&i| meta[i].1).sum();
            let pat_id = format!("pat-{}", &Uuid::new_v4().to_string()[..12]);
            let ts = now_ms();
            self.store.upsert_pattern(&PatternRow {
                id: pat_id.clone(),
                centroid: Some(centroids[j].clone()),
                count: Some(members.len() as i64),
                quality_sum: Some(q_sum),
                created_at: Some(ts),
            }).await?;
            patterns_written += 1;
            let strategy_text = self.summarize_cluster(members, &meta).await;
            self.store.insert_reasoning(&ReasoningRow {
                id: format!("rsn-{}", &Uuid::new_v4().to_string()[..12]),
                pattern_id: Some(pat_id),
                strategy: strategy_text,
                success_rate: Some(q_sum / members.len() as f64),
                created_at: Some(ts),
            }).await?;
            reasoning_written += 1;
        }

        let mut batch: Vec<TrainSample> = Vec::new();
        for (i, (_, q, dec, query)) in meta.iter().enumerate() {
            if *q < QUALITY_THRESHOLD as f64 { continue; }
            let model = serde_json::from_str::<serde_json::Value>(dec).ok()
                .and_then(|v| v.get("model").and_then(|m| m.as_str()).map(String::from));
            let Some(chosen) = model else { continue };
            let estimated_tokens = query.as_ref().map(|q| q.len() as u64).unwrap_or(0);
            batch.push(TrainSample { embedding: vectors[i].clone(), chosen_target: chosen, quality: *q as f32, estimated_tokens });
        }
        let trained_on = if !batch.is_empty() {
            let mut r = self.router.lock().await;
            let n = r.train(&batch)?;
            if n > 0 { r.save().await?; }
            n
        } else { 0 };

        let dur = t0.elapsed().as_millis();
        self.run_count.fetch_add(1, Ordering::Relaxed);
        self.last_k.store(patterns_written as u64, Ordering::Relaxed);
        self.last_duration_ms.store(dur as u64, Ordering::Relaxed);
        Ok(RunStats { clusters: kk, patterns_written, reasoning_written, trained_on, duration_ms: dur })
    }

    async fn summarize_cluster(&self, members: &[usize], meta: &[(String, f64, String, Option<String>)]) -> String {
        let samples: Vec<String> = members.iter()
            .filter_map(|&i| meta[i].3.clone())
            .filter(|q| !q.trim().is_empty())
            .take(5)
            .collect();
        let fallback = match samples.first() {
            Some(first) => {
                let prefix: String = first.chars().take(60).collect();
                format!("cluster of {} queries around: {}", members.len(), prefix.trim())
            }
            None => format!("cluster of {} trajectories", members.len()),
        };
        let Some(acp) = &self.acp else { return fallback; };
        if samples.is_empty() { return fallback; }
        let prompt = format!(
            "Summarize the shared intent of these user queries as a single reusable strategy (<100 chars):\n- {}",
            samples.join("\n- ")
        );
        match acp.generate("", &prompt, 20_000).await {
            Ok(v) => v.get("strategy").and_then(|s| s.as_str()).map(String::from)
                .unwrap_or_else(|| {
                    match &v {
                        serde_json::Value::String(s) if !s.trim().is_empty() => s.clone(),
                        other => {
                            let s = other.to_string();
                            if s.trim().is_empty() || s == "null" { fallback.clone() } else { s }
                        }
                    }
                }),
            Err(_) => fallback,
        }
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
mod tests {
    use super::*;
    use crate::store::types::TrajectoryRow;
    use kmeans::mulberry32;

    #[test]
    fn kmeans_deterministic_same_seed() {
        let mut rng = mulberry32(7);
        let vectors: Vec<Vec<f32>> = (0..30).map(|_| (0..IN).map(|_| rng() - 0.5).collect()).collect();
        let a = kmeans_plus_plus(&vectors, 4, 42);
        let b = kmeans_plus_plus(&vectors, 4, 42);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) { assert_eq!(x.cluster, y.cluster); }
    }

    #[tokio::test]
    async fn run_once_writes_patterns_and_trains() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap().to_string();
        drop(tmp);
        let store = Arc::new(Store::open(&path).await.unwrap());
        let targets = vec!["a".to_string(), "b".to_string()];
        let router = Arc::new(Mutex::new(Router::new(store.clone(), targets.clone())));
        let reasoning = Arc::new(ReasoningBank::new(store.clone()));

        let mut rng = mulberry32(11);
        for i in 0..8 {
            let emb: Vec<f32> = (0..IN).map(|_| rng() - 0.5).collect();
            let model = if i % 2 == 0 { "a" } else { "b" };
            store.insert_trajectory(&TrajectoryRow {
                id: format!("t{}", i),
                session_id: Some("s".into()),
                query: Some(format!("q{}", i)),
                query_embedding: Some(emb),
                retrieved_ids: None,
                router_decision: Some(format!("{{\"model\":\"{}\"}}", model)),
                response: Some("r".into()),
                activations: None,
                quality: Some(0.9),
                latency_ms: Some(1),
                created_at: Some(1000 + i as i64),
            }).await.unwrap();
        }

        let bg = BackgroundLoop::new(store.clone(), router.clone(), None, reasoning, None);
        let stats = bg.run_once().await.unwrap();
        assert!(stats.patterns_written > 0, "no patterns written");
        assert!(stats.trained_on > 0, "router.train not invoked on quality>=0.7 batch");
        let patterns = store.count_rows("patterns").await;
        assert!(patterns > 0, "patterns table empty");
        let reasoning = store.count_rows("reasoning_bank").await;
        assert!(reasoning > 0, "reasoning_bank empty");
    }

    #[tokio::test]
    async fn summarize_prompt_uses_query_content_not_ids() {
        use crate::backend::AgentBackend;
        use crate::errors::{LlmError, Result as LlmResult};
        use async_trait::async_trait;
        use std::sync::Mutex as StdMutex;

        struct CapBackend(Arc<StdMutex<Vec<String>>>);
        #[async_trait]
        impl AgentBackend for CapBackend {
            async fn generate(&self, _s: &str, u: &str, _t: u64) -> LlmResult<serde_json::Value> {
                self.0.lock().unwrap().push(u.to_string());
                Err(LlmError::Validation("stub".into()))
            }
            fn name(&self) -> &'static str { "cap" }
        }

        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap().to_string();
        drop(tmp);
        let store = Arc::new(Store::open(&path).await.unwrap());
        let targets = vec!["a".to_string()];
        let router = Arc::new(Mutex::new(Router::new(store.clone(), targets)));
        let reasoning = Arc::new(ReasoningBank::new(store.clone()));
        let hits = Arc::new(StdMutex::new(Vec::new()));
        let acp: Arc<dyn AgentBackend> = Arc::new(CapBackend(hits.clone()));

        let mut rng = kmeans::mulberry32(9);
        for i in 0..4 {
            let emb: Vec<f32> = (0..IN).map(|_| rng() - 0.5).collect();
            store.insert_trajectory(&TrajectoryRow {
                id: format!("t{}", i),
                session_id: Some("s".into()),
                query: Some(format!("how do I refactor {} module", i)),
                query_embedding: Some(emb),
                retrieved_ids: None,
                router_decision: Some("{\"model\":\"a\"}".to_string()),
                response: None,
                activations: None,
                quality: Some(0.9),
                latency_ms: Some(1),
                created_at: Some(100 + i as i64),
            }).await.unwrap();
        }
        let bg = BackgroundLoop::new(store.clone(), router, Some(acp), reasoning, None);
        let _ = bg.run_once().await.unwrap();
        let captured = hits.lock().unwrap().clone();
        assert!(!captured.is_empty(), "backend was not called");
        let p = &captured[0];
        assert!(p.contains("refactor"), "prompt lacks query content: {p}");
        assert!(!p.contains("t0,"), "prompt must not contain raw trajectory ids");
    }
}
