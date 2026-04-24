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

        struct ClusterPrep { j: usize, members: Vec<usize>, q_sum: f64, pat_id: String, ts: i64 }
        let preps: Vec<ClusterPrep> = groups.iter().enumerate()
            .filter(|(_, m)| !m.is_empty())
            .map(|(j, members)| {
                let q_sum: f64 = members.iter().map(|&i| meta[i].1).sum();
                ClusterPrep { j, members: members.clone(), q_sum, pat_id: format!("pat-{}", &Uuid::new_v4().to_string()[..12]), ts: now_ms() }
            }).collect();

        let strategy_futures: Vec<_> = preps.iter().map(|p| self.summarize_cluster(&p.members, &meta)).collect();
        let strategies: Vec<String> = futures::future::join_all(strategy_futures).await;

        let mut patterns_written = 0usize;
        let mut reasoning_written = 0usize;
        for (p, strategy_text) in preps.iter().zip(strategies.into_iter()) {
            self.store.upsert_pattern(&PatternRow {
                id: p.pat_id.clone(),
                centroid: Some(centroids[p.j].clone()),
                count: Some(p.members.len() as i64),
                quality_sum: Some(p.q_sum),
                created_at: Some(p.ts),
            }).await?;
            patterns_written += 1;
            self.store.insert_reasoning(&ReasoningRow {
                id: format!("rsn-{}", &Uuid::new_v4().to_string()[..12]),
                pattern_id: Some(p.pat_id.clone()),
                strategy: strategy_text,
                success_rate: Some(p.q_sum / p.members.len() as f64),
                created_at: Some(p.ts),
            }).await?;
            reasoning_written += 1;
        }

        if let Some(instant) = &self.instant {
            let seeds: Vec<(Vec<f32>, f32)> = preps.iter()
                .filter(|p| p.members.len() > 0 && p.q_sum / p.members.len() as f64 >= 0.8)
                .take(3)
                .map(|p| (centroids[p.j].clone(), (p.q_sum / p.members.len() as f64) as f32))
                .collect();
            if !seeds.is_empty() {
                let mut il = instant.lock().await;
                for (centroid, quality) in seeds {
                    il.seed_replay(centroid, quality);
                }
            }
        }

        let mut batch: Vec<TrainSample> = Vec::new();
        for (i, (_, q, dec, query)) in meta.iter().enumerate() {
            let qf = *q as f32;
            let model = serde_json::from_str::<serde_json::Value>(dec).ok()
                .and_then(|v| v.get("model").and_then(|m| m.as_str()).map(String::from));
            let Some(chosen) = model else { continue };
            let estimated_tokens = query.as_ref()
                .map(|q| (q.split_whitespace().count() * 4 / 3) as u64)
                .unwrap_or(0);
            batch.push(TrainSample { embedding: vectors[i].clone(), chosen_target: chosen, quality: qf, estimated_tokens });
        }
        let trained_on = if !batch.is_empty() {
            let mut r = self.router.lock().await;
            let n = r.train(&batch)?;
            if n > 0 { r.save().await?; }
            n
        } else { 0 };

        let ttl = std::env::var("RS_LEARN_REASONING_TTL_DAYS").ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(7);
        let evicted_reasoning = self.store.evict_stale_reasoning(ttl, 0.3).await.unwrap_or(0);
        let evicted_patterns = self.store.evict_noise_patterns().await.unwrap_or(0);
        let keep = std::env::var("RS_LEARN_TRAJ_KEEP").ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n >= 500)
            .unwrap_or(10_000);
        let _ = self.store.prune_trajectories(keep).await;
        let _ = self.store.prune_router_weights(5).await;

        let dur = t0.elapsed().as_millis();
        self.run_count.fetch_add(1, Ordering::Relaxed);
        self.last_k.store(patterns_written as u64, Ordering::Relaxed);
        self.last_duration_ms.store(dur as u64, Ordering::Relaxed);
        self.last_evicted_reasoning.store(evicted_reasoning + evicted_patterns, Ordering::Relaxed);
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
#[path = "bg_tests.rs"]
mod tests;
