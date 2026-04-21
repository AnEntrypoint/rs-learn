use crate::acp::AcpClient;
use crate::learn::instant::InstantLoop;
use crate::learn::reasoning_bank::ReasoningBank;
use crate::observability;
use crate::router::{Router, TrainSample, IN};
use crate::store::types::{PatternRow, ReasoningRow};
use crate::store::Store;
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
const MAX_ITER: usize = 25;
const QUALITY_THRESHOLD: f32 = 0.7;

#[derive(Debug, Clone, Default)]
pub struct RunStats {
    pub clusters: usize,
    pub patterns_written: usize,
    pub reasoning_written: usize,
    pub trained_on: usize,
    pub duration_ms: u128,
}

#[derive(Debug, Clone)]
pub struct ClusterAssignment {
    pub index: usize,
    pub cluster: usize,
}

fn mulberry32(seed: u32) -> impl FnMut() -> f32 {
    let mut a: u32 = seed;
    move || {
        a = a.wrapping_add(0x6D2B_79F5);
        let mut t = a;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        ((t ^ (t >> 14)) as f32) / 4_294_967_296.0
    }
}

fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0f32; let mut na = 0f32; let mut nb = 0f32;
    for i in 0..a.len() { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-9);
    1.0 - dot / denom
}

pub fn kmeans_plus_plus(vectors: &[Vec<f32>], k: usize, seed: u32) -> Vec<ClusterAssignment> {
    let n = vectors.len();
    if n == 0 || k == 0 { return vec![]; }
    let kk = k.min(n);
    let mut rng = mulberry32(seed);
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(kk);
    let first = (rng() * n as f32).floor() as usize;
    centroids.push(vectors[first.min(n - 1)].clone());
    let mut d2 = vec![0f32; n];
    while centroids.len() < kk {
        let mut sum = 0f32;
        for i in 0..n {
            let mut m = f32::INFINITY;
            for c in &centroids {
                let v = cosine_dist(&vectors[i], c);
                if v < m { m = v; }
            }
            d2[i] = m; sum += m;
        }
        if sum == 0.0 { break; }
        let mut r = rng() * sum;
        let mut pick = n - 1;
        for i in 0..n {
            r -= d2[i];
            if r <= 0.0 { pick = i; break; }
        }
        centroids.push(vectors[pick].clone());
    }
    let dim = vectors[0].len();
    let mut assign = vec![0usize; n];
    for _ in 0..MAX_ITER {
        let mut changed = 0usize;
        for i in 0..n {
            let mut best = 0usize; let mut bd = f32::INFINITY;
            for (j, c) in centroids.iter().enumerate() {
                let d = cosine_dist(&vectors[i], c);
                if d < bd { bd = d; best = j; }
            }
            if assign[i] != best { changed += 1; assign[i] = best; }
        }
        let mut sums: Vec<Vec<f32>> = centroids.iter().map(|_| vec![0f32; dim]).collect();
        let mut counts = vec![0usize; centroids.len()];
        for i in 0..n {
            let a = assign[i]; counts[a] += 1;
            for d in 0..dim { sums[a][d] += vectors[i][d]; }
        }
        for j in 0..centroids.len() {
            if counts[j] == 0 { continue; }
            for d in 0..dim { centroids[j][d] = sums[j][d] / counts[j] as f32; }
        }
        if changed == 0 { break; }
    }
    (0..n).map(|i| ClusterAssignment { index: i, cluster: assign[i] }).collect()
}

pub fn kmeans_centroids(vectors: &[Vec<f32>], assignments: &[ClusterAssignment], k: usize) -> Vec<Vec<f32>> {
    if vectors.is_empty() { return vec![]; }
    let dim = vectors[0].len();
    let kk = k.min(vectors.len());
    let mut sums: Vec<Vec<f32>> = (0..kk).map(|_| vec![0f32; dim]).collect();
    let mut counts = vec![0usize; kk];
    for a in assignments {
        if a.cluster >= kk { continue; }
        counts[a.cluster] += 1;
        for d in 0..dim { sums[a.cluster][d] += vectors[a.index][d]; }
    }
    for j in 0..kk {
        if counts[j] == 0 { continue; }
        for d in 0..dim { sums[j][d] /= counts[j] as f32; }
    }
    sums
}

pub struct BackgroundLoop {
    pub store: Arc<Store>,
    pub router: Arc<Mutex<Router>>,
    pub acp: Option<Arc<AcpClient>>,
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
    pub fn new(store: Arc<Store>, router: Arc<Mutex<Router>>, acp: Option<Arc<AcpClient>>, reasoning: Arc<ReasoningBank>, instant: Option<Arc<Mutex<InstantLoop>>>) -> Arc<Self> {
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
        let mut meta: Vec<(String, f64, String)> = Vec::new();
        for r in &rows {
            let Some(emb) = &r.query_embedding else { continue };
            if emb.len() != IN { continue; }
            vectors.push(emb.clone());
            meta.push((r.id.clone(), r.quality.unwrap_or(0.0), r.router_decision.clone().unwrap_or_default()));
        }
        if vectors.len() < 2 {
            self.run_count.fetch_add(1, Ordering::Relaxed);
            self.last_duration_ms.store(t0.elapsed().as_millis() as u64, Ordering::Relaxed);
            return Ok(RunStats { duration_ms: t0.elapsed().as_millis(), ..Default::default() });
        }
        let assigns = kmeans_plus_plus(&vectors, self.k, self.seed);
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
            self.store.upsert_pattern(&PatternRow {
                id: pat_id.clone(),
                centroid: Some(centroids[j].clone()),
                count: Some(members.len() as i64),
                quality_sum: Some(q_sum),
                created_at: None,
            }).await?;
            patterns_written += 1;
            let strategy_text = self.summarize_cluster(members, &meta).await;
            self.store.insert_reasoning(&ReasoningRow {
                id: format!("rsn-{}", &Uuid::new_v4().to_string()[..12]),
                pattern_id: Some(pat_id),
                strategy: strategy_text,
                success_rate: Some(q_sum / members.len() as f64),
                created_at: None,
            }).await?;
            reasoning_written += 1;
        }

        let mut batch: Vec<TrainSample> = Vec::new();
        for (i, (_, q, dec)) in meta.iter().enumerate() {
            if *q < QUALITY_THRESHOLD as f64 { continue; }
            let model = serde_json::from_str::<serde_json::Value>(dec).ok()
                .and_then(|v| v.get("model").and_then(|m| m.as_str()).map(String::from));
            let Some(chosen) = model else { continue };
            batch.push(TrainSample { embedding: vectors[i].clone(), chosen_target: chosen, quality: *q as f32 });
        }
        let trained_on = if !batch.is_empty() {
            let mut r = self.router.lock().await;
            r.train(&batch)?
        } else { 0 };

        let dur = t0.elapsed().as_millis();
        self.run_count.fetch_add(1, Ordering::Relaxed);
        self.last_k.store(patterns_written as u64, Ordering::Relaxed);
        self.last_duration_ms.store(dur as u64, Ordering::Relaxed);
        Ok(RunStats { clusters: kk, patterns_written, reasoning_written, trained_on, duration_ms: dur })
    }

    async fn summarize_cluster(&self, members: &[usize], meta: &[(String, f64, String)]) -> String {
        let Some(acp) = &self.acp else {
            return format!("auto-cluster n={}", members.len());
        };
        let sample: Vec<String> = members.iter().take(5).map(|&i| meta[i].0.clone()).collect();
        let prompt = format!("summarize these trajectories: {}", sample.join(", "));
        match acp.generate("", &prompt, 20_000).await {
            Ok(v) => v.get("strategy").and_then(|s| s.as_str()).map(String::from)
                .unwrap_or_else(|| v.to_string()),
            Err(_) => format!("auto-cluster n={}", members.len()),
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
}
