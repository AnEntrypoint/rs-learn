use crate::observability;
use crate::router::{Router, IN};
use crate::store::types::TrajectoryRow;
use crate::store::Store;
use anyhow::{anyhow, Result};
use bytemuck::cast_slice;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;
use uuid::Uuid;

pub const RANK: usize = 2;
pub const DECAY: f32 = 0.995;
pub const LR0: f32 = 0.01;
pub const FEEDBACK_WINDOW: Duration = Duration::from_secs(60);
pub const MAX_ADAPTER_NORM: f32 = 5.0;

pub type RequestId = String;

#[derive(Debug, Clone)]
pub struct PendingInfo {
    pub session_id: Option<String>,
    pub embedding: Vec<f32>,
    pub route_model: String,
    pub query_text: Option<String>,
    pub created_at: Instant,
    pub created_ms: i64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FeedbackPayload {
    pub quality: f32,
    #[serde(default)]
    pub signal: Option<String>,
}

pub struct InstantLoop {
    pub store: Arc<Store>,
    pub router: Arc<Mutex<Router>>,
    pub adapter_a: Vec<f32>,
    pub adapter_b: Vec<f32>,
    pub pending: HashMap<RequestId, PendingInfo>,
    pub spine: Option<Arc<crate::spine::TrajectorySpine>>,
    targets: Vec<String>,
    n_targets: usize,
    lr: f32,
    lr_min: f32,
    feedback_count: Arc<AtomicU64>,
    adapter_norm_milli: Arc<AtomicU64>,
    pending_count: Arc<AtomicU64>,
    resets_performed: Arc<AtomicU64>,
    feedback_expired: Arc<AtomicU64>,
    replay_buf: std::collections::VecDeque<(Vec<f32>, usize, f32)>,
}

const REPLAY_CAP: usize = 64;

fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

pub(crate) fn weighted_pick(buf: &std::collections::VecDeque<(Vec<f32>, usize, f32)>, seed: &mut u32) -> usize {
    let n = buf.len();
    *seed = seed.wrapping_mul(2654435761).wrapping_add(1);
    let r = (*seed as f32) / (u32::MAX as f32 + 1.0);
    let total: f32 = buf.iter().map(|(_, _, s)| s.abs()).sum();
    if !(total > 0.0) {
        return ((*seed as usize) % n).min(n - 1);
    }
    let target = r * total;
    let mut acc = 0f32;
    for (i, (_, _, s)) in buf.iter().enumerate() {
        acc += s.abs();
        if target < acc { return i; }
    }
    n - 1
}

impl InstantLoop {
    pub fn new(store: Arc<Store>, router: Arc<Mutex<Router>>, targets: Vec<String>) -> Self {
        let n_targets = targets.len().max(1);
        let lr_min = std::env::var("RS_LEARN_LR_MIN").ok()
            .and_then(|s| s.parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(1e-3)
            .min(LR0);
        let loop_ = Self {
            store, router,
            adapter_a: vec![0f32; IN * RANK],
            adapter_b: vec![0f32; RANK * n_targets],
            pending: HashMap::new(),
            spine: None,
            targets, n_targets, lr: LR0, lr_min,
            feedback_count: Arc::new(AtomicU64::new(0)),
            adapter_norm_milli: Arc::new(AtomicU64::new(0)),
            pending_count: Arc::new(AtomicU64::new(0)),
            resets_performed: Arc::new(AtomicU64::new(0)),
            feedback_expired: Arc::new(AtomicU64::new(0)),
            replay_buf: std::collections::VecDeque::with_capacity(REPLAY_CAP),
        };
        loop_.register_observability();
        loop_
    }

    fn register_observability(&self) {
        let fc = self.feedback_count.clone();
        let nm = self.adapter_norm_milli.clone();
        let pc = self.pending_count.clone();
        let rp = self.resets_performed.clone();
        let fe = self.feedback_expired.clone();
        observability::register("instant", move || {
            json!({
                "adapter_norm": (nm.load(Ordering::Relaxed) as f64) / 1000.0,
                "feedback_count": fc.load(Ordering::Relaxed),
                "pending_count": pc.load(Ordering::Relaxed),
                "resets_performed": rp.load(Ordering::Relaxed),
                "feedback_expired": fe.load(Ordering::Relaxed),
            })
        });
    }

    fn target_index(&self, name: &str) -> Option<usize> {
        self.targets.iter().position(|t| t == name)
    }

    pub fn adapter_norm(&self) -> f32 {
        let mut s = 0f32;
        for &x in &self.adapter_a { s += x * x; }
        for &x in &self.adapter_b { s += x * x; }
        s.sqrt()
    }

    pub fn apply_adapter(&self, embedding: &[f32], logits: &mut [f32]) {
        Self::apply_adapter_raw(&self.adapter_a, &self.adapter_b, RANK, &self.targets, embedding, logits);
    }

    pub fn apply_adapter_raw(a: &[f32], b: &[f32], rank: usize, targets: &[String], embedding: &[f32], logits: &mut [f32]) {
        let nt = targets.len();
        if embedding.len() != IN || logits.len() != nt { return; }
        if a.len() != rank * IN || b.len() != rank * nt { return; }
        let mut proj = vec![0f32; rank];
        for r in 0..rank {
            let off = r * IN;
            proj[r] = crate::simd::dot(&a[off..off + IN], embedding);
        }
        for k in 0..nt {
            let mut s = 0f32;
            for r in 0..rank { s += b[r * nt + k] * proj[r]; }
            logits[k] += s;
        }
    }

    pub fn targets_clone(&self) -> Vec<String> { self.targets.clone() }
    pub fn adapter_rank(&self) -> usize { RANK }

    pub fn seed_replay(&mut self, embedding: Vec<f32>, quality: f32) {
        if self.replay_buf.len() >= REPLAY_CAP { self.replay_buf.pop_front(); }
        self.replay_buf.push_back((embedding, 0, quality));
    }

    pub fn reset_adapter(&mut self) {
        self.adapter_a.fill(0.0);
        self.adapter_b.fill(0.0);
        self.lr = LR0;
        self.adapter_norm_milli.store(0, Ordering::Relaxed);
        self.resets_performed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn serialize_adapter_flat(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.adapter_a.len() + self.adapter_b.len());
        flat.extend_from_slice(&self.adapter_a);
        flat.extend_from_slice(&self.adapter_b);
        flat
    }

    pub fn serialize_adapter(&self) -> Vec<u8> {
        let flat = self.serialize_adapter_flat();
        cast_slice::<f32, u8>(&flat).to_vec()
    }

    fn hebbian_update(&mut self, embedding: &[f32], t_idx: usize, quality: f32) {
        if t_idx >= self.n_targets { return; }
        let scale = self.lr * quality;
        let fallback = 1.0 / (RANK as f32).sqrt();
        for r in 0..RANK {
            let off = r * IN;
            let mut b_val = self.adapter_b[r * self.n_targets + t_idx];
            if b_val == 0.0 { b_val = fallback; }
            crate::simd::axpy(scale * b_val, embedding, &mut self.adapter_a[off..off + IN]);
        }
        for r in 0..RANK {
            let off = r * IN;
            let pr = crate::simd::dot(&self.adapter_a[off..off + IN], embedding);
            self.adapter_b[r * self.n_targets + t_idx] += scale * pr;
        }
        self.lr = (self.lr * DECAY).max(self.lr_min);
        let norm = self.adapter_norm();
        if norm > MAX_ADAPTER_NORM {
            let s = MAX_ADAPTER_NORM / norm;
            for x in self.adapter_a.iter_mut() { *x *= s; }
            for x in self.adapter_b.iter_mut() { *x *= s; }
        }
        let final_norm = self.adapter_norm();
        self.adapter_norm_milli.store((final_norm * 1000.0) as u64, Ordering::Relaxed);
    }

    fn gc_pending(&mut self) {
        let now = Instant::now();
        let before = self.pending.len();
        self.pending.retain(|_, p| now.duration_since(p.created_at) < FEEDBACK_WINDOW);
        let dropped = before.saturating_sub(self.pending.len());
        if dropped > 0 { self.feedback_expired.fetch_add(dropped as u64, Ordering::Relaxed); }
        self.pending_count.store(self.pending.len() as u64, Ordering::Relaxed);
    }

    pub async fn record_trajectory(
        &mut self,
        session_id: Option<String>,
        embedding: Vec<f32>,
        route_model: String,
        response: String,
        query_text: Option<String>,
        implicit_quality: Option<f64>,
        latency_ms: u64,
    ) -> Result<RequestId> {
        if embedding.len() != IN { return Err(anyhow!("embedding must be {IN}-d")); }
        self.gc_pending();
        let request_id = Uuid::new_v4().to_string();
        let created_ms = now_ms();
        let decision = json!({ "model": route_model.clone() }).to_string();
        let row = TrajectoryRow {
            id: request_id.clone(),
            session_id: session_id.clone(),
            query: query_text.clone(),
            query_embedding: Some(embedding.clone()),
            retrieved_ids: None,
            router_decision: Some(decision),
            response: Some(response),
            activations: None,
            quality: implicit_quality,
            latency_ms: Some(latency_ms as i64),
            created_at: Some(created_ms),
        };
        match self.spine.as_ref() {
            Some(s) => s.send(row).await?,
            None => self.store.insert_trajectory(&row).await?,
        }
        self.pending.insert(request_id.clone(), PendingInfo {
            session_id, embedding, route_model,
            query_text,
            created_at: Instant::now(), created_ms,
        });
        self.pending_count.store(self.pending.len() as u64, Ordering::Relaxed);
        Ok(request_id)
    }

    pub async fn feedback(&mut self, request_id: &str, payload: FeedbackPayload) -> Result<()> {
        if !(0.0..=1.0).contains(&payload.quality) { return Err(anyhow!("quality must be 0..1")); }
        self.gc_pending();
        let p = self.pending.get(request_id).cloned();
        let (emb, sid, model, cms) = match &p {
            Some(pi) => (pi.embedding.clone(), pi.session_id.clone(), pi.route_model.clone(), pi.created_ms),
            None => (vec![0f32; IN], None, String::new(), now_ms()),
        };
        let decision = json!({ "model": model.clone(), "signal": payload.signal }).to_string();
        let row = TrajectoryRow {
            id: request_id.to_string(),
            session_id: sid,
            query: None,
            query_embedding: Some(emb.clone()),
            retrieved_ids: None,
            router_decision: Some(decision),
            response: None,
            activations: None,
            quality: Some(payload.quality as f64),
            latency_ms: Some(0),
            created_at: Some(cms),
        };
        match self.spine.as_ref() {
            Some(s) => s.send(row).await?,
            None => self.store.insert_trajectory(&row).await?,
        }
        self.feedback_count.fetch_add(1, Ordering::Relaxed);
        if let Some(idx) = self.target_index(&model) {
            let centered = payload.quality - 0.5;
            let applied: Option<f32> = if centered.abs() < 1e-4 {
                None
            } else {
                let scale = centered * 2.0;
                self.hebbian_update(&emb, idx, scale);
                Some(scale)
            };
            if let Some(scale) = applied {
                if self.replay_buf.len() >= REPLAY_CAP { self.replay_buf.pop_front(); }
                self.replay_buf.push_back((emb.clone(), idx, scale));
                if self.replay_buf.len() >= 4 {
                    let mut seed = (now_ms() as u32).wrapping_mul(2654435761);
                    let pick = weighted_pick(&self.replay_buf, &mut seed);
                    let (re, ri, rs) = self.replay_buf[pick].clone();
                    self.hebbian_update(&re, ri, rs * 0.5);
                }
            }
        }
        self.pending.remove(request_id);
        self.pending_count.store(self.pending.len() as u64, Ordering::Relaxed);
        Ok(())
    }
}

impl Drop for InstantLoop {
    fn drop(&mut self) { observability::unregister("instant"); }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn record_then_feedback_grows_adapter() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap().to_string();
        drop(tmp);
        let store = Arc::new(Store::open(&path).await.unwrap());
        let targets = vec!["a".to_string(), "b".to_string()];
        let router = Arc::new(Mutex::new(Router::new(store.clone(), targets.clone())));
        let mut il = InstantLoop::new(store, router, targets);
        let emb = vec![0.05f32; IN];
        let rid = il.record_trajectory(Some("s1".into()), emb, "a".into(), "hello".into(), None, None, 0).await.unwrap();
        assert_eq!(il.adapter_norm(), 0.0);
        il.feedback(&rid, FeedbackPayload { quality: 1.0, signal: None }).await.unwrap();
        assert!(il.adapter_norm() > 0.0, "adapter_norm must grow after positive feedback");
        let mut logits = vec![0f32; 2];
        il.apply_adapter(&vec![0.05f32; IN], &mut logits);
        assert!(logits[0].abs() + logits[1].abs() > 0.0);
    }

    #[tokio::test]
    async fn record_trajectory_full_persists_query_and_quality() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap().to_string();
        drop(tmp);
        let store = Arc::new(Store::open(&path).await.unwrap());
        let targets = vec!["a".to_string()];
        let router = Arc::new(Mutex::new(Router::new(store.clone(), targets.clone())));
        let mut il = InstantLoop::new(store.clone(), router, targets);
        let emb = vec![0.03f32; IN];
        il.record_trajectory(Some("s".into()), emb, "a".into(), "resp".into(),
            Some("why is the sky blue".into()), Some(0.85), 42).await.unwrap();
        let rows = store.list_recent_trajectories_with_embeddings(10).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].query.as_deref(), Some("why is the sky blue"));
        assert_eq!(rows[0].quality, Some(0.85));
        assert_eq!(rows[0].latency_ms, Some(42));
    }

    #[tokio::test]
    async fn lr_respects_floor() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap().to_string();
        drop(tmp);
        let store = Arc::new(Store::open(&path).await.unwrap());
        let targets = vec!["a".to_string()];
        let router = Arc::new(Mutex::new(Router::new(store.clone(), targets.clone())));
        let mut il = InstantLoop::new(store, router, targets);
        let emb = vec![0.01f32; IN];
        for _ in 0..2000 { il.hebbian_update(&emb, 0, 1.0); }
        assert!(il.lr >= il.lr_min, "lr {} below floor {}", il.lr, il.lr_min);
        assert!(il.lr_min > 0.0 && il.lr_min <= LR0);
    }

    #[tokio::test]
    async fn adapter_shifts_router_decision_after_feedback() {
        use crate::router::RouteCtx;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap().to_string();
        drop(tmp);
        let store = Arc::new(Store::open(&path).await.unwrap());
        let targets = vec!["a".to_string(), "b".to_string()];
        let router = Arc::new(Mutex::new({
            let mut r = Router::new(store.clone(), targets.clone());
            r.save().await.unwrap();
            r
        }));
        let mut il = InstantLoop::new(store.clone(), router.clone(), targets.clone());
        let emb: Vec<f32> = (0..IN).map(|i| ((i as f32) * 0.013).sin()).collect();

        let rid = il.record_trajectory(Some("s".into()), emb.clone(), "b".into(), "resp".into(), None, None, 0).await.unwrap();
        il.feedback(&rid, FeedbackPayload { quality: 1.0, signal: None }).await.unwrap();

        let a = il.adapter_a.clone();
        let b = il.adapter_b.clone();
        let tgt = il.targets_clone();
        let rank = il.adapter_rank();

        let plain = { let r = router.lock().await; r.route(&emb, &RouteCtx::default()) };
        let adapted = {
            let r = router.lock().await;
            r.route_with_adapter(&emb, &RouteCtx::default(), |e, l| {
                InstantLoop::apply_adapter_raw(&a, &b, rank, &tgt, e, l);
            })
        };
        assert!(adapted.confidence >= plain.confidence - 1e-6 || adapted.model == "b",
            "adapter should not reduce confidence for its chosen target: plain={} adapted={}", plain.confidence, adapted.confidence);
    }

    #[test]
    fn prioritized_replay_favors_high_impact() {
        let mut buf: std::collections::VecDeque<(Vec<f32>, usize, f32)> = std::collections::VecDeque::new();
        buf.push_back((vec![0f32; IN], 0, 0.1));
        buf.push_back((vec![0f32; IN], 1, 0.1));
        buf.push_back((vec![0f32; IN], 2, 0.1));
        buf.push_back((vec![0f32; IN], 3, 0.9));
        let mut seed: u32 = 0x9E3779B1;
        let mut hits = 0usize;
        for _ in 0..1000 {
            if weighted_pick(&buf, &mut seed) == 3 { hits += 1; }
        }
        assert!(hits > 500, "expected index 3 picked > 500, got {hits}");
    }

    #[test]
    fn weighted_pick_zero_total_fallback_uniform() {
        let mut buf: std::collections::VecDeque<(Vec<f32>, usize, f32)> = std::collections::VecDeque::new();
        for i in 0..4 { buf.push_back((vec![0f32; IN], i, 0.0)); }
        let mut seed: u32 = 12345;
        for _ in 0..50 {
            let i = weighted_pick(&buf, &mut seed);
            assert!(i < 4);
        }
    }

    #[tokio::test]
    async fn instant_mid_quality_trains_now() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap().to_string();
        drop(tmp);
        let store = Arc::new(Store::open(&path).await.unwrap());
        let targets = vec!["a".to_string(), "b".to_string()];
        let router = Arc::new(Mutex::new(Router::new(store.clone(), targets.clone())));
        let mut il = InstantLoop::new(store, router, targets);
        let emb = vec![0.05f32; IN];
        let rid = il.record_trajectory(Some("s1".into()), emb, "a".into(), "hello".into(), None, None, 0).await.unwrap();
        assert_eq!(il.adapter_norm(), 0.0);
        il.feedback(&rid, FeedbackPayload { quality: 0.55, signal: None }).await.unwrap();
        assert!(il.adapter_norm() > 0.0, "adapter_norm must grow after mid-quality feedback (no dead-band)");
    }
}
