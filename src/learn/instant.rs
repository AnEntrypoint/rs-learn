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
#[path = "instant_tests.rs"]
mod tests;
