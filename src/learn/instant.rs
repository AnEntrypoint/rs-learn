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

pub type RequestId = String;

#[derive(Debug, Clone)]
pub struct PendingInfo {
    pub session_id: Option<String>,
    pub embedding: Vec<f32>,
    pub route_model: String,
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
    feedback_count: Arc<AtomicU64>,
}

fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

impl InstantLoop {
    pub fn new(store: Arc<Store>, router: Arc<Mutex<Router>>, targets: Vec<String>) -> Self {
        let n_targets = targets.len().max(1);
        let loop_ = Self {
            store, router,
            adapter_a: vec![0f32; IN * RANK],
            adapter_b: vec![0f32; RANK * n_targets],
            pending: HashMap::new(),
            spine: None,
            targets, n_targets, lr: LR0,
            feedback_count: Arc::new(AtomicU64::new(0)),
        };
        loop_.register_observability();
        loop_
    }

    fn register_observability(&self) {
        let fc = self.feedback_count.clone();
        let norm = self.adapter_norm();
        let pending_count = self.pending.len();
        observability::register("instant", move || {
            json!({
                "adapter_norm": norm,
                "feedback_count": fc.load(Ordering::Relaxed),
                "pending_count": pending_count,
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
        if embedding.len() != IN || logits.len() != self.n_targets { return; }
        let mut proj = [0f32; RANK];
        for r in 0..RANK {
            let off = r * IN;
            proj[r] = crate::simd::dot(&self.adapter_a[off..off + IN], embedding);
        }
        for k in 0..self.n_targets {
            let mut s = 0f32;
            for r in 0..RANK { s += self.adapter_b[r * self.n_targets + k] * proj[r]; }
            logits[k] += s;
        }
    }

    pub fn reset_adapter(&mut self) {
        self.adapter_a.fill(0.0);
        self.adapter_b.fill(0.0);
        self.lr = LR0;
    }

    pub fn serialize_adapter(&self) -> Vec<u8> {
        let mut flat = Vec::with_capacity(self.adapter_a.len() + self.adapter_b.len());
        flat.extend_from_slice(&self.adapter_a);
        flat.extend_from_slice(&self.adapter_b);
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
        self.lr *= DECAY;
    }

    fn gc_pending(&mut self) {
        let now = Instant::now();
        self.pending.retain(|_, p| now.duration_since(p.created_at) < FEEDBACK_WINDOW);
    }

    pub async fn record_trajectory(&mut self, session_id: Option<String>, embedding: Vec<f32>, route_model: String, response: String) -> Result<RequestId> {
        if embedding.len() != IN { return Err(anyhow!("embedding must be {IN}-d")); }
        self.gc_pending();
        let request_id = Uuid::new_v4().to_string();
        let created_ms = now_ms();
        let decision = json!({ "model": route_model.clone() }).to_string();
        let row = TrajectoryRow {
            id: request_id.clone(),
            session_id: session_id.clone(),
            query: None,
            query_embedding: Some(embedding.clone()),
            retrieved_ids: None,
            router_decision: Some(decision),
            response: Some(response),
            activations: None,
            quality: None,
            latency_ms: Some(0),
            created_at: Some(created_ms),
        };
        match self.spine.as_ref() {
            Some(s) => s.send(row).await?,
            None => self.store.insert_trajectory(&row).await?,
        }
        self.pending.insert(request_id.clone(), PendingInfo {
            session_id, embedding, route_model,
            created_at: Instant::now(), created_ms,
        });
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
        if payload.quality > 0.7 {
            if let Some(idx) = self.target_index(&model) {
                self.hebbian_update(&emb, idx, payload.quality);
            }
        }
        self.pending.remove(request_id);
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
        let rid = il.record_trajectory(Some("s1".into()), emb, "a".into(), "hello".into()).await.unwrap();
        assert_eq!(il.adapter_norm(), 0.0);
        il.feedback(&rid, FeedbackPayload { quality: 1.0, signal: None }).await.unwrap();
        assert!(il.adapter_norm() > 0.0, "adapter_norm must grow after positive feedback");
        let mut logits = vec![0f32; 2];
        il.apply_adapter(&vec![0.05f32; IN], &mut logits);
        assert!(logits[0].abs() + logits[1].abs() > 0.0);
    }
}
