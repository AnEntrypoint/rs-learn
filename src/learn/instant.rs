#[path = "instant_io.rs"]
mod instant_io;

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
    pub retrieved_strategies: Vec<String>,
    pub dominant_relation: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FeedbackPayload {
    pub quality: f32,
    #[serde(default)]
    pub signal: Option<String>,
}

pub struct EwcState {
    pub fisher: Vec<f32>,
    pub snapshot: Vec<f32>,
    pub lambda: f32,
}

pub struct InstantLoop {
    pub store: Arc<Store>,
    pub router: Arc<Mutex<Router>>,
    pub adapter_a: Vec<f32>,
    pub adapter_b: Vec<f32>,
    pub pending: HashMap<RequestId, PendingInfo>,
    pub spine: Option<Arc<crate::spine::TrajectorySpine>>,
    pub ewc: Option<EwcState>,
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
            ewc: None,
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

    pub fn set_ewc_state(&mut self, fisher: Vec<f32>, snapshot: Vec<f32>, lambda: f32) {
        let expected = self.adapter_a.len() + self.adapter_b.len();
        if fisher.len() != expected || snapshot.len() != expected || !(lambda.is_finite() && lambda >= 0.0) { return; }
        self.ewc = Some(EwcState { fisher, snapshot, lambda });
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

}

impl Drop for InstantLoop {
    fn drop(&mut self) { observability::unregister("instant"); }
}

#[cfg(test)]
#[path = "instant_tests.rs"]
mod tests;
