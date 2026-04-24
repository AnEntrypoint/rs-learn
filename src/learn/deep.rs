use crate::observability;
use crate::store::Store;
use anyhow::Result;
use serde_json::json;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const FISHER_DECAY: f32 = 0.999;
const DEFAULT_LAMBDA: f32 = 2000.0;
const RING_CAP: usize = 10;
const BOUNDARY_Z: f32 = 2.5;
const MIN_STDDEV: f32 = 1e-4;

pub struct DeepLoop {
    pub store: Arc<Store>,
    pub lambda: f32,
    pub loss_ring: VecDeque<f32>,
    pub ring_cap: usize,
    pub fisher: HashMap<String, Vec<f32>>,
    pub params_snapshot: HashMap<String, Vec<f32>>,
    boundaries_detected: Arc<AtomicU64>,
    ring_shared: Arc<Mutex<Vec<f32>>>,
}

impl DeepLoop {
    pub fn new(store: Arc<Store>) -> Self {
        let lambda = std::env::var("RS_LEARN_EWC_LAMBDA").ok()
            .and_then(|s| s.parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(DEFAULT_LAMBDA);
        let boundaries_detected = Arc::new(AtomicU64::new(0));
        let ring_shared = Arc::new(Mutex::new(Vec::new()));
        let this = Self {
            store, lambda,
            loss_ring: VecDeque::with_capacity(RING_CAP),
            ring_cap: RING_CAP,
            fisher: HashMap::new(),
            params_snapshot: HashMap::new(),
            boundaries_detected: boundaries_detected.clone(),
            ring_shared: ring_shared.clone(),
        };
        let lam = lambda;
        observability::register("deep", move || {
            let ring = ring_shared.lock().map(|r| r.clone()).unwrap_or_default();
            let n = ring.len();
            let (mean, stddev) = if n == 0 {
                (0.0f32, 0.0f32)
            } else {
                let m = ring.iter().sum::<f32>() / n as f32;
                let v = ring.iter().map(|x| (x - m).powi(2)).sum::<f32>() / n as f32;
                (m, v.sqrt())
            };
            json!({
                "boundary_fires": boundaries_detected.load(Ordering::Relaxed),
                "window_mean": mean,
                "window_stddev": stddev,
                "threshold": BOUNDARY_Z,
                "samples_in_window": n,
                "lambda": lam,
                "loss_ring": ring,
            })
        });
        this
    }

    pub async fn consolidate(&mut self, param_id: &str, params: &[f32], grads: &[f32]) -> Result<()> {
        if params.len() != grads.len() {
            return Err(anyhow::anyhow!("consolidate: params/grads length mismatch"));
        }
        let prev = self.fisher.entry(param_id.to_string())
            .or_insert_with(|| vec![0f32; params.len()]);
        if prev.len() != params.len() { prev.resize(params.len(), 0.0); }
        for i in 0..params.len() {
            let g2 = grads[i] * grads[i];
            prev[i] = FISHER_DECAY * prev[i] + (1.0 - FISHER_DECAY) * g2;
        }
        let snapshot = prev.clone();
        self.store.save_fisher_vec(param_id, &snapshot).await?;
        self.store.save_params_snapshot_vec(param_id, params).await?;
        self.params_snapshot.insert(param_id.to_string(), params.to_vec());
        Ok(())
    }

    pub async fn record_loss(&mut self, loss: f32) -> Result<bool> {
        let prior: Vec<f32> = self.loss_ring.iter().copied().collect();
        if self.loss_ring.len() >= self.ring_cap { self.loss_ring.pop_front(); }
        self.loss_ring.push_back(loss);
        if let Ok(mut r) = self.ring_shared.lock() { *r = self.loss_ring.iter().copied().collect(); }
        if prior.len() < 3 { return Ok(false); }
        let n = prior.len() as f32;
        let mean = prior.iter().sum::<f32>() / n;
        let var = prior.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let stddev = var.sqrt().max(MIN_STDDEV);
        let z = (loss - mean).abs() / stddev;
        if z > BOUNDARY_Z {
            self.boundaries_detected.fetch_add(1, Ordering::Relaxed);
            return Ok(true);
        }
        Ok(false)
    }

    pub fn ewc_state(&self, param_id: &str) -> Option<(Vec<f32>, Vec<f32>, f32)> {
        let f = self.fisher.get(param_id)?.clone();
        let s = self.params_snapshot.get(param_id)?.clone();
        if f.len() != s.len() || f.is_empty() { return None; }
        Some((f, s, self.lambda))
    }

    pub fn ewc_penalty(&self, param_id: &str, params: &[f32]) -> f32 {
        let Some(f) = self.fisher.get(param_id) else { return 0.0 };
        let Some(snap) = self.params_snapshot.get(param_id) else { return 0.0 };
        let n = params.len().min(f.len()).min(snap.len());
        let mut sum = 0f32;
        for i in 0..n {
            let d = params[i] - snap[i];
            sum += f[i] * d * d;
        }
        self.lambda * sum
    }

    pub async fn load_fisher(&mut self, param_id: &str) -> Result<()> {
        let fisher_vec = self.store.load_fisher_vec(param_id).await?;
        let snap_vec = self.store.load_params_snapshot_vec(param_id).await?;
        self.fisher.insert(param_id.to_string(), fisher_vec);
        if !snap_vec.is_empty() {
            self.params_snapshot.insert(param_id.to_string(), snap_vec);
        }
        Ok(())
    }
}

impl Drop for DeepLoop {
    fn drop(&mut self) { observability::unregister("deep"); }
}

#[cfg(test)]
#[path = "deep_tests.rs"]
mod tests;
