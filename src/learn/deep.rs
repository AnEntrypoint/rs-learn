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
const BOUNDARY_DELTA: f32 = 0.5;

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
            json!({
                "lambda": lam,
                "loss_ring": ring,
                "boundaries_detected": boundaries_detected.load(Ordering::Relaxed),
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
        self.params_snapshot.insert(param_id.to_string(), params.to_vec());
        Ok(())
    }

    pub async fn record_loss(&mut self, loss: f32) -> Result<bool> {
        if self.loss_ring.len() >= self.ring_cap { self.loss_ring.pop_front(); }
        self.loss_ring.push_back(loss);
        if let Ok(mut r) = self.ring_shared.lock() { *r = self.loss_ring.iter().copied().collect(); }
        if self.loss_ring.len() < 2 { return Ok(false); }
        let mean = self.loss_ring.iter().sum::<f32>() / self.loss_ring.len() as f32;
        let delta = (loss - mean).abs();
        if delta > BOUNDARY_DELTA {
            self.boundaries_detected.fetch_add(1, Ordering::Relaxed);
            return Ok(true);
        }
        Ok(false)
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
        let v = self.store.load_fisher_vec(param_id).await?;
        self.fisher.insert(param_id.to_string(), v);
        Ok(())
    }
}

impl Drop for DeepLoop {
    fn drop(&mut self) { observability::unregister("deep"); }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    static ENV_GUARD: Mutex<()> = Mutex::new(());

    async fn tmp_store() -> Arc<Store> {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap().to_string();
        drop(tmp);
        Arc::new(Store::open(&path).await.unwrap())
    }

    #[tokio::test]
    async fn env_lambda_override() {
        let _g = ENV_GUARD.lock().unwrap_or_else(|p| p.into_inner());
        unsafe { std::env::set_var("RS_LEARN_EWC_LAMBDA", "42.5"); }
        let store = tmp_store().await;
        let dl = DeepLoop::new(store);
        assert_eq!(dl.lambda, 42.5);
        unsafe { std::env::remove_var("RS_LEARN_EWC_LAMBDA"); }
        let store2 = tmp_store().await;
        let dl2 = DeepLoop::new(store2);
        assert_eq!(dl2.lambda, DEFAULT_LAMBDA);
    }

    #[tokio::test]
    async fn consolidate_writes_fisher() {
        let _g = ENV_GUARD.lock().unwrap_or_else(|p| p.into_inner());
        unsafe { std::env::remove_var("RS_LEARN_EWC_LAMBDA"); }
        let store = tmp_store().await;
        let mut dl = DeepLoop::new(store.clone());
        let params = vec![0.1f32, 0.2, 0.3];
        let grads = vec![1.0f32, 2.0, 3.0];
        dl.consolidate("layer0", &params, &grads).await.unwrap();
        let loaded = store.load_fisher_vec("layer0").await.unwrap();
        assert_eq!(loaded.len(), 3);
        assert!(loaded[0] > 0.0 && loaded[1] > loaded[0] && loaded[2] > loaded[1]);
        let pen = dl.ewc_penalty("layer0", &vec![0.2f32, 0.2, 0.3]);
        assert!(pen > 0.0);
    }

    #[tokio::test]
    async fn record_loss_triggers_boundary() {
        let _g = ENV_GUARD.lock().unwrap_or_else(|p| p.into_inner());
        unsafe { std::env::remove_var("RS_LEARN_EWC_LAMBDA"); }
        let store = tmp_store().await;
        let mut dl = DeepLoop::new(store);
        for _ in 0..5 { let _ = dl.record_loss(0.1).await.unwrap(); }
        let boundary = dl.record_loss(1.0).await.unwrap();
        assert!(boundary, "delta>0.5 vs ring mean should trigger boundary");
    }
}
