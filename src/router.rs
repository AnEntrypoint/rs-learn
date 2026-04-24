use crate::observability;
use crate::store::Store;
use anyhow::{anyhow, Result};
use bytemuck::{cast_slice, cast_slice_mut};
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub const IN: usize = 768;
pub const DIM: usize = 128;
pub const RANK: usize = 8;
pub const SPARSITY: f32 = 0.9;
pub const CTX_BUCKETS: usize = 5;
const SEED: u32 = 0xC0A5;
const BUCKET_CAPS: [u64; 5] = [1000, 4000, 16000, 64000, u64::MAX];

#[derive(Debug, Clone)]
pub struct Route {
    pub model: String,
    pub context_bucket: u8,
    pub temperature: f32,
    pub top_p: f32,
    pub confidence: f32,
    pub algo: &'static str,
    pub exploration: bool,
}

#[derive(Debug, Default, Clone)]
pub struct RouteCtx {
    pub task_type: Option<String>,
    pub estimated_tokens: u64,
}

#[derive(Debug, Clone)]
pub struct TrainSample {
    pub embedding: Vec<f32>,
    pub chosen_target: String,
    pub quality: f32,
    pub estimated_tokens: u64,
}

#[path = "router_core.rs"]
mod router_core;
use router_core::*;

pub struct Router {
    store: Arc<Store>,
    targets: Vec<String>,
    w: Weights,
    heads: Heads,
    version: i64,
    trained: bool,
    threshold: u64,
    trajectory_count: u64,
    inference_count: Arc<AtomicU64>,
    total_us: Arc<AtomicU64>,
    threshold_obs: Arc<AtomicU64>,
    traj_count_obs: Arc<AtomicU64>,
    per_target_counts: Vec<Arc<AtomicU64>>,
    per_target_quality_milli: Vec<Arc<AtomicU64>>,
}

impl Router {
    pub fn new(store: Arc<Store>, targets: Vec<String>) -> Self {
        if targets.is_empty() { panic!("router: targets required"); }
        let threshold = std::env::var("RS_LEARN_ROUTER_THRESHOLD").ok()
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|&n| n >= 1 && n <= 10_000)
            .unwrap_or(200);
        let nt = targets.len();
        let per_target_counts: Vec<Arc<AtomicU64>> = (0..nt).map(|_| Arc::new(AtomicU64::new(0))).collect();
        let per_target_quality_milli: Vec<Arc<AtomicU64>> = (0..nt).map(|_| Arc::new(AtomicU64::new(500))).collect();
        let r = Self {
            store, w: init_weights(), heads: init_heads(nt),
            targets: targets.clone(), version: 0, trained: false, threshold,
            trajectory_count: 0,
            inference_count: Arc::new(AtomicU64::new(0)),
            total_us: Arc::new(AtomicU64::new(0)),
            threshold_obs: Arc::new(AtomicU64::new(threshold)),
            traj_count_obs: Arc::new(AtomicU64::new(0)),
            per_target_counts: per_target_counts.clone(),
            per_target_quality_milli: per_target_quality_milli.clone(),
        };
        let ic = r.inference_count.clone();
        let tu = r.total_us.clone();
        let thr = r.threshold_obs.clone();
        let tc = r.traj_count_obs.clone();
        let tnames = targets;
        observability::register("router", move || {
            let n = ic.load(Ordering::Relaxed);
            let t = tu.load(Ordering::Relaxed);
            let counts: Vec<u64> = per_target_counts.iter().map(|a| a.load(Ordering::Relaxed)).collect();
            let qualities: Vec<f64> = per_target_quality_milli.iter().map(|a| a.load(Ordering::Relaxed) as f64 / 1000.0).collect();
            let mut per_target = serde_json::Map::new();
            for (i, name) in tnames.iter().enumerate() {
                per_target.insert(name.clone(), json!({ "count": counts[i], "avg_quality": qualities[i] }));
            }
            json!({ "inferenceCount": n, "totalUs": t, "avgUs": if n>0 { t as f64 / n as f64 } else { 0.0 }, "targets": tnames.clone(), "threshold": thr.load(Ordering::Relaxed), "trajectoryCount": tc.load(Ordering::Relaxed), "per_target": per_target })
        });
        r
    }

    pub fn record_outcome(&self, target: &str, quality: f32) {
        let Some(idx) = self.targets.iter().position(|t| t == target) else { return };
        let q = quality.clamp(0.0, 1.0) as f64;
        let prior = self.per_target_quality_milli[idx].load(Ordering::Relaxed) as f64 / 1000.0;
        let alpha = 0.1;
        let new_q = (1.0 - alpha) * prior + alpha * q;
        self.per_target_quality_milli[idx].store((new_q * 1000.0) as u64, Ordering::Relaxed);
        self.per_target_counts[idx].fetch_add(1, Ordering::Relaxed);
    }

    pub fn route(&self, emb: &[f32], ctx: &RouteCtx) -> Route {
        self.route_with_adapter(emb, ctx, |_, _| {})
    }

    pub fn route_with_adapter<F: Fn(&[f32], &mut [f32])>(&self, emb: &[f32], ctx: &RouteCtx, adapter: F) -> Route {
        let t0 = std::time::Instant::now();
        let out = if !self.trained {
            Route { model: self.targets[0].clone(), context_bucket: bucket_for_tokens(ctx.estimated_tokens),
                    temperature: 0.7, top_p: 0.9, confidence: 0.5, algo: "rule", exploration: false }
        } else {
            let f = forward(&self.w, &self.heads, emb, self.targets.len());
            let mut ml = f.ml.clone();
            adapter(emb, &mut ml);
            let (argmax_idx, p) = softmax_argmax(&ml);
            let (cb, _) = softmax_argmax(&f.cl);
            let nt = self.targets.len();
            let eps = std::env::var("RS_LEARN_ROUTER_EPSILON").ok()
                .and_then(|v| v.parse::<f32>().ok())
                .map(|v| v.clamp(0.0, 1.0))
                .unwrap_or(0.0);
            let (idx, exploration) = if nt > 1 && eps > 0.0 {
                let seed = std::env::var("RS_LEARN_ROUTER_SEED").ok()
                    .and_then(|v| v.parse::<u32>().ok())
                    .unwrap_or_else(|| {
                        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_nanos() as u32).unwrap_or(0)
                            ^ self.inference_count.load(Ordering::Relaxed) as u32
                    });
                let mut rng = mulberry32(seed ^ SEED);
                if rng() < eps {
                    let mut alt = (rng() * (nt as f32 - 1.0)) as usize;
                    if alt >= argmax_idx { alt += 1; }
                    if alt >= nt { alt = nt - 1; }
                    (alt, true)
                } else { (argmax_idx, false) }
            } else { (argmax_idx, false) };
            Route { model: self.targets[idx].clone(), context_bucket: cb as u8,
                    temperature: f.tp, top_p: f.top_p, confidence: f.conf * p, algo: "fastgrnn",
                    exploration }
        };
        self.inference_count.fetch_add(1, Ordering::Relaxed);
        self.total_us.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
        out
    }

}

#[path = "router_persist.rs"]
mod router_persist;
#[path = "router_train.rs"]
mod router_train;

impl Drop for Router {
    fn drop(&mut self) { observability::unregister("router"); }
}

#[cfg(test)] #[path = "router_tests.rs"] mod tests;
