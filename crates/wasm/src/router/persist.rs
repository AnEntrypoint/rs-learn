#![cfg(target_arch = "wasm32")]

use crate::errors::Result;
use crate::router::core::{Router, Weights, Heads, init_weights, init_heads, RouterConfig};
use crate::wasm_host;
use serde::{Deserialize, Serialize};

const NS: &str = "rs-learn/router";

#[derive(Serialize, Deserialize)]
struct RouterBlob {
    in_dim: usize,
    targets: Vec<String>,
    epsilon: f32,
    threshold: u64,
    version: i64,
    trained: bool,
    trajectory_count: u64,
    inference_count: u64,
    per_target_counts: Vec<u64>,
    per_target_quality_milli: Vec<u64>,
    v: Vec<f32>, u: Vec<f32>, uh: Vec<f32>, bh: Vec<f32>, bz: Vec<f32>,
    h_model: Vec<f32>, h_model_b: Vec<f32>,
    h_ctx: Vec<f32>, h_ctx_b: Vec<f32>,
    h_temp: Vec<f32>, h_temp_b: Vec<f32>,
    h_top_p: Vec<f32>, h_top_p_b: Vec<f32>,
    h_conf: Vec<f32>, h_conf_b: Vec<f32>,
}

pub fn save_router(r: &Router, key: &str) -> Result<()> {
    let blob = RouterBlob {
        in_dim: r.config.in_dim,
        targets: r.config.targets.clone(),
        epsilon: r.config.epsilon,
        threshold: r.config.threshold,
        version: r.version,
        trained: r.trained,
        trajectory_count: r.trajectory_count,
        inference_count: r.inference_count,
        per_target_counts: r.per_target_counts.clone(),
        per_target_quality_milli: r.per_target_quality_milli.clone(),
        v: r.w.v.clone(), u: r.w.u.clone(), uh: r.w.uh.clone(),
        bh: r.w.bh.clone(), bz: r.w.bz.clone(),
        h_model: r.heads.model.clone(), h_model_b: r.heads.model_b.clone(),
        h_ctx: r.heads.ctx.clone(),     h_ctx_b: r.heads.ctx_b.clone(),
        h_temp: r.heads.temp.clone(),   h_temp_b: r.heads.temp_b.clone(),
        h_top_p: r.heads.top_p.clone(), h_top_p_b: r.heads.top_p_b.clone(),
        h_conf: r.heads.conf.clone(),   h_conf_b: r.heads.conf_b.clone(),
    };
    let json = serde_json::to_vec(&blob)?;
    wasm_host::kv_put(NS, key, &json)
}

pub fn load_router(key: &str) -> Result<Option<Router>> {
    let bytes = wasm_host::kv_get(NS, key)?;
    if bytes.is_empty() { return Ok(None); }
    let blob: RouterBlob = serde_json::from_slice(&bytes)?;
    let cfg = RouterConfig {
        in_dim: blob.in_dim,
        targets: blob.targets.clone(),
        threshold: blob.threshold,
        epsilon: blob.epsilon,
    };
    if cfg.targets.is_empty() {
        return Err(LlmError::Process("router blob has no targets".into()));
    }
    let mut r = Router::new(cfg).map_err(|e| crate::errors::LlmError::Process(e))?;
    let w = Weights {
        v: blob.v, u: blob.u, uh: blob.uh, bh: blob.bh, bz: blob.bz,
    };
    let h = Heads {
        model: blob.h_model, model_b: blob.h_model_b,
        ctx: blob.h_ctx, ctx_b: blob.h_ctx_b,
        temp: blob.h_temp, temp_b: blob.h_temp_b,
        top_p: blob.h_top_p, top_p_b: blob.h_top_p_b,
        conf: blob.h_conf, conf_b: blob.h_conf_b,
    };
    let weights_match = w.v.len() == r.w.v.len()
        && w.u.len() == r.w.u.len()
        && w.uh.len() == r.w.uh.len()
        && h.model.len() == r.heads.model.len();
    if weights_match {
        r.w = w;
        r.heads = h;
    }
    r.version = blob.version;
    r.trained = blob.trained && weights_match;
    r.trajectory_count = blob.trajectory_count;
    r.inference_count = blob.inference_count;
    r.per_target_counts = blob.per_target_counts;
    r.per_target_quality_milli = blob.per_target_quality_milli;
    Ok(Some(r))
}
