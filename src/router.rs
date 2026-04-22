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

struct Weights { v: Vec<f32>, u: Vec<f32>, uh: Vec<f32>, bh: Vec<f32>, bz: Vec<f32> }
struct Heads {
    model: Vec<f32>, model_b: Vec<f32>,
    ctx: Vec<f32>, ctx_b: Vec<f32>,
    temp: Vec<f32>, temp_b: Vec<f32>,
    top_p: Vec<f32>, top_p_b: Vec<f32>,
    conf: Vec<f32>, conf_b: Vec<f32>,
}

fn mulberry32(mut a: u32) -> impl FnMut() -> f32 {
    move || {
        a = a.wrapping_add(0x6D2B_79F5);
        let mut t = a;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        ((t ^ (t >> 14)) as f32) / 4_294_967_296.0
    }
}

fn randn(rnd: &mut dyn FnMut() -> f32) -> f32 {
    let u = rnd().max(1e-9);
    let v = rnd();
    (-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos()
}

fn init_weights() -> Weights {
    let mut rnd = mulberry32(SEED);
    let (sv, su, sh) = (1.0 / (IN as f32).sqrt(), 1.0 / (RANK as f32).sqrt(), 1.0 / (DIM as f32).sqrt());
    let mut v = vec![0f32; RANK * IN]; for x in v.iter_mut() { *x = randn(&mut rnd) * sv; }
    let mut u = vec![0f32; DIM * RANK]; for x in u.iter_mut() { *x = randn(&mut rnd) * su; }
    let mut uh = vec![0f32; DIM * DIM]; for x in uh.iter_mut() { *x = randn(&mut rnd) * sh; }
    let mut mrnd = mulberry32(SEED ^ 0x9E37);
    for i in 0..u.len() { if mrnd() < SPARSITY { u[i] = 0.0; } }
    Weights { v, u, uh, bh: vec![0f32; DIM], bz: vec![0f32; DIM] }
}

fn init_heads(n_targets: usize) -> Heads {
    let mut rnd = mulberry32(SEED ^ 0xA17C);
    let s = 1.0 / (DIM as f32).sqrt();
    let mut mk = |n: usize| {
        let mut a = vec![0f32; n * DIM];
        for x in a.iter_mut() { *x = randn(&mut rnd) * s; }
        a
    };
    let model = mk(n_targets);
    let ctx = mk(CTX_BUCKETS);
    let temp = mk(1);
    let top_p = mk(1);
    let conf = mk(1);
    Heads { model, model_b: vec![0f32; n_targets], ctx, ctx_b: vec![0f32; CTX_BUCKETS],
            temp, temp_b: vec![0f32; 1], top_p, top_p_b: vec![0f32; 1], conf, conf_b: vec![0f32; 1] }
}

#[inline] fn sig(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

struct Fwd { h: Vec<f32>, ml: Vec<f32>, cl: Vec<f32>, tp: f32, top_p: f32, conf: f32 }

fn forward(w: &Weights, hd: &Heads, x: &[f32], n_targets: usize) -> Fwd {
    let mut proj = vec![0f32; RANK];
    for r in 0..RANK { let off = r * IN; let mut s = 0.0; for i in 0..IN { s += w.v[off + i] * x[i]; } proj[r] = s; }
    let mut wx = vec![0f32; DIM];
    for d in 0..DIM { let off = d * RANK; let mut s = 0.0; for r in 0..RANK { s += w.u[off + r] * proj[r]; } wx[d] = s; }
    let mut h = vec![0f32; DIM];
    for d in 0..DIM { let pre = wx[d] + w.bh[d]; let z = sig(pre + w.bz[d]); h[d] = (1.0 - z) * pre.tanh(); }
    let head = |wm: &[f32], b: &[f32], n: usize| {
        let mut o = vec![0f32; n];
        for k in 0..n { let off = k * DIM; let mut s = b[k]; for d in 0..DIM { s += wm[off + d] * h[d]; } o[k] = s; }
        o
    };
    let ml = head(&hd.model, &hd.model_b, n_targets);
    let cl = head(&hd.ctx, &hd.ctx_b, CTX_BUCKETS);
    let tp = head(&hd.temp, &hd.temp_b, 1)[0];
    let tpp = head(&hd.top_p, &hd.top_p_b, 1)[0];
    let cf = head(&hd.conf, &hd.conf_b, 1)[0];
    Fwd { h, ml, cl, tp: 0.1 + sig(tp) * 1.4, top_p: 0.5 + sig(tpp) * 0.5, conf: sig(cf) }
}

fn softmax_argmax(a: &[f32]) -> (usize, f32) {
    let (mut mi, mut m) = (0usize, f32::NEG_INFINITY);
    for (i, &v) in a.iter().enumerate() { if v > m { m = v; mi = i; } }
    let mut s = 0.0; for &v in a { s += (v - m).exp(); }
    (mi, (a[mi] - m).exp() / s)
}

fn softmax(a: &[f32]) -> Vec<f32> {
    let m = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = a.iter().map(|&v| (v - m).exp()).collect();
    let s: f32 = out.iter().sum();
    if s > 0.0 { for x in out.iter_mut() { *x /= s; } }
    out
}

fn bucket_for_tokens(n: u64) -> u8 {
    for (i, &cap) in BUCKET_CAPS.iter().enumerate() { if n <= cap { return i as u8; } }
    4
}

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
}

impl Router {
    pub fn new(store: Arc<Store>, targets: Vec<String>) -> Self {
        if targets.is_empty() { panic!("router: targets required"); }
        let r = Self {
            store, w: init_weights(), heads: init_heads(targets.len()),
            targets: targets.clone(), version: 0, trained: false, threshold: 200,
            trajectory_count: 0,
            inference_count: Arc::new(AtomicU64::new(0)),
            total_us: Arc::new(AtomicU64::new(0)),
        };
        let ic = r.inference_count.clone();
        let tu = r.total_us.clone();
        let tnames = targets;
        observability::register("router", move || {
            let n = ic.load(Ordering::Relaxed);
            let t = tu.load(Ordering::Relaxed);
            json!({ "inferenceCount": n, "totalUs": t, "avgUs": if n>0 { t as f64 / n as f64 } else { 0.0 }, "targets": tnames.clone() })
        });
        r
    }

    pub fn route(&self, emb: &[f32], ctx: &RouteCtx) -> Route {
        self.route_with_adapter(emb, ctx, |_, _| {})
    }

    pub fn route_with_adapter<F: Fn(&[f32], &mut [f32])>(&self, emb: &[f32], ctx: &RouteCtx, adapter: F) -> Route {
        let t0 = std::time::Instant::now();
        let out = if !self.trained {
            Route { model: self.targets[0].clone(), context_bucket: bucket_for_tokens(ctx.estimated_tokens),
                    temperature: 0.7, top_p: 0.9, confidence: 0.5, algo: "rule" }
        } else {
            let f = forward(&self.w, &self.heads, emb, self.targets.len());
            let mut ml = f.ml.clone();
            adapter(emb, &mut ml);
            let (idx, p) = softmax_argmax(&ml);
            let (cb, _) = softmax_argmax(&f.cl);
            Route { model: self.targets[idx].clone(), context_bucket: cb as u8,
                    temperature: f.tp, top_p: f.top_p, confidence: f.conf * p, algo: "fastgrnn" }
        };
        self.inference_count.fetch_add(1, Ordering::Relaxed);
        self.total_us.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
        out
    }

    pub fn train(&mut self, batch: &[TrainSample]) -> Result<usize> {
        let mut applied = 0usize;
        let nt = self.targets.len();
        let base_lr = 0.05f32;
        for tr in batch {
            if tr.quality <= 0.7 || tr.embedding.len() != IN { continue; }
            let Some(t_idx) = self.targets.iter().position(|t| t == &tr.chosen_target) else { continue };
            let f = forward(&self.w, &self.heads, &tr.embedding, nt);
            let lr = base_lr * tr.quality;
            let model_probs = softmax(&f.ml);
            for k in 0..nt {
                let err = model_probs[k] - if k == t_idx { 1.0 } else { 0.0 };
                let off = k * DIM;
                for d in 0..DIM { self.heads.model[off + d] -= lr * err * f.h[d]; }
                self.heads.model_b[k] -= lr * err;
            }
            let ctx_target = (bucket_for_tokens(tr.estimated_tokens) as usize).min(CTX_BUCKETS - 1);
            let ctx_probs = softmax(&f.cl);
            for k in 0..CTX_BUCKETS {
                let err = ctx_probs[k] - if k == ctx_target { 1.0 } else { 0.0 };
                let off = k * DIM;
                for d in 0..DIM { self.heads.ctx[off + d] -= lr * err * f.h[d]; }
                self.heads.ctx_b[k] -= lr * err;
            }
            applied += 1;
        }
        self.trajectory_count += applied as u64;
        if self.trajectory_count >= self.threshold { self.trained = true; }
        Ok(applied)
    }

    fn pack(&self) -> Vec<u8> {
        let parts: [&[f32]; 15] = [
            &self.w.v, &self.w.u, &self.w.uh, &self.w.bh, &self.w.bz,
            &self.heads.model, &self.heads.model_b, &self.heads.ctx, &self.heads.ctx_b,
            &self.heads.temp, &self.heads.temp_b, &self.heads.top_p, &self.heads.top_p_b,
            &self.heads.conf, &self.heads.conf_b,
        ];
        let n: usize = parts.iter().map(|p| p.len()).sum();
        let mut flat = Vec::with_capacity(n);
        for p in parts { flat.extend_from_slice(p); }
        cast_slice::<f32, u8>(&flat).to_vec()
    }

    fn unpack(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() % 4 != 0 { return Err(anyhow!("router: blob not f32-aligned")); }
        let mut flat = vec![0f32; bytes.len() / 4];
        cast_slice_mut::<f32, u8>(&mut flat).copy_from_slice(bytes);
        let nt = self.targets.len();
        let sizes = [RANK*IN, DIM*RANK, DIM*DIM, DIM, DIM, nt*DIM, nt, CTX_BUCKETS*DIM, CTX_BUCKETS, DIM, 1, DIM, 1, DIM, 1];
        let total: usize = sizes.iter().sum();
        if flat.len() != total { return Err(anyhow!("router: blob size mismatch want {} got {}", total, flat.len())); }
        let mut o = 0usize;
        let mut take = |n: usize| -> Vec<f32> { let s = flat[o..o+n].to_vec(); o += n; s };
        self.w = Weights { v: take(RANK*IN), u: take(DIM*RANK), uh: take(DIM*DIM), bh: take(DIM), bz: take(DIM) };
        self.heads = Heads {
            model: take(nt*DIM), model_b: take(nt),
            ctx: take(CTX_BUCKETS*DIM), ctx_b: take(CTX_BUCKETS),
            temp: take(DIM), temp_b: take(1),
            top_p: take(DIM), top_p_b: take(1),
            conf: take(DIM), conf_b: take(1),
        };
        self.trained = true;
        Ok(())
    }

    pub async fn save(&mut self) -> Result<i64> {
        self.version += 1;
        let blob = self.pack();
        let meta = json!({ "dim": DIM, "sparsity": SPARSITY, "rank": RANK, "inputDim": IN, "nTargets": self.targets.len() });
        self.store.save_router_weights(self.version, &blob, "fastgrnn", &meta).await?;
        self.trained = true;
        Ok(self.version)
    }

    pub async fn load(&mut self) -> Result<Option<i64>> {
        let Some(row) = self.store.load_latest_router_weights().await? else { return Ok(None) };
        self.unpack(&row.blob)?;
        self.version = row.version;
        Ok(Some(row.version))
    }
}

impl Drop for Router {
    fn drop(&mut self) { observability::unregister("router"); }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn roundtrip() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap().to_string();
        drop(tmp);
        let store = Arc::new(Store::open(&path).await.unwrap());
        let targets = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut r = Router::new(store.clone(), targets.clone());
        let emb = vec![0.01f32; IN];
        let out = r.route(&emb, &RouteCtx { task_type: Some("default".into()), estimated_tokens: 500 });
        assert_eq!(out.algo, "rule");
        assert_eq!(out.context_bucket, 0);
        r.trained = true;
        let blob1 = r.pack();
        r.save().await.unwrap();
        drop(r);
        let mut r2 = Router::new(store, targets);
        let v = r2.load().await.unwrap();
        assert_eq!(v, Some(1));
        let blob2 = r2.pack();
        assert_eq!(blob1, blob2, "round-trip bytes must match");
        let t0 = std::time::Instant::now();
        for _ in 0..100 { let _ = r2.route(&emb, &RouteCtx::default()); }
        let per_us = t0.elapsed().as_micros() / 100;
        assert!(per_us < 1000, "inference p50 {}us >= 1000us", per_us);
    }

    #[tokio::test]
    async fn train_softmax_shifts_prediction_toward_chosen() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_str().unwrap().to_string();
        drop(tmp);
        let store = Arc::new(Store::open(&path).await.unwrap());
        let targets = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut r = Router::new(store, targets.clone());
        let emb: Vec<f32> = (0..IN).map(|i| ((i as f32) * 0.01).sin()).collect();
        let before = forward(&r.w, &r.heads, &emb, targets.len());
        let samples: Vec<TrainSample> = (0..200).map(|_| TrainSample {
            embedding: emb.clone(), chosen_target: "b".into(), quality: 0.95, estimated_tokens: 500,
        }).collect();
        let applied = r.train(&samples).unwrap();
        assert_eq!(applied, 200);
        let after = forward(&r.w, &r.heads, &emb, targets.len());
        assert!(after.ml[1] - before.ml[1] > after.ml[0] - before.ml[0],
            "logit for chosen target must grow faster than non-chosen");
        assert!(after.ml[1] - before.ml[1] > after.ml[2] - before.ml[2]);
        let max_abs = r.heads.model.iter().map(|x| x.abs()).fold(0f32, f32::max);
        assert!(max_abs < 10.0, "weights should stay bounded under softmax training, got max {max_abs}");
    }
}
