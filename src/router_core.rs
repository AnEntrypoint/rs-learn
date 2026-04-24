use super::*;

pub(super) struct Weights { pub v: Vec<f32>, pub u: Vec<f32>, pub uh: Vec<f32>, pub bh: Vec<f32>, pub bz: Vec<f32> }
pub(super) struct Heads {
    pub model: Vec<f32>, pub model_b: Vec<f32>,
    pub ctx: Vec<f32>, pub ctx_b: Vec<f32>,
    pub temp: Vec<f32>, pub temp_b: Vec<f32>,
    pub top_p: Vec<f32>, pub top_p_b: Vec<f32>,
    pub conf: Vec<f32>, pub conf_b: Vec<f32>,
}

pub(super) fn mulberry32(mut a: u32) -> impl FnMut() -> f32 {
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

pub(super) fn init_weights() -> Weights {
    let mut rnd = mulberry32(SEED);
    let (sv, su, sh) = (1.0 / (IN as f32).sqrt(), 1.0 / (RANK as f32).sqrt(), 1.0 / (DIM as f32).sqrt());
    let mut v = vec![0f32; RANK * IN]; for x in v.iter_mut() { *x = randn(&mut rnd) * sv; }
    let mut u = vec![0f32; DIM * RANK]; for x in u.iter_mut() { *x = randn(&mut rnd) * su; }
    let mut uh = vec![0f32; DIM * DIM]; for x in uh.iter_mut() { *x = randn(&mut rnd) * sh; }
    let mut mrnd = mulberry32(SEED ^ 0x9E37);
    for i in 0..u.len() { if mrnd() < SPARSITY { u[i] = 0.0; } }
    Weights { v, u, uh, bh: vec![0f32; DIM], bz: vec![0f32; DIM] }
}

pub(super) fn init_heads(n_targets: usize) -> Heads {
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

#[inline] pub(super) fn sig(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

pub(super) struct Fwd { pub h: Vec<f32>, pub ml: Vec<f32>, pub cl: Vec<f32>, pub tp: f32, pub top_p: f32, pub conf: f32 }

pub(super) fn forward(w: &Weights, hd: &Heads, x: &[f32], n_targets: usize) -> Fwd {
    let mut proj = vec![0f32; RANK];
    crate::simd::matvec(&w.v, RANK, IN, x, &mut proj);
    let mut wx = vec![0f32; DIM];
    crate::simd::matvec(&w.u, DIM, RANK, &proj, &mut wx);
    let mut h = vec![0f32; DIM];
    for d in 0..DIM { let pre = wx[d] + w.bh[d]; let z = sig(pre + w.bz[d]); h[d] = (1.0 - z) * pre.tanh(); }
    let head = |wm: &[f32], b: &[f32], n: usize| {
        let mut o = b.to_vec();
        for k in 0..n { o[k] += crate::simd::dot(&wm[k * DIM..(k + 1) * DIM], &h); }
        o
    };
    let ml = head(&hd.model, &hd.model_b, n_targets);
    let cl = head(&hd.ctx, &hd.ctx_b, CTX_BUCKETS);
    let tp = head(&hd.temp, &hd.temp_b, 1)[0];
    let tpp = head(&hd.top_p, &hd.top_p_b, 1)[0];
    let cf = head(&hd.conf, &hd.conf_b, 1)[0];
    Fwd { h, ml, cl, tp: 0.1 + sig(tp) * 1.4, top_p: 0.5 + sig(tpp) * 0.5, conf: sig(cf) }
}

pub(super) fn softmax_argmax(a: &[f32]) -> (usize, f32) {
    let (mut mi, mut m) = (0usize, f32::NEG_INFINITY);
    for (i, &v) in a.iter().enumerate() { if v > m { m = v; mi = i; } }
    let mut s = 0.0; for &v in a { s += (v - m).exp(); }
    (mi, (a[mi] - m).exp() / s)
}

pub(super) fn softmax(a: &[f32]) -> Vec<f32> {
    let m = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = a.iter().map(|&v| (v - m).exp()).collect();
    let s: f32 = out.iter().sum();
    if s > 0.0 { for x in out.iter_mut() { *x /= s; } }
    out
}

pub(super) fn bucket_for_tokens(n: u64) -> u8 {
    for (i, &cap) in BUCKET_CAPS.iter().enumerate() { if n <= cap { return i as u8; } }
    4
}
