pub const DIM: usize = 128;
pub const RANK: usize = 8;
pub const SPARSITY: f32 = 0.9;
pub const CTX_BUCKETS: usize = 5;
pub const SEED: u32 = 0xC0A5;
pub const BUCKET_CAPS: [u64; 5] = [1000, 4000, 16000, 64000, u64::MAX];

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
pub struct RouterConfig {
    pub in_dim: usize,
    pub targets: Vec<String>,
    pub threshold: u64,
    pub epsilon: f32,
}

impl RouterConfig {
    pub fn new(in_dim: usize, targets: Vec<String>) -> Self {
        Self { in_dim, targets, threshold: 200, epsilon: 0.0 }
    }
}

pub struct Weights {
    pub v: Vec<f32>,
    pub u: Vec<f32>,
    pub uh: Vec<f32>,
    pub bh: Vec<f32>,
    pub bz: Vec<f32>,
}

pub struct Heads {
    pub model: Vec<f32>, pub model_b: Vec<f32>,
    pub ctx: Vec<f32>,   pub ctx_b: Vec<f32>,
    pub temp: Vec<f32>,  pub temp_b: Vec<f32>,
    pub top_p: Vec<f32>, pub top_p_b: Vec<f32>,
    pub conf: Vec<f32>,  pub conf_b: Vec<f32>,
}

pub fn mulberry32(mut a: u32) -> impl FnMut() -> f32 {
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
    (-2.0 * u.ln()).sqrt() * (2.0 * core::f32::consts::PI * v).cos()
}

fn matvec(m: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    for r in 0..rows {
        let mut s = 0f32;
        let off = r * cols;
        for c in 0..cols { s += m[off + c] * x[c]; }
        out[r] = s;
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0f32;
    for i in 0..a.len() { s += a[i] * b[i]; }
    s
}

pub fn init_weights(in_dim: usize) -> Weights {
    let mut rnd = mulberry32(SEED);
    let (sv, su, sh) = (1.0 / (in_dim as f32).sqrt(), 1.0 / (RANK as f32).sqrt(), 1.0 / (DIM as f32).sqrt());
    let mut v = vec![0f32; RANK * in_dim]; for x in v.iter_mut() { *x = randn(&mut rnd) * sv; }
    let mut u = vec![0f32; DIM * RANK];     for x in u.iter_mut() { *x = randn(&mut rnd) * su; }
    let mut uh = vec![0f32; DIM * DIM];     for x in uh.iter_mut() { *x = randn(&mut rnd) * sh; }
    let mut mrnd = mulberry32(SEED ^ 0x9E37);
    for i in 0..u.len() { if mrnd() < SPARSITY { u[i] = 0.0; } }
    Weights { v, u, uh, bh: vec![0f32; DIM], bz: vec![0f32; DIM] }
}

pub fn init_heads(n_targets: usize) -> Heads {
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
    Heads {
        model, model_b: vec![0f32; n_targets],
        ctx, ctx_b: vec![0f32; CTX_BUCKETS],
        temp, temp_b: vec![0f32; 1],
        top_p, top_p_b: vec![0f32; 1],
        conf, conf_b: vec![0f32; 1],
    }
}

#[inline]
pub fn sig(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

pub struct Fwd {
    pub h: Vec<f32>,
    pub ml: Vec<f32>,
    pub cl: Vec<f32>,
    pub tp: f32,
    pub top_p: f32,
    pub conf: f32,
}

pub fn forward(w: &Weights, hd: &Heads, x: &[f32], in_dim: usize, n_targets: usize) -> Fwd {
    let mut proj = vec![0f32; RANK];
    matvec(&w.v, RANK, in_dim, x, &mut proj);
    let mut wx = vec![0f32; DIM];
    matvec(&w.u, DIM, RANK, &proj, &mut wx);
    let mut h = vec![0f32; DIM];
    for d in 0..DIM {
        let pre = wx[d] + w.bh[d];
        let z = sig(pre + w.bz[d]);
        h[d] = (1.0 - z) * pre.tanh();
    }
    let head = |wm: &[f32], b: &[f32], n: usize| {
        let mut o = b.to_vec();
        for k in 0..n { o[k] += dot(&wm[k * DIM..(k + 1) * DIM], &h); }
        o
    };
    let ml = head(&hd.model, &hd.model_b, n_targets);
    let cl = head(&hd.ctx, &hd.ctx_b, CTX_BUCKETS);
    let tp = head(&hd.temp, &hd.temp_b, 1)[0];
    let tpp = head(&hd.top_p, &hd.top_p_b, 1)[0];
    let cf = head(&hd.conf, &hd.conf_b, 1)[0];
    Fwd {
        h, ml, cl,
        tp: 0.1 + sig(tp) * 1.4,
        top_p: 0.5 + sig(tpp) * 0.5,
        conf: sig(cf),
    }
}

pub fn softmax_argmax(a: &[f32]) -> (usize, f32) {
    let (mut mi, mut m) = (0usize, f32::NEG_INFINITY);
    for (i, &v) in a.iter().enumerate() { if v > m { m = v; mi = i; } }
    let mut s = 0.0;
    for &v in a { s += (v - m).exp(); }
    (mi, (a[mi] - m).exp() / s)
}

pub fn bucket_for_tokens(n: u64) -> u8 {
    for (i, &cap) in BUCKET_CAPS.iter().enumerate() { if n <= cap { return i as u8; } }
    4
}

pub struct Router {
    pub config: RouterConfig,
    pub w: Weights,
    pub heads: Heads,
    pub version: i64,
    pub trained: bool,
    pub trajectory_count: u64,
    pub inference_count: u64,
    pub per_target_counts: Vec<u64>,
    pub per_target_quality_milli: Vec<u64>,
}

impl Router {
    pub fn new(config: RouterConfig) -> Result<Self, String> {
        if config.targets.is_empty() { return Err("router: targets required".into()); }
        let nt = config.targets.len();
        let w = init_weights(config.in_dim);
        let heads = init_heads(nt);
        Ok(Self {
            config,
            w, heads,
            version: 0,
            trained: false,
            trajectory_count: 0,
            inference_count: 0,
            per_target_counts: vec![0u64; nt],
            per_target_quality_milli: vec![500u64; nt],
        })
    }

    pub fn record_outcome(&mut self, target: &str, quality: f32) {
        let idx = match self.config.targets.iter().position(|t| t == target) {
            Some(i) => i, None => return,
        };
        let q = quality.clamp(0.0, 1.0) as f64;
        let prior = self.per_target_quality_milli[idx] as f64 / 1000.0;
        let alpha = 0.1;
        let new_q = (1.0 - alpha) * prior + alpha * q;
        self.per_target_quality_milli[idx] = (new_q * 1000.0) as u64;
        self.per_target_counts[idx] += 1;
    }

    pub fn route(&mut self, emb: &[f32], ctx: &RouteCtx) -> Route {
        self.route_with_adapter(emb, ctx, |_, _| {})
    }

    pub fn route_with_adapter<F: Fn(&[f32], &mut [f32])>(&mut self, emb: &[f32], ctx: &RouteCtx, adapter: F) -> Route {
        self.inference_count += 1;
        if !self.trained {
            return Route {
                model: self.config.targets[0].clone(),
                context_bucket: bucket_for_tokens(ctx.estimated_tokens),
                temperature: 0.7, top_p: 0.9, confidence: 0.5,
                algo: "rule", exploration: false,
            };
        }
        let f = forward(&self.w, &self.heads, emb, self.config.in_dim, self.config.targets.len());
        let mut ml = f.ml.clone();
        adapter(emb, &mut ml);
        let (argmax_idx, p) = softmax_argmax(&ml);
        let (cb, _) = softmax_argmax(&f.cl);
        let nt = self.config.targets.len();
        let eps = self.config.epsilon.clamp(0.0, 1.0);
        let (idx, exploration) = if nt > 1 && eps > 0.0 {
            let seed = SEED ^ (self.inference_count as u32);
            let mut rng = mulberry32(seed);
            if rng() < eps {
                let mut alt = (rng() * (nt as f32 - 1.0)) as usize;
                if alt >= argmax_idx { alt += 1; }
                if alt >= nt { alt = nt - 1; }
                (alt, true)
            } else {
                (argmax_idx, false)
            }
        } else {
            (argmax_idx, false)
        };
        Route {
            model: self.config.targets[idx].clone(),
            context_bucket: cb as u8,
            temperature: f.tp,
            top_p: f.top_p,
            confidence: f.conf * p,
            algo: "fastgrnn",
            exploration,
        }
    }

    pub fn set_trained(&mut self, trained: bool) {
        self.trained = trained;
        self.version += 1;
    }

    pub fn sparsity_fraction(&self) -> f32 {
        let zeros = self.w.u.iter().filter(|&&x| x == 0.0).count() as f32;
        zeros / self.w.u.len() as f32
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    fn mk_router(n_targets: usize, in_dim: usize) -> Router {
        let targets: Vec<String> = (0..n_targets).map(|i| format!("m{}", i)).collect();
        Router::new(RouterConfig::new(in_dim, targets)).unwrap()
    }

    fn mk_emb(dim: usize, seed: u8) -> Vec<f32> {
        (0..dim).map(|i| ((i as u8).wrapping_add(seed) as f32) / 255.0 - 0.5).collect()
    }

    #[test]
    fn new_initializes_dims() {
        let r = mk_router(3, 384);
        assert_eq!(r.config.targets.len(), 3);
        assert_eq!(r.w.v.len(), RANK * 384);
        assert_eq!(r.w.u.len(), DIM * RANK);
        assert_eq!(r.w.uh.len(), DIM * DIM);
        assert_eq!(r.heads.model.len(), 3 * DIM);
        assert_eq!(r.heads.ctx.len(), CTX_BUCKETS * DIM);
    }

    #[test]
    fn empty_targets_returns_err() {
        assert!(Router::new(RouterConfig::new(384, vec![])).is_err());
    }

    #[test]
    fn untrained_returns_rule_route_first_target() {
        let mut r = mk_router(3, 384);
        let route = r.route(&mk_emb(384, 1), &RouteCtx { task_type: None, estimated_tokens: 500 });
        assert_eq!(route.model, "m0");
        assert_eq!(route.algo, "rule");
        assert_eq!(route.context_bucket, 0);
        assert_eq!(route.temperature, 0.7);
        assert!(!route.exploration);
    }

    #[test]
    fn trained_returns_fastgrnn_route() {
        let mut r = mk_router(3, 384);
        r.set_trained(true);
        let route = r.route(&mk_emb(384, 2), &RouteCtx { task_type: None, estimated_tokens: 5000 });
        assert_eq!(route.algo, "fastgrnn");
        assert!(r.config.targets.contains(&route.model));
        assert!(route.temperature >= 0.1 && route.temperature <= 1.5);
        assert!(route.top_p >= 0.5 && route.top_p <= 1.0);
        assert!(route.confidence >= 0.0 && route.confidence <= 1.0);
    }

    #[test]
    fn record_outcome_updates_target_stats() {
        let mut r = mk_router(2, 384);
        r.record_outcome("m0", 1.0);
        r.record_outcome("m0", 1.0);
        r.record_outcome("m1", 0.0);
        assert_eq!(r.per_target_counts[0], 2);
        assert_eq!(r.per_target_counts[1], 1);
        assert!(r.per_target_quality_milli[0] > 500);
        assert!(r.per_target_quality_milli[1] < 500);
    }

    #[test]
    fn record_outcome_unknown_target_is_noop() {
        let mut r = mk_router(2, 384);
        r.record_outcome("not_a_target", 1.0);
        assert_eq!(r.per_target_counts[0], 0);
        assert_eq!(r.per_target_counts[1], 0);
    }

    #[test]
    fn sparsity_around_90_percent() {
        let r = mk_router(3, 384);
        let s = r.sparsity_fraction();
        assert!((s - SPARSITY).abs() < 0.15, "sparsity {} should be near {}", s, SPARSITY);
    }

    #[test]
    fn bucket_for_tokens_monotone() {
        assert_eq!(bucket_for_tokens(100), 0);
        assert_eq!(bucket_for_tokens(1000), 0);
        assert_eq!(bucket_for_tokens(1001), 1);
        assert_eq!(bucket_for_tokens(4000), 1);
        assert_eq!(bucket_for_tokens(16000), 2);
        assert_eq!(bucket_for_tokens(64000), 3);
        assert_eq!(bucket_for_tokens(1_000_000), 4);
    }

    #[test]
    fn softmax_argmax_picks_max() {
        let (i, p) = softmax_argmax(&[0.0, 5.0, 1.0]);
        assert_eq!(i, 1);
        assert!(p > 0.9, "softmax mass on max should dominate, got {}", p);
    }

    #[test]
    fn forward_uses_low_rank_path() {
        let w = init_weights(384);
        let heads = init_heads(3);
        let x = mk_emb(384, 1);
        let f = forward(&w, &heads, &x, 384, 3);
        assert_eq!(f.h.len(), DIM);
        assert_eq!(f.ml.len(), 3);
        assert_eq!(f.cl.len(), CTX_BUCKETS);
        assert!(f.tp >= 0.1 && f.tp <= 1.5);
        assert!(f.top_p >= 0.5 && f.top_p <= 1.0);
        assert!(f.conf >= 0.0 && f.conf <= 1.0);
    }

    #[test]
    fn epsilon_zero_never_explores() {
        let mut r = mk_router(3, 384);
        r.set_trained(true);
        r.config.epsilon = 0.0;
        for _ in 0..50 {
            let route = r.route(&mk_emb(384, 4), &RouteCtx::default());
            assert!(!route.exploration);
        }
    }

    #[test]
    fn epsilon_one_always_explores() {
        let mut r = mk_router(3, 384);
        r.set_trained(true);
        r.config.epsilon = 1.0;
        let mut exploration_count = 0;
        for _ in 0..50 {
            let route = r.route(&mk_emb(384, 5), &RouteCtx::default());
            if route.exploration { exploration_count += 1; }
        }
        assert!(exploration_count > 40, "epsilon=1.0 should explore almost always, got {}/50", exploration_count);
    }

    #[test]
    fn adapter_can_override_route() {
        let mut r = mk_router(3, 384);
        r.set_trained(true);
        let emb = mk_emb(384, 6);
        let base = r.route(&emb, &RouteCtx::default());
        let biased = r.route_with_adapter(&emb, &RouteCtx::default(), |_, ml| {
            let target = if base.model == "m0" { 2 } else { 0 };
            ml[target] += 100.0;
        });
        assert_ne!(biased.model, base.model, "huge adapter bump should flip the route");
    }
}
