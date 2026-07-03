use serde::Deserialize;

pub const RANK: usize = 2;
pub const DECAY: f32 = 0.995;
pub const LR0: f32 = 0.01;
pub const MAX_ADAPTER_NORM: f32 = 5.0;
pub const IN: usize = crate::embeddings::EMBED_DIM;
pub const REPLAY_CAP: usize = 64;

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

pub struct InstantCore {
    pub adapter_a: Vec<f32>,
    pub adapter_b: Vec<f32>,
    pub targets: Vec<String>,
    pub n_targets: usize,
    pub lr: f32,
    pub lr_min: f32,
    pub ewc: Option<EwcState>,
    pub replay_buf: std::collections::VecDeque<(Vec<f32>, usize, f32)>,
    pub feedback_count: u64,
    pub resets_performed: u64,
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0f32;
    for i in 0..a.len() { s += a[i] * b[i]; }
    s
}

fn axpy(scale: f32, src: &[f32], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    for i in 0..src.len() { dst[i] += scale * src[i]; }
}

fn weighted_pick(buf: &std::collections::VecDeque<(Vec<f32>, usize, f32)>, seed: &mut u32) -> usize {
    let n = buf.len();
    *seed = seed.wrapping_mul(2654435761).wrapping_add(1);
    let r = (*seed as f32) / (u32::MAX as f32 + 1.0);
    let total: f32 = buf.iter().map(|(_, _, s)| s.abs()).sum();
    if !(total > 0.0) { return ((*seed as usize) % n).min(n - 1); }
    let target = r * total;
    let mut acc = 0f32;
    for (i, (_, _, s)) in buf.iter().enumerate() {
        acc += s.abs();
        if target < acc { return i; }
    }
    n - 1
}

impl InstantCore {
    pub fn new(targets: Vec<String>) -> Self {
        let n_targets = targets.len().max(1);
        let lr_min = read_lr_min();
        Self {
            adapter_a: vec![0f32; IN * RANK],
            adapter_b: vec![0f32; RANK * n_targets],
            targets,
            n_targets,
            lr: LR0,
            lr_min,
            ewc: None,
            replay_buf: std::collections::VecDeque::with_capacity(REPLAY_CAP),
            feedback_count: 0,
            resets_performed: 0,
        }
    }

    pub fn target_index(&self, name: &str) -> Option<usize> {
        self.targets.iter().position(|t| t == name)
    }

    pub fn adapter_norm(&self) -> f32 {
        let mut s = 0f32;
        for &x in &self.adapter_a { s += x * x; }
        for &x in &self.adapter_b { s += x * x; }
        s.sqrt()
    }

    pub fn apply_adapter(&self, embedding: &[f32], logits: &mut [f32]) {
        let nt = self.n_targets;
        if embedding.len() != IN || logits.len() != nt { return; }
        let mut proj = vec![0f32; RANK];
        for r in 0..RANK {
            let off = r * IN;
            proj[r] = dot(&self.adapter_a[off..off + IN], embedding);
        }
        for k in 0..nt {
            let mut s = 0f32;
            for r in 0..RANK { s += self.adapter_b[r * nt + k] * proj[r]; }
            logits[k] += s;
        }
    }

    pub fn hebbian_update(&mut self, embedding: &[f32], t_idx: usize, quality: f32) {
        if t_idx >= self.n_targets || embedding.len() != IN { return; }
        let scale = self.lr * quality;
        let fallback = 1.0 / (RANK as f32).sqrt();
        for r in 0..RANK {
            let off = r * IN;
            let mut b_val = self.adapter_b[r * self.n_targets + t_idx];
            if b_val == 0.0 { b_val = fallback; }
            axpy(scale * b_val, embedding, &mut self.adapter_a[off..off + IN]);
        }
        for r in 0..RANK {
            let off = r * IN;
            let pr = dot(&self.adapter_a[off..off + IN], embedding);
            self.adapter_b[r * self.n_targets + t_idx] += scale * pr;
        }
        if let Some(ewc) = self.ewc.as_ref() {
            let a_len = self.adapter_a.len();
            let lam = ewc.lambda;
            let lr = self.lr;
            for i in 0..a_len {
                let d = self.adapter_a[i] - ewc.snapshot[i];
                self.adapter_a[i] -= lr * lam * ewc.fisher[i] * d;
            }
            for j in 0..self.adapter_b.len() {
                let k = a_len + j;
                let d = self.adapter_b[j] - ewc.snapshot[k];
                self.adapter_b[j] -= lr * lam * ewc.fisher[k] * d;
            }
        }
        self.lr = (self.lr * DECAY).max(self.lr_min);
        let norm = self.adapter_norm();
        if norm > MAX_ADAPTER_NORM {
            let s = MAX_ADAPTER_NORM / norm;
            for x in self.adapter_a.iter_mut() { *x *= s; }
            for x in self.adapter_b.iter_mut() { *x *= s; }
        }
    }

    pub fn feedback(&mut self, embedding: &[f32], model: &str, payload: FeedbackPayload, now_ms: i64) -> Result<(), &'static str> {
        if !(0.0..=1.0).contains(&payload.quality) { return Err("quality must be 0..1"); }
        if embedding.len() != IN { return Err("embedding wrong dim"); }
        let idx = match self.target_index(model) { Some(i) => i, None => return Err("unknown model name; not in registered targets") };
        self.feedback_count += 1;
        let centered = payload.quality - 0.5;
        if centered.abs() < 1e-4 { return Ok(()); }
        let scale = centered * 2.0;
        self.hebbian_update(embedding, idx, scale);
        if self.replay_buf.len() >= REPLAY_CAP { self.replay_buf.pop_front(); }
        self.replay_buf.push_back((embedding.to_vec(), idx, scale));
        if self.replay_buf.len() >= 4 {
            let mut seed = (now_ms as u32).wrapping_mul(2654435761);
            let pick = weighted_pick(&self.replay_buf, &mut seed);
            let (re, ri, rs) = self.replay_buf[pick].clone();
            self.hebbian_update(&re, ri, rs * 0.5);
        }
        Ok(())
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
        self.resets_performed += 1;
    }

    pub fn serialize_adapter_flat(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.adapter_a.len() + self.adapter_b.len());
        flat.extend_from_slice(&self.adapter_a);
        flat.extend_from_slice(&self.adapter_b);
        flat
    }
}

#[cfg(target_arch = "wasm32")]
fn read_lr_min() -> f32 {
    LR0.min(1e-3)
}

#[cfg(not(target_arch = "wasm32"))]
fn read_lr_min() -> f32 {
    std::env::var("RS_LEARN_LR_MIN").ok()
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(1e-3)
        .min(LR0)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    fn make_emb(seed: u8) -> Vec<f32> {
        (0..IN).map(|i| ((i as u8).wrapping_add(seed) as f32) / 255.0 - 0.5).collect()
    }

    #[test]
    fn adapter_norm_starts_zero() {
        let core = InstantCore::new(vec!["a".into(), "b".into()]);
        assert_eq!(core.adapter_norm(), 0.0);
    }

    #[test]
    fn positive_feedback_grows_norm_under_bound() {
        let mut core = InstantCore::new(vec!["m".into()]);
        let emb = make_emb(7);
        let mut prev = 0f32;
        for i in 0..100 {
            core.feedback(&emb, "m", FeedbackPayload { quality: 1.0, signal: None }, i as i64).unwrap();
            let n = core.adapter_norm();
            assert!(n <= MAX_ADAPTER_NORM + 1e-4, "norm {} exceeded bound at step {}", n, i);
            prev = n;
        }
        assert!(prev > 0.0, "norm did not grow under positive feedback");
    }

    #[test]
    fn norm_clamped_at_max() {
        let mut core = InstantCore::new(vec!["m".into()]);
        for x in core.adapter_a.iter_mut() { *x = 10.0; }
        for x in core.adapter_b.iter_mut() { *x = 10.0; }
        let emb = make_emb(3);
        core.feedback(&emb, "m", FeedbackPayload { quality: 1.0, signal: None }, 1).unwrap();
        assert!(core.adapter_norm() <= MAX_ADAPTER_NORM + 1e-4);
    }

    #[test]
    fn lr_floor_respected() {
        let mut core = InstantCore::new(vec!["m".into()]);
        let emb = make_emb(11);
        for i in 0..10_000 {
            core.feedback(&emb, "m", FeedbackPayload { quality: 0.6, signal: None }, i).unwrap();
        }
        assert!(core.lr >= core.lr_min - 1e-9, "lr {} fell below floor {}", core.lr, core.lr_min);
    }

    #[test]
    fn neutral_feedback_no_change() {
        let mut core = InstantCore::new(vec!["m".into()]);
        let emb = make_emb(5);
        core.feedback(&emb, "m", FeedbackPayload { quality: 0.5, signal: None }, 1).unwrap();
        assert_eq!(core.adapter_norm(), 0.0);
    }

    #[test]
    fn reset_zeros_adapter() {
        let mut core = InstantCore::new(vec!["m".into()]);
        let emb = make_emb(2);
        for i in 0..20 { core.feedback(&emb, "m", FeedbackPayload { quality: 1.0, signal: None }, i).unwrap(); }
        assert!(core.adapter_norm() > 0.0);
        core.reset_adapter();
        assert_eq!(core.adapter_norm(), 0.0);
        assert_eq!(core.lr, LR0);
        assert_eq!(core.resets_performed, 1);
    }

    #[test]
    fn unknown_target_errors() {
        let mut core = InstantCore::new(vec!["m".into()]);
        let emb = make_emb(9);
        let err = core.feedback(&emb, "unknown", FeedbackPayload { quality: 1.0, signal: None }, 1);
        assert!(err.is_err());
        assert_eq!(core.adapter_norm(), 0.0);
        assert_eq!(core.feedback_count, 0);
    }
}
