use std::collections::{HashMap, VecDeque};

pub const FISHER_DECAY: f32 = 0.999;
pub const DEFAULT_LAMBDA: f32 = 2000.0;
pub const RING_CAP: usize = 10;
pub const BOUNDARY_Z: f32 = 2.5;
pub const MIN_STDDEV: f32 = 1e-4;

pub struct DeepCore {
    pub lambda: f32,
    pub loss_ring: VecDeque<f32>,
    pub ring_cap: usize,
    pub fisher: HashMap<String, Vec<f32>>,
    pub params_snapshot: HashMap<String, Vec<f32>>,
    pub boundaries_detected: u64,
}

impl DeepCore {
    pub fn new() -> Self {
        Self::with_lambda(read_lambda())
    }

    pub fn with_lambda(lambda: f32) -> Self {
        Self {
            lambda,
            loss_ring: VecDeque::with_capacity(RING_CAP),
            ring_cap: RING_CAP,
            fisher: HashMap::new(),
            params_snapshot: HashMap::new(),
            boundaries_detected: 0,
        }
    }

    pub fn consolidate(&mut self, param_id: &str, params: &[f32], grads: &[f32]) -> Result<(), &'static str> {
        if params.len() != grads.len() {
            return Err("consolidate: params/grads length mismatch");
        }
        let prev = self.fisher.entry(param_id.to_string())
            .or_insert_with(|| vec![0f32; params.len()]);
        if prev.len() != params.len() { prev.resize(params.len(), 0.0); }
        for i in 0..params.len() {
            let g2 = grads[i] * grads[i];
            prev[i] = FISHER_DECAY * prev[i] + (1.0 - FISHER_DECAY) * g2;
        }
        self.params_snapshot.insert(param_id.to_string(), params.to_vec());
        Ok(())
    }

    pub fn record_loss(&mut self, loss: f32) -> bool {
        let prior: Vec<f32> = self.loss_ring.iter().copied().collect();
        if self.loss_ring.len() >= self.ring_cap { self.loss_ring.pop_front(); }
        self.loss_ring.push_back(loss);
        if prior.len() < 3 { return false; }
        let n = prior.len() as f32;
        let mean = prior.iter().sum::<f32>() / n;
        let var = prior.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let stddev = var.sqrt().max(MIN_STDDEV);
        let z = (loss - mean).abs() / stddev;
        if z > BOUNDARY_Z {
            self.boundaries_detected += 1;
            return true;
        }
        false
    }

    pub fn ewc_state(&self, param_id: &str) -> Option<(Vec<f32>, Vec<f32>, f32)> {
        let f = self.fisher.get(param_id)?.clone();
        let s = self.params_snapshot.get(param_id)?.clone();
        if f.len() != s.len() || f.is_empty() { return None; }
        Some((f, s, self.lambda))
    }

    pub fn ewc_penalty(&self, param_id: &str, params: &[f32]) -> f32 {
        let f = match self.fisher.get(param_id) { Some(v) => v, None => return 0.0 };
        let snap = match self.params_snapshot.get(param_id) { Some(v) => v, None => return 0.0 };
        let n = params.len().min(f.len()).min(snap.len());
        let mut sum = 0f32;
        for i in 0..n {
            let d = params[i] - snap[i];
            sum += f[i] * d * d;
        }
        self.lambda * sum
    }

    pub fn window_stats(&self) -> (f32, f32, usize) {
        let n = self.loss_ring.len();
        if n == 0 { return (0.0, 0.0, 0); }
        let mean = self.loss_ring.iter().sum::<f32>() / n as f32;
        let var = self.loss_ring.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        (mean, var.sqrt(), n)
    }
}

impl Default for DeepCore {
    fn default() -> Self { Self::new() }
}

#[cfg(target_arch = "wasm32")]
fn read_lambda() -> f32 {
    DEFAULT_LAMBDA
}

#[cfg(not(target_arch = "wasm32"))]
fn read_lambda() -> f32 {
    std::env::var("RS_LEARN_EWC_LAMBDA").ok()
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(DEFAULT_LAMBDA)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn new_has_zero_state() {
        let d = DeepCore::with_lambda(100.0);
        assert_eq!(d.boundaries_detected, 0);
        assert_eq!(d.loss_ring.len(), 0);
        assert!(d.fisher.is_empty());
    }

    #[test]
    fn fisher_ema_accumulates() {
        let mut d = DeepCore::with_lambda(1.0);
        let params = vec![0.5f32; 4];
        let grads = vec![1.0f32; 4];
        d.consolidate("p", &params, &grads).unwrap();
        let f1 = d.fisher.get("p").unwrap().clone();
        assert!(f1[0] > 0.0 && f1[0] < 1.0, "first EMA step should land between 0 and 1, got {}", f1[0]);

        for _ in 0..1000 { d.consolidate("p", &params, &grads).unwrap(); }
        let f2 = d.fisher.get("p").unwrap().clone();
        assert!(f2[0] > f1[0], "Fisher should grow under repeated unit grads");
        assert!(f2[0] < 1.001, "Fisher EMA bounded above by g^2 sup, got {}", f2[0]);
    }

    #[test]
    fn consolidate_rejects_length_mismatch() {
        let mut d = DeepCore::new();
        assert!(d.consolidate("p", &[1.0; 4], &[1.0; 3]).is_err());
    }

    #[test]
    fn ewc_penalty_zero_at_snapshot() {
        let mut d = DeepCore::with_lambda(1000.0);
        let params = vec![0.3f32; 5];
        d.consolidate("p", &params, &[1.0; 5]).unwrap();
        assert_eq!(d.ewc_penalty("p", &params), 0.0);
    }

    #[test]
    fn ewc_penalty_grows_with_distance() {
        let mut d = DeepCore::with_lambda(1.0);
        let params0 = vec![0.0f32; 3];
        d.consolidate("p", &params0, &[1.0; 3]).unwrap();
        let pen_near = d.ewc_penalty("p", &[0.1, 0.1, 0.1]);
        let pen_far  = d.ewc_penalty("p", &[1.0, 1.0, 1.0]);
        assert!(pen_far > pen_near);
        assert!(pen_near > 0.0);
    }

    #[test]
    fn z_score_boundary_fires_on_outlier() {
        let mut d = DeepCore::new();
        for _ in 0..5 { d.record_loss(0.5); }
        let fired = d.record_loss(0.5);
        assert!(!fired);
        let fired_outlier = d.record_loss(100.0);
        assert!(fired_outlier, "z-score outlier should fire boundary");
        assert_eq!(d.boundaries_detected, 1);
    }

    #[test]
    fn first_three_losses_never_fire() {
        let mut d = DeepCore::new();
        assert!(!d.record_loss(0.5));
        assert!(!d.record_loss(100.0));
        assert!(!d.record_loss(0.5));
        assert_eq!(d.boundaries_detected, 0);
    }

    #[test]
    fn ring_cap_enforced() {
        let mut d = DeepCore::new();
        for i in 0..30 { d.record_loss(i as f32); }
        assert_eq!(d.loss_ring.len(), RING_CAP);
        assert_eq!(*d.loss_ring.front().unwrap(), 20.0);
    }

    #[test]
    fn window_stats_match_ring() {
        let mut d = DeepCore::new();
        for x in [1.0, 2.0, 3.0, 4.0, 5.0] { d.record_loss(x); }
        let (mean, _stddev, n) = d.window_stats();
        assert_eq!(n, 5);
        assert!((mean - 3.0).abs() < 1e-5);
    }

    #[test]
    fn ewc_state_returns_full_triple() {
        let mut d = DeepCore::with_lambda(7.0);
        let p = vec![0.1, 0.2, 0.3];
        d.consolidate("p", &p, &[1.0, 1.0, 1.0]).unwrap();
        let (f, s, lam) = d.ewc_state("p").unwrap();
        assert_eq!(f.len(), 3);
        assert_eq!(s, p);
        assert_eq!(lam, 7.0);
    }

    #[test]
    fn ewc_state_missing_returns_none() {
        let d = DeepCore::with_lambda(1.0);
        assert!(d.ewc_state("nope").is_none());
    }
}
