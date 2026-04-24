use super::*;

impl Router {
    pub fn train(&mut self, batch: &[TrainSample]) -> Result<usize> {
        let mut applied = 0usize;
        let nt = self.targets.len();
        let base_lr = 0.05f32;
        let epochs: usize = std::env::var("RS_LEARN_ROUTER_EPOCHS").ok()
            .and_then(|v| v.parse().ok()).filter(|&n: &usize| n >= 1 && n <= 8).unwrap_or(2);
        let mut order: Vec<usize> = (0..batch.len()).collect();
        let mut rng = mulberry32(self.version.wrapping_add(1) as u32 ^ SEED);
        for _epoch in 0..epochs {
            for i in (1..order.len()).rev() {
                let j = (rng() * (i as f32 + 1.0)) as usize;
                order.swap(i, j.min(i));
            }
            for &bi in &order {
                let tr = &batch[bi];
                if tr.embedding.len() != IN { continue; }
                let centered = tr.quality - 0.5;
                if centered.abs() < 1e-4 { continue; }
                let positive = centered > 0.0;
                let Some(t_idx) = self.targets.iter().position(|t| t == &tr.chosen_target) else { continue };
                let f = forward(&self.w, &self.heads, &tr.embedding, nt);
                let sign = if positive { 1.0f32 } else { -1.0f32 };
                let strength = centered.abs() * 2.0;
                let lr = base_lr * strength;
                let model_probs = softmax(&f.ml);
                for k in 0..nt {
                    let err = model_probs[k] - if k == t_idx { 1.0 } else { 0.0 };
                    let off = k * DIM;
                    crate::simd::axpy(-sign * lr * err, &f.h, &mut self.heads.model[off..off + DIM]);
                    self.heads.model_b[k] -= sign * lr * err;
                }
                if positive {
                    let ctx_target = (bucket_for_tokens(tr.estimated_tokens) as usize).min(CTX_BUCKETS - 1);
                    let ctx_probs = softmax(&f.cl);
                    for k in 0..CTX_BUCKETS {
                        let err = ctx_probs[k] - if k == ctx_target { 1.0 } else { 0.0 };
                        let off = k * DIM;
                        crate::simd::axpy(-lr * err, &f.h, &mut self.heads.ctx[off..off + DIM]);
                        self.heads.ctx_b[k] -= lr * err;
                    }
                }
                applied += 1;
            }
        }
        let unique = applied / epochs.max(1);
        self.trajectory_count += unique as u64;
        self.traj_count_obs.store(self.trajectory_count, Ordering::Relaxed);
        if self.trajectory_count >= self.threshold { self.trained = true; }
        Ok(unique)
    }
}
