use super::*;

pub(crate) fn weighted_pick(buf: &std::collections::VecDeque<(Vec<f32>, usize, f32)>, seed: &mut u32) -> usize {
    let n = buf.len();
    *seed = seed.wrapping_mul(2654435761).wrapping_add(1);
    let r = (*seed as f32) / (u32::MAX as f32 + 1.0);
    let total: f32 = buf.iter().map(|(_, _, s)| s.abs()).sum();
    if !(total > 0.0) {
        return ((*seed as usize) % n).min(n - 1);
    }
    let target = r * total;
    let mut acc = 0f32;
    for (i, (_, _, s)) in buf.iter().enumerate() {
        acc += s.abs();
        if target < acc { return i; }
    }
    n - 1
}

impl InstantLoop {
    pub(super) fn hebbian_update(&mut self, embedding: &[f32], t_idx: usize, quality: f32) {
        if t_idx >= self.n_targets { return; }
        let scale = self.lr * quality;
        let fallback = 1.0 / (RANK as f32).sqrt();
        for r in 0..RANK {
            let off = r * IN;
            let mut b_val = self.adapter_b[r * self.n_targets + t_idx];
            if b_val == 0.0 { b_val = fallback; }
            crate::simd::axpy(scale * b_val, embedding, &mut self.adapter_a[off..off + IN]);
        }
        for r in 0..RANK {
            let off = r * IN;
            let pr = crate::simd::dot(&self.adapter_a[off..off + IN], embedding);
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
        let final_norm = self.adapter_norm();
        self.adapter_norm_milli.store((final_norm * 1000.0) as u64, Ordering::Relaxed);
    }

    pub(super) fn gc_pending(&mut self) {
        let now = Instant::now();
        let before = self.pending.len();
        self.pending.retain(|_, p| now.duration_since(p.created_at) < FEEDBACK_WINDOW);
        let dropped = before.saturating_sub(self.pending.len());
        if dropped > 0 { self.feedback_expired.fetch_add(dropped as u64, Ordering::Relaxed); }
        self.pending_count.store(self.pending.len() as u64, Ordering::Relaxed);
    }

    pub async fn record_trajectory(
        &mut self,
        session_id: Option<String>,
        embedding: Vec<f32>,
        route_model: String,
        response: String,
        query_text: Option<String>,
        implicit_quality: Option<f64>,
        latency_ms: u64,
    ) -> Result<RequestId> {
        self.record_trajectory_full(session_id, embedding, route_model, response, query_text, implicit_quality, latency_ms, Vec::new(), None).await
    }

    pub async fn record_trajectory_full(
        &mut self,
        session_id: Option<String>,
        embedding: Vec<f32>,
        route_model: String,
        response: String,
        query_text: Option<String>,
        implicit_quality: Option<f64>,
        latency_ms: u64,
        retrieved_strategies: Vec<String>,
        dominant_relation: Option<String>,
    ) -> Result<RequestId> {
        if embedding.len() != IN { return Err(anyhow!("embedding must be {IN}-d")); }
        self.gc_pending();
        let request_id = Uuid::new_v4().to_string();
        let created_ms = now_ms();
        let decision = json!({ "model": route_model.clone() }).to_string();
        let row = TrajectoryRow {
            id: request_id.clone(),
            session_id: session_id.clone(),
            query: query_text.clone(),
            query_embedding: Some(embedding.clone()),
            retrieved_ids: None,
            router_decision: Some(decision),
            response: Some(response),
            activations: None,
            quality: implicit_quality,
            latency_ms: Some(latency_ms as i64),
            created_at: Some(created_ms),
        };
        match self.spine.as_ref() {
            Some(s) => s.send(row).await?,
            None => self.store.insert_trajectory(&row).await?,
        }
        self.pending.insert(request_id.clone(), PendingInfo {
            session_id, embedding, route_model,
            query_text,
            created_at: Instant::now(), created_ms,
            retrieved_strategies,
            dominant_relation,
        });
        self.pending_count.store(self.pending.len() as u64, Ordering::Relaxed);
        Ok(request_id)
    }

    pub async fn feedback(&mut self, request_id: &str, payload: FeedbackPayload) -> Result<()> {
        if !(0.0..=1.0).contains(&payload.quality) { return Err(anyhow!("quality must be 0..1")); }
        self.gc_pending();
        let p = self.pending.get(request_id).cloned();
        let (emb, sid, model, cms) = match &p {
            Some(pi) => (pi.embedding.clone(), pi.session_id.clone(), pi.route_model.clone(), pi.created_ms),
            None => (vec![0f32; IN], None, String::new(), now_ms()),
        };
        let decision = json!({ "model": model.clone(), "signal": payload.signal }).to_string();
        let row = TrajectoryRow {
            id: request_id.to_string(),
            session_id: sid,
            query: None,
            query_embedding: Some(emb.clone()),
            retrieved_ids: None,
            router_decision: Some(decision),
            response: None,
            activations: None,
            quality: Some(payload.quality as f64),
            latency_ms: Some(0),
            created_at: Some(cms),
        };
        match self.spine.as_ref() {
            Some(s) => s.send(row).await?,
            None => self.store.insert_trajectory(&row).await?,
        }
        self.feedback_count.fetch_add(1, Ordering::Relaxed);
        if let Some(idx) = self.target_index(&model) {
            let centered = payload.quality - 0.5;
            let applied: Option<f32> = if centered.abs() < 1e-4 {
                None
            } else {
                let scale = centered * 2.0;
                self.hebbian_update(&emb, idx, scale);
                Some(scale)
            };
            if let Some(scale) = applied {
                if self.replay_buf.len() >= REPLAY_CAP { self.replay_buf.pop_front(); }
                self.replay_buf.push_back((emb.clone(), idx, scale));
                if self.replay_buf.len() >= 4 {
                    let mut seed = (now_ms() as u32).wrapping_mul(2654435761);
                    let pick = weighted_pick(&self.replay_buf, &mut seed);
                    let (re, ri, rs) = self.replay_buf[pick].clone();
                    self.hebbian_update(&re, ri, rs * 0.5);
                }
            }
        }
        self.pending.remove(request_id);
        self.pending_count.store(self.pending.len() as u64, Ordering::Relaxed);
        Ok(())
    }
}
