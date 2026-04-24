use super::{Orchestrator, QueryOpts, QueryResult, RouteSnapshot};
use crate::attention::{Context as AttnContext, Subgraph, SubgraphNode};
use crate::learn::instant::{FeedbackPayload, InstantLoop};
use crate::router::RouteCtx;
use anyhow::{anyhow, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::Instant;

impl Orchestrator {
    pub async fn query(&self, text: &str, opts: QueryOpts) -> Result<QueryResult> {
        let t0 = Instant::now();
        let mut stages: HashMap<String, u64> = HashMap::new();
        let sid = self.session(opts.session_id.clone()).await;

        let t_e = Instant::now();
        let emb = match self.embed_cache.as_ref() {
            Some(c) => c.embed(text).await?,
            None => self.embedder.embed(text)?,
        };
        stages.insert("embed".into(), t_e.elapsed().as_millis() as u64);

        let k = if opts.max_retrieved == 0 { 8 } else { opts.max_retrieved };
        let t_m = Instant::now();
        let neighbors = self.memory.search(&emb, k).await?;
        let subgraph = if neighbors.is_empty() {
            Default::default()
        } else {
            let seeds: Vec<&str> = neighbors.iter().take(3).map(|n| n.id.as_str()).collect();
            let mut sg = self.memory.expand(seeds[0], 1).await.unwrap_or_default();
            for seed in &seeds[1..] {
                let extra = self.memory.expand(seed, 1).await.unwrap_or_default();
                sg.nodes.extend(extra.nodes);
                sg.edges.extend(extra.edges);
            }
            sg.nodes.dedup_by(|a, b| a.id == b.id);
            sg
        };
        stages.insert("memory".into(), t_m.elapsed().as_millis() as u64);

        let t_a = Instant::now();
        let attn_result = self.attention.attend(&emb, &subgraph);
        let attn_text = match &attn_result {
            Ok(ctx) => attention_hint(ctx, &subgraph),
            Err(e) => { eprintln!("attention error (non-fatal): {e}"); String::new() }
        };
        let training_emb = match &attn_result {
            Ok(ctx) if !ctx.weights.is_empty() => ctx.vector.clone(),
            _ => emb.clone(),
        };
        stages.insert("attention".into(), t_a.elapsed().as_millis() as u64);

        let t_r = Instant::now();
        let ctx = RouteCtx {
            task_type: opts.task_type.clone(),
            estimated_tokens: opts.estimated_tokens.unwrap_or(text.len() as u64),
        };
        let route = {
            let il_snapshot = {
                let il = self.instant.lock().await;
                (il.adapter_a.clone(), il.adapter_b.clone(), il.targets_clone(), il.adapter_rank())
            };
            let r = self.router.lock().await;
            r.route_with_adapter(&emb, &ctx, |e, logits| {
                InstantLoop::apply_adapter_raw(&il_snapshot.0, &il_snapshot.1, il_snapshot.3, &il_snapshot.2, e, logits);
            })
        };
        let route_model = route.model.clone();
        let confidence = route.confidence;
        let snapshot: RouteSnapshot = route.into();
        stages.insert("route".into(), t_r.elapsed().as_millis() as u64);

        let t_h = Instant::now();
        let hints = self.reasoning.retrieve_for_query(text, 3).await.unwrap_or_default();
        let hint_text = if hints.is_empty() { String::new() } else {
            format!("\n\nReasoning hints:\n{}", hints.iter().map(|h| format!("- {}", h.strategy)).collect::<Vec<_>>().join("\n"))
        };
        let code_text = match (opts.include_code_search, self.rs_search.as_ref()) {
            (true, Some(rs)) => {
                let hits = rs.search(text, &self.search_root);
                if hits.is_empty() { String::new() } else {
                    format!("\n\nCode context:\n{}", hits.iter().take(3).map(|h| format!("- {}:{}-{}", h.file, h.line_start, h.line_end)).collect::<Vec<_>>().join("\n"))
                }
            }
            _ => String::new(),
        };
        stages.insert("hints".into(), t_h.elapsed().as_millis() as u64);

        let memory_text = if neighbors.is_empty() { String::new() } else {
            let items: Vec<String> = neighbors.iter().take(3).filter(|n| !n.payload.is_empty()).map(|n| format!("- {}", n.payload)).collect();
            if items.is_empty() { String::new() } else { format!("\n\nMemory context:\n{}", items.join("\n")) }
        };
        let t_acp = Instant::now();
        let sys = format!("You are rs-learn. Task type: {}.{}{}{}{}", opts.task_type.as_deref().unwrap_or("default"), memory_text, hint_text, code_text, attn_text);
        let response = self.acp.generate(&sys, text, 120_000).await.map_err(|e| anyhow!("acp: {e}"))?;
        stages.insert("acp".into(), t_acp.elapsed().as_millis() as u64);

        let t_l = Instant::now();
        let response_str = if let Value::String(s) = &response { s.clone() } else { response.to_string() };
        let latency_ms = t0.elapsed().as_millis() as u64;
        let grounding = neighbors.first().map(|n| n.score.clamp(0.0, 1.0));
        let implicit_quality = Some(implicit_quality_from(latency_ms, grounding, confidence));
        let request_id = {
            let mut il = self.instant.lock().await;
            il.record_trajectory(Some(sid.clone()), training_emb, route_model, response_str, Some(text.to_string()), implicit_quality, latency_ms).await?
        };
        stages.insert("learn".into(), t_l.elapsed().as_millis() as u64);

        { let mut map = self.sessions.write().await;
          if let Some(s) = map.get_mut(&sid) { s.last_embedding = Some(emb); } }
        self.queries.fetch_add(1, Ordering::Relaxed);
        self.total_ms.fetch_add(latency_ms, Ordering::Relaxed);

        Ok(QueryResult { text: response, request_id, session_id: sid, routing: snapshot, retrieved: neighbors, confidence, latency_ms, stage_breakdown: stages })
    }

    pub async fn feedback(&self, request_id: &str, payload: FeedbackPayload) -> Result<()> {
        let (sid_for_ema, emb_for_memory, query_for_memory) = {
            let il = self.instant.lock().await;
            let sid = il.pending.get(request_id).and_then(|p| p.session_id.clone());
            let emb = il.pending.get(request_id).map(|p| p.embedding.clone());
            let query = il.pending.get(request_id).and_then(|p| p.query_text.clone());
            (sid, emb, query)
        };
        let effective_quality = {
            let mut map = self.sessions.write().await;
            if let Some(ref sid) = sid_for_ema {
                if let Some(sess) = map.get_mut(sid) {
                    let smoothed = 0.7 * payload.quality + 0.3 * sess.quality_ema;
                    sess.quality_ema = smoothed;
                    smoothed
                } else { payload.quality }
            } else { payload.quality }
        };
        let smoothed_payload = FeedbackPayload { quality: effective_quality, ..payload };
        let loss = (1.0 - effective_quality).max(0.0);
        let boundary = { let mut dl = self.deep.lock().await; dl.record_loss(loss).await.unwrap_or(false) };
        {
            let mut il = self.instant.lock().await;
            let emb_before: Option<Vec<f32>> = il.pending.get(request_id).map(|p| p.embedding.clone());
            let quality = effective_quality;
            il.feedback(request_id, smoothed_payload).await?;
            if boundary {
                if let Some(emb) = emb_before {
                    let flat = il.serialize_adapter_flat();
                    let grads: Vec<f32> = flat.iter().enumerate().map(|(i, _)| emb[i % emb.len()] * quality).collect();
                    let mut dl = self.deep.lock().await;
                    let _ = dl.consolidate("adapter", &flat, &grads).await;
                }
                il.reset_adapter();
            }
        }
        if effective_quality >= 0.7 {
            if let (Some(emb), Some(text)) = (emb_for_memory, query_for_memory) {
                let _ = self.memory.add(crate::memory::NodeInput { id: None, payload: serde_json::json!({ "query": text }), embedding: emb, level: None }).await;
            }
        }
        Ok(())
    }
}

pub(super) fn attention_hint(ctx: &AttnContext, subgraph: &Subgraph) -> String {
    let valid: Vec<(usize, &SubgraphNode)> = subgraph.nodes.iter().enumerate()
        .filter(|(_, n)| n.embedding.as_ref().map(|e| e.len() == 768).unwrap_or(false))
        .collect();
    if valid.is_empty() || ctx.weights.is_empty() { return String::new(); }
    let n = valid.len();
    let mut mean = vec![0.0f32; n];
    for head in &ctx.weights {
        if head.len() != n { return String::new(); }
        for (i, w) in head.iter().enumerate() { mean[i] += *w; }
    }
    let h = ctx.weights.len() as f32;
    for v in mean.iter_mut() { *v /= h; }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|a, b| mean[*b].partial_cmp(&mean[*a]).unwrap_or(std::cmp::Ordering::Equal));
    let top: Vec<String> = idx.iter().take(3).map(|i| format!("- {} (attn={:.3})", valid[*i].1.id, mean[*i])).collect();
    format!("\n\nTop-attended context:\n{}", top.join("\n"))
}

pub(super) fn implicit_quality_from(latency_ms: u64, grounding: Option<f32>, confidence: f32) -> f64 {
    let latency_score = (5000.0f64 - (latency_ms as f64).min(5000.0)) / 5000.0;
    let conf = confidence.clamp(0.0, 1.0) as f64;
    match grounding {
        None => (0.35 * latency_score + 0.15 * conf + 0.50 * 0.5).clamp(0.0, 1.0),
        Some(g) => {
            let ground = g.clamp(0.0, 1.0) as f64;
            let q = 0.35 * latency_score + 0.50 * ground + 0.15 * conf;
            if ground < 0.15 { return (q * 0.5).clamp(0.0, 1.0); }
            q.clamp(0.0, 1.0)
        }
    }
}
