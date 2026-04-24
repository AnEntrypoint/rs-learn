use super::hints::{attention_hint, implicit_quality_from};
use super::{Orchestrator, QueryOpts, QueryResult, RouteSnapshot};
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
        let hint_ids: Vec<String> = hints.iter().map(|h| h.id.clone()).collect();
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
        let dominant_relation: Option<String> = {
            let mut counts: HashMap<String, u32> = HashMap::new();
            for e in &subgraph.edges {
                if let Some(r) = e.relation.as_deref() {
                    *counts.entry(r.split("-L").next().unwrap_or(r).to_string()).or_insert(0) += 1;
                }
            }
            counts.into_iter().max_by_key(|(_, c)| *c).map(|(r, _)| r)
        };
        let request_id = {
            let mut il = self.instant.lock().await;
            il.record_trajectory_full(Some(sid.clone()), training_emb, route_model, response_str, Some(text.to_string()), implicit_quality, latency_ms, hint_ids, dominant_relation).await?
        };
        stages.insert("learn".into(), t_l.elapsed().as_millis() as u64);

        { let mut map = self.sessions.write().await;
          if let Some(s) = map.get_mut(&sid) { s.last_embedding = Some(emb); } }
        self.queries.fetch_add(1, Ordering::Relaxed);
        self.total_ms.fetch_add(latency_ms, Ordering::Relaxed);

        Ok(QueryResult { text: response, request_id, session_id: sid, routing: snapshot, retrieved: neighbors, confidence, latency_ms, stage_breakdown: stages })
    }

    pub async fn feedback(&self, request_id: &str, payload: FeedbackPayload) -> Result<()> {
        let (sid_for_ema, emb_for_memory, query_for_memory, strategy_ids, route_model_for_outcome, dominant_relation) = {
            let il = self.instant.lock().await;
            let p = il.pending.get(request_id);
            let sid = p.and_then(|p| p.session_id.clone());
            let emb = p.map(|p| p.embedding.clone());
            let query = p.and_then(|p| p.query_text.clone());
            let strats = p.map(|p| p.retrieved_strategies.clone()).unwrap_or_default();
            let model = p.map(|p| p.route_model.clone()).unwrap_or_default();
            let rel = p.and_then(|p| p.dominant_relation.clone());
            (sid, emb, query, strats, model, rel)
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
                    if let Some((f, s, lam)) = dl.ewc_state("adapter") {
                        il.set_ewc_state(f, s, lam);
                    }
                }
                il.reset_adapter();
            }
        }
        if effective_quality >= 0.7 {
            if let (Some(emb), Some(text)) = (emb_for_memory, query_for_memory) {
                let _ = self.memory.add(crate::memory::NodeInput { id: None, payload: serde_json::json!({ "query": text }), embedding: emb, level: None }).await;
            }
        }
        if !strategy_ids.is_empty() {
            let _ = self.reasoning.record_outcome(&strategy_ids, effective_quality).await;
        }
        if !route_model_for_outcome.is_empty() {
            let r = self.router.lock().await;
            r.record_outcome(&route_model_for_outcome, effective_quality);
        }
        if let Some(rel) = dominant_relation.as_deref() {
            let signed = (effective_quality - 0.5) * 2.0;
            self.attention.nudge_relation(rel, signed);
        }
        if let Some(ref sid) = sid_for_ema {
            let ema = self.sessions.read().await.get(sid).map(|s| s.quality_ema).unwrap_or(0.5);
            let _ = self.store.insert_session(&crate::store::SessionRow {
                id: sid.clone(), created_at: None,
                meta: Some(serde_json::json!({ "quality_ema": ema })),
            }).await;
        }
        Ok(())
    }
}

