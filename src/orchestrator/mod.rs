mod types;

pub use types::{QueryOpts, QueryResult, RouteSnapshot, Session};

use crate::backend::{self, AgentBackend};
use crate::attention::{Attention, Context as AttnContext, Subgraph};
use crate::cache::EmbeddingCache;
use crate::embeddings::Embedder;
use crate::learn::background::BackgroundLoop;
use crate::learn::deep::DeepLoop;
use crate::learn::instant::{FeedbackPayload, InstantLoop};
use crate::learn::reasoning_bank::ReasoningBank;
use crate::memory::Memory;
use crate::observability;
use crate::router::{RouteCtx, Router};
use crate::rs_search_bridge::RsSearch;
use crate::spine::TrajectorySpine;
use crate::store::Store;
use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tokio::task::JoinHandle;

pub struct Orchestrator {
    pub embedder: Arc<Embedder>,
    pub memory: Arc<Memory>,
    pub attention: Arc<Attention>,
    pub router: Arc<Mutex<Router>>,
    pub instant: Arc<Mutex<InstantLoop>>,
    pub reasoning: Arc<ReasoningBank>,
    pub acp: Arc<dyn AgentBackend>,
    pub rs_search: Option<Arc<RsSearch>>,
    pub sessions: Arc<RwLock<HashMap<String, Session>>>,
    pub store: Arc<Store>,
    pub search_root: PathBuf,
    pub embed_cache: Option<Arc<EmbeddingCache>>,
    pub spine: Option<Arc<TrajectorySpine>>,
    pub background: Option<Arc<BackgroundLoop>>,
    pub deep: Arc<Mutex<DeepLoop>>,
    bg_task: std::sync::Mutex<Option<JoinHandle<()>>>,
    queries: Arc<AtomicU64>,
    total_ms: Arc<AtomicU64>,
}

impl Orchestrator {
    pub async fn new_default() -> Result<Self> {
        let db_path = std::env::var("RS_LEARN_DB_PATH").unwrap_or_else(|_| "./rs-learn.db".into());
        let store = Arc::new(Store::open(&db_path).await?);
        let targets: Vec<String> = std::env::var("RS_LEARN_TARGETS")
            .unwrap_or_else(|_| "default".into())
            .split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
        let embedder = Arc::new(Embedder::new());
        let memory = Arc::new(Memory::new(store.clone()));
        let attention = Arc::new(Attention::new(store.clone()));
        let router = Arc::new(Mutex::new(Router::new(store.clone(), targets.clone())));
        let instant = Arc::new(Mutex::new(InstantLoop::new(store.clone(), router.clone(), targets.clone())));
        let reasoning = Arc::new(ReasoningBank::with_embedder(store.clone(), embedder.clone()));
        let acp = backend::from_env().map_err(|e| anyhow!("backend: {e}"))?;
        let rs_search = if std::env::var("RS_LEARN_CODE_SEARCH").is_ok() { Some(Arc::new(RsSearch::new())) } else { None };
        let search_root: PathBuf = std::env::var("RS_LEARN_SEARCH_ROOT").map(PathBuf::from)
            .unwrap_or_else(|_| std::env::current_dir().unwrap_or_default());
        let embed_cache = if std::env::var("RS_LEARN_EMBED_CACHE").ok().as_deref() == Some("1") {
            Some(EmbeddingCache::new(embedder.clone(), 10_000, std::time::Duration::from_secs(3600)))
        } else { None };
        let spine = if std::env::var("RS_LEARN_ASYNC_TRAJECTORY").ok().as_deref() == Some("1") {
            Some(TrajectorySpine::new(store.clone(), crate::spine::DEFAULT_CAPACITY))
        } else { None };
        if let Some(s) = spine.as_ref() {
            let mut il = instant.lock().await;
            il.spine = Some(s.clone());
        }
        let background = BackgroundLoop::new(store.clone(), router.clone(), Some(acp.clone()), reasoning.clone(), Some(instant.clone()));
        let deep = Arc::new(Mutex::new(DeepLoop::new(store.clone())));
        { let mut dl = deep.lock().await; let _ = dl.load_fisher("adapter").await; }
        let bg_task = match std::env::var("RS_LEARN_BG_INTERVAL_SEC").ok().and_then(|s| s.parse::<u64>().ok()).filter(|&n| n > 0) {
            Some(secs) => Some(tokio::spawn(background.clone().schedule(Duration::from_secs(secs)))),
            None => None,
        };
        let sessions: Arc<RwLock<HashMap<String, Session>>> = Arc::new(RwLock::new(HashMap::new()));
        let queries = Arc::new(AtomicU64::new(0));
        let total_ms = Arc::new(AtomicU64::new(0));
        let q2 = queries.clone(); let t2 = total_ms.clone();
        let backend_name = acp.name();
        let sessions_obs = Arc::clone(&sessions);
        observability::register("orchestrator", move || {
            let n = q2.load(Ordering::Relaxed);
            let t = t2.load(Ordering::Relaxed);
            let sc = sessions_obs.try_read().map(|m| m.len()).unwrap_or(0);
            json!({
                "queries_count": n,
                "avg_latency_ms": if n > 0 { t as f64 / n as f64 } else { 0.0 },
                "backend": backend_name,
                "session_count": sc,
            })
        });
        Ok(Self {
            embedder, memory, attention, router, instant, reasoning, acp, rs_search,
            sessions,
            store, search_root, embed_cache, spine,
            background: Some(background),
            deep,
            bg_task: std::sync::Mutex::new(bg_task),
            queries, total_ms,
        })
    }

    async fn session(&self, id: Option<String>) -> String {
        let sid = id.unwrap_or_else(|| format!("sess-{}", &uuid::Uuid::new_v4().to_string()[..8]));
        let mut map = self.sessions.write().await;
        let entry = map.entry(sid.clone()).or_insert_with(|| Session {
            id: sid.clone(), created_at: crate::store::now_ms(), turns: 0, last_embedding: None, quality_ema: 0.5,
        });
        entry.turns += 1;
        sid
    }

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
            format!("\n\nReasoning hints:\n{}",
                hints.iter().map(|h| format!("- {}", h.strategy)).collect::<Vec<_>>().join("\n"))
        };
        let code_text = match (opts.include_code_search, self.rs_search.as_ref()) {
            (true, Some(rs)) => {
                let hits = rs.search(text, &self.search_root);
                if hits.is_empty() { String::new() } else {
                    format!("\n\nCode context:\n{}", hits.iter().take(3)
                        .map(|h| format!("- {}:{}-{}", h.file, h.line_start, h.line_end))
                        .collect::<Vec<_>>().join("\n"))
                }
            }
            _ => String::new(),
        };
        stages.insert("hints".into(), t_h.elapsed().as_millis() as u64);

        let memory_text = if neighbors.is_empty() { String::new() } else {
            let items: Vec<String> = neighbors.iter().take(3)
                .filter(|n| !n.payload.is_empty())
                .map(|n| format!("- {}", n.payload))
                .collect();
            if items.is_empty() { String::new() } else { format!("\n\nMemory context:\n{}", items.join("\n")) }
        };
        let t_acp = Instant::now();
        let sys = format!("You are rs-learn. Task type: {}.{}{}{}{}",
            opts.task_type.as_deref().unwrap_or("default"), memory_text, hint_text, code_text, attn_text);
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

        Ok(QueryResult {
            text: response, request_id, session_id: sid,
            routing: snapshot, retrieved: neighbors, confidence, latency_ms, stage_breakdown: stages,
        })
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
        let boundary = {
            let mut dl = self.deep.lock().await;
            dl.record_loss(loss).await.unwrap_or(false)
        };
        {
            let mut il = self.instant.lock().await;
            let emb_before: Option<Vec<f32>> = il.pending.get(request_id).map(|p| p.embedding.clone());
            let quality = effective_quality;
            il.feedback(request_id, smoothed_payload).await?;
            if boundary {
                if let Some(emb) = emb_before {
                    let flat = il.serialize_adapter_flat();
                    let grads: Vec<f32> = flat.iter().enumerate()
                        .map(|(i, _)| emb[i % emb.len()] * quality).collect();
                    let mut dl = self.deep.lock().await;
                    let _ = dl.consolidate("adapter", &flat, &grads).await;
                }
                il.reset_adapter();
            }
        }
        if effective_quality >= 0.7 {
            if let (Some(emb), Some(text)) = (emb_for_memory, query_for_memory) {
                let _ = self.memory.add(crate::memory::NodeInput {
                    id: None,
                    payload: serde_json::json!({ "query": text }),
                    embedding: emb,
                    level: None,
                }).await;
            }
        }
        Ok(())
    }
}

fn attention_hint(ctx: &AttnContext, subgraph: &Subgraph) -> String {
    let valid: Vec<(usize, &crate::attention::SubgraphNode)> = subgraph.nodes.iter().enumerate()
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
    let top: Vec<String> = idx.iter().take(3).map(|i| {
        let node = valid[*i].1;
        format!("- {} (attn={:.3})", node.id, mean[*i])
    }).collect();
    format!("\n\nTop-attended context:\n{}", top.join("\n"))
}

fn implicit_quality_from(latency_ms: u64, grounding: Option<f32>, confidence: f32) -> f64 {
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

impl Drop for Orchestrator {
    fn drop(&mut self) {
        if let Ok(mut g) = self.bg_task.lock() {
            if let Some(h) = g.take() { h.abort(); }
        }
        observability::unregister("orchestrator");
    }
}

#[cfg(test)]
mod tests {
    use super::{attention_hint, implicit_quality_from};
    use crate::attention::{Context as AttnContext, Subgraph, SubgraphNode};

    fn node(id: &str) -> SubgraphNode {
        SubgraphNode { id: id.into(), embedding: Some(vec![0.0; 768]), created_at: Some(0) }
    }

    #[test]
    fn attention_hint_empty_subgraph_no_panic() {
        let ctx = AttnContext { vector: vec![0.0; 768], weights: vec![] };
        let sg = Subgraph::default();
        assert_eq!(attention_hint(&ctx, &sg), "");
    }

    #[test]
    fn attention_hint_changes_prompt_vs_baseline() {
        let sg = Subgraph { nodes: vec![node("alpha"), node("beta"), node("gamma")], edges: vec![] };
        let ctx = AttnContext {
            vector: vec![0.0; 768],
            weights: vec![vec![0.1, 0.7, 0.2], vec![0.2, 0.6, 0.2]],
        };
        let baseline = AttnContext { vector: vec![0.0; 768], weights: vec![] };
        let with_attn = attention_hint(&ctx, &sg);
        let without = attention_hint(&baseline, &sg);
        assert_ne!(with_attn, without);
        assert!(with_attn.contains("Top-attended context:"));
        assert!(with_attn.contains("beta"));
        let beta_pos = with_attn.find("beta").unwrap();
        let alpha_pos = with_attn.find("alpha").unwrap();
        assert!(beta_pos < alpha_pos, "beta (highest weight) must rank first");
    }

    #[test]
    fn implicit_quality_rewards_fast_grounded_responses() {
        let fast_grounded = implicit_quality_from(200, Some(0.9), 0.9);
        let slow_ungrounded = implicit_quality_from(4800, Some(0.05), 0.1);
        assert!(fast_grounded > 0.7, "fast grounded should be high, got {fast_grounded}");
        assert!(slow_ungrounded < 0.3, "slow ungrounded should be low, got {slow_ungrounded}");
    }
    #[test]
    fn implicit_quality_low_grounding_caps_quality() {
        let fast_ungrounded = implicit_quality_from(100, Some(0.05), 0.9);
        let fast_grounded = implicit_quality_from(100, Some(0.5), 0.9);
        assert!(fast_ungrounded < 0.4, "fast but ungrounded must be penalized, got {fast_ungrounded}");
        assert!(fast_grounded > fast_ungrounded, "grounding must dominate latency");
    }
    #[test]
    fn implicit_quality_length_does_not_affect_score() {
        let q = implicit_quality_from(1000, Some(0.5), 0.5);
        assert!((0.0..=1.0).contains(&q));
    }
    #[test]
    fn implicit_quality_clamps_to_unit() {
        assert!((0.0..=1.0).contains(&implicit_quality_from(0, Some(1.0), 1.0)));
        assert!((0.0..=1.0).contains(&implicit_quality_from(u64::MAX, Some(-1.0), -1.0)));
    }
    #[test]
    fn implicit_quality_cold_start_no_penalty() {
        let cold = implicit_quality_from(1000, None, 0.8);
        let grounded = implicit_quality_from(1000, Some(0.8), 0.8);
        assert!((0.0..=1.0).contains(&cold), "cold start must be in range, got {cold}");
        assert!(cold > 0.3, "cold start must not be excessively penalized, got {cold}");
        assert!(grounded > cold, "grounded response should score higher than cold start");
    }

    #[test]
    fn session_ema_smooths_outlier_feedback() {
        let default_ema: f32 = 0.5;
        let raw_quality: f32 = 1.0;
        let expected = 0.7 * raw_quality + 0.3 * default_ema;
        let smoothed = 0.7 * raw_quality + 0.3 * default_ema;
        assert!((smoothed - expected).abs() < 1e-6, "EMA formula mismatch: {smoothed} vs {expected}");
        assert!((smoothed - 0.85).abs() < 1e-6, "first feedback on default EMA(0.5) with quality=1.0 should yield 0.85, got {smoothed}");
        let mut ema = default_ema;
        for &q in &[1.0f32, 0.0, 1.0, 0.0] {
            ema = 0.7 * q + 0.3 * ema;
        }
        assert!((0.0..=1.0).contains(&ema), "EMA must stay in [0,1], got {ema}");
    }
}
