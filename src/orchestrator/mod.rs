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
        let bg_task = match std::env::var("RS_LEARN_BG_INTERVAL_SEC").ok().and_then(|s| s.parse::<u64>().ok()).filter(|&n| n > 0) {
            Some(secs) => Some(tokio::spawn(background.clone().schedule(Duration::from_secs(secs)))),
            None => None,
        };
        let queries = Arc::new(AtomicU64::new(0));
        let total_ms = Arc::new(AtomicU64::new(0));
        let q2 = queries.clone(); let t2 = total_ms.clone();
        let backend_name = acp.name();
        observability::register("orchestrator", move || {
            let n = q2.load(Ordering::Relaxed);
            let t = t2.load(Ordering::Relaxed);
            json!({
                "queries_count": n,
                "avg_latency_ms": if n > 0 { t as f64 / n as f64 } else { 0.0 },
                "backend": backend_name,
            })
        });
        Ok(Self {
            embedder, memory, attention, router, instant, reasoning, acp, rs_search,
            sessions: Arc::new(RwLock::new(HashMap::new())),
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
            id: sid.clone(), created_at: crate::store::now_ms(), turns: 0, last_embedding: None,
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
        let subgraph = if let Some(first) = neighbors.first() {
            self.memory.expand(&first.id, 1).await.unwrap_or_default()
        } else { Default::default() };
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

        let t_acp = Instant::now();
        let sys = format!("You are rs-learn. Task type: {}.{}{}{}",
            opts.task_type.as_deref().unwrap_or("default"), hint_text, code_text, attn_text);
        let response = self.acp.generate(&sys, text, 120_000).await.map_err(|e| anyhow!("acp: {e}"))?;
        stages.insert("acp".into(), t_acp.elapsed().as_millis() as u64);

        let t_l = Instant::now();
        let response_str = if let Value::String(s) = &response { s.clone() } else { response.to_string() };
        let latency_ms = t0.elapsed().as_millis() as u64;
        let grounding = neighbors.first().map(|n| n.score.clamp(0.0, 1.0)).unwrap_or(0.0);
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
        let loss = (1.0 - payload.quality).max(0.0);
        let boundary = {
            let mut dl = self.deep.lock().await;
            dl.record_loss(loss).await.unwrap_or(false)
        };
        {
            let mut il = self.instant.lock().await;
            let emb_before: Option<Vec<f32>> = il.pending.get(request_id).map(|p| p.embedding.clone());
            let quality = payload.quality;
            il.feedback(request_id, payload).await?;
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
        Ok(())
    }
}

fn attention_hint(ctx: &AttnContext, subgraph: &Subgraph) -> String {
    let valid: Vec<&str> = subgraph.nodes.iter()
        .filter(|n| n.embedding.as_ref().map(|e| e.len() == 768).unwrap_or(false))
        .map(|n| n.id.as_str()).collect();
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
    let top: Vec<String> = idx.iter().take(3)
        .map(|i| format!("- {} (attn={:.3})", valid[*i], mean[*i])).collect();
    format!("\n\nTop-attended context:\n{}", top.join("\n"))
}

fn implicit_quality_from(latency_ms: u64, grounding: f32, confidence: f32) -> f64 {
    let latency_score = (5000.0f64 - (latency_ms as f64).min(5000.0)) / 5000.0;
    let ground = grounding.clamp(0.0, 1.0) as f64;
    let conf = confidence.clamp(0.0, 1.0) as f64;
    let q = 0.35 * latency_score + 0.50 * ground + 0.15 * conf;
    if ground < 0.15 { return (q * 0.5).clamp(0.0, 1.0); }
    q.clamp(0.0, 1.0)
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
        let fast_grounded = implicit_quality_from(200, 0.9, 0.9);
        let slow_ungrounded = implicit_quality_from(4800, 0.05, 0.1);
        assert!(fast_grounded > 0.7, "fast grounded should be high, got {fast_grounded}");
        assert!(slow_ungrounded < 0.3, "slow ungrounded should be low, got {slow_ungrounded}");
    }
    #[test]
    fn implicit_quality_low_grounding_caps_quality() {
        let fast_ungrounded = implicit_quality_from(100, 0.05, 0.9);
        let fast_grounded = implicit_quality_from(100, 0.5, 0.9);
        assert!(fast_ungrounded < 0.4, "fast but ungrounded must be penalized, got {fast_ungrounded}");
        assert!(fast_grounded > fast_ungrounded, "grounding must dominate latency");
    }
    #[test]
    fn implicit_quality_length_does_not_affect_score() {
        let q = implicit_quality_from(1000, 0.5, 0.5);
        assert!((0.0..=1.0).contains(&q));
    }
    #[test]
    fn implicit_quality_clamps_to_unit() {
        assert!((0.0..=1.0).contains(&implicit_quality_from(0, 1.0, 1.0)));
        assert!((0.0..=1.0).contains(&implicit_quality_from(u64::MAX, -1.0, -1.0)));
    }
}
