mod types;

pub use types::{QueryOpts, QueryResult, RouteSnapshot, Session};

use crate::backend::{self, AgentBackend};
use crate::attention::Attention;
use crate::cache::EmbeddingCache;
use crate::embeddings::Embedder;
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
use std::time::Instant;
use tokio::sync::{Mutex, RwLock};

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
            store, search_root, embed_cache, spine, queries, total_ms,
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
        let _ = self.attention.attend(&emb, &subgraph).ok();
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
        let sys = format!("You are rs-learn. Task type: {}.{}{}",
            opts.task_type.as_deref().unwrap_or("default"), hint_text, code_text);
        let response = self.acp.generate(&sys, text, 120_000).await.map_err(|e| anyhow!("acp: {e}"))?;
        stages.insert("acp".into(), t_acp.elapsed().as_millis() as u64);

        let t_l = Instant::now();
        let response_str = if let Value::String(s) = &response { s.clone() } else { response.to_string() };
        let request_id = {
            let mut il = self.instant.lock().await;
            il.record_trajectory(Some(sid.clone()), emb.clone(), route_model, response_str).await?
        };
        stages.insert("learn".into(), t_l.elapsed().as_millis() as u64);

        { let mut map = self.sessions.write().await;
          if let Some(s) = map.get_mut(&sid) { s.last_embedding = Some(emb); } }
        let latency_ms = t0.elapsed().as_millis() as u64;
        self.queries.fetch_add(1, Ordering::Relaxed);
        self.total_ms.fetch_add(latency_ms, Ordering::Relaxed);

        Ok(QueryResult {
            text: response, request_id, session_id: sid,
            routing: snapshot, retrieved: neighbors, confidence, latency_ms, stage_breakdown: stages,
        })
    }

    pub async fn feedback(&self, request_id: &str, payload: FeedbackPayload) -> Result<()> {
        let mut il = self.instant.lock().await;
        il.feedback(request_id, payload).await
    }
}

impl Drop for Orchestrator {
    fn drop(&mut self) { observability::unregister("orchestrator"); }
}
