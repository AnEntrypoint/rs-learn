mod hints;
mod pipeline;
mod types;
#[cfg(test)]
mod tests;

pub use types::{QueryOpts, QueryResult, RouteSnapshot, Session};

use crate::backend::{self, AgentBackend};
use crate::attention::Attention;
use crate::cache::EmbeddingCache;
use crate::embeddings::Embedder;
use crate::learn::background::BackgroundLoop;
use crate::learn::deep::DeepLoop;
use crate::learn::instant::InstantLoop;
use crate::learn::reasoning_bank::ReasoningBank;
use crate::memory::Memory;
use crate::observability;
use crate::router::Router;
use crate::rs_search_bridge::RsSearch;
use crate::spine::TrajectorySpine;
use crate::store::Store;
use anyhow::{anyhow, Result};
use serde_json::json;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
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
        { let mut r = router.lock().await; let _ = r.load().await; }
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

    pub(super) async fn session(&self, id: Option<String>) -> String {
        use crate::store::now_ms;
        use uuid::Uuid;
        let sid = id.unwrap_or_else(|| format!("sess-{}", &Uuid::new_v4().to_string()[..8]));
        {
            let map = self.sessions.read().await;
            if map.contains_key(&sid) {
                drop(map);
                let mut map = self.sessions.write().await;
                if let Some(s) = map.get_mut(&sid) { s.turns += 1; }
                return sid;
            }
        }
        let loaded = self.store.load_session(&sid).await.ok().flatten();
        let (created_at, ema) = match &loaded {
            Some(row) => {
                let ema = row.meta.as_ref()
                    .and_then(|m| m.get("quality_ema"))
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
                    .unwrap_or(0.5);
                (row.created_at.unwrap_or_else(now_ms), ema)
            }
            None => (now_ms(), 0.5),
        };
        let mut map = self.sessions.write().await;
        let entry = map.entry(sid.clone()).or_insert(Session {
            id: sid.clone(), created_at, turns: 0, last_embedding: None, quality_ema: ema,
        });
        entry.turns += 1;
        sid
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
