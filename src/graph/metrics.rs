use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

pub struct GraphMetrics {
    pub episodes_ingested: AtomicU64,
    pub nodes_upserted: AtomicU64,
    pub edges_upserted: AtomicU64,
    pub search_calls: AtomicU64,
    pub search_total_ms: AtomicU64,
    pub llm_calls: AtomicU64,
    pub llm_total_ms: AtomicU64,
    pub dedup_candidates_seen: AtomicU64,
}

impl GraphMetrics {
    const fn new() -> Self {
        Self {
            episodes_ingested: AtomicU64::new(0),
            nodes_upserted: AtomicU64::new(0),
            edges_upserted: AtomicU64::new(0),
            search_calls: AtomicU64::new(0),
            search_total_ms: AtomicU64::new(0),
            llm_calls: AtomicU64::new(0),
            llm_total_ms: AtomicU64::new(0),
            dedup_candidates_seen: AtomicU64::new(0),
        }
    }
}

static METRICS: OnceLock<GraphMetrics> = OnceLock::new();

pub fn metrics() -> &'static GraphMetrics {
    METRICS.get_or_init(GraphMetrics::new)
}

pub fn incr(field: &AtomicU64, by: u64) { field.fetch_add(by, Ordering::Relaxed); }

pub fn snapshot() -> Value {
    let m = metrics();
    let n = m.search_calls.load(Ordering::Relaxed).max(1);
    let l = m.llm_calls.load(Ordering::Relaxed).max(1);
    json!({
        "episodes_ingested": m.episodes_ingested.load(Ordering::Relaxed),
        "nodes_upserted": m.nodes_upserted.load(Ordering::Relaxed),
        "edges_upserted": m.edges_upserted.load(Ordering::Relaxed),
        "search_calls": m.search_calls.load(Ordering::Relaxed),
        "search_avg_ms": m.search_total_ms.load(Ordering::Relaxed) as f64 / n as f64,
        "llm_calls": m.llm_calls.load(Ordering::Relaxed),
        "llm_avg_ms": m.llm_total_ms.load(Ordering::Relaxed) as f64 / l as f64,
        "dedup_candidates_seen": m.dedup_candidates_seen.load(Ordering::Relaxed),
    })
}

pub fn register() {
    crate::observability::register("graph", || snapshot());
}
