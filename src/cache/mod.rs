use crate::embeddings::Embedder;
use crate::observability;
use anyhow::Result;
use moka::future::Cache;
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

pub struct EmbeddingCache {
    inner: Cache<String, Arc<Vec<f32>>>,
    embedder: Arc<Embedder>,
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
}

impl EmbeddingCache {
    pub fn new(embedder: Arc<Embedder>, capacity: u64, ttl: Duration) -> Arc<Self> {
        let inner = Cache::builder()
            .max_capacity(capacity)
            .time_to_live(ttl)
            .build();
        let hits = Arc::new(AtomicU64::new(0));
        let misses = Arc::new(AtomicU64::new(0));
        let s = Arc::new(Self { inner, embedder, hits: hits.clone(), misses: misses.clone() });
        let h2 = hits.clone(); let m2 = misses.clone();
        observability::register("embed_cache", move || {
            let h = h2.load(Ordering::Relaxed);
            let m = m2.load(Ordering::Relaxed);
            let total = h + m;
            json!({
                "hits": h,
                "misses": m,
                "hit_rate": if total > 0 { h as f64 / total as f64 } else { 0.0 },
            })
        });
        s
    }

    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let key = text.to_string();
        if let Some(arc) = self.inner.get(&key).await {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok((*arc).clone());
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        let embedder = self.embedder.clone();
        let key_for_init = key.clone();
        let arc = self.inner.get_with(key, async move {
            Arc::new(embedder.embed(&key_for_init).unwrap_or_default())
        }).await;
        Ok((*arc).clone())
    }

    pub fn hits(&self) -> u64 { self.hits.load(Ordering::Relaxed) }
    pub fn misses(&self) -> u64 { self.misses.load(Ordering::Relaxed) }
}

impl Drop for EmbeddingCache {
    fn drop(&mut self) { observability::unregister("embed_cache"); }
}
