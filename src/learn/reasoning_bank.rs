use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::embeddings::Embedder;
use crate::store::Store;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Strategy {
    pub id: String,
    pub pattern_id: Option<String>,
    pub strategy: String,
    pub success_rate: f64,
    pub created_at: i64,
    pub score: Option<f64>,
}

fn get_text(m: &HashMap<String, libsql::Value>, key: &str) -> Option<String> {
    match m.get(key)? {
        libsql::Value::Text(s) => Some(s.clone()),
        libsql::Value::Null => None,
        _ => None,
    }
}

fn get_int(m: &HashMap<String, libsql::Value>, key: &str) -> i64 {
    match m.get(key) {
        Some(libsql::Value::Integer(i)) => *i,
        Some(libsql::Value::Real(r)) => *r as i64,
        _ => 0,
    }
}

fn get_real(m: &HashMap<String, libsql::Value>, key: &str) -> f64 {
    match m.get(key) {
        Some(libsql::Value::Real(r)) => *r,
        Some(libsql::Value::Integer(i)) => *i as f64,
        _ => 0.0,
    }
}

fn rrf_fuse(a: Vec<Strategy>, b: Vec<Strategy>, k: usize) -> Vec<Strategy> {
    let rrf_k = 60.0f64;
    let mut scores: HashMap<String, f64> = HashMap::new();
    let mut by_id: HashMap<String, Strategy> = HashMap::new();
    for (i, s) in a.into_iter().enumerate() {
        *scores.entry(s.id.clone()).or_insert(0.0) += 1.0 / (rrf_k + (i as f64) + 1.0);
        by_id.entry(s.id.clone()).or_insert(s);
    }
    for (i, s) in b.into_iter().enumerate() {
        *scores.entry(s.id.clone()).or_insert(0.0) += 1.0 / (rrf_k + (i as f64) + 1.0);
        by_id.entry(s.id.clone()).or_insert(s);
    }
    let mut pairs: Vec<(String, f64)> = scores.into_iter().collect();
    pairs.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs.into_iter().take(k).filter_map(|(id, sc)| {
        by_id.remove(&id).map(|mut s| { s.score = Some(sc); s })
    }).collect()
}

fn row_to_strategy(m: &HashMap<String, libsql::Value>) -> Option<Strategy> {
    let id = get_text(m, "id")?;
    let strategy = get_text(m, "strategy").unwrap_or_default();
    let score = match m.get("score") {
        Some(libsql::Value::Real(r)) => Some(*r),
        Some(libsql::Value::Integer(i)) => Some(*i as f64),
        _ => None,
    };
    Some(Strategy {
        id,
        pattern_id: get_text(m, "pattern_id"),
        strategy,
        success_rate: get_real(m, "success_rate"),
        created_at: get_int(m, "created_at"),
        score,
    })
}

pub struct ReasoningBank {
    store: Arc<Store>,
    embedder: Option<Arc<Embedder>>,
}

impl ReasoningBank {
    pub fn new(store: Arc<Store>) -> Self {
        Self { store, embedder: None }
    }

    pub fn with_embedder(store: Arc<Store>, embedder: Arc<Embedder>) -> Self {
        Self { store, embedder: Some(embedder) }
    }

    pub async fn retrieve_for_query(&self, query: &str, k: usize) -> Result<Vec<Strategy>> {
        let k = k.max(1);
        let fetch = (k * 3).max(5);
        let fts_rows = self.store.fts_search("reasoning_bank", query, fetch).await.unwrap_or_default();
        let fts: Vec<Strategy> = fts_rows.iter().filter_map(row_to_strategy).collect();
        let semantic: Vec<Strategy> = match self.embedder.as_ref() {
            Some(emb) => self.semantic_hits(emb, query, fetch).await.unwrap_or_default(),
            None => Vec::new(),
        };
        Ok(rrf_fuse(fts, semantic, k))
    }

    async fn semantic_hits(&self, embedder: &Embedder, query: &str, k: usize) -> Result<Vec<Strategy>> {
        let emb = embedder.embed(query).unwrap_or_default();
        if emb.is_empty() { return Ok(vec![]); }
        let pat_rows = self.store.vector_top_k("patterns", &emb, k, None).await?;
        let mut ids: Vec<String> = Vec::new();
        for r in &pat_rows {
            if let Some(libsql::Value::Text(s)) = r.get("id") { ids.push(s.clone()); }
        }
        if ids.is_empty() { return Ok(vec![]); }
        let placeholders: Vec<String> = (1..=ids.len()).map(|i| format!("?{i}")).collect();
        let sql = format!(
            "SELECT id, pattern_id, strategy, success_rate, created_at FROM reasoning_bank WHERE pattern_id IN ({}) ORDER BY success_rate DESC",
            placeholders.join(",")
        );
        let params: Vec<libsql::Value> = ids.into_iter().map(libsql::Value::Text).collect();
        let mut rows = self.store.conn.query(&sql, params).await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(Strategy {
                id: row.get(0).unwrap_or_default(),
                pattern_id: row.get(1).ok(),
                strategy: row.get(2).unwrap_or_default(),
                success_rate: row.get::<f64>(3).unwrap_or(0.0),
                created_at: row.get::<i64>(4).unwrap_or(0),
                score: None,
            });
        }
        Ok(out)
    }

    pub async fn record_outcome(&self, strategy_ids: &[String], observed_quality: f32) -> Result<()> {
        let alpha = 0.2f64;
        let q = observed_quality.clamp(0.0, 1.0) as f64;
        for id in strategy_ids {
            let prior = self.store.get_reasoning_success_rate(id).await?.unwrap_or(0.5);
            let new_rate = (1.0 - alpha) * prior + alpha * q;
            self.store.update_reasoning_success_rate(id, new_rate).await?;
        }
        Ok(())
    }

    pub async fn top_strategies(&self, limit: usize) -> Result<Vec<Strategy>> {
        let lim = limit.max(1) as i64;
        let mut rows = self.store.conn.query(
            "SELECT id, pattern_id, strategy, success_rate, created_at FROM reasoning_bank ORDER BY success_rate DESC LIMIT ?1",
            libsql::params![lim],
        ).await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            let id: String = row.get(0)?;
            let pattern_id: Option<String> = row.get(1).ok();
            let strategy: String = row.get::<String>(2).unwrap_or_default();
            let success_rate: f64 = row.get::<f64>(3).unwrap_or(0.0);
            let created_at: i64 = row.get::<i64>(4).unwrap_or(0);
            out.push(Strategy { id, pattern_id, strategy, success_rate, created_at, score: None });
        }
        Ok(out)
    }
}

#[cfg(test)] #[path = "reasoning_bank_tests.rs"] mod tests;
