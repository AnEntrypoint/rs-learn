use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

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
}

impl ReasoningBank {
    pub fn new(store: Arc<Store>) -> Self {
        Self { store }
    }

    pub async fn retrieve_for_query(&self, query: &str, k: usize) -> Result<Vec<Strategy>> {
        let rows = self.store.fts_search("reasoning_bank", query, k.max(1)).await?;
        Ok(rows.iter().filter_map(row_to_strategy).collect())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{Store, ReasoningRow};

    #[tokio::test]
    async fn retrieve_and_top() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("rb.db");
        let store = Arc::new(Store::open(path.to_str().unwrap()).await?);

        store.insert_reasoning(&ReasoningRow {
            id: "s1".into(),
            pattern_id: Some("p1".into()),
            strategy: "decompose complex queries into substeps".into(),
            success_rate: Some(0.9),
            created_at: None,
        }).await?;
        store.insert_reasoning(&ReasoningRow {
            id: "s2".into(),
            pattern_id: None,
            strategy: "cache frequent lookups".into(),
            success_rate: Some(0.5),
            created_at: None,
        }).await?;

        let bank = ReasoningBank::new(store.clone());
        let hits = bank.retrieve_for_query("decompose", 5).await?;
        assert!(hits.iter().any(|s| s.id == "s1"), "fts should return s1");

        let top = bank.top_strategies(10).await?;
        assert_eq!(top.first().map(|s| s.id.as_str()), Some("s1"));
        assert!(top[0].success_rate >= top[1].success_rate);
        Ok(())
    }
}
