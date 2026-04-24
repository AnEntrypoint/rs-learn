use anyhow::{anyhow, Result};
use std::collections::HashMap;

use super::schema::{fts_query, vec_lit, FTS_MAP, VEC_INDEX};
use super::types::RouterWeightsRow;
use super::Store;

pub struct VectorFilter<'a> {
    pub sql: &'a str,
    pub args: Vec<libsql::Value>,
}

impl Store {
    pub async fn vector_top_k(
        &self,
        table: &str,
        query: &[f32],
        k: usize,
        filter: Option<VectorFilter<'_>>,
    ) -> Result<Vec<HashMap<String, libsql::Value>>> {
        let (_, col, idx) = VEC_INDEX.iter().find(|(t, _, _)| *t == table)
            .ok_or_else(|| anyhow!("vector_top_k: unknown table '{}'", table))?;
        let k_int = k.max(1);
        let fetch = k_int * 4;
        let lit = vec_lit(Some(query));
        let (where_clause, args) = match filter {
            Some(f) => (format!("WHERE {}", f.sql), f.args),
            None => (String::new(), vec![]),
        };
        let sql = format!(
            "SELECT b.*, vector_distance_cos(b.{col}, {lit}) AS dist
             FROM vector_top_k('{idx}', {lit}, {fetch}) AS t
             JOIN {table} b ON b.rowid = t.id
             {where_clause}
             ORDER BY dist ASC LIMIT {k_int}"
        );
        let mut rows = self.conn.query(&sql, args).await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(row_to_map(&row));
        }
        Ok(out)
    }

    pub async fn fts_search(
        &self,
        table: &str,
        query: &str,
        k: usize,
    ) -> Result<Vec<HashMap<String, libsql::Value>>> {
        let (_, fts, base, key) = FTS_MAP.iter().find(|(t, _, _, _)| *t == table)
            .ok_or_else(|| anyhow!("fts_search: unknown table '{}'", table))?;
        let sql = format!(
            "SELECT b.*, bm25({fts}) AS score
             FROM {fts} f JOIN {base} b ON b.{key} = f.{key}
             WHERE {fts} MATCH ?1
             ORDER BY score LIMIT ?2"
        );
        let mut rows = self.conn.query(&sql, libsql::params![fts_query(query), k as i64]).await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(row_to_map(&row));
        }
        Ok(out)
    }

    pub async fn load_latest_router_weights(&self) -> Result<Option<RouterWeightsRow>> {
        let mut rows = self.conn.query(
            "SELECT version, blob, algo, created_at, meta FROM router_weights ORDER BY version DESC LIMIT 1",
            (),
        ).await?;
        let Some(row) = rows.next().await? else { return Ok(None) };
        let version: i64 = row.get(0)?;
        let blob: Vec<u8> = row.get::<Vec<u8>>(1)?;
        let algo: Option<String> = row.get(2).ok();
        let created_at: i64 = row.get(3)?;
        let meta_str: String = row.get::<String>(4).unwrap_or_else(|_| "{}".into());
        let meta: serde_json::Value = serde_json::from_str(&meta_str).unwrap_or_else(|_| serde_json::json!({}));
        Ok(Some(RouterWeightsRow { version, blob, algo, created_at, meta }))
    }

    pub async fn load_fisher_vec(&self, param_id: &str) -> Result<Vec<f32>> {
        let prefix = format!("{}:", param_id);
        let like = format!("{}%", prefix);
        let mut rows = self.conn.query(
            "SELECT param_id, value FROM ewc_fisher WHERE param_id LIKE ?1",
            libsql::params![like],
        ).await?;
        let mut pairs: Vec<(usize, f32)> = Vec::new();
        while let Some(row) = rows.next().await? {
            let id: String = row.get(0)?;
            let v: f64 = row.get(1)?;
            let Some(idx_str) = id.strip_prefix(&prefix) else { continue };
            let Ok(idx) = idx_str.parse::<usize>() else { continue };
            pairs.push((idx, v as f32));
        }
        pairs.sort_by_key(|(i, _)| *i);
        let n = pairs.last().map(|(i, _)| *i + 1).unwrap_or(0);
        let mut out = vec![0f32; n];
        for (i, v) in pairs { out[i] = v; }
        Ok(out)
    }

    pub async fn load_params_snapshot_vec(&self, param_id: &str) -> Result<Vec<f32>> {
        let prefix = format!("snap:{}:", param_id);
        let like = format!("{}%", prefix);
        let mut rows = self.conn.query(
            "SELECT param_id, value FROM ewc_fisher WHERE param_id LIKE ?1",
            libsql::params![like],
        ).await?;
        let mut pairs: Vec<(usize, f32)> = Vec::new();
        while let Some(row) = rows.next().await? {
            let id: String = row.get(0)?;
            let v: f64 = row.get(1)?;
            let Some(idx_str) = id.strip_prefix(&prefix) else { continue };
            let Ok(idx) = idx_str.parse::<usize>() else { continue };
            pairs.push((idx, v as f32));
        }
        pairs.sort_by_key(|(i, _)| *i);
        let n = pairs.last().map(|(i, _)| *i + 1).unwrap_or(0);
        let mut out = vec![0f32; n];
        for (i, v) in pairs { out[i] = v; }
        Ok(out)
    }

    pub async fn load_fisher(&self) -> Result<HashMap<String, f64>> {
        let mut rows = self.conn.query("SELECT param_id, value FROM ewc_fisher", ()).await?;
        let mut out = HashMap::new();
        while let Some(row) = rows.next().await? {
            let id: String = row.get(0)?;
            let v: f64 = row.get(1)?;
            out.insert(id, v);
        }
        Ok(out)
    }

    pub async fn list_recent_trajectories_with_embeddings(&self, limit: usize) -> Result<Vec<super::types::TrajectoryRow>> {
        let sql = "SELECT id, session_id, query, vector_extract(query_embedding) AS emb, router_decision, quality, latency_ms, created_at
                   FROM trajectories
                   WHERE query_embedding IS NOT NULL
                   ORDER BY created_at DESC LIMIT ?1";
        let mut rows = self.conn.query(sql, libsql::params![limit as i64]).await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            let emb_str: Option<String> = row.get(3).ok();
            let embedding = emb_str.and_then(|s| parse_vec_text(&s));
            out.push(super::types::TrajectoryRow {
                id: row.get(0)?,
                session_id: row.get(1).ok(),
                query: row.get(2).ok(),
                query_embedding: embedding,
                retrieved_ids: None,
                router_decision: row.get(4).ok(),
                response: None,
                activations: None,
                quality: row.get::<f64>(5).ok(),
                latency_ms: row.get::<i64>(6).ok(),
                created_at: row.get(7).ok(),
            });
        }
        Ok(out)
    }

}

pub(super) fn parse_vec_text(s: &str) -> Option<Vec<f32>> {
    let trimmed = s.trim().trim_start_matches('[').trim_end_matches(']');
    let parts: Result<Vec<f32>, _> = trimmed.split(',').map(|p| p.trim().parse::<f32>()).collect();
    parts.ok()
}

fn row_to_map(row: &libsql::Row) -> HashMap<String, libsql::Value> {
    let mut map = HashMap::new();
    let n = row.column_count();
    for i in 0..n {
        let name = row.column_name(i).map(|s| s.to_string()).unwrap_or_else(|| format!("c{}", i));
        if let Ok(v) = row.get_value(i) {
            map.insert(name, v);
        }
    }
    map
}
