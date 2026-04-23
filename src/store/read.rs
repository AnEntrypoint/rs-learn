use anyhow::{anyhow, Result};
use std::collections::HashMap;

use super::schema::{fts_query, vec_lit, FTS_MAP, VEC_INDEX};
use super::types::{NodeEmbeddingRow, RouterWeightsRow};
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

    pub async fn get_edges_from(&self, ids: &[String]) -> Result<Vec<super::types::EdgeRow>> {
        if ids.is_empty() { return Ok(vec![]); }
        let placeholders: Vec<String> = (1..=ids.len()).map(|i| format!("?{}", i)).collect();
        let sql = format!(
            "SELECT id, src, dst, relation, weight, created_at FROM edges WHERE src IN ({})",
            placeholders.join(",")
        );
        let args: Vec<libsql::Value> = ids.iter().map(|s| libsql::Value::Text(s.clone())).collect();
        let mut rows = self.conn.query(&sql, args).await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(super::types::EdgeRow {
                id: row.get(0)?, src: row.get(1)?, dst: row.get(2)?,
                relation: row.get(3).ok(), weight: row.get::<f64>(4).ok(),
                created_at: row.get(5).ok(), ..Default::default()
            });
        }
        Ok(out)
    }

    pub async fn get_nodes_by_ids(&self, ids: &[String]) -> Result<Vec<super::types::NodeRow>> {
        if ids.is_empty() { return Ok(vec![]); }
        let placeholders: Vec<String> = (1..=ids.len()).map(|i| format!("?{}", i)).collect();
        let sql = format!(
            "SELECT id, name, type, summary, level, created_at FROM nodes WHERE id IN ({})",
            placeholders.join(",")
        );
        let args: Vec<libsql::Value> = ids.iter().map(|s| libsql::Value::Text(s.clone())).collect();
        let mut rows = self.conn.query(&sql, args).await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(super::types::NodeRow {
                id: row.get(0)?, name: row.get(1)?, r#type: row.get(2).ok(),
                summary: row.get(3).ok(), level: row.get(4).ok(),
                group_id: None,
                created_at: row.get(5).ok(), embedding: None,
            });
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

    pub async fn get_node_embeddings(&self, ids: &[String]) -> Result<HashMap<String, NodeEmbeddingRow>> {
        let mut out = HashMap::new();
        if ids.is_empty() { return Ok(out); }
        let placeholders: Vec<String> = (1..=ids.len()).map(|i| format!("?{}", i)).collect();
        let sql = format!(
            "SELECT id, vector_extract(embedding) AS emb, created_at FROM nodes WHERE id IN ({})",
            placeholders.join(",")
        );
        let args: Vec<libsql::Value> = ids.iter().map(|s| libsql::Value::Text(s.clone())).collect();
        let mut rows = self.conn.query(&sql, args).await?;
        while let Some(row) = rows.next().await? {
            let id: String = row.get(0)?;
            let emb_str: Option<String> = row.get(1).ok();
            let created_at: i64 = row.get(2).unwrap_or(0);
            let embedding = emb_str.and_then(|s| parse_vec_text(&s));
            out.insert(id, NodeEmbeddingRow { embedding, created_at });
        }
        Ok(out)
    }

    pub async fn graph_walk(
        &self,
        start_ids: &[String],
        depth: u32,
        group_id: Option<&str>,
    ) -> Result<Vec<super::types::NodeRow>> {
        if start_ids.is_empty() || depth == 0 { return Ok(vec![]); }
        let placeholders: Vec<String> = (1..=start_ids.len()).map(|i| format!("?{}", i)).collect();
        let depth_idx = start_ids.len() + 1;
        let mut args: Vec<libsql::Value> = start_ids.iter().map(|s| libsql::Value::Text(s.clone())).collect();
        args.push(libsql::Value::Integer(depth as i64));
        let group_clause = if let Some(g) = group_id {
            args.push(libsql::Value::Text(g.to_string()));
            format!(" AND n.group_id = ?{}", depth_idx + 1)
        } else { String::new() };
        let sql = format!(
            "WITH RECURSIVE walk(id, d) AS (
                SELECT id, 0 FROM nodes WHERE id IN ({placeholders})
                UNION
                SELECT CASE WHEN e.src = w.id THEN e.dst ELSE e.src END, w.d + 1
                FROM walk w JOIN edges e ON (e.src = w.id OR e.dst = w.id)
                WHERE w.d < ?{depth_idx} AND e.expired_at IS NULL
            )
            SELECT DISTINCT n.id, n.name, n.type, n.summary, n.level, n.created_at
            FROM walk w JOIN nodes n ON n.id = w.id
            WHERE w.d > 0{group_clause}",
            placeholders = placeholders.join(","),
        );
        let mut rows = self.conn.query(&sql, args).await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(super::types::NodeRow {
                id: row.get(0)?, name: row.get(1)?,
                r#type: row.get(2).ok(), summary: row.get(3).ok(),
                level: row.get(4).ok(), created_at: row.get(5).ok(),
                group_id: None,
                embedding: None,
            });
        }
        Ok(out)
    }

    pub async fn edges_between(
        &self,
        src_ids: &[String],
        tgt_ids: &[String],
        group_id: Option<&str>,
        as_of: Option<i64>,
    ) -> Result<Vec<super::types::EdgeRow>> {
        if src_ids.is_empty() || tgt_ids.is_empty() { return Ok(vec![]); }
        let src_ph: Vec<String> = (1..=src_ids.len()).map(|i| format!("?{}", i)).collect();
        let tgt_start = src_ids.len() + 1;
        let tgt_ph: Vec<String> = (tgt_start..tgt_start + tgt_ids.len()).map(|i| format!("?{}", i)).collect();
        let mut args: Vec<libsql::Value> = src_ids.iter().chain(tgt_ids.iter())
            .map(|s| libsql::Value::Text(s.clone())).collect();
        let mut where_extra = String::new();
        let mut next_idx = tgt_start + tgt_ids.len();
        if let Some(t) = as_of {
            where_extra.push_str(&format!(" AND valid_at <= ?{} AND (expired_at IS NULL OR expired_at > ?{}) AND (invalid_at IS NULL OR invalid_at > ?{})", next_idx, next_idx + 1, next_idx + 2));
            args.push(libsql::Value::Integer(t));
            args.push(libsql::Value::Integer(t));
            args.push(libsql::Value::Integer(t));
            next_idx += 3;
        } else {
            where_extra.push_str(" AND expired_at IS NULL");
        }
        if let Some(g) = group_id {
            where_extra.push_str(&format!(" AND group_id = ?{}", next_idx));
            args.push(libsql::Value::Text(g.to_string()));
        }
        let sql = format!(
            "SELECT id, src, dst, relation, fact, weight, created_at, valid_at, invalid_at
             FROM edges
             WHERE src IN ({}) AND dst IN ({}){}",
            src_ph.join(","), tgt_ph.join(","), where_extra,
        );
        let mut rows = self.conn.query(&sql, args).await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(super::types::EdgeRow {
                id: row.get(0)?, src: row.get(1)?, dst: row.get(2)?,
                relation: row.get(3).ok(), fact: row.get(4).ok(),
                weight: row.get::<f64>(5).ok(),
                group_id: None,
                created_at: row.get(6).ok(),
                valid_at: row.get(7).ok(), invalid_at: row.get(8).ok(),
                embedding: None,
            });
        }
        Ok(out)
    }
}

fn parse_vec_text(s: &str) -> Option<Vec<f32>> {
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
