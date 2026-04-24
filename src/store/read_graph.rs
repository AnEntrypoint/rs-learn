use anyhow::Result;
use std::collections::HashMap;

use super::types::{EdgeRow, NodeEmbeddingRow, NodeRow};
use super::Store;

impl Store {
    pub async fn get_edges_from(&self, ids: &[String]) -> Result<Vec<EdgeRow>> {
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
            out.push(EdgeRow {
                id: row.get(0)?, src: row.get(1)?, dst: row.get(2)?,
                relation: row.get(3).ok(), weight: row.get::<f64>(4).ok(),
                created_at: row.get(5).ok(), ..Default::default()
            });
        }
        Ok(out)
    }

    pub async fn get_nodes_by_ids(&self, ids: &[String]) -> Result<Vec<NodeRow>> {
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
            out.push(NodeRow {
                id: row.get(0)?, name: row.get(1)?, r#type: row.get(2).ok(),
                summary: row.get(3).ok(), level: row.get(4).ok(),
                group_id: None,
                created_at: row.get(5).ok(), embedding: None,
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
            let embedding = emb_str.and_then(|s| super::read::parse_vec_text(&s));
            out.insert(id, NodeEmbeddingRow { embedding, created_at });
        }
        Ok(out)
    }

    pub async fn graph_walk(
        &self,
        start_ids: &[String],
        depth: u32,
        group_id: Option<&str>,
    ) -> Result<Vec<NodeRow>> {
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
            out.push(NodeRow {
                id: row.get(0)?, name: row.get(1)?,
                r#type: row.get(2).ok(), summary: row.get(3).ok(),
                level: row.get(4).ok(), created_at: row.get(5).ok(),
                group_id: None, embedding: None,
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
    ) -> Result<Vec<EdgeRow>> {
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
            out.push(EdgeRow {
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
