use crate::embeddings::Embedder;
use crate::store::types::{EdgeRow, EpisodeRow};
use crate::store::{now_ms, Store};
use anyhow::Result;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;

use super::edges::{EdgeOps, ResolvedEdge};
use super::entities::{Entity, EntityOps};
use super::llm::LlmJson;

#[derive(serde::Deserialize)]
pub struct BulkEpisode {
    pub content: String,
    #[serde(default = "default_source")]
    pub source: String,
    #[serde(default)]
    pub reference_time: Option<String>,
}

fn default_source() -> String { "message".into() }

pub struct IngestResult {
    pub episode_id: String,
    pub node_count: usize,
    pub edge_count: usize,
    pub expired_edge_ids: Vec<String>,
}

pub struct Ingestor {
    pub store: Arc<Store>,
    pub embedder: Arc<Embedder>,
    pub llm: Arc<LlmJson>,
    pub entity_ops: Arc<EntityOps>,
    pub edge_ops: Arc<EdgeOps>,
    writer: Mutex<()>,
}

impl Ingestor {
    pub fn new(store: Arc<Store>, embedder: Arc<Embedder>, llm: Arc<LlmJson>) -> Arc<Self> {
        let entity_ops = Arc::new(EntityOps::new(store.clone(), embedder.clone(), llm.clone()));
        let edge_ops = Arc::new(EdgeOps::new(store.clone(), embedder.clone(), llm.clone()));
        Arc::new(Self {
            store,
            embedder,
            llm,
            entity_ops,
            edge_ops,
            writer: Mutex::new(()),
        })
    }

    pub async fn add_episode(
        &self,
        content: &str,
        source: &str,
        reference_time: Option<&str>,
    ) -> Result<IngestResult> {
        let _lock = self.writer.lock().await;
        let now = now_ms();
        let episode_id = Uuid::new_v4().to_string();
        let ref_time_str = reference_time
            .map(|s| s.to_string())
            .unwrap_or_else(|| format_iso(now));
        let episode = EpisodeRow {
            id: episode_id.clone(),
            content: content.to_string(),
            source: Some(source.to_string()),
            created_at: Some(now),
            valid_at: Some(now),
            invalid_at: None,
        };
        self.store.insert_episode(&episode).await?;

        let previous = self.load_previous_episodes(4).await.unwrap_or_else(|_| json!([]));
        let extracted = self
            .entity_ops
            .extract_entities(source, content, &previous)
            .await
            .unwrap_or_default();
        let entities = self
            .entity_ops
            .dedup_entities(extracted, content, &previous)
            .await
            .unwrap_or_default();
        for ent in &entities {
            let _ = self.entity_ops.upsert_node(ent).await;
        }

        let extracted_edges = self
            .edge_ops
            .extract_edges(content, &previous, &entities, &ref_time_str)
            .await
            .unwrap_or_default();
        let resolved = self
            .edge_ops
            .resolve_edges(extracted_edges, &entities)
            .await
            .unwrap_or_default();
        let expired_ids = self
            .edge_ops
            .resolve_temporal(&resolved)
            .await
            .unwrap_or_default();
        if !expired_ids.is_empty() {
            let _ = self.edge_ops.expire_edges(&expired_ids, now).await;
        }
        for e in &resolved {
            let _ = self.edge_ops.upsert_edge(e).await;
        }
        self.write_mentions(&episode_id, &entities, now).await?;

        Ok(IngestResult {
            episode_id,
            node_count: entities.len(),
            edge_count: resolved.len(),
            expired_edge_ids: expired_ids,
        })
    }

    async fn load_previous_episodes(&self, limit: usize) -> Result<Value> {
        let mut rows = self
            .store
            .conn
            .query(
                "SELECT id, content, source, created_at FROM episodes ORDER BY created_at DESC LIMIT ?1",
                libsql::params![limit as i64],
            )
            .await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            let id: String = row.get(0)?;
            let content: String = row.get(1)?;
            let source: Option<String> = row.get(2).ok();
            let created_at: i64 = row.get(3).unwrap_or(0);
            out.push(json!({
                "id": id,
                "content": content,
                "source": source,
                "created_at": created_at,
            }));
        }
        out.reverse();
        Ok(json!(out))
    }

    async fn write_mentions(&self, episode_id: &str, entities: &[Entity], now: i64) -> Result<()> {
        for ent in entities {
            let edge = EdgeRow {
                id: Uuid::new_v4().to_string(),
                src: episode_id.to_string(),
                dst: ent.id.clone(),
                relation: Some("MENTIONS".into()),
                fact: Some(String::new()),
                embedding: None,
                weight: Some(1.0),
                created_at: Some(now),
                valid_at: Some(now),
                invalid_at: None,
            };
            self.store.insert_edge(&edge).await?;
        }
        Ok(())
    }
}

impl Ingestor {
    pub async fn add_triplet(&self, src: Entity, dst: Entity, relation: &str, fact: &str) -> Result<String> {
        let _lock = self.writer.lock().await;
        let now = now_ms();
        let _ = self.entity_ops.upsert_node(&src).await;
        let _ = self.entity_ops.upsert_node(&dst).await;
        let emb = self.embedder.embed(fact).ok();
        let resolved = ResolvedEdge {
            id: Uuid::new_v4().to_string(),
            src: src.id.clone(),
            dst: dst.id.clone(),
            relation: relation.to_string(),
            fact: fact.to_string(),
            embedding: emb,
            valid_at: Some(now),
            invalid_at: None,
        };
        self.edge_ops.upsert_edge(&resolved).await?;
        Ok(resolved.id)
    }

    pub async fn add_episode_bulk(&self, items: Vec<BulkEpisode>) -> Result<Vec<IngestResult>> {
        let mut out = Vec::with_capacity(items.len());
        for it in items {
            let r = self.add_episode(&it.content, &it.source, it.reference_time.as_deref()).await?;
            out.push(r);
        }
        Ok(out)
    }

    pub async fn get_episodes(&self, group_id: Option<&str>, limit: usize) -> Result<Vec<serde_json::Value>> {
        let mut rows = match group_id {
            Some(gid) => self.store.conn.query(
                "SELECT id, content, source, created_at FROM episodes WHERE group_id = ?1 ORDER BY created_at DESC LIMIT ?2",
                libsql::params![gid.to_string(), limit as i64],
            ).await?,
            None => self.store.conn.query(
                "SELECT id, content, source, created_at FROM episodes ORDER BY created_at DESC LIMIT ?1",
                libsql::params![limit as i64],
            ).await?,
        };
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(json!({
                "id": row.get::<String>(0).unwrap_or_default(),
                "content": row.get::<String>(1).unwrap_or_default(),
                "source": row.get::<String>(2).ok(),
                "created_at": row.get::<i64>(3).ok(),
            }));
        }
        Ok(out)
    }

    pub async fn clear_graph(&self) -> Result<()> {
        let _lock = self.writer.lock().await;
        for tbl in [
            "edges",
            "nodes",
            "episodes",
            "communities",
            "community_members",
            "saga_episodes",
            "sagas",
        ] {
            self.store
                .conn
                .execute(&format!("DELETE FROM {tbl}"), ())
                .await?;
        }
        Ok(())
    }
}

fn format_iso(ms: i64) -> String {
    let s = ms / 1000;
    let millis = ms % 1000;
    let (y, mo, d, h, mi, se) = ms_to_parts(s);
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z", y, mo, d, h, mi, se, millis)
}

fn ms_to_parts(s: i64) -> (i64, i64, i64, i64, i64, i64) {
    let days = s.div_euclid(86400);
    let rem = s.rem_euclid(86400);
    let h = rem / 3600;
    let mi = (rem % 3600) / 60;
    let se = rem % 60;
    let (y, mo, d) = civil_from_days(days);
    (y, mo, d, h, mi, se)
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as i64;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as i64;
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}
