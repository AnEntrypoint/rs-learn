use crate::embeddings::Embedder;
use crate::store::types::EdgeRow;
use crate::store::Store;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use uuid::Uuid;

use super::entities::Entity;
use super::llm::{require_array_field, LlmJson};
use super::prompts::dedupe_edges as de;
use super::prompts::extract_edges as ee;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEdge {
    pub source_entity_name: String,
    pub target_entity_name: String,
    #[serde(default)]
    pub relation_type: String,
    #[serde(default)]
    pub fact: String,
    #[serde(default)]
    pub valid_at: Option<String>,
    #[serde(default)]
    pub invalid_at: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ResolvedEdge {
    pub id: String,
    pub src: String,
    pub dst: String,
    pub relation: String,
    pub fact: String,
    pub embedding: Option<Vec<f32>>,
    pub valid_at: Option<i64>,
    pub invalid_at: Option<i64>,
    pub group_id: Option<String>,
}

pub struct EdgeOps {
    pub store: Arc<Store>,
    pub embedder: Arc<Embedder>,
    pub llm: Arc<LlmJson>,
    pub edge_types: Option<Value>,
}

impl EdgeOps {
    pub fn new(store: Arc<Store>, embedder: Arc<Embedder>, llm: Arc<LlmJson>) -> Self {
        Self::with_types(store, embedder, llm, None)
    }

    pub fn with_types(store: Arc<Store>, embedder: Arc<Embedder>, llm: Arc<LlmJson>, edge_types: Option<Value>) -> Self {
        let edge_types = edge_types.or_else(|| {
            std::env::var("RS_LEARN_EDGE_TYPES_JSON").ok()
                .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        });
        Self { store, embedder, llm, edge_types }
    }

    pub async fn extract_edges(
        &self,
        episode_content: &str,
        previous_episodes: &Value,
        entities: &[Entity],
        reference_time: &str,
    ) -> Result<Vec<ExtractedEdge>> {
        self.extract_edges_with(episode_content, previous_episodes, entities, reference_time, None).await
    }

    pub async fn extract_edges_with(
        &self,
        episode_content: &str,
        previous_episodes: &Value,
        entities: &[Entity],
        reference_time: &str,
        edge_types_override: Option<&Value>,
    ) -> Result<Vec<ExtractedEdge>> {
        if entities.len() < 2 {
            return Ok(vec![]);
        }
        let nodes_json = json!(entities
            .iter()
            .map(|e| json!({"name": e.name}))
            .collect::<Vec<_>>());
        let prompt = ee::edge(&ee::EdgeCtx {
            previous_episodes,
            episode_content,
            nodes: &nodes_json,
            reference_time,
            edge_types: edge_types_override.or(self.edge_types.as_ref()),
            custom_extraction_instructions: None,
        });
        let v = self
            .llm
            .call(&prompt.system, &prompt.user, |v| {
                require_array_field(v, "edges")
            })
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let arr = v.get("edges").and_then(|a| a.as_array()).cloned().unwrap_or_default();
        let mut out = Vec::new();
        for item in arr {
            if let Ok(edge) = serde_json::from_value::<ExtractedEdge>(item) {
                if !edge.source_entity_name.is_empty()
                    && !edge.target_entity_name.is_empty()
                    && edge.source_entity_name != edge.target_entity_name
                {
                    out.push(edge);
                }
            }
        }
        Ok(out)
    }

    pub async fn resolve_edges(
        &self,
        extracted: Vec<ExtractedEdge>,
        entities: &[Entity],
    ) -> Result<Vec<ResolvedEdge>> {
        let mut resolved = Vec::new();
        for ex in extracted {
            let Some(src) = find_entity_id(&ex.source_entity_name, entities) else { continue };
            let Some(dst) = find_entity_id(&ex.target_entity_name, entities) else { continue };
            let fact = if ex.fact.is_empty() {
                format!("{} {} {}", ex.source_entity_name, ex.relation_type, ex.target_entity_name)
            } else {
                ex.fact
            };
            let emb = self.embedder.embed(&fact).ok();
            let valid_at = ex.valid_at.as_deref().and_then(parse_iso_ms);
            let invalid_at = ex.invalid_at.as_deref().and_then(parse_iso_ms);
            resolved.push(ResolvedEdge {
                id: Uuid::new_v4().to_string(),
                src,
                dst,
                relation: if ex.relation_type.is_empty() { "RELATED_TO".into() } else { ex.relation_type },
                fact,
                embedding: emb,
                valid_at,
                invalid_at,
                group_id: entities.first().and_then(|e| e.group_id.clone()),
            });
        }
        Ok(resolved)
    }

    pub async fn resolve_temporal(&self, new_edges: &[ResolvedEdge]) -> Result<Vec<String>> {
        let mut expired_ids = Vec::new();
        for e in new_edges {
            let existing = self.load_same_pair_edges(&e.src, &e.dst, &e.relation).await?;
            if existing.is_empty() {
                continue;
            }
            let existing_json = json!(existing
                .iter()
                .enumerate()
                .map(|(i, ex)| json!({
                    "idx": i,
                    "fact": ex.fact.clone().unwrap_or_default(),
                    "valid_at": ex.valid_at,
                    "invalid_at": ex.invalid_at,
                }))
                .collect::<Vec<_>>());
            let prompt = de::resolve_edge(&de::ResolveEdgeCtx {
                existing_edges: &existing_json,
                edge_invalidation_candidates: &json!([]),
                new_edge: &json!({"fact": e.fact, "valid_at": e.valid_at, "invalid_at": e.invalid_at}),
            });
            let v = self.llm.call(&prompt.system, &prompt.user, |_| Ok(())).await.ok();
            let Some(v) = v else { continue };
            let contradicted = v
                .get("contradicted_facts")
                .and_then(|a| a.as_array())
                .cloned()
                .unwrap_or_default();
            for c in contradicted {
                if let Some(i) = c.as_i64() {
                    if let Some(ex) = existing.get(i as usize) {
                        expired_ids.push(ex.id.clone());
                    }
                }
            }
        }
        Ok(expired_ids)
    }

    pub async fn upsert_edge(&self, e: &ResolvedEdge) -> Result<()> {
        let row = EdgeRow {
            id: e.id.clone(),
            src: e.src.clone(),
            dst: e.dst.clone(),
            relation: Some(e.relation.clone()),
            fact: Some(e.fact.clone()),
            embedding: e.embedding.clone(),
            weight: Some(1.0),
            group_id: e.group_id.clone(),
            valid_at: e.valid_at,
            invalid_at: e.invalid_at,
            created_at: None,
        };
        self.store.insert_edge(&row).await
    }

    pub async fn expire_edges(&self, ids: &[String], expired_at: i64) -> Result<()> {
        for id in ids {
            self.store
                .conn
                .execute(
                    "UPDATE edges SET expired_at = ?1 WHERE id = ?2",
                    libsql::params![expired_at, id.clone()],
                )
                .await?;
        }
        Ok(())
    }

    async fn load_same_pair_edges(&self, src: &str, dst: &str, _relation: &str) -> Result<Vec<EdgeRow>> {
        let mut rows = self
            .store
            .conn
            .query(
                "SELECT id, src, dst, relation, fact, weight, created_at, valid_at, invalid_at
                 FROM edges WHERE src = ?1 AND dst = ?2 AND (expired_at IS NULL)",
                libsql::params![src.to_string(), dst.to_string()],
            )
            .await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(EdgeRow {
                id: row.get(0)?,
                src: row.get(1)?,
                dst: row.get(2)?,
                relation: row.get(3).ok(),
                fact: row.get(4).ok(),
                weight: row.get::<f64>(5).ok(),
                group_id: None,
                created_at: row.get(6).ok(),
                valid_at: row.get(7).ok(),
                invalid_at: row.get(8).ok(),
                embedding: None,
            });
        }
        Ok(out)
    }
}

fn find_entity_id(name: &str, entities: &[Entity]) -> Option<String> {
    let target = name.trim().to_lowercase();
    entities
        .iter()
        .find(|e| e.name.trim().to_lowercase() == target)
        .map(|e| e.id.clone())
}

fn parse_iso_ms(s: &str) -> Option<i64> {
    let s = s.trim();
    if s.is_empty() { return None; }
    let normalized = if s.ends_with('Z') {
        s.replace('Z', "+00:00")
    } else { s.to_string() };
    chrono_like::parse_ms(&normalized)
}

mod chrono_like {
    pub fn parse_ms(s: &str) -> Option<i64> {
        let bytes = s.as_bytes();
        if bytes.len() < 19 { return None; }
        let year: i64 = std::str::from_utf8(&bytes[0..4]).ok()?.parse().ok()?;
        let month: i64 = std::str::from_utf8(&bytes[5..7]).ok()?.parse().ok()?;
        let day: i64 = std::str::from_utf8(&bytes[8..10]).ok()?.parse().ok()?;
        let hour: i64 = std::str::from_utf8(&bytes[11..13]).ok()?.parse().ok()?;
        let minute: i64 = std::str::from_utf8(&bytes[14..16]).ok()?.parse().ok()?;
        let second: i64 = std::str::from_utf8(&bytes[17..19]).ok()?.parse().ok()?;
        Some(days_from_civil(year, month, day) * 86_400_000
            + hour * 3_600_000
            + minute * 60_000
            + second * 1_000)
    }
    fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
        let y = if m <= 2 { y - 1 } else { y };
        let era = if y >= 0 { y } else { y - 399 } / 400;
        let yoe = (y - era * 400) as u64;
        let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) as u64 / 5 + d as u64 - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        era * 146097 + doe as i64 - 719468
    }
}
