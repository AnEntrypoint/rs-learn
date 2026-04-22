use crate::embeddings::Embedder;
use crate::store::types::NodeRow;
use crate::store::Store;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use uuid::Uuid;

use super::llm::{require_array_field, LlmJson};
use super::prompts::dedupe_nodes as dn;
use super::prompts::extract_nodes as en;

pub const NODE_DEDUP_CANDIDATE_LIMIT: usize = 15;
pub const MAX_NODES: usize = 30;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    pub name: String,
    #[serde(default)]
    pub entity_type_id: i64,
}

#[derive(Debug, Clone)]
pub struct Entity {
    pub id: String,
    pub name: String,
    pub entity_type: Option<String>,
    pub embedding: Option<Vec<f32>>,
    pub group_id: Option<String>,
}

pub struct EntityOps {
    pub store: Arc<Store>,
    pub embedder: Arc<Embedder>,
    pub llm: Arc<LlmJson>,
    pub entity_types: String,
}

impl EntityOps {
    pub fn new(store: Arc<Store>, embedder: Arc<Embedder>, llm: Arc<LlmJson>) -> Self {
        Self {
            store,
            embedder,
            llm,
            entity_types: default_entity_types(),
        }
    }

    pub async fn extract_entities(
        &self,
        source: &str,
        episode_content: &str,
        previous_episodes: &Value,
    ) -> Result<Vec<ExtractedEntity>> {
        let ctx = en::ExtractCtx {
            entity_types: &self.entity_types,
            previous_episodes,
            episode_content,
            custom_extraction_instructions: None,
            source_description: Some("episode"),
        };
        let prompt = match source {
            "message" => en::extract_message(&ctx),
            "json" => en::extract_json(&ctx),
            _ => en::extract_text(&ctx),
        };
        let v = self
            .llm
            .call(&prompt.system, &prompt.user, |v| {
                require_array_field(v, "extracted_entities")
            })
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let arr = v
            .get("extracted_entities")
            .and_then(|a| a.as_array())
            .cloned()
            .unwrap_or_default();
        let mut out = Vec::new();
        for item in arr {
            if let Ok(ent) = serde_json::from_value::<ExtractedEntity>(item) {
                if !ent.name.trim().is_empty() {
                    out.push(ent);
                }
            }
        }
        out.truncate(MAX_NODES);
        Ok(out)
    }

    pub async fn dedup_entities(
        &self,
        candidates: Vec<ExtractedEntity>,
        episode_content: &str,
        previous_episodes: &Value,
        group_id: Option<&str>,
    ) -> Result<Vec<Entity>> {
        let mut resolved: Vec<Entity> = Vec::with_capacity(candidates.len());
        for cand in candidates {
            let emb = self.embedder.embed(&cand.name).unwrap_or_default();
            let existing = if emb.is_empty() {
                vec![]
            } else {
                self.store.vector_top_k("nodes", &emb, NODE_DEDUP_CANDIDATE_LIMIT, None).await.unwrap_or_default()
            };
            if let Some(exact) = exact_name_match(&cand.name, &existing) {
                resolved.push(Entity {
                    id: exact,
                    name: cand.name,
                    entity_type: None,
                    embedding: Some(emb),
                    group_id: group_id.map(String::from),
                });
                continue;
            }
            if !existing.is_empty() {
                let candidates_json = json!(existing
                    .iter()
                    .enumerate()
                    .map(|(i, row)| json!({
                        "candidate_id": i,
                        "name": row.get("name").and_then(val_text),
                        "summary": row.get("summary").and_then(val_text),
                    }))
                    .collect::<Vec<_>>());
                let prompt = dn::node(&dn::NodeCtx {
                    previous_episodes,
                    episode_content,
                    extracted_node: &json!({"name": cand.name}),
                    entity_type_description: &Value::Null,
                    existing_nodes: &candidates_json,
                });
                let v = self
                    .llm
                    .call(&prompt.system, &prompt.user, |_| Ok(()))
                    .await
                    .ok();
                let dup_idx = v
                    .as_ref()
                    .and_then(|v| v.get("duplicate_candidate_id"))
                    .and_then(|n| n.as_i64())
                    .unwrap_or(-1);
                if dup_idx >= 0 {
                    if let Some(row) = existing.get(dup_idx as usize) {
                        if let Some(id) = row.get("id").and_then(val_text) {
                            resolved.push(Entity {
                                id,
                                name: cand.name,
                                entity_type: None,
                                embedding: Some(emb),
                                group_id: group_id.map(String::from),
                            });
                            continue;
                        }
                    }
                }
            }
            resolved.push(Entity {
                id: Uuid::new_v4().to_string(),
                name: cand.name,
                entity_type: None,
                embedding: Some(emb),
                group_id: group_id.map(String::from),
            });
        }
        Ok(resolved)
    }

    pub async fn upsert_node(&self, ent: &Entity) -> Result<()> {
        use super::prompts::snippets::MAX_SUMMARY_CHARS;
        use super::text::truncate_at_sentence;
        let row = NodeRow {
            id: ent.id.clone(),
            name: ent.name.clone(),
            r#type: ent.entity_type.clone(),
            summary: Some(truncate_at_sentence("", MAX_SUMMARY_CHARS)),
            embedding: ent.embedding.clone(),
            level: Some(0),
            group_id: ent.group_id.clone(),
            created_at: None,
        };
        self.store.insert_node(&row).await
    }

    pub async fn get_node(&self, id: &str) -> Result<Option<NodeRow>> {
        let v = self.store.get_nodes_by_ids(&[id.to_string()]).await?;
        Ok(v.into_iter().next())
    }
}

fn exact_name_match(
    name: &str,
    rows: &[std::collections::HashMap<String, libsql::Value>],
) -> Option<String> {
    let target = name.trim().to_lowercase();
    for row in rows {
        if let Some(row_name) = row.get("name").and_then(val_text) {
            if row_name.trim().to_lowercase() == target {
                return row.get("id").and_then(val_text);
            }
        }
    }
    None
}

fn val_text(v: &libsql::Value) -> Option<String> {
    match v {
        libsql::Value::Text(s) => Some(s.clone()),
        _ => None,
    }
}

pub fn default_entity_types() -> String {
    r#"[
  {"entity_type_id": 0, "name": "Entity", "description": "Default entity type for anything that doesn't fit the others."},
  {"entity_type_id": 1, "name": "Person", "description": "Named individual human."},
  {"entity_type_id": 2, "name": "Organization", "description": "Company, agency, team, club."},
  {"entity_type_id": 3, "name": "Location", "description": "Place, city, country, landmark."},
  {"entity_type_id": 4, "name": "Object", "description": "Concrete physical or digital object."},
  {"entity_type_id": 5, "name": "Topic", "description": "Field, subject, concept discussed."}
]"#
    .to_string()
}
