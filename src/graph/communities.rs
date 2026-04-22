use crate::embeddings::Embedder;
use crate::store::{now_ms, Store};
use anyhow::Result;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use uuid::Uuid;

use super::llm::LlmJson;
use super::prompts::summarize_nodes;

pub struct CommunityOps {
    pub store: Arc<Store>,
    pub embedder: Arc<Embedder>,
    pub llm: Arc<LlmJson>,
}

pub struct BuildResult {
    pub community_count: usize,
    pub member_count: usize,
}

impl CommunityOps {
    pub fn new(store: Arc<Store>, embedder: Arc<Embedder>, llm: Arc<LlmJson>) -> Self {
        Self { store, embedder, llm }
    }

    pub async fn build_communities(&self) -> Result<BuildResult> {
        let nodes = self.all_node_ids().await?;
        if nodes.is_empty() {
            return Ok(BuildResult { community_count: 0, member_count: 0 });
        }
        let adjacency = self.build_adjacency(&nodes).await?;
        let labels = label_propagation(&nodes, &adjacency, 10);
        let now = now_ms();
        let mut groups: HashMap<String, Vec<String>> = HashMap::new();
        for (node, label) in &labels {
            groups.entry(label.clone()).or_default().push(node.clone());
        }
        self.store.conn.execute("DELETE FROM community_members", ()).await?;
        self.store.conn.execute("DELETE FROM communities", ()).await?;
        let mut community_count = 0;
        let mut member_count = 0;
        for (_label, members) in groups.iter() {
            if members.len() < 2 { continue; }
            let cid = Uuid::new_v4().to_string();
            let (name, summary, emb) = self.summarize_group(members).await;
            self.insert_community(&cid, &name, &summary, emb.as_deref(), now).await?;
            for n in members {
                self.store.conn.execute(
                    "INSERT INTO community_members(community_id,node_id) VALUES(?1,?2)
                     ON CONFLICT DO NOTHING",
                    libsql::params![cid.clone(), n.clone()],
                ).await?;
                member_count += 1;
            }
            community_count += 1;
        }
        Ok(BuildResult { community_count, member_count })
    }

    pub async fn remove_communities(&self) -> Result<()> {
        self.store.conn.execute("DELETE FROM community_members", ()).await?;
        self.store.conn.execute("DELETE FROM communities", ()).await?;
        Ok(())
    }

    async fn all_node_ids(&self) -> Result<Vec<String>> {
        let mut rows = self.store.conn.query("SELECT id FROM nodes", ()).await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(row.get::<String>(0)?);
        }
        Ok(out)
    }

    async fn build_adjacency(&self, nodes: &[String]) -> Result<HashMap<String, Vec<String>>> {
        let node_set: HashSet<&str> = nodes.iter().map(String::as_str).collect();
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        for n in nodes { adj.insert(n.clone(), vec![]); }
        let mut rows = self.store.conn.query(
            "SELECT src, dst FROM edges WHERE relation != 'MENTIONS' AND (expired_at IS NULL)",
            (),
        ).await?;
        while let Some(row) = rows.next().await? {
            let src: String = row.get(0)?;
            let dst: String = row.get(1)?;
            if !node_set.contains(src.as_str()) || !node_set.contains(dst.as_str()) { continue; }
            adj.entry(src.clone()).or_default().push(dst.clone());
            adj.entry(dst).or_default().push(src);
        }
        Ok(adj)
    }

    async fn summarize_group(&self, members: &[String]) -> (String, String, Option<Vec<f32>>) {
        let node_rows = self.store.get_nodes_by_ids(members).await.unwrap_or_default();
        let summaries: Vec<Value> = node_rows.iter().map(|n| {
            json!({
                "uuid": n.id,
                "name": n.name,
                "summary": n.summary.clone().unwrap_or_default(),
            })
        }).collect();
        let names: Vec<String> = node_rows.iter().map(|n| n.name.clone()).collect();
        let fallback_name = names.iter().take(3).cloned().collect::<Vec<_>>().join(", ");
        let prompt = summarize_nodes::summarize_pair(&Value::Array(summaries));
        let v = self.llm.call(&prompt.system, &prompt.user, |_| Ok(())).await.ok();
        let summary = v
            .as_ref()
            .and_then(|v| v.get("summary"))
            .and_then(|s| s.as_str())
            .unwrap_or(&fallback_name)
            .to_string();
        let emb = self.embedder.embed(&summary).ok();
        let name = fallback_name.chars().take(80).collect();
        (name, summary, emb)
    }

    async fn insert_community(&self, id: &str, name: &str, summary: &str, emb: Option<&[f32]>, now: i64) -> Result<()> {
        let lit = crate::store::vec_lit(emb);
        let sql = format!(
            "INSERT INTO communities(id,name,summary,embedding,level,created_at)
             VALUES(?1,?2,?3,{},?4,?5)
             ON CONFLICT(id) DO UPDATE SET name=excluded.name, summary=excluded.summary,
               embedding=excluded.embedding",
            lit
        );
        self.store.conn.execute(&sql, libsql::params![
            id.to_string(), name.to_string(), summary.to_string(), 0_i64, now
        ]).await?;
        Ok(())
    }
}

fn label_propagation(
    nodes: &[String],
    adj: &HashMap<String, Vec<String>>,
    max_iter: usize,
) -> HashMap<String, String> {
    let mut labels: HashMap<String, String> = nodes.iter()
        .map(|n| (n.clone(), n.clone()))
        .collect();
    for _ in 0..max_iter {
        let mut changed = 0;
        let mut order: Vec<&String> = nodes.iter().collect();
        order.sort();
        for n in &order {
            let Some(neighbors) = adj.get(n.as_str()) else { continue };
            if neighbors.is_empty() { continue; }
            let mut counts: HashMap<String, u32> = HashMap::new();
            for nb in neighbors {
                if let Some(l) = labels.get(nb) {
                    *counts.entry(l.clone()).or_insert(0) += 1;
                }
            }
            let Some(best) = counts.into_iter().max_by_key(|(_, c)| *c).map(|(l, _)| l) else { continue };
            if labels.get(*n) != Some(&best) {
                labels.insert((*n).clone(), best);
                changed += 1;
            }
        }
        if changed == 0 { break; }
    }
    labels
}
