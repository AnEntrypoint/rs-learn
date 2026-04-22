use crate::embeddings::Embedder;
use crate::store::Store;
use anyhow::Result;
use libsql::Value as LV;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;

use super::llm::LlmJson;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Reranker {
    Rrf,
    Mmr,
    NodeDistance,
    EpisodeMentions,
    CrossEncoder,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    pub limit: usize,
    pub rrf_k: f32,
    pub mmr_lambda: f32,
    pub reranker: Reranker,
    pub as_of: Option<i64>,
    pub center_node_id: Option<String>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            limit: 10,
            rrf_k: 60.0,
            mmr_lambda: 0.5,
            reranker: Reranker::Rrf,
            as_of: None,
            center_node_id: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub id: String,
    pub score: f32,
    pub row: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchAll {
    pub nodes: Vec<SearchHit>,
    pub edges: Vec<SearchHit>,
    pub episodes: Vec<SearchHit>,
    pub communities: Vec<SearchHit>,
}

pub struct Searcher {
    pub store: Arc<Store>,
    pub embedder: Arc<Embedder>,
    pub llm: Option<Arc<LlmJson>>,
}

impl Searcher {
    pub fn new(store: Arc<Store>, embedder: Arc<Embedder>) -> Self {
        Self { store, embedder, llm: None }
    }

    pub fn with_llm(store: Arc<Store>, embedder: Arc<Embedder>, llm: Arc<LlmJson>) -> Self {
        Self { store, embedder, llm: Some(llm) }
    }

    pub async fn search_nodes(&self, query: &str, cfg: &SearchConfig) -> Result<Vec<SearchHit>> {
        self.search_table("nodes", query, cfg).await
    }

    pub async fn search_facts(&self, query: &str, cfg: &SearchConfig) -> Result<Vec<SearchHit>> {
        let hits = self.search_table("edges", query, cfg).await?;
        Ok(self.filter_as_of_edges(hits, cfg.as_of))
    }

    pub async fn search_episodes(&self, query: &str, cfg: &SearchConfig) -> Result<Vec<SearchHit>> {
        self.search_table("episodes", query, cfg).await
    }

    pub async fn search_communities(&self, query: &str, cfg: &SearchConfig) -> Result<Vec<SearchHit>> {
        self.search_table("communities", query, cfg).await
    }

    pub async fn search_all(&self, query: &str, cfg: &SearchConfig) -> Result<SearchAll> {
        let (nodes, edges, episodes, communities) = tokio::join!(
            self.search_nodes(query, cfg),
            self.search_facts(query, cfg),
            self.search_episodes(query, cfg),
            self.search_communities(query, cfg),
        );
        Ok(SearchAll {
            nodes: nodes?,
            edges: edges?,
            episodes: episodes?,
            communities: communities?,
        })
    }

    async fn search_table(&self, table: &str, query: &str, cfg: &SearchConfig) -> Result<Vec<SearchHit>> {
        let emb = self.embedder.embed(query).unwrap_or_default();
        let k_fetch = (cfg.limit * 3).max(10);
        let vec_rows = if emb.is_empty() {
            vec![]
        } else {
            self.store.vector_top_k(table, &emb, k_fetch, None).await.unwrap_or_default()
        };
        let fts_rows = self.store.fts_search(table, query, k_fetch).await.unwrap_or_default();
        let fused = rrf_fuse(&vec_rows, &fts_rows, cfg.rrf_k);
        let hits: Vec<SearchHit> = fused
            .into_iter()
            .map(|(id, score, row)| SearchHit {
                id,
                score,
                row: row_to_json_map(&row),
            })
            .collect();
        match cfg.reranker {
            Reranker::Rrf => Ok(hits.into_iter().take(cfg.limit).collect()),
            Reranker::Mmr => Ok(mmr(hits, cfg.limit, cfg.mmr_lambda, &emb)),
            Reranker::NodeDistance => Ok(self.node_distance_rerank(hits, cfg).await),
            Reranker::EpisodeMentions => Ok(self.episode_mentions_rerank(hits, cfg.limit).await),
            Reranker::CrossEncoder => Ok(self.cross_encoder_rerank(query, hits, cfg.limit).await),
        }
    }

    fn filter_as_of_edges(&self, hits: Vec<SearchHit>, as_of: Option<i64>) -> Vec<SearchHit> {
        let Some(t) = as_of else { return hits };
        hits.into_iter()
            .filter(|h| {
                let valid_at = h.row.get("valid_at").and_then(|v| v.as_i64()).unwrap_or(0);
                let expired_at = h.row.get("expired_at").and_then(|v| v.as_i64());
                let invalid_at = h.row.get("invalid_at").and_then(|v| v.as_i64());
                if valid_at > t { return false; }
                if let Some(e) = expired_at { if e <= t { return false; } }
                if let Some(i) = invalid_at { if i <= t { return false; } }
                true
            })
            .collect()
    }

    async fn node_distance_rerank(&self, hits: Vec<SearchHit>, cfg: &SearchConfig) -> Vec<SearchHit> {
        let Some(center) = cfg.center_node_id.as_deref() else {
            return hits.into_iter().take(cfg.limit).collect();
        };
        let ids: Vec<String> = hits.iter().map(|h| h.id.clone()).collect();
        let distances = self.shortest_hops(center, &ids).await;
        let mut scored: Vec<(SearchHit, f32)> = hits
            .into_iter()
            .map(|h| {
                let d = distances.get(&h.id).copied().unwrap_or(u32::MAX);
                let boost = 1.0 / (1.0 + d as f32);
                let combined = h.score + 0.5 * boost;
                (h, combined)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(cfg.limit).map(|(h, _)| h).collect()
    }

    async fn episode_mentions_rerank(&self, hits: Vec<SearchHit>, limit: usize) -> Vec<SearchHit> {
        let ids: Vec<String> = hits.iter().map(|h| h.id.clone()).collect();
        let counts = self.mention_counts(&ids).await;
        let mut scored: Vec<(SearchHit, f32)> = hits
            .into_iter()
            .map(|h| {
                let c = counts.get(&h.id).copied().unwrap_or(0);
                let boost = (c as f32).ln_1p();
                (h.clone(), h.score + 0.3 * boost)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(limit).map(|(h, _)| h).collect()
    }

    async fn cross_encoder_rerank(&self, query: &str, hits: Vec<SearchHit>, limit: usize) -> Vec<SearchHit> {
        let Some(llm) = self.llm.clone() else {
            return hits.into_iter().take(limit).collect();
        };
        if hits.is_empty() { return vec![]; }
        let items: Vec<serde_json::Value> = hits.iter().enumerate().map(|(i, h)| {
            let text = h.row.get("name").and_then(|v| v.as_str())
                .or_else(|| h.row.get("fact").and_then(|v| v.as_str()))
                .or_else(|| h.row.get("content").and_then(|v| v.as_str()))
                .or_else(|| h.row.get("summary").and_then(|v| v.as_str()))
                .unwrap_or("");
            json!({ "idx": i, "text": text })
        }).collect();
        let sys = "You are a relevance ranking assistant. Score each CANDIDATE for relevance to the QUERY on a 0.0..1.0 scale. Higher = more relevant. Return only JSON.";
        let user = format!(
            "<QUERY>\n{query}\n</QUERY>\n<CANDIDATES>\n{items}\n</CANDIDATES>\n\nReturn JSON: {{\"scores\":[{{\"idx\":0,\"score\":0.0}}]}}",
            query = query,
            items = serde_json::to_string_pretty(&items).unwrap_or_default(),
        );
        let v = llm.call(sys, &user, |_| Ok(())).await.ok();
        let mut scored: Vec<(SearchHit, f32)> = hits
            .into_iter()
            .enumerate()
            .map(|(i, h)| (h, i as f32))
            .collect();
        if let Some(v) = v {
            if let Some(arr) = v.get("scores").and_then(|a| a.as_array()) {
                let mut map: HashMap<usize, f32> = HashMap::new();
                for item in arr {
                    let idx = item.get("idx").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
                    let score = item.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0) as f32;
                    map.insert(idx, score);
                }
                scored = scored
                    .into_iter()
                    .enumerate()
                    .map(|(i, (h, _))| {
                        let s = map.get(&i).copied().unwrap_or(0.0);
                        (h, s)
                    })
                    .collect();
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
        }
        scored.into_iter().take(limit).map(|(h, _)| h).collect()
    }

    async fn shortest_hops(&self, center: &str, targets: &[String]) -> HashMap<String, u32> {
        let mut out = HashMap::new();
        out.insert(center.to_string(), 0);
        let mut frontier: Vec<String> = vec![center.to_string()];
        for hop in 1..=3u32 {
            if frontier.is_empty() { break; }
            let edges = self.store.get_edges_from(&frontier).await.unwrap_or_default();
            let mut next = Vec::new();
            for e in edges {
                if !out.contains_key(&e.dst) {
                    out.insert(e.dst.clone(), hop);
                    next.push(e.dst);
                }
            }
            frontier = next;
            if targets.iter().all(|t| out.contains_key(t)) { break; }
        }
        out
    }

    async fn mention_counts(&self, node_ids: &[String]) -> HashMap<String, u64> {
        let mut out = HashMap::new();
        if node_ids.is_empty() { return out; }
        let placeholders: Vec<String> = (1..=node_ids.len()).map(|i| format!("?{i}")).collect();
        let sql = format!(
            "SELECT dst, COUNT(*) FROM edges WHERE relation='MENTIONS' AND dst IN ({}) GROUP BY dst",
            placeholders.join(",")
        );
        let args: Vec<LV> = node_ids.iter().map(|s| LV::Text(s.clone())).collect();
        let Ok(mut rows) = self.store.conn.query(&sql, args).await else { return out };
        while let Ok(Some(row)) = rows.next().await {
            let id: String = row.get(0).unwrap_or_default();
            let c: i64 = row.get(1).unwrap_or(0);
            out.insert(id, c as u64);
        }
        out
    }
}

fn rrf_fuse(
    a: &[HashMap<String, LV>],
    b: &[HashMap<String, LV>],
    k: f32,
) -> Vec<(String, f32, HashMap<String, LV>)> {
    let mut scores: HashMap<String, f32> = HashMap::new();
    let mut rows: HashMap<String, HashMap<String, LV>> = HashMap::new();
    for (i, r) in a.iter().enumerate() {
        if let Some(id) = row_id(r) {
            *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (k + i as f32 + 1.0);
            rows.entry(id).or_insert_with(|| r.clone());
        }
    }
    for (i, r) in b.iter().enumerate() {
        if let Some(id) = row_id(r) {
            *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (k + i as f32 + 1.0);
            rows.entry(id).or_insert_with(|| r.clone());
        }
    }
    let mut fused: Vec<(String, f32, HashMap<String, LV>)> = scores
        .into_iter()
        .map(|(id, s)| {
            let row = rows.remove(&id).unwrap_or_default();
            (id, s, row)
        })
        .collect();
    fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    fused
}

fn mmr(hits: Vec<SearchHit>, limit: usize, lambda: f32, _query_emb: &[f32]) -> Vec<SearchHit> {
    if hits.is_empty() { return vec![]; }
    let mut remaining = hits;
    let mut selected: Vec<SearchHit> = Vec::with_capacity(limit);
    selected.push(remaining.remove(0));
    while selected.len() < limit && !remaining.is_empty() {
        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;
        for (i, cand) in remaining.iter().enumerate() {
            let relevance = cand.score;
            let max_sim = selected
                .iter()
                .map(|s| name_sim(cand, s))
                .fold(0.0f32, f32::max);
            let mmr_s = lambda * relevance - (1.0 - lambda) * max_sim;
            if mmr_s > best_score {
                best_score = mmr_s;
                best_idx = i;
            }
        }
        selected.push(remaining.remove(best_idx));
    }
    selected
}

fn name_sim(a: &SearchHit, b: &SearchHit) -> f32 {
    let an = a.row.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let bn = b.row.get("name").and_then(|v| v.as_str()).unwrap_or("");
    if an.is_empty() || bn.is_empty() { return 0.0; }
    if an == bn { return 1.0; }
    let al = an.to_lowercase();
    let bl = bn.to_lowercase();
    if al.contains(&bl) || bl.contains(&al) { 0.5 } else { 0.0 }
}

fn row_id(r: &HashMap<String, LV>) -> Option<String> {
    match r.get("id") {
        Some(LV::Text(s)) => Some(s.clone()),
        _ => None,
    }
}

fn row_to_json_map(r: &HashMap<String, LV>) -> HashMap<String, serde_json::Value> {
    let mut out = HashMap::new();
    for (k, v) in r {
        out.insert(k.clone(), lv_to_json(v));
    }
    out
}

fn lv_to_json(v: &LV) -> serde_json::Value {
    match v {
        LV::Null => serde_json::Value::Null,
        LV::Integer(i) => serde_json::Value::from(*i),
        LV::Real(f) => serde_json::Value::from(*f),
        LV::Text(s) => serde_json::Value::from(s.clone()),
        LV::Blob(b) => serde_json::Value::from(b.len()),
    }
}
