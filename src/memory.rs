use crate::observability;
use crate::store::read::VectorFilter;
use crate::store::{now_ms, EdgeRow, NodeRow, Store};
use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub use crate::attention::{Subgraph, SubgraphEdge, SubgraphNode};

const M: usize = 32;
const EF: usize = 64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInput {
    pub id: Option<String>,
    pub payload: serde_json::Value,
    pub embedding: Vec<f32>,
    pub level: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub id: String,
    pub payload: String,
    pub score: f32,
    pub level: i64,
}

struct Stats {
    add_count: AtomicU64,
    search_count: AtomicU64,
    expand_count: AtomicU64,
}

pub struct Memory {
    store: Arc<Store>,
    stats: Arc<Stats>,
}

fn m_l() -> f64 { 1.0 / (M as f64).ln() }

fn sample_level() -> i64 {
    let mut rng = rand::thread_rng();
    let r: f64 = rng.gen_range(1e-12f64..1.0);
    (-r.ln() * m_l()).floor() as i64
}

fn rid() -> String {
    use rand::distributions::Alphanumeric;
    let suffix: String = rand::thread_rng().sample_iter(&Alphanumeric).take(8).map(char::from).collect();
    format!("m_{}_{}", now_ms(), suffix.to_lowercase())
}

fn row_get_str(r: &HashMap<String, libsql::Value>, k: &str) -> Option<String> {
    match r.get(k)? { libsql::Value::Text(s) => Some(s.clone()), _ => None }
}
fn row_get_i64(r: &HashMap<String, libsql::Value>, k: &str) -> Option<i64> {
    match r.get(k)? { libsql::Value::Integer(i) => Some(*i), _ => None }
}
fn row_get_f64(r: &HashMap<String, libsql::Value>, k: &str) -> Option<f64> {
    match r.get(k)? { libsql::Value::Real(f) => Some(*f), libsql::Value::Integer(i) => Some(*i as f64), _ => None }
}

impl Memory {
    pub fn new(store: Arc<Store>) -> Self {
        let stats = Arc::new(Stats {
            add_count: AtomicU64::new(0),
            search_count: AtomicU64::new(0),
            expand_count: AtomicU64::new(0),
        });
        let s2 = stats.clone();
        observability::register("memory", move || serde_json::json!({
            "add_count": s2.add_count.load(Ordering::Relaxed),
            "search_count": s2.search_count.load(Ordering::Relaxed),
            "expand_count": s2.expand_count.load(Ordering::Relaxed),
            "M": M, "mL": m_l(),
        }));
        Self { store, stats }
    }

    async fn neighbors_at_level(&self, emb: &[f32], level: i64, k: usize, exclude: &str) -> Result<Vec<(String, f64)>> {
        let filter = VectorFilter {
            sql: "b.level >= ?1 AND b.id != ?2",
            args: vec![libsql::Value::Integer(level), libsql::Value::Text(exclude.to_string())],
        };
        let rows = self.store.vector_top_k("nodes", emb, k, Some(filter)).await?;
        Ok(rows.into_iter().filter_map(|r| {
            let id = row_get_str(&r, "id")?;
            let dist = row_get_f64(&r, "dist").unwrap_or(0.0);
            Some((id, dist))
        }).collect())
    }

    pub async fn add(&self, node: NodeInput) -> Result<String> {
        let nid = node.id.clone().unwrap_or_else(rid);
        let lvl = node.level.unwrap_or_else(sample_level);
        let summary = match &node.payload {
            serde_json::Value::String(s) => s.clone(),
            v => v.to_string(),
        };
        let name = node.payload.get("name").and_then(|v| v.as_str()).map(|s| s.to_string()).unwrap_or_else(|| nid.clone());
        let ts = now_ms();
        self.store.insert_node(&NodeRow {
            id: nid.clone(), name, r#type: Some("memory".into()),
            summary: Some(summary), embedding: Some(node.embedding.clone()),
            level: Some(lvl), group_id: None, created_at: Some(ts),
        }).await?;
        for l in 0..=lvl {
            let neigh = self.neighbors_at_level(&node.embedding, l, M, &nid).await?;
            for (nbr_id, dist) in neigh {
                let w = 1.0 - dist;
                let rel = format!("hnsw-neighbor-L{}", l);
                let e1 = EdgeRow {
                    id: format!("hnsw_{}_{}_L{}", nid, nbr_id, l),
                    src: nid.clone(), dst: nbr_id.clone(),
                    relation: Some(rel.clone()), weight: Some(w),
                    created_at: Some(ts), ..Default::default()
                };
                let e2 = EdgeRow {
                    id: format!("hnsw_{}_{}_L{}", nbr_id, nid, l),
                    src: nbr_id, dst: nid.clone(),
                    relation: Some(rel), weight: Some(w),
                    created_at: Some(ts), ..Default::default()
                };
                self.store.insert_edge(&e1).await?;
                self.store.insert_edge(&e2).await?;
            }
        }
        self.stats.add_count.fetch_add(1, Ordering::Relaxed);
        Ok(nid)
    }

    pub async fn search(&self, query_emb: &[f32], k: usize) -> Result<Vec<SearchHit>> {
        let fetch = EF.max(k * 4);
        let rows = self.store.vector_top_k("nodes", query_emb, fetch, None).await?;
        let mut out: Vec<SearchHit> = rows.into_iter().filter_map(|r| {
            let id = row_get_str(&r, "id")?;
            let payload = row_get_str(&r, "summary").unwrap_or_default();
            let level = row_get_i64(&r, "level").unwrap_or(0);
            let dist = row_get_f64(&r, "dist").unwrap_or(0.0);
            Some(SearchHit { id, payload, score: (1.0 - dist) as f32, level })
        }).collect();
        out.truncate(k);
        self.stats.search_count.fetch_add(1, Ordering::Relaxed);
        Ok(out)
    }

    pub async fn expand(&self, id: &str, hops: usize) -> Result<Subgraph> {
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(id.to_string());
        let mut frontier: Vec<String> = vec![id.to_string()];
        let mut all_edges: Vec<SubgraphEdge> = Vec::new();
        for _ in 0..hops {
            if frontier.is_empty() { break; }
            let edges = self.store.get_edges_from(&frontier).await?;
            let mut next: Vec<String> = Vec::new();
            for e in edges {
                all_edges.push(SubgraphEdge {
                    src: e.src.clone(), dst: e.dst.clone(),
                    relation: e.relation.clone(),
                    weight: e.weight.map(|w| w as f32),
                    created_at: e.created_at,
                });
                if !visited.contains(&e.dst) {
                    visited.insert(e.dst.clone());
                    next.push(e.dst);
                }
            }
            frontier = next;
        }
        let ids: Vec<String> = visited.into_iter().collect();
        let node_rows = self.store.get_nodes_by_ids(&ids).await?;
        let nodes: Vec<SubgraphNode> = node_rows.into_iter().map(|n| SubgraphNode {
            id: n.id, embedding: n.embedding, created_at: n.created_at,
        }).collect();
        self.stats.expand_count.fetch_add(1, Ordering::Relaxed);
        Ok(Subgraph { nodes, edges: all_edges })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn rand_emb(seed: u64) -> Vec<f32> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        (0..768).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
    }

    fn cos_dist(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        1.0 - dot / (na * nb + 1e-12)
    }

    #[tokio::test]
    async fn memory_hnsw_recall_and_expand() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("m.db");
        let store = Arc::new(Store::open(p.to_str().unwrap()).await.unwrap());
        let mem = Memory::new(store);
        let n = 200;
        let mut all: Vec<(String, Vec<f32>)> = Vec::new();
        for i in 0..n {
            let emb = rand_emb(i as u64);
            let id = mem.add(NodeInput {
                id: Some(format!("n{}", i)), payload: serde_json::json!({"i": i}),
                embedding: emb.clone(), level: None,
            }).await.unwrap();
            all.push((id, emb));
        }
        let q = rand_emb(99999);
        let hits = mem.search(&q, 10).await.unwrap();
        assert_eq!(hits.len(), 10);
        let mut brute: Vec<(String, f32)> = all.iter().map(|(id, e)| (id.clone(), cos_dist(&q, e))).collect();
        brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let truth: HashSet<String> = brute.iter().take(10).map(|(id, _)| id.clone()).collect();
        let got: HashSet<String> = hits.iter().map(|h| h.id.clone()).collect();
        let recall = truth.intersection(&got).count() as f32 / 10.0;
        println!("recall@10 = {}", recall);
        assert!(recall >= 0.9, "recall {} < 0.9", recall);

        let first_id = &all[0].0;
        let sg = mem.expand(first_id, 2).await.unwrap();
        assert!(!sg.nodes.is_empty(), "expand returned no nodes");
        assert!(!sg.edges.is_empty(), "expand returned no edges");
    }
}
