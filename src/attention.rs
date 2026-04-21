use crate::observability;
use crate::store::Store;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

const WEEK_MS: f32 = 7.0 * 24.0 * 60.0 * 60.0 * 1000.0;
const RELATION_VOCAB: &[&str] = &["hnsw-neighbor", "entity", "mention", "episode", "saga"];

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SubgraphNode {
    pub id: String,
    pub embedding: Option<Vec<f32>>,
    pub created_at: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SubgraphEdge {
    pub src: String,
    pub dst: String,
    pub relation: Option<String>,
    pub weight: Option<f32>,
    pub created_at: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Subgraph {
    pub nodes: Vec<SubgraphNode>,
    pub edges: Vec<SubgraphEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Context {
    pub vector: Vec<f32>,
    pub weights: Vec<Vec<f32>>,
}

struct Stats {
    calls: AtomicU64,
    latency_us_total: AtomicU64,
    last_subgraph_size: AtomicU64,
}

pub struct Attention {
    _store: Arc<Store>,
    dim: usize,
    heads: usize,
    head_dim: usize,
    proj_dim: usize,
    edge_feat_dim: usize,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    we: Vec<f32>,
    wo: Vec<f32>,
    inv_sqrt_head: f32,
    stats: Arc<Stats>,
}

fn mulberry32(seed: u32) -> impl FnMut() -> f32 {
    let mut a: u32 = seed;
    move || {
        a = a.wrapping_add(0x6D2B79F5);
        let mut t = a;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        ((t ^ (t >> 14)) as f32) / 4294967296.0
    }
}

fn rand_matrix(rows: usize, cols: usize, rng: &mut impl FnMut() -> f32, scale: f32) -> Vec<f32> {
    let mut m = vec![0.0f32; rows * cols];
    for v in m.iter_mut() { *v = (rng() * 2.0 - 1.0) * scale; }
    m
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut s = 0.0f32;
    for i in 0..n { s += a[i] * b[i]; }
    s
}

fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    for r in 0..rows {
        let base = r * cols;
        out[r] = dot(&w[base..base + cols], x);
    }
}

fn layer_norm(x: &[f32]) -> Vec<f32> {
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let inv = 1.0 / (var + 1e-5).sqrt();
    x.iter().map(|v| (v - mean) * inv).collect()
}

fn now_ms_f32() -> f32 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as f32).unwrap_or(0.0)
}

impl Attention {
    pub fn new(store: Arc<Store>) -> Self {
        let dim = 768usize;
        let heads = 8usize;
        let head_dim = 96usize;
        let proj_dim = heads * head_dim;
        let edge_feat_dim = RELATION_VOCAB.len() + 2;
        let mut rng = mulberry32(42);
        let s_q = 1.0 / (dim as f32).sqrt();
        let s_e = 1.0 / (edge_feat_dim as f32).sqrt();
        let s_o = 1.0 / (proj_dim as f32).sqrt();
        let wq = rand_matrix(proj_dim, dim, &mut rng, s_q);
        let wk = rand_matrix(proj_dim, dim, &mut rng, s_q);
        let wv = rand_matrix(proj_dim, dim, &mut rng, s_q);
        let we = rand_matrix(head_dim, edge_feat_dim, &mut rng, s_e);
        let wo = rand_matrix(dim, proj_dim, &mut rng, s_o);
        let stats = Arc::new(Stats {
            calls: AtomicU64::new(0),
            latency_us_total: AtomicU64::new(0),
            last_subgraph_size: AtomicU64::new(0),
        });
        let s2 = stats.clone();
        observability::register("attention", move || {
            let calls = s2.calls.load(Ordering::Relaxed);
            let lat = s2.latency_us_total.load(Ordering::Relaxed);
            serde_json::json!({
                "calls": calls,
                "avg_latency_ms": if calls > 0 { (lat as f64) / (calls as f64) / 1000.0 } else { 0.0 },
                "last_subgraph_size": s2.last_subgraph_size.load(Ordering::Relaxed),
                "dim": 768, "heads": 8, "head_dim": 96,
            })
        });
        Self { _store: store, dim, heads, head_dim, proj_dim, edge_feat_dim,
            wq, wk, wv, we, wo, inv_sqrt_head: 1.0 / (head_dim as f32).sqrt(), stats }
    }

    pub fn attend(&self, query_emb: &[f32], subgraph: &Subgraph) -> Result<Context> {
        let t0 = std::time::Instant::now();
        if query_emb.len() != self.dim {
            anyhow::bail!("attention: query_emb must be len {}", self.dim);
        }
        let valid: Vec<(&SubgraphNode, &[f32], i64)> = subgraph.nodes.iter()
            .filter_map(|n| n.embedding.as_ref().filter(|e| e.len() == self.dim)
                .map(|e| (n, e.as_slice(), n.created_at.unwrap_or(0))))
            .collect();
        let n = valid.len();
        self.stats.last_subgraph_size.store(n as u64, Ordering::Relaxed);
        if n == 0 {
            self.stats.calls.fetch_add(1, Ordering::Relaxed);
            return Ok(Context { vector: layer_norm(query_emb), weights: vec![] });
        }
        let mut q = vec![0.0f32; self.proj_dim];
        matvec(&self.wq, self.proj_dim, self.dim, query_emb, &mut q);
        let mut k_mat = vec![0.0f32; n * self.proj_dim];
        let mut v_mat = vec![0.0f32; n * self.proj_dim];
        for (i, (_, emb, _)) in valid.iter().enumerate() {
            matvec(&self.wk, self.proj_dim, self.dim, emb, &mut k_mat[i * self.proj_dim..(i + 1) * self.proj_dim]);
            matvec(&self.wv, self.proj_dim, self.dim, emb, &mut v_mat[i * self.proj_dim..(i + 1) * self.proj_dim]);
        }
        let now = now_ms_f32();
        let mut edge_by_dst: std::collections::HashMap<&str, &SubgraphEdge> = std::collections::HashMap::new();
        for e in &subgraph.edges { edge_by_dst.entry(e.dst.as_str()).or_insert(e); }
        let mut e_proj = vec![0.0f32; self.head_dim];
        let mut feat = vec![0.0f32; self.edge_feat_dim];
        for (i, (node, _, ts)) in valid.iter().enumerate() {
            let Some(edge) = edge_by_dst.get(node.id.as_str()) else { continue };
            for v in feat.iter_mut() { *v = 0.0; }
            let rel = edge.relation.as_deref().unwrap_or("").split("-L").next().unwrap_or("");
            if let Some(idx) = RELATION_VOCAB.iter().position(|r| *r == rel) { feat[idx] = 1.0; }
            feat[RELATION_VOCAB.len()] = (-(now - (*ts as f32)) / WEEK_MS).exp();
            feat[RELATION_VOCAB.len() + 1] = edge.weight.unwrap_or(1.0);
            matvec(&self.we, self.head_dim, self.edge_feat_dim, &feat, &mut e_proj);
            for h in 0..self.heads {
                let off = h * self.head_dim;
                for d in 0..self.head_dim { k_mat[i * self.proj_dim + off + d] += e_proj[d]; }
            }
        }
        let mut weights: Vec<Vec<f32>> = Vec::with_capacity(self.heads);
        for h in 0..self.heads {
            let off = h * self.head_dim;
            let q_slice = &q[off..off + self.head_dim];
            let mut scores = vec![0.0f32; n];
            let mut max_s = f32::NEG_INFINITY;
            for i in 0..n {
                let k_slice = &k_mat[i * self.proj_dim + off..i * self.proj_dim + off + self.head_dim];
                let s = dot(q_slice, k_slice) * self.inv_sqrt_head;
                scores[i] = s;
                if s > max_s { max_s = s; }
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                let x = (*s - max_s).max(-30.0);
                *s = x.exp();
                sum += *s;
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for s in scores.iter_mut() { *s *= inv; }
            weights.push(scores);
        }
        let mut concat = vec![0.0f32; self.proj_dim];
        for h in 0..self.heads {
            let off = h * self.head_dim;
            for i in 0..n {
                let w = weights[h][i];
                for d in 0..self.head_dim { concat[off + d] += w * v_mat[i * self.proj_dim + off + d]; }
            }
        }
        let mut proj = vec![0.0f32; self.dim];
        matvec(&self.wo, self.dim, self.proj_dim, &concat, &mut proj);
        for i in 0..self.dim { proj[i] += query_emb[i]; }
        let vector = layer_norm(&proj);
        self.stats.calls.fetch_add(1, Ordering::Relaxed);
        self.stats.latency_us_total.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
        Ok(Context { vector, weights })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn rand_emb(seed: u32, dim: usize) -> Vec<f32> {
        let mut rng = mulberry32(seed);
        (0..dim).map(|_| rng() * 2.0 - 1.0).collect()
    }

    #[tokio::test]
    async fn attend_basic() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("a.db");
        let store = Arc::new(Store::open(p.to_str().unwrap()).await.unwrap());
        let att = Attention::new(store);
        let nodes: Vec<SubgraphNode> = (0..16).map(|i| SubgraphNode {
            id: format!("n{}", i), embedding: Some(rand_emb(100 + i as u32, 768)), created_at: Some(0),
        }).collect();
        let edges: Vec<SubgraphEdge> = (0..16).map(|i| SubgraphEdge {
            src: "q".into(), dst: format!("n{}", i), relation: Some("entity".into()),
            weight: Some(1.0), created_at: Some(0),
        }).collect();
        let sg = Subgraph { nodes, edges };
        let q = rand_emb(7, 768);
        let t0 = std::time::Instant::now();
        let ctx = att.attend(&q, &sg).unwrap();
        let elapsed = t0.elapsed();
        assert_eq!(ctx.vector.len(), 768);
        assert!(ctx.vector.iter().all(|v| v.is_finite()));
        assert_eq!(ctx.weights.len(), 8);
        for w in &ctx.weights {
            assert_eq!(w.len(), 16);
            let sum: f32 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "head sum {}", sum);
            assert!(w.iter().all(|v| v.is_finite()));
        }
        println!("attend 16-node elapsed={:?}", elapsed);
    }
}
