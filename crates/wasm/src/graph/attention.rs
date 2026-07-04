use crate::graph::types::EdgeRow;
use serde::{Deserialize, Serialize};

pub const RELATION_VOCAB: &[&str] = &["hnsw-neighbor", "entity", "mention", "episode", "saga"];
pub const WEEK_MS: f32 = 7.0 * 24.0 * 60.0 * 60.0 * 1000.0;

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

impl From<&EdgeRow> for SubgraphEdge {
    fn from(e: &EdgeRow) -> Self {
        Self {
            src: e.src.clone(),
            dst: e.dst.clone(),
            relation: e.relation.clone(),
            weight: e.weight.map(|w| w as f32),
            created_at: e.created_at,
        }
    }
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

pub struct Attention {
    pub dim: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub proj_dim: usize,
    pub edge_feat_dim: usize,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub we: Vec<f32>,
    pub wo: Vec<f32>,
    pub inv_sqrt_head: f32,
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

pub fn layer_norm(x: &[f32]) -> Vec<f32> {
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    let inv = 1.0 / (var + 1e-5).sqrt();
    x.iter().map(|v| (v - mean) * inv).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0f32;
    for i in 0..a.len() { s += a[i] * b[i]; }
    s
}

fn axpy(scale: f32, src: &[f32], dst: &mut [f32]) {
    for i in 0..src.len() { dst[i] += scale * src[i]; }
}

fn matvec(m: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    for r in 0..rows {
        let mut s = 0f32;
        let off = r * cols;
        for c in 0..cols { s += m[off + c] * x[c]; }
        out[r] = s;
    }
}

impl Attention {
    pub fn new(dim: usize, heads: usize, head_dim: usize, seed: u32) -> Self {
        let proj_dim = heads * head_dim;
        let edge_feat_dim = RELATION_VOCAB.len() + 2;
        let mut rng = mulberry32(seed);
        let s_q = 1.0 / (dim as f32).sqrt();
        let s_e = 1.0 / (edge_feat_dim as f32).sqrt();
        let s_o = 1.0 / (proj_dim as f32).sqrt();
        let wq = rand_matrix(proj_dim, dim, &mut rng, s_q);
        let wk = rand_matrix(proj_dim, dim, &mut rng, s_q);
        let wv = rand_matrix(proj_dim, dim, &mut rng, s_q);
        let we = rand_matrix(head_dim, edge_feat_dim, &mut rng, s_e);
        let wo = rand_matrix(dim, proj_dim, &mut rng, s_o);
        Self {
            dim, heads, head_dim, proj_dim, edge_feat_dim,
            wq, wk, wv, we, wo,
            inv_sqrt_head: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    pub fn default_for(dim: usize) -> Self {
        Self::new(dim, 8, dim / 8, 42)
    }

    pub fn nudge_relation(&mut self, relation: &str, signed_quality: f32) {
        if !signed_quality.is_finite() { return; }
        let rel = relation.split("-L").next().unwrap_or("");
        let idx = match RELATION_VOCAB.iter().position(|r| *r == rel) {
            Some(i) => i, None => return,
        };
        let alpha: f32 = 0.05;
        let scale = 1.0 + signed_quality.clamp(-1.0, 1.0) * alpha;
        let stride = self.edge_feat_dim;
        for h in 0..self.head_dim {
            let off = h * stride + idx;
            if off < self.we.len() { self.we[off] *= scale; }
        }
    }

    pub fn attend(&self, query_emb: &[f32], subgraph: &Subgraph, now_ms: i64) -> Result<Context, String> {
        if query_emb.len() != self.dim {
            return Err(format!("attention: query_emb must be len {}", self.dim));
        }
        let valid: Vec<(&SubgraphNode, &[f32], i64)> = subgraph.nodes.iter()
            .filter_map(|n| n.embedding.as_ref().filter(|e| e.len() == self.dim)
                .map(|e| (n, e.as_slice(), n.created_at.unwrap_or(0))))
            .collect();
        let n = valid.len();
        if n == 0 {
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
        let now = now_ms as f32;
        let mut edge_by_dst: std::collections::HashMap<&str, &SubgraphEdge> = std::collections::HashMap::new();
        for e in &subgraph.edges { edge_by_dst.entry(e.dst.as_str()).or_insert(e); }
        let mut e_proj = vec![0.0f32; self.head_dim];
        let mut feat = vec![0.0f32; self.edge_feat_dim];
        for (i, (node, _, ts)) in valid.iter().enumerate() {
            let edge = match edge_by_dst.get(node.id.as_str()) { Some(e) => e, None => continue };
            for v in feat.iter_mut() { *v = 0.0; }
            let rel = edge.relation.as_deref().unwrap_or("").split("-L").next().unwrap_or("");
            if let Some(idx) = RELATION_VOCAB.iter().position(|r| *r == rel) { feat[idx] = 1.0; }
            feat[RELATION_VOCAB.len()] = (((*ts as f32) - now).min(0.0) / WEEK_MS).exp();
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
            let dst = &mut concat[off..off + self.head_dim];
            for i in 0..n {
                let w = weights[h][i];
                let src = &v_mat[i * self.proj_dim + off..i * self.proj_dim + off + self.head_dim];
                axpy(w, src, dst);
            }
        }
        let mut proj = vec![0.0f32; self.dim];
        matvec(&self.wo, self.dim, self.proj_dim, &concat, &mut proj);
        for i in 0..self.dim { proj[i] += query_emb[i]; }
        let vector = layer_norm(&proj);
        Ok(Context { vector, weights })
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    fn mk_emb(dim: usize, seed: u8) -> Vec<f32> {
        (0..dim).map(|i| ((i as u8).wrapping_add(seed) as f32) / 255.0 - 0.5).collect()
    }

    #[test]
    fn new_initializes_consistent_dims() {
        let a = Attention::new(64, 8, 8, 1);
        assert_eq!(a.dim, 64);
        assert_eq!(a.heads, 8);
        assert_eq!(a.head_dim, 8);
        assert_eq!(a.proj_dim, 64);
        assert_eq!(a.wq.len(), 64 * 64);
        assert_eq!(a.wk.len(), 64 * 64);
        assert_eq!(a.wv.len(), 64 * 64);
        assert_eq!(a.wo.len(), 64 * 64);
        assert_eq!(a.we.len(), 8 * (RELATION_VOCAB.len() + 2));
    }

    #[test]
    fn empty_subgraph_returns_normalized_query() {
        let a = Attention::new(64, 8, 8, 1);
        let q = mk_emb(64, 3);
        let ctx = a.attend(&q, &Subgraph::default(), 0).unwrap();
        assert_eq!(ctx.vector.len(), 64);
        assert!(ctx.weights.is_empty());
        let mean: f32 = ctx.vector.iter().sum::<f32>() / 64.0;
        assert!(mean.abs() < 1e-4, "layer_norm output should be ~zero mean, got {}", mean);
    }

    #[test]
    fn attend_rejects_wrong_dim() {
        let a = Attention::new(64, 8, 8, 1);
        let q = vec![0.0; 32];
        assert!(a.attend(&q, &Subgraph::default(), 0).is_err());
    }

    #[test]
    fn attention_weights_sum_to_one_per_head() {
        let a = Attention::new(64, 4, 16, 7);
        let q = mk_emb(64, 1);
        let nodes = vec![
            SubgraphNode { id: "n1".into(), embedding: Some(mk_emb(64, 2)), created_at: Some(1000) },
            SubgraphNode { id: "n2".into(), embedding: Some(mk_emb(64, 3)), created_at: Some(2000) },
            SubgraphNode { id: "n3".into(), embedding: Some(mk_emb(64, 4)), created_at: Some(3000) },
        ];
        let edges = vec![
            SubgraphEdge { src: "q".into(), dst: "n1".into(), relation: Some("entity".into()), weight: Some(1.0), created_at: Some(1000) },
            SubgraphEdge { src: "q".into(), dst: "n2".into(), relation: Some("mention".into()), weight: Some(0.5), created_at: Some(2000) },
            SubgraphEdge { src: "q".into(), dst: "n3".into(), relation: Some("episode".into()), weight: Some(0.3), created_at: Some(3000) },
        ];
        let ctx = a.attend(&q, &Subgraph { nodes, edges }, 5000).unwrap();
        assert_eq!(ctx.weights.len(), 4, "8 heads -> 4 weight rows");
        for (h, row) in ctx.weights.iter().enumerate() {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-3, "head {} weights sum {} should be 1", h, sum);
            assert_eq!(row.len(), 3);
        }
    }

    #[test]
    fn skips_nodes_with_wrong_embed_dim() {
        let a = Attention::new(64, 8, 8, 1);
        let q = mk_emb(64, 1);
        let nodes = vec![
            SubgraphNode { id: "ok".into(), embedding: Some(mk_emb(64, 2)), created_at: Some(0) },
            SubgraphNode { id: "bad".into(), embedding: Some(mk_emb(32, 3)), created_at: Some(0) },
            SubgraphNode { id: "none".into(), embedding: None, created_at: Some(0) },
        ];
        let ctx = a.attend(&q, &Subgraph { nodes, edges: vec![] }, 0).unwrap();
        for row in &ctx.weights {
            assert_eq!(row.len(), 1, "only 'ok' has valid embedding");
        }
    }

    #[test]
    fn nudge_relation_known_changes_we() {
        let mut a = Attention::new(64, 8, 8, 1);
        let before: Vec<f32> = a.we.clone();
        a.nudge_relation("entity", 1.0);
        assert_ne!(a.we, before);
    }

    #[test]
    fn nudge_relation_unknown_is_noop() {
        let mut a = Attention::new(64, 8, 8, 1);
        let before: Vec<f32> = a.we.clone();
        a.nudge_relation("does-not-exist", 1.0);
        assert_eq!(a.we, before);
    }

    #[test]
    fn nudge_relation_non_finite_is_noop() {
        let mut a = Attention::new(64, 8, 8, 1);
        let before: Vec<f32> = a.we.clone();
        a.nudge_relation("entity", f32::NAN);
        assert_eq!(a.we, before);
    }

    #[test]
    fn output_residual_adds_query() {
        let a = Attention::new(64, 4, 16, 13);
        let q = mk_emb(64, 1);
        let ctx_empty = a.attend(&q, &Subgraph::default(), 0).unwrap();
        let q_norm = layer_norm(&q);
        for i in 0..64 {
            assert!((ctx_empty.vector[i] - q_norm[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn edge_row_converts_to_subgraph_edge() {
        let e = EdgeRow {
            id: "e1".into(), src: "a".into(), dst: "b".into(),
            relation: Some("entity".into()), fact: None, embedding: None,
            weight: Some(0.5), group_id: None, created_at: Some(100),
            expired_at: None, valid_at: Some(100), invalid_at: None,
        };
        let s: SubgraphEdge = (&e).into();
        assert_eq!(s.src, "a");
        assert_eq!(s.weight, Some(0.5));
        assert_eq!(s.relation, Some("entity".into()));
    }
}
