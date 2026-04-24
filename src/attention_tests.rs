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

#[tokio::test]
async fn nudge_relation_changes_we_column() {
    let dir = tempdir().unwrap();
    let p = dir.path().join("a.db");
    let store = Arc::new(Store::open(p.to_str().unwrap()).await.unwrap());
    let att = Attention::new(store);
    let before: Vec<f32> = att.we.lock().unwrap().clone();
    for _ in 0..10 { att.nudge_relation("entity", 1.0); }
    let after: Vec<f32> = att.we.lock().unwrap().clone();
    let idx = RELATION_VOCAB.iter().position(|r| *r == "entity").unwrap();
    let stride = att.edge_feat_dim;
    let mut delta = 0f32;
    for h in 0..att.head_dim {
        let off = h * stride + idx;
        delta += (after[off] - before[off]).abs();
    }
    assert!(delta > 1e-6, "entity column must change after positive nudges, delta={}", delta);
    let mut untouched = 0f32;
    let other_idx = RELATION_VOCAB.iter().position(|r| *r == "episode").unwrap();
    for h in 0..att.head_dim {
        let off = h * stride + other_idx;
        untouched += (after[off] - before[off]).abs();
    }
    assert_eq!(untouched, 0.0, "other relation columns must not change");
}
