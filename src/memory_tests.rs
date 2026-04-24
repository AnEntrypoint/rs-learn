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
