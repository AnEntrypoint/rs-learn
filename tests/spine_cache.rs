use rs_learn::simd;
use rs_learn::cache::EmbeddingCache;
use rs_learn::embeddings::Embedder;
use rs_learn::spine::TrajectorySpine;
use rs_learn::store::types::TrajectoryRow;
use rs_learn::store::Store;
use std::sync::Arc;
use std::time::Duration;

fn rnd(seed: u32, n: usize) -> Vec<f32> {
    let mut a = seed;
    (0..n).map(|_| {
        a = a.wrapping_mul(1664525).wrapping_add(1013904223);
        ((a >> 8) as f32 / 16777216.0) - 0.5
    }).collect()
}

#[test]
fn simd_scalar_parity_full_kernels() {
    let a = rnd(11, 768);
    let b = rnd(12, 768);
    assert!((simd::dot(&a, &b) - simd::dot_scalar(&a, &b)).abs() < 1e-3);
    let mut y1 = rnd(13, 768);
    let mut y2 = y1.clone();
    simd::axpy(0.21, &a, &mut y1);
    simd::axpy_scalar(0.21, &a, &mut y2);
    let diff_axpy: f32 = y1.iter().zip(y2.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
    assert!(diff_axpy < 1e-5);
    let w = rnd(14, 768 * 768);
    let mut o1 = vec![0.0f32; 768];
    let mut o2 = vec![0.0f32; 768];
    simd::matvec(&w, 768, 768, &a, &mut o1);
    simd::matvec_scalar(&w, 768, 768, &a, &mut o2);
    let diff_mv: f32 = o1.iter().zip(o2.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
    assert!(diff_mv < 1e-2, "matvec max diff {}", diff_mv);
}

#[tokio::test]
async fn cache_hit_after_miss_and_coalesces() {
    let embedder = Arc::new(Embedder::new());
    let cache = EmbeddingCache::new(embedder, 100, Duration::from_secs(60));
    let v1 = cache.embed("hello world").await.unwrap();
    let v2 = cache.embed("hello world").await.unwrap();
    assert_eq!(v1, v2);
    assert_eq!(cache.misses(), 1);
    assert_eq!(cache.hits(), 1);
    let c2 = cache.clone();
    let c3 = cache.clone();
    let (r1, r2) = tokio::join!(
        tokio::spawn(async move { c2.embed("concurrent key").await.unwrap() }),
        tokio::spawn(async move { c3.embed("concurrent key").await.unwrap() }),
    );
    assert_eq!(r1.unwrap(), r2.unwrap());
}

#[tokio::test]
async fn spine_drops_on_overflow_and_counter_fires() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();
    drop(tmp);
    let store = Arc::new(Store::open(&path).await.unwrap());
    let spine = TrajectorySpine::new(store, 4);
    for i in 0..200u32 {
        let row = TrajectoryRow {
            id: format!("r{i}"),
            session_id: None,
            query: None,
            query_embedding: None,
            retrieved_ids: None,
            router_decision: None,
            response: None,
            activations: None,
            quality: None,
            latency_ms: None,
            created_at: None,
        };
        spine.send(row).await.unwrap();
    }
    spine.close().await;
    assert!(spine.dropped_count() > 0, "dropped counter must fire under flood, got {}", spine.dropped_count());
    assert!(spine.written_count() > 0, "some writes must succeed, got {}", spine.written_count());
}
