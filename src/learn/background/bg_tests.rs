use super::*;
use crate::store::types::TrajectoryRow;
use kmeans::mulberry32;

#[test]
fn kmeans_deterministic_same_seed() {
    let mut rng = mulberry32(7);
    let vectors: Vec<Vec<f32>> = (0..30).map(|_| (0..IN).map(|_| rng() - 0.5).collect()).collect();
    let a = kmeans_plus_plus(&vectors, 4, 42);
    let b = kmeans_plus_plus(&vectors, 4, 42);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) { assert_eq!(x.cluster, y.cluster); }
}

#[tokio::test]
async fn run_once_writes_patterns_and_trains() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();
    drop(tmp);
    let store = Arc::new(Store::open(&path).await.unwrap());
    let targets = vec!["a".to_string(), "b".to_string()];
    let router = Arc::new(Mutex::new(Router::new(store.clone(), targets.clone())));
    let reasoning = Arc::new(ReasoningBank::new(store.clone()));
    let mut rng = mulberry32(11);
    for i in 0..8 {
        let emb: Vec<f32> = (0..IN).map(|_| rng() - 0.5).collect();
        let model = if i % 2 == 0 { "a" } else { "b" };
        store.insert_trajectory(&TrajectoryRow {
            id: format!("t{}", i), session_id: Some("s".into()),
            query: Some(format!("q{}", i)), query_embedding: Some(emb), retrieved_ids: None,
            router_decision: Some(format!("{{\"model\":\"{}\"}}", model)),
            response: Some("r".into()), activations: None,
            quality: Some(0.9), latency_ms: Some(1), created_at: Some(1000 + i as i64),
        }).await.unwrap();
    }
    let bg = BackgroundLoop::new(store.clone(), router.clone(), None, reasoning, None);
    let stats = bg.run_once().await.unwrap();
    assert!(stats.patterns_written > 0, "no patterns written");
    assert!(stats.trained_on > 0, "router.train not invoked");
    assert!(store.count_rows("patterns").await > 0, "patterns table empty");
    assert!(store.count_rows("reasoning_bank").await > 0, "reasoning_bank empty");
}

#[tokio::test]
async fn run_once_trains_middle_quality() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();
    drop(tmp);
    let store = Arc::new(Store::open(&path).await.unwrap());
    let targets = vec!["a".to_string(), "b".to_string()];
    let router = Arc::new(Mutex::new(Router::new(store.clone(), targets)));
    let reasoning = Arc::new(ReasoningBank::new(store.clone()));
    let mut rng = mulberry32(17);
    let qualities = [0.2f64, 0.55, 0.65, 0.9];
    for (i, q) in qualities.iter().enumerate() {
        let emb: Vec<f32> = (0..IN).map(|_| rng() - 0.5).collect();
        let model = if i % 2 == 0 { "a" } else { "b" };
        store.insert_trajectory(&TrajectoryRow {
            id: format!("t{}", i), session_id: Some("s".into()),
            query: Some(format!("q{}", i)), query_embedding: Some(emb), retrieved_ids: None,
            router_decision: Some(format!("{{\"model\":\"{}\"}}", model)),
            response: Some("r".into()), activations: None,
            quality: Some(*q), latency_ms: Some(1), created_at: Some(2000 + i as i64),
        }).await.unwrap();
    }
    let bg = BackgroundLoop::new(store.clone(), router.clone(), None, reasoning, None);
    let stats = bg.run_once().await.unwrap();
    assert_eq!(stats.trained_on, 4, "expected all 4 mid-band samples trained, got {}", stats.trained_on);
}

#[tokio::test]
async fn summarize_prompt_uses_query_content_not_ids() {
    use crate::backend::AgentBackend;
    use crate::errors::{LlmError, Result as LlmResult};
    use async_trait::async_trait;
    use std::sync::Mutex as StdMutex;

    struct CapBackend(Arc<StdMutex<Vec<String>>>);
    #[async_trait]
    impl AgentBackend for CapBackend {
        async fn generate(&self, _s: &str, u: &str, _t: u64) -> LlmResult<serde_json::Value> {
            self.0.lock().unwrap().push(u.to_string());
            Err(LlmError::Validation("stub".into()))
        }
        fn name(&self) -> &'static str { "cap" }
    }

    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();
    drop(tmp);
    let store = Arc::new(Store::open(&path).await.unwrap());
    let targets = vec!["a".to_string()];
    let router = Arc::new(Mutex::new(Router::new(store.clone(), targets)));
    let reasoning = Arc::new(ReasoningBank::new(store.clone()));
    let hits = Arc::new(StdMutex::new(Vec::new()));
    let acp: Arc<dyn AgentBackend> = Arc::new(CapBackend(hits.clone()));
    let mut rng = kmeans::mulberry32(9);
    for i in 0..4 {
        let emb: Vec<f32> = (0..IN).map(|_| rng() - 0.5).collect();
        store.insert_trajectory(&TrajectoryRow {
            id: format!("t{}", i), session_id: Some("s".into()),
            query: Some(format!("how do I refactor {} module", i)),
            query_embedding: Some(emb), retrieved_ids: None,
            router_decision: Some("{\"model\":\"a\"}".to_string()),
            response: None, activations: None,
            quality: Some(0.9), latency_ms: Some(1), created_at: Some(100 + i as i64),
        }).await.unwrap();
    }
    let bg = BackgroundLoop::new(store.clone(), router, Some(acp), reasoning, None);
    let _ = bg.run_once().await.unwrap();
    let captured = hits.lock().unwrap().clone();
    assert!(!captured.is_empty(), "backend was not called");
    let p = &captured[0];
    assert!(p.contains("refactor"), "prompt lacks query content: {p}");
    assert!(!p.contains("t0,"), "prompt must not contain raw trajectory ids");
}

#[test]
fn estimated_tokens_word_count_approximation() {
    let query = "one two three four five six seven eight nine ten";
    let estimated = (query.split_whitespace().count() * 4 / 3) as u64;
    assert!(estimated >= 10 && estimated <= 20, "expected in [10,20], got {estimated}");
}

#[tokio::test]
async fn eviction_removes_stale_low_quality_reasoning() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();
    drop(tmp);
    let store = Arc::new(Store::open(&path).await.unwrap());
    let now = crate::store::now_ms();
    let eight_days_ago = now - (8 * 86_400_000);
    for i in 0..5 {
        store.insert_reasoning(&crate::store::types::ReasoningRow {
            id: format!("stale-{}", i), pattern_id: None, strategy: "bad".into(),
            success_rate: Some(0.1), created_at: Some(eight_days_ago),
        }).await.unwrap();
    }
    for i in 0..20 {
        store.insert_reasoning(&crate::store::types::ReasoningRow {
            id: format!("good-{}", i), pattern_id: None, strategy: "good".into(),
            success_rate: Some(0.9), created_at: Some(now),
        }).await.unwrap();
    }
    let deleted = store.evict_stale_reasoning(7, 0.3).await.unwrap();
    assert_eq!(deleted, 5, "expected exactly 5 stale rows deleted, got {deleted}");
}
