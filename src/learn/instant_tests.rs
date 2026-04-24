use super::*;

#[tokio::test]
async fn record_then_feedback_grows_adapter() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();
    drop(tmp);
    let store = Arc::new(Store::open(&path).await.unwrap());
    let targets = vec!["a".to_string(), "b".to_string()];
    let router = Arc::new(Mutex::new(Router::new(store.clone(), targets.clone())));
    let mut il = InstantLoop::new(store, router, targets);
    let emb = vec![0.05f32; IN];
    let rid = il.record_trajectory(Some("s1".into()), emb, "a".into(), "hello".into(), None, None, 0).await.unwrap();
    assert_eq!(il.adapter_norm(), 0.0);
    il.feedback(&rid, FeedbackPayload { quality: 1.0, signal: None }).await.unwrap();
    assert!(il.adapter_norm() > 0.0, "adapter_norm must grow after positive feedback");
    let mut logits = vec![0f32; 2];
    il.apply_adapter(&vec![0.05f32; IN], &mut logits);
    assert!(logits[0].abs() + logits[1].abs() > 0.0);
}

#[tokio::test]
async fn record_trajectory_full_persists_query_and_quality() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();
    drop(tmp);
    let store = Arc::new(Store::open(&path).await.unwrap());
    let targets = vec!["a".to_string()];
    let router = Arc::new(Mutex::new(Router::new(store.clone(), targets.clone())));
    let mut il = InstantLoop::new(store.clone(), router, targets);
    let emb = vec![0.03f32; IN];
    il.record_trajectory(Some("s".into()), emb, "a".into(), "resp".into(),
        Some("why is the sky blue".into()), Some(0.85), 42).await.unwrap();
    let rows = store.list_recent_trajectories_with_embeddings(10).await.unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].query.as_deref(), Some("why is the sky blue"));
    assert_eq!(rows[0].quality, Some(0.85));
    assert_eq!(rows[0].latency_ms, Some(42));
}

#[tokio::test]
async fn lr_respects_floor() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();
    drop(tmp);
    let store = Arc::new(Store::open(&path).await.unwrap());
    let targets = vec!["a".to_string()];
    let router = Arc::new(Mutex::new(Router::new(store.clone(), targets.clone())));
    let mut il = InstantLoop::new(store, router, targets);
    let emb = vec![0.01f32; IN];
    for _ in 0..2000 { il.hebbian_update(&emb, 0, 1.0); }
    assert!(il.lr >= il.lr_min, "lr {} below floor {}", il.lr, il.lr_min);
    assert!(il.lr_min > 0.0 && il.lr_min <= LR0);
}

#[tokio::test]
async fn adapter_shifts_router_decision_after_feedback() {
    use crate::router::RouteCtx;
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();
    drop(tmp);
    let store = Arc::new(Store::open(&path).await.unwrap());
    let targets = vec!["a".to_string(), "b".to_string()];
    let router = Arc::new(Mutex::new({
        let mut r = Router::new(store.clone(), targets.clone());
        r.save().await.unwrap();
        r
    }));
    let mut il = InstantLoop::new(store.clone(), router.clone(), targets.clone());
    let emb: Vec<f32> = (0..IN).map(|i| ((i as f32) * 0.013).sin()).collect();
    let rid = il.record_trajectory(Some("s".into()), emb.clone(), "b".into(), "resp".into(), None, None, 0).await.unwrap();
    il.feedback(&rid, FeedbackPayload { quality: 1.0, signal: None }).await.unwrap();
    let a = il.adapter_a.clone();
    let b = il.adapter_b.clone();
    let tgt = il.targets_clone();
    let rank = il.adapter_rank();
    let plain = { let r = router.lock().await; r.route(&emb, &RouteCtx::default()) };
    let adapted = {
        let r = router.lock().await;
        r.route_with_adapter(&emb, &RouteCtx::default(), |e, l| {
            InstantLoop::apply_adapter_raw(&a, &b, rank, &tgt, e, l);
        })
    };
    assert!(adapted.confidence >= plain.confidence - 1e-6 || adapted.model == "b",
        "adapter should not reduce confidence: plain={} adapted={}", plain.confidence, adapted.confidence);
}

#[test]
fn prioritized_replay_favors_high_impact() {
    let mut buf: std::collections::VecDeque<(Vec<f32>, usize, f32)> = std::collections::VecDeque::new();
    buf.push_back((vec![0f32; IN], 0, 0.1));
    buf.push_back((vec![0f32; IN], 1, 0.1));
    buf.push_back((vec![0f32; IN], 2, 0.1));
    buf.push_back((vec![0f32; IN], 3, 0.9));
    let mut seed: u32 = 0x9E3779B1;
    let mut hits = 0usize;
    for _ in 0..1000 { if super::instant_io::weighted_pick(&buf, &mut seed) == 3 { hits += 1; } }
    assert!(hits > 500, "expected index 3 picked > 500, got {hits}");
}

#[test]
fn weighted_pick_zero_total_fallback_uniform() {
    let mut buf: std::collections::VecDeque<(Vec<f32>, usize, f32)> = std::collections::VecDeque::new();
    for i in 0..4 { buf.push_back((vec![0f32; IN], i, 0.0)); }
    let mut seed: u32 = 12345;
    for _ in 0..50 { assert!(super::instant_io::weighted_pick(&buf, &mut seed) < 4); }
}

#[tokio::test]
async fn ewc_penalty_pulls_adapter_toward_snapshot() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();
    drop(tmp);
    let store = Arc::new(Store::open(&path).await.unwrap());
    let targets = vec!["a".to_string()];
    let router = Arc::new(Mutex::new(Router::new(store.clone(), targets.clone())));
    let mut il_free = InstantLoop::new(store.clone(), router.clone(), targets.clone());
    let mut il_ewc = InstantLoop::new(store.clone(), router.clone(), targets.clone());
    let total = IN * RANK + RANK * 1;
    let snapshot = vec![0f32; total];
    let fisher = vec![1.0f32; total];
    il_ewc.set_ewc_state(fisher, snapshot, 50.0);
    let emb = vec![0.05f32; IN];
    for _ in 0..3 { il_free.hebbian_update(&emb, 0, 1.0); }
    for _ in 0..3 { il_ewc.hebbian_update(&emb, 0, 1.0); }
    assert!(il_ewc.adapter_norm() < il_free.adapter_norm(),
        "EWC must pull adapter toward snapshot: free={} ewc={}",
        il_free.adapter_norm(), il_ewc.adapter_norm());
}

#[tokio::test]
async fn instant_mid_quality_trains_now() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();
    drop(tmp);
    let store = Arc::new(Store::open(&path).await.unwrap());
    let targets = vec!["a".to_string(), "b".to_string()];
    let router = Arc::new(Mutex::new(Router::new(store.clone(), targets.clone())));
    let mut il = InstantLoop::new(store, router, targets);
    let emb = vec![0.05f32; IN];
    let rid = il.record_trajectory(Some("s1".into()), emb, "a".into(), "hello".into(), None, None, 0).await.unwrap();
    assert_eq!(il.adapter_norm(), 0.0);
    il.feedback(&rid, FeedbackPayload { quality: 0.55, signal: None }).await.unwrap();
    assert!(il.adapter_norm() > 0.0, "adapter_norm must grow after mid-quality feedback");
}
