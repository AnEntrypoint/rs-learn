use super::*;

#[tokio::test]
async fn roundtrip() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();
    drop(tmp);
    let store = Arc::new(Store::open(&path).await.unwrap());
    let targets = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let mut r = Router::new(store.clone(), targets.clone());
    let emb = vec![0.01f32; IN];
    let out = r.route(&emb, &RouteCtx { task_type: Some("default".into()), estimated_tokens: 500 });
    assert_eq!(out.algo, "rule");
    assert_eq!(out.context_bucket, 0);
    r.trained = true;
    let blob1 = r.pack();
    r.save().await.unwrap();
    drop(r);
    let mut r2 = Router::new(store, targets);
    let v = r2.load().await.unwrap();
    assert_eq!(v, Some(1));
    let blob2 = r2.pack();
    assert_eq!(blob1, blob2, "round-trip bytes must match");
    let t0 = std::time::Instant::now();
    for _ in 0..100 { let _ = r2.route(&emb, &RouteCtx::default()); }
    let per_us = t0.elapsed().as_micros() / 100;
    assert!(per_us < 1000, "inference p50 {}us >= 1000us", per_us);
}

#[tokio::test]
async fn train_softmax_shifts_prediction_toward_chosen() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();
    drop(tmp);
    let store = Arc::new(Store::open(&path).await.unwrap());
    let targets = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let mut r = Router::new(store, targets.clone());
    let emb: Vec<f32> = (0..IN).map(|i| ((i as f32) * 0.01).sin()).collect();
    let before = forward(&r.w, &r.heads, &emb, targets.len());
    let samples: Vec<TrainSample> = (0..200).map(|_| TrainSample {
        embedding: emb.clone(), chosen_target: "b".into(), quality: 0.95, estimated_tokens: 500,
    }).collect();
    let applied = r.train(&samples).unwrap();
    assert_eq!(applied, 200);
    let after = forward(&r.w, &r.heads, &emb, targets.len());
    assert!(after.ml[1] - before.ml[1] > after.ml[0] - before.ml[0]);
    assert!(after.ml[1] - before.ml[1] > after.ml[2] - before.ml[2]);
    let max_abs = r.heads.model.iter().map(|x| x.abs()).fold(0f32, f32::max);
    assert!(max_abs < 10.0, "weights should stay bounded, got max {max_abs}");
}

#[tokio::test]
async fn epsilon_one_forces_non_argmax() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();
    drop(tmp);
    let store = Arc::new(Store::open(&path).await.unwrap());
    let targets = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let mut r = Router::new(store, targets.clone());
    r.trained = true;
    let emb: Vec<f32> = (0..IN).map(|i| ((i as f32) * 0.01).sin()).collect();
    std::env::set_var("RS_LEARN_ROUTER_EPSILON", "0.0");
    std::env::remove_var("RS_LEARN_ROUTER_SEED");
    let baseline = r.route(&emb, &RouteCtx::default());
    assert!(!baseline.exploration);
    std::env::set_var("RS_LEARN_ROUTER_EPSILON", "1.0");
    std::env::set_var("RS_LEARN_ROUTER_SEED", "12345");
    let explored = r.route(&emb, &RouteCtx::default());
    std::env::remove_var("RS_LEARN_ROUTER_EPSILON");
    std::env::remove_var("RS_LEARN_ROUTER_SEED");
    assert!(explored.exploration);
    assert_ne!(explored.model, baseline.model);
}

#[tokio::test]
async fn router_threshold_env_override() {
    std::env::set_var("RS_LEARN_ROUTER_THRESHOLD", "50");
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();
    drop(tmp);
    let store = Arc::new(Store::open(&path).await.unwrap());
    let targets = vec!["a".to_string(), "b".to_string()];
    let mut r = Router::new(store, targets);
    std::env::remove_var("RS_LEARN_ROUTER_THRESHOLD");
    assert!(!r.trained);
    let emb: Vec<f32> = (0..IN).map(|i| ((i as f32) * 0.01).sin()).collect();
    let samples: Vec<TrainSample> = (0..50).map(|_| TrainSample {
        embedding: emb.clone(), chosen_target: "a".into(), quality: 0.9, estimated_tokens: 500,
    }).collect();
    r.train(&samples).unwrap();
    assert!(r.trained);
    assert_eq!(r.threshold_obs.load(std::sync::atomic::Ordering::Relaxed), 50);
    assert_eq!(r.traj_count_obs.load(std::sync::atomic::Ordering::Relaxed), 50);
}

#[tokio::test]
async fn epsilon_single_target_no_op() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();
    drop(tmp);
    let store = Arc::new(Store::open(&path).await.unwrap());
    let targets = vec!["only".to_string()];
    let mut r = Router::new(store, targets);
    r.trained = true;
    let emb: Vec<f32> = (0..IN).map(|i| ((i as f32) * 0.01).sin()).collect();
    std::env::set_var("RS_LEARN_ROUTER_EPSILON", "0.5");
    std::env::set_var("RS_LEARN_ROUTER_SEED", "1");
    let out = r.route(&emb, &RouteCtx::default());
    std::env::remove_var("RS_LEARN_ROUTER_EPSILON");
    std::env::remove_var("RS_LEARN_ROUTER_SEED");
    assert!(!out.exploration);
    assert_eq!(out.model, "only");
}

#[tokio::test]
async fn record_outcome_updates_per_target_stats() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();
    drop(tmp);
    let store = Arc::new(Store::open(&path).await.unwrap());
    let targets = vec!["a".to_string(), "b".to_string()];
    let r = Router::new(store, targets);
    for _ in 0..20 { r.record_outcome("a", 0.9); }
    let count = r.per_target_counts[0].load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(count, 20);
    let q = r.per_target_quality_milli[0].load(std::sync::atomic::Ordering::Relaxed) as f64 / 1000.0;
    assert!(q > 0.5 && q < 0.9, "quality should EMA toward 0.9, got {}", q);
    assert_eq!(r.per_target_counts[1].load(std::sync::atomic::Ordering::Relaxed), 0);
}
