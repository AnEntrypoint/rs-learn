use super::*;
use std::sync::Mutex;
static ENV_GUARD: Mutex<()> = Mutex::new(());

async fn tmp_store() -> Arc<Store> {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();
    drop(tmp);
    Arc::new(Store::open(&path).await.unwrap())
}

#[tokio::test]
async fn env_lambda_override() {
    let _g = ENV_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    unsafe { std::env::set_var("RS_LEARN_EWC_LAMBDA", "42.5"); }
    let store = tmp_store().await;
    let dl = DeepLoop::new(store);
    assert_eq!(dl.lambda, 42.5);
    unsafe { std::env::remove_var("RS_LEARN_EWC_LAMBDA"); }
    let store2 = tmp_store().await;
    let dl2 = DeepLoop::new(store2);
    assert_eq!(dl2.lambda, DEFAULT_LAMBDA);
}

#[tokio::test]
async fn consolidate_writes_fisher_and_snapshot() {
    let _g = ENV_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    unsafe { std::env::remove_var("RS_LEARN_EWC_LAMBDA"); }
    let store = tmp_store().await;
    let mut dl = DeepLoop::new(store.clone());
    let params = vec![0.1f32, 0.2, 0.3];
    let grads = vec![1.0f32, 2.0, 3.0];
    dl.consolidate("layer0", &params, &grads).await.unwrap();
    let loaded = store.load_fisher_vec("layer0").await.unwrap();
    assert_eq!(loaded.len(), 3);
    assert!(loaded[0] > 0.0 && loaded[1] > loaded[0] && loaded[2] > loaded[1]);
    let pen = dl.ewc_penalty("layer0", &[0.2f32, 0.2, 0.3]);
    assert!(pen > 0.0);
    let snap = store.load_params_snapshot_vec("layer0").await.unwrap();
    assert_eq!(snap, params, "params_snapshot must persist to DB");
}

#[tokio::test]
async fn load_fisher_restores_params_snapshot() {
    let _g = ENV_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    unsafe { std::env::remove_var("RS_LEARN_EWC_LAMBDA"); }
    let store = tmp_store().await;
    let params = vec![0.5f32, 0.6, 0.7];
    let grads = vec![1.0f32, 1.0, 1.0];
    { let mut dl = DeepLoop::new(store.clone()); dl.consolidate("adapter", &params, &grads).await.unwrap(); }
    let mut dl2 = DeepLoop::new(store.clone());
    assert!(dl2.params_snapshot.is_empty(), "fresh instance has no snapshot");
    dl2.load_fisher("adapter").await.unwrap();
    assert!(!dl2.params_snapshot.is_empty(), "load_fisher must restore params_snapshot");
    let pen = dl2.ewc_penalty("adapter", &[0.6f32, 0.6, 0.7]);
    assert!(pen > 0.0, "ewc_penalty must be >0 after restart when params deviate from snapshot");
}

#[tokio::test]
async fn observability_reports_boundary_fires() {
    let _g = ENV_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    unsafe { std::env::remove_var("RS_LEARN_EWC_LAMBDA"); }
    let store = tmp_store().await;
    let mut dl = DeepLoop::new(store);
    let before = observability::dump();
    let b0 = before.get("deep").and_then(|v| v.get("boundary_fires")).and_then(|v| v.as_u64()).unwrap_or(0);
    for _ in 0..5 { let _ = dl.record_loss(0.1).await.unwrap(); }
    let _ = dl.record_loss(10.0).await.unwrap();
    let after = observability::dump();
    let deep = after.get("deep").expect("deep observability registered");
    let b1 = deep.get("boundary_fires").and_then(|v| v.as_u64()).unwrap();
    assert!(b1 > b0, "boundary_fires must increment: {} -> {}", b0, b1);
    assert!(deep.get("window_mean").is_some());
    assert!(deep.get("window_stddev").is_some());
    let n = deep.get("samples_in_window").and_then(|v| v.as_u64()).unwrap();
    assert!(n > 0);
}

#[tokio::test]
async fn record_loss_triggers_boundary() {
    let _g = ENV_GUARD.lock().unwrap_or_else(|p| p.into_inner());
    unsafe { std::env::remove_var("RS_LEARN_EWC_LAMBDA"); }
    let store = tmp_store().await;
    let mut dl = DeepLoop::new(store);
    for _ in 0..5 { let _ = dl.record_loss(0.1).await.unwrap(); }
    let spike = dl.record_loss(10.0).await.unwrap();
    assert!(spike, "z-score spike should trigger boundary");

    let store2 = tmp_store().await;
    let mut dl2 = DeepLoop::new(store2);
    for i in 0..5 { let _ = dl2.record_loss(0.1 + 0.01 * i as f32).await.unwrap(); }
    let calm = dl2.record_loss(0.13).await.unwrap();
    assert!(!calm, "within-noise loss must not trigger boundary");
}
