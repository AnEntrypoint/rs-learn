// End-to-end integration tests — port of validate.js to Rust.
// Each section mirrors a subsystem exercised in validate.js.
// Live ACP section gated behind env RS_LEARN_ACP_LIVE=1 + RS_LEARN_ACP_COMMAND.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use rs_learn::attention::{Attention, Subgraph, SubgraphEdge, SubgraphNode};
use rs_learn::embeddings::{Embedder, EMBED_DIM};
use rs_learn::export::{export_patterns, export_preferences, export_safetensors, push_to_hugging_face};
use rs_learn::learn::background::{kmeans_centroids, kmeans_plus_plus, BackgroundLoop};
use rs_learn::learn::deep::DeepLoop;
use rs_learn::learn::instant::{FeedbackPayload, InstantLoop};
use rs_learn::learn::reasoning_bank::ReasoningBank;
use rs_learn::memory::{Memory, NodeInput};
use rs_learn::observability;
use rs_learn::router::{RouteCtx, Router, TrainSample};
use rs_learn::store::{NodeRow, PatternRow, PreferenceRow, ReasoningRow, Store};
use tokio::sync::Mutex;

fn tmp_db(tag: &str) -> (tempfile::TempDir, String) {
    let dir = tempfile::tempdir().expect("tempdir");
    let p: PathBuf = dir.path().join(format!("{}.db", tag));
    let path = p.to_str().unwrap().to_string();
    (dir, path)
}

fn rand_emb(seed: u32) -> Vec<f32> {
    let mut s = seed as u64;
    (0..EMBED_DIM).map(|_| {
        s = (s.wrapping_mul(9301).wrapping_add(49297)) % 233280;
        (s as f32 / 233280.0) * 2.0 - 1.0
    }).collect()
}

fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0f32, 0f32, 0f32);
    for i in 0..a.len() { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    dot / (na.sqrt() * nb.sqrt() + 1e-12)
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Store — open/close, migrate idempotent, schema present.
// ─────────────────────────────────────────────────────────────────────────────
#[tokio::test]
async fn section_01_store_open_close_and_schema() {
    let (_g, path) = tmp_db("s1");
    let store = Store::open(&path).await.expect("store open");
    store.migrate().await.expect("migrate 1");
    store.migrate().await.expect("migrate 2 idempotent");

    let now = rs_learn::store::now_ms();
    store.insert_node(&NodeRow {
        id: "n1".into(), name: "alpha".into(), r#type: Some("concept".into()),
        summary: Some("hello".into()), embedding: Some(rand_emb(1)),
        level: Some(0), created_at: Some(now),
            group_id: None,
    }).await.expect("insert node");
    assert_eq!(store.count_rows("nodes").await, 1);
    store.close().await;
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Embeddings — embed('hello') -> 768, cache roundtrip.
// Requires rs-search GGUF model to be resolvable; if unavailable, we skip.
// ─────────────────────────────────────────────────────────────────────────────
#[tokio::test]
async fn section_02_embeddings_len_and_cache() {
    let embedder = Embedder::new();
    let v1 = match embedder.embed("hello") {
        Ok(v) => v,
        Err(e) => { eprintln!("skipping section_02: {e}"); return; }
    };
    assert_eq!(v1.len(), EMBED_DIM, "embedding must be {} dims", EMBED_DIM);
    for x in &v1 { assert!(x.is_finite(), "non-finite component in embedding"); }
    let (h0, _m0, _c0) = rs_learn::embeddings::stats_snapshot();
    let v2 = embedder.embed("hello").expect("embed hit cache");
    assert_eq!(v1, v2, "cache must return byte-identical vector");
    let (h1, _, _) = rs_learn::embeddings::stats_snapshot();
    assert!(h1 > h0, "cache hit count must grow");
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Memory — add 200 random 768d, search k=10 recall≥0.9 vs brute, expand.
// ─────────────────────────────────────────────────────────────────────────────
#[tokio::test]
async fn section_03_memory_hnsw_recall_and_expand() {
    let (_g, path) = tmp_db("s3");
    let store = Arc::new(Store::open(&path).await.unwrap());
    let mem = Memory::new(store);
    let mut truth_vecs: Vec<(String, Vec<f32>)> = Vec::with_capacity(200);
    for i in 0..200u32 {
        let emb = rand_emb(i.wrapping_mul(17).wrapping_add(1));
        let id = format!("m{}", i);
        mem.add(NodeInput {
            id: Some(id.clone()),
            payload: serde_json::json!({ "i": i }),
            embedding: emb.clone(), level: None,
        }).await.expect("memory add");
        truth_vecs.push((id, emb));
    }
    let q = rand_emb(99_999);
    let hits = mem.search(&q, 10).await.expect("memory search");
    assert_eq!(hits.len(), 10, "search must return exactly k hits");

    let mut brute: Vec<(String, f32)> = truth_vecs.iter()
        .map(|(id, e)| (id.clone(), cos_sim(&q, e))).collect();
    brute.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let truth: HashSet<String> = brute.iter().take(10).map(|(id, _)| id.clone()).collect();
    let got: HashSet<String> = hits.iter().map(|h| h.id.clone()).collect();
    let recall = truth.intersection(&got).count() as f32 / 10.0;
    assert!(recall >= 0.9, "recall@10 = {} < 0.9", recall);

    let sg = mem.expand(&truth_vecs[0].0, 2).await.expect("expand");
    assert!(!sg.nodes.is_empty(), "expand must yield subgraph nodes");
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Attention — attend with 16-node subgraph, weights sum 1 per head.
// ─────────────────────────────────────────────────────────────────────────────
#[tokio::test]
async fn section_04_attention_weights_sum_to_one_per_head() {
    let (_g, path) = tmp_db("s4");
    let store = Arc::new(Store::open(&path).await.unwrap());
    let att = Attention::new(store);
    let nodes: Vec<SubgraphNode> = (0..16u32).map(|i| SubgraphNode {
        id: format!("n{}", i),
        embedding: Some(rand_emb(100 + i)),
        created_at: Some(0),
    }).collect();
    let edges: Vec<SubgraphEdge> = (0..16u32).map(|i| SubgraphEdge {
        src: "q".into(), dst: format!("n{}", i),
        relation: Some("entity".into()), weight: Some(1.0), created_at: Some(0),
    }).collect();
    let sg = Subgraph { nodes, edges };
    let q = rand_emb(42);
    let ctx = att.attend(&q, &sg).expect("attend");
    assert_eq!(ctx.vector.len(), EMBED_DIM);
    for v in &ctx.vector { assert!(v.is_finite(), "NaN/Inf in context vector"); }
    assert_eq!(ctx.weights.len(), 8, "must produce one weight vector per head");
    for (h, w) in ctx.weights.iter().enumerate() {
        assert_eq!(w.len(), 16, "head {} must have weight per node", h);
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "head {} weights sum {} must be 1", h, sum);
    }

    let empty = att.attend(&q, &Subgraph { nodes: vec![], edges: vec![] }).expect("attend empty");
    assert_eq!(empty.vector.len(), EMBED_DIM);
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Router — route, train, save/load roundtrip byte-equal.
// ─────────────────────────────────────────────────────────────────────────────
#[tokio::test]
async fn section_05_router_route_train_save_load_roundtrip() {
    let (_g, path) = tmp_db("s5");
    let store = Arc::new(Store::open(&path).await.unwrap());
    let targets = vec!["default".to_string(), "think".to_string(), "background".to_string()];
    let mut r = Router::new(store.clone(), targets.clone());
    let route = r.route(&rand_emb(1), &RouteCtx { task_type: Some("default".into()), estimated_tokens: 500 });
    assert!(!route.model.is_empty(), "route model must be populated");

    let mut buckets: HashSet<u8> = HashSet::new();
    for et in &[500u64, 3_000, 10_000, 40_000, 200_000] {
        let r2 = r.route(&rand_emb(1), &RouteCtx { task_type: None, estimated_tokens: *et });
        buckets.insert(r2.context_bucket);
    }
    assert!(buckets.len() >= 4, "expected 4+ unique context buckets, got {}", buckets.len());

    let samples: Vec<TrainSample> = (0..10u32).map(|i| TrainSample {
        embedding: rand_emb(i + 10), chosen_target: targets[i as usize % targets.len()].clone(), quality: 0.9, estimated_tokens: 500,
    }).collect();
    let applied = r.train(&samples).expect("train");
    assert!(applied >= 1, "training must apply at least one sample");

    let v_saved = r.save().await.expect("router save");
    let mut r2 = Router::new(store, targets);
    let v_loaded = r2.load().await.expect("router load").expect("weights present");
    assert_eq!(v_saved, v_loaded, "version mismatch");
    // post-load route should behave equivalently (byte-equal weights ⇒ same routing output)
    let q = rand_emb(5);
    let ctx_q = RouteCtx { task_type: Some("default".into()), estimated_tokens: 1_000 };
    let a = r.route(&q, &ctx_q);
    let b = r2.route(&q, &ctx_q);
    assert_eq!(a.model, b.model);
    assert!((a.temperature - b.temperature).abs() < 1e-5);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. InstantLoop — record → feedback 1.0 → adapter_norm > 0.
// ─────────────────────────────────────────────────────────────────────────────
#[tokio::test]
async fn section_06_instant_loop_adapter_grows_on_positive_feedback() {
    let (_g, path) = tmp_db("s6");
    let store = Arc::new(Store::open(&path).await.unwrap());
    let targets = vec!["a".to_string(), "b".to_string()];
    let router = Arc::new(Mutex::new(Router::new(store.clone(), targets.clone())));
    let mut il = InstantLoop::new(store.clone(), router, targets);
    assert_eq!(il.adapter_norm(), 0.0, "fresh adapter norm must be 0");

    for i in 0..5u32 {
        let rid = il.record_trajectory(Some("sess".into()), rand_emb(i), "a".into(), "resp".into()).await
            .expect("record trajectory");
        il.feedback(&rid, FeedbackPayload { quality: 1.0, signal: None }).await.expect("feedback");
    }
    assert!(il.adapter_norm() > 0.0, "adapter norm must grow after positive feedback");
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. ReasoningBank — insert + retrieve via FTS.
// ─────────────────────────────────────────────────────────────────────────────
#[tokio::test]
async fn section_07_reasoning_bank_fts_retrieve_and_top() {
    let (_g, path) = tmp_db("s7");
    let store = Arc::new(Store::open(&path).await.unwrap());
    for (id, strat, sr) in [
        ("r1", "high quality code pattern", 0.9f64),
        ("r2", "low quality pattern", 0.3),
        ("r3", "medium pattern code", 0.6),
    ] {
        store.insert_reasoning(&ReasoningRow {
            id: id.into(), pattern_id: None, strategy: strat.into(),
            success_rate: Some(sr), created_at: Some(rs_learn::store::now_ms()),
        }).await.expect("insert reasoning");
    }
    let bank = ReasoningBank::new(store);
    let hits = bank.retrieve_for_query("pattern code", 5).await.expect("fts");
    assert!(hits.len() >= 2, "FTS must find at least 2 matching rows, got {}", hits.len());
    let top = bank.top_strategies(10).await.expect("top");
    assert_eq!(top[0].id, "r1", "highest success_rate row must be first");
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Background loop — deterministic k-means, run_once writes patterns.
// ─────────────────────────────────────────────────────────────────────────────
#[tokio::test]
async fn section_09_background_kmeans_deterministic_and_runs() {
    let vecs: Vec<Vec<f32>> = (0..20u32).map(rand_emb).collect();
    let a = kmeans_plus_plus(&vecs, 3, 42);
    let b = kmeans_plus_plus(&vecs, 3, 42);
    let a_assign: Vec<usize> = a.iter().map(|x| x.cluster).collect();
    let b_assign: Vec<usize> = b.iter().map(|x| x.cluster).collect();
    assert_eq!(a_assign, b_assign, "k-means++ must be deterministic across seeds");
    let centroids = kmeans_centroids(&vecs, &a, 3);
    assert_eq!(centroids.len(), 3);
    assert!(centroids.iter().all(|c| c.len() == EMBED_DIM));

    let (_g, path) = tmp_db("s9");
    let store = Arc::new(Store::open(&path).await.unwrap());
    let targets = vec!["a".to_string(), "b".to_string()];
    let router = Arc::new(Mutex::new(Router::new(store.clone(), targets.clone())));
    let il = Arc::new(Mutex::new(InstantLoop::new(store.clone(), router.clone(), targets)));
    // seed 20 trajectories with positive feedback so clustering has data
    {
        let mut ilg = il.lock().await;
        for i in 0..20u32 {
            let rid = ilg.record_trajectory(Some("s".into()), rand_emb(i * 11), "a".into(), "r".into()).await.unwrap();
            ilg.feedback(&rid, FeedbackPayload { quality: 0.85, signal: None }).await.unwrap();
        }
    }
    let reasoning = Arc::new(ReasoningBank::new(store.clone()));
    let bg = BackgroundLoop::new(store.clone(), router, None, reasoning, Some(il));
    let stats = bg.run_once().await.expect("bg run_once");
    assert!(stats.clusters >= 1, "run_once must produce at least 1 cluster");
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. Deep loop — consolidate writes Fisher, record_loss boundary trigger.
// ─────────────────────────────────────────────────────────────────────────────
#[tokio::test]
async fn section_10_deep_loop_fisher_and_boundary() {
    std::env::remove_var("RS_LEARN_EWC_LAMBDA");
    let (_g, path) = tmp_db("s10");
    let store = Arc::new(Store::open(&path).await.unwrap());
    let mut dl = DeepLoop::new(store.clone());
    let params = vec![0.1f32, 0.2, 0.3];
    let grads = vec![1.0f32, 2.0, 3.0];
    dl.consolidate("p0", &params, &grads).await.expect("consolidate");
    let loaded = store.load_fisher_vec("p0").await.expect("fisher load");
    assert_eq!(loaded.len(), 3);
    assert!(loaded.iter().all(|v| *v > 0.0), "Fisher values must be positive");
    let penalty = dl.ewc_penalty("p0", &[0.2f32, 0.2, 0.3]);
    assert!(penalty > 0.0, "ewc penalty must be positive when params diverge");

    for _ in 0..5 { dl.record_loss(0.1).await.unwrap(); }
    let boundary = dl.record_loss(1.5).await.expect("record_loss");
    assert!(boundary, "large loss delta must trigger boundary");
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. Observability — start_debug_server port 0, GET /healthz /debug/names /debug/:name (404).
// ─────────────────────────────────────────────────────────────────────────────
#[tokio::test]
async fn section_11_observability_endpoints() {
    observability::register("integ_probe", || serde_json::json!({ "v": 1 }));
    let srv = observability::start_debug_server("127.0.0.1", 0).await.expect("start server");
    let base = format!("http://127.0.0.1:{}", srv.port);

    let health: serde_json::Value = reqwest::get(format!("{base}/healthz")).await.unwrap()
        .json().await.unwrap();
    assert_eq!(health, serde_json::json!({ "ok": true }));

    let names: Vec<String> = reqwest::get(format!("{base}/debug/names")).await.unwrap()
        .json().await.unwrap();
    assert!(names.iter().any(|n| n == "integ_probe"), "registered key must be listed");

    let nf = reqwest::get(format!("{base}/debug/no_such_xxx")).await.unwrap();
    assert_eq!(nf.status(), reqwest::StatusCode::NOT_FOUND);

    observability::unregister("integ_probe");
    srv.close().await;
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. Export — safetensors F32, patterns.jsonl parseable, HF dry_run enumerates.
// ─────────────────────────────────────────────────────────────────────────────
#[tokio::test]
async fn section_12_exports() {
    let (_g, path) = tmp_db("s12");
    let store = Store::open(&path).await.unwrap();
    let now = rs_learn::store::now_ms();
    // seed router weights row so safetensors export succeeds
    let blob: Vec<u8> = (0..16u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    store.save_router_weights(1, &blob, "fastgrnn", &serde_json::json!({})).await.unwrap();
    store.update_fisher("p0", 0.5).await.unwrap();
    store.upsert_pattern(&PatternRow {
        id: "p1".into(), centroid: Some(rand_emb(1)),
        count: Some(10), quality_sum: Some(8.0), created_at: Some(now),
    }).await.unwrap();
    store.insert_reasoning(&ReasoningRow {
        id: "r1".into(), pattern_id: Some("p1".into()),
        strategy: "s".into(), success_rate: Some(0.8), created_at: Some(now),
    }).await.unwrap();
    store.add_preference_pair(&PreferenceRow {
        id: "pp1".into(), query: Some("q".into()),
        chosen: "yes".into(), rejected: "no".into(), created_at: Some(now),
    }).await.unwrap();

    let tmp = tempfile::tempdir().unwrap();
    let outdir = tmp.path().join("exports");
    let st_path = export_safetensors(&store, &outdir).await.expect("safetensors");
    let raw = std::fs::read(&st_path).unwrap();
    let hlen = u64::from_le_bytes(raw[0..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&raw[8..8 + hlen]).unwrap();
    assert_eq!(header["router_weights"]["dtype"], "F32", "router_weights dtype must be F32");
    assert_eq!(header["__metadata__"]["source"], "rs-learn");

    let pat_path = export_patterns(&store, &outdir).await.expect("patterns export");
    for line in std::fs::read_to_string(&pat_path).unwrap().lines() {
        serde_json::from_str::<serde_json::Value>(line).expect("patterns.jsonl line parseable");
    }

    let pref_path = export_preferences(&store, &outdir).await.expect("prefs export");
    for line in std::fs::read_to_string(&pref_path).unwrap().lines() {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        assert!(v.get("chosen").is_some() && v.get("rejected").is_some());
    }

    let files = push_to_hugging_face("ns/name", &outdir, "", true).await.expect("hf dry_run");
    assert!(files.iter().any(|f| f == "router.safetensors"));
    assert!(files.iter().any(|f| f == "patterns.jsonl"));
    assert!(files.iter().any(|f| f == "preferences.jsonl"));
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. Orchestrator — query returns QueryResult with stage_breakdown, session reuse, feedback.
// The Rust Orchestrator requires a live ACP client at construction, so this is
// gated behind RS_LEARN_ACP_LIVE=1 just like section 14.
// ─────────────────────────────────────────────────────────────────────────────
#[tokio::test]
async fn section_13_orchestrator_full_pipeline_live_gated() {
    if std::env::var("RS_LEARN_ACP_LIVE").ok().as_deref() != Some("1") {
        eprintln!("skipping section_13: set RS_LEARN_ACP_LIVE=1 + RS_LEARN_ACP_COMMAND to run");
        return;
    }
    if std::env::var("RS_LEARN_ACP_COMMAND").is_err() {
        std::env::set_var("RS_LEARN_ACP_COMMAND", "opencode acp");
    }
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("orch.db");
    std::env::set_var("RS_LEARN_DB_PATH", db.to_str().unwrap());
    let orch = rs_learn::Orchestrator::new_default().await.expect("orchestrator init");
    let res1 = orch.query("Reply with {\"ok\":true}", rs_learn::orchestrator::QueryOpts {
        session_id: Some("warm".into()), include_code_search: false,
        max_retrieved: 0, task_type: Some("default".into()), estimated_tokens: None,
    }).await.expect("query 1");
    assert!(!res1.request_id.is_empty());
    assert_eq!(res1.session_id, "warm");
    for key in ["embed", "memory", "attention", "route", "acp", "learn"] {
        assert!(res1.stage_breakdown.contains_key(key), "stage_breakdown missing '{}'", key);
    }
    let res2 = orch.query("second turn", rs_learn::orchestrator::QueryOpts {
        session_id: Some("warm".into()), include_code_search: false,
        max_retrieved: 0, task_type: Some("default".into()), estimated_tokens: None,
    }).await.expect("query 2");
    assert_eq!(res1.session_id, res2.session_id, "session must be reused");
    orch.feedback(&res1.request_id, FeedbackPayload { quality: 1.0, signal: None }).await.expect("feedback");
}

// ─────────────────────────────────────────────────────────────────────────────
// 14. Live ACP via opencode — end-to-end round-trip.
// ─────────────────────────────────────────────────────────────────────────────
#[tokio::test]
async fn section_14_live_acp_round_trip() {
    if std::env::var("RS_LEARN_ACP_LIVE").ok().as_deref() != Some("1") {
        eprintln!("skipping section_14: set RS_LEARN_ACP_LIVE=1 + RS_LEARN_ACP_COMMAND to run");
        return;
    }
    if std::env::var("RS_LEARN_ACP_COMMAND").is_err() {
        std::env::set_var("RS_LEARN_ACP_COMMAND", "opencode acp");
    }
    let client = rs_learn::AcpClient::from_env().expect("acp from_env");
    let t0 = std::time::Instant::now();
    let v = client.generate(
        "You are a JSON echo.",
        "Reply with {\"live\":true,\"n\":7}",
        120_000,
    ).await.expect("acp generate");
    let ms = t0.elapsed().as_millis();
    assert!(v.is_object() || v.is_array(), "ACP must return parseable JSON");
    eprintln!("live ACP round-trip ok in {}ms: {}", ms, v);
    client.close().await;
}

// ─────────────────────────────────────────────────────────────────────────────
// Baseline smoke — ACP missing env yields process error (non-live).
// ─────────────────────────────────────────────────────────────────────────────
#[tokio::test]
async fn acp_missing_env_yields_process_error() {
    std::env::remove_var("RS_LEARN_ACP_COMMAND");
    match rs_learn::AcpClient::from_env() {
        Err(rs_learn::LlmError::Process(_)) => {}
        other => panic!("expected Process error, got {:?}", other.as_ref().err()),
    }
}

#[tokio::test]
async fn router_ctx_bucket_trains_from_estimated_tokens() {
    use rs_learn::router::{RouteCtx, Router, TrainSample};
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();
    drop(tmp);
    let store = std::sync::Arc::new(rs_learn::Store::open(&path).await.unwrap());
    let targets = vec!["a".to_string(), "b".to_string()];
    let mut r = Router::new(store, targets);
    let emb = rand_emb(7);
    let small = vec![TrainSample { embedding: emb.clone(), chosen_target: "a".into(), quality: 0.95, estimated_tokens: 500 }; 120];
    assert_eq!(r.train(&small).unwrap(), 120);
    r.save().await.unwrap();
    let route_small = r.route(&emb, &RouteCtx { task_type: None, estimated_tokens: 500 });
    let route_huge = r.route(&emb, &RouteCtx { task_type: None, estimated_tokens: 200_000 });
    assert_eq!(route_small.algo, "fastgrnn");
    let low = route_small.context_bucket;
    let large = vec![TrainSample { embedding: emb.clone(), chosen_target: "a".into(), quality: 0.95, estimated_tokens: 200_000 }; 120];
    assert_eq!(r.train(&large).unwrap(), 120);
    let route_huge2 = r.route(&emb, &RouteCtx { task_type: None, estimated_tokens: 200_000 });
    assert!(route_huge2.context_bucket >= route_huge.context_bucket,
        "training on 200k-token samples must at least not shift ctx bucket below initial; small-only bucket={} after-large bucket={}", low, route_huge2.context_bucket);
}

#[tokio::test]
async fn instant_adapter_bounded_under_runaway_feedback() {
    use rs_learn::learn::instant::{FeedbackPayload, InstantLoop, MAX_ADAPTER_NORM};
    use rs_learn::router::Router;
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();
    drop(tmp);
    let store = std::sync::Arc::new(rs_learn::Store::open(&path).await.unwrap());
    let targets = vec!["a".to_string(), "b".to_string()];
    let router = std::sync::Arc::new(tokio::sync::Mutex::new(Router::new(store.clone(), targets.clone())));
    let mut il = InstantLoop::new(store, router, targets);
    let emb = rand_emb(99);
    for _ in 0..500 {
        let rid = il.record_trajectory(None, emb.clone(), "a".into(), "r".into()).await.unwrap();
        il.feedback(&rid, FeedbackPayload { quality: 1.0, signal: None }).await.unwrap();
    }
    let norm = il.adapter_norm();
    assert!(norm <= MAX_ADAPTER_NORM + 1e-3, "adapter norm {} exceeded cap {}", norm, MAX_ADAPTER_NORM);
    assert!(norm > 0.0);
}

#[tokio::test]
async fn router_learns_held_out_routing_from_trajectories() {
    let (_g, path) = tmp_db("held_out");
    let store = Arc::new(Store::open(&path).await.unwrap());
    let targets = vec!["alpha".to_string(), "beta".to_string()];
    let mut router = Router::new(store.clone(), targets.clone());

    let anchor_a: Vec<f32> = (0..EMBED_DIM).map(|i| ((i as f32) * 0.017).sin()).collect();
    let anchor_b: Vec<f32> = (0..EMBED_DIM).map(|i| ((i as f32) * 0.023).cos()).collect();
    let jitter = |base: &[f32], seed: u32| -> Vec<f32> {
        let noise = rand_emb(seed);
        base.iter().zip(noise.iter()).map(|(x, n)| x + 0.05 * n).collect()
    };

    let mut samples = Vec::with_capacity(200);
    for i in 0..100u32 {
        samples.push(TrainSample {
            embedding: jitter(&anchor_a, 100 + i),
            chosen_target: "alpha".into(), quality: 0.9, estimated_tokens: 500,
        });
        samples.push(TrainSample {
            embedding: jitter(&anchor_b, 500 + i),
            chosen_target: "beta".into(), quality: 0.9, estimated_tokens: 500,
        });
    }
    let applied = router.train(&samples).unwrap();
    assert_eq!(applied, 200);

    let held_out_a = jitter(&anchor_a, 9001);
    let held_out_b = jitter(&anchor_b, 9002);
    let ctx = RouteCtx { task_type: Some("default".into()), estimated_tokens: 500 };
    let pick_a = router.route(&held_out_a, &ctx);
    let pick_b = router.route(&held_out_b, &ctx);
    assert_eq!(pick_a.model, "alpha", "held-out A region must route to alpha, got {}", pick_a.model);
    assert_eq!(pick_b.model, "beta", "held-out B region must route to beta, got {}", pick_b.model);

    let cross_sim = cos_sim(&anchor_a, &anchor_b);
    assert!(cross_sim.abs() < 0.9, "anchors must be distinct, cos={}", cross_sim);
}
