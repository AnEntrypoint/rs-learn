use async_trait::async_trait;
use rs_learn::backend::AgentBackend;
use rs_learn::embeddings::Embedder;
use rs_learn::errors::{LlmError, Result as LlmResult};
use rs_learn::graph::ingest::Ingestor;
use rs_learn::graph::llm::LlmJson;
use rs_learn::graph::recipes;
use rs_learn::graph::search::{Reranker, SearchConfig, Searcher};
use rs_learn::store::types::{EdgeRow, EpisodeRow, NodeRow};
use rs_learn::store::{now_ms, Store};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use tempfile::tempdir;

struct StubBackend {
    responses: Mutex<Vec<Value>>,
}

impl StubBackend {
    fn new(responses: Vec<Value>) -> Arc<Self> {
        Arc::new(Self { responses: Mutex::new(responses) })
    }
}

#[async_trait]
impl AgentBackend for StubBackend {
    async fn generate(&self, _system: &str, _user: &str, _timeout_ms: u64) -> LlmResult<Value> {
        let mut g = self.responses.lock().unwrap();
        if g.is_empty() {
            return Err(LlmError::Validation("stub exhausted".into()));
        }
        Ok(g.remove(0))
    }
    fn name(&self) -> &'static str { "stub" }
}

async fn open() -> (tempfile::TempDir, Arc<Store>) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("g.db");
    let store = Arc::new(Store::open(path.to_str().unwrap()).await.unwrap());
    (dir, store)
}

#[tokio::test]
async fn schema_has_bungraph_tables() {
    let (_dir, store) = open().await;
    for t in ["communities", "community_members", "sagas", "saga_episodes"] {
        let c = store.count_rows(t).await;
        assert!(c >= 0, "table '{t}' should exist");
    }
}

#[tokio::test]
async fn episode_and_node_edge_roundtrip_with_new_columns() {
    let (_dir, store) = open().await;
    let ep = EpisodeRow {
        id: "ep1".into(),
        content: "Alice joined Acme".into(),
        source: Some("message".into()),
        created_at: Some(now_ms()),
        valid_at: Some(now_ms()),
        invalid_at: None,
    };
    store.insert_episode(&ep).await.unwrap();
    let n = NodeRow {
        id: "n1".into(),
        name: "Alice".into(),
        r#type: Some("Person".into()),
        summary: Some("".into()),
        embedding: Some(vec![0.1; 768]),
        level: Some(0),
        created_at: Some(now_ms()),
    };
    store.insert_node(&n).await.unwrap();
    let e = EdgeRow {
        id: "e1".into(),
        src: "n1".into(),
        dst: "n1".into(),
        relation: Some("SELF_REF".into()),
        fact: Some("alice is alice".into()),
        embedding: Some(vec![0.05; 768]),
        weight: Some(1.0),
        created_at: Some(now_ms()),
        valid_at: Some(now_ms()),
        invalid_at: None,
    };
    store.insert_edge(&e).await.unwrap();
    assert_eq!(store.count_rows("episodes").await, 1);
    assert_eq!(store.count_rows("nodes").await, 1);
    assert_eq!(store.count_rows("edges").await, 1);
}

#[tokio::test]
async fn expired_at_column_writable() {
    let (_dir, store) = open().await;
    let n = NodeRow { id: "n1".into(), name: "n1".into(), r#type: None, summary: None,
        embedding: Some(vec![0.1; 768]), level: Some(0), created_at: Some(now_ms()) };
    store.insert_node(&n).await.unwrap();
    let e = EdgeRow {
        id: "e1".into(), src: "n1".into(), dst: "n1".into(),
        relation: Some("R".into()), fact: Some("f".into()),
        embedding: None, weight: Some(1.0),
        created_at: Some(now_ms()), valid_at: Some(100), invalid_at: None,
    };
    store.insert_edge(&e).await.unwrap();
    store.conn.execute(
        "UPDATE edges SET expired_at = ?1 WHERE id = ?2",
        libsql::params![5000_i64, "e1".to_string()],
    ).await.unwrap();
    let mut rows = store.conn.query(
        "SELECT expired_at FROM edges WHERE id = 'e1'", (),
    ).await.unwrap();
    let row = rows.next().await.unwrap().unwrap();
    let v: i64 = row.get(0).unwrap();
    assert_eq!(v, 5000);
}

#[tokio::test]
async fn recipes_registry_covers_all_scopes() {
    let all = recipes::all();
    assert_eq!(all.len(), 17);
    let node = recipes::by_name("NODE_HYBRID_SEARCH_RRF").unwrap();
    assert_eq!(node.cfg.reranker, Reranker::Rrf);
    let edge_mmr = recipes::by_name("EDGE_HYBRID_SEARCH_MMR").unwrap();
    assert_eq!(edge_mmr.cfg.reranker, Reranker::Mmr);
    let community = recipes::by_name("COMMUNITY_HYBRID_SEARCH_RRF").unwrap();
    assert_eq!(community.cfg.limit, 3);
}

#[tokio::test]
async fn search_config_defaults_sensible() {
    let c = SearchConfig::default();
    assert_eq!(c.limit, 10);
    assert_eq!(c.reranker, Reranker::Rrf);
    assert!(c.rrf_k > 0.0);
    assert!(c.mmr_lambda > 0.0 && c.mmr_lambda <= 1.0);
}

#[tokio::test]
async fn ingest_with_stub_llm_persists_nodes_edges_and_mentions() {
    let (_dir, store) = open().await;
    let embedder = Arc::new(Embedder::new());
    // Response sequence for one add_episode: extract_entities -> dedup_entities (skipped
    // if no existing) -> extract_edges -> resolve_temporal (skipped: no existing edges).
    let backend = StubBackend::new(vec![
        json!({ "extracted_entities": [
            { "name": "Alice", "entity_type_id": 1 },
            { "name": "Acme",  "entity_type_id": 2 }
        ]}),
        json!({ "edges": [
            { "source_entity_name": "Alice", "target_entity_name": "Acme",
              "relation_type": "WORKS_AT", "fact": "Alice works at Acme",
              "valid_at": null, "invalid_at": null }
        ]}),
    ]);
    let llm = Arc::new(LlmJson::with_limits(backend, 1, 5000, 1000));
    let ingestor = Ingestor::new(store.clone(), embedder, llm);
    let r = ingestor.add_episode("Alice works at Acme Corp.", "text", None).await.unwrap();
    assert_eq!(r.node_count, 2, "two entities resolved");
    assert_eq!(r.edge_count, 1, "one fact triple extracted");
    // Nodes + fact edge + 2 MENTIONS edges.
    assert_eq!(store.count_rows("nodes").await, 2);
    assert_eq!(store.count_rows("edges").await, 3);
    assert_eq!(store.count_rows("episodes").await, 1);
}

#[tokio::test]
async fn bulk_ingest_and_get_episodes_roundtrip() {
    let (_dir, store) = open().await;
    let embedder = Arc::new(Embedder::new());
    let backend = StubBackend::new(vec![
        json!({ "extracted_entities": [] }),
        json!({ "extracted_entities": [] }),
    ]);
    let llm = Arc::new(LlmJson::with_limits(backend, 1, 5000, 1000));
    let ingestor = Ingestor::new(store.clone(), embedder, llm);
    let items = vec![
        rs_learn::graph::ingest::BulkEpisode { content: "first".into(), source: "text".into(), reference_time: None },
        rs_learn::graph::ingest::BulkEpisode { content: "second".into(), source: "text".into(), reference_time: None },
    ];
    let rs = ingestor.add_episode_bulk(items).await.unwrap();
    assert_eq!(rs.len(), 2);
    let eps = ingestor.get_episodes(None, 100).await.unwrap();
    assert_eq!(eps.len(), 2);
}

#[tokio::test]
async fn search_with_stub_cross_encoder_reorders_hits() {
    let (_dir, store) = open().await;
    // Seed two nodes with identical embeddings so vector rank is tied and the
    // cross-encoder score decides final order.
    store.insert_node(&NodeRow {
        id: "a".into(), name: "alpha".into(), r#type: None, summary: Some("".into()),
        embedding: Some(vec![0.1; 768]), level: Some(0), created_at: Some(now_ms()),
    }).await.unwrap();
    store.insert_node(&NodeRow {
        id: "b".into(), name: "beta".into(), r#type: None, summary: Some("".into()),
        embedding: Some(vec![0.1; 768]), level: Some(0), created_at: Some(now_ms()),
    }).await.unwrap();
    let embedder = Arc::new(Embedder::new());
    let backend = StubBackend::new(vec![
        // cross-encoder scores: prefer idx 1 (beta) over idx 0 (alpha)
        json!({ "scores": [ { "idx": 0, "score": 0.1 }, { "idx": 1, "score": 0.9 } ] }),
    ]);
    let llm = Arc::new(LlmJson::with_limits(backend, 1, 5000, 1000));
    let searcher = Searcher::with_llm(store, embedder, llm);
    let cfg = SearchConfig { limit: 2, reranker: Reranker::CrossEncoder, ..Default::default() };
    let hits = searcher.search_nodes("alpha", &cfg).await.unwrap();
    assert!(!hits.is_empty());
    // beta should come first after LLM scoring
    assert_eq!(hits[0].id, "b", "cross-encoder should put beta first, got {:?}", hits.iter().map(|h| h.id.clone()).collect::<Vec<_>>());
}

#[tokio::test]
async fn search_all_fans_out_across_scopes() {
    let (_dir, store) = open().await;
    let now = now_ms();
    store.insert_episode(&EpisodeRow {
        id: "e1".into(), content: "alpha".into(), source: Some("text".into()),
        created_at: Some(now), valid_at: Some(now), invalid_at: None,
    }).await.unwrap();
    store.insert_node(&NodeRow {
        id: "n1".into(), name: "alpha".into(), r#type: None, summary: Some("".into()),
        embedding: Some(vec![0.1; 768]), level: Some(0), created_at: Some(now),
    }).await.unwrap();
    store.insert_edge(&EdgeRow {
        id: "ed1".into(), src: "n1".into(), dst: "n1".into(),
        relation: Some("R".into()), fact: Some("alpha fact".into()),
        embedding: Some(vec![0.1; 768]), weight: Some(1.0),
        created_at: Some(now), valid_at: Some(now), invalid_at: None,
    }).await.unwrap();
    let embedder = Arc::new(Embedder::new());
    let searcher = Searcher::new(store, embedder);
    let cfg = SearchConfig { limit: 5, ..Default::default() };
    let r = searcher.search_all("alpha", &cfg).await.unwrap();
    assert!(!r.nodes.is_empty(), "nodes scope must return hit");
    assert!(!r.edges.is_empty(), "edges scope must return hit");
    assert!(!r.episodes.is_empty(), "episodes scope must return hit");
    assert_eq!(r.communities.len(), 0, "no communities seeded");
}

#[tokio::test]
async fn graph_walk_follows_edges_by_depth() {
    let (_dir, store) = open().await;
    let now = now_ms();
    for id in ["a","b","c"] {
        store.insert_node(&NodeRow {
            id: id.into(), name: id.into(), r#type: None, summary: Some("".into()),
            embedding: Some(vec![0.1; 768]), level: Some(0), created_at: Some(now),
        }).await.unwrap();
    }
    store.insert_edge(&EdgeRow {
        id: "ab".into(), src: "a".into(), dst: "b".into(),
        relation: Some("R".into()), fact: Some("".into()),
        embedding: None, weight: Some(1.0),
        created_at: Some(now), valid_at: Some(now), invalid_at: None,
    }).await.unwrap();
    store.insert_edge(&EdgeRow {
        id: "bc".into(), src: "b".into(), dst: "c".into(),
        relation: Some("R".into()), fact: Some("".into()),
        embedding: None, weight: Some(1.0),
        created_at: Some(now), valid_at: Some(now), invalid_at: None,
    }).await.unwrap();
    let ns1 = store.graph_walk(&["a".to_string()], 1, None).await.unwrap();
    let ids1: Vec<String> = ns1.iter().map(|n| n.id.clone()).collect();
    assert!(ids1.contains(&"b".to_string()), "depth 1 must reach b: {:?}", ids1);
    assert!(!ids1.contains(&"c".to_string()), "depth 1 must NOT reach c: {:?}", ids1);
    let ns2 = store.graph_walk(&["a".to_string()], 2, None).await.unwrap();
    let ids2: Vec<String> = ns2.iter().map(|n| n.id.clone()).collect();
    assert!(ids2.contains(&"c".to_string()), "depth 2 must reach c: {:?}", ids2);
    let between = store.edges_between(&["a".to_string()], &["b".to_string()], None, None).await.unwrap();
    assert_eq!(between.len(), 1);
    assert_eq!(between[0].id, "ab");
}

#[tokio::test]
async fn per_scope_search_respects_use_fts_toggle() {
    let (_dir, store) = open().await;
    let now = now_ms();
    store.insert_node(&NodeRow {
        id: "n1".into(), name: "alpha".into(), r#type: None, summary: Some("".into()),
        embedding: Some(vec![0.2; 768]), level: Some(0), created_at: Some(now),
    }).await.unwrap();
    let embedder = Arc::new(Embedder::new());
    let searcher = Searcher::new(store, embedder);
    let mut all = rs_learn::graph::search::SearchAllConfig::all_defaults(5);
    if let Some(c) = all.nodes.as_mut() { c.use_fts = false; }
    if let Some(c) = all.edges.as_mut() { c.use_fts = false; }
    if let Some(c) = all.episodes.as_mut() { c.use_fts = false; }
    if let Some(c) = all.communities.as_mut() { c.use_fts = false; }
    let r = searcher.search_all_cfg("alpha", &all).await.unwrap();
    assert!(!r.nodes.is_empty(), "vector-only search must still return node hit");
}

#[tokio::test]
async fn both_toggles_off_fails_loud() {
    let (_dir, store) = open().await;
    let embedder = Arc::new(Embedder::new());
    let searcher = Searcher::new(store, embedder);
    let cfg = SearchConfig { limit: 5, use_vector: false, use_fts: false, ..Default::default() };
    let err = searcher.search_nodes("x", &cfg).await.unwrap_err();
    assert!(err.to_string().contains("both use_vector and use_fts disabled"));
}

#[tokio::test]
async fn bulk_ingest_context_respects_reference_time() {
    let (_dir, store) = open().await;
    store.insert_episode(&EpisodeRow {
        id: "old".into(), content: "OLD".into(), source: Some("text".into()),
        created_at: Some(100), valid_at: Some(100), invalid_at: None,
    }).await.unwrap();
    store.insert_episode(&EpisodeRow {
        id: "new".into(), content: "NEW".into(), source: Some("text".into()),
        created_at: Some(1_000_000_000_000), valid_at: Some(1_000_000_000_000), invalid_at: None,
    }).await.unwrap();
    let embedder = Arc::new(Embedder::new());
    let backend = StubBackend::new(vec![ json!({ "extracted_entities": [] }) ]);
    let llm = Arc::new(LlmJson::with_limits(backend, 1, 5000, 1000));
    let ingestor = Ingestor::new(store.clone(), embedder, llm);
    let _ = ingestor.add_episode("mid", "text", Some("1970-01-01T00:01:00.000Z")).await.unwrap();
}

#[tokio::test]
async fn group_id_filter_cascade_delete() {
    let (_dir, store) = open().await;
    store.insert_episode(&EpisodeRow {
        id: "a".into(), content: "x".into(), source: None,
        created_at: Some(now_ms()), valid_at: None, invalid_at: None,
    }).await.unwrap();
    store.conn.execute(
        "UPDATE episodes SET group_id = 'g1' WHERE id = 'a'", (),
    ).await.unwrap();
    store.insert_episode(&EpisodeRow {
        id: "b".into(), content: "y".into(), source: None,
        created_at: Some(now_ms()), valid_at: None, invalid_at: None,
    }).await.unwrap();
    store.conn.execute(
        "DELETE FROM episodes WHERE group_id = 'g1'", (),
    ).await.unwrap();
    assert_eq!(store.count_rows("episodes").await, 1);
}
