use rs_learn::graph::recipes;
use rs_learn::graph::search::{Reranker, SearchConfig};
use rs_learn::store::types::{EdgeRow, EpisodeRow, NodeRow};
use rs_learn::store::{now_ms, Store};
use std::sync::Arc;
use tempfile::tempdir;

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
