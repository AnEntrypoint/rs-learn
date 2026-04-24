use super::*;
use crate::store::{Store, ReasoningRow};

#[tokio::test]
async fn retrieve_and_top() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("rb.db");
    let store = Arc::new(Store::open(path.to_str().unwrap()).await?);

    store.insert_reasoning(&ReasoningRow {
        id: "s1".into(), pattern_id: Some("p1".into()),
        strategy: "decompose complex queries into substeps".into(),
        success_rate: Some(0.9), created_at: None,
    }).await?;
    store.insert_reasoning(&ReasoningRow {
        id: "s2".into(), pattern_id: None,
        strategy: "cache frequent lookups".into(),
        success_rate: Some(0.5), created_at: None,
    }).await?;

    let bank = ReasoningBank::new(store.clone());
    let hits = bank.retrieve_for_query("decompose", 5).await?;
    assert!(hits.iter().any(|s| s.id == "s1"), "fts should return s1");

    let top = bank.top_strategies(10).await?;
    assert_eq!(top.first().map(|s| s.id.as_str()), Some("s1"));
    assert!(top[0].success_rate >= top[1].success_rate);
    Ok(())
}

#[tokio::test]
async fn hybrid_retrieval_bridges_via_pattern_centroid() -> Result<()> {
    use crate::embeddings::Embedder;
    use crate::store::types::PatternRow;
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("rbh.db");
    let store = Arc::new(Store::open(path.to_str().unwrap()).await?);
    let embedder = Arc::new(Embedder::new());

    let emb = embedder.embed("user wants to reorganize legacy code")?;
    store.upsert_pattern(&PatternRow {
        id: "p-refactor".into(), centroid: Some(emb.clone()),
        count: Some(5), quality_sum: Some(4.0), created_at: None,
    }).await?;
    store.insert_reasoning(&crate::store::ReasoningRow {
        id: "s-refactor".into(), pattern_id: Some("p-refactor".into()),
        strategy: "break the monolith into modules with clear seams".into(),
        success_rate: Some(0.85), created_at: None,
    }).await?;

    let bank = ReasoningBank::with_embedder(store, embedder);
    let hits = bank.retrieve_for_query("reorganize legacy code", 3).await?;
    assert!(hits.iter().any(|s| s.id == "s-refactor"));
    Ok(())
}

#[tokio::test]
async fn record_outcome_updates_success_rate() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("ro.db");
    let store = Arc::new(Store::open(path.to_str().unwrap()).await?);
    store.insert_reasoning(&ReasoningRow {
        id: "s-update".into(), pattern_id: None,
        strategy: "test".into(), success_rate: Some(0.5), created_at: None,
    }).await?;
    let bank = ReasoningBank::new(store.clone());
    bank.record_outcome(&["s-update".to_string()], 1.0).await?;
    let rate = store.get_reasoning_success_rate("s-update").await?.unwrap();
    assert!(rate > 0.5 && rate < 1.0, "EMA should move toward 1.0 but not reach it in one step: got {}", rate);
    Ok(())
}
