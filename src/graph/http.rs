use crate::embeddings::Embedder;
use crate::store::Store;
use anyhow::Result;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;
use tower_http::cors::CorsLayer;

use super::ingest::Ingestor;
use super::llm::LlmJson;
use super::search::{SearchConfig, SearchFilters, Searcher};
use super::time::parse_iso_ms;

const MAX_BODY_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone)]
pub struct HttpState {
    pub store: Arc<Store>,
    pub embedder: Arc<Embedder>,
    pub ingestor: Arc<Ingestor>,
    pub searcher: Arc<Searcher>,
}

impl HttpState {
    pub fn new(store: Arc<Store>, embedder: Arc<Embedder>, llm: Arc<LlmJson>) -> Self {
        let ingestor = Ingestor::new(store.clone(), embedder.clone(), llm.clone());
        let searcher = Arc::new(Searcher::with_llm(store.clone(), embedder.clone(), llm.clone()));
        Self { store, embedder, ingestor, searcher }
    }

    pub fn router(self) -> Router {
        Router::new()
            .route("/healthcheck", get(healthcheck))
            .route("/messages", post(post_messages))
            .route("/entity-node", post(post_entity_node))
            .route("/entity-edge/:uuid", get(get_entity_edge).delete(delete_entity_edge))
            .route("/episode/:uuid", delete(delete_episode))
            .route("/group/:gid", delete(delete_group))
            .route("/episodes/:gid", get(get_episodes))
            .route("/clear", post(post_clear))
            .route("/search", post(post_search))
            .route("/get-memory", post(post_get_memory))
            .route("/build-communities", post(post_build_communities))
            .route("/triplet", post(post_triplet))
            .layer(CorsLayer::permissive())
            .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
            .with_state(self)
    }
}

#[derive(Debug)]
pub struct Problem {
    pub status: StatusCode,
    pub title: &'static str,
    pub detail: String,
}

impl IntoResponse for Problem {
    fn into_response(self) -> axum::response::Response {
        let body = json!({
            "type": "about:blank",
            "title": self.title,
            "status": self.status.as_u16(),
            "detail": self.detail,
        });
        let mut r = (self.status, Json(body)).into_response();
        r.headers_mut().insert(
            axum::http::header::CONTENT_TYPE,
            axum::http::HeaderValue::from_static("application/problem+json"),
        );
        r
    }
}

fn bad(detail: impl Into<String>) -> Problem {
    Problem { status: StatusCode::BAD_REQUEST, title: "Bad Request", detail: detail.into() }
}
fn ise(detail: impl Into<String>) -> Problem {
    Problem { status: StatusCode::INTERNAL_SERVER_ERROR, title: "Internal Server Error", detail: detail.into() }
}

async fn healthcheck() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

#[derive(Deserialize)]
struct MessagesBody {
    content: String,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    reference_time: Option<String>,
    #[serde(default)]
    group_id: Option<String>,
    #[serde(default)]
    entity_types: Option<String>,
    #[serde(default)]
    edge_types: Option<Value>,
}

async fn post_messages(State(s): State<HttpState>, Json(body): Json<MessagesBody>) -> Result<Json<Value>, Problem> {
    let source = body.source.unwrap_or_else(|| "message".into());
    if let Some(g) = &body.group_id { super::validation::validate_group_id(g).map_err(|e| bad(e.to_string()))?; }
    super::validation::validate_content(&body.content).map_err(|e| bad(e.to_string()))?;
    let res = s.ingestor.add_episode_with(
        &body.content, &source, body.reference_time.as_deref(), body.group_id.as_deref(),
        body.entity_types.as_deref(), body.edge_types.as_ref(),
    ).await.map_err(|e| ise(e.to_string()))?;
    Ok(Json(json!({
        "episode_id": res.episode_id,
        "nodes": res.node_count,
        "edges": res.edge_count,
        "expired_edge_ids": res.expired_edge_ids,
    })))
}

#[derive(Deserialize)]
struct TripletBody {
    src_id: String,
    src_name: String,
    dst_id: String,
    dst_name: String,
    relation: String,
    fact: String,
    #[serde(default)]
    group_id: Option<String>,
}

async fn post_triplet(State(s): State<HttpState>, Json(body): Json<TripletBody>) -> Result<Json<Value>, Problem> {
    use super::entities::Entity;
    if let Some(g) = &body.group_id { super::validation::validate_group_id(g).map_err(|e| bad(e.to_string()))?; }
    let src_emb = s.embedder.embed(&body.src_name).ok();
    let dst_emb = s.embedder.embed(&body.dst_name).ok();
    let src = Entity { id: body.src_id, name: body.src_name, entity_type: None, embedding: src_emb, group_id: body.group_id.clone() };
    let dst = Entity { id: body.dst_id, name: body.dst_name, entity_type: None, embedding: dst_emb, group_id: body.group_id.clone() };
    let eid = s.ingestor.add_triplet(src, dst, &body.relation, &body.fact, body.group_id.as_deref()).await.map_err(|e| ise(e.to_string()))?;
    Ok(Json(json!({ "edge_id": eid })))
}

#[derive(Deserialize)]
struct EntityNodeBody {
    name: String,
    #[serde(default)]
    r#type: Option<String>,
    #[serde(default)]
    group_id: Option<String>,
}

async fn post_entity_node(State(s): State<HttpState>, Json(body): Json<EntityNodeBody>) -> Result<Json<Value>, Problem> {
    use crate::store::types::NodeRow;
    use uuid::Uuid;
    if let Some(g) = &body.group_id { super::validation::validate_group_id(g).map_err(|e| bad(e.to_string()))?; }
    let id = Uuid::new_v4().to_string();
    let emb = s.embedder.embed(&body.name).ok();
    let row = NodeRow {
        id: id.clone(),
        name: body.name,
        r#type: body.r#type,
        summary: Some(String::new()),
        embedding: emb,
        level: Some(0),
        group_id: body.group_id,
        created_at: None,
    };
    s.store.insert_node(&row).await.map_err(|e| ise(e.to_string()))?;
    Ok(Json(json!({ "id": id })))
}

async fn get_entity_edge(State(s): State<HttpState>, Path(uuid): Path<String>) -> Result<Json<Value>, Problem> {
    let mut rows = s.store.conn.query(
        "SELECT id, src, dst, relation, fact, weight, created_at, valid_at, invalid_at, expired_at FROM edges WHERE id = ?1",
        libsql::params![uuid.clone()],
    ).await.map_err(|e| ise(e.to_string()))?;
    let Some(row) = rows.next().await.map_err(|e| ise(e.to_string()))? else {
        return Err(Problem { status: StatusCode::NOT_FOUND, title: "Not Found", detail: "edge".into() });
    };
    Ok(Json(json!({
        "id": row.get::<String>(0).unwrap_or_default(),
        "src": row.get::<String>(1).unwrap_or_default(),
        "dst": row.get::<String>(2).unwrap_or_default(),
        "relation": row.get::<String>(3).ok(),
        "fact": row.get::<String>(4).ok(),
        "weight": row.get::<f64>(5).ok(),
        "created_at": row.get::<i64>(6).ok(),
        "valid_at": row.get::<i64>(7).ok(),
        "invalid_at": row.get::<i64>(8).ok(),
        "expired_at": row.get::<i64>(9).ok(),
    })))
}

async fn delete_entity_edge(State(s): State<HttpState>, Path(uuid): Path<String>) -> Result<Json<Value>, Problem> {
    s.store.conn.execute("DELETE FROM edges WHERE id = ?1", libsql::params![uuid])
        .await.map_err(|e| ise(e.to_string()))?;
    Ok(Json(json!({ "status": "ok" })))
}

async fn delete_episode(State(s): State<HttpState>, Path(uuid): Path<String>) -> Result<Json<Value>, Problem> {
    s.store.conn.execute("DELETE FROM episodes WHERE id = ?1", libsql::params![uuid])
        .await.map_err(|e| ise(e.to_string()))?;
    Ok(Json(json!({ "status": "ok" })))
}

async fn delete_group(State(s): State<HttpState>, Path(gid): Path<String>) -> Result<Json<Value>, Problem> {
    for tbl in ["edges", "nodes", "episodes"] {
        s.store.conn.execute(&format!("DELETE FROM {tbl} WHERE group_id = ?1"), libsql::params![gid.clone()])
            .await.map_err(|e| ise(e.to_string()))?;
    }
    Ok(Json(json!({ "status": "ok" })))
}

async fn get_episodes(State(s): State<HttpState>, Path(gid): Path<String>) -> Result<Json<Value>, Problem> {
    let mut rows = s.store.conn.query(
        "SELECT id, content, source, created_at FROM episodes WHERE group_id = ?1 ORDER BY created_at DESC LIMIT 100",
        libsql::params![gid],
    ).await.map_err(|e| ise(e.to_string()))?;
    let mut out = Vec::new();
    while let Some(row) = rows.next().await.map_err(|e| ise(e.to_string()))? {
        out.push(json!({
            "id": row.get::<String>(0).unwrap_or_default(),
            "content": row.get::<String>(1).unwrap_or_default(),
            "source": row.get::<String>(2).ok(),
            "created_at": row.get::<i64>(3).ok(),
        }));
    }
    Ok(Json(json!({ "episodes": out })))
}

#[derive(Deserialize, Default)]
struct ClearBody {
    #[serde(default)]
    group_ids: Option<Vec<String>>,
}

async fn post_clear(State(s): State<HttpState>, body: Option<Json<ClearBody>>) -> Result<Json<Value>, Problem> {
    let gids = body.and_then(|b| b.0.group_ids);
    if let Some(g) = &gids { for x in g { super::validation::validate_group_id(x).map_err(|e| bad(e.to_string()))?; } }
    s.ingestor.clear_graph(gids.as_deref()).await.map_err(|e| ise(e.to_string()))?;
    Ok(Json(json!({ "status": "ok" })))
}

#[derive(Deserialize, Default)]
struct SearchBody {
    query: String,
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    as_of: Option<String>,
    #[serde(default)]
    nodes: Option<ScopeBody>,
    #[serde(default)]
    edges: Option<ScopeBody>,
    #[serde(default)]
    episodes: Option<ScopeBody>,
    #[serde(default)]
    communities: Option<ScopeBody>,
    #[serde(default)]
    center_node_ids: Vec<String>,
}

#[derive(Deserialize, Clone)]
struct ScopeBody {
    #[serde(default)]
    reranker: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    use_vector: Option<bool>,
    #[serde(default)]
    use_fts: Option<bool>,
    #[serde(default)]
    mmr_lambda: Option<f32>,
    #[serde(default)]
    filters: Option<SearchFilters>,
}

fn parse_reranker(s: &str) -> Result<super::search::Reranker, String> {
    use super::search::Reranker::*;
    Ok(match s {
        "rrf" => Rrf, "mmr" => Mmr,
        "node_distance" => NodeDistance,
        "episode_mentions" => EpisodeMentions,
        "cross_encoder" => CrossEncoder,
        other => return Err(format!("unknown reranker '{other}'")),
    })
}

fn scope_body_to_cfg(sb: &ScopeBody, default_limit: usize) -> Result<SearchConfig, String> {
    let mut c = SearchConfig { limit: sb.limit.unwrap_or(default_limit), ..Default::default() };
    if let Some(r) = &sb.reranker { c.reranker = parse_reranker(r)?; }
    if let Some(v) = sb.use_vector { c.use_vector = v; }
    if let Some(v) = sb.use_fts { c.use_fts = v; }
    if let Some(v) = sb.mmr_lambda { c.mmr_lambda = v; }
    if let Some(f) = sb.filters.clone() { c.filters = f; }
    Ok(c)
}

async fn post_search(State(s): State<HttpState>, Json(body): Json<SearchBody>) -> Result<Json<Value>, Problem> {
    if body.query.trim().is_empty() { return Err(bad("query required")); }
    let cfg = SearchConfig {
        limit: body.limit.unwrap_or(10),
        as_of: body.as_of.as_deref().and_then(parse_iso_ms),
        ..Default::default()
    };
    let scope = body.scope.as_deref().unwrap_or("nodes");
    if scope == "all" {
        let limit = body.limit.unwrap_or(10);
        let mut all = super::search::SearchAllConfig::all_defaults(limit);
        all.as_of = cfg.as_of;
        all.center_node_ids = body.center_node_ids.clone();
        let apply = |slot: &mut Option<SearchConfig>, sb: Option<&ScopeBody>| -> Result<(), Problem> {
            if let Some(sb) = sb {
                let built = scope_body_to_cfg(sb, limit).map_err(|e| bad(e.to_string()))?;
                *slot = Some(SearchConfig { as_of: cfg.as_of, ..built });
            }
            Ok(())
        };
        apply(&mut all.nodes, body.nodes.as_ref())?;
        apply(&mut all.edges, body.edges.as_ref())?;
        apply(&mut all.episodes, body.episodes.as_ref())?;
        apply(&mut all.communities, body.communities.as_ref())?;
        let r = s.searcher.search_all_cfg(&body.query, &all).await.map_err(|e| ise(e.to_string()))?;
        return Ok(Json(json!({
            "nodes": r.nodes,
            "edges": r.edges,
            "episodes": r.episodes,
            "communities": r.communities,
        })));
    }
    let hits = match scope {
        "nodes" => s.searcher.search_nodes(&body.query, &cfg).await,
        "facts" | "edges" => s.searcher.search_facts(&body.query, &cfg).await,
        "episodes" => s.searcher.search_episodes(&body.query, &cfg).await,
        "communities" => s.searcher.search_communities(&body.query, &cfg).await,
        other => return Err(bad(format!("unknown scope '{other}'"))),
    }.map_err(|e| ise(e.to_string()))?;
    Ok(Json(json!({ "hits": hits })))
}

async fn post_get_memory(State(s): State<HttpState>, Json(body): Json<SearchBody>) -> Result<Json<Value>, Problem> {
    let cfg = SearchConfig { limit: body.limit.unwrap_or(10), ..Default::default() };
    let nodes = s.searcher.search_nodes(&body.query, &cfg).await.map_err(|e| ise(e.to_string()))?;
    let facts = s.searcher.search_facts(&body.query, &cfg).await.map_err(|e| ise(e.to_string()))?;
    Ok(Json(json!({ "nodes": nodes, "facts": facts })))
}

#[derive(Deserialize, Default)]
struct BuildCommunitiesBody {
    #[serde(default)]
    force: bool,
}

async fn post_build_communities(State(s): State<HttpState>, body: Option<Json<BuildCommunitiesBody>>) -> Result<Json<Value>, Problem> {
    use super::communities::CommunityOps;
    let force = body.map(|b| b.0.force).unwrap_or(false);
    let llm = s.ingestor.llm.clone();
    let ops = CommunityOps::new(s.store.clone(), s.embedder.clone(), llm);
    let r = if force { ops.build_communities().await } else { ops.build_communities_if_dirty().await }
        .map_err(|e| ise(e.to_string()))?;
    Ok(Json(json!({ "communities": r.community_count, "members": r.member_count })))
}
