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
use super::search::{SearchConfig, Searcher};

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
        let searcher = Arc::new(Searcher::new(store.clone(), embedder.clone()));
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
}

async fn post_messages(State(s): State<HttpState>, Json(body): Json<MessagesBody>) -> Result<Json<Value>, Problem> {
    let source = body.source.unwrap_or_else(|| "message".into());
    let res = s.ingestor.add_episode(&body.content, &source, body.reference_time.as_deref())
        .await.map_err(|e| ise(e.to_string()))?;
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
}

async fn post_triplet(State(s): State<HttpState>, Json(body): Json<TripletBody>) -> Result<Json<Value>, Problem> {
    use super::entities::Entity;
    let src_emb = s.embedder.embed(&body.src_name).ok();
    let dst_emb = s.embedder.embed(&body.dst_name).ok();
    let src = Entity { id: body.src_id, name: body.src_name, entity_type: None, embedding: src_emb };
    let dst = Entity { id: body.dst_id, name: body.dst_name, entity_type: None, embedding: dst_emb };
    let eid = s.ingestor.add_triplet(src, dst, &body.relation, &body.fact).await.map_err(|e| ise(e.to_string()))?;
    Ok(Json(json!({ "edge_id": eid })))
}

#[derive(Deserialize)]
struct EntityNodeBody {
    name: String,
    #[serde(default)]
    r#type: Option<String>,
}

async fn post_entity_node(State(s): State<HttpState>, Json(body): Json<EntityNodeBody>) -> Result<Json<Value>, Problem> {
    use crate::store::types::NodeRow;
    use uuid::Uuid;
    let id = Uuid::new_v4().to_string();
    let emb = s.embedder.embed(&body.name).ok();
    let row = NodeRow {
        id: id.clone(),
        name: body.name,
        r#type: body.r#type,
        summary: Some(String::new()),
        embedding: emb,
        level: Some(0),
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

async fn post_clear(State(s): State<HttpState>) -> Result<Json<Value>, Problem> {
    s.ingestor.clear_graph().await.map_err(|e| ise(e.to_string()))?;
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
}

async fn post_search(State(s): State<HttpState>, Json(body): Json<SearchBody>) -> Result<Json<Value>, Problem> {
    if body.query.trim().is_empty() { return Err(bad("query required")); }
    let cfg = SearchConfig {
        limit: body.limit.unwrap_or(10),
        as_of: body.as_of.as_deref().and_then(parse_iso_ms),
        ..Default::default()
    };
    let scope = body.scope.as_deref().unwrap_or("nodes");
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

async fn post_build_communities(State(s): State<HttpState>) -> Result<Json<Value>, Problem> {
    use super::communities::CommunityOps;
    let llm = s.ingestor.llm.clone();
    let ops = CommunityOps::new(s.store.clone(), s.embedder.clone(), llm);
    let r = ops.build_communities().await.map_err(|e| ise(e.to_string()))?;
    Ok(Json(json!({ "communities": r.community_count, "members": r.member_count })))
}

fn parse_iso_ms(s: &str) -> Option<i64> {
    let s = s.trim();
    if s.len() < 19 { return None; }
    let b = s.as_bytes();
    let year: i64 = std::str::from_utf8(&b[0..4]).ok()?.parse().ok()?;
    let month: i64 = std::str::from_utf8(&b[5..7]).ok()?.parse().ok()?;
    let day: i64 = std::str::from_utf8(&b[8..10]).ok()?.parse().ok()?;
    let hour: i64 = std::str::from_utf8(&b[11..13]).ok()?.parse().ok()?;
    let minute: i64 = std::str::from_utf8(&b[14..16]).ok()?.parse().ok()?;
    let second: i64 = std::str::from_utf8(&b[17..19]).ok()?.parse().ok()?;
    let days = days_from_civil(year, month, day);
    Some(days * 86_400_000 + hour * 3_600_000 + minute * 60_000 + second * 1_000)
}

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) as u64 / 5 + d as u64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe as i64 - 719468
}
