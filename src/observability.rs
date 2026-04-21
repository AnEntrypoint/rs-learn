use std::net::SocketAddr;
use std::sync::{Arc, OnceLock};

use arc_swap::ArcSwap;
use axum::extract::Path;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use dashmap::DashMap;
use serde_json::{json, Value};
use tokio::sync::oneshot;
use tower_http::cors::CorsLayer;

pub type Provider = Arc<dyn Fn() -> Value + Send + Sync>;
type Registry = DashMap<String, Provider>;

fn registry() -> &'static ArcSwap<Registry> {
    static R: OnceLock<ArcSwap<Registry>> = OnceLock::new();
    R.get_or_init(|| ArcSwap::from_pointee(DashMap::new()))
}

pub fn register<F>(key: &str, provider: F)
where
    F: Fn() -> Value + Send + Sync + 'static,
{
    registry().load().insert(key.to_string(), Arc::new(provider));
}

pub fn unregister(key: &str) {
    registry().load().remove(key);
}

pub fn names() -> Vec<String> {
    let r = registry().load();
    let mut v: Vec<String> = r.iter().map(|e| e.key().clone()).collect();
    v.sort();
    v
}

pub fn dump() -> Value {
    let r = registry().load();
    let mut map = serde_json::Map::new();
    for entry in r.iter() {
        let val = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (entry.value())()))
            .unwrap_or_else(|_| json!({ "error": "provider panicked" }));
        map.insert(entry.key().clone(), val);
    }
    Value::Object(map)
}

fn lookup(key: &str) -> Option<Value> {
    let r = registry().load();
    let entry = r.get(key)?;
    let provider = entry.value().clone();
    let val = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || provider()))
        .unwrap_or_else(|_| json!({ "error": "provider panicked" }));
    Some(val)
}

async fn healthz() -> Json<Value> {
    Json(json!({ "ok": true }))
}

async fn debug_all() -> Json<Value> {
    Json(dump())
}

async fn debug_names() -> Json<Vec<String>> {
    Json(names())
}

async fn debug_one(Path(name): Path<String>) -> Response {
    match lookup(&name) {
        Some(v) => Json(v).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "unknown subsystem", "available": names() })),
        )
            .into_response(),
    }
}

pub struct DebugServer {
    shutdown: Option<oneshot::Sender<()>>,
    pub port: u16,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl DebugServer {
    pub async fn close(mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(h) = self.handle.take() {
            let _ = h.await;
        }
    }
}

pub async fn start_debug_server(host: &str, port: u16) -> anyhow::Result<DebugServer> {
    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/debug", get(debug_all))
        .route("/debug/names", get(debug_names))
        .route("/debug/:name", get(debug_one))
        .layer(CorsLayer::permissive());

    let addr: SocketAddr = format!("{host}:{port}").parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let bound = listener.local_addr()?.port();

    let (tx, rx) = oneshot::channel::<()>();
    let handle = tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = rx.await;
            })
            .await;
    });

    Ok(DebugServer {
        shutdown: Some(tx),
        port: bound,
        handle: Some(handle),
    })
}
