use crate::memory::SearchHit as MemoryHit;
use crate::router::Route;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct Session {
    pub id: String,
    pub created_at: i64,
    pub turns: u64,
    pub last_embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Default)]
pub struct QueryOpts {
    pub session_id: Option<String>,
    pub include_code_search: bool,
    pub max_retrieved: usize,
    pub task_type: Option<String>,
    pub estimated_tokens: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryResult {
    pub text: Value,
    pub request_id: String,
    pub session_id: String,
    pub routing: RouteSnapshot,
    pub retrieved: Vec<MemoryHit>,
    pub confidence: f32,
    pub latency_ms: u64,
    pub stage_breakdown: HashMap<String, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteSnapshot {
    pub model: String,
    pub context_bucket: u8,
    pub temperature: f32,
    pub top_p: f32,
    pub confidence: f32,
    pub algo: String,
}

impl From<Route> for RouteSnapshot {
    fn from(r: Route) -> Self {
        Self {
            model: r.model,
            context_bucket: r.context_bucket,
            temperature: r.temperature,
            top_p: r.top_p,
            confidence: r.confidence,
            algo: r.algo.to_string(),
        }
    }
}
