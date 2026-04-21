use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EpisodeRow {
    pub id: String,
    pub content: String,
    pub source: Option<String>,
    pub created_at: Option<i64>,
    pub valid_at: Option<i64>,
    pub invalid_at: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NodeRow {
    pub id: String,
    pub name: String,
    pub r#type: Option<String>,
    pub summary: Option<String>,
    pub embedding: Option<Vec<f32>>,
    pub level: Option<i64>,
    pub created_at: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EdgeRow {
    pub id: String,
    pub src: String,
    pub dst: String,
    pub relation: Option<String>,
    pub fact: Option<String>,
    pub embedding: Option<Vec<f32>>,
    pub weight: Option<f64>,
    pub created_at: Option<i64>,
    pub valid_at: Option<i64>,
    pub invalid_at: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrajectoryRow {
    pub id: String,
    pub session_id: Option<String>,
    pub query: Option<String>,
    pub query_embedding: Option<Vec<f32>>,
    pub retrieved_ids: Option<Vec<String>>,
    pub router_decision: Option<String>,
    pub response: Option<String>,
    pub activations: Option<Vec<u8>>,
    pub quality: Option<f64>,
    pub latency_ms: Option<i64>,
    pub created_at: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PatternRow {
    pub id: String,
    pub centroid: Option<Vec<f32>>,
    pub count: Option<i64>,
    pub quality_sum: Option<f64>,
    pub created_at: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReasoningRow {
    pub id: String,
    pub pattern_id: Option<String>,
    pub strategy: String,
    pub success_rate: Option<f64>,
    pub created_at: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PreferenceRow {
    pub id: String,
    pub query: Option<String>,
    pub chosen: String,
    pub rejected: String,
    pub created_at: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionRow {
    pub id: String,
    pub created_at: Option<i64>,
    pub meta: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterWeightsRow {
    pub version: i64,
    pub blob: Vec<u8>,
    pub algo: Option<String>,
    pub created_at: i64,
    pub meta: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeEmbeddingRow {
    pub embedding: Option<Vec<f32>>,
    pub created_at: i64,
}
