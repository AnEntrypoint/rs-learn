pub mod snippets;
pub mod extract_edges;
pub mod extract_nodes;
pub mod dedupe_nodes;
pub mod dedupe_edges;
pub mod summarize_nodes;
pub mod summarize_sagas;
pub mod eval;

use serde_json::Value;

pub fn to_prompt_json(v: &Value) -> String {
    serde_json::to_string_pretty(v).unwrap_or_else(|_| "null".into())
}

pub struct Prompt {
    pub system: String,
    pub user: String,
    pub schema: &'static str,
}
