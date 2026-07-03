use serde::{Deserialize, Serialize};

// No dispatch verb constructs, inserts, or queries EpisodeRow: there is no
// episode storage in TemporalGraph (no NS_EPISODES namespace, no
// insert_episode/get_episode), so this type is currently unreachable from
// any wasm-boundary call. Wiring it needs a storage layer added to
// TemporalGraph first, not just a dispatch handler.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct EpisodeRow {
    pub id: String,
    pub content: String,
    pub source: Option<String>,
    pub group_id: Option<String>,
    pub created_at: Option<i64>,
    pub valid_at: Option<i64>,
    pub invalid_at: Option<i64>,
}

// No dispatch verb constructs, inserts, or queries NodeRow: TemporalGraph
// defines NS_NODES but never reads or writes it (only NS_EDGES/_BY_SRC/_BY_DST
// are used by insert_edge/get_edge/lookup_index), so the node layer of the
// graph is currently unreachable from any wasm-boundary call. Wiring a
// node_insert/node_query verb needs real node storage methods added to
// TemporalGraph first (mirroring insert_edge/get_edge), not just a dispatch
// handler over EdgeRow's existing pattern.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct NodeRow {
    pub id: String,
    pub name: String,
    pub r#type: Option<String>,
    pub summary: Option<String>,
    pub embedding: Option<Vec<f32>>,
    pub level: Option<i64>,
    pub group_id: Option<String>,
    pub created_at: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct EdgeRow {
    pub id: String,
    pub src: String,
    pub dst: String,
    pub relation: Option<String>,
    pub fact: Option<String>,
    pub embedding: Option<Vec<f32>>,
    pub weight: Option<f64>,
    pub group_id: Option<String>,
    pub created_at: Option<i64>,
    pub expired_at: Option<i64>,
    pub valid_at: Option<i64>,
    pub invalid_at: Option<i64>,
}

impl EdgeRow {
    pub fn active_at(&self, t: i64) -> bool {
        let va = self.valid_at.unwrap_or(i64::MIN);
        if va > t { return false; }
        match self.invalid_at {
            Some(iv) => t < iv,
            None => true,
        }
    }

    pub fn is_invalidated(&self) -> bool {
        self.invalid_at.is_some()
    }
}
