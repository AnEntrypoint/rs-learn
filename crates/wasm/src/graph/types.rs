use serde::{Deserialize, Serialize};

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
