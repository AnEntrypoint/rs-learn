use crate::observability;
use serde_json::json;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct SearchHit {
    pub file: String,
    pub line_start: usize,
    pub line_end: usize,
    pub score: f64,
    pub snippet: String,
}

pub struct RsSearch {
    calls: Arc<AtomicU64>,
    last_hits: Arc<AtomicU64>,
}

impl RsSearch {
    pub fn new() -> Self {
        let calls = Arc::new(AtomicU64::new(0));
        let last_hits = Arc::new(AtomicU64::new(0));
        let c = calls.clone();
        let h = last_hits.clone();
        observability::register("rs-search", move || {
            json!({
                "calls": c.load(Ordering::Relaxed),
                "last_hits": h.load(Ordering::Relaxed),
            })
        });
        Self { calls, last_hits }
    }

    pub fn search(&self, query: &str, root: &Path) -> Vec<SearchHit> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let raw = rs_search::run_search(query, root);
        let hits: Vec<SearchHit> = raw
            .into_iter()
            .map(|r| SearchHit {
                file: r.chunk.file_path,
                line_start: r.chunk.line_start,
                line_end: r.chunk.line_end,
                score: r.score,
                snippet: r.chunk.content,
            })
            .collect();
        self.last_hits.store(hits.len() as u64, Ordering::Relaxed);
        hits
    }
}

impl Default for RsSearch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn search_finds_fixture_content() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        fs::write(
            root.join("alpha.rs"),
            "fn alpha_unique_marker() { println!(\"hello\"); }\n",
        )
        .expect("write alpha");
        fs::write(
            root.join("beta.rs"),
            "fn beta_other() { let x = 1; }\n",
        )
        .expect("write beta");

        let rs = RsSearch::new();
        let hits = rs.search("alpha_unique_marker", root);
        assert!(
            hits.iter().any(|h| h.file.contains("alpha.rs")),
            "expected alpha.rs in hits, got {:?}",
            hits.iter().map(|h| &h.file).collect::<Vec<_>>()
        );
    }
}
