use crate::graph::types::EdgeRow;
use std::collections::HashMap;

pub const NS_EDGES: &str = "rs-learn/graph/edges";
pub const NS_EDGES_BY_SRC: &str = "rs-learn/graph/edges_by_src";
pub const NS_EDGES_BY_DST: &str = "rs-learn/graph/edges_by_dst";

pub trait KvBackend {
    fn get(&self, namespace: &str, key: &str) -> Option<Vec<u8>>;
    fn put(&mut self, namespace: &str, key: &str, val: &[u8]) -> Result<(), String>;
    fn list_prefix(&self, namespace: &str, prefix: &str) -> Vec<String>;
}

#[derive(Debug, Default)]
pub struct MemKv {
    inner: HashMap<(String, String), Vec<u8>>,
}

impl KvBackend for MemKv {
    fn get(&self, namespace: &str, key: &str) -> Option<Vec<u8>> {
        self.inner.get(&(namespace.to_string(), key.to_string())).cloned()
    }
    fn put(&mut self, namespace: &str, key: &str, val: &[u8]) -> Result<(), String> {
        self.inner.insert((namespace.to_string(), key.to_string()), val.to_vec());
        Ok(())
    }
    fn list_prefix(&self, namespace: &str, prefix: &str) -> Vec<String> {
        let mut out: Vec<String> = self.inner.keys()
            .filter(|(ns, k)| ns == namespace && k.starts_with(prefix))
            .map(|(_, k)| k.clone())
            .collect();
        out.sort();
        out
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum InvalidationOutcome {
    Inserted { edge_id: String },
    Invalidated { invalidated_edge_id: String, new_edge_id: String },
}

pub struct TemporalGraph<B: KvBackend> {
    pub kv: B,
}

impl<B: KvBackend> TemporalGraph<B> {
    pub fn new(kv: B) -> Self {
        Self { kv }
    }

    pub fn insert_edge(&mut self, edge: EdgeRow) -> Result<(), String> {
        let id = edge.id.clone();
        if id.is_empty() { return Err("edge.id required".into()); }
        if id.contains(',') { return Err("edge.id must not contain a comma (used as index delimiter)".into()); }
        if edge.src.is_empty() || edge.dst.is_empty() { return Err("edge.src and edge.dst required".into()); }
        if edge.valid_at.is_none() { return Err("edge.valid_at required (bi-temporal)".into()); }
        if edge.created_at.is_none() { return Err("edge.created_at required (system-time)".into()); }
        if let (Some(va), Some(iv)) = (edge.valid_at, edge.invalid_at) {
            if iv < va { return Err("edge.invalid_at must be >= edge.valid_at".into()); }
        }
        let blob = serde_json::to_vec(&edge).map_err(|e| format!("serialize: {}", e))?;
        self.kv.put(NS_EDGES, &id, &blob)?;
        self.append_index(NS_EDGES_BY_SRC, &edge.src, &id)?;
        self.append_index(NS_EDGES_BY_DST, &edge.dst, &id)?;
        Ok(())
    }

    fn append_index(&mut self, ns: &str, key: &str, edge_id: &str) -> Result<(), String> {
        const MAX_RETRIES: u32 = 8;
        for attempt in 0..MAX_RETRIES {
            let before = self.kv.get(ns, key).unwrap_or_default();
            let mut s = String::from_utf8(before.clone()).unwrap_or_default();
            if s.split(',').any(|e| e == edge_id) {
                return Ok(());
            }
            if !s.is_empty() { s.push(','); }
            s.push_str(edge_id);
            self.kv.put(ns, key, s.as_bytes())?;
            let after = self.kv.get(ns, key).unwrap_or_default();
            if after == s.as_bytes() {
                return Ok(());
            }
            let after_s = String::from_utf8(after).unwrap_or_default();
            if after_s.split(',').any(|e| e == edge_id) {
                return Ok(());
            }
            if attempt + 1 < MAX_RETRIES {
                let jitter = (edge_id.len() as u64 + attempt as u64 * 7) % 5;
                let backoff_ms = (attempt as u64 + 1) * 2 + jitter;
                std::thread::sleep(std::time::Duration::from_millis(backoff_ms));
            }
        }
        Err(format!(
            "append_index: failed to converge for key {} after {} retries",
            key, MAX_RETRIES
        ))
    }

    pub fn get_edge(&self, edge_id: &str) -> Option<EdgeRow> {
        let blob = self.kv.get(NS_EDGES, edge_id)?;
        serde_json::from_slice(&blob).ok()
    }

    pub fn edges_by_src(&self, src: &str) -> Vec<EdgeRow> {
        self.lookup_index(NS_EDGES_BY_SRC, src)
    }

    pub fn edges_by_dst(&self, dst: &str) -> Vec<EdgeRow> {
        self.lookup_index(NS_EDGES_BY_DST, dst)
    }

    fn lookup_index(&self, ns: &str, key: &str) -> Vec<EdgeRow> {
        let raw = match self.kv.get(ns, key) { Some(b) => b, None => return Vec::new() };
        let s = match String::from_utf8(raw) { Ok(s) => s, Err(_) => return Vec::new() };
        s.split(',').filter(|e| !e.is_empty())
            .filter_map(|id| self.get_edge(id))
            .collect()
    }

    pub fn query_at(&self, src: &str, t: i64) -> Vec<EdgeRow> {
        self.edges_by_src(src).into_iter().filter(|e| e.active_at(t)).collect()
    }

    pub fn query_at_bounded(&self, src: &str, t: i64, limit: usize) -> Vec<EdgeRow> {
        let mut edges: Vec<EdgeRow> = self.edges_by_src(src).into_iter().filter(|e| e.active_at(t)).collect();
        if limit == 0 || edges.len() <= limit {
            return edges;
        }
        edges.sort_by(|a, b| b.created_at.unwrap_or(0).cmp(&a.created_at.unwrap_or(0)));
        edges.truncate(limit);
        edges
    }

    pub fn invalidate_edge(&mut self, edge_id: &str, invalid_at: i64, expired_at: i64) -> Result<(), String> {
        let mut edge = self.get_edge(edge_id).ok_or_else(|| format!("edge {} not found", edge_id))?;
        if let Some(existing_iv) = edge.invalid_at {
            if existing_iv < invalid_at {
                return Ok(());
            }
        }
        if let Some(va) = edge.valid_at {
            if invalid_at < va {
                return Err("invalid_at < edge.valid_at; would create negative interval".into());
            }
        }
        edge.invalid_at = Some(invalid_at);
        edge.expired_at = Some(expired_at);
        let blob = serde_json::to_vec(&edge).map_err(|e| format!("serialize: {}", e))?;
        self.kv.put(NS_EDGES, edge_id, &blob)?;
        Ok(())
    }

    pub fn insert_with_contradiction(
        &mut self,
        new_edge: EdgeRow,
        contradicted_edge_ids: &[String],
        now_ms: i64,
    ) -> Result<InvalidationOutcome, String> {
        let new_valid_at = new_edge.valid_at.ok_or("new edge.valid_at required")?;
        let new_id = new_edge.id.clone();
        for old_id in contradicted_edge_ids {
            self.invalidate_edge(old_id, new_valid_at, now_ms)?;
        }
        self.insert_edge(new_edge)?;
        if let Some(first) = contradicted_edge_ids.first() {
            Ok(InvalidationOutcome::Invalidated {
                invalidated_edge_id: first.clone(),
                new_edge_id: new_id,
            })
        } else {
            Ok(InvalidationOutcome::Inserted { edge_id: new_id })
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    fn mk_edge(id: &str, src: &str, dst: &str, valid_at: i64) -> EdgeRow {
        EdgeRow {
            id: id.into(),
            src: src.into(),
            dst: dst.into(),
            relation: Some("works_at".into()),
            fact: Some(format!("{} works at {}", src, dst)),
            embedding: None,
            weight: Some(1.0),
            group_id: Some("default".into()),
            created_at: Some(valid_at),
            expired_at: None,
            valid_at: Some(valid_at),
            invalid_at: None,
        }
    }

    #[test]
    fn insert_and_get() {
        let mut g = TemporalGraph::new(MemKv::default());
        let e = mk_edge("e1", "alice", "acme", 1000);
        g.insert_edge(e.clone()).unwrap();
        assert_eq!(g.get_edge("e1"), Some(e));
    }

    #[test]
    fn query_at_bounded_returns_k_most_recent_active() {
        let mut g = TemporalGraph::new(MemKv::default());
        for i in 0..5 {
            g.insert_edge(mk_edge(&format!("e{}", i), "hub", &format!("d{}", i), 1000 + i * 100)).unwrap();
        }
        let full = g.query_at("hub", 9999);
        assert_eq!(full.len(), 5);
        let bounded = g.query_at_bounded("hub", 9999, 3);
        assert_eq!(bounded.len(), 3);
        let ids: Vec<&str> = bounded.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"e4") && ids.contains(&"e3") && ids.contains(&"e2"));
        assert!(!ids.contains(&"e0") && !ids.contains(&"e1"));
        assert_eq!(g.query_at_bounded("hub", 9999, 0).len(), 5);
        assert_eq!(g.query_at_bounded("hub", 9999, 10).len(), 5);
    }

    #[test]
    fn insert_rejects_missing_required_fields() {
        let mut g = TemporalGraph::new(MemKv::default());
        let mut e = mk_edge("e1", "alice", "acme", 1000);
        e.valid_at = None;
        assert!(g.insert_edge(e).is_err());

        let mut e2 = mk_edge("e2", "", "acme", 1000);
        e2.src = String::new();
        assert!(g.insert_edge(e2).is_err());
    }

    #[test]
    fn insert_rejects_inverted_interval() {
        let mut g = TemporalGraph::new(MemKv::default());
        let mut e = mk_edge("e1", "alice", "acme", 2000);
        e.invalid_at = Some(1000);
        assert!(g.insert_edge(e).is_err());
    }

    #[test]
    fn edges_by_src_indexes_correctly() {
        let mut g = TemporalGraph::new(MemKv::default());
        g.insert_edge(mk_edge("e1", "alice", "acme", 1000)).unwrap();
        g.insert_edge(mk_edge("e2", "alice", "globex", 2000)).unwrap();
        g.insert_edge(mk_edge("e3", "bob", "acme", 1500)).unwrap();
        let alice = g.edges_by_src("alice");
        assert_eq!(alice.len(), 2);
        assert!(alice.iter().any(|e| e.id == "e1"));
        assert!(alice.iter().any(|e| e.id == "e2"));
    }

    #[test]
    fn point_in_time_query_filters_by_interval() {
        let mut g = TemporalGraph::new(MemKv::default());
        let mut e1 = mk_edge("e1", "alice", "acme", 1000);
        e1.invalid_at = Some(2000);
        g.insert_edge(e1).unwrap();
        let e2 = mk_edge("e2", "alice", "globex", 2000);
        g.insert_edge(e2).unwrap();

        assert_eq!(g.query_at("alice", 500).len(), 0);
        let at_1500: Vec<_> = g.query_at("alice", 1500);
        assert_eq!(at_1500.len(), 1);
        assert_eq!(at_1500[0].id, "e1");
        let at_2500: Vec<_> = g.query_at("alice", 2500);
        assert_eq!(at_2500.len(), 1);
        assert_eq!(at_2500[0].id, "e2");
    }

    #[test]
    fn invalidate_sets_invalid_and_expired() {
        let mut g = TemporalGraph::new(MemKv::default());
        g.insert_edge(mk_edge("e1", "alice", "acme", 1000)).unwrap();
        g.invalidate_edge("e1", 2000, 2500).unwrap();
        let e = g.get_edge("e1").unwrap();
        assert_eq!(e.invalid_at, Some(2000));
        assert_eq!(e.expired_at, Some(2500));
        assert!(e.is_invalidated());
        assert!(!e.active_at(2500));
        assert!(e.active_at(1500));
    }

    #[test]
    fn invalidate_idempotent_if_already_earlier() {
        let mut g = TemporalGraph::new(MemKv::default());
        let mut e = mk_edge("e1", "alice", "acme", 1000);
        e.invalid_at = Some(1500);
        g.insert_edge(e).unwrap();
        g.invalidate_edge("e1", 2000, 2500).unwrap();
        assert_eq!(g.get_edge("e1").unwrap().invalid_at, Some(1500));
    }

    #[test]
    fn invalidate_rejects_negative_interval() {
        let mut g = TemporalGraph::new(MemKv::default());
        g.insert_edge(mk_edge("e1", "alice", "acme", 2000)).unwrap();
        assert!(g.invalidate_edge("e1", 1000, 1500).is_err());
    }

    #[test]
    fn graphiti_style_contradiction_preserves_history() {
        let mut g = TemporalGraph::new(MemKv::default());
        let acme = mk_edge("e_acme", "alice", "acme", 1000);
        g.insert_edge(acme).unwrap();

        let globex = mk_edge("e_globex", "alice", "globex", 2000);
        let outcome = g.insert_with_contradiction(globex, &["e_acme".into()], 2050).unwrap();
        match outcome {
            InvalidationOutcome::Invalidated { invalidated_edge_id, new_edge_id } => {
                assert_eq!(invalidated_edge_id, "e_acme");
                assert_eq!(new_edge_id, "e_globex");
            }
            other => panic!("expected Invalidated, got {:?}", other),
        }

        let old = g.get_edge("e_acme").unwrap();
        assert_eq!(old.invalid_at, Some(2000), "old edge t_invalid must equal new edge t_valid (Graphiti rule)");
        assert!(old.expired_at.is_some(), "old edge gets system-time stamp");
        assert!(!old.active_at(2500), "old edge no longer active after contradiction");
        assert!(old.active_at(1500), "old edge still active in its historic window");

        let new_edge = g.get_edge("e_globex").unwrap();
        assert!(!new_edge.is_invalidated());
        assert!(new_edge.active_at(2500));

        let active_now: Vec<_> = g.query_at("alice", 2500);
        assert_eq!(active_now.len(), 1);
        assert_eq!(active_now[0].id, "e_globex");

        let historical_at_1500: Vec<_> = g.query_at("alice", 1500);
        assert_eq!(historical_at_1500.len(), 1);
        assert_eq!(historical_at_1500[0].id, "e_acme");
    }

    #[test]
    fn insert_without_contradiction_lists_no_invalidation() {
        let mut g = TemporalGraph::new(MemKv::default());
        let outcome = g.insert_with_contradiction(
            mk_edge("e1", "alice", "acme", 1000),
            &[],
            1000,
        ).unwrap();
        assert_eq!(outcome, InvalidationOutcome::Inserted { edge_id: "e1".into() });
    }

    #[test]
    fn insert_rejects_comma_in_id() {
        let mut g = TemporalGraph::new(MemKv::default());
        let e = mk_edge("e1,evil", "alice", "acme", 1000);
        assert!(g.insert_edge(e).is_err());
    }

    #[test]
    fn invalidate_equal_timestamp_applies_update() {
        let mut g = TemporalGraph::new(MemKv::default());
        let mut e = mk_edge("e1", "alice", "acme", 1000);
        e.invalid_at = Some(2000);
        g.insert_edge(e).unwrap();
        g.invalidate_edge("e1", 2000, 2500).unwrap();
        assert_eq!(g.get_edge("e1").unwrap().expired_at, Some(2500));
    }

    #[test]
    fn append_index_survives_many_sequential_inserts_same_key() {
        let mut g = TemporalGraph::new(MemKv::default());
        for i in 0..50 {
            g.insert_edge(mk_edge(&format!("e{}", i), "hub", &format!("d{}", i), 1000 + i)).unwrap();
        }
        let edges = g.edges_by_src("hub");
        assert_eq!(edges.len(), 50, "all 50 edges must be reachable via the src index, none dropped");
    }
}
