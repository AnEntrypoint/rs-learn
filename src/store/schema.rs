pub use super::triggers::TRIGGERS;

pub const EMBED_DIM: u32 = 768;
pub const SCHEMA_VERSION: u32 = 2;

pub const PRAGMAS: &[&str] = &[
    "PRAGMA journal_mode = WAL",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA temp_store = MEMORY",
    "PRAGMA mmap_size = 268435456",
    "PRAGMA cache_size = -65536",
    "PRAGMA busy_timeout = 5000",
    "PRAGMA foreign_keys = ON",
    "PRAGMA wal_autocheckpoint = 1000",
];

pub const TABLES: &[&str] = &[
    "CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at INTEGER NOT NULL
    )",
    "CREATE TABLE IF NOT EXISTS episodes (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        source TEXT,
        group_id TEXT DEFAULT 'default',
        ref_time INTEGER,
        created_at INTEGER NOT NULL,
        valid_at INTEGER,
        invalid_at INTEGER
    )",
    "CREATE INDEX IF NOT EXISTS episodes_valid ON episodes(valid_at)",
    "CREATE INDEX IF NOT EXISTS episodes_group ON episodes(group_id)",
    "CREATE TABLE IF NOT EXISTS nodes (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT,
        summary TEXT DEFAULT '',
        attributes TEXT DEFAULT '{}',
        embedding F32_BLOB(768),
        level INTEGER DEFAULT 0,
        group_id TEXT DEFAULT 'default',
        created_at INTEGER NOT NULL
    )",
    "CREATE INDEX IF NOT EXISTS nodes_type ON nodes(type)",
    "CREATE INDEX IF NOT EXISTS nodes_level ON nodes(level)",
    "CREATE INDEX IF NOT EXISTS nodes_group ON nodes(group_id)",
    "CREATE INDEX IF NOT EXISTS nodes_vec ON nodes(libsql_vector_idx(embedding))",
    "CREATE TABLE IF NOT EXISTS edges (
        id TEXT PRIMARY KEY,
        src TEXT NOT NULL,
        dst TEXT NOT NULL,
        relation TEXT,
        fact TEXT DEFAULT '',
        embedding F32_BLOB(768),
        weight REAL DEFAULT 1.0,
        group_id TEXT DEFAULT 'default',
        episode_ids TEXT DEFAULT '[]',
        created_at INTEGER NOT NULL,
        valid_at INTEGER,
        invalid_at INTEGER,
        expired_at INTEGER
    )",
    "CREATE INDEX IF NOT EXISTS edges_src ON edges(src)",
    "CREATE INDEX IF NOT EXISTS edges_dst ON edges(dst)",
    "CREATE INDEX IF NOT EXISTS edges_relation ON edges(relation)",
    "CREATE INDEX IF NOT EXISTS edges_group ON edges(group_id)",
    "CREATE INDEX IF NOT EXISTS edges_valid ON edges(valid_at)",
    "CREATE INDEX IF NOT EXISTS edges_expired ON edges(expired_at)",
    "CREATE INDEX IF NOT EXISTS edges_vec ON edges(libsql_vector_idx(embedding))",
    "CREATE TABLE IF NOT EXISTS trajectories (
        id TEXT PRIMARY KEY,
        session_id TEXT,
        query TEXT,
        query_embedding F32_BLOB(768),
        retrieved_ids TEXT DEFAULT '[]',
        router_decision TEXT,
        response TEXT,
        activations BLOB,
        quality REAL DEFAULT 0,
        latency_ms INTEGER,
        created_at INTEGER NOT NULL
    )",
    "CREATE INDEX IF NOT EXISTS trajectories_session ON trajectories(session_id)",
    "CREATE INDEX IF NOT EXISTS trajectories_created ON trajectories(created_at)",
    "CREATE INDEX IF NOT EXISTS trajectories_quality ON trajectories(quality)",
    "CREATE INDEX IF NOT EXISTS trajectories_vec ON trajectories(libsql_vector_idx(query_embedding))",
    "CREATE TABLE IF NOT EXISTS patterns (
        id TEXT PRIMARY KEY,
        centroid F32_BLOB(768),
        count INTEGER DEFAULT 0,
        quality_sum REAL DEFAULT 0,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
    )",
    "CREATE INDEX IF NOT EXISTS patterns_vec ON patterns(libsql_vector_idx(centroid))",
    "CREATE TABLE IF NOT EXISTS reasoning_bank (
        id TEXT PRIMARY KEY,
        pattern_id TEXT,
        strategy TEXT NOT NULL,
        success_rate REAL DEFAULT 0,
        created_at INTEGER NOT NULL
    )",
    "CREATE INDEX IF NOT EXISTS reasoning_pattern ON reasoning_bank(pattern_id)",
    "CREATE TABLE IF NOT EXISTS router_weights (
        version INTEGER PRIMARY KEY,
        blob BLOB NOT NULL,
        algo TEXT,
        created_at INTEGER NOT NULL,
        meta TEXT DEFAULT '{}'
    )",
    "CREATE TABLE IF NOT EXISTS ewc_fisher (
        param_id TEXT PRIMARY KEY,
        value REAL NOT NULL,
        updated_at INTEGER NOT NULL
    )",
    "CREATE TABLE IF NOT EXISTS preference_pairs (
        id TEXT PRIMARY KEY,
        query TEXT,
        chosen TEXT NOT NULL,
        rejected TEXT NOT NULL,
        created_at INTEGER NOT NULL
    )",
    "CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        created_at INTEGER NOT NULL,
        meta TEXT DEFAULT '{}'
    )",
    "CREATE TABLE IF NOT EXISTS communities (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        summary TEXT DEFAULT '',
        embedding F32_BLOB(768),
        level INTEGER DEFAULT 0,
        group_id TEXT DEFAULT 'default',
        created_at INTEGER NOT NULL
    )",
    "CREATE INDEX IF NOT EXISTS communities_group ON communities(group_id)",
    "CREATE INDEX IF NOT EXISTS communities_vec ON communities(libsql_vector_idx(embedding))",
    "CREATE TABLE IF NOT EXISTS community_members (
        community_id TEXT NOT NULL,
        node_id TEXT NOT NULL,
        PRIMARY KEY (community_id, node_id)
    )",
    "CREATE INDEX IF NOT EXISTS community_members_node ON community_members(node_id)",
    "CREATE TABLE IF NOT EXISTS sagas (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        summary TEXT DEFAULT '',
        group_id TEXT DEFAULT 'default',
        created_at INTEGER NOT NULL
    )",
    "CREATE TABLE IF NOT EXISTS saga_episodes (
        saga_id TEXT NOT NULL,
        episode_id TEXT NOT NULL,
        seq INTEGER NOT NULL,
        PRIMARY KEY (saga_id, episode_id)
    )",
    "CREATE INDEX IF NOT EXISTS saga_episodes_seq ON saga_episodes(saga_id, seq)",
    "CREATE VIRTUAL TABLE IF NOT EXISTS communities_fts USING fts5(id UNINDEXED, name, summary, tokenize='porter unicode61')",
    "CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(id UNINDEXED, name, summary, tokenize='porter unicode61')",
    "CREATE VIRTUAL TABLE IF NOT EXISTS edges_fts USING fts5(id UNINDEXED, fact, tokenize='porter unicode61')",
    "CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(id UNINDEXED, content, tokenize='porter unicode61')",
    "CREATE VIRTUAL TABLE IF NOT EXISTS reasoning_fts USING fts5(id UNINDEXED, strategy, tokenize='porter unicode61')",
];

pub const COUNT_TABLES: &[&str] = &[
    "episodes","nodes","edges","trajectories","patterns","reasoning_bank",
    "router_weights","ewc_fisher","preference_pairs","sessions",
    "communities","community_members","sagas","saga_episodes",
];

pub const VEC_INDEX: &[(&str, &str, &str)] = &[
    ("nodes", "embedding", "nodes_vec"),
    ("edges", "embedding", "edges_vec"),
    ("trajectories", "query_embedding", "trajectories_vec"),
    ("patterns", "centroid", "patterns_vec"),
    ("communities", "embedding", "communities_vec"),
];

pub const FTS_MAP: &[(&str, &str, &str, &str)] = &[
    ("nodes", "nodes_fts", "nodes", "id"),
    ("edges", "edges_fts", "edges", "id"),
    ("episodes", "episodes_fts", "episodes", "id"),
    ("reasoning_bank", "reasoning_fts", "reasoning_bank", "id"),
    ("communities", "communities_fts", "communities", "id"),
];

pub const ALTERS_V2: &[&str] = &[
    "ALTER TABLE episodes ADD COLUMN group_id TEXT DEFAULT 'default'",
    "ALTER TABLE episodes ADD COLUMN ref_time INTEGER",
    "ALTER TABLE nodes ADD COLUMN attributes TEXT DEFAULT '{}'",
    "ALTER TABLE nodes ADD COLUMN group_id TEXT DEFAULT 'default'",
    "ALTER TABLE edges ADD COLUMN group_id TEXT DEFAULT 'default'",
    "ALTER TABLE edges ADD COLUMN episode_ids TEXT DEFAULT '[]'",
    "ALTER TABLE edges ADD COLUMN expired_at INTEGER",
];

pub fn vec_lit(arr: Option<&[f32]>) -> String {
    match arr {
        None => "NULL".to_string(),
        Some(a) => {
            let parts: Vec<String> = a.iter().map(|v| format!("{}", v)).collect();
            format!("vector32('[{}]')", parts.join(","))
        }
    }
}

pub fn fts_query(q: &str) -> String {
    q.replace('"', "\"\"")
        .split_whitespace()
        .filter(|s| !s.is_empty())
        .map(|t| format!("\"{}\"*", t))
        .collect::<Vec<_>>()
        .join(" OR ")
}
