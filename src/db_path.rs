use std::path::{Path, PathBuf};

/// Resolve the database path for rs-learn.
///
/// Persistent DBs live in gm-data/ (tracked in git), ephemeral state in .gm/ (excluded).
/// Legacy .gm/rs-learn.db is auto-migrated to gm-data/rs-learn.db on first access.
pub fn resolve_db_path() -> String {
    resolve_db_path_for(None, None)
}

pub fn resolve_db_path_for(root: Option<&Path>, discipline: Option<&str>) -> String {
    if let Ok(v) = std::env::var("RS_LEARN_DB_PATH") {
        return v;
    }
    let base: PathBuf = match root {
        Some(r) => r.to_path_buf(),
        None => std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    };

    // Persistent data goes in gm-data/ (git-tracked)
    let data_dir = base.join("gm-data");
    let target_dir = match discipline {
        Some(name) if !name.is_empty() => data_dir.join("disciplines").join(name),
        _ => data_dir.clone(),
    };
    let target = target_dir.join("rs-learn.db");

    // Legacy paths to migrate from
    let legacy_gm = base.join(".gm").join("rs-learn.db");
    let legacy_root = base.join("rs-learn.db");

    let _ = std::fs::create_dir_all(&target_dir);

    // Migrate legacy DBs to gm-data/
    if discipline.is_none() {
        if !target.exists() {
            if legacy_gm.exists() {
                let _ = std::fs::rename(&legacy_gm, &target);
            } else if legacy_root.exists() {
                let _ = std::fs::rename(&legacy_root, &target);
            }
        }
    }

    if target.exists() || std::fs::create_dir_all(&target_dir).is_ok() {
        target.to_string_lossy().to_string()
    } else {
        // Fallback to .gm/ if gm-data/ is not writable
        let fallback = base.join(".gm").join("rs-learn.db");
        let _ = std::fs::create_dir_all(base.join(".gm"));
        fallback.to_string_lossy().to_string()
    }
}

/// Resolve the code-search index path.
/// Persistent index lives in gm-data/code-search/ (git-tracked).
pub fn resolve_code_search_path(root: Option<&Path>) -> PathBuf {
    let base: PathBuf = match root {
        Some(r) => r.to_path_buf(),
        None => std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    };
    let data_dir = base.join("gm-data");
    let cs_dir = data_dir.join("code-search");
    let _ = std::fs::create_dir_all(&cs_dir);

    // Migrate legacy .gm/code-search/ to gm-data/code-search/
    let legacy = base.join(".gm").join("code-search");
    if legacy.exists() && !cs_dir.exists() {
        let _ = std::fs::rename(&legacy, &cs_dir);
    }

    cs_dir
}
