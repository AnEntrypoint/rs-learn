use std::path::{Path, PathBuf};

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
    let gm_dir = base.join(".gm");
    let target_dir = match discipline {
        Some(name) if !name.is_empty() => gm_dir.join("disciplines").join(name),
        _ => gm_dir.clone(),
    };
    let target = target_dir.join("rs-learn.db");
    let legacy = base.join("rs-learn.db");
    if std::fs::create_dir_all(&target_dir).is_ok() {
        if discipline.is_none() && legacy.exists() && !target.exists() {
            let _ = std::fs::rename(&legacy, &target);
        }
        target.to_string_lossy().to_string()
    } else {
        legacy.to_string_lossy().to_string()
    }
}
