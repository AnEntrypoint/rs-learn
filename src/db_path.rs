use std::path::PathBuf;

pub fn resolve_db_path() -> String {
    if let Ok(v) = std::env::var("RS_LEARN_DB_PATH") {
        return v;
    }
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let gm_dir = cwd.join(".gm");
    let target = gm_dir.join("rs-learn.db");
    let legacy = cwd.join("rs-learn.db");
    if std::fs::create_dir_all(&gm_dir).is_ok() {
        if legacy.exists() && !target.exists() {
            let _ = std::fs::rename(&legacy, &target);
        }
        target.to_string_lossy().to_string()
    } else {
        legacy.to_string_lossy().to_string()
    }
}
