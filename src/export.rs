use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use anyhow::{anyhow, Result};
use safetensors::tensor::{Dtype, TensorView};
use safetensors::serialize_to_file;
use serde_json::json;
use tokio::fs;
use tokio::io::AsyncWriteExt;
use crate::observability;
use crate::store::Store;

fn register_once() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        observability::register("export", || json!({
            "module": "export", "formats": ["safetensors", "jsonl"], "hf": true
        }));
    });
}

pub async fn export_safetensors(store: &Store, dir: &Path) -> Result<PathBuf> {
    register_once();
    fs::create_dir_all(dir).await?;
    let row = store.load_latest_router_weights().await?
        .ok_or_else(|| anyhow!("export: no router weights"))?;
    if row.blob.len() % 4 != 0 { return Err(anyhow!("blob not f32-aligned")); }
    let router = TensorView::new(Dtype::F32, vec![row.blob.len() / 4], &row.blob)
        .map_err(|e| anyhow!("router view: {e}"))?;
    let fisher = store.load_fisher().await?;
    let mut keys: Vec<String> = fisher.keys().cloned().collect();
    keys.sort();
    let fvec: Vec<f32> = keys.iter().map(|k| fisher[k] as f32).collect();
    let fbytes: Vec<u8> = fvec.iter().flat_map(|f| f.to_le_bytes()).collect();
    let fview = TensorView::new(Dtype::F32, vec![fvec.len()], &fbytes)
        .map_err(|e| anyhow!("fisher view: {e}"))?;
    let abytes: Vec<u8> = Vec::new();
    let aview = TensorView::new(Dtype::F32, vec![0], &abytes)
        .map_err(|e| anyhow!("adapter view: {e}"))?;
    let out = dir.join("router.safetensors");
    let meta: std::collections::HashMap<String, String> = [
        ("format", "safetensors"), ("source", "rs-learn"),
    ].iter().map(|(k, v)| (k.to_string(), v.to_string()))
      .chain([("version".to_string(), row.version.to_string()),
              ("fisher_keys".to_string(), keys.join(","))]).collect();
    serialize_to_file(
        vec![("router_weights", router), ("adapter", aview), ("fisher", fview)],
        &Some(meta), &out
    ).map_err(|e| anyhow!("safetensors: {e}"))?;
    Ok(out)
}

pub async fn export_patterns(store: &Store, dir: &Path) -> Result<PathBuf> {
    register_once();
    fs::create_dir_all(dir).await?;
    let sql = "SELECT p.id, p.count, p.quality_sum, r.strategy, r.success_rate \
               FROM patterns p LEFT JOIN reasoning_bank r ON r.pattern_id = p.id \
               ORDER BY p.updated_at DESC";
    let mut rows = store.conn.query(sql, ()).await?;
    let out = dir.join("patterns.jsonl");
    let mut f = fs::File::create(&out).await?;
    while let Some(row) = rows.next().await? {
        let id: String = row.get(0)?;
        let count: i64 = row.get::<i64>(1).unwrap_or(0);
        let qs: f64 = row.get::<f64>(2).unwrap_or(0.0);
        let strategy: Option<String> = row.get(3).ok();
        let sr: Option<f64> = row.get(4).ok();
        let mean = if count > 0 { qs / count as f64 } else { 0.0 };
        let line = json!({ "id": id, "count": count, "quality_mean": mean,
            "strategy": strategy, "success_rate": sr });
        f.write_all(serde_json::to_string(&line)?.as_bytes()).await?;
        f.write_all(b"\n").await?;
    }
    f.flush().await?;
    Ok(out)
}

pub async fn export_preferences(store: &Store, dir: &Path) -> Result<PathBuf> {
    register_once();
    fs::create_dir_all(dir).await?;
    let sql = "SELECT query, chosen, rejected, created_at FROM preference_pairs ORDER BY created_at DESC";
    let mut rows = store.conn.query(sql, ()).await?;
    let out = dir.join("preferences.jsonl");
    let mut f = fs::File::create(&out).await?;
    while let Some(row) = rows.next().await? {
        let prompt: Option<String> = row.get(0).ok();
        let chosen: String = row.get(1)?;
        let rejected: String = row.get(2)?;
        let line = json!({ "prompt": prompt, "chosen": chosen, "rejected": rejected });
        f.write_all(serde_json::to_string(&line)?.as_bytes()).await?;
        f.write_all(b"\n").await?;
    }
    f.flush().await?;
    Ok(out)
}

fn url_enc(s: &str) -> String {
    s.chars().map(|c| match c {
        'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '/' => c.to_string(),
        _ => format!("%{:02X}", c as u32),
    }).collect()
}

pub async fn push_to_hugging_face(repo: &str, dir: &Path, token: &str, dry_run: bool) -> Result<Vec<String>> {
    register_once();
    if repo.split('/').count() != 2 {
        return Err(anyhow!("push: repo must be 'namespace/name', got '{repo}'"));
    }
    let mut rd = fs::read_dir(dir).await?;
    let mut files: Vec<PathBuf> = Vec::new();
    while let Some(e) = rd.next_entry().await? {
        if e.file_type().await?.is_file() { files.push(e.path()); }
    }
    files.sort();
    if files.is_empty() { return Err(anyhow!("push: no files in {}", dir.display())); }
    let names: Vec<String> = files.iter()
        .map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default())
        .collect();
    if dry_run { return Ok(names); }
    if token.is_empty() { return Err(anyhow!("push: HF token required")); }
    let client = reqwest::Client::new();
    let check = client.get(format!("https://huggingface.co/api/models/{repo}"))
        .bearer_auth(token).send().await?;
    if check.status() == reqwest::StatusCode::NOT_FOUND {
        let (ns, name) = repo.split_once('/').unwrap();
        let c = client.post("https://huggingface.co/api/repos/create").bearer_auth(token)
            .json(&json!({"name": name, "organization": ns, "type": "model", "private": true}))
            .send().await?;
        if !c.status().is_success() { return Err(anyhow!("HF create: {}", c.status())); }
    } else if !check.status().is_success() {
        return Err(anyhow!("HF check: {}", check.status()));
    }
    let mut uploaded = Vec::new();
    for (path, name) in files.iter().zip(names.iter()) {
        let body = fs::read(path).await?;
        let url = format!("https://huggingface.co/api/models/{repo}/upload/main/{}", url_enc(name));
        let res = client.post(url).bearer_auth(token)
            .header("Content-Type", "application/octet-stream").body(body).send().await?;
        if !res.status().is_success() { return Err(anyhow!("HF upload {name}: {}", res.status())); }
        uploaded.push(name.clone());
    }
    Ok(uploaded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    async fn seed(store: &Store) {
        let now = crate::store::now_ms();
        let blob: Vec<u8> = (0..16u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
        store.conn.execute("INSERT INTO router_weights(version,blob,algo,created_at,meta) VALUES(?1,?2,?3,?4,?5)",
            libsql::params![1i64, blob, "fastgrnn".to_string(), now, "{}".to_string()]).await.unwrap();
        store.conn.execute("INSERT INTO ewc_fisher(param_id,value,updated_at) VALUES(?1,?2,?3)",
            libsql::params!["w0".to_string(), 0.5f64, now]).await.unwrap();
        store.conn.execute("INSERT INTO patterns(id,count,quality_sum,created_at,updated_at) VALUES(?1,?2,?3,?4,?5)",
            libsql::params!["p1".to_string(), 4i64, 2.0f64, now, now]).await.unwrap();
        store.conn.execute("INSERT INTO preference_pairs(id,query,chosen,rejected,created_at) VALUES(?1,?2,?3,?4,?5)",
            libsql::params!["pr1".to_string(), "q".to_string(), "c".to_string(), "r".to_string(), now]).await.unwrap();
    }

    #[tokio::test]
    async fn export_roundtrip() {
        let tmp = tempdir().unwrap();
        let store = Store::open(tmp.path().join("s.db").to_str().unwrap()).await.unwrap();
        seed(&store).await;
        let dir = tmp.path().join("out");
        let st = export_safetensors(&store, &dir).await.unwrap();
        let b = std::fs::read(&st).unwrap();
        let hl = u64::from_le_bytes(b[0..8].try_into().unwrap()) as usize;
        let h: serde_json::Value = serde_json::from_slice(&b[8..8 + hl]).unwrap();
        assert_eq!(h["router_weights"]["dtype"], "F32");
        assert!(h.get("adapter").is_some());
        assert!(h.get("fisher").is_some());
        let pp = export_patterns(&store, &dir).await.unwrap();
        for l in std::fs::read_to_string(&pp).unwrap().lines() {
            let _: serde_json::Value = serde_json::from_str(l).unwrap();
        }
        let prf = export_preferences(&store, &dir).await.unwrap();
        for l in std::fs::read_to_string(&prf).unwrap().lines() {
            let _: serde_json::Value = serde_json::from_str(l).unwrap();
        }
        let files = push_to_hugging_face("ns/name", &dir, "", true).await.unwrap();
        assert!(files.iter().any(|f| f == "router.safetensors"));
        assert!(files.iter().any(|f| f == "patterns.jsonl"));
        assert!(files.iter().any(|f| f == "preferences.jsonl"));
    }
}
