use crate::store::Store;
use anyhow::Result;
use moka::future::Cache;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

const CACHE_CAP: u64 = 8;

type Key = (PathBuf, Option<String>);

static CACHE: OnceLock<Cache<Key, Arc<Store>>> = OnceLock::new();

fn cache() -> &'static Cache<Key, Arc<Store>> {
    CACHE.get_or_init(|| Cache::builder().max_capacity(CACHE_CAP).build())
}

fn norm_discipline(d: Option<&str>) -> Option<String> {
    d.and_then(|s| if s.is_empty() { None } else { Some(s.to_string()) })
}

pub async fn resolve(root: Option<&Path>, discipline: Option<&str>) -> Result<Arc<Store>> {
    let path_str = crate::resolve_db_path_for(root, discipline);
    let key: Key = (PathBuf::from(&path_str), norm_discipline(discipline));
    let c = cache();
    if let Some(s) = c.get(&key).await {
        return Ok(s);
    }
    let store = Arc::new(Store::open(&path_str).await?);
    c.insert(key.clone(), store.clone()).await;
    Ok(store)
}

pub fn capacity() -> u64 { CACHE_CAP }
