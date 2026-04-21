use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{anyhow, Result};
use serde_json::json;

use crate::observability;

pub const EMBED_DIM: usize = 768;
const CACHE_MAX: usize = 1024;
const MODEL_TAG: &str = "rs-search::nomic-embed-text";

#[derive(Default)]
struct Lru {
    map: HashMap<[u8; 32], Vec<f32>>,
    order: VecDeque<[u8; 32]>,
}

impl Lru {
    fn get(&mut self, k: &[u8; 32]) -> Option<Vec<f32>> {
        let v = self.map.get(k).cloned()?;
        if let Some(pos) = self.order.iter().position(|x| x == k) {
            self.order.remove(pos);
        }
        self.order.push_back(*k);
        Some(v)
    }
    fn put(&mut self, k: [u8; 32], v: Vec<f32>) {
        if self.map.contains_key(&k) {
            if let Some(pos) = self.order.iter().position(|x| *x == k) {
                self.order.remove(pos);
            }
        }
        self.map.insert(k, v);
        self.order.push_back(k);
        while self.order.len() > CACHE_MAX {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            }
        }
    }
    fn len(&self) -> usize { self.map.len() }
}

struct Stats {
    hits: AtomicU64,
    misses: AtomicU64,
    calls: AtomicU64,
}

fn stats() -> &'static Stats {
    static S: OnceLock<Stats> = OnceLock::new();
    S.get_or_init(|| Stats {
        hits: AtomicU64::new(0),
        misses: AtomicU64::new(0),
        calls: AtomicU64::new(0),
    })
}

fn cache() -> &'static Mutex<Lru> {
    static C: OnceLock<Mutex<Lru>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(Lru::default()))
}

fn register_once() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        observability::register("embeddings", || {
            let s = stats();
            let size = cache().lock().map(|c| c.len()).unwrap_or(0);
            json!({
                "cache_hits": s.hits.load(Ordering::Relaxed),
                "cache_misses": s.misses.load(Ordering::Relaxed),
                "total_calls": s.calls.load(Ordering::Relaxed),
                "cache_size": size,
                "model": MODEL_TAG,
                "dim": EMBED_DIM,
            })
        });
    });
}

fn key_for(text: &str) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(MODEL_TAG.as_bytes());
    h.update(&[0u8]);
    h.update(text.as_bytes());
    *h.finalize().as_bytes()
}

fn embed_raw(text: &str) -> Result<Vec<f32>> {
    let v = rs_search::embed::embed_query(text, Path::new(""))
        .ok_or_else(|| anyhow!("rs_search::embed::embed_query returned None (vector feature disabled or model unavailable)"))?;
    if v.len() != EMBED_DIM {
        return Err(anyhow!(
            "rs_search returned embedding of len {}, expected {}",
            v.len(),
            EMBED_DIM
        ));
    }
    Ok(v)
}

pub struct Embedder {
    _priv: (),
}

impl Embedder {
    pub fn new() -> Self {
        register_once();
        Self { _priv: () }
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        stats().calls.fetch_add(1, Ordering::Relaxed);
        let k = key_for(text);
        {
            let mut c = cache().lock().map_err(|e| anyhow!("cache poisoned: {e}"))?;
            if let Some(v) = c.get(&k) {
                stats().hits.fetch_add(1, Ordering::Relaxed);
                return Ok(v);
            }
        }
        stats().misses.fetch_add(1, Ordering::Relaxed);
        let v = embed_raw(text)?;
        {
            let mut c = cache().lock().map_err(|e| anyhow!("cache poisoned: {e}"))?;
            c.put(k, v.clone());
        }
        Ok(v)
    }

    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            out.push(self.embed(t)?);
        }
        Ok(out)
    }
}

impl Default for Embedder {
    fn default() -> Self { Self::new() }
}

pub fn stats_snapshot() -> (u64, u64, u64) {
    let s = stats();
    (
        s.hits.load(Ordering::Relaxed),
        s.misses.load(Ordering::Relaxed),
        s.calls.load(Ordering::Relaxed),
    )
}

// Suppress unused Arc warning in minimal builds.
#[allow(dead_code)]
fn _arc_anchor() -> Option<Arc<()>> { None }
