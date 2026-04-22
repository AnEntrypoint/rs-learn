use anyhow::{anyhow, Result};

pub const MAX_GROUP_ID_LEN: usize = 128;
pub const MAX_CONTENT_BYTES: usize = 200_000;
pub const MAX_LIMIT: usize = 1000;

pub fn validate_group_id(s: &str) -> Result<()> {
    if s.is_empty() { return Err(anyhow!("group_id must be non-empty")); }
    if s.len() > MAX_GROUP_ID_LEN { return Err(anyhow!("group_id exceeds {MAX_GROUP_ID_LEN} chars")); }
    let ok = s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'));
    if !ok { return Err(anyhow!("group_id may contain only [A-Za-z0-9_.-]")); }
    Ok(())
}

pub fn validate_content(s: &str) -> Result<()> {
    if s.is_empty() { return Err(anyhow!("content must be non-empty")); }
    if s.len() > MAX_CONTENT_BYTES {
        return Err(anyhow!("content exceeds {MAX_CONTENT_BYTES} bytes (got {})", s.len()));
    }
    Ok(())
}

pub fn validate_limit(n: usize) -> Result<()> {
    if n == 0 || n > MAX_LIMIT { return Err(anyhow!("limit must be 1..={MAX_LIMIT}")); }
    Ok(())
}

pub fn validate_iso_date(s: &str) -> Result<()> {
    if super::time::parse_iso_ms(s).is_some() { Ok(()) } else { Err(anyhow!("invalid ISO date: {s}")) }
}

pub fn validate_reranker(s: &str) -> Result<()> {
    match s {
        "rrf" | "mmr" | "node_distance" | "episode_mentions" | "cross_encoder" => Ok(()),
        other => Err(anyhow!("unknown reranker '{other}'")),
    }
}
