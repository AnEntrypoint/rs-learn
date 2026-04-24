use anyhow::Result;

use super::types::*;
use super::{now_ms, Store};

impl Store {
    pub async fn evict_stale_reasoning(&self, ttl_days: u64, min_success_rate: f64) -> Result<u64> {
        let cutoff = now_ms() - (ttl_days as i64 * 86_400_000);
        let mut rows = self.conn.query("SELECT COUNT(*) FROM reasoning_bank", ()).await?;
        let total: i64 = rows.next().await?.map(|r| r.get::<i64>(0).unwrap_or(0)).unwrap_or(0);
        if total < 20 { return Ok(0); }
        let affected = self.conn.execute(
            "DELETE FROM reasoning_bank WHERE success_rate < ?1 AND created_at < ?2",
            libsql::params![min_success_rate, cutoff],
        ).await?;
        Ok(affected)
    }

    pub async fn evict_noise_patterns(&self) -> Result<u64> {
        Ok(self.conn.execute("DELETE FROM patterns WHERE count < 3 AND (quality_sum / NULLIF(count, 0)) < 0.2", ()).await?)
    }

    pub async fn prune_trajectories(&self, keep: usize) -> Result<u64> {
        let keep = keep.max(500) as i64;
        Ok(self.conn.execute(
            "DELETE FROM trajectories WHERE id NOT IN (
               SELECT id FROM (
                 SELECT id FROM trajectories WHERE quality > 0.7
                 UNION ALL
                 SELECT id FROM trajectories ORDER BY created_at DESC LIMIT ?1
               )
             )",
            libsql::params![keep],
        ).await?)
    }

    pub async fn prune_router_weights(&self, keep_versions: usize) -> Result<u64> {
        Ok(self.conn.execute(
            "DELETE FROM router_weights WHERE version NOT IN (SELECT version FROM router_weights ORDER BY version DESC LIMIT ?1)",
            libsql::params![keep_versions.max(1) as i64],
        ).await?)
    }

    pub async fn load_session(&self, id: &str) -> Result<Option<SessionRow>> {
        let mut rows = self.conn.query(
            "SELECT id, created_at, meta FROM sessions WHERE id=?1",
            libsql::params![id.to_string()],
        ).await?;
        Ok(match rows.next().await? {
            Some(row) => {
                let meta_str: String = row.get::<String>(2).unwrap_or_else(|_| "{}".into());
                let meta = serde_json::from_str(&meta_str).unwrap_or(serde_json::json!({}));
                Some(SessionRow { id: row.get(0)?, created_at: row.get(1).ok(), meta: Some(meta) })
            }
            None => None,
        })
    }

    pub async fn insert_session(&self, s: &SessionRow) -> Result<()> {
        let meta = s.meta.clone().unwrap_or_else(|| serde_json::json!({}));
        self.conn.execute(
            "INSERT INTO sessions(id,created_at,meta) VALUES(?1,?2,?3) ON CONFLICT(id) DO UPDATE SET meta=excluded.meta",
            libsql::params![s.id.clone(), s.created_at.unwrap_or_else(now_ms), serde_json::to_string(&meta)?],
        ).await?;
        Ok(())
    }
}
