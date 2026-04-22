use crate::store::{now_ms, Store};
use anyhow::Result;
use std::sync::Arc;
use uuid::Uuid;

use super::llm::LlmJson;
use super::prompts::summarize_sagas;

pub struct SagaOps {
    pub store: Arc<Store>,
    pub llm: Arc<LlmJson>,
}

impl SagaOps {
    pub fn new(store: Arc<Store>, llm: Arc<LlmJson>) -> Self {
        Self { store, llm }
    }

    pub async fn create_saga(&self, name: &str) -> Result<String> {
        let id = Uuid::new_v4().to_string();
        let now = now_ms();
        self.store.conn.execute(
            "INSERT INTO sagas(id,name,summary,created_at) VALUES(?1,?2,?3,?4)",
            libsql::params![id.clone(), name.to_string(), String::new(), now],
        ).await?;
        Ok(id)
    }

    pub async fn add_episode_to_saga(&self, saga_id: &str, episode_id: &str) -> Result<i64> {
        let next_seq = self.next_seq(saga_id).await?;
        self.store.conn.execute(
            "INSERT INTO saga_episodes(saga_id,episode_id,seq) VALUES(?1,?2,?3)
             ON CONFLICT(saga_id,episode_id) DO UPDATE SET seq=excluded.seq",
            libsql::params![saga_id.to_string(), episode_id.to_string(), next_seq],
        ).await?;
        let now = now_ms();
        self.store.conn.execute(
            "INSERT INTO edges(id,src,dst,relation,fact,weight,created_at,valid_at)
             VALUES(?1,?2,?3,?4,?5,?6,?7,?8)
             ON CONFLICT(id) DO NOTHING",
            libsql::params![
                Uuid::new_v4().to_string(), saga_id.to_string(), episode_id.to_string(),
                "HAS_EPISODE".to_string(), String::new(), 1.0_f64, now, now
            ],
        ).await?;
        if next_seq > 0 {
            let prev = self.episode_at(saga_id, next_seq - 1).await?;
            if let Some(prev_id) = prev {
                self.store.conn.execute(
                    "INSERT INTO edges(id,src,dst,relation,fact,weight,created_at,valid_at)
                     VALUES(?1,?2,?3,?4,?5,?6,?7,?8)
                     ON CONFLICT(id) DO NOTHING",
                    libsql::params![
                        Uuid::new_v4().to_string(), prev_id, episode_id.to_string(),
                        "NEXT_EPISODE".to_string(), String::new(), 1.0_f64, now, now
                    ],
                ).await?;
            }
        }
        Ok(next_seq)
    }

    pub async fn summarize_saga(&self, saga_id: &str) -> Result<String> {
        let name = self.saga_name(saga_id).await?.unwrap_or_default();
        let existing = self.saga_summary(saga_id).await?.unwrap_or_default();
        let episodes = self.saga_episodes_text(saga_id).await?;
        let prompt = summarize_sagas::summarize_saga(&summarize_sagas::SagaCtx {
            saga_name: &name,
            existing_summary: &existing,
            episodes: &episodes,
        });
        let v = self.llm.call(&prompt.system, &prompt.user, |_| Ok(())).await.ok();
        let summary = v
            .and_then(|v| v.get("summary").and_then(|s| s.as_str()).map(String::from))
            .unwrap_or_default();
        let summary = super::text::truncate_at_sentence(&summary, super::prompts::snippets::MAX_SUMMARY_CHARS);
        self.store.conn.execute(
            "UPDATE sagas SET summary = ?1 WHERE id = ?2",
            libsql::params![summary.clone(), saga_id.to_string()],
        ).await?;
        Ok(summary)
    }

    async fn next_seq(&self, saga_id: &str) -> Result<i64> {
        let mut rows = self.store.conn.query(
            "SELECT COALESCE(MAX(seq), -1) + 1 FROM saga_episodes WHERE saga_id = ?1",
            libsql::params![saga_id.to_string()],
        ).await?;
        if let Some(row) = rows.next().await? {
            return Ok(row.get::<i64>(0).unwrap_or(0));
        }
        Ok(0)
    }

    async fn episode_at(&self, saga_id: &str, seq: i64) -> Result<Option<String>> {
        let mut rows = self.store.conn.query(
            "SELECT episode_id FROM saga_episodes WHERE saga_id = ?1 AND seq = ?2",
            libsql::params![saga_id.to_string(), seq],
        ).await?;
        if let Some(row) = rows.next().await? {
            return Ok(row.get::<String>(0).ok());
        }
        Ok(None)
    }

    async fn saga_name(&self, saga_id: &str) -> Result<Option<String>> {
        let mut rows = self.store.conn.query(
            "SELECT name FROM sagas WHERE id = ?1",
            libsql::params![saga_id.to_string()],
        ).await?;
        if let Some(row) = rows.next().await? {
            return Ok(row.get::<String>(0).ok());
        }
        Ok(None)
    }

    async fn saga_summary(&self, saga_id: &str) -> Result<Option<String>> {
        let mut rows = self.store.conn.query(
            "SELECT summary FROM sagas WHERE id = ?1",
            libsql::params![saga_id.to_string()],
        ).await?;
        if let Some(row) = rows.next().await? {
            return Ok(row.get::<String>(0).ok());
        }
        Ok(None)
    }

    async fn saga_episodes_text(&self, saga_id: &str) -> Result<Vec<String>> {
        let mut rows = self.store.conn.query(
            "SELECT e.content FROM saga_episodes se
             JOIN episodes e ON e.id = se.episode_id
             WHERE se.saga_id = ?1
             ORDER BY se.seq ASC",
            libsql::params![saga_id.to_string()],
        ).await?;
        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(row.get::<String>(0)?);
        }
        Ok(out)
    }
}
