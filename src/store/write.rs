use anyhow::Result;

use super::schema::vec_lit;
use super::types::*;
use super::{now_ms, Store};

impl Store {
    pub async fn insert_episode(&self, e: &EpisodeRow) -> Result<()> {
        let gid = e.group_id.clone().unwrap_or_else(|| "default".into());
        self.conn.execute(
            "INSERT INTO episodes(id,content,source,group_id,created_at,valid_at,invalid_at) VALUES(?1,?2,?3,?4,?5,?6,?7)
             ON CONFLICT(id) DO UPDATE SET content=excluded.content, source=excluded.source,
               group_id=excluded.group_id, valid_at=excluded.valid_at, invalid_at=excluded.invalid_at",
            libsql::params![e.id.clone(), e.content.clone(), e.source.clone(), gid, e.created_at.unwrap_or_else(now_ms), e.valid_at, e.invalid_at],
        ).await?;
        Ok(())
    }

    pub async fn insert_node(&self, n: &NodeRow) -> Result<()> {
        let lit = vec_lit(n.embedding.as_deref());
        let gid = n.group_id.clone().unwrap_or_else(|| "default".into());
        let sql = format!(
            "INSERT INTO nodes(id,name,type,summary,embedding,level,group_id,created_at) VALUES(?1,?2,?3,?4,{},?5,?6,?7)
             ON CONFLICT(id) DO UPDATE SET name=excluded.name, type=excluded.type,
               summary=excluded.summary, embedding=excluded.embedding, level=excluded.level,
               group_id=excluded.group_id",
            lit
        );
        self.conn.execute(&sql, libsql::params![
            n.id.clone(), n.name.clone(), n.r#type.clone(), n.summary.clone().unwrap_or_default(),
            n.level.unwrap_or(0), gid, n.created_at.unwrap_or_else(now_ms)
        ]).await?;
        Ok(())
    }

    pub async fn insert_edge(&self, e: &EdgeRow) -> Result<()> {
        let lit = vec_lit(e.embedding.as_deref());
        let gid = e.group_id.clone().unwrap_or_else(|| "default".into());
        let sql = format!(
            "INSERT INTO edges(id,src,dst,relation,fact,embedding,weight,group_id,created_at,valid_at,invalid_at)
             VALUES(?1,?2,?3,?4,?5,{},?6,?7,?8,?9,?10)
             ON CONFLICT(id) DO UPDATE SET relation=excluded.relation, fact=excluded.fact,
               embedding=excluded.embedding, weight=excluded.weight, group_id=excluded.group_id,
               valid_at=excluded.valid_at, invalid_at=excluded.invalid_at",
            lit
        );
        self.conn.execute(&sql, libsql::params![
            e.id.clone(), e.src.clone(), e.dst.clone(), e.relation.clone(), e.fact.clone().unwrap_or_default(),
            e.weight.unwrap_or(1.0), gid, e.created_at.unwrap_or_else(now_ms), e.valid_at, e.invalid_at
        ]).await?;
        Ok(())
    }

    pub async fn insert_trajectory(&self, t: &TrajectoryRow) -> Result<()> {
        let lit = vec_lit(t.query_embedding.as_deref());
        let retrieved = serde_json::to_string(&t.retrieved_ids.clone().unwrap_or_default())?;
        let sql = format!(
            "INSERT INTO trajectories(id,session_id,query,query_embedding,retrieved_ids,router_decision,response,activations,quality,latency_ms,created_at)
             VALUES(?1,?2,?3,{},?4,?5,?6,?7,?8,?9,?10)
             ON CONFLICT(id) DO UPDATE SET quality=excluded.quality, response=excluded.response",
            lit
        );
        self.conn.execute(&sql, libsql::params![
            t.id.clone(), t.session_id.clone(), t.query.clone().unwrap_or_default(),
            retrieved, t.router_decision.clone().unwrap_or_else(|| "{}".into()),
            t.response.clone().unwrap_or_default(), t.activations.clone(),
            t.quality.unwrap_or(0.0), t.latency_ms.unwrap_or(0), t.created_at.unwrap_or_else(now_ms)
        ]).await?;
        Ok(())
    }

    pub async fn upsert_pattern(&self, p: &PatternRow) -> Result<()> {
        let ts = now_ms();
        let lit = vec_lit(p.centroid.as_deref());
        let sql = format!(
            "INSERT INTO patterns(id,centroid,count,quality_sum,created_at,updated_at)
             VALUES(?1,{},?2,?3,?4,?5)
             ON CONFLICT(id) DO UPDATE SET centroid=excluded.centroid,
               count=excluded.count, quality_sum=excluded.quality_sum, updated_at=excluded.updated_at",
            lit
        );
        self.conn.execute(&sql, libsql::params![
            p.id.clone(), p.count.unwrap_or(0), p.quality_sum.unwrap_or(0.0), p.created_at.unwrap_or(ts), ts
        ]).await?;
        Ok(())
    }

    pub async fn insert_reasoning(&self, r: &ReasoningRow) -> Result<()> {
        self.conn.execute(
            "INSERT INTO reasoning_bank(id,pattern_id,strategy,success_rate,created_at)
             VALUES(?1,?2,?3,?4,?5)
             ON CONFLICT(id) DO UPDATE SET strategy=excluded.strategy,
               success_rate=excluded.success_rate, pattern_id=excluded.pattern_id",
            libsql::params![r.id.clone(), r.pattern_id.clone(), r.strategy.clone(), r.success_rate.unwrap_or(0.0), r.created_at.unwrap_or_else(now_ms)],
        ).await?;
        Ok(())
    }

    pub async fn save_router_weights(&self, version: i64, blob: &[u8], algo: &str, meta: &serde_json::Value) -> Result<()> {
        self.conn.execute(
            "INSERT INTO router_weights(version,blob,algo,created_at,meta) VALUES(?1,?2,?3,?4,?5)
             ON CONFLICT(version) DO UPDATE SET blob=excluded.blob, algo=excluded.algo, meta=excluded.meta",
            libsql::params![version, blob.to_vec(), algo.to_string(), now_ms(), serde_json::to_string(meta)?],
        ).await?;
        Ok(())
    }

    pub async fn update_fisher(&self, param_id: &str, value: f64) -> Result<()> {
        self.conn.execute(
            "INSERT INTO ewc_fisher(param_id,value,updated_at) VALUES(?1,?2,?3)
             ON CONFLICT(param_id) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            libsql::params![param_id.to_string(), value, now_ms()],
        ).await?;
        Ok(())
    }

    pub async fn save_fisher_vec(&self, param_id: &str, values: &[f32]) -> Result<()> {
        let ts = now_ms();
        for (i, v) in values.iter().enumerate() {
            let key = format!("{}:{}", param_id, i);
            self.conn.execute(
                "INSERT INTO ewc_fisher(param_id,value,updated_at) VALUES(?1,?2,?3)
                 ON CONFLICT(param_id) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                libsql::params![key, *v as f64, ts],
            ).await?;
        }
        Ok(())
    }

    pub async fn add_preference_pair(&self, p: &PreferenceRow) -> Result<()> {
        self.conn.execute(
            "INSERT INTO preference_pairs(id,query,chosen,rejected,created_at) VALUES(?1,?2,?3,?4,?5)
             ON CONFLICT(id) DO UPDATE SET chosen=excluded.chosen, rejected=excluded.rejected",
            libsql::params![p.id.clone(), p.query.clone().unwrap_or_default(), p.chosen.clone(), p.rejected.clone(), p.created_at.unwrap_or_else(now_ms)],
        ).await?;
        Ok(())
    }

    pub async fn insert_session(&self, s: &SessionRow) -> Result<()> {
        let meta = s.meta.clone().unwrap_or_else(|| serde_json::json!({}));
        self.conn.execute(
            "INSERT INTO sessions(id,created_at,meta) VALUES(?1,?2,?3)
             ON CONFLICT(id) DO UPDATE SET meta=excluded.meta",
            libsql::params![s.id.clone(), s.created_at.unwrap_or_else(now_ms), serde_json::to_string(&meta)?],
        ).await?;
        Ok(())
    }
}
