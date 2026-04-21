pub mod schema;
pub mod triggers;
pub mod types;
pub mod read;
pub mod write;

pub use types::*;

use anyhow::Result;
use libsql::{Builder, Connection, Database};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub use schema::{SCHEMA_VERSION, EMBED_DIM, vec_lit, fts_query};

pub fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

pub struct Store {
    pub db: Database,
    pub conn: Connection,
    pub path: String,
}

impl Store {
    pub async fn open(path: &str) -> Result<Self> {
        let p = Path::new(path);
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let db = Builder::new_local(path).build().await?;
        let conn = db.connect()?;
        for p in schema::PRAGMAS {
            let _ = conn.execute(p, ()).await;
        }
        let store = Self { db, conn, path: path.to_string() };
        store.migrate().await?;
        Ok(store)
    }

    pub async fn migrate(&self) -> Result<()> {
        for q in schema::TABLES {
            self.conn.execute(q, ()).await?;
        }
        for t in schema::TRIGGERS {
            self.conn.execute(t, ()).await?;
        }
        self.conn.execute(
            "INSERT INTO schema_version(version,applied_at) VALUES(?1,?2) ON CONFLICT(version) DO NOTHING",
            libsql::params![SCHEMA_VERSION as i64, now_ms()],
        ).await?;
        let _ = self.conn.execute("PRAGMA optimize", ()).await;
        Ok(())
    }

    pub async fn count_rows(&self, table: &str) -> i64 {
        let sql = format!("SELECT COUNT(*) AS c FROM {}", table);
        match self.conn.query(&sql, ()).await {
            Ok(mut rows) => match rows.next().await {
                Ok(Some(row)) => row.get::<i64>(0).unwrap_or(-1),
                _ => -1,
            },
            Err(_) => -1,
        }
    }

    pub async fn close(self) {
        drop(self.conn);
        drop(self.db);
    }
}
