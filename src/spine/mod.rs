use crate::observability;
use crate::store::types::TrajectoryRow;
use crate::store::Store;
use anyhow::Result;
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;

pub const DEFAULT_CAPACITY: usize = 1024;

pub struct TrajectorySpine {
    tx: Mutex<Option<mpsc::Sender<TrajectoryRow>>>,
    dropped: Arc<AtomicU64>,
    written: Arc<AtomicU64>,
    handle: Mutex<Option<JoinHandle<()>>>,
    capacity: usize,
}

impl TrajectorySpine {
    pub fn new(store: Arc<Store>, capacity: usize) -> Arc<Self> {
        let (tx, mut rx) = mpsc::channel::<TrajectoryRow>(capacity);
        let dropped = Arc::new(AtomicU64::new(0));
        let written = Arc::new(AtomicU64::new(0));
        let d2 = dropped.clone(); let w2 = written.clone();
        observability::register("spine", move || {
            json!({
                "dropped": d2.load(Ordering::Relaxed),
                "written": w2.load(Ordering::Relaxed),
                "capacity": capacity,
            })
        });
        let w_writer = written.clone();
        let store_writer = store.clone();
        let handle = tokio::spawn(async move {
            while let Some(row) = rx.recv().await {
                if let Err(e) = store_writer.insert_trajectory(&row).await {
                    tracing::warn!(error=%e, id=%row.id, "spine insert_trajectory failed");
                    continue;
                }
                w_writer.fetch_add(1, Ordering::Relaxed);
            }
        });
        Arc::new(Self {
            tx: Mutex::new(Some(tx)),
            dropped,
            written,
            handle: Mutex::new(Some(handle)),
            capacity,
        })
    }

    pub async fn send(&self, row: TrajectoryRow) -> Result<()> {
        let g = self.tx.lock().await;
        let tx = g.as_ref().ok_or_else(|| anyhow::anyhow!("spine: closed"))?;
        match tx.try_send(row) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                Err(anyhow::anyhow!("spine: consumer closed"))
            }
        }
    }

    pub async fn close(&self) {
        {
            let mut g = self.tx.lock().await;
            let _ = g.take();
        }
        let handle_opt = {
            let mut g = self.handle.lock().await;
            g.take()
        };
        if let Some(h) = handle_opt {
            let _ = h.await;
        }
    }

    pub fn dropped_count(&self) -> u64 { self.dropped.load(Ordering::Relaxed) }
    pub fn written_count(&self) -> u64 { self.written.load(Ordering::Relaxed) }
    pub fn capacity(&self) -> usize { self.capacity }
}

impl Drop for TrajectorySpine {
    fn drop(&mut self) { observability::unregister("spine"); }
}
