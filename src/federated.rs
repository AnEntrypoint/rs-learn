use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::observability::{register, unregister};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    pub agent_id: String,
    pub embedding: Vec<f32>,
    pub quality: f32,
    pub timestamp: i64,
    #[serde(default)]
    pub meta: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedExport {
    pub agent_id: String,
    pub trajectories: Vec<Trajectory>,
    pub quality_avg: f32,
}

pub struct EphemeralAgent {
    id: String,
    buffer: Vec<Trajectory>,
}

impl EphemeralAgent {
    pub fn new<S: Into<String>>(id: S) -> Self {
        Self { id: id.into(), buffer: Vec::new() }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn record(&mut self, mut trajectory: Trajectory) {
        if trajectory.agent_id.is_empty() {
            trajectory.agent_id = self.id.clone();
        }
        self.buffer.push(trajectory);
    }

    pub fn export_state(&mut self) -> FederatedExport {
        let trajectories: Vec<Trajectory> = std::mem::take(&mut self.buffer);
        let quality_avg = if trajectories.is_empty() {
            0.0
        } else {
            let sum: f32 = trajectories.iter().map(|t| t.quality).sum();
            sum / trajectories.len() as f32
        };
        FederatedExport { agent_id: self.id.clone(), trajectories, quality_avg }
    }
}

struct CoordinatorState {
    pool: VecDeque<Trajectory>,
    agents_seen: u64,
    consolidations: u64,
}

pub struct FederatedCoordinator {
    capacity: usize,
    quality_threshold: f32,
    consolidation_interval: u64,
    state: Arc<Mutex<CoordinatorState>>,
    key: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AggregateResult {
    pub accepted: usize,
    pub pool_size: usize,
    pub consolidate: bool,
}

impl FederatedCoordinator {
    pub fn new(capacity: usize, quality_threshold: f32, consolidation_interval: u64) -> Self {
        let state = Arc::new(Mutex::new(CoordinatorState {
            pool: VecDeque::new(),
            agents_seen: 0,
            consolidations: 0,
        }));
        let key = "federated".to_string();
        let s = state.clone();
        let cap = capacity;
        let qt = quality_threshold;
        let ci = consolidation_interval;
        register(&key, move || {
            let g = s.lock().expect("federated state poisoned");
            json!({
                "poolSize": g.pool.len(),
                "capacity": cap,
                "qualityThreshold": qt,
                "consolidationInterval": ci,
                "agentsSeen": g.agents_seen,
                "consolidations": g.consolidations,
            })
        });
        Self { capacity, quality_threshold, consolidation_interval, state, key }
    }

    pub fn default_coordinator() -> Self {
        Self::new(50_000, 0.4, 50)
    }

    pub fn aggregate(&self, export: FederatedExport) -> AggregateResult {
        let mut g = self.state.lock().expect("federated state poisoned");
        g.agents_seen += 1;
        let mut accepted = 0usize;
        for t in export.trajectories.into_iter() {
            if t.quality >= self.quality_threshold {
                g.pool.push_back(t);
                accepted += 1;
            }
        }
        while g.pool.len() > self.capacity {
            g.pool.pop_front();
        }
        let trigger = self.consolidation_interval > 0
            && g.agents_seen % self.consolidation_interval == 0;
        if trigger {
            g.consolidations += 1;
        }
        AggregateResult { accepted, pool_size: g.pool.len(), consolidate: trigger }
    }

    pub fn share_patterns(&self) -> Vec<Trajectory> {
        let g = self.state.lock().expect("federated state poisoned");
        g.pool.iter().cloned().collect()
    }

    pub fn pool_size(&self) -> usize {
        self.state.lock().expect("federated state poisoned").pool.len()
    }

    pub fn consolidations(&self) -> u64 {
        self.state.lock().expect("federated state poisoned").consolidations
    }
}

impl Drop for FederatedCoordinator {
    fn drop(&mut self) {
        unregister(&self.key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn traj(q: f32) -> Trajectory {
        Trajectory {
            agent_id: "a".into(),
            embedding: vec![0.0; 4],
            quality: q,
            timestamp: 0,
            meta: Value::Null,
        }
    }

    #[test]
    fn filters_by_quality_and_triggers_consolidation() {
        let coord = FederatedCoordinator::new(50_000, 0.4, 50);
        let mut agent = EphemeralAgent::new("agent-1");
        agent.record(traj(0.1));
        agent.record(traj(0.5));
        agent.record(traj(0.9));
        let res = coord.aggregate(agent.export_state());
        assert_eq!(res.accepted, 2);
        assert_eq!(coord.pool_size(), 2);
        assert!(coord.share_patterns().iter().all(|t| t.quality >= 0.4));

        for _ in 0..49 {
            let mut a = EphemeralAgent::new("x");
            a.record(traj(0.8));
            coord.aggregate(a.export_state());
        }
        assert_eq!(coord.consolidations(), 1);
    }
}
