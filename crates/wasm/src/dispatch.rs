use crate::graph::temporal_core::{KvBackend, TemporalGraph, InvalidationOutcome};
use crate::graph::types::EdgeRow;
use crate::graph::attention::{Attention, Subgraph};
use crate::learn::deep_core::DeepCore;
use crate::learn::instant_core::{FeedbackPayload, InstantCore};
use crate::router::core::{Route, RouteCtx, Router, RouterConfig};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Deserialize)]
pub struct DispatchRequest {
    pub verb: String,
    #[serde(default)]
    pub body: Value,
}

#[derive(Serialize)]
pub struct DispatchResponse {
    pub ok: bool,
    pub verb: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub struct LearnSession<B: KvBackend> {
    pub graph: TemporalGraph<B>,
    pub instant: Option<InstantCore>,
    pub deep: DeepCore,
    pub router: Option<Router>,
    pub attention: Option<Attention>,
}

impl<B: KvBackend> LearnSession<B> {
    pub fn new(kv: B) -> Self {
        Self {
            graph: TemporalGraph::new(kv),
            instant: None,
            deep: DeepCore::new(),
            router: None,
            attention: None,
        }
    }

    pub fn dispatch(&mut self, req: DispatchRequest) -> DispatchResponse {
        let verb = req.verb.clone();
        let result: Result<Value, String> = match verb.as_str() {
            "insert_edge"     => self.handle_insert_edge(req.body),
            "query_at"        => self.handle_query_at(req.body),
            "invalidate_edge" => self.handle_invalidate(req.body),
            "contradict"      => self.handle_contradict(req.body),
            "get_edge"        => self.handle_get_edge(req.body),
            "init_instant"    => self.handle_init_instant(req.body),
            "feedback"        => self.handle_feedback(req.body),
            "apply_adapter"   => self.handle_apply_adapter(req.body),
            "reset_adapter"   => self.handle_reset_adapter(req.body),
            "record_loss"     => self.handle_record_loss(req.body),
            "consolidate"     => self.handle_consolidate(req.body),
            "ewc_penalty"     => self.handle_ewc_penalty(req.body),
            "init_router"     => self.handle_init_router(req.body),
            "route"           => self.handle_route(req.body),
            "record_outcome"  => self.handle_record_outcome(req.body),
            "init_attention"  => self.handle_init_attention(req.body),
            "attend"          => self.handle_attend(req.body),
            "nudge_relation"  => self.handle_nudge_relation(req.body),
            "health"          => Ok(self.health()),
            _ => Err(format!("unknown verb: {}", verb)),
        };
        match result {
            Ok(data) => DispatchResponse { ok: true, verb, data: Some(data), error: None },
            Err(e) => DispatchResponse { ok: false, verb, data: None, error: Some(e) },
        }
    }

    fn health(&self) -> Value {
        json!({
            "instant_ready": self.instant.is_some(),
            "router_ready": self.router.is_some(),
            "attention_ready": self.attention.is_some(),
            "deep_fisher_keys": self.deep.fisher.keys().count(),
            "loss_window_size": self.deep.loss_ring.len(),
            "boundaries_detected": self.deep.boundaries_detected,
        })
    }

    fn handle_insert_edge(&mut self, body: Value) -> Result<Value, String> {
        let edge: EdgeRow = serde_json::from_value(body).map_err(|e| format!("parse edge: {}", e))?;
        self.graph.insert_edge(edge.clone())?;
        Ok(json!({ "edge_id": edge.id }))
    }

    fn handle_query_at(&self, body: Value) -> Result<Value, String> {
        let src = body.get("src").and_then(|v| v.as_str()).ok_or("src required")?;
        let t = body.get("t").and_then(|v| v.as_i64()).ok_or("t required")?;
        let limit = body.get("limit").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let edges = if limit > 0 { self.graph.query_at_bounded(src, t, limit) } else { self.graph.query_at(src, t) };
        Ok(json!({ "edges": edges, "count": edges.len() }))
    }

    fn handle_invalidate(&mut self, body: Value) -> Result<Value, String> {
        let id = body.get("edge_id").and_then(|v| v.as_str()).ok_or("edge_id required")?;
        let invalid_at = body.get("invalid_at").and_then(|v| v.as_i64()).ok_or("invalid_at required")?;
        let expired_at = body.get("expired_at").and_then(|v| v.as_i64()).ok_or("expired_at required")?;
        self.graph.invalidate_edge(id, invalid_at, expired_at)?;
        Ok(json!({ "edge_id": id, "invalid_at": invalid_at }))
    }

    fn handle_contradict(&mut self, body: Value) -> Result<Value, String> {
        let edge: EdgeRow = serde_json::from_value(body.get("new_edge").cloned().ok_or("new_edge required")?)
            .map_err(|e| format!("parse new_edge: {}", e))?;
        let old_ids: Vec<String> = body.get("contradicts")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_default();
        let now = body.get("now_ms").and_then(|v| v.as_i64()).unwrap_or(0);
        let outcome = self.graph.insert_with_contradiction(edge, &old_ids, now)?;
        Ok(match outcome {
            InvalidationOutcome::Inserted { edge_id } => json!({ "kind": "inserted", "edge_id": edge_id }),
            InvalidationOutcome::Invalidated { invalidated_edge_id, new_edge_id } => json!({
                "kind": "invalidated",
                "invalidated_edge_id": invalidated_edge_id,
                "new_edge_id": new_edge_id,
            }),
        })
    }

    fn handle_get_edge(&self, body: Value) -> Result<Value, String> {
        let id = body.get("edge_id").and_then(|v| v.as_str()).ok_or("edge_id required")?;
        match self.graph.get_edge(id) {
            Some(e) => Ok(json!({ "edge": e })),
            None => Ok(json!({ "edge": null })),
        }
    }

    fn handle_init_instant(&mut self, body: Value) -> Result<Value, String> {
        let targets: Vec<String> = body.get("targets").and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
            .ok_or("targets required (string array)")?;
        if targets.is_empty() { return Err("targets must not be empty".into()); }
        let core = InstantCore::new(targets.clone());
        #[cfg(target_arch = "wasm32")]
        crate::learn::persist::save_adapter(&core, "default").map_err(|e| format!("save_adapter: {:?}", e))?;
        self.instant = Some(core);
        Ok(json!({ "n_targets": targets.len() }))
    }

    fn handle_feedback(&mut self, body: Value) -> Result<Value, String> {
        #[cfg(target_arch = "wasm32")]
        if self.instant.is_none() {
            if let Some(c) = crate::learn::persist::load_adapter("default").map_err(|e| format!("load_adapter: {:?}", e))? {
                self.instant = Some(c);
            }
        }
        let core = self.instant.as_mut().ok_or("instant not initialized; call init_instant first")?;
        let emb: Vec<f32> = serde_json::from_value(body.get("embedding").cloned().ok_or("embedding required")?)
            .map_err(|e| format!("embedding parse: {}", e))?;
        let model = body.get("model").and_then(|v| v.as_str()).ok_or("model required")?.to_string();
        let payload: FeedbackPayload = serde_json::from_value(body.get("payload").cloned().ok_or("payload required")?)
            .map_err(|e| format!("payload parse: {}", e))?;
        let now = body.get("now_ms").and_then(|v| v.as_i64()).unwrap_or(0);
        core.feedback(&emb, &model, payload, now)?;
        let resp = json!({
            "adapter_norm": core.adapter_norm(),
            "feedback_count": core.feedback_count,
            "lr": core.lr,
        });
        #[cfg(target_arch = "wasm32")]
        {
            let core = self.instant.as_ref().unwrap();
            crate::learn::persist::save_adapter(core, "default").map_err(|e| format!("save_adapter: {:?}", e))?;
        }
        Ok(resp)
    }

    fn handle_apply_adapter(&mut self, body: Value) -> Result<Value, String> {
        #[cfg(target_arch = "wasm32")]
        if self.instant.is_none() {
            if let Some(c) = crate::learn::persist::load_adapter("default").map_err(|e| format!("load_adapter: {:?}", e))? {
                self.instant = Some(c);
            }
        }
        let core = self.instant.as_ref().ok_or("instant not initialized")?;
        let emb: Vec<f32> = serde_json::from_value(body.get("embedding").cloned().ok_or("embedding required")?)
            .map_err(|e| format!("embedding parse: {}", e))?;
        let mut logits = vec![0f32; core.n_targets];
        core.apply_adapter(&emb, &mut logits);
        Ok(json!({ "logits": logits, "targets": core.targets }))
    }

    fn handle_reset_adapter(&mut self, _body: Value) -> Result<Value, String> {
        let core = self.instant.as_mut().ok_or("instant not initialized")?;
        core.reset_adapter();
        #[cfg(target_arch = "wasm32")]
        crate::learn::persist::save_adapter(core, "default").map_err(|e| format!("save_adapter: {:?}", e))?;
        Ok(json!({ "resets_performed": core.resets_performed }))
    }

    fn handle_record_loss(&mut self, body: Value) -> Result<Value, String> {
        let loss = body.get("loss").and_then(|v| v.as_f64()).ok_or("loss required")? as f32;
        let fired = self.deep.record_loss(loss);
        let (mean, stddev, n) = self.deep.window_stats();
        Ok(json!({
            "boundary_fired": fired,
            "window_mean": mean,
            "window_stddev": stddev,
            "samples": n,
            "boundaries_total": self.deep.boundaries_detected,
        }))
    }

    fn handle_consolidate(&mut self, body: Value) -> Result<Value, String> {
        let param_id = body.get("param_id").and_then(|v| v.as_str()).ok_or("param_id required")?.to_string();
        let params: Vec<f32> = serde_json::from_value(body.get("params").cloned().ok_or("params required")?)
            .map_err(|e| format!("params parse: {}", e))?;
        let grads: Vec<f32> = serde_json::from_value(body.get("grads").cloned().ok_or("grads required")?)
            .map_err(|e| format!("grads parse: {}", e))?;
        self.deep.consolidate(&param_id, &params, &grads)?;
        #[cfg(target_arch = "wasm32")]
        crate::learn::persist::save_fisher(&self.deep, &param_id).map_err(|e| format!("save_fisher: {:?}", e))?;
        Ok(json!({ "param_id": param_id, "fisher_len": self.deep.fisher.get(&param_id).map(|v| v.len()).unwrap_or(0) }))
    }

    fn handle_ewc_penalty(&mut self, body: Value) -> Result<Value, String> {
        let param_id = body.get("param_id").and_then(|v| v.as_str()).ok_or("param_id required")?.to_string();
        let params: Vec<f32> = serde_json::from_value(body.get("params").cloned().ok_or("params required")?)
            .map_err(|e| format!("params parse: {}", e))?;
        #[cfg(target_arch = "wasm32")]
        if !self.deep.fisher.contains_key(&param_id) {
            let _ = crate::learn::persist::load_fisher_into(&mut self.deep, &param_id);
        }
        let penalty = self.deep.ewc_penalty(&param_id, &params);
        Ok(json!({ "penalty": penalty }))
    }

    fn handle_init_router(&mut self, body: Value) -> Result<Value, String> {
        let in_dim = body.get("in_dim").and_then(|v| v.as_u64()).ok_or("in_dim required")? as usize;
        let targets: Vec<String> = body.get("targets").and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
            .ok_or("targets required (string array)")?;
        if targets.is_empty() { return Err("targets must not be empty".into()); }
        let mut cfg = RouterConfig::new(in_dim, targets);
        if let Some(eps) = body.get("epsilon").and_then(|v| v.as_f64()) { cfg.epsilon = eps as f32; }
        if let Some(thr) = body.get("threshold").and_then(|v| v.as_u64()) { cfg.threshold = thr; }
        let trained = body.get("trained").and_then(|v| v.as_bool()).unwrap_or(false);
        let mut r = Router::new(cfg).map_err(|e| e)?;
        if trained { r.set_trained(true); }
        #[cfg(target_arch = "wasm32")]
        crate::router::persist::save_router(&r, "default").map_err(|e| format!("save_router: {:?}", e))?;
        self.router = Some(r);
        Ok(json!({ "ready": true, "trained": trained }))
    }

    fn handle_route(&mut self, body: Value) -> Result<Value, String> {
        #[cfg(target_arch = "wasm32")]
        if self.router.is_none() {
            if let Some(r) = crate::router::persist::load_router("default").map_err(|e| format!("load_router: {:?}", e))? {
                self.router = Some(r);
            }
        }
        let r = self.router.as_mut().ok_or("router not initialized; call init_router first")?;
        let emb: Vec<f32> = serde_json::from_value(body.get("embedding").cloned().ok_or("embedding required")?)
            .map_err(|e| format!("embedding parse: {}", e))?;
        if emb.len() != r.config.in_dim {
            return Err(format!("embedding must be len {}, got {}", r.config.in_dim, emb.len()));
        }
        let mut ctx = RouteCtx::default();
        if let Some(tt) = body.get("task_type").and_then(|v| v.as_str()) { ctx.task_type = Some(tt.into()); }
        if let Some(et) = body.get("estimated_tokens").and_then(|v| v.as_u64()) { ctx.estimated_tokens = et; }
        let route: Route = r.route(&emb, &ctx);
        Ok(json!({
            "model": route.model,
            "context_bucket": route.context_bucket,
            "temperature": route.temperature,
            "top_p": route.top_p,
            "confidence": route.confidence,
            "algo": route.algo,
            "exploration": route.exploration,
        }))
    }

    fn handle_record_outcome(&mut self, body: Value) -> Result<Value, String> {
        #[cfg(target_arch = "wasm32")]
        if self.router.is_none() {
            if let Some(r) = crate::router::persist::load_router("default").map_err(|e| format!("load_router: {:?}", e))? {
                self.router = Some(r);
            }
        }
        let r = self.router.as_mut().ok_or("router not initialized")?;
        let target = body.get("target").and_then(|v| v.as_str()).ok_or("target required")?;
        let quality = body.get("quality").and_then(|v| v.as_f64()).ok_or("quality required")? as f32;
        r.record_outcome(target, quality);
        #[cfg(target_arch = "wasm32")]
        crate::router::persist::save_router(r, "default").map_err(|e| format!("save_router: {:?}", e))?;
        let idx = r.config.targets.iter().position(|t| t == target);
        Ok(json!({
            "target": target,
            "count": idx.map(|i| r.per_target_counts[i]).unwrap_or(0),
            "quality_milli": idx.map(|i| r.per_target_quality_milli[i]).unwrap_or(0),
        }))
    }

    fn handle_init_attention(&mut self, body: Value) -> Result<Value, String> {
        let dim = body.get("dim").and_then(|v| v.as_u64()).ok_or("dim required")? as usize;
        let heads = body.get("heads").and_then(|v| v.as_u64()).unwrap_or(8) as usize;
        let head_dim = body.get("head_dim").and_then(|v| v.as_u64()).unwrap_or((dim / heads) as u64) as usize;
        let seed = body.get("seed").and_then(|v| v.as_u64()).unwrap_or(42) as u32;
        if heads == 0 || head_dim == 0 || dim == 0 { return Err("dim/heads/head_dim must be > 0".into()); }
        let a = Attention::new(dim, heads, head_dim, seed);
        #[cfg(target_arch = "wasm32")]
        crate::graph::attention_persist::save_attention(&a, seed, "default").map_err(|e| format!("save_attention: {:?}", e))?;
        self.attention = Some(a);
        Ok(json!({ "dim": dim, "heads": heads, "head_dim": head_dim }))
    }

    fn handle_nudge_relation(&mut self, body: Value) -> Result<Value, String> {
        let relation = body.get("relation").and_then(|v| v.as_str()).ok_or("relation required")?.to_string();
        let quality = body.get("signed_quality").and_then(|v| v.as_f64()).ok_or("signed_quality required")? as f32;
        #[cfg(target_arch = "wasm32")]
        {
            let (a, seed) = crate::graph::attention_persist::load_attention("default")
                .map_err(|e| format!("load_attention: {:?}", e))?
                .ok_or("attention not initialized; call init_attention first")?;
            let mut a = a;
            a.nudge_relation(&relation, quality);
            crate::graph::attention_persist::save_attention(&a, seed, "default").map_err(|e| format!("save_attention: {:?}", e))?;
            self.attention = Some(a);
            return Ok(json!({ "relation": relation, "nudged": true }));
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let a = self.attention.as_mut().ok_or("attention not initialized")?;
            a.nudge_relation(&relation, quality);
            Ok(json!({ "relation": relation, "nudged": true }))
        }
    }

    fn handle_attend(&mut self, body: Value) -> Result<Value, String> {
        #[cfg(target_arch = "wasm32")]
        if self.attention.is_none() {
            if let Some((a, _s)) = crate::graph::attention_persist::load_attention("default").map_err(|e| format!("load_attention: {:?}", e))? {
                self.attention = Some(a);
            }
        }
        let a = self.attention.as_ref().ok_or("attention not initialized")?;
        let q: Vec<f32> = serde_json::from_value(body.get("query").cloned().ok_or("query required")?)
            .map_err(|e| format!("query parse: {}", e))?;
        let sub: Subgraph = serde_json::from_value(body.get("subgraph").cloned().ok_or("subgraph required")?)
            .map_err(|e| format!("subgraph parse: {}", e))?;
        let now = body.get("now_ms").and_then(|v| v.as_i64()).unwrap_or(0);
        let ctx = a.attend(&q, &sub, now)?;
        Ok(json!({ "vector": ctx.vector, "weights": ctx.weights, "n_nodes": sub.nodes.len() }))
    }
}

pub fn dispatch_json<B: KvBackend>(session: &mut LearnSession<B>, raw: &[u8]) -> Vec<u8> {
    let req: DispatchRequest = match serde_json::from_slice(raw) {
        Ok(r) => r,
        Err(e) => {
            let resp = DispatchResponse {
                ok: false, verb: "?".into(), data: None,
                error: Some(format!("parse: {}", e)),
            };
            return serde_json::to_vec(&resp).unwrap_or_default();
        }
    };
    let resp = session.dispatch(req);
    serde_json::to_vec(&resp).unwrap_or_default()
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod integration_tests {
    use super::*;
    use crate::graph::temporal_core::MemKv;

    fn mk_emb(dim: usize, seed: u8) -> Vec<f32> {
        (0..dim).map(|i| ((i as u8).wrapping_add(seed) as f32) / 255.0 - 0.5).collect()
    }

    fn call(session: &mut LearnSession<MemKv>, verb: &str, body: Value) -> Value {
        let req = DispatchRequest { verb: verb.into(), body };
        let resp = session.dispatch(req);
        assert!(resp.ok, "verb {} failed: {:?}", verb, resp.error);
        resp.data.unwrap_or(json!(null))
    }

    fn call_err(session: &mut LearnSession<MemKv>, verb: &str, body: Value) -> String {
        let req = DispatchRequest { verb: verb.into(), body };
        let resp = session.dispatch(req);
        assert!(!resp.ok, "expected failure for verb {}, got data: {:?}", verb, resp.data);
        resp.error.unwrap_or_default()
    }

    #[test]
    fn end_to_end_lora_feedback_loop_grows_then_resets() {
        let mut s = LearnSession::new(MemKv::default());
        call(&mut s, "init_instant", json!({ "targets": ["fast", "slow"] }));
        let emb = mk_emb(384, 1);
        for i in 0..50 {
            let r = call(&mut s, "feedback", json!({
                "embedding": emb, "model": "fast",
                "payload": { "quality": 1.0 },
                "now_ms": i,
            }));
            let norm = r.get("adapter_norm").and_then(|v| v.as_f64()).unwrap();
            assert!(norm <= 5.001, "norm {} exceeded bound at step {}", norm, i);
        }
        let after = call(&mut s, "feedback", json!({
            "embedding": emb, "model": "fast",
            "payload": { "quality": 1.0 }, "now_ms": 100,
        }));
        assert!(after.get("adapter_norm").and_then(|v| v.as_f64()).unwrap() > 0.0);

        let reset = call(&mut s, "reset_adapter", json!({}));
        assert_eq!(reset.get("resets_performed").and_then(|v| v.as_u64()), Some(1));
    }

    #[test]
    fn end_to_end_temporal_graph_contradiction_and_pit_query() {
        let mut s = LearnSession::new(MemKv::default());
        call(&mut s, "insert_edge", json!({
            "id": "e_acme", "src": "alice", "dst": "acme",
            "relation": "works_at",
            "valid_at": 1000, "created_at": 1000,
        }));

        let outcome = call(&mut s, "contradict", json!({
            "new_edge": {
                "id": "e_globex", "src": "alice", "dst": "globex",
                "relation": "works_at",
                "valid_at": 2000, "created_at": 2000,
            },
            "contradicts": ["e_acme"],
            "now_ms": 2050,
        }));
        assert_eq!(outcome.get("kind").and_then(|v| v.as_str()), Some("invalidated"));
        assert_eq!(outcome.get("invalidated_edge_id").and_then(|v| v.as_str()), Some("e_acme"));

        let at_1500 = call(&mut s, "query_at", json!({ "src": "alice", "t": 1500 }));
        let count_1500 = at_1500.get("count").and_then(|v| v.as_u64()).unwrap();
        assert_eq!(count_1500, 1, "at t=1500 only acme should be active");
        let edges_1500 = at_1500.get("edges").and_then(|v| v.as_array()).unwrap();
        assert_eq!(edges_1500[0].get("id").and_then(|v| v.as_str()), Some("e_acme"));

        let at_2500 = call(&mut s, "query_at", json!({ "src": "alice", "t": 2500 }));
        let edges_2500 = at_2500.get("edges").and_then(|v| v.as_array()).unwrap();
        assert_eq!(edges_2500.len(), 1);
        assert_eq!(edges_2500[0].get("id").and_then(|v| v.as_str()), Some("e_globex"));

        let acme = call(&mut s, "get_edge", json!({ "edge_id": "e_acme" }));
        let acme_edge = acme.get("edge").unwrap();
        assert_eq!(acme_edge.get("invalid_at").and_then(|v| v.as_i64()), Some(2000));
        assert!(acme_edge.get("expired_at").and_then(|v| v.as_i64()).is_some());
    }

    #[test]
    fn end_to_end_router_routes_and_records() {
        let mut s = LearnSession::new(MemKv::default());
        call(&mut s, "init_router", json!({
            "in_dim": 384, "targets": ["gpt", "claude", "local"], "trained": false,
        }));
        let route = call(&mut s, "route", json!({
            "embedding": mk_emb(384, 1), "estimated_tokens": 500,
        }));
        assert_eq!(route.get("algo").and_then(|v| v.as_str()), Some("rule"));
        assert_eq!(route.get("model").and_then(|v| v.as_str()), Some("gpt"));

        let outcome = call(&mut s, "record_outcome", json!({
            "target": "gpt", "quality": 0.9,
        }));
        assert_eq!(outcome.get("count").and_then(|v| v.as_u64()), Some(1));
        assert!(outcome.get("quality_milli").and_then(|v| v.as_u64()).unwrap() > 500);
    }

    #[test]
    fn end_to_end_trained_router_uses_fastgrnn() {
        let mut s = LearnSession::new(MemKv::default());
        call(&mut s, "init_router", json!({
            "in_dim": 384, "targets": ["a", "b", "c"], "trained": true,
        }));
        let route = call(&mut s, "route", json!({
            "embedding": mk_emb(384, 2), "estimated_tokens": 1000,
        }));
        assert_eq!(route.get("algo").and_then(|v| v.as_str()), Some("fastgrnn"));
        let model = route.get("model").and_then(|v| v.as_str()).unwrap();
        assert!(["a", "b", "c"].contains(&model));
    }

    #[test]
    fn end_to_end_deep_ewc_boundary_detection() {
        let mut s = LearnSession::new(MemKv::default());
        for _ in 0..5 {
            let r = call(&mut s, "record_loss", json!({ "loss": 0.5 }));
            assert_eq!(r.get("boundary_fired").and_then(|v| v.as_bool()), Some(false));
        }
        let outlier = call(&mut s, "record_loss", json!({ "loss": 100.0 }));
        assert_eq!(outlier.get("boundary_fired").and_then(|v| v.as_bool()), Some(true));

        call(&mut s, "consolidate", json!({
            "param_id": "adapter", "params": [0.1, 0.2, 0.3], "grads": [1.0, 1.0, 1.0],
        }));
        let penalty = call(&mut s, "ewc_penalty", json!({
            "param_id": "adapter", "params": [1.0, 1.0, 1.0],
        }));
        let p = penalty.get("penalty").and_then(|v| v.as_f64()).unwrap();
        assert!(p > 0.0, "ewc penalty for distance should be > 0, got {}", p);

        let zero = call(&mut s, "ewc_penalty", json!({
            "param_id": "adapter", "params": [0.1, 0.2, 0.3],
        }));
        assert_eq!(zero.get("penalty").and_then(|v| v.as_f64()), Some(0.0));
    }

    #[test]
    fn end_to_end_attention_with_subgraph() {
        let mut s = LearnSession::new(MemKv::default());
        call(&mut s, "init_attention", json!({
            "dim": 64, "heads": 4, "head_dim": 16, "seed": 7,
        }));
        let result = call(&mut s, "attend", json!({
            "query": mk_emb(64, 1),
            "subgraph": {
                "nodes": [
                    { "id": "n1", "embedding": mk_emb(64, 2), "created_at": 1000 },
                    { "id": "n2", "embedding": mk_emb(64, 3), "created_at": 2000 },
                ],
                "edges": [
                    { "src": "q", "dst": "n1", "relation": "entity", "weight": 1.0, "created_at": 1000 },
                    { "src": "q", "dst": "n2", "relation": "mention", "weight": 0.5, "created_at": 2000 },
                ],
            },
            "now_ms": 5000,
        }));
        let v = result.get("vector").and_then(|v| v.as_array()).unwrap();
        assert_eq!(v.len(), 64);
        let weights = result.get("weights").and_then(|v| v.as_array()).unwrap();
        assert_eq!(weights.len(), 4);
        for row in weights {
            let row_arr = row.as_array().unwrap();
            assert_eq!(row_arr.len(), 2);
            let sum: f64 = row_arr.iter().filter_map(|v| v.as_f64()).sum();
            assert!((sum - 1.0).abs() < 1e-3, "weights should sum to 1, got {}", sum);
        }
    }

    #[test]
    fn end_to_end_full_pipeline_lora_plus_graph_plus_router() {
        let mut s = LearnSession::new(MemKv::default());
        call(&mut s, "init_instant", json!({ "targets": ["fast", "slow"] }));
        call(&mut s, "init_router", json!({ "in_dim": 384, "targets": ["fast", "slow"], "trained": true }));
        call(&mut s, "insert_edge", json!({
            "id": "e1", "src": "user", "dst": "doc",
            "relation": "entity", "valid_at": 1000, "created_at": 1000,
        }));

        let emb = mk_emb(384, 9);
        let route = call(&mut s, "route", json!({ "embedding": emb, "estimated_tokens": 200 }));
        let chosen = route.get("model").and_then(|v| v.as_str()).unwrap().to_string();

        let fb = call(&mut s, "feedback", json!({
            "embedding": emb, "model": chosen,
            "payload": { "quality": 0.95 }, "now_ms": 1500,
        }));
        assert!(fb.get("adapter_norm").and_then(|v| v.as_f64()).unwrap() > 0.0);

        call(&mut s, "record_outcome", json!({ "target": chosen, "quality": 0.95 }));

        let query_now = call(&mut s, "query_at", json!({ "src": "user", "t": 2000 }));
        assert_eq!(query_now.get("count").and_then(|v| v.as_u64()), Some(1));

        let h = call(&mut s, "health", json!({}));
        assert_eq!(h.get("instant_ready").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(h.get("router_ready").and_then(|v| v.as_bool()), Some(true));
    }

    #[test]
    fn dispatch_unknown_verb_errors() {
        let mut s = LearnSession::new(MemKv::default());
        let err = call_err(&mut s, "no_such_verb", json!({}));
        assert!(err.contains("unknown verb"));
    }

    #[test]
    fn dispatch_missing_required_field_errors() {
        let mut s = LearnSession::new(MemKv::default());
        call(&mut s, "init_instant", json!({ "targets": ["m"] }));
        let err = call_err(&mut s, "feedback", json!({ "model": "m" }));
        assert!(err.contains("embedding"));
    }

    #[test]
    fn dispatch_json_roundtrip() {
        let mut s = LearnSession::new(MemKv::default());
        let raw = serde_json::to_vec(&json!({
            "verb": "health", "body": {},
        })).unwrap();
        let resp_bytes = dispatch_json(&mut s, &raw);
        let resp: Value = serde_json::from_slice(&resp_bytes).unwrap();
        assert_eq!(resp.get("ok").and_then(|v| v.as_bool()), Some(true));
        assert_eq!(resp.get("verb").and_then(|v| v.as_str()), Some("health"));
    }

    #[test]
    fn cross_session_persistence_via_kv_backend() {
        let mut kv = MemKv::default();
        {
            let mut s = LearnSession::new(MemKv::default());
            std::mem::swap(&mut s.graph.kv, &mut kv);
            call(&mut s, "insert_edge", json!({
                "id": "persist1", "src": "x", "dst": "y",
                "relation": "rel", "valid_at": 100, "created_at": 100,
            }));
            std::mem::swap(&mut s.graph.kv, &mut kv);
        }
        let mut s2 = LearnSession::new(kv);
        let got = call(&mut s2, "get_edge", json!({ "edge_id": "persist1" }));
        let edge = got.get("edge").unwrap();
        assert_eq!(edge.get("id").and_then(|v| v.as_str()), Some("persist1"));
        assert_eq!(edge.get("src").and_then(|v| v.as_str()), Some("x"));
    }
}
