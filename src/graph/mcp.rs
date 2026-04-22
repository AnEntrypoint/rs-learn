use crate::embeddings::Embedder;
use crate::store::Store;
use anyhow::Result;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use super::communities::CommunityOps;
use super::entities::Entity;
use super::ingest::Ingestor;
use super::llm::LlmJson;
use super::sagas::SagaOps;
use super::search::{SearchConfig, Searcher};

pub struct McpServer {
    pub store: Arc<Store>,
    pub embedder: Arc<Embedder>,
    pub ingestor: Arc<Ingestor>,
    pub searcher: Arc<Searcher>,
    pub community_ops: Arc<CommunityOps>,
    pub saga_ops: Arc<SagaOps>,
    pub llm: Arc<LlmJson>,
}

impl McpServer {
    pub fn new(store: Arc<Store>, embedder: Arc<Embedder>, llm: Arc<LlmJson>) -> Arc<Self> {
        let ingestor = Ingestor::new(store.clone(), embedder.clone(), llm.clone());
        let searcher = Arc::new(Searcher::new(store.clone(), embedder.clone()));
        let community_ops = Arc::new(CommunityOps::new(store.clone(), embedder.clone(), llm.clone()));
        let saga_ops = Arc::new(SagaOps::new(store.clone(), llm.clone()));
        Arc::new(Self { store, embedder, ingestor, searcher, community_ops, saga_ops, llm })
    }

    pub async fn serve_stdio(self: Arc<Self>) -> Result<()> {
        let stdin = tokio::io::stdin();
        let mut reader = BufReader::new(stdin).lines();
        let mut stdout = tokio::io::stdout();
        while let Some(line) = reader.next_line().await? {
            let line = line.trim().to_string();
            if line.is_empty() { continue; }
            let response = match serde_json::from_str::<Value>(&line) {
                Ok(req) => self.handle(req).await,
                Err(e) => parse_error(e.to_string()),
            };
            let bytes = serde_json::to_vec(&response)?;
            stdout.write_all(&bytes).await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        }
        Ok(())
    }

    async fn handle(&self, req: Value) -> Value {
        let id = req.get("id").cloned().unwrap_or(Value::Null);
        let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let params = req.get("params").cloned().unwrap_or(Value::Null);
        match method {
            "initialize" => ok(id, json!({
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "rs-learn", "version": env!("CARGO_PKG_VERSION") }
            })),
            "tools/list" => ok(id, json!({ "tools": tool_list() })),
            "tools/call" => {
                let name = params.get("name").and_then(|s| s.as_str()).unwrap_or("").to_string();
                let args = params.get("arguments").cloned().unwrap_or(Value::Null);
                match self.dispatch(&name, args).await {
                    Ok(result) => ok(id, json!({
                        "content": [{"type":"text","text": result.to_string()}],
                        "structuredContent": result,
                    })),
                    Err(e) => err(id, -32000, e.to_string()),
                }
            }
            "ping" => ok(id, json!({})),
            _ => err(id, -32601, format!("method not found: {method}")),
        }
    }

    async fn dispatch(&self, tool: &str, args: Value) -> Result<Value> {
        match tool {
            "add_episode" => {
                let content = str_arg(&args, "content")?;
                let source = args.get("source").and_then(|s| s.as_str()).unwrap_or("message").to_string();
                let ref_time = args.get("reference_time").and_then(|s| s.as_str()).map(String::from);
                let r = self.ingestor.add_episode(&content, &source, ref_time.as_deref()).await?;
                Ok(json!({
                    "episode_id": r.episode_id,
                    "nodes": r.node_count,
                    "edges": r.edge_count,
                    "expired_edge_ids": r.expired_edge_ids,
                }))
            }
            "add_triplet" => {
                let src_id = str_arg(&args, "src_id")?;
                let src_name = str_arg(&args, "src_name")?;
                let dst_id = str_arg(&args, "dst_id")?;
                let dst_name = str_arg(&args, "dst_name")?;
                let relation = str_arg(&args, "relation")?;
                let fact = args.get("fact").and_then(|s| s.as_str()).unwrap_or("").to_string();
                let src = Entity { id: src_id, name: src_name.clone(), entity_type: None, embedding: self.embedder.embed(&src_name).ok() };
                let dst = Entity { id: dst_id, name: dst_name.clone(), entity_type: None, embedding: self.embedder.embed(&dst_name).ok() };
                let eid = self.ingestor.add_triplet(src, dst, &relation, &fact).await?;
                Ok(json!({ "edge_id": eid }))
            }
            "search" | "search_nodes" => self.do_search(args, "nodes").await,
            "search_facts" => self.do_search(args, "facts").await,
            "search_episodes" => self.do_search(args, "episodes").await,
            "search_communities" => self.do_search(args, "communities").await,
            "get_node" => self.get_single("nodes", args).await,
            "get_edge" => self.get_single("edges", args).await,
            "get_episode" => self.get_single("episodes", args).await,
            "clear_graph" => { self.ingestor.clear_graph().await?; Ok(json!({ "status": "ok" })) }
            "build_communities" => {
                let r = self.community_ops.build_communities().await?;
                Ok(json!({ "communities": r.community_count, "members": r.member_count }))
            }
            "remove_communities" => { self.community_ops.remove_communities().await?; Ok(json!({ "status": "ok" })) }
            "create_saga" => {
                let name = str_arg(&args, "name")?;
                let id = self.saga_ops.create_saga(&name).await?;
                Ok(json!({ "saga_id": id }))
            }
            "summarize_saga" => {
                let saga_id = str_arg(&args, "saga_id")?;
                let s = self.saga_ops.summarize_saga(&saga_id).await?;
                Ok(json!({ "summary": s }))
            }
            "delete_episode" => { self.delete_by_id("episodes", args).await?; Ok(json!({ "status": "ok" })) }
            "delete_entity_edge" => { self.delete_by_id("edges", args).await?; Ok(json!({ "status": "ok" })) }
            "delete_entity_node" => { self.delete_by_id("nodes", args).await?; Ok(json!({ "status": "ok" })) }
            "debug_state" => Ok(crate::observability::dump()),
            other => Err(anyhow::anyhow!("unknown tool '{other}'")),
        }
    }

    async fn do_search(&self, args: Value, scope: &str) -> Result<Value> {
        let query = str_arg(&args, "query")?;
        let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
        let cfg = SearchConfig { limit, ..Default::default() };
        let hits = match scope {
            "nodes" => self.searcher.search_nodes(&query, &cfg).await?,
            "facts" => self.searcher.search_facts(&query, &cfg).await?,
            "episodes" => self.searcher.search_episodes(&query, &cfg).await?,
            "communities" => self.searcher.search_communities(&query, &cfg).await?,
            _ => vec![],
        };
        Ok(json!({ "hits": hits }))
    }

    async fn get_single(&self, table: &str, args: Value) -> Result<Value> {
        let id = str_arg(&args, "id")?;
        let sql = format!("SELECT * FROM {table} WHERE id = ?1");
        let mut rows = self.store.conn.query(&sql, libsql::params![id]).await?;
        let Some(row) = rows.next().await? else { return Ok(Value::Null); };
        let n = row.column_count();
        let mut map = serde_json::Map::new();
        for i in 0..n {
            let name = row.column_name(i).map(|s| s.to_string()).unwrap_or_else(|| format!("c{i}"));
            if let Ok(v) = row.get_value(i) { map.insert(name, libsql_value_to_json(&v)); }
        }
        Ok(Value::Object(map))
    }

    async fn delete_by_id(&self, table: &str, args: Value) -> Result<()> {
        let id = str_arg(&args, "id")?;
        let sql = format!("DELETE FROM {table} WHERE id = ?1");
        self.store.conn.execute(&sql, libsql::params![id]).await?;
        Ok(())
    }
}

fn str_arg(args: &Value, key: &str) -> Result<String> {
    args.get(key).and_then(|v| v.as_str()).map(String::from)
        .ok_or_else(|| anyhow::anyhow!("missing string arg '{key}'"))
}

fn libsql_value_to_json(v: &libsql::Value) -> Value {
    match v {
        libsql::Value::Null => Value::Null,
        libsql::Value::Integer(i) => Value::from(*i),
        libsql::Value::Real(f) => Value::from(*f),
        libsql::Value::Text(s) => Value::from(s.clone()),
        libsql::Value::Blob(b) => Value::from(b.len()),
    }
}

fn ok(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn err(id: Value, code: i64, message: String) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

fn parse_error(msg: String) -> Value {
    json!({ "jsonrpc": "2.0", "id": null, "error": { "code": -32700, "message": msg } })
}

fn tool_list() -> Vec<Value> {
    let schema = |props: Value, required: Vec<&str>| json!({
        "type": "object",
        "properties": props,
        "required": required,
    });
    let s = || json!({"type":"string"});
    let num = || json!({"type":"number"});
    vec![
        json!({"name":"add_episode","description":"Ingest an episode","inputSchema": schema(json!({
            "content": s(), "source": s(), "reference_time": s()
        }), vec!["content"])}),
        json!({"name":"add_triplet","inputSchema": schema(json!({
            "src_id": s(), "src_name": s(), "dst_id": s(), "dst_name": s(),
            "relation": s(), "fact": s()
        }), vec!["src_id","src_name","dst_id","dst_name","relation"])}),
        json!({"name":"search","inputSchema": schema(json!({"query": s(), "limit": num()}), vec!["query"])}),
        json!({"name":"search_nodes","inputSchema": schema(json!({"query": s(), "limit": num()}), vec!["query"])}),
        json!({"name":"search_facts","inputSchema": schema(json!({"query": s(), "limit": num()}), vec!["query"])}),
        json!({"name":"search_episodes","inputSchema": schema(json!({"query": s(), "limit": num()}), vec!["query"])}),
        json!({"name":"search_communities","inputSchema": schema(json!({"query": s(), "limit": num()}), vec!["query"])}),
        json!({"name":"get_node","inputSchema": schema(json!({"id": s()}), vec!["id"])}),
        json!({"name":"get_edge","inputSchema": schema(json!({"id": s()}), vec!["id"])}),
        json!({"name":"get_episode","inputSchema": schema(json!({"id": s()}), vec!["id"])}),
        json!({"name":"clear_graph","inputSchema": schema(json!({}), vec![])}),
        json!({"name":"build_communities","inputSchema": schema(json!({}), vec![])}),
        json!({"name":"remove_communities","inputSchema": schema(json!({}), vec![])}),
        json!({"name":"create_saga","inputSchema": schema(json!({"name": s()}), vec!["name"])}),
        json!({"name":"summarize_saga","inputSchema": schema(json!({"saga_id": s()}), vec!["saga_id"])}),
        json!({"name":"delete_episode","inputSchema": schema(json!({"id": s()}), vec!["id"])}),
        json!({"name":"delete_entity_edge","inputSchema": schema(json!({"id": s()}), vec!["id"])}),
        json!({"name":"delete_entity_node","inputSchema": schema(json!({"id": s()}), vec!["id"])}),
        json!({"name":"debug_state","inputSchema": schema(json!({}), vec![])}),
    ]
}
