use rs_learn::backend;
use rs_learn::embeddings::Embedder;
use rs_learn::graph::http::HttpState;
use rs_learn::graph::ingest::Ingestor;
use rs_learn::graph::llm::LlmJson;
use rs_learn::graph::mcp::McpServer;
use rs_learn::graph::search::{SearchConfig, Searcher};
use rs_learn::learn::instant::FeedbackPayload;
use rs_learn::orchestrator::{Orchestrator, QueryOpts};
use rs_learn::store::Store;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("ready");
    match cmd {
        "ready" | "start" | "" => {
            let _ = Orchestrator::new_default().await?;
            println!("rs-learn ready");
        }
        "query" => cmd_query(&args[1..]).await?,
        "feedback" => cmd_feedback(&args[1..]).await?,
        "debug" => cmd_debug(&args[1..]).await?,
        "add" => cmd_add(&args[1..]).await?,
        "search" => cmd_search(&args[1..]).await?,
        "forget" => cmd_forget(&args[1..]).await?,
        "episodes" => cmd_episodes(&args[1..]).await?,
        "clear" => cmd_clear().await?,
        "build-communities" => cmd_build_communities().await?,
        "serve" => cmd_serve(&args[1..]).await?,
        "mcp" => cmd_mcp().await?,
        "help" | "-h" | "--help" => { print_help(false); }
        "version" | "-V" | "--version" => {
            println!("rs-learn {}", env!("CARGO_PKG_VERSION"));
        }
        other => {
            eprintln!("unknown subcommand '{other}'");
            print_help(true);
            std::process::exit(2);
        }
    }
    Ok(())
}

fn print_help(to_stderr: bool) {
    let lines = [
        "rs-learn <subcommand> [args]",
        "  ready | start               boot orchestrator and exit",
        "  query <text>                run one query through the orchestrator (prints JSON)",
        "  feedback <request_id> <quality 0..1> [signal]   record quality for a prior query",
        "  debug [subsystem]           dump /debug snapshot (all or one subsystem)",
        "  add <text> [--source S] [--file <path>|-] [--chunk-size N] [--no-extract]  ingest episode(s)",
        "  search <query> [--scope S] [--limit N]",
        "  forget <id|directive> [--source S] [--query Q] [--hard] [--limit N]   invalidate episodes (soft by default)",
        "  episodes [--source S] [--limit N]   list episodes by source (active only)",
        "  clear                       drop all graph data",
        "  build-communities           run label propagation + summarize",
        "  serve [--port N]            start HTTP REST server",
        "  mcp                         start MCP stdio server",
        "  version | --version         print version",
    ];
    for l in lines {
        if to_stderr { eprintln!("{l}"); } else { println!("{l}"); }
    }
}

async fn cmd_query(args: &[String]) -> anyhow::Result<()> {
    let Some(text) = args.first() else { anyhow::bail!("query requires text"); };
    let orch = Orchestrator::new_default().await?;
    let r = orch.query(text, QueryOpts::default()).await?;
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "request_id": r.request_id,
        "session_id": r.session_id,
        "text": r.text,
        "confidence": r.confidence,
        "latency_ms": r.latency_ms,
        "routing": r.routing,
        "stage_breakdown": r.stage_breakdown,
    }))?);
    Ok(())
}

async fn cmd_feedback(args: &[String]) -> anyhow::Result<()> {
    let Some(request_id) = args.first() else { anyhow::bail!("feedback requires <request_id>"); };
    let Some(q_str) = args.get(1) else { anyhow::bail!("feedback requires <quality>"); };
    let quality: f32 = q_str.parse().map_err(|_| anyhow::anyhow!("quality must be f32 in 0..1"))?;
    let signal = args.get(2).cloned();
    let orch = Orchestrator::new_default().await?;
    orch.feedback(request_id, FeedbackPayload { quality, signal }).await?;
    println!("ok");
    Ok(())
}

async fn cmd_debug(args: &[String]) -> anyhow::Result<()> {
    let _orch = Orchestrator::new_default().await?;
    let v = match args.first() {
        Some(name) => rs_learn::observability::dump()
            .get(name.as_str()).cloned()
            .ok_or_else(|| anyhow::anyhow!("unknown subsystem '{name}'; try one of: {:?}", rs_learn::observability::names()))?,
        None => rs_learn::observability::dump(),
    };
    println!("{}", serde_json::to_string_pretty(&v)?);
    Ok(())
}

async fn open_graph() -> anyhow::Result<(Arc<Store>, Arc<Embedder>, Arc<LlmJson>)> {
    let db_path = rs_learn::resolve_db_path();
    let store = Arc::new(Store::open(&db_path).await?);
    let embedder = Arc::new(Embedder::new());
    let backend = backend::from_env().map_err(|e| anyhow::anyhow!("backend: {e}"))?;
    let llm = Arc::new(LlmJson::new(backend));
    Ok((store, embedder, llm))
}

async fn cmd_add(args: &[String]) -> anyhow::Result<()> {
    let source = flag(args, "--source").unwrap_or_else(|| "message".into());
    let chunk_size: usize = flag(args, "--chunk-size").and_then(|s| s.parse().ok()).unwrap_or(0);
    let no_extract = args.iter().any(|a| a == "--no-extract");

    let text = if let Some(path) = flag(args, "--file") {
        if path == "-" {
            use std::io::Read;
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s)?;
            s
        } else {
            std::fs::read_to_string(&path)
                .map_err(|e| anyhow::anyhow!("cannot read '{}': {}", path, e))?
        }
    } else if args.first().map(|a| !a.starts_with('-')).unwrap_or(false) {
        args[0].clone()
    } else {
        anyhow::bail!("add requires text, --file <path>, or --file -  (stdin)");
    };

    let (store, embedder, llm) = open_graph().await?;
    let ingestor = Ingestor::new(store, embedder, llm);

    let chunks: Vec<String> = if chunk_size > 0 {
        chunk_text(&text, chunk_size).into_iter().map(String::from).collect()
    } else {
        vec![text.clone()]
    };

    let total = chunks.len();
    if no_extract {
        let futs: Vec<_> = chunks.iter().enumerate().map(|(i, chunk)| {
            let src = if total > 1 { format!("{} [{}/{}]", source, i + 1, total) } else { source.clone() };
            let ingestor = ingestor.clone();
            let chunk = chunk.clone();
            async move {
                let r = ingestor.add_episode_fast(&chunk, &src, None).await?;
                anyhow::Ok((i + 1, r))
            }
        }).collect();
        let results = futures::future::join_all(futs).await;
        for res in results {
            let (n, r) = res?;
            println!("episode={} chunk={}/{} (fast)", r.episode_id, n, total);
        }
    } else {
        for (i, chunk) in chunks.iter().enumerate() {
            let src = if total > 1 { format!("{} [{}/{}]", source, i + 1, total) } else { source.clone() };
            let r = ingestor.add_episode(chunk, &src, None, None).await?;
            println!(
                "episode={} nodes={} edges={} expired={} chunk={}/{}",
                r.episode_id, r.node_count, r.edge_count, r.expired_edge_ids.len(), i + 1, total
            );
        }
    }
    Ok(())
}

async fn cmd_forget(args: &[String]) -> anyhow::Result<()> {
    let hard = args.iter().any(|a| a == "--hard");
    let limit: usize = flag(args, "--limit").and_then(|s| s.parse().ok()).unwrap_or(20);
    let source = flag(args, "--source");
    let query = flag(args, "--query");

    // First positional that's not a flag. Two-word forms like `by-source <tag>` are also accepted.
    let positional: Vec<&String> = args.iter()
        .filter(|a| !a.starts_with("--"))
        .collect();

    let mut ids: Vec<String> = Vec::new();
    let mut effective_source = source.clone();
    let mut effective_query = query.clone();

    if positional.len() == 2 {
        match positional[0].as_str() {
            "by-source" => effective_source = effective_source.or(Some(positional[1].clone())),
            "by-query"  => effective_query  = effective_query.or(Some(positional[1].clone())),
            "by-id"     => ids.push(positional[1].clone()),
            _ => {}
        }
    } else if positional.len() == 1 && effective_source.is_none() && effective_query.is_none() {
        // Bare UUID = by-id
        ids.push(positional[0].clone());
    }

    let (store, embedder, llm) = open_graph().await?;
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64).unwrap_or(0);

    if let Some(src) = effective_source.as_deref() {
        let mut rows = store.conn.query(
            "SELECT id FROM episodes WHERE source = ?1 AND (invalid_at IS NULL OR invalid_at = 0)",
            libsql::params![src.to_string()],
        ).await?;
        while let Some(row) = rows.next().await? {
            if let Ok(id) = row.get::<String>(0) { ids.push(id); }
        }
    }
    if let Some(q) = effective_query.as_deref() {
        let searcher = Searcher::with_llm(store.clone(), embedder.clone(), llm.clone());
        let cfg = SearchConfig { limit, ..Default::default() };
        let hits = searcher.search_episodes(q, &cfg).await?;
        for h in hits {
            if let Some(id) = h.row.get("id").and_then(|v| v.as_str()) {
                ids.push(id.to_string());
            }
        }
    }
    ids.sort();
    ids.dedup();
    if ids.is_empty() {
        println!("forgot 0 episodes (no matches)");
        return Ok(());
    }
    let mut count = 0u64;
    for id in &ids {
        let r = if hard {
            store.conn.execute("DELETE FROM episodes WHERE id = ?1", libsql::params![id.clone()]).await?
        } else {
            store.conn.execute(
                "UPDATE episodes SET invalid_at = ?1 WHERE id = ?2 AND (invalid_at IS NULL OR invalid_at = 0)",
                libsql::params![now, id.clone()],
            ).await?
        };
        count += r;
    }
    println!("forgot {} episodes ({})", count, if hard { "hard" } else { "soft" });
    Ok(())
}

async fn cmd_episodes(args: &[String]) -> anyhow::Result<()> {
    let limit: usize = flag(args, "--limit").and_then(|s| s.parse().ok()).unwrap_or(50);
    let source = flag(args, "--source");
    let (store, _embedder, _llm) = open_graph().await?;
    let mut rows = match source.as_deref() {
        Some(src) => store.conn.query(
            "SELECT id, content, source, created_at, invalid_at FROM episodes \
             WHERE source = ?1 AND (invalid_at IS NULL OR invalid_at = 0) \
             ORDER BY created_at DESC LIMIT ?2",
            libsql::params![src.to_string(), limit as i64],
        ).await?,
        None => store.conn.query(
            "SELECT id, content, source, created_at, invalid_at FROM episodes \
             WHERE invalid_at IS NULL OR invalid_at = 0 \
             ORDER BY created_at DESC LIMIT ?1",
            libsql::params![limit as i64],
        ).await?,
    };
    let mut out = Vec::new();
    while let Some(row) = rows.next().await? {
        out.push(serde_json::json!({
            "id": row.get::<String>(0).unwrap_or_default(),
            "content": row.get::<String>(1).unwrap_or_default(),
            "source": row.get::<String>(2).ok(),
            "created_at": row.get::<i64>(3).ok(),
        }));
    }
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

fn chunk_text(text: &str, max_chars: usize) -> Vec<&str> {
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let end = (start + max_chars).min(text.len());
        // extend to next paragraph boundary if possible
        let slice = &text[start..end];
        let cut = if end == text.len() {
            slice.len()
        } else if let Some(p) = slice.rfind("\n\n") {
            p + 2
        } else if let Some(p) = slice.rfind('\n') {
            p + 1
        } else {
            slice.len()
        };
        chunks.push(&text[start..start + cut]);
        start += cut;
    }
    chunks
}

async fn cmd_search(args: &[String]) -> anyhow::Result<()> {
    let Some(query) = args.first() else {
        anyhow::bail!("search requires query");
    };
    let scope = flag(args, "--scope").unwrap_or_else(|| "nodes".into());
    let limit: usize = flag(args, "--limit").and_then(|s| s.parse().ok()).unwrap_or(10);
    let (store, embedder, llm) = open_graph().await?;
    let searcher = Searcher::with_llm(store, embedder, llm);
    let cfg = SearchConfig { limit, ..Default::default() };
    let hits = match scope.as_str() {
        "nodes" => searcher.search_nodes(query, &cfg).await?,
        "facts" | "edges" => searcher.search_facts(query, &cfg).await?,
        "episodes" => searcher.search_episodes(query, &cfg).await?,
        "communities" => searcher.search_communities(query, &cfg).await?,
        other => anyhow::bail!("unknown scope '{other}'"),
    };
    println!("{}", serde_json::to_string_pretty(&hits)?);
    Ok(())
}

async fn cmd_clear() -> anyhow::Result<()> {
    let (store, embedder, llm) = open_graph().await?;
    Ingestor::new(store, embedder, llm).clear_graph(None).await?;
    println!("cleared");
    Ok(())
}

async fn cmd_build_communities() -> anyhow::Result<()> {
    use rs_learn::graph::communities::CommunityOps;
    let (store, embedder, llm) = open_graph().await?;
    let ops = CommunityOps::new(store, embedder, llm);
    let r = ops.build_communities().await?;
    println!("communities={} members={}", r.community_count, r.member_count);
    Ok(())
}

async fn cmd_serve(args: &[String]) -> anyhow::Result<()> {
    let port: u16 = flag(args, "--port").and_then(|s| s.parse().ok()).unwrap_or(8000);
    rs_learn::graph::metrics::register();
    let (store, embedder, llm) = open_graph().await?;
    let state = HttpState::new(store, embedder, llm);
    let app = state.router();
    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("rs-learn serving on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

async fn cmd_mcp() -> anyhow::Result<()> {
    rs_learn::graph::metrics::register();
    let (store, embedder, llm) = open_graph().await?;
    let server = McpServer::new(store, embedder, llm);
    server.serve_stdio().await
}

fn flag(args: &[String], key: &str) -> Option<String> {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == key {
            return it.next().cloned();
        }
    }
    None
}
