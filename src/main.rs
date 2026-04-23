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
        "  add <text> [--source S]     ingest an episode",
        "  search <query> [--scope S] [--limit N]",
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
    let db_path = std::env::var("RS_LEARN_DB_PATH").unwrap_or_else(|_| "./rs-learn.db".into());
    let store = Arc::new(Store::open(&db_path).await?);
    let embedder = Arc::new(Embedder::new());
    let backend = backend::from_env().map_err(|e| anyhow::anyhow!("backend: {e}"))?;
    let llm = Arc::new(LlmJson::new(backend));
    Ok((store, embedder, llm))
}

async fn cmd_add(args: &[String]) -> anyhow::Result<()> {
    let Some(text) = args.first() else {
        anyhow::bail!("add requires text");
    };
    let source = flag(args, "--source").unwrap_or_else(|| "message".into());
    let (store, embedder, llm) = open_graph().await?;
    let ingestor = Ingestor::new(store, embedder, llm);
    let r = ingestor.add_episode(text, &source, None, None).await?;
    println!(
        "episode={} nodes={} edges={} expired={}",
        r.episode_id, r.node_count, r.edge_count, r.expired_edge_ids.len()
    );
    Ok(())
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
