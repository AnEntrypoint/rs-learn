use rs_learn::backend;
use rs_learn::embeddings::Embedder;
use rs_learn::graph::http::HttpState;
use rs_learn::graph::ingest::Ingestor;
use rs_learn::graph::llm::LlmJson;
use rs_learn::graph::mcp::McpServer;
use rs_learn::graph::search::{SearchConfig, Searcher};
use rs_learn::orchestrator::Orchestrator;
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
        "ready" | "" => {
            let _ = Orchestrator::new_default().await?;
            println!("rs-learn ready");
        }
        "add" => cmd_add(&args[1..]).await?,
        "search" => cmd_search(&args[1..]).await?,
        "clear" => cmd_clear().await?,
        "build-communities" => cmd_build_communities().await?,
        "serve" => cmd_serve(&args[1..]).await?,
        "mcp" => cmd_mcp().await?,
        "help" | "-h" | "--help" => print_help(),
        other => {
            eprintln!("unknown subcommand '{other}'");
            print_help();
            std::process::exit(2);
        }
    }
    Ok(())
}

fn print_help() {
    eprintln!("rs-learn <subcommand> [args]");
    eprintln!("  ready                       boot orchestrator and exit");
    eprintln!("  add <text> [--source S]     ingest an episode");
    eprintln!("  search <query> [--scope S] [--limit N]");
    eprintln!("  clear                       drop all graph data");
    eprintln!("  build-communities           run label propagation + summarize");
    eprintln!("  serve [--port N]            start HTTP REST server");
    eprintln!("  mcp                         start MCP stdio server");
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
