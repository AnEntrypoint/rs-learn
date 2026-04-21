use rs_learn::orchestrator::Orchestrator;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).init();
    let _ = Orchestrator::new_default().await?;
    println!("rs-learn ready");
    Ok(())
}
