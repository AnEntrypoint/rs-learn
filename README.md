# rs-learn

A continual-learning orchestrator wrapped around any [Agent Client Protocol](https://agentclientprotocol.com) stdio agent. Gives an otherwise-stateless ACP agent persistent memory, adaptive routing, and three background learning loops that compress trajectories into reusable strategies and exportable LoRA / DPO training artefacts.

**Pure Rust.** Single binary, no Node, no npm.

Layers:

- **libsql** vector + graph + FTS5 — single-file memory store, zero external services
- **HNSW-equivalent** ANN over libsql `vector_top_k`, M=32 level sampling
- **8-head graph attention** with edge features (relation one-hot + recency decay + weight)
- **FastGRNN router** — sparse (90%) + low-rank (rank-8) matrices, softmax over ACP targets + context bucket + temperature + topP + confidence
- **Three rs-learn loops**:
  - **Instant (per-request)** — trajectory capture + rank-2 MicroLoRA Hebbian adapter
  - **Background (hourly)** — k-means++ pattern extraction + BaseLoRA via router retrain + ACP-summarized reasoning bank
  - **Deep (weekly)** — EWC++ consolidation with online Fisher information (EMA decay 0.999)
- **Federated** — EphemeralAgent + FederatedCoordinator (quality-filtered 50k pool)
- **Observability** — HTTP `/debug/<subsystem>` per subsystem, structured tracing
- **Exports** — safetensors router weights, patterns.jsonl, preferences.jsonl (DPO), HF push

## Install

```bash
cargo install rs-learn
```

Or clone and build:

```bash
git clone https://github.com/AnEntrypoint/rs-learn
cd rs-learn
cargo build --release
```

Binaries produced: `rs-learn`, `rs-learn-validate`.

## Usage

```rust
use rs_learn::{openStore, Orchestrator, OrchestratorConfig, ACPClient};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let store = openStore(".rs-learn.db").await?;
    let acp = ACPClient::from_env()?;
    let orch = Orchestrator::builder()
        .store(store)
        .acp(acp)
        .build()
        .await?;

    let r = orch.query("What's Alice's role?", Default::default()).await?;
    println!("{} {:?}", r.text, r.stage_breakdown);
    orch.feedback(&r.request_id, 0.9, true).await?;
    Ok(())
}
```

## ACP spawn

Set `RS_LEARN_ACP_COMMAND` to the ACP stdio command of any compatible agent:

```bash
# opencode
RS_LEARN_ACP_COMMAND="opencode acp" rs-learn

# kilo (requires prior `kilo auth login`)
RS_LEARN_ACP_COMMAND="kilo acp" rs-learn

# any ACP stdio agent
RS_LEARN_ACP_COMMAND="claude-cli acp" rs-learn
RS_LEARN_ACP_COMMAND="gemini-cli acp" rs-learn
RS_LEARN_ACP_COMMAND="codex acp" rs-learn
```

Live E2E mode requires `RS_LEARN_ACP_LIVE=1`; otherwise the orchestrator uses the stub handler for tests.

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `RS_LEARN_DB_PATH` | `.rs-learn.db` | libsql file path |
| `RS_LEARN_ACP_COMMAND` | (required for live) | ACP stdio command, e.g. `opencode acp` |
| `RS_LEARN_ACP_ARGS` | — | extra args (whitespace-separated) |
| `RS_LEARN_ACP_LIVE` | `0` | `1` = spawn real ACP subprocess, `0` = stub |
| `RS_LEARN_EWC_LAMBDA` | `2000` | EWC++ regularization strength (100–15000) |
| `RS_LEARN_LLM_TIMEOUT_MS` | `120000` | per-turn timeout |
| `RS_LEARN_DEBUG_ACP` | — | log ACP stderr |
| `HF_TOKEN` | — | Hugging Face token for `push_to_hugging_face` |

## Observability

HTTP server binds at `127.0.0.1:7878` on startup.

```bash
curl http://127.0.0.1:7878/debug              # all subsystems
curl http://127.0.0.1:7878/debug/acp          # ACP state
curl http://127.0.0.1:7878/debug/router       # router state
curl http://127.0.0.1:7878/debug/memory       # memory stats
curl http://127.0.0.1:7878/debug/learn-bg     # background loop
curl http://127.0.0.1:7878/debug/learn-deep   # deep loop
curl http://127.0.0.1:7878/debug/store        # store table counts
curl http://127.0.0.1:7878/debug/attention    # attention head stats
curl http://127.0.0.1:7878/debug/reasoning    # reasoning bank
curl http://127.0.0.1:7878/debug/federated    # federated pool
```

Every subsystem (store, memory, router, attention, instant/bg/deep loops, reasoning-bank, federated, acp, orchestrator, rs-search) registers into the debug registry on construction.

## Tests

```bash
cargo test --release
```

Integration suite lives in `tests/integration.rs` and exercises the full pipeline against an in-memory libsql DB.

## CI / Release

- `.github/workflows/test.yml` — cargo build + test matrix across 6 targets (linux/windows/macos × x86_64 + aarch64).
- `.github/workflows/release.yml` — on tag `v*`, builds release binaries per target, attaches them to the GitHub Release, and (if `CARGO_REGISTRY_TOKEN` secret is set) publishes to crates.io.

## License

MIT

## Claude Code plugin

This repo is also a Claude Code plugin. Load directly:

```bash
claude --plugin-dir /path/to/rs-learn -p "your prompt"
```

Or via marketplace:

```bash
claude plugin marketplace add AnEntrypoint/rs-learn
claude plugin install rs-learn@rs-learn
```

Provides:
- **Skill**: `rs-learn` — continual-learning orchestrator guidance (skills/rs-learn/)
- **Commands**: `/rs-learn-status`, `/rs-learn-query`, `/rs-learn-feedback`

Requires the `rs-learn` binary on PATH (`cargo install rs-learn` or built from this repo) and `RS_LEARN_ACP_COMMAND` pointing to an ACP stdio agent.
