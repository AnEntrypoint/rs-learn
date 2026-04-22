---
name: rs-learn
description: Continual-learning memory and adaptive routing. Use when the user wants persistent memory across sessions, pattern extraction from trajectories, FastGRNN routing of ACP agents, or LoRA/DPO export from live interactions. Wraps the rs-learn Rust binary (libsql + HNSW + 8-head graph attention + three learning loops). Requires rs-learn binary on PATH or RS_LEARN_BIN env var.
---

# rs-learn — Continual-Learning Orchestrator

Persistent memory, adaptive routing, background pattern extraction for any ACP stdio agent.

## What it gives you

- **libsql memory** — single-file vector + graph + FTS5 store
- **HNSW retrieval** — ANN over libsql vector_top_k
- **8-head graph attention** — edge features (relation + recency decay + weight)
- **FastGRNN router** — sparse 90% + rank-8 over ACP targets, context, temperature, topP, confidence
- **Three learning loops** — instant (per-request Hebbian MicroLoRA), background (hourly k-means + BaseLoRA), deep (weekly EWC++ consolidation)
- **Exports** — safetensors router weights, patterns.jsonl, DPO preferences.jsonl

## Environment

Set `RS_LEARN_ACP_COMMAND` to any ACP stdio agent command, e.g.:

```
RS_LEARN_ACP_COMMAND="opencode acp"
RS_LEARN_ACP_COMMAND="kilo acp"
RS_LEARN_ACP_COMMAND="claude-cli acp"
```

Set `RS_LEARN_DB_PATH` to the libsql file (default `.rs-learn.db`).

## Binary discovery

Plugin looks for rs-learn in this order:
1. `RS_LEARN_BIN` env var (absolute path)
2. `rs-learn` on PATH
3. `~/.cargo/bin/rs-learn`

Install via `cargo install rs-learn` or build from https://github.com/AnEntrypoint/rs-learn.

## Usage

See `/rs-learn-query`, `/rs-learn-feedback`, `/rs-learn-status` slash commands.

For library usage (embed the orchestrator in your own Rust crate):

```toml
[dependencies]
rs-learn = { git = "https://github.com/AnEntrypoint/rs-learn", branch = "main" }
```

```rust
use rs_learn::{Orchestrator, Store};
let store = Store::open(".rs-learn.db").await?;
let orch = Orchestrator::builder().store(store).build().await?;
let r = orch.query("What's Alice's role?", Default::default()).await?;
```
