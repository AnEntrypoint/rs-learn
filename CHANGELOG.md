# Changelog

## [Unreleased]

- graph: absorb remaining bungraph behaviour gaps.
  - Per-scope unified search config via `SearchAllConfig { nodes, edges, episodes, communities }`; each scope carries its own `reranker`, `limit`, `use_vector`, `use_fts`, `mmr_lambda`. Old `Searcher::search_all(&SearchConfig)` still works (back-compat).
  - `SearchConfig` gains `use_vector`, `use_fts`, `center_node_ids` fields; disabling both fails loud instead of silently returning empty.
  - Reranker center auto-resolution: when `NodeDistance` reranker is selected without `center_node_id`/`center_node_ids`, searcher seeds top-3 centers from a vector query on nodes (matches bungraph search.js behaviour).
  - `Store::graph_walk(start_ids, depth, group_id)` and `Store::edges_between(src, tgt, group_id, as_of)` — recursive-CTE neighborhood queries, bitemporal when `as_of` is set.
  - Bulk and single-episode ingest now filter the prior-episode context window by `reference_time` (via `load_previous_episodes_before`) — re-ingesting historical episodes no longer leaks post-reference context back into extraction.
  - MCP `search` tool and HTTP `POST /search` (with `scope=all`) accept optional `{nodes, edges, episodes, communities}` objects carrying `reranker`, `limit`, `use_vector`, `use_fts`, `mmr_lambda`; reranker names `rrf|mmr|node_distance|episode_mentions|cross_encoder`. Unknown reranker string returns a structured error.
  - Shared ISO datetime helpers extracted to `src/graph/time.rs` (deduplicated from `http.rs` + `ingest.rs`).
- graph: unified `search` surface (MCP tool + HTTP `/search` with `scope=all`) fans out across nodes, edges, episodes, communities in parallel via `Searcher::search_all` — closes last bungraph mcp.js parity gap.
- Added `AgentBackend` trait and `ClaudeCliClient` (src/backend/claude_cli.rs): run `claude -p --output-format json` as an alternative to ACP stdio agents. Selection via `RS_LEARN_BACKEND=claude-cli|acp` or auto (ACP if `RS_LEARN_ACP_COMMAND` set, else claude-cli). Configurable model (`RS_LEARN_CLAUDE_MODEL`, default `haiku`), plugin dir (`RS_LEARN_CLAUDE_PLUGIN_DIR`), extra args (`RS_LEARN_CLAUDE_ARGS`). Prompt delivered via stdin to bypass Windows `.cmd` argv parsing.
- Orchestrator and BackgroundLoop now hold `Arc<dyn AgentBackend>` instead of `Arc<AcpClient>`. `AcpClient` still exported for direct use; `AgentBackend` + `ClaudeCliClient` added to public API.
- `/debug/orchestrator` now reports selected backend name.

## [Unreleased] - Full Rust port

- Deleted all JavaScript sources: `package.json`, `package-lock.json`, `node_modules/`, `test.js`, `validate.js`, every `src/*.js`, and `src/rs-learn/*.js`.
- All subsystems (store, embeddings, memory, attention, router, orchestrator, acp-client, export, federated, observability, rs-search bridge, instant/background/deep loops, reasoning bank) now implemented in Rust under `src/`.
- Integration coverage consolidated into `tests/integration.rs`.
- Rewrote `.github/workflows/test.yml` as a cargo build + test matrix over 6 targets (linux/windows/macos × x86_64 + aarch64), with stable + nightly toolchain install and cargo registry/target caching.
- Rewrote `.github/workflows/release.yml` to build per-target release binaries on `v*` tag push, upload them to the GitHub Release via `gh`, and conditionally `cargo publish` when `CARGO_REGISTRY_TOKEN` is present.
- README updated to Rust-only surface: `cargo install`, Rust usage example, ACP spawn env vars, and full `/debug/*` endpoint inventory.

Known upstream note: local `cargo build --release` requires rustc >= 1.88 because transitive deps (`darling` 0.23, `icu_*` 2.2, `serde_with` 3.18, `home` 0.5.12) bumped their MSRV. CI installs current stable via `rustup toolchain install stable`, so matrix builds resolve the toolchain automatically. Developers on older rust should run `rustup update stable`.

## 0.0.1 - initial scaffold
- package.json (ESM, node>=20) with @libsql/client, @xenova/transformers, @agentclientprotocol/sdk
- src/ tree with placeholder ESM modules for every planned subsystem
- root test.js stub
- .gitignore, README.md

- memory: HNSW-equivalent over libsql vector+graph. createMemory(store) exposes add({id?,payload,embedding,level?}), search(embedding,{ef,k,filter}) and expand(nodeId,depth). Levels sampled via exponential distribution (M=32, mL=1/ln(M)); bidirectional hnsw-neighbor-L{l} edges written in a single db.batch per add. debug-registry 'memory' publishes {addCount,searchCount,avgSearchMs,expandCount,M,mL}. Verified on 300 synthetic 384d vectors: recall@10=1.0, search p50 30ms p99 52ms, expand returns connected subgraph.
- embeddings: MiniLM-L6 embedder with sha256-keyed LRU cache (2048), 512-token truncation, mean pool + L2 norm, debug-registry {cacheSize,hits,misses,modelLoaded}
- store: libsql schema (episodes, nodes, edges, trajectories, patterns, reasoning_bank, router_weights, ewc_fisher, preference_pairs, sessions) with F32_BLOB(384) embeddings, libsql_vector_idx on all embedding cols, FTS5 virtual tables + triggers for nodes/edges/episodes/reasoning, idempotent migrations via schema_version, openStore() facade (insert/upsert/vectorTopK/ftsSearch/router/fisher/preference), debug-registry 'store:<path>' with per-table counts
