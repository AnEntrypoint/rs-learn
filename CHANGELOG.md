## [Unreleased]

### Added
- **Instant adapter LR floor** — `RS_LEARN_LR_MIN` (default 1e-3, clamped to `(0, LR0]`) prevents hebbian learning rate from decaying toward zero; `hebbian_update` now does `lr = max(lr * DECAY, lr_min)`. Test `lr_respects_floor` asserts invariant after 2000 updates.
- **Router epsilon-greedy exploration** — `RS_LEARN_ROUTER_EPSILON=0.0..0.5` (default 0.0) samples non-argmax targets to break greedy lock-in. `RouteSnapshot.exploration` exposes per-query firing. Seeded via `RS_LEARN_ROUTER_SEED` for test determinism.
- **Attention feedthrough** — orchestrator now averages per-head attention weights across the retrieved subgraph and appends a "Top-attended context" block (top-3 node ids + weights) to the ACP system prompt. Previously `attend()` output was discarded — pure silent no-op.
- **Embed-cache size telemetry** — `/observability` `embed_cache` module now reports `size` (moka entry count) alongside hits/misses/hit_rate.
- **DeepLoop full window telemetry** — `/observability` `deep` now reports `boundary_fires`, `window_mean`, `window_stddev`, `threshold` (BOUNDARY_Z=2.5), `samples_in_window` — previously only `loss_ring` + `boundaries_detected` visible.

### Changed
- **Remove BC wrapper** — `InstantLoop::record_trajectory_full` was a rename shim for `record_trajectory`. Deleted the thin wrapper; single entry point `record_trajectory(session, embedding, model, response, query, implicit_quality, latency_ms)`. All 4 call sites updated.
- **CLAUDE.md corrected** — "dead modules" section was stale: DeepLoop is wired into `Orchestrator::feedback`, FederatedCoordinator already deleted. Replaced with current loop-wiring truth.

### Added (prior)
- **Router multi-epoch shuffled training** — `Router::train` now runs 2 epochs (env `RS_LEARN_ROUTER_EPOCHS=1..8`) with per-epoch Fisher-Yates shuffle, extracting more signal per background tick without growing batch size.
- **InstantLoop replay buffer (64-slot ring)** — each positive/negative feedback appends `(embedding, target_idx, scale)` and replays one random prior sample at half scale to fight catastrophic forgetting across regime shifts.
- **Feedback-expiry observability** — `feedback_expired` counter in `/observability` tracks pending trajectories GC'd past `FEEDBACK_WINDOW` (60s); previously silent signal loss.
- **Background trajectory limit env-tunable** — `RS_LEARN_BG_LIMIT=100..100000` (default 1000) overrides FIFO window used for clustering/training.

# Changelog

## [Unreleased]

- learn: negative-feedback now trains. Previously low-quality trajectories were silently dropped in three places (BackgroundLoop quality gate, Router.train quality gate, InstantLoop.feedback gate), so the system only learned *what to do* and never *what not to do* — half the learning signal wasted.
  - `Router::train`: `quality >= 0.7` still trains toward chosen target; `quality <= 0.3` now trains *away* (gradient sign reversed, strength `0.3 - quality`). Context-bucket head only trains on positive examples since token count is ground truth regardless of response quality.
  - `BackgroundLoop::run_once`: passes through both ends of the quality distribution (`>= 0.7` or `<= 0.3`); middle band (0.3..0.7) still filtered as noisy.
  - `InstantLoop::feedback`: negative hebbian update (`scale = -(1-quality)`) when `quality < 0.3` pushes the adapter away from the wrong target immediately, instead of waiting for the background consolidation pass.

- chore: drop vestigial rs-search binary-fetch workflow. rs-search has been consumed as an SDK crate (`rs_search::run_search`) via `src/rs_search_bridge.rs` since integration; no subprocess path exists. Removed `.github/workflows/fetch-rs-search.yml` which pulled prebuilt binaries from the rs-search release page weekly.

- perf: attention + router-train simd/rayon.
  - `attention::attend`: K/V projections for n subgraph nodes now run in parallel via `rayon::par_chunks_mut`. Previously serial n×2 matvecs (768→768). Weighted V accumulation uses `simd::axpy` instead of scalar per-dim loops.
  - `router::train`: per-class SGD update replaced scalar `heads[off+d] -= lr*err*h[d]` loop with `simd::axpy(-lr*err, h, row)`. Applies to both model and context-bucket heads.
  - Net: attention per-query scales with cores on subgraph size; router batch training shrinks inner loop to a single SIMD kernel call per head row.

- perf: k-means rewrite — SIMD cosine + rayon-parallel + pre-normalized vectors.
  - `cosine_dist` now calls `simd::dot` instead of manual scalar loops.
  - Vectors are L2-normalized once up front; inner loop uses `1 - dot(unit_a, unit_b)` (no per-call norm recompute). Kills the O(n·k·iter) norm redundancy that dominated BackgroundLoop CPU.
  - k-means++ seeding: squared-distance update is parallel via `rayon::par_iter_mut`. With n=1000 points, the per-centroid O(n·dim) scan now scales across cores.
  - Assignment pass (O(n·k·dim)) is `par_iter` — each point picks its cluster in parallel.
  - Centroid accumulation uses `fold/reduce` with `simd::axpy` for per-point sums; O(n·dim) add pass is now SIMD + parallel. Centroids re-normalized per iter so cosine_unit stays valid.
  - Added `rayon = "1"` dependency.
  - End-to-end: a full BackgroundLoop run on 1000 trajectories × 768-dim × 25 iterations previously spent ~6 GFLOPs serial scalar on distance computation; now parallel SIMD.

- perf: simd router forward, parallel cluster summaries, grounding-gated quality signal.
  - `router::forward()` now uses `simd::matvec` + `simd::dot` for all matrix ops instead of manual nested loops.
  - `BackgroundLoop::run_once` parallelizes all per-cluster LLM summarizations via `futures::future::join_all`. Previously serialized: N clusters = N sequential LLM calls. Now concurrent.
  - `implicit_quality_from`: grounding weight 0.40→0.50, latency 0.45→0.35. Hard gate: `grounding < 0.15` halves quality — prevents fast-ungrounded responses polluting router training.
  - `EmbeddingCache::embed`: fast-path `get().await` before `get_with` avoids TOCTOU double-lock on hits.

- cli: add `query`, `feedback`, `debug`, `--version`. CLI parity with library API.
  - `rs-learn query <text>` — runs one full orchestrator pass, prints JSON with request_id, confidence, latency, routing, stage breakdown. No Rust required to exercise the learning loop end-to-end.
  - `rs-learn feedback <request_id> <quality 0..1> [signal]` — records explicit feedback.
  - `rs-learn debug [subsystem]` — dumps `/debug` snapshot (all or one) without starting the HTTP server.
  - `rs-learn --version` / `-V` / `version` — prints package version.
  - `rs-learn start` alias for `ready` (more intuitive).
  - `help` now goes to stdout (was stderr, violated Unix convention); stderr still used for unknown-subcommand error path.

- learn: adaptive k-means k, pattern/reasoning timestamps.
  - `BackgroundLoop::run_once` capped `k = min(100, max(2, n_points / 4))`. Previously `k=100` hardcoded — 8 trajectories produced 8 singleton clusters, wasting compute and producing useless per-point "patterns".
  - `upsert_pattern` and `insert_reasoning` now carry `created_at = now_ms()`. Previously `None` — silently zero-timestamped, breaking recency ordering downstream.

- learn: tighten implicit quality signal, expose reset counter, fix stale README, add held-out routing test.
  - **Implicit quality replaces `response_len` with retrieval grounding**. The old weighting (`0.4*latency + 0.3*length + 0.3*confidence`) rewarded verbose responses — a concise correct answer scored lower than a rambling one. New weighting is `0.45*latency + 0.40*grounding + 0.15*confidence`, where `grounding = top-neighbor similarity score` (the actual retrieval quality for the query). Confidence was downweighted because softmax-of-logits fed back as quality creates an overconfidence loop. Length is gone entirely. Signature changed: `implicit_quality_from(latency_ms, grounding: f32, confidence: f32)`.
  - **InstantLoop `resets_performed` counter** — `reset_adapter()` used to wipe rank-2 Hebbian state on every boundary with no observable trace. Now surfaced at `/debug/instant.resets_performed`. `DeepLoop /debug/deep.boundaries_detected` already existed; the two together fully trace boundary→reset.
  - **README `feedback` example was wrong** — compiled against a removed `(id, f64, bool)` signature. Updated to use `FeedbackPayload { quality, signal }` matching current API.
  - **Held-out routing accuracy test** — `router_learns_held_out_routing_from_trajectories`: seeds 200 training samples from two jittered embedding anchors, trains via `Router::train`, asserts that held-out queries from each region route to the correct target. First regression test that validates *learning outcome*, not just mechanics.
- tests: 3 replace/rename (`implicit_quality_rewards_fast_grounded_responses`, `implicit_quality_length_does_not_affect_score`, `implicit_quality_clamps_to_unit`) + 1 new integration test. 30 lib + 17 integration green locally.

- learn: close the remaining gaps from the prior audit — implicit quality, deep-loop wiring, federated prune.
  - **Implicit quality signal**: `Orchestrator::query` now derives a per-request quality from latency, response length, and route confidence and stores it on the trajectory (previously `quality=None` until explicit `feedback()`). Exposed as `implicit_quality_from(latency_ms, response_len, confidence)`. Fast-confident replies score high, slow-short-unconfident ones score low. Without this, `BackgroundLoop.run_once` no-oped on every read-only deployment because nothing ever reached the 0.7 training threshold.
  - **Trajectory query + latency persisted**: `InstantLoop::record_trajectory_full` (new) writes `query`, `quality`, and `latency_ms` into the trajectory row. Old `record_trajectory` delegates to it with nulls to preserve the existing call sites / tests. `Store::list_recent_trajectories_with_embeddings` now selects `latency_ms` too (was silently `None`).
  - **Deep loop wired into feedback path**: `Orchestrator` now owns `Arc<Mutex<DeepLoop>>`. On every `feedback()`, the orchestrator feeds `1.0 - quality` as loss to `DeepLoop::record_loss`; when the z-score boundary fires, it calls `InstantLoop::reset_adapter()` so the rank-2 Hebbian adapter does not carry assumptions across a distribution shift. No new config — on by default.
  - **Federated module deleted**: `src/federated.rs`, `EphemeralAgent`, `FederatedCoordinator`, and `Trajectory` were never constructed outside their own tests; no orchestrator consumer exists and no downstream relies on them. Removed module, lib.rs re-export, integration test section 8, and doc references. If multi-agent trajectory sharing is wanted in future, it should be designed against a real caller.
- tests: 3 new regression tests — `record_trajectory_full_persists_query_and_quality`, `implicit_quality_rewards_fast_confident_responses`, `implicit_quality_clamps_to_unit`. All 29 lib + 16 integration + 23 graph + 3 spine tests green locally.

- learn: activate the 3-loop architecture end-to-end and close remaining correctness gaps.
  - **BackgroundLoop now runs in production**: previously constructed nowhere in `Orchestrator`; pattern extraction, reasoning bank writes, and router batch training never fired outside tests. `Orchestrator::new_default` now owns `Arc<BackgroundLoop>` and, when `RS_LEARN_BG_INTERVAL_SEC=N` (N>0), spawns a scheduler task that runs `run_once` every N seconds. `Drop(Orchestrator)` aborts the task. Unset env = opt-out (default).
  - **Router persists after batch training**: `BackgroundLoop::run_once` now calls `router.save()` when `train()` applied >0 samples. Previously batch learning was lost on restart — heads mutated in memory but never flushed.
  - **Context-bucket head learns real token counts**: `Router::train` used `bucket_for_tokens(embedding.len())` — always `IN=768` — so every sample trained the same ctx bucket. `TrainSample` now carries `estimated_tokens: u64` (derived from trajectory query length in `BackgroundLoop`). Ctx head actually discriminates small vs large context now.
  - **Hebbian adapter bounded**: `InstantLoop::hebbian_update` now clamps `||a, b||` to `MAX_ADAPTER_NORM=5.0`. Long sessions of positive feedback previously drove the adapter toward infinity, collapsing softmax routing to a single target.
  - **InstantLoop observability reads live state**: `adapter_norm` and `pending_count` in the registered debug snapshot were captured as compile-time constants at `register()` time and never updated. Now backed by `Arc<AtomicU64>` fields kept in sync with every `record_trajectory` / `feedback` / `hebbian_update` / `reset_adapter`.
- tests: 2 new regression tests — `router_ctx_bucket_trains_from_estimated_tokens` (asserts 200k-token samples do not degrade the bucket learned from 500-token samples) and `instant_adapter_bounded_under_runaway_feedback` (500 positive feedbacks stay under norm cap).

- learn: critical correctness fixes across router, instant loop, background loop, reasoning bank, and deep loop.
  - **Router training is now real**: `Router::train` previously did unbounded Hebbian push on the chosen target only (no loss, no error signal), biasing the argmax toward whichever target was chosen most often regardless of embedding. Replaced with softmax cross-entropy gradient on the model head plus the context-bucket head (previously never trained). Learning rate scales with trajectory quality. Weights now stay bounded under training.
  - **Instant adapter now consumed**: orchestrator previously called `router.route()` which never merged the rank-2 Hebbian adapter into the routing logits. Feedback was silent dead code. Added `Router::route_with_adapter` and wired `InstantLoop::apply_adapter_raw` through the orchestrator hot path. Positive feedback on target T now actually steers subsequent routes toward T.
  - **Background cluster summaries use real query text**: `summarize_cluster` previously sent opaque trajectory UUIDs to the LLM (`"summarize these trajectories: t1, t2, t3"`), producing unusable summaries. Now loads `query` column into cluster meta and prompts on actual query strings. Fallback summary derives from query prefix instead of `auto-cluster n=N`.
  - **Reasoning bank retrieval hybridized**: `ReasoningBank::retrieve_for_query` was FTS-only, missing semantic matches without lexical overlap. Added `with_embedder` constructor; retrieval now fuses FTS + vector route (via `patterns.centroid` + `pattern_id` bridge — no schema change) via RRF.
  - **Deep-loop boundary detection uses z-score**: `record_loss` previously flagged boundaries on absolute delta 0.5, brittle on unknown-scale losses. Now computes z-score over prior ring (spike not self-included) with min-stddev floor, threshold 2.5σ. Scale-invariant.
  - **background.rs split** at 312 → 200 LOC hygiene: `src/learn/background/` module with `kmeans.rs` (kmeans++ + cosine_dist + mulberry32) and `mod.rs` (BackgroundLoop).
- tests: 4 new regression tests — `train_softmax_shifts_prediction_toward_chosen`, `adapter_shifts_router_decision_after_feedback`, `summarize_prompt_uses_query_content_not_ids`, `hybrid_retrieval_bridges_via_pattern_centroid`. Existing `record_loss_triggers_boundary` updated for z-score semantics. 26+ lib tests + 23 graph + 15 integration green locally.

- graph: final bungraph-parity pass — search filters, custom type schemas, community dirty-skip, saga auto-summary.
  - `SearchFilters { node_labels, edge_types, created_at_min/max, valid_at_min/max }` on `SearchConfig`. Applied post-`rrf_fuse` pre-rerank in `Searcher::search_table`. HTTP `POST /search` `ScopeBody.filters` + MCP `search` per-scope `filters` object route through. Empty filters = no-op back-compat.
  - Custom extraction schemas via env + per-request override. `EntityOps::with_types(_, _, _, Option<String>)` reads `RS_LEARN_ENTITY_TYPES_JSON` when no arg. `EdgeOps::with_types(_, _, _, Option<Value>)` reads `RS_LEARN_EDGE_TYPES_JSON`. `Ingestor::with_types(store, embedder, llm, entity_types, edge_types)` bundles both. `Ingestor::add_episode_with(..., entity_types_override, edge_types_override)` allows per-call override. HTTP `POST /messages` body fields `entity_types` (string JSON) + `edge_types` (Value) and MCP `add_episode` args pass through.
  - `CommunityOps::build_communities_if_dirty()` short-circuits when `MAX(nodes.created_at) <= MAX(communities.created_at)` — zero new schema, uses existing columns. HTTP `POST /build-communities { force: bool }` and MCP `build_communities { force }` default to dirty-skip; `force=true` forces full rebuild.
  - `SagaOps::add_episode_to_saga(self: &Arc<Self>, ...)` auto-spawns `summarize_saga` when `(seq+1) % RS_LEARN_SAGA_SUMMARY_EVERY == 0` (default 10). Non-blocking via `tokio::spawn`. `SagaOps::with_threshold(store, llm, n)` lets callers override.
  - Downgrade: local ONNX cross-encoder (bge-reranker-v2-m3) explicitly out of scope — existing `Reranker::CrossEncoder` (LLM-backed in `Searcher::cross_encoder_rerank`) already produces quality reranks. ONNX would require 500MB model download + `ort` crate + `tokenizers` + per-platform runtime setup; complexity greatly exceeds benefit when the LLM path works. Revisit as separate infra scope.
- tests: 4 new regression tests — `search_filters_by_node_label_and_date_range`, `custom_entity_types_reach_prompt`, `communities_skip_when_not_dirty`, `saga_auto_summary_fires_at_threshold`. 23 graph + 15 integration + 3 spine = 41 tests green.

- graph: full multi-tenancy + boundary hardening pass (closes prior bungraph→rs-learn gaps the previous "absorption" missed).
  - `NodeRow`, `EdgeRow`, `EpisodeRow`, `Entity`, `ResolvedEdge` carry `group_id: Option<String>`. `Store::insert_node` / `insert_edge` / `insert_episode` now thread it into the column list (was always defaulting to `'default'` regardless of caller).
  - `Ingestor::add_episode(content, source, reference_time, group_id)`, `add_triplet(..., group_id)`, `add_episode_bulk(items, group_id)` — `BulkEpisode` also gains optional per-item `group_id` (item override beats batch-level).
  - HTTP `/messages`, `/triplet`, `/entity-node`, `/clear` and MCP `add_episode` / `add_episode_bulk` / `add_triplet` / `clear_graph` accept `group_id` (or `group_ids` for clear) and route it through ingest. Previously parsed and dropped — silent multi-tenant collapse fixed.
  - `Ingestor::clear_graph(group_ids: Option<&[String]>)` — when `Some`, scoped `DELETE WHERE group_id IN (...)` across `edges/nodes/episodes/communities`; `None` wipes globals too.
- graph: validation module at boundaries (`src/graph/validation.rs`).
  - `validate_group_id` (regex `[A-Za-z0-9_.-]`, ≤128 chars), `validate_content` (≤200_000 bytes), `validate_limit` (1..=1000), `validate_iso_date`, `validate_reranker`. Wired into MCP `add_episode` / `add_episode_bulk` / `add_triplet` / `clear_graph` and HTTP `/messages` / `/triplet` / `/entity-node` / `/clear`. Errors return structured 400 / JSON-RPC error.
- graph: runtime summary truncation (`src/graph/text.rs`).
  - `truncate_at_sentence(s, max)` clamps at last `.!?` boundary within `max` chars, falls back to char-cut if no sentence end. Applied to community summaries (`CommunityOps::insert_community`) and saga summaries (`SagaOps::summarize_saga`).
  - `MAX_SUMMARY_CHARS` bumped 500 → 1000 (parity with bungraph).
- graph: observability via `src/graph/metrics.rs`.
  - Atomic counters: `episodes_ingested`, `nodes_upserted`, `edges_upserted`, `search_calls`, `search_total_ms`, `llm_calls`, `llm_total_ms`, `dedup_candidates_seen`. Recorded in `Ingestor::add_episode`, `Searcher::search_table`, `LlmJson::call`. Registered as `/debug/graph` provider via `metrics::register()` on `cmd_serve` + `cmd_mcp` startup; also surfaced via MCP `debug_state`.
- tests: 6 new regression tests — `group_id_persisted_through_ingest`, `clear_graph_per_group_isolates`, `validation_rejects_bad_group_id`, `truncate_at_sentence_clamps_to_boundary`, `metrics_counters_advance_after_ingest`, plus updated `group_id_filter_cascade_delete`. 19 graph + 15 integration + 3 spine tests all green.

- graph: absorb remaining bungraph behaviour gaps.
  - Per-scope unified search config via `SearchAllConfig { nodes, edges, episodes, communities }`; each scope carries its own `reranker`, `limit`, `use_vector`, `use_fts`, `mmr_lambda`. Old `Searcher::search_all(&SearchConfig)` still works (back-compat).
  - `SearchConfig` gains `use_vector`, `use_fts`, `center_node_ids` fields; disabling both fails loud instead of silently returning empty.
  - Reranker center auto-resolution: when `NodeDistance` reranker is selected without `center_node_id`/`center_node_ids`, searcher seeds top-3 centers from a vector query on nodes (matches bungraph search.js behaviour).
  - `Store::graph_walk(start_ids, depth, group_id)` and `Store::edges_between(src, tgt, group_id, as_of)` — recursive-CTE neighborhood queries, bitemporal when `as_of` is set.
  - Bulk and single-episode ingest now filter the prior-episode context window by `reference_time` (via `load_previous_episodes_before`) — re-ingesting historical episodes no longer leaks post-reference context back into extraction.
  - MCP `search` tool and HTTP `POST /search` (with `scope=all`) accept optional `{nodes, edges, episodes, communities}` objects carrying `reranker`, `limit`, `use_vector`, `use_fts`, `mmr_lambda`; reranker names `rrf|mmr|node_distance|episode_mentions|cross_encoder`. Unknown reranker string returns a structured error.
  - Shared ISO datetime helpers extracted to `src/graph/time.rs` (deduplicated from `http.rs` + `ingest.rs`).
  - Dedup candidate limit bumped to match bungraph: `NODE_DEDUP_CANDIDATE_LIMIT=15` (was inline `5`), `MAX_NODES=30` cap applied to extraction.
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
