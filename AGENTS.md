rs-learn is a wasm32-wasip1 cdylib. Storage and embeddings live in the host (rs-plugkit); this crate exposes kv + vector recall verbs via the host_* extern surface in `crates/wasm/src/wasm_host.rs`.

`EMBED_DIM = 384` (MiniLM-L6-v2, F16). Schema is `F32_BLOB(384)` with `libsql_vector_idx` on the host side; dimension changes auto-migrate the on-disk schema.

`.gm/rs-learn.db` is tracked. The gitignore parent-re-include caveat applies: `.gm/` is never bulk-ignored; per-file ignores live between the `# >>> gm managed` markers. `.code-search/` and root `rs-learn.db` are never ignored.

Memory writes go through the orchestrator's `memorize` verb, not direct kv_put. The classifier in rs-plugkit rejects changelog-shaped facts from AGENTS.md ingestion; the rs-learn store accepts them when written directly.

`is_derivable_state` filter (rs-plugkit) rejects hex hashes, FIXED markers, historical framing, changelog/commit/blame refs before they reach the embedder.

Auto-recall on prompt-submit calls the rs-learn `Searcher` directly via the shared tokio Runtime in rs-plugkit, not over HTTP.

Per-discipline isolation: each discipline owns `<project>/.gm/disciplines/<name>/rs-learn.db`. Default discipline writes go to `.gm/rs-learn.db`. Cross-discipline reads are forbidden when a `@<name>` sigil is present.

Comments in source are forbidden — no `//`, `/* */`, or doc-comments in shipped code.

Push to trigger CI; never run `cargo build` or `cargo update` locally.

@.gm/next-step.md
