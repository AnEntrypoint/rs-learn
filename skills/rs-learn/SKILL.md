---
name: rs-learn
description: >
  Persistent memory and recall, wasm-only. Compiled to wasm32 and consumed by
  rs-plugkit; no native CLI, no MCP server, no HTTP. Activate when the user
  wants to remember facts across sessions, recall prior context mid-task, or
  wire an agent to a queryable knowledge store. Dispatched via plugkit spool
  verbs `recall` and `memorize`. Per-discipline isolation under
  `<project>/.gm/disciplines/<name>/rs-learn.db`. Embeddings 384-dim
  (MiniLM-L6-v2).
version: 0.2.0
---

# rs-learn — wasm recall/memorize

Library, not a binary. Host-served through plugkit; the agent never invokes rs-learn directly.

## Surface

`recall` — semantic retrieval. Body `{query, limit?}`. Returns `[{id, text, score, namespace?}]`.

`memorize` — append a fact. Body `{text, namespace}`. Embeds + persists.

## Dispatch

Write `.gm/exec-spool/in/recall/<N>.txt` or `.gm/exec-spool/in/memorize/<N>.txt`; read `.gm/exec-spool/out/<N>.json`. Synchronous from the agent's view.

```
recall      { "query": "terrain palette switching", "limit": 5 }
memorize    { "text": "fact body", "namespace": "default" }
```

## Auto-recall

Prompt-submit derives a 2-6 word query from the user prompt and injects hits into the `instruction` response under `## Recall for this prompt`. No explicit dispatch required at turn start.

## Discipline isolation

`@<name>` sigil in the request → reads and writes restricted to `<project>/.gm/disciplines/<name>/rs-learn.db`. Without a sigil: reads fan across `default` + every line of `<project>/.gm/disciplines/enabled.txt`, merge-ranked with `[discipline:<name>]` prefix; writes land in `default`.

## Storage

`<project>/.gm/rs-learn.db` (default discipline) or `<project>/.gm/disciplines/<name>/rs-learn.db`. Committed to git — shared across machines and sessions. Never add to `.gitignore`.

Each `memorize` persists: episode text, 384-dim embedding, namespace, timestamp. No graph extraction, no LLM call, no classifier in the wasm path — classification happens upstream in the `memorize` orchestrator verb before bytes hit this library.

## Constraints

- wasm32 target only. Native build deleted post-`dd065b9`.
- Host imports required: `host_kv_get`, `host_kv_put`, `host_kv_query`, `host_vec_search`, `host_log`, `host_now_ms`. Supplied by plugkit-wasm-wrapper.
- `EMBED_DIM = 384`. Changing the embedder requires reindex.
- No subcommands. No env vars. No `RS_LEARN_BACKEND`, no `RS_LEARN_ACP_COMMAND`, no `RS_LEARN_DB_PATH`.
- Recall returns `[]` on empty index; never errors on cold start.
- Memorize is append-only; dedup happens at query time via embedding similarity.

## Types

```rust
pub struct RecallHit {
    pub id: String,
    pub text: String,
    pub score: f32,
    pub namespace: Option<String>,
}

pub struct Learn;
impl Learn {
    pub fn recall(&self, query: &str, limit: usize) -> Result<Vec<RecallHit>>;
    pub fn memorize(&self, text: &str, namespace: &str) -> Result<()>;
}
```
