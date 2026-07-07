# rs-learn (retired)

**This crate is retired.** rs-plugkit no longer depends on it: memory is human-readable md files (`.gm/memories/`, `.gm/disciplines/<name>/memories/`) with a derived per-project vector index, and the memorize/recall/prune paths live in-tree at [AnEntrypoint/rs-plugkit](https://github.com/AnEntrypoint/rs-plugkit). This repo no longer triggers the release cascade.

Continual-learning primitives for LLM agents, shipped as a small wasm cdylib that runs inside a host providing key-value storage and vector search via a handful of imports.

This crate is the **algorithm layer**. The host (rs-plugkit, or anything that satisfies the imports) owns IO, scheduling, embedding, and agent transport. Together they provide LoRA optimization over time (in the spirit of ruvnet's sona) and a temporal knowledge graph (in the spirit of getzep's graphiti).

## What's in the box

Five algorithm cores, all native-Rust testable:

| Module | Algorithm | Inspiration |
| --- | --- | --- |
| `learn::instant_core` | Rank-2 Hebbian MicroLoRA adapter; norm-bound; LR floor; `\|scale\|`-weighted prioritized replay; optional EWC correction | sona |
| `learn::deep_core` | EWC++ with online Fisher EMA (decay 0.999); z-score boundary detection over a sliding loss window | EWC++ |
| `graph::temporal_core` | Bi-temporal edge store (`valid_at` / `invalid_at` / `created_at` / `expired_at`); point-in-time query; invalidate-don't-delete contradiction handling | graphiti |
| `graph::attention` | 8-head graph attention with edge features (relation one-hot + recency decay + weight); per-head softmax weights | graphiti / GAT |
| `router::core` | FastGRNN sparse (90%) + low-rank (rank-8) router; five heads (model, context bucket, temperature, top_p, confidence); epsilon-greedy exploration; per-target outcome EMA | original |

One JSON dispatch surface (`dispatch::LearnSession`) maps verbs to the cores; the wasm export wires `rs_learn_dispatch` to a singleton `LearnSession<HostKv>` backed by the host's `host_kv_*` shim.

## Host contract

The wasm module imports six functions:

```c
host_kv_get   (ns_ptr, ns_len, key_ptr, key_len)                        -> u64  // packed ptr+len
host_kv_put   (ns_ptr, ns_len, key_ptr, key_len, val_ptr, val_len)      -> u32  // nonzero = ok, 0 = failure
host_kv_query (ns_ptr, ns_len, query_ptr, query_len)                    -> u64
host_log      (level, msg_ptr, msg_len)                                  -> u32
host_now_ms   ()                                                          -> i64
```

`host_kv_query` returns a JSON array of `[{key, value?}]` entries. The `query` parameter is not a pure key-prefix filter at the host level: the reference host (gm-plugkit's JS wrapper) substring-matches `query` against key **or value** content and scans the entire namespace on every call (`O(namespace)` IO per query, no index). This crate never relies on the host doing prefix-only matching -- every caller (`graph::host_kv_backend::HostKv::list_prefix`) re-filters the returned entries client-side with `key.starts_with(prefix)` before use, so correctness holds regardless of host-side matching semantics. Treat `host_kv_query` as a coarse, whole-namespace substring scan; do not assume prefix-only semantics or sub-linear cost when implementing a new host.

## Wasm exports

```c
rs_learn_alloc    (len)      -> *mut u8
rs_learn_free     (ptr, len)
rs_learn_dispatch (ptr, len) -> u64        // packed ptr+len of response JSON
rs_learn_version  ()         -> *const u8  // NUL-terminated semver
```

Hosts pass an in-buffer (`{"verb": "...", "body": {...}}`) and read the response from the packed return.

## Dispatch verbs

Every call is `{ "verb": "<name>", "body": { ... } }` in, `{ "ok": bool, "verb": "<name>", "data"?: ..., "error"?: "..." }` out.

| Verb | Body | Effect |
| --- | --- | --- |
| `health` | `{}` | Reports core readiness, Fisher key count, boundary count |
| `init_instant` | `{targets: [String]}` | Initialize MicroLoRA adapter |
| `feedback` | `{embedding, model, payload:{quality}, now_ms}` | Hebbian update + replay |
| `apply_adapter` | `{embedding}` | Run adapter forward, return logits |
| `reset_adapter` | `{}` | Zero adapter, reset LR |
| `record_loss` | `{loss}` | Add to loss ring; returns whether z-score boundary fired |
| `consolidate` | `{param_id, params, grads}` | Update Fisher EMA + snapshot |
| `ewc_penalty` | `{param_id, params}` | Compute `lambda * sum(fisher * (params - snapshot)^2)` |
| `init_router` | `{in_dim, targets, trained?, epsilon?}` | Initialize FastGRNN router |
| `route` | `{embedding, estimated_tokens?}` | Return chosen model + sampling params |
| `record_outcome` | `{target, quality}` | EMA-update per-target quality |
| `init_attention` | `{dim, heads?, head_dim?, seed?}` | Initialize 8-head attention |
| `attend` | `{query, subgraph:{nodes, edges}, now_ms}` | Multi-head attended context vector |
| `insert_edge` | `{id, src, dst, relation?, valid_at, created_at, ...}` | Insert bi-temporal edge |
| `query_at` | `{src, t}` | Edges where `valid_at <= t < invalid_at` |
| `invalidate_edge` | `{edge_id, invalid_at, expired_at}` | Set invalidation stamps |
| `contradict` | `{new_edge, contradicts:[edge_id], now_ms}` | Insert new edge + invalidate olds with `old.invalid_at = new.valid_at` |
| `get_edge` | `{edge_id}` | Fetch one edge |

## Bi-temporal model

Edges carry four timestamps:

- **`valid_at` / `invalid_at`** — the interval during which the fact held in the world (valid time)
- **`created_at` / `expired_at`** — when we knew or stopped knowing the fact (system time)

A point-in-time query at `t` returns edges with `valid_at <= t < invalid_at` (or `invalid_at IS NULL`). Contradictions never delete history: when fact B contradicts fact A starting at `t = B.valid_at`, we set `A.invalid_at = t` and `A.expired_at = now`, keeping A queryable for any time before `t`.

This is the Graphiti invariant; the canonical test (`graphiti_style_contradiction_preserves_history`) walks the Alice@Acme → Alice@Globex scenario and witnesses both edges retrievable at their respective historical windows.

## LoRA optimization

`instant_core::InstantCore` keeps a rank-2 Hebbian adapter `(A: 2 x IN, B: 2 x n_targets)`. Each `feedback(emb, model, {quality, ...})` call:

1. Maps quality `[0,1]` to a centered `scale = (q - 0.5) * 2`. Neutral (`q ~ 0.5`) is a no-op.
2. Runs a Hebbian update on `A` and `B` for the chosen target, scaled by `lr * scale`.
3. Applies the optional EWC correction: `delta -= lr * lambda * fisher * (params - snapshot)`.
4. Decays `lr` by `0.995`, floored at `RS_LEARN_LR_MIN` (default `1e-3`).
5. Clamps `||A,B||` to `MAX_ADAPTER_NORM = 5.0`.
6. Buffers `(emb, idx, scale)` for `|scale|`-weighted prioritized replay (one extra Hebbian step on a sampled past transition).

`apply_adapter(emb, logits)` adds `B @ (A @ emb)` to incoming logits. This is what the router calls per route to bend its decision toward locally-validated patterns.

## Build

```bash
cargo build --target wasm32-wasip1 --release -p rs-learn
```

The artifact is `target/wasm32-wasip1/release/rs_learn.wasm`. Hosts load it and call `rs_learn_dispatch`.

## Testing

```bash
cargo test -p rs-learn
```

65 tests covering algorithm invariants and end-to-end JSON dispatch through every verb, including:

- LoRA: convergence under positive feedback, norm-bound clamp, LR-floor respect, neutral-feedback no-op, reset
- Bi-temporal: insert validation, point-in-time query, idempotent invalidation, Graphiti contradiction scenario, cross-session KV persistence
- EWC: Fisher EMA bound, penalty zero-at-snapshot, z-score boundary fires only after warm-up
- Attention: weights-sum-to-one per head, embedding-dim filtering, relation nudge effect, query residual
- Router: sparsity ~90%, untrained/trained behavior, epsilon=0/1 extremes, adapter override
- Dispatch: unknown-verb error, missing-field error, JSON roundtrip, full pipeline (init_instant + init_router + insert_edge + route + feedback + query_at + health)

## License

MIT
