# rs-learn Agent Coordination

## Non-Obvious Implementation Details

### Loop Training Improvements (Session ~2026-04-24)

The following changes to the learning loops were shipped to improve training signal and prevent catastrophic forgetting:

1. **InstantLoop Prioritized Replay** (`src/learn/instant.rs`)
   - `weighted_pick()` samples replay buffer by cumulative `|scale|` weight
   - Falls back to uniform when all scales are zero
   - Rationale: high-magnitude adapter updates merit more training iterations

2. **BackgroundLoop Dead-band Removed** (`src/learn/background/mod.rs`)
   - Previously skipped training on `0.3 < quality < 0.7` to avoid noisy mid-quality signals
   - Now trains on all qualities [0, 1]
   - Rationale: provides richer gradient signal across full spectrum; router centering handles squashing

3. **Router Continuous Centered Gradient** (`src/router.rs`)
   - Loss computation: `centered = quality - 0.5; strength = centered.abs() * 2.0`
   - Covers full quality [0,1] range with no dead-band zone
   - Symmetric gradient at quality=0.5 (loss=0), maximum gradient at extremes (loss=1)
   - Rationale: smooth everywhere, complements background loop training full band

4. **EWC Consolidation on Boundary** (`src/orchestrator/mod.rs`)
   - Before `reset_adapter()` on z-score boundary fire: `deep.consolidate("adapter", &flat, &grads)`
   - Where `grads = emb[i % emb.len()] * quality`
   - Rationale: Fisher-weighted consolidation prevents catastrophic forgetting; regularizes solution space before reinitialization

5. **DeepLoop Fully Wired** (`src/orchestrator/mod.rs`)
   - DeepLoop is live in `Orchestrator::new_default` (not library-only)
   - On every `feedback()`: loss = `1.0 - quality` fed to `DeepLoop::record_loss`
   - On z-score boundary: `InstantLoop::reset_adapter()` called
   - No environment gate; always active
   - Note: README.md had stale "library-only" claim which is incorrect

### Observability Endpoints

New debug endpoint available when DeepLoop is wired:
- `GET /debug/deep` — Fisher EMA state, consolidation history, z-score boundary stats

Existing endpoints document their subsystems via `curl http://127.0.0.1:7878/debug` (all) or per-subsystem (instant, background, router, etc.).

### Windows Subprocess Issues (Already in CLAUDE.md)

See CLAUDE.md for:
- Chocolatey Rust shadowing (requires PATH prepend, CLAUDE.md section 1)
- `.cmd` shim argument parsing (stdin workaround, CLAUDE.md section 2)
- gm-cc sandbox exec:sleep/exec:status unreliability (use `gh run view` directly, CLAUDE.md section 3)
