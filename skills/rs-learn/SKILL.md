---
name: rs-learn
description: >
  Use this skill when the user wants persistent memory, knowledge retrieval, or
  continual learning for their project — even if they don't explicitly say
  "memory" or "graph". Activate when the user wants to: remember facts across
  sessions, search previously ingested notes or docs, ask a question with
  context from stored knowledge, ingest a file or document into a searchable
  store, or wire an AI agent to a persistent knowledge base. Also activate when
  replacing a file-based memory system (e.g. MEMORY.md + markdown files) with
  a semantic, queryable alternative. Zero install: runs via `bun x rs-learn`
  or `npx rs-learn`. Creates rs-learn.db in the project root.
version: 0.1.37
---

# rs-learn — Continual-Learning Memory

Zero-install persistent memory via `bun x rs-learn` or `npx rs-learn`. Creates `rs-learn.db` in the current directory — one file, no server, no daemon.

## Install this skill

```bash
bun x skills add AnEntrypoint/rs-learn
# or
npx skills add AnEntrypoint/rs-learn
```

## Command map

| Goal | Command |
|---|---|
| Persist a fact across sessions | `gm:memorize` agent (background, classifies); or `bun x rs-learn add … --no-extract` for synchronous capture |
| Recall mid-task in a gm session | `exec:recall <query>` |
| Search ingested content | `bun x rs-learn search "query" --scope episodes` (default scope returns node names only) |
| Question answered via stored memory | `bun x rs-learn query` (needs `RS_LEARN_ACP_COMMAND`) |
| Train the router on a prior query | `bun x rs-learn feedback <request_id> 0..1` |
| Drop the store | `bun x rs-learn clear` |
| Inspect state | `bun x rs-learn debug [subsystem]` |

`bun x rs-learn add` fits file/batch ingestion (`--file`, seeding). Per-turn agent work flows through `gm:memorize` so classification and AGENTS.md updates land together.

## Workflow

**Store a fact (fast, ~1s):**
```bash
bun x rs-learn add "terrain shader palette switching is forbidden" --source "tip" --no-extract
```

**Store a file (full graph extraction, 20-40s/chunk):**
```bash
bun x rs-learn add --file AGENTS.md --source "AGENTS.md" --chunk-size 2000
```

**Retrieve — always use `--scope episodes` for full content:**
```bash
bun x rs-learn search "biome color blending" --scope episodes --limit 5
```

**Query through an ACP agent (requires `RS_LEARN_ACP_COMMAND`):**
```bash
bun x rs-learn query "why do terrain borders show stitch artifacts?"
```

## Gotchas

- **Default search scope returns entity names only, not content.** `search "query"` (no `--scope`) searches nodes and returns entity names with no summary text — nearly useless for recall. Always pass `--scope episodes` when you want the actual stored text back.
- **DB path resolution:** `.gm/rs-learn.db` under the current working directory by default. Falls back to `./rs-learn.db` only if `.gm/` cannot be created. Override with `RS_LEARN_DB_PATH`. Wrong CWD = separate disconnected database. Legacy `./rs-learn.db` is auto-migrated to `.gm/rs-learn.db` on first run.
- **Sharing memory across sessions/machines:** commit `.gm/rs-learn.db` to your repo. Add `!.gm/rs-learn.db` to `.gitignore` (after broader `.gm/` and `*.db` rules). Pulling the repo brings every prior session's facts. Conflicts resolve by re-running `bun x rs-learn add` — the DB is write-mostly, append-shape friendly.
- **`--no-extract` and `--scope episodes` are a pair.** Facts stored with `--no-extract` have no graph nodes — they only exist as episodes. They are invisible to `search` without `--scope episodes`.
- **Shell arg-length limit.** `bun x rs-learn add "$(cat file)"` crashes bun when the file exceeds ~2048 chars. Use `--file <path>` or `--file -` (stdin) instead.
- **LLM extraction is sequential per chunk.** With `--chunk-size 2000` and a 12KB file (~6 chunks), full extraction takes 4+ minutes. Use `--no-extract` when speed matters and graph structure is not needed.
- **`RS_LEARN_BACKEND` defaults to `claude-cli`.** Entity extraction calls Claude via the local CLI. If `claude` is not in PATH, extraction silently fails and episodes store with zero nodes/edges.

## add subcommand

```bash
bun x rs-learn add <text>                              # inline text
bun x rs-learn add --file <path>                       # read from file
bun x rs-learn add --file -                            # read from stdin
bun x rs-learn add --file doc.md --chunk-size 2000     # chunk at paragraph boundaries
bun x rs-learn add <text> --source "label"             # tag the source
bun x rs-learn add <text> --no-extract                 # fast: skip LLM, episode+embedding only (~1s)
```

`--chunk-size N` splits on paragraph (`\n\n`) or line boundaries. Each chunk is a separate episode tagged `"source [i/total]"`. Each chunk triggers LLM entity+edge extraction — 20-40s per chunk.

`--no-extract` skips all LLM calls. Episode stored with its embedding only. Use for short memory facts where `search --scope episodes` retrieval is sufficient and latency matters.

## Search scopes

```bash
bun x rs-learn search "query" --scope episodes       # full episode content (use this for recall)
bun x rs-learn search "query"                        # nodes/entities only (names, no content)
bun x rs-learn search "query" --scope facts          # edges/relations
bun x rs-learn search "query" --scope communities    # cluster summaries
bun x rs-learn search "query" --limit 20
```

## Environment

| Var | Default | Purpose |
|-----|---------|---------|
| `RS_LEARN_DB_PATH` | `.gm/rs-learn.db` (auto-created) | libsql file path; falls back to `./rs-learn.db` if `.gm/` not writable |
| `RS_LEARN_ACP_COMMAND` | — | ACP stdio agent command (enables `query`) |
| `RS_LEARN_BACKEND` | `claude-cli` | LLM backend for entity/edge extraction |

### ACP agent examples

```bash
export RS_LEARN_ACP_COMMAND="kilo acp"
export RS_LEARN_ACP_COMMAND="opencode acp"
export RS_LEARN_ACP_COMMAND="claude --print -p"
```

## MCP server

Expose rs-learn as an MCP tool so agents can call add/search without spawning subprocesses:

```bash
bun x rs-learn mcp   # start MCP stdio server
```

Wire into `.mcp.json`:
```json
{
  "mcpServers": {
    "rs-learn": {
      "command": "bun",
      "args": ["x", "rs-learn", "mcp"]
    }
  }
}
```

## All subcommands

| Command | Purpose |
|---------|---------|
| `add` | Ingest episode(s) — file, stdin, or inline text |
| `search` | Semantic search over nodes/facts/episodes/communities |
| `query` | Route query through ACP agent with memory context |
| `feedback` | Record quality signal for a prior query (trains router) |
| `debug` | Dump internal state (attention, memory, router, loops) |
| `build-communities` | Run label propagation + summarize clusters |
| `serve` | Start HTTP REST server (default port 8000) |
| `mcp` | Start MCP stdio server |
| `clear` | Drop all graph data |
| `version` | Print version |

## What gets stored

Each `add` call (without `--no-extract`) creates:
- **Episode** — raw content record (always)
- **Nodes** — named entities extracted by LLM (people, concepts, files, APIs)
- **Edges** — typed relations between nodes (USES, IMPLEMENTS, CONTRADICTS, etc.)
- **Embeddings** — 768-dim vectors for HNSW retrieval (always)

## Feedback loop

After a `query`, record quality to train the router:

```bash
bun x rs-learn feedback <request_id> 0.9              # good response
bun x rs-learn feedback <request_id> 0.2 "off topic"  # bad response
```

The router learns which ACP targets, temperatures, and context buckets produce high-quality answers over time.
