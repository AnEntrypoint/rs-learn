---
name: rs-learn
description: Continual-learning memory for any project. Ingests text/files into a local graph+vector store, retrieves semantically, and routes queries through any ACP stdio agent. Zero install — invoke via bun x rs-learn or npx rs-learn.
version: 0.1.35
---

# rs-learn — Continual-Learning Memory

Zero-install persistent memory via `bun x rs-learn` or `npx rs-learn`. Creates `rs-learn.db` in the current directory — one file, no server, no daemon.

## Install this skill

```bash
bun x skills add AnEntrypoint/rs-learn
# or
npx skills add AnEntrypoint/rs-learn
```

## Quick start

```bash
# Ingest a file (chunked, paragraph-aware)
bun x rs-learn add --file AGENTS.md --source "AGENTS.md" --chunk-size 2000

# Ingest from stdin
cat notes.md | bun x rs-learn add --file - --source "notes"

# Ingest a short snippet directly
bun x rs-learn add "terrain shader palette switching is forbidden" --source "tip"

# Semantic search
bun x rs-learn search "biome color blending" --limit 5

# Query through an ACP agent (requires RS_LEARN_ACP_COMMAND)
bun x rs-learn query "why do terrain borders show stitch artifacts?"

# Show graph/memory stats
bun x rs-learn debug
```

## Environment

| Var | Default | Purpose |
|-----|---------|---------|
| `RS_LEARN_DB_PATH` | `./rs-learn.db` | libsql file path |
| `RS_LEARN_ACP_COMMAND` | — | ACP stdio agent command (enables `query`) |
| `RS_LEARN_BACKEND` | `claude-cli` | LLM backend for entity/edge extraction |

### ACP agent examples

```bash
export RS_LEARN_ACP_COMMAND="kilo acp"
export RS_LEARN_ACP_COMMAND="opencode acp"
export RS_LEARN_ACP_COMMAND="claude --print -p"
```

## add subcommand

```bash
bun x rs-learn add <text>                              # inline text
bun x rs-learn add --file <path>                       # read from file
bun x rs-learn add --file -                            # read from stdin
bun x rs-learn add --file doc.md --chunk-size 2000     # chunk at paragraph boundaries
bun x rs-learn add <text> --source "label"             # tag the source
```

`--chunk-size N` splits on paragraph (`\n\n`) or line boundaries, ingesting each chunk as a separate episode tagged `"source [i/total]"`. Use for files larger than ~1 KB. Each chunk triggers LLM entity+edge extraction — expect 20-40s per chunk.

## Search scopes

```bash
bun x rs-learn search "query"                        # nodes (default)
bun x rs-learn search "query" --scope facts          # edges/facts
bun x rs-learn search "query" --scope episodes       # raw episodes
bun x rs-learn search "query" --scope communities
bun x rs-learn search "query" --limit 20
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

Each `add` call creates:
- **Episode** — raw content record
- **Nodes** — named entities extracted by LLM (people, concepts, files, APIs)
- **Edges** — typed relations between nodes (USES, IMPLEMENTS, CONTRADICTS, etc.)
- **Embeddings** — 768-dim vectors for HNSW retrieval

## Memory location

`rs-learn.db` is created in the **current working directory**. Run from your project root to co-locate memory with the project.

```
/my-project/
  rs-learn.db       ← created automatically on first run
  AGENTS.md         ← ingest this for project-specific memory
```

## Feedback loop

After a `query`, record quality to train the router:

```bash
bun x rs-learn feedback <request_id> 0.9              # good response
bun x rs-learn feedback <request_id> 0.2 "off topic"  # bad response
```

The router learns which ACP targets, temperatures, and context buckets produce high-quality answers over time.
