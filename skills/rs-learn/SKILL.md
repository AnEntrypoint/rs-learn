---
name: rs-learn
description: Continual-learning memory for any project. Ingests text/files into a local graph+vector store, retrieves semantically, routes queries through an ACP agent. Zero install — uses bunx rs-learn (or npx). Compatible with hermes-agent, gm-cc, and any ACP stdio agent.
version: 0.1.34
platforms: [macos, linux, windows]
---

# rs-learn — Continual-Learning Memory

Zero-install persistent memory via `bunx rs-learn` (or `npx rs-learn`). Creates `rs-learn.db` in the current directory — one file, no server.

## Quick start

```bash
# Ingest a file (chunked, paragraph-aware)
bunx rs-learn add --file AGENTS.md --source "AGENTS.md" --chunk-size 2000

# Ingest from stdin
cat notes.md | bunx rs-learn add --file - --source "notes"

# Ingest a short snippet directly
bunx rs-learn add "terrain shader palette switching is forbidden — causes white stitch lines" --source "tip"

# Semantic search
bunx rs-learn search "biome color blending" --limit 5

# Query through an ACP agent (set RS_LEARN_ACP_COMMAND first)
bunx rs-learn query "why do terrain borders show stitch artifacts?"

# Show graph/memory stats
bunx rs-learn debug
```

## Environment

| Var | Default | Purpose |
|-----|---------|---------|
| `RS_LEARN_DB_PATH` | `./rs-learn.db` | libsql file path |
| `RS_LEARN_ACP_COMMAND` | — | ACP stdio agent (enables `query`) |
| `RS_LEARN_BACKEND` | `claude-cli` | LLM backend for entity/edge extraction |

### ACP agent examples

```bash
export RS_LEARN_ACP_COMMAND="hermes acp"          # hermes-agent
export RS_LEARN_ACP_COMMAND="kilo acp"            # kilocode
export RS_LEARN_ACP_COMMAND="opencode acp"        # opencode
export RS_LEARN_ACP_COMMAND="claude --print -p"   # claude CLI non-interactive
```

## Add subcommand

```
bunx rs-learn add <text>                          # inline text
bunx rs-learn add --file <path>                   # read from file
bunx rs-learn add --file -                        # read from stdin
bunx rs-learn add --file doc.md --chunk-size 2000 # chunk at paragraph boundaries
bunx rs-learn add <text> --source "label"         # tag source
```

`--chunk-size N` splits on paragraph (`\n\n`) or line boundaries, ingesting each chunk as a separate episode tagged `"source [i/total]"`. Recommended for files >1KB.

Each `add` call runs LLM entity+edge extraction — expect 20-40s per chunk depending on backend.

## Search scopes

```bash
bunx rs-learn search "query"                  # nodes (default)
bunx rs-learn search "query" --scope facts    # edges/facts
bunx rs-learn search "query" --scope episodes # raw episodes
bunx rs-learn search "query" --scope communities
bunx rs-learn search "query" --limit 20
```

## Full subcommand list

```
add       Ingest episode(s) — file, stdin, or inline text
search    Semantic search over nodes/facts/episodes/communities
query     Route query through ACP agent with memory context
feedback  Record quality signal for a prior query (trains router)
debug     Dump internal state (attention, memory, router, loops)
build-communities  Run label propagation + summarize clusters
serve     Start HTTP REST server (default port 8000)
mcp       Start MCP stdio server
clear     Drop all graph data
version   Print version
```

## What gets stored

Each `add` call creates:
- **Episode** — raw content record
- **Nodes** — named entities extracted by LLM (people, concepts, files, APIs)
- **Edges** — typed relations between nodes (USES, IMPLEMENTS, CONTRADICTS, etc.)
- **Embeddings** — 768-dim nomic-embed-text vectors for HNSW retrieval

## Memory location

`rs-learn.db` is created in the **current working directory** when any subcommand runs. Run from your project root to keep memory co-located with the project.

Per-project memory pattern:
```
/my-project/
  rs-learn.db          ← created automatically
  AGENTS.md            ← ingest this for project-specific memory
  CLAUDE.md
```

## Hermes integration

Install this skill into hermes:
```bash
mkdir -p ~/.hermes/skills/rs-learn
cp SKILL.md ~/.hermes/skills/rs-learn/SKILL.md
```

Then in hermes: `/rs-learn` loads these instructions. Use alongside `RS_LEARN_ACP_COMMAND="hermes acp"` to route queries back through hermes with persistent memory context.

## Feedback loop

After a `query`, record quality to train the router:
```bash
bunx rs-learn feedback <request_id> 0.9          # good response
bunx rs-learn feedback <request_id> 0.2 "missed context"  # bad response
```

Router learns which ACP targets, temperatures, and context buckets produce high-quality answers.
