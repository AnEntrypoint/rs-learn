---
description: Run a query through the rs-learn orchestrator and stream stage breakdown
argument-hint: <question>
---

Execute a query against the orchestrator. Requires RS_LEARN_ACP_COMMAND set and a live ACP agent.

```bash
rs-learn query "$ARGUMENTS"
```

Returns JSON with `text`, `stage_breakdown`, `request_id`, and retrieval telemetry.
