---
description: Show rs-learn orchestrator status — DB path, ACP command, HTTP debug endpoints
---

Check the rs-learn installation and current configuration.

Run:
```bash
which rs-learn || echo "not on PATH"
echo "DB: ${RS_LEARN_DB_PATH:-.rs-learn.db}"
echo "ACP: ${RS_LEARN_ACP_COMMAND:-not set}"
echo "LIVE: ${RS_LEARN_ACP_LIVE:-0}"
```

If rs-learn is running with observability enabled, probe the HTTP endpoints:
```bash
curl -s http://127.0.0.1:${RS_LEARN_OBS_PORT:-8787}/debug/memory
curl -s http://127.0.0.1:${RS_LEARN_OBS_PORT:-8787}/debug/router
```
