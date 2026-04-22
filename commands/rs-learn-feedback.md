---
description: Record reward + success signal for a prior rs-learn request_id (trains instant/background loops)
argument-hint: <request_id> <reward 0..1> <success true|false>
---

```bash
rs-learn feedback $ARGUMENTS
```

Positive feedback grows the rank-2 MicroLoRA adapter. Negative feedback weakens it. Accumulated feedback drives the hourly background kmeans + BaseLoRA retrain.
