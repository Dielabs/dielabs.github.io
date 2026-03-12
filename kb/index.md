---
title: Knowledge Base
layout: default
---

# Knowledge Base

Reference documentation on LLM inference engineering, produced from real lab work at Dielabs.

---

## Documents

### [LLM Parameter Topology](inference-parameters.md)
A structured framework for understanding where every LLM parameter lives: Artifact, Startup, or Request. Covers the full parameter flow from model weights to runtime enforcement, with conflict zones, troubleshooting tables and a mental checklist for inference engineers.

### [Inference Observability](inference-observability.md)
Monitoring, diagnostics and incident response for LLM inference systems. Covers the golden metrics (TTFT, ITL, E2E), percentile statistics, root-cause diagnostic tree, PromQL queries for Grafana, and a pre-benchmark checklist. Based on real measurements on an RTX 4070 Super with vLLM + Prometheus + DCGM.

### [KV Cache Architecture](kv-cache-architecture-kb.md)
Deep dive into KV cache internals: PagedAttention, prefix sharing, eviction policies, and memory layout. Reference document for understanding vLLM memory management.

### [KV Cache Sizing](kv-cache-sizing.md)
Capacity planning guide for KV cache on constrained hardware. Covers GPU memory budgeting, block calculations, and practical sizing for the RTX 4070 Super homelab.

---

*All content is original Dielabs work by Diego Bardella.*
