---
title: Papers
layout: default
---

# Papers

Technical papers and long-form research notes produced from real lab work at Dielabs.

---

## Documents

### [CPU Inference Architecture Comparison](cpu-inference-architecture-comparison.md)
Benchmark and analysis of three LLM sharding architectures (Single-Node, Tensor Parallel, Data Parallel) on a Dell PowerEdge R730 dual-socket server with vLLM. Identifies the TP anti-pattern on QPI, the DP crossover point at ~6 concurrent requests, and demonstrates the structural isomorphism between NUMA-aware CPU inference and GPU distributed inference.

### [Inference Engineering Manual](inference-engineering-manual.md)
A comprehensive technical manual covering the full inference stack: memory wall physics, KV cache economics, batching strategies, parallelism, and deployment topologies. Targeting infrastructure professionals transitioning into AI inference engineering.

### [KV Cache Offloading Research](kv-cache-offloading-research.md)
Research and benchmark findings on KV cache CPU offloading with LMCache on an RTX 4070 Super. Covers A/B comparison methodology, configuration mechanics, and performance impact analysis.

---

*All content is original Dielabs work by Diego Bardella.*
