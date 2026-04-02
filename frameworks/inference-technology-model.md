---
layout: default
title: "Inference Technology Model"
---

# Inference Technology Model

A competency framework (Layers A–G) mapping what an inference engineer needs to know and operate. Unlike the Inference Diagnostic Framework (which maps components via L0–L6), this maps *skills* — and deliberately cuts across multiple L-layers.

---

**Layer A — Inference Pipeline** The lifecycle of a request: HTTP request → tokenize → batch → prefill → decode → stream. Covers tokenizer mechanics, KV cache lifecycle (allocation, fill, reuse, eviction), the prefill vs decode distinction (compute-bound vs memory-bound), continuous batching, streaming response handling.

**Layer B — Inference Engines** The concrete engines: vLLM (PagedAttention, scheduler, block manager), SGLang (RadixAttention, constrained decoding), TensorRT-LLM (engine compilation, max throughput), TGI (HuggingFace native serving), llama.cpp (GGUF, edge/CPU inference).

**Layer C — Memory Engineering** VRAM napkin math (weights + KV cache + activations + overhead), quantization formats (FP16, BF16, INT8, INT4, GPTQ, AWQ, GGUF), quality/compression tradeoff (perplexity degradation), PagedAttention as virtual memory for KV cache, eviction policies under memory pressure, OOM diagnosis and prevention.

**Layer D — Optimization Techniques** Speculative decoding (draft model + verification), FlashAttention (memory-efficient attention with tiling), chunked prefill (long-context without blocking decode), prefix caching (prompt reuse across requests), tensor/pipeline/sequence parallelism, profiling tools (Nsight, torch profiler, DCGM metrics).

**Layer E — Distributed Inference** Tensor parallelism (intra-layer split), pipeline parallelism (inter-layer split), disaggregated prefill-decode (prefill and decode on separate GPUs), NCCL collectives (all-reduce, all-gather), NVLink vs PCIe bandwidth ceiling, multi-node networking (ConnectX, InfiniBand, RoCE).

**Layer F — Performance Measurement** TTFT, TBT/ITL, E2E latency, throughput (tokens/sec, requests/sec), throughput vs latency as distinct targets, SLA definition and measurement, benchmarking methodology (warmup, percentiles, load profiles).

**Layer G — Production Engineering** Capacity planning (concurrent users → KV cache → VRAM ceiling), load balancing across inference replicas, monitoring and alerting (signal vs noise), cost modeling (GPU hours per million tokens, $/token), autoscaling policies, health checks and graceful degradation.

---

The framework cuts across the L0–L6 Stack Model: for example Layer C touches L0, L1 and L3 simultaneously, Layer F traverses nearly all L-layers. It is designed to map *competencies*, not components.
