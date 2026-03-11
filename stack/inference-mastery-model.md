# Inference Mastery Model (IMM) — Layers A–G

A competency framework for LLM inference engineering. Unlike the ISM (which maps components), the IMM maps **skills** — what an inference engineer needs to understand and operate at each level of depth.

The IMM cuts across the ISM stack: Layer C touches L0, L1 and L3 simultaneously; Layer F spans almost all ISM layers. It was designed to map *competencies*, not components.

---

## Layer A — Inference Pipeline

The lifecycle of a request: HTTP request → tokenize → batch → prefill → decode → stream.

**Covers:**
- Tokenizer mechanics
- KV cache lifecycle (allocation, fill, reuse, eviction)
- Prefill vs decode distinction (compute-bound vs memory-bound)
- Continuous batching
- Streaming response handling

---

## Layer B — Inference Engines

The concrete serving engines in production use.

**Covers:**
- **vLLM** — PagedAttention, scheduler, block manager
- **SGLang** — RadixAttention, constrained decoding
- **TensorRT-LLM** — engine compilation, max throughput
- **TGI** — HuggingFace native serving
- **llama.cpp** — GGUF format, edge/CPU inference

---

## Layer C — Memory Engineering

How GPU memory is allocated, consumed and managed under pressure.

**Covers:**
- VRAM napkin math (weights + KV cache + activations + overhead)
- Quantization formats (FP16, BF16, INT8, INT4, GPTQ, AWQ, GGUF)
- Quality/compression tradeoffs (perplexity degradation)
- PagedAttention as virtual memory for KV cache
- Eviction policies under memory pressure
- OOM diagnosis and prevention

---

## Layer D — Optimization Techniques

Advanced techniques to push throughput and reduce latency beyond baseline.

**Covers:**
- Speculative decoding (draft model + verification)
- FlashAttention (memory-efficient attention with tiling)
- Chunked prefill (long-context without blocking decode)
- Prefix caching (prompt reuse across requests)
- Tensor / pipeline / sequence parallelism
- Profiling tools (Nsight, torch profiler, DCGM metrics)

---

## Layer E — Distributed Inference

Operating inference across multiple GPUs and nodes.

**Covers:**
- Tensor parallelism (intra-layer split)
- Pipeline parallelism (inter-layer split)
- Disaggregated prefill-decode (prefill and decode on separate GPUs)
- NCCL collectives (all-reduce, all-gather)
- NVLink vs PCIe bandwidth ceiling
- Multi-node networking (ConnectX, InfiniBand, RoCE)

---

## Layer F — Performance Measurement

Defining, measuring and interpreting inference performance correctly.

**Covers:**
- TTFT (Time To First Token)
- TBT / ITL (Time Between Tokens / Inter-Token Latency)
- E2E latency
- Throughput (tokens/sec, requests/sec)
- Throughput vs latency as distinct optimization targets
- SLA definition and measurement
- Benchmarking methodology (warmup, percentiles, load profiles)

---

## Layer G — Production Engineering

Operating inference systems at scale with reliability and cost awareness.

**Covers:**
- Capacity planning (concurrent users → KV cache → VRAM ceiling)
- Load balancing across inference replicas
- Monitoring and alerting (signal vs noise)
- Cost modeling (GPU hours per million tokens, $/token)
- Autoscaling policies
- Health checks and graceful degradation

---

*Part of the Dielabs framework. See also: [Inference Stack Model](inference-stack-model.md), [Inference Diagnostic Framework](inference-diagnostic-framework.md)*
