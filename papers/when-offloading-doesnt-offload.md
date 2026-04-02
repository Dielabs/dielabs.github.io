---
layout: default
title: "When Offloading Doesn't Offload"
---

# When Offloading Doesn't Offload

## KV Cache Offloading in vLLM 0.15.1

_Analysis, benchmarks and real limits — An experimental investigation into KV cache offloading mechanisms on NVIDIA RTX 4070 Super (12 GB VRAM, PCIe 4.0)_

---

## Abstract

This paper documents an experimental investigation into KV cache offloading mechanisms in vLLM 0.15.1, conducted on a consumer GPU node (NVIDIA RTX 4070 Super, 12 GB VRAM, PCIe 4.0 x16).

The investigation revealed three main results: **(1)** LMCache operates as an external prefix cache, not as a swap mechanism; **(2)** the Native OffloadingConnector (introduced in v0.11.0) implements a true preemption recovery offloading, but requires specific conditions to activate; **(3)** on consumer hardware with PCIe 4.0, the measurable benefit manifests on tail latency (ITL p99 reduced by 23%) but not on average throughput.

The paper describes the methodological path and the unexpected results that led to these conclusions. Formulas are included to support key concepts, but the text remains readable without them. Basic familiarity with inference engineering fundamentals is assumed.

---

## 1. Introduction

The KV cache is the most critical data structure in LLM inference. For each request, the model generates key-value pairs for every token processed, with linear growth relative to context length. On consumer GPUs with limited VRAM (12 GB), the KV cache becomes the primary bottleneck for concurrency.

The idea of "offloading" — moving part of the KV cache from GPU to CPU RAM — is conceptually simple and frequently promoted as a solution. However, the term conceals significant architectural complexity. This paper documents the attempt to activate and measure KV cache offloading in vLLM on real homelab hardware, revealing that the concept comprises at least three completely different mechanisms.

---

## 2. Background

### 2.1 KV Cache and Admission Control

For each token processed, the model generates a key-value pair. The KV cache size per request is:

> `kv_cache_per_request = num_tokens × num_layers × 2 × num_kv_heads × head_dim × bytes_per_element`

For Qwen3-8B-AWQ (36 layers, 8 KV heads, head_dim 128, FP16), each token occupies approximately 144 KB. A request with 4096 input tokens generates ~576 MB of KV cache during prefill.

The VRAM available for the KV cache:

> `kv_cache_budget = (VRAM_total × gpu_memory_utilization) − model_weights − cuda_graphs − overhead`

The V1 scheduler in vLLM calculates the maximum theoretical concurrency as kv_cache_budget divided by the worst-case allocation per request (max_model_len × kv_per_token). If there are not enough blocks, the new request enters the wait queue. This admission control is the primary gate governing the number of active requests — and the first thing to consider when attempting to force offloading. As shown in Section 5, the scheduler operates more optimistically than this formula suggests.

### 2.2 Preemption

Preemption intervenes when KV cache space becomes insufficient for all active requests. Understanding it is fundamental because preemption is the exact point where offloading can intervene: the cost of preemption is what offloading attempts to reduce.

**Recompute (default V1).** When a request is preempted, its memory blocks are freed and the request returns to the wait queue. When space becomes available, the request is rescheduled and its prefill recomputed from scratch. This design reflects V1's optimization for reduced CPU overhead — synchronous GPU→CPU swap was removed to simplify the core system.

**Preemption lifecycle:** (1) Saturation — the scheduler detects insufficient free blocks for the next decode step. (2) Eviction — the scheduler selects requests to suspend (FCFS by default). (3) Block release — KV blocks return to the free queue; with prefix caching enabled, blocks remain identifiable via hash until overwritten. (4) Resumption — when memory frees up, the request resumes. If its blocks are still in cache (cache hit), part of the recompute is skipped; otherwise full prefill recompute occurs.

**Performance impact.** Preemption guarantees robustness (no OOM) but at a significant cost. It disproportionately affects p99 metrics: most requests complete normally, but preempted ones see latencies far above the average. During recompute, even non-preempted requests experience increased ITL because the GPU executes a compute-bound prefill job among their decode steps. This asymmetry is why offloading benefits manifest specifically on tail latency.

### 2.3 The Three Offloading Mechanisms

The investigation identified three distinct mechanisms, often confused under the generic term "KV cache offloading."

**A) LMCache — External Prefix Cache**

LMCache operates as an external cache of pre-computed KV pairs, organized by token prefix hash. It is not integrated into vLLM's scheduler.

- **Store:** at each completed prefill, LMCache captures the KV blocks and saves them to its backends (CPU RAM, disk, Redis). The store occurs always, regardless of load.
- **Retrieve:** when a new request is admitted, LMCache searches for hash matches. If the prefix match exceeds the minimum threshold, cached blocks are loaded into the GPU, skipping redundant prefill.
- **When it doesn't help:** if requests have unique prefixes, no hash match occurs. The store happens anyway but produces no benefits.
- **Relationship with admission control:** none. The scheduler is not aware of LMCache's existence.

**B) Native OffloadingConnector — Preemption Recovery**

Introduced in vLLM v0.11.0 (RFC #19854), the OffloadingConnector is a native component of vLLM's KV Transfer framework. Its function is to save KV blocks to CPU RAM, enabling reload instead of recompute when a request is preempted.

- **Store (proactive mirroring):** the connector copies KV blocks layer-by-layer via DMA as each attention layer completes, while the GPU computes the next layer. The DMA engine operates independently from the GPU's compute cores (SMs). This means that if the scheduler suddenly preempts a request, CPU RAM already contains an up-to-date copy.
- **Load (reactive):** activated only when a preempted request is rescheduled. Blocks are reloaded from CPU RAM to GPU VRAM via PCIe, layer-by-layer. If the reload fails, the system falls back to standard recompute.
- **Mirroring cost:** continuous PCIe bandwidth consumption, CPU RAM usage (~8 GB in our setup), and marginal HBM bandwidth — even when no preemption occurs. vLLM treats this as an "insurance premium": using cheap resources (CPU RAM, typically underutilized PCIe bandwidth) to avoid expensive recompute. This explains the 9% overhead on TTFT observed in benchmarks.
- **Relationship with admission control:** indirect. The connector does not modify admission decisions, but when preemption occurs, the recovery cost becomes a PCIe transfer instead of a full recompute.

**C) FlexGen / InfiniGen / KVSwap — Full KV Swap**

Research frameworks that implement full KV cache swap (and sometimes model weight swap) between GPU, CPU and disk. Designed for offline throughput scenarios on very large models. Not integrable as a vLLM serving backend. Outside the scope of this study.

**Mechanism Comparison**

| Characteristic | LMCache | Native Offload | FlexGen/InfiniGen |
|---|---|---|---|
| Type | Prefix cache | Preemption recovery | Full KV swap |
| vLLM integration | External plugin | Native (v0.11.0+) | Separate framework |
| Store trigger | Every completed prefill | Proactive (continuous mirroring) | Always (by design) |
| Load trigger | Prefix hash match | Preempted request rescheduled | Every attention step |
| Requires shared prefixes | Yes (essential) | No | No |
| Requires preemption | No | Yes (essential) | No |
| Modifies admission control | No | No | N/A (own scheduler) |
| Optimal scenario | RAG, multi-turn, system prompt | High concurrency, bursts | Offline batch, large models |

---

## 3. Experimental Setup

### Hardware and Software

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX 4070 Super (12 GB GDDR6X) |
| GPU interface | PCIe 4.0 x16 (~21 GB/s bidirectional) |
| CPU RAM | 32 GB DDR4 |
| CPU | Intel i7 (8 performance cores) |
| vLLM | 0.15.1 (V1 architecture) |
| Model | Qwen3-8B-AWQ (Marlin quantization) |
| Benchmark | GuideLLM 0.5.3 |
| Observability | Prometheus + Grafana + DCGM Exporter |

### Workload

All tests use the same GuideLLM workload: concurrent mode, rate 10 req/s, 300 seconds per run. Prompt: 4096 input tokens + ≤512 output tokens (synthetic, random, no shared prefixes). The high input/low output ratio ensures deterministic pressure on the KV cache during prefill, while keeping requests active long enough to observe saturation during the decode phase.

---

## 4. Phase 1: LMCache — The Prefix Cache Misunderstanding

The initial hypothesis was that LMCache operated as an overflow mechanism: when the GPU runs out of KV space, blocks move to CPU RAM, allowing the scheduler to admit more requests.

Analysis of the official documentation revealed a completely different picture. LMCache does not modify the admission control logic. The scheduler continues to calculate maximum concurrency based exclusively on available GPU blocks. LMCache intervenes only after admission, to avoid re-prefilling tokens already computed when a subsequent request shares a prefix (hash match).

Since GuideLLM generates random prompts without shared prefixes, LMCache never finds a cache hit in this workload. This explains why results with LMCache enabled vs. disabled were virtually identical in previous tests (not published here).

---

## 5. Phase 2: Native OffloadingConnector

### 5.1 First Test — Conservative Configuration (Failed)

The first test used the same conservative configuration as the LMCache tests: `gpu-memory-utilization=0.75`, `max-model-len=8192`. Result: performance identical to vanilla. Host RAM rose from 5.5 GB to 13.6 GB (confirming that blocks were being reserved), but TTFT, throughput and latency remained unchanged.

### 5.2 The Scheduler Paradox

Analysis of the startup logs revealed the root cause. With the conservative configuration, vLLM reported a maximum theoretical concurrency of **1.36x** for 8192-token requests. The scheduler admitted at most 2 simultaneous requests out of 10 concurrent submissions.

For the conservative configuration:

> `kv_cache = (12 GB × 0.75) − 5.7 GB (weights) − 0.6 GB (CUDA graph) − overhead ≈ 1.5 GB`
>
> `max_concurrency ≈ 1.36x`

The intuition "provide less VRAM for the KV cache to force the offload" produced the opposite effect: less VRAM → fewer blocks → fewer requests admitted → less contention → no preemption → offloading never activates. **Static resource scarcity prevents the dynamic conflict needed to activate offloading.**

### 5.3 The Solution: Dynamic Over-subscription

The correct strategy was the opposite: maximize admitted requests so that the KV cache saturates _during generation_, forcing preemptions. Three changes applied simultaneously:

- **`--gpu-memory-utilization 0.95`** (from 0.75): the KV cache pool grew from ~1.5 GB to ~4.1 GB.
- **`--enforce-eager`**: disabled CUDA graphs, freeing ~0.6 GB of fixed VRAM for the KV cache.
- **`--max-model-len 5120`** (from 8192): the scheduler reserves less space per request, admitting more within the same budget.

### 5.4 The Optimistic Scheduler

A crucial result: the V1 scheduler admitted **up to 7 simultaneous requests**, exceeding the theoretical maximum by 4–5x. This was validated against vLLM's technical documentation.

The max_concurrency reported at startup is the **worst-case** planning value — the scenario where every request simultaneously occupies max_model_len. The scheduler uses **on-demand allocation**: physical KV blocks are allocated only when needed, not pre-allocated for max_model_len at admission. The admission check verifies current actual occupancy:

> `admit_request = (blocks_needed_now ≤ blocks_free)`

This enables over-subscription: requests arrive at different phases, and the scheduler sees available blocks based on each request's current (not maximum) occupancy. When all requests generate tokens simultaneously, the KV cache grows in parallel and exceeds the physical budget — triggering preemption. This is exactly the scenario the OffloadingConnector is designed for.

Grafana data confirms: **5.9 running requests on average, max 7** (vs. 2.1 in the conservative configuration), with KV cache at 98.9% (vanilla) and 99.9% (offload).

---

## 6. Benchmark Results

### 6.1 Conservative Configuration

_gpu-memory-utilization=0.75, max-model-len=8192, CUDA graphs enabled_

| Metric | Vanilla | Native Offload | Delta |
|---|---|---|---|
| Requests completed | 63 | 63 | = |
| Running requests (avg) | 2.1 | 1.9 | -10% |
| Average TTFT (ms) | 36,040 | 37,017 | +2.7% |
| Output tok/s | 107.1 | 104.9 | -2.1% |
| Host RAM (GB) | 5.51 | 13.59 | +8.08 GB |
| KV cache % max | 87.2% | 81.9% | |

Offloading is active (RAM rises by 8 GB) but produces no measurable benefit. The scheduler admits only 2 requests; no dynamic pressure on the KV cache, preemptions are rare.

### 6.2 Aggressive Configuration

_gpu-memory-utilization=0.95, max-model-len=5120, --enforce-eager_

| Metric | Vanilla Aggr. | Offload Aggr. | Delta |
|---|---|---|---|
| **Requests completed** | **103** | **103** | = |
| Running requests (avg) | 5.9 | 5.5 | -7% |
| Average TTFT (ms) | 12,202 | 13,352 | +9.4% |
| Median TTFT (ms) | 16,325 | 17,857 | +9.4% |
| Average request latency (s) | 27.60 | 27.75 | = |
| Output tok/s | 175.4 | 173.5 | -1% |
| Throughput req/s | 0.34 | 0.34 | = |
| Average ITL (ms) | 30.1 | 28.2 | -6.5% |
| **ITL p99 (ms)** | **328.9** | **253.0** | **-23%** |
| **E2E latency p99 (s)** | **38.60** | **35.59** | **-7.8%** |
| KV cache % max | 98.9% | 99.9% | |
| Host RAM (GB) | 4.81 | 12.98 | +8.17 GB |
| ITL p50 (ms) | 17.7 | 38.2 | +116% |

Average throughput is identical, but **tail latency improves**: ITL p99 drops 23% (329 → 253 ms), E2E p99 drops 7.8%. Preempted requests recover faster via PCIe reload compared to full recompute. Average TTFT worsens by 9.4%, reflecting the continuous mirroring overhead of the connector.

### 6.3 Configuration Impact on the Scheduler

| Metric | Vanilla Conserv. | Vanilla Aggr. | Delta |
|---|---|---|---|
| **Requests completed** | 63 | **103** | **+63%** |
| **Running requests** | 2.1 | **5.9** | **+181%** |
| **Average TTFT (ms)** | 36,040 | **12,202** | **-66%** |
| **Output tok/s** | 107.1 | **175.4** | **+64%** |
| Request latency (s) | 44.3 | 27.6 | -38% |

The aggressive configuration produces dramatic improvement across all metrics **regardless of offloading**. On consumer hardware with limited VRAM, vLLM parameter tuning has an orders-of-magnitude greater impact than enabling offloading.

---

## 7. Lessons Learned

**The term "offloading" is misleading.** It is used interchangeably for at least three architecturally different mechanisms: prefix caching, preemption recovery, and full KV swap. This ambiguity generates incorrect expectations, especially among infrastructure engineers who associate "offload" with memory swap/overflow.

**Admission control governs everything.** The vLLM scheduler is the primary gatekeeper. All offloading mechanisms operate downstream of admission. No currently available mechanism in vLLM bypasses admission control to admit more requests than the GPU can physically handle.

**PCIe 4.0 is the limiting factor.** At ~21 GB/s, the CPU→GPU transfer cost for reloading KV blocks is comparable to the recompute cost. The benefit manifests only on tail latency, not on average throughput. On hardware with faster interconnects (e.g. NVLink C2C at 900 GB/s on Grace Hopper/Blackwell systems), we would expect the ratio to shift substantially in favor of offloading, although this remains untested in our environment.

**Parameter tuning surpasses offloading.** The move from conservative to aggressive configuration produced +63% completed requests and -66% TTFT — without any offloading enabled. On consumer hardware, base parameter optimization has a far greater impact than experimental features.

**The value of negative results.** Demonstrating that a feature does not produce benefits in a specific scenario is a valid contribution. Few documents in the LLM inference community critically analyze the real limits of offloading on consumer hardware. Most benchmarks and guides assume datacenter hardware (H100, A100) with radically different bandwidth and memory characteristics.

---

## 8. Conclusions

**LMCache** operates as an external prefix cache, not as swap. Useful for workloads with shared prefixes (RAG, multi-turn) but invisible with random prompts.

**The Native OffloadingConnector** implements true preemption recovery offloading with asynchronous, layer-by-layer architecture. On RTX 4070 Super with PCIe 4.0, it reduces tail latency (ITL p99 by 23%, E2E p99 by 7.8%) but does not improve average throughput.

**The vLLM scheduler** is the dominant factor. With conservative configuration, no offloading mechanism produces effects. With aggressive configuration, the offloading benefit is real but marginal compared to the improvement from parameter tuning alone.

**Recommendation for consumer hardware operators:** before investing time in configuring offloading mechanisms, optimize base vLLM parameters (gpu-memory-utilization, max-model-len, enforce-eager). The benefit will be orders of magnitude greater. Native offloading becomes relevant only after maximizing scheduler concurrency, and its real value is destined to grow as hardware evolves toward faster CPU-GPU interconnects.

### Future Work

- **Testing with shared-prefix workloads:** design a benchmark with shared prefixes (simulating RAG) to measure LMCache in its optimal scenario.
- **DGX Spark analysis:** repeat benchmarks on unified memory hardware (128 GB, 273 GB/s) to quantify the bandwidth impact on offloading.
- **Progressive KV Cache Onloading:** monitor vLLM RFC #33526 (February 2026), which introduces progressive block loading to reduce head-of-line blocking.
- **Dedicated monitoring:** develop Prometheus/Grafana metrics to track OffloadingConnector store/load events in real time.

---

## Appendix A — Input/Output Length Choice

Input length is fixed at 4096 tokens; output is limited to ≤512 tokens (max_tokens). The model may produce fewer tokens due to EOS or other stop conditions.

Every token — input or output — contributes equally to KV cache size. In this benchmark, most of the KV cache derives from the input because prompt length (4096) far exceeds maximum generation length (512). This design ensures deterministic and immediate pressure on the KV cache during prefill, while keeping requests active long enough to observe decode phase dynamics.

This configuration (long input, moderate output) is standard in LLM serving benchmarks. It enables clear observation of KV cache saturation, scheduler behavior, and preemption events. Increasing output length would amplify these effects without modifying the fundamental system dynamics.

## Appendix B — Unexplored Parameters and Known Limitations

**kv_load_failure_policy.** By default, failed KV block loads from CPU fall back to local recompute. Setting it to _fail_ could isolate the pure connector effect by eliminating the recompute safety net. Not recommended in production but potentially useful for cleaner benchmark data.

**Chunked Prefill tuning (max_num_batched_tokens).** Controls how many tokens are processed per prefill step. Smaller values improve ITL but worsen TTFT, directly interacting with the OffloadingConnector's work to reinsert preempted requests. Not varied in this benchmark.

**CUDA Graphs with reduced max-model-len.** An alternative to --enforce-eager: re-enable CUDA graphs while further reducing max-model-len to compensate. CUDA graphs improve decode throughput; this configuration could produce higher overall throughput while still triggering frequent preemptions. The trade-off between decode throughput and preemption frequency was not explored.

**NIXL backend.** vLLM supports the NixlConnector for highly efficient cross-process KV transfers. Relevant for disaggregated prefill/decode multi-instance configurations, outside the scope of this single-GPU study.

**Hardware-specific applicability.** The RTX 4070 Super with PCIe 4.0 sits at a specific bandwidth tier where transfer cost and recompute cost are approximately comparable. On hardware with lower bandwidth (PCIe 3.0), we would expect offloading to become counterproductive as transfer costs exceed recompute costs. On hardware with substantially higher bandwidth (NVLink C2C, Grace Hopper), we would expect offloading benefits to increase significantly. These are projections based on bandwidth ratios; they have not been experimentally validated.

### Study Limitations

- Single consumer GPU with 12 GB VRAM
- PCIe 4.0 interconnect
- Single model (Qwen3-8B-AWQ)
- Single synthetic workload profile
- Random prompts (no prefix reuse)
- Focus on vLLM 0.15.1 behavior within this perimeter

Results should not be automatically extended to NVLink platforms, Grace Hopper, Blackwell, or to multi-turn workloads with shared prefixes.

**The practical conclusion for consumer hardware operators is clear: optimize the scheduler first, then evaluate offloading. Without the first step, the second risks remaining technically active but operationally marginal.**
