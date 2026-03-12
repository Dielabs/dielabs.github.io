---
layout: default
title: "KV Cache Offloading Research"
---

**DIELABS Discovery Paper**

DDP-2026-001

**KV Cache Offloading on Consumer Hardware**

**Analysis, Benchmarks, and Real Limits**

*An experimental analysis of KV cache offloading mechanisms in vLLM 0.15.1*
*on NVIDIA RTX 4070 Super (12 GB VRAM, PCIe 4.0)*

March 2026

---

# Abstract

This paper documents an experimental investigation into KV cache offloading mechanisms available in vLLM 0.15.1, conducted on a consumer GPU node (NVIDIA RTX 4070 Super, 12 GB VRAM, PCIe 4.0 x16).

The initial objective was to measure the performance benefit of KV cache offloading to CPU RAM under load via GuideLLM benchmarks.

The investigation revealed three main findings: **(1)** LMCache, commonly described as an "offloading" solution, actually operates as an external prefix cache and not as a swap mechanism; **(2)** vLLM's Native OffloadingConnector (introduced in v0.11.0) reintroduces true offloading for preemption, but requires specific conditions to activate; **(3)** on consumer hardware with PCIe 4.0, the measurable benefit manifests on tail latency (ITL p99 reduced by 23%) but not on average throughput.

The paper expanded beyond initial predictions, also describing the methodological path and unexpected findings that led to these initially unforeseen conclusions.

Formulas are introduced to support key concepts, but the paper remains readable without examining them in detail. The text assumes basic familiarity with inference engineering fundamentals.

---

# 1. Introduction

The KV cache is the most critical data structure in LLM inference. For each request, the model generates key-value pairs for every processed token, whose size grows linearly with context length. On consumer GPUs with limited VRAM (12 GB), the KV cache becomes the primary concurrency bottleneck.

The idea of "offloading" — moving part of the KV cache from GPU to CPU RAM, exploiting the greater capacity of system memory — is conceptually simple and often promoted as a solution to extend serving capacity on limited hardware. However, the term "offloading" hides significant architectural complexity.

This paper originates from a practical attempt to trigger and measure KV cache offloading in vLLM on a real homelab. The path revealed that the concept of "offloading" encompasses at least three completely different mechanisms, each with distinct assumptions and use scenarios.

---

# 2. Technical Background

## 2.1 KV Cache and Admission Control

For each processed token, the model generates a key-value pair. The KV cache size for a single request is:

```
kv_cache_per_request = num_tokens × num_layers × 2 × num_kv_heads × head_dim × bytes_per_element
```

For Qwen3-8B-AWQ (36 layers, 8 KV heads, head_dim 128, FP16), each token occupies approximately 144 KB of KV cache. A request with 4096 input tokens immediately generates ~576 MB of KV cache during prefill.

Available VRAM for KV cache is calculated by subtracting all fixed allocations:

```
kv_cache_budget = (VRAM_total × gpu_memory_utilization) − model_weights − cuda_graphs − overhead
```

The vLLM scheduler at startup (V1 architecture) calculates maximum theoretical concurrency as:

```
max_concurrency = kv_cache_budget / (max_model_len × kv_per_token_weight)
```

This value represents the **worst case**: how many requests can coexist if all simultaneously reach `max_model_len` tokens. If VRAM is insufficient to guarantee this space for a new request, the scheduler places it in the waiting queue.

## 2.2 Preemption and Recompute

Preemption is the resource management mechanism that intervenes when KV cache space becomes insufficient for all active requests. Because of the autoregressive nature of Transformer models, the KV cache of each request grows with every generated token. If the sum of these growths exceeds available physical GPU memory, the scheduler must suspend some requests to free space.

Understanding preemption is fundamental because it represents the exact point where offloading can intervene: the cost of preemption is what offloading attempts to reduce *(from full recompute to a simple CPU-RAM to GPU-RAM transfer)*.

**Recompute: the default mechanism in V1**

Unlike V0, which supported swapping (transferring KV blocks from GPU to CPU RAM), vLLM V1 in standard configuration uses exclusively the **recompute** mechanism. When a request is selected for preemption, its memory blocks are freed and the request is returned to the waiting queue. When space becomes available again, the request is rescheduled and its prefill is recalculated from scratch.

**Life cycle of a preempted request**

The complete preemption cycle traverses four phases:

1. **Saturation:** The scheduler detects insufficient free blocks for the next decode step.
2. **Eviction:** The scheduler selects one or more requests to suspend, following a priority-based or FCFS policy.
3. **Block release:** The KV blocks associated with the request are returned to the free queue. Thanks to prefix caching, these blocks are not immediately deleted but remain in cache identified by hash, until overwritten by new allocations.
4. **Resumption:** When memory frees up, the request returns to execution. If its blocks are still in cache (cache hit), vLLM can skip part of the recalculation. If blocks were overwritten, prefill is fully recalculated.

**Performance impact**

Preemption guarantees system robustness by avoiding OOM errors, but has significant cost:

- **Latency:** Increases end-to-end latency of the suspended request due to rescheduling and prefill recalculation time.
- **Inter-token latency:** Non-preempted requests experience increased ITL during recompute of a rescheduled request, because the GPU must execute a compute-bound prefill job in the middle of other requests' decode steps.
- **Tail latency:** *Preemption disproportionately impacts p99 metrics.* Most requests complete normally, but those undergoing preemption see latencies significantly above average.

This asymmetry between average case and worst case explains why offloading produces measurable benefits specifically on tail latency (p99) rather than on average throughput — as confirmed by benchmarks in this study.

## 2.3 The Three Offloading Mechanisms

The investigation identified three distinct mechanisms, often confused under the generic term "KV cache offloading." Each has radically different architectural assumptions, activation conditions, and use scenarios.

**A) LMCache — External Prefix Cache**

LMCache operates as an external cache of already-computed KV values, organized by token prefix hash. It is not integrated with the vLLM scheduler and does not modify its behavior.

**When it activates (store):** Every time vLLM completes the prefill of a request, LMCache captures the generated KV blocks and saves them to its backends (CPU RAM, local disk, Redis, remote storage).

**When it activates (retrieve):** When a new request is admitted by the scheduler, LMCache checks if the request's token prefix matches (via hash match) blocks already in cache. If the match exceeds the minimum threshold (`min_retrieve_tokens`), blocks are loaded to GPU, avoiding prefill recalculation for that portion.

**When it does NOT activate:** If requests have unique prefixes (random prompts, no shared system prompts, no repeated RAG context), no hash match ever occurs. Store happens anyway, and in this scenario LMCache brings no benefit.

**B) Native OffloadingConnector — Preemption Recovery**

Introduced in vLLM v0.11.0 (RFC #19854), the OffloadingConnector is a native component of vLLM's KV Transfer framework. Its primary function is saving the KV blocks of requests to CPU RAM, allowing their reload instead of recompute when a request is preempted and then rescheduled.

**Proactive store mechanism (continuous mirroring)**

The most important — and least intuitive — aspect of the OffloadingConnector is that the store **does not wait for preemption** to activate. The connector operates as a **real-time mirroring system** that copies KV blocks from GPU to CPU during normal request execution:

- **Streaming layer-by-layer:** Not as soon as an attention calculation for a single layer is completed, the connector initiates the transfer of those KV blocks toward CPU RAM via DMA, while the GPU is already computing the next layer.
- **Transfer via DMA:** The copy does not use GPU compute cores (Streaming Multiprocessors). The transfer is managed by the PCIe bus DMA engine, which operates completely independently of compute.
- **Ready-to-preempt state:** Thanks to this continuous copy, if the scheduler suddenly decides to preempt a request, CPU RAM already contains an updated (or nearly complete) copy of its KV cache.

**The mirroring cost: an insurance premium**

This proactive mechanism has a continuous cost even when preemption does not occur. The system constantly consumes:

- **PCIe bandwidth:** Constant GPU→CPU data transfer during prefill and decode.
- **CPU RAM (reserved memory):** Blocks are written to page-locked memory. In our setup, host RAM goes from ~5 GB to ~13 GB with offload active.
- **GPU memory bandwidth (HBM):** DMA must read blocks from VRAM to send them via PCIe.

vLLM considers this cost an acceptable **"insurance premium"**: the system prefers to use "cheap" resources (CPU RAM and often-underutilized PCIe bandwidth) to avoid the "very expensive" cost of full recompute in case of preemption. This explains the 9% overhead on TTFT observed in benchmarks.

**When it activates (load):** The load is the only *reactive* operation of the connector. It is triggered only when a preempted request is rescheduled. The scheduler signals the connector to reload blocks from CPU to GPU. Transfer happens via PCIe, layer-by-layer, asynchronously.

**When the load does NOT activate:** If no preemptions occur, the proactive store happens anyway (CPU RAM fills up) but the load never fires because there are no preempted requests to restore.

**C) FlexGen / InfiniGen / KVSwap — Full KV Swap**

Research frameworks implementing a complete swap of KV cache (and in some cases model weights) between GPU, CPU, and disk. Designed for offline throughput scenarios (batch processing on very large models) where per-request latency is not critical. Not integrable as vLLM backends for online serving.

**Summary comparison:**

| **Characteristic** | **LMCache** | **Native Offload** | **FlexGen/InfiniGen** |
|---|---|---|---|
| Type | Prefix cache | Preemption recovery | Full KV swap |
| vLLM integration | External plugin | Native (v0.11.0+) | Separate framework |
| Store trigger | Every completed prefill | Proactive (continuous mirroring) | Always (by design) |
| Load trigger | Prefix hash match | Preempted request rescheduled | Every attention step |
| Requires shared prefixes | Yes (essential) | No | No |
| Requires preemption | No | Yes (essential) | No |
| Optimal scenario | RAG, multi-turn, system prompt | High concurrency, burst | Offline batch, large models |
| Worst scenario | Unique random prompts | Low concurrency | Interactive serving |

---

# 3. Experimental Setup

## 3.1 Hardware

| **Component** | **Specification** |
|---|---|
| GPU | NVIDIA RTX 4070 Super (12 GB GDDR6X) |
| GPU Interface | PCIe 4.0 x16 (~21 GB/s bidirectional) |
| CPU RAM | 32 GB DDR4 |
| Server/CPU | Workstation, Intel i7, 8 performance cores |
| OS | Ubuntu 24.04, NVIDIA Driver 590 |

## 3.2 Software

| **Component** | **Version** |
|---|---|
| vLLM | 0.15.1 (V1 architecture) |
| LMCache | Integrated in custom Docker image |
| Model | Qwen3-8B-AWQ (AWQ Marlin quantization) |
| Benchmark tool | GuideLLM 0.5.3 |
| Observability | Prometheus + Grafana + DCGM Exporter |
| Container runtime | Docker Compose |

## 3.3 Benchmark Workload

All tests use the same GuideLLM workload for comparability:

- **Profile:** concurrent, rate 10 requests/second
- **Duration:** 300 seconds per run
- **Prompt:** 4096 input tokens + 512 output tokens (synthetic, random, no shared prefixes)
- **Collected metrics:** TTFT, ITL, TPOT, throughput, KV cache %, host RAM, requests running/waiting (via Grafana/Prometheus)

---

# 4. Phase 1: LMCache — The Prefix Cache Misunderstanding

## 4.1 Initial Hypothesis

The starting hypothesis was that LMCache operated as an overflow mechanism: when the GPU exhausts KV space, blocks are moved to CPU RAM, allowing the scheduler to admit more requests. Under this view, offloading should increase throughput under pressure.

## 4.2 Discovery

Analysis of official documentation revealed a completely different picture:

- **Admission Control unchanged:** LMCache does not modify scheduler admission logic. vLLM continues to calculate maximum concurrency based exclusively on available GPU blocks.
- **Store/Retrieve logic:** LMCache saves computed KV values (store) and retrieves them (retrieve) only when a subsequent request presents an already-seen prefix (hash match). It operates as a CDN for KV cache, not as a swap.
- **No bypass:** The vLLM scheduler does not know CPU RAM exists.

## 4.3 Implication for GuideLLM Benchmark

The GuideLLM workload generates random prompts without shared prefixes. Under these conditions, LMCache never finds a cache hit and offloading produces no effect.

---

# 5. Phase 2: The Native OffloadingConnector

## 5.1 Identification

Research in vLLM documentation and the official blog (January 2026, author Or Ozeri of IBM Research) revealed a second mechanism: the native OffloadingConnector, introduced in v0.11.0 (RFC #19854).

## 5.2 Architecture

The Native OffloadingConnector operates with an asymmetric two-phase architecture: a **proactive and continuous store** (GPU→CPU) and a **reactive and on-demand load** (CPU→GPU).

**Comparison: Recompute vs OffloadingConnector**

| **Aspect** | **Recompute (default)** | **OffloadingConnector** |
|---|---|---|
| Operation at resumption | Recalculates entire prefill | Reloads blocks from CPU via PCIe |
| Resource used | GPU SMs (compute-bound) | DMA + PCIe bandwidth (I/O-bound) |
| Impact on other requests | High: prefill competes for SMs | Low: DMA does not use SMs |
| Cost during execution | None | Continuous mirroring (insurance premium) |
| Cost at preemption | Full prefill recalculation time | Only GPU deallocation |
| Cost at resumption | Prefill time (compute) | PCIe transfer time (I/O) |

The key difference is in the resource consumed: recompute is **compute-bound** (competes with decode of other requests for GPU cores), while reload is **I/O-bound** (uses PCIe bus, which is largely unused during decode). This explains why the benefit manifests specifically on ITL p99.

## 5.3 First Failed Test (Conservative Configuration)

The first test with Native OffloadingConnector used the same *conservative* configuration as LMCache tests: `gpu-memory-utilization=0.75`, `max-model-len=8192`. Result: performance identical to vanilla. Host RAM rose from 5.5 GB to 13.6 GB (confirming blocks were being reserved), but TTFT, throughput, and latency were unchanged.

---

# 6. The Scheduler Paradox

Analysis of startup logs revealed the fundamental problem: with conservative configuration, vLLM reported maximum concurrency of **1.36x** for 8192-token requests. The scheduler admitted at most 2 simultaneous requests out of 10 concurrent requests sent.

## 6.1 The Admission Control Formula

```
max_concurrency = kv_cache_size / max_model_len

kv_cache = (VRAM_total × gpu_memory_utilization) − model_weights − cuda_graphs − overhead
```

**Conservative configuration:**
```
kv_cache = (12 GB × 0.75) − 5.7 GB − 0.6 GB − overhead ≈ 1.5 GB
max_concurrency ≈ 1.36x
```

**Aggressive configuration:**
```
kv_cache = (12 GB × 0.95) − 5.7 GB − 0 GB (eager) − overhead ≈ 4.1 GB
max_concurrency ≈ 4–5x
```

## 6.2 The Static Scarcity Vicious Cycle

The initial intuition — "provide less VRAM to the KV cache to force offloading" — produced the opposite effect:

- Less VRAM for KV → fewer blocks available for the scheduler
- Fewer blocks → fewer simultaneously admitted requests
- Fewer active requests → less KV cache conflict
- Less conflict → no preemption
- No preemption → offloading never activates

**In summary:** Static resource scarcity *prevents* the dynamic conflict necessary to activate offloading.

## 6.3 The Solution: Dynamic Over-Subscription

The correct strategy was the opposite of what was hypothesized: maximize the number of admitted requests, so the KV cache saturates **during generation**, forcing preemptions. Three simultaneous modifications:

- **`--gpu-memory-utilization 0.95`** (from 0.75): VRAM pool for KV cache went from ~1.5 GB to ~4.1 GB
- **`--enforce-eager`:** Disables CUDA graphs, freeing ~0.6 GB of fixed VRAM
- **`--max-model-len 5120`** (from 8192): Scheduler reserves less space per request, admitting more within the same budget

## 6.4 The Optimistic Scheduler

A crucial aspect emerging from data: the vLLM V1 scheduler admitted **up to 7 concurrent requests**, exceeding the theoretical maximum concurrency of 4-5x. This behavior was validated with vLLM technical guides.

The `max_concurrency` value reported in startup logs represents the limit in the **worst case** — the scenario where every request instantly occupies the entire `max_model_len` length. This is for capacity planning, **not as a hard admission limit**.

The V1 scheduler operates with an **on-demand allocation** mechanism: physical KV cache blocks are allocated only when actually needed, not pre-allocated for entire `max_model_len` at admission time.

The timeline generating over-subscription:
- **t₀:** Request A arrives. Scheduler sees free blocks, admits it.
- **t₁:** While A is in partial prefill, B and C arrive. A's real occupancy is less than `max_model_len`. Scheduler admits them.
- **t₂–t₄:** 6–7 requests are now active in different phases. Real occupancy of each is below `max_model_len`. KV cache grows in parallel. Total sum exceeds physical budget → preemption.

Grafana data confirm: **5.9 requests running average, max 7** (vs. 2.1 average in previous configuration), with KV cache at **98.9% in vanilla and 99.9% with offload** — near-complete saturation.

---

# 7. Benchmark Results

## 7.1 Conservative Configuration Comparison

*Configuration: `gpu-memory-utilization=0.75`, `max-model-len=8192`, CUDA graphs enabled*

| **Metric** | **Vanilla** | **Native Offload** | **Delta** |
|---|---|---|---|
| Requests completed | 63 | 63 | = |
| Requests running (mean) | 2.1 | 1.9 | -10% |
| **TTFT mean (ms)** | 36,040 | 37,017 | +2.7% |
| TPOT mean (ms) | 86.5 | 88.3 | +2.1% |
| Output tok/s | 107.1 | 104.9 | -2.1% |
| ITL mean (ms) | 16.2 | 16.0 | -1.2% |
| **RAM host (GB)** | 5.51 | 13.59 | +8.08 GB |
| KV cache % max | 87.2% | 81.9% | |

**Analysis:** Offloading is active (RAM jumps 8 GB) but produces no measurable benefits. The scheduler admits only 2 requests — no dynamic pressure on the KV cache, no preemptions.

## 7.2 Aggressive Configuration Comparison

*Configuration: `gpu-memory-utilization=0.95`, `max-model-len=5120`, `--enforce-eager`*

| **Metric** | **Vanilla Aggr.** | **Offload Aggr.** | **Delta** |
|---|---|---|---|
| **Requests completed** | **103** | **103** | = |
| Requests running (mean) | 5.9 | 5.5 | -7% |
| TTFT mean (ms) | 12,202 | 13,352 | +9.4% |
| TTFT median (ms) | 16,325 | 17,857 | +9.4% |
| Request latency mean (s) | 27.60 | 27.75 | = |
| Output tok/s | 175.4 | 173.5 | -1% |
| Throughput req/s | 0.34 | 0.34 | = |
| ITL mean (ms) | 30.1 | 28.2 | -6.5% |
| **ITL p99 (ms)** | **328.9** | **253.0** | **-23%** |
| **E2E latency p99 (s)** | **38.60** | **35.59** | **-7.8%** |
| KV cache % max | 98.9% | 99.9% | |
| **RAM host (GB)** | 4.81 | 12.98 | +8.17 GB |

**Analysis:** With aggressive configuration, the picture changes significantly. Average throughput remains identical, but **tail latency improves**: ITL p99 drops 23% (from 329 ms to 253 ms) and E2E p99 drops 7.8%. TTFT mean worsens by 9.4%, indicating connector overhead from saving blocks to CPU even when not immediately necessary.

## 7.3 Scheduler Configuration Impact

Comparison between the two vanilla configurations shows the effect of unlocking the scheduler:

| **Metric** | **Vanilla Conserv.** | **Vanilla Aggr.** | **Delta** |
|---|---|---|---|
| **Requests completed** | 63 | **103** | **+63%** |
| **Requests running** | 2.1 | **5.9** | **+181%** |
| **TTFT mean (ms)** | 36,040 | **12,202** | **-66%** |
| Output tok/s | 107.1 | **175.4** | **+64%** |

The aggressive configuration produces dramatic improvement on all metrics, independent of offloading. This demonstrates that on consumer hardware with limited VRAM, **vLLM parameter tuning has an impact orders of magnitude larger than enabling offloading**.

## 7.4 Two Distinct Phenomena: Scheduler Oversubscription and Offloading

**The scheduler determines how many requests can run.** The most significant performance improvement (+63% completed requests, -66% TTFT) **does not come from offloading, but from unlocking the scheduler in the aggressive configuration.**

**Offloading determines how much it costs to stop and resume a request.** Offloading comes into play only after the scheduler has admitted enough requests to generate preemptions. At that point, the Native OffloadingConnector reduces the resumption cost (reload from CPU instead of recompute), improving tail latency without altering average throughput.

The complete causal chain:

```
oversubscription → KV cache pressure → preemption → useful offloading
```

Without the first link (oversubscription), the chain does not activate.

---

# 8. Lessons Learned

## 8.1 The Term "Offloading" is Misleading

"KV cache offloading" is used interchangeably to describe at least three architecturally different mechanisms: prefix caching (LMCache), preemption recovery (Native OffloadingConnector), and full KV swap (FlexGen). This terminological ambiguity generates incorrect expectations, especially among infrastructure engineers who associate "offload" with the concept of memory swap/overflow.

## 8.2 Admission Control Governs Everything

The vLLM scheduler is the primary gatekeeper. Any offloading mechanism operates *downstream* of admission, not as a replacement for it. No currently available mechanism in vLLM allows bypassing admission control to admit more requests than the GPU can physically handle.

## 8.3 The Static Scarcity Paradox

Reducing available VRAM does not force offloading — it prevents it. Preemption requires dynamic conflict between active requests. If the scheduler admits few requests, the conflict never occurs. The correct strategy is dynamic over-subscription: admit many requests and let the KV cache saturate during generation.

## 8.4 PCIe 4.0 is the Bottleneck

On consumer hardware with PCIe 4.0 (~21 GB/s), the CPU→GPU transfer cost for reloading KV blocks is comparable to recompute cost. The offloading benefit manifests only on tail latency (p99), not average throughput. On hardware with faster interconnects (NVLink C2C at 900 GB/s in Grace Hopper/Blackwell systems) the balance would change radically.

## 8.5 Parameter Tuning Outperforms Offloading

Moving from conservative to aggressive configuration (`--gpu-memory-utilization`, `--enforce-eager`, `--max-model-len`) produced 63% improvement on completed requests and 66% on TTFT — without activating any offloading. On consumer hardware, **optimizing base vLLM parameters has impact orders of magnitude larger than enabling experimental features**.

## 8.6 The Value of Negative Findings

Demonstrating that a feature produces no benefits in a specific scenario is a valid research contribution. Few documents in the LLM inference community critically analyze the real limits of offloading on consumer hardware. Most benchmarks and guides assume datacenter hardware (H100, A100) with radically different bandwidth and memory characteristics.

---

# 9. Conclusions

- **LMCache** operates as an external prefix cache, not as a swap. Useful for workloads with shared prefixes (RAG, multi-turn) but invisible with random prompts.
- **The Native OffloadingConnector** implements true offloading for preemption, with asynchronous layer-by-layer architecture. On RTX 4070 Super with PCIe 4.0, it reduces tail latency (ITL p99) by 23% and E2E p99 by 7.8%, but does not improve average throughput.
- **The vLLM scheduler** is the dominant factor. With conservative configuration, no offloading mechanism produces effects. With aggressive configuration, the offloading benefit is real but marginal compared to the improvement obtained from parameter tuning alone.

**Recommendation for consumer hardware operators:** Before investing time in configuring offloading mechanisms, optimize base vLLM parameters (`gpu-memory-utilization`, `max-model-len`, `enforce-eager`). The benefit will be orders of magnitude larger. Native offloading becomes relevant only after maximizing scheduler concurrency, and its real value will manifest with hardware evolution toward faster CPU-GPU interconnects.

## 9.1 Future Work

- **Test with prefix-sharing workloads:** Design a benchmark with shared prefixes (simulating RAG) to measure LMCache effect in its optimal scenario.
- **Analysis on DGX Spark:** Repeat benchmarks on hardware with unified memory (128 GB, 273 GB/s) to quantify bandwidth impact on offloading.
- **Progressive KV Cache Onloading:** Monitor vLLM RFC #33526 (February 2026) introducing progressive block loading, reducing head-of-line blocking.
- **Dedicated monitoring:** Develop Prometheus/Grafana metrics to trace OffloadingConnector store/load events in real time.

---

# Appendix A — Input and Output Length Choice in Benchmarking

Input length was fixed at 4096 tokens; output generation was limited to a maximum of 512 tokens. This value represents an **upper bound** (`max_tokens`). The model can produce up to 512 tokens, but generation may stop earlier due to EOS or other stop conditions.

From a memory perspective, every token — whether input or output — contributes equally to KV cache size:

```
KV_cache ∝ input_tokens + output_tokens
```

In this benchmark, most KV cache comes from input because prompt length (4096 tokens) is much larger than maximum generation length (512 tokens). Each request enters the system with an already significantly populated KV cache during the prefill phase.

**Temporal KV cache allocation:**
- **Prefill (input):** KV cache for input tokens is created immediately during prefill. Generates high, deterministic initial memory pressure.
- **Decode (output):** During generation, KV cache grows progressively, one token at a time.

This combination allows clearly observing phenomena such as KV cache saturation, scheduler behavior, and preemption events. This `4096 input / ≤512 output` configuration is standard in serving benchmarks and proved particularly suited to this study's objective.

---

# Appendix B — Unexplored Parameters and Known Limitations

## kv_load_failure_policy

By default, if KV block loading from CPU fails, vLLM falls back to local recompute. Setting this parameter to `fail` could isolate the pure connector effect by eliminating the recompute fallback. Not recommended in production but could provide cleaner benchmark data.

## Chunked Prefill Tuning (max_num_batched_tokens)

In vLLM V1, chunked prefill is active by default. The `max_num_batched_tokens` parameter controls how many tokens are processed in a single prefill step. Smaller values improve ITL but worsen TTFT, directly interacting with the OffloadingConnector's work of reinserting preempted requests into the batch. This parameter was not varied in our benchmark.

## CUDA Graphs with Reduced max-model-len

The `--enforce-eager` flag was used to free VRAM from CUDA graphs. An alternative would be to re-enable CUDA graphs while further reducing `max-model-len` to compensate. CUDA graphs improve decode throughput, and this configuration might produce higher overall throughput while still triggering more frequent preemptions. The trade-off between decode throughput and preemption frequency was not explored.

## NIXL Backend

Beyond the `native` backend, vLLM supports the NixlConnector, designed for extremely efficient cross-process KV transfers. In multi-instance configurations (disaggregated prefill/decode), NIXL would offer lower latency through integration with advanced transport libraries. This scenario falls outside the scope of the present study (single instance, single GPU) but represents a future research area.

**Study limitations:**

- Single consumer GPU with 12 GB VRAM
- PCIe 4.0 interconnect
- Single primary model
- Single synthetic workload profile
- Random prompts, no prefix reuse
- Focus on vLLM 0.15.1

Results should not be automatically extended to platforms with NVLink, Grace Hopper, Blackwell, or environments with multi-turn workloads and shared prefixes.

The practical conclusion, for those working on consumer hardware, is clear: **optimize the scheduler first, then evaluate offloading**. Without the first step, the second risks remaining a technically active but operationally marginal mechanism.

---

*Dielabs — [dielabs.eu](https://dielabs.eu)*

