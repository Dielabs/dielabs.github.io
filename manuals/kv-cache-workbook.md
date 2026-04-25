---
layout: default
title: "KV Cache Manual"
---

# KV Cache Manual

_Version 1.1 — April 2026_  
_Author: Diego @ Dielabs_  
_Stack reference: Dielabs Inference Stack Model (L0–L6)_

---

## Table of Contents

1. [Why the KV cache is the real bottleneck](#1)
2. [Taxonomy: session, turn, request, prefill, decode](#2)
3. [The three layers of parameters](#3)
4. [Anatomy of the KV cache](#4)
5. [Request scope: the default and why it exists](#5)
6. [PagedAttention and pool management in vLLM](#6)
7. [Behavior under load](#7)
8. [Prefix caching](#8)
9. [Tuning parameters and operational trade-offs](#9)
10. [Policy levers: TTL, quotas, eviction](#10)
11. [Management profiles compared](#11)
12. [KV cache offload and storage hierarchy](#12)
13. [Software ecosystem 2026](#13)
14. [The economic cost of long chats](#14)
15. [Observability metrics](#15)
16. [Scale: from single-node to distributed inference](#16)
17. [Relevance for Dielabs and roadmap](#17)

---

<a id="1"></a>

## 1. Why the KV cache is the real bottleneck

In an LLM inference system the temptation is to attribute performance limits to the model weights — how big the model is, how much VRAM it takes. This is a wrong simplification. Model weights are fixed: they take space once and don't grow. The resource that determines effective concurrency, perceived latency and cost per token is the **KV cache**.

The KV cache is the structure that lets a transformer model generate tokens efficiently without recomputing the full prior history at every step. It lives in GPU VRAM — the fastest and scarcest space in the system — and it grows with context length and with the number of concurrent requests. When it runs out, the system enters degraded regimes: queueing, preemption, recomputation.

This manual starts from the foundational concepts and arrives at the 2026 software ecosystem (vLLM, LMCache, NIXL, Dynamo, CMX). It deliberately stays neutral with respect to hardware and workload: Dielabs-specific cases are collected in the final section.

---

<a id="2"></a>

## 2. Taxonomy: session, turn, request, prefill, decode

To reason correctly about the KV cache you have to keep two viewpoints distinct: the application side (what the user sees) and the inference engine side (what vLLM or an equivalent engine sees).

### Application side

| Level | Description |
|---|---|
| **Session / Conversation** | Contains N turns. It's an application-level concept. The engine has no visibility into the application session. |
| **Turn** | A back-and-forth: user message + model response. Each turn generates a request. |

### Inference engine side

| Level | Description |
|---|---|
| **Request** | A single HTTP call to `/v1/chat/completions`. The atomic unit for the engine. |
| **Prompt** | The content of the request that the model "reads". |
| **Prefill** | The phase in which the engine processes the entire prompt and builds the KV cache. |
| **Decode** | The phase in which the engine generates response tokens one by one, using the KV cache. |

### The envelope paradox

A session contains turns over time, but every request contains the entire session as data. At turn 20, the app re-sends the previous 19 turns inside the prompt of a single request. The engine only sees that request — it does not know it's the twentieth turn. This distinction is the key to understanding why the KV cache, by default, does not persist across successive turns: for the engine there are no turns, only independent requests.

---

<a id="3"></a>

## 3. The three layers of parameters

An HTTP request to an inference engine is not governed only by the parameters passed in the body. Three distinct layers contribute to the final behavior.

| Layer | Scope | When decided | Examples |
|---|---|---|---|
| **Model params** | Fixed in the model | At training / model selection | Number of layers, hidden state size, attention heads, vocabulary size |
| **Engine params** | Per engine instance | At vLLM launch | `max_model_len`, `gpu_memory_utilization`, `tensor_parallel_size`, `enable_prefix_caching`, `kv_cache_dtype` |
| **Request params** | Per single request | At every HTTP call | Prompt, `temperature`, `top_p`, `max_tokens`, `stop` |

Model params are not configurable at runtime, but they determine the structural cost of everything else. Engine params define the resource budget. Request params operate inside that budget.

### What consumes tokens in the prompt

System prompt, history of previous turns re-sent by the app, current user message, tool/function definitions, RAG context injected by the app, few-shot examples. All of these enter the prefill and occupy KV cache.

### What does not consume tokens

Sampling parameters (`temperature`, `top_p`, `top_k`, `max_tokens`, `stop`, `frequency_penalty`, `presence_penalty`) only influence decode, not prefill. They do not occupy KV cache.

---

<a id="4"></a>

## 4. Anatomy of the KV cache

During autoregressive generation, for every token the model computes two vectors: **K** (Key) and **V** (Value). Without a KV cache the model would have to recompute attention over the entire sequence at every step. The KV cache avoids this recomputation, drastically reducing the incremental cost per token.

The impact is concrete: lower TTFT on long prompts, more stable inter-token latency during generation.

### Per-token consumption formula

```
KV_per_token = 2 × num_layers × num_kv_heads × head_dim × dtype_bytes
```

The `dtype_bytes` factor depends on the precision configured for the KV cache (see `kv_cache_dtype` in §9), which is independent from the model weights' quantization:

| dtype | bytes | Notes |
|---|---|---|
| FP32 | 4 | Rarely used in serving |
| BF16 / FP16 | 2 | Default in most engines |
| FP8 | 1 | Halves KV consumption — requires Ada Lovelace+ |
| INT8 | 1 | Engine-dependent support |

For a 7B model with GQA in float16: `2 × 32 × 8 × 128 × 2 = ~128 KB per token`. With `kv_cache_dtype=fp8` the same model drops to ~64 KB per token.

### Consumption examples at realistic context lengths

At FP16:

| Model | 1K tokens | 10K tokens | 100K tokens | Notes |
|---|---|---|---|---|
| Qwen3-8B | 0.15 GB | 1.47 GB | 14.75 GB | Dense |
| LLaMA 3.3 70B | 0.33 GB | 3.28 GB | 32.77 GB | Dense |
| LLaMA 3.1 405B | 0.52 GB | 5.16 GB | 51.6 GB | Dense |
| DeepSeek-R1 685B | 0.07 GB | 0.7 GB | 7 GB | MoE |

> **Box — KV cache in MoE models.** In MoE (Mixture of Experts) models, the per-token cost depends on the architecture actually active. In many cases the KV footprint can be smaller than equivalent dense models, but it has to be verified model by model.

These numbers explain why the KV cache — not the model weights — is the real bottleneck on concurrency. At 8192 tokens of context a 7B model already takes ~1 GB of KV per single active sequence.

---

<a id="5"></a>

## 5. Request scope: the default and why it exists

**Without optimizations at L4a/L4b in the Dielabs Inference Stack Model, the KV cache is strictly request-scoped:**

- **Born** at prefill
- **Serves** the decode (avoiding recomputation of attention over all prior tokens at every new token generated)
- **Dies** at the end of the response

It does not persist across successive requests, not even within the same active session. From the inference engine's point of view there is no difference between an "active session" and a "historical session": both are new requests with full prefill.

This is the default — and it's a coherent design choice. Keeping the KV cache of millions of inactive sessions in VRAM is economically unsustainable. Everything we will see in sections 8, 12, 13 (prefix caching, offload, LMCache, Dynamo) is a layer on top of this default to selectively extend the cache lifetime where it pays off.

---

<a id="6"></a>

## 6. PagedAttention and pool management in vLLM

In a naive implementation, the KV cache is a contiguous block of memory pre-allocated per request. This causes high fragmentation and limits concurrency. **PagedAttention** is the innovation that defined vLLM: the KV cache is not allocated as a monolithic block but in **pages (blocks)** managed dynamically, in analogy with the virtual memory of an operating system.

### What occupies VRAM at boot

At startup, VRAM is split into three components:

- **Model weights** — fixed, determined by size and quantization (~5–6 GB for a 7–8B model in AWQ 4-bit)
- **KV cache pool** — fixed after boot, but the internal distribution changes dynamically
- **System overhead** — CUDA driver, PyTorch, GPU processes: about 0.5–1 GB

System overhead is the reason `gpu_memory_utilization` is never set to 1.0.

### The two phases of management

**Phase 1 — Pool reservation (at boot).** vLLM pre-allocates a fixed pool of KV blocks. The pool is fixed at boot and does not vary during the lifetime of the process.

```
KV Cache Pool = (total_VRAM × gpu_memory_utilization) − model_weights_VRAM
```

If the model is too large or `gpu_memory_utilization` is too high, vLLM fails **at boot** — not at runtime. If it starts, the system is stable.

**Phase 2 — Dynamic allocation (at runtime).** Within the fixed pool, PagedAttention assigns blocks dynamically:

- Every active sequence receives blocks on demand, not in advance
- The sequence grows block by block (typically 16 tokens at a time)
- When it terminates, the blocks are freed and reassigned

vLLM does **not** reserve space for `max_model_len` per request. If a request generates 200 tokens instead of 8192, it occupies ~13 blocks instead of 512.

> **Analogy.** The pool is a parking lot with N spots. Every car (request) does not reserve N spots at the entrance: it occupies one at a time as it advances. If the cars are small, far more than N can fit in the lot.

### Concrete effects of PagedAttention

- Lower fragmentation → GPU memory is not wasted in oversized blocks
- Higher concurrency → more requests can coexist on the same GPU
- Technical foundation for prefix reuse → identical KV pages can be shared across requests

### Worst-case concurrency in the logs

The value shown at startup (e.g. "Maximum concurrency for 8,192 tokens: 2.41x") is the **worst case**: how many requests would coexist if each used the full context window. In practice, with short prompts and limited output, effective concurrency is much higher.

---

<a id="7"></a>

## 7. Behavior under load

The scheduler of an inference engine decides at every step which requests to admit and which to hold back. KV cache saturation is only one of the constraints.

### Why requests wait

A request can stay in the queue **even with available blocks**. The scheduler evaluates multiple constraints:

| Constraint | What it controls |
|---|---|
| KV pool saturation | Available physical blocks |
| `max_num_seqs` | Concurrent sequence limit in the batch |
| `max_num_batched_tokens` | Token budget per iteration (especially prefill) |
| Chunked prefill | Latency protection for sequences in decode |

In a well-sized configuration, waiting is typically caused by batch and token-budget limits, not by memory. **Waiting is part of the admission control mechanism**: the client simply waits.

### Preemption: the last resort

When sequences in the batch grow faster than they can be released, waiting is not enough. The scheduler resorts to **preemption**: it suspends active requests, frees (or swaps to CPU) their KV blocks, and resumes them when possible.

Preemption is expensive (it requires recomputation or reload) and is the signal that the configuration is undersized for the load.

### Intervention hierarchy

```
Admission control (waiting)        ← normal behavior
        ↓ insufficient
KV cache pool saturation (waiting) ← memory constraint
        ↓ insufficient
Preemption (swap/recompute)        ← last resort, to be avoided
```

**Operational rule:** in a well-configured system, `num_preemptions_total` stays at zero.

---

<a id="8"></a>

## 8. Prefix caching

When two requests share the same prefix, the engine can reuse already-computed KV blocks instead of recomputing them. This is the first optimization that partially breaks request scope.

### Content-based, not session-aware

Prefix caching is **not tied to the session**. It does not know who you are, it does not know you are continuing a chat. It performs an **exact token-level match** of the prefix of the current request against prefixes already in cache, typically via content hashing (SHA-256).

Implications:

- If two different users send the same system prompt, the prefix is shareable
- If you change a single comma in the system prompt compared to the previous request, the match misses
- Reuse is purely content-based: same bytes/tokens → hit, any token-level difference invalidates the match

### Opportunistic lifetime

KV pages in cache do not have a configured TTL. Their lifetime is **opportunistic**, governed by:

- GPU memory pressure (more concurrent requests → more eviction)
- Eviction policy (LRU-like in vLLM)
- Model size and `gpu_memory_utilization`

Under low load, pages can persist for minutes. Under intense load they are evicted in seconds. There is no guarantee of reuse.

### Impact on benchmarking

Prefix caching distorts TTFT measurements: if you use the same prompt on multiple runs, the second run has an artificially low TTFT.

**Strategy:** vary the prompt across runs, or restart the container between one model and the next to reset the pool. Monitor `prefix_cache_hits_total / prefix_cache_queries_total` — ideally below 5% during benchmarks.

---

<a id="9"></a>

## 9. Tuning parameters and operational trade-offs

The two most relevant parameters for tuning — `max_num_seqs` and `max_model_len` — protect different aspects of the system:

| Parameter | Mainly protects | Sacrifices |
|---|---|---|
| `max_num_seqs` ↓ | **Latency** (per-request ITL) — fewer sequences in the batch → less GPU contention | Aggregate throughput |
| `max_model_len` ↓ | **Pool / sustainable throughput** — each sequence can grow less → more sequences in the pool | Long-context capacity |

In practice, `max_num_seqs` is the main operational lever: you use it to decide whether you are optimizing for latency (low value) or throughput (high value). `max_model_len` is set once based on the workload profile and left fixed.

### gpu_memory_utilization

Percentage of VRAM used as budget (weights + KV pool). The most important parameter. It only reduces the available KV budget — it does not free VRAM from the weights. Setting it too high causes OOM at boot.

```
--gpu-memory-utilization 0.85   # recommended on 12 GB consumer
```

### kv_cache_dtype

Controls the precision of K/V vectors in the pool, independently from the model weights' quantization. With `fp8` the per-token KV consumption is halved, effectively doubling the pool capacity at the same VRAM.

```
--kv-cache-dtype auto   # default: follows model dtype
--kv-cache-dtype fp8    # halves KV consumption — requires Ada Lovelace+ architecture
```

The impact on generation quality is generally negligible, but it has to be validated for the specific model.

### max_model_len

Maximum context window per request (input + output). Reducing it does not change the pool size, but it increases theoretical concurrency because every request can grow less.

| Value | Trade-off |
|---|---|
| 8192 | Supports long conversations, low worst-case concurrency |
| 4096 | Good compromise for latency/throughput benchmarking |
| 2048 | Maximizes concurrency, but truncates long contexts |

### max_num_seqs

Hard cap on simultaneously active sequences. Even if the pool would have room for more requests, serving too many in parallel degrades the ITL of each.

```
--max-num-seqs 256   # vLLM default (not recommended for benchmarks)
--max-num-seqs 64    # controlled benchmark
--max-num-seqs 8     # single user, optimal latency
```

### max_num_batched_tokens and chunked prefill

Limits the tokens processable in a single step. Chunked prefill (active by default since vLLM 0.15.x) splits long prompts into chunks to prevent a single prompt from blocking all other requests.

### Upstream guardrails

Engine parameters operate inside the backend. A complete strategy includes upstream limits: rate limiting (nginx, Traefik, LiteLLM), reject vs queue policy, protection against infinite chats (cap `max_tokens` per response, truncate history client-side, monitor `request_prompt_tokens`).

---

<a id="10"></a>

## 10. Policy levers: TTL, quotas, eviction

So far we've discussed engine behavior. Above the engine there is a **policy** dimension that decides how the KV cache is managed with respect to users, not with respect to the GPU. **These policies are not handled by the inference engine but by upper layers — gateway, orchestrator or cache manager** (typically L4a/L4b in the Dielabs Inference Stack Model).

**How long to keep the cache (TTL).** Low TTL means more repeated prefills but free VRAM and a stable system. High TTL means fewer prefills and better latency for the user, but VRAM occupancy and the risk of forced eviction.

**How much to keep per user (quotas).** Low quotas guarantee fairness and predictability. High quotas improve the experience for intensive users but risk a few users consuming all available VRAM.

**What to keep.** Prefix caching — keeping only the cache of static prefixes such as the system prompt — is almost always advantageous: low cost, guaranteed reuse. Full session caching, keeping the entire conversation, only pays off if the session is active and continuous.

**Where to keep it.** VRAM is optimal but scarce. System RAM offers more capacity but introduces transfer penalties to the GPU. Disk is viable only where latency is not critical. This hierarchy is formalized in §12.

**Eviction policy.** When VRAM saturates, you have to decide what to evict. LRU and TTL are simple and robust. More sophisticated policies keep in memory those who are actively generating or those with high probability of returning soon. In enterprise systems you can define service classes — critical users with higher TTL and lower eviction priority.

---

<a id="11"></a>

## 11. Management profiles compared

There is no universal configuration. Context determines everything. Three representative profiles:

**Hyperscale** — providers with millions of simultaneous users. Stateless by default, opportunistic prefix caching, maximum batching. The cache is freed after every response and rebuilt on the next turn. You pay in compute, you gain in scalability.

**On-prem enterprise** — a hundred employees, dedicated server, predictable load. Stateful with a window: TTL of 10–20 minutes, per-user quotas, LRU eviction, sticky routing to send each user to the same node where their cache is already present.

**UX-first** — few users, latency is the priority. Long TTL (60+ minutes), high quotas, reduced concurrency. You maximize the quality of the individual experience at the expense of hardware density.

### Recommended benchmark

To find the optimal point in your own context, define the three profiles and measure them:

- **Profile 1 — Hyperscale-like:** zero TTL, only prefix caching active
- **Profile 2 — Enterprise balanced:** TTL 10–20 minutes, per-user quota, LRU eviction
- **Profile 3 — UX-first:** TTL 60 minutes, higher quotas, reduced concurrency

Four KPIs: TTFT, tokens/s, peak VRAM utilization, cache hit-rate. Two hours of benchmarking yield a real curve of the trade-off.

---

<a id="12"></a>

## 12. KV cache offload and storage hierarchy

The key insight is simple: instead of evicting the KV cache and losing it, you move (offload) it to more capacious storage. When the session resumes, the system reads the KV cache from storage instead of recomputing it.

The necessary condition for this to make sense:

```
Storage read time < Prefill recomputation time
```

For long contexts this equation tends to be favorable — recomputation is too expensive. For short contexts the advantage is marginal.

### Tier hierarchy (NVIDIA nomenclature)

| Tier | Layer | Speed | Capacity | Use |
|---|---|---|---|---|
| G1 | GPU VRAM (HBM) | ~2 TB/s | 12–192 GB | Active sessions |
| G2 | CPU RAM (DRAM) | ~100 GB/s | 128–512 GB | Recently idle sessions |
| G3 | Local NVMe SSD | ~10–50 GB/s | TB-scale | Intermediate cache, pre-staging |
| G3.5 | CMX (ICMS) | ~10–50 GB/s | TB-scale | Pod-level dedicated tier |
| G4 | Network storage (RDMA, S3) | ~10–40 GB/s | Petabyte | Archived sessions, shared cache |

The G3.5 tier (CMX) was introduced by NVIDIA at CES 2026 as a dedicated layer: NVMe flash managed by BlueField-4 DPU, optimized specifically for ephemeral KV cache within an inference pod.

### Lifecycle with offload

With an active management layer, the KV cache follows a configurable lifecycle instead of being evicted and lost:

- **Active conversation** → KV in VRAM (G1)
- **Idle conversation** → offload to CPU RAM (G2), VRAM freed
- **Long-paused conversation** → migration to storage (G3/G4), CPU RAM freed
- **Resumed conversation** → read from storage and reload to VRAM, no prefill recomputation
- **Expired retention** → permanent deletion

---

<a id="13"></a>

## 13. Software ecosystem 2026

The ecosystem for KV cache management has expanded significantly compared to the simple vLLM stack. Four main components, each on a different layer of the Dielabs Inference Stack.

### Inference Engines (L3)

**vLLM** — open source inference engine, manages the KV cache in VRAM via PagedAttention. Default behavior: aggressive eviction. Includes integration with LMCache via the KV connector interface.

**SGLang** — alternative to vLLM, supports the same offload mechanisms. To be considered for specific scenarios (RadixAttention for efficient prefix sharing).

### KV Cache Management — LMCache (L4b)

A management layer that sits on top of inference engines. It adds configurable retention policies, RAM/storage hierarchy management, and the mechanism to recover the KV cache instead of evicting it. Open source, part of the PyTorch Foundation ecosystem.

Key features:

- Native integration in vLLM via the kv-connector interface
- Multimodal support (caching of vision tokens via mm_hashes hashing)
- S3 backend via AWS CRT
- Disaggregated prefill/decode with NIXL
- Content-addressed storage (hash-based lookup, no centralized cache map)

### Data Transport — NIXL (L4b)

NVIDIA Inference Transfer Library. **It is not limited to RDMA**: it provides a unified vendor-agnostic API that abstracts transfers between GPU, CPU and storage on heterogeneous backends (RDMA, GPUDirect Storage, NVMe-oF, object storage). Supported by AWS, Azure and Google Cloud. Includes dedicated benchmarking tools (NIXLBench for raw metrics, KVBench for LLM-specific profiling).

### Orchestration — NVIDIA Dynamo (L4a)

Open source distributed inference serving framework, version 1.0 at GTC 2026. It positions itself as an orchestration layer on top of inference engines — it does not replace them, it coordinates them in a multi-node system.

Key features:

- **Disaggregated serving**: prefill/decode separation on distinct GPUs
- **KV-aware routing**: routes requests toward GPUs that already have the KV cache
- **KV Block Manager (KVBM)**: memory management across the tier hierarchy
- **Dynamic GPU scheduling**: resource allocation based on real-time demand
- **Multimodal**: disaggregated encode/prefill/decode with embedding cache

Compatible with vLLM, SGLang, TensorRT-LLM. Adopted by AWS, Azure, GCP, OCI, Perplexity, PayPal, Pinterest, ByteDance.

### Infrastructure — NVIDIA CMX (L0–L1)

AI-native storage platform announced at CES 2026. It introduces a dedicated tier (G3.5) for ephemeral pod-level KV cache, based on NVMe flash managed by BlueField-4 DPU (availability planned for H2 2026), DOCA Memos as the SDK for management, Spectrum-X as the RDMA-accelerated fabric, and STX as the modular reference architecture. NVIDIA-claimed results: up to 5x throughput and 5x power efficiency vs traditional storage.

### Relationship between components

```
NVIDIA Dynamo (orchestration — routing, scheduling, disaggregation)
  ├── vLLM / SGLang / TensorRT-LLM (inference engine — generates tokens)
  ├── LMCache (KV cache management — retention, offload, sharing)
  ├── NIXL (data transport — moves KV cache between GPU, CPU, storage)
  └── CMX / BlueField-4 (storage infrastructure — dedicated G3.5 tier)
```

Mapping to Dielabs layers:

- **L0 (Hardware):** GPU, CPU, NVMe SSD, BlueField-4 DPU, Spectrum-X
- **L1 (Driver/GPU Runtime):** CUDA, DOCA, GPUDirect
- **L3 (Inference Backend):** vLLM, SGLang, TensorRT-LLM
- **L4a (Request Orchestration):** Dynamo
- **L4b (GPU Workload Optimization):** LMCache, NIXL

---

<a id="14"></a>

## 14. The economic cost of long chats

Prefill has computational cost approximately **O(n)** with respect to prompt length (where n = number of input tokens). Since at every turn the app re-sends the entire history, the prefill cost grows linearly with the number of turns.

| Turn | Tokens in history (example) | Relative prefill cost |
|---|---|---|
| 1 | ~200 (system + message) | 1× |
| 5 | ~2,000 | 10× |
| 20 | ~10,000 | 50× |
| 50 | ~25,000 | 125× |

Without prefix caching, every turn pays the full cost. Direct implications:

- **Token economics**: per-turn cost is not constant. Providers bill input tokens → per-turn cost grows linearly, but cumulatively in a significant way over multi-turn conversations
- **Agent design**: multi-step agents with many turns accumulate significant prefill costs. Context window design (what to keep, what to discard, what to summarize) becomes an economic choice, not just a technical one
- **Window management**: strategies like sliding window, summarization of old turns, or history truncation exist precisely to contain this growth

The intra-request KV cache masks the problem during decode, but does not eliminate it: prefill is the structural cost of long conversations. The optimizations of §8 (prefix caching) and §12–13 (offload, LMCache) are ultimately attempts to break this linear growth.

---

<a id="15"></a>

## 15. Observability metrics

The Prometheus metrics exposed by vLLM (and by compatible engines) are the fundamental observation point.

| Metric | Type | What to observe |
|---|---|---|
| `kv_cache_usage_perc` | Gauge | > 80% sustained → saturation risk |
| `num_requests_running` | Gauge | Compare with configured `max_num_seqs` |
| `num_requests_waiting` | Gauge | > 0 sustained → system under pressure (verify with `kv_cache_usage_perc` to distinguish from normal admission control) |
| `num_preemptions_total` | Counter | > 0 → configuration to be revisited |
| `prefix_cache_hits_total` / `prefix_cache_queries_total` | Counter | Hit rate — during benchmark ideally < 5% |
| `time_to_first_token_seconds` | Histogram | Monitor p50 and p99; large gap = variance from concurrency or prefix caching |
| `inter_token_latency_seconds` | Histogram | Rises with concurrency: more requests → more crowded decode batch |
| `e2e_request_latency_seconds` | Histogram | Sum of queue time + prefill + decode |

**Patterns to recognize:**

- High `kv_cache_usage_perc` + `num_requests_waiting` > 0 + stable `num_preemptions_total` → system in controlled saturation, probably to relax with `max_num_seqs` or `max_model_len`
- `num_preemptions_total` growing → undersizing, urgent intervention
- `time_to_first_token_seconds` p99 much greater than p50 → high variance, probably noisy prefix cache hits/misses or full queues

---

<a id="16"></a>

## 16. Scale: from single-node to distributed inference

The mechanism is identical at all scales. What changes is absolute scale and the additional optimizations available.

| Aspect | Consumer single-node | Datacenter single-node | Datacenter multi-node |
|---|---|---|---|
| Reference GPU | RTX 4070/4090 (12–24 GB) | H100 (80 GB) | H100/B200 cluster |
| KV Cache pool | ~5 GB | ~50–60 GB | TB-scale aggregate |
| Worst-case concurrency (8K tok) | 2–3 requests | 50–100+ | Thousands |
| Tensor Parallelism | Not applicable | Optional | Standard |
| KV cache offload | Not necessary | Optional | Critical |
| Disaggregated prefill/decode | Not applicable | Not applicable | Standard with Dynamo |
| Orchestration layer | Direct | Optional | Required |

The single-node ceiling is physical, tied to available VRAM: VRAM out, concurrency out. You don't go past it without adding GPUs. The transition from single-node to multi-node is not an incremental upgrade: it introduces an orchestration layer (Dynamo or equivalent) that changes the mental model of request management.

---

<a id="17"></a>

## 17. Relevance for Dielabs and roadmap

The current lab setup (single-node vLLM, consumer GPU, mostly single-user traffic) does not yet justify LMCache, Dynamo, or KV cache offload. The immediate eviction observed in Grafana is the correct behavior for this context — there isn't enough concurrency to justify the extra complexity.

### When it will become relevant

- When the lab gets real multi-user traffic with concurrent sessions
- When RAG workloads with very long contexts (>10,000 tokens) are explored
- When agentic workflows with extended multi-turn conversations are implemented
- When a second GPU node is added and disaggregated prefill/decode can be experimented with
- When the robotics project with Qwen3-VL uses repeated vision sessions — LMCache supports caching of vision tokens, reducing TTFT for recurring webcam images

### Dielabs roadmap

- **Short term:** keep the current vLLM setup as a baseline, document the prefix caching behavior across existing benchmarks
- **Medium term:** experiment with LMCache as an additional layer on top of vLLM, starting from official documentation and the Dell Technologies reference as a methodological baseline
- **Medium-long term:** study NVIDIA Dynamo as the framework for disaggregated serving in view of an eventual second GPU node
- **Market observation:** monitor the availability of CMX/BlueField-4 (H2 2026) as a potential professional-positioning topic — the chain NIXL → Dynamo → CMX → BlueField-4 is an area where infrastructure engineering and inference engineering competencies converge

### Useful references

- LMCache: [github.com/LMCache/LMCache](https://github.com/LMCache/LMCache) | [docs.lmcache.ai](https://docs.lmcache.ai/) | Paper [arXiv:2510.09665](https://arxiv.org/abs/2510.09665)
- NIXL: [github.com/ai-dynamo/nixl](https://github.com/ai-dynamo/nixl)
- Dynamo: [github.com/ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo) | [developer.nvidia.com/dynamo](https://developer.nvidia.com/dynamo)
- Dell Technologies — _Scaling Multi-Turn LLM Inference with KV Cache Storage Offload_ (January 2026)
- VAST Forward 2026 — _Breaking Through the GPU Memory Wall_

---

*dielabs.github.io · github.com/Dielabs*
