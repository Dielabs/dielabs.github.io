---
layout: default
title: "KV Cache Workbook"
---

# KV Cache Workbook

Every time a model generates text, it needs to remember what it has already processed. The KV Cache is the structure that holds this memory — and it is the primary factor determining how many requests a server can serve in parallel.

This document explains how it works, how vLLM manages it, and which parameters control system behavior under load.

---

## 1. What Occupies VRAM

At vLLM startup, VRAM is divided into three components:

- **Model weights** — fixed, determined by model size and quantization (~5–6 GB for a 7–8B model in AWQ 4-bit).
- **KV Cache pool** — fixed after boot, but internal distribution changes dynamically request by request.
- **System overhead** — CUDA driver, PyTorch, GPU processes: approximately 0.5–1 GB, always present.

System overhead is the reason you never set `gpu_memory_utilization` to 1.0: that margin covers implicit costs that exist regardless of the loaded model.

---

## 2. What is the KV Cache

During autoregressive generation, for each token the model computes two vectors: **K** (Key) and **V** (Value). Without KV Cache, to generate the Nth token the model would need to reprocess the entire sequence from the beginning — O(N²) cost. With KV Cache, previously computed vectors are retrieved directly: O(N) cost per token.

The impact is concrete: lower TTFT on long prompts, more stable inter-token latency during generation.

### Per-Token Consumption

KV Cache size per token depends on model architecture:

```
KV_per_token = 2 × num_layers × num_kv_heads × head_dim × dtype_bytes
```

The `dtype_bytes` value depends on KV cache precision (configurable via `kv_cache_dtype`, see section 6):

| dtype | bytes | Notes |
|---|---|---|
| FP32 | 4 | Rarely used in serving |
| BF16 / FP16 | 2 | Default in most engines |
| FP8 | 1 | Halves KV consumption — requires Ada Lovelace+ |
| INT8 | 1 | Engine-dependent support |

For a 7B model with GQA in float16: 2 × 32 × 8 × 128 × 2 = **~128 KB per token**. With `kv_cache_dtype=fp8` the same model drops to **~64 KB per token**.

At 4096 context tokens → ~512 MB per active sequence. At 8192 → ~1 GB. These numbers explain why the KV Cache — not model weights — is the real bottleneck on concurrency.

> Note: at equal quantization and context window, models with different architectures (e.g. Qwen3-8B vs Qwen2.5-7B) can have different KV consumption. The vLLM startup log shows the number of allocated blocks.

---

## 3. How vLLM Manages the Pool

KV memory management occurs in two distinct phases with different logic.

### 3.1 Phase 1: Pool Reservation (at boot)

When vLLM starts, it pre-allocates a fixed pool of KV blocks. It does not grow or shrink during the server's lifetime.

```
KV Cache Pool = (VRAM_total × gpu_memory_utilization) − model_weight_VRAM
```

**Example on RTX 4070 Super (12 GB) with Qwen2.5-7B AWQ:**

| Item | Value |
|---|---|
| vLLM budget (12 × 0.85) | 10.2 GB |
| Model weights (AWQ 4-bit) | ~5.5 GB |
| **KV Cache Pool** | **~4.7 GB → ~3763 blocks of 16 tokens** |

If the model is too large or `gpu_memory_utilization` is too high, vLLM fails at boot — not during inference. If it starts, the system is stable at runtime.

### 3.2 Phase 2: Dynamic Allocation (at runtime)

Within the fixed pool, **PagedAttention** assigns blocks dynamically:

- Each active sequence receives blocks on demand, not in advance.
- The sequence grows block by block (16 tokens at a time).
- When it finishes, blocks are freed and reassigned.

vLLM **does not** reserve space for `max_model_len` per request. If a request generates 200 tokens instead of 8192, it occupies ~13 blocks instead of 512. This is why effective concurrency is almost always higher than the theoretical worst case.

> **Analogy:** the pool is a parking lot with N spaces. Each car (request) does not reserve N spaces on entry: it occupies one at a time as it progresses. If the cars are small, far more than N fit in the lot.

### 3.3 Maximum Concurrency in Logs

The value vLLM shows at startup (e.g. "Maximum concurrency for 8,192 tokens: 2.41x") is the **worst case**: how many requests would coexist if each used the full context window. In practice, with short prompts and limited output, effective concurrency is much higher.

---

## 4. Behavior Under Load

The vLLM scheduler decides at each step which requests to admit and which to hold. KV Cache saturation is only one of the constraints.

### 4.1 Why Requests Wait

A request can remain queued **even with available blocks**. The scheduler evaluates multiple constraints:

| Constraint | What it controls |
|---|---|
| KV pool saturation | Available physical blocks |
| `max_num_seqs` | Concurrent sequence limit in the batch |
| `max_num_batched_tokens` | Token budget per iteration (especially prefill) |
| Chunked prefill | Latency protection for sequences in decode |

In a well-dimensioned configuration, waiting is typically caused by batch and token budget limits, not memory. **Waiting is the physiological control mechanism**: no data loss, no OOM — the client simply waits.

### 4.2 Preemption: Last Resort

When batch sequences grow faster than they free, waiting is not enough. The scheduler resorts to **preemption**: suspends active requests, frees (or swaps to CPU) their KV blocks, resumes them when possible.

Preemption is expensive (requires recomputation or reload) and is the signal that the configuration is undersized for the load.

### 4.3 Intervention Hierarchy

```
Admission control (waiting)        ← normal behavior
        ↓ insufficient
KV cache pool saturation (waiting) ← memory constraint
        ↓ insufficient
Preemption (swap/recompute)        ← last resort, to be avoided
```

**Operational rule:** in a well-configured system, `num_preemptions_total` stays at zero.

---

## 5. Prefix Caching

When two requests share the same prefix, vLLM can reuse already-computed KV blocks instead of recomputing them. Enabled by default in the Dielabs lab.

The mechanism is based on **content hashing** (SHA-256), not session IDs. Completed blocks are marked as reusable. If a new request with the same prefix arrives before the pool has recycled them, reuse is automatic and transparent.

### Impact on Benchmarks

Prefix caching distorts TTFT measurements: if you use the same prompt across multiple runs, the second run has artificially low TTFT.

**Strategy:** vary the prompt between runs, or restart the container between models to reset the pool.

| Prometheus Metric | What it indicates |
|---|---|
| `prefix_cache_hits_total` / `prefix_cache_queries_total` | Hit rate — ideally < 5% during benchmarks |

---

## 6. Configuration Parameters

The two most relevant parameters for tuning — `max_num_seqs` and `max_model_len` — protect different aspects of the system:

| Parameter | Primarily protects | Sacrifices |
|---|---|---|
| `max_num_seqs` ↓ | **Latency** (ITL per request) — fewer sequences in batch → less GPU contention | Aggregate throughput |
| `max_model_len` ↓ | **Pool / sustainable throughput** — each sequence can grow less → more sequences in pool | Long context capability |

In practice, `max_num_seqs` is the primary operational lever: use it to decide whether you're optimizing for latency (low value) or throughput (high value). `max_model_len` is set once based on workload profile and left fixed.

### gpu_memory_utilization

Percentage of VRAM used as budget (weights + KV pool). The most important parameter. Lowering it reduces only the pool, not VRAM from weights. Raising it too high causes OOM at boot.

```
--gpu-memory-utilization 0.85   # recommended on 12 GB consumer
```

### kv_cache_dtype

Controls precision of K/V vectors in the pool, independent of model weight quantization. With `fp8` the KV consumption per token halves (~128 KB → ~64 KB in the 7B example), effectively doubling pool capacity at equal VRAM.

```
--kv-cache-dtype auto   # default: follows model dtype (typically float16)
--kv-cache-dtype fp8    # halves KV consumption — requires Ada Lovelace+ architecture
```

On RTX 4070 Super (Ada Lovelace) FP8 support is native. Impact on generation quality is generally negligible, but should be validated for the specific model — which is why it's a parameter to test in benchmarks, not enable blindly.

Together with `gpu_memory_utilization`, this parameter determines effective pool size: the first decides how much VRAM to dedicate, the second how efficiently it is used.

### max_model_len

Maximum context window per request (input + output). Reducing it does not change pool size, but increases theoretical concurrency because each request can grow less.

| Value | Trade-off |
|---|---|
| 8192 | Supports long conversations, low worst-case concurrency |
| 4096 | Good compromise for latency/throughput benchmarks |
| 2048 | Maximizes concurrency, but truncates long contexts |

### max_num_seqs

Hard cap on simultaneously active sequences. Even if the pool would have room for more requests, serving too many in parallel worsens each one's ITL.

```
--max-num-seqs 256   # vLLM default (not recommended for benchmarks)
--max-num-seqs 64    # controlled benchmark
--max-num-seqs 8     # single user, optimal latency
```

### max_num_batched_tokens and chunked prefill

Limits tokens processable in a single step. Chunked prefill (active by default since vLLM 0.15.x) splits long prompts into chunks to prevent a single prompt from blocking all other requests.

---

## 7. Upstream Guardrails: Admission Control

vLLM parameters operate inside the backend. A complete strategy includes upstream limitations:

- **Rate limiting** (nginx, Traefik, LiteLLM): requests per minute per API key, max tokens per request, timeout.
- **Reject vs queue policy**: vLLM uses the internal queue by default; a proxy can explicitly reject (HTTP 429) before passing the request to the backend.
- **Infinite chat protection**: without limits, each new message includes the entire history — after 50–100 turns the prompt can saturate `max_model_len`. Limit `max_tokens` per response, truncate history client-side, monitor `request_prompt_tokens`.

---

## 8. Prometheus Metrics

| Metric | Type | What to observe |
|---|---|---|
| `kv_cache_usage_perc` | Gauge | > 80% stable → saturation risk |
| `num_requests_running` | Gauge | Compare with configured `max_num_seqs` |
| `num_requests_waiting` | Gauge | > 0 constant → system under pressure (but check `kv_cache_usage_perc` to distinguish from normal admission control) |
| `num_preemptions_total` | Counter | > 0 → configuration needs review |
| `time_to_first_token_seconds` | Histogram | Monitor p50 and p99; high gap = variance from concurrency or prefix caching |
| `inter_token_latency_seconds` | Histogram | Rises with concurrency: more requests → more crowded decode batch |
| `e2e_request_latency_seconds` | Histogram | Sum of queue time + prefill + decode |

---

## 9. Scale: Homelab vs Datacenter

The mechanism is identical. What changes is scale and additional optimizations.

| Aspect | Homelab (4070 Super) | Datacenter (H100 80 GB) |
|---|---|---|
| KV Cache Pool | ~4.7 GB | ~50–60 GB |
| Worst-case concurrency (8192 tok) | 2–3 requests | 50–100+ requests |
| Tensor Parallelism | Not applicable (1 GPU) | Pool distributed across N GPUs |
| KV Cache offloading | Not necessary | CPU RAM or NVMe for inactive sequences |
| Disaggregated prefill | Not applicable | Separate nodes for prefill and decode |

The single-node physical limit is mathematical: once VRAM runs out, concurrency ends. You cannot surpass it without adding GPUs.

---

*Dielabs Knowledge Base — KV Cache Workbook v2 — March 2026*
