---
layout: default
title: "KV Cache Architecture"
---


Unified Technical Workbook

Framework L0–L6 | Layers L3 – L4b – L5

February 2026 | vLLM 0.15.x | RTX 4070 Super

---

This workbook unifies and deepens knowledge on the KV Cache in the context of the Dielabs lab, with reference to real infrastructure (RTX 4070 Super, vLLM 0.15.x) and comparative benchmarks across five models: Qwen2.5-3B, Qwen2.5-7B, Qwen3-8B, Llama 3.1 8B, and Mistral 7B v0.3.

The document covers three distinct but connected areas: internal KV cache mechanics and memory management, the reservation and dynamic allocation mechanism via PagedAttention, and guardrail parameters for making the system stable and predictable under load.

---

# 1. What Occupies VRAM: The Full Picture

When vLLM starts, VRAM is occupied by components with very different behaviors. Understanding the distinction is the first step to correctly configuring the system.

| **Component** | **Typical Size** | **When Allocated** | **Variable?** |
|---|---|---|---|
| Model weights (AWQ 4bit) | ~5-6 GB for 7-8B model | At model loading | No — fixed for server lifetime |
| KV Cache pool | Variable (see formula) | At vLLM boot (pre-allocation) | No — fixed after boot, but internal distribution dynamic |
| System overhead | 0.5-1 GB | Always present | No — CUDA driver, PyTorch, GPU processes |

System overhead is often underestimated. On consumer hardware it can be 0.5-1 GB of VRAM and is always present regardless of the loaded model. This is why `gpu_memory_utilization=1.0` is never used — that margin covers this implicit overhead.

## 1.1 Position in the L0-L6 Framework

In the Dielabs framework, the KV cache operates primarily at L3 (Inference Backend), but its implications propagate upward:

| **Layer** | **Component** | **Role in KV Cache** |
|---|---|---|
| L3 | Inference Backend (vLLM) | Allocates KV cache pool, manages blocks with PagedAttention, executes preemption if necessary |
| L4a | Request Orchestration | Decides how many requests to admit based on pool availability, manages waiting queue |
| L4b | GPU Workload Optimization | Optimizes reuse via prefix caching, manages physical blocks and their reassignment |
| L5 | API Gateway | Upstream rate limiting and admission control, before the request enters the vLLM pool |

---

# 2. What the KV Cache Is: Mechanism and Formula

During autoregressive inference, for every processed token the model computes two vectors: K (Key) and V (Value). These vectors are stored in the KV Cache to avoid recalculating them at every generation step.

Without KV Cache, to generate the N-th token the model would need to reprocess the entire sequence from the beginning: computational cost would be O(N²) per sequence. With KV Cache, already-computed K and V vectors are retrieved directly, reducing cost to O(N) per generated token. The impact on latency is direct: lower TTFT on long prompts, more stable ITL during generation.

## 2.1 Per-Token Size Formula

The KV Cache size for every processed token depends on model architecture:

```
KV_per_token = 2 × num_layers × num_kv_heads × head_dim × dtype_bytes
```

The factor 2 represents K and V. The product of other factors depends on the specific model architecture.

**Example for a 7B model with GQA (float16):**

| **Parameter** | **Value** | **Notes** |
|---|---|---|
| num_layers | 32 | Number of transformer layers |
| num_kv_heads | 8 | With GQA (Grouped Query Attention), reduced vs. total attention heads |
| head_dim | 128 | Dimension of each head |
| dtype_bytes | 2 | float16 = 2 bytes (default with AWQ) |
| KV_per_token | 2 × 32 × 8 × 128 × 2 = 131,072 bytes | ~128 KB per token per active sequence |

*With a 4096-token context this becomes ~512 MB for a single active sequence. At 8192 tokens it rises to ~1 GB. These numbers justify why the KV cache is the limiting factor on concurrency, not model weights.*

## 2.2 Variation Between Architectures: Benchmark Impact

The 5 Dielabs benchmark models have different architectures. At equal AWQ 4bit quantization and context window, per-token KV cache consumption varies because `num_layers`, `num_kv_heads`, and `head_dim` differ between model families.

This means Qwen3-8B (with its updated architecture) may have different KV consumption than Qwen2.5-7B despite similar parameter counts. The vLLM startup log shows number of allocated blocks and maximum concurrency value — comparing these values across the 5 models is already informative benchmark data.

*In the Qwen3-8B log: KV cache memory 2.71 GiB, 19712 total tokens, maximum concurrency 2.41× at 8192 tokens. In the benchmark, annotate these values for every model.*

---

# 3. Pool Reservation and Dynamic Allocation

This is the most critical point to understand, and also the one generating the most confusion. Two distinct mechanisms operate at different times and with different logic.

## 3.1 Pool Reservation: Server Startup (L3)

When vLLM starts, it does not allocate KV Cache progressively during requests — it pre-allocates it all in a fixed pool. The parameter controlling this phase is `gpu_memory_utilization`.

```
--gpu-memory-utilization 0.85   # Dielabs current setting
```

vLLM uses 85% of total VRAM as the overall budget: model weights + KV Cache pool. The pool is the difference between that budget and the weights.

**Formula:**

```
KV Cache Pool = (VRAM_total × gpu_memory_utilization) - VRAM_model_weights
```

**Calculation on RTX 4070 Super (12 GB) with Qwen2.5-7B AWQ:**

| **Item** | **Value** |
|---|---|
| Total VRAM | 12 GB |
| vLLM budget (12 × 0.85) | 10.2 GB |
| Model weights VRAM (AWQ 4bit) | ~5.5 GB |
| Available KV Cache pool | ~4.7 GB |
| PagedAttention block size | 16 tokens per block |
| Number of blocks in pool | ~3763 (per vLLM log) |

This pool never changes during server lifetime. It is fixed, reserved, occupying VRAM from startup even if no requests are active. If the model is larger or `gpu_memory_utilization` is too high, vLLM fails at boot, not during inference.

*The critical case on 12 GB is the first: large model + high `gpu_memory_utilization`. If vLLM starts correctly, the system is stable at runtime. OOM only occurs at boot.*

## 3.2 Dynamic Per-Request Allocation: PagedAttention (L4b)

Within the fixed pool, allocation happens dynamically block by block. PagedAttention is vLLM's central innovation: instead of allocating contiguous memory for each sequence, it divides the pool into fixed blocks of 16 tokens and assigns them dynamically.

**How it works at runtime:**

- At startup vLLM calculates how many 16-token blocks fit in the available pool
- Each active sequence receives blocks on demand, not in advance
- The sequence grows block by block as it generates tokens
- When a sequence ends, its blocks are freed and reassigned
- `max_model_len` limits how many blocks a single sequence can receive

vLLM does not reserve `max_model_len` space in advance for each request. If a request generates 200 tokens instead of 8192, it occupies ~13 blocks instead of 512. **This is why effective concurrency is almost always higher than the theoretical maximum value.**

*Practical analogy: the pool is like a parking lot with N spots. Each car (request) does not reserve N spots at the entrance — it occupies one at a time as it advances. If the cars are small, many more than N fit in the lot.*

## 3.3 The Maximum Concurrency Value in Logs

In the vLLM startup log a line appears like:

```
Maximum concurrency for 8,192 tokens per request: 2.41x
```

This is the theoretical worst case: how many requests can coexist if each simultaneously used the full context window. The value is not rounded to 3: with 3 full-context requests the pool would exhaust. With 2 full-context requests a margin remains.

In practice, with short prompts and limited output the effective concurrency is much higher. During Dielabs benchmarks with controlled prompts and output up to 2000 tokens, pool pressure will be well below the worst case.

## 3.4 The Relationship Between max_model_len and the Pool

`max_model_len` defines the maximum length of a single sequence (input + output).

| **What max_model_len does** | **What max_model_len does NOT do** |
|---|---|
| Limits maximum length of a single request | Does NOT pre-allocate KV Cache for max_len tokens per request |
| If a request exceeds this value, vLLM rejects it | Does NOT influence total KV Cache pool size |
| Indirectly influences how many blocks a sequence can use | Is NOT a static memory ceiling per request |

The KV Cache pool is determined by `gpu_memory_utilization`, not by `max_model_len`. Reducing `max_model_len` does not increase the pool — it only increases how many requests can theoretically fit, because each request can grow less.

---

# 4. Pool Saturation and Behavior Under Load

When available blocks in the pool are exhausted, vLLM has two options in order of preference:

**Option 1: Waiting (preferred)**

New requests are queued and wait until a block frees. No data loss, no OOM. The client simply waits longer.

**Option 2: Preemption (under extreme pressure)**

An active request is suspended, its blocks are freed or swapped to CPU RAM, and it will resume when the situation clears. Preemption is expensive: already-generated tokens must be recalculated or reloaded from CPU.

| **Scenario** | **vLLM Behavior** | **Prometheus Metric** |
|---|---|---|
| Pool < 80% full | Normal serving, new requests immediately accepted | `kv_cache_usage_perc` < 0.80 |
| Pool 80-95% full | Possible queueing, increasing latency | `num_requests_waiting` > 0 |
| Pool > 95% full | Mandatory waiting for new requests | `num_requests_waiting` growing |
| Pool 100% full | Possible preemption, performance degradation | `num_preemptions_total` > 0 |

In a well-configured system, `num_preemptions_total` should remain at zero. If it rises, the system is sized insufficiently for the received load.

## 4.1 Theoretical Batch Pressure

`max_num_seqs` (default 256 in vLLM) defines how many sequences can be processed in parallel. Theoretical maximum pool pressure:

```
Theoretical pressure = max_num_seqs × max_model_len × KV_per_token
256 sequences × 4096 tokens × 128 KB/token = ~131 GB  ← impossible to satisfy
```

This does NOT cause OOM because PagedAttention allocates blocks only when actually needed. Theoretical pressure is an abstract ceiling; real utilization depends on traffic.

---

# 5. Prefix Caching

vLLM supports prefix caching, enabled by default in the Dielabs lab (`enable_prefix_caching=True` visible in startup log). When two requests share the same prefix, vLLM does not recalculate KV blocks for that portion — it reuses them directly from the pool.

## 5.1 How It Works

The mechanism is based on content hash (SHA-256 in current log), not on session ID. Completed blocks are marked as reusable instead of being immediately deallocated. If a new request with the same prefix arrives before those blocks are removed to make space, recycling is automatic and transparent.

This differs from prefix caching in services like ChatGPT, which maintains persistent KV cache by session ID. vLLM uses a content-based opportunistic approach: if content matches and blocks are still available, they are reused. If pool pressure has removed them, they are recalculated.

## 5.2 Impact on Dielabs Benchmarks

Prefix caching is relevant for comparative benchmarks across 5 models because it distorts TTFT measurements. If you use the same prompt across multiple runs of the same model, the second run has artificially low TTFT because the prefix KV blocks are already computed.

| **Prometheus Metric** | **What It Measures** | **Benchmark Relevance** |
|---|---|---|
| `vllm:prefix_cache_hits_total` | Tokens reused from prefix cache (cumulative counter) | High = similar prompts between runs, TTFT biased downward |
| `vllm:prefix_cache_queries_total` | Total tokens processed in prefill (cumulative counter) | Denominator for calculating hit rate |
| Hit rate (hits / queries) | Percentage of reused tokens | Ideally < 5% during benchmark to avoid TTFT distortion |

*Benchmark strategy: always vary the prompt between runs, or restart the container between models. Restart clears the pool and guarantees clean TTFT measurements.*

---

# 6. Guardrails: vLLM Configuration Parameters (L3-L4b)

Correctly configuring vLLM does not mean only starting the server with defaults. It means setting limits that prevent pool saturation, protect latency, and make the system predictable under load.

## 6.1 gpu_memory_utilization

Controls the percentage of VRAM used as the overall budget for weights + KV cache pool. It is the single most important parameter.

- Lowering it does not free VRAM from model weights — it only reduces available KV cache pool
- Raising it too high causes OOM at boot if no margin for system overhead
- On consumer hardware with 12 GB, 0.85 is the recommended value

```
--gpu-memory-utilization 0.85   # Dielabs current setting
```

## 6.2 max_model_len

Defines the maximum context window per request.

| **Value** | **Effect** | **Trade-off** |
|---|---|---|
| 8192 (Dielabs current) | Supports long conversations and extended documents | Lower worst-case concurrency, pool exhausts faster with long requests |
| 4096 (recommended for benchmark) | Halves per-sequence ceiling, more blocks available for concurrent requests | Limits very long contexts, but sufficient for latency/throughput benchmarks |
| 2048 | Maximizes theoretical concurrency | Long conversations are truncated or rejected |

```
--max-model-len 8192   # Dielabs current setting
```

## 6.3 max_num_seqs

Limits the maximum number of simultaneously active sequences. It is a hard cap: beyond this value requests go to waiting regardless of KV block availability.

Useful for protecting latency: even if the pool had space for 10 requests, serving 10 concurrent requests increases ITL for each. Limiting to 4-8 on homelab improves per-request latency.

```
--max-num-seqs 256    # vLLM default (near-unlimited, not recommended for benchmarks)
--max-num-seqs 64     # Dielabs benchmark value (controlled concurrency)
--max-num-seqs 8      # Conservative example for single-user lab
```

## 6.4 max_num_batched_tokens and Chunked Prefill

Limits the total number of tokens processable in a single batch step. Controls pool pressure during the prefill phase, when many long requests enter simultaneously.

Chunked prefill (automatically enabled by vLLM 0.15.x, visible in log as `max_num_batched_tokens=2048`) splits long prompts into chunks. Prevents a single long prompt from blocking the batch for all other active requests.

```
--max-num-batched-tokens 2048   # default with chunked prefill enabled
```

---

# 7. Guardrails at L4a-L5: Admission Control

vLLM parameters operate within the backend (L3-L4b). A complete guardrail strategy also includes the higher level: limiting requests before they enter the pool.

## 7.1 Per-User Rate Limiting (L5)

An API gateway or reverse proxy (nginx, Traefik, Kong, LiteLLM) can limit requests per API key or IP before they reach vLLM:

- Requests per minute per user/API key
- Maximum tokens per request (`max_tokens` in JSON body)
- Timeout for requests stuck in waiting

## 7.2 Reject vs Queue Policy (L4a)

When the system is saturated there are two distinct approaches. The queue policy returns HTTP 429 (Too Many Requests) and invites the client to retry. The reject policy immediately refuses. vLLM uses the internal queue by default, but a L4a proxy can implement explicit reject before passing the request to the backend.

## 7.3 Infinite Chat Protection

Without explicit limits, a user can hold a very long conversation that grows with each turn. Every new message includes the entire previous history as prefix: after 50-100 messages the prompt can approach `max_model_len`, saturating KV blocks of that single session.

Mitigation strategies:
- Limit `max_tokens` per response in the request body
- Truncate history on the client side after N messages
- Set conservative `max_model_len` for the specific use case
- Monitor `vllm:request_prompt_tokens` and alert when approaching the limit

---

# 8. Prometheus Metrics: Interpretation and Operational Thresholds

In the Dielabs lab, Grafana monitors vLLM metrics in real time. This section provides complete interpretive context for KV cache-related metrics.

| **Metric** | **Type** | **What It Measures** | **Alert Threshold** |
|---|---|---|---|
| `vllm:kv_cache_usage_perc` | Gauge | % of KV cache pool occupied right now | > 80% stable = imminent saturation risk |
| `vllm:num_requests_running` | Gauge | Requests currently in generation phase | Depends on configured `max_num_seqs` |
| `vllm:num_requests_waiting` | Gauge | Requests queued for lack of KV blocks | > 0 consistently = system under pressure |
| `vllm:num_preemptions_total` | Counter | Cumulative preemptions since startup | > 0 = configuration needs urgent review |
| `vllm:prefix_cache_hits_total` | Counter | Tokens reused from prefix cache (cumulative) | High during benchmark = TTFT distorted |
| `vllm:prefix_cache_queries_total` | Counter | Total tokens in prefill (cumulative) | Denominator for hit rate |
| `vllm:request_queue_time_seconds` | Histogram | Time in waiting before entering execution | High p99 = admission control needed at L4a/L5 |
| `vllm:time_to_first_token_seconds` | Histogram | Prefill latency per request | Correlated with prompt length and active concurrency |
| `vllm:inter_token_latency_seconds` | Histogram | Time between consecutive tokens | Rises with high concurrency: more requests = more crowded decode batch |
| `vllm:e2e_request_latency_seconds` | Histogram | Total latency from request to last token | Sum of queue time + prefill + decode |

*For histogram metrics (TTFT, ITL, E2E), always monitor both p50 (typical case) and p99 (worst case). A high difference between p50 and p99 indicates high variance, typically caused by concurrency or prefix caching.*

---

# 9. Recommended Dielabs Lab Configuration

Configuration for comparative benchmark on RTX 4070 Super 12 GB with AWQ 4bit 7-8B models. Two profiles: one for formal benchmarking, one for daily single-user use.

## 9.1 Benchmark Profile (5 Comparative Models)

| **Parameter** | **Value** | **Rationale** |
|---|---|---|
| `gpu_memory_utilization` | 0.85 | Safe margin for system overhead on 12 GB |
| `max_model_len` | 8192 | Uniform across all models for correct comparison |
| `max_num_seqs` | 64 | Controlled concurrency, comparable results between models |
| `max_num_batched_tokens` | 2048 (default) | Chunked prefill already automatically enabled by vLLM 0.15.x |
| `enable_prefix_caching` | True (default) | Left active but vary prompts between runs |
| temperature | 0.7 | Same variability across all models |
| top_p | 0.9 | Consistent sampling across different architectures |

## 9.2 Daily Single-User Profile

| **Parameter** | **Value** | **Rationale** |
|---|---|---|
| `gpu_memory_utilization` | 0.85 | Unchanged |
| `max_model_len` | 8192 | Long context for multi-turn conversations |
| `max_num_seqs` | 8-16 | Balances concurrency and latency for single user |
| `max_num_batched_tokens` | 2048 (default) | Unchanged |
| `enable_prefix_caching` | True (default) | Useful: reuses KV of common prompts and system prompts |

*Benchmark warning: to obtain TTFT measurements unaffected by prefix cache, always vary the prompt between runs or restart the container between models. Restart clears the pool and guarantees identical starting conditions.*

---

# 10. Scale: Same Principle, Datacenter Context

The KV Cache pool pre-allocation mechanism and management via PagedAttention are identical in production datacenters. Scale and some additional optimizations change, not the architectural principle.

| **Aspect** | **Homelab (4070 Super)** | **Datacenter (H100 80GB)** | **Notes** |
|---|---|---|---|
| KV Cache pool | ~4.7 GB | ~50-60 GB per node | Same mechanism, different scale |
| Max concurrency (worst case) | 2-3 requests at 8192 tokens | 50-100+ requests at 8192 tokens | Proportional to pool |
| Tensor Parallelism | Not applicable (1 GPU) | Pool distributed across N GPUs in node | KV Cache sharding between cards |
| KV Cache offloading | Not necessary | CPU RAM or NVMe for inactive sequences | Increases effective capacity at latency cost |
| Prefix caching | Available (SHA-256) | Advanced systems with RadixAttention | Same principle, more sophisticated implementation |
| Disaggregated prefill | Not applicable | Separate nodes for prefill and decode | Optimization for specialized loads |

The single-node physical limit is mathematically determined by pool size. On H100 with Llama-3-8B, up to ~14 concurrent requests can be managed at full context. When VRAM runs out, concurrency runs out: it is a hard limit that cannot be exceeded without adding GPUs.

---

# Appendix: Complete Allocation Flow

**vLLM Boot:**

```
Boot vLLM
  |
  +-- Load AWQ model weights --> ~5.5 GB fixed in VRAM
  |
  +-- Calculate KV Cache pool --> (12 × 0.85) - 5.5 = ~4.7 GB
  |
  +-- Divide into 16-token blocks --> ~3763 PagedAttention blocks
  |
  +-- Pool reserved and fixed. Server ready.
```

**Runtime (per request):**

```
New request arrives
  |
  +-- Blocks available? --> YES --> Assign blocks to sequence
  |                                   |
  |                                   +-- Generate tokens
  |                                   +-- Allocate new block every 16 tokens
  |                                   +-- Sequence complete --> free blocks
  |
  +-- Blocks available? --> NO --> Waiting queue (optional HTTP 429 at L5)
                                    |
                                    +-- Wait for block release
                                    +-- If extreme pressure --> Preemption
```

*OOM only occurs at boot, never at runtime. If vLLM starts correctly, the system manages load via waiting and preemption, not via crashes.*

---

*Dielabs Knowledge Base | KV Cache Unified Workbook | February 2026*
*[dielabs.eu](https://dielabs.eu)*

