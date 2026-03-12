---
layout: default
title: "KV Cache Sizing"
---

# KV Cache — Sizing and Calculations

Memory calculation is not just about getting the model to start — it determines the **load capacity** (concurrency) before the system hits OOM (Out of Memory).

---

## 1. Quick & Dirty Formula

For a fast calculation, use this three-block structure:

> **Total VRAM = Static Base + (Cost per User × Concurrent Users) + Margin**

---

## 2. The 3 Memory Components

### A. Static Base — Model Weights

The memory occupied as soon as the model is loaded, before the first request arrives.

- **Formula:** `Parameters (billions) × bytes per precision`
- **Bytes per precision:**

| dtype | bytes |
|---|---|
| FP32 | 4 |
| BF16 / FP16 | 2 |
| INT8 | 1 |
| FP8 | 1 |
| INT4 | 0.5 |

- **Fixed overhead:** always add ~2 GB for the system (CUDA, driver, compute graphs).

### B. KV Cache — Cost per User

The dynamic memory that grows with every generated token. This is where the scalability game is played.

The KV Cache stores the Key and Value matrices of every transformer layer for already-processed tokens, avoiding recalculation at every generation step. Q (Query) is not saved.

Modern models with **GQA (Grouped Query Attention)** — Llama 3, Mistral, Qwen — consume significantly less KV Cache than classic MHA models, allowing more concurrent users per GPU.

### C. Activations — Temporary Peak

Memory used during actual computation. Low in serving (**0.5–1 GB**), but peaks during **Prefill** (when the GPU processes a long prompt all at once).

---

## 3. Formula 1 — KV Cache per Token (model property)

Calculates how much VRAM a single cached token occupies. This is a constant for a given model and dtype.

```
kv_per_token = 2 × num_layers × num_kv_heads × head_dim × bytes_per_element
```

**Example: Llama 3.1 8B (bf16)**

| Parameter | Value | Notes |
|---|---|---|
| num_layers | 32 | |
| num_kv_heads | 8 | GQA (not MHA) |
| head_dim | 128 | |
| bytes_per_element | 2 | bf16 |

Step by step:

```
2 × 32 = 64
64 × 8 = 512
512 × 128 = 65,536
65,536 × 2 = 131,072 bytes = 128 KB per token
```

---

## 4. Formula 2 — KV Cache per Workload (deployment property)

Estimates total VRAM needed for a given load. Uses Formula 1 as input.

```
memory_total = model_weights + (concurrent_users × avg_context_tokens × kv_per_token)
```

**Numerical example — Llama 3.1 8B, 10 users, 4K context**

| Item | Calculation | Result |
|---|---|---|
| Model weights (bf16) | 8B × 2 bytes | ~16 GB |
| KV per token (Formula 1) | | 128 KB |
| KV cache (10u × 4K) | 10 × 4096 × 128 KB | ~5 GB |
| **TOTAL** | | **~21 GB** |

**Relationship between the two formulas:**

Formula 1 is a fixed property of the model (architecture + dtype).
Formula 2 varies based on real workload.
Formula 1 is always an input parameter of Formula 2.

```
memory_total = model_weights + (concurrent_users × avg_context × [Formula 1])
```

---

## 5. Full Operational Example — Llama 3 27B, 10 users

| Step | Component | Calculation | Result |
|---|---|---|---|
| 1 | Model weights | 27B × 2 bytes (BF16) | 54 GB |
| 2 | Fixed overhead | Runtime + CUDA | 2 GB |
| 3 | KV Cache | 10 users × 1.3 GB (average estimate) | 13 GB |
| 4 | Safety margin | 10% of total | ~7 GB |
| **TOT** | **Required VRAM** | | **76 GB** |

---

## 6. Practical Rules

| Scenario | Impact |
|---|---|
| Double the context length | KV cache doubles linearly |
| Double concurrent users | KV cache doubles linearly |
| Switch from bf16 to fp8 | KV cache is halved |
| GQA vs MHA | Fewer kv_heads → smaller KV cache |

---

## 7. Choosing the GPU

- **Under 40 GB:** small models (7B–8B) or quantized (4-bit).
- **40–80 GB (A100/H100/A6000):** mid-size models (27B–30B) with a reasonable user count.
- **Over 80 GB:** requires **Tensor Parallelism** (splitting the model across multiple GPUs).

**Golden rule:** if memory is not enough, don't rush to buy a bigger GPU. Try first:
1. **Quantize** — from BF16 to INT8 or INT4
2. **Reduce maximum context length** in server settings
3. **Enable GQA** (if the model supports it)

---

## 8. Empirical Monitoring

Don't rely on theory alone. Engines like vLLM manage memory in blocks (PagedAttention).

1. Deploy the model
2. Run a stress test (e.g. with 10 simulated users)
3. Check `avg_kv_cache_usage` via Prometheus/Grafana
4. **Real formula:** `(Used VRAM - Base VRAM) / Num Users` = real cost per user

---

## Appendix — Production Notes

The formulas above assume an ideal linear model. In production systems, several techniques alter this behavior.

### A. Allocated vs Used vs Fragmented KV Cache

| State | Definition | Practical impact |
|---|---|---|
| Allocated | Memory reserved by the runtime at startup for the KV pool | Fixed, determined by `--gpu-memory-utilization` in vLLM |
| Used | Memory actually occupied by tokens of active requests | Variable, depends on instantaneous workload |
| Fragmented | Allocated memory that cannot be used for new requests due to block layout | Unavoidable overhead, reduces effective capacity vs theoretical |

**PagedAttention (vLLM):** the KV cache is not a monolithic linear block. Memory is divided into fixed-size physical blocks (`block_size`, typically 16 or 32 tokens). Each request occupies an integer number of blocks, even if the last block is not full — this introduces internal fragmentation. The benefit is that requests of different lengths can coexist without contiguous allocation waste, enabling much higher throughput than static allocation.

```
memory_used ≤ memory_allocated  (always)
memory_effective = memory_allocated - fragmentation_overhead
```

### B. Implicit Assumptions of Formula 2

| Assumption | Technique that violates it | Effect on the linear model |
|---|---|---|
| No persistent prefix caching | Prefix caching / RadixAttention | Reduces effectively generated KV cache: shared prefix tokens are reused across requests |
| No speculative decoding | Speculative decoding (draft model) | Generates temporary draft tokens that occupy additional KV cache not accounted for by the base formula |
| Entire context window in cache | Sliding window attention | Only a window of recent tokens is kept in cache: consumption grows to window size then stabilizes, not linearly with seq_len |
| KV cache entirely in VRAM | KV offloading (CPU/NVMe) | Part of the KV cache is offloaded to RAM or NVMe: occupied VRAM is lower than theoretical, but introduces swap latency |

In a standard vLLM deployment without advanced configurations, the assumptions of Formula 2 are substantially respected.

---

*Dielabs KB — KV Cache Sizing and Calculations v1.0*

