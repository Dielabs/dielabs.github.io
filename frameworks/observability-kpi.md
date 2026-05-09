---
layout: default
title: "Observability KPI"
---

# Observability KPI — Monitoring, Diagnostics and Incident Response for LLM Inference Systems

**Version 1.2**

Stack: vLLM + Prometheus + Grafana + DCGM Exporter + Node Exporter + LMCache

Hardware: NVIDIA RTX 4070 Super (12 GB VRAM, 363 GB/s)

ISM scope: L0 (Hardware) → L4b (GPU Workload Optimization)

---

## 1. The Golden Metrics Glossary

Three metrics define the user experience and the health of the inference system.

| Metric | Full Name | LLM Phase | Description |
|---|---|---|---|
| **TTFT** | Time To First Token | Prefill | Reaction time. The engine processes the prompt and generates the first token. Depends on prompt length and GPU availability. |
| **ITL** | Inter-Token Latency | Decode | Time between one token and the next. Determines the fluidity perceived by the user. Depends on model weight and the active batch. |
| **E2E** | End-to-End Latency | Total | Total time from HTTP request to last token. Includes queuing, prefill, decode and network overhead. |

Note: **TPOT** is the average ITL of a single request — the per-request decode-side counterpart of system-level ITL.

### Supporting Metrics

| Metric | Description | Why It Matters |
|---|---|---|
| **Waiting Requests** | Requests in queue before entering Running | Early saturation indicator. If it grows, TTFT will follow. |
| **Running Requests** | Requests currently in inference on the GPU | Shows the level of active parallelism (continuous batching). |
| **Preempted Requests** | Requests preempted by the inference engine | Direct signal of pressure on the KV cache — the engine is forcibly freeing slots. |
| **GPU KV Cache Usage** | Percentage of KV cache allocated | At ~95-100%, vLLM starts preemption or rejects requests. |
| **GPU Utilization** | GPU compute utilization (DCGM) | Distinguishes whether the GPU is the bottleneck or the problem is elsewhere. |
| **GPU Memory Used** | VRAM occupied (DCGM) | Baseline to understand the margin after model loading. |
| **LMCache Hit Rate** | Ratio of cache hits to total queries (external prefix cache) | Indicates KV cache offloading effectiveness. Low hit rate = the cache is not serving, investigate prompt patterns. |
| **Observed Throughput** | Effective tokens/sec observed (prompt + generation separated) | Real end-to-end throughput, includes scheduling and queuing overhead. |
| **Compute Throughput** | Pure compute tokens/sec (prefill and decode separated) | Raw GPU throughput, isolated from overhead. The delta with Observed Throughput reveals the system overhead. |

---

## 2. Observed vs Compute Throughput: Two Measures, Two Questions

The system exposes two classes of throughput metrics that answer different questions.

### Prefill and Decode: Why Two Measures Are Needed

Prefill (prompt processing) is **compute-bound**: the GPU executes a parallel forward pass on all input tokens. Decode (output generation) is **memory-bandwidth-bound**: at each step the GPU reads model weights from VRAM to produce a single token. Monitoring them as an aggregate hides the nature of the bottleneck.

### Observed Throughput (System Throughput)

Tokens processed or generated per unit of **wall-clock time**. Includes everything: GPU compute, scheduling, queue time, preemption/swap, CPU-side tokenization, PCIe transfers, idle time between batch iterations.

Answers the question: **how many tokens/sec is the system producing right now?**

### Compute Throughput (Engine Throughput)

Tokens processed or generated per unit of **effective GPU compute time**. Excludes any orchestration overhead.

Answers the question: **when the GPU is working, how fast does it work?**

### Efficiency Gap

Under ideal conditions (no queue, no overhead), Observed ≈ Compute. Under load they diverge:

```
Efficiency Gap = 1 - (Observed Throughput / Compute Throughput)
```

- **Gap ≈ 0%** → efficient system, almost all time is useful compute.
- **Gap 20-40%** → significant orchestration overhead.

### Sources of the Overhead (the Delta)

The delta between Observed and Compute captures the sum of:

- **Scheduling delay** — the continuous batching scheduler decides which requests to include; the GPU waits.
- **Queue time** — requests in waiting do not produce tokens, but time keeps flowing.
- **Preemption and KV cache swap** — manifests only under VRAM pressure or with KV offload active (e.g. LMCache `local_cpu: true`). The scheduler interrupts a request, copies the KV state via PCIe (swap out/in). Dead time for compute.
- **KV cache block management** — PagedAttention block allocation/deallocation (CPU-side).
- **Tokenization/Detokenization** — text↔token ID conversion (CPU-side).
- **PCIe host↔device transfers** — input token IDs to GPU, logits/output to CPU.
- **Batch padding and tensor reorganization** — requests with different lengths in continuous batching.

On a single GPU node with controlled benchmark loads, the gap is typically negligible. In production with thousands of concurrent requests and frequent swap, it can become significant.

### Diagnostics: TTFT × Dual Throughput

| TTFT | Compute Throughput | Diagnosis |
|---|---|---|
| Rises | Stable | Orchestration problem (queue, scheduling). Check Waiting Requests and Efficiency Gap. |
| Rises | Drops | GPU problem (compute saturation). Check `DCGM_FI_DEV_GPU_UTIL`. |
| Stable | Stable | Healthy system. |
| Stable | Drops | Anomaly — possible thermal throttling or GPU error. Check temperature and DCGM logs. |

### Operational Note

Compute Throughput metrics produce values only when there are active requests. At rest, `rate()` returns 0 or NaN (division by zero when there is no compute time). This is expected behavior, not an error.

---

## 3. Applied Statistics: Percentiles in Inference Systems

Never look only at the average: percentiles tell different stories.

### p50 (Median)

The central value. Tells the story of the "typical" user. It is the first indicator to normalize when a problem ends. If p50 is good, most users are satisfied.

### p99 (Edge Case)

The worst 1% of requests. Reveals bottlenecks, queues and saturation problems that the median hides. It is the KPI that matters for production SLAs.

### The Echo Effect

p99 stays high in graphs even after a problem ends because slow data points remain in the temporal aggregation window until they exit the calculation. If the window is 5 minutes, it takes 5 minutes for p99 to normalize.

**How to distinguish it from a real problem:** if p50 is already back to normal but p99 stays high, it is almost certainly an echo effect. Wait for the window to drain.

### Counter Reset vs Echo Effect

A vertical drop in the p99 line can mean two things:

1. Statistical flush: slow data points have exited the window — the system is back to healthy.
2. vLLM restart: Prometheus counters were zeroed by a process restart.

**Verification query:**

```promql
resets(vllm:time_to_first_token_seconds_count[5m])
```

If the result is > 0, there has been a restart. Annotate it to avoid confusing it with an organic recovery.

---

## 4. Diagnostic Tree: From Metric to Root Cause

Operational table for incident response. Cross-reference symptoms to identify the cause.

| Symptom | Probable Cause | Layer | Immediate Action |
|---|---|---|---|
| **High ITL**, low TTFT | Slow GPU on decode. Model too heavy or insufficient quantization. | L1 | Check GPU utilization (DCGM). Consider more aggressive quantization (FP16 → AWQ/GPTQ 4-bit). Lower max_model_len. |
| **Very high TTFT**, low ITL | Queue saturation. Requests wait in Waiting too long. | L3 | Check waiting_requests. Reduce concurrent load. Check KV cache usage — if at 100%, prefill is blocked. |
| **High p99**, low p50 | Healthy system, but outliers (very long prompts, past queue) pollute statistics. | — | Check if echo effect (§3). Check prompt length distribution. |
| **High TTFT**, high ITL | Total saturation. GPU at max, queue full, no margin. | L1+L3 | Reduce load immediately. Verify whether the model is appropriate for the hardware. |
| **KV Cache > 95%** | Risk of preemption or request rejection. Insufficient memory. | L3 | Lower max_model_len, reduce concurrency, evaluate smaller model or quantization. Check `vllm:num_requests_preempted` to confirm preemption is active. |
| **GPU Util < 30%** with high ITL | Memory bandwidth bottleneck (memory-bound). | L1 | RTX 4070 Super has 363 GB/s — for a 7B FP16 (~14 GB) decode is memory-bound. Quantizing reduces the data volume to read. |
| **Preempted > 0** growing | KV cache under active pressure. The engine is evicting requests. | L3 | Check KV cache usage. Reduce concurrency or max_model_len. If preemption policy is recompute, TTFT of evicted requests will rise. |
| **Low LMCache Hit Rate** (< 0.3) with high KV Cache | The external cache is not serving. Prompts have insufficient prefix overlap. | L3 | Check prompt patterns — if very different from each other, prefix cache is ineffective. Evaluate whether LMCache is configured correctly. |
| **Observed Throughput << Compute Throughput** | High system overhead (scheduling, queuing, preemption). | L3+L4a | The GPU has free capacity but the system is not using it. Check waiting requests, preemption, continuous batching configuration. |

---

## 5. Operational PromQL Queries for Grafana

### Configuration Principles

The current dashboard uses the following conventions:

- **Time interval:** `[15s]` for maximum reactivity.
- **Function:** `rate` on all queries.
- **Technical note:** With a 15s scrape interval, a `[15s]` window typically contains 1 sample (at the edge). Prometheus requires at least 2 data points to calculate a rate. In practice it works because Prometheus can use the last sample preceding the window, but the behavior is fragile: if a scrape fails, the rate returns `NaN`. For production environments, evaluate `[30s]` as a compromise between reactivity and robustness. Annotate the scrape interval in the dashboard.
- **Instance:** All vLLM queries filter by `instance="192.168.4.250:8000"`.

### Golden Metrics Dashboard

**TTFT p99:**

```promql
histogram_quantile(0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket{instance="192.168.4.250:8000"}[15s])))
```

**TTFT p50:**

```promql
histogram_quantile(0.50, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket{instance="192.168.4.250:8000"}[15s])))
```

**ITL p99:**

```promql
histogram_quantile(0.99, sum by (le) (rate(vllm:inter_token_latency_seconds_bucket{instance="192.168.4.250:8000"}[15s])))
```

**ITL p50:**

```promql
histogram_quantile(0.50, sum by (le) (rate(vllm:inter_token_latency_seconds_bucket{instance="192.168.4.250:8000"}[15s])))
```

**E2E Latency p99:**

```promql
histogram_quantile(0.99, sum by (le) (rate(vllm:e2e_request_latency_seconds_bucket{instance="192.168.4.250:8000"}[15s])))
```

**E2E Latency p50:**

```promql
histogram_quantile(0.50, sum by (le) (rate(vllm:e2e_request_latency_seconds_bucket{instance="192.168.4.250:8000"}[15s])))
```

### System State Dashboard

**Waiting Requests:**

```promql
vllm:num_requests_waiting{instance="192.168.4.250:8000"}
```

**Running Requests:**

```promql
vllm:num_requests_running{instance="192.168.4.250:8000"}
```

**Preempted Requests:**

```promql
vllm:num_requests_preempted{instance="192.168.4.250:8000"}
```

**KV Cache Usage (%):**

```promql
vllm:kv_cache_usage_perc{instance="192.168.4.250:8000"} * 100
```

**Observed Throughput — Prompt (tokens/sec):**

```promql
rate(vllm:prompt_tokens_total{instance="192.168.4.250:8000"}[15s])
```

**Observed Throughput — Generation (tokens/sec):**

```promql
rate(vllm:generation_tokens_total{instance="192.168.4.250:8000"}[15s])
```

**Compute Throughput — Prefill (tokens/sec):**

```promql
rate(vllm:request_prompt_tokens_sum{instance="192.168.4.250:8000"}[15s]) / rate(vllm:request_prefill_time_seconds_sum{instance="192.168.4.250:8000"}[15s])
```

**Compute Throughput — Decode (tokens/sec):**

```promql
rate(vllm:request_generation_tokens_sum{instance="192.168.4.250:8000"}[15s]) / rate(vllm:request_decode_time_seconds_sum{instance="192.168.4.250:8000"}[15s])
```

### LMCache Dashboard

**LMCache Hit Rate:**

```promql
sum(rate(vllm:external_prefix_cache_hits_total{instance="192.168.4.250:8000"}[15s])) / sum(rate(vllm:external_prefix_cache_queries_total{instance="192.168.4.250:8000"}[15s]))
```

**LMCache Query Rate:**

```promql
rate(vllm:external_prefix_cache_queries_total{instance="192.168.4.250:8000"}[15s])
```

### GPU Dashboard (DCGM)

**GPU Utilization:**

```promql
DCGM_FI_DEV_GPU_UTIL{instance=~"${instance}", gpu=~"${gpu}"}
```

**GPU Memory Used:**

```promql
DCGM_FI_DEV_FB_USED{instance=~"${instance}", gpu=~"${gpu}"}
```

> **Note:** The current dashboard does not include a panel for `DCGM_FI_DEV_GPU_TEMP`. Adding the panel is recommended for the pre-benchmark checklist (§9), which requires verifying the temperature baseline.

### Node Dashboard (Node Exporter)

**CPU Usage (%):**

```promql
100 * (1 - avg by (instance) (irate(node_cpu_seconds_total{job="gpu-node", mode="idle"}[5m])))
```

**RAM Usage (GB):**

```promql
(node_memory_MemTotal_bytes{job="gpu-node"} - node_memory_MemAvailable_bytes{job="gpu-node"}) / 1024 / 1024 / 1024
```

**SATA Disk IOPS — Read:**

```promql
rate(node_disk_reads_completed_total{instance="192.168.4.250:9100", device="sda"}[1m])
```

**SATA Disk IOPS — Write:**

```promql
rate(node_disk_writes_completed_total{instance="192.168.4.250:9100", device="sda"}[1m])
```

---

## 6. Cross-Verification: Logs vs Charts

### The Sanity Check

The textual logs of vLLM are the "real-time truth". If the log reports TTFT = 140ms but the Grafana chart shows 35s, the problem is the chart's aggregation window, not the server.

**Procedure:**

1. Identify the anomaly in the Grafana chart.
2. Consult vLLM logs for the same temporal interval.
3. If logs confirm normal values → echo effect (§3). Wait for window flush.
4. If logs confirm anomalous values → real problem. Proceed with the diagnostic tree (§4).

### Useful Commands

**Latest vLLM logs (Docker):**

```bash
docker logs --tail 100 --timestamps <container_name>
```

**Filter by specific timestamp:**

```bash
docker logs <container_name> 2>&1 | grep "2025-06-15T14:3"
```

**Check recent restarts:**

```bash
docker inspect --format='{{.State.StartedAt}}' <container_name>
```

---

## 7. Hardware Operational Thresholds — RTX 4070 Super

Reference values for the GPU node (gpu.dielabs.eu).

| Parameter | Value | Alert Threshold | Notes |
|---|---|---|---|
| Total VRAM | 12 GB GDDR6X | — | Budget for model + KV cache + CUDA overhead (~300-500 MB). |
| Memory Bandwidth | 363 GB/s | — | Limits decode of memory-bound models (e.g. 7B FP16). |
| GPU Temp (idle) | ~35-45°C | — | Pre-benchmark baseline. |
| GPU Temp (load) | ~65-80°C | > 83°C | Above 83°C the GPU starts thermal throttling, reducing clock and throughput. |
| GPU Utilization (active inference) | 60-95% | < 30% with high ITL | Below 30% with high latency → memory-bound, not compute-bound. |
| KV Cache Usage | Variable | > 95% | At saturation, preemption or request rejection. |
| Power Limit | 220W (stock) | — | Verify with `nvidia-smi` that it has not been reduced. |

---

## 8. Incident Response Procedure

Operational workflow when an alert is received or an anomaly is noticed.

**Step 1:** Check p50 and p99 TTFT + ITL

```
┌───────────────────────────────────────────────────────────┐
│  p50 OK, p99 high     → Echo effect / outliers            │
│                          Check logs. Wait for flush.      │
├───────────────────────────────────────────────────────────┤
│  TTFT high, ITL low   → Queue saturation                  │
│                          Waiting Requests + KV Cache %.   │
├───────────────────────────────────────────────────────────┤
│  ITL high, TTFT low   → GPU bottleneck                    │
│                          DCGM GPU Util + quantization.    │
├───────────────────────────────────────────────────────────┤
│  Everything high       → Total saturation                  │
│                          Reduce load immediately.         │
└───────────────────────────────────────────────────────────┘
```

**Step 2:** Check context metrics: Preempted Requests, LMCache Hit Rate, Observed vs Compute Throughput.

**Step 3:** Verify vLLM logs for confirmation (§6).

**Step 4:** Apply action from the diagnostic tree (§4).

**Step 5:** Monitor recovery — p50 normalizes first, p99 follows after temporal window flush.

---

## 9. Pre-Benchmark Checklist

To be executed before every benchmark session to ensure clean and comparable data.

- [ ] vLLM just started (Prometheus counters zeroed)
- [ ] No other GPU-intensive process active (`nvidia-smi`)
- [ ] Prometheus scrape interval verified and annotated (current: 15s)
- [ ] Grafana dashboard with `[15s]` windows + `rate`
- [ ] Stable GPU temperature baseline — verify `DCGM_FI_DEV_GPU_TEMP` (no thermal throttling, see §7)
- [ ] `max_model_len` and vLLM parameters documented
- [ ] Model quantization documented
- [ ] Warm-up completed (2-3 warm-up requests discarded; criterion: ITL stabilized and GPU temp on plateau)
