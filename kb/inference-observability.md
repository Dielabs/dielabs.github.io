# Inference Observability — Monitoring, Diagnostics and Incident Response

**Version:** 1.0  
**Stack:** vLLM + Prometheus + Grafana + DCGM Exporter  
**Hardware:** NVIDIA RTX 4070 Super (12 GB VRAM, 363 GB/s)  
**ISM Scope:** L0 (Hardware) → L4b (GPU Workload Optimization)

---

## 1. The Golden Metrics

Three metrics define user experience and the health of an inference system.

| Metric | Full Name | LLM Phase | Description |
|---|---|---|---|
| **TTFT** | Time To First Token | Prefill | Reaction time. The engine processes the prompt and generates the first token. Depends on prompt length and GPU availability. |
| **ITL** | Inter-Token Latency | Decode | Time between one token and the next. Determines perceived fluidity for the user. Depends on model weight and current batch. |
| **E2E** | End-to-End Latency | Total | Total time from HTTP request to last token. Includes queuing, prefill, decode and network overhead. |

### Support Metrics

| Metric | Description | Why It Matters |
|---|---|---|
| **Waiting Requests** | Requests queued before entering Running state | Early saturation indicator. If it grows, TTFT will follow. |
| **Running Requests** | Requests currently in GPU inference | Shows active parallelism level (continuous batching). |
| **GPU KV Cache Usage** | Percentage of allocated KV cache | At ~95–100%, vLLM starts preemption or rejects requests. |
| **GPU Utilization** | GPU compute usage (DCGM) | Distinguishes whether GPU is the bottleneck or the problem is elsewhere. |
| **GPU Memory Used** | Occupied VRAM (DCGM) | Baseline to understand the margin after model loading. |

---

## 2. Applied Statistics: Percentiles in Inference Systems

Never look at the average alone — percentiles tell different stories.

### p50 (Median)
The central value. Describes the "typical" user experience. The first indicator to normalize when a problem ends. If p50 is good, most users are satisfied.

### p99 (Edge Case)
The worst 1% of requests. Reveals bottlenecks, queues and saturation problems that the median hides. The KPI that matters for production SLAs.

### The Echo Effect
p99 remains high in graphs even after a problem ends, because slow data points remain in the temporal aggregation window until they exit the calculation. If the window is 5 minutes, it takes 5 minutes for p99 to normalize.

**How to distinguish it from a real problem:** if p50 has already returned to normal but p99 remains high, it's almost certainly an echo effect. Wait for the window to flush.

### Counter Reset vs Echo Effect
A vertical drop in the p99 line can mean two things:
1. Statistical flush: slow data has exited the window — the system is healthy again.
2. vLLM restart: Prometheus counters were reset by a process restart.

**Verification query:**
```promql
resets(vllm:time_to_first_token_seconds_count[5m])
```
If the result is > 0, there was a restart. Annotate it to avoid confusing it with an organic recovery.

---

## 3. Diagnostic Tree: From Metric to Root Cause

Operational table for incident response. Cross-reference symptoms to identify the cause.

| Symptom | Probable Cause | Layer | Immediate Action |
|---|---|---|---|
| **High ITL**, low TTFT | Slow GPU during decode. Model too heavy or insufficient quantization. | L1 | Check DCGM GPU utilization. Consider more aggressive quantization (FP16 → AWQ/GPTQ 4-bit). Reduce `max_model_len`. |
| **Very high TTFT**, low ITL | Queue saturation. Requests waiting in Waiting state too long. | L3 | Check `waiting_requests`. Reduce concurrent load. Verify KV cache usage — if at 100%, prefill is blocked. |
| **High p99**, low p50 | System healthy, but outliers (very long prompts, past queue) contaminate the statistic. | — | Check if it's echo effect (§2). Check prompt length distribution. |
| **High TTFT** + **high ITL** | Total saturation. GPU maxed out, full queue, no margin. | L1+L3 | Immediately reduce load. Verify whether model is appropriate for the hardware. |
| **KV Cache > 95%** | Preemption risk or request rejection. Insufficient memory. | L3 | Reduce `max_model_len`, reduce concurrency, consider smaller model or quantization. |
| **GPU Util < 30%** with high ITL | Memory bandwidth bottleneck (memory-bound). | L1 | RTX 4070 Super has 363 GB/s — for a 7B FP16 (~14 GB) decode is memory-bound. Quantization reduces the data volume to read. |

---

## 4. Operative PromQL Queries for Grafana

### Configuration Principles

- **Time interval:** use `[1m]` instead of `[5m]` for immediate reactivity in operational dashboards.
- **Function:** use `irate` for instantaneous variations (last 2 samples), `rate` for stable trends.
- **Technical note:** `irate` requires at least 2 data points in the window. With scrape interval at 15s, `[1m]` provides ~4 samples (safe). With 30s, `[1m]` provides 2 samples (borderline). Annotate scrape interval in the dashboard.

### Golden Metrics Dashboard

**TTFT p99 (reactive):**
```promql
histogram_quantile(0.99, sum by (le) (irate(vllm:time_to_first_token_seconds_bucket[1m])))
```

**TTFT p50 (reactive):**
```promql
histogram_quantile(0.50, sum by (le) (irate(vllm:time_to_first_token_seconds_bucket[1m])))
```

**ITL p99:**
```promql
histogram_quantile(0.99, sum by (le) (irate(vllm:time_per_output_token_seconds_bucket[1m])))
```

**E2E Latency p99:**
```promql
histogram_quantile(0.99, sum by (le) (irate(vllm:e2e_request_latency_seconds_bucket[1m])))
```

### System Status Dashboard

**Waiting Requests:**
```promql
vllm:num_requests_waiting
```

**Running Requests:**
```promql
vllm:num_requests_running
```

**KV Cache Usage (%):**
```promql
vllm:gpu_cache_usage_perc
```

**Throughput (tokens/sec):**
```promql
irate(vllm:generation_tokens_total[1m])
```

### GPU Dashboard (DCGM)

**GPU Utilization:**
```promql
DCGM_FI_DEV_GPU_UTIL
```

**GPU Memory Used:**
```promql
DCGM_FI_DEV_FB_USED
```

**GPU Temperature:**
```promql
DCGM_FI_DEV_GPU_TEMP
```

---

## 5. Cross-Checks: Logs vs Graphs

### The Sanity Check
vLLM text logs are the "real-time truth". If the log reports TTFT = 140ms but the Grafana graph shows 35s, the problem is the graph's aggregation window, not the server.

**Procedure:**
1. Identify the anomaly in the Grafana graph.
2. Consult vLLM logs for the same time interval.
3. If logs confirm normal values → echo effect (§2). Wait for the window to flush.
4. If logs confirm anomalous values → real problem. Proceed with the diagnostic tree (§3).

### Useful Commands

**Last vLLM logs (Docker):**
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

## 6. Reference Thresholds — RTX 4070 Super

> ⚠️ Indicative thresholds from internal tests. Vary by quantization, prompt length, concurrency and vLLM config.

| Condition | TTFT p50 | ITL p50 | Notes |
|---|---|---|---|
| **Healthy** (1 req, short prompt) | < 200 ms | < 30 ms | Optimal baseline, no batching |
| **Normal Load** (2–4 req) | < 500 ms | < 50 ms | Continuous batching active |
| **Initial Saturation** | 500 ms – 2s | 50–100 ms | Waiting requests > 0 steadily |
| **Critical Saturation** | **> 2s** | **> 100 ms** | KV cache > 90%, intervention needed |

📝 *Populate with specific values for: Qwen2.5-3B, Qwen2.5-7B, Qwen3-8B, Llama 3.1 8B, Mistral 7B v0.3.*

---

## 7. Incident Response Procedure

Operational workflow when an alert fires or an anomaly is noticed.

**Step 1:** Check p50 and p99 TTFT + ITL

```
p50 OK, p99 high      → Echo effect / outlier
                         Check logs. Wait for flush.

TTFT high, ITL low    → Queue saturation
                         Check Waiting Requests + KV Cache %.

ITL high, TTFT low    → GPU bottleneck
                         Check DCGM GPU Util + quantization.

Everything high       → Total saturation
                         Reduce load immediately.
```

**Step 2:** Verify vLLM logs for confirmation (§5)

**Step 3:** Apply action from diagnostic tree (§3)

**Step 4:** Monitor recovery — p50 normalizes first, p99 follows after window flush.

---

## 8. Pre-Benchmark Checklist

Run before every benchmark session to guarantee clean, comparable data.

- [ ] vLLM freshly started (Prometheus counters reset)
- [ ] No other GPU-intensive processes active (`nvidia-smi`)
- [ ] Prometheus scrape interval verified and annotated
- [ ] Grafana dashboard with `[1m]` windows + `irate`
- [ ] Baseline GPU temperature stable (no thermal throttling)
- [ ] `max_model_len` and vLLM parameters documented
- [ ] Model quantization documented
- [ ] Warmup completed (2–3 warmup requests discarded)

---

## Appendix A: Inference Stack Model Mapping

| Layer | Name | Runbook Coverage |
|---|---|---|
| **L0** | Hardware | RTX 4070 Super thresholds (§6) |
| **L1** | Driver/GPU | DCGM metrics, GPU Util, memory bandwidth (§3, §4) |
| **L2** | Runtime/Pkg | CUDA, driver — implicit prerequisites |
| **L3** | Inference Backend | vLLM metrics, KV cache, queue management (§1–§5) |
| **L4a** | Request Orchestration | Waiting/Running requests (§3, §4) |
| **L4b** | GPU Workload Opt | Continuous batching, preemption (§3) |
| **L5** | API Gateway | — (out of scope) |
| **L6** | Client | — (out of scope) |

## Appendix B: Inference Mastery Model Mapping

| Area | Name | Relevant Sections |
|---|---|---|
| **A** | Inference Pipeline | §1, §7 |
| **B** | Inference Engines | §3, §4 (vLLM specific) |
| **C** | Memory Engineering | §3 (KV cache), §6 |
| **D** | Optimization Techniques | §3 (quantization, max_model_len) |
| **E** | Distributed Inference | — (out of scope, single-GPU) |
| **F** | Performance Measurement | §1, §2, §4, §6 |
| **G** | Production Engineering | §5, §7, §8 |

---

*Dielabs — Inference Observability Runbook v1.0*
