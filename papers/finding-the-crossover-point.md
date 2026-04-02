---
layout: default
title: "Finding the Crossover Point"
---

# Finding the Crossover Point

## Abstract

This paper documents a systematic benchmark of an NVIDIA RTX 4070 Super (12 GB VRAM) with Llama 3.1 8B Instruct AWQ via vLLM, with the objective of identifying the crossover point (Cp) between latency and throughput. **The crossover point is the concurrency level at which the cost in latency begins to exceed the gain in throughput.** Knowing this point is essential for capacity planning and operational configuration of any inference deployment.

The benchmark was conducted in two phases: a GuideLLM sweep to map the system profile, followed by targeted concurrent tests at fixed concurrency levels (8, 10, 12, 14, 16, 18) to localize the Cp. In this benchmark configuration, the crossover point falls at 14 concurrent requests, with the sweet spot (best latency/throughput compromise) at 12 concurrent requests, delivering 107 tok/s with a TTFT of ~280 ms.

---

## 1. Hardware and Software Configuration

### 1.1 Hardware

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Super |
| VRAM | 12 GB GDDR6X |
| Memory bandwidth | 504 GB/s |
| Host CPU | Intel i7 (gpu_node) |
| Host RAM | 32 GB DDR5 |
| Network | 192.168.4.250 |

### 1.2 Software Stack

| Layer | Component | Version / Config |
|---|---|---|
| Inference Engine | vLLM | v0.17.1 |
| Model | Meta-Llama-3.1-8B-Instruct-AWQ-INT4 | hugging-quants |
| Quantization | AWQ 4-bit | ~5.5 GB model footprint |
| Benchmark Tool | GuideLLM | v0.5.3 |
| Observability | Prometheus + Grafana + DCGM | gpu-dcgm exporter |
| Container Runtime | Docker / Portainer | |

### 1.3 vLLM Launch Parameters

| Parameter | Value | Rationale |
|---|---|---|
| --gpu-memory-utilization | 0.85 | Conservative: 15% buffer to avoid OOM |
| --max-model-len | 2048 | Sufficient for benchmark, leaves room for KV cache |
| --tensor-parallel-size | 1 | Single GPU |
| --dtype | half | FP16 compute with AWQ 4-bit weights |
| --quantization | awq | AWQ format for the model |

---

## 2. Methodology

### 2.1 The Crossover Point

The crossover point (Cp) is defined as the first concurrency level at which the percentage increase in latency (TTFT) exceeds the percentage increase in throughput. Below the Cp, the system gains more capacity than it loses in responsiveness. **Above the Cp, each additional concurrent request costs more in user experience than it adds in capacity.**

> _Cp = first step where ΔTTFT% > ΔThroughput%_

This definition, formalized in the Dielabs Inference Operating Envelope framework, provides a concrete and measurable decision point for capacity planning.

### 2.2 Two-Phase Benchmark Design

The benchmark was designed in two phases, each with a distinct purpose.

**Phase 1: Sweep.** A GuideLLM sweep (profile: sweep, rate: 10, max-seconds: 90) maps the system's general behavior across a range of requests per second. The sweep distributes test points linearly between the synchronous rate and the saturation rate. It answers the question: how does this system behave as load increases? The sweep serves to determine how a given workload performs on a given infrastructure, within its boundary values, and is not a precision instrument for finding the Cp which presumably lies above it.

**Phase 2: Concurrent tests.** Targeted tests at fixed concurrency levels (8, 10, 12, 14, 16, 18) localize the Cp by forcing the system, controlling the exact number of concurrent requests. Each test lasted 90 seconds with fixed random seed (42), fixed prompt length (512 input tokens) and fixed output length (256 output tokens). This eliminates variability from prompt length and ensures comparability across runs.

The two phases are complementary. The sweep reveals the approximate zone where the system transitions from latency-friendly to throughput-limited. The concurrent tests zoom in from that zone and identify the transition by climbing upward.

### 2.3 Why the Sweep Alone Is Not Enough

The sweep distributes its test points linearly between the minimum and maximum sustainable rate. If the system comfortably handles most of that range (as the RTX 4070 does with this model), the majority of sweep points fall in the stable zone, and the transition zone is sampled with only one or two points. In our sweep, the system showed TTFT between 280 and 335 ms from rate 0.10 to 0.49, with throughput scaling linearly from 21 to 94 tok/s. The Cp was somewhere between the last stable sweep point (~9 concurrent) and the saturation test (~17 concurrent), but the sweep did not provide data points in that gap.

This is not a limitation of GuideLLM. The sweep does exactly what it is designed for: **mapping the workload profile on the hardware**. The limitation lies in using the sweep as the sole instrument to find the Cp. The concurrent tests fill the gap.

### 2.4 Benchmark Parameters

| Parameter | Value |
|---|---|
| Input tokens | 512 (fixed, synthetic) |
| Output tokens | 256 (fixed) |
| Random seed | 42 |
| Warmup | 2% |
| Max seconds per test | 90 |
| Backend | openai_http |
| Saturation detection | Enabled (sweep only) |

---

## 3. Results

### 3.1 Phase 1: Sweep Results

The sweep established the performance baseline and the general scaling profile of the system.

| Rate (req/s) | Completed | Avg conc. | Avg TTFT (ms) | Avg ITL (ms) | Throughput (tok/s) |
|---|---|---|---|---|---|
| sync | 5 | 1.0 | 199 | 85.7 | 14.5 |
| 0.102 | 8 | 1.9 | 313 | 91.8 | 20.9 |
| 0.158 | 12 | 3.0 | 313 | 92.6 | 32.5 |
| 0.270 | 19 | 5.0 | 318 | 94.2 | 52.9 |
| 0.326 | 22 | 5.9 | 324 | 94.9 | 61.7 |
| 0.382 | 26 | 7.1 | 324 | 96.2 | 73.3 |
| 0.438 | 30 | 8.3 | 328 | 96.8 | 84.9 |
| 0.495 | 33 | 9.2 | 335 | 97.8 | 93.6 |

Key observations from the sweep: TTFT jumps from 199 ms (synchronous) to ~313 ms at the first constant rate, then remains substantially flat (313–335 ms) for all subsequent rates. Meanwhile, throughput scales nearly linearly from 21 to 94 tok/s. ITL rises gradually from 86 to 98 ms. The system shows no signs of degradation within the sweep range.

### 3.2 Phase 2: Concurrent Test Results

The concurrent tests targeted the zone between 8 and 18 simultaneous requests, where the sweep could not reach.

| Concurrency | Completed | Avg TTFT (ms) | TTFT p90 (ms) | Avg ITL (ms) | Throughput (tok/s) |
|---|---|---|---|---|---|
| 8 | 25 | 723 | 1022 | 93.6 | 72.6 |
| 10 | 31 | 823 | 1107 | 94.6 | 90.0 |
| 12* | 37 | 277 | 363 | 93.0 | 107.4 |
| 14 | 43 | 859 | 1546 | 96.2 | 124.8 |
| 16 | 49 | 960 | 1828 | 97.8 | 142.2 |
| 18 | 49 | 3479 | 24163 | 98.4 | 142.2 |

_* Confirmed with a second run: average TTFT 277 ms, throughput 107.4 tok/s (identical). The conc 12 result is reproducible. See Section 3.4 for interpretation._

### 3.3 Visual Analysis

**Figure 1:** TTFT and throughput vs concurrency. The green dashed line indicates the sweet spot (conc 12), the red dashed line indicates the crossover point (conc 14). Bars exceeding the chart scale show the actual value as labels.

**Figure 2:** Inter-token latency vs concurrency. ITL rises gradually (+15% total) and shows no discontinuity, confirming that the bottleneck is in the prefill phase (TTFT), not in token generation.

### 3.4 On the Conc 8–10 vs Conc 12 Latency Pattern

The concurrent results show a non-monotonic TTFT pattern: conc 8 (723 ms) and conc 10 (823 ms) have higher TTFT than conc 12 (277 ms). This is counterintuitive and requires explanation.

The conc 12 result is not an artifact. A second run confirmed identical throughput (107.4 tok/s) and consistent TTFT (277 ms vs 328 ms in the first run, well within normal TTFT variance). Throughput and ITL are stable across both runs.

The most probable interpretation is a scheduling efficiency boundary in vLLM's continuous batching. At conc 8–10, the scheduler may operate in a less favorable batching regime: enough requests to create prefill contention, but not enough to fill the batch efficiently, resulting in partially occupied prefill iterations and higher average TTFT. At conc 12, the batch may reach a size where the prefill pipeline is more fully utilized per iteration, amortizing scheduling overhead over more requests and producing lower per-request TTFT.

**This hypothesis is consistent with known continuous batching scheduler behavior, where batch efficiency is not a monotonically increasing function of concurrency but can exhibit local optima dependent on the interaction between the prefill/decode pipeline, the model's KV cache footprint and the scheduler's chunked prefill policy.** A definitive explanation would require profiling vLLM's internal scheduling decisions, which is outside the scope of this benchmark.

**For the purposes of this analysis, we treat the conc 12 result as valid (reproducible) and the conc 8–10 results as reflecting a less efficient scheduling regime rather than measurement error.** The Cp determination at 14 is not affected by this pattern, as it depends on the 12→14 delta, where both the numerator (conc 12) and denominator (conc 14) are individually stable.

### 3.5 Delta Analysis and Crossover Point

Applying the Cp definition (ΔTTFT% > ΔThroughput%) to the concurrent test results:

| Step | ΔThroughput % | ΔTTFT % | ΔITL % | Crossover? |
|---|---|---|---|---|
| 8 → 10 | +24.0% | +13.9% | +1.1% | No |
| 10 → 12 | +19.4% | −60.2% | −1.2% | No |
| 12 → 14 | +16.2% | +162.4% | +2.9% | **YES ← Cp** |
| 14 → 16 | +14.0% | +11.7% | +1.7% | No |
| 16 → 18 | +0.0% | +262.3% | +0.6% | YES (collapse) |

**The crossover point is at 14 concurrent requests.** At step 12→14, TTFT increases by 162% while throughput gains only 16%. This is the first step where the latency cost decisively exceeds the throughput benefit.

**The sweet spot is at 12 concurrent requests.** It is the last concurrency level before the Cp. At conc 12, the system delivers 107.4 tok/s with a TTFT of ~280 ms and an ITL of 93 ms. In this benchmark configuration, this represents the best achievable compromise: maximum throughput without entering the degradation zone.

**The collapse point is at 18 concurrent requests.** Throughput stops growing (142.2 tok/s, identical to conc 16) while TTFT p90 explodes to 24 seconds. The system is generating queue, not capacity. Beyond this point, additional concurrency produces no benefit and produces catastrophic latency.

---

## 4. GPU Telemetry

Prometheus/DCGM telemetry during the benchmark provides hardware-level context. Data from the Grafana dashboard (GPU Node Monitoring for Benchmarks, UID: Oxed_c6Wz).

| Metric | Idle | Under Load | Notes |
|---|---|---|---|
| GPU Utilization | 0% | 100% | Saturated for all tests |
| Memory BW Util. | 8% | 87–88% | Stable, no bandwidth saturation |
| VRAM Used | 11,149 MB | 11,149 MB | Constant: vLLM preallocates at startup |
| Power Draw | 4 W | 165–182 W | Higher at higher concurrency |
| Temperature | 41°C | 68–76°C | Well within thermal limits |

The GPU is compute-saturated (100% utilization) during all test phases, but memory bandwidth utilization remains at 87–88% and does not appear to be the bottleneck. VRAM is entirely preallocated by vLLM at startup and does not fluctuate during the benchmark. Performance degradation at high concurrency is consistent with scheduling contention and prefill queueing rather than hardware saturation, although definitive attribution would require deeper profiling.

---

## 5. Analysis

### 5.1 The Operating Envelope

Based on these results, the Inference Operating Envelope for this configuration can be characterized as follows:

| Zone | Concurrency | TTFT | Throughput | Behavior |
|---|---|---|---|---|
| Optimal | 1–12 | ≤ 335 ms | ≤ 107 tok/s | Latency stable, throughput scales |
| Transition | 12–14 | 335 → 859 ms | 107 → 125 tok/s | Cp zone: latency cost > throughput gain |
| Degradation | 14–16 | 859 → 960 ms | 125 → 142 tok/s | Diminishing returns, high tail latency |
| Collapse | 18+ | > 3,000 ms | ~142 tok/s | Queue buildup, no throughput gain |

### 5.2 The TTFT Step Function

A notable result is the TTFT behavior between synchronous mode (199 ms) and the first constant rate (313 ms). It is not a gradual increase but a discrete step: **a fixed ~120 ms penalty for operating in batched mode.** Once this penalty is paid, TTFT remains stable across a wide range of concurrency levels (up to ~12 in the concurrent tests and up to ~9 in the sweep). This suggests the penalty is caused by the transition from exclusive to shared GPU scheduling, not by incremental resource contention.

**The practical implication is that there is no benefit in operating at very low concurrency (2–3 requests) if the use case can tolerate ~300 ms TTFT. The system behaves similarly at conc 3 and conc 12 in terms of latency, but throughput at conc 12 is 3–5x higher.**

### 5.3 ITL Stability

Inter-token latency (ITL) rises gradually from 85.7 ms (synchronous) to 98.4 ms (conc 18), a total increase of 15% across the entire concurrency range. This indicates that the decode phase (token-by-token generation) is not the primary bottleneck in this configuration. The bottleneck appears to be in the prefill phase (TTFT), where new requests must be scheduled and their initial attention computation executed. Once a request is generating tokens, it proceeds at a relatively stable rate regardless of how many other requests are in-flight.

### 5.4 Throughput Plateau

In this benchmark configuration, throughput reaches a plateau at ~142 tok/s (conc 16 and 18 produce identical throughput). This appears to represent the capacity limit of the RTX 4070 Super with this model, this quantization and these vLLM settings. Beyond this point, additional requests queue without producing additional output tokens per second.

**The difference between throughput at the sweet spot (107 tok/s at conc 12) and the plateau (142 tok/s) is approximately 33%. This gap quantifies what the system would sacrifice in user-side latency to reach maximum capacity — under these specific test conditions.**

---

## 6. Methodological Insights

### 6.1 Sweep + Concurrent: A Two-Phase Approach

This benchmark validates a two-phase methodology for finding the crossover point:

**Phase 1 (Sweep)** maps the system profile and identifies the approximate transition zone. It is fast (15–20 minutes) and requires no prior knowledge of hardware limits.

**Phase 2 (Concurrent)** targets the transition zone with fixed concurrency levels and produces the Cp. It requires Phase 1 results to set the correct concurrency range.

Neither phase is sufficient alone. The sweep without concurrent tests leaves the Cp unlocated. Concurrent tests without the sweep risk testing the wrong range.

### 6.2 Test Duration Matters

An initial sweep with 30-second tests produced only 1–4 completed requests per step, making statistical analysis impossible. Increasing to 90 seconds yielded 8–33 completed requests per step, providing meaningful averages and percentiles. For models with long generation times (~24 seconds per request at 512/256), 90 seconds per test point is the practical minimum.

### 6.3 Reproducibility

The conc 12 test was repeated to verify an unexpectedly favorable result. Both runs produced identical throughput (107.4 tok/s) and nearly identical ITL (93.4 vs 93.0 ms), with TTFT within normal variance (328 vs 277 ms). This confirms the result is reproducible and validates the benchmark methodology.

---

## 7. Limitations and Scope

The results reported in this paper are specific to the benchmark configuration used. The following factors define the validity scope and should be considered when interpreting or generalizing these results:

**Workload:** All tests used fixed-length synthetic prompts (512 input tokens, 256 output tokens). Real workloads have variable prompt and output lengths, which affect KV cache pressure, prefill time and scheduling behavior differently. The crossover point and sweet spot may shift with variable-length workloads.

**Model and quantization:** Results apply to Meta-Llama-3.1-8B-Instruct-AWQ-INT4, an AWQ 4-bit quantized model with approximately 5.5 GB footprint. A model of different size, architecture or quantization format (GPTQ, FP16, FP8) would produce different latency and throughput profiles and potentially a different Cp.

**vLLM configuration:** Tests used gpu-memory-utilization=0.85 and max-model-len=2048. Modifying these parameters directly affects KV cache capacity, which in turn affects the number of concurrent sequences the system can sustain before scheduling degrades. A higher gpu-memory-utilization (e.g. 0.95) would increase available KV cache but also OOM risk. A different max-model-len would change per-request memory footprint.

**Software versions:** Results were obtained with vLLM v0.17.1 and GuideLLM v0.5.3 on Linux 6.8.0. Scheduler behavior, batching strategies and memory management can change between vLLM releases, potentially shifting the Cp.

**Hardware:** The RTX 4070 Super has 12 GB VRAM and 504 GB/s memory bandwidth. GPUs with different memory capacity or bandwidth will exhibit different scaling behavior and different crossover points.

**Scheduling anomaly:** The non-monotonic TTFT pattern at conc 8–10 vs conc 12 (Section 3.4) remains an observation without a definitive causal explanation. While the conc 12 result is reproducible, the underlying scheduling dynamics would require vLLM internal profiling to be fully understood.

These limitations do not invalidate the methodology. The two-phase approach (sweep + concurrent) and the Cp definition are portable to other configurations. What changes is the specific Cp value, not the method for finding it.

---

## 8. Conclusions

In this benchmark configuration, the RTX 4070 Super with Llama 3.1 8B Instruct AWQ via vLLM has a crossover point at 14 concurrent requests and a sweet spot at 12 concurrent requests. At the sweet spot, the system delivers 107 tok/s with a TTFT of ~280 ms and an ITL of 93 ms.

The operating envelope has four distinct zones: optimal (1–12 concurrent), transition (12–14), degradation (14–16) and collapse (18+). Throughput reaches its plateau at ~142 tok/s, achieved at 16 concurrent requests. Beyond that point, additional concurrency produces no throughput gain and produces catastrophic latency.

The two-phase methodology (sweep + concurrent) proved effective and reproducible. The sweep maps the profile; the concurrent tests find the Cp. This methodology is portable to any single-GPU inference deployment and can be applied to different models, hardware and configurations by repeating the same process.

> _The question is not what the maximum throughput is. The question is: at what concurrency does this system transition from serving well to struggling, and what throughput does it deliver at that boundary? For this configuration, the answer is 12 concurrent requests at 107 tok/s._

This is the concurrency level I would use as the initial production operating target for this node.

---

## Appendix: Environment Details

**Benchmark tool:** GuideLLM v0.5.3, Python 3.12.3, Linux 6.8.0-101-generic x86_64.

**Model repository:** hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 (HuggingFace).

**Observability:** Prometheus + Grafana with nvidia-dcgm-exporter on gpu_node (192.168.4.250:9400). Grafana dashboard: GPU Node Monitoring for Benchmarks (UID: Oxed_c6Wz, v45). Prometheus datasource UID: bfanhgsrp7chsc.

All timestamps in CET (UTC+1). Benchmark executed March 20, 2026.

**Dielabs Inference Stack:** L0=Hardware (RTX 4070 Super), L1=Driver/CUDA, L2=Docker/Portainer, L3=vLLM, L4a=GuideLLM, L5=OpenAI-compatible API, L6=Benchmark client.
