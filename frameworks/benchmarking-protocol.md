---
layout: default
title: "Benchmarking Protocol"
---
# Dielabs Benchmarking Protocol

**Standardized LLM Inference Validation via GuideLLM**

---

## 1. Strategic Vision and Performance Governance

Systematic validation mitigates operational risk and optimizes Total Cost of Ownership (TCO). By using load profiles such as the sweep, we identify the **knee** of the performance curve: the point where the price-performance ratio is maximized, avoiding costly over-provisioning or service degradation.

The framework targets the optimal balance between performance, resource utilization, and cost-effectiveness, providing stakeholders with infrastructure-level guarantees that are independent of model behavior.

---

## 2. Environment Setup

### Setup Checklist

1. **Requirements**: Linux/MacOS, Python 3.10–3.13. Conda environments recommended for dependency isolation.
2. **Installation**: `pip install guidellm` (PyPI) for stability, or source install for experimental features.
3. **Verification**: Run `guidellm --version` to confirm binary integrity.
4. **Backend Validation**: Start the target (e.g., vLLM) and verify endpoint accessibility:

bash

```bash
curl http://localhost:8000/v1/models
```

A valid JSON response listing available models is required before proceeding.

---

## 3. Standard Data Profiles

The dataset selection determines whether we are running a stress test (synthetic data) or a business validation (real data). Synthetic data provides granular control over tokenization, essential for measuring pure decoding efficiency.

### Reference Profiles

|Use Case|prompt_tokens|output_tokens|Primary Metric|Rationale|
|---|---|---|---|---|
|Chat|256|128|TTFT|Balanced interactive exchange; responsiveness is critical|
|RAG|1024|256|TTFT + Throughput|Large retrieved context, moderate generation|
|Summarization|2048|128|Throughput (tok/s)|Massive input, concise output|
|Code Generation|256|512|Throughput (tok/s)|Short prompt, significant output block|
|Code Completion|64|32|TTFT + ITL|IDE-speed latency on very short snippets|
|Agentic / Instruction|512|256|Request latency e2e|Complex instructions, variable output|

### Usage

bash

```bash
--data "prompt_tokens=256,output_tokens=128"    # Chat profile
--data "prompt_tokens=1024,output_tokens=256"   # RAG profile
```

### Pre-processing and Standardization

Note: The `preprocess` command and its `short-prompt-strategy` option are required only when using custom real-world datasets with heterogeneous prompt lengths. When using synthetic profiles via `--data`, token standardization is implicit — no preprocessing or short-prompt handling is needed.

To eliminate bias from heterogeneous prompt lengths, use the `preprocess` command. This ensures each request respects the load parameters defined in the test plan.

bash

```bash
guidellm preprocess input.jsonl output.csv \
  --processor <model_id> \
  --config "prompt_tokens=256,output_tokens=128" \
  --short-prompt-strategy concatenate
```

|Option|Function|Values|
|---|---|---|
|`prompt_tokens`|Target input length|e.g., 256, 512, 1024, 2048|
|`output_tokens`|Tokens to generate|Defines decoding load|
|`short-prompt-strategy`|Short prompt handling|`ignore`, `concatenate`, `pad`, `error`|

### Custom Profiles

If real production data is available, use GuideLLM preprocessing to resize prompts to target token lengths. When using custom data, document the actual token distribution in the benchmark report.

### Protocol Rule

**Always use the same data profile across all runs in a benchmark session.** Changing the profile between phases invalidates Cp identification. Cross-profile comparisons require separate full protocol executions.

---

## 5. The `--rate` Parameter

**Key definitions:**

|Term|Definition|
|---|---|
|Arrival rate|Requests per second submitted to the system.|
|In-flight requests (client-side concurrency)|Requests sent by the benchmark for which a response has not yet been received.|
|Server-side concurrency|Requests actively being processed by the inference engine. Depends on internal scheduling (e.g., vLLM continuous batching) and is not visible to the benchmark.|
|Throughput|Rate of completed requests or generated tokens per unit of time. Always a measurement, never an input.|
|SLO|Service Level Objective. A target performance threshold (e.g., TTFT < 200ms) that defines acceptable service quality for a given use case.|

The meaning of `--rate` changes by profile. Misunderstanding this parameter is one of the most common errors in GuideLLM usage.

|Profile|`--rate` means|Controlled variable|Emergent variable|
|---|---|---|---|
|`sweep`|Number of test points|Depends on underlying profile|—|
|`concurrent`|Concurrency level|In-flight requests (constant)|Arrival rate|
|`constant`|Request rate (RPS)|Arrival rate (fixed)|Concurrency|
|`poisson`|Request rate (RPS)|Arrival rate (average)|Concurrency|
|`synchronous`|N/A|Concurrency = 1|—|

**Critical distinction:** In `concurrent` mode, the benchmark replaces each completed request immediately to maintain the target level. In `constant`/`poisson` modes, requests are emitted at the specified rate regardless of pending completions.

---

## 6. Cp (Critical Point)

The concurrency level (client-side) at which latency degrades non-linearly while throughput gains flatten. The Cp represents the last sustainable operating point — the highest concurrency that is both economically viable (resource cost vs. throughput gain) and qualitatively acceptable (SLO compliance). Beyond the Cp, every additional request degrades the experience for all concurrent users.

**Cp is conditional, not hardware-intrinsic.** It is a property of the hardware + model + data profile combination. Changing the data profile shifts the Cp:

- Higher `prompt_tokens` (e.g., RAG at 1024) → Cp shifts left (prefill cost + KV cache pressure).
- Higher `output_tokens` (e.g., Code Generation at 512) → Cp shifts left (longer slot occupancy).
- Lightweight profiles (e.g., Code Completion at 64/32) → Cp shifts right.

This is why cross-profile Cp comparisons are invalid and the protocol mandates a single data profile per benchmark session.

**Why Cp is defined on client-side concurrency.** The benchmark has no visibility into server-side internals (GPU utilization, batch size, queue depth). This is by design: it mirrors production observability (client/load balancer perspective), ensures reproducibility across engines without depending on engine-specific telemetry, and keeps cross-engine comparisons valid. Server-side observability (e.g., Grafana/Prometheus) explains _why_ the system saturates at a given Cp, but does not participate in its definition.

---

## 7. Three-Phase Benchmarking Protocol

### Phase 1 — Sweep (Macro Profile)

**Objective**: Map the full workload profile across the hardware's sustainable rate range. Identify the approximate region of the Cp.

bash

```bash
guidellm benchmark \
  --target <endpoint> \
  --model <model> \
  --profile sweep \
  --rate <N> \
  --data <profile> \
  --max-seconds <T>
```

GuideLLM explores an arrival rate range determined by internal heuristics and distributes N test points linearly across it. Each point runs at a specific arrival rate; concurrency and throughput are emergent. Note that the range selection is heuristic — GuideLLM does not guarantee optimal coverage of the critical transition zone.

**What to extract:**

1. **Stable zone**: the range of rates where TTFT and throughput scale linearly.
2. **Last stable point**: the highest-rate sweep point still in the stable zone. Note its emergent concurrency — this is the **lower bound** estimate for Cp.
3. **Saturation point**: the sweep point (if any) where TTFT spikes or throughput flattens. Its emergent concurrency is the **upper bound** estimate.

**Limitation**: The sweep distributes points uniformly across the rate range, not across the behavior space. If the system is stable for most of the range, the transition zone may receive minimal or no sampling. This is the fundamental reason Phase 2 exists: the sweep maps the macro profile, Phase 2 resolves the Cp.

The `poisson` profile is particularly useful here for simulating unpredictable web traffic patterns within specific sweep points.

### Phase 2 — Concurrent (Cp Identification)

**Objective**: Surgically identify the Cp by directly controlling concurrency in the transition zone identified by Phase 1.

Take the emergent concurrency from the last stable sweep point. Start slightly below and scan upward:

bash

```bash
# Example: last stable sweep point showed ~9 concurrent
# Scan: 8, 10, 12, 14, 16

guidellm benchmark \
  --target <endpoint> \
  --model <model> \
  --profile concurrent \
  --rate 8 \
  --data <profile> \
  --max-seconds <T>

# Repeat for --rate 10, 12, 14, 16
```

**What to extract** — for each concurrency level, record:

- TTFT (Time to First Token)
- ITL (Inter-Token Latency)
- Throughput (tok/s)
- Request latency (end-to-end)

**Cp Identification Criteria** — the Cp is the concurrency level where:

1. TTFT increases disproportionately relative to the previous step
2. Throughput gains flatten or reverse
3. ITL begins to degrade

The Cp is typically expressed as a range (e.g., Cp ≈ 10–12) rather than a single value.

### Phase 3 — SLO Mapping & Capacity Planning

**Objective**: Anchor the Cp to concrete Service Level Objectives and derive capacity planning parameters for production deployment.

See sections 8 and 9 below.

---

## 8. Metrics and SLO Framework

### Metric Definitions and Business Correlation

|Metric|Measures|Business Impact|
|---|---|---|
|**TTFT** (Time to First Token)|Responsiveness|Critical for user perception in chat and RAG.|
|**ITL** (Inter-Token Latency)|Fluidity|Defines whether streamed text appears choppy or natural.|
|**Throughput** (RPS / tok/s)|Capacity|In batch scenarios, maximizing this minimizes GPU cluster idle time, optimizing hardware amortization.|

### SLO Reference Targets (p99)

|Use Case|TTFT Target|ITL Target|Notes|
|---|---|---|---|
|Chat|< 200 ms|< 30 ms|User-facing responsiveness|
|RAG|< 500 ms|< 50 ms|Tolerable wait for retrieval context|
|Summarization|< 2000 ms|N/A (batch)|Throughput-driven, latency secondary|
|Code Completion|< 100 ms|< 20 ms|IDE real-time autocomplete|
|Code Generation|< 500 ms|< 40 ms|Developer workflow tolerance|
|Agentic|< 1000 ms|< 50 ms|Multi-step pipeline latency budget|

These are **Dielabs internal reference values**, not industry standards. Actual SLO targets vary significantly depending on model size, quantization, streaming vs. non-streaming mode, network latency between client and inference endpoint, and specific product requirements. Use these as a starting baseline and calibrate to measured conditions.

### Max Sustainable RPS

From Phase 2 results, find the highest concurrency level at which **all** defined SLOs are met. The corresponding throughput is the Max Sustainable RPS for that configuration.

> **Max Sustainable RPS** = highest measured RPS where TTFT ≤ SLO_TTFT AND ITL ≤ SLO_ITL

---

## 9. Capacity Planning and Capacity Card

With Max Sustainable RPS per node, derive horizontal scaling requirements:

```
Required nodes = ceil(Target RPS / Max Sustainable RPS per node)
```

### Capacity Card

The primary deliverable for infrastructure planning decisions. A single-page summary table containing:

- Hardware configuration per node
- Model and quantization
- Data profile used
- Cp range identified (Phase 2)
- SLO thresholds applied
- Max Sustainable RPS per node
- Scaling factor for target deployment

---

## 10. Over-Saturation Detection (OSD)

The OSD algorithm is a useful guardrail for safeguarding computational budget. By analyzing the statistical slope of TTFT and concurrent requests, GuideLLM attempts to detect when the system enters an "infinite queue" state — where additional load produces no useful data while consuming cluster time.

**Caveat**: OSD reliability depends on GuideLLM's internal heuristics and has not been independently validated across all hardware configurations and model sizes. Treat OSD as a **reasonable safety net, not a guaranteed oracle**. Always cross-check OSD behavior against Grafana/Prometheus metrics to confirm saturation is real and not a false positive.

**Enterprise configuration recommendation:**

|Parameter|Value|Rationale|
|---|---|---|
|`min_seconds`|30.0|Minimum test duration before OSD can trigger|
|`moe_threshold`|2.0|Margin of error threshold for saturation detection|

This prevents saturated tests from continuing to consume hours of H100 cluster time producing distorted, unusable data.

---

## 11. Resource Optimization and Scaling Strategies

Benchmark data guides parallelism configuration in the backend (e.g., vLLM) to optimize TCO.

| Strategy                  | When to Use                      | Effect                                                                                                                                                                                                                                                         |
| ------------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tensor Parallel (TP)**  | Reduce latency on massive models | Distributes model layers across GPUs. Effective when model size exceeds single-GPU memory or when per-request latency must decrease.                                                                                                                           |
| **Context Parallel (CP)** | Long-context architectures       | Distributes the prefill computation and KV cache across devices for requests with large context windows. Effectiveness depends heavily on implementation (framework-specific); not all backends support CP, and overhead may negate gains on shorter contexts. |
| **Expert Parallel (EP)**  | MoE models (e.g., DeepSeek)      | Balances token load among experts, avoiding bottlenecks on individual GPUs. Requires careful routing to prevent expert imbalance.                                                                                                                              |

### Linking Cp to Scaling Decisions

The Cp identified in Phase 2 is a direct diagnostic input for scaling strategy:

|Cp Symptom|Likely Bottleneck|Recommended Action|
|---|---|---|
|Low Cp with high TTFT|Prefill-bound (memory bandwidth or context size)|Evaluate CP, or disaggregated prefill/decode instances|
|Low Cp with high ITL|Decode-bound (compute or scheduling contention)|Increase TP, or scale horizontally|
|Cp scales linearly with batch size|System is well-balanced|Optimize at current parallelism; scale horizontally for capacity|

**Scaling directive**: If sweep tests show ITL exceeding 100ms under load, increase `tensor_parallel_size` or evaluate disaggregated prefill/decode instances.

**CPU-specific note**: For CPU configurations with multiple architectures (SN, TP, DP), run the full three-phase protocol independently for each architecture. Cross-architecture Cp and capacity comparison is a primary deliverable (ref. KB-INF-004).

---

## 12. Operational Notes

- **Warmup before measurement.** Run a short warmup phase (e.g., 10–30 requests at low concurrency) before starting any benchmark run. Cold-start effects — CUDA kernel compilation, memory allocator initialization, KV cache first-allocation — introduce significant bias in TTFT and throughput measurements. Discard warmup data. On CPU, NUMA memory allocation and thread pool stabilization produce similar cold-start artifacts.
- Always use the same dataset, data profile, and `--max-seconds` across all runs in a benchmark session.
- Record hardware state (GPU utilization, temperature, memory) via Grafana/Prometheus during runs.
- The sweep does not control arrival rate directly — it determines the range autonomously. The `--rate` parameter only controls how many points to sample.
- Concurrent mode tests are independent — each `--rate` value is a separate run with a fixed concurrency level.
- Save benchmark outputs in JSON format for post-analysis and Grafana import.
- When using custom datasets, run GuideLLM preprocessing to standardize token distributions before benchmarking.

---

## 14. Benchmark Anti-Patterns

Common errors that invalidate results or lead to incorrect infrastructure decisions.

|#|Anti-Pattern|Why It's Wrong|Correct Approach|
|---|---|---|---|
|1|**Changing data profile between runs**|Cp is profile-dependent. Mixing profiles within a session makes Phase 1→Phase 2 triangulation meaningless.|One profile per full protocol execution. Cross-profile comparison requires separate sessions.|
|2|**Comparing RPS across different data profiles**|A system producing 100 RPS at 64/32 tokens is not comparable to one producing 40 RPS at 2048/128. The workload is fundamentally different.|Compare only within the same profile. Use tok/s as secondary normalization if cross-profile context is needed.|
|3|**Reporting throughput without TTFT**|High throughput can coexist with unacceptable latency. A system batching aggressively may show excellent tok/s while individual users wait seconds for the first token.|Always report TTFT, ITL, and throughput together. Throughput alone is a vanity metric.|
|4|**Ignoring emergent concurrency in sweep results**|Reading only RPS from sweep output misses the actual system state. Two sweep points with the same RPS can have different concurrency — and different user-facing behavior.|Extract emergent concurrency for every sweep point. It is the bridge to Phase 2.|
|5|**Not isolating variables**|Changing model, quantization, and backend version simultaneously between runs makes it impossible to attribute performance differences.|Change one variable at a time. Document the full stack (model, quantization, backend version, parallelism config, OS tuning) in every benchmark report.|
|6|**Skipping warmup**|Cold-start artifacts (kernel compilation, allocator init, KV cache first-allocation) inflate TTFT by 2–10x on the first requests.|Run warmup phase, discard initial data. See section 13.|
|7|**Using wall-clock duration as the only stopping criterion**|A 60-second run at low concurrency may generate too few requests for statistical significance. A 60-second run at high concurrency may generate thousands — but under saturation, most are queued, not served.|Combine `--max-seconds` with minimum request count validation. Check that the sample size is sufficient for p99 reporting.|
|8|**Benchmarking without observability**|GuideLLM measures client-side metrics. Without server-side telemetry, you cannot distinguish whether TTFT degradation is caused by scheduling contention, memory pressure, thermal throttling, or network latency.|Run Grafana/Prometheus with GPU (DCGM) or CPU (node-exporter, IPMI) metrics during every benchmark session.|

---

## Protocol Summary

|Phase|Mode|`--rate` means|Controls|Measures|
|---|---|---|---|---|
|1 — Sweep|`sweep`|Test point count|Arrival rate (auto)|Emergent concurrency, throughput, latency|
|2 — Concurrent|`concurrent`|Concurrency level|Concurrency (direct)|Throughput, latency at each level|
|3 — SLO Mapping|—|—|SLO thresholds|Max Sustainable RPS, capacity estimate|

```
Phase 1: Sweep
  └─ Maps macro profile
  └─ Identifies Cp region (lower bound from last stable point)

Phase 2: Concurrent
  └─ Scans transition zone with controlled concurrency
  └─ Pinpoints Cp with precision

Phase 3: SLO Mapping & Capacity Planning
  └─ Anchors Cp to use-case SLO thresholds
  └─ Derives Max Sustainable RPS and scaling factor
  └─ Produces Capacity Card
```

---

## Applicability

This protocol applies to:

- GPU inference (single GPU, multi-GPU)
- CPU inference (single-node, TP, DP configurations)
- Any vLLM-compatible backend
---

## Operational Tuning: Latency vs Throughput Levers

Once the Crossover Point is identified, the next decision is how to operate: latency-first or throughput-first. Two vLLM parameters define this operational profile.

### The Two Primary Levers

| Parameter | Protects | Sacrifices |
|---|---|---|
| `max_num_seqs` ↓ | **Latency** (ITL per request) — fewer sequences in the batch → less GPU contention | Aggregate throughput |
| `max_model_len` ↓ | **Sustainable throughput** — shorter max context per sequence → more sequences fit in the KV cache pool | Long-context capability |

### How They Work

**`max_num_seqs`** is the primary operational lever. It sets a hard cap on concurrent sequences in the batch, regardless of available KV cache capacity. Lowering it reduces per-request latency (less contention per decode step); raising it increases aggregate throughput (more tokens produced per forward pass, amortizing the memory bandwidth cost).

- **Latency-optimized profile:** `max_num_seqs` set at or below the Crossover Point. Each user gets fast responses, but total system throughput is capped.
- **Throughput-optimized profile:** `max_num_seqs` set above the Crossover Point. The system serves more users simultaneously, but individual latency degrades.

**`max_model_len`** is set once based on workload profile and left fixed. It controls the maximum context length per sequence. Lowering it means each sequence consumes fewer KV cache blocks, freeing capacity for more concurrent sequences. It does not directly affect per-token latency — it affects how many sequences the pool can sustain.

### Decision Framework

After running the Benchmarking Protocol and identifying the Crossover Point (Cp):

1. **Latency-first deployment** (chat, interactive): set `max_num_seqs` ≤ Cp. Monitor ITL p99.
2. **Throughput-first deployment** (batch processing, async): set `max_num_seqs` > Cp. Monitor aggregate tokens/sec.
3. **Balanced deployment**: set `max_num_seqs` = Cp (the sweet spot). Monitor both.

The Cp is what makes this decision quantitative rather than guesswork. Without it, tuning `max_num_seqs` is trial and error. With it, you know exactly where the latency/throughput trade-off inflects.
