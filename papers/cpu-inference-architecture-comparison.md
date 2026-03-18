---
title: "Inference Architecture Comparison – Finding the Cross-Over Point"
subtitle: "NUMA-Aware Parallel Architectures on Dell PowerEdge R730 with vLLM"
author: Diego – Dielabs
date: 2026-03
tags: [dielabs, inference, cpu, vllm, numa, tensor-parallel, data-parallel, benchmark, r730]
---

# Inference Architecture Comparison

**Finding the Cross-Over Point**  
NUMA-Aware Parallel Architectures on Dell PowerEdge R730 with vLLM

*Diego – Dielabs | March 2026*

---

## Abstract

Analysis and benchmarking of LLM sharding architectures on a Dell PowerEdge R730 dual-socket server (2× Xeon E5-2680 v4, 128 GB DDR4). The goal is to use CPU inference as an analytical instrument to understand the internal mechanics of the inference stack: how architectural patterns behave as workload changes, where real bottlenecks emerge, and which principles are universal regardless of GPU hardware.

**Three operational findings:**

1. Cross-socket Tensor Parallel via QPI is a **negative pattern** for small models, with degradation up to 79%.
2. Data Parallel is the **recommended pattern** on multi-socket architectures, with a crossover point consistently observed at ~6 concurrent requests.
3. Distributed inference patterns behave as **general principles** transferable across different substrates. The analogy NUMA node ↔ GPU, DDR4 ↔ HBM, QPI ↔ NVLink is conceptual and structural, not performance-based.

---

## 1. Why CPU Inference

**As an analytical tool.** Removing the GPU exposes mechanisms that are masked by raw speed on GPU: scheduler behavior, KV cache management, communication overhead in parallel patterns. It is reverse engineering of the inference pipeline.

**As an infrastructure option.** Edge deployments, regulated environments, development and testing: scenarios where GPUs are unavailable. Not fast, but functional.

**As a mental model.** Understanding why Tensor Parallel fails on QPI makes it clearer why it works on NVLink.

---

## 2. Test Environment

### 2.1 Hardware

| Parameter | Value |
|---|---|
| **Server** | Dell PowerEdge R730 |
| **CPU** | 2× Intel Xeon E5-2680 v4 @ 2.40 GHz (14C/28T, Broadwell) |
| **ISA** | AVX2 (no AVX-512, no native BF16) |
| **RAM** | 128 GB DDR4-2133 ECC (8×16 GB, ~2/4 channels per socket) |
| **Estimated bandwidth** | ~34 GB/s per socket (50% of theoretical maximum) |
| **L3 Cache** | 35 MB per socket (70 MB total) |
| **Storage** | PERC H730 Mini, 931 GB SATA HDD |
| **Idle power** | ~147W (measured via IPMI) |

> **Intrinsic limitation:** the RAM configuration uses ~2/4 channels per socket → ~34 GB/s vs ~68 GB/s theoretical. LLM inference on CPU is **memory-bandwidth bound**: this limitation affects all tested configurations uniformly. Absolute throughput results could improve by up to ~2× with fully populated memory channels.

### 2.2 Software Stack

| Component | Version |
|---|---|
| **OS** | Ubuntu 24.04.4 LTS, kernel 6.8.0-106 |
| **Python** | 3.12.3 |
| **PyTorch** | 2.10.0+cpu |
| **vLLM** | 0.17.1rc1.dev136 (build from source, CPU backend, AVX2) |
| **GCC** | 13.3.0 |
| **GuideLLM** | Latest stable release |
| **Model** | microsoft/Phi-4-mini-instruct (bfloat16, ~4B params) |
| **Observability** | Prometheus + Grafana + IPMI exporter + node_exporter |

> The `bfloat16` dtype is **emulated in software** on Broadwell (AVX2). Native AVX512_BF16 instructions are available from Cooper Lake/Sapphire Rapids onward. This penalizes absolute performance but does not invalidate relative comparisons.

---

## 3. The Three Architectures

A dual-socket server shares the same base topology as a multi-GPU system: two independent compute units, each with local memory, connected by an interconnect bus. Distributed inference patterns apply identically.

### 3.1 Single-Node Inference (SN)

The entire model resides on a single NUMA node (Socket 0). Zero inter-socket communication. The local DDR4 bandwidth is the only constraint. This is the **baseline**: the bare cost of inference with no parallelism overhead.

- 12 physical cores assigned to inference
- 2 cores reserved for OS and HTTP frontend process
- Hyperthreads (CPUs 28–55) excluded from binding (contention on AVX2 vector units)

### 3.2 Intra-Node Tensor Parallel (TP-2)

Model weights are sharded across both sockets. Each layer is computed in parallel on the two NUMA nodes with an **all-reduce via QPI** after every layer. Communication library: Gloo (not NCCL). vLLM treats each NUMA node as a virtual GPU card.

> KV cache is allocated per-rank: each socket reserves a full cache space in its local RAM. Attention heads are distributed across ranks, but the cache is not truly shared → linear increase in total memory usage.

### 3.3 Intra-Node Data Parallel (DP-2)

Two complete copies of the model, one per NUMA node. Each replica serves a subset of requests independently. **Zero inter-socket communication** during inference. vLLM handles load balancing via an internal router process (ZMQ). Each rank has its own fully independent KV cache.

### 3.4 Structural Comparison

|  | **SN** | **TP-2** | **DP-2** |
|---|---|---|---|
| **Weights in RAM** | ~8 GB (1×) | ~8 GB | ~16 GB (2×) |
| **KV Cache** | Local, single | Per-rank (40 GB × 2) | Independent per rank |
| **Communication** | Zero | All-reduce every layer | Zero (frontend LB only) |
| **Active cores** | 12 (1 socket) | 24 (2 sockets) | 24 (2 sockets) |
| **RAM used** | ~51 GB | ~92 GB* | ~81 GB |

*\*In TP, KV cache is not shared and total memory grows more due to per-rank buffers, activations, and overhead.*

### 3.5 CPU vs GPU: Time-Sharing vs True Parallel Batching

**GPU:** executes request batches leveraging massive parallelism (thousands of CUDA cores). Continuous batching maximizes occupancy and HBM bandwidth utilization. Increasing concurrency improves efficiency up to saturation, with stable ITL.

**CPU:** applies continuous batching but with reduced effectiveness due to:
- limited core count
- DDR bandwidth constraints
- scheduling overhead (threading, NUMA, synchronization)

Increasing concurrency does not scale linearly: it introduces cache/memory contention and internal queueing, with progressive ITL growth.

---

## 4. Test Configuration

### 4.1 NUMA Binding

```
Topology (lscpu -e):
  Socket 0: even physical cores  → 0, 2, 4, ..., 26
  Socket 1: odd physical cores   → 1, 3, 5, ..., 27
  Hyperthreads: 28-55 (excluded)
```

Binding managed via `VLLM_CPU_OMP_THREADS_BIND` with pipe syntax (`|`) to separate ranks. vLLM automatically infers the OpenMP thread count from the binding.

### 4.2 Launch Commands

**Single-Node (SN)**
```bash
VLLM_CPU_KVCACHE_SPACE=40 VLLM_CPU_OMP_THREADS_BIND="0,2,4,...,22" \
numactl --cpunodebind=0 --membind=0 vllm serve ... --dtype bfloat16 -tp 1
```

**Tensor Parallel (TP-2)**
```bash
VLLM_CPU_KVCACHE_SPACE=40 VLLM_CPU_OMP_THREADS_BIND="0,2,...,22|1,3,...,23" \
vllm serve ... --dtype bfloat16 -tp 2 --distributed-executor-backend mp
```

**Data Parallel (DP-2)**
```bash
VLLM_CPU_KVCACHE_SPACE=30 VLLM_CPU_OMP_THREADS_BIND="0,2,...,22|1,3,...,23" \
vllm serve ... --dtype bfloat16 --data-parallel-size 2 --distributed-executor-backend mp
```

> KV cache: 40 GB for SN and TP (single instance), 30 GB per DP rank (two instances, ~62 GB available per socket minus weights and OS).

### 4.3 Benchmark Profiles

| Profile | Parameters | Goal |
|---|---|---|
| **Light** | prompt=256, output=128, concurrent=1 | Baseline latency (TTFT, ITL) without contention |
| **Medium** | prompt=512, output=256, concurrent=4 | Contention under moderate load |
| **Sweep** | prompt=256, output=128, rate=1,2,4,6,8 | Saturation point and crossover identification |

**Concurrent vs Rate:**
- **Concurrent** = simultaneous in-flight requests (direct concurrency control, ideal behavior)
- **Rate** = req/s arrival rate independent of system capacity (concurrency is an emergent variable, real-world scenario)

Saturation, queueing, and degradation emerge in the rate sweep, enabling identification of the **crossover point**.

---

## 5. Results

### 5.1 Light Workload (concurrent=1)

| Metric | SN | TP-2 | DP-2 |
|---|---|---|---|
| **Completed** | 9/14 (64%) | 6/11 (55%) | 7/12 (58%) |
| **E2E latency p50** | 32.7s | 57.2s | 43.6s |
| **TTFT p50** | 278 ms | 5,948 ms | 10,734 ms |
| **ITL p50** | 255 ms | 403 ms | 258 ms |
| **TPOT p50** | 256 ms | 447 ms | 340 ms |
| **Output tok/s** | 3.91 | 2.48 | 3.87 |
| **Active CPU cores** | ~12 | ~17 | ~13 |
| **RAM (GB)** | 51 | 92 | 81 |
| **Power (W)** | 203 | 202 | 211 |

**Reading:** At low concurrency, SN wins across the board. TP-2 is the worst (ITL +58%, throughput -37%): on a model with low compute-per-layer, the QPI synchronization overhead becomes comparable to or exceeds the computation time itself. DP-2 shows ITL nearly identical to SN (258 vs 255 ms): each rank operates in isolation on its own local RAM. The anomalous TTFT in DP-2 (>10s) appears to be a warmup/scheduling artifact in the initial request phase, not a structural limitation of data parallelism.

### 5.2 Medium Workload (concurrent=4)

| Metric | SN | TP-2 | DP-2 |
|---|---|---|---|
| **Completed** | 12/28 (43%) | 5/20 (25%) | 9/24 (38%) |
| **Errors** | 4 | 0 | 0 |
| **E2E latency p50** | 97.8s | 165.7s | 113.6s |
| **TTFT p50** | 23.0s | 46.7s | 21.8s |
| **ITL p50** | 292 ms | 467 ms | 360 ms |
| **Output tok/s** | 8.11 | 1.67 | 6.38 |
| **Req/s** | 0.016 | 0.013 | 0.022 |
| **Active CPU cores** | ~12 | ~17-22 | ~25 |
| **Power (W)** | 224 | 223 | 285 |

**Reading:** Under load, DP-2 emerges with the best req/s (0.022), best TTFT (21.8s), and zero errors. Both sockets work at maximum (~25 active cores). TP-2 completes only 25% of requests. SN begins showing errors (4 timeouts) → signal that the single socket is reaching saturation.

---

## 6. Scaling Sweep and Crossover Point

Sweep with increasing rates (1, 2, 4, 6, 8 req/s), 300s duration per step, identical parameters for SN and DP-2.

### 6.1 Sweep Data: Single-Node

| Rate | OK/Tot | E2E p50 | TTFT p50 | ITL p50 | tok/s | req/s | Errors |
|---|---|---|---|---|---|---|---|
| 1 | 7/12 | 43.2s | 10,711 ms | 256 ms | 3.90 | 0.023 | 0 |
| 2 | 13/21 | 44.5s | 673 ms | 264 ms | 6.23 | 0.045 | 0 |
| 4 | 21/36 | 37.3s | 798 ms | 287 ms | 7.07 | 0.034 | 0 |
| 6 | 25/52 | 47.4s | 1,038 ms | 365 ms | 5.80 | 0.042 | 5 |
| 8 | 33/62 | 60.2s | 1,305 ms | 464 ms | 4.95 | 0.040 | 0 |

### 6.2 Sweep Data: Data Parallel (DP-2)

| Rate | OK/Tot | E2E p50 | TTFT p50 | ITL p50 | tok/s | req/s | Errors |
|---|---|---|---|---|---|---|---|
| 1 | 7/12 | 43.3s | 10,692 ms | 257 ms | 3.90 | 0.023 | 0 |
| 2 | 14/22 | 44.7s | 10,751 ms | 267 ms | 7.25 | 0.045 | 0 |
| 4 | 19/34 | 58.3s | 11,082 ms | 414 ms | 6.42 | 0.045 | 0 |
| 6 | 26/48 | 68.7s | 1,014 ms | 420 ms | 8.87 | 0.064 | 0 |
| 8 | 31/60 | 44.1s | 809 ms | 295 ms | 8.33 | 0.058 | 0 |

### 6.3 Direct Comparison and Crossover

| Rate | SN tok/s | DP-2 tok/s | Delta | Winner |
|---|---|---|---|---|
| 1 | 3.90 | 3.90 | 0% | Tie |
| 2 | 6.23 | 7.25 | +16% | DP-2 |
| 4 | 7.07 | 6.42 | -9% | SN |
| **6** | **5.80** | **8.87** | **+53%** | **DP-2 ✅** |
| **8** | **4.95** | **8.33** | **+68%** | **DP-2 ✅** |

### 6.4 Reading the Scaling Curve

**The entire system is memory-bandwidth bound.** This is the root cause of every observed behavior.

**Regime 1 – Scaling (rate 1→2):** DDR4 bandwidth is not saturated, ITL is stable (~256–267 ms). Both SN and DP-2 scale almost linearly. Architecture is irrelevant: the bottleneck has not yet been reached.

**Regime 2 – Saturation (rate 4):** SN reaches peak throughput (7.07 tok/s). Single-socket DDR4 bandwidth begins to saturate; ITL starts growing (+12%). DP-2 underperforms here (6.42 tok/s, ITL 414 ms) because the coordinator overhead outweighs the distribution benefit with only 4 requests.

**Regime 3 – Degradation (rate 6→8):** SN collapses (ITL +81%, tok/s -30%, errors appear). Queueing becomes the dominant phenomenon. DP-2 scales instead: at rate=8 it produces 8.33 tok/s (+68% vs SN), ITL 295 ms (-36%), zero errors. Each socket has its own dedicated DDR4 bandwidth → aggregate bandwidth doubles, queueing halves.

### 6.5 The Crossover Point as an Engineering Parameter

The crossover at ~6 concurrent requests is an **operational parameter**, not just a metric.

- If load is consistently below 4–5 requests: **SN** is the correct choice.
- If load can exceed 6 requests (even intermittently): **DP must be configured proactively**. The cost of collapse (errors, timeouts, 3× latency) exceeds the cost of DP overhead at low load.

> **Operational rule:** the crossover point (~6 concurrent on this hardware) must be **measured** for each configuration. It depends on DDR4 bandwidth, model size, and context length.

---

## 7. Energy Efficiency: Infra Economics

Power consumption measured in real time via IPMI (`ipmi_dcmi_power_consumption_watts`), integrated into the Prometheus/Grafana stack. Idle consumption: ~147W.

Key metrics:
- **Tok/W total** = tokens produced per total watt consumed
- **Tok/ΔW** = tokens per incremental watt above idle (isolates energy dedicated to inference)

| Benchmark | Avg W | ΔW | tok/s | Tok/W total | Tok/ΔW | Note |
|---|---|---|---|---|---|---|
| **SN light** | 203 | 56 | 3.91 | 0.019 | 0.070 | |
| **SN medium** | 224 | 77 | 8.11 | 0.036 | **0.105** | **Best** |
| **TP-2 light** | 202 | 55 | 2.48 | 0.012 | 0.045 | |
| **TP-2 medium** | 223 | 76 | 1.67 | 0.008 | **0.022** | **Worst** |
| **DP-2 light** | 211 | 64 | 3.87 | 0.018 | 0.060 | |
| **DP-2 medium** | 285 | 138 | 6.38 | 0.022 | 0.046 | |

**Reading:** SN medium is the most energy-efficient (0.105 tok/ΔW, ~5× better than TP-2 medium). DP-2 medium produces more absolute throughput, but at a per-token cost **2.3× higher** than SN.

> **Implication:** SN is the most cost-efficient choice unless concurrent load requires DP-2. The additional energy cost of DP-2 must be justified by actual throughput needs, not enabled by default.

---

## 8. Design Patterns: Operational Rules

### 8.1 Anti-Pattern: Tensor Parallel over a Slow Interconnect

TP cross-socket produces **worse** performance than a single socket. This is not a vLLM issue — the Gloo-based TP implementation works correctly. The problem is the **combination** of two factors:

1. **Small model (~4B params):** compute-per-layer is insufficient to hide communication latency. With ~32 layers, every generated token pays 32 all-reduce operations.
2. **Slow interconnect (QPI ~38 GB/s, NUMA distance 2.1×):** every cross-socket all-reduce introduces measurable latency. On NVLink (~900 GB/s, ~24× QPI bandwidth), the same operation is nearly free.

This is consistent with the general GPU principle: **TP is effective only over high-bandwidth intra-node interconnects** (NVLink). Over slower interconnects (PCIe), TP degrades performance. The principle is universal; this study quantifies it on CPU. **The limit emerges when synchronization cost exceeds compute per layer.**

> **Rule:** do not use `--tensor-parallel-size > 1` on dual-socket CPU systems with QPI/UPI for low-parameter-count models. TP is only justified when the model does not fit in a single socket's RAM and no alternative exists.

### 8.2 Recommended Pattern: Data Parallel on Multi-Socket

DP-2 is the recommended pattern for multi-socket architectures when concurrent load exceeds the crossover point. Each socket operates as an independent engine with its own model copy and KV cache. Zero inter-socket communication during inference.

**Benefits:** throughput up to +68% under load, ITL -36% vs SN at rate=8, zero errors where SN produces them, both sockets running at full capacity.

**Costs:** double RAM for weights (~16 GB vs ~8 GB), coordinator overhead at low load (TTFT 10.7s vs 278 ms of SN at rate=1), higher power consumption (285W vs 224W, 2.3× cost per token), configuration complexity (NUMA binding, KV cache sizing per rank).

> **Rule:** on dual-socket CPU with load >6 concurrent, use `--data-parallel-size 2` with explicit NUMA binding. Below this threshold, use SN. The crossover point must be measured for each hardware configuration.

### 8.3 Default Pattern: Single-Node for Low Concurrency

For single-user or low-concurrency workloads (<6 requests): minimum latency, maximum energy efficiency (0.105 tok/ΔW), zero coordination overhead. The second socket can remain in sleep state or be dedicated to other services.

> **Rule:** for workloads with concurrency < 6, use a single socket with `numactl --cpunodebind --membind`. Do not activate DP or TP unless necessary. Architectural simplicity is not just a convenience — it is a measurable performance advantage.

---

## 9. Practical Implications

### 9.1 When CPU Inference Makes Sense

**Edge and air-gapped environments:** where GPUs are unavailable or uncertified. A Xeon server can serve 3–8B models with acceptable latency for non-real-time applications.

**Regulated environments:** finance, healthcare, defense — where hardware is compliance-constrained and GPUs are not in the approved inventory.

**Development and testing:** validating inference pipelines, testing prompts, debugging the application stack without occupying expensive GPUs.

**Backup and resilience:** automatic failover from GPU to CPU in case of hardware failure, ensuring service continuity at reduced throughput.

### 9.2 How to Choose Hardware for CPU Inference

**Memory bandwidth > core count.** LLM inference on CPU is memory-bandwidth bound. **Populating all DIMM channels is more important than having more cores.** Our test with 2/4 channels demonstrates the impact: throughput could double with full channel population.

**IPC > clock speed.** More recent CPU generations (Sapphire Rapids, Genoa) offer better ISAs (AVX-512, AMX, native BF16) that directly accelerate inference operations. A Xeon 4th Gen at 2.0 GHz can outperform a Broadwell at 3.3 GHz due to native hardware support.

**Dual-socket only if DP is needed.** A second socket adds cost, power consumption, and complexity. It only makes sense if concurrent load justifies Data Parallel. For low-concurrency workloads, a single socket with full RAM is the better choice.

---

## 10. Demonstration: The CPU-GPU Isomorphism

The most significant result of this work is not in the absolute performance figures — obviously inadequate compared to a GPU — but in the demonstration that distributed inference patterns are **general principles, not GPU hardware artifacts.**

> *A NUMA node is a compute unit with fast-access local memory and a communication bus toward other units — exactly like a GPU.*

| Concept | GPU World | CPU World (this study) |
|---|---|---|
| **Compute unit** | GPU | NUMA Node (Socket) |
| **Local memory** | HBM (~2–3 TB/s) | DDR4 (~34 GB/s, 2/4 ch) |
| **Interconnect** | NVLink (~900 GB/s) | QPI (~38 GB/s) |
| **Local/inter BW ratio** | ~3× (HBM/NVLink) | ~0.9× (DDR4/QPI) |
| **TP effective?** | Yes (NVLink is fast) | No (QPI ≈ DDR4, overhead dominant) |
| **DP effective?** | Multi-GPU, multi-node | Multi-socket (>6 concurrent) |
| **Primary bottleneck** | HBM bandwidth | DDR4 bandwidth |
| **Scheduling** | True parallelism + batching | Time-sharing + queueing |

**The local/interconnect bandwidth ratio is the key metric.** On GPU, HBM is ~3× faster than NVLink: there is headroom to hide TP communication cost inside compute. On CPU, DDR4 and QPI have nearly equal bandwidth (~34 vs ~38 GB/s): there is no headroom, and every TP communication is a net cost.

The quantitative gap is enormous (DDR4 ~34–68 GB/s vs HBM ~2–3 TB/s, 50–90×; QPI ~38 GB/s vs NVLink ~900 GB/s, ~24×). But the **decision-making principles** — when to use TP, when to use DP, where the bottleneck lies — follow the same dynamics.

---

## 11. Study Limitations

**Broadwell hardware (2016).** No AVX-512, AMX, or native BF16. On more recent CPUs the absolute results would be significantly better and the DP/SN crossover point might shift. Architectural principles remain valid.

**Software-emulated BF16.** Penalizes throughput uniformly across all tested architectures. On CPUs with AVX512_BF16 the TP vs SN delta might shrink (but not reverse: the bottleneck is bandwidth, not compute).

**RAM not fully populated.** With ~2/4 channels per socket, bandwidth is ~50% of maximum. At full population, SN throughput would improve significantly and the DP/SN crossover point would shift toward higher concurrency levels.

**Single model tested (~4B params).** TP results are specific to small models with low compute-per-layer. For models >30B, the compute/communication ratio improves and TP could become advantageous even on QPI.

**No formal statistical analysis.** Single-run benchmarks per configuration; inter-run variance was not quantified. The observed deltas (37–79% for TP, 53–68% for DP) are large enough to be significant, but validation with repeated runs and confidence intervals would strengthen the conclusions.

---

## 12. Lessons Learned

### The Principle: The Bottleneck Is Memory, Not the CPU

In LLM inference, cores are not the bottleneck. DDR4 bandwidth is. Every generated token requires reading the entire weight set from RAM. More cores do not accelerate this read — more DIMM channels do.

A Xeon with 14 cores and 4 fully populated DDR4 channels outperforms a Xeon with 28 cores and 2 channels. Everything we measured — ITL, degradation, crossover — derives from this constraint.

### The Five Rules

**Rule 1:** Do not use cross-socket Tensor Parallel for models below ~10B parameters. QPI/UPI overhead cannot be amortized. TP is reserved for cases where the model does not fit in a single socket.

**Rule 2:** Use Data Parallel when concurrency exceeds the single-socket crossover point. On our hardware: ~6 concurrent requests. The exact value depends on DDR4 bandwidth, model size, and context length. Must be measured per configuration.

**Rule 3:** Keep concurrency below the single-socket saturation point when using SN. Beyond the peak (~rate 4 in our tests), throughput degrades and errors appear. The safe operating range must be identified with a sweep before production deployment.

**Rule 4:** Populate all DIMM channels before adding cores or sockets. DDR4 bandwidth is the first investment to make. A second socket with partial RAM does not beat a single socket with full RAM.

**Rule 5:** Measure, do not assume. The crossover point, safe operating range, and energy efficiency all depend on the specific configuration. Every deployment must be validated with structured benchmarks before going to production.

---

*dielabs.github.io • github.com/Dielabs*
