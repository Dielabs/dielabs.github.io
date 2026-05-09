---
layout: default
title: "CPU-GPU Memory Topology for AI Inference"
---

# CPU-GPU Memory Topology for AI Inference

**Scope:** Taxonomy of CPU-GPU memory architectures relevant to LLM inference, with focus on modern unified/coherent patterns and their implications for data placement, KV cache tiering, and platform selection.

**Status:** Reference document. Vendor-verified facts where indicated; forward-looking sections are interpretive and marked as such.

---

## 1. The Problem

LLM inference is dominated by **memory-bound workloads**, not compute-bound ones:

- KV cache grows linearly with context length × batch size, eventually saturating HBM
- Models exceeding HBM capacity (70B+ at FP16, MoE such as DeepSeek-V3 671B) require tiering
- Decode is bandwidth-bound: every generated token reads attention from the entire accumulated KV cache

Consequence: **CPU-GPU memory topology** — and the nature of the interconnect between the two — becomes a primary architectural factor, not an implementation detail.

---

## 2. Conceptual Framing: Two Orthogonal Axes

The terms "shared", "unified", and "coherent" are often used interchangeably in marketing material. They are **distinct and orthogonal properties**:

| Axis | Question | Property type |
|---|---|---|
| **Unified** | Is there a single virtual address space? | Programming model |
| **Coherent** | Are views synchronized in hardware? | Hardware implementation |

Four theoretical combinations:

1. **Neither unified nor coherent** → Discrete + classic PCIe (x86 + NVIDIA/AMD GPU via PCIe)
2. **Unified but not coherent** → CUDA UVM over PCIe (software-simulated unification with on-demand page migration)
3. **Coherent but not unified** → Classic multi-CPU NUMA (rare in CPU-GPU context)
4. **Both unified AND coherent in hardware** → Apple Silicon, GH200/GB200, GB10 (Spark), MI300A

Combination (4) is the dominant pattern for modern AI compute.

### 2.1 Why the Distinction Matters

"Unified" describes **what software sees**. "Coherent" describes **how it's implemented underneath**.

You can have software-simulated unification (UVM) on top of non-coherent hardware — it works, but with hidden copy costs. True hardware coherence eliminates both the copy and the page-fault handling overhead.

---

## 3. Operational 4-Quadrant Taxonomy

| Pattern | Address space | HW coherence | Physical memory | CPU↔GPU interconnect | Examples | Typical use case |
|---|---|---|---|---|---|---|
| **Discrete + PCIe** | Separate (UVM abstracts above) | No | Heterogeneous (HBM + DDR) | PCIe Gen4/5 (~64 GB/s unidirectional) | x86 + H100/B200 PCIe, A100 | Mainstream training & inference, DGX H100/B200 nodes |
| **Homogeneous unified** | Unified | Yes | Homogeneous (LPDDR only) | Internal SoC fabric (~400-800 GB/s) | Apple M-series, DGX Spark (GB10) | Dev workstation, edge inference, prototyping |
| **Heterogeneous unified** | Unified | Yes | Heterogeneous (HBM + LPDDR) | NVLink-C2C (~900 GB/s) | GH200, GB200, MI300A | Inference of large models (>HBM), MoE, memory-bound training |
| **Legacy shared** | Unified but statically partitioned | Partial | Homogeneous (subtracted from CPU) | Internal SoC bus | Older Intel/AMD iGPU | Consumer/office, not relevant for AI |

---

## 4. Deep-Dive on the Three AI-Relevant Patterns

### 4.1 Discrete + PCIe (Row 1)

**Characteristics:**

- CPU and GPU as separate domains with distinct address spaces
- Data transfers via DMA over PCIe, I/O-class semantics
- Software sees UVM (Unified Virtual Memory) as an abstraction, but page migration and copies happen underneath

**Limits for inference:**

- PCIe Gen5 x16: ~64 GB/s unidirectional → an order of magnitude below HBM (3-8 TB/s)
- I/O-class latency: TLP overhead, completion, ordering
- CPU-side KV cache offload is practical only at coarse granularity (entire prefixes, dormant sessions)

**When it fits:**

- Most of the current market (DGX H100/B200, cloud GPU instances)
- Workloads that fit comfortably in HBM
- Horizontal scaling via NVLink/InfiniBand across nodes (the PCIe bottleneck is intra-node)

### 4.2 Homogeneous Unified (Row 2) — Apple Silicon, DGX Spark

**Characteristics:**

- Single LPDDR pool on the SoC package
- CPU, GPU (and Apple's Neural Engine) access the same physical addresses
- **Dynamic** allocation: no static pre-partitioning, the allocator distributes on demand
- The GPU can claim almost the entire pool if the workload requires it

**Concrete examples:**

- **Apple M3 Ultra**: up to 512 GB unified, ~800 GB/s bandwidth
- **DGX Spark (GB10)**: 128 GB LPDDR5X unified, Blackwell GPU + 20-core ARM (Cortex-X925/A725), ~$3,000 at launch. _Specs to be verified at actual launch — some details may have changed._

**Advantages:**

- Simple mental model: "one pool, take what you need"
- No data placement decisions: everything is at the same bandwidth
- GPU-addressable capacity = full system memory limit (no separate HBM cap)

**Limits:**

- Uniform bandwidth = no fast tier for hot weights
- LPDDR ~500-800 GB/s vs HBM ~5 TB/s → compute-bound workloads are penalized
- Spark/Mac do not replace H100/B200 in HBM — they are **complementary** platforms

**macOS caveat:** the system reserves a minimum quota; the GPU ceiling can be raised via `sysctl iogpu.wired_limit_mb` but a floor remains.

### 4.3 Heterogeneous Unified (Row 3) — GH200, GB200, MI300A

**Characteristics:**

- **Two physically distinct memory tiers**, logically unified
- HBM on the GPU package + LPDDR on the CPU package, joined by a coherent interconnect
- Same virtual address space (NVIDIA: ATS — Address Translation Services)
- Hardware coherence via NVLink-C2C (NVIDIA) or Infinity Fabric (AMD)

**Concrete examples:**

**NVIDIA GH200:**

| Memory | Technology | Capacity | Bandwidth |
|---|---|---|---|
| HBM3e | stacked DRAM on Hopper GPU | 96-144 GB | ~4.9 TB/s |
| LPDDR5X | DIMM-like on Grace CPU | 480 GB | ~500 GB/s |
| C2C interconnect | NVLink-C2C | — | 900 GB/s |

**NVIDIA GB200:** same pattern, Blackwell GPU with larger HBM3e, identical Grace LPDDR.

**AMD MI300A:** single-package APU with Zen4 CPU + CDNA3 GPU + unified coherent HBM3. Used in El Capitan (LLNL). Pattern similar to GH200, but single-die instead of CPU+GPU split across C2C.

**Important:** the enabling factor is **Grace + NVLink-C2C**, not Hopper/Blackwell per se. Hopper/Blackwell as discrete GPUs (in x86 PCIe nodes) **do not** have unified coherent memory with the host CPU.

**Advantages:**

- HBM stays where extreme bandwidth matters (active weights, attention compute)
- LPDDR extends capacity (warm KV cache, dormant MoE experts, models >HBM) without copy penalty
- Software sees a single pool → the allocator (or the informed developer) decides tiering
- Enables single-node serving of large models (DeepSeek-V3 671B MoE) without inter-node disaggregation

**Trade-offs:**

- HBM capacity is design-capped (not expandable)
- Performance depends on **data placement**: wrong data in LPDDR = decode bandwidth-bound at 500 GB/s instead of 5 TB/s
- Software stack complexity: today's vLLM CPU-offload is designed for the PCIe model and doesn't fully exploit coherence

---

## 5. The Key Insight: "Not an Access Problem, an Access-Cost Problem"

The GPU has **always been able to reach CPU RAM**, even over PCIe (DMA, UVM, mapped memory). The PCIe → NVLink-C2C → unified SoC fabric evolution did not change _what_ the GPU can reach. It changed:

- **How fast** (bandwidth)
- **At what latency** (memory-class vs I/O-class)
- **With what semantics** (load/store vs DMA, copy vs direct access)
- **At what software overhead** (driver/copy engine vs direct access)

**Operational framing:**

> The "memory wall" of AI inference is not a question of access feasibility. It's a question of access cost.

A natural consequence: **classical storage tiering** (RAM → SSD → HDD → tape) and **modern memory tiering** (HBM → coherent LPDDR → remote CPU RAM → CXL → storage) are the same pattern applied at different scales. Everything is reachable; cost scales with distance.

### 5.1 The CPU is Not a Broker — It's a Peer

A frequently misunderstood point: when the GPU accesses memory physically located "on the CPU side", it **does not ask the CPU for permission**. The CPU as a processor is out of the data path.

**What actually happens:**

- **On PCIe:** the GPU's copy engine (DMA) issues PCIe transactions directly to the CPU's memory controller. Transactions traverse the PCIe root complex (passive hardware for this flow) and reach the memory controller. The CPU as a processor is not interrupted. This is identical to an NVMe doing DMA into RAM — the CPU does not "know" it's happening.
- **On NVLink-C2C:** the GPU issues a **load instruction** (not a DMA). The address goes to the GPU's MMU; if the physical address lands in Grace LPDDR, the request travels over C2C and the Grace memory controller responds. It's the same instruction as a load on HBM — the GPU neither knows nor cares that the data is physically in LPDDR. Only latency changes.
- **On Apple/Spark:** memory is physically singular; GPU and CPU are peers on the same internal SoC bus.

**What the CPU actually does in the access lifecycle:**

- **Initial setup:** the CUDA driver prepares page tables, allocates buffers, configures the copy engine
- **Page fault handling:** if the page is not in RAM or not pinned, the CPU kernel intervenes (this is why pinned memory is used in AI data paths)
- **Coherence protocol:** on coherent systems, hardware protocols manage cache coherence — hardware, not CPU software

Everything else is **GPU-driven data plane**.

**Practical implication:** in inference, the CPU as a compute device matters only for pre/post-processing, tokenization, and the scheduler/serving framework (vLLM is Python on CPU). For **data movement**, the CPU is passive infrastructure. Adding more or faster CPU cores does not help inference throughput on an already-configured data path. What helps is interconnect bandwidth (PCIe Gen5 vs Gen4) or replacing the interconnect entirely (C2C, unified fabric).

**Correct mental model:** CPU and GPU are **independent peers that share only memory**, not master/slave. Anyone thinking "the GPU asks the CPU" reveals a CPU-centric mindset still rooted in the pre-AI-compute era.

### 5.2 Physical Path as Architectural Metaphor

The difference between the three patterns is not just bandwidth or absolute latency in numbers. It's the **physical and logical length of the path** the GPU must travel to reach non-local memory.

**x86 + PCIe GPU (Discrete):**

```
GPU memory controller
  ↓
GPU PCIe root port
  ↓
PCIe link (PCB trace, ~cm distance)
  ↓
CPU PCIe root complex
  ↓
Mesh / Ring / Infinity Fabric inside the CPU
  ↓
CPU memory controller (IMC)
  ↓
DDR DIMM
```

Every hop with its own protocols (PCIe TLP, internal mesh, DDR command/address). The GPU is dialoguing with a memory subsystem **designed to serve the CPU as primary client**.

**NVLink-C2C (GH200):**

```
GPU memory controller
  ↓
NVLink-C2C interface (on package, ~mm distance)
  ↓
Grace CPU memory controller
  ↓
LPDDR5X
```

Physically shorter hops, no root complex, no CPU mesh to traverse. C2C is designed as a **memory fabric**, not an I/O bus — different semantics and latency.

**Apple Silicon / Spark (Homogeneous unified):**

```
GPU
  ↓
Internal SoC fabric (intra-die or intra-package)
  ↓
Shared memory controller
  ↓
LPDDR
```

Path effectively collapsed. Everything on the same silicon or package.

**Typical latency for GPU access to CPU-side memory** (approximate orders of magnitude — verify against vendor documentation for formal deliverables):

| Pattern | Typical latency | Notes |
|---|---|---|
| x86 + PCIe Gen5 | ~1-2 μs | Round-trip via PCIe + CPU mesh + IMC |
| NVLink-C2C | ~hundreds of ns | Memory-class, no I/O protocol |
| Unified SoC fabric | ~hundreds of ns or less | Quasi-local access |
| Local HBM (reference) | ~tens to ~100 ns | For comparison |

**Conceptual transition reflected by these physical paths:**

- **x86 + PCIe** → "GPU as accelerator attached to a CPU system" (Lego pattern: flexible, modular, not optimized for any specific workload)
- **GH200 / GB200** → "CPU+GPU as co-designed system" (Grace dimensioned with GPU traffic in mind, C2C is not a bottleneck, address space genuinely peer-to-peer)
- **Apple / Spark** → "Unified system where there is no longer a CPU memory and a GPU memory" (single memory designed to serve both simultaneously)

**Visual synthesis:**

> On x86+PCIe, the GPU is a guest of the CPU's memory subsystem. On Grace-Hopper, they are roommates. On Apple Silicon, they live in the same room.

Three different paradigms of hardware co-design, directly reflected in the physical geometry of access paths.

**Why this makes strategic sense for vendors:** when the GPU is the heart of the system (AI inference/training), the Lego pattern shows its weaknesses. NVIDIA with Grace, Apple with M-series, and AMD with MI300A are all making the same bet: if the primary workload is AI, the system must be designed _around_ the workload, not assembled from general-purpose pieces.

---

## 6. Implications for KV Cache Offload

KV cache offload works on all three relevant rows, but with profoundly different mechanisms and costs.

### 6.1 Offload on Discrete + PCIe (Row 1)

- Tools: vLLM `--cpu-offload-gb`, LMCache CPU tier
- Mechanism: **explicit copy** HBM ↔ system RAM via PCIe DMA
- Penalty: ~64 GB/s, I/O-class latency
- Practical granularity: coarse (entire prefixes, dormant sessions, cross-request prefix caching)
- Decode with offloaded KV requires back-and-forth transfer → significant TPOT impact

### 6.2 Offload on Homogeneous Unified (Row 2)

- "Offload" is almost a misnomer: KV cache lives in the unique pool from the start
- There's no fast tier and slow tier — it's all LPDDR at the same bandwidth
- You don't offload: **you just allocate**
- Limit: uniform bandwidth ceiling across the entire pool

### 6.3 Offload on Heterogeneous Unified (Row 3)

- Mechanism: **coherent load/store** over NVLink-C2C, no copies
- ~900 GB/s vs ~64 GB/s of PCIe → ~14× more bandwidth
- Memory-class latency, not I/O-class
- Practical granularity: fine, layer-by-layer or beyond
- **The GPU can perform attention computation with K and V physically resident in Grace LPDDR**, simply slower

**Typical strategy on GH200:**

- Hot KV cache (active requests, last N tokens): HBM
- Warm KV cache (shared prefixes, paused sessions): Grace LPDDR
- Pattern similar to LMCache, but intra-node and copy-free

### 6.4 Mental Model Shift

| Row 1 (PCIe) | Row 3 (Coherent C2C) |
|---|---|
| "Move data at the right time" | "Leave it where it is, let it be read at lower speed" |
| Optimize **when** to transfer | Optimize **where** to allocate initially |
| Data movement | Data placement |

On Row 3 the operative phrase becomes: _"How do I move data?" → "Where should data reside?"_

---

## 7. Software Stack Status (interpretive)

**Fact:**

- vLLM has CPU offload (`--cpu-offload-gb`) designed for the PCIe model with explicit copies
- LMCache implements CPU-side KV cache tiering, originally for PCIe nodes
- NIXL (NVIDIA InferenceXfer Library) abstracts the transport layer for KV cache transfers
- NVIDIA Dynamo and llm-d explicitly aim to expose the coherent memory hierarchy as a native tier

**Roadmap/trend (interpretive):**

- On GH200, current stacks _work_ but are sub-optimal — they don't fully leverage HW coherence (they continue to perform logically explicit copies)
- Expected direction: software aware of the coherent tier, native KV cache allocation on Grace LPDDR without pseudo-offload
- Software maturity gap is still significant (as of April 2026, to be revisited)

---

## 8. Spark vs GH200: Direct Comparison (Row 2 vs Row 3)

|  | DGX Spark / Mac (Row 2) | GH200 (Row 3) |
|---|---|---|
| Mental model | "One pool, take what you need" | "Two tiers, decide where to place what" |
| GPU capacity ceiling | Almost the entire system RAM | HBM + LPDDR (sum) |
| Bandwidth ceiling | LPDDR bandwidth for everything (~500-800 GB/s) | HBM (~5 TB/s) for hot tier |
| Data placement complexity | None | Significant |
| Sweet spot | Models that fit but don't require HBM bandwidth | Large models with hot working set in HBM |
| Cost (order of magnitude) | $3K-$10K | $40K+ per single GH200 |

---

## 9. References for Further Reading

**Vendor documentation (fact):**

- NVIDIA Grace Hopper Superchip Architecture Whitepaper
- NVIDIA Blackwell Architecture Whitepaper
- AMD MI300A documentation (CDNA3 + Zen4 APU)
- Apple Silicon Unified Memory Architecture (developer docs)

**Operational/practitioner:**

- vLLM documentation: CPU offload, prefix caching
- LMCache documentation
- NIXL / Dynamo / llm-d project READMEs

**Roadmap/trend (to verify):**

- DGX Spark specifications at actual launch
- Software stack maturity on GH200/GB200 for native KV cache on LPDDR

---

_All bandwidth, capacity, and latency figures are order-of-magnitude approximations. For formal deliverables, always verify against current vendor documentation or workload-specific benchmarks._
