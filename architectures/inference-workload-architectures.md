---
layout: default
title: "Inference Workload Architectures"
---

# Inference Workload Architectures

**Version:** 1.2  
**Scope:** Primary focus on L0–L1 in the Dielabs Inference Stack Model, with controlled extensions to upper layers (L2/L3) when needed to explain emergent behavior (e.g. NIXL/runtime interaction, KV cache scheduling, serving-engine constraints).

---

## Abstract

LLM inference at scale requires distributing computation across multiple processors. This document covers the full path — from the physical interconnect between chips, to the libraries that govern it, to network behavior under load, to the parallelism strategies that depend on it — providing a unified decision framework.

The guiding principle is that **parallelism choices are bounded by the physical layer**: no software strategy can exceed the bandwidth and latency limits imposed by the underlying fabric. Understanding the hardware hierarchy and knowing _when_ the network enters the token-generation loop is a prerequisite for any architectural decision at the upper layers of the stack.

---

## Part I — The GPU Compute Fabric

### 1. The underlying problem: scaling computation, not just GPUs

In early accelerated systems a GPU worked almost in isolation, the CPU orchestrated, communication was sporadic. With LLMs, distributed training and persistent inference, GPUs have to synchronize continuously, GPU↔GPU traffic dominates total time, and latency becomes more critical than nominal bandwidth.

This is where the need for a **compute fabric**, not a simple network, comes from.

### 2. NVLink: GPU↔GPU interconnect at the node level

NVLink is a **point-to-point, very-high-bandwidth, low-latency** interconnect between chips, born to connect GPUs to each other inside a single node.

#### Generational evolution

| Generation | GPU architecture | Bidirectional bandwidth per GPU | Notes |
|---|---|---|---|
| NVLink 3.0 | Ampere (A100) | 600 GB/s | 12 links × 50 GB/s |
| NVLink 4.0 | Hopper (H100/H200) | 900 GB/s | 18 links × 50 GB/s |
| NVLink 5.0 | Blackwell (B200/GB200) | 1,800 GB/s | 18 links × 100 GB/s |

These numbers define the **physical ceiling** for local parallelism: no Tensor Parallelism or Pipeline Parallelism strategy can exceed the NVLink bandwidth available between the involved GPUs.

In 4/8 GPU systems, every GPU is connected to every other through NVLink, often with the help of **NVSwitch**, creating a near-fully-connected GPU↔GPU domain.

**What NVLink is not:** it is not a network, it does not involve the CPU, it does not introduce CPU↔GPU coherence. It solves the local-parallelism problem, not the system-scale problem.

### 3. NVLink-C2C: coherent CPU↔GPU

With the Grace platforms, NVIDIA redefined NVLink-C2C as a **coherent** very-high-bandwidth interconnect between Grace CPU and GPU.

| Parameter | NVLink-C2C (Grace Hopper GH200) | PCIe Gen5 x16 |
|---|---|---|
| Bidirectional bandwidth | 900 GB/s | ~128 GB/s |
| Delta | **~7×** vs PCIe | baseline |
| Memory coherence | Hardware, native | Not supported |
| Semantics | Compute datapath | I/O bus |

NVLink-C2C does **not** connect GPUs to each other, and does **not** scale beyond a single node. It exists to remove the CPU↔GPU bottleneck. It's a local paradigm shift, not a scale-out fabric.

#### Note on terminology confusion

Historically NVIDIA used "chip-to-chip (C2C)" generically for direct chip-to-chip links (GPU↔GPU NVLink direct, no PCIe, no network switch). Today "NVLink-C2C" has a much more specific meaning — the coherent Grace CPU ↔ GPU link.

### 4. NVFabric: extending NVLink beyond the node

NVFabric is the extension of the NVLink concept beyond a single server, at rack or pod level, through **dedicated NVLink Switches**.

The term was formalized by NVIDIA starting from the **Blackwell** architecture, in particular with the **GB200 NVL72** platform (72 GPUs interconnected via NVLink in a single rack-level domain). The concept already existed in earlier generations (DGX SuperPOD with NVSwitch on Hopper), but without this unified branding.

#### NVLink Switch

NVLink Switches route NVLink traffic with deterministic latency, support collective compute primitives, and are not traditional network switches. Without NVLink Switches, NVFabric does not exist.

#### NVLink cables

NVLink cables are dedicated and proprietary, carry point-to-point signals, do not encapsulate packets, and do not implement a network stack. Conceptually they are **physical extensions of the interconnect bus**, not network cables. This makes them extremely fast but extremely constraining in terms of distance and layout.

### 5. NVFabric vs traditional networks

| Aspect | NVFabric | Networks (InfiniBand / Ethernet) |
|---|---|---|
| Semantics | Compute and synchronization | Message-based |
| Switching | Minimal, deterministic latency | Packet switching |
| Orchestration | Hardware-accelerated collectives | Explicit software orchestration |
| Scale | Restricted physical domain (rack/pod) | Geographic scale-out |

NVFabric does **not replace** networks: it **complements** them for compute-critical traffic. It is not a general-purpose network, but a specialized fabric for compute-synchronous traffic. With Blackwell it is effectively becoming a system fabric — but it remains bound to restricted physical domains and to workloads with known communication patterns.

### 6. Complete interconnect hierarchy

| Level | Domain | Technology | Function |
|---|---|---|---|
| 1 | Inside the GPU | Internal interconnect | SM/HBM communication |
| 2 | Between GPUs in the same node | NVLink / NVSwitch | Local parallelism (TP, PP) |
| 3 | Between nodes in the same rack/pod | NVFabric (NVLink Switch + NVLink cables) | Synchronous multi-node scaling |
| 4 | Between racks and across the datacenter | InfiniBand / Ethernet | Scale-out, storage, control |

Each level solves a different problem and is not interchangeable. Trying to use a network as a compute fabric leads to structural inefficiencies; trying to use NVFabric as a network leads to unmanageable constraints.

### 7. Design implications for the datacenter

The presence of a compute fabric implies that the rack becomes a unit of design, the cabling is rigid and pre-defined, the facility (power, cooling, space) is part of the system, and traditional IT converges toward HPC models. The compute fabric **reduces flexibility** but **increases efficiency** on extremely parallel workloads.

---

## Part II — NVIDIA Communication Libraries

The three NVIDIA libraries that govern GPU-to-GPU communication in the inference ecosystem operate at distinct purposes. The choice between them is not a preference — it is determined by the deployment architecture.

### 8. NCCL — Communication inside a distributed model

**NVIDIA Collective Communications Library.** Historical library, born for distributed training and adopted for inference with tensor parallelism. Distributes the model itself across multiple GPUs that cooperate to execute a single forward pass. The GPUs are not independent — they work together as a single compute unit.

It implements collective primitives (all-gather, all-reduce, broadcast, reduce-scatter, send/recv) optimized for PCIe, NVLink and InfiniBand. NCCL uses dedicated copy engines with SM assistance: contention for compute resources is therefore workload-dependent, not systematic. On compute-heavy workloads contention is minimal; on memory-bound workloads with frequent communication it can become significant.

**When to use it:** every time the model is distributed across multiple GPUs with TP, PP, or EP. Communication is synchronous and happens at every layer/step of the forward pass.

### 9. NIXL — Communication between independent instances

**NVIDIA Inference Transfer Library.** New library, part of the NVIDIA Dynamo ecosystem. Its purpose is to transfer the KV cache between independent vLLM instances in the context of disaggregated inference.

It operates between independent models — it does not distribute the model, it coordinates state transfer between separate processes. Each node has the complete model loaded. The orchestrator assigns roles: prefill to node A, decode to node B. Node A computes the KV cache, NIXL enables direct transfer via GPUDirect RDMA, reducing CPU overhead and minimizing impact on GPU compute. Node B receives the state and starts directly from token generation.

**When to use it:** disaggregated serving (prefill/decode separation). RDMA is not strictly mandatory at functional level (NIXL can operate over TCP), but it is practically necessary to obtain real benefits: without RDMA, KV cache transfer time wipes out the disaggregation advantage in most production scenarios.

### 10. NVSHMEM — Direct access to remote GPU memory

**NVIDIA Symmetric Hierarchical Memory.** Operates on a PGAS (Partitioned Global Address Space) programming model where each GPU can directly access the memory of others through one-sided operations (put/get) without explicit collective synchronization.

**When to use it:** primarily in HPC contexts. In distributed inference its use is marginal compared to NCCL and NIXL.

### 11. Library map

| Library | Purpose | Pattern | Typical context |
|---|---|---|---|
| NCCL | Communication inside a distributed model | Collective (all-reduce, all-gather) | TP, PP, EP |
| NIXL | State transfer between independent instances | Point-to-point (KV cache transfer) | Disaggregated serving |
| NVSHMEM | Direct access to remote GPU memory | One-sided (put/get, PGAS) | HPC |

The key conceptual distinction: NCCL = collective inside a model. NIXL = state transfer between instances. NVSHMEM = direct remote memory access.

### 12. Note: UCCL P2P

UCCL P2P (UC Berkeley / UCCL Project) is an alternative KV cache transfer engine that offers NCCL/RCCL-style collective APIs while minimizing impact on GPU compute resources, with performance comparable to NIXL for messages in the typical KV cache transfer size range (256KB–1MB). Emerging approach, not yet mainstream. Worth monitoring as a fourth option in the transfer-engine landscape.

---

## Part III — Network Critical Path

Part I describes the fabric and its theoretical bandwidths. Part II describes the libraries that use it. This part answers the operational question: **when does the network actually enter the token-generation loop, and what happens when it does?**

This distinction is more important than the technological choice between RoCE and InfiniBand. First you understand whether the network is in the critical path, then you decide how to optimize it.

### 13. Definition of network-critical path

The network enters the critical path when a communication is required to proceed with execution and cannot be hidden or overlapped with compute. Every network delay translates directly into additional user-perceived latency.

This happens in three main scenarios, each tied to a parallelism strategy or architectural pattern described in other parts of this document:

- **Tensor Parallelism (§20):** nodes have to synchronize with all-reduce at every layer. Crucial distinction: if TP is confined to an intra-node NVLink domain, communication happens on the fabric (§2) — not on the network — and is not a bottleneck. If TP extends inter-node (InfiniBand, Ethernet, QPI), the actual network enters the critical path at every step of the forward pass, for every token.
- **Disaggregated serving:** the KV cache produced by the prefill node has to be transferred to the decode node before generation can start. The transfer is blocking.
- **Expert Parallelism on MoE models (§22):** token-to-expert routing generates all-to-all traffic between nodes. The pattern scales worse than TP's all-reduce because each node potentially communicates with every other, and the volume depends on dynamic router decisions — therefore less predictable.

When the network is **not** in the critical path — typically in pure Data Parallelism (§19), where each replica is independent — it serves only to distribute requests and collect responses. In that case the network choice has negligible impact on inference latency.

### 14. Latency and jitter: why stability matters more than speed

What matters is not just how fast the network is, but how stable it is.

A network with low average latency but variable (high jitter) can cause more problems than a slightly slower but stable one. In distributed systems, the final time is often determined by the slowest node (straggler effect). This effect is amplified in LLMs because decode is sequential: a single delay propagates over the entire generation.

The relevant metrics are not averages, but **p95 and p99**.

### 15. RoCE vs InfiniBand

Both technologies offer RDMA, so they share the fundamental characteristics: kernel bypass, zero-copy, very low latency.

The main difference is operational. RoCE uses Ethernet and requires careful network configuration (PFC, ECN, DSCP) to operate optimally. InfiniBand uses a dedicated fabric with native congestion-control mechanisms, resulting in greater stability under load.

InfiniBand is not necessarily faster under ideal conditions, but it is **more predictable when the system is under stress** — and predictability is what matters when the network is in the critical path (§14).

### 16. Operational signals: how to recognize that the network is the bottleneck

There are clear signals that indicate the network has become the limiting factor:

- Latency grows with concurrency even though GPUs are not saturated.
- The gap between p50 and p99 widens (rising jitter).
- Throughput stops scaling linearly.
- Decode-side queues or synchronization stalls appear.

These symptoms indicate that the limit is no longer compute, but communication. Increasing the number of GPUs or compute power produces no benefit if the network does not keep up.

### 17. Connection to the IOE framework

In the IOE (Inference Operating Envelope) framework, the network represents a **third axis** of the operating envelope, alongside concurrency and memory pressure. The point at which the network becomes a bottleneck defines a boundary of the envelope on par with the Cp (crossover point): beyond that point, scaling compute is useless.

Monitoring network signals (§16) together with throughput and TTFT metrics enables identifying which zone of the envelope the system is operating in and which resource is the real constraint.

---

## Part IV — Parallelism Strategies in LLM Inference

### 18. Why parallelism is not optional

A 70-billion-parameter LLM requires roughly 140 GB of memory for weights alone in FP16. A single H100 with 80 GB of HBM3 cannot hold it. Even when a model fits on a single device, the KV cache produced during inference grows with sequence length and batch size, rapidly exhausting memory.

Parallelism in LLM inference is not an optimization technique. It is a structural constraint.

#### The second axis: the KV cache

Model weights are the static component of memory consumption. The KV cache is the dynamic component: it grows with sequence length and number of concurrent requests. For long-context models or high-concurrency deployments, the KV cache can consume as much memory as the weights — or more.

Memory "freed" by distributing weights across multiple devices via TP or PP is often immediately consumed by the KV cache. A deployment with TP=4 does not offer 4× total capacity — it offers 4× capacity for the weights, but the KV cache on each device still grows with the workload. The practical question is not just "do the model weights fit?" but "after loading the weights, how much memory is left for the KV cache, and is that enough for the target concurrency?"

Every parallelism decision must be evaluated on both axes: weight distribution and KV cache headroom.

The KV cache is not just a memory consumer — it is also a **scheduling driver**. The serving engine's scheduler decides how many requests to admit into the batch based on available KV cache, when to preempt/evict, and how to prioritize prefill vs decode. KV cache pressure directly affects throughput, latency and system fairness, regardless of the parallelism strategy chosen.

### 19. Data Parallelism (DP)

#### Mechanism

The full model is replicated across multiple devices or nodes, and requests are distributed among the replicas. Each replica operates independently, with no inter-device communication during inference.

**Relationship with the network:** the network is not in the critical path (§13). It only serves to distribute requests and collect responses. This makes DP the most robust architecture with respect to interconnect quality.

#### Strengths

- **Linear throughput scaling.** Adding a replica doubles aggregate throughput (uniform load-balancer distribution).
- **No communication overhead.** Replicas are independent — no sensitivity to interconnect bandwidth or latency.
- **Operational simplicity.** Each replica is autonomous. A replica failure does not impact the others. Rolling updates, canary deployments and autoscaling are straightforward.

#### Failure modes

- **Memory duplication.** Each replica holds a complete copy of the weights. Four replicas of a 70B model = 560 GB of aggregate storage for weights.
- **No single-request acceleration.** DP does not reduce the latency of any individual request.
- **Load imbalance.** With requests of variable processing time, round-robin leads to uneven utilization. Smarter routing strategies (e.g. KV-cache-aware routing) mitigate this but add complexity.

#### When to use it

The model fits on the available device and the goal is to increase aggregate throughput. It is the default scaling mechanism for production serving — the first strategy to consider before introducing intra-model parallelism.

### 20. Tensor Parallelism (TP)

#### Mechanism

Individual matrix operations are split across multiple devices. The large weight matrices in attention and feed-forward layers are partitioned horizontally, each device computes a portion of the result. Partial results are combined through all-reduce or all-gather collective operations (via **NCCL**, §8).

For an `[H, H]` matrix distributed across `N` devices, each device holds a shard `[H, H/N]` or `[H/N, H]`. Every forward pass requires synchronization across all `N` devices.

#### The fabric constraint

TP requires high-bandwidth, low-latency interconnect. The synchronization step happens at every layer — for an 80-layer model with TP=4, more than 80 synchronization events per forward pass per token. **The network is in the critical path at every step** (§13).

The actual hierarchy of available bandwidths makes the constraint concrete:

| Interconnect | Bandwidth | Context |
|---|---|---|
| NVLink (H100) | 900 GB/s | Regime where TP works well |
| InfiniBand NDR | ~50 GB/s | ~18× less than NVLink |
| Ethernet 100 GbE | ~12 GB/s | ~75× less than NVLink |
| QPI/UPI between CPU sockets | 30–40 GB/s theoretical | Often less under contention |

These are not marginal differences. Going from NVLink to InfiniBand means a synchronization that took microseconds now takes tens of microseconds — per layer, per token. Over 40 or 80 layers, the accumulated overhead can dominate the forward pass. Here network stability (§14) becomes critical: even moderate jitter multiplies by the number of synchronizations.

TP over CPU-class interconnect (QPI/UPI) is not scaling: it is controlled degradation. Practical bandwidth under inference workloads is often a fraction of theoretical peak, and TP across CPU sockets has to be evaluated as a fundamentally different proposition from TP across GPUs in an NVLink domain — not assumed to scale by analogy.

#### Strengths

- Allows serving models larger than a single device.
- Reduces memory per device (`1/N` of weight matrices), freeing space for the KV cache.
- Reduces single-request latency (with adequate interconnect).

#### Failure modes

- **Interconnect dependency.** If the interconnect is slow, the forward pass becomes serialized on communication. The most common failure mode, often invisible in synthetic benchmarks. Operational signals from §16 are the most reliable way to detect it.
- **Diminishing returns.** Doubling the TP degree halves computation but not communication. Past a certain point, adding devices increases latency.
- **Complexity.** Tight coupling between devices. A failure on any device blocks the entire replica.

#### When to use it

The model does not fit on a single device and high-bandwidth interconnect (NVLink or equivalent) is available. Inside a node, TP is almost always the right choice for large models. Across nodes or across CPU sockets with limited bandwidth, approach with caution.

### 21. Pipeline Parallelism (PP)

#### Mechanism

The model is partitioned by layers, assigning consecutive groups to different devices. A request flows sequentially through the pipeline. Micro-batching mitigates the inefficiency of sequential processing, keeping multiple devices simultaneously active.

**Relationship with the network:** point-to-point communication between adjacent stages (via **NCCL** send/recv). The network is in the critical path only at the boundaries between stages, not at every layer as in TP — significantly lower interconnect requirements.

#### Strengths

- **Lower interconnect requirements.** Point-to-point communication, not all-to-all collectives.
- **Scales across nodes.** Practical for multi-node deployments without NVLink.
- **Uniform memory distribution.** Each device holds about `1/N` of the layers.

#### Failure modes

- **Pipeline bubble.** At the start and end of batch processing, some stages are idle. Significant with few micro-batches.
- **Increased latency.** For a single request, PP adds sequential latency at every stage boundary. Direct impact on TTFT.
- **Poor compatibility with MoE.** Heterogeneous layer costs make balanced partitioning hard.

#### When to use it

The model has to be distributed across multiple nodes without high-bandwidth interconnect. Often combined with TP (TP intra-node, PP inter-node).

### 22. Expert Parallelism (EP)

#### Mechanism

Applies specifically to Mixture-of-Experts (MoE) architectures. Experts are distributed across devices. When a token is routed to an expert on a remote device, the embedding is sent, processed, and returned.

**Relationship with the network:** routing generates all-to-all traffic between nodes (§13). This pattern scales worse than TP's all-reduce: every node potentially communicates with every other, and the volume depends on dynamic router decisions. Traffic is data-dependent and time-varying — this breaks static capacity-planning models, because network load is not predictable a priori from the workload. The network is in the critical path in a less obvious but equally real way. The unpredictability of routing makes network jitter (§14) particularly impactful. Given the market direction toward MoE architectures, this pattern will become increasingly relevant.

#### Strengths

- **Memory efficiency for large MoE models.** No single device has to hold all experts.
- **Reduced compute per token.** Only activated experts compute on each token.

#### Failure modes

- **Routing imbalance.** If certain experts are "hot", the devices holding them become bottlenecks. EPLB techniques mitigate this by dynamically replicating the most-requested experts.
- **All-to-all communication.** More complex than TP's all-reduce or PP's point-to-point.
- **Limited applicability.** MoE models only.

#### When to use it

Large MoE models where the total parameter count of experts exceeds the memory of a single device. Typically combined with TP for the attention layers.

EP is architecturally different from the other three strategies. DP, TP and PP are general-purpose. EP is model-architecture-dependent: it exists because MoE models have a specific internal structure (discrete expert sub-networks) that can be physically separated.

---

## Part V — Combinations and Decision Framework

### 23. Hybrid parallelism

Production deployments rarely use a single strategy in isolation.

| Combination | Pattern | Context |
|---|---|---|
| **TP + DP** | TP intra-node (NVLink), DP inter-node | Most common production pattern |
| **TP + PP** | TP intra-node, PP inter-node | Model too large for a single node |
| **TP + EP + DP** | TP for attention, EP for experts, DP for throughput | Large MoE models |
| **TP + PP + EP + DP** | Full combination | Larger MoE on multi-node clusters |

**Guiding principle:** use the fastest interconnect for the strategy with the highest communication intensity. TP requires the highest bandwidth → NVLink. PP tolerates lower bandwidth → can extend across nodes. DP requires no inter-device communication. EP sits between TP and PP in bandwidth sensitivity.

### 24. Decision framework

#### Step 0: Estimate the KV cache footprint at target context length and concurrency

Before asking whether the model "fits" on a device, calculate KV cache consumption at target context length and expected concurrency. A model that fits in memory but leaves no headroom for the KV cache will limit concurrency, increase queuing and degrade throughput under load — exactly the conditions parallelism was meant to solve. If projected KV cache exceeds available memory after weights, more memory is needed (TP/PP) or offload strategies (KV cache to CPU/disk), regardless of whether the weights fit.

#### Step 1: Does the model fit on a single device (with sufficient KV headroom)?

Yes → **DP** to scale throughput. No need for intra-model parallelism.

No → Step 2.

#### Step 2: Does the model fit on a single node (multiple devices with high-bandwidth interconnect)?

Yes → **TP** inside the node. **DP** across nodes for throughput.

No → Step 3.

#### Step 3: Multi-node distribution required.

**TP** inside each node (NVLink). **PP** across nodes (InfiniBand/Ethernet). **DP** to replicate the TP+PP configuration.

#### Step 4: Is the model MoE?

Yes → **EP** for the expert layers, combined with TP for the attention layers.

#### Cross-cutting concern: interconnect quality

At every step, verify that the interconnect bandwidth supports the chosen strategy (see hierarchy §6 and bandwidths §20). TP on slow interconnects can be worse than no TP at all. When interconnect is the bottleneck, prefer PP over TP for inter-device distribution, or accept the memory cost of DP with quantization. If operational signals from §16 indicate the network is the bottleneck, no software optimization will solve the problem.

#### Cross-cutting concern: the gray zone

A model could technically fit on a single device but leave almost no room for the KV cache. An interconnect could be fast enough for TP at low concurrency but degrade under load (here p95/p99 monitoring from §14 is decisive). In these borderline cases, the deciding factor is often operational cost: how many failure modes does the configuration introduce? How complex is scheduling? When performance differences between two valid configurations are within 10–15%, the simpler architecture is usually the right choice.

### 25. Quick rules of thumb

| Problem | Solution | Fabric level |
|---|---|---|
| Local GPU↔GPU | NVLink | L0 — node |
| Coherent CPU↔GPU | NVLink-C2C | L0 — node |
| Synchronous multi-GPU scaling | NVFabric | L0 — rack/pod |
| Generic communication and distance | Network (IB / Ethernet) | Infrastructure |
| Distributed model, synchronous communication | NCCL | TP, PP, EP |
| KV cache transfer between independent instances | NIXL (or UCCL P2P) | Disaggregated serving |
| Direct remote GPU memory access | NVSHMEM | HPC |
| Network in the critical path, stability needed | InfiniBand > RoCE | TP, EP, disaggregation |
| Network out of the critical path | Any | Pure DP |

### 26. Note on disaggregated serving

Disaggregating prefill/decode is not a parallelism strategy in the traditional sense, but an architectural pattern that changes how strategies are applied. Prefill workers can use aggressive TP on compute-intensive GPUs, while decode workers use GPUs optimized for memory bandwidth. The KV cache is transferred via **NIXL**/GPUDirect RDMA. The network is in the critical path for cache transfer (§13).

This pattern is covered in detail in a separate document on disaggregated inference. It is mentioned here because it interacts directly with parallelism and fabric choices.

### 27. Serving-engine constraints

Parallelism theory is broader than the actual capabilities of serving engines. The taxonomy above describes strategies as freely composable. In practice, the serving engine imposes its own constraints and implementation quality varies significantly. As of early 2026: TP and DP are well supported by the main engines (vLLM, TensorRT-LLM, SGLang). PP support in inference is less mature and often comes with performance caveats. EP support is emerging and tightly coupled to specific MoE architectures. The gap between what is architecturally correct and what a given framework can efficiently execute is real and must be empirically validated before committing to a deployment topology.

---

## Strategy summary table

| Strategy | Partitions | Communication pattern | Interconnect requirement | Network in critical path | Library | Primary use case |
|---|---|---|---|---|---|---|
| **DP** | Requests across replicas | None between replicas | None | No | N/A | Throughput scaling |
| **TP** | Weight matrices across devices | All-reduce per layer | High (NVLink) | Yes, every layer | NCCL | Models > single-device memory |
| **PP** | Layers across devices | Point-to-point between stages | Moderate | Yes, between stages | NCCL | Multi-node distribution |
| **EP** | MoE experts across devices | All-to-all for token routing | Moderate-high | Yes, unpredictable | NCCL | Large MoE models |

---

*dielabs.github.io · github.com/Dielabs*
