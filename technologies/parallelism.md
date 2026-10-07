---
layout: default
title: "Parallelism — Fabric, Libraries and Strategies"
redirect_from:
  - /technologies/inference-workload-architectures
  - /technologies/inference-workload-architectures.html
  - /architectures/inference-workload-architectures.html
---

# Parallelism — Fabric, Libraries and Strategies

> The moving layer of parallelism in inference: physical interconnect, communication libraries, when the network enters the critical path, and the parallelism strategies — the four classic ones plus the new context-parallel ones for long contexts and MoE. Reference versions: vLLM 0.29, NVLink 5 / Blackwell, September 2026.

Foundations: [The Inference Engineering Manual](/foundations/inference-engineering-manual), Chapter 4 (why a model must be distributed and what it costs), and the [KV Cache Manual](/foundations/kv-cache-workbook) (the second memory axis). This page holds what changes: fabric generations, libraries, the strategies available in engines and their flags.

The guiding principle: **parallelism choices are constrained by the physical layer**. No software strategy overcomes the bandwidth and latency limits of the underlying fabric. First understand when the network enters the generation loop, then pick the strategy.

---

## Part I — The GPU Compute Fabric

### 1. The Underlying Problem

In early accelerated systems a GPU worked almost in isolation, the CPU orchestrated, communication was occasional. With LLMs, distributed training and persistent inference, GPUs must synchronize continuously, GPU↔GPU traffic dominates total time, and latency becomes more critical than nominal bandwidth. What you need is a **compute fabric**, not a network.

### 2. NVLink — GPU↔GPU Inside the Node

Point-to-point interconnect with very high bandwidth and low latency between chips.

| Generation | Architecture | Bidirectional bandwidth per GPU | Notes |
|---|---|---|---|
| NVLink 3.0 | Ampere (A100) | 600 GB/s | 12 links × 50 GB/s |
| NVLink 4.0 | Hopper (H100/H200) | 900 GB/s | 18 links × 50 GB/s |
| NVLink 5.0 | Blackwell (B200/GB200/B300) | 1,800 GB/s | 18 links × 100 GB/s |

These numbers are the **physical ceiling** of local parallelism. In 4/8-GPU systems every GPU is connected to the others via NVLink, often through NVSwitch, in an almost fully connected domain. NVLink is not a network, does not involve the CPU, and does not add CPU↔GPU coherence: it solves local parallelism, not system scale.

### 3. NVLink-C2C — Coherent CPU↔GPU

With Grace, NVLink-C2C is the coherent interconnect between the Grace CPU and the GPU.

| Parameter | NVLink-C2C (GH200) | PCIe Gen5 x16 |
|---|---|---|
| Bidirectional bandwidth | 900 GB/s | ~128 GB/s |
| Delta | ~7× | baseline |
| Memory coherence | hardware, native | not supported |
| Semantics | compute datapath | I/O bus |

It does not connect GPUs to each other and does not scale across nodes: it removes the CPU↔GPU bottleneck. Historically "C2C" meant any direct chip-to-chip link; today "NVLink-C2C" specifically means Grace ↔ GPU.

### 4. NVFabric — NVLink Beyond the Node

The extension of NVLink to rack or pod level through dedicated NVLink Switches; formalized with Blackwell and GB200 NVL72 (72 GPUs in one rack-level NVLink domain), but already present in Hopper DGX SuperPODs without this name.

NVLink Switches route NVLink traffic with deterministic latency and support collective primitives in hardware; they are not network switches. NVLink cables are proprietary, point-to-point, and do not encapsulate packets: they are physical extensions of the bus, not network cables — very fast, and very constraining in distance and layout.

### 5. NVFabric vs Traditional Networks

| Aspect | NVFabric | Networks (InfiniBand / Ethernet) |
|---|---|---|
| Semantics | compute and synchronization | message-based |
| Switching | minimal, deterministic latency | packet switching |
| Orchestration | hardware-accelerated collectives | explicit software |
| Scale | narrow physical domain (rack/pod) | geographic scale-out |

NVFabric does not replace networks: it sits alongside them for compute-critical traffic. With Blackwell it effectively becomes a system fabric, but it stays bound to narrow domains and to communication patterns known in advance.

### 6. The Interconnect Hierarchy

| Level | Domain | Technology | Function |
|---|---|---|---|
| 1 | inside the GPU | internal interconnect | SM ↔ HBM |
| 2 | between GPUs in the node | NVLink / NVSwitch | local parallelism (TP, DCP) |
| 3 | between nodes in the rack/pod | NVFabric | synchronous multi-node scale |
| 4 | between racks and data centers | InfiniBand / Ethernet | scale-out, storage, control |

Each level solves a different problem and they are not interchangeable. Using a network as a fabric leads to structural inefficiencies; using NVFabric as a network leads to unmanageable constraints.

### 7. Implications for the Data Center

The rack becomes the unit of design, cabling is rigid and predefined, the facility (power, cooling, space) is part of the system, and traditional IT converges towards HPC. The fabric reduces flexibility and increases efficiency on highly parallel workloads.

---

## Part II — Communication Libraries

The choice between libraries is not a preference: it is set by the deployment architecture.

### 8. NCCL — Inside a Distributed Model

NVIDIA Collective Communications Library. Born for training, adopted for inference with TP: it distributes the model across several GPUs that cooperate in a single forward pass. Collective primitives (all-gather, all-reduce, broadcast, reduce-scatter, send/recv) optimized for PCIe, NVLink and InfiniBand. It uses dedicated copy engines with SM assistance: contention with compute is workload-dependent — minimal on compute-heavy workloads, significant on memory-bound workloads with frequent communication.

**When:** whenever the model is distributed with TP, PP, EP or CP. Synchronous communication at every layer or step.

### 9. NIXL — Between Independent Instances

NVIDIA Inference Transfer Library, part of Dynamo but also adopted by native vLLM, LMCache and SGLang. It transfers the KV cache between independent instances: it does not distribute the model, it coordinates state handoff between separate processes, each holding the full model. It enables GPUDirect RDMA and, with storage backends, tiering onto NVMe and S3.

**When:** disaggregated serving and KV tiering. RDMA is not mandatory (NIXL also runs over TCP) but practically necessary: without it, the transfer cancels out the benefit of disaggregation.

### 10. NVSHMEM — Remote GPU Memory

PGAS model: each GPU accesses the memory of the others directly with one-sided operations (put/get). Mostly HPC; marginal in distributed inference, but it is the basis of the **symmetric memory** that vLLM uses to fuse DCP collectives into the attention kernels (§22).

### 11. Library Map

| Library | Purpose | Pattern | Context |
|---|---|---|---|
| NCCL | communication inside a distributed model | collective | TP, PP, EP, CP |
| NIXL | state transfer between instances | point-to-point | disaggregation, KV tiering |
| NVSHMEM | direct access to remote memory | one-sided | HPC; fused DCP kernels |

### 12. Note — UCCL P2P

UCCL P2P (UC Berkeley) is an alternative transfer engine with NCCL/RCCL-style collective APIs and minimal compute impact, with performance comparable to NIXL at typical KV transfer sizes (256 KB–1 MB). Emerging, not mainstream.

---

## Part III — The Network Critical Path

When the network enters the generation loop, and what happens when it does. This distinction matters more than the choice between RoCE and InfiniBand.

### 13. Definition

The network is in the critical path when a communication is required to proceed and cannot be hidden or overlapped with compute: every network delay becomes perceived latency. Four scenarios:

- **TP (§20):** all-reduce at every layer. If TP is confined to an NVLink domain, communication stays on the fabric and is not a bottleneck; if it extends across nodes (IB, Ethernet, QPI), the network is in the critical path at every step, for every token.
- **DCP (§22):** all-gather of the query and reduce-scatter of the output at every attention layer during decode. Same regime as TP: it only lives inside the NVLink domain.
- **Disaggregation ([Disaggregated Inference](disaggregated-inference.md)):** the KV from prefill must reach decode before generation starts. Blocking.
- **EP on MoE (§21.3):** token → expert routing with all-to-all traffic. It scales worse than all-reduce: every node can potentially talk to every other, and volume depends on the router's dynamic decisions.

In pure DP (§19) the network is not in the critical path: it distributes requests and collects responses.

### 14. Latency and Jitter

Stability matters, not just speed. A network with low but variable average latency causes more problems than a slower, stable one: in distributed systems the final time is set by the slowest node (the straggler), and in sequential decode a single delay propagates through the whole generation. Metrics: p95 and p99, never averages.

### 15. RoCE and InfiniBand

Both offer RDMA (kernel bypass, zero-copy, low latency). The difference is operational: RoCE runs on Ethernet and needs careful configuration (PFC, ECN, DSCP); InfiniBand has a dedicated fabric with native congestion control. InfiniBand is not faster under ideal conditions, it is **more predictable under stress** — which is what counts when the network is in the critical path. On Ethernet, Spectrum-X with adaptive routing and NIC↔switch congestion control narrows the gap.

### 16. Operational Signals

The network has become the limit when: latency grows with concurrency while GPUs are not saturated; the p50–p99 gap widens; throughput stops scaling linearly; queues build up on the decode side or synchronization stalls appear. In that regime, adding GPUs brings no benefit.

### 17. The Network as a Third Constraint

Alongside concurrency and memory pressure, the network is a third operational constraint. The point where it becomes the bottleneck defines a boundary just like [CrossP](/frameworks/one-capacity-is-not-enough): beyond it, scaling compute is useless. Monitoring the signals of §16 together with throughput and TTFT tells you which regime you are operating in.

---

## Part IV — Parallelism Strategies

### 18. Why Parallelism Is Not Optional

A 70B model in FP16 is ~140 GB of weights; an H100 has 80. And even when the weights fit, the KV cache grows with sequences and batch. Parallelism is not an optimization: it is a structural constraint.

**The second axis — the KV cache.** Weights are the static component; the KV is the dynamic one, and for long context or high concurrency it can exceed the weights. The memory freed by distributing the weights is often consumed right away by the KV: TP=4 gives 4× capacity for weights, not 4× total capacity. The question is "after the weights, how much memory is left for the KV, and is it enough for the target concurrency?". The KV is also a scheduling driver: it decides how many requests to admit, when to preempt, how to balance prefill and decode.

### 19. Data Parallelism

The model is replicated, requests are distributed across independent replicas. The network is out of the critical path: the architecture most robust to interconnect quality.

Strengths: linear throughput scaling, no communication overhead, operational simplicity (isolated failures, rolling updates, autoscaling). Weaknesses: memory duplication (four replicas of a 70B = 560 GB of weights), no speed-up for a single request, imbalance with round-robin — mitigated by KV-aware routing.

**When:** the model fits on the device and the goal is aggregate throughput. It is the first strategy to consider.

### 20. Tensor Parallelism

The weight matrices of attention and feed-forward are partitioned across devices; partial results are combined with all-reduce or all-gather via NCCL. For an `[H, H]` matrix across N devices, each device holds an `[H, H/N]` shard; each forward pass synchronizes all N devices.

**The fabric constraint.** Synchronization happens at every layer: 80 layers with TP=4 means more than 80 events per token.

| Interconnect | Bandwidth | Regime |
|---|---|---|
| NVLink 4 (H100) | 900 GB/s | TP works |
| NVLink 5 (B200) | 1,800 GB/s | TP works |
| InfiniBand NDR | ~50 GB/s | ~18× less than NVLink 4 |
| Ethernet 100 GbE | ~12 GB/s | ~75× less |
| QPI/UPI between CPU sockets | 30–40 GB/s theoretical | often less under contention |

Moving from NVLink to InfiniBand takes a synchronization from microseconds to tens of microseconds, per layer, per token; over 40–80 layers the overhead dominates. TP on a CPU-class interconnect is not scaling: it is controlled degradation. The lab measured this on a dual-socket server in [What CPUs Teach About GPU Inference](/papers/what-cpus-teach-about-gpu-inference).

Strengths: models larger than one device; `1/N` of the weights per device, leaving room for the KV; lower single-request latency with an adequate interconnect. Weaknesses: dependence on the interconnect (often invisible in synthetic benchmarks — the signals of §16 reveal it); diminishing returns (doubling TP halves compute, not communication); tight coupling, one failure stops the replica. On MLA models, TP replicates the latent KV on every rank instead of splitting it: memory-inefficient, and the reason DCP exists (§22).

**When:** the model does not fit on one device and NVLink is available. Inside a node it is almost always the right choice for large dense models.

**A measured exception: two DGX Sparks (September 2026).** The table above suggests TP outside NVLink should always be avoided. On two DGX Sparks connected by a direct QSFP cable (RoCE at 200 GbE, about 23 GB/s measured with NCCL) that is not the case. On four models, TP=2 was 14–37% faster than PP=2, despite moving 100 to 290 times more data over the link. A separate NCCL test also shows that on this link PP's point-to-point exchange is not noticeably faster than TP's all-reduce. TP=2 is also the path of the NVIDIA playbooks and of every published recipe for GLM-5.3-Flash on two Sparks. The cost of the link remains, though, so the rule is: join nodes to make the model fit, replicate to scale.

### 21. Pipeline and Expert Parallelism

#### 21.1 Pipeline Parallelism

The model is partitioned by layer, with consecutive groups on different devices; the request flows through them in sequence, and micro-batching keeps several stages active. Point-to-point communication between adjacent stages (NCCL send/recv): the network is in the critical path only at stage boundaries — far lower interconnect requirements than TP.

Strengths: scales across nodes without NVLink; memory evenly distributed. Weaknesses: pipeline bubbles (idle stages at the start and end of a batch, heavy with few micro-batches); added latency at every boundary, with a direct impact on TTFT; poor fit with MoE (heterogeneous layer costs).

**The AgentX lesson (September 2026).** PP, including chunked pipeline parallelism, performs well on long, cold prompts: large prefills fill the stages and scaling is almost linear. But most agentic turns are warm — system prompt and history already in cache, a few hundred or thousand new tokens — and there is not enough fresh compute to fill the pipeline: bubbles eat the gain. PP remains valid for cold, compute-heavy prefills; it should not be the default for warm, prefix-heavy turns.

**When:** a model to distribute across several nodes without a high-bandwidth interconnect; typically TP intra-node + PP inter-node; or dedicated prefill workers for cold requests.

#### 21.2 Chunked Pipeline Parallelism

CPP (or context/sequence pipeline parallelism): fine-grained PP in which the prefill is split into chunks sized on compute load and streamed through the pipeline, with SLO-driven scheduling. Designed for ultra-long (1M-token) and variable-length sequences, where classic context parallelism degrades. In vLLM it is under development (issue #28912, paper 2409.17264); in the AgentX roadmap it is paired with PCP for first-turn workers.

#### 21.3 Expert Parallelism

MoE only: experts are distributed across devices; a token routed to a remote expert is sent, processed and returned. All-to-all traffic that is data-dependent and time-varying: it breaks static capacity planning models, and network jitter weighs more than in TP.

Strengths: memory-efficient for large MoEs; reduced compute per token. Weaknesses: routing imbalance ("hot" experts; EPLB dynamically replicates the most requested ones); all-to-all is more complex than all-reduce; MoE only. The share of experts read per layer grows with concurrency — ≈ 1 − (1 − k/E)^B — so the "few active parameters" advantage erodes at high batch; EP is combined with TP or DP for attention.

### 22. Context Parallelism — DCP, PCP, DEP

The classic strategies partition weights (TP), layers (PP), experts (EP) or requests (DP). **Context parallelism** partitions the **sequence**: the KV cache is sharded along the token dimension, and each rank holds 1/N of it. It was born for long contexts and for MLA models, where TP replicates the latent KV. vLLM has had it since late 2025 (PR #23734), but it is with the agentic workloads of 2026 that it became central.

#### 22.1 Decode Context Parallelism

DCP reuses the GPUs of the TP group without changing the world size: `--decode-context-parallel-size N` (`-dcp`), with `tp % dcp == 0`; the TP group splits into `tp/dcp` DCP groups, and the KV token budget grows `dcp` times. The pattern for every attention layer during decode: all-gather of the query (cheap: a single token), local attention on its own KV shard, correction with the exchanged log-sum-exp values, reduce-scatter of the output. With symmetric memory (§10) vLLM fuses these steps into the attention kernels, avoiding NCCL: ~13% lower latency per layer.

Two benefits for agents: a shorter decode (MLA attention is memory-bound and grows with context; sharding it shortens the step) and more KV per GPU (no replication → more sequences in flight). vLLM data: ~3× long-context throughput versus TP with the same GPUs; on Kimi K3, DCP8 beats TP8 in decode latency and scales to higher concurrency. Constraints: only inside the NVLink domain (§13); for GQA it requires `(tp // kv_heads) % dcp == 0`; cannot be combined with attention DP; sliding window not supported.

#### 22.2 Prefill Context Parallelism

PCP shards the prompt across ranks during prefill, expanding the world size (`--prefill-context-parallel-size`; world = tp × pcp). It distributes the compressor and indexer of sparse-MLA models and gives attention a more efficient head-local shape: on DeepSeek V4 with 32K prompts, PCP8 prefills 2.65× faster than TP8. But it replicates decode state across ranks: suited to dedicated prefill workers, not to decode.

#### 22.3 Data + Expert Parallelism

DEP: requests and their KV are assigned to different data-parallel ranks, attention stays fully local, and MoE experts are sharded across ranks. It avoids DCP's attention collectives and scales better on large scale-up domains: on NVL72, DEP16 overtakes DCP8 as soon as the batch per rank exceeds ~3. It is vLLM's default for most DeepSeek V4 configurations. The price: KV cache isolated per rank (no prefix reuse across ranks without a shared pool) and ranks in lockstep on the MoE all-to-all, which requires a controlled prefill cadence in the scheduler.

#### 22.4 Parallelism Follows the Model Architecture

The cross-cutting AgentX lesson: a strategy that works on one latent-attention model may not transfer to another. DCP pays off on pure MLA models (DeepSeek R1, Kimi K2.x) and hybrids (Kimi K3), but on DeepSeek V4 — with compressor, indexer and sparse attention to coordinate — it at best ties with DEP, after heavy kernel investment. There is no optimal configuration per model family: you measure per model.

#### 22.5 Note — Attention-FFN Disaggregation

AFD separates attention and FFN/MoE onto different GPUs instead of replicating attention on every EP rank. In vLLM it is a plugin (GPU and Ascend backends, execution via connector). Emerging: it is the direction in which parallelism and disaggregation converge.

---

## Part V — Combinations and Choice

### 23. Hybrid Parallelism

| Combination | Pattern | Context |
|---|---|---|
| TP + DP | TP intra-node, DP inter-node | production, dense models |
| TP + PP | TP intra-node, PP inter-node | dense model too large for one node |
| TP + DCP | DCP inside the TP group | MLA and long context, decode |
| DEP (+ TP attention) | DP for requests, EP for experts | large MoE, NVL72 |
| PCP on prefillers + DCP/DEP on decoders | disaggregated, with per-phase parallelism | agentic at scale |
| TP + PP + EP + DP | full | largest multi-node MoE |

Principle: give the fastest interconnect to the strategy with the highest communication intensity. TP and DCP → NVLink; EP sits between TP and PP in sensitivity; PP tolerates lower bandwidth; DP does not communicate.

### 24. Decision Framework

**Step 0 — KV cache footprint.** Before asking whether the model "fits", compute the KV at the target context and concurrency. A model that fits but leaves no headroom will limit concurrency, grow queues and degrade under load — exactly the conditions parallelism was meant to solve. If the projected KV exceeds the memory left after the weights: more memory (TP/DCP) or offload.

**Step 1 — Does it fit on one device?** Yes → DP. No → step 2.

**Step 2 — Does it fit on one node?** Yes → TP inside the node, DP across nodes. If the model is MLA or the workload is long-context/agentic → TP + DCP. No → step 3.

**Step 3 — Multi-node.** TP inside the node (NVLink), PP across nodes, DP to replicate. On a rack-level NVLink domain (NVL72), evaluate DEP before PP.

**Step 4 — Is it MoE?** EP for the experts with TP for attention, or DEP; measure expert imbalance on real traffic.

**Step 5 — Is it disaggregated?** Per-phase parallelism: PCP or chunked PP on the prefillers, DCP or DEP on the decoders. See [Disaggregated Inference](disaggregated-inference.md).

**Cross-cutting — interconnect quality.** At every step, check that the bandwidth supports the strategy (§6, §20). TP on a slow interconnect can be worse than no TP; if the network is the bottleneck (§16), no software optimization solves it.

**Cross-cutting — the grey zone.** A model that fits but leaves no room for the KV; an interconnect adequate at low concurrency but degrading under load (p95/p99 are decisive). In borderline cases, operational cost decides: how many failure modes, how much scheduling. If two valid configurations differ by less than 10–15%, the simpler one wins.

### 25. Quick Rules

| Problem | Solution | Level |
|---|---|---|
| local GPU↔GPU | NVLink | node |
| coherent CPU↔GPU | NVLink-C2C | node |
| synchronous multi-GPU scale | NVFabric | rack/pod |
| general communication and distance | IB / Ethernet network | infrastructure |
| distributed, synchronous model | NCCL | TP, PP, EP, CP |
| KV transfer between instances | NIXL (or UCCL P2P) | disaggregation, tiering |
| network in the critical path, stability needed | InfiniBand > RoCE | TP, EP, disaggregation |
| network out of the critical path | anything | pure DP |
| MLA or long-context model, decode | DCP inside the TP group | NVLink |
| long, cold prefill | PCP or chunked PP on prefillers | disaggregated |
| large MoE on NVL72 | DEP | NVLink rack |

### 26. Disaggregation and Parallelism

Disaggregation is not a parallelism strategy but a pattern that changes how strategies are applied: prefillers with PCP or aggressive TP on compute-intensive GPUs, decoders with DCP or DEP on memory-bandwidth GPUs, KV transferred via NIXL. The P/D ratio is found with vLLM's two-phase methodology: separate saturation of prefill-only and decode-only workers across parallelism and size, then a concurrency sweep on the combined system. Covered in [Disaggregated Inference](disaggregated-inference.md).

### 27. Engine Constraints — September 2026

The taxonomy describes strategies as composable; the engine imposes its own constraints. In vLLM 0.29: TP and DP mature; PP supported, with the caveats of §21.1; EP and EPLB in production on the main MoEs; DCP mature on MLA and GQA with FlashMLA/FlashInfer backends, PCP available, CPP in development; DEP the default on frontier MoEs; AFD via plugin. TensorRT-LLM and SGLang follow with their own CP implementations. The gap between what is architecturally correct and what an engine runs well remains real: validate empirically before committing to a topology.

### 28. Summary Table

| Strategy | Partitions | Communication | Interconnect | Network in critical path | Use case |
|---|---|---|---|---|---|
| DP | requests across replicas | none | none | no | throughput |
| TP | weight matrices | all-reduce per layer | high (NVLink) | yes, every layer | dense > one device's memory |
| PP | layers across stages | point-to-point | moderate | yes, between stages | multi-node without NVLink; cold prefills |
| EP | MoE experts | all-to-all | moderate-high | yes, unpredictable | large MoE |
| DCP | KV along the sequence (decode) | all-gather Q + reduce-scatter per layer | high (NVLink) | yes, every layer | MLA, long context, agents |
| PCP | prompt along the sequence (prefill) | collectives on prefill | high | yes | dedicated prefillers |
| DEP | requests + experts | MoE all-to-all, local attention | high (NVLink rack) | yes, lockstep | frontier MoE on NVL72 |

---

## Sources

- NVIDIA: NVLink, NVLink-C2C, NVLink Switch, GB200 NVL72, NCCL, NIXL, NVSHMEM documentation.
- vLLM Blog: "Efficient Decode Context Parallelism with vLLM for Long Context Workloads" (7 Aug 2026); "vLLM x AgentX" (8 Sep 2026) — DCP, PCP, DEP, lessons on PP and DCP; PR #23734 (DCP), RFC #25749 (CP), issue #28912 (CPP); Context Parallel docs.
- UCCL Project (UC Berkeley), UCCL P2P.
- TP vs PP measurements on two DGX Sparks: github.com/myers-dev/tp-vs-pp-on-two-dgx-sparks (Jun 2026); NCCL send/recv vs all-reduce test: multimodalflow.net/en/blog/dgx-spark-dual-node-nccl-rdma (Jun 2026); NVIDIA vLLM multi-node playbook: build.nvidia.com/playbooks/vllm/multi-node.
- Dielabs: [What CPUs Teach About GPU Inference](/papers/what-cpus-teach-about-gpu-inference) — SN vs TP vs DP crossover measured on bandwidth-constrained hardware.

*Original Dielabs work by Diego Bardella.*
