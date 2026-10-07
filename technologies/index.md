---
redirect_from:
  - /architectures/
  - /architectures/index.html
  - /technologies/cpu-gpu-memory-topology
  - /technologies/cpu-gpu-memory-topology.html
  - /architectures/cpu-gpu-memory-topology.html
title: Technologies
layout: default
---

# Technologies

The moving layer: products, versions, topologies and measured numbers. GPU fabric, communication libraries, network behavior under load, parallelism strategies, disaggregation. These pages get rewritten when the landscape moves; the mechanics they rest on are in [Foundations](/foundations/).

---

## Documents

### [Parallelism — Fabric, Libraries and Strategies](parallelism.md) <span class="read-time">19 min</span>
The physical and software layers of parallelism in inference: NVLink generations, NVLink-C2C and NVFabric, the communication libraries (NCCL, NIXL, NVSHMEM), when the network enters the token-generation critical path, the four classic strategies (DP, TP, PP, EP) and the context-parallel ones for long contexts and MoE (DCP, PCP, DEP), with a decision framework that starts from the KV cache footprint and the engine constraints as of September 2026.

### [Disaggregated Inference](disaggregated-inference.md) <span class="read-time">20 min</span>
Architectural reference for the disaggregated serving pattern: prefill/decode separation, KV cache transfer over RDMA, NIC vs DPU on the data path, KV pooling tiers (NVMe-oF today, CXL as direction), performance analysis with fair-baseline methodology, speculative decoding interactions, and a decision framework for when disaggregation pays off versus when it adds complexity without benefit. The guiding principle: KV cache transfer dominates the design space, and TTFT — not GB/s — is the KPI that matters.

---

*All content is original Dielabs work by Diego Bardella.*
