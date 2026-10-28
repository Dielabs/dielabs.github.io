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

The moving layer: products, versions, topologies and measured numbers. GPU fabric, communication libraries, network behavior under load, parallelism strategies. These pages get rewritten when the landscape moves; the mechanics they rest on are in [Foundations](/foundations/).

---

## Documents

### [Parallelism — Fabric, Libraries and Strategies](parallelism.md)
The physical and software layers of parallelism in inference: NVLink generations, NVLink-C2C and NVFabric, the communication libraries (NCCL, NIXL, NVSHMEM), when the network enters the token-generation critical path, the four classic strategies (DP, TP, PP, EP) and the context-parallel ones for long contexts and MoE (DCP, PCP, DEP), with a decision framework that starts from the KV cache footprint and the engine constraints as of September 2026.

---

*All content is original Dielabs work by Diego Bardella.*
