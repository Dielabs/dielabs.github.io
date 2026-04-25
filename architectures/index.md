---
title: Architectures
layout: default
---

# Architectures

Architectural references for AI inference infrastructure: GPU fabric, communication libraries, network behavior under load, parallelism strategies, and deployment patterns. The connective tissue between datacenter infrastructure and LLM serving.

---

## Documents

### [Inference Workload Architectures](inference-workload-architectures.md)
Unified decision framework covering the full path from physical interconnect to parallelism strategy: NVLink and NVFabric topology, NVIDIA communication libraries (NCCL, NIXL, NVSHMEM), when the network enters the token-generation critical path, and the four parallelism strategies (DP, TP, PP, EP) with their fabric constraints and failure modes. Layer L0–L1 focus in the Dielabs Inference Stack Model, with controlled extensions to L2/L3.

---

*All content is original Dielabs work by Diego Bardella.*
