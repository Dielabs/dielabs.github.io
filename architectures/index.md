---
title: Architectures
layout: default
---

# Architectures

Architectural references for AI inference infrastructure: GPU fabric, communication libraries, network behavior under load, parallelism strategies, deployment patterns, and memory topology. The connective tissue between datacenter infrastructure and LLM serving.

---

## Documents

### [Inference Workload Architectures](inference-workload-architectures.md)
Unified decision framework covering the full path from physical interconnect to parallelism strategy: NVLink and NVFabric topology, NVIDIA communication libraries (NCCL, NIXL, NVSHMEM), when the network enters the token-generation critical path, and the four parallelism strategies (DP, TP, PP, EP) with their fabric constraints and failure modes. Layer L0–L1 focus in the Dielabs Inference Stack Model, with controlled extensions to L2/L3.

### [Disaggregated Inference](disaggregated-inference.md)
Architectural reference for the disaggregated serving pattern: prefill/decode separation, KV cache transfer over RDMA, NIC vs DPU on the data path, KV pooling tiers (NVMe-oF today, CXL as direction), performance analysis with fair-baseline methodology, speculative decoding interactions, and a decision framework for when disaggregation pays off versus when it adds complexity without benefit. The guiding principle: KV cache transfer dominates the design space, and TTFT — not GB/s — is the KPI that matters.

### [CPU-GPU Memory Topology for AI Inference](cpu-gpu-memory-topology.md)
Taxonomy of CPU-GPU memory architectures and their implications for LLM inference. Distinguishes "unified" (programming model) from "coherent" (hardware implementation) as orthogonal axes, and maps the four operational quadrants — Discrete + PCIe, Homogeneous Unified (Apple, DGX Spark), Heterogeneous Unified (GH200, GB200, MI300A), Legacy Shared. Covers KV cache offload mechanisms across patterns, the access-cost framing, and the mental shift from data movement to data placement.

---

*All content is original Dielabs work by Diego Bardella.*
