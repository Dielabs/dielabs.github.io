---
layout: default
title: "Inference Diagnostic Framework"
---

# Inference Diagnostic Framework

## The Principle

Every problem in an LLM inference system has a precise origin. The L0–L6 framework provides a layered model to quickly locate where the problem lies, who owns it and which component needs attention.

**Quick rule:** who decides sits on top (L4a/L4b), who computes sits at L3, who makes it possible sits at L2/L1/L0.

---

## The Layers

**L6 — Client / Entry Point** GUI, CLI, SDK, API client, agent. Collects input, manages client-side session, renders output. The data entry point.

**L5 — API Gateway / Edge** Reverse proxy, ingress, authN/Z, rate limiting, routing, TLS termination. Datacenter perimeter. CPU-only. Never touches the GPU.

**L4a — Request Orchestration** Queueing, prioritization, admission control, multi-tenant policy, load shedding, backpressure. Transforms variable and unpredictable traffic into a governed flow before it touches the GPU.

**L4b — GPU Workload Optimization** Continuous batching, chunked prefill, token scheduling, KV cache lifecycle (alloc/evict/reuse), prefix caching, speculative decoding orchestration, GPU assignment. Transforms governed requests into efficient GPU workload.

**L3 — Inference Backend / Execution Engine** PyTorch, Transformers, TensorRT-LLM, attention/MLA kernels. Executes the forward pass, consumes KV cache during attention, produces logits and tokens. This is where the math happens.

**L2 — Runtime & Packaging** Container runtime, nvidia-container-runtime, device mapping, driver/library injection, cgroups/namespace. Prepares the process environment before any computation starts.

**L1 — Driver & GPU Runtime** NVIDIA kernel driver, CUDA driver API, NCCL, NVML/DCGM. Manages GPU memory, kernel launch, low-level scheduling, GPU↔GPU communication and telemetry.

**L0 — Hardware** GPU, VRAM, PCIe, NVLink, InfiniBand/RoCE, CPU, RAM, storage, power, cooling. The physical constraints that determine the ceiling of the entire system.

---

## How to Use It

When a problem manifests, start from the highest visible layer and descend to the root cause.

| Symptom | Probable Layer |
|---|---|
| High first-token latency (TTFT) | L4a, L4b |
| Low throughput, GPU underutilized | L4b |
| Queue growing without control | L4a |
| Slow forward pass, inefficient compute | L3 |
| CUDA errors, OOM, driver crash | L1, L2 |
| Bandwidth saturated, VRAM exhausted | L0 |
