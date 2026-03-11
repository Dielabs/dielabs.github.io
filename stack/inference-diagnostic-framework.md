# Inference Diagnostic Framework — L0–L6

A structured approach for root-cause analysis in LLM inference systems.

## Principle

Every problem in an LLM inference system has a precise origin. The L0–L6 framework provides a layered model for quickly locating where a problem lives, who owns it, and which component needs to be touched.

**Quick rule:** who decides sits on top (L4a/L4b), who computes sits at L3, who makes it possible sits at L2/L1/L0.

---

## Layers

**L6 — Client / Entry Point**
GUI, CLI, SDK, API client, agent. Collects input, manages client-side session, renders output. The entry point of the data.

**L5 — API Gateway / Edge**
Reverse proxy, ingress, authN/Z, rate limiting, routing, TLS termination. Datacenter perimeter. CPU-only. Never touches the GPU.

**L4a — Request Orchestration**
Queueing, prioritization, admission control, multi-tenant policy, load shedding, backpressure. Transforms variable and unpredictable traffic into a governed flow before it reaches the GPU.

**L4b — GPU Workload Optimization**
Continuous batching, chunked prefill, token scheduling, KV cache lifecycle (alloc/evict/reuse), prefix caching, speculative decoding orchestration, GPU assignment. Transforms governed requests into efficient GPU workload.

**L3 — Inference Backend / Execution Engine**
PyTorch, Transformers, TensorRT-LLM, attention/MLA kernels. Runs the forward pass, consumes the KV cache during attention, produces logits and tokens. This is where the math happens.

**L2 — Runtime & Packaging**
Container runtime, nvidia-container-runtime, device mapping, driver/library injection, cgroups/namespaces. Prepares the process environment before any computation starts.

**L1 — Driver & GPU Runtime**
NVIDIA kernel driver, CUDA driver API, NCCL, NVML/DCGM. Manages GPU memory, kernel launch, low-level scheduling, GPU↔GPU communication and telemetry.

**L0 — Hardware**
GPU, VRAM, PCIe, NVLink, InfiniBand/RoCE, CPU, RAM, storage, power, cooling. The physical constraints that determine the ceiling of the entire system.

---

## How to use it

When a problem surfaces, start from the highest visible layer and descend to root cause.

| Symptom | Probable layer |
|---|---|
| High first-token latency (TTFT) | L4a, L4b |
| Low throughput, underutilized GPU | L4b |
| Queue growing uncontrolled | L4a |
| Slow forward pass, inefficient compute | L3 |
| CUDA errors, OOM, driver crash | L1, L2 |
| Saturated bandwidth, exhausted VRAM | L0 |

---

*Part of the Dielabs framework. See also: [Inference Stack Model](inference-stack-model.md), [Inference Mastery Model](inference-mastery-model.md)*
