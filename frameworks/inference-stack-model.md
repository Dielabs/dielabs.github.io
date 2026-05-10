---
layout: default
title: "The LLM Inference Stack Model"
---

# The LLM Inference Stack Model

## The Principle

An LLM inference stack reads like a pyramid: physical at the bottom, user at the top. Every layer does one thing and rests on the one below.

**Reading rule:** bottom-up, each layer *enables* the one above. Top-down, each layer *depends* on the one below.

---

## The Layers

### L6 — Client
GUI, CLI, SDK, agent framework. The user request enters here and the response returns here. Manages client-side session (history, context, token streaming) and renders the output. The point where the system meets the human or the application using it.

### L5 — Gateway
Reverse proxy, ingress, authentication, rate limiting, per-model or per-tenant routing, TLS termination. The datacenter perimeter: protects, dispatches and enforces access policies before the request reaches the inference engine. Runs on CPU, never touches the GPU.

### L4 — Serving Engine
vLLM, TensorRT-LLM, SGLang, TGI. The operational brain of the system. Receives requests from the gateway, queues them, batches them together (continuous batching), manages the KV cache lifecycle (allocation, eviction, prefix reuse) and decides moment by moment which sequences advance on the GPU. This is where throughput and latency are won or lost.

### L3 — Model Execution
Frameworks (PyTorch, TensorRT-LLM engine) and CUDA kernels (FlashAttention, MLA, PagedAttention). Runs the forward pass: takes the input tokens, performs the model math on the GPU and produces the logits from which the next token is sampled. The layer where the actual computation happens.

### L2.5 — Model Artifact
Model weights, numerical format (FP16, BF16, FP8, AWQ, GPTQ, GGUF), tokenizer, chat template, optional LoRA adapters. The raw material the runtime executes. Not hardware, not scheduling, not kernels: it''s the *what* gets executed. Same GPU and same engine produce different results depending on the artifact loaded.

### L2 — Container Runtime
Docker or containerd, nvidia-container-runtime, device mapping, driver and library injection. Prepares the isolated environment where the inference process runs and exposes the GPU to it. The bridge between the host operating system and the containerized application.

### L1 — Driver & CUDA
NVIDIA kernel driver, CUDA driver and runtime API, NCCL for collective communication, NVML/DCGM for telemetry. The software layer that talks directly to the GPU: allocates VRAM, launches kernels, coordinates GPU-to-GPU communication and collects hardware health signals.

### L0.5 — Interconnect & Fabric
NVLink/NVSwitch intra-node, InfiniBand HDR/NDR or RoCEv2 inter-node, switch topology. The fabric connecting GPUs to one another, within the same server and across servers. Almost invisible in single-node; in multi-node or disaggregated prefill/decode scenarios it often becomes the real bottleneck, because the KV cache has to travel between GPUs on different nodes.

### L0 — Hardware
GPU (compute + VRAM), CPU, system RAM, PCIe, NVMe storage, power, cooling. The physical constraints that determine the ceiling of the entire stack. No optimization at higher layers can exceed the limits set here.

---

## How to Read the Pyramid

**Who consumes** sits at **L5 → L6** (gateway, client). **Who decides** sits at **L4** (serving engine). **Who computes** sits at **L3** (model execution). **What makes it possible** sits at **L2.5 → L0** (artifact, runtime, driver, fabric, hardware).