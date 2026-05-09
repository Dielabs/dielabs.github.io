---
title: Papers
layout: default
---

# Papers

Technical papers produced from real lab work at Dielabs. Each paper documents original findings from hands-on experimentation with LLM inference systems.

---

## Documents

### [When Good GPUs Produce Bad Tokens](when-good-gpus-produce-bad-tokens.md)
Degenerative decoding loops in vLLM: diagnosis, three-plane framework, and cross-model comparison (Qwen2.5-7B vs Mistral-7B). Identifies sampling entropy collapse and model-specific coherence thresholds.

### [When Offloading Doesn't Offload](when-offloading-doesnt-offload.md)
Experimental investigation of KV cache offloading mechanisms in vLLM 0.15.1 on consumer hardware. Reveals three distinct mechanisms commonly confused under one term. Demonstrates that scheduler tuning outperforms offloading by orders of magnitude.

### [What CPUs Teach About GPU Inference](what-cpus-teach-about-gpu-inference.md)
Benchmark of SN / TP-2 / DP-2 architectures on Dell PowerEdge R730 with vLLM. Proves the CPU-GPU isomorphism: distributed inference patterns are general principles, not GPU artifacts. Identifies TP as anti-pattern on slow interconnects.

### [The Shifting Bottleneck](the-shifting-bottleneck.md)
Three benchmarks on a single RTX 4070 Super with Qwen3-8B-AWQ via vLLM. Shows how the binding constraint moves from KV capacity (FP16, prompt-heavy) to a higher KV ceiling (FP8) to memory bandwidth (FP8, chat-like). A methodology for inference sizing that starts from workload and SLO.

---

*All content is original Dielabs work by Diego Bardella.*
