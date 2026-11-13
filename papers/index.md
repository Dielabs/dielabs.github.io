---
title: Papers
layout: default
wide: true
---

<style>
  .content > h3 { margin-top: 3rem; }
  .content .paper-featured { border: 1.5px solid rgba(0,87,255,0.18); border-left: 4px solid var(--blue); background: linear-gradient(120deg, #F3F7FF 0%, #FFFFFF 70%); border-radius: 12px; padding: 0.2rem 1.4rem 0.4rem; margin: 2.5rem 0; }
  .content .paper-featured h3 { margin-top: 1rem; }
  .content .paper-featured p { margin-bottom: 0.8rem; }
</style>

# Papers

Technical papers produced from real lab work at Dielabs. Each paper documents original findings from hands-on experimentation with LLM inference systems.

---

### [When Good GPUs Produce Bad Tokens](when-good-gpus-produce-bad-tokens.md)
Degenerative decoding loops in vLLM: diagnosis, three-plane framework, and cross-model comparison (Qwen2.5-7B vs Mistral-7B). Identifies sampling entropy collapse and model-specific coherence thresholds.

### [When Offloading Doesn't Offload](when-offloading-doesnt-offload.md)
Experimental investigation of KV cache offloading mechanisms in vLLM 0.15.1 on consumer hardware. Reveals three distinct mechanisms commonly confused under one term. On this setup, tuning the base scheduler parameters gave a much larger gain than any offloading mechanism.

### [What CPUs Teach About GPU Inference](what-cpus-teach-about-gpu-inference.md)
Benchmark of SN / TP-2 / DP-2 architectures on Dell PowerEdge R730 with vLLM. Shows the same single-node, TP and DP patterns seen on GPUs, suggesting they are general principles rather than GPU artifacts. Identifies TP as anti-pattern on slow interconnects.

### [The Shifting Bottleneck](the-shifting-bottleneck.md)
Three benchmarks on a single RTX 4070 Super with Qwen3-8B-AWQ via vLLM. Shows how the binding constraint moves from KV capacity (FP16, prompt-heavy) to a higher KV ceiling (FP8) to memory bandwidth (FP8, chat-like). A methodology for inference sizing that starts from workload and SLO.

<div class="paper-featured" markdown="1">

### [The Small Slice That Matters](the-small-slice-that-matters.md)
Local AI on a consumer GPU: a 35B MoE model (Qwen3.6-35B-A3B) served by llama.cpp on a 12 GB RTX 4070 Super, with experts offloaded to RAM. Shows that the first 3.5 GB of VRAM are worth more than the next 7, that prefill is bound by PCIe and decode by GPU-CPU handovers, and how to spend VRAM on the metric you care about.

</div>

<div class="paper-featured" markdown="1">

### [What It Costs to Be Right](what-it-costs-to-be-right.md) <span class="new-flag">New</span>
Coding Benchmark @Dielabs: 17 LLM configurations on 50 Aider Polyglot exercises, local (llama.cpp on a 12 GB GPU) and remote (pinned providers via OpenRouter). Quality, cost and time per solved exercise; what model size, reasoning, sampling and quantization really change, and how to choose a model for code.

</div>

<div class="paper-featured" markdown="1">

### [Making the Most of PCIe GPUs](making-the-most-of-pcie-gpus.md) <span class="new-flag">New</span>
Measured best practices for an eight-GPU, air-cooled PCIe cloud server: discover the effective topology, test contention, place communicating workers within fast domains, and qualify RAM paths and whole-node thermals before the LLM benchmark.

</div>

<div class="paper-featured" markdown="1">

### [When LLM Evals Become Harness Evals](when-llm-evals-become-harness-evals.md) <span class="new-flag">New</span>
End-to-end eval of PMI AI Excel, the Dielabs assistant that reads, analyses and edits Excel files inside OpenWebUI, with GLM 5.3, Kimi K3.1 Pro and DeepSeek V4 Pro on a real administrative workbook. From 5 successes out of 18 to 30 out of 36 without changing the model: the eval ended up testing the harness more than the models. Why an agent's cost follows its turns, why the parser accepts what Excel refuses, and how the model was chosen last.

</div>

---

*All content is original Dielabs work by Diego Bardella.*
