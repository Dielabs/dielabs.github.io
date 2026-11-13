---
title: Papers
layout: default
wide: true
---

<style>
  .paper-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; margin: 1.3rem 0 2.4rem; }
  .paper-grid article { padding: 1.3rem 1.4rem; border: 1px solid var(--border); border-radius: 10px; background: var(--white); }
  .content .paper-grid h3 { margin: 0 0 0.8rem; font-size: 0.94rem; line-height: 1.45; letter-spacing: 0.01em; text-transform: none; }
  .content .paper-grid p { margin: 0; font-size: 0.88rem; line-height: 1.7; }
  .paper-grid a:focus-visible { outline: 2px solid var(--blue); outline-offset: 4px; }
  @media (max-width: 700px) { .paper-grid { grid-template-columns: 1fr; } }
</style>

# Papers

Original technical papers by Dielabs on hardware topology, LLM inference, and model and agent evaluation. Each paper draws on hands-on lab work and benchmarks to explain the findings and what they mean for engineering decisions.

## Hardware & Interconnects

How does hardware topology affect inference performance?

<div class="paper-grid">
  <article>
    <h3><a href="making-the-most-of-pcie-gpus.html">Making the Most of PCIe GPUs</a></h3>
    <p>On an eight-GPU PCIe server, bandwidth tests reveal which GPUs share fast links and where simultaneous transfers compete for bandwidth.</p>
  </article>
  <article>
    <h3><a href="what-cpus-teach-about-gpu-inference.html">What CPUs Teach About GPU Inference</a></h3>
    <p>On a Dell PowerEdge R730, the cost of synchronizing across a slow link explains why splitting a model across two workers can hurt inference performance.</p>
  </article>
</div>

## Inference Performance & Diagnostics

What limits inference speed, capacity, or output quality?

<div class="paper-grid">
  <article>
    <h3><a href="the-shifting-bottleneck.html">The Shifting Bottleneck</a></h3>
    <p>On a 12 GB RTX 4070 Super, changing the KV cache format and workload shifts the limiting factor from cache capacity to memory bandwidth.</p>
  </article>
  <article>
    <h3><a href="when-offloading-doesnt-offload.html">When Offloading Doesn't Offload</a></h3>
    <p>Tests of three vLLM offloading mechanisms on consumer hardware found that tuning the scheduler delivered the largest gain.</p>
  </article>
  <article>
    <h3><a href="the-small-slice-that-matters.html">The Small Slice That Matters</a></h3>
    <p>For a 35B mixture-of-experts model on a 12 GB GPU, the first 3.5 GB of VRAM allocated to experts delivered a larger gain than the next 7 GB.</p>
  </article>
  <article>
    <h3><a href="when-good-gpus-produce-bad-tokens.html">When Good GPUs Produce Bad Tokens</a></h3>
    <p>A comparison of Qwen and Mistral in vLLM traces repetitive output to a collapse in sampling entropy and shows how the threshold for coherent output varies by model.</p>
  </article>
</div>

## Model & Agent Evaluation

How reliably can a model or agent complete a task, at what cost, and in how much time?

<div class="paper-grid">
  <article>
    <h3><a href="what-it-costs-to-be-right.html">What It Costs to Be Right</a></h3>
    <p>Across 50 coding exercises, enabling reasoning on local Qwen3.6 raised its score from 22 to 35 solves while average time per exercise rose from 72 to 261 seconds.</p>
  </article>
  <article>
    <h3><a href="when-llm-evals-become-harness-tuning.html">When LLM Evals Become Harness Tuning</a></h3>
    <p>Success on an Excel workflow rose from 5 of 18 runs to 30 of 36 after changes to the agent harness, with the same models under test.</p>
  </article>
</div>

---

*All content is original Dielabs work by Diego Bardella.*
