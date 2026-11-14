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
    <p>What does it cost to move data across NUMA domains on a PCIe server? With no NVLink, pairs within a domain held ~53 GB/s even with four transferring at once, about 1/17 of NVLink 5 on paper. Across domains, they fell to 21–25 GB/s. Host RAM showed the same pattern. Keep KV handoffs within a domain, and spread offload workers across domains.</p>
  </article>
  <article>
    <h3><a href="what-cpus-teach-about-gpu-inference.html">What CPUs Teach About GPU Inference</a></h3>
    <p>Splitting a 4B model across two CPU sockets made every token slower (255 → 403 ms) and cut throughput by up to 79%. The socket link is about as fast as local RAM, so each per-layer sync is pure overhead. One copy per socket worked instead: +68% throughput under heavy load. Every workload needs its own runtime architecture, matched to the hardware it runs on.</p>
  </article>
</div>

## Inference Performance & Diagnostics

What limits inference speed, capacity, or output quality?

<div class="paper-grid">
  <article>
    <h3><a href="the-shifting-bottleneck.html">The Shifting Bottleneck</a></h3>
    <p>Same 12 GB GPU, same model, two workloads. With long prompts (RAG-style), the KV cache filled up at ~10 streams and output stalled at ~255 tok/s. With chat traffic and an FP8 KV cache, output reached ~1,300 tok/s, and memory bandwidth became the limit. Prompt-heavy workloads need VRAM capacity; chat workloads need memory bandwidth. Choose the hardware for the workload.</p>
  </article>
  <article>
    <h3><a href="when-offloading-doesnt-offload.html">When Offloading Doesn't Offload</a></h3>
    <p>KV offloading in vLLM doesn't add capacity: the scheduler still admits only what fits in VRAM. It works as insurance, reloading preempted requests from RAM instead of recomputing them, which cut p99 time between tokens by 23%. Completed requests didn't change. Tuning three basic settings did: from 63 to 103.</p>
  </article>
  <article>
    <h3><a href="the-small-slice-that-matters.html">The Small Slice That Matters</a></h3>
    <p>Running a 35B MoE on 12 GB, the first 3.5 GB of VRAM did most of the work: decode went from 18 to 50 tok/s. Adding 7 GB of experts gave only 30% more, because the layers left on the CPU still set the pace. Keeping all experts in RAM was the best trade: most of the speed, with ~8 GB of VRAM left free for KV cache, longer context or more users.</p>
  </article>
  <article>
    <h3><a href="when-good-gpus-produce-bad-tokens.html">When Good GPUs Produce Bad Tokens</a></h3>
    <p>Grafana can't see bad tokens. With a calm GPU and KV cache at 20%, Qwen 7B looped in Italian until the repetition penalty was raised. Then it held for ~5,000 tokens, well below its 32K context. Mistral 7B broke after 200–300 tokens with the same settings. Fixing one problem can expose the next one.</p>
  </article>
</div>

## Model & Agent Evaluation

How reliably can a model or agent complete a task, at what cost, and in how much time?

<div class="paper-grid">
  <article>
    <h3><a href="what-it-costs-to-be-right.html">What It Costs to Be Right</a></h3>
    <p>Across 17 configurations on 50 coding exercises, six cheap models landed at 62–70%, with cost per solved exercise varying more than tenfold. Only Opus 5.5 stood apart, with 49 of 50. On a 12 GB local GPU, thinking took Qwen3.6 from 22 to 35 solved, at 3.6× the time per exercise. Being right costs money or time: compare models by cost per solved task, not price per token.</p>
  </article>
  <article>
    <h3><a href="when-llm-evals-become-harness-tuning.html">When LLM Evals Become Harness Tuning</a></h3>
    <p>Same three models, two rounds: 5 successes out of 18, then 30 out of 36. The models didn't change: 8 of the first 12 failures came from the harness and the client. Once those were fixed, all three succeeded in 10 of 12 runs with no calculation errors, and the choice came down to cost and error severity. In an agent, evaluate the harness before the model.</p>
  </article>
</div>

---

*All content is original Dielabs work by Diego Bardella.*
