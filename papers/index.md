---
title: Papers
layout: default
wide: true
---

<style>
  .paper-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; margin: 1.3rem 0 2.4rem; }
  .content .paper-grid .paper-card { display: block; text-decoration: none; padding: 1.3rem 1.4rem; border: 1px solid var(--border); border-radius: 10px; background: var(--white); }
  .content .paper-grid .paper-card:hover { border-color: var(--blue); }
  .content .paper-grid h3 { color: var(--blue); margin: 0 0 0.8rem; font-size: 0.94rem; line-height: 1.45; letter-spacing: 0.01em; text-transform: none; }
  .content .paper-grid p { margin: 0; font-size: 0.88rem; line-height: 1.7; }
  .paper-grid a:focus-visible { outline: 2px solid var(--blue); outline-offset: 4px; }
  @media (max-width: 700px) { .paper-grid { grid-template-columns: 1fr; } }
</style>

# Papers

Original technical papers by Dielabs on hardware topology, LLM inference, and model and agent evaluation. Each paper draws on hands-on lab work and benchmarks to explain the findings and what they mean for engineering decisions.

## Hardware & Interconnects

How does hardware topology affect inference performance?

<div class="paper-grid">
  <a class="paper-card" href="making-the-most-of-pcie-gpus.html" aria-labelledby="making-the-most-of-pcie-gpus">
    <h3 id="making-the-most-of-pcie-gpus">Making the Most of PCIe GPUs</h3>
    <p>On a PCIe server without NVLink, GPU-to-GPU transfers held full speed inside a NUMA domain, even with four pairs at once, and lost more than half of it across domains. Keep KV handoffs within a domain, and spread offload workers across domains.</p>
  </a>
  <a class="paper-card" href="what-cpus-teach-about-gpu-inference.html" aria-labelledby="what-cpus-teach-about-gpu-inference">
    <h3 id="what-cpus-teach-about-gpu-inference">What CPUs Teach About GPU Inference</h3>
    <p>Splitting a model across two CPU sockets made every token slower, even though the socket link is as fast as local RAM: each per-layer sync is pure overhead. One copy per socket raised throughput instead. Shard or replicate? The same choice comes up with multiple GPUs: choose the right architecture.</p>
  </a>
</div>

## Inference Performance & Diagnostics

What limits inference speed, capacity, or output quality?

<div class="paper-grid">
  <a class="paper-card" href="the-shifting-bottleneck.html" aria-labelledby="the-shifting-bottleneck">
    <h3 id="the-shifting-bottleneck">The Shifting Bottleneck</h3>
    <p>"How many users can this GPU serve?" has no answer until you name the workload. Same GPU, same model, same settings: with long prompts it stopped at ~18 concurrent streams; with chat traffic it served 32 and was still scaling. Even the limit changed: first VRAM capacity, then memory bandwidth.</p>
  </a>
  <a class="paper-card" href="when-offloading-doesnt-offload.html" aria-labelledby="when-offloading-doesnt-offload">
    <h3 id="when-offloading-doesnt-offload">When Offloading Doesn't Offload</h3>
    <p>In vLLM, KV offloading is not extra memory: the scheduler still admits only what fits in VRAM. It works as insurance: preempted requests reload from RAM instead of being recomputed. Measured effect: 23% lower p99 time between tokens, same completed requests.</p>
  </a>
  <a class="paper-card" href="the-small-slice-that-matters.html" aria-labelledby="the-small-slice-that-matters">
    <h3 id="the-small-slice-that-matters">The Small Slice That Matters</h3>
    <p>A 22 GB MoE model runs well on a 12 GB GPU. Most of the speed comes from a small slice: 3.5 GB of VRAM for attention and shared weights nearly tripled decode speed, while 7 GB more of experts added only 30%. Once the base is covered, you'd better spend VRAM on context and concurrency.</p>
  </a>
  <a class="paper-card" href="when-good-gpus-produce-bad-tokens.html" aria-labelledby="when-good-gpus-produce-bad-tokens">
    <h3 id="when-good-gpus-produce-bad-tokens">When Good GPUs Produce Bad Tokens</h3>
    <p>Grafana can't see bad tokens: with the GPU calm and KV cache at 20%, the model looped. Fixing the loop exposed a deeper limit: text degraded after ~5,000 generated tokens, far below the 32K context window. The context window doesn't tell you how long the output stays coherent.</p>
  </a>
</div>

## Model & Agent Evaluation

How reliably can a model or agent complete a task, at what cost, and in how much time?

<div class="paper-grid">
  <a class="paper-card" href="what-it-costs-to-be-right.html" aria-labelledby="what-it-costs-to-be-right">
    <h3 id="what-it-costs-to-be-right">What It Costs to Be Right</h3>
    <p>Being right costs money or time. Six cloud models solved about the same share of 50 coding exercises, at a cost per solved exercise that varied more than tenfold. A local model on a 12 GB GPU landed in the same range with thinking on, paying in time: over four minutes per exercise.</p>
  </a>
  <a class="paper-card" href="when-llm-evals-become-harness-tuning.html" aria-labelledby="when-llm-evals-become-harness-tuning">
    <h3 id="when-llm-evals-become-harness-tuning">When LLM Evals Become Harness Tuning</h3>
    <p>We set out to choose a model and ended up fixing the harness. Same three models, two rounds: from 5 successes out of 18 to 30 out of 36. Eight of the first 12 failures came from the harness and the client, not the model. In an agent, evaluate the harness before the model.</p>
  </a>
</div>

---

*All content is original Dielabs work by Diego Bardella.*
