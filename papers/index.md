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
    <p>How much does GPU placement affect PCIe transfer bandwidth? On an eight-GPU server, the VM exposed a single NUMA node, yet measurements revealed two distinct GPU groups. With four pairs transferring at once, each pair sustained about 53 GB/s within a group and 21–25 GB/s across groups. <strong>With PCIe, choose your GPU pairs carefully. Placement can more than double your transfer bandwidth.</strong></p>
  </article>
  <article>
    <h3><a href="what-cpus-teach-about-gpu-inference.html">What CPUs Teach About GPU Inference</a></h3>
    <p><strong>When can model sharding slow down inference?</strong> Even with a single request, tensor parallelism across two CPU sockets increased median time between tokens from 255 ms to 403 ms. With four concurrent requests, the gap widened to 292 ms on one socket versus 467 ms on two. Choose a sharding technique that fits both the workload and the interconnect.</p>
  </article>
</div>

## Inference Performance & Diagnostics

What limits inference speed, capacity, or output quality?

<div class="paper-grid">
  <article>
    <h3><a href="the-shifting-bottleneck.html">The Shifting Bottleneck</a></h3>
    <p>Load tests on a 12 GB GPU showed how the bottleneck shifted from KV cache capacity to memory bandwidth. Switching to an FP8 KV cache and moving from long prompts to a chat workload brought total output throughput from about 255 to nearly 1,300 tokens per second on the same GPU.</p>
  </article>
  <article>
    <h3><a href="when-offloading-doesnt-offload.html">When Offloading Doesn't Offload</a></h3>
    <p>In one of my first vLLM studies, I tested what happens when part of the KV cache moves from GPU memory to system RAM. Offloading improved tail latency but added no completed requests. Tuning vLLM’s basic settings alone raised the count from 63 to 103.</p>
  </article>
  <article>
    <h3><a href="the-small-slice-that-matters.html">The Small Slice That Matters</a></h3>
    <p><strong>All experts in system RAM can be enough.</strong> We tested three offload configurations in llama.cpp for a 35B MoE model. With just 3.5 GB of VRAM in use, the model generated 50 tokens/s. Moving some experts onto the GPU raised that to 65 tokens/s but used 10.6 GB of VRAM.</p>
  </article>
  <article>
    <h3><a href="when-good-gpus-produce-bad-tokens.html">When Good GPUs Produce Bad Tokens</a></h3>
    <p>A sampling stress test in vLLM showed how long two 7B models could sustain a numbered narrative in Italian. With the same decoding settings, Qwen stayed coherent for about 5,000 tokens. Mistral broke down after 200–300, yet produced a coherent technical explanation in Italian. Test each model on the task, language and output length you expect to serve.</p>
  </article>
</div>

## Model & Agent Evaluation

How reliably can a model or agent complete a task, at what cost, and in how much time?

<div class="paper-grid">
  <article>
    <h3><a href="what-it-costs-to-be-right.html">What It Costs to Be Right</a></h3>
    <p>Does thinking mode help with code? On the same 50 coding exercises, enabling it took Qwen3.6-35B from 22 solves to 35. That made this general-purpose model my pick for a local setup with 12 GB of VRAM and experts offloaded to RAM. It solved more exercises than either dedicated coding model I tested locally.</p>
  </article>
  <article>
    <h3><a href="when-llm-evals-become-harness-tuning.html">When LLM Evals Become Harness Tuning</a></h3>
    <p>I set out to compare three frontier open-weight models on a real Excel workbook and ended up refining the agent harness around them. After those fixes, each model succeeded in 10 of 12 runs. I chose GLM 5.3 for reliability because none of its failures in these tests were attributable to the model itself.</p>
  </article>
</div>

---

*All content is original Dielabs work by Diego Bardella.*
