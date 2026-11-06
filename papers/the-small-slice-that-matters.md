---
layout: default
title: "The Small Slice That Matters"
description: "MoE expert offloading in llama.cpp on a 12 GB GPU: a 35B model on a consumer card, and what the VRAM actually buys."
---

<style>
  .content .tldr { border: 1.5px solid rgba(0,87,255,0.18); border-left: 4px solid var(--blue); background: linear-gradient(120deg, #F3F7FF 0%, #FFFFFF 70%); border-radius: 12px; padding: 1.4rem 1.6rem 0.6rem; margin: 2rem 0; }
  .content .tldr h2 { margin-top: 0; border-bottom: none; }
</style>

# The Small Slice That Matters

## MoE expert offloading in llama.cpp on a 12 GB GPU

**Lab:** Dielabs **Hardware:** NVIDIA RTX 4070 Super (12 GB GDDR6X), Intel Core i7-14700F, 32 GB DDR5-5600 **Model:** Qwen3.6-35B-A3B, Unsloth UD-Q4_K_XL GGUF (22.4 GB) **Stack:** llama.cpp b12063 on k3s, custom measuring proxy, Prometheus + Grafana, DCGM, Intel RAPL **Date:** November 2026

<div class="tldr" markdown="1">

## In short

A 35B Mixture-of-Experts model can run on a 12 GB consumer GPU with part of its weights in system RAM. We tested three placements of the same model: all CPU, all experts in RAM with 3.5 GB of VRAM used, and experts split between RAM and VRAM with 10.6 GB used. The main findings:

- **The first 3.5 GB of VRAM deliver the biggest gain.** Keeping attention, KV cache and shared weights on the GPU, while leaving all experts in RAM, raised short-prompt decode from 18 to 50 tok/s and cut long-prompt time to first token from 48.2 to 10.34 seconds.
- **Another 7.1 GB helps, with smaller returns.** Moving more experts into VRAM raised short-prompt decode from 50 to 65 tok/s and reduced long-prompt time to first token to 7.38 seconds, about 29% less.
- **Prefill and decode need different explanations.** During prefill, experts held in RAM are copied to the GPU; keeping more in VRAM avoids traffic. During decode, little data crosses PCIe. The measurements suggest that CPU work and GPU–CPU handovers limit speed, although that diagnosis still needs a controlled test.
- **Spend VRAM on the objective.** Once the GPU base is covered, experts in VRAM help long-prompt response time; context, concurrency and model precision are other possible uses to evaluate. Estimated energy per generated token fell from about 3.8 to 3.1 J between the two GPU configurations.

These are short, single-user tests on one model and one machine, with thinking disabled. The CPU-only baseline also suffered memory pressure and disk reads; power and transfer-volume figures include estimates. The results do not establish multi-user capacity.

</div>

---

## 1. The question

The model file is 22.4 GB. The GPU has 12 GB. The usual answer is "it doesn't fit, buy a bigger card". With a Mixture-of-Experts model and llama.cpp, the real answer is different: it runs, and it runs well. The interesting question is what the VRAM is actually buying you.

This paper documents three runs of the same model on the same machine. Only one thing changes across them: **where the weights live**. Nothing on the GPU, the experts all in system RAM, or the experts split between VRAM and RAM.

The intuitive expectation is that moving more experts into VRAM should make everything proportionally faster. It doesn't. The first few gigabytes of VRAM are worth far more than the rest, and each user-facing metric is limited by a different part of the system.

---

## 2. The lab setup

| Component | Detail |
| --- | --- |
| GPU | RTX 4070 Super 12 GB, PCIe 4.0 x16 |
| CPU | Intel i7-14700F (8 P-cores + 12 E-cores, 28 threads) |
| RAM | 32 GB DDR5-5600, dual channel (~90 GB/s theoretical) |
| Model | Qwen3.6-35B-A3B, MoE: 35B total, ~3B active per token, 40 layers, 256 experts per layer, 8 active |
| Serving | llama.cpp b12063, 32k context, 4 slots, flash attention |
| Measurement | Python proxy in front of llama-server (response times), Prometheus scraped every 5 s, DCGM (NVIDIA's GPU metrics exporter), Intel RAPL (the CPU's built-in power counters), nvidia-smi for PCIe traffic |

The model is a hybrid architecture: only 10 of the 40 layers use classic attention with a KV cache, the others use a lighter mechanism with a fixed-size memory instead of a cache that grows with the text. The KV cache is therefore light, about 20 KB per token, roughly 0.6 GB for the full 32k context. It plays no role in the results below.

### The three configurations

| Scenario | llama.cpp flags | VRAM used |
| --- | --- | --- |
| S0, everything on CPU | `--device none --no-op-offload` | 0 GB |
| S2, all experts in RAM | `--fit on --cpu-moe` | 3.5 GB |
| S1, experts split VRAM and RAM | `--fit on` | 10.6 GB |

In S2 the GPU still holds attention, the KV cache and the shared (non-expert) weights, and it still computes the prefill. Only the experts live in RAM. S0 is the only configuration where the GPU does nothing at all.

### The test

One user, streaming responses, thinking disabled. Two prompt shapes: a short prompt of ~140 tokens (up to 300 output tokens) and a long prompt of ~5,500 tokens (up to 150 output tokens). Five requests per shape in S1 and S2, three in S0, which is much slower. Every prompt carries a random prefix, so the prompt cache is never reused. All traffic goes through the measuring proxy, and each run is marked in Grafana so hardware metrics can be cut by run.

Two numbers describe what the user feels:

- **TTFT** (time to first token): the wait before the answer starts to appear. It is mostly the **prefill**, the phase where the model reads the whole prompt.
- **Decode speed**: tokens per second while the answer streams. Its inverse is the **ITL** (inter-token latency), the time between two tokens.

---

## 3. The results

### User experience

Medians, measured by the client and confirmed by the proxy.

| Metric | S0, all CPU (0 GB) | S2, experts in RAM (3.5 GB) | S1, half in VRAM (10.6 GB) |
| --- | --- | --- | --- |
| TTFT, short prompt | 1.43 s | 1.00 s | 0.62 s |
| TTFT, long prompt (5,462 tokens) | 48.2 s | 10.34 s | 7.38 s |
| Prefill speed, long prompt | ~113 tok/s | ~530 tok/s | ~740 tok/s |
| Decode speed, short prompt | 18 tok/s | 50 tok/s | 65 tok/s |
| Decode speed, long prompt | 15 tok/s | 50 tok/s | 64 tok/s |
| ITL p50 / p95 | 62.5 / 74.0 ms | 20.3 / 24.5 ms | 15.8 / 19.6 ms |

With attention on the GPU, decode speed does not depend on context length (65 vs 64 tok/s). On CPU only it drops from 18 to 15 tok/s, because reading the long context at every token becomes CPU work too.

### What the hardware does, per phase

For S1 and S2 I split each run into prefill and decode using PCIe traffic as the phase marker: traffic from system RAM to the GPU above 1 GB/s means prefill. Averages over the samples in each phase.

| Metric | Prefill S1 | Prefill S2 | Decode S1 | Decode S2 |
| --- | --- | --- | --- | --- |
| PCIe RAM to GPU | 14.8 GB/s | 14.4 GB/s | 0.1 GB/s | 0.2 GB/s |
| PCIe GPU to RAM | 1.0 GB/s | 1.2 GB/s | 0.3 GB/s | 0.1 GB/s |
| GPU utilization | 77% | 71% | 48% | 53% |
| GPU memory bandwidth used | 22% | 8% | 33% | 6% |
| CPU utilization (28 threads) | 7% | 8% | 9% | 9% |
| GPU power | 85 W | 69 W | 67 W | 50 W |
| CPU package power | 66 W | 70 W | 81 W | 87 W |

In both S1 and S2 the page cache stayed flat at 24.1 GB (the model file), disk reads stayed at zero, swap did not grow, and the GPU never went above 57 °C.

A direct look with `top` during an S1 decode at 67 tok/s showed **8 llama.cpp compute threads at ~35% each**, plus one main thread at 93% driving the GPU.

### What happened in S0

With no PCIe traffic there is no phase marker, so S0 numbers cover the whole run (234 s):

- GPU idle: 0% utilization, 4 W, no PCIe traffic.
- CPU at ~24% of all threads (about 7 cores fully busy), drawing **140 to 190 W**, twice the other scenarios.
- **Memory pressure.** At load time process memory jumped from 4.4 to 17.5 GB and the page cache dropped from 24 to 16 GB. My reading is that llama.cpp, without a GPU, builds a CPU-optimized copy of the weights on top of the file mapped in page cache. Part of the file got evicted and re-read from a SATA SSD: 70 to 170 MB/s at load, then 1 to 20 MB/s during the test.

S0 is therefore slightly penalized by disk reads. Its numbers are an order of magnitude, not a precise measurement. The gap with the other scenarios is far larger than anything the disk can explain.

---

## 4. What the VRAM actually buys

Two steps, two very different returns.

| Step | VRAM added | Decode speed | TTFT, long prompt |
| --- | --- | --- | --- |
| From all CPU to the GPU base (S0 to S2) | +3.5 GB | 18 to 50 tok/s (×2.8) | 48 to 10 s (÷4.7) |
| From the base to experts in VRAM (S2 to S1) | +7.1 GB | 50 to 65 tok/s (+30%) | 10 to 7 s (−29%) |

The first 3.5 GB, which hold attention, the KV cache and the shared weights and let the GPU run the prefill, are worth far more than the 7.1 GB of experts added afterwards.

The reason is architectural. A GPU is built for operations that touch a lot of data at once: the prefill (thousands of tokens in one pass) and attention over the context. Experts in decode are a small amount of work per token, 8 out of 256 per layer, and a CPU can carry them. Removing the GPU entirely moves exactly the heavy parts onto the CPU: prefill falls from ~530 to ~113 tok/s.

Moving experts into VRAM always helps, but each metric is decided by a different resource:

| Metric | Effect of experts in VRAM | Resource that decides it |
| --- | --- | --- |
| Decode speed | +30% | the layers left on the CPU, plus the GPU-CPU handovers |
| TTFT on long prompts | −29% | volume of data to move over PCIe |
| Energy per token | −19% | mostly fixed power, divided over more tokens |

---

## 5. How the serving stack works with offload

### Where weights live and where math runs

llama.cpp takes two separate decisions, and mixing them up is the easiest way to misread this lab.

**Where the weights live is decided once, at load time.** The model is made of tensors, the blocks of numbers that hold its weights. When the server starts, each tensor is placed either in GPU memory or in system RAM. In llama.cpp terms, each tensor goes into the memory of a *backend*, the device that runs the math: the CUDA backend for the GPU, the CPU backend for the processor. Two flags decide the split:

- `--fit on` and `-ngl` decide how many layers go to the GPU;
- `--cpu-moe` keeps all the experts in RAM (it is a shortcut for an `--override-tensor` rule that matches the expert tensors).

The split is per layer, not per expert: a layer has all 256 experts in VRAM, or all of them in RAM. Attention, KV cache and shared weights always stay on the GPU. The weights in RAM are not an extra copy: they are the GGUF file mapped into memory (`mmap`) and served from the page cache. This placement never changes while the server runs. In particular, llama.cpp does not move the most-used experts into VRAM: the idea is discussed in the project, but it is not implemented.

**Where the math runs is decided operation by operation.** For every batch of tokens, llama.cpp builds the list of operations to perform (the *compute graph*), and a component called the *backend scheduler* decides which device runs each one. Normally an operation runs where its weights are. The exception is called **op offload**, and it is on by default (`--op-offload`): if an operation uses weights stored in RAM but has at least 32 tokens to process, the GPU runs it anyway. The weights are copied from RAM to the GPU over PCIe, used for that batch, and thrown away. The 32-token threshold is fixed in the code of the GPU backends.

What this means for each phase:

- **Prefill.** The prompt is processed in chunks called micro-batches, 512 tokens by default (`-ub`). Every chunk is well above 32 tokens, so for every chunk the experts stored in RAM are copied to the GPU again. That is where the ~100 to 145 GB of PCIe traffic for one long prompt comes from (estimated, see below).
- **Decode.** One token at a time, far below 32. The experts stored in RAM are computed by the CPU right where they are, and only the intermediate results passed from layer to layer (the *activations*) cross PCIe.
- **S0.** `--device none` puts no weights on the GPU, and `--no-op-offload` switches the mechanism off: everything, prefill included, runs on the CPU.

> **Box: The shelf and the photocopies**
>
> Think of the experts as books. Where each book is shelved is decided once, at load time, and never changes: a few shelves are in VRAM, most are in RAM. For a big meeting (prefill), llama.cpp photocopies the books from the RAM shelves, carries the copies to the GPU, uses them and throws them away; at the next meeting it photocopies them again. For a quick question (decode), it reads the book where it sits. The shelves never move; only the reading room changes.

Sources: the 32-token threshold and its discussion, [llama.cpp issue #18530](https://github.com/ggml-org/llama.cpp/issues/18530); proposals for a VRAM cache of frequently used experts, [discussion #24528](https://github.com/ggml-org/llama.cpp/discussions/24528) and [issue #20757](https://github.com/ggml-org/llama.cpp/issues/20757).

### Prefill is a transfer problem

Because of op offload, the GPU does the prefill math in both S1 and S2. What changes is how much has to be copied first.

- PCIe runs at 14 to 15 GB/s in both S1 and S2, which is the practical ceiling of a PCIe 4.0 x16 link. Over a long prompt that is roughly 100 GB moved in S1 and 145 GB in S2 (average bandwidth times duration): the same experts are copied again for every chunk.
- The CPU stays almost idle (7 to 8%).
- With more experts in RAM there is more to copy over the same link, so prefill slows down roughly in proportion: +40% time for about +50% more expert bytes in RAM.

**A check on the numbers.** A careful reader will do the count: 5,462 tokens make 11 micro-batches, and 11 copies of ~13 GB (S1) and ~20 GB (S2) would give ~143 and ~220 GB. The measured traffic is ~100 and ~145 GB, about 0.7 of that in both scenarios. The naive count would not even fit in the measured TTFT: at 15 GB/s, 143 GB takes ~9.5 s against 7.4 s. So the mechanism holds, since traffic grows with the expert bytes held in RAM, but the exact bytes per copy are not reconciled. Expert sizes and PCIe traffic are both approximate (see §6 and §9): treat these figures as estimates.

> **Box: In prefill, VRAM is a transfer cache**
>
> It is tempting to read the S1 vs S2 prefill gap as "the GPU is faster than the CPU". That is not what happens. In both scenarios the GPU does the prefill math. The difference is how many bytes have to cross PCIe first: experts already in VRAM don't need to be copied. In this phase VRAM works as a cache that avoids traffic, not as faster compute.

### Decode is a serial path

Decode produces one token at a time. Each token crosses the 40 layers in order, and no layer can start before the previous one finishes. For each layer the GPU computes attention, then the 8 selected experts are computed where they live: on the GPU if the layer is in VRAM, on the CPU if it is in RAM. Only activations cross PCIe, 0.1 to 0.3 GB/s.

The time per token is the sum of the time of each layer, plus a synchronization cost at every handover between GPU and CPU. Moving layers to VRAM shortens only those layers. The layers left on the CPU, and the handovers, stay on the critical path, the chain of steps that sets the total time.

> **Box: The relay race**
>
> Think of each token as a 40-leg relay. The GPU runs its legs very fast, the CPU runs its legs slower, and every change of runner costs time. Make half the legs very fast and the total time still drops only a little, because the slow legs and the baton changes are still there. The real jump only comes when every leg runs on the GPU: no CPU, no handovers. On this card, with this model, that is not possible.

The decode bottleneck is not where one would first look:

- **Not RAM bandwidth.** S1 reads roughly 0.4 GB of experts from RAM per token, S2 roughly 0.6 GB (estimates, see §6). At the measured speeds that is ~27 and ~32 GB/s, about a third of the ~90 GB/s theoretical. If bandwidth were the limit, speed would drop about as much as the bytes grow (+50%). It dropped 24%.
- **Not CPU compute.** The 8 compute threads sit at ~35% each.
- **Not PCIe.** It carries about 1% of its capacity during decode.

What remains is latency: GPU and CPU waiting for each other, dozens of times per token. The GPU shows 48 to 53% "utilization" with little memory bandwidth used, which fits a device working in short bursts and waiting in between. This is a hypothesis supported by consistent evidence, not a proof (see §9).

---

## 6. Why the MoE runs on so little VRAM

In each layer a small gating network, the router, picks 8 experts out of 256 for every token. The other 248 are not even read. Of the ~35B parameters, about 3B work for each token: attention and shared weights, always on the GPU, plus 1/32 of the experts. Experts are not topic specialists (code, poetry): the selection changes token by token and the specialization is weak.

| Estimate | S1 | S2 |
| --- | --- | --- |
| Experts held in RAM | ~13 GB | ~20 GB |
| Read from RAM per token (1/32) | ~0.4 GB | ~0.6 GB |
| RAM bandwidth used in decode | ~27 GB/s | ~32 GB/s |
| Share of theoretical bandwidth (~90 GB/s) | ~30% | ~35% |

The amount of experts in RAM comes from the VRAM difference between S1 and S2 (7.1 GB of experts moved) and from the size of the model file.

The comparison that explains everything, estimated and not measured: a dense model of the same size reads all its weights at every token, ~20 GB instead of ~0.6 GB. Even at the full theoretical RAM bandwidth it would stay below 5 tok/s on this machine. That ~30× gap in bytes per token is what lets the MoE run on a consumer GPU.

---

## 7. Energy

| Metric | S0, all CPU | S2, experts in RAM | S1, half in VRAM |
| --- | --- | --- | --- |
| Wall power during the test (estimate) | ~230 W | ~190 W | ~202 W |
| Energy per generated token | ~14 J | ~3.8 J | ~3.1 J |
| Cost per million generated tokens | ~0.95 € | ~0.26 € | ~0.21 € |
| Energy per prompt token | ~2.0 J | ~0.36 J | ~0.28 J |

Wall power = GPU (DCGM) + CPU package (RAPL) + an estimated 30 W for the rest of the platform, divided by an estimated 0.88 PSU efficiency. Electricity at 0.247 €/kWh, the variable component of my residential contract (Plenitude, VAT included, valid until 30/11/2026). The idle node draws about 77 W at the wall. S0 power is averaged over the whole run, without a prefill/decode split.

Between S1 and S2 wall power changes by only 6%, while tokens per second change by 30%. A large part of the power is drawn anyway (GPU powered while waiting for the CPU, motherboard, RAM, disks, fans, PSU losses). Produce more tokens in the same time and that fixed share is spread over more of them. On CPU only, the CPU alone goes up to 140 to 190 W while producing a third of the tokens: each token costs ~4.5× S1.

These are single-user numbers. With more concurrent users the cost per token should fall, since power grows less than token output.

---

## 8. The principles

### 8.1 The first gigabytes of VRAM are worth the most

Attention, KV cache, shared weights and the ability to run prefill on the GPU are the base. Going from zero to that base multiplied decode speed by 2.8 and cut long-prompt TTFT by 4.7. Everything above the base brought +30% and −29%. On a consumer card hosting an offloaded MoE, VRAM has steeply decreasing returns once the base is covered.

### 8.2 Every metric has its own bottleneck

Prefill is bound by the PCIe link. Decode is bound by the layers left on the CPU and by the GPU-CPU handovers. Energy per token is dominated by fixed power. A single knob, "more VRAM", moves all three, but by different amounts and for different reasons. Sizing without naming the metric is sizing blind.

### 8.3 In prefill, VRAM avoids traffic

Experts in VRAM shorten TTFT because they don't need to be copied, not because the GPU replaces the CPU. That makes long-prompt workloads (RAG, document analysis, agents with large tool outputs) the ones where experts in VRAM pay off most.

### 8.4 A split model decodes at the speed of its slow part

As long as any layer runs on the CPU, the decode speed is pinned by those layers and by the handovers. Moving more layers to VRAM helps in proportion to the share of the critical path they remove, not in proportion to the gigabytes moved. The big jump only arrives when the whole model fits in VRAM.

### 8.5 Spend VRAM on the objective

Once the base is covered, the best use of the remaining VRAM depends on what you are optimizing:

- **TTFT on long prompts:** experts in VRAM, to cut PCIe traffic.
- **More context or more users:** KV cache and slots.
- **More quality:** a higher-precision quantization, keeping the same VRAM and putting more weights in RAM.
- **Faster decode:** not more VRAM, but less CPU work on the critical path or cheaper handovers (thread placement, llama.cpp tuning), or a GPU that holds the whole model.

---

## 9. What this lab didn't test

- **Concurrency.** One user only. With several users each token picks its own experts, so the CPU has more distinct experts to compute per step. VRAM may matter more under load; a capacity sweep is the natural next run.
- **Short runs and coarse sampling.** 74 s (S1), 104 s (S2), 234 s (S0), sampled every 5 s: the per-phase averages rest on 6 to 11 samples each. PCIe traffic comes from 1-second nvidia-smi samples averaged into those 5-second scrapes, so total bytes moved per prompt are estimates (§5). The user-experience numbers are solid; the per-phase hardware numbers are indicative.
- **Synthetic, repetitive text** with thinking disabled. Real prompts and thinking enabled may change both speed and perceived waiting time.
- **Declared estimates:** expert bytes in RAM, bytes read per token, platform power and PSU efficiency.
- **The decode hypothesis.** That decode is limited by GPU-CPU handovers is supported by the thread load and bandwidth data, not proven. Thread count and core placement were not varied; the CPU has 8 P-cores and 12 E-cores, and where llama.cpp's threads land was not controlled.
- **S0 memory pressure.** As described in §3, S0 suffered page cache eviction and disk reads.
- **Dense baseline.** The dense-model comparison in §6 is an estimate. Measuring it would need a dense model of similar size.
- **Prefill micro-batch size.** Larger `-ub` values should reduce repeated expert copies over PCIe and shorten long-prompt TTFT. Not tested.

---

## 10. Reproducibility

### llama-server, S1 baseline

```yaml
args:
  - -m
  - /models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
  - -c
  - "32768"
  - --fit
  - "on"
  - -fa
  - "on"
  - --metrics
  - --slots
```

S2 adds `--cpu-moe`. S0 adds `--device none --no-op-offload`. Batch sizes left at their defaults (`-b 2048`, `-ub 512`). Image `ghcr.io/ggml-org/llama.cpp:server-cuda`, build b12063, deployed on k3s with the NVIDIA GPU Operator.

### Benchmark driver

A small Python script sends streaming chat completions through the measuring proxy, one request at a time: a warm-up, then N short prompts (~140 tokens in, up to 300 out) and N long prompts (~5,500 tokens in, up to 150 out). Each prompt starts with a random UUID so the prompt cache never hits, and thinking is disabled with `chat_template_kwargs.enable_thinking=false`. The script records TTFT and decode speed per request and reports medians. Runs are wrapped in a marker so Grafana and Prometheus can filter by run name.

### Key Prometheus queries

```promql
# TTFT p50 for one run, from the proxy histogram
histogram_quantile(0.5, sum by (le)(increase(llmproxy_ttft_seconds_bucket{run="moe-1-vram-ram"}[5m])))

# PCIe RAM to GPU, used as prefill marker (> 1 GB/s)
max(gpu_pcie_rx_bytes_per_second)

# GPU utilization and memory bandwidth use
avg(DCGM_FI_DEV_GPU_UTIL)
avg(DCGM_FI_DEV_MEM_COPY_UTIL)

# CPU package power from RAPL
sum(rate(cpu_rapl_joules_total{domain="package"}[10s]))

# Page cache and disk reads
node_memory_Cached_bytes
sum(rate(node_disk_read_bytes_total[10s]))
```

---

_Comments and corrections welcome. The numbers belong to one card and one model; the method of asking "which resource decides this metric" is the part meant to travel._
