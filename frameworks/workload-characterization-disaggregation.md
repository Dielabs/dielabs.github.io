---
layout: default
title: "Workload Characterization in Disaggregation"
---

# Workload Characterization in Disaggregation

## Key Points

- Sizing modern inference doesn't start from the model, it starts from the workload: the same infrastructure can serve very different request mixes depending on prompt length, prefix reuse and arrival pattern. "How many GPUs for Llama 70B" is the wrong question; the right one is "how many replicas for *this* traffic profile under *these* SLOs".
- Every characterization question maps to a precise architectural decision in the Disaggregation OS: the input/output ratio decides whether to disaggregate prefill and decode; prefix reuse sizes the KV cache hierarchy and justifies KV-aware routing; the arrival pattern defines the autoscaling signal.
- The bridge between characterization and sizing is per-replica capacity. This framework uses a two-value convention: **Cr_closed** is the replica's baseline capacity, measured in closed loop under controlled conditions and anchored to the hardware; **Cr_open** is the sustainable capacity under realistic traffic in open loop, anchored to SLOs (TTFT/TPOT). Cr_closed alone sizes nothing — it's the reference against which Cr_open quantifies the "discount" imposed by the specific workload.

---

## 1. Why Workload Comes Before Model

The two phases of inference have opposite natures: prefill is compute-bound (processes all input tokens in parallel, saturates the tensor cores), decode is memory-bound (generates one token at a time, the bottleneck is bandwidth to HBM and KV cache residency). As a practical rule, this boundary shifts with batch size, sequence length, quantization, attention backend and hardware — but as an architectural simplification it holds, and it's what the rest of this framework reasons on. The mix between the two phases — determined entirely by the workload, not the model — decides where the bottleneck forms and therefore which architecture is the right fit.

A summarization workload (8K-token prompts, 200-token outputs) and a creative-generation workload (300-token prompts, 2K-token outputs) on the same model and the same GPU have completely different saturation profiles. The first is prefill-heavy and benefits from disaggregation; the second is decode-heavy, and disaggregation can actually make things worse by adding the cost of KV transfer. Sizing without knowing the mix means sizing blind.

The Disaggregation OS — the system layer that orchestrates the scheduler, memory hierarchy and KV transport (Dynamo, llm-d, or equivalents) — exists precisely to adapt the infrastructure to the workload profile. But configuring it requires measuring that profile first. This framework lists the questions to ask and the architectural decision each answer supports.

---

## 2. The Seven Questions

### Q1 — What is the input/output length distribution (ISL/OSL)?

**Why it matters.** The ISL:OSL ratio determines the ratio between prefill work and decode work. The average isn't enough: you need the histograms, because a long tail of 32K-token prompts in an otherwise short-traffic mix saturates prefill intermittently and wrecks TTFT for everyone else (head-of-line blocking).

**Architectural decision.** This is the question that decides *aggregated vs. disaggregated*. With high ISL:OSL (RAG, summarization, document analysis) prefill/decode disaggregation pays off: dedicated prefill workers absorb long prompts without interfering with the inter-token latency of decodes in flight. With low ISL:OSL, chunked prefill in an aggregated deployment is often enough and simpler. If you disaggregate, this same distribution sets the prefill-to-decode worker ratio — the single most important tuning parameter in the entire system.

### Q2 — How much of the prompt is shared across requests?

**Why it matters.** Common system prompts, RAG templates, few-shot examples, repeated enterprise context: the share of shared prefix is the theoretical ceiling on cache hit ratio. A hit on the prefix eliminates the prefill for those tokens — on a 2K-token system prompt reused across thousands of requests, the capacity difference is dramatic.

**Architectural decision.** Three decisions cascade from this. First: whether and how to size prefix caching. Second: routing — with high prefix reuse, round-robin routing wastes the cache by scattering related requests across different replicas; you need a KV-aware router that assigns requests to the replica that already holds the prefix (prefix scoring). Third: cache hierarchy depth — if the working set of hot prefixes exceeds HBM (G1), you fall back to G2/G3/G4 with managed offload (KVBM or equivalent).

### Q3 — Is traffic multi-turn or one-shot?

**Why it matters.** Conversations reuse the KV cache across turns, but with time gaps of seconds to minutes during which keeping it in HBM is wasteful. One-shot traffic (stateless APIs, batch) doesn't have this problem, but doesn't get this benefit either.

**Architectural decision.** Retention policy and tiering: how long to hold a session's KV, and at which tier (evict → recompute, or offload to CPU/NVMe and restore on the next turn). In parallel, session affinity in the router: turn N+1 must land back on the replica — or tier — that holds turn N's cache, otherwise the reuse is only theoretical.

### Q4 — What is the request arrival pattern?

**Why it matters.** Flat traffic, marked diurnality, unpredictable bursts: the same average capacity requires very different headroom. And bursts in inference don't show up as 100% CPU — they show up as a lengthening queue and degrading TTFT.

**Architectural decision.** The autoscaling signal. GPU utilization is a poor signal for inference (a GPU in decode can look "underutilized" on tensor cores while being fully saturated on KV cache). The right signals are system metrics: queue depth, TTFT p95, KV cache pressure. From here also come decisions on capacity headroom, scale-out speed (cold start for a replica loading tens of GB of model weights isn't instant), and preemption priority during spikes.

### Q5 — What are the SLOs for each use case?

**Why it matters.** An interactive chatbot lives and dies by TTFT (sub-second) and smooth TPOT; an overnight batch job lives by throughput and has no latency SLO. Mixing them in the same pool without distinction means sizing everything for the worst case.

**Architectural decision.** Service classes: priority queues, preemption of batch jobs when interactive traffic arrives, or physically separate pools when the profiles are too far apart. This is also where Cr_open comes in: per-replica capacity must be measured *per SLO class*, not in aggregate — the same replica has a much lower Cr_open under TTFT p95 < 800ms than under "best effort".

### Q6 — How many models, how many variants?

**Why it matters.** A single flagship model is the simple case. Multiple models, canary versions, per-tenant LoRA adapters change the nature of the problem: GPU memory becomes contested between weights and KV cache across different models.

**Architectural decision.** Dedicated pools per model vs. multiplexing (with hot-swapping of LoRA adapters), and the resulting scheduling policy. It also affects the cache hierarchy: each model has its own prefix space, and the aggregate working set grows.

### Q7 — How will traffic evolve over the next 6–18 months?

**Why it matters.** The current profile sizes day 1; the trajectory sizes the topology. Going from 1 to N nodes isn't linear: disaggregation across nodes introduces KV cache transfer over the network.

**Architectural decision.** Cluster topology and interconnect. If growth will push prefill and decode onto different nodes, the KV transport layer (NIXL or equivalent: RDMA, NVLink, or TCP as fallback) and network bandwidth become part of sizing. This is the question that separates a deployment that scales from one that has to be rebuilt.

---

## 3. From Signal to Decision

| Signal from the workload | Architectural decision |
|---|---|
| High ISL:OSL (prefill-heavy) | P/D disaggregation; prefill:decode worker ratio |
| Low ISL:OSL (decode-heavy) | Aggregated + chunked prefill; disaggregation not justified |
| High prefix reuse | Prefix caching + KV-aware router + G1–G4 hierarchy |
| Multi-turn with time gaps | Offload/restore to G2–G3, session affinity |
| Bursty traffic | Autoscaling on queue depth / TTFT p95, not GPU util |
| Mixed interactive + batch | Service classes, preemption, per-class Cr_open |
| Multi-model / multi-tenant | Pool vs. multiplexing, LoRA swapping |
| Multi-node growth | KV transport interconnect, topology |

---

## 4. What If the Client Doesn't Know the Answers?

This is the normal case. Few organizations know their own ISL/OSL histograms or their potential prefix reuse. Three paths, in order of preference:

**Measure from existing traffic.** If a service is already in production (even against external APIs), the gateway logs contain almost everything: input and output lengths, arrival timestamps, per-endpoint distribution. One week of logs beats any estimate.

**Derive from the use case.** Absent real traffic, the use case constrains the profile: an enterprise RAG has ISL dominated by chunk size and structurally high prefix reuse (fixed template + system prompt); a coding assistant has strong multi-turn behavior and ISL that grows through the session. From this you build a synthetic profile.

**Benchmark on the synthetic profile.** With the profile (real or derived) in hand, measure Cr_open in open loop — with GuideLLM or equivalent — on candidate replicas, under the stated SLOs. Cr_closed provides the hardware baseline; the ratio between the two quantifies how much the specific workload "discounts" theoretical capacity.

The methodological point: characterization isn't a questionnaire you fill out once. It's a loop — estimate, deploy, measure from real traffic, recalibrate worker ratios, cache retention and autoscaling thresholds. The Disaggregation OS exposes the knobs; the workload tells you how to turn them.

---

## 5. Worked Example

**Profile.** Enterprise document RAG: 8–16K-token prompts (fixed template + retrieved chunks), short output (200–500 tokens), high prefix reuse (system prompt and template common to all requests), diurnal traffic with morning peaks, SLO of TTFT < 1s for the user-facing interface.

**Reading the profile.** ISL:OSL heavily skewed toward input → prefill-heavy (Q1). Structural shared prefix → high potential cache hit ratio (Q2). One-shot or short multi-turn (Q3). Predictable morning burst (Q4). Tight latency SLO on the interactive path (Q5).

**Decisions.** P/D disaggregation with an oversized prefill pool relative to decode (short output frees decode workers quickly); prefix caching with a KV-aware router to capitalize on the shared template; short session KV retention; prefill pool autoscaling on TTFT p95 and queue depth, with scheduled pre-scaling for the morning peak. Cr_open measured in open loop on a synthetic 12K-in/300-out profile under TTFT p95 < 1s: that number — not the GPU spec sheet — sets the replica count.

---

## 6. Discovery Checklist

Compact form of the seven questions, usable during qualification:

1. ISL/OSL distribution per use case (histogram, not average — watch the tails)
2. Share of prompt shared across requests (system prompt, template, few-shot)
3. Multi-turn or one-shot? If multi-turn: average session length and gap between turns
4. Arrival pattern: flat, diurnal, bursty? Peak-to-average ratio
5. SLOs per use case: TTFT and TPOT targets, or throughput only?
6. Number of models, variants, adapters; tenant isolation requirements
7. Growth trajectory over 6–18 months (requests/day, new use cases)
