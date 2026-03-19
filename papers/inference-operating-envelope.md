---
title: "The Inference Operating Envelope"
subtitle: "Latency vs Throughput — Finding the Crossover Point"
author: Diego Bardella – Dielabs
date: 2026-03
tags: [dielabs, inference, latency, throughput, crossover, kv-cache, operating-envelope, manifesto]
---

# The Inference Operating Envelope

**Latency vs Throughput in LLM Inference — Finding the Crossover Point**

*Diego Bardella – Dielabs, Casale Monferrato | March 2026*

---

## The Problem

Every LLM inference system has two performance metrics that matter: how fast each user gets a response (latency), and how many users the system can serve at once (throughput). These two goals are in tension. Serving more users simultaneously makes each individual response slower.

The industry benchmarks for throughput: tokens per second at maximum load. But most production systems spend most of their time at low load. The result is that systems are configured for a peak that rarely happens, at the cost of degrading the everyday experience. *Because throughput-oriented configurations increase latency through larger batches and high concurrency levels, which increase the number of active sequences per step and amplify contention and scheduling delays*.

The question that matters is not "how fast can this system go at full load?" but **"at what point does adding more concurrent users start hurting everyone's experience?"**

***That point exists on every inference system. We call it the crossover point.***

---

## The Crossover Point

At low concurrency, each additional user barely affects latency. The system has spare capacity. Throughput grows and latency stays flat.

At some concurrency level, this changes. This transition is driven by finite memory bandwidth: as concurrency increases, requests compete for the same bandwidth, making each step slower. Latency starts climbing faster than throughput grows. *Each new user costs more in latency than they add in throughput.* That is the crossover point (Cp).

> **Cp is the first concurrency level where the percentage increase in latency exceeds the percentage increase in throughput.**

To find it, run a concurrency sweep (1, 2, 4, 8, 16, 32 concurrent requests). At each step, compute:

- **ΔThroughput %** — how much aggregate throughput increased from the previous step
- **ΔTTFT %** — how much time-to-first-token (p90) increased from the previous step

**Cp = first step where ΔTTFT % > ΔThroughput %.**

If the data is noisy, two fallback heuristics work: the "knee" where the latency curve bends upward, or the point where throughput gains drop below 10% per doubling of concurrency.

The exact number matters less than the order of magnitude. Knowing your crossover is at 4 or 8 or 32 changes how you configure the system. Not knowing it means you are guessing.

---

## What To Do With It

> **The Crossover Decision**
>
> **Below Cp** → optimize for latency. Keep configuration simple, memory conservative, batching minimal.
>
> **Above Cp** → optimize for throughput. Scale **horizontally even internally** *(adding independent replicas, e.g. across GPUs on the same node or across multiple nodes)*, increase batching, push utilization. Saturated but stable zone.
>
> **Around Cp** → unstable zone. Small load changes cause big performance swings. Avoid sustained operation here.
>
> **Unknown Cp** → you are dimensioned blind.

**Most production systems — enterprise assistants, clinical tools, RAG pipelines — operate below Cp for 90% of their lifetime.** For these systems, throughput optimization is the wrong target. The right target is keeping latency low under normal conditions and handling bursts gracefully.

---

## Two Scenarios

### Scalable Architecture

If you can add hardware (a second GPU, a second instance), you have a choice between architectures: one big instance vs multiple smaller ones. Below Cp, the single instance wins — no coordination overhead, minimum latency. Above Cp, multiple instances win — they distribute the load and keep each replica in its latency-optimal zone.

The crossover tells you when to scale. Scaling before Cp is wasteful. Scaling after Cp is necessary.

Not all scaling is beneficial. Splitting a model across devices (tensor parallelism) on slow interconnects can make things worse at every concurrency level. More hardware does not automatically mean a better operating envelope.

### Fixed Architecture

If you have one GPU and no option to add another, there is no architecture choice. But the intra-configuration crossover still exists: the concurrency level where your single instance transitions from "serving well" to "struggling."

Fixed architectures are more fragile because there is no horizontal escape valve. When concurrency exceeds Cp, **the only options will be: degrade latency for everyone, or reject excess requests**. This makes the next point critical.

---

## The Safety Margin

The crossover point defines performance. The second axis of the operating envelope is memory pressure.

Inference engines allow control over how much memory is reserved for KV cache. Pushing utilization high (e.g. 95%) maximizes KV cache capacity and supported concurrency, but this is not a strict hard limit — it is a target.

In practice, only that portion of memory is actively controlled. The remaining memory (the "unreserved" 5%) is not strictly managed and can fluctuate due to runtime overheads such as CUDA context, temporary buffers, fragmentation, and other allocations. Under load variability, this portion can grow unexpectedly, leaving insufficient headroom and triggering out-of-memory conditions.

Keeping utilization conservative (e.g. 85%) introduces a safety buffer. This absorbs variability in request size, concurrency, and runtime allocations, preventing memory exhaustion. Instead of crashing, the system can reject excess requests.

| Configuration | Failure Mode | Recovery |
|---|---|---|
| **Aggressive (95%)** | Out-of-memory → process termination → model reload | 10–30 seconds of zero service |
| **Conservative (85%)** | Request rejected (503) → client retries | Immediate — no downtime |

A 503 is a system saying "I'm full, try again." An OOM event is a system going dark.

The trade-off is quantifiable: a conservative margin lowers Cp — because less memory is available for KV cache, reducing the number of concurrent sequences the system can sustain — but eliminates catastrophic failure modes.

This defines a second control knob: not performance, but failure behavior.

For most real-world deployments, where availability matters more than peak capacity, this is the correct choice.

---

## The Point

Every inference system has a crossover point. It separates two regimes that need different engineering strategies. Operating in the wrong regime — or not knowing which regime you are in — leads to either degraded user experience or wasted resources.

Measuring the crossover is simple: a concurrency sweep, two metrics, one comparison. It takes hours, not weeks. It should be a standard deliverable for any inference deployment, like a load test for a web service.

The Inference Operating Envelope is the combination of the crossover point and the safety margin. Together they answer: "how many users can this system serve well, and what happens when it can't?"

> The question should not be "what is the maximum throughput?" Instead, "in which regime is this system operating, and have I measured the crossover?"
>
> Any deployment that cannot answer this is potentially dimensioned blind.
