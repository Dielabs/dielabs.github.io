---
layout: default
title: "One Capacity Is Not Enough"
---

# One Capacity Is Not Enough

> The Dielabs benchmark framework for LLM inference capacity planning: why a single capacity number does not exist, and how CrossP, Cr_closed and Cr_open break it down into three measurable quantities, anchored to the three vLLM benchmarks.

## Key Points

The conventional LLM inference benchmark — maximum throughput at saturation — answers a question that capacity planning does not ask. The capacity of an inference system is not a property of the hardware: it is a property of the hardware + workload + SLO combination, and no single number can express it. The Dielabs framework breaks it down into three measurable quantities: **CrossP**, the concurrency boundary where the system changes regime; **Cr_closed**, the per-replica capacity anchored to the hardware, measured in closed loop; **Cr_open**, the per-replica capacity anchored to the SLOs, measured in open loop. Cr_open is almost always lower than Cr_closed, and it is almost always the number that matters for user-facing workloads. Sizing — replicas = target / Cr, plus headroom — is the conclusion of the exercise, not a fourth quantity.

---

## The Wrong Question

The industry measures inference with a datasheet number: tokens per second at maximum load. It is a true, reproducible number, and almost always irrelevant to anyone who has to size a real deployment — because most production systems do not run at saturation, and because the configurations that maximize throughput (large batches, high concurrency) are the same ones that degrade the latency each individual user perceives.

The capacity planning question is not "how fast can this system go at most?" but "how many users can it serve *well*, what defines 'well', and what happens when it can't?". Answering it means turning the conventional benchmark logic upside down: instead of looking for maximum throughput and reporting latency as a side effect, fix the service quality constraints first and derive capacity from them. The resulting number is almost always lower than maximum throughput — and almost always more useful.

---

## Three Tools, Three Numbers

The simplest way to understand the framework is to look at what the engine itself does. vLLM, the most widely used inference engine, has three benchmark commands, and they return three different numbers:

- `throughput`: sends all requests at once and measures what comes out. It is the slide number: true, but measured in a way no user ever works.
- `latency`: a few requests, short input, measures time only. Useful to compare two engines, not to size a deployment.
- `serve`: simulates users arriving at random, with a concurrency cap, and measures time to first token, time between tokens, and requests served within target. It is the only one that looks like a customer.

The Dielabs framework starts here and adds what the engine does not need: Cr_closed is a disciplined reading of the `throughput` regime (not a single point, but the knee found with a sweep); Cr_open is the `serve` regime anchored to the customer's SLOs; CrossP, the boundary between the two, does not exist in any command. Whoever presents a number must say which tool measured it. A number without a measurement mode is not a number.

---

## The Regime Boundary — CrossP

Every inference system has two competing goals: per-user latency and aggregate throughput. At low concurrency, each additional user costs little: the system has spare capacity, throughput grows, latency stays flat. At some concurrency level this changes — contention on memory bandwidth and scheduling means each new user costs more in latency than it adds in throughput.

That point is the **Crossover Point (CrossP)**: operationally, the first concurrency level at which the percentage increase in latency exceeds the percentage increase in throughput. It is not a capacity and it is not a sharp point — it is a transition zone, measured with a concurrency sweep by comparing ΔTTFT% and ΔThroughput% at each step. The exact number matters less than the order of magnitude: knowing whether the crossover is at 4, 8 or 32 radically changes configuration and scaling strategy. Not knowing it means sizing blind.

Below CrossP you optimize for latency; above CrossP you optimize for throughput and scale horizontally; around CrossP the system is unstable and should not be operated in a sustained way. CrossP tells you *where* the system changes nature. It does not tell you how much it delivers in each regime: for that you need two distinct quantities.

---

## Two Quantities, Two Anchors

The conceptual mistake behind the "single capacity number" is mixing two different physical phenomena. There are two ways to stress an inference system — keep it under constant pressure, or expose it to traffic that arrives independently of completions — and they measure different properties, producing different numbers.

**Cr_closed** is the per-replica capacity measured in closed loop: the client keeps in-flight concurrency fixed, and as soon as a request finishes another one starts. This regime keeps the system close to saturation and answers "how hard can I push this replica": the knee of the throughput/concurrency curve, beyond which throughput stops growing while latency degrades fast. It is the **hardware-anchored** quantity — SLOs are not part of the definition — and it serves throughput-first workloads, batch processing, cost per token and economic sizing. €/Mtok, the metric that closes sizing conversations with a customer, can only be derived from here: computing it from a non-saturated regime systematically overestimates it.

**Cr_open** is the per-replica capacity measured in open loop: arrivals are governed by an arrival rate (steady or Poisson), concurrency is not imposed but emerges — governed to a first approximation by Little's law (L = λ × W), with the caveat that W grows with load because of queueing, which makes any low-load estimate optimistic. Cr_open answers "how much can I promise to a number of users": the maximum sustainable arrival rate while meeting all latency SLOs (TTFT, ITL, E2E at the target percentiles). It is the **SLO-anchored** quantity, expressed in RPS, and it serves chat, RAG, agentic systems and every user-facing workload.

Why is Cr_open almost always lower than Cr_closed? Because of GPU physics. At low concurrency, the time of each decode step is dominated by loading the weights from memory: serving 1 or 10 users costs almost the same, latency stays flat and throughput grows for free. Beyond a certain batch threshold the GPU becomes compute-bound: each additional user lengthens the step for everyone. The throughput knee (Cr_closed) lies beyond that threshold; the latency knee (Cr_open) lies before it. This is not a measurement flaw: they are two different curves by construction. Bursty traffic widens the gap, it does not create it. The gap is informative: it measures the margin between where you operate and where the hardware saturates.

|                       | Cr_closed                              | Cr_open                                                 |
| --------------------- | -------------------------------------- | ------------------------------------------------------- |
| **Anchor**            | Hardware                               | SLO                                                     |
| **Regime**            | Closed loop (fixed concurrency)        | Open loop (fixed arrival rate, emergent concurrency)    |
| **Answers**           | How hard can I push the replica        | How much can I promise to users                         |
| **Unit**              | Concurrency (range)                    | RPS                                                     |
| **Used for**          | Batch, €/Mtok, saturation              | Chat, RAG, agentic, latency-first                       |
| **vLLM equivalent**   | `vllm bench throughput` (saturation)   | `vllm bench serve` (Poisson arrivals)                   |

---

## Capacity Does Not Belong to the Hardware

The most important consequence of the double anchor: **the same hardware produces different capacities depending on the SLOs applied**. A replica with CrossP at 8 concurrent requests can sustain 6 users for a chat with a tight TTFT, and 12 for a batch job with a relaxed E2E constraint. This is not measurement noise: they are two different capacities, both true, determined by the choice of SLOs.

This is why the framework does not accept a "default" benchmark: the choice between Cr_closed, Cr_open or both follows from the workload. And it is why the output of a benchmark is not a number but a **capacity card**: hardware, model, runtime, data profile, SLOs, measured Cr value and the metrics at the measurement point — the unit you can compare over time, across engine versions, models and GPU generations.

Sizing is the conclusion of the exercise, not an additional quantity: replicas = ceil(target / Cr), with consistent units (RPS on Cr_open for latency-first, tok/s measured at Cr_closed for throughput-first) and an operating margin for bursts, failover and rolling updates. Cr is never used at its limit.

---

## The Agentic Extension

The framework was born for chat and batch workloads, where the cost per request is fairly predictable from the data profile. Agentic workloads break this assumption in two ways. The dynamic fan-out of agents makes the inference load per query no longer deterministic: Cr is still the right concept, but estimating it requires characterizing the agent loop. And the prefill cost — dominant in trajectories with long, growing prefixes — depends on the cache hit rate, which is a property of the application harness, not of the hardware: two harnesses with the same task success rate can produce Cr_open values that differ by multiples on the same hardware.

As a consequence, for agentic workloads Cr_open becomes a property of the model + hardware + harness triple, and the assessment must include reading how the agent behaves. More on what agentic workloads do to serving in [Agentic](/agentic/).

---

## What Gets Automated and What Doesn't

vLLM already ships an auto-tuning script: given an SLO ("p99 under 500 ms"), it searches on its own for the configuration that meets it and maximizes throughput. It is a signal of where the market is going: sizing a single replica will become a command. What no command does is the step before — translating the customer's real traffic into a profile (lengths, arrivals, context reuse) and into an SLO the customer is willing to sign. The value of the framework is not in measuring, which gets automated, but in deciding what to measure and with which promise. The capacity card is the contract for that promise.

---

## The Point

A single capacity number for an inference system does not exist. There is a regime boundary (CrossP), a saturation capacity (Cr_closed) and a service capacity (Cr_open) — three quantities measurable in hours, not weeks, with a concurrency sweep and an open-loop benchmark. Maximum throughput is a datasheet number; the maximum sustainable within SLOs is a capacity planning number. Whoever sizes with the first buys hardware for a peak that will never come, at the price of a degraded experience every day.

---

*External reference for the three benchmark tools, auto-tuning and the roofline model: A. Gordić, "Inside vLLM: Anatomy of a High-Throughput LLM Inference System" (August 2025).*

*Original Dielabs work by Diego Bardella.*
