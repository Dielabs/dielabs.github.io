---
layout: default
title: "From Idea to Production"
---

# From Idea to Production

> LLM Inference Sizing in 11 Steps. From a business need to an inference system that is sized and empirically validated. Steps 1–4 are inputs from the customer, steps 5–9 are the architectural response (iterative loop), steps 10–11 close the cycle with measurement and conscious scaling.

## Flow

```
1. Use case
      │
      ▼
2. Workload profile  ──►  3. Traffic profile  ──►  4. Workload SLO
                                                          │
                                                          ▼
                              5. Model architecture
                                                          │
                                                          ▼
                              6. Sizing  (6a capacity → 6b performance)
                                                          │
                                                          ▼
                              7. Runtime architecture
                                                          │
                                                          ▼
                         8. Hardware & fabric architecture
                                                          │
                                                          ▼
                              9. Serving stack
                                                          │
                                                          ▼
            10. Benchmark & baseline  ──►  11. Scaling decision
                                                          │
                                                          ▼
                                                   loop on 5–9
```

## The 11 steps in one sentence each

1. **Use case** — Fix the business problem, the actor, the type of task, the deployment context (greenfield/brownfield) and the success criteria, without naming models or GPUs.
2. **Workload profile** — Quantify the shape of the request — context, input/output distribution at p95, prefill/decode ratio — independently of how many requests arrive over time.
3. **Traffic profile** — Quantify the volume over time and its pattern, deriving concurrency — the number of requests simultaneously present in the system — from users × duty cycle or from RPS × E2E duration when it is not declared directly, and applying an explicit provisioning headroom.
4. **Workload SLO** — Write the service contract in terms of latency (TTFT, ITL, E2E) and availability: SLOs are not a direct input to sizing but a validation constraint on capacity and performance, and their profile determines the benchmark scenario of step 10 (latency-first, throughput-first, full envelope).
5. **Model architecture** — Decide the model independently from serving — family, weight precision, KV quantization, context, attention type, license — because the model constrains everything that follows.
6. **Sizing** — Size **capacity** first (6a, VRAM-first: weights + KV cache + activations + overhead → maximum concurrency per replica), then **verify performance** (6b, sanity check on compute-bound prefill and bandwidth-bound decode against SLOs): the same target concurrency must be sustainable both in memory (capacity) and in time (performance), and when the two axes require different GPUs the dominant constraint wins.
7. **Runtime architecture** — Define the **execution model** on the replica (parallelism TP/PP/EP/DP, KV strategy, scheduling, **aggregated vs disaggregated fork**): the runtime establishes _how_ you want to execute, and in greenfield it drives the hardware while in enterprise on-prem it must adapt to the existing fleet.
8. **Hardware & fabric architecture** — Fix the **constraint surface** (GPU class, HBM per GPU, intra/inter-node fabric, storage fabric for weights and KV offload): the hardware establishes _what is actually possible_, and the fabric must sustain the runtime choices — multi-node TP without NVLink, or disaggregated KV transport without RDMA, are structural traps.
9. **Serving stack** — Configure the platform on top of the runtime and tune the three families of levers (throughput/prefill, latency/decode, VRAM/capacity) **one at a time** starting from vLLM defaults, because modifying multiple parameters together makes effect attribution impossible.
10. **Benchmark & baseline** — Empirically validate the sizing with real datasets and distributions, including the queue in E2E, and produce **one or more operating envelopes** (Cr_closed in closed-loop regime for throughput-first workloads, Cr_open in open-loop regime for latency-first workloads, both for discovery or a dual business metric) that become the baseline for capacity management and SRE.
11. **Scaling decision / iteration** — Identify the **dominant bottleneck** from metrics and choose the lever consciously (stack tuning → runtime → hardware → replicas → model change), treating replicas as a decision and not as a default, because they multiply both capacity and upstream inefficiencies.

## Guiding principles (10)

1. **Worst case, not average** — peaks break SLOs.
2. **Explicit provisioning headroom** — the maximum between expected growth and burst absorption.
3. **N+1 failure domain** — N+2 for critical workloads.
4. **Capacity ≠ Performance** — two axes (6a vs 6b), two possible benchmarks (Cr_closed and Cr_open), executed as a function of the SLO profile: the same target concurrency must be validated on both axes when the scenario requires it.
5. **TCO, not throughput** — €/1M tokens as the business metric, computed from Cr_closed.
6. **Observability from day 1.**
7. **Assume that assumptions are wrong.**
8. **Heterogeneous fleet as a TCO lever.**
9. **Plan the exit, not just the entry.**
10. **Watch the seams between layers.**

## The 5 framework assumptions

1. **Workload profile and traffic profile are two distinct inputs** — the shape of the load and the volume over time are estimated and negotiated separately.
2. **Runtime, hardware and serving stack are three distinct concerns.**
3. **Sizing is an estimate validated empirically.**
4. **Runtime drives hardware in greenfield; hardware constrains runtime in enterprise.**
5. **Scaling is a decision, not a default.**

---

_Framework developed at Dielabs — personal research lab on LLM inference infrastructure._
