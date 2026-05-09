---
layout: default
title: "LLM Inference Sizing in 10 Steps"
---

# LLM Inference Sizing in 10 Steps

> From a business need to an inference system that is sized and empirically validated. Steps 1–3 are inputs from the customer, 4–8 are the architectural response (iterative loop), 9–10 close the cycle with measurement and conscious scaling.

## Flow

```
1. Use case  ──►  2. Workload profile  ──►  3. Workload SLO
                                                    │
                                                    ▼
                            4. Model architecture
                                                    │
                                                    ▼
                            5. Sizing  (5a capacity → 5b performance)
                                                    │
                                                    ▼
                            6. Runtime architecture
                                                    │
                                                    ▼
                       7. Hardware & fabric architecture
                                                    │
                                                    ▼
                            8. Serving stack
                                                    │
                                                    ▼
                  9. Benchmark & baseline  ──►  10. Scaling decision
                                                                │
                                                                ▼
                                                         loop on 5–8
```

## The 10 steps in one sentence each

1. **Use case** — Fix the business problem, the actor, the type of task, the deployment context (greenfield/brownfield) and the success criteria, without naming models or GPUs.
2. **Workload profile** — Quantify the expected traffic (context, input/output distribution, p95 concurrency) with explicit provisioning headroom, deriving concurrent requests from users × duty cycle or from RPS × E2E duration when not declared directly.
3. **Workload SLO** — Write the service contract in terms of latency (TTFT, ITL, E2E) and availability: SLOs are not a direct input to sizing but a validation constraint on capacity and performance, and their profile determines the type of benchmark in step 9 (latency-first, throughput-first, full envelope).
4. **Model architecture** — Decide the model independently from serving — family, weight precision, KV quantization, context, attention type, license — because the model constrains everything that follows.
5. **Sizing** — Size **capacity** first (5a, VRAM-first: weights + KV cache + activations + overhead → max concurrency per replica), then **verify performance** (5b, sanity check on compute-bound prefill and bandwidth-bound decode against SLOs): the same target concurrency must be sustainable both in memory (capacity) and in time (performance), and when the two axes require different GPUs the dominant constraint wins.
6. **Runtime architecture** — Define the **execution model** on the replica (parallelism TP/PP/EP/DP, KV strategy, scheduling, **aggregated vs disaggregated fork**): the runtime establishes _how_ you want to execute, and in greenfield it drives the hardware while in enterprise on-prem it must adapt to the existing fleet.
7. **Hardware & fabric architecture** — Fix the **constraint surface** (GPU class, HBM per GPU, intra/inter-node fabric, storage fabric for weights and KV offload): the hardware establishes _what is actually possible_, and the fabric must sustain the runtime choices — multi-node TP without NVLink, or disaggregated KV transport without RDMA, are structural traps.
8. **Serving stack** — Configure the platform on top of the runtime and tune the three families of levers (throughput/prefill, latency/decode, VRAM/capacity) **one at a time** starting from vLLM defaults, because modifying multiple parameters together makes effect attribution impossible.
9. **Benchmark & baseline** — Empirically validate the sizing with real datasets and distributions, including the queue in E2E, and produce **one or more operating envelopes** (Cp_closed in closed-loop regime for throughput-first workloads, Cp_open in open-loop regime for latency-first workloads, both for discovery or dual business metric) that become the baseline for capacity management and SRE.
10. **Scaling decision / iteration** — Identify the **dominant bottleneck** from metrics and choose the lever consciously (stack tuning → runtime → hardware → replicas → model change), treating replicas as a decision and not as a default, because they multiply both capacity and upstream inefficiencies.

## Guiding principles (10)

1. Worst case, not average — peaks break SLOs.
2. Explicit provisioning headroom (max between growth and bursts).
3. N+1 failure domain (N+2 for critical workloads).
4. Capacity ≠ Performance — two axes (5a vs 5b), two possible benchmarks (Cp_closed and Cp_open), executed as a function of the SLO profile: the same target concurrency must be validated on both axes when the scenario requires it.
5. TCO, not throughput — €/1M tokens as the business metric (computed from Cp_closed).
6. Observability from day 1.
7. Assume that assumptions are wrong.
8. Heterogeneous fleet as a TCO lever.
9. Plan the exit, not just the entry.
10. Watch the seams between layers.

## The 4 opinionated points

1. Runtime, hardware and serving stack are three distinct concerns.
2. Sizing is an estimate validated empirically.
3. Runtime drives hardware in greenfield; hardware constrains runtime in enterprise.
4. Scaling is a decision, not a default.

---

_Synthesis of the full framework. References: Dielabs — personal research lab on LLM inference infrastructure._
