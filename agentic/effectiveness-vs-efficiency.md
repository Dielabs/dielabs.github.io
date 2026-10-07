---
layout: default
title: "Effectiveness vs Efficiency in an Agentic Architecture"
---

# Effectiveness vs Efficiency in an Agentic Architecture

> Quality of outcomes and cost of outcomes are two different axes of an agentic system. Context engineering is the hinge between them — and it is where agentic sizing lives.

*Assumes infrastructure and LLM serving context (prefill, decode, KV cache, batching, SLO); no agent engineering specialization required. The `[D]` marker flags a Dielabs extension or position, not industry consensus.*

## Key Points

- **Effectiveness** and **efficiency** are two distinct axes of an agentic system. **Effectiveness = quality of outcomes** (is the task solved well?). **Efficiency = cost of outcomes** (at what cost?). Confusing them is the most common cause of wrong sizing and off-target technical proposals.
- The two dimensions have different metrics and owners, but they are **not independent**. Between them lives **context engineering**: the discipline that acts on both at once. It is the hinge of the system.
- The agentic workload **breaks classic sizing** along four dimensions (tool call rate, context growth, compaction, sub-agent fan-out). This is where a presales reader realizes that chatbot sizing cannot be reused for an agent.
- **Some agentic design choices set the conditions within which serving efficiency can be expressed.** `[D]` A perfectly tuned engine on a badly designed workload delivers less than an average engine on a well-structured one.
- Sizing an agentic workload lives **in the hinge** between the two axes. Measuring them separately and ignoring the context engineering that binds them leads to sizing that works in synthetic benchmarks and fails in production. `[D]`

---

## 1. The Two Axes — Quality vs Cost of Outcomes

In an agentic system, the question "does it work well?" splits into two questions that never meet unless you separate the axes.

- **Effectiveness — quality of outcomes.** Does the agent solve the *right* task, *correctly*, without diverging or falsely declaring it is done?
- **Efficiency — cost of outcomes.** Is that result produced at the lowest possible resource cost, while staying within the SLO?

They are orthogonal in the sense that they can move independently *in principle*: an agent can be effective but extremely expensive (it solves everything, but burns 100 tool calls and 500K tokens per task), or efficient but ineffective (it consumes little, but fails half the tasks). A production-grade system aims high on both.

The central point here is that **the two axes are bound by a common layer — context engineering** — and that this is exactly where the sizing of an agentic deployment is decided. First, though, it is worth seeing *why* the agentic workload differs from what you already know how to size (§2).

---

## 2. Why the Agentic Workload Breaks Classic Sizing

Classic (chat-style) sizing assumes that 1 user-visible request = 1 prefill + 1 decode. The agentic workload breaks this equation along four dimensions — tool call rate, context growth, compaction, sub-agent fan-out — and produces **prefill amplification**: an N-step agent runs N prefills over a growing context. The right metric is not `requests/sec` but **`agent-tasks/sec under SLO at the 99th percentile`**.

The four dimensions, prefill amplification and the metric are covered in [Agentic Systems for Inference Infrastructure People](agentic-systems.md), Chapter 8; capacity measured in open loop under SLO (Cr_open) is the one defined in [One Capacity Is Not Enough](/frameworks/one-capacity-is-not-enough).

All four dimensions stem from **agentic design choices** (how many tools, how context is managed, whether sub-agents are used), and how much they weigh on cost depends on *how* those choices are made. That is §3.

---

## 3. Context Engineering as the Hinge Between the Two Axes `[D]`

The four dimensions of §2 are not fixed facts. Their intensity — and therefore the cost they push onto serving — depends on the layer that sits between effectiveness and efficiency: **context engineering**.

A common mistake is to classify context engineering, compaction, sub-agents and external memory as "effectiveness levers" (they help the agent reason better). That is only half the truth: **they act on both axes at the same time.**

| Technique | Effect on effectiveness (quality) | Effect on efficiency (cost) |
|---|---|---|
| **Compaction** | Avoids lost-in-the-middle, keeps the model focused | Reduces the tokens to re-prefill at each iteration |
| **Sub-agents** | Isolates exploration context → cleaner reasoning | Protects the parent's prefix → prefix caching preserved |
| **Skills (lazy load)** | Gives the model only the relevant capabilities | Pays tokens only for what is needed |
| **External memory** | Authoritative state, anti-divergence | Takes state out of the context → less prefill |

Context engineering is therefore the **hinge**: the layer that, done well, raises quality *and* lowers cost; done badly, degrades both.

**The strong claim, stated in a defensible way:**

> Some agentic design choices set the conditions within which serving efficiency can be expressed. `[D]`

In other words: efficiency is not just "how good the engine is". It is "how well the agentic workload *lends itself* to being served". A perfectly tuned engine on a badly designed workload delivers less than an average engine on a well-structured one.

The coupling mechanism — a stable prefix keeping the prefix cache alive, a reworked prefix destroying it, and the resulting cache hit rate as an input of Cr_open — is in [Context Engineering vs KV Cache Engineering](context-vs-kv-cache-engineering.md).

---

## 4. The Effectiveness Axis

**Governing question:** did I solve the *right* task, *correctly*, without wasting iterations?

Effectiveness is not a property of the model, it is a property of the **harness** that orchestrates it. An LLM is a stateless function `f(context) → P(next_token)`; the quality of the outcome emerges from how that function is guided and constrained.

**Effectiveness-specific levers** (beyond the shared context engineering of §3):

- **Reasoning policy** — constraints in the prompt that steer the model's probability distribution towards disciplined decisions (e.g. "don't mark it complete until it really is", "don't propose changes to code you haven't read"). *Soft* guidance: it shifts probability, it does not guarantee.
- **Hard guardrails** — deterministic limits in the harness code: max iterations, timeouts, permission gates, tool call format validation. The model proposes, the harness disposes.
- **Verify step** — the mechanism that forces the loop to close before moving on, to avoid divergence.

**Metrics:** task success rate, `N_steps`, degenerative looping rate, and — the summary KPI that binds the two axes — **cost per successful outcome** (total cost / successful tasks, not per request).

**Owner:** agent engineer / whoever designs harness and prompts. **Hardware-agnostic.**

---

## 5. The Efficiency Axis

**Governing question:** am I serving the requests generated by the loop at the lowest cost, within SLO?

Efficiency is a property of the **inference engine** (vLLM, TensorRT-LLM, etc.) and of the hardware. The engine does not know, and does not need to know, whether a request is turn 3 of an agentic refactoring or a one-off chatbot question. It sees incoming tokens, KV cache to manage, outgoing tokens. It is **content-agnostic**.

**Levers:** batching (continuous / in-flight), prefix caching, KV cache management and tiering, scheduling, quantization, parallelism (TP/PP), disaggregated prefill/decode.

**Metrics:**

| Metric | Meaning |
|---|---|
| Throughput | Aggregate tokens/sec |
| TTFT / ITL / TPOT | Latency: time to first token, inter-token, time per output token |
| Cr_closed `[D]` | Capacity per replica, hardware-anchored, closed loop |
| Cr_open `[D]` | Capacity per replica, SLO-anchored, open loop |
| GPU utilization | Actual use of the hardware |

**Owner:** inference engineer / platform team. **Lives in the inference engine.**

---

## 6. The Blind Spot of Inferring the Workload From the Interface

Two systems with identical UX ("upload a file and analyze it") can have opposite serving profiles:

- **Single-shot:** the host injects the content into the context *before* inference (deterministically or via RAG). 1 heavy prefill + 1 decode. Prefill-dominated, monolithic.
- **Agentic:** the model reads selectively via tool calls *inside* the loop. N prefills over a growing context. Prefill-amplified, prefix caching critical.

From the outside they are indistinguishable. **You cannot infer the workload profile from the interface.** The right assessment questions:

- Is tool use inside the loop, or is it host pre-processing?
- How many tool calls per task on average (`N_steps`)?
- Does the context grow within the session?
- Is there compaction? How often?
- Is there sub-agent fan-out?

Only these answers tell you whether to size for `requests/sec` (chat) or for `agent-tasks/sec under SLO` (agentic).

---

## 7. Implications for Sovereign On-Premises Deployments `[D]`

In the cloud, hardware is elastic: imprecise sizing gets corrected by scaling, at a cost. In a sovereign on-premises deployment — fixed hardware, upfront capex, no elasticity — the effectiveness/efficiency hinge becomes **the** variable that decides whether the project stays within budget.

Most counterparts sit entirely on one axis (the agent engineer who ignores the KV cache) or entirely on the other (the inference engineer who sees abstract requests without realizing they come from a loop). The differentiating value is to own the hinge:

> "Your agentic design choice has a computable infrastructure cost, and I can quantify it — with Cr_open, SLO-anchored, on real hardware."

---

## 8. Operational Summary

| Axis | Effectiveness (quality of outcomes) | Efficiency (cost of outcomes) |
|---|---|---|
| Question | Right task, solved well? | Served at the lowest cost within SLO? |
| Own levers | Reasoning policy, guardrails, verify | Batching, KV management, scheduling, quantization |
| Metrics | Task success, N_steps, cost/outcome | Throughput, TTFT/ITL/TPOT, Cr_closed/Cr_open |
| Owner | Agent engineer / harness | Inference engineer / platform |
| Agnostic to | Hardware | Content |
| **Common hinge** | **Context engineering** (skills, sub-agents, compaction, external memory) — acts on both | |

**The lines to remember:**

> Effectiveness is the quality of outcomes, efficiency is their cost. In between lives context engineering, which moves both — and that is where agentic sizing lives. `[D]`

> Some agentic design choices set the conditions within which serving efficiency can be expressed. A great engine on a badly designed workload delivers less than an average engine on a well-structured one. `[D]`

---

## Appendix — Dielabs Experimental Direction `[D]`

A falsifiable hypothesis for the coupling of §3: with the same task and hardware, two context engineering strategies produce measurably different Cr_open values.

Minimal protocol:

1. Same agentic task, same model, same hardware (e.g. RTX 4070 Super, Qwen3-8B-AWQ, vLLM).
2. Variant A: disciplined context engineering (stable prefix, structured eviction).
3. Variant B: naive growth (reworked prefix, disorderly eviction).
4. Measure: prefix cache hit rate, prefill amplification factor, Cr_open per variant.
5. Expected: a significant delta in Cr_open — empirical proof that context engineering moves efficiency, not just quality.

Candidate output: an extension of [The Shifting Bottleneck](/papers/the-shifting-bottleneck) — how agentic design shifts the serving bottleneck. Demonstrating the hinge, not just asserting it.

---

*Original Dielabs work by Diego Bardella.*
