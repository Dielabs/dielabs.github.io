---
layout: default
title: "Context Engineering vs KV Cache Engineering"
---

# Context Engineering vs KV Cache Engineering

> Why designing an agent is a capacity planning decision.

## Key Points

The cost and performance of an agentic LLM workload are decided at two distinct layers of the stack: the application layer, where **context engineering** decides *what* enters the context window and in what shape, and the infrastructure layer, where **KV cache engineering** decides *how much it costs* to materialize and re-materialize that context in GPU memory. The two layers are owned by different roles that rarely talk to each other, but they are tightly coupled: every agent design choice — compaction, sub-agents, loop structure — changes the cache reuse pattern, and therefore the marginal cost per token, the TTFT and ultimately the infrastructure sizing. The expected cache hit rate is a property of the application, not of the hardware: this is why in the Dielabs framework it is an explicit input of **Cr_open**, the SLO-anchored capacity per replica. You cannot size the rack without reading the agent's code.

---

## Two Disciplines That Don't Talk

Whoever designs the agent decides the cost of serving without knowing it. Whoever pays that cost — whoever sizes the infrastructure — gets the bill without knowing which decisions produced it. This page is about the missing interface between the two.

In recent months the public debate on agentic AI has moved from prompt engineering to context engineering: how to structure an agent's context window so that it holds the most useful signal with the least noise. It is a conversation that lives entirely at the application layer — agent frameworks, harnesses, orchestration — and typically ignores what happens underneath: how that context is materialized in GPU memory, reused across requests, moved between storage tiers.

Symmetrically, people who do inference serving reason about prefix caching, KV offload, cache-aware routing and disaggregation, but treat the workload as an external given: requests arrive with certain input/output length distributions and a certain prefix reuse rate, and you optimize on those. Where that pattern comes from — which application design choices produced it — stays out of scope.

This separation was tolerable in the chat era: relatively short sessions, context dominated by the system prompt and a few turns, simple reuse patterns. **With agentic workloads it becomes a design error.** An agent working for tens of minutes on a task generates trajectories with very long, growing prefixes, sub-agent fan-out, massive tool result injection: the shape of these trajectories — decided entirely at the application layer — is the dominant factor in serving cost. **The two worlds need to sit at the same table.**

---

## Context Engineering — The Shape of Trajectories

Context engineering is the set of techniques the application harness uses to decide what enters the context window, when, and in what form. The main levers, seen from the side of whoever will have to serve that traffic:

**Compaction.** When the context approaches its limit, the harness summarizes it and restarts from a compressed version. It reduces the tokens carried into later turns, but rewrites the conversation prefix: from the context's point of view, it is a total discontinuity.

**Sub-agents and fan-out.** A complex task is broken down by delegating parts of it to sub-agents, each of which typically starts from a shared subset of the parent's context (instructions, task state) plus a specific delta. It is an isolation technique: the sub-agent works in its own clean context and returns only the result, without polluting the main context.

**Progressive disclosure (skills, lazy documentation).** Instead of loading all potentially useful material upfront, the harness exposes lightweight indexes to the model and loads full content only when needed. The context grows on demand, by appending.

**Tool result design.** How verbose a tool's output is, and whether it gets truncated, structured or summarized before entering the context. In an agent loop, tool results are often the dominant part of the context — far more than the system prompt or user messages.

**Loop structure.** The deepest choice: does the context grow strictly append-only (each turn adds at the end without touching what came before), or does the harness rewrite, reorder, prune? It is a design choice that looks neutral functionally, and is anything but neutral for serving.

The implicit metric at this layer is the **signal-per-token ratio**: maximize the probability of task success while minimizing the context carried along. But every lever above has a second effect, invisible at this layer: it changes the prefix reuse pattern.

---

## KV Cache Engineering — The Cost of Trajectories

At the inference layer, every token in the context has a physical counterpart: its Key and Value tensors, computed during prefill and kept in memory for the whole generation. KV cache engineering is the set of techniques that govern where these tensors live and how much it costs to reproduce them:

**Prefix caching.** If two requests share an identical token prefix, the KV of that prefix can be computed once and reused. The benefit is twofold: you avoid recomputation (the compute cost of prefill) and you cut TTFT. The condition is strict: the prefix must be identical, byte for byte, up to the point of divergence.

**Cache tiering.** In the Dielabs nomenclature, the G1–G4 hierarchy: from GPU HBM (G1) to host memory (G2), down to local storage (G3) and remote/shared storage (G4). Tiering lets you keep the KV of "warm" prefixes outside HBM and fetch them when needed, paying transfer latency instead of recomputation — a trade-off that almost always pays off for long prefixes.

**Cache-aware routing.** In a multi-replica deployment, the benefit of prefix caching depends on where the request lands: a scheduler that routes requests with a shared prefix to the replica already holding its KV turns a potential hit into an actual hit. Without cache awareness in routing, the theoretical reuse gets diluted.

**Prefill/decode disaggregation.** Separating the two phases onto distinct resource pools, with KV transfer between them. Agentic workloads, with their huge and recurring prefills on growing prefixes, are exactly the profile that makes this architecture interesting.

The metric at this layer is marginal cost: how many FLOPs and how much bandwidth it takes to serve the next token of the trajectory, given everything already computed. And here is the point: that "given everything already computed" — the cache hit rate — is not decided by the infrastructure. It is decided by the harness.

---

## The Two-Column Frame

| | Context Engineering | KV Cache Engineering |
|---|---|---|
| **Where it lives** | Application / harness | Rack / inference stack |
| **Who owns it** | Agent engineer, platform engineer | Inference engineer, infra architect |
| **Object** | What enters the context, and when | Where the KV lives and how much it costs to re-materialize |
| **Levers** | Compaction, sub-agents, progressive disclosure, tool result design, loop structure | Prefix caching, G1–G4 tiering, cache-aware routing, P/D disaggregation |
| **Metric** | Useful signal per token, task success rate | Cache hit rate, TTFT, effective cost per token |
| **Determines** | The *shape* of trajectories | The *marginal cost* of trajectories |

The two columns look independent. The interface between them is where the economics of the workload are decided.

---

## The Interface — Three Concrete Couplings

***Compaction against prefix caching.*** Compaction is the most intuitive context engineering technique and the most hostile to the cache. Rewriting the context means invalidating the accumulated prefix: on the next request no existing KV is reusable, and prefill restarts from zero on the whole compacted context. To be precise: the compacted context in turn becomes a new stable prefix, cacheable in later turns — compaction does not make the session permanently cache-hostile, but it pays a full prefill *at every compaction event* and wipes out, each time, the value of the KV accumulated in tiers G1–G4. The trade-off can be quantified: tokens saved in all future turns against the repeated cost of these resets. A harness that compacts often and aggressively can cost more in prefill than it saves in context — and the break-even point depends on the expected remaining length of the session, something only the application layer knows.

**Sub-agents as prefix fan-out.** When an agent launches N sub-agents starting from the same base context, it creates the ideal scenario for prefix caching: one prefill, N reuses. But the benefit only materializes if the infrastructure can capture it: you need cache-aware routing that sends sub-agents to the replica holding the parent's KV, **or a shared tier (G3/G4) from which any replica can fetch it.** A cache-unaware round-robin load balancer turns N potential hits into N full prefills. Here responsibility is reversed: the trajectory shape is optimal, and it is the infrastructure that must be designed to exploit it.

**The append-only loop as a cache contract.** An agent whose context grows strictly by appending — stable system prompt, history never rewritten, tool results queued at the end — offers the infrastructure an implicit contract: every request in the loop shares with the previous one a prefix equal to the whole context minus the last delta. The hit rate tends towards very high values, and the cost of the session converges towards the incremental cost of new tokens alone. It is the opposite workload profile to multi-user chat with short, divergent contexts, and it needs completely different sizing: **less prefill compute per token served, more pressure on memory capacity and on bandwidth between tiers.**

*The common thread of the three cases: the decision is taken in the left column, the bill arrives in the right column.*

**A mini-scenario** (stylized, not measured). Two harnesses run the same task with the same task success rate, same model, same GPU. The first keeps an append-only loop: by step twenty the context is a ~40k-token prefix almost entirely in cache, and each step pays prefill only for the new tokens — a few hundred. The second compacts every three steps to keep the context under 15k tokens: at each compaction it pays a cold 15k-token prefill, six or seven times over the task. At the application layer the two harnesses are equivalent; at the rack layer, the second generates a per-session prefill load that is a multiple of the first, with worse TTFT exactly at reset time. Same application as the user sees it, two completely different workloads as the GPU sees it — and two different Cr_open values.

---

## The Cache Hit Rate Is an Input of Cr_open

In the Dielabs framework, [Cr_open](/frameworks/one-capacity-is-not-enough) expresses the SLO-anchored capacity per replica: how many requests per second a replica sustains while keeping TTFT and ITL within threshold, measured in open loop. Deriving Cr_open depends on the average cost per request — and for an agentic workload that cost is dominated by prefill, which in turn is a direct function of the cache hit rate.

The operational consequence is that Cr_open is not a property of the model + hardware pair: it is a property of the model + hardware + **harness** triple. **The same model on the same GPU can deliver per-replica capacities that differ by multiples, depending on whether traffic comes from an append-only harness behind cache-aware routing or from a harness that compacts aggressively behind a cache-unaware load balancer.** A sizing exercise that assumes a "typical" hit rate without having analyzed the application is estimating the wrong variable.

*In practice, this means that a capacity planning assessment for agentic workloads must include a phase that traditionally does not exist: reading the harness. How does it manage context? Does it compact, and how often? Does it use sub-agents, and with how much shared prefix? Are tool results truncated or injected whole? From these answers you derive an estimate of the reuse pattern, and from that — not from generic benchmarks — the expected Cr_open.*

---

## Conclusion

Context engineering and KV cache engineering are two disciplines with different owners, metrics and languages, but they describe the same object from two sides: the first draws the shape of trajectories, the second determines their marginal cost. In the chat era their separation was harmless; in the agent era it is the most expensive blind spot in the stack. The cache hit rate — the variable that more than any other decides the economics of an agentic deployment — is born in the application and paid for in the rack.

For those who design agents: every context management choice is also an infrastructure cost choice, and it is worth knowing its price. For those who size infrastructure: the point is not to optimize the model or buy more GPUs. In agentic workloads the application draws the trajectory and the rack pays for it — which is why capacity planning cannot start from the GPU datasheet. It has to start from the shape of the agent.

---

**Related:** context management strategies (skills, sub-agents, summarization, external memory) are covered in [Agentic Systems for Inference Infrastructure People](agentic-systems.md), Chapter 5, and the layers of the prefix with their expected hit rate in Chapter 8; the quality/cost axes in [Effectiveness vs Efficiency](effectiveness-vs-efficiency.md); how Cr_open uses the hit rate in [One Capacity Is Not Enough](/frameworks/one-capacity-is-not-enough).

*Original Dielabs work by Diego Bardella.*
