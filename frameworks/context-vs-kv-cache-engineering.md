---
layout: default
title: Context Engineering vs KV Cache Engineering
redirect_from:
- /agentic/context-vs-kv-cache-engineering
- /agentic/context-vs-kv-cache-engineering.html
---

# Context Engineering vs KV Cache Engineering

> Why designing an agent is a capacity planning decision.

## Key Points

The cost and performance of an agentic LLM workload are decided at two distinct layers of the stack: the application layer, where **context engineering** decides *what* enters the context window and in what shape, and the infrastructure layer, where **KV cache engineering** decides *how much it costs* to materialize and re-materialize that context in GPU memory. The two layers are owned by different roles that rarely talk to each other, but they are tightly coupled: every agent design choice — compaction, sub-agents, loop structure — changes the cache reuse pattern, and therefore the marginal cost per token, the TTFT and ultimately the infrastructure sizing. The harness shapes potential prefix reuse; the runtime, available cache and routing determine the hit rate actually achieved. These are conditions of the measured **Cr_open**, the SLO-anchored capacity per replica. Sizing requires understanding both the agent's behaviour and the serving deployment.

---

## Two Disciplines That Don't Talk

Whoever designs the agent decides the cost of serving without knowing it. Whoever pays that cost — whoever sizes the infrastructure — gets the bill without knowing which decisions produced it. This page is about the missing interface between the two.

In recent months the public debate on agentic AI has moved from prompt engineering to context engineering: how to structure an agent's context window so that it holds the most useful signal with the least noise. It is a conversation that lives entirely at the application layer — agent frameworks, harnesses, orchestration — and typically ignores what happens underneath: how that context is materialized in GPU memory, reused across requests, moved between storage tiers.

Symmetrically, people who do inference serving reason about prefix caching, KV offload, cache-aware routing and disaggregation, but treat the workload as an external given: requests arrive with certain input/output length distributions and a certain prefix reuse rate, and you optimize on those. Where that pattern comes from — which application design choices produced it — stays out of scope.

This separation was tolerable in the chat era: relatively short sessions, context dominated by the system prompt and a few turns, simple reuse patterns. **With agentic workloads it becomes a design error.** An agent working for tens of minutes on a task generates trajectories with very long, growing prefixes, sub-agent fan-out, massive tool result injection: the shape of these trajectories — decided entirely at the application layer — is the dominant factor in serving cost. **The two worlds need to sit at the same table.**

---

## Context Engineering — The Shape of Trajectories

Context engineering is the set of techniques the application harness uses to decide what enters the context window, when, and in what form. The main levers, seen from the side of whoever will have to serve that traffic:

**Compaction.** When the context approaches its limit, the harness summarizes it and restarts from a compressed version. It reduces the tokens carried into later turns, but rewrites the history: prefix matching stops at the first modified token, while a stable initial prefix — typically the system prompt — can still be reused.

**Sub-agents and fan-out.** A complex task is broken down by delegating parts of it to sub-agents, each of which typically starts from a shared subset of the parent's context (instructions, task state) plus a specific delta. It is an isolation technique: the sub-agent works in its own clean context and returns only the result, without polluting the main context.

**Progressive disclosure (skills, lazy documentation).** Instead of loading all potentially useful material upfront, the harness exposes lightweight indexes to the model and loads full content only when needed. The context grows on demand, by appending.

**Tool result design.** How verbose a tool's output is, and whether it gets truncated, structured or summarized before entering the context. In an agent loop, tool results are often the dominant part of the context — far more than the system prompt or user messages.

**Loop structure.** The deepest choice: does the context grow strictly append-only (each turn adds at the end without touching what came before), or does the harness rewrite, reorder, prune? It is a design choice that looks neutral functionally, and is anything but neutral for serving.

The implicit metric at this layer is the **signal-per-token ratio**: maximize the probability of task success while minimizing the context carried along. But every lever above has a second effect, invisible at this layer: it changes the prefix reuse pattern.

---

## KV Cache Engineering — The Cost of Trajectories

At the inference layer, every token in the context has a physical counterpart: its Key and Value tensors, computed during prefill and kept in memory for the whole generation. KV cache engineering is the set of techniques that govern where these tensors live and how much it costs to reproduce them:

**Prefix caching.** If two requests share an identical token prefix, the KV of that prefix can be computed once and reused. The benefit is twofold: you avoid recomputation (the compute cost of prefill) and you cut TTFT. The condition is strict: the prefix tokens must match, along with whatever else the runtime includes in the cache identity (adapters, multimodal inputs), and the KV must still exist and be reachable by the replica serving the request. Identical text alone does not describe every condition.

**Cache tiering.** In the Dielabs nomenclature, the G1–G4 hierarchy: from GPU HBM (G1) to host memory (G2), down to local storage (G3) and remote/shared storage (G4). Tiering lets you keep the KV of "warm" prefixes outside HBM and fetch them when needed, paying transfer latency instead of recomputation — a trade-off to measure, not assume: retrieval time, bytes moved and contention against the recomputation avoided, weighted by the chance that the prefix is requested again before eviction.

**Cache-aware routing.** In a multi-replica deployment, the benefit of prefix caching depends on where the request lands: a scheduler that routes requests with a shared prefix to the replica already holding its KV turns a potential hit into an actual hit. Without cache awareness in routing, the theoretical reuse gets diluted.

**Prefill/decode disaggregation.** Separating the two phases onto distinct resource pools, with KV transfer between them. Agentic workloads, with their huge and recurring prefills on growing prefixes, are exactly the profile that makes this architecture interesting.

The metric at this layer is marginal cost: how many FLOPs and how much bandwidth it takes to serve the next token of the trajectory, given the reusable KV available to the serving replica. The harness determines which prefixes could be reused; cache identity, retention, placement and routing determine which are actually reused. The realized hit rate is a result of both layers.

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

***Compaction against prefix caching.*** Compaction reduces the context carried into later turns, but can interrupt prefix reuse. Rewriting the history breaks matching from the first modified point: a stable initial prefix — typically the system prompt — can remain reusable if its KV is available. The rewritten suffix requires prefill for any part that has no matching cached KV. The amount of recomputation depends on where the rewrite starts, the size of the new context and the cache available to the serving replica; a large or fully cold prefill is not an inevitable consequence of compaction. The compacted context can itself become a reusable prefix in subsequent turns, and the rewrite does not imply physical deletion of the previous KV. Compare the summarization cost and subsequent recomputation with the savings over the remaining turns, including decode and memory costs. Task quality belongs in that comparison too, since a summary can drop useful information. Whether compaction pays off depends on the workload and expected remaining session length.

**Sub-agents as prefix fan-out.** When an agent launches N sub-agents starting from the same base context, it creates a favourable scenario for prefix caching: in the best case, one prefill and N reuses — provided the sub-agents really share the same token sequence and the blocks are still available. Simultaneous starts, different templates or a cold cache can duplicate work. But the benefit only materializes if the infrastructure can capture it: you need cache-aware routing that sends sub-agents to the replica holding the parent's KV, **or a shared tier (G3/G4) from which any replica can fetch it.** A cache-unaware round-robin load balancer reduces locality and can turn many of those potential hits into recomputation; affinity to a warm replica helps, but must be balanced against that replica's queue, and transfers from a shared tier can cancel the gain. Here responsibility is reversed: the trajectory shape is favourable, and it is the infrastructure that must be designed to exploit it.

**The append-only loop as a cache contract.** An agent whose context grows strictly by appending — stable system prompt, history never rewritten, tool results queued at the end — offers the infrastructure an implicit contract: every request in the loop shares with the previous one a prefix equal to the whole context minus the last delta. The potential hit rate is very high, and the prefill cost of each step can approach the cost of the new tokens alone — subject to template stability, eviction, pauses and how requests are spread across replicas. The context keeps growing, though: memory, decode cost and noise grow with it, and a high hit rate does not prove that the loop beats a strategy that compacts and solves the same task with fewer resources overall. It is the opposite workload profile to multi-user chat with short, divergent contexts, and it needs completely different sizing: **less prefill compute per token served, more pressure on memory capacity and on bandwidth between tiers.**

*The common thread of the three cases: the decision is taken in the left column, the bill arrives in the right column.*

**A mini-scenario** (stylized, not measured). Suppose two harnesses achieve the same task success rate with the same model and GPU. The first keeps an append-only loop whose context reaches 40k tokens, with its earlier KV retained on the serving replica. Each turn then requires prefill for the newly appended tokens, while decode still operates over the growing context. The second compacts every three steps to a 15k-token context. Assume 2k tokens form an unchanged, cached initial prefix and the remaining 13k-token suffix has no matching cached KV: the first request after each compaction must prefill that suffix, and later requests can reuse it if retained. This illustrates a possible recomputation cost, not a guaranteed ranking. Compare summarization, prefill, decode, transfers, memory and task quality over the full run, then measure TTFT and Cr_open under representative traffic. Different cache conditions or longer remaining sessions can change the result.

---

## Cache Reuse Is a Condition of Cr_open

In the Dielabs framework, [Cr_open](/frameworks/one-capacity-is-not-enough.html) expresses the SLO-anchored capacity per replica: how many requests per second a replica sustains within the declared latency targets, measured in open loop. Cache reuse is one condition of that measurement, alongside arrivals, input/output lengths, runtime and routing. Prefill can dominate some agentic workloads, but an assumed hit rate alone does not determine request cost or capacity.

The operational consequence is that Cr_open is not a property of the model + hardware pair alone. Its conditions include the **harness**, serving configuration, traffic profile and SLOs. **The same model on the same GPU can deliver different per-replica capacities when context management, cache retention or routing changes.** The size of that difference must be measured; neither append-only history nor compaction guarantees the better result. Serving requests/s also differ from completed tasks/s: a task may make several dependent or parallel inference calls, so validate task capacity end to end.

*In practice, an assessment must characterize the harness: how it manages context, how often it compacts, how sub-agents share prefixes and how tool results enter the context. Use those answers to build a representative traffic profile, preserving relevant prefixes and pauses, then measure Cr_open on the target serving configuration. A generic benchmark remains a useful baseline, but cannot establish capacity for that workload by itself.*

---

## Conclusion

Context engineering and KV cache engineering are two disciplines with different owners, metrics and languages, but they describe the same workload from two sides: the first shapes trajectories and opportunities for reuse, the second determines how that reuse is realized and what it costs. Their interaction matters for agentic deployments: cache hit rate, latency and total cost emerge from both application choices and serving conditions, and must be assessed alongside task quality.

For those who design agents: every context management choice is also an infrastructure cost choice, and it is worth knowing its price. For those who size infrastructure: the point is not to optimize the model or buy more GPUs. In agentic workloads the application draws the trajectory and the rack pays for it — which is why capacity planning cannot start from the GPU datasheet. It has to start from the shape of the agent.

---

**Related:** [Minimum Requirements of an Agentic Harness](agentic-harness-requirements.md) covers memory, control and verification; [One Capacity Is Not Enough](/frameworks/one-capacity-is-not-enough.html) defines SLO-anchored capacity and how cache reuse affects it.

*Original Dielabs work by Diego Bardella.*
