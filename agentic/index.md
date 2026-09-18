---
layout: default
title: Agentic
---

# Agentic <span class="new-flag">New</span>

Infrastructure for agentic systems and retrieval. Two tracks that meet in one place: how to design an agent, and what an agentic workload does to the serving stack underneath it.

This is the newest part of the lab. It reads the same problems as the inference sections from the other end: not how fast a token comes out, but how many calls a task costs, how the context grows, and who is allowed to make the calls.

---

## Two tracks

**Agents** — start from design, end at serving. What an agent is made of, then what it does to prefill, TTFT and observability once it runs in production.

**Retrieval** — start from how a RAG chain works, end at choosing its depth. Four models, two flows, and the decision of how much retrieval a use case actually needs.

---

## Documents

### [Agentic Systems for Inference Infrastructure People](agentic-systems.md)
From the stateless model to the agentic harness, written for people who know inference but not agents. The model as a pure function, tool use as structured generation plus external execution, the gather-act-verify loop, context engineering as the central discipline, the four context management strategies, MCP, reasoning policy — then what an agentic workload does to serving: prefill amplification, the bottleneck shifting from memory bandwidth to compute, TTFT times N_steps, layered prefix caching, and six families of agentic observability.

### [RAG for Dummies](rag-for-dummies.md)
How a system that answers from company documents actually works, with the library analogy: the four models of the chain (rewriter, embedding, reranker, generator), the ingestion and query flows, the judge and its metrics, chunking, and a tour of the advanced variants — hybrid search, GraphRAG, agentic, adaptive, multimodal, federated. No maturity framework and no sizing: this is the one to read before deciding how deep a retrieval system needs to go.

<div class="private-card">
  <div class="private-title-row">
    <span class="private-title">From Zero to RAG</span>
    <a class="private-pill" href="mailto:info@dielabs.eu?subject=From%20Zero%20to%20RAG%20%E2%80%94%20request" title="Request access via email">Available on request</a>
  </div>
  <p class="private-desc">The quantitative companion to the page above: a four-level framework (Production layer, Knowledge Representation, Retrieval Intelligence, Agent Orchestration) with a requirement &rarr; pattern &rarr; cost selection table, and the impact of each level on inference sizing.</p>
  <p class="private-note">Not published. Reserved for direct conversations &mdash; reach out if relevant to your context.</p>
</div>

<div class="private-card">
  <div class="private-title-row">
    <span class="private-title">Designing AI Agents from Scratch</span>
    <a class="private-pill" href="mailto:info@dielabs.eu?subject=Designing%20AI%20Agents%20%E2%80%94%20request" title="Request access via email">Available on request</a>
  </div>
  <p class="private-desc">A design handbook: the agent contract (objective, input/output, boundaries, budget, risk), the components (system prompt, context, skills, RAG, tools, deterministic code, state, memory, policy, validator, observability), the control loop, planning, security by design, human-in-the-loop, evaluation, and a production readiness checklist.</p>
  <p class="private-note">Not published. Reserved for direct conversations &mdash; reach out if relevant to your context.</p>
</div>

---

*All content is original Dielabs work by Diego Bardella.*