---
layout: default
title: "The Agentic Ecosystem — MCP, A2A and Observability Tooling"
---

# The Agentic Ecosystem — MCP, A2A and Observability Tooling

*The technology companion of [From the Stateless Model to the Agentic Harness](agentic-systems.md). Reference date: April 2026.*

This is the part of the agentic corpus that ages. Products, protocols, versions and tools: it gets rewritten when the landscape moves. The foundations — the model as a pure function, tool use, the loop, context engineering, reasoning policy, the implications for serving — live in [From the Stateless Model to the Agentic Harness](agentic-systems.md) and are not repeated here.

---

## MCP in practice

The problem MCP solves and its minimal architecture are covered in [From the Stateless Model to the Agentic Harness](agentic-systems.md). The operational side is in [MCP — The Integration Protocol for Agentic Platforms](mcp-integration-protocol.md): the end-to-end flow with a concrete example (an agent connected to Postgres), why MCP is just ordinary tool use from the model's point of view, the compatibilities to validate, transports, deployment topologies, governance, identity and attack vectors. It is not repeated here.

---

## Agent-to-Agent: state of the art and open tensions

**Key points.**

- **A2A** (Agent-to-Agent) is the idea of letting agents from different vendors talk to each other over a standard protocol
- Google promoted A2A in 2025 as a horizontal complement to MCP
- In the real production systems we can observe today, the winning patterns **look hierarchical**, not peer-to-peer
- A peer-to-peer A2A mesh **has not yet shown widespread operational value** — which is not the same as saying "it will never work"
- Four non-trivial problems remain open: identity, cost accounting, latency, error semantics
- Some domains (federated enterprise workflows, sovereign AI with data residency) are exploring federated orchestration; it is an evolving area

### The A2A dream

The idea, in its marketing version:

> "Imagine an agent that manages your email. Another one that manages your calendar. A third that books travel. All from different vendors. They talk to each other over a standard protocol and coordinate to organize your business trip, with no central orchestrator."

It is an attractive vision. The four problems below are not proof that A2A "will never work" — they are real operational obstacles that today push production systems toward hierarchical patterns. Several initiatives (Google's A2A protocol, IBM research on agent communication, European sovereign AI consortia) are trying to tackle them. They are worth knowing so as not to talk about A2A as if it were a mature solution.

### Problem 1 — Identity and trust

When agent A calls agent B:

- Who is A? With which credentials does it present itself to B?
- How does B verify that A is authorized to ask for what it asks?
- If A asks B to perform an expensive or sensitive action, should B trust it?
- Is the identity that of A's end user, or of A itself? How is delegation handled?

The enterprise world has OAuth, SAML, OIDC. But they are designed for **human users authenticating applications**, not for **autonomous agents invoking other agents**. The semantics of delegation are different.

A concrete example: your banking agent receives a request from the travel agent of an external collaborator. Should it execute it? Even if it is "on your behalf"? Within which limits?

No A2A protocol answers these questions satisfactorily today.

### Problem 2 — Cost accounting

If A calls B, which calls C, which calls an LLM, who pays for the tokens?

- A? Then A must know how much B costs before calling it (impossible upfront for open-ended tasks)
- B? Then B pays to serve A. A hostile business model.
- Pay-per-call at fixed rates? It works for traditional REST APIs, not for LLM calls with variable costs

Even internally, Anthropic solves this by **keeping everything inside a single orchestrator**: parent and subagents are part of the same process, and the same user pays. A subagent is NOT "another vendor" — it is a subroutine.

### Problem 3 — Latency stacking

Every hop between agents adds:

- Network latency (10–100 ms and more)
- The latency of the called model (several seconds)
- Protocol parsing/validation latency

A chain A → B → C → LLM can easily add up to 10+ seconds of overhead. For interactive tasks that is unacceptable. For batch tasks it is fine, but then a traditional pipeline is enough.

**A frequent anti-pattern**: A2A is proposed as "modular and scalable", but in practice:

- A local subagent in a coding agent has an overhead of hundreds of milliseconds
- A remote subagent over A2A would have an overhead of seconds
- A chain of three A2A hops is impractical for interactive use

### Problem 4 — Error semantics

If B fails while serving A, what happens?

- A receives a generic error ("B has failed")? Useless for debugging.
- A receives a structured error that allows retry/fallback? Then A must know B's error semantics, and we are back to the N×M problem.
- A receives the error in natural language and interprets it? It works, but it is fragile and hostile to debugging.

On top of that: if B fails **halfway through** an operation with side effects (for example, it has already sent 3 emails out of 5), how do you recover? Distributed transactions have never been easy. With agents performing open-ended actions, they are even harder.

### What the mature systems do instead — hierarchy plus MCP

The pattern visible in Claude Code is hierarchical:

```
        PARENT AGENT (orchestrator)
       /          |           \
   Subagent    Subagent     Subagent
   (Explore)   (Plan)       (Bash)
       |          |           |
       └─ MCP ────┴─── MCP ───┘
              (tool layer)
```

- The "children" are **subagents of the same process**, not external agents
- All the fan-out is controlled by the parent
- The "horizontal" interface is MCP (the tool layer), not A2A
- Identity, cost, latency and error semantics are all handled centrally by the parent

No peer-to-peer A2A protocol is involved in this design. It is a precise architectural choice, not a universal law — but it is significant, because Anthropic has every technical and organizational means to implement A2A if it considered it strategic.

### Where A2A could make sense

A2A has concrete potential in **specific scenarios** where the operating conditions offset the open problems.

**Scenario 1: inter-domain handoff with pre-established trust**

Example: an insurance agent that, on receiving a medical reimbursement request, hands the task to the attending physician's agent for validation. The two domains (insurance, healthcare) are distinct, have pre-established trust through B2B contracts, and the handoff is explicit (not continuous peer orchestration). Here A2A is a natural direction.

**Scenario 2: federation between sovereign deployments**

Example: a platform serves customer A in Italy and customer B in Germany. The two deployments cannot share context (data residency, sovereignty). An agent for A can send tasks to an agent for B over a controlled A2A protocol, with full audit. Here A2A is almost mandatory, because you cannot unify the state.

**Scenario 3: federated enterprise workflows**

Example: several enterprise platforms are exploring orchestration where agents from different departments (with different systems of record and different privacy boundaries) coordinate process workflows. It is an area of active experimentation in 2025–2026, with the outcome still open.

In all these cases, A2A is **structured, audited handoff**, not **continuous peer-to-peer orchestration**. The distinction matters.

### State of the art in short

Peer-to-peer A2A, as a general pattern for composing agents, has not yet established itself in production: the operational obstacles on identity, cost accounting, latency and error semantics are real and unsolved. In the mature production systems we can observe today — Claude Code, Cursor, Devin — the dominant pattern is hierarchical: a parent orchestrator with child subagents, all inside the same trust domain. The only horizontal protocol is MCP, which is agent-to-tool, not agent-to-agent.

That said, A2A is an evolving area. It makes structural sense in specific scenarios: inter-domain handoff with pre-established trust (regulated B2B), federation between sovereign deployments with separate data residency, departmental enterprise workflows with privacy boundaries. The question is not "A2A yes or no", but "when does the value of federation exceed the operational cost of peer coordination". Today, for general-purpose use cases, the answer is almost always "no". For specific use cases, it is "maybe, and it deserves exploration".

---

## The 2026 tooling stack

The six families of agentic telemetry — what to measure and why — are covered in [From the Stateless Model to the Agentic Harness](agentic-systems.md). Here are the tools that implement them.

The space is consolidating. An indicative categorization (the space moves fast, and the uncertainty is stated):

**Commercial tracing and observability platforms:**

- **LangSmith** (LangChain): the pioneer, strong on trace navigation, debug-first UX
- **Langfuse**: open source first, focused on evaluation plus tracing
- **Weights & Biases Weave**: part of the W&B stack, integrated with the ML life cycle
- **Arize Phoenix / Helicone / Honeycomb**: other players, each with a different angle

**Standardization:**

- **OpenTelemetry GenAI semantic conventions**: stabilizing, the interoperable pivot. Worth betting on for the long term.
- **OpenInference**: an Arize project for standardized agentic spans
- **MCP trace propagation**: support still evolving across MCP implementations

**Cost and usage:**

- Almost always custom-built on top of the tracing platforms
- The Anthropic and OpenAI APIs return `usage` in every response, the basis for aggregation

**Replay and debugging:**

- LangSmith has a good replay UI
- Open source: limited, still an experimentation ground

**For a sovereign platform specifically**: you will most likely need a **proprietary observability stack** (for data residency and sovereignty), built on the OTel GenAI conventions, with a UI inspired by the LangSmith pattern but deployable on-prem. It is an architectural product component, not just a SaaS integration.

---

*All content is original Dielabs work by Diego Bardella.*
