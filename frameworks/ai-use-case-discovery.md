---
layout: default
title: "AI Use-Case Discovery"
---

# AI Use-Case Discovery

> How to qualify an AI use case before anything gets sized: the value chain it must close, the five coordinates that place a system between a chat and an agent, and the eight discovery areas that turn a customer's idea into inputs a design can rest on. The front end of [From Idea to Production](from-idea-to-production.md).

## Key Points

- Sizing starts too late if it starts from the model. Before any GPU math, discovery has to establish, in this order: whether the problem is worth solving, which system it actually needs, whether it is feasible, how it will be measured, and who decides whether it goes to production.
- A use case is valid only when it closes three links at once: business value, operating value and technical value. If one is missing, the project is not ready to start.
- "Agentic" is not one scale. A system is placed on five independent coordinates — flow autonomy, knowledge access, flow control, action level, supervision. Their combination, not any single one, drives complexity, cost, risk, governance and infrastructure sizing.
- The acceptance method is agreed **before** the PoV starts. Without a shared measure, acceptance stays negotiable and the PoV never closes.

---

## 1. The Principle — Close the Value Chain

| Link | The question | Without it |
|---|---|---|
| **Business value** | Which KPI improves, and what is the improvement worth? | A solution looking for a problem |
| **Operating value** | How does the process work today, who runs it, where does it lose time, where must a person step in? | You don't know what to automate |
| **Technical value** | Which data, integrations and constraints make it feasible? | The value stays theoretical |

---

## 2. Five Coordinates of an Agentic AI System

The coordinates are **independent**: they combine, but one does not imply another. Do not collapse them into a single "maturity" scale.

### Axis 1 — Flow autonomy
How many steps the system can take before returning to the user.

| Level | Description |
|---|---|
| **L1 — Chat** | Produces a text answer, no tools. |
| **L2 — Chat with tools** | One cycle: request → tool choice → result → answer. The user waits. |
| **L3 — Free-loop agent** | Chains tools on its own and decides when to stop. |
| **L4 — Orchestrated workflow** | Follows a path designed upfront; AI works inside each step. |
| **L5 — Orchestrated multi-agent** | Coordinates specialized agents, with state, retries and resumable processes. |

**Minimum sufficient level.** Never climb a level without a process need. Every step up adds cost, error surface and governance burden.

**Autonomy is not retrieval depth.** A system can run an advanced RAG, call the model several times, rewrite and verify answers — and still be **L2 on axis 1**, as long as everything happens inside one user request.

### Axis 2 — Knowledge access
What the system can know beyond the base model: (1) model knowledge only, (2) manually provided context, (3) dynamic retrieval via RAG, (4) live access to systems such as CRM, ERP and databases. Independent from axis 1: an L2 can have RAG and an L5 may not.

### Axis 3 — Flow control
Who decides the next step. **Model-driven** (typical of L3): variable path, harder to reproduce and audit. **Logic-driven** (typical of L4–L5): path fixed at design time, more predictable, more engineering. **Hybrid**: fixed structure, bounded freedom at specific points.

> High potential damage with a model-driven flow is usually a bad first PoV.

### Axis 4 — Action level
What the system can change: (1) nothing, (2) read data, (3) modify data, (4) operational actions, (5) high-impact or irreversible actions. The harder an action is to undo, the more supervision, traceability and control it needs.

### Axis 5 — Supervision and execution
**Supervision:** recommendation → human approval (HITL) → human oversight (HOTL) → autonomous execution. **Execution:** synchronous (someone waits) or asynchronous/background (triggered by an event or a schedule). Background execution changes who detects errors, how load is sized and who owns monitoring.

> Autonomy + background + irreversible actions is the maximum risk profile, and rarely fits a first PoV.

---

## 3. Eight Discovery Areas

Each area ends with the rule it enforces.

### A. Process and economic value
How the process works without AI, who runs it, how often, where errors and rework happen, what a failure costs. Why now, what has been tried, whether a budget exists or the business case must be built.
**Rule:** no project starts without an economic estimate, even a rough one.

### B. Nature of the task
Answer or action? Is the base model enough, or are documents, RAG or live data needed? One step or a chain — and if a chain, stable or case-by-case? Deterministic or interpretive output, structured or free, automatically verifiable?
**Rule:** the shape of the flow places the system on axis 1 (one step → L2, model-chosen chain → L3, designed chain → L4, coordinated agents with state → L5). Fine-tuning teaches behavior and format; it does not replace an up-to-date knowledge source.

### C. Risk, actions and autonomy
Worst possible damage, what happens if nobody notices, how fast it must be caught. For each action: what it changes, whether it is reversible, who approves, who is accountable. Can inputs, decisions, tools, model, prompt and data versions be reconstructed? In background mode, who detects a failure and how fast?
**Rule:** a non-deterministic path requires *more* traceability, not more trust. Mandatory in legal, HR, medical and financial domains — even when the system is read-only.

### D. Data and privacy
Which data, where it lives, who owns it, how often it changes. Native PDFs, scans, Office files, email, structured data. Personal or confidential? Can everyone see everything? Cloud, private cloud or on-premise constraints?
**Rule:** if not everyone can see everything, retrieval must enforce permissions — raising ingestion complexity, technical complexity and governance risk together. Low-quality scans can sink a PoV before the model is ever involved.

### E. Integrations and technical constraints
APIs, MCP, webhooks, databases, queues; legacy systems limited to CSV or batch; authentication, service accounts, network constraints or air gap; whether on-premise is mandatory; whether the model handles tool calling reliably.
**Rule:** a web UI with connectors covers many L2 cases. L4 and L5 need a real orchestration engine — not a configuration.

### F. Volumes, load and SLOs
Tasks per day, average and peak, concurrent users, steady or bursty traffic. Synchronous or background, acceptable latency, windows where the system cannot slow down. How much the context grows per session, how many flow steps and how many model calls per request.
**Rule:** this area qualifies, it does not compute — its answers become the workload, traffic and SLO inputs of [From Idea to Production](from-idea-to-production.md). Flow steps and model calls are not the same count: an L2 with heavy retrieval can cost more than a simple L4. In true agentic systems the number of calls is a distribution, so size on mean fan-out, variance, p95/p99, step limit, timeouts and retries. **Sizing an agent is not sizing a RAG.**

### G. Success, adoption and decision
Which KPI defines success and at what threshold; who approves production, who sponsors, who buys, who uses it daily, who runs it after go-live; what would lead to a "no".
**Rule:** buyer, sponsor and daily user are often different people. Value only materializes if the people who use the process actually adopt it.

### H. Evaluation and acceptance
What "correct" means, who defines it, whether you judge only the result or also the path (from L3 on: tool chosen, step sequence, wasted steps, cost, time, recovered errors). A **golden set** of real cases with expected answers, coverage, owners, version and sign-off. Judging by human experts, automatic rules, LLM-as-judge or a mix — with LLM-as-judge validated by sampling. Baseline of the human process, metric, threshold, calculation method, exclusions and final decision owner, all agreed before kick-off.
**Rule:** the golden set is the technical contract of the PoV and, later, its regression suite. Compare the AI with the real human process, not with theoretical perfection.

---

## Where This Fits

Discovery answers *what to build and how it will be judged*. The answers in areas B and F hand off to [From Idea to Production](from-idea-to-production.md) for sizing; the platform is then accepted with the benchmark method in [One Capacity Is Not Enough](one-capacity-is-not-enough.md). Answer quality and platform performance are accepted separately: the golden set for the first, the benchmark for the second.

---

*Original Dielabs work by Diego Bardella.*
