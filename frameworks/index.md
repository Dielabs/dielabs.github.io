---
title: Frameworks
layout: default
---

# Frameworks

Proprietary frameworks for reasoning about LLM inference systems. Four of them follow the life of a deployment, from the first customer conversation to production; the others are the reference models the lab uses to structure analysis, design and engineering work.

---

## The Cycle

### 01 · [AI Use-Case Discovery](ai-use-case-discovery.md)
How to qualify an AI use case before anything gets sized: the business–operating–technical value chain, five independent coordinates that place a system between a chat and an agent (autonomy, knowledge access, flow control, action level, supervision), and eight discovery areas from economic value to acceptance. The golden set is agreed before the PoV starts.

### 02 · [From Idea to Production](from-idea-to-production.md)
An 11-step methodology that goes from a business need to an empirically validated inference deployment. Distinguishes customer inputs (use case, workload, traffic, SLO) from the architectural response (model, sizing, runtime, hardware, stack) and closes the loop with benchmark and conscious scaling.

### 03 · [One Capacity Is Not Enough](one-capacity-is-not-enough.md)
The Dielabs benchmark framework. Why a single capacity number does not exist, and how CrossP (the regime boundary), Cr_closed (hardware-anchored, closed loop) and Cr_open (SLO-anchored, open loop) break it down into three measurable quantities, anchored to the three vLLM benchmarks. The output is a capacity card, not a number.

### 04 · [Observability KPI](observability-kpi.md)
A monitoring, diagnostics and incident response framework for LLM inference systems built on vLLM + Prometheus + Grafana + DCGM. Covers the golden metrics (TTFT, ITL, TPOT, E2E), the TPOT vs ITL distinction, Observed vs Compute throughput, percentile statistics, diagnostic tree from symptom to root cause, and operational PromQL queries.

---

## Reference Models

### [The LLM Inference Stack Model](inference-stack-model.md)
A layered model of an LLM inference system, from physical hardware (L0) to the client (L6). Each layer does one thing and enables the one above. The conceptual map used across the lab to reason about where every component sits and how dependencies flow.

### [Inference Technology Model](inference-technology-model.md)
A competency framework (Layers A–G) mapping what an inference engineer needs to know and operate. Maps skills rather than components — deliberately cuts across multiple L-layers.

### [LLM Parameter Topology](llm-parameter-topology.md)
A structured framework for understanding where every LLM parameter lives: Artifact, Startup, or Request. Covers the full parameter flow from model weights to runtime enforcement, with conflict zones and troubleshooting tables.

### [Workload Characterization in Disaggregation](workload-characterization-disaggregation.md)
Seven discovery questions about the workload (ISL/OSL, prefix reuse, multi-turn, arrival pattern, SLO, multi-model, growth) and the architectural decision each answer supports in the Disaggregation OS. Read this before sizing a disaggregated deployment without traffic data.

<div class="private-card">
  <div class="private-title-row">
    <span class="private-title">From Idea to Production &mdash; the manual</span>
    <a class="private-pill" href="mailto:info@dielabs.eu?subject=From%20Idea%20to%20Production%20%E2%80%94%20request" title="Request access via email">Available on request</a>
  </div>
  <p class="private-desc">The full operational manual behind the <em>Inference Sizing in 11 Steps</em> framework. End-to-end presales architect playbook: from business pain statement to validated production deployment, with discovery templates, sizing worksheets, runtime decision matrices, hardware constraint tables, benchmark protocols, and scaling decision trees. Each of the 11 steps is expanded into actionable artifacts usable in real customer engagements.</p>
  <p class="private-note">Not published. Reserved for direct conversations &mdash; reach out if relevant to your context.</p>
</div>

<div class="private-card">
  <div class="private-title-row">
    <span class="private-title">AI Technical Presales Framework</span>
    <a class="private-pill" href="mailto:info@dielabs.eu?subject=AI%20Technical%20Presales%20Framework%20%E2%80%94%20request" title="Request access via email">Available on request</a>
  </div>
  <p class="private-desc">The full presales path that <em>AI Use-Case Discovery</em> is taken from, from first contact to delivery in one document: commercial qualification with stakeholder map and red flags, use-case discovery, a Conceptual Model with requirements, constraints, assumptions and risks, physical design and BOM, the offer with the benchmark as platform acceptance, handover to delivery, production and drift.</p>
  <p class="private-note">Not published. Reserved for direct conversations &mdash; reach out if relevant to your context.</p>
</div>

---

*All frameworks are original Dielabs work by Diego Bardella.*
