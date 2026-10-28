---
redirect_from:
  - /frameworks/inference-technology-model
  - /frameworks/inference-technology-model.html
  - /frameworks/inference-stack-model
  - /frameworks/inference-stack-model.html
  - /frameworks/ai-use-case-discovery
  - /frameworks/ai-use-case-discovery.html
title: Frameworks
layout: default
---

# Frameworks

Proprietary frameworks for reasoning about LLM inference systems. Three of them follow the life of a deployment, from sizing to production; the other is the reference model the lab uses to structure analysis, design and engineering work.

---

## The Cycle

### 01 · [From Idea to Production](from-idea-to-production.md)
An 11-step methodology that goes from a business need to an empirically validated inference deployment. Distinguishes customer inputs (use case, workload, traffic, SLO) from the architectural response (model, sizing, runtime, hardware, stack) and closes the loop with benchmark and conscious scaling.

### 02 · [One Capacity Is Not Enough](one-capacity-is-not-enough.md)
The Dielabs benchmark framework. Why a single capacity number does not exist, and how CrossP (the regime boundary), Cr_closed (hardware-anchored, closed loop) and Cr_open (SLO-anchored, open loop) break it down into three measurable quantities, anchored to the three vLLM benchmarks. The output is a capacity card, not a number.

### 03 · [Observability KPI](observability-kpi.md)
A monitoring, diagnostics and incident response framework for LLM inference systems built on vLLM + Prometheus + Grafana + DCGM. Covers the golden metrics (TTFT, ITL, TPOT, E2E), the TPOT vs ITL distinction, Observed vs Compute throughput, percentile statistics, diagnostic tree from symptom to root cause, and operational PromQL queries.

---

## Reference Model

### [LLM Parameter Topology](llm-parameter-topology.md)
A structured framework for understanding where every LLM parameter lives: Artifact, Startup, or Request. Covers the full parameter flow from model weights to runtime enforcement, with conflict zones and troubleshooting tables.

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
  <p class="private-desc">The full presales path, from first contact to delivery in one document: commercial qualification with stakeholder map and red flags, use-case discovery, a Conceptual Model with requirements, constraints, assumptions and risks, physical design and BOM, the offer with the benchmark as platform acceptance, handover to delivery, production and drift.</p>
  <p class="private-note">Not published. Reserved for direct conversations &mdash; reach out if relevant to your context.</p>
</div>

---

*All frameworks are original Dielabs work by Diego Bardella.*
