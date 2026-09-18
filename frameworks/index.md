---
title: Frameworks
layout: default
---

# Frameworks

Proprietary frameworks for reasoning about LLM inference systems. These define the conceptual models used throughout the lab to structure analysis, design and engineering work.

---

## Documents

### [The LLM Inference Stack Model](inference-stack-model.md)
A layered model of an LLM inference system, from physical hardware (L0) to the client (L6). Each layer does one thing and enables the one above. The conceptual map used across the lab to reason about where every component sits and how dependencies flow.

### [Inference Technology Model](inference-technology-model.md)
A competency framework (Layers A–G) mapping what an inference engineer needs to know and operate. Maps skills rather than components — deliberately cuts across multiple L-layers.

### [LLM Parameter Topology](llm-parameter-topology.md)
A structured framework for understanding where every LLM parameter lives: Artifact, Startup, or Request. Covers the full parameter flow from model weights to runtime enforcement, with conflict zones and troubleshooting tables.

### [From Idea to Production](from-idea-to-production.md)
An 11-step methodology that goes from a business need to an empirically validated inference deployment. Distinguishes customer inputs (use case, workload, traffic, SLO) from the architectural response (model, sizing, runtime, hardware, stack) and closes the loop with benchmark and conscious scaling.

### [Observability KPI](observability-kpi.md)
A monitoring, diagnostics and incident response framework for LLM inference systems built on vLLM + Prometheus + Grafana + DCGM. Covers the golden metrics (TTFT, ITL, TPOT, E2E), the TPOT vs ITL distinction, Observed vs Compute throughput, percentile statistics, diagnostic tree from symptom to root cause, and operational PromQL queries.

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

---

*All frameworks are original Dielabs work by Diego Bardella.*
