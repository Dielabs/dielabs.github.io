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

---

*All frameworks are original Dielabs work by Diego Bardella.*
