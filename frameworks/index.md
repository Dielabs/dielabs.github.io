---
title: Frameworks
layout: default
---

# Frameworks

Proprietary frameworks for reasoning about LLM inference systems. These define the conceptual models used throughout the lab to structure analysis, diagnostics and engineering work.

---

## Documents

### [Inference Diagnostic Framework](inference-diagnostic-framework.md)
A 7-layer architectural model (L0–L6) mapping every component involved in serving an LLM — from physical hardware to the client application. Used to understand system boundaries, ownership and root-cause analysis.

### [Inference Technology Model](inference-technology-model.md)
A competency framework (Layers A–G) mapping what an inference engineer needs to know and operate. Maps skills rather than components — deliberately cuts across multiple L-layers.

### [LLM Parameter Topology](llm-parameter-topology.md)
A structured framework for understanding where every LLM parameter lives: Artifact, Startup, or Request. Covers the full parameter flow from model weights to runtime enforcement, with conflict zones and troubleshooting tables.

### [LLM Inference Sizing in 10 Steps](inference-sizing-10-steps.md)
A 10-step methodology that goes from a business need to an empirically validated inference deployment. Distinguishes customer inputs (use case, workload, SLO) from architectural response (model, sizing, runtime, hardware, stack) and closes the loop with benchmark and conscious scaling.

### [Observability KPI](observability-kpi.md)
A monitoring, diagnostics and incident response framework for LLM inference systems built on vLLM + Prometheus + Grafana + DCGM. Covers the golden metrics (TTFT, ITL, E2E), Observed vs Compute throughput, percentile statistics, diagnostic tree from symptom to root cause, and operational PromQL queries.

---

*All frameworks are original Dielabs work by Diego Bardella.*
