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

### [Benchmarking Protocol](benchmarking-protocol.md)
A three-phase LLM inference validation methodology (Sweep, Concurrent, SLO Mapping) built on GuideLLM. Identifies the Crossover Point between latency and throughput, and produces a Capacity Card for each deployment configuration.

---

*All frameworks are original Dielabs work by Diego Bardella.*
