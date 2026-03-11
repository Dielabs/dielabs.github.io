# Knowledge Base

Reference documentation on LLM inference engineering, produced from real lab work at Dielabs.

These documents cover the operational and infrastructural side of AI inference — not model training or data science.

---

## Documents

### [LLM Parameter Topology](inference-parameters.md)
A structured framework for understanding where every LLM parameter lives: Artifact, Startup, or Request. Covers the full parameter flow from model weights to runtime enforcement, with conflict zones, troubleshooting tables and a mental checklist for inference engineers. Applies to vLLM, TGI, TensorRT-LLM, SGLang, Ollama and any OpenAI-compatible engine.

### [Inference Observability](inference-observability.md)
Monitoring, diagnostics and incident response for LLM inference systems. Covers the golden metrics (TTFT, ITL, E2E), percentile statistics and the echo effect, a root-cause diagnostic tree, operative PromQL queries for Grafana, log cross-checks and a pre-benchmark checklist. Based on real measurements on an RTX 4070 Super with vLLM + Prometheus + DCGM.

---

*All content is original Dielabs work by Diego Bardella.*
