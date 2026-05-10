# Dielabs

**Dielabs** is a personal engineering lab focused on AI inference infrastructure.

The goal is to explore, test and document how modern inference systems behave in real environments, with a strong focus on infrastructure, runtime behavior and performance.

Rather than focusing on model training or data science, Dielabs investigates the operational and infrastructural side of AI systems — from GPU memory physics to scheduler tuning, from distributed parallelism patterns to cost-per-token economics.

---

## Repository Structure

### [Papers](papers/)
Original technical papers from real lab work. Topics include degenerative decoding analysis, KV cache offloading investigation, CPU-GPU inference isomorphism, and bottleneck migration in capacity sizing.

### [Manuals](manuals/)
Reference manuals covering operational inference engineering knowledge — KV cache mechanics, vLLM tuning, Prometheus metrics.

### [Frameworks](frameworks/)
Proprietary conceptual models for reasoning about inference systems: the L0–L6 diagnostic framework, the A–G technology model, the Artifact/Startup/Request parameter topology, a 10-step sizing methodology, and an observability KPI framework for incident response.

---

## Philosophy

**Understanding systems by building, measuring and documenting them.**

Every finding published here comes from hands-on experimentation on real hardware — RTX 4070 Super, Dell PowerEdge R730, vLLM, Prometheus, Grafana. No synthetic benchmarks, no theoretical-only analysis.

---

## Author

Diego Bardella — [dielabs.github.io](https://dielabs.github.io)
