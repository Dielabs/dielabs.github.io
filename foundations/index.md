---
redirect_from:
  - /manuals/
  - /manuals/index.html
  - /papers/testing-is-the-product.html
title: Foundations
layout: default
---

# Foundations

The part of inference that does not age: mechanics, math, and trade-offs. Product names, versions and measured numbers live in [Technologies](/technologies/) and get rewritten when the landscape moves. These pages do not.

---

## Documents

### [The Inference Engineering Manual](inference-engineering-manual.md)
Comprehensive reference covering the full LLM inference stack: from model lifecycle and quantization to the memory wall, KV cache economics, batching strategies, parallelism topologies, deployment architectures, and cost modeling. Six chapters plus epilogue, built from first principles for inference engineers and AI infrastructure architects.

### [KV Cache Manual](kv-cache-workbook.md)
End-to-end manual on KV cache mechanics and management. Covers anatomy and per-token consumption, PagedAttention and pool management in vLLM, behavior under load, prefix caching, tuning parameters, policy levers (TTL, quotas, eviction), the storage tier hierarchy and offload lifecycle, the 2026 software ecosystem (LMCache, NIXL, Dynamo, CMX), the economics of long chats, observability metrics, and the path from single-node to distributed inference.

### [Disaggregated Inference](disaggregated-inference.md)
Architectural reference for the disaggregated serving pattern: prefill/decode separation, KV cache transfer over RDMA, NIC vs DPU on the data path, KV pooling tiers (NVMe-oF today, CXL as direction), performance analysis with fair-baseline methodology, speculative decoding interactions, and a decision framework for when disaggregation pays off versus when it adds complexity without benefit. The guiding principle: KV cache transfer dominates the design space, and TTFT — not GB/s — is the KPI that matters.

### [Semantic Routing](semantic-routing.md)
Why a semantic router must be fine-tuned on the tier boundaries of its real domain: the decision layer (a small embedding model) is separate from the deterministic tier-to-backend map, and misrouting is born in the first. Why pre-trained embeddings confuse similarity with complexity, what fine-tuning really costs, routing accuracy as an SLO rather than uptime, and why in sovereign deployments routing is where data residency becomes configurable.

<div class="private-card">
  <div class="private-title-row">
    <span class="private-title">Testing Is the Product</span>
    <a class="private-pill" href="mailto:info@dielabs.eu?subject=Testing%20Is%20the%20Product%20%E2%80%94%20request" title="Request access via email">Available on request</a>
  </div>
  <p class="private-desc">A manifesto for on-premise inference due diligence, built from a public case study: a viral four-node DGX Spark deployment and the four hundred comments it drew. Extracts eleven principles, an eight-question qualification checklist, and a five-axis acceptance protocol (performance, quality, resilience, operations, governance).</p>
  <p class="private-note">Not published. Reserved for direct conversations &mdash; reach out if relevant to your context.</p>
</div>

---

*All content is original Dielabs work by Diego Bardella.*