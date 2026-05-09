---
title: Manuals
layout: default
---

# Manuals

Reference manuals for LLM inference engineering, covering operational knowledge from real lab work at Dielabs.

---

## Documents

### [The Inference Engineering Manual](inference-engineering-manual.md)
Comprehensive reference covering the full LLM inference stack: from model lifecycle and quantization to the memory wall, KV cache economics, batching strategies, parallelism topologies, deployment architectures, and cost modeling. Six chapters plus epilogue, built from first principles for inference engineers and AI infrastructure architects.

### [KV Cache Manual](kv-cache-workbook.md)
End-to-end manual on KV cache mechanics and management. Covers anatomy and per-token consumption, PagedAttention and pool management in vLLM, behavior under load, prefix caching, tuning parameters, policy levers (TTL, quotas, eviction), the storage tier hierarchy and offload lifecycle, the 2026 software ecosystem (LMCache, NIXL, Dynamo, CMX), the economics of long chats, observability metrics, and the path from single-node to distributed inference.

<div class="private-card">
  <div class="private-title-row">
    <span class="private-title">From Idea to Production</span>
    <span class="private-pill">Available on request</span>
  </div>
  <p class="private-desc">The full operational manual behind the <em>Inference Sizing in 10 Steps</em> framework. End-to-end presales architect playbook: from business pain statement to validated production deployment, with discovery templates, sizing worksheets, runtime decision matrices, hardware constraint tables, benchmark protocols, and scaling decision trees. Each of the 10 steps is expanded into actionable artifacts usable in real customer engagements.</p>
  <p class="private-note">Not published. Reserved for direct conversations &mdash; reach out if relevant to your context.</p>
</div>

---

*All content is original Dielabs work by Diego Bardella.*
