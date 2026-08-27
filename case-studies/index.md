---
title: Case Studies
layout: default
---

# Case Studies

Applied systems designed, built and shipped at Dielabs — where the frameworks meet a real deployment. Each case study documents the architecture decisions, the measured trade-offs behind them, and the incidents solved along the way.

---

## Documents

### [Platone — Hybrid Edge/Cloud Voice AI](platone.html)
A voice assistant grounded on a technical knowledge base, split across an edge GPU (VAD, STT, retrieval, TTS) and cloud EU inference (query rewriting, generation, judging). Documents the retrieve-then-generate cascade beyond naive RAG, per-role model routing, the edge/cloud placement rule, and four production incidents — from a GPU memory conflict between STT and TTS to a retrieval collapse on conversational follow-ups.

---

*All content is original Dielabs work by Diego Bardella. Platone's repository is private; this write-up documents the architecture and decisions in full.*
