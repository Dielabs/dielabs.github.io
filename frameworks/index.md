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

<style>
  .content .framework-intro { margin-bottom: 2rem; }
  .content .framework-group { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; margin: 1.3rem 0 2.4rem; }
  .content .framework-card { display: flex; flex-direction: column; padding: 1.3rem 1.4rem; border: 1px solid var(--border); border-radius: 10px; background: var(--white); min-width: 0; }
  .content .framework-card h3 { margin: 0 0 0.8rem; font-size: 0.94rem; line-height: 1.45; letter-spacing: 0.01em; text-transform: none; color: var(--text); }
  .content .framework-card p { margin: 0 0 1rem; font-size: 0.88rem; line-height: 1.7; }
  .content .framework-action { align-self: flex-start; margin-top: auto; padding: 0.2rem 0; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.04em; }
  .content .framework-action:focus-visible { outline: 2px solid var(--blue); outline-offset: 4px; }
  .content .framework-card:last-child:nth-child(odd) { grid-column: 1 / -1; }
  .content .framework-group.single { grid-template-columns: 1fr; }
  @media (max-width: 700px) { .content .framework-group { grid-template-columns: 1fr; } }
</style>

# Frameworks

<p class="framework-intro">The methods behind Dielabs projects: qualify the need, design the system, measure what it can deliver and keep it dependable in production. Public guides sit alongside specialist materials available on request.</p>

## Assess & Design

<div class="framework-group">
  <article class="framework-card">
    <h3>From Idea to Production</h3>
    <p>Turn a business need, traffic profile and service targets into a sized inference architecture, then validate it through benchmarks before deciding how to scale.</p>
    <a class="framework-action" href="from-idea-to-production-manual.html" aria-label="Read From Idea to Production">Read &rarr;</a>
  </article>
  <article class="framework-card">
    <h3>AI Technical Presales Framework</h3>
    <p>Qualify the opportunity and use case, turn requirements and risks into a proposal with explicit acceptance criteria, and prepare the handover to delivery.</p>
    <a class="framework-action" href="mailto:info@dielabs.eu?subject=AI%20Technical%20Presales%20Framework%20%E2%80%94%20request" aria-label="Request access to AI Technical Presales Framework">Request access &rarr;</a>
  </article>
  <article class="framework-card">
    <h3>LLM Model Selection</h3>
    <p>Choose an open-weight model for the actual workload and serving stack, using failure-driven benchmarks, measured speed and explicit uncertainty before validating the finished agent.</p>
    <a class="framework-action" href="llm-model-selection.html" aria-label="Read LLM Model Selection">Read &rarr;</a>
  </article>
</div>

## Benchmark & Operate

<div class="framework-group">
  <article class="framework-card">
    <h3>One Capacity Is Not Enough</h3>
    <p>Separate hardware saturation from capacity at acceptable latency, and produce a capacity card tied to the workload and its service targets.</p>
    <a class="framework-action" href="one-capacity-is-not-enough.html" aria-label="Read One Capacity Is Not Enough">Read &rarr;</a>
  </article>
  <article class="framework-card">
    <h3>Observability KPI</h3>
    <p>Connect user-facing latency and serving metrics to hardware signals, with diagnostic paths and monitoring queries for investigating performance incidents.</p>
    <a class="framework-action" href="observability-kpi.html" aria-label="Read Observability KPI">Read &rarr;</a>
  </article>
</div>

## Agents & Delivery

<div class="framework-group">
  <article class="framework-card">
    <h3>Designing AI Agents from Scratch</h3>
    <p>Decide whether an agent is needed, define its contract and control loop, and specify permissions, verification and recovery before implementation.</p>
    <a class="framework-action" href="mailto:info@dielabs.eu?subject=Designing%20AI%20Agents%20%E2%80%94%20request" aria-label="Request access to Designing AI Agents from Scratch">Request access &rarr;</a>
  </article>
  <article class="framework-card">
    <h3>Delivery Frameworks</h3>
    <p>Four practical methods for defining agents, qualifying retrieval needs, preparing knowledge bases and governing workflows with explicit human oversight.</p>
    <a class="framework-action" href="mailto:info@dielabs.eu?subject=Delivery%20Frameworks%20%E2%80%94%20request" aria-label="Request access to Delivery Frameworks">Request access &rarr;</a>
  </article>
</div>

## Technical Reference

<div class="framework-group single">
  <article class="framework-card">
    <h3>LLM Parameters — Model, Engine, Request</h3>
    <p>Locate each parameter in the model, engine or request layer to understand when it takes effect, who controls it and where conflicting settings must be resolved.</p>
    <a class="framework-action" href="llm-parameter-topology.html" aria-label="Read LLM Parameters — Model, Engine, Request">Read &rarr;</a>
  </article>
</div>

---

*All frameworks are original Dielabs work by Diego Bardella.*
