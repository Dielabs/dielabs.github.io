---
title: "Frameworks — Build & operate"
layout: default
wide: true
---

<style>
  .content .framework-group { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; margin: 1.3rem 0 1rem; }
  .content .framework-card { display: flex; flex-direction: column; padding: 1.3rem 1.4rem; border: 1px solid var(--border); border-radius: 10px; background: var(--white); min-width: 0; scroll-margin-top: 5rem; }
  .content .framework-card h3 { margin: 0 0 0.8rem; font-size: 0.94rem; line-height: 1.45; letter-spacing: 0.01em; text-transform: none; color: var(--text); }
  .content .framework-card p { margin: 0 0 1rem; font-size: 0.88rem; line-height: 1.7; }
  .content .framework-action { align-self: flex-start; margin-top: auto; padding: 0.2rem 0; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.04em; }
  .content .framework-action:focus-visible { outline: 2px solid var(--blue); outline-offset: 4px; }
  .content .framework-card:last-child:nth-child(odd) { grid-column: 1 / -1; }
  .content .framework-group.single { grid-template-columns: 1fr; }
  .content .framework-card.framework-path { border: 2px solid var(--blue); }
  .content .framework-label { margin: 0 0 0.5rem; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--blue); }
  .content .framework-question { margin: 0 0 1rem; }
  .content .framework-also { margin: 0 0 2.4rem; font-size: 0.8rem; line-height: 1.6; }
  .content .framework-group + h2, .content .framework-group + hr { margin-top: 2.4rem; }
  @media (max-width: 700px) { .content .framework-group { grid-template-columns: 1fr; } }
</style>

<p class="framework-back"><a href="/frameworks/">&larr; Frameworks</a></p>

# Build &amp; operate

<p class="framework-question">How do we get it running and keep it dependable? Prepare the documents the system will read, hand it over so someone else can run it, and monitor it in production to diagnose slowdowns before users notice.</p>

<div class="framework-group">
  <article class="framework-card" id="data-preparation">
    <h3>Data Preparation for Knowledge Bases</h3>
    <p>Use it to get documents ready for an LLM, whichever way it will read them: browsing them through MCP, searching them with a RAG, or loading them straight into the context. Each way needs the data shaped differently, and a seven-step pipeline covers the move from MCP to RAG.</p>
    <a class="framework-action" href="mailto:info@dielabs.eu?subject=Data%20Preparation%20for%20Knowledge%20Bases%20%E2%80%94%20request" aria-label="Request access to Data Preparation for Knowledge Bases">Request access &rarr;</a>
  </article>
  <article class="framework-card" id="observability-kpi">
    <h3>Observability KPI</h3>
    <p>Use it when users say the AI has become slow: start from what they notice, such as a late first word or a stuttering answer, then follow the path down to serving metrics and GPU signals, with ready-made monitoring queries for vLLM.</p>
    <a class="framework-action" href="mailto:info@dielabs.eu?subject=Observability%20KPI%20%E2%80%94%20request" aria-label="Request access to Observability KPI">Request access &rarr;</a>
  </article>
  <article class="framework-card" id="llm-parameters">
    <h3>LLM Parameters — Model, Engine, Request</h3>
    <p>Use it when a setting doesn’t behave as expected: every LLM parameter lives in one of three places (the model, the engine or the request), and knowing which tells you when it takes effect, who can change it, and which value wins when two settings conflict.</p>
    <a class="framework-action" href="llm-parameter-topology.html" aria-label="Read LLM Parameters — Model, Engine, Request">Read &rarr;</a>
  </article>
</div>
