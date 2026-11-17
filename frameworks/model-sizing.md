---
title: "Frameworks — Choose the model & size it"
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

# Choose the model &amp; size it

<p class="framework-question">Which model, which resources, and what can we prove? Selection and sizing first, then measurements that confirm or correct them.</p>

<div class="framework-group">
  <article class="framework-card" id="model-selection">
    <h3>LLM Model Selection</h3>
    <p>Use it to choose an open-weight model for your real workload and serving stack: test candidates on the cases where they fail, measure their speed on your hardware, and state how sure the choice is before validating the finished agent.</p>
    <a class="framework-action" href="llm-model-selection.html" aria-label="Read LLM Model Selection">Read &rarr;</a>
  </article>
  <article class="framework-card" id="context-vs-kv-cache">
    <h3>Context Engineering vs KV Cache Engineering</h3>
    <p>Use it when an agent and its infrastructure are designed by different people: it shows how choices like compaction, sub-agents and skills change cache reuse, latency and GPU capacity, so the two are sized together. Designing an agent is a capacity planning decision.</p>
    <a class="framework-action" href="context-vs-kv-cache-engineering.html" aria-label="Read Context Engineering vs KV Cache Engineering">Read &rarr;</a>
  </article>
  <article class="framework-card" id="idea-to-production">
    <h3>From Idea to Production</h3>
    <p>Use it to plan an inference deployment from use case to production in eleven steps: workload and traffic, service targets, model, sizing, runtime, hardware, serving stack, benchmarks and scaling. It explains what to decide, in which order, and why.</p>
    <a class="framework-action" href="mailto:info@dielabs.eu?subject=From%20Idea%20to%20Production%20%E2%80%94%20request" aria-label="Request access to From Idea to Production">Request access &rarr;</a>
  </article>
  <article class="framework-card" id="one-capacity">
    <h3>One Capacity Is Not Enough</h3>
    <p>Use it to answer “how many users can this system serve well?” instead of quoting peak throughput: it splits capacity into three measurable numbers, tied to the workload and its service targets, and turns them into a capacity card.</p>
    <a class="framework-action" href="one-capacity-is-not-enough.html" aria-label="Read One Capacity Is Not Enough">Read &rarr;</a>
  </article>
</div>
