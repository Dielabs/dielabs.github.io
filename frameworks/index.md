---
redirect_from:
  - /agentic/
  - /agentic/index.html
  - /frameworks/inference-technology-model
  - /frameworks/inference-technology-model.html
  - /frameworks/inference-stack-model
  - /frameworks/inference-stack-model.html
  - /frameworks/ai-use-case-discovery
  - /frameworks/ai-use-case-discovery.html
title: Frameworks
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
  .content .framework-card.framework-path { border: 2px solid #E0A82E; }
  .content .framework-label { margin: 0 0 0.5rem; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--blue); }
  .content .framework-question { margin: 0 0 1rem; }
  .content .framework-also { margin: 0 0 2.4rem; font-size: 0.8rem; line-height: 1.6; }
  .content .framework-group + h2, .content .framework-group + hr { margin-top: 2.4rem; }
  @media (max-width: 700px) { .content .framework-group { grid-template-columns: 1fr; } }
  .content .phase-list { display: flex; flex-direction: column; gap: 1.6rem; margin: 2.6rem 0 1rem; }
  .content a.phase-card { display: block; padding: 1.8rem 2rem; border: 1px solid var(--border); border-radius: 12px; background: var(--white); text-decoration: none; color: inherit; transition: border-color 0.2s, transform 0.2s, box-shadow 0.2s; }
  .content a.phase-card:hover { border-color: var(--blue); transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.06); }
  .content a.phase-card:focus-visible { outline: 2px solid var(--blue); outline-offset: 4px; }
  .content a.phase-card h2 { margin: 0 0 0.7rem; }
  .content a.phase-card p { margin: 0 0 1rem; font-size: 0.95rem; line-height: 1.7; color: var(--muted); }
  .content .phase-meta { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.04em; color: var(--blue); }</style>

# Frameworks

<p>The methods behind Dielabs projects, in the order a project needs them: qualify the use case, design the solution, size and validate it, build and hand it over, then keep it dependable in production. Public guides sit alongside specialist materials available on request.</p>

<div class="framework-group single">
  <article class="framework-card framework-path" id="ai-technical-presales">
    <p class="framework-label">The end-to-end path</p>
    <h3>AI Technical Presales Framework</h3>
    <p>Use it to take an AI deal from the first customer meeting to delivery: qualify the opportunity and the use case, turn requirements and risks into a proposal with clear acceptance criteria, and hand the project over. The sections below are the methods it uses at each step.</p>
    <a class="framework-action" href="mailto:info@dielabs.eu?subject=AI%20Technical%20Presales%20Framework%20%E2%80%94%20request" aria-label="Request access to AI Technical Presales Framework">Request access &rarr;</a>
  </article>
</div>

<div class="phase-list">
  <a class="phase-card" href="qualify-design.html">
    <h2>Qualify &amp; design the solution</h2>
    <p>Is it worth building, and what must the system do? Need, constraints, agents, retrieval and controls.</p>
    <span class="phase-meta">6 frameworks &middot; Open &rarr;</span>
  </a>
  <a class="phase-card" href="model-sizing.html">
    <h2>Choose the model &amp; size it</h2>
    <p>Which model, which resources, and what can we prove? Selection and sizing first, then measurements that confirm or correct them.</p>
    <span class="phase-meta">4 frameworks &middot; Open &rarr;</span>
  </a>
  <a class="phase-card" href="build-operate.html">
    <h2>Build &amp; operate</h2>
    <p>How do we prepare the data, hand the system over and keep it dependable in production?</p>
    <span class="phase-meta">3 frameworks &middot; Open &rarr;</span>
  </a>
</div>

<script>
  (function () {
    var map = {
      'rag-discovery': 'qualify-design', 'agent-design': 'qualify-design', 'designing-agents': 'qualify-design',
      'workflow-governance': 'qualify-design', 'harness-requirements': 'qualify-design', 'zero-to-rag': 'qualify-design',
      'model-selection': 'model-sizing', 'context-vs-kv-cache': 'model-sizing', 'idea-to-production': 'model-sizing', 'one-capacity': 'model-sizing',
      'data-preparation': 'build-operate', 'observability-kpi': 'build-operate', 'llm-parameters': 'build-operate'
    };
    var id = location.hash.slice(1);
    if (map[id]) location.replace('/frameworks/' + map[id] + '.html#' + id);
  })();
</script>