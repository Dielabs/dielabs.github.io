---
title: "Frameworks — Qualify & design the solution"
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

# Qualify &amp; design the solution

<p class="framework-question">Is it worth building, and what must the system do? Start from the real need, the data and the constraints, then design the solution: which agents, how retrieval works, what the system may do on its own and where a person stays in control.</p>

<div class="framework-group">
  <article class="framework-card" id="rag-discovery">
    <h3>RAG Discovery Framework</h3>
    <p>Use it before building a RAG, to find out what the use case really needs: what the documents are like, which enhanced features the questions call for beyond a basic RAG, how strictly answers must stick to sources and cite them, who may see what, and where the system stops.</p>
    <a class="framework-action" href="mailto:info@dielabs.eu?subject=RAG%20Discovery%20Framework%20%E2%80%94%20request" aria-label="Request access to RAG Discovery Framework">Request access &rarr;</a>
  </article>
  <article class="framework-card" id="agent-design">
    <h3>Agent Design Framework</h3>
    <p>Use it to define which agents a business needs: one per business area, never a universal “company GPT”. Each gets a clear role, its own data, only the permissions it needs, and human checks that grow with the risk of the action: read freely, act with approval, write under control.</p>
    <a class="framework-action" href="mailto:info@dielabs.eu?subject=Agent%20Design%20Framework%20%E2%80%94%20request" aria-label="Request access to Agent Design Framework">Request access &rarr;</a>
  </article>
  <article class="framework-card" id="designing-agents">
    <h3>Designing AI Agents from Scratch</h3>
    <p>Use it to engineer an agent before writing code: first check whether a single LLM call or a fixed workflow would do, then define the agent’s goal, limits and budget, its control loop, and how it checks its own results and recovers from errors.</p>
    <a class="framework-action" href="mailto:info@dielabs.eu?subject=Designing%20AI%20Agents%20%E2%80%94%20request" aria-label="Request access to Designing AI Agents from Scratch">Request access &rarr;</a>
  </article>
  <article class="framework-card" id="workflow-governance">
    <h3>Agentic Workflow Governance</h3>
    <p>Use it to automate a business process with AI without losing control: the model reads and interprets, code takes the decisions, and a person approves the critical actions. Autonomy grows step by step, only when measured approval rates justify it. Built for processes that must stay auditable, such as invoices, contracts and email triage.</p>
    <a class="framework-action" href="mailto:info@dielabs.eu?subject=Agentic%20Workflow%20Governance%20%E2%80%94%20request" aria-label="Request access to Agentic Workflow Governance">Request access &rarr;</a>
  </article>
  <article class="framework-card" id="harness-requirements">
    <h3>Minimum Requirements of an Agentic Harness</h3>
    <p>Use it to evaluate or design an agent harness, or to decide whether an agent is ready for production: a checklist of what it must remember, the limits and budgets it works within, and how it proves its work, so that someone else can pick it up cold.</p>
    <a class="framework-action" href="agentic-harness-requirements.html" aria-label="Read Minimum Requirements of an Agentic Harness">Read &rarr;</a>
  </article>
  <article class="framework-card" id="zero-to-rag">
    <h3>From Zero to RAG</h3>
    <p>Once the requirements are clear, use it to choose how to build the RAG, from a basic RAG to agent orchestration: a table from requirement to pattern and cost, and how each step up changes inference sizing. It starts from why a basic RAG fails in production.</p>
    <a class="framework-action" href="mailto:info@dielabs.eu?subject=From%20Zero%20to%20RAG%20%E2%80%94%20request" aria-label="Request access to From Zero to RAG">Request access &rarr;</a>
  </article>
</div>
