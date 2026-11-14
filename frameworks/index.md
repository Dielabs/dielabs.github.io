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
  .content .framework-card.framework-path { border: 2px solid var(--blue); }
  .content .framework-label { margin: 0 0 0.5rem; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: var(--blue); }
  .content .framework-question { margin: 0 0 1rem; }
  .content .framework-also { margin: 0 0 2.4rem; font-size: 0.8rem; line-height: 1.6; }
  .content .framework-group + h2, .content .framework-group + hr { margin-top: 2.4rem; }
  @media (max-width: 700px) { .content .framework-group { grid-template-columns: 1fr; } }
</style>

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

## Qualify the use case

<p class="framework-question">Is it worth building, and what exactly should it do? Need, constraints and success criteria.</p>

<div class="framework-group single">
  <article class="framework-card" id="rag-discovery">
    <h3>RAG Discovery Framework</h3>
    <p>Use it before building a RAG, to find out what the use case really needs: what the documents are like, which enhanced features the questions call for beyond a basic RAG, how strictly answers must stick to sources and cite them, who may see what, and where the system stops.</p>
    <a class="framework-action" href="mailto:info@dielabs.eu?subject=RAG%20Discovery%20Framework%20%E2%80%94%20request" aria-label="Request access to RAG Discovery Framework">Request access &rarr;</a>
  </article>
</div>

<p class="framework-also">Also used here: <a href="#workflow-governance">Agentic Workflow Governance</a></p>

## Design the solution

<p class="framework-question">What must the system do, and how must it behave? Model, agents, retrieval and controls.</p>

<div class="framework-group">
  <article class="framework-card" id="model-selection">
    <h3>LLM Model Selection</h3>
    <p>Use it to choose an open-weight model for your real workload and serving stack: test candidates on the cases where they fail, measure their speed on your hardware, and state how sure the choice is before validating the finished agent.</p>
    <a class="framework-action" href="llm-model-selection.html" aria-label="Read LLM Model Selection">Read &rarr;</a>
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
  <article class="framework-card" id="context-vs-kv-cache">
    <h3>Context Engineering vs KV Cache Engineering</h3>
    <p>Use it when an agent and its infrastructure are designed by different people: it shows how choices like compaction, sub-agents and skills change cache reuse, latency and GPU capacity, so the two are sized together. Designing an agent is a capacity planning decision.</p>
    <a class="framework-action" href="context-vs-kv-cache-engineering.html" aria-label="Read Context Engineering vs KV Cache Engineering">Read &rarr;</a>
  </article>
</div>

<p class="framework-also">Also used here: <a href="#idea-to-production">From Idea to Production</a></p>

## Size &amp; validate

<p class="framework-question">Which resources does it need, and what can we prove? Sizing calculations first, then measurements that confirm or correct them.</p>

<div class="framework-group">
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

<p class="framework-also">Also used here: <a href="#model-selection">LLM Model Selection</a> · <a href="#context-vs-kv-cache">Context Engineering vs KV Cache Engineering</a> · <a href="#observability-kpi">Observability KPI</a></p>

## Build &amp; hand over

<p class="framework-question">How do we prepare the data, implement and deliver a solution someone else can run?</p>

<div class="framework-group single">
  <article class="framework-card" id="data-preparation">
    <h3>Data Preparation for Knowledge Bases</h3>
    <p>Use it to get documents ready for an LLM, whichever way it will read them: browsing them through MCP, searching them with a RAG, or loading them straight into the context. Each way needs the data shaped differently, and a seven-step pipeline covers the move from MCP to RAG.</p>
    <a class="framework-action" href="mailto:info@dielabs.eu?subject=Data%20Preparation%20for%20Knowledge%20Bases%20%E2%80%94%20request" aria-label="Request access to Data Preparation for Knowledge Bases">Request access &rarr;</a>
  </article>
</div>

## Govern &amp; operate

<p class="framework-question">How do we keep it dependable? Monitor, diagnose degradation and recover.</p>

<div class="framework-group single">
  <article class="framework-card" id="observability-kpi">
    <h3>Observability KPI</h3>
    <p>Use it when users say the AI has become slow: start from what they notice, such as a late first word or a stuttering answer, then follow the path down to serving metrics and GPU signals, with ready-made monitoring queries for vLLM.</p>
    <a class="framework-action" href="mailto:info@dielabs.eu?subject=Observability%20KPI%20%E2%80%94%20request" aria-label="Request access to Observability KPI">Request access &rarr;</a>
  </article>
</div>

<p class="framework-also">Also used here: <a href="#harness-requirements">Minimum Requirements of an Agentic Harness</a></p>

## Technical references

<div class="framework-group single">
  <article class="framework-card" id="llm-parameters">
    <h3>LLM Parameters — Model, Engine, Request</h3>
    <p>Use it when a setting doesn’t behave as expected: every LLM parameter lives in one of three places (the model, the engine or the request), and knowing which tells you when it takes effect, who can change it, and which value wins when two settings conflict.</p>
    <a class="framework-action" href="llm-parameter-topology.html" aria-label="Read LLM Parameters — Model, Engine, Request">Read &rarr;</a>
  </article>
</div>

---

*Methods and guides by Diego Bardella, Dielabs, including original work and adaptations with sources credited in each document.*
