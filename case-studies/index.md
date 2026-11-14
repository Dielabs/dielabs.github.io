---
title: Works
layout: default
wide: true
---

# Works

<div class="poc-wrap" markdown="0">
<style>
  .poc-wrap{--amber:#E2A33D; --teal:#57C2D6; --green:#1F6B4F; --mint:#2BA38A; --violet:#6B4FD8; --navy:#1F4F9A; --sky:#2B88C9;}
  .poc-intro{font-size:1.08rem; line-height:1.6; color:#3a3a55; margin:0 0 2.4rem;}
  .poc-label{font-family:'IBM Plex Mono',monospace; font-size:.74rem; letter-spacing:.18em; text-transform:uppercase; color:#8a8aa3; margin:2.2rem 0 .9rem; display:flex; align-items:center; gap:.7rem;}
  .poc-label::after{content:""; flex:1; height:1px; background:rgba(0,0,0,.08);}
  .poc-grid{display:grid; grid-template-columns:1fr 1fr; gap:1.1rem;}
  .poc-card{position:relative; display:block; border-radius:10px; padding:1.4rem 1.4rem 1.3rem; text-decoration:none !important; color:inherit; border:1px solid rgba(0,0,0,.07); overflow:hidden; transition:transform .18s, box-shadow .18s;}
  .poc-card::before{content:""; position:absolute; left:0; top:0; right:0; height:4px;}
  a.poc-card:hover{transform:translateY(-2px); box-shadow:0 10px 26px rgba(20,20,60,.10);}
  .poc-card h3{font-family:'Bebas Neue',sans-serif; font-size:1.75rem; letter-spacing:.02em; margin:0 0 .15rem; line-height:1.05; color:#1a1a2e;}
  .poc-card .sub{font-family:'IBM Plex Mono',monospace; font-size:.74rem; letter-spacing:.06em; text-transform:uppercase; margin-bottom:.8rem; display:block;}
  .poc-card p{margin:0; font-size:.95rem; line-height:1.55; color:#4a4a66;}
  .poc-card .go{display:inline-block; margin-top:.9rem; font-family:'IBM Plex Mono',monospace; font-size:.78rem; font-weight:600; letter-spacing:.06em;}
  .c-platone{background:linear-gradient(160deg,#FFF8EC 0%,#EEF9FB 100%);}
  .c-platone::before{background:linear-gradient(90deg,var(--amber),var(--teal));}
  .c-platone .sub,.c-platone .go{color:#9A6A12;}
  .c-pmi{background:linear-gradient(160deg,#EAF0FB 0%,#F3F7FD 100%);}
  .c-pmi::before{background:linear-gradient(90deg,var(--navy),var(--sky));}
  .c-pmi .sub,.c-pmi .go{color:var(--navy);}
  .c-news{background:linear-gradient(160deg,#EAF7F0 0%,#EFF9FA 100%);}
  .c-news::before{background:linear-gradient(90deg,var(--green),var(--mint));}
  .c-news .sub,.c-news .go{color:var(--green);}
  .lab{background:#FAFAFC; border:1px dashed rgba(0,0,0,.16);}
  .lab::before{display:none;}
  .lab h3{color:#2a2a44;}
  .lab .sub{color:#8a8aa3;}
  .pill{font-family:'IBM Plex Mono',monospace; font-size:.68rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase; padding:.18rem .55rem; border-radius:999px; display:inline-block; margin-bottom:.75rem;}
  .p-prog{color:#9A6412; background:#FBF0D9;}
  .p-des{color:var(--violet); background:#EFEAFD;}
  .p-beta{color:#1F6B4F; background:#E3F1EA;}
  .poc-grid.three{grid-template-columns:repeat(3,1fr);}
  .poc-foot{margin-top:2.6rem; padding-top:1.2rem; border-top:1px solid rgba(0,0,0,.08); font-size:.9rem; color:#7a7a96; font-style:italic;}
  @media (max-width:760px){ .poc-grid,.poc-grid.three{grid-template-columns:1fr;} }
</style>

<p class="poc-intro">Proofs of concept built in Dielabs — the first step from idea to production. Some are already presented to companies, others are still in the making. The published write-ups cover architecture, decisions, problems solved and measured results.</p>

<div class="poc-label">Published</div>
<div class="poc-grid three">
  <a class="poc-card c-platone" href="platone.html">
    <h3>Platone</h3>
    <span class="sub">Hybrid edge/cloud voice AI</span>
    <p>A voice assistant for your company's knowledge, down to the manuals of complex machinery. Load the PDFs, and operators on the production floor can ask questions with their headphones on. Before answering, it checks that the documents really cover the question; when they don't, it says so, and turns to the web or offers an opinion only if you agree.</p>
    <span class="go">Read the write-up →</span>
  </a>
  <a class="poc-card c-pmi" href="pmi-ai-gateway.html">
    <h3>PMI AI Gateway</h3>
    <span class="sub">Private AI for small businesses</span>
    <p>Open WebUI turned into a private AI product for small businesses: one isolated instance per company, Microsoft SSO, open models served in the EU, and Excel analysed and returned in chat. Two dozen defaults changed and measured, and the whole setup tuned through end-to-end evals. A 30k-token prompt now answers in 9 s instead of failing.</p>
    <span class="go">Read the write-up →</span>
  </a>
  <a class="poc-card c-news" href="newsrag.html">
    <h3>NewsRAG</h3>
    <span class="sub">Newspaper archives</span>
    <p>Natural-language search over newspaper archives, with answers you can verify: each citation opens the original article, date and excerpt. Captions are searchable, names and places are matched exactly, results can be filtered by period, and too little evidence means “no source found” rather than a guess.</p>
    <span class="go">Read the write-up →</span>
  </a>
</div>

<div class="poc-label">In the lab</div>
<div class="poc-grid">
  <div class="poc-card lab">
    <span class="pill p-prog">In progress</span>
    <h3>Dielabs RAG</h3>
    <span class="sub">Versioned docs RAG</span>
    <p>Answers only from the official docs of fast-moving AI software (vLLM, GuideLLM, lm-eval and LiteLLM), with one knowledge base per version, so releases never mix. Every answer links to the exact guide section for that version. What changed between two versions is computed by code, not guessed by the model.</p>
    <a class="go" href="https://github.com/Dielabs/dielabs-rag" target="_blank" rel="noopener">View the repository →</a>
  </div>
  <div class="poc-card lab">
    <span class="pill p-prog">In progress</span>
    <h3>Managed personal agent</h3>
    <span class="sub">Agentic harness</span>
    <p>A personal agent in daily use, reachable on Telegram and email, built one permission at a time. It runs scheduled jobs and manages its own calendar and mailbox, always signing as the agent, never as me. It lives in its own VM with a spending cap, its logs are copied outside that VM, and anything beyond the sandbox needs a human OK. The lab behind a future agent starter kit.</p>
  </div>
</div>

<p class="poc-foot">All content is original Dielabs work by Diego Bardella. Most project repositories are private; Dielabs RAG is public on GitHub. The published write-ups document the architecture and decisions in full.</p>
</div>
