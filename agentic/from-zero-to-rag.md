---
layout: default
title: "From Zero to RAG"
---

# From Zero to RAG

The quantitative companion to [RAG for Dummies](rag-for-dummies.md). Part I explains the problem and how a basic RAG works; Part II presents a four-level framework, a requirement → pattern → cost selection table, and what each level does to inference sizing. Read it to choose how deep a retrieval system needs to go for a given use case.

---

**PART I — From zero to RAG**

## 1. The limits of an LLM on its own

An LLM is trained on a huge corpus, but one frozen at a certain date. On its own it has three structural limits:

- **It doesn't know your private data** — contracts, internal documentation, tickets, the company knowledge base. They weren't in the training set.
- **It isn't up to date** — everything that happened after the knowledge cutoff is out of reach.
- **It can't cite its source** — it produces plausible text, but it can't tell you *where* a statement comes from. This opens the door to hallucinations: when it doesn't know, it tends to answer anyway.

The "brute force" route would be to retrain or fine-tune the model on your own data. Expensive, slow, and it has to be redone every time the data changes. And it doesn't solve citability.

## 2. The idea behind RAG

RAG = **Retrieval-Augmented Generation**. The insight is to flip the problem: instead of putting knowledge *inside* the model, you keep it outside and *hand it over at question time*.

The flow, in words:

> The user asks a question → the system searches the documents for the relevant pieces → it puts them into the prompt after the question → the model receives them and answers based on that context → the user gets the answer

The model stops answering "from memory" and answers by reading material supplied on the fly. The three limits fall: the data can be private, updated at any time, and every statement can be traced back to the chunk it came from — the right technology for the job.

> **📦 Box — "With 1M-token context windows, why not load the whole corpus into the prompt?"**
>
> A legitimate objection, and the most frequent one in 2026. Four reasons why long context doesn't replace RAG:
>
> - **Cost per query.** Loading the corpus into the prompt means paying the prefill of the *whole corpus on every question*. RAG pays only for the few relevant chunks. At enterprise volumes the difference is orders of magnitude.
> - **Quality.** On very long contexts models degrade (*lost in the middle*): they remember the beginning and the end well, and lose the middle. More noise in the context = a worse answer, not a better one.
> - **Citability.** With the whole corpus in the prompt you don't know *where* a statement comes from. RAG gives you chunk-by-chunk granularity (§7.1).
> - **Size.** A real enterprise corpus (tens of GB of documents) exceeds any window anyway.
>
> Long context and RAG are **complementary**, not alternatives: a bigger window allows more generous chunks and more candidates in the prompt — it doesn't remove retrieval.

## 3. How it works, concretely

RAG has two distinct phases. The first happens **once** (or when the data changes) and prepares the knowledge — the **offline phase**; the second runs on every question and produces the answer — the **online phase**.

Before looking at the flow, two concepts hold the whole mechanism together: what *embeddings* are, and why they make it possible to "search by meaning".

### The two key concepts

**An embedding turns text into a position.** An embedding model (the **encoder**) converts a piece of text into a vector — a list of numbers, usually from a few hundred to a few thousand — which is equivalent to a coordinate in a high-dimensional space. The encoder is trained on purpose so that texts with similar meaning end up close together in that space, even when they use different words. "How do I reset my password?" and "credential recovery procedure" share almost no terms, yet their vectors are close because they say the same thing.

**Search becomes about closeness, not words.** Classic search engines *(and SQL database queries)* look for exact word matches: either the term is there or it isn't. Vector search does something else — it measures **how close two vectors are** in the space. This makes it possible to find the right chunk even when the question is phrased completely differently from the document's text. It's the reason a dedicated index is needed *(the vector DB)*: finding the nearest neighbors among millions of high-dimensional vectors is a specific problem that a relational database doesn't solve efficiently.

> **📦 Technical box — Vector DB ≠ Relational DB**
>
> They are two different families, optimized for two opposite questions.
>
> The **relational DB** answers exact matches and filters: "all orders with `status = 'shipped'` and `amount > 100`". Either a record meets the condition or it doesn't — a boolean match. Its indexes are built for exact values and ranges.
>
> The **vector DB** answers "give me the vectors *closest* to this one". There's no yes/no: there's an ordering by distance, and you take the first ones. It's *nearest neighbor search* in high-dimensional spaces, where traditional indexes degrade until they become useless.

| Aspect | Relational DB | Vector DB |
|---|---|---|
| Question | Exact match / range | Most similar vectors |
| Match | Boolean | By proximity (ranking) |
| *Index* | *B-tree, hash* | *ANN: HNSW, IVF, PQ* |
| Result | Rows that match | Top-k closest |

### Offline phase — preparing the knowledge (ingestion)

It runs once, and is repeated only when the documents change. The naive RAG offline pipeline:

```
Documents → clean text → chunks → embeddings → vector index
```

1. **Parsing** — structured text is extracted from PDFs, HTML and scans (typically Markdown, which preserves headings and tables).

2. **Chunking** — documents are split into small but self-contained units (a paragraph, a section). Size is a trade-off: a chunk that's too big produces a vector mixing too many concepts — similarity gets diluted and redundant material ends up in the prompt; a chunk that's too small loses its context and stops standing on its own at retrieval time.

3. **Embedding** — each chunk is converted into its vector with the embedding model. This is where "meaning" becomes a position in space.

4. **Indexing** — the vectors are loaded into an index (the vector DB) built to answer quickly the question "which vectors are closest to this one?".

*At the end of the phase, knowledge is no longer text: it's a cloud of points in a space where closeness means affinity of meaning — or almost: sometimes closeness is misleading, and that's why re-ranking exists, as we'll see later.*

### Online phase — answering the query

It runs on every user question. The naive RAG online pipeline:

```
Question → embedding (same encoder model) → similarity search (top k) → chunk retrieval → prompt injection (question + retrieved context) → model answer
```

1. **Embedding the question** — the question is turned into a vector with the **same model** used for the chunks. This detail is non-negotiable: question and documents must live in the same space and logic, otherwise closeness makes no sense.

2. **Retrieval** — the system looks for the **k** chunk-vectors closest to the question-vector (e.g. k=5). "Closest" = most similar in meaning. These are the candidates that may contain the answer.

3. **Prompt injection** — the retrieved chunks are inserted into the prompt together with the question, *usually with an instruction like "answer using only the context provided"*. This is the "A" in *Augmented*: the prompt is enriched with external knowledge on the fly.

4. **Generation** — the model produces the answer based on that context instead of on its internal memory alone. This gives **source traceability**: you know *which chunks* were provided as the basis.

*Careful, though* — knowing which chunks you passed doesn't guarantee that the answer is actually faithful to them: the model can hallucinate or mix them with its internal memory. *Verifying* faithfulness is a separate problem, which can be tackled with dedicated additional technologies (Self-RAG, CRAG → §9), not part of the foundation.

### The models involved

The flow leaves implicit a point worth making explicit: RAG **doesn't use a single model**. The "model" components are distinct, with different roles and profiles, and they shouldn't be confused.

**Embedding model — the encoder.** It turns text into vectors, both offline (on the chunks) and online (on the question). It isn't an LLM in the classic sense: it's typically an *encoder* (BERT family), it doesn't generate text. It receives a chunk, internally produces a vector for each token, and a **pooling** step (mean pooling, or the `[CLS]` token) collapses them into the single vector — of **fixed size** (e.g. 384, 768, 1024), independent of the chunk's length — that goes into the index. The exact same operation, with the **same model**, runs on the question: that's what guarantees that question and chunks live in the same space and can be compared by closeness.

> Note: some recent embedding models (e.g. the Qwen3-Embedding family) are *derived* from decoder-only LLMs. The technical boundary is getting thinner, but the role remains that of an embedding model — vectors, not generation — so they're classified as encoders, not as a second LLM.

**Generative LLM — the decoder.** It's the only real LLM in the proper sense: an autoregressive decoder that receives the augmented prompt and produces the answer. It's **independent** of the embedding model: you can index with one encoder and generate with a completely different LLM — in fact that's the norm.

**Re-ranker — the cross-encoder** (optional, but almost always present in production; §7). In retrieval, the question is vectorized by the encoder and compared by closeness, on the index, with the chunk vectors already computed at ingestion: fast and precomputable, but approximate, because question and chunk never see each other. The result is a set of chunks that will *probably* help answer. The cross-encoder improves the final result because it reads the user's question together with each candidate chunk, one at a time, concatenated in a single forward pass — much more precise, but not precomputable (one forward pass per pair, at query time). That's why the two run in series: retrieval skims the whole DB down to a shortlist of candidate chunks, and the cross-encoder *reorders only those and keeps the best ones for the prompt*.

The misconception to avoid is thinking that the generative LLM produces the embeddings. It doesn't: embedding is done by the **encoder**, reordering by the **cross-encoder**, generation by the **decoder**. Three models, three inference profiles — the reason why sizing a RAG doesn't coincide with sizing the generator alone (→ appendix).

### Naive vs Production RAG

The flow seen so far already contains a fork. In its minimal form — embed → top chunks → stuff into context, **without a re-ranker** — it's **Naive RAG**: direct, and perfect for a demo on a few clean documents. Adding the cross-encoder (and, more generally, the hybrid search and re-ranking of the Foundation, §7) takes it to production, into **Production RAG**.

The distinction isn't academic: it's the first step of the *capability ladder*. Naive RAG on its own breaks in production — and understanding *where* it breaks is what brings out all the later patterns, starting with the re-ranker.

## 4. Why basic RAG fails in production

The point that separates a demo from an enterprise system: **quality isn't won on the model, it's won on the data**. The typical ways of failing, in order of frequency:

- **Dirty ingestion** — if extraction produces broken tables, or text that is really a picture, every downstream step inherits the damage. No model recovers corrupted input data.

- **Naive chunking** — fixed-length chunks that cut tables, sentences or logical sections in half. Retrieval returns fragments with no context.

- **Semantic search only** — dense search on its own misses queries based on exact words: product codes, acronyms, proper names, part numbers. "Error E-4023" has no useful semantic neighbors. That's why in production semantic search almost never works alone: it's paired with an **exact-word search**, the one used by classic search engines. The two run in parallel and their results are merged — the so-called ***hybrid search***, a row of the Foundation (§7). Exact-word search catches codes and acronyms literally; semantic search captures meaning even when the words change. Together they cover the failures each one has on its own.

- **No reordering** — top-k returns candidates by raw similarity, but the most relevant ones aren't always at the top.

The practical consequence: the bottleneck is rarely the vector DB. It's almost always upstream. **Whoever has clean data and a solid pipeline has already solved 80% of the problem.**

## 5. The leap — from naive RAG to a structured system

Fixing those failures means adding steps to the basic pipeline. And here a structure emerges naturally: every improvement answers a specific limit. Instead of a list of "tricks", it's better to read them as stacked **levels**, each of which solves a class of problems the level below can't.

It's this structure — not a taxonomy from a paper — that forms the backbone of Part II.

---

**PART II — RAG types and how to match them to use cases**

## 6. The four-level framework

```
Level 4   Agent Orchestration        reasons and coordinates several tools
             ↑
Level 3   Retrieval Intelligence     decides whether and how to retrieve
             ↑
Level 2   Knowledge Representation   how knowledge is modeled
             ↑
Level 1   Production                 finds the documents
```

"Production RAG" is Naive done well (hybrid search + re-ranking), the mature starting point from which you may climb. The four-level ladder measures a different axis: how many decision-making capabilities you add on top of a solid Production layer, not how polished the Production layer itself is.

The structural distinction that holds everything together:

- **Production** and **Knowledge Representation** change *how you represent* knowledge.
- **Retrieval Intelligence** and **Agent Orchestration** change *how the system retrieves and uses it*.

They are two independent decisions: you can have a sophisticated Production layer under fixed-rule retrieval, or a minimal Production layer under an agent that decides everything at runtime. Representation and decision are sized separately — and it's the second axis, the decision axis, that makes sizing difficulty explode on the way up to Level 4 (§12).

From here on, each component is presented as **requirement → pattern → rationale → cost**. Every pattern exists to solve a specific limit; *if that limit doesn't show up in the use case, the pattern only adds complexity, latency and operating cost with nothing in return.* The final selection table (§13) puts the same logic back together in summary form, adding the level column.

## 7. Production layer — always on

You've already met the elements of this layer in Part I, but there they served to explain *how* RAG works. Here we read them again as a **production chain**, where each link puts a quality ceiling on everything downstream — and reading them this way shows **what separates a demo pipeline (Naive) from one that holds real load (Foundation)**. **Below this level, you're not in production.**

The Foundation isn't a different pipeline from Naive: it's the **same skeleton with three more stages switched on** — **Metadata** (offline), **Hybrid search** and **Re-ranking** (online). Naive RAG is exactly the Foundation with those three switched off. The other three links (Ingestion, Chunking, Indexing) are already in Naive: the Foundation doesn't introduce them, it raises the bar on *how* you do them.

| Step | In short | Where mediocre RAGs fail |
|---|---|---|
| **Ingestion** *(shared with Naive)* | PDF/HTML/scans → structured text (§3) | It's the quality ceiling of the whole system. *Garbage in, garbage retrieved*: no downstream stage recovers a dirty extraction. |
| **Chunking** *(shared with Naive)* | Splits documents into retrievable units (§3) | Chunks that cut tables or sections → noisy retrieval. The Markdown structure defines sensible boundaries; blind fixed-length cutting doesn't. |
| **Metadata** *(+ Foundation)* | Adds attributes: source, date, section, type | *Conditional on the use case.* It enables filters and disambiguation (e.g. "only documents after 2024"). Without it, retrieval is blind to context and can't tell an old version from a new one. In enterprise it also includes **ACLs**: if different users can read different documents, the permission must be stored as per-chunk metadata and applied as a filter at query time — it's prepared here, at ingestion, not added later (security is covered in a dedicated document). |
| **Indexing** *(shared with Naive)* | Embedding + building the vector index | *This is where the real choices are:* embedding model, dimensionality, index type (HNSW/IVF), quantization. **These are the parameters that govern the cost, latency and quality of retrieval.** Naive already has them, but it's the Foundation that treats them as decisions, not defaults. |
| **Hybrid search** *(+ Foundation)* | Semantic search **+** exact-word search. Naive uses only the semantic half. | Semantics alone misses queries based on exact tokens: product codes, acronyms, proper names. Exact-word search catches them literally; merged, they cover both failures. |
| **Re-ranking** *(+ Foundation)* | A cross-encoder that reorders the candidates (§ *The models involved*) | The single lever that moves perceived quality the most. It reorders the retrieval shortlist, but it doesn't recover what retrieval has already lost: it's only as good as the layer below. |

The three stages the Foundation **adds** to Naive are therefore **Metadata**, **Hybrid search** and **Re-ranking**: the first enables filters (retrieving by meaning *inside* a subset already narrowed down by attribute), the second covers the failures of purely semantic search, the third reorders the shortlist before the prompt. Ingestion, Chunking and Indexing aren't new — they're the skeleton shared with Naive; what matters is *how well* you do them, and it's in **Indexing** that the engineering choices with a direct impact on cost and latency are concentrated.

A note on chunking, worth knowing without going deeper here: when the size/context trade-off of §3 isn't enough — typically on corpora with long, interdependent sections — there are finer levers. The two most used: **parent-document retrieval** (small chunks are indexed for matching precision, but the larger "parent" they come from is passed to the prompt) and **contextual retrieval** (at ingestion, each chunk is prefixed with a short summary of its context before its embedding is computed). They are levers to switch on for difficult corpora, not by default — but when a client complains that "retrieval finds the right spot but the answer is incomplete", the problem is almost always here.

> **Key points** — Most of the quality gains in production come from here, not from the advanced patterns. Before climbing the *capability ladder* (§11), a well-built Foundation — that is, the three stages that separate it from Naive: **metadata + hybrid search + re-ranking** — is already the difference between a demo RAG and one that holds real load.

### 7.1 Explicit source citation (offline-anchored)

A very common requirement in enterprise — legal, compliance, support: the answer doesn't just have to be correct, it has to **say which document (and ideally which part of the document) each statement comes from**. It's worth keeping it separate from the two nearby concepts already seen, because they get confused:

- **Traceability** (§3) — *which chunks I passed to the model*. It's a fact of the system, and free: you always know the retrieved chunks.
- **Faithfulness** (Self-RAG/CRAG, §9) — *is the answer really supported by those chunks*, or did the model hallucinate or mix in internal memory? It's a downstream check, and costs an extra inference round.
- **Citation** — *the answer shows the user the verifiable source* (file, page, section) of every statement. It's what the typical requirement calls "must state its sources", and it's mostly an **offline-phase** problem, not a model problem.

The key point: fine-grained citability is **prepared upstream**, at ingestion. If at chunking time you didn't keep *where each chunk came from*, no prompt can recover it at query time — the information is already lost. The lever is therefore **per-chunk metadata** (§7, Metadata row), taken down to citation granularity.

**What to do, concretely:**

1. **Per-chunk provenance metadata, at ingestion.** Each chunk gets a `source_id` / file name and — this is what gives "the part of the document" — a positional anchor: page (PDF), heading/section (from the parsed Markdown, §3), possibly a character offset or chunk number. They're the same Foundation metadata, pushed down to citation level. *This is where the maximum granularity you can get downstream is decided:* if you anchor only to the file, you'll cite only the file; if you anchor to page/section, you'll be able to cite the portion.

2. **Citation-aware chunking.** Chunk size also becomes the **resolution of the citation**: big chunks → vague citations ("document X"), small, well-bounded chunks → precise citations ("X, §4.2"). This adds a constraint to the usual chunking trade-off (§3): align boundaries to citable units (section, clause, article), don't cut through the middle.

3. **Carry the metadata all the way to the prompt.** The chunks injected into the context must reach the model *labeled* with their provenance (e.g. `[source: contract_2024.pdf, §4.2]` at the head of each chunk), together with an explicit instruction: *cite the source of every statement using the labels provided, don't invent others*. The model doesn't "know" the provenance: you pass it, together with the text.

4. **Citation verification (optional, but it's the critical point).** A model that cites can still get the attribution wrong — cite the wrong chunk, or cite a plausible one that doesn't really support the statement. If citation correctness is critical (legal/compliance), you need a downstream check, which is exactly **Self-RAG / CRAG territory (§9)**: validating that the statement is actually supported by the cited chunk. *This is where citation and faithfulness join:* a correct citation is faithfulness made explicit and verifiable.

> **📐 Box — "Cites" ≠ "cites correctly"**
>
> A system that adds source labels *looks* fine in a demo. But showing a citation and citing the *right* source are two different things, and in a legal/compliance context the second one is what counts. Three ways to get it wrong, in order of severity:
>
> - **Unsupported citation** — a cited source that *no provided chunk* supports. It's the failure a legal reviewer notices first.
> - **Incorrect citation** — it cites a chunk that doesn't really support the statement (plausible but wrong).
> - **Missing citation** — a true statement that isn't attributed, or only partly attributed (the claim rests on three chunks, it cites one).
>
> Measuring it requires a judgment for every (statement, cited chunk) pair: *does that chunk really support that statement?* It's done with human review on a reference sample, or by automating that judgment with a second model — the same logic as the Self-RAG/CRAG round of point 4, used here as a metric instead of a correction. The details of *how* it's quantified are beyond the scope of this document.
>
> **Why it matters in presales:** it moves the conversation from *"our RAG cites its sources"* (everyone says that, it's a checkbox) to *"it cites correctly, measured on your corpus"* — a reproducible number on a proof of value, defensible in a tender. It's what separates a measured integration from a slideware one.

> **Key points** — "Stating the sources" isn't a feature of the generative model: it's a **metadata chain** that starts at ingestion. It's enabled offline (per-chunk provenance + chunking aligned to citable units), made effective in the prompt (labeled chunks + an instruction to cite only those labels), and — if attribution has to be *guaranteed* — verified downstream with Self-RAG/CRAG. The operating rule: *the citation granularity you get at query time can't exceed what you saved at ingestion.* If you need to cite the section, you have to decide to save it **before**, when you split the documents — not after.

### 7.2 Measuring the Foundation — evaluation

The whole chapter says that hybrid search and re-ranking "move quality". The obvious question: **how do you prove it?** Without measurement, every choice in the pipeline — chunk size, value of k, re-ranker on or off, embedding model — is made by feel, and every "improvement" is an opinion. You can't tune what you don't measure.

There's no need to go into the details of the metrics here, but you do need to know that evaluation lives on **two distinct levels**, and that confusing them leads to wrong diagnoses:

- **Retrieval quality** — *did the right chunk make it into the shortlist?* The basic metric is **recall@k**: for a question whose answer chunk you know, is it among the top-k retrieved? If not, no downstream stage can save the answer — the re-ranker reorders, it doesn't resurrect (§7). There are finer ranking metrics (MRR, NDCG), but recall@k alone already separates a retrieval that works from one that doesn't.
- **End-to-end quality** — *is the final answer faithful and relevant?* Here you measure faithfulness (is the answer supported by the chunks?) and answer relevance (does it answer the question?). It's typically automated with a second model as judge (*LLM-as-judge*); there are dedicated frameworks (RAGAS is the most cited reference) that package these metrics.

The prerequisite for both levels is a **golden set**: a collection of question → expected answer (and expected chunk) pairs, built on the client's real corpus. It doesn't need to be big — 50–100 well-chosen pairs are enough to steer decisions — but without it, evaluation can't start. It's a setup cost to plan for at the beginning, not an add-on afterwards.

> **Key points** — A correct diagnosis needs the two levels kept separate: if recall@k is low the problem is upstream (chunking, embedding, hybrid search); if recall is high but faithfulness is low, the problem is downstream (prompt, model, packing). Measuring only end-to-end tells you *that* something is wrong, not *where*. In presales, it's the same logic as the §7.1 box extended to the whole system: "retrieval measured on your corpus, with reproducible numbers" is what a serious proof of value puts on the table.

### 7.3 Index lifecycle

The offline phase of §3 looks "one-off", but in production the index is a **living system**: documents change, and every change has to be reflected. Three warnings worth knowing before signing a project, because they're recurring costs the client typically discovers at the first update:

1. **Document updates → re-indexing.** New or modified documents require re-ingestion (parsing → chunking → embedding → index). The choice is between **incremental** (only the documents touched — the norm) and **full** (the whole corpus). Incremental, though, requires knowing *what* changed: you need a change-detection mechanism on the source, which has to be designed, not improvised.

2. **Changing the embedding model = re-indexing everything. Always.** Vectors from different encoders live in different spaces and **can't be compared or mixed** in the same index — this holds even between different versions of the same model. Upgrading the encoder (for quality, cost, or the model's end of life) means re-embedding the entire corpus: on large corpora it's a significant batch job, to be planned and budgeted. It's the most underestimated operational constraint of RAG, and it ties back to the Indexing row of §7: choosing the embedding model is a decision with an exit cost, not an interchangeable default.

3. **Corpus versioning.** For citation (§7.1) and for compliance you need to know *which version* of a document generated which chunk: if you cite "contract_2024.pdf §4.2" but the file was replaced yesterday, the citation points to nothing. Document version and ingestion timestamp are per-chunk metadata to plan from the start.

> **Key points** — Ingestion isn't a one-shot cost: it's a recurring operating cost with three items (incremental re-indexing, re-embedding on encoder change, versioning). In an offer, you size the update pipeline, not just the first-load one — it's the difference between selling a proof of value and selling a system that lives.

## 8. Knowledge Representation

These change the **form** in which knowledge is represented offline, while you're building the **knowledge base** — plain text, an entity graph, or multimodal content. It's the form of the *index you draw from*, not of what ends up in the prompt: the LLM's context always receives only the subset retrieved at query time, never the whole representation. This choice changes *what you can retrieve and how*, not the way the system searches when the question arrives. They can be chosen as alternatives or combined.

### Standard RAG — chunk repository

- **Requirement:** a text corpus, answers located in one place in the corpus.
- **Rationale:** knowledge is a set of independent chunks; retrieving the few chunks most similar to the question is enough to answer.
- **When it's enough:** FAQs, internal KB, homogeneous documentation. With a solid Production layer it covers most enterprise cases.
- **Classification note:** it's the **default representation of the Production layer (Level 1)**, not a Level 2 option — the chunk repository is what the Foundation already builds on its own. It appears here as a **point of comparison**: it's the baseline GraphRAG and Multimodal depart from, and choosing it means not climbing a level. That's why the selection table (§13) and the composition examples (§14) classify it as L1.

### GraphRAG — graph representation

- **Requirement:** documents that reference each other, or **multi-hop**/global questions on the corpus.
- **Examples:** "which clauses of the contract contradict regulation X?", "recurring themes across 500 documents".
- **Rationale:** it builds a knowledge graph of entities and relationships and traverses it. Plain similarity search fails by construction: the answer isn't in a single chunk, but in the *relationship* between distant chunks — and retrieving the most similar pieces, one by one, doesn't bring that connection out.
- **Cost:** heavy ingestion (entity/relationship extraction, graph construction).

### Multimodal RAG — multimodal representation

- **Requirement:** meaning also lives in tables, figures, diagrams, scans.
- **Rationale:** it retrieves over non-textual representations (visual embeddings, descriptions, table extraction).
- **Classification note:** it has a foot in two levels. The *cost* lives in the Foundation (OCR, captioning, table extraction); the *capability* lives here, because it changes what the system can represent and query. It's classified by its systemic effect, not by where the work happens.

### Notes

GraphRAG is the RAG for **multi-hop** questions: the ones no single chunk answers, because the answer is the relationship between pieces of information spread over several documents. It builds an entity-and-relationship graph at ingestion and traverses it — more powerful on connections, but with a much higher preparation cost.

GraphRAG and Multimodal are two **independent** levers of the same level: both act on the *form* of knowledge, but on different dimensions. Multimodal changes the **modality** (it also retrieves tables and figures, not only text); GraphRAG changes the **structure** (the relationship graph seen above). Neither is mandatory or a consequence of the other — one comes from the type of content, the other from the type of question — so you switch them on separately, according to the requirement. They can also be combined (**Multimodal GraphRAG**: a graph with nodes extracted from figures and tables too), but it's the most expensive configuration, to be kept for cases that really need both.

## 9. Retrieval Intelligence — Level 3

At L2 you chose the **form of the index** — text, graph, multimodal. At L3 that index stays **identical**, whatever form the levels below gave it: you don't touch the form of the knowledge, you change **how the system decides to query it** when the user's request arrives.

Up to here, retrieval was a **fixed** procedure: the question arrives, the most similar chunks are searched, the answer is generated — always the same round. At L3 the system gains a **decision logic** at query time, which comes down to three questions: *whether* to retrieve, *how* to search, *whether* what it found is enough. The four patterns answer these questions at two different moments — **before** retrieval (deciding whether/how to search) and **after** it (judging whether the result is good enough):

- *before:* **Adaptive** (whether to retrieve), **RAG-Fusion** (how to search)
- *after:* **Self-RAG** (is the answer supported?), **CRAG** (is the retrieval reliable?)

They are refinements to add **when the specific problem they solve shows up**, not by default: each costs at least one more inference round, and on a case that doesn't need it it only adds latency and cost (→ §12, this is where query-time fan-out starts to weigh on sizing).

### Adaptive RAG — upstream semantic router

- **Requirement:** not every query needs retrieval (a mix of chit-chat and knowledge questions).
- **Rationale:** a router classifies the question *before* acting and chooses its path — from "no retrieval" (chit-chat) to simple or multi-step retrieval, depending on complexity. Always retrieving, in the same way, is waste and noise when the question doesn't need it. The decision is made once, upstream.

### RAG-Fusion — several phrasings of the same question

- **Requirement:** vague or under-specified questions.
- **Rationale:** it generates several rephrasings of the same query, retrieves for each one and **merges the results**. A single way of phrasing the question misses relevant chunks that a different phrasing would have found.
- **Cost:** it multiplies retrieval by the number of variants — query-time fan-out, but by a known and predictable factor (§12).

### Self-RAG — the model evaluates itself

- **Requirement:** faithfulness to the source is critical, hallucination is unacceptable.
- **Rationale:** it's the **model itself**, during generation, that judges whether retrieval is needed and whether what it retrieved is relevant and sufficient to support the answer. A reflective decision (introspection on its own output), not an external router.
- **Cost:** +one round of self-evaluation.

### CRAG (Corrective RAG) — it corrects retrieval

- **Requirement:** potentially incomplete corpus coverage, where "I don't know" is better than an invention.
- **Rationale:** it evaluates the **confidence of the retrieval** and, when it's low, triggers a **fallback** instead of answering in the void: it rephrases the query or searches an external source. It's *corrective* — it doesn't just judge, it acts to fix.
- **Cost:** +one round of evaluation, +the fallback when it triggers.

> **Key points** — Self-RAG and CRAG look close but judge different things: Self-RAG checks *its own answer* (is it supported by what I have?), CRAG checks the *retrieval* (did I retrieve reliable material?) and corrects it if not. Adaptive and RAG-Fusion sit *upstream* instead, on whether/how to search. None of them touches the index: they only change the querying strategy.

## 10. Agent Orchestration — Level 4

The agent isn't a variant of retrieval: it's the top layer, which **uses retrieval as one tool among many**.

```
Agent
 ├─ RAG / GraphRAG        (knowledge retrieval)
 ├─ Web search
 ├─ SQL query
 ├─ API call
 └─ other tools
```

A clarification, because this is often confused. A RAG, even an advanced one with the levels enabled, is a **pipeline**: the steps and their branches are **predefined** by the programmer. An L3 pattern can change path — Adaptive routes based on the question, CRAG falls back if confidence is low — but it always chooses between branches **already written in the code**, it doesn't create new ones.

The agent replaces the pipeline with a **runtime loop**: there's no longer a path written in advance, but a control logic that decides the next step at every round — which tool to use, whether it's enough or another round is needed — and repeats until the question is solved. The difference isn't "there's a model / there isn't": it's **fixed branches decided beforehand** versus **a flow decided at runtime by the agent's logic**.

> **Watch out for *agentic-washing*.** Many systems sold as "AI Agents" are actually RAG pipelines with predefined workflows. The presence of retrieval, tool calls or workflows **doesn't imply** an agentic architecture: a system that calls three tools in an order hard-wired by the programmer is a pipeline, even if each of those tools is an LLM. There's only one discriminant — *who decides the flow*: if the branches are written in the code (even many of them, even with LLMs inside), it's L3 or below; it's L4 only when the model decides **at runtime** which steps to take and in what order, choosing among possibilities not all foreseen in advance. The distinction isn't academic: it changes sizing (§12, fan-out becomes a distribution) and it changes what you're buying — a deterministic orchestration you govern, or a loop whose cost and behavior have to be measured at runtime.

It also changes **who owns the context**. In the pipeline it's the programmer who decided, in advance, what goes into the prompt: system prompt, the retrieved chunks, any history, according to a fixed packing rule. The shape of the context is predictable because someone hard-wired it. With the agent, ownership moves to the runtime loop: it's the agent that decides, round by round, what to accumulate in the context — the results of which tools, of how many steps, what to keep and what to drop. The context is no longer assembled by a fixed rule, but built dynamically by its decisions. That's the structural reason why at L4 prefill length — and therefore sizing — stops being predictable (→ appendix, §12).

- **Requirement:** the question has to be broken down, needs several retrievals in sequence, several heterogeneous sources, or tools beyond retrieval (querying a DB, calling an API, acting on a system).
- **Rationale:** retrieval becomes **one tool among the others**; the agent decides *at runtime* which tools to call, in what order and how many times. The value is no longer "doing RAG" but combining knowledge retrieval + tool use + workflow execution + reasoning.
- **Orthogonality (the key point):** L4 **doesn't presuppose** L2–3. An agent can orchestrate Standard RAG + web + SQL without any GraphRAG or CRAG underneath. You **can** go from L1 straight to L4: the agentic layer is orthogonal to the depth of the underlying retrieval — it's the *decision/orchestration* axis, not the *representation* axis.
- **Cost:** this is where sizing breaks. Fan-out is no longer predictable as it is at L3 (variants ×N, +1 fallback round): the agent decides at runtime how many tools and how many rounds, so the inference load per query becomes a **distribution, not a number** (→ §12).

## 11. The progression as maturity (capability ladder)

```
Foundation          Representation          Retrieval Intel.       Agent Orchestration
Finds documents  →  Represents them better → Retrieves them better → Reasons and coordinates
   (L1)                  (L2)                    (L3)                  (L4)
```

It's a **ladder of capabilities, not a mandatory migration path**. Most enterprise systems live well at **L1 with a solid Foundation**, climbing to L2 only if the content (graph, multimodal) requires it: that's where most of the value already is. You climb only when the requirement forces you to.

And the "ladder" is a convenience for explaining, not an order to follow in sequence. As seen in §6 and §10, the two axes — *representation* (L1–L2) and *decision/orchestration* (L3–L4) — are **independent**: you can be at L4 with L1 retrieval underneath, without going through L2–L3. You don't climb along a single line, you climb **on the axis the requirement touches**.

The criterion for deciding whether a component deserves a higher level:

> A component moves up a level when it changes **what the system can answer**, not when it only adds work to the pipeline.

In practice: if a more sophisticated pattern doesn't unlock a *new question* you couldn't answer before — but only makes the flow more elaborate — you're not climbing a level, you're adding cost and latency. It's the same logic as §15: complexity goes where it changes the result, not where it's technically interesting.

## 12. The axis that changes inference sizing

Climbing the levels, three axes grow together. The first two are well known; the third ties this framework to inference engineering and is rarely made explicit — and it's the one that makes **sizing a RAG different from sizing an agent**.

| Axis | Foundation + Representation (L1–L2) | Retrieval Intelligence (L3) | Agent Orchestration (L4) |
|---|---|---|---|
| **Decision autonomy** | fixed rules | the system chooses the strategy among foreseen branches | the agent decides the flow at runtime |
| **Inference fan-out** | 1:1 with the query (or offline) | limited but predictable | dynamic, N calls per query |
| **Sizing difficulty** | Cr_closed estimable in advance | Cr still computable (×known factor) | Cr no longer deterministic |

*Cr_closed and Cr (capacity per replica) are the capacity measures of the Dielabs benchmark framework — see [One Capacity Is Not Enough](/frameworks/one-capacity-is-not-enough.html).*

L1 and L2 sit in the same column not because they're the same level, but because they behave the same way on **query-time fan-out**: neither changes it. The difference between them is where the cost falls, not how many calls per query.

- **Foundation (L1)** is largely *offline* work — batch ingestion. It's throughput-first, **Cr_closed** territory: you maximize pages per second at saturation, with no per-request latency SLO.
- **Knowledge Representation (L2)** doesn't touch query-time fan-out: it moves the cost *upstream*, into ingestion (graph extraction for GraphRAG, OCR and captioning for Multimodal). At query time there's still one round, as in L1.
- **Retrieval Intelligence (L3)** introduces query-time fan-out, but **limited and predictable**: CRAG adds roughly one evaluation round, RAG-Fusion multiplies by the number of variants — a factor known in advance. Cr stays estimable by applying a constant multiplier.
- **Agent Orchestration (L4)** breaks predictability: fan-out is *dynamic*, decided at runtime by the agent's logic (§10). Cr is no longer 1:1, nor can it be multiplied by a fixed factor — how many calls the agent will make is known only while it runs. And this is where the circular dependency of Little's Law bites hardest: you can't size concurrency without knowing latency, which in turn depends on a per-query load that isn't fixed.

> **Key points** — The sizing jump isn't gradual across the levels: it's all in the move to L4. Up to L3, the inference load per query is a **number** (1, or ×N with N known, or +1 fallback), and Cr can be computed in advance. At L4 it becomes a **distribution**: sizing is no longer done on an expected value, but on mean fan-out and percentiles (p95/p99 of calls per query). Sizing an agentic system ≠ sizing a RAG.

## 13. Selection table — requirement → level → pattern → cost

| Starting requirement | Level | Pattern | Dominant cost |
|---|---|---|---|
| Text corpus, answers located in one place | L1 | Standard RAG | low — all in the Foundation |
| Documents that reference each other, multi-hop questions | L2 | GraphRAG | heavy ingestion (graph construction) |
| Meaning also in tables, figures, scans | L2 | Multimodal RAG | ingestion (OCR, captioning, table extraction) |
| Not every query needs retrieval | L3 | Adaptive RAG | query-time fan-out, limited |
| Vague or under-specified questions | L3 | RAG-Fusion | retrieval ×N variants, known factor |
| The answer must state its sources (file + portion) | L1 | Foundation + per-chunk provenance metadata (§7.1) | offline (citation-aware chunking + metadata); +L3 verification if guaranteed |
| Retrieval quality to be proven with numbers | L1 | Evaluation: golden set + recall@k + faithfulness (§7.2) | offline (golden set construction) + evaluation rounds |
| Faithfulness critical, hallucination unacceptable | L3 | Self-RAG | +1 self-evaluation round |
| Incomplete coverage, "I don't know" better than an invention | L3 | CRAG | +1 evaluation round, +fallback when it triggers |
| Decomposition, several sources/tools, workflows | L4 | Agent Orchestration | dynamic fan-out, non-deterministic Cr |

The table reads one row per pattern, but the patterns **aren't mutually exclusive**: they're chosen based on the requirement that shows up, and they add up when requirements coexist (§14). The "dominant cost" column says *where* the weight falls — ingestion for L1–L2, query-time fan-out for L3–L4 — and it's the same split as the sizing chapter (§12).

## 14. Composition examples

Patterns **add up**, they don't exclude each other — and in most real cases the sum is short. The point of each example isn't how many levels to switch on, but how many you can *leave switched off*.

- **FAQ / simple internal KB** → Foundation + Standard RAG (L1). A well-built Foundation (hybrid search + re-ranking) is enough; no reason to climb.
- **First-level support on a company KB** → Foundation + Standard RAG (L1). The flow is fixed: request → retrieval → answer. It stays L1 as long as the system always answers the same way from a single source — "agentic AI" here is marketing, not architecture. It climbs only when a *decision* shows up: telling resolvable tickets apart from those to escalate (→ L3), or also querying CRM and ticketing besides the KB (→ L4).
- **Law firm** → Foundation + Standard RAG as the baseline. Many real firms live perfectly well at L1 with hybrid search + re-ranking. The higher levers are evaluated by requirement, not by default: GraphRAG **only if** multi-hop references between contracts and regulations are needed; Self-RAG/CRAG **only if** every statement has to be citable and verifiable (that's faithfulness to the source, not multi-hop — two separate needs); the agentic layer **only if** the work requires coordinating several systems.
- **Technical documentation with datasheets** → Foundation + Multimodal (L2). The agentic layer is added only if the questions are *comparative* across several datasheets (decomposition + several retrievals).
- **General-purpose company assistant** → Foundation + Adaptive + RAG-Fusion (L3). Adaptive because traffic mixes chit-chat and knowledge questions; RAG-Fusion because questions often arrive under-specified.

## 15. Summary position

> In production RAG, most quality gains come from ingestion quality, hybrid retrieval and re-ranking. Advanced patterns such as GraphRAG, CRAG or Agentic RAG add latency and operational cost, so they should be introduced only when a specific retrieval limitation appears — not as a default architectural choice.

Put complexity where it's needed, not where it's technically interesting: it's the difference between a paper architecture and an enterprise system that holds. Every level above the Foundation has to be justified by a new question you couldn't answer before (§11) — not by the fact that the pattern exists. And it has to be counted twice: once in quality, once in **inference cost** (§12), because every extra round is load on a GPU that someone has to size and pay for.

---

## Appendix — RAG as a multi-model inference workload

§12 looks at sizing from the side of the *capability ladder* (the higher you climb, the less deterministic the fan-out). Here the circle is closed from the infrastructure side: what it means, for serving, that a RAG is a **multi-model** system and that the context isn't owned by the client.

#### The three inference profiles in sequence

Every query goes through up to three models, with different load profiles that **don't add up in a trivial way**:

| Stage | Model | Inference profile |
|---|---|---|
| Query embedding | encoder | single forward pass, no KV cache, batch-friendly, low and predictable latency |
| Retrieval | *(no model)* | closeness search on the index — **not GPU inference**, it lives on the vector DB |
| Re-ranking | cross-encoder | forward pass on each candidate pair, medium cost, grows with the number of candidates |
| Generation | decoder | autoregressive, growing KV cache, the heavy stage |

Two sizing consequences:

- **Retrieval isn't GPU serving load**: it's I/O and CPU on the vector DB index. On large corpora it weighs, but on the *database* budget, not the LLMs'. It's the typical mistake of people who size "only the models".
- Encoder and re-ranker are **throughput-first**, with contained latency; the generator is **latency-bound** (TTFT + ITL) and dominates the cost per query. Characterizing them with a single Cr is misleading — each has its own. It's the operational reason why **Cr(RAG) ≠ Cr(generator)**: sizing a RAG means sizing three distinct profiles, not just the model that answers.

#### The context owner isn't the UI

As in the agentic runtime, the **model is stateless**: on every call it receives a block of tokens and forgets it right afterwards. What goes into that block is assembled by an **intermediate orchestration tier**, not by the UI — which is a thin client: it captures the raw input and renders the output.

```
UI                Orchestrator (context owner)               Model
─────             ───────────────────────────────────       ───────
raw input     →   system prompt                          →  stateless:
                  + history / memory                         receives the block,
                  + retrieved chunks (RAG)                   answers,
                  + tool results (agent)                     forgets
final output  ←   + truncation policy / token budget     ←  answer
```

**Sizing implication — the user doesn't decide the prefill length.** The user types a few words; the block that hits the GPU in prefill can be thousands of tokens (system prompt + chunks + history). The distribution of *input lengths* — the one that governs TTFT, prefill compute and the KV cache footprint — is therefore a **packing policy of the orchestrator**, not a property of the user. In **RAG** (L1–L3) the policy is static (chunks × size + a constant): predictable input length, and Cr_closed with it. In the **agent** (L4) it becomes dynamic — the context owner decides at runtime how much to pack and how many rounds to run — and it's the other side of §12: what breaks the predictability of Cr at L4 is exactly the context assembly that stops being fixed, and Little's Law biting on a *distribution*, no longer on a number.

> Terminology note (tentative): this tier is starting to be called *context engineering* / *context assembly layer* — a recent term, not yet standardized; used here for what it points to, not as a settled label.

---

*Original Dielabs work by Diego Bardella.*
