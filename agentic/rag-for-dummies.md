---
layout: default
title: "RAG for Dummies"
---

# RAG for Dummies

How a system that answers questions from a company's own documents actually works.

## The starting problem

An LLM knows an enormous amount in general, and nothing at all about your company's documents — contracts, procedures, mail, manuals. Ask it something it does not know and it will often invent an answer, with great confidence.

**RAG** (Retrieval-Augmented Generation) fixes this: first it goes and finds the right documents, then it asks the LLM to answer using only those.

The analogy running through this document is a **library**: someone catalogues, someone selects, someone writes the answer.

## The four models, in working order

> The four models are listed in the order they work in when a question arrives. But the embedding model has already done most of its work long before, while the archive was being prepared — it appears twice: first on the documents, then on the question. Details in the section on the two flows.

### Rewriter, the question translator

Users write badly: *"and the earlier one?"*, or *"how much does it cost"* without saying what. The rewriter rewrites the question into a clear, self-contained form, puts the conversation context back into it, and adds synonyms or in-house terminology.

If the question enters wrong, the whole rest of the chain works beautifully on the wrong thing. It is the one error that cannot be recovered downstream.

### Embedding, the cataloguer

It reads every document and assigns it a position by meaning: similar texts end up close together even when the words differ. This is what lets you find the right document when the user does not use the words written inside it.

### Reranker, the selector

The initial search pulls out 20-50 candidates, fast but rough. The reranker reads them one by one alongside the question and picks the 3-5 that are genuinely best.

This is the piece most often missing from mediocre RAG systems: it costs little to add and improves quality a lot.

### Generator, the explainer

The LLM itself. It takes the few selected documents and writes the answer in natural language, citing sources. It is the part the user sees, and the part that depends most on everything that happened before it.

### The judge, the reviewer outside the chain

It takes no part in the answer. It runs separately, on a sample of answers already given, and measures two distinct things.

**Did retrieval work?**

- **Recall** — of the documents that were needed, how many did it find? If this is low, the problem is upstream: chunking, embedding, or a question that entered wrong.
- **Precision** — of those handed to the generator, how many were actually useful? If this is low, the LLM is drowning in irrelevant material and a better reranker is needed.

**Did the answer work?**

- **Groundedness** — is every claim supported by the retrieved documents? Measured with a second LLM, asked whether each sentence of the answer is justified by the attached documents. When this drops, the system is starting to invent.
- **Relevance** — does the answer actually answer the question, or is it talking about something else?

The distinction matters because the remedies differ: low recall is not cured by tweaking the generator prompt, and low groundedness is not cured by changing the embedding model. Without a judge there is no way to notice that the system is degrading — and it degrades slowly, as documents are added.

A note on names: **retrieval** is the phase, not a model. The models in that phase are embedding and reranker.
## The two working flows

A RAG system is not one process: it is two, running at completely different moments.

### Ingestion flow, offline and ahead of time

```
Documents (PDF, web pages, mail, databases)
   |
Text extraction + cleaning
   |
Chunking - cut into cards of a couple of paragraphs
   |
Embedding model -> a position by meaning for each card
   |
Write to the vector database (position + text + housekeeping fields)
```

It runs whenever you like, overnight if you want, and it can take hours: nobody is waiting. It is re-run when new documents arrive.

#### Cutting into cards

The archive cannot be read in full every time, so it is cut into cards — **chunking**. This is the moment where half the quality of the system is quietly decided.

One rule: every card must make sense on its own. If a sentence is left cut in half, the card speaks into the void.

Size is a trade-off:

- **Cards too small** — you find the right one but it lacks context (*"as set out in paragraph 4"* — paragraph 4 is in another card).
- **Cards too large** — the average topic of the card no longer resembles anything specific, and questions stop finding it.

The judge tells you whether you chose well: retrieval that misses useful documents means cards are too small; cards where the useful part is buried in unrelated material means they are too large.

### Query flow, online and in real time

```
User question
   |
Rewriter -> rewritten, self-contained question
   |
Embedding model -> position of the question (same model as the ingestion flow)
   |
Vector database search -> 20-50 candidates
   |
Reranker -> the 3-5 cards that are genuinely useful
   |
Generator (LLM) + selected cards -> answer with sources
   |
Answer to the user
```

The user is staring at the screen: a total budget of 2-5 seconds, split across four steps in a row.

#### The generator system prompt

In front of the documents you always put a fixed text, written once, that sets the rules of the game:

```
You are the assistant for the company documentation.

Answer the question using ONLY the documents below.
For every claim, state which document it comes from.

If the documents do not contain the answer, write:
"I cannot find it in the documents." - and nothing else.
Do not use what you know in general, do not guess, do not fill gaps.

=== DOCUMENTS ===
{the cards found by the search}
=== END OF DOCUMENTS ===

Question: {the user question}

Answer:
```

Three details that are not merely stylistic:

- **The positive rule first, the way out second** — "use only the documents" says what to do; "do not invent" on its own only says what not to do, and an LLM with no positive instruction fills the gaps anyway. "I do not know" has to be an explicitly permitted answer.
- **"and nothing else" carries weight** — without it, the model tends to admit it does not know and then attempt an answer anyway, immediately after.
- **Mandatory citation makes the error visible, not impossible** — an LLM can invent the citation too. Its job is to make errors checkable, by a person or by the judge, not to prevent them.

Treat this text as configuration: versioned, dated, changed only with a measurement before and after.

### The bridge between the two flows

The **vector database**: the ingestion flow writes it, the query flow reads it. The only model they share is the embedding model, which must be the same in both.
## Things that matter

**The embedding model has to stay the same.** Same version in both flows — two rulers that must measure in the same unit. Changing it usually means rebuilding the whole archive from scratch: on millions of documents, hours or days. Rare exception, to be verified case by case: some later versions of the same model stay compatible with each other.

**The judge runs separately, but it is needed.** It is the only instrument that tells you whether the system still works well after six months of added documents.

**The archive has to be kept current.** A document that is edited or deleted must disappear from the archive too, otherwise the system answers with stale information and nobody notices. You need a mechanism that updates only the cards that changed.

**Every answer must cite its sources.** This is not decoration: without citations, a wrong answer and a right one look alike.

**The system must be able to say "I do not know".** It does not happen on its own: it has to be imposed in the system prompt. It is the most frequent difference between a RAG system you can trust and one that invents.

**Cutting into cards is blind.** Even with a well-chosen size, the cut does not understand what it is cutting: a table gets split in half, a contract condition ends up separated from its exception. The system retrieves the right card and still answers badly, because the other half is missing. Three remedies, all in ingestion: let consecutive cards overlap a little, cut along the structure of the document instead of by word count, and attach a header to each card saying where it came from.

**Permissions have to be respected.** If a document is restricted to a few people, the filter belongs inside the search, not after it.

**The time budget is tight.** 2-5 seconds in total, split across four models in a row: every piece you add is latency that accumulates.

**Quality depends on the data more than on the models.** Messy, duplicated or contradictory documents: no model compensates for that. Data cleaning is the largest line of work and the least visible.

## Advanced features

Classic RAG is enough for the large majority of cases. These are the directions to take when it is not.

### Hybrid search

Alongside search by meaning (embedding), it runs exact-word search, for product codes, proper nouns and acronyms — where meaning does not help and the exact letters are what count. The two result lists are then fused. Almost always better than either technique alone, and it is the first upgrade to make.

### GraphRAG, the map of relationships

It first builds a map of who is who and what relates to what — people, companies, contracts, ties — then answers by navigating that map. It is for questions that no single document contains, such as *"which suppliers are connected to this group?"*. It pays off on relationship questions inside large archives; for simple facts, classic RAG is faster, cheaper and just as accurate. Building and maintaining the map costs a good deal.

### Agentic RAG, the RAG that thinks again

Classic RAG searches, answers, done. Agentic RAG can take several passes: it evaluates what it found, decides it is not enough, reformulates, searches again, and stops only when satisfied or out of budget. It is for complex multi-step questions. The price is time — from seconds to tens of seconds — and the cost multiplies.

### Adaptive RAG, the switchboard operator

A classifier reads the question and routes it: simple ones to classic RAG, relationship ones to GraphRAG, complex ones to agentic RAG. An emerging good practice, because it gives the best cost/quality trade-off — most questions are simple and get fast answers, only the rest get the full treatment. It is for when you already have several pipelines and meaningful volume.

### Multimodal RAG

The archive also holds images, diagrams, tables, screenshots and video, catalogued and retrieved like text. It is for technical manuals, documentation with drawings, catalogues.

### Federated RAG

The data cannot be centralised, for privacy or regulatory reasons. The system searches where the data lives, without moving it. It is for healthcare, banking, groups with separate legal entities, data subject to sovereignty rules.

### Memory and long documents

Techniques for making the system remember past conversations, and for handling documents too long to split naively, building multi-level summaries like a tree-shaped index. It is for continuous conversational assistants, or archives with documents hundreds of pages long.

## In short

| | |
|---|---|
| Where to start | Hybrid search + reranker, nothing else |
| What to measure immediately | Retrieval recall and precision, groundedness of the answer |
| What to add only if needed | Graph, agentic, adaptive routing |
| Where the bulk of the work goes | Cleaning and updating the data, not the models |

The golden rule: start from the simplest thing that works, measure, and add complexity only where the numbers say it is needed.

---

*Original Dielabs work by Diego Bardella.*