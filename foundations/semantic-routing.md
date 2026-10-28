---
layout: default
title: "Semantic Routing — Why the Decision Layer Must Be Fine-Tuned on the Domain"
---

# Semantic Routing — Why the Decision Layer Must Be Fine-Tuned on the Domain

## Key points

> - A semantic router sends each request to the most suitable model (lightweight vs reasoning, local vs cloud) by classifying it by task complexity.
> - The component that decides is not an LLM but a small embedding model that measures **semantic similarity**, not **task complexity**: two different things.
> - Out of the box, the pre-trained model mistakes "a prompt that looks complex" for "a prompt that needs a powerful model", producing systematic misrouting.
> - The fix is to fine-tune the embedding model on data from the real domain. The dominant cost is not compute (hours of CPU) but defining the boundaries between tiers.
> - **The router must be fine-tuned on the tier boundaries of the real domain it will work in, not on generic data: it has to learn where the line between one tier and the next falls in the very traffic it will route.**
> - **The weak point — and therefore where to invest — is the decision layer that classifies upstream, not the infrastructure around it: the latter can be perfect and still produce systematic misrouting.**
> - Routing accuracy is an SLO and cost problem, not an uptime problem: a router can always be available and still get one request in five wrong.

---

## 1. What a semantic router is and why it is needed

Not all requests to an inference system weigh the same. A dry factual question ("what is the capital of France?") and a multi-step reasoning problem ("design the microservice architecture of an e-commerce site") need very different resources. If both land on the largest model available, you burn compute and budget for nothing and worsen the latency of the whole system.

The semantic router adds a layer that **classifies the incoming request and routes it to the appropriate backend**: simple queries go to light, cheap models, complex ones to models capable of reasoning. In an on-premises or sovereign context the same mechanism takes on a second meaning — it can decide **what stays inside the infrastructure and what goes out to the cloud**, effectively becoming a data residency control point.

## 2. Anatomy — two components of opposite nature

The most common confusion is to think of the router as a single, "intelligent" block. In reality it is made of two parts with opposite characteristics, and keeping them apart is essential to understand where the value is and where the risk is.

### 2.1 The decision layer — embedding model plus anchors

The decision core **is not a generative LLM**. It is an **embedding model** — a small encoder (in the order of tens of millions of parameters) that turns a text into a numeric vector. The router keeps a set of **anchors**: example sentences already labeled by complexity tier (e.g. SIMPLE, MEDIUM, COMPLEX, REASONING). When a query arrives, the embedding model vectorizes it and the router measures how close that vector is to the anchors of each tier, typically with cosine similarity. The query inherits the tier of the nearest anchors.

> **Analogy.** It is like sorting mail. You do not compare each letter with the destination offices, but with examples of the kind "this goes to department A, this to department B". The semantic part decides the department; which desk serves it, inside that department, is a fixed rule written on the wall.

### 2.2 The tier → backend map — a deterministic lookup

Once the tier is decided, picking the model is a **deterministic table**, a plain if/else you configure in a file: SIMPLE → local model, REASONING → cloud model, and so on. This part is not probabilistic and never errs: it faithfully executes the rule you gave it.

| Tier | Example request | Backend (example) |
|---|---|---|
| SIMPLE | "What is the capital of France?" | Light local model |
| MEDIUM | "Write a Python function that parses a CSV" | Local model |
| COMPLEX | "Design a microservice architecture for e-commerce" | Capable cloud model |
| REASONING | "Prove by induction that the sum 1..n = n(n+1)/2" | Cloud reasoning model |

**Where the risk is.** The table is trivial and reliable; all the risk is in the decision layer. If the classification picks the wrong tier, the table — while working perfectly — routes the query to the wrong model. The table faithfully executes a wrong decision. That is why you work on the semantic component, not on the map.

## 3. Why the pre-trained model fails

A generic pre-trained embedding model is optimized for **semantic similarity**, not for **task complexity**. It groups texts by how they look and read, not by what it takes to execute them. It is a subtle but decisive distinction: two sentences can "look alike" while needing completely different models.

The resulting error pattern is structural and predictable, not random noise:

- Longer prompts look more complex.
- More descriptive language looks more complex.
- Structured instructions look more complex.

The typical result is over-classification: mid-tier requests are pushed into the top tier because they contain more context or detail, not because they really need more reasoning. The economic effect is silent over-provisioning — you pay for the expensive models (or the GPUs that serve them) for a load inflated upstream by the router.

> **Example: the Red Hat experiment (June 2026)**
>
> An experiment published by Red Hat on the vLLM Semantic Router quantifies the phenomenon. On a four-tier classification, a pre-trained embedding model (all-MiniLM-L6-v2, ~23M parameters) reached about 80% accuracy — a 20% misrouting rate, one request in five sent to the wrong model. After fine-tuning on domain data, accuracy rose to ~98.5% (misrouting ~1.5%), with the gains concentrated exactly on the ambiguous boundaries between tiers.
>
> *Note: the tiers in the experiment were well separated, almost caricatures. In real domains (legal, manufacturing) the boundaries are much blurrier, and similar numbers are not guaranteed. What is replicable is the method, not the figure.*

## 4. The principle of domain-targeted fine-tuning

The two previous sections converge on a single statement, worth isolating because it is the core of the whole argument: **the router must be fine-tuned on the tier boundaries of the real domain it will work in, not on generic data.** Two consequences follow, easy to confuse but to be kept apart.

### 4.1 Tier boundaries, not similarity

Measuring how much two texts resemble each other is a different problem from deciding whether they need the same level of model. A pre-trained embedding does the first; you need the second. The difference shows on the **boundaries**: the point where SIMPLE ends and MEDIUM begins, or where COMPLEX slides into REASONING. Fine-tuning exists precisely to teach the model *where those lines fall in your domain* — not in a generic domain, in yours.

> **Example (legal domain).** "Summarize this clause" and "assess whether this clause can be challenged" look very similar: same vocabulary, similar length, identical structure. But they sit on two different tiers — the first is extraction, the second is legal reasoning. A generic embedding places them close together and sends them to the same model; an embedding fine-tuned on *your* domain anchors learns that a boundary runs right there. This is why the gains in the Red Hat experiment concentrated on the ambiguous boundaries: that is where the generic model is blind and the fine-tuned one sees.

"Not on generic data" means the pre-training dataset is not enough: you need examples representative of the real traffic the router will handle. Not raw traffic in volume, but the **labeled anchors** that encode where the boundaries run in that specific domain — the intellectual work described in the next section.

### 4.2 The decision layer decides upstream; the architecture does not

The second misconception is blaming routing errors on the architecture. The router can be **stateless, replicated, scaled and always available** — and still get the classification wrong. The tier → backend map (section 2.2) executes faithfully: given a wrong tier, it faithfully routes to the wrong model. The error is born upstream, in the classifying component, not in the infrastructure engineering around it.

The operational implication is clear: **invest in the decision layer fine-tuned on the domain, not in hardening the router's infrastructure.** You can have the perfect architecture and still lose one request in five. It is the same logic by which, in section 2, all the risk sat in the decision layer and not in the deterministic table.

## 5. Fine-tuning — what it does, what it costs, what it needs

The solution is not to change the architecture but to **fine-tune the embedding model** so it learns the boundaries between the tiers of the specific domain instead of generic similarity. In practice, you train it to **tell one tier from another** rather than to group similar sentences. The model stays the same (compatible with the existing deployment, a drop-in replacement); only what it has learned to separate changes.

### 5.1 The typical pipeline

1. Start from a small set of hand-written seed anchors, a few dozen per tier, representing the starting truth of the domain.
2. Expand them with synthetic generation (paraphrases, domain transfers, edge cases, hard negatives), filtering for quality and label consistency.
3. Train the embedding model with contrastive learning (e.g. a loss that forces every batch to contain all tiers, so it learns the boundaries).
4. Evaluate on a held-out test set: accuracy, per-tier performance, calibration.
5. Release the model as a direct replacement in the router, changing a single configuration field.

### 5.2 Where the cost really is

It helps to separate two phases with opposite requirements, because this is where the most common misconception hides.

| Phase | Real requirement |
|---|---|
| **Training the embedding** | Light. It runs on CPU in a few hours, no GPU needed. This is not the bottleneck. |
| **Data generation (SDG)** | Expensive. It needs one or more capable LLMs to produce and validate the synthetic examples. This is the phase that consumes resources; in a lab you replace them with smaller local models, accepting lower data quality. |
| **Defining the anchors** | Intellectual work, not compute. It is the highest-value step: translating domain knowledge into labeled examples per tier. It cannot be bought with hardware. |

**Practical implication.** Reusing an existing fine-tuning pipeline (the know-how) and changing only the seed anchors to those of your own domain is the most efficient path. The engine belongs to whoever published the pipeline; the fuel — the domain examples — is what makes the result yours and defensible.

## 6. Routing accuracy as an SLO, not uptime

Routing accuracy is easy to mistake for an infrastructure reliability problem. It is not. A router can be **stateless, replicated, scaled and always available** — and still make systematically wrong decisions. Uptime does not tell the whole story: the system is available, but not reliable in the sense that matters.

For routing, **correctness is an SLO**. Seen this way, the problem becomes clear: the architecture is solid, but the decision layer is not accurate enough to carry it. And since misrouting tends to over-route toward the expensive models, the quality of the decision layer goes straight into **sizing**: you are sizing capacity for a load that the router itself inflates. In the Dielabs methodology this ties routing to **Cr_open** (SLO-anchored capacity, see [One Capacity Is Not Enough](/frameworks/one-capacity-is-not-enough.html)): the correctness of upstream classification is a sizing variable, not an application detail.

There is also an effect on guardrails. At high misrouting rates the guardrails are overloaded: they spend their time catching traffic that should have been handled correctly from the start. As accuracy rises, guardrails go back to their proper role — covering edge cases and enforcing policy boundaries — instead of compensating for the router's errors.

## 7. Implications for on-premises and sovereign deployments

In regulated environments routing is part of the **security model**, not only of efficiency. If the tier → backend map keeps some tiers on local models and sends others to the cloud, then a wrong classification does not just degrade performance: it can **violate a compliance policy, a data residency requirement or an internal control**, sending to the cloud a request that should have stayed inside the infrastructure.

This leads to an observation that matters for anyone integrating sovereign AI: routing is where data sovereignty becomes concrete and configurable. Defining with the customer where the line runs between what stays on-premises and what may leave — and encoding it in the anchors and the map — is the operational translation of data residency. Domain anchors are not a technical detail: they are that conversation turned into code.

---

> **In short**
>
> Semantic routing is the right architectural pattern for serving several models economically and compliantly. Its limit is not the architecture but the accuracy of the decision layer, which out of the box confuses similarity with complexity. The fix has a precise direction: fine-tune the embedding on the tier boundaries of the real domain it will work in — on what it will have to route, not on generic data. The valuable work is not compute but defining those boundaries. And since those boundaries also decide what stays in and what goes out, routing is at once a cost, an SLO and a sovereignty problem.

---

*All content is original Dielabs work by Diego Bardella.*
