---
layout: default
title: "Testing Is the Product"
---

# Testing Is the Product

## A manifesto for on-premise inference due diligence

**Lab:** Dielabs **Focus:** on-premise inference, sovereign AI, presales due diligence **Date:** July 2026, updated October 2026

> **Thesis.** The industry has learned to assemble sovereign inference. It hasn't learned to certify it. The gap between *"it works"* and *"I can put my signature on it"* is where the value gets created — and it's empty.

---

## 1. The case

In July 2026, a CISO posts a photo on LinkedIn of four NVIDIA DGX Spark units stacked in his office, connected over a 100-gigabit network with RDMA. Running on top is GLM-5.2 — a 744-billion-parameter model — compressed to 4 bits and distributed across the four nodes, with a 200,000-token context window. The message is blunt: I have a frontier model in my office and my data never leaves the building.

The post gathers nearly four hundred comments in four days. CISOs, enterprise architects, an NVIDIA Senior Director, quant trading engineers, a legal engineer building institutional memory for law firms, healthcare consultants, defense people — all weigh in.

**The post isn't the interesting document. The thread is.** Those four hundred replies contain, for free and unsolicited, the complete list of objections, questions, and blind spots of anyone evaluating on-premise inference. It should be read as discovery material, not as a technical debate.

This piece pulls the first eleven principles (I–XI) out of that material, checks them against public third-party data, and turns them into an operational protocol. The later principles (XII–XIV) come from our own experimental investigation, from a second industry field report and from an analysis of a serving engine's code: they don't come from the DGX Spark thread, but from the same verification criterion — separating what can be sourced from what is a field report before certifying it in an acceptance test. The document stays anchored to a founding case, but is no longer limited to it: it grows every time a verified source adds a principle that stands on its own.

---

## 2. I — Platform metrics don't measure the workload

The numbers the author publishes are three: 97.97 Gb/s of RDMA bandwidth between nodes, a cold start of 4 minutes 49 seconds, no reboots since launch.

All three are infrastructure numbers. They say *"I set it up correctly, it's standing, it boots fast."* That's deployability and availability. They say nothing about whether the system is useful to anyone.

The numbers that decide whether the system is worth anything are all missing:

- how many tokens per second it generates while writing
- how long it takes to produce the first token with a full context
- how many concurrent requests it can hold before latency breaks its limits
- how much capacity the model lost by being compressed to 4 bits

**Principle.** A cluster gets accepted on the metrics of the load it will serve, not on the metrics of its own setup. Network bandwidth measured with a synthetic benchmark is a prerequisite, not a result.

---

## 3. II — Reading and writing are two different machines

The only performance figure in the post is disguised as something else: *it reads a 160,000-token document in four minutes*. It sounds like a response speed. It isn't.

A model, when it works, does two things with opposite computational profiles.

**Reading the prompt (prefill)** is massively parallel: thousands of tokens are processed together, the bottleneck is compute. It scales well, and it looks good in demos.

**Generating the response (decode)** is sequential by construction: one token at a time, and for every token the model's active weights have to be re-read from memory. The bottleneck is memory bandwidth, not compute. It doesn't parallelize, and it's what determines the user's actual experience.

160,000 tokens in four minutes works out to roughly 667 tokens per second, and it's **prefill**. It's the easy number. The hard number was never published — not even after a dozen people asked for it explicitly in the comments, separately, repeatedly, and without ever getting an answer.

In the replies, though, the author gives it away: the machine works on *"a complex problem taking 90–100 passes at a time, in the background."* Translation: this isn't an interactive system, it's a batch system. A legitimate and even sensible choice — but a different one from what the post implies.

**Principle.** Any throughput figure that doesn't separate prefill from decode is unusable for sizing. Whoever publishes only the prefill number is — knowingly or not — showing the machine's best side.

---

## 4. III — The ceiling gets calculated before you buy

This is the part worth taking away as a method, because it makes the unanswered question moot.

The public facts:

- Each DGX Spark has 128 GB of unified LPDDR5X memory at **273 GB/s** of bandwidth.
- GLM-5.2 is a Mixture-of-Experts model: **~744 billion total parameters, ~40 billion active per token**. It only lights up a fraction of itself on each step.
- The model runs quantized to 4 bits (NVFP4), the standard on DGX Spark.
- With the model split tensor-parallel across four nodes, each node has to read **roughly 5 GB of memory for every generated token** (10 billion active parameters per node × 0.5 bytes).

From this: 273 ÷ 5 ≈ **56 tokens/second as the absolute theoretical ceiling**, unreachable in practice since it ignores every overhead.

And in practice? On the same exact hardware, others have published measured numbers:

| Configuration | Decode (single-stream) |
|---|---|
| 4× Spark, aggressive quantization IQ4_XS | **6.28 tok/s** |
| 2× Spark, 2-bit experts + pruning 256→208 | **~15–21.5 tok/s** |
| 4× Spark, 4-way KV sharding, no pruning | **~24.7 tok/s** |
| 4× Spark, compact NVFP4 KV + MTP-5 | **~30 tok/s at 64K, 42 peak** (prefill ~819 tok/s) |

The system in the post almost certainly generates somewhere between 20 and 30 tokens per second. About a fifth of what users are used to from cloud models. For asynchronous work it's fine; for an interactive chat it's painful.

**Principle.** With three parameters — per-node memory bandwidth, active parameters per token, parallelism scheme — you can estimate the performance ceiling *before* signing a purchase order. It's a two-minute calculation that prevents twenty-thousand-euro mistakes. Do it at qualification time, not at acceptance time.

---

**Technical note (August 2026) on the compression starting point.** Not every model starts from the same precision level. Some (e.g. Kimi K3) are trained from the outset in an already-reduced version of their weights, instead of full precision like most open models: the reduction that normally happens after release has, in these cases, already happened upstream. Compressing an already-reduced model further, down to 1 bit, isn't the same operation — nor does it carry the same risk — as compressing a model that starts from full precision. The "1-bit" label alone isn't enough to compare two models: where they started from matters too.

## 5. IV — The dominant constraint isn't the one being discussed

A large share of the technical comments focus on the network: *why 100 gigabit and not 200, given the ConnectX-7 cards support it?* At least five people ask this, including an NVIDIA Senior Director and several network architects.

It's the wrong discussion, twice over.

**First, on the facts.** On this platform the link does negotiate 200 GbE, but in real tests multi-stream TCP traffic tops out around 106 Gbit/s, and in several configurations the collective communication library can't even use RDMA and falls back to ordinary sockets. It's a limit of the box's internal bus, not a design flaw. The measured 97.97 Gb/s is already close to the practical maximum.

**Second, and far more important: the network isn't the bottleneck.** The comparison speaks for itself — roughly 12 GB/s of fabric versus 273 GB/s of local memory per node, a 22-to-1 ratio. But the network moves relatively little data per token, while memory gets read *entirely* for every single token. The dominant constraint is memory, by an order of magnitude.

Out of nearly four hundred comments, written largely by qualified professionals, two or three people identify the right constraint. Everyone else optimizes the wrong part of the system.

**Principle.** This is exactly what happens in a meeting room with a client. The architect's job isn't to supply every number — it's to say **which number matters**, and why the rest is noise. It's the one part of the job that can't be delegated to a spec sheet.

---

## 6. V — Not capex versus opex, but load profile

The loudest thread of the discussion is economic: *"twenty thousand dollars once versus two hundred a month."* It's a badly framed comparison on both sides.

The cloud's defenders are right about most workloads and wrong about some. The purchase's defenders confuse ownership with cost-effectiveness. And the numbers circulating are wrong: some say two hundred thousand dollars, when the real math is four machines at roughly $4,700 each — starting from $3,999 and rising 18% due to the global LPDDR5X memory shortage — plus switches and cabling, so **around twenty thousand total**.

The real distinguishing factor isn't the shape of the spend. It's the load profile:

**On-premise holds up when the context is long and repetitive, the work is asynchronous or batch, the volume is predictable, the cost needs to be fixed, and the data can't leave for regulatory or contractual reasons.**

**Cloud wins almost every time when** the load is interactive, bursty and unpredictable, latency-sensitive, and requires the best quality available on the market that quarter.

Another commenter frames it well as an insurance argument: on-premise is also a hedge against price risk — the possibility that cloud inference costs spike once an organization has built processes on top of it. It's a risk-management argument, not a TCO argument — and should be evaluated with risk-management tools, accordingly.

**Principle. The question "should I buy or rent?" has no answer without the load profile.** Anyone who answers before measuring it is selling something.

---

## 7. VI — Cost gets measured in work delivered

The smartest line in the whole thread comes almost in passing from the author, replying to a critic: in the first twenty-four hours the system cleared two to three weeks of technical backlog. Elsewhere he adds he worked ten straight hours without stopping and without running out of memory.

This is the only value metric in the entire discussion, and it's expressed in units a CFO understands. Not tokens per second: **weeks of backlog cleared**.

**Principle.** Presales for on-premise inference is won by changing the unit of measure. Tokens/second is the engineer's metric. Work delivered per unit of time, at fixed cost, with data that never leaves, is the buyer's metric. Converting between the two is your job, and it should be done explicitly when building the business case.

---

**Note (August 2026).** The same objection shows up, independently, in a separate thread months later: a commenter notes that **publishing tokens/second without energy cost and total hardware cost can make a very expensive system look efficient.** Same principle, different vendor, different model — cost should always be reported per unit of work delivered, never in isolation.

## 8. VII — Localizing the data isn't controlling its use

The most mature thread of the discussion is raised by two security architects, and it's the sharpest objection in the whole exchange: **data locality is not access control.**

*"Nothing leaves the building"* solves residency. It doesn't solve:

- **Who is authorized** to query the system, and with what data in the context.
- **What gets logged** about what enters and exits the context window.
- **What happens if the supply chain is compromised.** If the model weights or a Python dependency installed during setup are poisoned, exfiltration happens through a port that's already allowed. A perimeter firewall in *allow-then-inspect* mode doesn't see it. What's needed is an explicit, per-node default-deny egress posture.

And there's the flip side of the sales pitch: a machine that reads 160,000 tokens in four minutes is also a machine that can be pointed at any document on the network in four minutes. The ingestion speed that makes the system attractive is the same thing that turns it into a risk multiplier if the authorization perimeter isn't designed first.

**Principle. Data sovereignty and access governance are two separate projects.** Selling the first while leaving the second implicit is the single most common mistake in this category of solutions — and it's a real, sellable product gap.

---

**Specific case: the shared KV cache.** The same principle applies inside the system, not just at its perimeter. The `fs`/`obj` KV cache offload tiers in vLLM can share blocks across different sessions or tenants to reuse prefixes: without robust content-addressed naming, a predictably fixed `PYTHONHASHSEED` or weak isolation between tenants opens a side channel — one tenant can infer or read fragments of another tenant's context from the shared cache.

---

## 9. VIII — The missing product is day two

The most valuable comment in the whole thread comes from someone running eight Sparks on the same stack. The account is disarming: CUDA driver issues, unstable integration between the inference engine and the model, malformed and mis-nested JSON in the responses, low-level crashes. The conclusion — *if you manage to appease the patch gods and get it working, then don't touch it again* — is the exact opposite of what a company can accept.

Public repositories confirm he isn't exaggerating. Getting this stack running requires:

- a patch to the attention kernel, because the chip's shared-memory limit (101,376 bytes) is smaller than what the standard kernel requires;
- orchestrator timeouts raised to as much as an hour, because MoE initialization can take twenty minutes or more;
- and when a node runs out of memory, it has to be power-cycled **physically**, because it doesn't come back on its own after a crash.

None of these is exotic. They're the ordinary problems of day two, and that's precisely the band the market has no offer for: everyone sells day one.

**Principle.** In this category, maintainability and reproducibility matter more than peak performance. A system that does 40 tokens/second and needs to be hand-nursed back to life every two weeks is worth less than one that does 25 and restarts itself. Tell the client this **before** they discover it themselves — it's the only moment it sounds like expertise instead of an excuse.

---

## 10. IX — The right model is the smallest one that's enough

Two commenters arrive at the same point from opposite directions, and it's probably the most useful architectural lesson in the thread: **the thing worth copying is the locality, not the 744 billion parameters.**

For most real enterprise tasks — structured extraction, classification, routing, document search, tool calls — a much smaller model, backed by a good retrieval system and output validation checks, gets the job done. And it gets it done faster, on a single machine, with a much smaller failure domain.

The guarantee on the data comes from the architecture (inference running on hardware you control), not from model size. These are two independent decisions the post merges into one, and nearly every commenter follows the author into the same mistake.

Worth keeping as a side note: one commenter observes that models are moving faster than hardware. A year ago the local reference point was around 35 billion parameters; today it's hundreds. If this trend holds, the stable strategy isn't chasing the largest model that fits in memory — it's **a modest model plus a sophisticated retrieval layer**, where the last mile of domain knowledge comes from retrieval rather than from the weights.

**Principle.** The first sizing question isn't "how big can I fit?" It's "what's the smallest model that clears the task's acceptance test?" Two questions, two quotes that differ by a factor of five.

---

**Note (August 2026).** The same principle resurfaces in an independent thread, about a different model (Kimi K3, Moonshot AI, Unsloth's 1-bit quantization): a commenter notes that treating models as interchangeable competitors is the mistake — they're tools with different edges, and the useful question was never "which is best" but "best at what, and at what cost." Two threads, two vendors, two models, same objection: not an isolated opinion, a pattern.

## 11. X — The four questions nobody asked

Across four hundred comments, these four points never come up. Not by coincidence, they're the four a regulated company raises within the first hour of due diligence.

**1. Quality after compression.** Four hundred comments about speed, zero about accuracy. Yet to fit the model into available memory, some recipes circulating don't just compress the weights to 2 bits — they **prune the experts from 256 down to 208 per layer**. That's not losing numerical precision, that's removing model capacity. A system that works but answers worse than the model you think you bought is the silent risk of this entire category — and nobody in the whole thread proposed **a regression evaluation** against the reference checkpoint.

**2. The lifecycle.** There's already talk of a successor beyond a trillion parameters, with projections for summer 2026. Hardware doesn't grow alongside models. This cluster's depreciation isn't measured in time, it's **tied to the next release**: in six months the same hardware might no longer fit the reference model of its category. A three-year depreciation schedule on an asset whose relevance is measured in quarters is a financial-model error, not an engineering one.

**3. The failure domain.** Four nodes working as one machine means if one goes down, the service goes down. Zero redundancy, overall availability equal to the product of the four. Nobody asked about recovery time, continuity target, or what happens during a firmware update. For a company, that's the first question, not the last.

**4. Weight provenance.** The model is openly licensed and runs in-house, but the vendor is on the U.S. Entity List. In a thread devoted entirely to technological sovereignty, nobody mentions it. For a regulated client — banking, healthcare, defense, public sector — it's a mandatory due-diligence point, and the fact that the weights are downloadable doesn't resolve it: they're still an opaque artifact of foreign provenance inside the perimeter.

**Principle.** The questions that don't show up in a public technical discussion are the ones that will show up in the purchasing committee's room. Whoever brings them first controls the conversation.

---

## 12. XI — The buyer isn't the loudest voice

The last reading of the thread isn't technical. It's about segmentation.

Look at **who** comments: security leads, bank CISOs, a legal engineer building institutional memory for law firms, healthcare consultants, enterprise architects, defense people, trading quants. And look at what they ask for: concurrency, access control, stability, confidentiality.

Then look at who's shouting that two hundred dollars a month gets you more. They're technically right. They're not the buyer.

**Whoever buys sovereign inference isn't doing it to save money.** They're doing it because they have a compliance, confidentiality, IP, or continuity constraint the cloud doesn't solve for them. The savings, when they show up, are a welcome side effect, not the purchasing motivation.

**Principle.** The commercial message shouldn't be built on TCO. It should be built on constraint → architecture → proof. TCO comes after, to justify the number to whoever signs, not to convince whoever decides.

---

## 13. XII — The declared concurrency number isn't the one the system holds

*From here on, the principles no longer come from the DGX Spark thread. The source changes — an industry talk on the serving stack, checked against our own experimental investigation — but the criterion stays the same: separate what can be sourced from what is a field report before treating it as a principle.*

An industry talk on inference serving architectures describes a field-report case: a node with eight H100 GPUs, configured for a maximum context of 4,000 tokens, held 370 concurrent users. Raising that limit to 128,000 tokens — a seemingly harmless product request, *"we want to support longer contexts"* — made the observed capacity collapse to 12 users. No error, no crash, no log. The system stayed green on every infrastructure metric while losing 97% of its serving capacity.

The mechanism isn't a failure: it's planning arithmetic. The memory pool dedicated to the conversation cache is a fixed quantity. The share each session can take at its declared limit grows with the configured maximum context. If that maximum rises by a factor of 32, the number of sessions the system can guarantee to hold at the same time drops by the same order of magnitude — not because physical memory disappeared, but because the system has to reserve, for every admitted session, the space for the case where that session actually reaches its declared limit.

This is principle IX applied not to the model's weights but to per-user context: *"supports 128k context"* and *"holds 370 concurrent users"* are two claims sold separately that have to be multiplied, not added, at sizing time. A well-designed inference engine can be more optimistic than the planning number it declares at startup — allocating memory based on how much a session is actually using, not on its theoretical maximum — but that margin of optimism has to be verified under real load, not assumed from the spec sheet.

**Principle.** The concurrency number a vendor or a benchmark declares is almost always a worst-case planning value, calculated assuming every active session fills its maximum context. The number that matters for acceptance is the concurrency observed under the client's real traffic, with the real distribution of context lengths — not the theoretical one, not the worst case, not the best case.

---

## 14. XIII — The load isn't just one, and the layer that refuses comes before the layer that computes

The same industry talk cited in principle XII offers a taxonomy that's useful even to people who will never run that benchmark: four load profiles, with requirements that contradict each other on the same stack.

- **High-concurrency chat** — short inputs, medium outputs, thousands of users. The bottleneck is memory bandwidth while writing. You optimize by increasing the batch size.
- **Long context, document search** — large inputs, short outputs. The tightest constraint is time to first token. The typical flaw is prefix reuse close to zero, because every retrieval request pulls different chunks.
- **Multi-turn agents** — system prompt, schema and history repeated almost identically at every turn, tens of thousands of tokens per turn. The prefix cache hit rate is the whole economics of the workload: not a technical detail, the number that decides whether the service is sustainable.
- **Deferred processing** — evaluations, embeddings, data pipelines. No latency limit to meet; the only metric is how many tokens are produced per euro spent.

No production system serves a single profile: the typical pattern is chat and agents during the day, deferred processing at night, on the same cluster. An infrastructure sized for one load profile is oversized for the other three and undersized for the one it didn't foresee.

**The layer that decides whether a request starts is separate from the one that computes it, and it has a property no other layer has: it can say no before the compute cost is already spent.** Once a request is past that gate, the GPU is already committed — whether the answer arrives in one second or in nine no longer changes what was spent to compute it. A system without that gate doesn't crash under load: it queues silently, and the user waits nine seconds instead of getting an immediate refusal in a millisecond. It's a failure that no log records as a failure.

The same distinction applies to routing the request to the right machine. Sending a request to the least busy server, as a traditional load balancer would, is almost always a mistake in this domain: the free server is free because it doesn't have the session history already loaded. The time saved in the queue is paid for with a full recomputation of the context from scratch — a trade-off that is almost always unfavorable once the context goes beyond a few thousand tokens. Correct routing sends the request to the server that already holds the system prompt, the tool definitions and the history in cache: not a marginal optimization, but the difference between a tenth of a second and a full second of repeated compute.

**Principle.** The layer that admits or refuses a request, and the one that decides which machine to route it to, matter as much as the compute engine downstream — and they have to be sized on the specific load profile, not on a generic profile assumed to be the same for chat, search, agents and batch.

### The failure that leaves no trace

The pattern generalizes. No admission control, blind routing, the context reading and writing phases put on the same device, missed cache reuse, fixed-size instead of adaptive compute batches, a slow network interconnect: none of these produces an error. All of them silently degrade throughput or latency while every health indicator of the system stays green. It's the same family of behavior as the two sources of non-determinism in inference: neither ever throws an exception, both silently change what the system returns or how much load it holds. *"The system is green"* doesn't imply *"the system does what the client expects"*, and that equivalence has to be verified explicitly in the acceptance test, not assumed.

A concrete field-report case, useful as an illustration even without verifying its numbers: four requests to a ticketing system all come back successful at the transport level. Three pass validation but take an order of magnitude longer than the agreed limit; one fails validation because of a missing character in the data structure — there was no upstream check — and is still forwarded to the compute engine, which produces an answer nobody asked for. None of the four generates a system error. All four are a failure for the end user.

Two quantitative corollaries, reported as field-report orders of magnitude and not as certified thresholds. Latency under load doesn't degrade linearly but much more steeply close to saturation, which makes unused capacity headroom a structural cost of serving a load with latency limits, not waste to eliminate. And the time between a traffic spike arriving and a fully operational node being available adds up to several minutes — provisioning, image download, weight loading, engine warm-up — a time that no standard autoscaling manages to bring below thirty seconds, whatever the optimization applied.

---

## 15. XIV — The vendor's number and the acceptance number come from two different tools

*Source: analysis of the vLLM engine code (technical post reviewed by core contributors, August 2025). Not a field report: it's what the engine's benchmark tools do by construction.*

The most widely used inference engine, vLLM, has three benchmark commands. `throughput` sends all the requests at once, as if traffic were infinite, and reports tokens per second: it's the number you see in slides. `latency` uses a small batch and a short input, and measures time only: it's the demo number. `serve` simulates users arriving at random intervals, with a concurrency cap, and measures time to first token, time between tokens and how many requests meet the declared target: it's the only one that resembles the client. On the same hardware and the same model, the three commands give three different numbers — and none of the three is wrong.

There's one more detail. In benchmarks the engine is told to ignore the end-of-response signal and to generate a fixed number of tokens anyway. That's correct for comparing two configurations, because it makes the load repeatable; but it isn't real traffic, where answers end when they end. A number measured this way is clean and optimistic at the same time.

The mechanism is the same as in XII: a true value, obtained in conditions no user experiences, presented without the conditions. The difference is that here you don't need a talk to find out. You only have to read which command was run.

**Principle.** A performance number without its measurement mode isn't a number. Whoever presents a throughput has to state which tool produced it, with which arrival distribution and which end-of-response rule. The acceptance test uses only the mode that simulates users — random arrivals, capped concurrency, declared SLO — with the client's traffic profile. The infinite-traffic number stays useful for one thing only: knowing how much margin there is between where you operate and where the hardware saturates. The method to derive both is in [One Capacity Is Not Enough](/frameworks/one-capacity-is-not-enough.html).

---

## 16. The eight-question checklist

Counting the questions commenters spontaneously ask the author produces a recurring list. It isn't a questionnaire designed at a desk: it's **free discovery, validated by a sample of four hundred professionals in the middle of an actual purchase evaluation**. Use it as a qualification grid.

1. What is single-stream decode, reported separately from prefill?
2. What is time-to-first-token at the stated context values?
3. How many concurrent users does it hold, and at what degradation?
4. What is idle and under-load power draw, and what's the thermal management?
5. How much management time does it need per week, and how stable is it between updates?
6. How much quality is lost with the chosen quantization, measured how?
7. What governs what enters the context window, and what gets tracked?
8. What is the real total cost, including the cost of the time of whoever maintains it?

Whoever answers these eight makes a sale. Whoever answers the first and the fourth makes a LinkedIn post.

---

## 17. The acceptance protocol

The operational consequence of all of the above is that an artifact is missing, and the missing artifact isn't a benchmark: **it's an acceptance test report.**

Given any cluster — Spark, appliance, traditional GPU server — an acceptance test produces, in half a day:

**Performance.** Context/throughput curve with prefill and decode separated and reported distinctly. Capacity measured two ways: hardware-anchored in closed loop (*Cr_closed*) to know what the hardware can do, and SLO-anchored in open loop (*Cr_open*) to know how many users it actually holds at acceptable latency. The theoretical ceiling calculated from memory bandwidth, reported alongside the measured one, as a sanity check. For every number, the benchmark mode, the arrival distribution and the end-of-response rule are stated (principle XIV).

**Quality.** Regression comparison between the quantized checkpoint in production and the uncompressed reference model, on a task set representative of the client's use. This is the piece everyone is missing, and it's the piece that tells the client **how much intelligence they actually bought**.

**Resilience.** Behavior on node failure, measured recovery time, behavior after memory exhaustion, zero-downtime update procedure (or an explicit statement that none exists).

**Operations.** Power draw and thermal profile under sustained load, estimated person-hours for routine maintenance, an explicit list of the patches and deviations from standard needed to make the stack work — because every deviation is debt somebody will end up paying.

**Governance.** Egress posture, authorization perimeter on the context, tracking, provenance of weights and dependencies.

All of it mapped onto the *-ilities* (AMPRS), so the report speaks the language of design, not the language of a benchmark.

---

## 18. The commercial consequence

Don't sell the hardware. Clients buy that on their own, and the thread proves they can even assemble it themselves: four nodes over RoCE, RDMA at 98% of line rate, first bring-up successful. Assembly competence is widespread and growing fast.

What nobody in that thread knows how to do is **certify** what they built and **maintain** it without every update becoming an event. The comment about the eight unstable Sparks and the total absence of questions about post-quantization quality are the same gap seen from two sides.

The positioning, then:

- **At qualification**, the ceiling calculation before purchase, and the eight questions. Costs nothing, saves the client a mistake worth tens of thousands of euros, and establishes who leads the technical conversation.
- **At delivery**, the acceptance test as a signable artifact.
- **In operations**, day two: reproducibility, updates, recovery.

The thread is the proof, written by four hundred qualified people, that the industry knows how to assemble these systems and knows how to neither test nor keep them running.

That gap is the product.

---

## Sources and figures

All quantitative data cited comes from public third-party sources, not from the analyzed post. Reported performance varies significantly with quantization scheme, expert pruning, KV cache format, and use of speculative decoding: figures are only comparable at a stated configuration. GLM-5.2's parameter count appears as either 744B or 753B depending on the source; the active-per-token figure is consistently around 40B.

- DGX Spark specs and memory bandwidth: [storagereview.com — NVIDIA DGX Spark review](https://www.storagereview.com/review/nvidia-dgx-spark-review-the-ai-appliance-bringing-datacenter-capabilities-to-desktops)
- Updated pricing and positioning: [ifactoryapp.com — DGX Spark enterprise review](https://ifactoryapp.com/sap-integration/on-prem-ai/nvidia-dgx-spark-review-enterprise)
- Practical fabric limit on GB10 (NVIDIA Developer thread): [forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/connectx-7-200gbe-via-mikrotik-crs812-qsfp-dd-400g-2xqsfp56-200g-breakout/357162)
- GLM-5.2 architecture and parameters: [recipes.vllm.ai/zai-org/GLM-5.2](https://recipes.vllm.ai/zai-org/GLM-5.2)
- Measured numbers, 4× Spark, memory-bandwidth ceiling, compact NVFP4 KV: [github.com/0xdfi/GLM-5.2-1M-4x-DGX-Spark](https://github.com/0xdfi/GLM-5.2-1M-4x-DGX-Spark)
- No-pruning configuration, 4-way KV sharding: [github.com/bird/GLM-spark](https://github.com/bird/GLM-spark)
- 2-bit quantization with 256→208 expert pruning: [github.com/tonyd2wild/GLM5.2-2bit-2-DGX-Spark--21.5tok-s](https://github.com/tonyd2wild/GLM5.2-2bit-2-DGX-Spark--21.5tok-s)
- Performance with aggressive quantization: [forums.developer.nvidia.com — GLM 5.2 IQ4_XS on 4x GB10](https://forums.developer.nvidia.com/t/glm-5-2-iq4-xs-on-4x-gb10-6-28-tok-s-dsa-active-full-recipe/373933)
- Model provenance and regulatory position: [theairankings.com/zhipu/glm-5](https://theairankings.com/zhipu/glm-5/)

**Added sources (August 2026), Kimi K3 case.** Data on Kimi K3 (parameters, quantized size, stated accuracy) comes from vendor documentation and repository (Unsloth), not from an independent third-party source — a lower evidence tier than the other references in this note. Treat as a qualified field report, not as citable data for acceptance testing purposes.

- Kimi K3 documentation (Unsloth): [unsloth.ai/docs/models/kimi-k3](https://unsloth.ai/docs/models/kimi-k3)
- GGUF repository (Hugging Face): [huggingface.co/unsloth/Kimi-K3-GGUF](https://huggingface.co/unsloth/Kimi-K3-GGUF)

**Added source (September 2026), principle XII.** The 370→12 concurrent users figure comes from an industry talk, reported as the speaker's direct experience on their own experiment (70B model, 8×H100 node). The talk states no inference engine version, scheduler configuration or measurement methodology: field-report evidence tier, not citable for acceptance purposes. The number is used here to illustrate the size of the effect, not as a reference figure for real sizing — sizing always has to be redone on the client's hardware, engine and traffic.

Our own experimental investigation — [When Offloading Doesn't Offload](/papers/when-offloading-doesnt-offload.html), section 5.4, on vLLM 0.15.1 — isolated the underlying mechanism with a stated methodology: the concurrency number the inference engine reports at startup is a worst-case planning value, and the scheduler can admit substantially more requests at runtime through on-demand allocation. It's possible, though not verified here, that the talk's number describes exactly that planning value and not the concurrency observable under real load — in which case the apparent collapse would be partly an artifact of the formula, not only a physical memory limit.

**Added source (September 2026), principle XIII.** The taxonomy of the four load profiles, the gateway/router distinction, the cache-aware routing pattern and the list of silent failures come from the same industry talk cited in principle XII. The architectural principle (the admission layer can refuse before the compute cost is spent; blind routing is counterproductive when the session cache matters) is structural and verifiable regardless of the source. The corollary numbers — latency degradation by utilization, cold-start times, the specific case of a 98% cache hit rate with TTFT going from 786 ms to 16 ms — remain field reports that can't be reproduced without a stated configuration, reported to illustrate the size of the effect and not as acceptance thresholds.

**Added source (September 2026), principle XIV.** The description of vLLM's three benchmark commands (`throughput`, `latency`, `serve`), of the `ignore_eos` behavior and of the auto-tuning script comes from A. Gordić, *"Inside vLLM: Anatomy of a High-Throughput LLM Inference System"* (August 2025), an analysis of the source code at commit 42172ad, reviewed by core project contributors before publication. Evidence tier: citable, because it describes how the tools behave by construction, not a measured result; no performance number is taken from this source.
