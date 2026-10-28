---
layout: default
title: "Disaggregation OS — NVIDIA Dynamo"
---

# Disaggregation OS — NVIDIA Dynamo

**Distributed orchestration for LLM inference.** *A study manual — from model serving to inference systems engineering.*

> **As of June 2026** (Dynamo workshop, Dynamo 1.x). This is the technology companion of [Disaggregated Inference](disaggregated-inference.md): that page covers why and when to disaggregate, vendor-neutral; this one covers how it is done on NVIDIA Dynamo. Product names, versions and numbers here will age.

> **Audience:** infrastructure people with general LLM context, not inference engineering specialists. **Suite scope:** runtime + Router + Planner + KVBM + NIXL + Grove + AIConfigurator/AIPerf + integrations with vLLM / SGLang / TensorRT-LLM.

> **Maturity legend.** To read the manual's claims correctly, every "borderline" piece of content is tagged with one of these levels. It separates what is stable from what is emerging or interpretive:
>
> - **[Baseline]** — documented, stable behavior in current Dynamo versions (1.x). You can size on this.
> - **[Session/Roadmap]** — an emerging feature, in beta, or coming from technical sessions/GTC: a credible direction, but no guarantee of availability in every deployment.
> - **[Vendor claim]** — numbers or capabilities stated by a vendor (e.g. VAST): treat them as a specific benchmark, not as a generalizable sizing assumption.
> - **[Dielabs interpretation]** — my own architectural or strategic reading, not a fact from the documentation.
>
> Where no tag is present, the content is to be read as **[Baseline]**.

---

**Part I — Fundamentals**

This manual builds a solid mental model of NVIDIA Dynamo. The goal is not to memorize component names but to understand *why* they exist and *how* they cooperate. The sequence is didactic: first the problem, then the vocabulary, then the architectural choices, then the individual components placed in the plane they belong to, and finally sizing and operations. Every advanced concept rests on the previous ones.

## 1. The core idea

Dynamo exists because modern LLM inference is no longer simply "load a model on a GPU and serve requests". As model size, context length, concurrency and traffic variability grow, the problem stops being local and becomes systemic: routing, KV cache, data transfer, autoscaling, physical placement, fault tolerance and cost per token.

> **The right mental sentence.** vLLM, SGLang and TensorRT-LLM optimize **model execution**; Dynamo optimizes **the distributed system around** model execution.

This distinction is the key to everything else. Dynamo does not necessarily replace the inference engine: it turns it into a coordinated multi-GPU / multi-node system with disaggregated serving, intelligent routing, multi-tier KV caching and autoscaling. NVIDIA describes it as an open-source framework for distributed inference, Kubernetes-native for production but also usable locally or as separate components.

In the Dielabs stack model, Dynamo lives mostly between `L4a Request Orchestration`, `L4b GPU Workload Optimization` and part of the state / memory / data movement layer. The actual engines sit below, at the execution backend level. The full mapping onto the layers is in Part VI.

## 2. The problem Dynamo sets out to solve

In a traditional LLM deployment a single server handles the whole request cycle: it receives the prompt, tokenizes, runs prefill, generates tokens in decode and returns the answer. It is simple and adequate for many cases, but it forces the same group of GPUs to handle two phases with very different load profiles.

> **Prefill** — processes the input prompt and builds the KV cache. It is mostly **compute-bound**, the more so the longer the prompt.
>
> **Decode** — generates tokens one at a time, autoregressively. It is mostly **memory-bandwidth-bound**: it depends on memory, bandwidth, KV cache and concurrency.

Forcing two such different phases into the same hardware block inevitably leads to compromises: the configuration that is optimal for prefill is rarely optimal for decode. Hence the need to treat them separately.

Dynamo answers with three main techniques, which the documentation presents as **composable** — each helps on its own, but combined they reinforce each other because they reduce recomputation, raise cache reuse and let prefill and decode scale independently:

1. **Disaggregated serving** — physically separating the prefill and decode pools.
2. **KV cache-aware routing** — routing requests also by which context is already present on a worker.
3. **KV cache offloading** — extending cache capacity beyond HBM.

These three techniques are the thread of the manual: every component we meet serves at least one of them.

## 3. Minimal vocabulary

Before going on, it pays to pin down the terms. A frequent confusion — to be avoided — is to treat workload parameters as if they were SLAs. They are not.

### Workload parameters (the "shape" of the load)

| Term | Meaning | Example |
|---|---|---|
| **ISL** — Input Sequence Length | Input length in tokens. Part of the workload shape in benchmarks. | ISL 4000 → a ~4000-token prompt |
| **OSL** — Output Sequence Length | Length of the generated output in tokens. | OSL 500 → a ~500-token answer |

ISL and OSL **are not SLAs**: they describe the load, not a performance target.

### Performance targets (SLA / SLO)

| Metric | What it measures | Sensitive to |
|---|---|---|
| **TTFT** — Time To First Token | Time before the user receives the first token. | prefill, queue, routing, cache hit |
| **TPOT / ITL** — Time Per Output Token / Inter-Token Latency | The pace at which the following tokens are generated. | decode, concurrency, memory, bandwidth |
| **tokens/s/user** | Speed perceived by the single user. | user experience |
| **tokens/s/GPU** | Infrastructure efficiency. | system economics |

### The state that makes it all interesting

> **KV cache** — the intermediate state produced by the model during attention. It is valuable because it avoids recomputing context already processed, but it takes a lot of memory, especially with long contexts and many concurrent requests. Much of Dynamo's engineering revolves around a single question: how to reuse, move and keep the KV cache efficiently.

## 4. Aggregated vs disaggregated serving

This is the first real architectural choice, and it must be understood thoroughly because everything else follows from it.

### Aggregated

Each instance serves the whole cycle: prefill plus decode. It is simpler to deploy and debug, and it is often the right choice for small workloads, single-node environments, quick POCs, or cases where traffic does not justify the complexity.

### Disaggregated

Prefill and decode become separate pools. The typical flow is:

1. The router sends the request to a prefill worker.
2. The prefill worker computes the prompt's KV cache.
3. The KV cache is transferred or made available to the decode worker.
4. The decode worker generates the answer, streaming it.

The benefit is sizing the two pools differently. Long prompts and short answers push toward more *prefill* capacity; highly concurrent chats with long generation push toward more *decode* capacity. The two pools can also use different parallelism configurations, if backend and KV layout allow it — for example prefill with `TP=2` and decode with `TP=4` or `TP=8`.

> ⚠️ **Careful: disaggregated does not mean "always better".** It introduces real complexity: KV transfer, worker compatibility, scheduling, topological placement, coordinated autoscaling and more advanced observability. It pays off when traffic and scale justify the separation — not as a default.

### Practical thresholds — when to disaggregate

NVIDIA's operational guidance yields two concrete decision axes, useful as a rule of thumb (not as law):

- **Model size** — the model should be big enough to span several GPUs: roughly dense ≥ ~70B (e.g. Llama 70B), MoE ≥ ~100B. Below this threshold the overhead of moving state between GPUs tends to eat the benefit.
- **Context and traffic** — a typical inflection point is around contexts in the order of **~100K tokens** with enough concurrency. That is where prefill/decode separation starts to repay its complexity.

Two practical consequences to remember:

1. On small or low-scale models, disaggregating is often **overkill**: the state communication overhead exceeds the gain. In demos with very small models (e.g. 1.7B) the benefit does not show at all — they only serve to show the mechanics.
2. **Cache-aware routing**, on the other hand, almost always pays off, even on small models: it improves reuse and balance without requiring physical pool separation.

> **A number that gives the scale.** [Session/Roadmap] Monolithic (aggregated) serving can leave about **30–40% of performance on the table** on models like Qwen, GLM or DeepSeek compared to a well-configured disaggregated setup. It is the order of magnitude that explains why, beyond a certain scale, the complexity of disaggregation pays for itself.

---

**Part II — Architecture**

With the fundamentals in hand, we can look at the overall structure. The best way to study Dynamo is to think of it as three cooperating planes. Once the planes are internalized, every component naturally finds its place.

## 5. The three planes

| Plane | Role | Main components |
|---|---|---|
| **Request plane** (hot path) | The request path: it must be light, fast and stable, because tokens and streams to the client flow through it. | Frontend, Router, prefill worker, decode worker |
| **Control plane** | Decides how the system adapts: replica count, placement, scaling, liveness, convergence toward the desired state. | Planner, Operator, CRDs, discovery, Grove, scheduling |
| **Storage / events plane** | Manages cache state: how to make it visible, reuse it, move it and offload it. | KV events, KVBM, NIXL |

This split avoids the most common mistake when studying Dynamo: reducing it to "a router" or "an autoscaler". It is a system that coordinates request, control and state at the same time.

## 6. The flow of a request

Let us follow a request from start to finish — the most direct way to see the planes at work.

- The client talks to an **OpenAI-compatible frontend**, which normalizes the request.
- The **router** chooses where to send it. With KV-aware routing it weighs not only the worker's load but how much context can be reused on that worker.
- If prefill is needed, a **prefill worker** computes the prompt's KV cache.
- The KV cache is handed over, shared or made accessible to the **decode worker**.
- The **decode worker** continues autoregressive generation and streams tokens to the client.

> ⚠️ **Where disaggregated is fragile.** The handoff between prefill and decode is the delicate point. If the KV transfer is slow, you lose part of the benefit. If the router chooses badly, you pay for useless recomputation. If the pools are unbalanced, one of the two becomes the bottleneck. Keep these three risks in mind: they explain why the Router, NIXL and the Planner exist.

---

**Part III — The request plane in detail**

Now the components, grouped by plane. We start with the hot path, because it is what the client sees directly.

## 7. Router — the brain of the request path

The Dynamo KV Router picks workers by weighing three factors together: computational cost, load and KV cache overlap. The point is not round-robin distribution but reducing useless work: if a request has a prefix already present on a worker, it pays to send it there to avoid rebuilding the KV cache.

This is crucial in real workloads where part of the prompt repeats often — system prompts, instructions, application context, conversation history, uploaded documents: multi-turn chat, agents, RAG, coding assistants, repeated requests, similar prompt templates.

### How the Router decides, in practice

The cache-vs-load balance is not a black box: it is a cost function with a configurable weight. **[Baseline]** The base knob, given by the current documentation as the primary parameter, is `--router-kv-overlap-score-weight` (default 1.0). The cost formula is:

`cost = overlap_score_weight × prefill_blocks + decode_blocks`

and the Router picks the worker with the lowest cost. What the weight means:

- **High `overlap_score_weight` (> 1)** — favors cache reuse (less prefill to recompute → better TTFT). The typical logic of **routing toward prefill**.
- **Low `overlap_score_weight` (< 1)** — favors load distribution → better ITL. The typical logic of **routing toward decode**.
- **`overlap_score_weight = 0`** — ignores prefix caches entirely: pure load balancing.

The weight can also be overridden per request (via agent hints), useful when different request types in the same deployment want different trade-offs.

Under the hood, the global view of the cache is kept by the **KvIndexer**, which builds a prefix tree (in some documentation pages also called a radix tree) from the KV events published by the workers. This is what lets the Router estimate, for every worker, both the load and the KV hit rate it would get by routing there. The heuristics are **open source and overridable**: `KV match + GPU load` is the default, but you can write custom policies.

> **[Session/Roadmap] Tier-aware and shared-pool routing.** In configurations with a hierarchical cache or a shared pool (e.g. HiCache + Mooncake) routing becomes **tier-aware**: the Router looks not only at the device-local overlap of a single worker, but also at blocks recoverable from external tiers or a shared pool, discounting the prefill cost accordingly. In this regime additional parameters come into play, such as `overlap_score_credit` (device-local overlap credit) and `shared_cache_multiplier`. Read them as an **extension** of the base model for multi-tier scenarios, not a replacement: the `overlap_score_weight` knob remains the primary reference in the standard case.

> **The conceptual leap of modern routing.** The question is no longer *"which worker is least loaded?"*, but ***"which worker minimizes the overall cost of the request while respecting latency?"***

> **Why it matters at design time.** This is a point where the system exposes an explicit lever, not opaque automatic behavior. In a disaggregated setup the two sides (prefill and decode) typically use opposite weights — maximum cache on one side, maximum balance on the other — and being able to explain it is a good sign of mastering the request path.

## 8. Backend engines — vLLM, SGLang, TensorRT-LLM

Dynamo relies on inference engines that do the model execution work: internal scheduling, batching, kernels, engine-specific memory management, paged attention or equivalents, parallelism and runtime optimizations. On top, Dynamo adds a common layer: routing, distributed runtime, discovery, KV events, transfer, Planner and deployment orchestration. It is engine-agnostic, with extensible support.

> **Engine** = *how* I execute the model. **Dynamo** = *how* I coordinate many engine workers inside a distributed inference system.

An important practical consequence: you do not position Dynamo *against* vLLM. Dynamo **orchestrates** vLLM. The distinction avoids a very common misunderstanding.

---

**Part IV — The storage/events plane: managing state**

This is the part that sets Dynamo apart from a simple load balancer. Here it is decided how state — the KV cache — is managed, moved and extended. The fundamental distinction to keep throughout the section is just one: **whoever manages the cache is different from whoever transfers it.**

## 9. KVBM — KV Block Manager

The KVBM is the component that manages KV cache blocks: allocation, layout, lookup, life cycle, state transitions, reuse, eviction and offload. It is the logical cache management layer.

Mentally, associate the KVBM with these questions:

- Where do I keep the KV cache?
- Which block can I reuse?
- What do I evict?
- What do I offload?
- How do I extend effective memory beyond HBM?
- How do I enable long context and more concurrency without saturating GPU memory?

The KVBM acts as a unified memory layer across several tiers — GPU HBM, pinned host memory, RDMA-accessible memory, local or distributed SSDs, remote / object / cloud storage — depending on the integration. NVIDIA speaks explicitly of KV cache offloading from HBM to host memory, local disk or remote storage. The tier hierarchy is the bridge to the *offloading* section.

> **Naming note: the "G1–G4" tiers.** In NVIDIA technical sessions and benchmarks the KVBM tiers are often referred to with short labels worth recognizing, even though this manual keeps the descriptive names (HBM / host RAM / SSD / remote storage):
>
> - **G1** = GPU HBM · **G2** = CPU host memory · **G3** = local SSD/NVMe · **G4** = network / remote storage.
>
> A **G3.5** is now joining them: the name Dynamo gives to the CMX tier (see below), a pod-level layer sitting between local SSD and remote storage. Same ideas as the classic tiers, with more compact labels: keep them as vocabulary for reading benchmarks and slides, not as a different hierarchy.
>
> *Snippets from NVIDIA benchmarks (read as orders of magnitude, not guarantees):* enabling G2 alone (host memory) can give about a 31% TTFT reduction and a few percentage points of output TPS/GPU; the **G2 + G3** combination tends to beat G2 alone at lower TCO, but requires tuning the block size per tier; **G4** (remote storage, demonstrated with the VAST file system) went GA with Dynamo 1.0. An indirect confirmation of the offload warning: in tests with a small working set, the cache hit on the CPU tier stayed at 1–2%, i.e. almost useless — offload only pays when there is real reuse.

### A tier dedicated to context memory (CMX)

Beyond the classic tiers (HBM → host RAM → SSD → remote storage), the idea is emerging of formalizing a layer dedicated to *context memory* shared at **pod level**, distinct from the tiers above. NVIDIA presents it as **CMX (Context Memory)**: a tier designed for warm / shared reuse of the KV cache across several GPUs and servers, based on BlueField-4, Spectrum-X Ethernet and key-value APIs via DOCA Memos. Conceptually it is an attempt to treat the KV cache as a first-class infrastructure resource, no longer as an internal detail of the engine.

> ⚠️ **Future-proof, not a 1.2.x baseline.** CMX is an **emerging NVIDIA platform/architecture** for the context-memory tier (publicly presented as the Context Memory Storage Platform), not a feature to assume as available and stable in every Dynamo deployment. Read it as the *direction* of KV tiering — a "pod-level" tier sitting conceptually between HBM and remote storage — not as a tier available in today's deployment. In spirit it is a relative of software solutions like LMCache (reusing and persisting the KV cache beyond HBM, "prefill once, reuse everywhere"), but placed much lower in the stack: DPUs, fabric and flash storage instead of a software layer above the engine.
>
> *Architectural interpretation:* the value is not "magically expanding the context window", but raising the cache hit rate and reducing recompute. When the model really needs the KV cache, it still has to come back onto the accelerated path to GPU/HBM — the external tier keeps it and makes it recoverable, it does not execute it.

> ⚠️ **The tiers are not equivalent.** Keep in mind, already at this level, that the tiers have very different latency and bandwidth, and that not every engine / version / backend combination is equally mature. It is the same warning as in the *ecosystem* section, but worth recalling here because it is a concrete operational choice: the "right" tier depends on working set, expected reuse and fabric quality, not only on available capacity.

## 10. NIXL — the data movement layer

NIXL is the data transfer library/layer. It moves KV cache and other data between GPUs, CPUs, remote memory, storage and nodes, and is designed to make data movement efficient in distributed scenarios.

It has two visible roles: in disaggregated serving it transfers the KV cache between prefill and decode workers; in KV offloading it enables movement between memory tiers. The documentation cites unified support for data/storage transactions, P2P GPU transfer, RDMA, NVLink remote memory sharing, dynamic block registration, metadata exchange and plugins for storage backends.

> **Implementation detail.** Out of the box NIXL uses **UCX** as the backend for P2P transfer, with the option of integrating other communication libraries. An important property for inference is that the transfer is a **two-party, peer-to-peer, asynchronous handshake**: to move KV between two workers you only need the other's handle, with no global NCCL-style "comm world" and no consensus across all nodes. This is what lets you add or remove prefill/decode workers on the fly without restarting the system — it enables dynamic scaling and fault tolerance.

> **The formula to remember.** NIXL **is not the cache policy: it is the transport**. KVBM (and LMCache / HiCache / FlexKV) **manage** the cache; NIXL **transfers** blocks between memory and storage domains.

This clean separation between memory management and data movement is exactly what makes the architecture composable.

## 11. KV cache offloading

Offloading means moving part of the KV cache out of GPU HBM to cheaper or larger tiers: CPU RAM, local SSD, remote filesystem, distributed storage or object storage.

The reason is direct: HBM is extremely fast but limited and expensive. With long contexts, many concurrent sessions and multi-turn, the KV cache often becomes the real bottleneck. Offloading raises effective capacity, reduces recomputation and can improve TTFT when the system manages to reuse state already computed. NVIDIA presents it as a technique to increase the effective retention and reuse capacity of the KV cache — making long-context and high-concurrency workloads more sustainable — and to cut TCO by reusing precomputed state.

> ⚠️ **When offload does NOT pay.** If the working set is small and fits comfortably in HBM, offload only adds overhead. If the workload has little reuse, it helps less. If storage and network are not adequate, offload itself becomes the bottleneck. The rule: offload is a lever, not a guarantee.

---

**Part V — The control plane: configuration, scaling, scheduling**

What remains is how the system is configured at the start and how it adapts over time. This is the plane that turns Dynamo from a runtime into a platform.

## 12. Planner — pipeline-aware autoscaling

The Planner watches traffic and metrics and decides how to scale prefill and decode to meet SLAs/SLOs such as TTFT and TPOT/ITL. After an offline configuration (with AIConfigurator or DGDR), real traffic can vary a lot: the Planner looks at live traffic and makes scaling decisions to absorb spikes while keeping the SLAs.

It is worth spelling out what makes it more than an autoscaler: the Planner does not work on a single proxy metric, but combines profiling, GPU utilization and traffic patterns, and makes scaling decisions **aiming directly at TTFT and ITL targets**, not at an indirect indicator. This is the qualitative difference from an HPA, and the reason it is a control plane component and not a simple trigger.

> **Why a Kubernetes HPA is not enough.** An HPA sees a metric on a resource and scales that resource. The Planner knows that prefill and decode are two parts of the same pipeline. If it scales only prefill while decode is saturated, the system does not improve. If it scales only decode while prefill capacity is missing, TTFT stays high.
>
> In practice there is an **optimal prefill/decode ratio** that changes with traffic: scaling prefill 3× without scaling decode may only *move* the bottleneck, not remove it.

> **[Session/Roadmap] The Global Planner and SLA pools.** Recent versions introduce a **Global Planner** built on the concept of *pools*: several compute pools can serve the **same model** with different configurations (e.g. one optimized for latency, one for throughput). The Planner can then move workers from one pool to another as load changes — useful when you want to serve differentiated SLAs (premium low-latency requests vs lower-priority ones). It is in trials with selected partners: treat it as a direction, not a baseline. In the field, NVIDIA cites partners (e.g. Alibaba) reporting fewer TTFT and TPOT SLA violations with fewer GPUs thanks to pipeline-aware autoscaling.

## 13. AIConfigurator — offline configuration

AIConfigurator is the suite's offline performance/configuration tool. It finds a good starting configuration for a given model, hardware, GPU budget, workload and SLO. It does not replace a real benchmark: it narrows the search space by removing guesswork in disaggregated clusters, recommending prefill/decode configurations and model parallelism strategies against model, GPU budget and SLO.

It takes as input a workload shape (ISL/OSL), a hardware system, a GPU count and targets such as TTFT/TPOT; as output it proposes a promising configuration: aggregated or disaggregated, number of pools, parallelism, prefill/decode ratio and the deployment YAML/DGDR.

> **[Session/Roadmap] The DGDR / Profiler umbrella.** In the most recent material, AIConfigurator and AIPerf should be read inside a broader workflow: the **Dynamo Profiler**, orchestrated by a Kubernetes custom resource called **DGDR (DynamoGraphDeploymentRequest)**, a "deploy-by-intent" approach. In a single YAML you declare model, backend, hardware and target SLA (TTFT, ITL); the Dynamo Operator then does hardware discovery, runs profiling (online with real deployments, or offline via AIConfigurator in ~20–30 s), generates the deployment DGD, applies it and — optionally — enables the Planner. In practice DGDR folds the *simulation → benchmark → autoscaling* triad into a single declarative flow. It is in **beta**: the mental model (offline config → validation → runtime autoscaling) stays the one in §14, but the modern operational packaging is DGDR.

## 14. AIPerf — benchmark and validation

AIPerf is the benchmark/performance report side: it measures the performance of generative models with detailed metrics via CLI and reports. It is the step that validates in the field what AIConfigurator assumed.

> **The sizing triad.** **AIConfigurator** → "start from this configuration" (simulation). **AIPerf** → "this config, on this hardware, with this workload, produces these numbers" (benchmark). **Planner** → "in production, with variable traffic, I adapt the replica count" (runtime autoscaling).

The cycle is therefore **simulation → benchmark → runtime autoscaling**: the backbone of any serious approach to sizing.

## 15. Grove — Kubernetes-native orchestration

Grove is the Kubernetes layer for describing and orchestrating complex inference workloads. It is not the router and not the inference engine: it is an API/operator that tells Kubernetes "this inference application is made of router, prefill, decode and possibly other components; they have dependencies, separate scaling, gang scheduling and topology constraints".

It lets you describe a system ranging from a single pod to multi-node disaggregated deployments as a single custom resource. The vocabulary to remember:

| Grove object | Meaning |
|---|---|
| **PodCliqueSet** | The whole inference application described as a set of cliques. |
| **PodClique** | A role/component: router, prefill or decode. |
| **PodCliqueScalingGroup** | Scaling of multi-pod groups, useful for TP groups. |
| **PodGang** | Atomic/coordinated scheduling. |
| **topologyConstraint** | A request for coherent physical placement (e.g. same rack). |

Why it is needed: if you model prefill and decode as separate Kubernetes resources, the scheduler can place them but **does not know** they are part of the same pipeline — you would have to manage affinity, coordinated rollouts and relationships by hand. Grove brings that relationship into a single resource (a PodCliqueSet), coordinating startup, life cycle, scaling and topology constraints. It is not "more elegant YAML": it expresses application structure and physical constraints that base Kubernetes does not natively know.

## 16. Scheduling — why standard Kubernetes is not enough

For a multi-pod LLM workload, asking for "8 pods with GPUs" is not enough. You must make sure the pods forming a single tensor-parallel instance start together, sit close to each other and are never left half-scheduled. Three capabilities are needed that base Kubernetes does not offer natively:

| Capability | What it is for |
|---|---|
| **Gang scheduling** | Avoids partial deployments that hold GPUs but cannot serve. |
| **Hierarchical gang scheduling** | For nested groups: a TP group inside prefill or decode, plus the overall pipeline. |
| **Topology-aware placement** | Places coupled components on favorable network/interconnect domains. |

This is central in NVLink/NVSwitch/RDMA architectures: if prefill and decode exchange KV cache, their **physical distance matters**. About Grove, NVIDIA cites the goal of co-locating prefill and decode in the same NVL72 rack to optimize KV cache transfer.

## 17. Fault tolerance

Dynamo treats failure as a normal event, not an exception, with mechanisms on several levels: request migration/cancellation, health checks, graceful shutdown, endpoint draining, load shedding, lease expiry and rerouting. In practice they come down to four ideas:

1. The router/frontend can have several replicas.
2. Workers report health and stop receiving traffic when terminating or unhealthy.
3. Request migration moves in-flight work to healthy workers, where supported.
4. Load shedding rejects requests above a threshold, instead of making everything worse by propagating overload.

> **The guiding principle.** The goal is not *"no component fails"*, but **"the system degrades in a controlled way"**.

## 18. Elasticity between pools

Dynamo introduces separate scaling between pools and, with the right components, can rebalance capacity between prefill and decode. It must be stated precisely, though: do not imagine that "a prefill server becomes decode live" without constraints.

More correctly: Dynamo/Grove/Planner can scale one role down and another up, or allocate new replicas on available resources, if hardware, backend, model parallelism, container image, configuration and KV layout allow it.

> **Roadmap vs baseline.** Recent versions point toward more dynamic forms of conditional prefill/decode and more flexible workers (workers serving prefill or decode on demand, policy-driven routing). Treat it as **technical evolution**, not as a universal production-ready guarantee: what is roadmap should be read as direction, not as a baseline to count on today.

The correct summary: Dynamo enables independent scaling and coordinated rebalancing of capacity between prefill and decode pools, subject to backend, hardware and deployment constraints. In one line, to keep as a boundary: **Dynamo coordinates separate scaling and capacity reallocation, but it should not be understood as universal hot role switching between prefill and decode** — a decode worker does not become a "live" prefill worker without considering engine, parallelism, container, model, KV layout and scheduling.

---

**Part VI — Deployment, ecosystem and placement**

## 19. Deployment modes

Dynamo is used in several postures, and the choice depends on the maturity of the environment.

| Mode | When | What it includes |
|---|---|---|
| **Local / container** | Development, testing, incremental adoption, standalone components. | Frontend, router, Python/Rust workers, simple discovery |
| **Kubernetes** | Multi-node production, GPU fleets, autoscaling, life cycle management. | CRDs, operator, Gateway API, Grove, ModelExpress, observability |

Dynamo is **Kubernetes-native for production, but not Kubernetes-only**. There are also two routing modes:

- **Standalone mode** — client → Frontend → Router → workers.
- **Gateway mode** — Dynamo integrates with the Kubernetes Gateway API Inference Extension; the KV-aware logic can live in the gateway layer through an endpoint picker/plugin.

Gateway mode is often more natural in enterprise environments that already have standards for ingress/gateway, policy, auth, rate limiting and centralized observability.

> **Worker discovery: etcd vs Kubernetes-native.** Dynamo originally registered workers via **etcd**. At customers' request, **Kubernetes-native discovery via endpoint slices** was added — and made the default — keeping etcd as an option. The practical advantage: a new prefill worker that passes its health check becomes immediately discoverable and the Router can send it traffic; likewise decode workers discover each other ad hoc. This is what makes the system truly dynamic — workers are added and removed without restarting everything — and it fits better with the Kubernetes standards already in place.

## 20. ModelExpress — the cold start problem

ModelExpress is about model loading and weight transfer. It matters because autoscaling and fault recovery do not depend only on the replica count: if starting a new replica takes a long time to load huge weights, real elasticity is slow.

The documentation positions it as a component for loading model weights quickly through caching and transfer via NIXL to other GPUs, also used in fault tolerance. In practice: the Planner may decide to scale and Grove may schedule, but if the model takes too long to load, the response to the spike arrives late. ModelExpress works precisely on **cold start**.

## 21. Ecosystem and integrations

The suite is modular and faces a broad ecosystem:

| Area | Components |
|---|---|
| Inference engines | SGLang, TensorRT-LLM, vLLM |
| Kubernetes | Inference Gateway |
| Memory management | LMCache, HiCache, FlexKV (besides KVBM) |
| Networking / storage | Mooncake, DOCA NetIO, GDS, POSIX, S3, 3FS |
| Context-memory tier *(future-proof)* | NVIDIA CMX (reference architecture); enterprise backends e.g. VAST Data |
| Multi-hardware | Intel XPU, AMD |
| Emerging components *(preview)* | FlexTensor (weight streaming for GPUs with little memory), AITune (automatic backend tuning for custom models) |

> ⚠️ **Caution on what "support" means.** "Ecosystem support" **does not imply** the same maturity, the same performance or the same level of production-grade validation for every combination. In a real architecture always check the support matrix: engine version, backend, model, hardware and feature interactions.

### A concrete case — a data platform as a KV tier

An example of how the storage ecosystem is moving *toward the inference runtime* is the integration of enterprise data platforms (e.g. VAST Data) with Dynamo for KV cache offload/reuse, and their declared adoption of the NVIDIA CMX platform. The two things should be kept apart:

- **[Vendor claim] Today:** VAST integrates with Dynamo as a storage tier for KV cache offload/reuse. In KVBM terms, the current integration is declared at **tier G3** (local storage); the roadmap includes integration at **G4** (remote storage) and at the new **G3.5** (CMX, announced at CES). At G4 the KVBM manages the tier "opaquely", without exposing its complexity to the rest of the system. In the test scenario cited by the vendor (Llama ~5B, ~128K context, 8×H100, 200 Gb network), it declares a ~20× TTFT improvement (from ~65 s to ~3 s for a 128K prefill) and ~90% of GPU time saved. ⚠️ These numbers are a **vendor benchmark on a specific scenario, not a sizing assumption**: do not use them to size a different deployment without measuring again.
- **[Session/Roadmap] Next step:** VAST declares support for CMX with clusters based on BlueField-4 and Spectrum-X, positioning itself as an *enterprise implementation* of the context-memory tier: single namespace, data reduction, encryption, retention, multi-tenancy and observability on top of the NVIDIA data path.

> ⚠️ **Where NVIDIA ends and the vendor begins.** The hardware / fabric / base-software stack of CMX remains NVIDIA's (BlueField-4, Spectrum-X, DOCA Memos, NIXL/Dynamo). The storage vendor **does not own CMX**: it is one possible enterprise backend for it, compatible with NVIDIA's libraries and data path. In short: NVIDIA defines the architectural standard of the context-memory tier; the vendor makes it usable in production with storage, governance and enterprise services.
>
> *Strategic implication (interpretation):* if CMX becomes a stable part of AI factories, vendors such as VAST / Dell / HPE / NetApp / DDN stop selling only "storage for datasets/checkpoints" and start selling a component close to the inference runtime — the **backend of context memory**. It is also the point where the KV cache stops being a "technical cache" and becomes business data (prompt fragments, conversations, documents, agent state), with all the implications for encryption, multi-tenancy, retention and audit.

### The KV cache as business data

This point deserves its own section because it has direct operational implications, not only strategic ones. **When the KV cache leaves HBM**, it stops being internal engine state and becomes persisted data containing sensitive information: prompt fragments, conversation history, context documents, agent state. Some concrete consequences follow:

- **Encryption** of the KV cache at rest and in transit, seen as a requirement for alignment with regulatory frameworks — the **EU AI Act** is cited explicitly in this context.
- **SLA-driven retention**: the life cycle of the context must be governed (how long it is kept, when it is deleted), not left undefined.
- **Multi-tenancy and isolation**: on shared infrastructure, clear policies are needed to avoid prompt leakage between tenants.

> **Why it matters for an integrator in the EU.** For a system integrator bringing on-prem or sovereign-cloud inference to the European market, "treating the KV cache as enterprise data" is not just a technical caveat: it is a positioning angle. Governance, encryption and retention of the context tier are exactly the topics a regulated customer (healthcare, finance, public sector) raises — and they are hard to address with do-it-yourself offloading and no adequate data layer.

### Offloading economics — the order of magnitude

**[Vendor claim]** The economic argument for offload is not "making the GPU faster" but **making it available more often** by avoiding recomputation: storage becomes a *multiplier* of compute. The numbers projected by the vendor (estimates, not promises) speak of **+60–130% tokens per dollar** with conservative cache hit rates of 40–60% for enterprise flows; in agentic/coding tasks, where context reuse is higher, the gain grows. On top of that comes **~1.4× data reduction** measured on offloaded KV cache (mixed dataset: documents, code, chat), which translates into more capacity → more sessions, more users, higher hit rates.

> ⚠️ **The honest physical limit.** All this holds as long as network and storage keep up. In tests, the bottleneck moves to the network (the KV pattern is read-heavy with large blocks, so network-bound rather than IOPS-bound); accelerations such as GPUDirect Storage, RDMA and S3-over-RDMA exist precisely so that the data path does not become the new bottleneck. It is the "economic" version of the same warning in §11: offload is a lever, not magic.

## 22. Where Dynamo sits in the Dielabs stack

We close the architecture by mapping the components onto the L0–L6 layers. The layers are defined in [The Inference Engineering Manual](inference-engineering-manual.md), Chapter 3; this table says where each Dynamo component falls.

| Layer | Content |
|---|---|
| **L6** — Client / Entry Point | OpenAI-compatible clients, API consumers, agent frameworks |
| **L5** — API Gateway / Edge | Inference Gateway, auth, rate limiting, policy, endpoint picker |
| **L4a** — Request Orchestration | Dynamo Frontend, Router, request flow, KV-aware routing |
| **L4b** — GPU Workload Optimization | Planner, AIConfigurator, prefill/decode pools, parallelism strategy, autoscaling |
| **L3** — Inference Backend / Engine | vLLM, SGLang, TensorRT-LLM |
| **L2.5** — Model Artifact | Weights, numeric format, tokenizer, chat template, LoRA adapters; fast weight loading at cold start is ModelExpress (§20) |
| **L2** — Runtime & Packaging | Containers, nvidia-container-runtime, Kubernetes pods, Helm, operator |
| **L1** — Driver & CUDA | Drivers, CUDA, communication stack (NCCL) where applicable |
| **L0.5** — Interconnects and Fabric | NVLink/NVSwitch, PCIe, RDMA fabric (InfiniBand, RoCE); this is where the KV cache travels between prefill and decode pools (NIXL, §10) |
| **L0** — Hardware | GPU, HBM, CPU, RAM, NVMe |

NIXL and the KVBM **cross several layers**: they sit near L4b/L3 as a state/memory/data movement function, but depend heavily on the lower layers — HBM, RAM, storage, RDMA, NVLink, GDS. It is the stack-level translation of the fact that state management is transversal to the whole system.

> *Future-proof note:* the CMX/VAST tier described in §9 and §21 would sit as a downward extension of this state/memory function — a "pod-level" tier anchored at L0/L0.5 (BlueField-4 DPU, Spectrum-X fabric, flash) but logically orchestrated by KVBM/NIXL at L4b/L3. Today it is an architectural direction, not a stable row of the production stack.

---

**Part VII — Sizing and operations**

Theory becomes useful when you can size and observe a deployment. This part lays out the workflow and the metrics in sequence, then collects the conceptual mistakes to avoid.

## 23. How to size a Dynamo deployment

The clean workflow is an ordered sequence. Each step prepares the next.

1. **Define the workload**: model, quantization/dtype, ISL, OSL, request rate, concurrency, prompt distribution, percentiles, multi-turn rate, expected cache reuse, streaming yes/no, tool calling yes/no.
2. **Define the SLOs**: target TTFT, target TPOT/ITL, minimum throughput, p95/p99 latency, GPU budget, maximum cost per token or per request.
3. **Use AIConfigurator** for the initial configuration: aggregated or disaggregated, GPU count, parallelism, prefill/decode ratio.
4. **Validate with AIPerf** or realistic benchmarks, ideally with a traffic trace or a dataset similar to production.
5. **Deploy with Dynamo/Grove** if you are on Kubernetes.
6. **Enable the Planner** to adapt prefill/decode to real traffic.
7. **Observe**: TTFT, ITL, queue depth, KV hit rate, KV cache utilization, GPU utilization, memory pressure, transfer latency, prefill/decode imbalance, rejected requests, migration/cancellation, cold start.

## 24. Operational metrics to watch

For Dynamo, watching "GPU utilization" is not enough: on its own it does not tell where the problem is. The key metrics, and what they tell you:

| Metric | What it tells you |
|---|---|
| **TTFT** | Whether prefill, routing, cache and queue are working well. |
| **TPOT / ITL** | Whether decode and concurrency are holding up. |
| **Queue depth per role** | Separates a prefill bottleneck from a decode one. |
| **KV cache hit rate** | How much routing and cache are avoiding recomputation. |
| **KV transfer latency** | In disaggregated setups it can eat the benefit. |
| **GPU memory pressure** | The KV cache can saturate HBM. |
| **tokens/s/GPU** | Economic efficiency. |
| **tokens/s/user** | User experience. |
| **Replica count per pool** | How the Planner is reacting. |
| **Load shedding / rejections** | Saturation, or active protection. |

## 25. Conceptual mistakes to avoid

These are the most common misunderstandings. Recognizing them is a good test of understanding.

**Mistake 1 — Dynamo is "an alternative vLLM".** No. Dynamo sits above the engines: it orchestrates them, it does not replace them.

**Mistake 2 — ISL/OSL are SLAs.** No. They are workload shape. SLAs/SLOs are TTFT and TPOT/ITL.

**Mistake 3 — Disaggregated is always better.** No. It helps when traffic and scale justify the separation.

**Mistake 4 — NIXL and KVBM are the same thing.** No. NIXL moves data; the KVBM manages KV blocks and policies.

**Mistake 5 — Base Kubernetes schedules multi-pod LLM workloads well.** No. It schedules pods, but it does not understand TP groups, prefill/decode, rack locality, NVLink domains and pipelines.

**Mistake 6 — Dynamo is perfectly hardware-agnostic.** No. It is moving toward a broad ecosystem, but every model-engine-hardware-feature combination must be verified.

**Mistake 7 — CMX/VAST "replace HBM" or are already baseline.** No. They are an external context-memory tier: they do not run the model and do not replace HBM, they make it less wasted. And as of today CMX is a reference architecture / direction, not a stable feature to size on.

**Mistake 8 — Disaggregation pays at any scale.** No. Below certain thresholds (small model, low context and concurrency) the state communication overhead exceeds the benefit: the gain from prefill/decode separation shows beyond a certain model and context size (§4). Cache-aware routing, instead, is almost always useful.

**Mistake 9 — If pods are slow to start, it is Dynamo's problem.** Often not: it is the cold start of the weights (§20) plus Kubernetes' default timeouts. In deployments with large models, the probes' `failureThreshold` must be raised (the ~30 s default is not enough for warm-up), otherwise you run into 503s and restarts while the model is still loading. It is configuration, not a runtime limit.

---

**Part VIII — Summary and consolidation**

## 26. Dynamo in one sentence

> NVIDIA Dynamo is a distributed inference orchestration stack that sits above engines like vLLM, SGLang and TensorRT-LLM, adding disaggregated prefill/decode serving, KV-aware routing, KV cache offload, low-latency KV transfer, SLA-based autoscaling and Kubernetes-native orchestration through Grove.

## 27. Suggested study path

If you had to go through the material from scratch, this is the order that builds understanding cumulatively:

1. Prefill vs decode, ISL/OSL, TTFT/TPOT, KV cache, and why decode is autoregressive.
2. Aggregated vs disaggregated serving: being able to explain when it pays and when it does not.
3. The Router: load-aware, KV-aware, cache overlap, queue.
4. KVBM and NIXL: who manages the blocks, who transfers the data, what offload means — and where an external tier like CMX fits.
5. Planner and AIConfigurator: offline configuration vs online autoscaling.
6. Grove/Kubernetes: PodCliqueSet, gang scheduling, topology-aware placement, startup order, per-role scaling.
7. Fault tolerance and observability.

> **The maturity test.** You should be able to explain what happens when the workload changes from `ISL 4000 / OSL 500` to `ISL 500 / OSL 4000`, and which components react: Router, Planner, prefill/decode pools, KVBM, NIXL, Grove and the related metrics. If you can tell this transition end to end, you have internalized the system.

## 28. The point, in short

Dynamo matters because it marks the shift from model serving to inference systems engineering.

The value is not "running an LLM", but running complex LLM workloads on many GPUs, with variable traffic, long contexts, reusable cache, measurable SLAs, controlled costs, handled failures and a coherent Kubernetes deployment.

For a path in AI infrastructure, Dynamo is one of the most useful frameworks to study because it brings together all the topics that matter: GPU infrastructure, inference performance, Kubernetes, scheduling, the storage/memory hierarchy, networking, observability and autoscaling. It is the point where these worlds stop being separate topics and become a single system.

---

**Part IX — Reasoning scenarios**

The previous parts explain the components. This one puts them to work. There is no new theory: there are three recurring traffic situations, each read the same way — what becomes the bottleneck, which metrics you watch, how the system should react. It is direct training for the "maturity test" in §27: being able to tell end to end what changes when the workload changes.

## 29. How to read a scenario

For each scenario it pays to answer four questions, in order:

1. **Where does the load fall?** Prefill or decode — i.e. which of the two phases dominates given the shape of the workload.
2. **Which metrics confirm it?** The metrics in §24 are not all watched together: each scenario has a diagnostic subset.
3. **How does the control plane react?** What the Planner should do, and in which direction to tilt capacity.
4. **What risk lies in wait?** The typical way the scenario degrades if the reaction is wrong.

Keeping this scheme fixed is the point: the same reasoning applies to workloads that are not in this list.

## 30. Scenario 1 — High ISL, low OSL

**Long prompts, short answers.** Typical of summarization, classification over documents, structured extraction, reranking, judging long texts.

- **Where the load falls** — on **prefill**. Building the KV cache of a long prompt is compute-heavy; generation, being short, weighs little. The system spends most of its time before the first token.
- **Metrics to watch** — **TTFT** first (it is dominated by prefill), **prefill pool queue depth**, **KV transfer latency** (in disaggregated setups the freshly built cache must reach decode), and **cache reuse / KV hit rate** if documents or templates repeat.
- **How the control plane reacts** — the Planner should increase **prefill** capacity more than decode. It is the textbook case in which tilting toward prefill is correct, and in which scaling decode would bring little benefit.
- **Risk in wait** — saturating prefill while decode stays half idle: TTFT rises and the user waits, even though the decode GPUs look "free". This is exactly the situation disaggregation exists to solve.

## 31. Scenario 2 — Low ISL, high OSL

**Short prompts, long generation.** Typical of open chat, content writing, extended reasoning, long code generation.

- **Where the load falls** — on **decode**. Prefill is quick; the cost is in token-by-token autoregressive generation, which is memory-bandwidth-bound and grows with concurrency.
- **Metrics to watch** — **ITL / TPOT** first (they measure the pace of generation), **decode pool queue depth**, **memory bandwidth** and **GPU memory pressure** (the KV cache grows with every token and every concurrent session), **concurrency** and **tokens/s/user** as an indicator of perceived experience.
- **How the control plane reacts** — the Planner should tilt toward **decode**. Here the lever is not just adding replicas: what also counts is managing the swelling KV cache (offload if needed, §11) and how many concurrent sessions each worker can hold without ITL rising.
- **Risk in wait** — more concurrent requests saturate HBM with KV cache, ITL degrades for everyone and the system enters memory pressure. Without offload or decode scaling, quality of service collapses across the board, not on single requests.

## 32. Scenario 3 — Multi-turn, agents, RAG with repeated prefixes

**Part of the prompt repeats systematically:** system prompt, instructions, conversation history, context documents, application templates. Typical of multi-turn assistants, agents, coding assistants, RAG pipelines.

- **Where the load falls** — it is no longer (only) a matter of prefill vs decode: the dominant factor becomes **routing**. A large share of prefill is *avoidable recomputation* if the prefix is already cached on some worker.
- **Metrics to watch** — **KV cache hit rate** as the main metric (it tells how much routing and cache are actually avoiding work), **TTFT** (a prefix hit lowers it noticeably), and the load distribution across workers to check that KV-aware routing is not concentrating too much traffic on a few "hot" nodes.
- **How the control plane reacts** — here the protagonist is **KV-aware routing**, not the Planner. The principle of §7 becomes operational: **the least loaded worker is not always the best one**. It may pay to send the request where useful cache already is, accepting a slightly busier worker but saving the full rebuild of the prefix.
- **Risk in wait** — two opposite risks. Purely load-based routing that ignores the cache → continuous recomputation and high TTFT. Routing too attached to the cache → hotspots on a few workers and a queue growing there while others sit idle. Good behavior lies in balancing cache overlap and load, which is exactly what the Router does by weighing the two factors together.

> **The thread of the three scenarios.** As the shape of the workload changes, the component carrying the weight changes: prefill-bound → Planner toward prefill; decode-bound → Planner toward decode plus KV management; repeated prefix → KV-aware Router. If you can say, for any workload, *which of these three regimes prevails and why*, you have the mental model the whole manual was built to give you.

---

*All content is original Dielabs work by Diego Bardella.*
