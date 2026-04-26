---
layout: default
title: "Disaggregated Inference"
---

# Disaggregated Inference

---

## 1. Foundation: Prefill and Decode Are Two Different Phases

Autoregressive LLM generation consists of two phases with opposite computational profiles.

### 1.1 Prefill

The model receives the entire prompt and processes it in a single parallel forward pass. For every token in the prompt it computes the Key (K) and Value (V) vectors of the attention mechanism and stores them in the KV cache. The output of this forward pass is the **first generated token** — there is no separate "generation start" step. The first token is the direct product of reading the prompt, exactly as a model in training predicts the next token given a sequence.

TTFT (Time To First Token) measures the duration of this phase.

**Profile:** compute-bound. High density of matrix operations, all tokens processed in parallel. Benefits from hardware with high compute capacity (FLOPS). An H100 in prefill saturates the tensor cores with massive matmuls.

### 1.2 Decode

From the second token onwards, the model generates one token at a time. At each step it computes Q, K and V only for the new token, appends K and V to the cache, and computes the attention of the new token against all previous K and V already in the cache. The forward pass processes a single token, not the entire sequence.

TPOT (Time Per Output Token) is the average time per token in this phase. ITL (Inter-Token Latency) is the punctual latency between consecutive tokens.

**Profile:** memory-bandwidth-bound. The bottleneck is reading from memory at every step: both the model weights (which must be read for every forward pass) and the KV cache (which grows with the sequence and becomes the dominant factor on long contexts). Compute per single token is minimal — time is dominated by data movement. An H100 in decode can stay below 20% compute utilization while HBM bandwidth is saturated. Benefits from hardware with high memory bandwidth.

### 1.3 Why the KV Cache Is Essential

Without KV cache, every decode step would be equivalent to a full prefill on the entire sequence accumulated up to that point: to generate token N, the model would have to re-read and recompute K and V for all preceding tokens (prompt + N-1 generated tokens). Each step costs O(n) in compute (attention over n tokens), and there are n steps → cumulative cost grows as O(n³) with sequence length.

The KV cache eliminates the recomputation of K and V: values for previous tokens are already in memory, each step processes only the new token. Attention remains O(n) per step (the new token still has to "look at" all previous ones in the cache), but per-step cost is dominated by reading from the cache, not by recomputation. Total cost drops to O(n²) — a change in order of magnitude, not just a lower constant. This transforms decode from compute-bound (many repeated prefills) to memory-bound (reading model weights and cache), making autoregressive inference practical.

The KV cache is not an optimization added a posteriori by inference servers. It is a direct consequence of the math of attention in transformers: K and V of previous tokens do not change when a new token is added, so recomputing them is pure compute waste. The innovation of inference engines (e.g. PagedAttention in vLLM) lies in the **efficient management** of the cache, not in its existence.

---

## 2. The Conflict: Two Profiles on a Single Piece of Hardware

In the traditional configuration, prefill and decode happen on the same GPU. This creates a structural compromise on two levels, and the second is architecturally more important than the first.

**Level 1 — Instantaneous utilization mismatch.** Prefill wants all available compute capacity; decode wants all memory bandwidth. The two phases compete for the same resources (compute units, memory, bandwidth, batch scheduler). With continuous batching (e.g. vLLM), requests in prefill and requests in decode coexist in the same batch. This improves GPU utilization but creates interference, not only at the compute level but especially at the scheduling and memory bandwidth level: requests in prefill and decode compete for access to weights and KV cache in the same execution cycle. Prefill requests (compute-intensive) slow down decode ones (latency-sensitive) and vice versa. Continuous batching mitigates but does not eliminate.

**Level 2 — Independent scaling patterns.** Prefill scales with the volume of incoming prompts (length × request rate), decode scales with the number of concurrent sessions and their generation length. A workload with long contexts is prefill-heavy; a conversational workload with long replies is decode-heavy. Sizing them together means oversizing one of the two to cover the peak of the other.

Disaggregated serving is born as a structural response: physically separate the two phases on dedicated hardware, scaled independently.

---

## 3. Three Architectures Compared

Given a budget of two GPUs, there are three architecturally distinct ways to use them to serve an LLM model.

### 3.1 Single Node

A single GPU handles the entire prefill + decode cycle. The second GPU stays unused. It is the minimum baseline, useful as a reference to measure throughput per hardware unit.

**Flow:** Client → vLLM Instance (prefill + decode) → Response

### 3.2 Replicated Serving

Two GPUs each run an independent and complete instance of the model. A stateless proxy distributes requests in round-robin. Each instance handles prefill and decode internally.

**Flow:** Client → Proxy (round-robin) → Instance A or B (prefill + decode) → Response

The proxy is simple: it keeps no state, does not know request content, just alternates between the two backends. Additional cost over single node is exclusively the proxy hop (~6% measured overhead).

### 3.3 Disaggregated Serving

The two GPUs specialize: one is dedicated to prefill, the other to decode. The KV cache generated by the prefill node is transferred to the decode node, which uses it for token generation.

**Flow:** Client → Proxy → Prefill Node → [KV cache transfer] → Decode Node → Response

Here the proxy becomes a stateful component that orchestrates a two-phase protocol:

- **Phase 1:** the proxy sends the prompt to the prefill node with the flag `do_remote_decode: true`. The prefill node processes the prompt, generates the KV cache, and returns localization metadata to the proxy (`remote_engine_id`, `remote_block_ids`).
- **Phase 2:** the proxy forwards the metadata to the decode node, which performs a pull of the KV cache via NIXL/RDMA and starts token generation.

### 3.4 Specific Advantages of Disaggregation

**Independent hardware optimization.** Each pool uses hardware suited to its profile. Prefill nodes can be dense GPUs (e.g. H100 SXM with high compute), decode nodes can be GPUs with high HBM bandwidth or even alternative hardware optimized for sequential reads.

**Independent scaling.** Prefill is bursty (a compute peak, then it ends), decode is prolonged. The two pools can be scaled differently based on load. Prefill nodes can even run on spot/preemptible hardware, since their work is atomic.

**Interference elimination.** No contention between prefill and decode requests in the same batch. Each pool handles a single workload type.

**Differentiated pricing.** Input tokens and output tokens have different production costs. Physically separating them makes pricing more transparent and aligned to real costs. This is already the norm in commercial APIs (e.g. input tokens cheaper than output tokens).

---

## 4. The Critical Role of the Interconnect

Disaggregation introduces an architectural cost not present in the other configurations: the KV cache transfer between nodes. End-to-end transfer must be kept in the millisecond range or below — beyond that, disaggregated becomes worse than colocated. This is the operational constraint that dominates the entire data path design.

### 4.1 KV Cache Size: Orders of Magnitude

Size is computable from model parameters with the formula:

```
KV_size = layers × kv_heads × head_dim × seq_len × bytes_per_element × 2  (K + V)
```

Two examples at the extremes of the operational spectrum:

|Model|Context|Precision|KV cache per request|
|---|---|---|---|
|Llama-3.1-8B (GQA, 8 KV heads, 32 layers, head_dim 128)|512 tokens|FP16|**~32 MB**|
|Llama-3-70B (GQA, 32 effective layers per attention, standard configuration)|32k tokens|FP16|**~10 GB**|

With quantized KV (FP8, INT4) size drops by 2-4×. The order of magnitude is still such that the transfer must be fast, because it adds directly to TTFT — the KPI the end client perceives.

### 4.2 Bandwidth and Latency: Operational References

|Technology|Bandwidth|Latency (32 MB)|Impact|
|---|---|---|---|
|RDMA (RoCE / InfiniBand)|100-800 Gbps|< 1 ms|Negligible|
|Standard TCP/IP|1-25 Gbps|10-100 ms|Significant|

Additional references:

- **Intra-rack RDMA latency for small payloads:** tens of µs.
- **Aggregate throughput of a node with 400/800 Gbps NICs:** hundreds of GB/s, depending on number of NICs, fabric tuning and topology.
- **NVLink:** operates on a different scale (TB/s intra-server) and must be kept separate from the "cross-node network" discussion.
- **True KPI for the client:** end-to-end TTFT and sustained throughput in tokens/s under SLO. Bandwidth is a means, not the end.

**Conclusion:** RDMA is an architectural prerequisite for medium/large models with long contexts. On small models (~7B) with short contexts, even 10-25 Gbps TCP can be sufficient. But without a high-speed, low-latency interconnect, the cost of KV cache transfer can exceed the gain from hardware specialization, making disaggregation a more complex architecture that performs worse than simple replication.

### 4.3 NIC vs DPU: What's Actually Needed on the Data Path

The marketing narrative tends to present the DPU as a necessary condition for disaggregated inference. **It is not.** The correct framing is this:

**For the pure KV-cache data path, an RDMA-capable NIC is enough.** The transfer between GPUs of different nodes goes via GPUDirect RDMA:

```
GPU_A → NIC_A → fabric → NIC_B → GPU_B
```

without involving the host CPU or system memory. NVIDIA ConnectX-6/7/8 do this just fine. The DPU (BlueField) is not required to enable this pattern.

**The DPU (BlueField) adds services on the data path, not the data path itself:**

- **Multi-tenant isolation** via DOCA/eBPF
- **Granular per-flow telemetry**
- **Policy offload** for congestion control and QoS
- **NVMe-oF target in hardware** (see §5)

Useful in shared cloud environments and managed platforms; less relevant on a dedicated single-tenant training cluster.

### 4.4 The Fabric: Spectrum-X as a System

NVIDIA Spectrum-X is a **fabric-level** platform, not a single-DPU component. It combines Spectrum-4 switches, NICs (ConnectX-8 or BlueField-3/4) and software stack that together implement adaptive routing and congestion control optimized for RoCE.

The NIC/DPU participates in the system, but the value is in the **switch ↔ NIC** interaction, not in the DPU taken in isolation.

For disaggregated inference this matters because the KV-cache transfer traffic pattern is **bursty and many-to-many** — exactly the case where classical ECMP collapses and adaptive routing is required. On standard RoCE fabrics without adaptive routing, real throughput under synchronous bursts is typically much lower than nominal due to congestion on shared paths.

---

## 5. Storage Tier and KV Cache Pooling

Beyond direct prefill → decode transfer, an architectural pattern relevant from 2025-2026 is **KV-cache pooling**: instead of transferring the KV-cache directly between nodes, it is written to a tier of shared remote memory, where it can be reused **cross-request** (cache hit when the same prompt or prefix appears again — e.g. long system prompts, recurring RAG documents, multi-turn sessions).

### 5.1 NVMe-oF as the Dominant Implementation

Today the pattern is typically implemented via **NVMe over Fabrics**. It is the choice of Mooncake (Kimi/Moonshot) in production and is mature engineering.

**Here the DPU is actually enabling**, not just useful: it can be the NVMe-oF target in hardware, exposing to the decode node "virtual" NVMe devices that actually point to remote memory (shared pool or dedicated storage tier). Doing it in software on the host CPU would be prohibitive in terms of latency and CPU overhead.

It is the same pattern that Lambda Labs (and AWS Nitro, GCP, OCI) use for bare metal instance volumes, applied to the KV-cache.

### 5.2 CXL as Direction, Not State of the Art

**Memory pooling via CXL 3.0 over fabric** is conceptually cleaner than NVMe-oF (memory-semantic access instead of block-semantic, theoretically lower latency). As of 2026 it remains predominantly **direction** rather than widespread practice in production for inference.

It is worth mentioning as a roadmap in conversations with customers/prospects, not as the current state of the art. Anyone presenting it as deployment-ready today is selling vaporware.

### 5.3 Economic Implications of Pooling

KV-cache pooling changes the economics of inference in non-trivial ways:

- **Cross-request cache hit** → prefill is not repeated for prompts/prefixes already processed. This is why OpenAI, Anthropic, and other providers have introduced discounted pricing for "cached input tokens".
- **Reduced load on the prefill pool** → fewer prefill GPUs needed at the same SLO.
- **TTFT trade-off:** lookup in the pool adds latency; if the pool is fast (NVMe-oF via DPU, RDMA) the cost is largely offset by prefill savings.

Whoever designs disaggregated infrastructure should treat the pooling tier as a **first-class component**, not as an optional optimization.

---

## 6. Performance Analysis

### 6.1 The Fair Baseline Question

Most published benchmarks on disaggregation compare 2 disaggregated GPUs against 1 single-node GPU. This comparison is methodologically incorrect: you are doubling the hardware and declaring it goes faster.

The correct comparison is: given a fixed budget of N GPUs, which architecture (replication vs disaggregation) performs better? Only this comparison reveals whether disaggregation offers a real architectural advantage or simply a hardware advantage.

This distinction is often ignored in blog posts and papers, not necessarily in bad faith, but because the focus is to demonstrate that disaggregation works, not to quantify when it is worth it. For those who must make infrastructure decisions, it is a critical gap.

### 6.2 Single Request (Latency Test)

With a single request, there is no concurrency and therefore no possibility to overlap pipeline phases. The expected result is that disaggregation is the slowest configuration:

- **Single node:** minimal latency, no network overhead.
- **Replicated:** the request goes through the proxy and lands on only one of the two nodes. The second is idle. Overhead ~6% (proxy layer cost).
- **Disaggregated:** the request crosses both nodes by design (prefill → transfer → decode). Two network hops with no compensating parallelism.

This test measures the worst case of disaggregation and it is essential to include it: it is the cost paid always, even when the system is idle.

### 6.3 Under Concurrent Load (Throughput Test)

Under concurrent load, the pipeline activates. While the decode node generates tokens for request N, the prefill node is already processing request N+1. The two GPUs work in parallel on different phases of different requests.

Reference data (Thomas/TokenLabs benchmark, 8 concurrent requests, Llama-3.1-8B):

|Configuration|Throughput (tok/s)|Note|
|---|---|---|
|Single Node|Baseline|1 active GPU|
|Replicated|26.5 tok/s|2 GPUs, round-robin|
|Disaggregated|**101.9 tok/s**|2 GPUs, prefill/decode split|

The 26.5 tok/s replication figure is notable: it indicates that at 8 concurrent requests, round-robin between two nodes does not scale linearly. Each node handles 4 requests internally, but contention between prefill and decode in the same GPU caps throughput. Replication does not eliminate the prefill/decode conflict — it replicates it across more nodes. Each node continues to suffer the same internal contention, limiting real scalability.

Reference papers (DistServe, Splitwise, Mooncake) report throughput improvements at equivalent SLO in the order of **2-4×** over colocated on large models, provided the transfer does not degrade TTFT.

### 6.4 The Missing Crossover Point

The benchmark presents only two points: 1 and 8 concurrent requests. This demonstrates that the crossover between replication and disaggregation exists, but does not identify where it falls. A curve with 1, 2, 4, 8, 16, 32 requests would have allowed determining the exact concurrency level at which disaggregation overtakes replication — critical information for capacity planning.

---

## 7. Speculative Decoding in Disaggregated Context

Speculative Decoding (SD) and disaggregated serving are complementary.

### 7.1 The SD Problem in Traditional Configuration

SD uses a small draft model to propose N candidate tokens, which the target model verifies in a single forward pass (mechanically identical to a mini-prefill incremental over the candidate tokens). This multiplies decode throughput.

However SD has costs:

- **TTFT worsens:** the overhead on TTFT is implementation-dependent. In some implementations the draft model executes a complete prefill of the prompt to populate its own KV cache (effectively a double prefill). In others, the draft is initialized in a lighter way (partial priming, embedding sharing, or lazy startup after the first token). On top of these costs come pipeline overhead (scheduler coordination, buffer synchronization between the two models) and the allocation of the double KV cache. The net result is a worsened TTFT, but the magnitude depends on the specific implementation.
- **Double KV cache:** two models in memory = double KV cache footprint, which reduces space available for larger batches.
- **Acceptance rate:** the real gain depends on how many tokens proposed by the draft are accepted by the target. Without this metric, SD benchmarks are incomplete. SD shifts the bottleneck from decode compute to draft model prediction quality: acceptance rate becomes the dominant factor in determining net gain.

For short outputs, the cost of double prefill can exceed the gain in decode. For long outputs, the decode gain widely compensates.

### 7.2 SD + Disaggregated: The Natural Complement

With disaggregated serving, SD initialization overhead (target prefill + draft priming, in whatever form the implementation requires) happens on prefill nodes optimized for compute. Additional cost is absorbed by hardware designed for that workload type.

Decode with SD happens on decode nodes, where the draft model proposes and the target verifies, producing token bursts. Decode nodes are optimized for bandwidth — exactly what is needed to read the KV caches of both models quickly.

Result: SD's weak point (degraded TTFT) is not eliminated — initialization cost still exists — but is shifted to compute-optimized hardware where it is amortized far more efficiently. The strong point (accelerated decode) remains intact on decode nodes.

---

## 8. Implications for Hybrid CPU+GPU Architectures

Disaggregated serving opens interesting scenarios with heterogeneous hardware.

### 8.1 CPU as Decode Node — Limited Scenario

Server CPUs (e.g. dual Xeon with high DDR4/DDR5 capacity) have large RAM capacity but significantly lower bandwidth than HBM: DDR4 at ~68 GB/s per socket vs HBM3 at ~3 TB/s on H100. Memory latency is also higher, and the massive parallelism of GPUs is missing. CPU decode quickly becomes limited not only by bandwidth, but also by the lack of vector parallelism comparable to GPUs, making it difficult to sustain high throughput even with sufficient memory.

This makes the CPU a decode candidate **only in specific scenarios:**

- Small/medium models (where model weight read does not dominate)
- Low batch size (few concurrent requests)
- Long contexts where the advantage is RAM capacity (KV cache that doesn't fit in VRAM)

A hybrid architecture in these scenarios could provide:

- **Prefill on GPU:** high compute density, massive parallelism, optimal TTFT.
- **Decode on CPU:** large memory capacity for extended KV cache, lower hardware cost. Functional for small models with low batch.

### 8.2 Dielabs Context

The Dielabs lab has:

- **GPU node** (RTX 4070 Super, 12GB VRAM): natural candidate as prefill node for small/medium models.
- **R730** (dual Xeon E5-2680v4, 128GB DDR4): candidate as decode node, with ample RAM for KV cache and DDR4 bandwidth at ~68 GB/s per socket.

The bottleneck in this configuration would be the KV cache transfer from the GPU node to the R730 over the local network. With a 1GbE network the transfer would be prohibitive; with 10GbE it becomes feasible for small models. For experimentation, a pragmatic approach would be to measure the cost of KV cache serialization/transfer against the gain from hardware specialization.

This scenario represents a "homelab" disaggregated serving on recycled enterprise hardware — an area with very little public documentation.

---

## 9. Open Challenges

- **KV Cache Transfer — the dominant problem:** KV cache transfer between nodes is not just a bottleneck — it is the factor that can completely cancel the disaggregation advantage. KV cache size scales as O(seq_len × num_layers × hidden_dim): for a 32B parameter model with 8K context, KV cache can reach several GB per single request. Without ultra-high bandwidth interconnects (≥100 Gbps), transfer time exceeds the time saved by hardware specialization. Solutions under exploration: KV cache compression/quantization, pre-allocation on decode nodes, transfer pipelining in parallel with first-token generation, KV pooling (§5).
- **Coordinated Scheduling:** an orchestrator must decide which node performs prefill, which node performs decode, and manage cache transfer. It increases system complexity. Detail in §12.
- **Fault Tolerance:** if a decode node fails during generation, the KV cache must be rebuilt (repeating prefill). Physical separation increases failure modes.
- **Convenience Threshold:** for small models and short prompts, coordination overhead can exceed the separation advantage. Disaggregated serving makes more sense at scale (large models, high throughput, long outputs).

---

## 10. From Manual Disaggregation to Production

The manual implementation (vLLM + Python proxy + NixlConnector) demonstrates the principles but lacks the components needed for a production environment:

- **KV-aware routing:** route follow-up requests to nodes that already hold the KV cache, enabling cross-request reuse and avoiding redundant transfers.
- **Dynamic scaling:** add prefill workers when the prompt processing phase is the bottleneck, decode workers when the generation queue grows.
- **Service discovery:** automatic worker registration and deregistration, without static configuration.

Frameworks like NVIDIA AI Dynamo implement these components. However, understanding the underlying protocol is essential before adopting higher-level abstractions: if you don't know the proxy → prefill → metadata → decode flow, you cannot diagnose when something doesn't work or doesn't perform.

---

## 11. Decision Model

### 11.1 When to Use Each Architecture

||**Single Node**|**Replicated**|**Disaggregated**|
|---|---|---|---|
|**Proxy complexity**|None|Stateless round-robin|Stateful, 2-phase protocol|
|**Network requirements**|N/A|TCP sufficient|RDMA for medium/large models; TCP 10-25 Gbps for small models|
|**Latency (1 req)**|Minimal|~+6%|Higher|
|**Throughput (N req)**|Limited to 1 GPU|~2× (linear)|High (pipeline)|
|**Granular scaling**|No|Only whole replicas|Independent prefill/decode|
|**Ideal use case**|Dev, test, low concurrency|Medium load, simplicity|High concurrency, asymmetric workloads|

### 11.2 Conditions Favorable to Disaggregation

|Condition|Why|
|---|---|
|Long input sequences|Heavy prefill → benefits from dedicated compute nodes|
|Long output|Lots of time in decode → amortizes setup and transfer cost|
|High concurrency|Prefill/decode interference in the batch becomes the dominant bottleneck|
|Fast network (≥100 Gbps RDMA)|KV cache transfer does not cancel the advantage|
|Large models (>30B)|Prefill cost is significant, separation has real impact|
|Asymmetric workloads|Long prompts with short replies (or vice versa), where one phase dominates the other|
|Recurring shared prefixes|System prompt, RAG → KV pooling (§5) amplifies the gain|

### 11.3 Unfavorable Conditions

|Condition|Why|
|---|---|
|Small models (<7B)|Prefill already fast, little to gain from separation|
|Short prompts + short outputs|Too little time in either phase to amortize overhead|
|Low concurrency|No interference to eliminate|
|Slow network (<10 Gbps)|KV cache transfer becomes the new bottleneck|
|Single-node deployment|Coordination overhead exceeds the benefit|

### 11.4 Gray Area (To Validate with Benchmarks)

|Condition|Approach|
|---|---|
|Medium models (7B-30B)|Measure if KV transfer time < time saved|
|Medium concurrency (10-50)|Depends on prefill/decode ratio in the workload|
|10-25 Gbps network|Feasible with KV compression, validate empirically|
|SD + disaggregated|The double SD + transfer overhead requires high acceptance rate to be net positive|

---

## 12. Scheduling, DPU-Side Logic and Roadmap

### 12.1 Current State: CPU-Side Router

Every disaggregated system has a router that decides:

1. Which **prefill** node to send a request to (ideally one that already has part of the prefix in local cache).
2. Which **decode** node to route the resulting session to (ideally the same node that received the just-generated KV cache, to avoid additional transfers).
3. How to handle **follow-up turns** (same decode node as the session, if possible, to reuse the KV cache).

Today this router is almost always **CPU-side**: vLLM router, NVIDIA Dynamo frontend, custom gateways. It works, it's the default, scalability is known.

### 12.2 Direction: DPU-Side Scheduling

The direction — pushed by NVIDIA via DOCA — is to move part of the dispatch logic onto the DPU to:

- **Reduce jitter** on dispatch (host CPU can be contended by other workloads).
- **Free host CPU** for other functions (monitoring, control plane, application).
- **Integrate fabric telemetry** directly into decision making (e.g. choose path and destination based on fabric congestion state at that microsecond).

**Watch the framing:** it's early adopter territory in 2026, not standard practice. Worth flagging in technical conversations as a plausible evolution, not as consolidated reality. Anyone presenting it as deployment-ready today is anticipating the roadmap.

### 12.3 Fact vs Direction — Quick Reference

To distinguish in technical conversations (presales, architecture review):

**Fact today (2026):**

- GPUDirect RDMA on ConnectX
- NVMe-oF for KV-cache pooling (e.g. Mooncake in production)
- Spectrum-X in production deployments (large-scale AI clusters)
- Dynamo as orchestration framework

**Direction (roadmap, early adopter):**

- Networked CXL for memory pooling
- DPU-side scheduling
- Dynamo at full regime with native integration of all components
- Hardware-accelerated KV cache compression

---

## 13. Three Takeaways

Three statements that distinguish an engineer's discourse from a brochure's:

1. **The DPU is an amplifier, not a prerequisite.** GPUDirect RDMA runs on ConnectX. The DPU adds multi-tenant isolation, storage offload (NVMe-oF target in hardware), and — prospectively — scheduling. Presenting it as a necessary condition for disaggregated inference is marketing, not architecture.
2. **Distinguish fact from direction.** Fact today: GPUDirect RDMA, NVMe-oF for KV pool, Spectrum-X, Dynamo. Direction: networked CXL, DPU-side scheduling, memory pooling over fabric.
3. **The KPI is TTFT, not GB/s.** Bandwidth serves to meet end-to-end latency; latency serves to meet perceived SLO. Whoever sells bandwidth without contextualizing the SLO is selling the wrong thing. TTFT and sustained throughput in tokens/s under SLO are the numbers that matter to the end client.

Disaggregated inference is one of the few topics where network infrastructure, storage and GPU compute converge on the same problem. Whoever works in AI infrastructure — pre-sales, solutions architecture, inference engineering — must know how to tell the story knowing where the technical fact ends and the roadmap begins.

---

## 14. References

- Vaswani et al., "Attention Is All You Need" (2017) — transformer architecture and Q/K/V attention mechanism
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023) — vLLM and efficient KV cache management
- Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving" — foundational paper on disaggregation
- Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting" — Microsoft/ASPLOS, disaggregated serving in production
- Qin et al., "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving" — Moonshot/Kimi, KV pooling via NVMe-oF
- Thomas, E. (2026). Replication vs Disaggregation: Benchmarking Two-Node LLM Inference with vLLM and NIXL. LinkedIn / TokenLabs.
- vLLM Project. Disaggregated Prefill documentation.
- NVIDIA. NIXL — Network Interface for Accelerated Libraries.
- NVIDIA. AI Dynamo — Framework for disaggregated inference orchestration.
- NVIDIA. Spectrum-X platform documentation.
- NVIDIA. DOCA SDK — DPU programming framework.
- CPU inference benchmarking on enterprise hardware (TP, DP, NUMA analysis) — covered in detail in the dedicated document on CPU inference.
- Dielabs Inference Engineering Manual — memory wall, KV cache economics, parallelism

---

_Dielabs — AI Inference Engineering Lab_ _dielabs.github.io_
