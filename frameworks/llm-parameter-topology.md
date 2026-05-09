---
layout: default
title: "LLM Parameter Topology"
---

# LLM Parameter Topology — Artifact, Startup, Request

> **Dielabs core principle**
>
> Every LLM parameter lives in exactly one of three places: **Artifact**, **Startup**, or **Request**. If you don't know where a parameter lives, you can't understand when it takes effect or who controls it.

---

## The Complete Parameter Flow

The real path of a request through an inference system:

```
[A] Model Artifact
        ↓
[S] Server Startup Configuration
        ↓
[R] Incoming Request Parameters
        ↓
    Server Enforcement & Generation
        ↓
      Response
```

This schema represents the real parameter topology in LLM systems (vLLM, TGI, TensorRT-LLM, SGLang, Ollama, etc.).

**Convention:** each parameter is labeled `[A]`, `[S]` or `[R]`. Where relevant, `enforced-by-server: yes` is added for parameters that travel in the request but are applied by the server during generation.

---

## 1. [A] Artifact — Model Artifact

Defines the model as an object/version. This is what gets downloaded from HuggingFace or saved as a checkpoint. Contains weights, architectural configuration, tokenizer, metadata and any chat template.

**Changing a parameter at this level means changing the model.**

### Architecture / Model Family
Examples: Llama, Qwen, Mistral, Mixtral. Influences: model behavior, engine compatibility, tokenizer format.

### Parameter Count
Examples: 2B, 7B, 70B. Influences: VRAM required for weights, throughput and logical capacity.

### Checkpoint Type
- `base` — raw text completion
- `instruct` / `chat` — aligned for instruction following and dialogue

Determines output style and degree of alignment.

### Tokenizer
Tokenization scheme, special tokens, BOS/EOS. Influences: token cost, linguistic behavior, multi-byte character handling.

### Position Encoding / RoPE Variant
Position encoding mechanism in the context. Critical for coherence on long contexts and semantic stability.

### Training Context Length
Maximum context seen during training (e.g. 8k, 32k, 128k). The server can force longer contexts, but quality may degrade beyond this limit.

### Quantization Format (Weights)
Weight format: FP16, BF16, INT8, AWQ, GPTQ. Trade-off: memory ↔ speed ↔ quality.

### Chat Template
Dialogue formatting rules (system/user/assistant roles, special tokens, EOS). **Common critical error:** a template mismatch leads to incoherent output and failed EOS stopping.

---

## 2. [S] Startup — Inference Engine Configuration

Defines how the server runs. These parameters are set at container startup, in the engine process or Kubernetes pod, and **remain fixed for the entire instance lifetime**.

This level is pure Inference Engineering.

### Memory and Capacity

**`max_model_len`** Maximum context manageable by the server: `prompt + output ≤ max_model_len`. Primary ceiling for memory and compatibility. One of the first parameters to calculate during capacity planning. Lowering it protects sustainable throughput: each sequence consumes fewer KV cache blocks, freeing capacity for more concurrent sequences. Set once based on workload profile and left fixed — unlike `max_num_seqs`, it is not a runtime tuning lever.

**`gpu_memory_utilization`** (or equivalent) Percentage of VRAM the server can use. Directly influences: KV cache size and sustainable concurrency.

**KV Cache management** Memory used to save intermediate model states (avoids recalculating prefill for each new token). Scales with: `context length × concurrency`. Often the real operational limit of inference systems.

**`kv_cache_dtype`** KV cache precision: FP16, FP8. FP8 reduces VRAM for the same context/concurrency, but may slightly degrade quality on long outputs.

### Concurrency and Scheduling

**`max_num_seqs`** Maximum number of sequences the continuous batching scheduler can keep active (resident) simultaneously in the engine. This is the primary lever to balance latency and throughput.

Lowering it protects latency: fewer active sequences mean less pressure on the KV cache and less GPU contention during decode steps, but aggregate throughput drops. Raising it increases throughput at the cost of latency; beyond a certain threshold VRAM saturates, forcing preemption or sequence swap with non-linear degradations.

The Crossover Point (Cp) — the concurrency at which the system transitions from latency-first to throughput-first behavior — guides tuning: below Cp → latency-first, above Cp → throughput-first. The correspondence between external concurrency and active sequences in the scheduler is not 1:1 (it depends on the input/output length distribution), so the value must be validated empirically against the target workload.

**`max_num_batched_tokens`** Token limit processable in the same batch. Trade-off: TTFT ↔ throughput. On engines like vLLM, this parameter can transform a "slow" server into a powerhouse without touching the model or the request.

**Dynamic / Continuous batching** Mechanism that dynamically combines compatible requests in the same batch. Serves to increase GPU throughput and improve efficiency. Requires tuning to not worsen TTFT.

**Prefix caching** (if supported) Reuse of common prefixes (fixed system prompt, RAG templates). Reduces prefill work on repeated patterns.

### Parallelism (multi-GPU)

**Tensor Parallel** Distributes model weights across multiple GPUs.

**Pipeline Parallel** Distributes model layers across different GPUs.

### Offload / Swap (if available)
Ability to move part of memory to CPU RAM / host memory. Helps avoid OOM, but can increase latency. (Example: LMCache on vLLM.)

### Server-side Request Guardrails
The server can impose limits on request parameters: cap on `max_tokens`, stop policy, response format policy. **These limits can ignore or override values sent in [R].**

### Observability / Logging
Metrics, tracing, log level. Does not change output, but is fundamental for debugging, capacity planning and regression detection.

---

## 3. [R] Request — Decoding / Sampling Parameters

Parameters sent with each individual request. Can change with every call. Control how the model chooses tokens during generation.

### Sampling

**`temperature`** Generation randomness. `0` → deterministic, `1` → creative/varied.

**`top_p`** (nucleus sampling) Considers only tokens within a cumulative probability mass (e.g. 0.9).

**`top_k`** Considers only the K most probable tokens.

**`min_p`** Filters tokens much less probable than the best token. Increases coherence without sacrificing variety.

**`repetition_penalty`** Reduces loops and repetitions in generated text.

**`presence_penalty` / `frequency_penalty`** Push the model toward greater lexical variety and reduce redundancy. Engine-dependent implementation.

**`seed`** (if supported) Enables reproducibility in benchmarks and tests. Not always 100% guaranteed across all engines.

---

## 4. [R] Termination & Response Contract

Parameters that define when to stop and how to return the response. Sent in the request **but applied by the server** during generation.

### Termination & Length Control

**`max_tokens` / `max_new_tokens`** Maximum generation limit. `enforced-by-server: yes`

**`min_tokens`** Minimum length before allowing stop. `enforced-by-server: yes`

**`stop` / stop sequences** Strings that immediately interrupt generation if encountered. `enforced-by-server: yes`

**EOS handling** How to handle the end-of-sequence token (stop / ignore). `enforced-by-server: yes`

### Delivery & Metadata

**`stream`** Token-by-token streaming response vs complete response in one block.

**`logprobs`** (if supported) Token probabilities for debugging, routing and qualitative evaluation.

**`response_format` / JSON mode / schema** (if supported) Output format constraints. Often a combination of engine capability and server-side policy.

> **Note:** client-side post-processing (UI that truncates, formats, hides special tokens) also exists. This is not the model's [R]: it is interface behavior, outside this framework's scope.

---

## 5. Control Hierarchy (Enforcement Hierarchy)

The real hierarchy is descending:

```
[S] Startup       ← commands everything (defines the operational fence)
      ↓
[A] Artifact      ← defines the model's physical limits
      ↓
[R] Request       ← the user plays inside the fence built by A and S
```

**Practical example:**

```
[S] server started with max_tokens = 512
[R] request sends max_tokens = 2048
→ Result: 512 (the server wins)
```

**[A] defines physical limits** — you cannot ask a 2B model to have the same logical coherence as a 70B, regardless of S and R.

---

## 6. Conflict Zones Between Levels

Many problems do not reside in a single parameter, but in **misalignment between levels**.

### Context Conflict [A vs S vs R]

```
[A] model trained for 32k tokens
[S] max_model_len = 8k (to save VRAM)
[R] prompt of 10k tokens

→ Result: 400 error or aggressive truncation,
  despite the model "on paper" supporting 32k
```

### Template Mismatch [A vs R]

```
[A] model trained with Llama-3 (<|begin_of_text|>)
[R] client sends ChatML format (<|im_start|>)

→ Result: hallucinations, incoherent output,
  failed EOS stopping
```

### Precision Tradeoff [A vs S]

```
[A] FP16 weights
[S] kv_cache_dtype = FP8 (to double concurrency)

→ Result: slight degradation on long responses,
  invisible in short tests, emerges only under real load
```

---

## 7. Quick Troubleshooting

| Symptom | Probable Cause | Level |
|---|---|---|
| Model hallucinates or speaks in another language | Chat template or tokenizer mismatch | **[A]** |
| Out of Memory (OOM) at startup | Insufficient VRAM for weights or `gpu_memory_utilization` too high | **[S]** |
| Very high TTFT with many users | Unoptimized batching or saturated KV cache | **[S]** |
| Response truncated mid-sentence | `max_tokens` too low or stop sequence encountered | **[R]** |
| Model keeps repeating the same phrase | `temperature = 0` without `repetition_penalty`, or unhandled loop | **[R]** |
| Malformed JSON returned | `response_format` not supported or prompt not aligned | **[S] / [R]** |
| Performance degrades gradually | KV cache full, sub-optimal batching | **[S]** |
| Different behavior with the same prompt | Checkpoint changed, different template or quantization | **[A]** |

---

## 8. Mental Checklist for Inference Engineering

When analyzing a problem, follow this order:

1. **Is it a model problem?** → check `[A]` Artifact (checkpoint, template, tokenizer, quantization)
2. **Is it a performance or memory problem?** → check `[S]` Startup (max_model_len, KV cache, batching, concurrency limits)
3. **Is it a generative behavior problem?** → check `[R]` Request (temperature, sampling, stop, max_tokens)

**Quick rule:**

- Quality/style changes "at same prompt" → **[A]**
- Latency/throughput/concurrency changes → **[S]**
- Creativity/variance/loop changes → **[R]** sampling
- Stops "weird", cuts, broken format → **[R]** output + guardrail **[S]**

---

## Supported Engines

This framework applies to any LLM serving stack:

- vLLM
- TGI (Text Generation Inference)
- TensorRT-LLM
- SGLang
- Ollama
- any OpenAI API-compatible engine

---

*Dielabs KB — LLM Parameter Topology v2.0*
