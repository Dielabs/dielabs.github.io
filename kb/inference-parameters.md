# LLM Parameter Topology — Artifact, Startup, Request

> **Dielabs core principle**
>
> Every LLM parameter lives in exactly one of three places: **Artifact**, **Startup**, or **Request**. If you don't know where a parameter lives, you can't understand when it takes effect or who controls it.

---

## The complete parameter flow

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

This schema represents the real parameter topology in LLM serving systems (vLLM, TGI, TensorRT-LLM, SGLang, Ollama, etc.).

**Convention:** each parameter is labeled `[A]`, `[S]` or `[R]`. Where relevant, `enforced-by-server: yes` is added for parameters that travel in the request but are applied by the server during generation.

---

## 1. [A] Artifact — Model Artifact

Defines the model as an object/version. It's what gets downloaded from HuggingFace or saved as a checkpoint. Contains weights, architectural configuration, tokenizer, metadata and the chat template.

**Changing a parameter at this level means changing the model.**

### Architecture / Model Family
Examples: Llama, Qwen, Mistral, Mixtral. Influences: model behavior, engine compatibility, tokenizer format.

### Parameter Count
Examples: 2B, 7B, 70B. Influences: VRAM required for weights, throughput and logical capability.

### Checkpoint Type
- `base` — raw text completion
- `instruct` / `chat` — aligned for instruction following and dialogue

Determines output style and alignment degree.

### Tokenizer
Tokenization scheme, special tokens, BOS/EOS. Influences: token cost, linguistic behavior, multi-byte character handling.

### Position Encoding / RoPE Variant
Position encoding mechanism in context. Critical for coherence on long contexts and semantic stability.

### Training Context Length
Maximum context seen during training (e.g. 8k, 32k, 128k). The server can force longer contexts, but quality may degrade beyond this limit.

### Quantization Format (Weights)
Weight format: FP16, BF16, INT8, AWQ, GPTQ. Trade-off: memory ↔ speed ↔ quality.

### Chat Template
Dialogue formatting rules (system/user/assistant roles, special tokens, EOS). **Common critical error:** a template mismatch leads to incoherent output and failure to stop on EOS.

---

## 2. [S] Startup — Inference Engine Configuration

Defines how the server runs. These parameters are set at container startup, in the engine process or in the Kubernetes pod, and **remain fixed for the entire life of the instance**.

This level is pure Inference Engineering.

### Memory and capacity

**`max_model_len`**
Maximum context the server can handle: `prompt + output ≤ max_model_len`. Primary ceiling for memory and compatibility. One of the first parameters to calculate during capacity planning.

**`gpu_memory_utilization`** (or equivalent)
Percentage of VRAM the server can use. Directly influences: KV cache size and sustainable concurrency.

**KV Cache management**
Memory used to store intermediate model states (avoids recomputing prefill on each new token). Scales with: `context length × concurrency`. Often the real operational limit of inference systems.

**`kv_cache_dtype`**
KV cache precision: FP16, FP8. FP8 reduces VRAM at equal context/concurrency, but may slightly degrade quality on long outputs.

### Concurrency and scheduling

**`max_num_seqs`**
Maximum number of simultaneous requests in the batch. Influences latency under load and throughput.

**`max_num_batched_tokens`**
Limit of tokens processable in the same batch. Trade-off: TTFT ↔ throughput. On engines like vLLM, this parameter can transform a "slow" server into a high-throughput machine without touching the model or the request.

**Dynamic / Continuous batching**
Mechanism that dynamically joins compatible requests into the same batch. Used to increase GPU throughput and improve efficiency. Requires tuning to avoid worsening TTFT.

**Prefix caching** (if supported)
Reuse of common prefixes (fixed system prompt, RAG templates). Reduces prefill work on repeated patterns.

### Parallelism (multi-GPU)

**Tensor Parallel** — distributes model weights across multiple GPUs.

**Pipeline Parallel** — distributes model layers across different GPUs.

### Offload / Swap (if available)
Ability to move part of memory to CPU RAM / host memory. Helps avoid OOM, but may increase latency. (Example: LMCache on vLLM.)

### Server-side guardrails on requests
The server can impose limits on request parameters: cap on `max_tokens`, stop policy, response format policy. **These limits can override or ignore values sent in [R].**

### Observability / Logging
Metrics, tracing, log level. Does not change output, but essential for debugging, capacity planning and regression detection.

---

## 3. [R] Request — Decoding / Sampling Parameters

Parameters sent with each individual request. Can change on every call. Control how the model selects tokens during generation.

### Sampling

**`temperature`** — generation randomness.
- `0` → deterministic
- `1` → creative/varied

**`top_p`** (nucleus sampling) — considers only tokens within a cumulative probability mass (e.g. 0.9).

**`top_k`** — considers only the K most probable tokens.

**`min_p`** — filters tokens much less probable than the best token. Increases coherence without sacrificing variety.

**`repetition_penalty`** — reduces loops and repetitions in generated text.

**`presence_penalty` / `frequency_penalty`** — push the model toward greater lexical variety and reduce redundancy. Engine-dependent implementation.

**`seed`** (if supported) — enables reproducibility in benchmarks and tests. Not always 100% guaranteed across all engines.

---

## 4. [R] Termination & Response Contract

Parameters that define when to stop and how to return the response. Sent in the request **but applied by the server** during generation.

### Termination & length control

**`max_tokens` / `max_new_tokens`** — maximum generation limit. `enforced-by-server: yes`

**`min_tokens`** — minimum length before allowing stop. `enforced-by-server: yes`

**`stop` / stop sequences** — strings that immediately stop generation when encountered. `enforced-by-server: yes`

**EOS handling** — how to handle the end-of-sequence token (stop / ignore). `enforced-by-server: yes`

### Delivery & metadata

**`stream`** — token-by-token streaming response vs complete response in one block.

**`logprobs`** (if supported) — token probabilities for debugging, routing and qualitative evaluation.

**`response_format` / JSON mode / schema** (if supported) — output format constraints. Often a combination of engine capability and server-side policy.

> **Note:** client-side post-processing also exists (UI that truncates, formats, hides special tokens). This is not [R] of the model: it's interface behavior, outside the scope of this framework.

---

## 5. Control Hierarchy (Enforcement Hierarchy)

The real hierarchy is descending:

```
[S] Startup       ← commands over everything (defines the operational boundary)
      ↓
[A] Artifact      ← defines the physical limits of the model
      ↓
[R] Request       ← the user plays within the boundary built by A and S
```

**Practical example:**
```
[S] server started with max_tokens = 512
[R] request sends max_tokens = 2048
→ Result: 512 (server wins)
```

**[A] defines physical limits** — you can't ask a 2B model to have the same logical coherence as a 70B, regardless of S and R.

---

## 6. Cross-level conflict zones

Many problems don't reside in a single parameter, but in **misalignment between levels**.

### Context conflict [A vs S vs R]
```
[A] model trained for 32k tokens
[S] max_model_len = 8k (to save VRAM)
[R] 10k token prompt

→ Result: 400 error or aggressive truncation,
  despite the model "on paper" supporting 32k
```

### Template mismatch [A vs R]
```
[A] model trained with Llama-3 (<|begin_of_text|>)
[R] client sends ChatML format (<|im_start|>)

→ Result: hallucinations, incoherent output,
  failure to stop on EOS
```

### Precision tradeoff [A vs S]
```
[A] FP16 weights
[S] kv_cache_dtype = FP8 (to double concurrency)

→ Result: slight degradation on long responses,
  invisible in short tests, emerges only under real load
```

---

## 7. Quick troubleshooting

| Symptom | Probable cause | Level |
|---|---|---|
| Model hallucinates or speaks wrong language | Wrong chat template or tokenizer | **[A]** |
| Out of Memory (OOM) at startup | Insufficient VRAM for weights or `gpu_memory_utilization` too high | **[S]** |
| Very high TTFT with many users | Unoptimized batching or saturated KV cache | **[S]** |
| Response truncated mid-sentence | `max_tokens` too low or stop sequence encountered | **[R]** |
| Model keeps repeating the same phrase | `temperature = 0` without `repetition_penalty`, or unhandled loop | **[R]** |
| Malformed JSON returned | `response_format` not supported or misaligned prompt | **[S] / [R]** |
| Performance degrades gradually | Full KV cache, sub-optimal batching | **[S]** |
| Different behavior with same prompt | Changed checkpoint, different template or quantization | **[A]** |

---

## 8. Mental checklist for inference engineering

When analyzing a problem, follow this order:

1. **Is it a model problem?** → check `[A]` Artifact (checkpoint, template, tokenizer, quantization)
2. **Is it a performance or memory problem?** → check `[S]` Startup (max_model_len, KV cache, batching, concurrency limits)
3. **Is it a generative behavior problem?** → check `[R]` Request (temperature, sampling, stop, max_tokens)

**Quick rule:**
- Quality/style changes "with same prompt" → **[A]**
- Latency/throughput/concurrency changes → **[S]**
- Creativity/variance/loop changes → **[R]** sampling
- Stops strangely, truncates, broken format → **[R]** output + **[S]** guardrails

---

## Supported engines

This framework applies to any LLM serving stack:
- vLLM
- TGI (Text Generation Inference)
- TensorRT-LLM
- SGLang
- Ollama
- Any OpenAI API-compatible engine

---

*Dielabs KB — LLM Parameter Topology v2.0*
