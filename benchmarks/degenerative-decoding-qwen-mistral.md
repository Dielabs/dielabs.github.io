---
layout: default
title: "Benchmark: Degenerative Decoding Qwen vs Mistral"
---

# Narrative Loop and Degenerative Decoding in vLLM
## Complete Case Study — Parts 1 and 2

_Qwen2.5-7B-Instruct-AWQ · Mistral-7B-Instruct-v0.3-FP8 · RTX 4070 Super 12GB · vLLM_

---

# Part 1 — Qwen2.5-7B: Phenomenon, Diagnosis and Framework

## 1. Infrastructure Context

This case was observed on a homelab node with the following configuration:

- Hardware: NVIDIA RTX 4070 Super 12GB VRAM
- Engine: vLLM (OpenAI-compatible API)
- Model: Qwen/Qwen2.5-7B-Instruct-AWQ
- max_model_len = 32768
- gpu_memory_utilization = 0.85 (~10.2GB allocated)
- Concurrency = 1
- KV cache usage: ~20% throughout all tests

The system remained stable for the entire duration of the tests. No memory saturation, no KV cache spikes, no CUDA errors, no OOM events.

## 2. Observed Phenomenon

During a single long generation, the output began repeating itself. At a certain point the narrative entered a loop: phrases and structures replicated cyclically, with no technical error detected.

The phenomenon is NOT caused by:
- KV cache saturation
- GPU or CPU memory pressure
- Compute overload
- vLLM scheduler
- CUDA instability

It is a degenerative decoding phenomenon — a statistical behavior of the model, not a hardware problem.

## 3. Technical Mechanism

### 3.1 Probabilistic Attractor

An LLM generates tokens by maximizing the conditional probability P(token_t | token_1..t-1). When the text enters a repetitive pattern and no repetition penalty is applied, the model can find a very high-probability sequence and stay there.

The autoregressive cycle closes as follows:
- Generates a highly probable sequence
- The sequence is already in context → remains at high probability
- The model repeats it
- Probability stays high → repeats again

With repetition_penalty = 1.0 (default) there is no mechanism that penalizes tokens already seen in the context. The loop becomes stable and persistent.

### 3.2 Hypothesis: Degradation at Long Sequences

**Observed fact:** degradation emerges in the final portion of long generations, not at the beginning.

**Hypothesis — the cause is not determinable from the tests conducted.** Plausible explanations, not mutually exclusive:
- *Sampling entropy collapse:* as the sequence grows longer, the probability distribution narrows and previously generated tokens dominate (see §5.6)
- *Training distribution effects:* the model has seen fewer examples of that text type at that length
- *Rigid prompt structure:* formal constraints narrow the space of valid tokens, accelerating distribution collapse
- *Attention behavior on long sequences:* structural hypothesis, not verifiable from external observation of the output

Determining which factor is dominant would require controlled tests with isolated variables — for example: same prompt without rigid structure, token-level probability distribution analysis, context length ablation.

### 3.3 Language Drift as Side Effect

**Observed fact:** with repetition_penalty > 1.0 and no explicit language constraint, the output drifted toward English beyond ~5000 tokens. Adding the constraint in the system prompt eliminated the drift.

**Hypothesis on the cause:** when plausible alternatives in the current language are exhausted under the penalty's effect, the model samples from alternative distributions. Plausible explanations for why English emerges as an alternative include pretraining corpus distribution, tokenizer efficiency for Italian, instruction tuning balance, and effective sampling temperature in that region of the context. No single factor can be isolated from the tests conducted.

**Practical implication — independent of the cause:** repetition_penalty is not a neutral parameter. On multilingual models it acts on the overall language distribution, not just on lexical repetition.

## 4. Test Results

### 4.1 Comparative Test: penalty 1.0 vs 1.2 (target 75 paragraphs)

Fixed parameters: default temperature (1.0), default top_p (0.9), no system prompt.

| Parameter | Test 1 — penalty 1.0 | Test 3 — penalty 1.2 |
|---|---|---|
| Loop onset | Paragraph 16 | No loop |
| Completion | No (infinite loop) | Yes — Paragraph 75 reached |
| Language coherence | IT stable (until loop) | IT → EN switch from ~§66 |
| Side effects | Identical token-for-token infinite loop | Language drift in tail |
| Modified parameters | None (all defaults) | Only repetition_penalty |

### 4.2 Stress Test: 150 paragraphs (penalty 1.2 + IT system prompt)

Added system prompt: "Reply exclusively in Italian. Do not use other languages." The language constraint eliminated the IT→EN drift, but tail degradation did not disappear — it manifested differently.

| Zone | Paragraphs | Observed phenomenon |
|---|---|---|
| Phase 1 — Stable | 1–63 | Coherent IT, progressive narrative |
| Phase 2 — Language drift | 64–83 | Progressive IT→EN switch |
| Phase 3 — Collapse | 84–86 | Intra-paragraph loop, token flooding, complete degeneration |

Critical finding: language degradation begins when the generated context reaches approximately 5000–6000 tokens, regardless of the target declared in the prompt and well below the technical context window limit (32768 tokens). A practical coherence threshold therefore exists for this model on this task type, separate from and independent of the technical context window.

## 5. Interpretive Framework: Three Distinct Layers

The analysis revealed that output quality and infrastructure state are completely separate layers. A system can be perfectly stable — KV cache at 20%, GPU quiet, scheduler regular — and still produce degenerative output. The most common diagnostic mistake is looking for the cause at the wrong layer.

The three-layer framework helps quickly identify the correct level to investigate before intervening.

### 5.1 System Layer

Covers everything measurable at the hardware and orchestration level. It is the most visible layer: Grafana covers it entirely and problems manifest with clear signals — OOM, latency spikes, saturated KV cache, GPU at 100%.

**Diagnostic question:** is there pressure on physical resources or orchestration?

**Metrics:** GPU utilization, VRAM used, KV cache %, latency per token, throughput, scheduler queue depth, CUDA errors.

**Typical solutions:** reduce concurrency, increase VRAM, reduce max_model_len, tune gpu_memory_utilization, update CUDA drivers.

_In the documented case: KV cache at 20%, GPU stable, no errors. The System Layer was clean. The cause was elsewhere._

### 5.2 Model / Decoding Layer

Covers the statistical behavior of the model during generation. Not visible on Grafana: requires observing the output directly and knowing the active sampling parameters. Problems here produce no technical errors — they produce low-quality output: loops, incoherence, language drift, hallucination.

**Diagnostic question:** is the anomalous behavior reproducible with a stable system? Are sampling parameters correctly configured for the task?

**Metrics:** repetition rate in output, narrative coherence, lexical diversity, entropy of generated tokens, output language.

**Typical solutions:** calibrate repetition_penalty, temperature, top_p; add stop sequences; set max_tokens; strengthen system prompt with language constraints and format instructions.

_In the documented case: repetition_penalty = 1.0 (default) allowed the loop at paragraph 16. Raising it to 1.2 eliminated the loop. The problem was entirely at this layer._

### 5.3 Architecture / Context Layer

The most subtle layer: covers model behavior as a function of generated context length. Problems here produce no technical errors and are not detectable from sampling parameters — they emerge only with long outputs, on a stable system with correct parameters.

**Diagnostic question:** does degradation appear only beyond a certain output length, even with a stable system and correct parameters?

**Metrics:** number of generated tokens at degradation onset, position in context window, type of degradation (syntactic compression, language drift, intra-paragraph collapse).

**Typical solutions:** set max_tokens hard limit below the empirically measured degradation threshold; split the task into multiple shorter requests; use models with higher training length for tasks requiring very long outputs.

_In the documented case: with 150 paragraphs and correct parameters, degradation emerged around 5000–6000 generated tokens — well below the technical limit of 32768. The Architecture/Context Layer defined the practical limit of the task._

### 5.4 Three-Layer Summary Table

| System Layer | Model/Decoding Layer | Architecture/Context Layer |
|---|---|---|
| GPU utilization | Repetition rate | Position in context |
| KV cache % | Token entropy | Behavior on long sequences |
| vLLM scheduler | Narrative divergence | Type of observed degradation |
| Latency / throughput | Sampling parameters | Actual generated length |

### 5.5 The Causal Chain: Layers Masking Each Other

The three layers are not independent problems: they can be overlapping and sequential. A problem at a higher layer can completely mask a problem at a lower layer, preventing its observation.

**Phase 1 — Model Layer problem visible, Architecture Layer hidden.** With repetition_penalty = 1.0, the loop appeared at paragraph 16 — approximately 800–1000 generated tokens. The test never reached enough length to stress the Architecture Layer. The third layer already existed but was masked by the second.

**Phase 2 — Model Layer resolved, Architecture Layer emerges.** Raising the penalty to 1.2 eliminated the loop. The model generated freely up to 75, then 150 paragraphs. Only at that point did the Architecture Layer limit become observable: degradation at ~5000 tokens, independent of decoding parameters and infrastructure state.

**Practical implication:** after resolving a problem at any layer, do not declare the system fixed without explicitly verifying the layers below. Diagnosis must be sequential and complete.

### 5.6 Hypothesis: Sampling Entropy Collapse

A useful interpretive framework for explaining the observed phenomena is **Sampling Entropy Collapse**: the tendency of the model's probability distribution to progressively narrow as the generated sequence grows longer.

Hypothesized mechanism:

```
long sequence
↓
accumulated context constrains the distribution
↓
distribution becomes increasingly narrow
↓
few tokens dominate at high probability
↓
loop or degeneration
```

This hypothesis is consistent with three test observations:

**Why the penalty resolves the loop.** The repetition_penalty artificially lowers the probability of already-seen tokens, reopening the sampling space. It does not correct the cause of the collapse — it circumnavigates it by forcing diversity.

**Why the problem emerges late.** Entropy does not collapse immediately: a sufficiently long sequence is needed for the distribution to narrow enough to trigger the loop. In the tests the threshold was ~800 tokens (no penalty) and ~5000 tokens (with penalty 1.2).

**Why it depends on the task.** A narrative task with rigid structure and progressive numbering constrains the space of valid sequences more than a Q&A or a technical task. Fewer degrees of freedom = narrower distribution faster = earlier collapse.

This hypothesis was not verified by directly measuring entropy during generation, and it is not the only possible explanation. It remains a useful interpretive framework for practical diagnosis and for designing future tests.

## 6. Implications for Inference Engineering in Production

### 6.1 Decoding Parameters

- repetition_penalty = 1.0 is insufficient for long generations on this model
- The minimum effective threshold is between 1.15 and 1.2 (empirically confirmed)
- At values > 1.15 on Qwen2.5, the risk of lexical degradation in technical texts increases — correctly repeated technical terms get penalized
- For non-English texts, the penalty should be accompanied by an explicit language constraint in the system prompt

### 6.2 Quality Control Strategy for Production

Recommended priority order:
1. First defense: max_tokens hard limit calibrated for the task + explicit stop sequences
2. Second defense: repetition_penalty = 1.2 with language constraint in system prompt
3. Monitoring: quality metrics kept separate from infrastructure metrics

### 6.3 Observed Practical Limit

For Qwen2.5-7B-Instruct-AWQ on long Italian narrative tasks:
- Coherence threshold: ~5000–6000 generated tokens
- Beyond this threshold: language degradation, then intra-paragraph collapse
- Independent of technical context window (32768 tokens)
- Independent of infrastructure state (KV cache, GPU, scheduler)

## 7. The Coherence Threshold Is a Property of the Task, Not the Model

The ~5000–6000 token threshold observed is not a fixed architectural limit of the model. It is an emergent property of the specific model–task–language combination tested: Italian prose narrative with rigid numbered structure.

### 7.1 Factors Influencing the Threshold

**Text type:** prose narrative requires causal and emotional continuity. Structured technical texts (documentation, Q&A, commented code) have more semantically independent sections — context pressure is lower and the threshold may be higher.

**Language:** the tests showed that language influences behavior — the IT→EN drift is an observed fact. Plausible causes include pretraining corpus distribution, tokenizer efficiency for Italian, and instruction tuning balance. No single factor can be isolated from the tests conducted.

**Formal prompt constraints:** the prompt used in the tests imposed rigid structure (progressive numbering, minimum length per paragraph). Multiple formal constraints narrow the space of valid sequences, increasing the probability that the model finds a dominant attractor earlier. More open prompts tend to delay collapse.

### 7.2 Methodological Implication

No universal threshold valid for all tasks exists. In a production context, the coherence threshold must be measured empirically for each relevant combination of model, task, and language. The correct procedure: define the target task and language, run stress tests with progressively longer outputs, empirically identify the degradation threshold, set max_tokens hard limit below that threshold with an adequate safety margin.

---

# Part 2 — Comparison with Mistral-7B-FP8: Deployment and Benchmark

## 8. Context and Motivation

In Case Study 1, a practical coherence threshold of ~5000–6000 generated tokens was documented for Qwen2.5-7B, independent of the technical context window and infrastructure state.

The logical next step is to verify whether that threshold is specific to Qwen2.5-7B or a general behavior of models at that parameter scale on that type of task. To do this, the same tests must be run on a different model, with equal hardware and decoding parameters.

**Objective:** determine whether a model with different characteristics produces a significantly different coherence threshold.

## 9. Model Selection

### 9.1 Selection Criteria

For a valid comparison the model must meet three criteria: same parameter size (7B), compatibility with the available hardware (12GB VRAM), and characteristics different enough from Qwen2.5 to make the comparison informative.

The selected model is `neuralmagic/Mistral-7B-Instruct-v0.3-FP8`, produced by NeuralMagic and specifically optimized for vLLM.

### 9.2 Technical Comparison

| Parameter | Qwen2.5-7B-Instruct-AWQ | Mistral-7B-Instruct-v0.3-FP8 |
|---|---|---|
| Quantization | AWQ 4-bit | FP8 8-bit |
| Disk size | ~4 GB | ~7 GB |
| VRAM usage | ~4 GB | ~7 GB |
| Remaining VRAM for KV cache | ~6 GB | ~2.4 GB |
| Supported max_model_len | 32768 tokens | 19000 tokens (VRAM-constrained) |
| Quality vs full precision | Greater degradation | 65.85 vs 66.33 benchmark — minimal |
| CUDA kernel | AWQ kernel | CutlassFP8ScaledMMLinearKernel |
| Architecture family | Qwen2ForCausalLM | MistralForCausalLM |

The main trade-off is quality vs VRAM footprint. FP8 better preserves the quality of the original model (minimal benchmark loss: 0.48 points) but occupies almost double the VRAM compared to AWQ 4-bit, reducing the servable context window on the same GPU.

### 9.3 Why FP8 is the Middle Ground

On 12GB VRAM the choice was between three options: AWQ 4-bit (~4GB), full precision BF16 (~14GB, incompatible), and FP8 (~7GB, compatible with margin). FP8 is the format recommended by NeuralMagic for production deployment with vLLM.

## 10. Troubleshooting: From First Error to Successful Startup

### 10.1 Event Sequence

| Step | Event | Corrective action |
|---|---|---|
| 1 | Initial configuration with max_model_len = 32768 | — |
| 2 | Model download: 416 seconds (~7GB from HuggingFace) | Expected — first startup downloads from HF Hub |
| 3 | ERROR: ValueError — insufficient KV cache. Required 4.0 GiB, available 2.41 GiB for max_model_len 32768 | Reduce max_model_len. vLLM suggests estimated maximum: 19696 tokens |
| 4 | Updated docker-compose: max_model_len = 19000 (safety margin) | Container restart with new config |
| 5 | Second startup: model already cached, loaded in 16 seconds | Successful startup — Application startup complete |

### 10.2 Error Analysis

The error at first startup demonstrates how the quantization format directly impacts the VRAM available for the KV cache — and therefore the servable context window.

The causal chain: FP8 occupies 7GB VRAM → 2.41GB remaining → serving max_model_len 32768 requires 4.0GB KV cache → 2.41 < 4.0 → error.

vLLM autonomously provided the estimated maximum (19696 tokens), making the correction immediate.

**Note:** this is a concrete example of interaction between System Layer and Architecture Layer. The choice of quantization format (System Layer) determines the VRAM available for the KV cache, which in turn constrains the maximum context window (Architecture Layer). The layers are never completely independent.

## 11. Detailed Analysis of Startup Logs

vLLM startup logs contain dense diagnostic information. Below, each relevant entry with its technical meaning.

| Log entry | Technical meaning |
|---|---|
| version 0.15.1 | vLLM V1 engine active — internally rewritten architecture, more efficient for scheduling and memory than previous versions. |
| Resolved architecture: MistralForCausalLM | vLLM identified the model family. Determines which CUDA kernel is selected and how attention operations are optimized. |
| quantization=fp8 / Selected CutlassFP8ScaledMMLinearKernel | FP8 requires specific kernels for matrix multiplications. CutlassFP8 is the NVIDIA-optimized kernel for Ada Lovelace architectures (RTX 4070 Super). |
| Using FLASH_ATTN attention backend | FlashAttention automatically selected among 4 alternatives. Reduces memory usage during attention and increases speed — optimal choice for RTX 4070 Super. |
| Model loading took 7.01 GiB and 15.976617 seconds | 15 seconds because the model was already in local cache. At first startup with download it took 429 seconds. |
| torch.compile takes 9.61 s in total | vLLM compiles the computational graph with PyTorch Inductor. Result cached — subsequent runs use the cache. |
| Available KV cache memory: 2.41 GiB / GPU KV cache size: 19,696 tokens | With 2.41 GiB remaining, vLLM can manage at most 19,696 total KV cache tokens. This is the physical constraint that forced the max_model_len reduction. |
| Maximum concurrency: 1.04x for 19,000 tokens | With 19k-token requests, the KV cache supports just over one simultaneous request. |
| Capturing CUDA graphs: 51/51 + 35/35 | vLLM captures 86 CUDA graphs for batch sizes from 1 to 512 tokens. Eliminates CUDA scheduling overhead — improves decode latency. Cost: 0.47 GiB additional VRAM. |
| Chunked prefill enabled, max_num_batched_tokens=2048 | Prefill is split into 2048-token chunks. Prevents long prompts from monopolizing the GPU while blocking ongoing decode requests. |
| enable_prefix_caching=True | KV cache reuse for identical prefixes across requests. With a fixed system prompt in the tests, each new chat benefits from this. |
| Application startup complete. | Engine ready to accept requests. This is the target line to look for in logs to confirm successful startup. |

## 12. docker-compose Configuration

### 12.1 Explicitly Declared Parameters

| Parameter | Value | Meaning |
|---|---|---|
| --model | neuralmagic/Mistral-7B-Instruct-v0.3-FP8 | HuggingFace model identifier. |
| --max-model-len | 19000 | Maximum servable context window — reduced due to VRAM constraint. |
| --gpu-memory-utilization | 0.85 | 85% of 12GB = 10.2GB allocated to vLLM. |
| --host / --port | 0.0.0.0 / 8000 | Network binding to reach the service from the host. |

```yaml
version: "3.9"
services:
  vllm:
    image: vllm/vllm-openai:latest
    container_name: vllm
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - vllm_models:/root/.cache/huggingface
    command:
      - "--model"
      - "neuralmagic/Mistral-7B-Instruct-v0.3-FP8"
      - "--host"
      - "0.0.0.0"
      - "--port"
      - "8000"
      - "--max-model-len"
      - "19000"
      - "--gpu-memory-utilization"
      - "0.85"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ipc: host
```

### 12.2 Features Automatically Activated by vLLM

Everything not in the docker-compose is automatically configured by vLLM at runtime, based on the detected model and hardware.

| Feature | How and why it is automatically activated |
|---|---|
| FlashAttention backend | vLLM detects the GPU and selects the most efficient attention backend. On RTX 4070 Super (Ada Lovelace), FlashAttention is always the optimal choice. |
| CutlassFP8 kernel | FP8 quantization detected, vLLM automatically selects the optimized CUDA kernel. |
| Chunked prefill | Active by default in vLLM 0.15.x with V1 engine. |
| Prefix caching | Active by default. Particularly useful with a fixed system prompt. |
| torch.compile + Inductor | V1 engine compiles the computational graph at first startup and caches the result. |
| CUDA graphs (86 graphs) | Automatically captured for batch sizes from 1 to 512. Reduce decode latency. |
| Asynchronous scheduling | Active by default in V1 engine. Decouples scheduling from GPU execution. |

**Key principle:** the docker-compose is the minimum contract with vLLM. Everything else is automatic optimization. Adding unnecessary parameters introduces conflict risk and reduces readability without concrete benefit.

## 13. Comparative Test Results

Tests were run with the same parameters as Part 1: repetition_penalty = 1.2, numbered narrative prompt for 75 paragraphs, Italian and French.

### 13.1 Test A — 75-Paragraph Italian Narrative

Collapse occurred at paragraph 3–4, approximately 200–300 generated tokens. Paragraph 1 was comprehensible Italian with syntactic anomalies at the end. From Paragraph 3: fragmented Italian, invented neologisms, token flooding, insertion of Chinese characters. By Paragraph 4 the text was completely unreadable.

Direct comparison: Qwen2.5-7B with the same penalty had maintained coherence up to paragraph 63 (~5000 tokens). The ratio is approximately **25:1** in favor of Qwen on this task.

### 13.2 Test B — 75-Paragraph French Narrative

The hypothesis was that Mistral, developed by a French company, would perform significantly better in French. Result: collapse at §4 instead of §3 — one paragraph more. The difference is minimal and insufficient to draw conclusions about language as a causal factor.

### 13.3 Test C — Short Technical Task in Italian (~500 words)

Clearly positive result: fluent Italian, coherent logical structure, correct coverage of key concepts. Some minor terminological imprecisions, acceptable for a 7B on a divulgative task. Zero degeneration, zero language drift.

This demonstrates that the collapse observed in narrative tests is not a general limitation of Mistral — it is specific to the combination model + long output + rigid structure.

### 13.4 Summary Table

| Model | Task | Penalty | Collapse | Outcome |
|---|---|---|---|---|
| Qwen2.5-7B-AWQ | IT narrative 75 §  | 1.2 | §63 (~5000 tok) | Complete, degrades at tail |
| Mistral-7B-FP8 | IT narrative 75 § | 1.2 | §3–4 (~200 tok) | Immediate collapse |
| Mistral-7B-FP8 | FR narrative 75 § | 1.2 | §4 (~300 tok) | Immediate collapse |
| Mistral-7B-FP8 | Short technical IT ~500 words | 1.2 | None (~400 tok) | Coherent, acceptable quality |

## 14. Comparison Conclusions

**The threshold is model-specific — observed fact.** On this task, Qwen2.5-7B maintained coherence for ~5000 tokens. Mistral-7B collapsed within 200–300 tokens. Ratio 25:1. Same parameter size, same hardware, same decoding parameters — completely different results.

The causes of this difference are not isolable from the tests conducted. Candidate factors include: dataset and instruction tuning, sampling sensitivity, quantization differences (AWQ vs FP8), tokenizer. The test demonstrates the difference, not the cause.

**Language as a factor — insufficient data.** French delayed collapse by one paragraph compared to Italian. The difference is minimal and does not allow concluding that language is a relevant causal factor in this comparison.

**The threshold is task-specific — observed fact.** Mistral on a short technical task (~400 tokens) produces coherent output of acceptable quality. The collapse observed in narrative tests is not a general model limitation — it is specific to the combination model + long output + rigid structure.

**Deployment implication:** model selection must explicitly consider task type and expected output length. An English benchmark on generic tasks is insufficient to predict behavior on specific tasks in different languages. The coherence threshold must be measured empirically for each relevant model-task-language combination.

---

*Produced as part of the AI Infrastructure laboratory — Dielabs*

