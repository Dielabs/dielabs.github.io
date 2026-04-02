---
layout: default
title: "When Good GPUs Produce Bad Tokens"
---

# When Good GPUs Produce Bad Tokens

#### Narrative Loops and Degenerative Decoding in vLLM
_Qwen2.5-7B-Instruct-AWQ · Mistral-7B-Instruct-v0.3-FP8 · RTX 4070 Super 12GB · vLLM_

---

# Part 1 — Qwen2.5-7B: Phenomenon, Diagnosis and Framework

## 1. Infrastructure Context

The case was observed on a homelab node with the following characteristics:

- Hardware: NVIDIA RTX 4070 Super 12GB VRAM
- Engine: vLLM (OpenAI-compatible API)
- Model: Qwen/Qwen2.5-7B-Instruct-AWQ
- max_model_len = 32768
- gpu_memory_utilization = 0.85 (~10.2GB allocated)
- Concurrency = 1
- KV cache utilization: ~20% during tests

The system was stable throughout the entire test duration. No memory saturation, no KV cache spikes, no CUDA errors, no OOM.

## 2. Observed Phenomenon

During a single long generation, the text began repeating itself. After a certain point the narrative entered a loop: phrases and structures replicated cyclically, with no technical error detected.

The phenomenon is NOT related to:
- KV cache saturation
- GPU or CPU memory pressure
- Compute overload
- vLLM scheduler
- CUDA instability

It is a degenerative decoding phenomenon — a statistical behavior of the model, not a hardware issue.

## 3. Technical Mechanism

### 3.1 Probabilistic Attractor

An LLM generates tokens by maximizing the conditional probability P(token_t | token_1..t-1). When the text enters a repetitive pattern and there is no repetition penalty, the model can find a very high-probability sequence and remain trapped in it.

The autoregressive cycle closes as follows:
- Generates a very high-probability sequence
- The sequence is already in the context → remains at high probability
- The model repeats it
- The probability stays high → it repeats again

With repetition_penalty = 1.0 (default) there is no mechanism to penalize tokens already seen in the context. The loop becomes stable and persistent.

### 3.2 Hypothesis on Long-Sequence Degradation

**Observed data:** degradation emerges in the final portion of long generations, not at the beginning.

**Hypothesis — the cause cannot be determined from the tests conducted.** Plausible explanations, not mutually exclusive:
- *Sampling entropy collapse:* as the sequence lengthens, the probability distribution narrows and previously seen tokens dominate (see §5.6)
- *Training distribution effects:* the model has seen fewer examples of that type of text at that length
- *Rigid prompt structure:* formal constraints narrow the space of valid tokens, accelerating the distribution collapse
- *Attention behavior on long sequences:* structural hypothesis, not verifiable from external observation of the output

Determining which factor is dominant would require controlled tests with isolated variables — for example: same prompt without rigid structure, token-by-token probability distribution analysis, context length ablation.

### 3.3 Linguistic Drift as a Side Effect

**Observed data:** with repetition_penalty > 1.0 and no explicit language constraint, the text drifted toward English beyond ~5000 tokens. Adding the constraint in the system prompt eliminated the drift.

**Hypothesis on the cause:** when plausible alternatives in the current language are exhausted under the penalty effect, the model samples from alternative distributions. Plausible causes for why English emerges as an alternative include the pretraining corpus distribution, tokenizer efficiency for Italian, instruction tuning balance, and effective sampling temperature in that context region. It is not possible to isolate a single factor from the tests conducted.

**Practical implication — regardless of the cause:** repetition_penalty is not a neutral parameter. On multilingual models it acts on the overall linguistic distribution, not just on lexical repetition.

## 4. Test Results

### 4.1 Comparative Test: penalty 1.0 vs 1.2 (target 75 paragraphs)

Fixed parameters: default temperature (1.0), default top_p (0.9), no system prompt.

| Parameter | Test 1 — penalty 1.0 | Test 3 — penalty 1.2 |
|---|---|---|
| Loop onset | Paragraph 16 | No loop |
| Completion | No (infinite loop) | Yes — Paragraph 75 reached |
| Linguistic coherence | IT stable (until loop) | IT → EN switch from ~§66 |
| Side effects | Infinite loop, identical token by token | Linguistic drift at the tail |
| Modified parameters | None (all defaults) | Only repetition_penalty |

### 4.2 Stress Test: 150 paragraphs (penalty 1.2 + IT system prompt)

Added system prompt: "Rispondi esclusivamente in italiano. Non usare altre lingue." The linguistic constraint eliminated the IT→EN drift, but tail degradation did not disappear — it manifested differently.

| Zone | Paragraphs | Observed phenomenon |
|---|---|---|
| Phase 1 — Stable | 1–63 | Coherent IT, progressive narrative |
| Phase 2 — Language drift | 64–83 | Progressive IT→EN switch |
| Phase 3 — Collapse | 84–86 | Intra-paragraph loop, token flooding, complete degeneration |

Critical finding: linguistic degradation begins when the generated context reaches approximately 5000–6000 tokens, regardless of the declared target in the prompt and well below the technical context window limit (32768 tokens). There exists therefore a practical coherence threshold for this model on this type of task, separate and independent from the technical context window.

## 5. Interpretive Schema: Three Distinct Planes

The analysis revealed that output quality and infrastructure state are two completely separate planes. A system can be perfectly stable — KV cache at 20%, GPU calm, scheduler regular — and still produce degenerative output. The most common diagnostic error is looking for the cause on the wrong plane.

The three-plane framework serves to quickly identify the correct level where the cause lies, before intervening.

### 5.1 System Plane

Concerns everything measurable at the hardware and orchestration level. It is the most visible plane: Grafana covers it entirely and problems manifest with clear signals — OOM, latency spikes, saturated KV cache, GPU at 100%.

**Diagnostic question:** is there pressure on physical resources or orchestration?

**Metrics:** GPU utilization, VRAM used, KV cache %, latency per token, throughput, scheduler queue depth, CUDA errors.

**Typical solutions:** reduce concurrency, increase VRAM, reduce max_model_len, optimize gpu_memory_utilization, update CUDA drivers.

_In the documented case: KV cache at 20%, GPU stable, no errors. The System Plane was clean. The cause was elsewhere._

### 5.2 Model / Decoding Plane

Concerns the statistical behavior of the model during generation. Not visible on Grafana: requires observing the output directly and knowing the active sampling parameters. Problems here do not produce technical errors — they produce low-quality output: loops, incoherence, linguistic drift, hallucination.

**Diagnostic question:** is the anomalous behavior reproducible with a stable system? Are the sampling parameters configured correctly for the task?

**Metrics:** repetition rate in the output, narrative coherence, lexical diversity, entropy of generated tokens, output language.

**Typical solutions:** calibrate repetition_penalty, temperature, top_p; add stop sequences; set max_tokens; reinforce the system prompt with linguistic constraints and format instructions.

_In the documented case: repetition_penalty = 1.0 (default) allowed the loop at paragraph 16. Raising it to 1.2 eliminated the loop. The problem was entirely on this plane._

### 5.3 Architecture / Context Plane

The subtlest plane: concerns model behavior as a function of generated context length. Problems here produce no technical errors and are not detectable from sampling parameters — they emerge only with long outputs, on a stable system with correct parameters.

**Diagnostic question:** does degradation emerge only beyond a certain output length, even with a stable system and correct parameters?

**Metrics:** number of tokens generated at the moment of degradation, position in the context window, type of degradation (syntactic compression, linguistic drift, intra-paragraph collapse).

**Typical solutions:** set a max_tokens hard limit below the empirically measured degradation threshold; break the task into multiple shorter requests; use models with superior training length for tasks requiring very long outputs.

_In the documented case: with 150 paragraphs and correct parameters, degradation emerged around 5000–6000 generated tokens — well below the technical limit of 32768. The Architecture/Context Plane defined the practical task limit._

### 5.4 Summary Table of the Three Planes

| System Plane | Model/Decoding Plane | Architecture/Context Plane |
|---|---|---|
| GPU utilization | Repetition rate | Position in context |
| KV cache % | Token entropy | Long-sequence behavior |
| vLLM scheduler | Narrative divergence | Type of observed degradation |
| Latency / throughput | Sampling parameters | Effective generated length |

### 5.5 The Causal Chain: Planes Mask Each Other

The three planes are not independent problems: they can be overlapping and sequential. A problem on a higher plane can completely mask a problem on a lower plane, preventing its observation.

**Phase 1 — Model Plane problem visible, Architecture Plane hidden.** With repetition_penalty = 1.0, the loop emerged at paragraph 16 — approximately 800–1000 generated tokens. The test never reached enough length to stress the Architecture Plane. The third layer already existed, but was masked by the second.

**Phase 2 — Model Plane resolved, Architecture Plane emerges.** Raising the penalty to 1.2, the loop disappeared. The model generated freely up to 75, then 150 paragraphs. Only at that point did the Architecture Plane limit become observable: degradation at ~5000 tokens, independent of decoding parameters and infrastructure state.

**Practical implication:** after resolving a problem on any plane, do not declare the system resolved without explicitly verifying the underlying planes. Diagnosis must be sequential and complete.

### 5.6 Hypothesis: Sampling Entropy Collapse

A useful interpretive framework to explain the observed phenomena is **Sampling Entropy Collapse**: the tendency of the model's probability distribution to progressively narrow as the generated sequence lengthens.

The hypothesized mechanism:

```
long sequence
↓
accumulated context constrains the distribution
↓
the distribution becomes increasingly narrow
↓
few tokens dominate with high probability
↓
loop or degeneration
```

This hypothesis is consistent with three test observations:

**Why the penalty resolves the loop.** The repetition_penalty artificially lowers the probability of previously seen tokens, reopening the sampling space. It does not correct the cause of the collapse — it circumnavigates it by forcing diversity.

**Why the problem emerges late.** Entropy does not collapse immediately: a sufficiently long sequence is needed for the distribution to narrow enough to trigger the loop. In the tests the threshold was ~800 tokens (without penalty) and ~5000 tokens (with penalty 1.2).

**Why it depends on the task.** A narrative task with rigid structure and progressive numbering constrains the space of valid sequences more than a Q&A or technical task. Fewer degrees of freedom = narrower distribution faster = earlier collapse.

This hypothesis has not been verified by directly measuring entropy during generation, and is not the only possible explanation. It remains a useful interpretive framework for practical diagnosis and for designing future tests.

## 6. Implications for Production Inference Engineering

### 6.1 Decoding Parameters

- repetition_penalty = 1.0 is insufficient for long generations on this model
- The minimum effective threshold is between 1.15 and 1.2 (empirically confirmed)
- At values > 1.15 on Qwen2.5, the risk of lexical degradation in technical texts increases — correctly repeated technical terms get penalized
- For non-English texts, the penalty should be accompanied by an explicit linguistic constraint in the system prompt

### 6.2 Quality Control Strategy for Production

Recommended priority order:
1. First defense: max_tokens hard limit calibrated on the task + explicit stop sequences
2. Second defense: repetition_penalty = 1.2 with linguistic constraint in the system prompt
3. Monitoring: quality metrics separate from infrastructure metrics

### 6.3 Observed Practical Limit

For Qwen2.5-7B-Instruct-AWQ on long narrative task in Italian:
- Coherence threshold: ~5000–6000 generated tokens
- Beyond this threshold: linguistic degradation, then intra-paragraph collapse
- Independent of the technical context window (32768 tokens)
- Independent of infrastructure state (KV, GPU, scheduler)

## 7. The Coherence Threshold is a Property of the Task, Not the Model

The ~5000–6000 token threshold observed is not a fixed architectural limit of the model. It is an emergent property of the specific model–task–language combination tested: Italian prose narrative with rigid numbered structure.

### 7.1 Factors Influencing the Threshold

**Text type:** prose narrative requires causal and emotional continuity. Structured technical texts (documentation, Q&A, commented code) have more semantically independent sections — context pressure is lower and the threshold may rise.

**Language:** the tests showed that language influences behavior — the IT→EN drift is an observed fact. Plausible causes include pretraining corpus distribution, tokenizer efficiency for Italian, and instruction tuning balance. It is not possible to isolate a single factor from the tests conducted.

**Formal prompt constraints:** the prompt used in tests imposed rigid structure (progressive numbering, minimum paragraph length). Multiple formal constraints narrow the space of valid sequences, increasing the probability that the model finds a dominant attractor sooner. More open prompts tend to postpone the collapse.

### 7.2 Methodological Implication

There is no universal threshold valid for all tasks. In a production context, the coherence threshold must be measured empirically for each relevant combination of model, task and language. The correct procedure: define the target task and language, conduct stress tests with progressively longer outputs, empirically identify the degradation threshold, set a max_tokens hard limit below that threshold with an adequate safety margin.

---

# Part 2 — Comparison with Mistral-7B-FP8: Deployment and Benchmark

## 8. Context and Motivation

In Case Study 1 a practical coherence threshold of ~5000–6000 generated tokens was documented on Qwen2.5-7B, independent of the technical context window and infrastructure state.

The next logical step is to verify whether that threshold is a specific characteristic of Qwen2.5-7B or a general behavior of models at that parameter count on that type of task. To do this, it is necessary to run the same tests on a different model, with identical hardware and decoding parameters.

**Objective:** determine whether a model with different characteristics produces a significantly different coherence threshold.

## 9. Model Selection

### 9.1 Selection Criteria

For a valid comparison the model must satisfy three criteria: same parameter size (7B), compatibility with available hardware (12GB VRAM), and characteristics sufficiently different from Qwen2.5 to make the comparison informative.

The selected model is `neuralmagic/Mistral-7B-Instruct-v0.3-FP8`, produced by NeuralMagic and specifically optimized for vLLM.

### 9.2 Technical Comparison

| Parameter | Qwen2.5-7B-Instruct-AWQ | Mistral-7B-Instruct-v0.3-FP8 |
|---|---|---|
| Quantization | AWQ 4-bit | FP8 8-bit |
| Size on disk | ~4 GB | ~7 GB |
| VRAM occupied | ~4 GB | ~7 GB |
| Residual VRAM for KV cache | ~6 GB | ~2.4 GB |
| Supported max_model_len | 32768 tokens | 19000 tokens (VRAM-constrained) |
| Quality vs full precision | Greater degradation | 65.85 vs 66.33 benchmark — minimal |
| CUDA kernel | AWQ kernel | CutlassFP8ScaledMMLinearKernel |
| Architecture family | Qwen2ForCausalLM | MistralForCausalLM |

The main trade-off is quality vs VRAM footprint. FP8 better preserves the quality of the original model (minimal benchmark loss: 0.48 points) but occupies nearly twice the VRAM compared to AWQ 4-bit, reducing the serviceable context window on the same GPU.

### 9.3 Why FP8 is the Middle Ground

On 12GB VRAM the choice was between three options: AWQ 4-bit (~4GB), full precision BF16 (~14GB, not compatible), and FP8 (~7GB, compatible with margin). FP8 is the format recommended by NeuralMagic for production deployment with vLLM.

## 10. Troubleshooting: From First Error to Successful Startup

### 10.1 Sequence of Events

| Step | Event | Corrective action |
|---|---|---|
| 1 | Initial configuration with max_model_len = 32768 | — |
| 2 | Model download: 416 seconds (~7GB from HuggingFace) | Expected — first startup downloads from HF Hub |
| 3 | ERROR: ValueError — Insufficient KV cache. Required 4.0 GiB, available 2.41 GiB for max_model_len 32768 | Reduce max_model_len. vLLM suggests estimated maximum: 19696 tokens |
| 4 | Updated docker-compose: max_model_len = 19000 (safety margin) | Container restart with new config |
| 5 | Second startup: model already cached, loading in 16 seconds | Successful startup — Application startup complete |

### 10.2 Error Analysis

The error that emerged at first startup demonstrates how the quantization format directly impacts VRAM available for the KV cache — and therefore the serviceable context window.

The causal chain: FP8 occupies 7GB of VRAM → 2.41GB remain free → serving max_model_len 32768 requires 4.0GB of KV cache → 2.41 < 4.0 → error.

vLLM autonomously provided the estimate of the maximum supportable value (19696 tokens), making the correction immediate.

**Note:** this is a concrete example of interaction between the System Plane and the Architecture Plane. The choice of quantization format (System Plane) determines the VRAM available for the KV cache, which in turn constrains the maximum context window (Architecture Plane). The planes are never completely independent.

## 11. Detailed Startup Log Analysis

vLLM startup logs contain dense diagnostic information. Below each relevant entry with its technical meaning.

| Log entry | Technical meaning |
|---|---|
| version 0.15.1 | vLLM V1 engine active — rewritten internal architecture, more efficient for scheduling and memory than previous versions. |
| Resolved architecture: MistralForCausalLM | vLLM identified the model family. Determines which CUDA kernel is selected and how attention operations are optimized. |
| quantization=fp8 / Selected CutlassFP8ScaledMMLinearKernel | FP8 requires specific kernels for matrix multiplications. CutlassFP8 is the NVIDIA kernel optimized for Ada Lovelace architectures (RTX 4070 Super). |
| Using FLASH_ATTN attention backend | FlashAttention selected automatically among 4 alternatives. Reduces memory usage during attention and increases speed — optimal choice for RTX 4070 Super. |
| Model loading took 7.01 GiB and 15.976617 seconds | 15 seconds because the model was already in local cache. At first startup with download it had taken 429 seconds. |
| torch.compile takes 9.61 s in total | vLLM compiles the computational graph with PyTorch Inductor. Result cached — subsequent executions use the cache. |
| Available KV cache memory: 2.41 GiB / GPU KV cache size: 19,696 tokens | With 2.41 GiB remaining, vLLM can manage at most 19,696 total KV cache tokens. This is the physical constraint that required reducing max_model_len. |
| Maximum concurrency: 1.04x for 19,000 tokens | With 19k token requests, the KV cache supports slightly more than one concurrent request. |
| Capturing CUDA graphs: 51/51 + 35/35 | vLLM captures 86 CUDA graphs for batch sizes from 1 to 512 tokens. They eliminate CUDA scheduling overhead — improve decode latency. Cost: 0.47 GiB of additional VRAM. |
| Chunked prefill enabled, max_num_batched_tokens=2048 | Prefill is split into 2048-token chunks. Prevents long prompts from monopolizing the GPU and blocking ongoing decode requests. |
| enable_prefix_caching=True | KV cache reuse for identical prefixes across requests. With a fixed system prompt in tests, each new chat benefits from this. |
| Application startup complete. | Engine ready to receive requests. This is the target line to look for in logs to confirm successful startup. |

## 12. Docker-compose Configuration

### 12.1 Explicitly Declared Parameters

| Parameter | Value | Meaning |
|---|---|---|
| --model | neuralmagic/Mistral-7B-Instruct-v0.3-FP8 | HuggingFace model identifier. |
| --max-model-len | 19000 | Maximum serviceable context window — reduced due to VRAM constraint. |
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

### 12.2 Features Activated Automatically by vLLM

Everything not in the docker-compose is configured automatically by vLLM at runtime, based on the detected model and hardware.

| Feature | How and why it is activated automatically |
|---|---|
| FlashAttention backend | vLLM detects the GPU and selects the most efficient attention backend. On RTX 4070 Super (Ada Lovelace), FlashAttention is always the optimal choice. |
| CutlassFP8 kernel | Detected FP8 quantization, vLLM automatically selects the optimized CUDA kernel. |
| Chunked prefill | Active by default in vLLM 0.15.x with V1 engine. |
| Prefix caching | Active by default. Particularly useful with fixed system prompts. |
| torch.compile + Inductor | V1 engine compiles the computational graph at first startup and caches the result. |
| CUDA graphs (86 graphs) | Captured automatically for batch sizes from 1 to 512. Reduce decode latency. |
| Asynchronous scheduling | Active by default in V1 engine. Decouples scheduling from GPU execution. |

**Key principle:** the docker-compose is the minimum contract with vLLM. Everything else is automatic optimization. Adding unnecessary parameters introduces conflict risk and reduces readability without concrete benefits.

## 13. Comparative Test Results

Tests were run with the same parameters as Part 1: repetition_penalty = 1.2, numbered narrative prompt of 75 paragraphs, Italian and French languages.

### 13.1 Test A — 75-Paragraph Narrative in Italian

Collapse occurred at paragraphs 3–4, approximately 200–300 generated tokens. Paragraph 1 was comprehensible Italian but with syntactic anomalies at the tail. From Paragraph 3: fragmented Italian, invented neologisms, token flooding, insertion of Chinese characters. By Paragraph 4 the text was completely illegible.

Direct comparison: Qwen2.5-7B with the same penalty had maintained coherence up to paragraph 63 (~5000 tokens). The ratio is approximately **25:1** in favor of Qwen on this task.

### 13.2 Test B — 75-Paragraph Narrative in French

The hypothesis was that Mistral, developed by a French company, would perform significantly better in French. Result: collapse at §4 instead of §3 — one additional paragraph. The difference is minimal and insufficient to draw conclusions about language as a causal factor.

### 13.3 Test C — Short Technical Task in Italian (~500 words)

Markedly positive result: fluent Italian, coherent logical structure, correct coverage of key concepts. Some minor terminological inaccuracies but acceptable for a 7B on a popularization task. Zero degeneration, zero linguistic drift.

This demonstrates that the collapse observed in narrative tests is not a general limitation of Mistral — it is specific to the combination of model + long output + rigid structure.

### 13.4 Summary Table

| Model | Task | Penalty | Collapse | Outcome |
|---|---|---|---|---|
| Qwen2.5-7B-AWQ | IT narrative 75 § | 1.2 | §63 (~5000 tok) | Complete, degrades at tail |
| Mistral-7B-FP8 | IT narrative 75 § | 1.2 | §3–4 (~200 tok) | Immediate collapse |
| Mistral-7B-FP8 | FR narrative 75 § | 1.2 | §4 (~300 tok) | Immediate collapse |
| Mistral-7B-FP8 | IT technical task ~500 words | 1.2 | None (~400 tok) | Coherent, acceptable quality |

## 14. Comparison Conclusions

**The threshold is model-specific — observed data.** On this task, Qwen2.5-7B maintained coherence for ~5000 tokens. Mistral-7B collapsed within 200–300 tokens. 25:1 ratio. Same parameter count, same hardware, same decoding parameters — completely different results.

The causes of this difference cannot be isolated from the tests conducted. Candidate factors include: dataset and instruction tuning, sampling sensitivity, quantization differences (AWQ vs FP8), tokenizer. The test demonstrates the difference, not the cause.

**Language as a factor — insufficient data.** French postponed the collapse by one paragraph compared to Italian. The difference is minimal and does not allow concluding that language is a relevant causal factor in this comparison.

**The threshold is task-specific — observed data.** Mistral on a short technical task (~400 tokens) produces coherent and acceptable quality output. The collapse observed in narrative tests is not a general limitation of the model — it is specific to the combination of model + long output + rigid structure.

**Implication for deployment:** model selection must explicitly consider the type of task and expected output length. An English benchmark on generic tasks is not sufficient to predict behavior on specific tasks in different languages. The coherence threshold must be measured empirically for each relevant model-task-language combination.

---

*Document produced as part of the AI Infrastructure lab — Dielabs Academy*
