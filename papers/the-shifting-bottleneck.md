---
layout: default
title: "The Shifting Bottleneck"
---

# The Shifting Bottleneck

## Three benchmarks on a single GPU, and what they teach about inference sizing

**Lab:** Dielabs **Hardware:** NVIDIA RTX 4070 Super (12 GB GDDR6X) **Model:** Qwen3-8B-AWQ **Stack:** vLLM 0.20.1, Docker, Prometheus + Grafana **Date:** May 2026

---

## 1. The wrong question

When a customer asks "_how many concurrent users can this system handle?_", the honest answer is: **the question is ill-posed until you specify the workload and the SLO**. The same hardware, with the same model and the same inference serving platform, behaves like three different machines depending on what you throw at it.

This paper documents three benchmarks I ran in my Dielabs setup over the course of a single day. The hardware never changed. The model never changed. Only two things changed across runs: the KV cache datatype (inference serving setting), and the prompt-to-output token ratio (workload). The conclusions go well beyond the specific numbers — they could form a methodology that applies to any inference sizing exercise.

The narrative arc is simple:

1. A baseline incremental run (must) reveals a clear bottleneck.
2. Targeted intervention (FP8 KV cache) raises the ceiling of the same bottleneck
3. A workload change moves the bottleneck.

---

## 2. The lab setup

The serving stack:

```yaml
services:
  vllm:
    image: vllm/vllm-openai:latest
    command:
      - "--model"
      - "Qwen/Qwen3-8B-AWQ"
      - "--max-model-len"
      - "8192"
```

The benchmark driver is GuideLLM, run in `concurrent` mode (closed-loop): N clients keep **N requests in flight at all times**, sending a new one as soon as the previous returns. This is the classic stress-test profile — it tells me what the system can sustain under continuous pressure, which is the right shape for a **capacity benchmark**.

For each run I sweep concurrency from 4 to 28 (or 32), running each level for 240–480 seconds. Observability comes from vLLM's Prometheus metrics scraped every 10s, plus DCGM for GPU-level signals. I look at six things, in roughly this order of importance:

- `num_requests_running` and `num_requests_waiting` (scheduler state)
- `kv_cache_usage_perc` (memory pressure)
- TTFT, ITL, E2E latency percentiles (user experience)
- Output throughput (system productivity)
- GPU utilization and VRAM allocation (hardware saturation)

What follows is what each run revealed.

---

## 3. Benchmark 1 — Baseline FP16 KV cache, prompt-heavy workload

### Configuration

- KV cache dtype: default (FP16)
- Workload: `prompt_tokens=2048, output_tokens=256` (8:1 ratio, prefill-heavy)
- Concurrency sweep: 4, 8, 12, 16, 20, 24, 28, 32
- Duration: 240s per level

This workload approximates RAG or summarization use cases: long context in, short generation out. Compute-heavy on prefill, light on decode.

### Results

|Conc|Req/s|OutTok/s|TTFT p50 (ms)|TTFT p95 (ms)|ITL p50 (ms)|E2E p50 (s)|
|--:|--:|--:|--:|--:|--:|--:|
|4|0.72|184|1361|1797|16.5|5.6|
|8|1.00|254|1817|1820|24.4|8.0|
|12|1.03|255|2752|7706|30.2|10.5|
|16|1.03|258|9480|9546|30.2|17.2|
|20|1.03|257|11356|11403|30.2|19.1|
|24|1.03|257|13598|18161|30.2|22.5|
|28|1.03|259|19969|20028|30.2|27.7|
|32|1.03|255|21832|26767|30.2|29.5|

Three things jump out:

1. **Throughput hits a ceiling at c=12** (~255 tok/s) and never moves again, regardless of how many concurrent clients I throw at it.
2. **TTFT explodes** from 1.3s at c=4 to 21.8s at c=32 — a 16× increase.
3. **ITL plateaus at exactly 30.2 ms** from c=12 onwards, dead flat.

### Diagnosis

Grafana confirmed what the metrics suggested: `num_requests_running` capped at ~10 from c=12 onwards, while `num_requests_waiting` grew linearly with the target concurrency, reaching 22 at c=32. The KV cache usage sat at 90–99% throughout the saturated regime.

The bottleneck is unambiguous: **VRAM available for KV cache**. With ~5–6 GB of weights occupying the 12 GB framebuffer, the KV pool cannot host more than ~10 simultaneous sequences at this max-model-len setting. Everything else queues server-side.

The plateau in ITL is the cleanest evidence. ITL measures the time _between consecutive tokens of an in-flight request_ — it doesn't see the queue. Once the system stabilizes at 10 active streams, the per-token decode cost is fixed at ~30 ms, and adding more clients doesn't change it. They just wait longer to enter.

The TPOT metric, which GuideLLM computes as latency-divided-by-output-tokens, _does_ absorb queue time and grows linearly with concurrency. The distinction between ITL and TPOT becomes important here: ITL tells you how the system runs, TPOT tells you how the user experiences it.

> **Box: Capacity vs scheduling — how to tell them apart**
> 
> When `num_requests_running` plateaus, two hypotheses fit: the scheduler is being conservative, or the system is genuinely full. The disambiguation is in the KV usage. If running plateaus _while_ KV is at 90+%, capacity is the binding constraint. If running plateaus while KV sits at 60%, the scheduler has hit some other limit (max_num_seqs, CUDA graph capture sizes, etc.). In this run, the two metrics moved together — capacity bound, no doubt.

The operating range where this system is healthy is **c=8–12**. Below that, it's underutilized. Above, it's queueing.

---

## 4. Benchmark 2 — FP8 KV cache, same workload

### Hypothesis

If KV capacity is the binding constraint, halving the bytes per KV block should roughly double the number of admissible sequences. FP8 storage on Ada Lovelace doesn't accelerate compute (the kernel still dequantizes to FP16/BF16 internally), but it cuts the storage footprint cleanly in half.

### Configuration

Single change: `--kv-cache-dtype fp8_e5m2`. I picked E5M2 over E4M3 because it doesn't require calibration scales — for a capacity benchmark the marginal precision loss is irrelevant, and I wanted to isolate the storage variable.

I also dropped c=32 from the sweep after the baseline showed it adds no information beyond c=28: same saturation story, marginally longer queue. Three points past the knee is enough to confirm the plateau; four is just verbose.

### Results (with delta vs baseline)

|Conc|Req/s|Δ%|OutTok/s|Δ%|TTFT p50 (ms)|ITL p50 (ms)|
|--:|--:|--:|--:|--:|--:|--:|
|4|0.77|+7%|197|+7%|1353|15.1|
|8|1.10|+10%|281|+11%|1793|21.5|
|12|1.28|+24%|317|+24%|1803|30.1|
|16|1.33|+30%|341|+32%|1808|39.1|
|20|1.34|+30%|329|+28%|2272|48.4|
|24|1.35|+31%|325|+26%|4084|48.3|
|28|1.35|+31%|326|+26%|6114|48.4|

Throughput plateau moved from ~255 to ~330 tok/s (+28%). `num_requests_running` plateau moved from ~10 to ~18–19 (+80%). The knee shifted from c=12 to c=20.

### Diagnosis

The hypothesis is confirmed: the bottleneck was KV storage capacity, and halving the bytes-per-token roughly doubled the admissible parallelism. The system now serves ~18 simultaneous streams instead of ~10.

But the bottleneck **didn't change in nature** — it just got pushed up. ITL still plateaus (this time at 48 ms instead of 30 ms), `num_requests_running` still saturates, and KV usage still parks at 90–99% past c=20. The shape of the curves is identical to the baseline, just shifted.

> **Box: ITL vs concurrency — reading the trade-off**
> 
> ITL plateau went from 30 ms to 48 ms (+60%) when moving to FP8. Why did it go up if FP8 doesn't make the kernel slower? Because at FP8 the system is now running ~18 sequences in the batch instead of ~10. Each decode step has to read the KV of more sequences. Decode is memory-bandwidth-bound: more sequences in the batch = more KV to read per step = higher ITL. **The cost of FP8 isn't on the kernel, it's on the increased batch density it enables.**
> 
> Single-stream FP8 (c=4 with `num_running` actually equal to 4) is slightly _faster_ than FP16 (15.1 vs 16.5 ms). The penalty only kicks in under load, when the batch is full of parallel streams competing for memory bandwidth.

The interesting question this run raises: what happens when the KV cache storage bottleneck is no longer binding? Is there _another_ bottleneck waiting underneath, or does the system just keep scaling? The next run answers this — by changing the workload instead of the configuration.

---

## 5. Benchmark 3 — FP8 KV cache, chat-like workload

### Hypothesis

If I keep the same hardware and same FP8 configuration, but invert the workload from prompt-heavy to decode-heavy, two things should happen:

1. KV per sequence drops (from 2304 to 1536 tokens, –33%), so more sequences fit.
2. Prefill cost drops by ~4× (from 2048 to 512 tokens), so TTFT improves dramatically.

What I didn't fully predict: the _temporal pattern_ of KV usage would change too, in a way that materially affects scheduler behavior.

### Configuration

- KV cache dtype: `fp8_e5m2` (same as run 2)
- Workload: `prompt_tokens=512, output_tokens=1024` (1:2 ratio, decode-heavy)
- Concurrency sweep: 4, 8, 12, 16, 20, 24, 28, 32
- Duration: 480s per level **(longer because each request takes ~4× more time to complete)**

### Results

|Conc|Req/s|OutTok/s|TTFT p50 (ms)|TTFT p99 (ms)|ITL p50 (ms)|E2E p50 (s)|
|--:|--:|--:|--:|--:|--:|--:|
|4|0.29|303|345|451|12.9|13.5|
|8|0.54|567|567|889|13.6|14.4|
|12|0.75|777|786|1003|14.7|15.8|
|16|0.96|951|1004|1198|15.9|17.2|
|20|1.04|1075|1012|1541|17.6|19.0|
|24|1.19|1217|1015|1638|18.7|20.1|
|28|1.24|1214|1017|1641|20.2|21.7|
|32|1.33|1298|1757|3886|21.4|23.8|

The numbers tell a different story from the previous two runs:

- **Throughput keeps climbing** from c=4 to c=32 (303 → 1298 tok/s), with only a brief flat spot at c=24–28.
- **TTFT stays low** (under 1.1s) until c=28, then jumps at c=32.
- ***ITL grows linearly with concurrency*** (12.9 → 21.4) without plateauing.
- **`num_requests_running` follows the target** all the way to 32, no saturation cap.
- **KV usage oscillates** in a sawtooth pattern between 35% and 95%, never sitting flat at the ceiling.
- **The queue stays empty** for most of the run, with brief spikes only at c=32.

### Diagnosis

The bottleneck has **changed in nature**. KV capacity is no longer binding — the system has headroom. What's saturating now is **GPU memory bandwidth on the decode step**. The evidence is in the ITL: it grows continuously and approximately linearly with `num_running`, without reaching a plateau within the tested range. **Each additional sequence in the batch adds a measurable cost per token, because each decode step has to read more KV from memory.**

The TTFT break at c=32 is the visible symptom of this. Once the bandwidth-bound decode batch is full, new arrivals start waiting longer in the admission queue — not because there's no KV space (there is), but because the scheduler is reluctant to admit more sequences when each existing one is already costing measurable bandwidth. The brief queue spike to 6 sequences at c=32 confirms it.

The chain of causation runs:

> bandwidth saturated → each decode step slower → batch slots held longer → new sequences wait longer → TTFT rises

TTFT is the _downstream effect_, not the cause. ITL is the cleanest leading indicator.

> **Box: The decode-heavy advantage isn't in the ratio, it's in the timing**
> 
> *A naive reading would conclude "decode-heavy workloads are more efficient than prompt-heavy". The real story is more nuanced.*
> 
> A prompt-heavy sequence (2048+256) requires the scheduler to allocate KV for ~2048 tokens _immediately_ on admission. If those blocks aren't free, the request queues. A chat-like sequence (512+1024) requires only ~512 blocks on admission, with the rest accumulated incrementally one decode step at a time.
> 
> When client arrivals are _staggered in time_ (which they are in any closed-loop benchmark, and in any organic production traffic), each sequence in the batch **is at a different point in its decode** lifecycle. Some are at token 50, others at 500, others at 950. The total KV in use at any instant is far below the sum of per-sequence peaks.
> 
> The sawtooth pattern in KV usage is the direct fingerprint of this behavior: sequences finishing free up large chunks of KV while others are still climbing slowly. The system breathes.
> 
> **The advantage disappears under synchronized burst arrivals.** If 32 chat-like clients all submit requests at the same instant, they all start prefilling 512 tokens together, then all decode in lockstep, all hitting their 1536-token peak simultaneously. At that moment KV looks just as saturated as in the prompt-heavy run.
> 
> So the precise principle is: _staggered arrivals + decode-dominant workloads_ let the system serve more concurrent streams than the static peak-KV calculation would suggest. All three conditions matter.

The operating sweet spot for this workload, looking purely at sustainable latency: **c=20–24**. Past c=28 the TTFT degradation isn't worth the marginal throughput gain.

---

## 6. The single-stream baseline as a ruler

A point I underweighted at first: every benchmark sweep should include a c=1 run as a reference floor. Not for throughput (it's by definition the minimum), but for the latency components. Without a single-stream baseline, you can't tell how much of the latency at higher concurrency is structural versus introduced by batching and queueing.

For the chat-like workload, the c=1 latency floor is approximately:

```
E2E_min = TTFT_min + (ITL_min × output_tokens)
        ≈ 300 ms + (13 ms × 1024)
        ≈ 13.5 s
```

The actual c=4 run measured 13.5s E2E p50. With four streams already in flight, the system is effectively already running at the latency floor — the four sequences barely interfere with each other. This is what "underutilized" looks like from the latency perspective.

> **Box: Demo number vs production number**
> 
> Single-stream metrics are the numbers you show in a demo: clean, fast, impressive. They are also _non-representative_ of production behavior. A serious sizing conversation distinguishes between the two:
> 
> - **Demo number**: c=1, ITL ~13 ms, TTFT ~345 ms. What a single user feels with the system to themselves.
> - **Production number**: c=24, ITL ~19 ms, TTFT ~1015 ms. What 24 concurrent users feel sharing the system.
> 
> Both are true. Showing both, and being clear about which is which, is the difference between a vendor pitch and an architectural conversation.

The latency floor is _unbeatable_ on this hardware with this model and this output length. To get below 13s E2E for 1024-token outputs, you need one of:

1. **Generate fewer tokens** — change the workload, not the hardware.
2. **Speculative decoding** — use a draft model to skip decode steps. Worth 1.5–2× on this class of model.
3. **More memory bandwidth** — H100 has 3.35 TB/s versus ~500 GB/s on the 4070 Super. ITL would drop ~6×, E2E to ~3s.

This is the most important sizing conversation a customer can have, and it's almost never had explicitly.

---

## 7. The principles

Five things this lab made undeniable.

### 7.1 Bottlenecks don't go away — they move

Each intervention removed _the currently binding constraint_ and exposed a different one. Three runs, three bottlenecks:

1. Baseline FP16, prompt-heavy → KV capacity, knee at c=10.
2. FP8 KV, prompt-heavy → KV capacity (raised ceiling), knee at c=18.
3. FP8 KV, chat-like → memory bandwidth, knee at c=24–28.

Past the bandwidth bottleneck (which I didn't push further in this lab), the next layers in the stack are: attention kernel compute, scheduler overhead at high parallelism, interconnect bandwidth in tensor-parallel setups, network bandwidth in disaggregated prefill/decode architectures, storage IOPS in tiered KV cache deployments. Each layer demands a different lever, different hardware, different cost.

Knowing which layer is binding for a specific customer determines what to propose — and, crucially, what _not_ to propose.

### 7.2 Latency has a floor

The minimum end-to-end latency for any inference request decomposes cleanly into:

```
E2E_min = TTFT + (ITL × output_tokens)
```

TTFT is bounded below by the prefill cost (compute-bound, scales with input length). ITL is bounded below by memory bandwidth (scales with model size and context length). Neither can be optimized away with batching, prefix caching, or quantization tricks. They can only be moved by:

- Changing the workload (fewer output tokens, shorter prompts)
- Changing the algorithm (speculative decoding, parallel sampling)
- Changing the hardware (more bandwidth, more compute)

For chat-like workloads with long generations, the latency floor is the dominant user-facing metric. Concurrency doesn't help — it just gives you that floor for more users simultaneously.

### 7.3 Sizing without an SLO is impossible

The same hardware running the same model achieves wildly different "concurrent user counts" depending on what counts as acceptable:

|SLO|Concurrent users supportable (chat-like, FP8)|
|---|---|
|ITL < 15 ms|~8|
|ITL < 20 ms|~24|
|TTFT < 1s|~24|
|TTFT < 2s|~32|
|E2E < 15s for 1024 tokens|~8|
|E2E < 20s for 1024 tokens|~24|

There's no single "capacity" number. There's a family of capacity numbers, one per SLO. The customer who answers "how many users?" without specifying the SLO is asking the wrong question.

### 7.4 Arrival timing is an architectural variable

Two systems with identical peak-load metrics can behave very differently under realistic traffic, depending on whether arrivals are staggered or synchronized. Decode-heavy workloads with organic arrivals can sustain concurrency levels that the static peak-KV calculation would forbid — but the same workload with synchronized burst arrivals collapses back to the static calculation.

This means sizing exercises need two numbers, not one:

- **Conservative (peak-burst) sizing**: assume worst-case synchronized arrivals. Robust but expensive.
- **Optimistic (steady-state) sizing**: assume staggered arrivals at average rate. Realistic for organic traffic, fragile under burst.

The right number depends on the customer's traffic shape. Internal corporate chatbots tend toward steady-state. RAG pipelines fed by batch jobs tend toward burst. The conversation about _which_ matters more than the conversation about _how many_.

### 7.5 Synthetic benchmarks measure capacity, not robustness

All three runs in this lab used uniform synthetic prompts and fixed output lengths. This is the right shape for _capacity_ benchmarks — it isolates the system's nominal throughput under stable pressure. But it systematically _under-stresses_ the system on robustness dimensions: variance in output length, prefix cache reuse patterns, mixed workload types, traffic bursts.

A complete benchmark suite needs both:

- **Capacity benchmarks**: synthetic, uniform, stable-pressure. Tell you the ceiling.
- **Robustness benchmarks**: realistic distributions, varied workloads, burst patterns. Tell you how the system behaves when reality intrudes.

Production sizing requires both. 

---

## 8. From "how many users?" to a sizing conversation that works

Replacing the wrong question with the right one means asking, in order:

1. **What workload?** Prompt-heavy (RAG, summarization, classification) or decode-heavy (chat, agent loops, generation)?
2. **What output length distribution?** Average and tail. The latency floor depends on this.
3. **What SLO targets?** TTFT, ITL, and E2E thresholds. Pick the binding one.
4. **What traffic shape?** Steady-state organic, scheduled batch, peak-burst. Defines conservative vs optimistic sizing.
5. **What concurrent user count under those constraints?** Now this question has a defensible answer.

With those five pieces of input, sizing becomes deterministic. Without them, every number is either invented or copied from a vendor slide.

---

## 9. What this lab didn't test

For methodological honesty, the following questions are explicitly not answered here:

- **Workload variance**: all runs used fixed prompt and output token counts. Real traffic has distributions, and distributions can trigger preemption events that this lab never observed (`num_requests_preempted` was 0 across all runs).
- **Prefix cache effectiveness**: GuideLLM generates random synthetic prompts, so the vLLM prefix cache hit rate was effectively zero. A run with shared system prompts would tell a meaningfully different story for RAG/agent workloads.
- **Speculative decoding**: not enabled in any run. On Qwen3-8B with a small draft model, the expected ITL improvement is 1.5–2×, which would meaningfully change the latency-floor conversation.
- **Multi-GPU configurations**: single-GPU only. Tensor parallelism, pipeline parallelism, and disaggregated prefill/decode are different bottleneck regimes entirely.
- **Hardware comparisons**: only the RTX 4070 Super was tested. Extrapolation to L40S, H100, or H200 is informed but not measured.

Each of these is a candidate for follow-up work. None of them changes the methodology — they just add more data points to the same framework.

---

## 10. Reproducibility

### vLLM serving (run 2 and 3 configuration)

```yaml
services:
  vllm:
    image: vllm/vllm-openai:latest
    container_name: vllm_kvfp8
    restart: "no"
    ports:
      - "8000:8000"
    volumes:
      - vllm_models:/root/.cache/huggingface
    command:
      - "--model"
      - "Qwen/Qwen3-8B-AWQ"
      - "--served-model-name"
      - "qwen3-8b-awq"
      - "--max-model-len"
      - "8192"
      - "--kv-cache-dtype"
      - "fp8_e5m2"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ipc: host
volumes:
  vllm_models:
    external: true
```

### Benchmark commands

Run 1 (FP16 KV, prompt-heavy):

```bash
guidellm benchmark \
  --target http://localhost:8000 \
  --model qwen3-8b-awq \
  --processor Qwen/Qwen3-8B-AWQ \
  --profile concurrent \
  --rate 4,8,12,16,20,24,28,32 \
  --data "prompt_tokens=2048,output_tokens=256" \
  --max-seconds 300 \
  --warmup 0.1 --cooldown 0.1 \
  --output-path qwen3_closed_baseline.yaml
```

Run 2 (FP8 KV, prompt-heavy): same as run 1, output path `qwen3_kvfp8_baseline.yaml`, `--rate 4,8,12,16,20,24,28`.

Run 3 (FP8 KV, chat-like):

```bash
guidellm benchmark \
  --target http://localhost:8000 \
  --model qwen3-8b-awq \
  --processor Qwen/Qwen3-8B-AWQ \
  --profile concurrent \
  --rate 4,8,12,16,20,24,28,32 \
  --data "prompt_tokens=512,output_tokens=1024" \
  --max-seconds 480 \
  --warmup 0.1 --cooldown 0.1 \
  --output-path qwen3_chatlike_fp8.yaml
```

### Key Prometheus queries

```promql
# Active sequences in batch
vllm:num_requests_running

# Queued requests waiting for admission
vllm:num_requests_waiting

# KV cache occupancy
vllm:kv_cache_usage_perc * 100

# Output throughput
rate(vllm:generation_tokens_total[15s])

# Preemption count (was zero across all runs)
vllm:num_requests_preempted
```

### Raw data

The three GuideLLM YAML output files are available in the Dielabs repository [link to be added when published]. Each file contains per-request latency traces in addition to the aggregated metrics shown in this paper.

---

_Comments and corrections welcome. The methodology is more important than any specific number — the goal is to make inference sizing a conversation that actually converges._
