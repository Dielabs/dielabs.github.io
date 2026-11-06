---
layout: default
title: "LLM Model Selection"
description: "A repeatable framework for selecting an open-weight LLM for a specific workload, hardware and serving stack: failure modes, evaluation tracks, uncertainty and reproducibility."
---

<style>
  .content h1 { white-space: normal; }
  .content .table-scroll { overflow-x: auto; margin: 1.5rem 0; }
  .content .table-scroll table { margin: 0; min-width: 1050px; font-size: 0.78rem; }
  .content .table-scroll th, .content .table-scroll td { padding: 8px; }
  .content .table-scroll th { font-size: 0.64rem; letter-spacing: 0.03em; }
  @media (max-width: 700px) { .content > table { display: block; overflow-x: auto; } }
</style>

# LLM Model Selection

## A repeatable evaluation framework for on-premise production

**Author:** Diego Bardella · **Lab:** Dielabs · **Reference date:** November 2026

How do you choose an open-weight model to run on your own GPUs, and make the decision repeatable whenever a new candidate arrives? This framework starts with the workload's failure modes and ends with a justified shortlist on the actual hardware and serving stack.

The starting point is Fabio Carta's talk for Agile Lab at MLOpsDay 2026, Bologna, 29 October 2026. Its running example is CWYD, *Chat With Your Data*: an agent that answers business questions by querying SQL databases. Product names remain as concrete examples. The pipeline, timings and model scores attributed to the talk are **Agile Lab's reported work, not measurements made by Dielabs**. Dielabs adds the agent acceptance track, the connection to capacity planning and customer golden sets, and the discussion of sampling uncertainty and limitations.

## 1. What this framework does—and what it does not

**It does:** select among shortlisted open-weight models on the real stack using repeatable criteria. The output is a table with one row per candidate and a reasoned shortlist.

**It does not, by itself, establish:**

- The finished agent's quality on customer data. That is track C, using a golden set of real questions and expected outcomes, as defined in the AI Technical Presales Framework's evaluation and acceptance process.
- Platform acceptance against the performance promised in a proposal. That belongs to the agreed acceptance criteria and benchmark protocol.
- Hardware sizing. The measured throughput becomes an input to sizing; it is not the sizing calculation itself.

**Where it fits:** after discovery identifies failure modes and the workload, before hardware sizing. The selected model changes the sizing inputs, so the capability decision comes first. [From Idea to Production](from-idea-to-production-manual.html) describes the broader technical path. The AI Technical Presales Framework is [available on request](mailto:info@dielabs.eu?subject=AI%20Technical%20Presales%20Framework%20%E2%80%94%20request).

## 2. Why leaderboards are not enough

Public rankings answer someone else's question. Four reasons matter:

| Problem | What it means |
| --- | --- |
| Wrong workload | The prompts, use cases and generation settings are theirs, not yours. |
| Wrong stack | The same weights can behave differently with another engine, chat template, parser or quantization. These settings are often not disclosed. |
| Wrong cadence | Models arrive faster than rankings settle. |
| Missing axis | A leaderboard does not tell you what it costs to serve the candidate on your GPUs. |

The framework therefore asks: which model is best on this hardware, behind this stack, for this work?

## 3. From the use case to the benchmarks

### 3.1 Failure modes, capabilities, benchmarks

Build the chain by identifying what the agent must do, how it fails when it cannot, and which measurement catches that failure.

| The agent must… | Otherwise, in production… | Measure with… | Benchmark used for CWYD |
| --- | --- | --- | --- |
| Write correct SQL | Plausible but wrong numbers | Text-to-SQL scored by executing the query | BIRD |
| Plan across steps | Incorrect joins and filters | Reasoning tests | BBH |
| Respect the output format | The downstream parser breaks | Instruction following | IFEval, IFBench |
| Call tools correctly | Invented or incorrect actions | Function calling | BFCL v3 |
| Understand the domain | Misread business terms | Knowledge tests | MMLU-Pro |
| Serve the intended load | More GPUs are needed than available | Serving performance | vllm_bench with ShareGPT |

**Rule:** no requirement, no benchmark. If a test covers no relevant failure mode, leave it out.

The table changes with the use case; the method does not. A document RAG agent replaces “write SQL” with “find the right passage and cite it”. Its direct evaluation must assess retrieval and grounded answers with citations.

### 3.2 How strong is the proxy?

Not every benchmark measures the actual job. State how close each one is to it.

| Strength | Meaning | CWYD example |
| --- | --- | --- |
| Direct task or serving test | Measures the target task class or the serving path, though the dataset and traffic must still match production | BIRD, with SQL execution; vllm_bench, with throughput and waiting times |
| Strong proxy | Exercises behavior needed on each request | BFCL for schema/query tool calls; IFBench for parser-compatible output; IFEval as an easier instruction-following gate |
| Partial proxy | General capability rather than the specific task | BBH for planning joins and interpreting dates; GSM8K for deltas and ratios |
| Weak proxy | General knowledge or auxiliary skills rather than the customer's domain | MMLU-Pro for business vocabulary; HumanEval for possible post-processing code |

Weak proxies are the first place to improve when adapting the pipeline to a customer. Even a direct task benchmark on public databases does not replace evaluation on that customer's schema.

### 3.3 CWYD example: one question, eight capabilities

Question: *“Which region increased its margin the most in Q3 compared with Q2? Answer with a table.”*

| Step | What the agent does | Benchmark |
| --- | --- | --- |
| Understand | Business terms and relative dates, such as Q3 versus Q2 | MMLU-Pro: multiple-choice knowledge across 14 subjects; BBH: logic, dates and word problems |
| Plan | Choose tables, joins and filters | BBH: multi-step reasoning |
| Use tools | Select the correct tool and arguments, or use no tool | BFCL: select a function and populate its arguments |
| Write SQL | Query a real, messy schema | BIRD: SQL on real databases, checked through execution |
| Calculate | Compute percentage growth and rank returned rows | GSM8K: mathematical word problems |
| Respond | Follow format, length and language constraints | IFEval: verifiable rules; IFBench: additional instruction constraints |

Two capabilities cut across the steps: operating at the intended load, assessed through serving tests, and any Python post-processing, assessed through executable coding tests such as HumanEval.

## 4. Six criteria for choosing a benchmark

Covering a failure mode is necessary but not sufficient. In this framework, a benchmark enters the suite only if it meets all six criteria.

| Criterion | Rule | Why |
| --- | --- | --- |
| Failure mode | No requirement, no benchmark | Otherwise GPU time is wasted. |
| Harness integration | Available as a task in the chosen evaluation framework at evaluation time | Between tests with the same purpose, prefer the one already integrated and easier to rerun. |
| Alongside the engine | Runs on the model's VM within the intended resource budget | No additional judge GPU or large fleet of sandboxes for the basic suite. |
| Hours, not days | The suite for one model fits within a morning or afternoon | It can be rerun at each release. |
| Verifiable scoring | Execution, AST comparison, unit tests, constraint checks or exact matching; no external LLM judge | Lower cost and easier auditing, including in an air-gapped environment once dependencies are staged. |
| Discrimination | Differences between models must exceed measurement noise | A test where every candidate scores the same cannot support selection. |

Harness integration is a practical criterion, not a scientific one, and is stated as such: ease of repetition takes priority over an ideal measurement that is never repeated. Deterministic scoring does not guarantee deterministic model generation.

## 5. Three evaluation tracks

| Track | Question | Method | Responsibility |
| --- | --- | --- | --- |
| A — Capability | Can the model do it? | Public benchmarks behind a unified evaluation framework | This model-selection framework |
| B — Task and speed | Can it do the required kind of work fast enough? | A direct task benchmark, BIRD for CWYD, plus load tests on the actual hardware | This framework for selection; the benchmark protocol for load measurement |
| C — Agent | Does the resulting agent work on customer data? | A golden set with real questions, expected answers and the customer's table metadata | The presales acceptance process and agent-design method |

**A and B select the model; C accepts the agent.** The talk covers A and B. Track C is a Dielabs addition.

**Track B and platform capacity.** The talk measures requests per second with ShareGPT prompts. For sizing, measure both hardware-anchored capacity in closed loop, Cr_closed, and capacity under service-level objectives in open loop, Cr_open, using the method in [One Capacity Is Not Enough](one-capacity-is-not-enough.html). The result can then inform replica and GPU counts. Prompt and response lengths, traffic and concurrency must come from discovery.

**Calibrate proxies against track C.** Once a golden set exists, compare the A/B ranking with C. Downgrade or remove benchmarks that do not predict customer-task performance. Until this comparison is made, it is unknown how well the selection pipeline approximates the final agent's evaluation.

## 6. Pipeline architecture

### 6.1 One endpoint, one container per harness

The talk uses an on-premise VM with one A100 80 GB and no external calls during evaluation:

```text
YAML configuration → launcher → harness containers → adapter → inference engine

Target: one OpenAI-compatible endpoint
Execution: local
Harnesses: BFCL v3 (AST), bigcode-evaluation-harness (HumanEval+),
           lm-evaluation-harness (IFEval, GSM8K, BBH, MMLU-Pro), IFBench
Serving: vLLM on the same VM, /v1/chat/completions
Per-model settings: context, chat template, tool parser; thinking disabled
Outputs: collected results in a common format
```

A harness combines a dataset, prompt templates and a scoring rule. The important choices are:

- **The evaluation framework does not load the model.** The engine serves the model once, at a pinned version, through one endpoint. All benchmarks target it and may run concurrently within controlled limits.
- **One version-pinned container per harness.** Each test tool runs in its own reproducible environment.
- **One command and configuration for the suite.** Results use a common format, avoiding reconciliation scripts.
- **One serving configuration per model.** Record context limits, chat template and parsers rather than assuming identical settings work across families.

The tool used in the talk is [NVIDIA NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator), an open-source SDK for running evaluations through different harnesses. The launcher manages configuration, execution and result export. Track B uses a custom BIRD implementation on 11 SQLite databases and vllm_bench with ShareGPT traffic.

**Results:** one row per model, including scores, settings and responses, versioned in Git. In the talk, a CSV is packaged into a shareable MLflow bundle that colleagues can import without a running server.

### 6.2 The adapter

A small configurable layer between the test and model can:

- Add a format instruction, such as “answer with the letter only”.
- Cache responses.
- Log requests and responses.
- Separate reasoning from the final answer when a model mixes them.

Write shared transformations once in configuration and apply them consistently. Record model-specific parsing explicitly so it remains auditable. A fair comparison does not silently repair one candidate more than another.

### 6.3 Concurrency and timing

On one GPU, too many simultaneous tests fill the serving queue and cause timeouts. The talk reports:

| Phase | Parallelism | Example duration |
| --- | --- | --- |
| Fast tests | Five at a time | 21 minutes |
| Slow tests | Two at a time, after the first group | 25 minutes |
| SQL and speed | Afterwards | 30 minutes |

The total is approximately 75 minutes for a fast non-thinking model, and several hours for slower or thinking models. Duration depends strongly on how much the model generates, not just how many questions it receives. Reasoning tokens therefore affect the evaluation budget as well as production latency and capacity.

## 7. A zero score often points to the setup

Treat a near-zero or unexpectedly low score as a possible integration failure until investigated. Two families of causes matter.

**Test-side failures**

| Symptom | Possible cause | Remedy |
| --- | --- | --- |
| Correct answer, zero score | The checker expects another format | State the required format through the adapter. |
| A capable coding model scores zero | Harness bug | Try another implementation of the same benchmark. |
| The test hangs | Incompatible dependency inside the container | Pin a compatible version. |
| Truncated answer | Reasoning consumed the output budget | Adjust the token limit and record it. |

**Server-side failures**

| Symptom | Possible cause | Remedy |
| --- | --- | --- |
| Incoherent output | Wrong chat template | Use the model's correct template. |
| Reasoning appears inside the answer | Missing or incorrect reasoning parser | Configure a compatible parser. |
| Unexpected outputs cause crashes | A rigid or incompatible parser | Validate a compatible parser version. |

**Rule:** check the setup before blaming the model. Compatibility with the actual server is itself a requirement: parsing problems on the deployment stack matter in production too.

A unified evaluation framework may expose multiple implementations of a benchmark, allowing a harness issue to be isolated without rewriting the entire pipeline.

## 8. Sampling and uncertainty

Full benchmarks on a single GPU can take days. Use a fixed sample, balanced across benchmark topics where appropriate. The talk reports 84 questions from roughly 12,000 in MMLU-Pro and 138 from roughly 6,500 in BBH. Choose sample size from the time budget, allowing margin for slow models.

**A sample score is an estimate.** A simple approximate 95% margin for a proportion, with p as a fraction and n as the number of questions, is:

```text
margin ≈ 1.96 × sqrt(p × (1 − p) / n)
```

| Example | p | n | Approximate margin |
| --- | --- | --- | --- |
| BIRD example from the talk | 0.57 | 500 | ±4.3 percentage points |
| Sampled BBH | 0.55 | 138 | ±8 percentage points |
| Sampled MMLU-Pro | 0.65 | 84 | ±10 percentage points |
| Terminal-Bench 2.0 subset | 0.50 | 30 | ±18 percentage points |

Conversely, around p = 0.5, n ≈ 1 / margin² is a useful rounded planning rule: about 400 questions for ±5 points, or 1,100 for ±3 points.

**Decision rule:** do not rank candidates confidently on small score gaps. Overlapping individual intervals are a conservative prompt to investigate further or prefer operational criteria; they do not prove equivalence. When models face the same questions, inspect paired wins and losses. For small samples or proportions near zero or one, use a more appropriate interval, such as Wilson, rather than relying on the normal approximation. Correlated tasks and stratified sampling also require care.

The talk's leading BIRD candidates cluster between 54.0 and 57.8. Those point estimates alone do not establish a reliable ordering. This qualification makes explicit the limits of the source note's shorthand, “overlapping bars = tie”.

## 9. Reading the results

### 9.1 Gates, capability profile and two axes

Read results in this order:

1. **Minimum gates.** A required capability below threshold disqualifies a candidate regardless of its average.
2. **Verify before rejecting.** Investigate a failing cell using section 7: is it the model or the integration?
3. **Profile before average.** A balanced candidate can be preferable to a higher average with a critical weakness.
4. **Quality and serving efficiency together.** Quality alone can select a model too slow to deploy; speed alone can select one unable to do the job. Shortlist candidates with strong quality and speed after the gates.
5. **Use operational criteria for close candidates.** Consider speed, failure types, license and engine support maturity, alongside statistical uncertainty.

A composite score, such as the mean of min-max-normalized benchmark scores, can help draw a chart. It must not replace the gates or the individual capability profile.

### 9.2 CWYD example output

The following is the Agile Lab pipeline snapshot described in the source notes as spring 2026, reported in the October talk: one A100 80 GB, with a separate serving configuration for each model. BFCL is the average of their selected categories, not the official overall score. **This is an illustration of the pipeline's output, not a current recommendation or a Dielabs benchmark.** Values are reproduced from the supplied notes; raw runs were not independently reproduced here. Columns use the source's scales: BIRD and BFCL are percentage-style scores, while the other quality columns are fractions.

<div class="table-scroll" markdown="1">

| Model | BIRD | req/s | BFCL | IFBench | IFEval | HumanEval | GSM8K | BBH | MMLU-Pro |
|---|---|---|---|---|---|---|---|---|---|
| gemma-4-26B-A4B | **57.8** | 2.6 | 69.0 | 0.43 | 0.89 | 0.98 | 0.86 | 0.56 | **0.75** |
| Qwen3-Coder-30B-A3B | 57.0 | 5.7 | 67.3 | 0.29 | 0.83 | 0.92 | 0.82 | 0.59 | 0.67 |
| Qwen3.5-27B | 56.2 | 1.8 | **74.5** | **0.61** | **0.89** | **0.98** | 0.78 | 0.53 | 0.64 |
| Qwen3.6-35B-A3B | 54.6 | 3.0 | 74.3 | 0.36 | 0.84 | **0.98** | 0.86 | 0.52 | 0.59 |
| Qwen3.5-35B-A3B | 54.0 | **7.9** | 70.0 | 0.44 | 0.84 | 0.93 | 0.79 | 0.51 | 0.63 |
| Qwen3-30B-A3B | 53.4 | 3.1 | 66.0 | 0.30 | 0.82 | 0.88 | 0.69 | 0.58 | 0.64 |
| gpt-oss-20b | 48.8 | 7.3 | 47.0 | 0.27 | 0.48 | 0.62 | 0.73 | **0.64** | 0.55 |
| Mistral-Small-4-119B | 44.6 | 0.6 | 73.5 | 0.27 | 0.76 | 0.91 | 0.90 | 0.57 | 0.65 |
| IQuest-Coder-V1-14B | 42.6 | 3.2 | 10.5 | 0.18 | 0.54 | 0.79 | 0.74 | 0.46 | 0.36 |
| Devstral-Small-2-24B | 42.2 | 2.0 | 67.1 | 0.22 | 0.74 | 0.90 | **0.96** | 0.59 | 0.68 |
| Ministral-3-14B | 41.4 | 5.5 | 64.9 | 0.25 | 0.63 | 0.87 | 0.94 | 0.54 | 0.59 |
| GLM-4.7-Flash | 38.8 | 1.7 | 68.1 | 0.41 | 0.80 | 0.90 | 0.73 | 0.58 | 0.63 |
| NVIDIA-Nemotron-3-Nano-30B-A3B | 37.2 | 1.7 | 65.3 | 0.21 | 0.54 | 0.62 | 0.83 | 0.55 | 0.52 |
| Nemotron-Cascade-2-30B-A3B | 13.2 | 6.1 | 56.8 | 0.29 | 0.66 | 0.61 | 0.55 | 0.42 | 0.44 |

</div>

Bold values reproduce the source table's highlighted values. Read the table as follows:

- No candidate wins every column.
- Reported throughput ranges from 0.6 to 7.9 requests/s on the same A100, a roughly 13-fold spread using the displayed rounded values.
- In the talk's quality/speed trade-off, Qwen3.5-35B-A3B emerges as the main balanced candidate. Qwen3.5-27B offers stronger quality but lower speed; gpt-oss-20b and Nemotron-Cascade are fast but miss required capabilities. These are interpretations of that snapshot, not universal model rankings.
- Very low cells in otherwise capable models, such as IQuest's BFCL result, require the setup checks in section 7 first.

## 10. Pipeline reproducibility

Agile Lab's pipeline audit provides a checklist for other implementations:

| Finding | Risk | Remedy |
| --- | --- | --- |
| Unpinned launcher, unrecorded image tags | The run cannot be reproduced | Pin versions and record image digests. |
| Datasets downloaded from changing URLs | Silent dataset changes | Pin a revision and checksum. |
| Sample sizes or benchmark variants changed mid-campaign | Rows are not comparable | Version the suite for each campaign. |
| Serving tests used different prompt counts | Throughput is not comparable | Fix the load preset. |
| CSV rows not linked to commits and run dates | A result's provenance is lost | Keep a manifest per row. |
| Configuration comments disagree with actual values | Documentation misleads | Generate documentation from configuration. |

These are inexpensive fixes. Without them, the example table cannot support a defensible customer decision.

## 11. The benchmark lifecycle

### 11.1 Saturation

A benchmark loses selection value when leading candidates cluster too closely to distinguish with the available evidence. It may remain useful as a minimum gate.

In the talk's data:

- The leading seven IFEval scores span 0.80–0.89, compared with 0.29–0.61 on IFBench. The latter shows more separation in that snapshot; uncertainty must still be considered.
- The top three HumanEval scores are 0.98.
- BIRD, BFCL and IFBench retain more discriminating value in the reported campaign.

### 11.2 Proposed replacements

Run a new benchmark alongside the old one for several campaigns before replacing it. This preserves a bridge to historical results. Candidates must still satisfy the six criteria in section 4; the list below is a set of proposals to qualify, not a pre-approved suite.

| Existing test | Proposed candidate | Constraint to check |
| --- | --- | --- |
| HumanEval | Terminal-Bench 2.0 subset | An agent loop and task containers; higher infrastructure cost |
| HumanEval | LiveCodeBench v6 | Reported as being trialled in the talk, n = 100 |
| IFEval | IFBench, retaining IFEval as a gate | Already part of the reported suite |
| BFCL v3 | BFCL v4 | Exclude live web-search components where incompatible with air-gapped operation |
| GSM8K | MATH-500 or AIME | Small samples; verify the scoring path and any judge dependency |
| MMLU-Pro | GPQA Diamond | 198 questions; dataset access conditions |
| BIRD Mini-Dev | AMBROSIA and customer schemas | Dataset curation cost |
| vllm_bench | Add llama-benchy | Context-depth latency for a single request complements a load test |

Notes accompanying the November 2026 framework:

- [Terminal-Bench 2.0](https://arxiv.org/abs/2601.11868) comprises 89 terminal tasks. [Harbor](https://github.com/harbor-framework/harbor) provides the evaluation infrastructure. The talk proposes a fixed, stratified subset of about 30 tasks in Git, fewer simultaneous long-context sessions, and turn/time limits, reserving the full set for the shortlist. Thirty tasks give a wide uncertainty band; they support coarse screening more readily than precise ranking.
- BFCL v4 expands into agentic evaluation, including web search and memory. The source note flags ageing expected web answers as a reason to review or exclude that portion. Check the pinned dataset and scoring implementation before adoption.
- [IFBench](https://arxiv.org/abs/2507.02833) introduces 58 verifiable instruction constraints and investigates generalization beyond IFEval-style instructions.
- [AMBROSIA](https://arxiv.org/abs/2406.19073), Saparina and Lapata, NeurIPS 2024, evaluates ambiguous text-to-SQL questions, such as whether “margin” means gross or net margin.
- [llama-benchy](https://github.com/eugr/llama-benchy) measures prompt-processing and generation speed at different context depths through OpenAI-compatible endpoints, useful when agent context grows over time.

## 12. Declared limitations

- **Thinking disabled.** The talk reports non-thinking runs to manage memory and runtime on one A100. Shorter contexts allow more concurrency. If production uses reasoning, repeat track B with it enabled: context use, latency and throughput can change substantially.
- **Temperature and repeated runs.** Temperature zero and thinking disabled reduce variation but do not guarantee identical runs. With active sampling or reasoning, run each candidate repeatedly and report central tendency and dispersion.
- **No external LLM judge in model selection.** This is a cost, repeatability and air-gap choice. A judge may be used in track C if checked against human-reviewed examples.
- **Quantization.** BF16, FP8 and INT4 versions are separate deployment candidates and need separate rows, even when the model name is the same.
- **Generality.** The talk's suite targets a use-case category, not a particular customer. Real business vocabulary and schemas enter through track C or customer-specific benchmarks.
- **Reported evidence.** The talk's results and timings are retained as attributed examples. This English publication does not provide or claim an independent reproduction of its raw runs.

## 13. Operational checklist

1. From discovery, record the use case's failure modes and workload profile.
2. Map failure modes to capabilities and benchmarks, including proxy strength.
3. Apply the six selection criteria. Record why each excluded test was rejected.
4. Shortlist candidates, including quantization variants.
5. Fix the serving configuration for each model: templates, tool and reasoning parsers, context limits and engine version.
6. Version the suite: harnesses, image digests, dataset revisions/checksums, sample sizes and load presets.
7. Run A with bounded concurrency, then B. Keep a manifest for every result row.
8. Investigate zero scores and failed gates before rejecting candidates.
9. Read gates, capability profile, quality/speed trade-offs and uncertainty, in that order.
10. Run C on the leading candidate and one or two alternatives using the customer's golden set. Compare with A/B to calibrate the proxies.
11. Rerun the same suite for each new model, and review benchmark saturation at every campaign.

## Sources and attribution

- Fabio Carta, Agile Lab, *Benchmarking open-source LLMs: a data-driven approach to model selection for on-premise production*, MLOpsDay 2026, Bologna, 29 October 2026. Slides and transcript supplied by Diego Bardella; source of the CWYD example, pipeline, timing examples and result table. No public slide or recording URL is provided here.
- [NVIDIA NeMo Evaluator repository](https://github.com/NVIDIA-NeMo/Evaluator) and [documentation](https://docs.nvidia.com/nemo/evaluator/latest/index.html).
- [Terminal-Bench paper](https://arxiv.org/abs/2601.11868) and [Harbor](https://github.com/harbor-framework/harbor).
- [BFCL project](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard). The original note also cites an [Epoch AI review](https://epoch.ai/benchmarks/berkeley-function-calling-leaderboard/review) for its caution about web-search tasks; that is a secondary assessment.
- [Generalizing Verifiable Instruction Following](https://arxiv.org/abs/2507.02833) and [IFBench repository](https://github.com/allenai/IFBench).
- [AMBROSIA paper](https://arxiv.org/abs/2406.19073) and [repository](https://github.com/saparina/ambrosia).
- [llama-benchy repository](https://github.com/eugr/llama-benchy).

*Translated and edited from the Dielabs model-selection note of 3 November 2026. The English edition clarifies statistical uncertainty, uses the displayed throughput values for the spread calculation, and replaces vault-only references with public links or named internal methods. The Italian source note remains unchanged.*
