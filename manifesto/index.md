---
layout: default
wide: true
title: "Testing Is the Product"
description: "Nine principles for building useful, measurable and dependable private AI."
redirect_from:
  - /technologies/
  - /technologies/index.html
  - /architectures/
  - /architectures/index.html
  - /technologies/cpu-gpu-memory-topology
  - /technologies/cpu-gpu-memory-topology.html
  - /architectures/cpu-gpu-memory-topology.html
  - /manuals/
  - /manuals/index.html
  - /papers/testing-is-the-product.html
  - /foundations/
  - /foundations/index.html
  - /foundations/testing-is-the-product
  - /foundations/testing-is-the-product.html
  - /manifesto/testing-is-the-product.html
---

# Testing Is the Product

Private AI earns its place by solving a real problem. Until a system has been tested on the real work, nobody knows whether it does: the evidence that it works is what we deliver, together with the system. These are the principles that guide what we build, measure and deliver at Dielabs.

## 1. Start with the problem.

Before choosing a model or buying hardware, understand the work: who needs it, what a good result looks like, and which constraints matter. The architecture follows from those answers.

*Measured in [The Shifting Bottleneck](/papers/the-shifting-bottleneck.html): the same GPU and model stopped at ~18 concurrent streams with long prompts and served 32 with chat traffic.*

## 2. Measure useful work.

Cost per completed task. Time to a usable answer. Corrections a person still has to make. These connect a benchmark to the reason someone would pay for the system. Publish the conditions behind the numbers so others can judge them.

*Measured in [What It Costs to Be Right](/papers/what-it-costs-to-be-right.html): six models solved about the same share of exercises, at a cost per solved exercise that varied more than tenfold.*

## 3. Run the right Evaluation Loops.

Build a golden test set from real production tasks, with expected outcomes checked by domain experts. Use it to compare alternatives, catch regressions and guide improvements to the system.

A capacity benchmark with synthetic prompts tells you how much the system can carry; only an Evaluation Loop on real tasks tells you whether the answers are good. You need both.

*Measured in [When LLM Evaluation Loops Become Harness Tuning](/papers/when-llm-evals-become-harness-tuning.html): on a real workbook, files that the models' own check had accepted were refused by Excel.*

## 4. Evaluate the system, not the model.

The model is one part. The prompt, the tools, the client and the limits around it decide the result as much as the model does. Test the whole path, from the request to the output the user receives, before comparing models.

*Measured in [When LLM Evaluation Loops Become Harness Tuning](/papers/when-llm-evals-become-harness-tuning.html): the same three models went from 5 successes out of 18 to 30 out of 36 when only the harness and the briefs changed.*

## 5. Choose the smallest model that passes.

Define what good enough means and test for it. A smaller model that reliably completes the task leaves more room for speed, capacity and a simpler system. Increase the size when the evidence calls for it.

*Measured in [What It Costs to Be Right](/papers/what-it-costs-to-be-right.html): models very different in size and price landed in the same range, and size alone did not predict the result.*

## 6. Find the constraint that matters.

The most impressive component may have little influence on the result. Measure where time, capacity or quality is actually lost, then fix the part that changes the outcome.

*Measured in [When Offloading Doesn't Offload](/papers/when-offloading-doesnt-offload.html): tuning three basic settings raised completed requests by 63%; the offloading feature did not raise them at all.*

## 7. Watch the output, not only the machine.

A healthy dashboard does not mean a healthy answer. GPU, memory and latency metrics say whether the system is running, not whether what it writes is good. Monitor output quality with its own checks.

*Measured in [When Good GPUs Produce Bad Tokens](/papers/when-good-gpus-produce-bad-tokens.html): the model looped with a calm GPU and the KV cache at 20%, and no hardware metric showed it.*

## 8. Make control concrete.

Running locally is a choice about where computation happens. Control also requires clear answers about who can access the data, what is retained and what can leave the system. Make those choices explicit and verify them.

## 9. Build for the day after the demo.

A system has to survive updates, failures and the absence of the person who built it. Make it reproducible, recoverable and maintainable. When the model, prompts or software change, repeat the tests that earned your trust.

*Measured in [When Good GPUs Produce Bad Tokens](/papers/when-good-gpus-produce-bad-tokens.html): fixing one problem exposed the next: once the loop was gone, the text degraded far below the context window.*

---

The experiments behind these principles are in [Papers](/papers/). The methods we use to apply them are in [Frameworks](/frameworks/).
