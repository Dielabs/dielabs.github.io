---
layout: default
wide: true
title: "Testing Is the Product"
description: "Seven principles for building useful, measurable and dependable private AI."
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

Private AI earns its place by solving a real problem. These are the principles that guide what we build, measure and deliver at Dielabs.

## 1. Start with the problem.

Before choosing a model or buying hardware, understand the work: who needs it, what a good result looks like, and which constraints matter. The architecture follows from those answers.

## 2. Choose the smallest model that passes.

Define what good enough means and test for it. A smaller model that reliably completes the task leaves more room for speed, capacity and a simpler system. Increase the size when the evidence calls for it.

## 3. Find the constraint that matters.

The most impressive component may have little influence on the result. Measure where time, capacity or quality is actually lost, then fix the part that changes the outcome.

## 4. Measure useful work.

Cost per completed task. Time to a usable answer. Corrections a person still has to make. These connect a benchmark to the reason someone would pay for the system. Publish the conditions behind the numbers so others can judge them.

## 5. Make control concrete.

Running locally is a choice about where computation happens. Control also requires clear answers about who can access the data, what is retained and what can leave the system. Make those choices explicit and verify them.

## 6. Build for the day after the demo.

A system has to survive updates, failures and the absence of the person who built it. Make it reproducible, recoverable and maintainable. When the model, prompts or software change, repeat the tests that earned your trust.

## 7. Run the right evals.

Build a golden test set from real production tasks, with expected outcomes checked by domain experts. Use it to compare alternatives, catch regressions and guide improvements to the system.

---

The experiments behind these principles are in [Papers](/papers/). The methods we use to apply them are in [Frameworks](/frameworks/).
