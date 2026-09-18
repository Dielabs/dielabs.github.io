---
layout: default
title: "Agentic Systems for Inference Infrastructure People"
---

# Agentic Systems for Inference Infrastructure People

*From the stateless model to the agentic harness.*

## Preface

This manual has one precise goal: to take the reader from "I know inference but not agents" to an operational understanding of how a modern agentic harness is built, and of what it does to the serving workload.

**What the reader is assumed to know:**

- How an LLM works at inference level: prefill, decode, KV cache, tokenization
- Serving concepts: batching, throughput vs latency, observability
- Classic sizing patterns (Little's Law, queueing)
- The two capacity-per-replica metrics used throughout: one hardware-anchored, measured in closed loop; one SLO-anchored, measured in open loop

**What is not assumed:**

- What an agent is, why it exists, how it is built
- What tool use, function calling and MCP are
- What Skills and subagents are
- How these patterns hit the serving workload

**Style**: mid-level senior, technical but not specialist. No over-explanation of fundamentals — KV cache, prefill and decode are used without redefining them — but no assumptions about agentic patterns either.

This is the foundations volume: the mechanics that do not age. Products, protocols, the state of the art on agent-to-agent, and the observability tooling landscape live in a separate technology companion, which is not published.

---

## The model as a pure function

**Key points.**

- An LLM is a **stateless** function mapping context to a probability distribution over the next token
- Everything we call an "agent" is **built on top of** that primitive, not inside the model
- An agent's state lives outside the model, in the code that orchestrates the calls
- Practical implication: there is no such thing as "the model that remembers". Every request is self-contained

### The model as a function

Formally, an LLM at inference time is simply:

```
f(context) -> P(next_token | context)
```

where `context` is a sequence of tokens (the prompt plus everything generated so far), and the output is a probability distribution to sample the next token from. The decoding loop repeats this until a stop token is emitted or a limit is exceeded.

**Three properties that matter for agents:**

1. **Stateless**: the model does not "remember" previous calls. If you want it to remember something, you have to pass it in the context.
2. **Deterministic up to sampling**: given the same context and the same sampling parameters (temperature, top-p, seed), the output is reproducible. The variation comes from the sampler, not from the model.
3. **Context-bounded**: there is a hard limit on context length (8K, 200K, 1M tokens depending on the model). Beyond that limit, the model cannot see.

### Why this matters for agents

Whenever you hear "the agent decided to...", "the agent remembers...", "the agent learned...", translate:

- "The agent decided" — the code called `f(context)` and interpreted the output as a decision
- "The agent remembers" — the code put something into the `context` of the next call
- "The agent learned" — almost never true at inference time; the model does not learn during a session

This is the most important starting point in the whole manual: **all of an agent's state lives in the code that orchestrates the calls to the model, not in the model**.

### Infrastructure consequences, in advance

If the agent's state lives in the context, and the context has to be re-transmitted on every call, then:

- The **prefill** of each iteration includes all the accumulated context
- The **KV cache** can potentially be reused if iterations share a common prefix
- The **token budget** is a scarce resource that grows monotonically in a naive agent

These three points become the thread running through the serving chapter.

### In short

An agent is not a new kind of model. It is external orchestration of a stateless model. All the apparently emergent intelligence — planning, memory, tool use — lives in the code that manages the context, not in the weights.

---

## Tool use, or how a stateless model does things

**Key points.**

- Tool use is not an extension of the model, it is **structured generation plus external execution**
- The model emits JSON describing an action; an external runtime executes it and re-injects the result into the context
- Function calling, tool calling and action calling are different names for the same mechanism
- From the serving point of view, every tool call is a separate request with a growing context

### The problem tool use solves

On its own, an LLM cannot:

- Read a file from the filesystem
- Query a database
- Call a REST API
- Execute code

All the model can do is **generate tokens**. So how do you give it the ability to act? The answer is elegant and almost trivial: you teach the model to **generate tokens that describe an action**, and you delegate execution to an external runtime.
### The mechanism, concretely

A call with tool use is structured like this.

**Step 1 — declare the tools in the system prompt:**

```json
{
  "tools": [
    {
      "name": "read_file",
      "description": "Read a file from disk",
      "input_schema": {
        "type": "object",
        "properties": {
          "path": {"type": "string"}
        },
        "required": ["path"]
      }
    },
    {
      "name": "list_directory",
      "description": "List files in a directory",
      "input_schema": {
        "type": "object",
        "properties": {
          "path": {"type": "string"}
        }
      }
    }
  ]
}
```

This schema is serialized and injected into the prompt. The model "sees" the list of available tools as text, with some special tagging depending on the provider.

**Step 2 — the user asks a question:**

> "Read the file /etc/hosts and tell me what it contains"

**Step 3 — the model emits a tool call, not text for the user:**

```json
{
  "type": "tool_use",
  "id": "tu_01ABC",
  "name": "read_file",
  "input": {"path": "/etc/hosts"}
}
```

Note: this is NOT text going to the user. It is structured output that the runtime intercepts.

**Step 4 — the runtime executes the tool.**

The agent loop code recognises the tool_use block, calls the Python function (or whatever it is) `read_file("/etc/hosts")`, and gets the content back.

**Step 5 — the result is re-injected into the context:**

```json
{
  "type": "tool_result",
  "tool_use_id": "tu_01ABC",
  "content": "127.0.0.1 localhost\n::1 localhost\n..."
}
```

**Step 6 — a new call is made to the model** with the updated context (original request + tool_use + tool_result), and the model finally generates text for the user:

> "The /etc/hosts file contains the standard localhost mapping..."

### Two calls, not one

This is the point that changes everything from the infrastructure side.

**Without tool use**: 1 user message, 1 call to the model, 1 response. The classic traffic pattern.

**With tool use**: 1 user message, N calls to the model interleaved with tool executions, 1 final response.

And N can be large. On a complex task, a coding agent can make 50 or more tool calls before answering the user. Each one is a separate call to the serving stack, with a context that grows every time.

### Anatomy of a growing context

Iteration 1:

```
[system prompt + tools] + [user msg]
```

Iteration 2, after the first tool call:

```
[system prompt + tools] + [user msg] + [assistant: tool_use] + [tool_result]
```

Iteration 3:

```
[system prompt + tools] + [user msg] + [assistant: tool_use_1] + [tool_result_1] + [assistant: tool_use_2] + [tool_result_2]
```

And so on. The **prefix is stable** — system prompt, tools, history up to a point — but it grows with every iteration.

### Serving implications, in advance

- **Prefix caching becomes critical**: if the KV cache is preserved, iterations 2 through N pay prefill only on the delta. Without a prefix cache, every iteration re-prefills everything.
- **Latency stacking**: total latency as perceived by the user is the sum of the latencies of the N calls plus the tool execution time. Even if each call is fast, N=20 calls means 20 times the base latency.
- **Throughput patterns**: an agent is not "1 user = 1 request". It is "1 user = N sequential requests with a growing context". Classic sizing formulas — requests per second through Little's Law — get the estimate wrong.

### In short

Tool use adds no capability to the model. It adds an external execution loop around a model that remains stateless. The model emits structured JSON describing an action, a runtime executes it, the result goes back into the context, and a new call is made. From the serving point of view, an agentic session is a sequence of calls with a shared prefix that grows monotonically.
---

## The gather, act, verify loop

**Key points.**

- The agent loop is the minimal primitive: gather context, plan, act, verify, repeat
- It exists to resolve the **mismatch between a finite context and an open-ended task horizon**
- It is not "ReAct" in the academic sense — it is a much tighter operational design pattern
- The verify step is the difference between an agent that works and one that diverges

### Why the loop exists

The key question: if I can pass all the context in a single call, why do I need a loop?

Because **you do not know in advance what the context will need to contain**.

Take this task: "Find the bug in my code and fix it."

With a single call, you would have to preload:

- The entire codebase, potentially millions of tokens
- All the documentation
- All the logs
- All the tests

Even if the model had an infinite context window, this would be **wasteful**: the bug is in 3 lines of 1 file. 99.99% of the context would be noise.

The loop solves this: the model **explores the context progressively**, materialising only what is relevant in light of what it has already found.

### The minimal loop

```python
def agent_loop(user_request):
    context = initial_context(user_request)
    while not done(context):
        action = model.generate(context)        # gather + plan + act
        if action.is_tool_call:
            result = execute_tool(action)
            context = append(context, action, result)
        elif action.is_final_answer:
            return action.text
        elif action.is_question:
            answer = ask_user(action.text)
            context = append(context, action, answer)
    return None
```

This is the **minimal loop**. Every modern agent is a variation on this theme, with additional constraints.

### The stack that runs the loop

A natural operational question for anyone coming from serving: **where does this loop actually run?**

An agent in production is not a single monolithic process. It is a three-layer stack, and each layer runs on different resources:

```
        +--------------------------+
        |   User (Frontend / UI)   |
        +------------+-------------+
                     |
        +------------v-------------+
        |      AGENT RUNTIME       |   <- CPU, lightweight
        |  - Tool call parser      |      (LangGraph, Semantic Kernel,
        |  - State management      |       Claude Agent SDK, custom)
        |  - Connector routing     |
        |  - Loop control          |
        +------+-----------+-------+
               |           |
       +-------v--+   +----v---------------+
       |   LLM    |   |  Tools / connectors|
       | serving  |   |  (MCP, REST, DB,   |
       | (vLLM)   |   |   web search, ...) |
       |   GPU    |   |   CPU + network    |
       +----------+   +--------------------+
```

**Three layers, three resource domains:**

1. **Agent runtime** (CPU): orchestrates the loop, parses the tool calls the model emits, routes to connectors, manages session state. It is the procedural brain sitting above the model. Light on compute, but it adds latency overhead on every round trip.

2. **LLM serving** (GPU): the inference engine proper. It runs prefill and decode on every iteration. This is where prefix caching, KV cache and all the traditional inference sizing live.

3. **Tools and connectors** (CPU + network): HTTP calls to external APIs, database queries, MCP servers, web search. Often dominated by network latency or by external backends — a crucial point we return to in the serving chapter.

**Three operational observations for serving people:**

- The **runtime does not consume GPU**. It sits on standard CPU, next to the frontend or as a dedicated microservice. Sizing-wise it is negligible.
- The **two arrows towards LLM and tools** are separate latency points. Optimising one without measuring the other is blind.
- **Model-runtime compatibility** is a critical point that comes back later: the model must be fine-tuned to emit tool calls in the format the runtime parser knows how to read. Without that, the loop breaks silently.

### What an agent loop is not

**Academic ReAct**: "Reasoning + Acting", Yao et al. 2022. A more rigid scheme, where every step has a "Thought:", "Action:", "Observation:" block. It works, but it is verbose and tied to a single paper. Modern agents, post-2024, have moved past it, using structured tool calling instead of markdown parsing.

**Chain-of-thought**: the model reasons out loud before answering, but in a **single call**, with no tool use. Useful, but not a loop.

**A single function call**: one call, one tool, one result, one answer. That is not a loop, it is a two-call pipeline.

A **real agent loop** has three characteristics:

1. Iteration **not bounded in advance** — it can take 1 step or 100
2. **Dynamic decisions** about what to do next, based on accumulated context
3. An **explicit or implicit verify step** to avoid divergence

### The verify step, the most underrated part

In a naive loop, the model can:

- Run a tool, get an error, ignore it and carry on
- Modify a file and never check whether the modification is correct
- Claim to have completed a task that is in fact partial

The verify step forces the model to **close the loop** before moving on. From the leaked Claude Code system prompt:

> "ONLY mark a task as completed when you have FULLY accomplished it. If you encounter errors, blockers, or cannot finish, keep the task as in_progress."

That is not a suggestion, it is an operational constraint. Without it the loop diverges: the model piles up uncorrected errors and arrives at the end convinced it has done things it has not done.

**Three forms of verification in real systems:**

1. **Self-verification**: the model interrogates itself — did I really do what was asked? Reliable only when the prompt forces it.
2. **Tool-based verification**: after an action, a tool is called to check its effect — after `write_file`, call `read_file` to confirm.
3. **External verification**: another model, or a deterministic test, checks the output.

Production coding systems mostly use the first two. The TodoWrite pattern is an externalised form of the first.

### The full loop, with a plan

In practice the minimal loop becomes:

```python
def agent_loop(user_request):
    context = initial_context(user_request)
    plan = None

    while not done(context):
        # Phase 1: gather context
        # (only if the model decides to explore)

        # Phase 2: plan, explicit or implicit
        if not plan or plan_needs_update(context):
            plan = model.generate_plan(context)
            context = append(context, plan)

        # Phase 3: act
        action = model.generate(context)
        result = execute(action)
        context = append(context, action, result)

        # Phase 4: verify
        if action.was_state_changing:
            verification = model.verify(context, action, result)
            if not verification.ok:
                context = append(context, verification)
                # the loop retries or changes strategy

    return final_answer(context)
```

### What the loop actually solves

**Agent loops resolve the structural mismatch between two things:**

1. **The model's finite context window** — large, but bounded
2. **The open-ended, not-known-in-advance horizon** of the task

In a single call, you have to preload every piece of context that *might* be needed. With a loop, you **materialise the relevant context just in time**, as a function of what you have discovered so far.

Put differently: the loop turns a **context preloading** problem, unsolvable for non-trivial tasks, into a **context exploration** problem, solvable with the right gathering strategy.

Everything else — Skills, subagents — is a strategy built on top of this primitive, not an independent innovation.

### In short

The loop exists because you cannot preload all the context you will need. You explore it progressively. The loop turns an unsolvable problem, load everything, into a solvable one, load what is needed when it is needed. Skills and subagents are optimisations of that exploration, not independent innovations.