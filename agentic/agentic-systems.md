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
---

## Context engineering, the real problem

**Key points.**

- In a naive loop, the context grows **monotonically** on every iteration
- At some point it saturates the window, and everything breaks
- "Context engineering" is the discipline of **choosing what goes in, what stays, what comes out**
- It is the central problem of modern agents, and what separates a toy system from a production-grade one

### Anatomy of the context in an agent loop

At iteration N of an agent loop, the context holds:

```
+--------------------------------------+
|  System prompt                       |  <- stable, large (5-50K tokens)
+--------------------------------------+
|  Tool descriptions                   |  <- stable, medium (1-10K tokens)
+--------------------------------------+
|  Original user message               |  <- stable, small (100-1K)
+--------------------------------------+
|  Assistant turn 1: thought + action  |  <- grows
|  Tool result 1                       |  <- grows
|  Assistant turn 2: thought + action  |  <- grows
|  Tool result 2                       |  <- grows
|  ...                                 |
|  Assistant turn N-1: action          |
|  Tool result N-1                     |
+--------------------------------------+
|  Assistant turn N: thought (current) |  <- being generated
+--------------------------------------+
```

**Two crucial observations:**

1. **The top part is stable**: system prompt, tools and user message do not change between iteration 1 and iteration N. This is what makes prefix caching possible.

2. **The bottom part grows**: every iteration adds a thought, an action and a tool result. And tool results can be large — a `read_file` on a 5,000-line file is roughly 50K tokens for that single result.

### The monotonic growth problem

A concrete example. Task: "Refactor this Python module to use async/await instead of threading."

- Iteration 1: `list_directory(".")` sees 50 files
- Iteration 2: `read_file("module.py")`, 3,000 lines, about 30K tokens
- Iteration 3: `grep("threading", ".")`, 80 matches, about 5K tokens
- Iteration 4: `read_file("utils.py")`, 1,500 lines, about 15K tokens
- Iteration 5: `read_file("worker.py")`, 2,000 lines, about 20K tokens

In 10 iterations you have accumulated roughly 150K tokens of **historical tool results**, which the model may no longer need for the next steps. But they are there, they occupy space, and every new call re-prefills them.

At some point you either:

- **Saturate the context window** — 300K, 1M, whatever the limit is
- **Degrade model performance** — the lost-in-the-middle problem: models remember the beginning and the end of a context better than its middle
- **Blow up the cost** — per-token pricing, expensive prefill

### What does not work

**Naive sliding window**: cutting the oldest tokens past a threshold. It breaks because the system prompt and the original request are at the beginning — cut them and the agent no longer knows what it was doing.

**Random truncation**: removing blocks at random. It breaks causal coherence — a tool_use without its tool_result is a format error.

**Relying on a huge context window**: scaling hardware until the context reaches 10M. It does not scale in cost, and the model degrades in the middle anyway.

### What does work

The term **context engineering** emerged in 2024-2025 as a discipline. It is the art of answering three questions on every iteration of the loop:

1. **What has to go in** the context for this iteration? (gather)
2. **What can come out** without compromising the task? (eviction)
3. **What gets compressed** versus kept verbatim? (compaction)

Each answer is a context engineering strategy. The four main ones:

| Strategy | Question it answers |
|---|---|
| **Skills** | What goes in (lazy loading) |
| **Subagents** | What does NOT go in (context isolation) |
| **Summarization** | What gets compressed |
| **External memory** | What gets moved outside |

### Context engineering is the real core skill

An uncomfortable truth of the industry in 2026: **the model is close to a commodity**. Claude Sonnet 4.6, GPT-5, Gemini 3 are all good enough for most tasks. What separates a production-grade agent from a toy is not which model it uses, but **how well it manages the context**.

That is also why Anthropic has published explicit engineering posts on context engineering, and opened up tooling like MCP. They are moving the value from the model to the **harness** around the model.

### Serving implications, in advance

For anyone sizing infrastructure:

- **Context size distribution**: in an agentic workload, context size is NOT a fixed variable. It is a long-tailed distribution — some tasks close in 3 steps, others in 50. You measure percentiles, not averages.
- **Prefix caching becomes layered**: the system prompt is 99% cacheable. The Skills loaded depend on the task. The conversation history is almost always a miss.
- **Compaction events**: when an agent runs automatic summarization, it is a prefill tax on a long context followed by the decode of a summary. A bursty traffic pattern. It has to be modelled.

### In short

The real problem with agents is not model power, it is context engineering. In a naive loop the context grows monotonically until it breaks the window or degrades the model. Skills, subagents, summarization and external memory are the four strategies production-grade systems use to manage that growth. That is where it is won or lost, not in the weights.
---

## Four strategies for context management

**Key points.**

- Four main strategies in production-grade systems: **Skills**, **subagents**, **summarization**, **external memory**
- Each answers a different context engineering question
- They are **complementary, not alternatives**: a mature coding agent uses all four
- Understanding these four is understanding 90% of modern agents

### Strategy 1, Skills and progressive loading

**Question answered**: what goes into the context, and *when*?

The problem: if you have 50 different capabilities — editing Word files, handling PDFs, running SQL queries, talking to MCP servers X, Y and Z — you cannot describe them all in the system prompt. It would blow up to hundreds of thousands of tokens.

The solution visible in production systems: **two or three level loading**. The layered structure below is a reasoned reconstruction of the pattern visible in the leaked system prompt and in public Anthropic documentation, not a quotation from an official design document.

**Level 1, the skill manifest**, always in the context:

- Skill name
- A short description, one or two sentences
- Semantic triggers, describing when to activate it
- For example: `pdf-skill: Use when working with .pdf files. Triggers: read PDF, extract text from PDF, fill PDF form.`

This costs a few dozen tokens per skill. You can have a hundred without trouble.

**Level 2, the skill body**, loaded on demand:

- Detailed instructions
- Examples
- References to code files
- Schemas for the tools specific to that skill

Loaded **only if the model decides to activate the skill** based on the triggers. It can run to thousands of tokens, but you pay for it only when it is needed.

**Level 3, referenced resources**, loaded on a tool call:

- Utility code files
- Assets such as templates or datasets
- In-depth documentation

Loaded only when the skill body explicitly asks to read them.

**Why it works:**

1. **Context economy**: you pay tokens only for relevant capabilities
2. **Modularity**: adding a skill does not require touching the main system prompt
3. **Composition**: several skills can activate together when their triggers match

**What Skills are not:**

- They are not fine-tuning. They do not modify the weights.
- They are not plugins. They do not execute arbitrary code outside the agent runtime.
- They are not RAG. There is no vector similarity search — the match is based on language semantics in the prompt.

### Strategy 2, subagents and hierarchical isolation

**Question answered**: what does NOT enter the parent's context?

The problem: some tasks require exploring a lot of context to arrive at very little information. "Find all the files that import `requests`" — you might read 100 files to discover that only 5 import it. Do those 100 files stay in the context?

Without subagents: yes. They pollute the context for the rest of the task.

With subagents:

```
PARENT AGENT
  context: [system + tools + user_msg + task_so_far]
  |
  calls subagent("explore", "find files importing requests")
  |
  SUBAGENT (separate context)
    context: [system_subagent + task]
    reads 100 files
    accumulates 200K tokens in its own context
    produces a final synthesis: "5 files: a.py, b.py, c.py, d.py, e.py"
  |
  back to the PARENT:
    added to the parent context: ONLY the synthesis (~50 tokens)
```

The 200K tokens the subagent read **never travel back up to the parent**. The parent sees only the distillate.

**The tradeoff, stated explicitly in the leak:**

> "For broader codebase exploration and deep research, use the Task tool with subagent_type=Explore. **This is slower than calling Glob or Grep directly** so use this only when a simple, directed search proves to be insufficient..."

Anthropic admits it: the subagent is **slower**. It goes through a second model, adds latency, and costs more tokens in total. **But it protects the parent's context**, which is the scarcer resource.

This is a fundamental design tradeoff of modern agents: **trade latency for context economy**.

**Types of subagent in the leak:**

- `general-purpose`: generic multi-step search
- `Explore`: codebase exploration, at three depths — quick, medium, very thorough
- `Plan`: software architect for planning complex tasks
- `Bash`: shell command execution
- `statusline-setup`: a very specific configuration job

They are functionally identical to the parent — LLMs with tools — but with a **reduced tool set** and a **specialised prompt**. The specialisation is in the prompt, not in the model.

### Strategy 3, summarization and lossy compression

**Question answered**: what gets compressed when the context grows too far?

The problem: even with Skills and subagents, a long session can accumulate context beyond any reasonable limit. At some point the model itself degrades.

The solution: **automatic summarization** of the historical context, past a threshold.

From the leak:

> "The conversation has unlimited context through automatic summarization."

The leak confirms the mechanism exists; the implementation details that follow — the threshold, the criteria for choosing which sections to compress — are a **plausible reconstruction** of the pattern, not officially stated by Anthropic.

That sentence is enormous. Anthropic does NOT rely on large context windows: it runs **automatic compaction** as the context grows.

**How it works, as a general pattern:**

```python
def maybe_compact(context):
    if token_count(context) > THRESHOLD:
        # Identify the compactable sections
        # (typically: old tool_results, successfully completed turns)
        sections_to_compact = identify_compactable(context)

        # Call the model to summarize
        summary = model.summarize(sections_to_compact)

        # Replace them in the context
        context = replace(context, sections_to_compact, summary)
    return context
```

**What makes compaction good:**

1. **It preserves causality**: never compress across a tool_use / tool_result pair
2. **It preserves the user-agent contract**: the original request stays verbatim
3. **It preserves relevant intermediate results**: if a file was created at turn 3, the agent has to remember it exists
4. **It compresses aggressively**: exploration tool results — reads, searches — are excellent compression candidates

**The infrastructure cost of compaction.** A compaction event is:

1. An **expensive prefill** over the large context, as input to the summarization
2. A **decode** of the summary
3. A **new prefill** at the next iteration, over the compacted context

It is a **periodic tax** that has to be modelled in sizing. It is not a normal request.

### Strategy 4, external memory

**Question answered**: what can live outside the context, reachable on demand?

The problem: even the three strategies above are not enough for very long tasks with structured state — "implement 20 features following this plan".

The solution: **move the state into persistent storage**, reachable through tools.

The TodoWrite pattern in the leak is exactly this:

```
[the model writes a todo list at the start of the task]
  TodoWrite([
    "Fix authentication bug",
    "Add unit tests",
    "Update docs"
  ])

[the state lives in an external file or structure, not in the context]

[as the model works, it updates the state through tools]
  TodoUpdate(1, status="completed")
  TodoUpdate(2, status="in_progress")

[when needed, it reads the state back]
  TodoList() -> [{id:1, status:"completed"}, {id:2, status:"in_progress"}, ...]
```

**Advantages:**

1. The list is **authoritative**, independent of the model's memory
2. It survives compaction, because it is in storage, not in the context
3. It is visible to the user, which is a UX bonus
4. It forces the model to be explicit about the plan, which fights divergence

**Extensions of the pattern.** The same idea applies to:

- **Memory** (Mem0, Letta, custom): persistent user facts across sessions
- **Working files**: a scratchpad where the agent writes intermediate results
- **Vector DB**: an external knowledge base reachable through similarity search

In every case the principle is the same: **the agent's state does not live only in the context, it lives in external storage reachable through tools**.

### How they combine

A modern agent loop uses all four together:

```
TASK START
  |
[skills manifest always in the context]
[skill body loaded on demand from the triggers]
  |
[iterative loop]
  |- complex task?        -> generate todos in external memory
  |- heavy exploration?   -> delegate to a subagent (context isolation)
  |- context > threshold? -> trigger summarization (compaction)
  |- continue
  |
TASK END
```

All four exist to answer the central problem of the previous chapter: **how to manage a context that would otherwise grow monotonically**.

### In short

Modern agents manage context with four complementary strategies. Skills load capabilities progressively (what goes in). Subagents isolate exploration in separate contexts (what does not go in). Summarization compresses historical context (what gets compacted). External memory moves structured state into storage (what leaves the context). Production coding systems use all four. Understanding that combination is understanding 90% of production-grade agent design.
---

## MCP and the tool ecosystem

**Key points.**

- **MCP (Model Context Protocol)** is an open protocol for connecting agents to external tools
- Published by Anthropic in November 2024, adopted across the ecosystem through 2025-2026
- It is not "tool use" — it is the **standardised wire protocol** between an agent and a tool server
- It is agent-to-tool, not agent-to-agent, and that distinction carries weight

### The problem MCP solves

Without MCP, every agent talks to tools in its own proprietary way:

```
Claude Code -> custom tool API
ChatGPT     -> OpenAI plugins API (deprecated)
Cursor      -> custom MCP-like protocol
Windsurf    -> ...
```

For a tool provider — GitHub, Notion, Slack — that means **N separate integrations, one per agent**. Unsustainable.

MCP is an open protocol standardising:

- **Tool discovery**: how the agent finds out which tools exist
- **Tool invocation**: how the agent calls one
- **Resource access**: how the agent reads structured data — files, databases, query results
- **Prompts**: how the server can hand prompt templates to the agent

In practice: an **MCP server** exposes tools and resources, and any **MCP client** — any agent — can consume them.

### Minimal MCP architecture

```
+------------------+     JSON-RPC over     +--------------------+
|  Agent (client)  | <---- stdio / SSE --> |   MCP server       |
|  (Claude Code)   |                       |  (Notion, GitHub,  |
+------------------+                       |   custom, ...)     |
                                           +--------------------+
```

**Supported transports:**

- **stdio**, for local servers spawned as a process by the agent
- **SSE / HTTP**, for remote servers

**Key protocol operations:**

- `tools/list`: discover the available tools
- `tools/call`: invoke a tool with parameters
- `resources/list`: discover readable resources
- `resources/read`: read one
- `prompts/list` and `prompts/get`: prompt templates

### Why MCP matters for agentic demand

Three points.

**1. It decouples the agent from the tool ecosystem.** A company can build an MCP server once, and every agent can use it. That reduces vendor lock-in and accelerates the ecosystem.

**2. It turns tool integration into an infrastructure problem.** An MCP server is a service like any other: deployment, monitoring, scaling, SLA. Familiar ground for anyone from a solution architecture or DevOps background.

**3. It is not agent-to-agent.** MCP is **agent to tool**. The MCP server is NOT an agent. It is a provider of structured capability. It has no autonomy, no loop, no decisions of its own.

### Infrastructure implications

For anyone sizing infrastructure:

- **MCP servers are workload components**: an agent using N MCP servers makes N kinds of call, often in parallel. It becomes a distributed system with multiple dependencies.
- **A shared latency budget**: total perceived latency is LLM latency plus MCP call latency plus tool execution. Every MCP server is a latency point and a failure point.
- **Authentication and secrets**: MCP servers often reach sensitive systems — databases, enterprise APIs. Credential management, audit and RBAC become part of the agentic infrastructure.
- **Cold start**: local MCP servers are spawned at boot. Remote ones scale like any web API.

For a sovereign AI platform, offering a curated **MCP server registry** — certified servers, monitored, with governance built in — is probably an architectural asset, not just a technical integration.

### In short

MCP is the standardised wire protocol between agents and tools. It is not tool use, it is the open and interoperable version of tool use. It turns the N×M problem — N agents times M tool providers — into N+M. And it is agent-to-tool, not agent-to-agent: the foundation for a capability ecosystem, not a protocol for composing agents.
---

## Reasoning policy, or how the parent decides

**Key points.**

- The brain deciding what to do inside an agent loop is, in the systems we can observe today, **almost entirely prompt-driven**, with hardcoded guardrails for sensitive actions
- Loop termination is a mix of criteria — completion signal, iteration budget, verify failure, cost budget — typically all present at once
- Degenerative looping is a real pathology, and the production mitigations are pragmatic, not sophisticated
- For an enterprise customer, **reasoning policy is a configurable surface**, not a fixed property of the system

### Who decides which tool or subagent to invoke

On every iteration, the agent chooses between:

- Primitive tools (read_file, grep, write, bash)
- A subagent
- Loading a Skill
- Asking the user
- Closing the loop with a final answer

In the production systems we can observe, **the decision is driven by the prompt**, not by an external policy engine. The prompt carries natural-language directives such as "prefer the Task tool for searches across many files, to reduce context usage" — the model reads them and applies them. For someone coming from serving, the operational observation is simple: the reasoning policy is not a separate runtime component, it is material travelling inside the context on every request, with all the caching and prefill implications that follow.

In mature systems it is **hybrid**: the prompt as primary logic, plus hard hardcoded guardrails for actions that need strong guarantees — a filesystem permission system, a sandbox for code execution, a blocklist of destructive operations. Hard guardrails are never delegated to the model.

### Termination, or when the loop closes

A loop with no closing criterion diverges. Four criteria are typically present together:

1. **A task-complete signal from the model**: it emits a final answer — text, not a tool call — and the runtime reads that as the end. The primary criterion, but fallible in both directions: closing early, or carrying on past the point of usefulness.

2. **An iteration budget**: a hard cap on the number of iterations, say 50. Crude but effective as a safety net.

3. **Repeated verify failure**: if verification fails N times in a row, escalate or abort.

4. **A cost or token budget**: a hard cap on total tokens per task. Common in enterprises with cost control.

The important part: these four are **configurable parameters**, not constants. A customer can tune them per use case.

### Degenerative looping

The pathology: the model retries the same action, or minimal variations of it, without making progress. Typical causes are a context that already holds the failed attempts without the model recognising the pattern, a prompt that never explicitly encourages a change of strategy, and the absence of external detection.

Production-grade mitigations, all pragmatic, no magic:

- **An explicit prompt**: "if you have tried the same approach three times without success, stop and change strategy or ask the user"
- **Aggressive verification after N failures**, forcing escalation
- **Cost-aware termination**: if the task has consumed X tokens with no measurable progress, abort

Nothing exotic. The difference between a system that works and one that does not is applying these **together and with discipline**, not inventing new ones.

### A concrete enterprise example

To anchor all this, take a realistic scenario: a customer wants an internal AI copilot for its developers.

**Reasoning policy that changes per user profile:**

| Profile | Max iterations | Cost cap | Escalation | Hard guardrail |
|---|---|---|---|---|
| Senior R&D developer | 100 | $5/task | On request only | Sandboxed bash, no production credentials |
| Junior developer | 30 | $1/task | After 3 verify failures | Sandbox, read-only on critical repositories |
| CI/CD operator | 20 | $0.50/task | Always before merge | Tool whitelist, no writes to main |

That table is **not a technical detail**, it is a product argument: the difference between a generic LLM API and a platform that lets you express enterprise policy per user profile.

**What the platform needs in order to implement that grid:**

- Reasoning policy as versioned configuration, not as a hidden prompt
- Telemetry showing where each profile consumes its budget
- An audit log of escalation triggers
- The ability to tune the policy without redeploying the model

### Architectural implications

Three angles.

**1. Reasoning policy is a configurable surface.** Not an internal opacity, but a product parameter the customer tunes per use case and per user profile.

**2. Telemetry on reasoning is crucial for operational debugging.** When an agent misbehaves you need to know: did it loop? did it terminate early? was it stopped by a guardrail? Without dedicated observability the debugging is blind.

**3. Hard guardrails are compliance territory.** A regulated customer — a bank, a hospital, a public administration — cannot accept an agent loop governed entirely by prompts. Hard guardrails exposed as explicit configuration are an enterprise requirement, not a nice-to-have. A sovereign platform has to treat them as a first-class feature.

### In short

An agent's reasoning policy — which tool to call, when to stop, how to avoid degenerative loops — is, in the systems observable today, almost entirely prompt-driven, with hard guardrails hardcoded for sensitive actions. For an enterprise customer the relevant question is not how it works internally, but how it can be configured for different user profiles and use cases. For a sovereign platform, exposing reasoning policy and guardrails as versioned configuration rather than hidden prompts is a significant enterprise differentiator.
---

## What this does to serving infrastructure

**Key points.**

- Agentic workloads **break the classic sizing patterns** of LLM inference
- Four dimensions are missing from the traditional formulas: tool call rate, context growth, compaction frequency, subagent fan-out
- **Prefix caching becomes layered, and critical**
- The **open-loop, SLO-anchored** metric is particularly well suited to this workload

### From chat completion to agentic session

**Classic workload, chat completion:**

- 1 user message, 1 request to the serving stack, 1 response
- Context size: typically under 10K tokens
- Requests per second: directly measurable
- Caching: useful, but marginal — every session has a different system prompt
- Pattern: bursty, but stationary over short windows

**Agentic workload:**

- 1 user task, N requests to the serving stack, where N is 5 to 50 or more
- Context size: grows monotonically within a session, from 10K to 200K and beyond
- Requests per second: varies by session, depending on task complexity
- Caching: **critical** — the prefix cache hit rate determines cost and latency
- Pattern: long-tailed, with highly variable session duration

The jump is structural, not quantitative. Classic sizing metrics — X requests per second, Y average tokens — lose their meaning.

### Enterprise use cases, from patterns to workload

Before the four sizing dimensions, it helps to anchor the conceptual patterns to real use cases. One possible classification:

**1. Internal AI copilot for developers.** Dominant patterns: agent loop with verification, intensive tool use (read, grep, edit), subagents for codebase exploration, prefix cache critical on the system prompt and development-focused Skills. Workload profile: long sessions of dozens of tool calls, context growing to 100-200K, frequent compaction, bursty distribution as a developer works then pauses. Serving implication: throughput is not the bottleneck, p99 latency is. High TTFT breaks the UX. Layered prefix caching is the main efficiency factor.

**2. Ticket analysis and support augmentation.** Dominant patterns: a shorter loop, targeted tool use — knowledge base lookup over MCP, ticket database queries, possibly RAG — a smaller context, and no complex subagents in most cases. Workload profile: short sessions of 3 to 10 tool calls, context usually under 30K, traffic tied to helpdesk working hours. Serving implication: throughput is more predictable, prefix cache hit rate is very high because the system prompt and KB loader are stable, and the bottleneck is often the MCP layer rather than the model.

**3. HR and ITSM workflows with transactional actions.** Dominant patterns: a SHORT loop but with rigid guardrails, because the actions write to systems of record. MCP towards enterprise systems. An explicit verify step before every state-changing action. Workload profile: few tokens, but high overall latency because of the downstream MCP calls, and a very conservative reasoning policy. Serving implication: the model is a small part of the total cost. Observability across the tool chain is critical. Compliance and audit are often the real requirement, not performance.

**4. Hybrid technical or document RAG with an agent loop.** Dominant patterns: a loop with iterative retrieval. The model decides what to search for, gets results through a RAG or vector database tool, reasons, and may search again. Not the classic one retrieval plus one generation, but a multi-step loop. Workload profile: the context grows quickly because every retrieval injects 5 to 20K tokens, so compaction can trigger early. Serving implication: prefix cache hit rate on retrieval results is almost always a miss, since different queries pull different chunks. The bottleneck is prefill, not decode.

**5. Batch code assistant** — mass refactoring, migration, audit. Dominant patterns: long autonomous loops, run overnight or on demand, with aggressive subagent fan-out, a high iteration budget and a high cost cap. Workload profile: throughput-bound, not latency-bound. Maximum context, multiple compactions per task. Serving implication: prioritise total throughput over p99 latency. Batch scheduling works well, and hardware can be time-shared with interactive workloads during quiet hours.

**Summary:**

| Use case | Session | Bottleneck | Cost driver | Reasoning policy |
|---|---|---|---|---|
| Developer copilot | Long | p99 latency | Context growth | Permissive |
| Ticket analysis | Short | MCP downstream | Tool call frequency | Standard |
| HR / ITSM workflow | Short | Guardrails, audit | Compliance overhead | Conservative |
| Hybrid RAG | Medium | Prefill throughput | Retrieval volume | Standard |
| Batch code assistant | Long | Total throughput | Token consumption | Aggressive |

Every row is a distinct workload: different sizing, different policy configuration, different capacity card. This is the mental map that separates "GPUs for AI" from sizing a specific agent workload.
### Dimension 1, tool call rate

In an agentic workload, every user task generates N tool calls, and N varies enormously. The table below is an **order-of-magnitude estimate** drawn from public observation of agentic coding systems, not from official benchmarks. Treat it as a reference for reasoning about the workload, to be validated by direct measurement in a real case:

| Task type | Typical tool calls |
|---|---|
| Q&A on a specific file | 1-3 |
| Refactoring a module | 5-15 |
| Implementing a complex feature | 20-50 |
| Deep debugging or exploration | 30-100+ |

Every tool call is a **separate request** to the serving stack. With an average of 20 tool calls per task and one task per minute per user, the serving rate is **20 requests per minute per user**, not one.

**Sizing implication**: the concurrent-users figure has to be multiplied by the tool call multiplier before estimating required throughput.

```
Effective requests/sec = concurrent users x tasks/min/user x tool calls/task / 60
```

For 100 users doing one task a minute at 20 tool calls each: **33 sustained requests per second**, with peaks at multiples of that.

### Dimension 2, context growth

The context grows monotonically within a session. The curve below is an **illustrative distribution** for a typical coding agent session — absolute numbers depend on model, task and configuration, and have to be measured in the specific case:

```
Iteration 1:   ~20K tokens  (system prompt + tools + skills manifest)
Iteration 5:   ~50K tokens  (a few tool results accumulated)
Iteration 15:  ~120K tokens (more tool results, code read)
Iteration 30:  ~180K tokens (approaching the context limit)
               -- compaction trigger --
Iteration 31:  ~80K tokens  (post-compaction)
Iteration 50:  ~150K tokens (growing again)
```

**Prefill implications:**

- At iteration 30, every new request prefills 180K tokens
- Without prefix caching, that is expensive recomputation
- With prefix caching, you pay prefill only on the **delta**

This changes the throughput formula:

```
Without prefix cache: prefill time is proportional to context_size
With prefix cache:    prefill time is proportional to delta_size
```

For an iteration adding 5K tokens to a 150K context, that is a **30x speedup** on prefill when the prefix cache hits.

### Dimension 3, compaction frequency

When the context passes the compaction threshold — say 80% of the window — the runtime runs summarization:

1. A **large prefill** over the long context, as input to the summarizer
2. A **decode** of the summary, roughly 1 to 5K tokens
3. A **new prefill** over the compacted context at the next iteration

A compaction event is a **tax request**: it pays prefill at maximum context, decodes a summary, and invalidates the prefix cache up to that point.

**Typical observed frequency**: in an hour of intensive agentic work you might see 2 to 5 compaction events. That figure is not an official measurement; it is there to reason about cadence, not as a sizing parameter.

**Implication**: your workload model has to include these tax requests as a separate category. They are not normal — their prefill/decode profile is heavily skewed towards prefill.

### Dimension 4, subagent fan-out

When the parent invokes a subagent, it starts a **sub-session** with its own system prompt, its own tool results accumulated in its own context and invisible to the parent, and several internal iterations of its own loop.

In practice, invoking a subagent can generate several additional requests — call it 5 to 20 — in a short window, in parallel with the main flow.

If the parent's orchestration invokes **several subagents in parallel**, as it can, you get a **fan-out** that multiplies the instantaneous request rate.

**Implication**: the bursting pattern of an agentic workload has far higher parallelism peaks than a chat workload. Systems tuned for chat — steady throughput — can suffer head-of-line blocking in the queue during those bursts.
### Prefill amplification

The four dimensions above describe effects that have a single name in serving systems: **prefill amplification**. It is worth naming explicitly, because it immediately separates the conversation from anyone who only talks about "long context".

```
Classic chat:
  1 prefill  ->  1 decode  ->  user-visible answer

N-step agent:
  prefill1 -> decode1 -> tool -> prefill2 -> decode2 -> tool -> ... -> prefillN -> decodeN
  |--------------------------  N prefills, N decodes  --------------------------|
```

On every iteration the model runs a new forward pass over a context that grows monotonically. The decode stays short — a tool call in JSON, a few hundred tokens — stopping at `stop_reason: tool_use`. Only the last decode produces the final answer for the user.

The input/output ratio per turn runs at typical orders of 10:1, 50:1, up to 100:1 in deep agents, against 2-5:1 for standard chat.

### The bottleneck shifts from memory bandwidth to compute

This is the most important observation in the chapter for anyone coming from serving, and it rarely shows up in generic agent pitches.

- **Prefill is compute-bound**: large GEMMs that saturate the matrix units
- **Decode is memory-bandwidth-bound**: reading the KV cache for every generated token

Direct consequence: a chat workload, decode-heavy, is limited by HBM bandwidth. An agentic workload, prefill-heavy, moves the bottleneck towards **GPU compute**. Operational implications:

- GPUs with high FLOPS do **proportionally better** on agentic workloads than on chat. Their compute advantage expresses itself more fully.
- KV offload to slower tiers — CPU memory, NVMe — becomes **more tolerable** in agentic workloads: short decodes mean fewer KV reads, so less sensitivity to offload bandwidth.
- Sizing that optimises only for bandwidth, the natural instinct when assuming a chat workload, undervalues peak compute when the workload is agentic.

This changes the hardware conversation. "You need compute, not bandwidth" is a strong technical argument, and one rarely heard.

### TTFT times N_steps, the latency that is actually perceived

A direct consequence of prefill amplification: the latency the user of an N-step agent perceives is roughly

```
perceived latency = N x TTFT_per_step + N x tool_latency + final decode
```

An 8-step agent with a 2-second TTFT is **16 perceived seconds**, even with instant decode and fast tools. Which means TTFT is the queen metric in agentic workloads, not TPOT.

Implications for SLOs:

- TTFT under 500ms becomes the realistic operational target for interactive agents
- Under compaction, prefilling 180K or more, the target relaxes to 2-3 seconds
- If the perceived SLO is "task done in under 30 seconds" and the agent takes 10 steps, each step has at most 3 seconds of budget

That calculation belongs in discovery, not at the end of the project.

### Characterising by N_steps

"Agentic" is not a monolithic category. The most operational way to characterise a workload is along the axis of **average N_steps per task**:

| N_steps | Profile | Analogy | Sizing reference |
|---|---|---|---|
| 1 | Chat-tool (single-loop) | RAG | Standard closed-loop chat, light overhead |
| 2-4 | Light agent | ITSM workflow | Moderate prefill amplification, cache useful |
| 5-15 | Full agent | Coding agent | Prefill-driven, cache critical, aggregate KV pressure |
| 15+ | Deep agent | Deep research | Disaggregated serving and KV offload nearly mandatory |

The right posture in discovery:

> "It depends on the average N_steps for your use case. A single-call copilot I size like RAG. A multi-step coding agent is a different exercise. Let us measure N_steps on real traffic before estimating the GPU footprint."

That sentence closes the wrong conversation — how many GPUs do we need for AI — and opens the right one: measure before sizing.
### Workload morphology, what the serving side actually sees

The four dimensions describe the behaviour of a single task. Anyone proposing a sizing has to reason at the aggregate level — how the workload presents itself to the serving cluster under real load. Three recurring operational observations.

**1. The arrival curve is not Poisson.**

In a classic chat workload, requests arrive approximately as a Poisson process, an assumption that holds up the classic queueing formulas. In an agentic workload, a single user activation generates a **correlated burst**: the first request triggers N tool results which trigger M more requests, and so on. Interarrival time between requests of the same session is near zero, milliseconds, while between different sessions it stays Poisson-like.

Operational implication: estimating aggregate throughput as a sum of Poisson processes underestimates the p99 of the queue. A cluster sized on the average goes into overload during correlated bursts.

**2. Context size is bimodally distributed.**

Under real mixed load, the context size of requests arriving at the serving stack tends to be **bimodal**:

- A "low" peak: opening requests of new sessions, around 20-30K — system prompt, tools, skills manifest, user message
- A "high" peak: advanced requests of sessions in progress, 80-200K and up

It is not a normal distribution around the mean. Sizing based on average context is wrong at both peaks: it overestimates memory for the low requests and underestimates prefill for the high ones.

**3. KV cache is the scarce resource.**

In a serving engine such as vLLM, the KV cache is finite GPU memory. Every active session occupies KV cache equal to its context size. In chat workloads with short sessions, turnover is high and the cache frees up frequently. In agentic workloads, an hour-long session steadily occupies 100K or more.

Implication: the number of concurrent sessions the cluster can sustain is limited by the **sum of committed KV cache**, not by theoretical tokens per second. When a customer says "1000 concurrent developers", the first question is: how many sessions are actually active at any moment, and what is the average KV cache footprint?

### Four operational pressures worth naming

When proposing an agentic-ready architecture, there are four pressures worth naming explicitly, because nobody mentions them in generic "AI for enterprise" pitches.

**Pressure 1, tail latency versus throughput.** The agentic workload is naturally sensitive to p99, not to the average. A compaction taking 4 seconds of prefill over 180K tokens is a user-visible event that breaks the flow of work. Systems tuned for aggregate throughput, with aggressive batching, make the tail worse.

**Pressure 2, KV cache eviction policy.** Who gets evicted when GPU memory saturates? In chat workloads, natural LRU works. In agentic workloads, evicting an active session means forcing a large prefill recomputation when that session comes back. LRU-based policies need adjusting, or mitigations: offload to CPU or NVMe, priority for active sessions.

**Pressure 3, cost attribution.** In a cluster shared across N tenants, "who consumed what" is not a trivial question. An hour-long agentic session that fans out into subagents crosses several jobs, several requests, several layers of shared prefix cache. Accurate attribution requires dedicated observability infrastructure.

**Pressure 4, failure isolation.** What happens when a single tool result is anomalous — a 50MB file read in one go? Without context size guardrails, one request can saturate an entire serving slot. You need checks at the tool runtime level, to split or filter before it enters the context.

These four are concrete technical arguments that separate a grounded conversation from a generic pitch. They are also the direct antecedents of the observability metrics in the next chapter.
### Tool latency, where the latency actually lives

One operational observation is worth naming explicitly, because it inverts the usual framing of enterprise AI conversations: **in an agentic workload, the GPU is often not the bottleneck**.

Decomposing the end-to-end latency of a single loop iteration:

```
E2E_step = TTFT + decode_time + tool_latency + runtime_overhead

where typically:
  TTFT             = prefill of this iteration (GPU)          ~50-500ms
  decode_time      = generating the tool call (GPU, short)    ~50-300ms
  tool_latency     = external tool execution (network)        ~100ms-5s
  runtime_overhead = parsing, routing, state mgmt (CPU)       ~10-50ms
```

In real systems, **tool_latency is often the dominant term**. The typical sources in enterprise platforms:

- **External SaaS APIs**: rate limiting, geographic region, large payloads. Typical latency 200ms to 2s per call, with high variance.
- **Vendor throttling**: per-minute quotas and fair use policies that kick in under load.
- **MCP serialization**: marshalling and unmarshalling complex payloads, especially with large structured result sets.
- **Cold database or vector store queries**: the first query after idle, cold cache, embeddings not resident.
- **Auth and token exchange overhead**: OAuth refresh, SSO federation, an mTLS handshake on every call if sessions are not reused.
- **Connection pool exhaustion** towards saturated internal backends.

**The crucial implication for discovery**: when a customer says "our agentic AI proof of concept is too slow", the first question is not how many GPUs they need. It is where the latency is. Measuring TTFT separately from tool latency requires dedicated observability, but it is the precondition for a correct diagnosis.

Put bluntly: a system with excellent prefix caching and p99 TTFT under 500ms, whose MCP tools round-trip to a service in a US region for an Italian customer, will still be slow. That is not a GPU sizing problem. It is a network topology and backend problem.

### Model and runtime compatibility, the silent failure mode

A practical point often discovered only in production, and worth raising in discovery.

For an agent to work, three things have to line up:

1. The **model** must be fine-tuned to emit tool calls in a structured format — special tokens, dedicated JSON, XML tags.
2. The **runtime parser** must know how to interpret that specific format.
3. The **tool definition schema** in the system prompt must be consistent with what the model expects.

The formats are **not universal**: each model family has its own. One uses a specific wrapper for tool use blocks, another a `tool_calls` field with its own structure, another XML tags or JSON depending on the fine-tuning.

**The typical failure mode**, seen in many proofs of concept: the model was fine-tuned to emit tool calls as XML, but the runtime expects JSON in a dedicated field. The parser does not recognise the pattern, so the tool call reaches the user as raw text. The agent "looks stupid": instead of doing something, it narrates what it would do.

**The problem is not the model, it is architectural.** Without an explicit model-runtime compatibility matrix, these mismatches only surface under real load. For a sovereign platform that might offer bring-your-own-model or several sovereign-tuned models, **stating that compatibility matrix explicitly** is a product requirement, not a technical detail.

In discovery, three concrete questions for a customer proposing a custom model: has the model been fine-tuned for function calling? In what format does it emit tool calls? Has the proposed runtime been tested with this specific model? Three questions, three potential sources of silent failure.
### Layered prefix caching

Now the crucial point of agentic infrastructure. The context of an agent session has **layers** of stability:

```
+------------------------------------+
| Main system prompt                 |  <- STABLE across all users
+------------------------------------+
| Tool definitions (base)            |  <- STABLE across all users
+------------------------------------+
| Skills manifest                    |  <- STABLE across all users
+------------------------------------+
| Loaded skill bodies                |  <- VARIES by task
+------------------------------------+
| Original user message              |  <- STABLE for the session
+------------------------------------+
| Iteration 1: action + result       |  <- STABLE from iteration 2 on
+------------------------------------+
| Iteration 2: action + result       |  <- STABLE from iteration 3 on
+------------------------------------+
| ...                                |
+------------------------------------+
| Iteration N: being generated       |  <- VARIES
+------------------------------------+
```

**The optimal caching strategy** exploits that layering. The hit rates below are **reasoned estimates**, not measurements: they depend heavily on the runtime implementation, the serving engine and the real workload pattern.

1. **Layer 1**, system prompt plus tools plus manifest: globally cacheable, expected hit rate very high, near 99% in stable conditions
2. **Layer 2**, skill bodies: cached per skill set, hit rate depends on which skills are popular
3. **Layer 3**, user message: cached per session, hit rate 100% for every iteration after the first
4. **Layer 4**, history: cached per session, hit rate decaying after each compaction

Without layered caching, the KV cache prefix tree is inefficient. With it well designed, a request at an advanced iteration can reach a very high hit rate on the prefix — in well-tuned scenarios above 90% — prefilling only the delta. That is an orienting estimate, always to be validated by direct measurement in the specific deployment.

This layering is a strong design argument: an agent-aware inference platform has to do multi-level prefix caching, not just session-level. It is an infrastructural design choice, not a nice-to-have.

### How this connects to the SLO-anchored metric

Four points, each one a differentiator.

**1. Closed-loop benchmarks measure the wrong thing.** The hardware-anchored closed-loop metric measures maximum throughput at constant saturation. Perfect for chat. For agents it **understates** the real problem: an agent does not generate stationary load, it generates bursts of N requests with growing context.

**2. Open-loop with an SLO is the only honest metric.** The SLO-anchored open-loop metric asks: at what rate can I accept new tasks while keeping TTFT under X and ITL under Y at the 99th percentile? That is exactly what an agent needs — the task is the user-visible unit of work, not the individual request; p99 TTFT captures the worst cases, post-compaction and subagent fan-out; ITL captures streaming quality during decode.

**3. The workload model has to be enriched.** The four missing dimensions become benchmark parameters:

- `tool_calls_per_task`: a distribution, with mean, p50, p95, p99
- `context_growth_curve`: context size as a function of iteration
- `compaction_rate`: events per minute per active session
- `subagent_fanout`: the distribution of fan-out sizes

A synthetic workload for an open-loop agentic benchmark is a **multi-dimensional distribution**, not a single number.

**4. The capacity card includes caching efficiency.** When you produce a capacity card, a key metric is the **layered prefix cache hit rate**: per-layer hit rates, and the effective prefill reduction they produce.

That is what converts "we have X GPUs" into "we can serve Y agent-tasks per second at this SLO". It is the correct translation from hardware to business metric.

### Anticipating the customer's question

The customer asks: "How many GPUs do we need to serve 1000 developers using a CLI agent?"

The wrong answer: "It depends on the model, on average X tokens per second per GPU, so N GPUs."

The right answer:

1. I need the real workload: average tool calls per task, session duration distribution, peak fan-out
2. I need the target SLO: max TTFT, max ITL, at which percentile
3. I need to know the model and its prefill/decode characteristics
4. I need to know whether the infrastructure supports layered prefix caching
5. With those inputs I can build a synthetic open-loop workload and measure
6. The capacity card output answers in units the customer understands: Y agent-tasks per second sustained at p99, with M GPUs of type Z

That is enabling a business decision, not doing arithmetic.
### A worked example, a copilot for an Italian bank

A realistic scenario, composite and not referring to any specific customer: an Italian bank, 600 internal developers, wants a sovereign AI copilot deployable on-premise or in a sovereign cloud. Requirements: EU data residency, an open or sovereign-tuned model, banking compliance with audit and operational restrictions on critical systems.

**Step 1, map the workload.** From the discovery conversation:

- Developers active at any moment: an estimated 150, a quarter of 600
- Sessions active at any moment: around 80 — not everyone active has an agentic session open
- Tasks per active session: one every 5 to 10 minutes, with wide variance
- Tool calls per task: bimodal, a fast cluster at 3-5 and a complex cluster at 20-40
- Cost cap target: no task above 3 euros of inference
- SLO target: TTFT p95 under 2 seconds, even after compaction

**Step 2, estimate the operational pressures.** Applying the patterns above:

- Average aggregate throughput: on the order of 30 to 60 requests per second on the cluster
- Average KV cache footprint per active session: 100-150K tokens
- Burst peaks from subagent fan-out: an estimated 2 to 3 times the average rate, in windows of a few seconds
- Expected compaction events: roughly one every 15 minutes per active session

**Step 3, configure the platform.** Given the banking profile:

- Conservative reasoning policy: maximum 30 iterations, early escalation, extensive hard guardrails with no writes to core banking repositories
- A restricted skill set: only certified Skills with an audit log
- MCP servers limited to approved integrations, no arbitrary ones
- Full observability with 90-day retention for compliance
- KV cache offload to NVMe for paused sessions

**Step 4, the resulting capacity card.** The final output of the open-loop benchmark process:

```
Workload - Banking copilot
Profile: 600 developers, ~150 concurrent active, conservative policy
SLO: TTFT p95 < 2s, ITL p95 < 50ms, task completion p95 < 90s

Capacity: 12 tasks/sec sustained, 28 tasks/sec peak (3s burst)
Hardware: 8x high-end datacentre GPUs (sovereign-deployable)
Effective prefix cache hit rate: 78-85% (layers 1-2 stable, layer 3 variable)
Cost per successful task (p50): EUR 0.34
Cost per successful task (p95): EUR 1.20
Compliance: full audit trail, guardrails active, EU data residency
```

That card gives the bank's CTO three things: a concrete answer to "how many GPUs", a cost structure to budget against, and an operational guarantee with explicit SLOs.

**What must not be promised, and why saying so matters:**

- No absolute numbers on models not measured directly
- No TTFT figure without measuring the customer's real workload
- No prefix cache hit rate guarantee without knowing the organisation's skill usage pattern

That operational honesty is exactly what separates a technically credible proposal from a sales deck.

### In short

Agentic workloads break classic sizing along four dimensions: tool call rate, context growth, compaction frequency, subagent fan-out. The right metric is not requests per second but agent-tasks per second under a TTFT and ITL SLO. Layered prefix caching — system, tools, skills, history — is what decides the real unit cost. The open-loop SLO-anchored metric is designed for exactly this, and produces a capacity card that translates hardware into a business metric.
---

## Agentic observability

**Key points.**

- Without observability on an agent, debugging and tuning are blind — and these workloads are too complex for improvised post-mortems
- Six families of metrics to capture: hierarchical **tracing**, **context**, **cost**, **quality**, **performance**, and **replay and debugging**
- For a sovereign platform, agentic observability is also a **compliance surface**: audit, attribution, governance
- It is natural ground for anyone from infrastructure: the same observability patterns as distributed systems, applied to a new domain

### Why agentic observability differs from LLM observability

Observability on a single LLM prompt is relatively simple: an input, an output, and metrics — TTFT, ITL, tokens in and out, total latency, cost.

Observability on an agent loop is far richer:

- N model calls, with a growing context
- M tool calls, with variable results
- K subagents invoked, each with its own sub-loop
- Compaction events
- Reasoning policy decisions: terminate, continue, escalate
- Internal failure modes: looping, verify failure, timeout

The right analogy is not "the log of a REST API" but "the distributed trace of a microservice with N hops". Familiar patterns for anyone from the cloud-native world.

### Family 1, hierarchical tracing

**What it captures**: the whole execution tree of a task, from user prompt to final output.

**Conceptual model**: the same as OpenTelemetry, applied to agents. Hierarchical spans:

```
Trace: task_completion (user: "refactor module X")
+-- Span: parent_agent_loop
    +-- Span: llm_call (iter 1, prefill 5K, decode 200)
    +-- Span: tool_call (read_file)
    +-- Span: llm_call (iter 2, prefill 5.5K, decode 150)
    +-- Span: tool_call (Task -> subagent_explore)
    |   +-- Span: subagent_loop (Explore)
    |       +-- Span: llm_call (iter 1)
    |       +-- Span: tool_call (grep)
    |       +-- Span: llm_call (iter 2)
    |       +-- Span: subagent_result (summary, 200 tokens)
    +-- Span: llm_call (iter 3, prefill 6K + summary)
    +-- Span: tool_call (edit_file)
    +-- Span: final_response
```

**What belongs in the span attributes**: iteration count, tool name, subagent type, model, input and output tokens, cache hit rate and which layer hit (system, tools, skills, history), context size in tokens, and which skill was activated.

**OpenTelemetry GenAI semantic conventions**, stabilising through 2025-2026, standardise these attributes. They are not fully settled, but they are the pivot to bet on.

**Trace context propagation through MCP** is a delicate point. When the agent calls an MCP server, the trace context has to travel inside the MCP protocol, so the server call appears in the same trace. Recent MCP supports this through standard headers; older implementations do not.

### Family 2, context telemetry

**What it captures**: how the context grows, compresses and stratifies during a session.

| Metric | What it measures |
|---|---|
| `context_size_over_time` | Tokens against iteration |
| `context_growth_rate` | Tokens added per iteration (mean, p50, p95) |
| `compaction_events` | Count, timing, size before and after |
| `prefix_cache_hit_layer1..N` | Hit rate per layer |
| `prefix_cache_byte_reuse` | Bytes actually reused from cache |
| `skill_load_frequency` | Which skills load, and how often |
| `subagent_context_isolation` | Tokens the parent was spared by the subagent |

**Why it matters**: context telemetry is the primary data for **modelling the workload**. Without it, sizing rests on assumptions. With it, you have empirical distributions to feed the open-loop benchmark.

A practical example: you discover the prefix cache hit rate on the skills layer is only 35%, because there are too many skills and they vary too much. You consolidate similar skills, the hit rate rises to 70%, average prefill drops by 40%, and the sizing improves accordingly.

### Family 3, cost telemetry

**What it captures**: who consumes what, and what **a useful outcome** costs.

**Three levels of granularity.**

**Per request**, the base level: tokens in, tokens out, cost, model used.

**Per task**, the agentic level: total tokens across all N requests, total cost per task, and a breakdown by agent (parent versus subagent), by tool, and by loaded skill.

**Per outcome**, the business level:

- **Cost per successful outcome**: total cost divided by successful tasks
- **Cost of failure**: what is spent on tasks that fail, which has to be counted
- **Cost of retry and looping**: what is spent on unproductive iterations

The last level is the business-relevant one and is rarely captured. For example: "the agent costs 0.50 per task on average, but 30% of tasks fail, so cost per successful outcome is 0.71". That is the number the customer's CFO wants to see, not 0.50.

**Token attribution per task** is what ties this back to sizing. Cost telemetry has to sum tokens across parent, subagents and every skill loaded for the same task. That requires a correlation ID propagated through the whole stack.

### Family 4, quality telemetry

**What it captures**: is the agent working well, and how well?

| Metric | What it measures |
|---|---|
| `task_success_rate` | Share of tasks completed successfully (needs ground truth) |
| `verify_failure_rate` | Share of verify steps returning not-OK |
| `looping_detection_rate` | Share of tasks where degenerative looping is detected |
| `abandonment_rate` | Share aborted on iteration or cost budget |
| `escalation_rate` | Share requiring user escalation |
| `user_satisfaction` | Explicit user feedback |
| `time_to_completion_p50/p95` | End-to-end latency per task |

**The ground truth challenge**: `task_success_rate` needs an external judge — human or LLM-as-judge — to decide whether the task was really completed. The agent declaring "done" is not enough. For coding workloads, automated tests are the natural judge; in other domains it is harder.

### Family 5, performance telemetry

**What it captures**: the operational SLOs of serving under agentic load. It overlaps with classic inference telemetry, with extra dimensions:

| Metric | Note |
|---|---|
| `TTFT_p50/p99` | Per request, but also aggregated per task |
| `ITL_p50/p99` | Inter-token latency during decode |
| `prefill_throughput` | Tokens/sec, split by cache hit versus miss |
| `decode_throughput` | Output tokens/sec |
| `queue_depth` | Requests queued — bursty under subagent fan-out |
| `gpu_utilization`, `kv_cache_pressure` | Hardware saturation |
| `agent_task_throughput` | Tasks per second completed, not requests per second |

**The critical point**: you need to correlate serving performance metrics with agentic task metrics. "p99 TTFT has degraded — is it because we have tasks with context over 150K missing the prefix cache?" Without correlation, that question has no answer.

### Family 6, replay and debugging

**What it captures**: the ability to faithfully reproduce a task execution for post-mortem debugging.

Four components are needed: a **complete persisted trace** of every input, output, tool call and subagent invocation; a **determinism handle** — sampling seed, model version, prompt template version — so the run can be repeated; a **replay interface** to step through the execution; and **attribution per decision**, understanding which part of the context influenced each step.

That last one is active research. Attribution interpretation is far from solved, but heuristic methods — attention analysis, ablation testing on parts of the context — are starting to appear in production debugging.

**The business use case**: a customer reports that the agent did something strange yesterday. Without a complete replay and attribution you are log-scraping. With them you have a structured triage workflow. That is compliance-grade, not a nice-to-have.
### How observability ties back to sizing

Three connections.

**1. Observability is the tap the workload modelling data comes out of.** A synthetic open-loop workload is not invented, it is measured. From observability on real workloads — even synthetic ones built in a lab — you get the distribution of tool calls per task, the context growth curve, the compaction rate, and the subagent fan-out pattern. Without layered observability those distributions are guesses. With it, they are data.

**2. Observability is the feedback loop for capacity tuning.** Measure the real workload, compare it with the forecast, identify the drift, tune. That loop does not work without rich observability.

**3. Cost per successful outcome is the business KPI.** The capacity card should carry not only "X tasks/sec at Y SLO" but also the cost per successful outcome. That requires performance telemetry and quality telemetry to be integrated. It is the most mature level of observability.

### What this is worth for a sovereign platform

For a sovereign platform with enterprise or public sector customers, agentic observability has three distinct values.

**Operational**: debugging, tuning, capacity planning — everything above.

**Compliance**: a complete audit trail of who did what, when, with which prompt and which result. For regulated sectors this is a regulatory requirement, not a feature.

**Governance**: administrator visibility into use cases, costs and anomalies, enabling policy enforcement — budget per department, restrictions on sensitive use, audit of critical actions.

Presenting a platform where observability is **first class** rather than an add-on is a strong enterprise argument. For regulated customers it is close to a must-have.

### Natural ground for infrastructure people

An honest note: agentic observability sits **very close** to the ground anyone from datacentre or serving operations already knows. The primitives are familiar — distributed tracing, SLO and SLA frameworks, metrics versus logging, anomaly detection.

The jump is applying those primitives to a new domain, the agent loop, with its own peculiarities: hierarchy, context, reasoning policy. For anyone from modern infrastructure, it is a natural transfer.

And that is precisely the difference from someone talking about "AI agents" in general: being able to describe agentic observability with the same rigour as the observability of a distributed application. Few people do.

### In short

Agentic observability is an engineering surface as rich as distributed system observability: six families of metrics, hierarchical tracing, per-task attribution, cost per successful outcome. For a sovereign platform it is also a compliance and governance surface. The underlying pattern — hierarchical spans, trace context propagation, multi-level SLOs — is familiar to anyone from serving operations. The challenge is applying it where the primitives are new.