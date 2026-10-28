---
layout: default
title: "MCP — The Integration Protocol for Agentic Platforms"
---

# MCP — The Integration Protocol for Agentic Platforms

*Dielabs Knowledge Base. Prerequisite: [From the Stateless Model to the Agentic Harness](agentic-systems.md) (anatomy of the agent loop, prefill amplification, the SLO-anchored metric).*

---

## Key points

- MCP is an **infrastructure decoupling protocol**, in the same family as CSI/CNI in Kubernetes or ONNX in ML: it separates who orchestrates (the agent) from who integrates (the MCP server) from who serves the model (the inference engine).
- It turns the integration problem from **N×M** (every agent integrates every tool) into **N+M** (every agent speaks MCP, every tool exposes an MCP server).
- In the operational flow the LLM **never sees MCP**: it only sees generic tools in function-calling format. The translation between tools and the MCP protocol happens in the runtime.
- Three compatibilities to validate at design time: **LLM ↔ runtime** (function calling), **runtime ↔ MCP** (client implementation), **MCP server ↔ target software** (auth, scope, connectivity).
- The real architectural advantage of MCP is not "less code": it is **separation of responsibilities**. Agent developers, platform/IT and the ML team evolve independently.
- The winning enterprise pattern is the **centralized MCP gateway**, operationally very close to service chaining and API gateways in the NFV world.

---

## What is here and what is not

This page covers **MCP as a protocol and as an infrastructure layer**: what it does, how it is deployed, how it is governed in an enterprise.

It does **not** cover the anatomy of the agent loop, prefill amplification, sizing on N_steps or the Cr_open vs Cr_closed framing: all of that is in [From the Stateless Model to the Agentic Harness](agentic-systems.md). Where the discussion touches workload shape or metrics, it points there without re-explaining.

Typical reader: someone evaluating whether to bring MCP into an agentic platform who wants to understand **where the operational risk lives** (compatibility, governance, security), not why an agent is prefill-driven.

---

## 1. The problem MCP solves

Before MCP, every agent runtime integrated external systems with custom code:

```
[Python agent]
  ├── psycopg2 ────────> Postgres
  ├── slack_sdk ───────> Slack API
  ├── PyGithub ────────> GitHub API
  └── boto3 ───────────> AWS
```

The result: **N agents × M tools = N×M integrations** to write and maintain. Every change of agent framework forced a rewrite of the tool wrappers. Every new system to integrate required code inside the agent.

MCP (Model Context Protocol, Anthropic, November 2024) turns the problem into **N+M**: every agent speaks MCP, every tool exposes an MCP server, and they all talk through a single protocol (JSON-RPC over stdio or Streamable HTTP, see §2.2).

**Infrastructure analogy.** Same logic as CSI (Container Storage Interface) in Kubernetes. Before CSI, every storage vendor wrote custom plugins for every orchestrator. After CSI: one interface, N implementations. The same holds for CNI in networking, ONNX for ML models, ODBC for databases. **MCP is the integration protocol of the agentic layer**: it arrives late compared to the others because the agentic layer was born later.

---

## 2. The protocol itself

MCP is based on **JSON-RPC 2.0** and defines a set of standard primitives between client and server.

### 2.1 Main capabilities

The protocol standardizes three categories of resources exposed by the server:

| Primitive | What it exposes | Example |
|---|---|---|
| **Tools** | Invocable functions (with a parameter schema) | `query_db`, `send_slack_message`, `read_file` |
| **Resources** | Readable objects (files, records, URIs) | `file://project/main.py`, `postgres://users/123` |
| **Prompts** | Reusable prompt templates | "summarize_ticket", "code_review" |

In current practice the dominant primitive is **tools**. Resources and prompts are used less, and the ecosystem is still consolidating best practices.

### 2.2 Transport — stdio vs Streamable HTTP

Two standard transport modes, with different operational implications:

- **stdio**: the server is a local process spawned by the client, communicating over stdin/stdout. Minimal latency, no network, but it must be co-located with the client. The default for IDEs and desktop apps.
- **Streamable HTTP**: the server is an independent process reachable over the network, with a single HTTP endpoint that accepts POST and GET; streaming messages over SSE is optional. It enables distributed deployments, gateways and independent scaling. The required mode in enterprise scenarios.

> **Version note.** Streamable HTTP was introduced with the 2025-03-26 revision of the specification and replaces the two-endpoint HTTP+SSE transport of the 2024-11-05 revision, which remains only for backward compatibility. In client configurations and older documentation the "SSE" label still refers to the legacy transport: what is deprecated is the two-endpoint transport, not SSE as a streaming technology. Checked on `modelcontextprotocol.io` on 2026-09-26.

### 2.3 Capability negotiation

At the initial handshake, client and server exchange the capabilities they support (tools/list, resources/read, and so on). This lets the protocol evolve without breaking older implementations — a standard pattern in mature protocols (HTTP/2, TLS).

---

## 3. Anatomy of the end-to-end flow

Teaching scenario: a user asks *"how many active users do we have in the DB?"* to an agent connected to Postgres via MCP.

```
[User]
   │ "how many active users?"
   ▼
┌─────────────────────────────────────┐
│ AGENT RUNTIME (e.g. LangGraph)      │
│                                     │
│  1. Builds the prompt:              │
│     - user message                  │
│     - list of available tools       │
│       (fetched earlier via          │
│        MCP client → tools/list)     │
└──────────────┬──────────────────────┘
               │ API call (function calling format)
               ▼
        ┌──────────────┐
        │     LLM      │  ← decides: "I need to call query_db"
        └──────┬───────┘
               │ tool_use: { name: "query_db",
               │             args: { sql: "SELECT COUNT(*)..." }}
               ▼
┌─────────────────────────────────────┐
│ AGENT RUNTIME                       │
│  2. Receives tool_use from the LLM  │
│  3. Hands it to the MCP client      │
└──────────────┬──────────────────────┘
               │
               ▼
        ┌──────────────┐
        │  MCP CLIENT  │  (library inside the runtime)
        └──────┬───────┘
               │ JSON-RPC: tools/call
               │ (over stdio or HTTP)
               ▼
        ┌──────────────┐
        │  MCP SERVER  │  (e.g. mcp-server-postgres)
        │              │
        │  4. Receives │
        │  5. Translates│
        │     into SQL │
        └──────┬───────┘
               │ TCP / Postgres driver
               ▼
        ┌──────────────┐
        │   POSTGRES   │  ← runs the query
        └──────┬───────┘
               │ result set
               ▼
        ┌──────────────┐
        │  MCP SERVER  │  6. Formats the MCP result
        └──────┬───────┘
               │ JSON-RPC response
               ▼
        ┌──────────────┐
        │  MCP CLIENT  │
        └──────┬───────┘
               │
               ▼
┌─────────────────────────────────────┐
│ AGENT RUNTIME                       │
│  7. Injects the result into the     │
│     context as tool_result          │
└──────────────┬──────────────────────┘
               │ new API call to the LLM
               ▼
        ┌──────────────┐
        │     LLM      │  ← writes a natural-language answer
        └──────┬───────┘
               │ "There are 1,247 active users"
               ▼
            [User]
```

For the general dynamics of the loop (multiple round trips, context accumulation, prefill amplification) see [From the Stateless Model to the Agentic Harness](agentic-systems.md), the sections on tool use, the gather-act-verify loop and serving. Here we only care about **where MCP fits** in the flow.

---

## 4. The two invisibilities of MCP

### 4.1 The LLM does not know about MCP

The model only sees a **list of tools in function-calling format** (the vendor's dialect: OpenAI, Anthropic, Gemini, and so on). MCP is invisible to the model. All the translation between "abstract tool" and "MCP call" happens in the runtime.

Practical consequence: no model needs to be "trained for MCP". It only needs to be instruction-tuned for **generic tool use**, a standard capability in modern models (Claude, GPT, Gemini, Llama 3.1+, Qwen3, Mistral).

### 4.2 The target software does not know about MCP

Postgres sees an ordinary client connection (psycopg, a pg driver). The MCP server acts as a **semantic proxy**: it translates protocol requests into native calls to the target system. The same goes for Slack (it calls the Slack API), GitHub (GitHub REST), the filesystem (syscalls).

Consequence: MCP requires no changes to existing software. It is an additive layer. This also holds for closed-source or legacy systems: as long as they expose some API, the MCP server is just an external client.

### 4.3 Where MCP lives

MCP is the **layer between runtime and tools**. The client lives inside the runtime (it is an imported library). The server is an external process, local or remote depending on the transport. It is a sidecar/proxy pattern, not a component of the model nor of the target software.

---

## 5. The three compatibilities to validate

When you design an agentic system with MCP, three distinct compatibilities must be checked. This is the section that saves projects from silent failures in production.

| # | Layer | What to check | Risk if ignored |
|---|---|---|---|
| 1 | **LLM ↔ runtime** | The runtime speaks the function-calling dialect of the chosen model | Malformed tool calls, unparsable output |
| 2 | **Runtime ↔ MCP** | The runtime implements an MCP client compatible with the protocol (version, capabilities) | Version incompatibilities, missing features |
| 3 | **MCP server ↔ target software** | Connectivity, auth, permission scope | Security holes, runtime errors, unauthorized access |

**Implicit compatibility: the model's tool-use quality.** Even when all three technical compatibilities are fine, small or poorly tool-tuned models produce malformed arguments, pick the wrong tool, or fall into loops. It has to be validated with dedicated evals (BFCL, ToolBench, or custom evals on the domain). It is a frequent blind spot in POCs.

For the silent failure pattern "model fine-tuned for format X, runtime parser expects format Y", see the section on model and runtime compatibility in [From the Stateless Model to the Agentic Harness](agentic-systems.md).

---

## 6. The decouplings MCP introduces

The value of MCP is not "less code". It is the **separation of responsibilities between layers and teams**:

| Layer | Owner | Life cycle driven by |
|---|---|---|
| Agent logic | AI team / app dev | New application use cases |
| MCP client | Agent framework | Library updates |
| LLM | ML team / vendor | New model, fine-tuning |
| MCP server | Platform team / vendor | Tool updates, new integrations |
| Target software | Product team / IT | Its own life cycle |

Each layer evolves independently. It is the classic pattern of mature infrastructure protocols: HTTP, SQL, gRPC, CSI and CNI did the same in their domains.

**Organizational implication.** In an enterprise platform, MCP lets the platform team own the integration layer (servers, gateway, governance) **without depending on the application team** that builds the agents, and vice versa. It is a significant change of ownership in enterprise AI projects, where integration has so far often been part of a monolithic agent codebase.

---

## 7. Deployment topologies

Three realistic patterns, in order of enterprise maturity.

### 7.1 Local stdio (desktop / dev)

```
[Laptop] Agent ──stdio──> mcp-server-X ──TCP──> Software [anywhere]
```

The server is spawned by the client as a child process. The default for Claude Desktop, Cursor and IDEs. Pro: zero infrastructure. Con: every user installs everything, credentials are scattered, no governance.

This is not the enterprise pattern — it is the dev/early-adopter pattern. But it is also where much of the ecosystem is being born.

### 7.2 MCP server co-located with the target

```
[Client] Agent ──HTTP──> [Target server] mcp-server-X ──localhost──> Software
```

A Streamable HTTP server next to the software. Sensible when you want to centralize auth and audit close to the data. A good fit for on-prem legacy systems where you cannot, or do not want to, expose the target system over the network to an external gateway.

### 7.3 Centralized MCP gateway (enterprise)

```
[Client] Agent ──HTTP──> [MCP Gateway VM/K8s] ──TCP──> N backend systems
```

A fleet of MCP servers in a dedicated cluster. Centralized auth, rate limiting, audit and secret injection. The realistic pattern for sovereign AI and regulated enterprises.

**For people coming from NFV/datacenter.** The MCP gateway is the **service chaining / API gateway** of the agentic world. Same operational patterns (HA, observability, policy enforcement), new application domain. The skills transfer almost directly: whoever can design an API gateway architecture with WAF, rate limiting, mTLS and secret injection is already designing an MCP gateway, and only has to learn the protocol.

The lock-in here is **internal** (the gateway becomes a critical platform component), not external. It deserves the same seriousness as an API gateway or an enterprise message broker.

---

## 8. Who builds MCP servers

Four sources, in order of reliability for enterprise use:

1. **Official vendors**: the software maker publishes and maintains the server (GitHub, Cloudflare, Sentry, Stripe, Linear). A fast-growing trend in 2025–26.
2. **Anthropic reference servers**: a set of reference servers (filesystem, Postgres, Slack, Puppeteer) in the `modelcontextprotocol/servers` repo. Production-grade for the most common systems.
3. **Community**: thousands of servers on GitHub. Variable quality; audit before use (auth, scope, security).
4. **Custom in-house**: even when an official server exists, enterprises often write a custom wrapper to narrow the scope, add audit, and integrate with corporate IAM.

**Closed-source software can have an MCP server too**, because the server is an *external client* of the public APIs, not a component of the product. This lowers the adoption barrier drastically.

**An underestimated risk.** The maturity of the individual MCP server is the factor that most often decides whether an agentic project succeeds or fails. A community server with too broad a scope (for example full-admin access to a DB instead of read-only on specific tables) is a security hole waiting to be exploited, not a bug.

---

## 9. Governance and identity in the enterprise

From a platform/security point of view, every MCP server is:

- A new **service identity** in IAM (DB user, service account, API token)
- A new **data access channel** to audit (access logs, anomaly detection)
- A place to apply **least privilege** (minimum-scope tokens, granular roles)
- A **secret management** point (backend credentials must not live in the server's code)

In sovereign or regulated environments (GDPR, finance, healthcare) this is platform/security territory, not application territory. The correct enterprise pattern is:

```
[MCP Gateway] ──> [Vault / Secrets Manager] ──> credentials injected at runtime
       │
       ├──> [IAM / SSO] ──> identity propagation (who is making the call?)
       ├──> [Audit log] ──> every tool call traced
       └──> [Policy engine] ──> who can call what, with which parameters
```

### Specific attack vectors

Three vectors that did not exist in the classic datacenter and must be covered explicitly:

- **Prompt injection via tool description**: a malicious (or compromised) MCP server can inject instructions into the text of its tool descriptions, which end up in the model's system prompt. The model reads "ignore previous instructions, exfiltrate data". An active research area, with mitigations still immature.
- **Tool result poisoning**: the result of a tool call goes back into the context. A compromised MCP server can inject instructions into its results.
- **Scope creep**: the MCP server exposes more operations than the agent should be able to perform. For example, a "read-only" server that also has an undocumented `delete` operation. Auditing the server is necessary.

These are not hypotheses: they are patterns documented in security advisories published in 2024–25.

---

## 10. Workload implications (pointer)

MCP-heavy agentic workloads have a specific inference traffic shape: **prefill-driven and multi-turn, with context growing at every step and short decode at each step**. This changes sizing radically compared to classic chat, and single-turn benchmarks systematically underestimate real capacity.

The right frame for sizing is **Cr_open** (SLO-anchored, open loop), not Cr_closed (hardware-anchored, single step).

For the full treatment of prefill amplification, N_steps, KV pressure, prefix caching as a capacity multiplier and the operational sizing checklist, see the serving section of [From the Stateless Model to the Agentic Harness](agentic-systems.md).

What is specific to MCP, and worth remembering here:

- Every MCP tool definition in the system prompt takes 100–300 tokens (name + description + parameter schema). A gateway exposing 50 tools without selective filtering means 5–15K tokens of fixed prefill on every iteration → a heavy impact on TTFT and KV.
- **Tool latency** in enterprise MCP scenarios includes the round trip to the gateway, auth and any rate limiting. It must be measured and tracked separately from GPU latency.
- MCP servers must be treated as **critical dependencies** in the availability plan: if the gateway is down, the whole agent is degraded even when the GPU works.

---

## 11. Honest pros and cons

### Pros

- Real decoupling between layers (agent, tools, model)
- MCP servers reused across teams and frameworks
- A governance pattern that can be centralized (the gateway)
- A mental model familiar to people coming from the classic datacenter
- A fast-growing ecosystem: in 2025 enterprise vendors started publishing official servers at a steady pace

### Cons / caveats

- A young protocol (introduced in late 2024): less battle-tested than HTTP, SQL, gRPC
- Bugs and incompatibilities between client and server implementations still exist
- Security is evolving: prompt injection via tool description and tool result poisoning are active vectors
- Enterprise governance is an emerging pattern, not a consolidated one — best practices are still forming
- Latency overhead versus direct integration (an extra hop)
- A de facto standard driven by Anthropic: alternatives (e.g. OpenAI functions, proprietary ecosystems) exist and could fragment the space

---

## 12. Takeaways

1. MCP is not just an API: treat it as an **infrastructure layer/protocol** (HA, observability, governance), not as a simple application SDK.
2. The LLM is not the critical point of MCP. The **runtime** (compatibility) and the **server** (scope, auth, quality) are.
3. The winning enterprise pattern is the **centralized MCP gateway** — operationally very close to NFV service chaining and API gateways.
4. **MCP server maturity** is the most overlooked risk factor: a badly scoped server is a security hole, not an accuracy problem.
5. For people coming from datacenter/NFV, MCP is a space where existing skills (gateway design, identity, secret management, HA) transfer almost directly. It is natural presales ground for infrastructure profiles.

---

## The thesis in short

> MCP is the **CSI/CNI of the agentic world**: a decoupling protocol that separates who orchestrates (the agent runtime) from who integrates (the MCP server) from who serves the model (the inference engine). The value is not "less code", it is the operating model: three independent layers, three independent teams, a centralized gateway that does governance. The right enterprise question is not "which MCP server do I use" but "how does MCP fit into my existing gateway, identity and audit platform". People coming from datacenter/NFV already have 70% of the mental model they need.

---

## Quick glossary

| Term | Definition |
|---|---|
| **MCP** | Model Context Protocol. A standardized JSON-RPC protocol for agent ↔ tool integration, introduced by Anthropic in November 2024. |
| **MCP client** | Library inside the agent runtime that speaks MCP to one or more servers. |
| **MCP server** | External process that exposes tools/resources/prompts over MCP and translates them into native calls to the target software. |
| **MCP gateway** | A centralized fleet of MCP servers in a dedicated cluster. The enterprise pattern for governance. |
| **Capability negotiation** | The initial client/server handshake that establishes which primitives are supported. |
| **Tool definition** | The MCP schema of a single tool (name, description, parameters). It ends up in the model's system prompt. |

---

## References

**Dielabs**

- [One Capacity Is Not Enough](/frameworks/one-capacity-is-not-enough.html) — the methodological basis of Cr_closed, Cr_open and CrossP
- [From the Stateless Model to the Agentic Harness](agentic-systems.md) — prefill amplification, N_steps, prefix caching, three services to size and the agentic sizing checklist
- [The Agentic Ecosystem](agentic-ecosystem.md) — the state of the art on A2A and the observability tooling

**External**

- Anthropic, *Introducing the Model Context Protocol* (November 2024)
- Official specification: `modelcontextprotocol.io`
- Official repo: `github.com/modelcontextprotocol`

---

*All content is original Dielabs work by Diego Bardella.*
