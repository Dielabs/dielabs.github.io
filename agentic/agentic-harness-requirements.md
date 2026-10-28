---
layout: default
title: "Minimum Requirements of an Agentic Harness"
---

# Minimum Requirements of an Agentic Harness

A harness is everything around the model: what it can do, what it remembers, when it stops. The framework is the easy part, and by now it is a commodity. What decides whether an agent works in production are the requirements below, and almost all of them stay with whoever uses it.

Use this list when you evaluate or design a harness (Deep Agents, Claude Code, Opencode, Pi), or when you need a criterion to say whether an agent is ready for production.

---

## Three families — memory, control, verification

Every requirement falls into one of three families. The first two answer "what does it know" and "what can it do". The third answers "how does it prove it did well", and it is the one most often missing.

## Memory — what the harness must remember

- **Project context.** What is being built, for whom, under which constraints. Without it, every session starts from zero.
- **The project split into phases.** Each phase has a goal and an exit condition. The agent knows which phase it is in and what closes it.
- **Decisions taken, with the why.** Not just the choice, but the reason. The reason is what keeps the decision from being reopened in every session.
- **Blocked decisions, with the why.** See the dedicated section below: they have one extra field.

## Control — boundaries, budgets, stop criteria

- **Tracing.** Every action leaves a trace that a person can read without reconstructing the whole run. When a subagent gets stuck or produces conflicting work, the harness must show what happened.
- **Clear boundaries, separate for read and write.** Reading the whole project is almost always acceptable. Writing only in declared zones. Secrets never in the context.
- **Explicit budgets.** Tokens, time, cost, number of attempts. An agent without a budget does not fail, it wears you out.
- **Stop and escalation criteria.** For each phase and each subagent: when to stop and ask. Boundaries say what you cannot do; stop criteria say when you must not go on. It is the piece no framework ships ready-made. A concrete example of a stop rule: halt after two consecutive failed write attempts.
- **Identity and permissions fixed per session, not per request.** At session start you decide on whose behalf the agent acts and what it can do, and you do not re-decide it turn by turn. For every agent, two things written down before it starts: **who it is** (identity: a user, a service, a delegation) and **what it can do** (read-only permissions by default; writing opens per zone and per session). The block lives in the permission system (IAM, gateway), not in the agent's instructions or skill: an instruction can be worked around, a denied permission cannot.

## Verification — external proof, rollback, replay

- **External truth, not self-declared.** The agent does not say "done": it produces evidence that someone else can check. Passing tests, a command with the expected output, an applied diff. Testing is the product here too.
- **Rollback.** Checkpoints of the state, and resume from there, not from scratch. Every action that changes something has a known way back before it runs.
- **Replay.** A traced session must be replayable with another model and compared. Without replay, tracing is a diary, not a tool.

## Blocked decisions have three fields

What, why, and what would unblock it. With two fields the list is an archive. With the third it becomes a work queue for the person: they know what they must provide or decide to get the agent moving again.

## There is only one quality criterion

The list above is the same one you would apply to a new person joining a team: context, phases, limits, written decisions, when to ask. A harness is good if it makes the project readable to anyone who comes after, human or model. The measure is one: how long it takes someone to pick up the work cold.

---

**Related:** [From the Stateless Model to the Agentic Harness](agentic-systems.md) for the mechanics of the loop, context engineering and agentic observability; [MCP — The Integration Protocol for Agentic Platforms](mcp-integration-protocol.md) for identity, gateways and permission scoping on the tool side.

*All content is original Dielabs work by Diego Bardella.*
