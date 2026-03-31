# Agent Runtime

This document defines the canonical agent-runtime direction for the live
`backend/botadvisor` backend after retrieval evaluation is in place.

## Goal

Add a tool-capable agent layer that can grow into more modern agent features
without turning the backend into a framework-heavy platform.

The first implementation must stay small:

- retrieval tool only
- single tool call per turn
- streaming chat contract preserved
- retrieval quality remains the primary gate

## Design Position

BotAdvisor should not jump directly from retrieval-backed chat to a general
multi-step agent framework.

The correct direction is:

- design an expansion-capable runtime
- implement a minimal tool-aware kernel first
- defer planner loops, retries, and multi-tool orchestration until real product
  pressure exists

This keeps the codebase small enough for a solo developer while preserving a
credible path toward modern agent features.

## Runtime Layers

Phase 12 introduces three explicit layers.

### 1. Tool Runtime Kernel

The kernel owns:

- tool definition contracts
- tool input and output schemas
- registry and lookup
- execution boundary
- trace hooks
- error boundary for tool execution

The kernel must not contain product-specific orchestration logic.

### 2. Tool Adapters

Tool adapters wrap product capabilities as tools.

Phase 12 includes exactly one first-class adapter:

- retrieval tool

The retrieval tool wraps the canonical retrieval service and exposes it through
the kernel contract.

The retrieval module still owns retrieval behavior.
The tool adapter only converts between tool contracts and retrieval contracts.

### 3. Agent Orchestration

The orchestration layer decides whether the LLM should:

- answer directly without a tool call
- call the retrieval tool once
- generate the final answer from tool output

Phase 12 orchestration is explicitly single-step.

It must not implement:

- planner loops
- repeated tool retries
- multi-tool routing
- self-reflection chains
- graph execution

## Canonical Flow

Phase 12 runtime flow is:

1. user message enters the canonical chat path
2. the agent orchestrator asks the LLM whether a retrieval tool call is needed
3. the orchestrator optionally executes the retrieval tool once
4. the LLM produces the final answer
5. the API streams the final answer through the existing NDJSON chat contract

This preserves the existing API shape while replacing internal orchestration
with a tool-aware path.

## Module Boundaries

Recommended module shape:

- `backend/botadvisor/app/tools/`
  - `contracts.py`
  - `registry.py`
  - `executor.py`
- `backend/botadvisor/app/tools/retrieval_tool.py`
- `backend/botadvisor/app/agent/`
  - `schemas.py`
  - `service.py`
  - `orchestration.py`

Existing modules keep these roles:

- `retrieval/`
  owns retrieval quality and hybrid search behavior
- `chat/`
  keeps request and response shaping, streaming chunks, and user-facing schemas
- `llm/`
  grows the model-facing interface to support tool-aware message exchange
- `api/`
  remains thin and delegates to the runtime service

## Ownership Rules

These boundaries are strict.

- Retrieval logic does not move into `tools/`.
- Tool registry logic does not move into `chat/`.
- Agent orchestration does not own retrieval scoring behavior.
- Routers do not decide tool policy.
- Tool adapters must stay thin and product-specific.

If a tool adapter starts containing retrieval heuristics or prompt policy, the
boundary is wrong and should be refactored.

## Phase 12 Execution Constraints

Phase 12 must enforce:

- at most one tool call per user turn
- only one registered first-class tool: retrieval
- a bounded and explicit tool result schema
- direct-answer fallback when the model does not request a tool
- continued `/chat` streaming support

This is the minimum viable agent runtime.

## Quality Gate

The new agent runtime must not weaken the retrieval-first quality bar.

Phase 12 quality gates are:

- Phase 11 retrieval evaluation still passes
- retrieval tool contract preserves citation metadata
- single-step tool orchestration remains deterministic and testable
- tool-aware chat path keeps the NDJSON streaming contract

Tool calling is blocked from expanding until these checks are stable.

## Deferred Expansion Path

The design intentionally leaves room for later phases:

- bounded multi-step tool loops
- additional first-class tools
- approval gates for risky tools
- planner or executor splits
- LangGraph or graph-shaped orchestration

Those are future layers, not Phase 12 requirements.

## Future Roadmap

These are future candidate phases, not active implementation commitments.

### Phase 13: Bounded Multi-Step Tool Loop

Purpose:
Allow a small number of repeated tool calls with explicit stop conditions.

Deferred reason:
Phase 12 must first prove that single-step tool orchestration is stable and
maintainable.

Entry condition:
Phase 12 tool-aware chat must be stable under regression tests and retrieval
quality gates.

### Phase 14: Additional First-Class Tools

Purpose:
Introduce a small number of high-value tools beyond retrieval.

Deferred reason:
Additional tools increase runtime surface area, policy decisions, and test
burden before the kernel is proven.

Entry condition:
The retrieval tool kernel must remain small, explicit, and easy to extend
without naming or boundary drift.

### Phase 15: Agent Safety And Approval Gates

Purpose:
Add policy checks, human approval gates, and stronger execution controls for
riskier tool actions.

Deferred reason:
Phase 12 uses a low-risk retrieval-only tool set, so safety workflow complexity
is not yet justified.

Entry condition:
The runtime must start supporting tools whose effects or data exposure require
stronger control.

### Phase 16: Planner-Oriented Orchestration

Purpose:
Add planner or executor splits, or graph-shaped orchestration, when multi-step
reasoning becomes product-critical.

Deferred reason:
Planner-style orchestration would add architectural weight before the single-step
runtime is operationally mature.

Entry condition:
Repeated multi-step workflows must appear often enough that single-step
orchestration becomes the limiting factor.

## Non-Goals

Phase 12 does not attempt to deliver:

- a generic plugin marketplace
- dynamic tool loading
- many built-in operational tools
- multi-agent collaboration
- graph orchestration by default

## Success Condition

The agent runtime is successful when:

- retrieval remains the central product capability
- one retrieval tool can be invoked through an explicit kernel
- the chat API streams as before
- the code stays understandable without framework sprawl
- future agent capabilities can be added without reopening the retrieval core
