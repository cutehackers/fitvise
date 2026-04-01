# Agent Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a retrieval-first, tool-capable agent runtime that supports one retrieval tool call per turn while preserving the canonical streaming chat contract.

**Architecture:** The implementation introduces a small tool kernel, a thin retrieval tool adapter, and a single-step agent orchestrator. Retrieval remains the product core, the API stays thin, and future multi-step agent features remain deferred behind the new runtime boundary.

**Tech Stack:** Python, existing BotAdvisor retrieval/chat/llm modules, LangChain-compatible orchestration patterns, pytest, structured logging, LangFuse

---

### Task 1: Add Tool Runtime Contracts

**Files:**
- Create: `backend/botadvisor/app/tools/contracts.py`
- Create: `backend/botadvisor/tests/unit/test_tool_contracts.py`

- [ ] **Step 1: Write the failing test**

Define tests for explicit tool definitions, request payloads, result payloads, and
rejection of vague or incomplete tool metadata.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest botadvisor/tests/unit/test_tool_contracts.py -q`

- [ ] **Step 3: Write minimal implementation**

Add explicit tool contract models with clear field names and no vague helper or
manager naming.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest botadvisor/tests/unit/test_tool_contracts.py -q`

- [ ] **Step 5: Commit**

Commit: `feat(botadvisor): add tool runtime contracts`

### Task 2: Add Tool Registry And Executor

**Files:**
- Create: `backend/botadvisor/app/tools/registry.py`
- Create: `backend/botadvisor/app/tools/executor.py`
- Create: `backend/botadvisor/tests/unit/test_tool_registry.py`

- [ ] **Step 1: Write the failing test**

Add tests for tool registration, duplicate-name rejection, explicit lookup, and
single tool execution through the executor boundary.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest botadvisor/tests/unit/test_tool_registry.py -q`

- [ ] **Step 3: Write minimal implementation**

Implement a focused registry and executor with explicit tracing and error
boundaries.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest botadvisor/tests/unit/test_tool_registry.py -q`

- [ ] **Step 5: Commit**

Commit: `feat(botadvisor): add tool registry and executor`

### Task 3: Add Retrieval Tool Adapter

**Files:**
- Create: `backend/botadvisor/app/tools/retrieval_tool.py`
- Create: `backend/botadvisor/tests/unit/test_retrieval_tool.py`

- [ ] **Step 1: Write the failing test**

Add tests that verify the retrieval tool maps tool input to
`RetrieverRequest`, preserves citation metadata, and returns bounded result
payloads.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest botadvisor/tests/unit/test_retrieval_tool.py -q`

- [ ] **Step 3: Write minimal implementation**

Add a thin adapter around the canonical retrieval service. Keep retrieval logic
inside `retrieval/`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest botadvisor/tests/unit/test_retrieval_tool.py -q`

- [ ] **Step 5: Commit**

Commit: `feat(botadvisor): add retrieval tool adapter`

### Task 4: Add Single-Step Agent Orchestration

**Files:**
- Create: `backend/botadvisor/app/agent/schemas.py`
- Create: `backend/botadvisor/app/agent/orchestration.py`
- Create: `backend/botadvisor/app/agent/service.py`
- Create: `backend/botadvisor/tests/unit/test_agent_orchestration.py`

- [ ] **Step 1: Write the failing test**

Add tests for direct-answer mode, one retrieval tool call mode, and rejection of
multiple tool calls in a single turn.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest botadvisor/tests/unit/test_agent_orchestration.py -q`

- [ ] **Step 3: Write minimal implementation**

Implement a single-step orchestrator that can either answer directly or call the
retrieval tool once before final answer generation.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest botadvisor/tests/unit/test_agent_orchestration.py -q`

- [ ] **Step 5: Commit**

Commit: `feat(botadvisor): add single-step agent orchestration`

### Task 5: Extend LLM Boundary For Tool Decisions

**Files:**
- Modify: `backend/botadvisor/app/llm/factory.py`
- Modify: `backend/botadvisor/app/llm/ollama.py`
- Create: `backend/botadvisor/tests/unit/test_tool_aware_llm.py`

- [ ] **Step 1: Write the failing test**

Add tests that define the model-facing contract for tool decision output and
final answer generation after tool execution.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest botadvisor/tests/unit/test_tool_aware_llm.py -q`

- [ ] **Step 3: Write minimal implementation**

Extend the existing LLM seam just enough to support tool-aware orchestration
without introducing graph logic or planner loops.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest botadvisor/tests/unit/test_tool_aware_llm.py -q`

- [ ] **Step 5: Commit**

Commit: `feat(botadvisor): add tool-aware llm contract`

### Task 6: Route Chat Through The Agent Runtime

**Files:**
- Modify: `backend/botadvisor/app/chat/service.py`
- Modify: `backend/botadvisor/app/api/router.py`
- Modify: `backend/botadvisor/app/api/deps.py`
- Create: `backend/botadvisor/tests/unit/test_agent_chat_api.py`

- [ ] **Step 1: Write the failing test**

Add tests that verify `/chat` still streams NDJSON while internally using the
new agent runtime.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest botadvisor/tests/unit/test_agent_chat_api.py -q`

- [ ] **Step 3: Write minimal implementation**

Route canonical chat execution through the single-step agent path while keeping
the external API contract stable.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest botadvisor/tests/unit/test_agent_chat_api.py -q`

- [ ] **Step 5: Commit**

Commit: `feat(botadvisor): route chat through agent runtime`

### Task 7: Close Phase 12 With Verification

**Files:**
- Modify: `backend/botadvisor/docs/agent_runtime/tasks.md`
- Modify: `backend/botadvisor/docs/agent_runtime.md`
- Modify: `backend/botadvisor/README.md`

- [ ] **Step 1: Run focused verification**

Run: `cd backend && uv run pytest botadvisor/tests/unit/test_tool_contracts.py botadvisor/tests/unit/test_tool_registry.py botadvisor/tests/unit/test_retrieval_tool.py botadvisor/tests/unit/test_agent_orchestration.py botadvisor/tests/unit/test_tool_aware_llm.py botadvisor/tests/unit/test_agent_chat_api.py -q`

- [ ] **Step 2: Run full backend verification**

Run: `cd backend && uv run pytest -q`

- [ ] **Step 3: Run lint**

Run: `cd backend && uv run ruff check botadvisor/app botadvisor/scripts botadvisor/tests`

- [ ] **Step 4: Update docs and status**

Mark Phase 12 complete in `backend/botadvisor/docs/agent_runtime/tasks.md` and document the
canonical agent-runtime usage path.

- [ ] **Step 5: Commit**

Commit: `feat(botadvisor): complete agent runtime foundation`
