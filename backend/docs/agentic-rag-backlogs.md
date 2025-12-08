Agentic RAG Backlog
===================

# How to use this backlog
# Prompt 
# Natural next steps: pick Phase 0 items from agentic-rag-backlogs.md and implement/tests, then proceed to Phase 1 to stand up the retriever tool and agent prompt.

Purpose: incrementally evolve the current RAG chat (`app/application/use_cases/chat/rag_chat_use_case.py`) into an agentic RAG flow aligned with LangChain’s agentic retrieval pattern.

Phase 0 – Stabilize current RAG (short term)
- RAG retrieval/citations: keep doc ordering stable for citations, ensure metadata includes `document_id`, `chunk_id`, `_distance` before trimming; cap citation content length and normalize whitespace.
- Context windowing: upgrade `ContextWindowManager` to handle empty/None context gracefully, expose a hook for future summarization, and log token budget vs. actual tokens used per request.
- Session management: add max-turn enforcement in `SessionService` (drop oldest turns beyond `turns_window`), and include a guardrail for maximum total messages per session.
- Error handling: wrap retrieval/LLM errors with typed `ChatOrchestratorError` causes; add clearer warnings when context is empty or truncated.
- Streaming: preserve chunk order; when the model returns `AIMessageChunk`, concatenate content in a consistent buffer before emitting.
- Tests: add unit tests for citations, context fitting edge cases, empty retrieval, and session turn trimming.

Phase 1 – Prepare agent tooling
- Retriever tool: wrap the existing retriever with `create_retriever_tool`, name/describe it for fitness use cases, and ensure tool output carries citation-ready metadata.
- Tool registry: add a lightweight registry module (e.g., `app/application/use_cases/chat/tools.py`) to house tool creation and descriptions; keep it injectable for tests.
- Agent prompt: craft a system prompt that teaches tool use, citation format `[n]`, when to retrieve vs. answer from history, and how to handle empty results.
- Configuration: add knobs for max tool calls, tool call timeout, and whether to allow multiple retrieval passes; wire them into settings or the use case init.
- Tests: add unit tests for tool creation, prompt rendering, and configuration defaults.

Phase 2 – Agent executor integration
- Agent construction: build an async agent via `create_tool_calling_agent` with the retriever tool; wrap it in `AgentExecutor` with streaming enabled.
- Execution path: refactor `chat()` to route through the agent executor (plan → tool call(s) → final), preserving session history and streaming partials.
- History/citations: ensure tool outputs include indices so the agent’s final answer can cite `[1]`, `[2]`; map the agent’s final message to `SourceCitation` entries consistently.
- Rephrasing strategy: optionally retain the rephrase chain as a tool or pre-step, but let the agent decide when to call it; avoid forced pre-rephrase.
- Guardrails: enforce max tool calls per request; short-circuit with graceful responses when retrieval is empty or tool calls fail.
- Tests: add agent loop tests (no tools, single tool call, multi-call), streaming assertions, and citation propagation tests with mocked retriever outputs.

Phase 3 – Observability and resiliency
- Tracing/metrics: emit spans for `rephrase`, `retrieve`, `agent.plan`, `agent.final` with timing; track token usage per step when available from the LLM client.
- Structured logs: include `session_id`, `rephrased_question`, documents retrieved, tool call counts, and truncation decisions.
- Fallback paths: define behavior when retriever errors (return no-context answer with disclaimer), when LLM refuses tool calls, and when context window overflow occurs.
- Tests: observability hooks are called; fallback responses are well-formed and cited as “no sources”.

Phase 4 – Rollout hygiene
- Configuration surface: document new settings in `README.md`/env examples; expose feature flag to toggle agentic mode vs. legacy RAG.
- Performance checks: benchmark latency with and without agent loop; adjust max tool calls/timeouts accordingly.
- Documentation: add developer doc summarizing the agentic flow, tool catalog, and testing guidance.
