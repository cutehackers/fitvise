# Agent Runtime Tasks

## Phase 12: Agent Runtime Foundation

- [x] Add a canonical tool runtime kernel with explicit tool contracts, registry, and executor boundaries
- [x] Add a retrieval tool adapter that wraps the canonical retrieval service without moving retrieval logic out of `retrieval/`
- [x] Add a single-step agent orchestration path that supports at most one retrieval tool call per turn
- [x] Extend the LLM boundary to support tool-aware decision and final-answer generation contracts
- [x] Keep `/chat` streaming stable while routing chat execution through the tool-aware agent runtime
- [x] Add regression tests for tool selection, retrieval tool payloads, citation preservation, and NDJSON streaming behavior
- [x] Document the Phase 12 agent runtime rules in `backend/botadvisor/docs/agent_runtime.md`
- [ ] Track completion state for each implementation task in `backend/botadvisor/docs/agent_runtime/plan.md`

## Future Candidate Phases

- [ ] Phase 13 candidate: bounded multi-step tool loop once single-step orchestration is stable
- [ ] Phase 14 candidate: additional first-class tools once retrieval tool boundaries remain clean
- [ ] Phase 15 candidate: agent safety and approval gates once higher-risk tools exist
- [ ] Phase 16 candidate: planner-oriented orchestration once single-step execution is no longer enough
