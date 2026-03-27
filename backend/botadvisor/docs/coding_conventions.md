# BotAdvisor Coding Conventions

## Goal

Every surviving BotAdvisor module must be tidy, testable, and maintainable for a solo developer.
Functional code is not enough.
Code must also satisfy a strict readability and responsibility standard.

## Core Quality Rules

- Prefer small, focused modules over large generic layers.
- Every file must have one clear responsibility.
- Every function must have one clear reason to change.
- Every abstraction must justify its existence through real runtime or testing value.
- Favor explicit data flow over hidden global behavior.
- Favor boring code over clever code.

## SOLID Expectations

### Single Responsibility

- A router should only translate HTTP to service calls.
- A service should only coordinate one business concern.
- A repository or adapter should only handle one external system.
- Parsing, validation, IO, orchestration, and formatting should not be mixed in one function.

### Open/Closed

- Extension points are allowed only where real variation exists.
- Do not create abstract base classes for hypothetical future providers.

### Liskov and Interface Segregation

- Keep interfaces narrow.
- If an interface forces consumers to ignore half its methods, the interface is wrong.

### Dependency Inversion

- Depend on small local contracts where needed.
- Do not introduce a full dependency-injection framework unless plain constructors and small factories are insufficient.

## File and Module Rules

- Prefer files under roughly 200 lines when practical.
- Split files once a second responsibility emerges.
- Group code by feature first, not by abstract technical layer.
- Shared modules must stay small. If shared code keeps growing, it probably belongs in a feature module.
- Do not import from `backend/deprecated/legacy_backend/app` in the final `botadvisor` runtime.

## Function Rules

- A function should do one thing.
- If a function needs comments to explain multiple phases of work, split it.
- Return explicit values instead of mutating hidden shared state.
- Keep control flow shallow.
- Avoid boolean flag arguments that switch behavior.

## Class Rules

- Classes are for cohesive state plus behavior, not namespacing.
- If a class has only static methods, prefer functions.
- If a class mostly passes through to another object, remove or collapse it.

## Error Handling

- Fail early on invalid input.
- Raise domain-appropriate errors with useful messages.
- Do not swallow exceptions unless converting them into a clearer boundary-level error.
- Logging is not error handling.

## Logging and Observability

- Use structured logging where possible.
- Bind request or trace identifiers at runtime boundaries.
- Log decisions and failures, not noisy internal chatter.
- LangFuse tracing should wrap meaningful operations, not every tiny helper.

## Retrieval Rules

- Retrieval logic must stay inside retrieval modules.
- LangChain adapters must not reimplement retrieval behavior.
- Retrieval responses must preserve source metadata required for citations.
- Hybrid search behavior belongs in retrieval configuration, not API handlers.

## API Rules

- Routers stay thin.
- Validation belongs at the schema or boundary level.
- Orchestration belongs in services.
- Persistence and network calls do not belong in routers.

## Script Rules

- Scripts are production code and must follow the same quality bar as API code.
- Separate argument parsing from business logic.
- Scripts must return machine-readable summaries when practical.
- Retries, logging, and exit codes must be explicit.

## Testing Rules

- No production change without focused tests.
- Prefer unit tests around contracts and orchestration seams.
- Use integration tests only where adapter behavior or runtime wiring matters.
- Tests must describe behavior, not implementation details.
- A partially working feature with no trustworthy tests is not complete.

## Naming Rules

- Names must describe business purpose, not implementation trivia.
- Avoid vague names such as `manager`, `helper`, `utils`, `processor`, or `service` unless the context is truly precise.
- Prefer `EmbedUpsertRunner` over `EmbedManager` and `retrieve_context` over `handle_query_data`.

## Forbidden Patterns

- dead abstraction layers
- god services
- routers with business logic
- functions that parse input, call external systems, and format output all at once
- feature code importing legacy runtime code directly
- speculative extension systems
- silent fallback behavior that hides failures
- large files that collect unrelated concerns

## Refactoring Gate

Existing code may be functionally correct but still fail the quality bar.
Such code must be treated as `partial`, not `done`, when planning migration into BotAdvisor.

Code is considered acceptable only if it is:

- understandable without reading the whole repository
- testable without awkward global setup
- isolated by responsibility
- aligned with the canonical architecture
- free from obvious duplication and dead layers
