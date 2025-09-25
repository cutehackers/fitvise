# Repository Guidelines

## Project Structure & Module Organization
Fitvise backend is centered in `app/`, with HTTP interfaces in `app/api/v1`, orchestration and services in `app/application`, domain logic in `app/domain`, infrastructure adapters in `app/infrastructure`, and shared schemas in `app/schemas`. `main.py` wires FastAPI, while `models/` keeps persisted ML assets and `docs/` houses architectural notes. Tests live under `tests/` with `unit`, `integration`, `e2e`, and reusable fixtures inside `tests/fixtures`.

## Build, Test, and Development Commands
Install dependencies with `uv sync` (or run `./boot.sh -i` to create `.venv` and sync automatically). Start the API via `uv run python run.py` or `uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000` for live reload, and verify configuration using `uv run python test_settings.py`. After setup, `./boot.sh` is the quickest way to bring the server up with consistent tooling.

## Coding Style & Naming Conventions
Code targets Python 3.11+, four-space indentation, and explicit typing on public interfaces. Apply `uv run black .` (120-character lines) and `uv run isort . --profile black`; keep modules snake_case, routers in `router.py`, and Pydantic models suffixed with `Schema`. Tests and fixtures should stay snake_case with filenames like `test_workout.py` and factory helpers in `tests/fixtures/*`.

## Testing Guidelines
Execute `uv run pytest` to run the suite; the included `tests/pytest.ini` enforces 80% coverage across `app`, emitting reports in `htmlcov/` and `coverage.xml`. Use markers to focus runs, e.g. `uv run pytest -m unit` or `uv run pytest tests/integration -m "integration and not external"`. Mirror production modules in the matching test package and document any required external services with `external` or `skip_ci` markers.

## Commit & Pull Request Guidelines
Commits follow Conventional style as seen in history (`feat:`, `refactor:`, `chore:`), so prefer a type, optional scope, and imperative summary such as `feat: add workout prompt endpoint`. Separate formatting-only changes and keep related work together for easier review. Pull requests should summarise behaviour changes, note affected endpoints or configs, list verification steps (`uv run pytest`, sample curl), and link the relevant issue or doc.

## Configuration & Security Tips
Environment variables belong in `.env`; use the template in `README.md`, keep secrets out of git, and run `uv run python test_settings.py` after edits to catch misconfiguration. Respect `ENVIRONMENT` flags to avoid exposing docs in production, and never commit generated artifacts (`htmlcov/`, `models/` exports) or API credentials.
