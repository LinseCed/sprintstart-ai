# SprintStart AI Service

FastAPI service that ingests project artifacts, indexes them via RAG, and
answers questions / generates personalized onboarding paths.

## Setup

```bash
uv sync
cp .env.example .env   # fill in values
uv run python -m src.main   # dev server on :8000, docs at /docs
```

Docker: `docker-compose up --build` (overrides `OLLAMA_BASE_URL` to
`http://host.docker.internal:11434` automatically).

Production Docker CMD (Dockerfile): `uvicorn api.app:app --app-dir src`.

## Commands

| Purpose | Command |
|---|---|
| Run tests | `uv run pytest` |
| Tests + coverage | `uv run pytest --cov=src --cov-report=term-missing` |
| Integration tests (needs Ollama) | `uv run pytest -m integration` |
| Lint | `uv run ruff check .` |
| Format check | `uv run ruff format --check .` |
| Auto-format | `uv run ruff format .` |
| Type-check | `uv run pyright src/` |

Run **lint → format check → type-check → tests** before considering a change
done. CI enforces this order:
`gitleaks → quality (ruff lint + ruff format check + pyright) → pytest`.

## Branch protection

PRs to `main` must come from `dev` or `hotfix/*`. Push to `main`/`dev`
triggers Docker image publish to `ghcr.io/sprintstartproject/sprintstart-ai`.

## Architecture

`src/` layout:

- **`api/`** — FastAPI app (`app.py`), DI (`dependencies.py`), `schemas.py`,
  SSE helpers (`sse.py`), and route modules under `routes/`.
- **`agents/`** — `Agent` base class (`base.py`) with a `decision_role` /
  `answer_system` prompt pair and `max_steps` cap (default 5).
  `OrchestratorAgent` decides whether to delegate to sub-agents or answer
  directly. Tools live in `agents/tools/`, registered via `ToolRegistry`.
  `ChatOrchestrator` wraps the agent run into SSE events for `/api/v1/chat`.
- **`ingestion/`** — Per-filetype parsers (`text_parser`, `pdf_parser`,
  `code_parser`, `image_parser`) behind `parser.py`, then `chunker.py` and
  `metadata_store.py`.
- **`rag/`** — `retriever.py`, `hybrid.py` (BM25 + vector hybrid retrieval
  with RRF fusion), `citation.py`, `query_expansion.py`, `prompt.py`.
- **`llm/`** — `LLMClient` protocol (`base.py`) with implementations:
  `ollama_client.py`, `openai_client.py`, `anthropic_client.py`.
  `SplitLLMClient` lets chat and embeddings use different backends
  (`LLM_BACKEND` vs `EMBED_BACKEND`).
- **`store/`** — `VectorStore` protocol (`base.py`), `chroma_store.py`.
- **`onboarding/`** — Deterministic staged pipeline (not agentic):
  `select → filter → retrieve → synthesize → validate → emit`, yielding
  `StageProgress` markers. Blueprints are owned by the backend and passed in
  on each request — the service is stateless.

## Conventions

- `src` is on `pythonpath` for tests — import as `from agents.base import
  Agent`, **not** `from src.agents...`.
- Don't assume Ollama-only: check `llm/base.py`'s `LLMClient` protocol when
  touching LLM calls. Backend is configurable per deployment via
  `LLM_BACKEND` / `EMBED_BACKEND`.
- New sub-agents should follow the `SynthesisAgent` / `OrchestratorAgent`
  pattern (`Agent` base + `ToolRegistry`/`AgentTool`), not a bespoke loop.
- `AGENT_DEBUG=1` logs each agent's reasoning (LLM text + tool calls) to
  stderr — useful when debugging agent behavior.
- The onboarding pipeline is intentionally deterministic/staged rather than
  agentic: a bad LLM output degrades to a blueprint-only path instead of
  breaking. Keep that property when modifying it.
- `data/` is gitignored — don't commit local ChromaDB state or test fixtures
  there.

## Tests

`tests/` mirrors `src/`. Key utilities in `conftest.py`:

- `llm_required` / `vision_required` markers skip tests when Ollama is
  unreachable or vision model unconfigured.
- `parse_sse_events()` parses SSE streams into `dict` lists.
- `clear_dependency_caches` (autouse fixture) resets `lru_cache` on
  `get_llm`, `get_store`, `get_ingestion_metadata_store` before each test.

Reuse the fakes in `tests/stubs/llm.py` (`StubLLMClient`, `ScriptedLLMClient`)
and `tests/stubs/store.py` (`StubVectorStore`) instead of hand-rolling mocks.