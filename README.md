# sprintstart-ai

The AI and RAG pipeline service for [SprintStart](https://sprintstart.readthedocs.io/en/latest/), an AI-assisted onboarding and knowledge-retrieval platform for software development teams.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally with the required models pulled:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text

# Optional — required only for image captioning
ollama pull llava:7b        # lightweight, good for development
# ollama pull qwen2-vl:7b  # recommended for production (better on diagrams/text-heavy images)
```

## Getting Started

### Local

```bash
# 1. Install dependencies
uv sync

# 2. Configure environment
cp .env.example .env
# Edit .env and fill in the values

# 3. Run the service
uv run python -m src.main
```

The service runs on port `8000`. Interactive docs are available at `/docs`.

### Docker

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env and fill in the values

# 2. Start the service
docker-compose up --build
```

The service runs on port `8000`.

> `OLLAMA_BASE_URL` is automatically overridden to `http://host.docker.internal:11434` inside the container, so no manual change is needed.

## Environment Variables

| Variable | Example value | Description |
|---|---|---|
| `LLM_BACKEND` | `ollama` | LLM backend to use. Currently only `ollama` is supported. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL of the Ollama instance. Use `http://host.docker.internal:11434` when running via Docker with Ollama on the host. |
| `OLLAMA_MODEL` | `gemma4:e4b` | Chat model to use for generation. |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text:latest` | Embedding model to use for ingestion and retrieval. |
| `OLLAMA_VISION_MODEL` | `llava:7b` (dev) / `qwen2-vl:7b` (prod) | Vision model for image captioning. Optional — if unset, image files are accepted but produce no chunks (`chunk_count=0`). |
| `CHROMA_PATH` | `./data/chroma` | Path for ChromaDB persistent storage. If unset, an in-memory store is used and data will not persist. |
| `AGENT_DEBUG` | `0` | When set to a truthy value, logs each agent's reasoning step (LLM text and tool calls) to stderr. Disabled by default and for `0`/`false`/`no`/`off`/empty. |
| `CHUNK_SIZE` | `512` | Maximum number of characters per chunk |
| `CHUNK_OVERLAP` | `64` | Number of characters reused between consecutive chunks to preserve context when splitting large chunks |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Reports service health including LLM backend status. Returns `503` if Ollama is unreachable. |
| `POST` | `/api/v1/ingest` | Parses, chunks, and embeds a document and stores it in the vector store. Re-ingesting the same `artifact_id` replaces existing chunks. Supports text files and images (send image content as base64; requires `OLLAMA_VISION_MODEL`). |
| `POST` | `/api/v1/chat` | Retrieves relevant chunks and streams a generated answer as Server-Sent Events (SSE). |
| `POST` | `/api/v1/title` | Generates a short descriptive title from a user prompt. |
| `POST` | `/api/v1/onboarding/path` | Generates a personalized onboarding path (SSE). Blueprints are passed in by the backend on each request — the service is stateless. |
| `POST` | `/api/v1/onboarding/blueprints/generate` | Batch job: generates `source: generated` blueprints from the corpus and returns them. The backend owns persistence — it passes its active blueprints in and stores the results. |

### Chat SSE stream

The `/api/v1/chat` endpoint streams newline-delimited JSON events:

| Event type | Description |
|---|---|
| `tool_use` | A capability the orchestrator invoked, in order. Has `name` and `kind` (`agent` for a sub-agent, `tool` for a leaf tool) |
| `token` | A single token fragment of the answer |
| `citation` | A source chunk used to generate the answer |
| `done` | Signals the end of the stream |
| `error` | Emitted on failure instead of the above |

## AI-proposed onboarding blueprints

Onboarding paths are assembled from versioned **blueprints** scoped by `global` or `area:<name>`. Blueprints carry a `source`: `authored` (human-written) or `generated` (drafted by the AI from the ingested corpus). The backend owns blueprint persistence, versioning, and rollback — the AI service is stateless and only returns generated data.

The generation job analyzes the corpus and returns proposed blueprints for the backend to persist. Every generated step is **grounded** (cites an ingested chunk), the job is **idempotent** (a blueprint records the `corpus_fingerprint` it was drafted from, so an unchanged corpus is a no-op), and **human-owned invariants are protected** — it may not remove or downgrade a `required` or `invariant: true` step; such changes are re-injected and the outcome is escalated, never silently applied.

Blueprint management (trigger generation, list versions, rollback) is exposed through the backend API at `POST /api/v1/onboarding/blueprints/generate`, `GET /api/v1/onboarding/blueprints/{scope}/versions`, and `POST /api/v1/onboarding/blueprints/{scope}/rollback`.

### Cost / performance

A full refresh is **O(number of scopes)**, not O(corpus size) or per onboarding request. Per scope it performs:

- **1 hybrid retrieval** (one embedding call + a cached in-memory BM25 pass over the corpus), and
- **1 LLM `generate` call** drafting the blueprint from the top ~12 retrieved chunks.

So a refresh of *global + N areas* is `N+1` retrievals and `N+1` generate calls — typically single-digit LLM calls for a whole organization. Scopes whose corpus fingerprint is unchanged are skipped entirely (no LLM call). The job is batch and schedulable; the latency-sensitive `/onboarding/path` request path never invokes generation.

## Running Tests

```bash
uv run pytest
```

With coverage:

```bash
uv run pytest --cov=src --cov-report=term-missing
```
