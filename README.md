# sprintstart-ai

The AI and RAG pipeline service for [SprintStart](https://sprintstart.readthedocs.io/en/latest/), an AI-assisted onboarding and knowledge-retrieval platform for software development teams.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally with the required models pulled:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
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

| Variable | Description |
|---|---|
| `LLM_BACKEND` | LLM backend to use. Currently only `ollama` is supported. |
| `OLLAMA_BASE_URL` | Base URL of the Ollama instance. Use `http://host.docker.internal:11434` when running via Docker with Ollama on the host. |
| `OLLAMA_MODEL` | Chat model to use for generation. |
| `OLLAMA_EMBED_MODEL` | Embedding model to use for ingestion and retrieval. |
| `CHROMA_PATH` | Path for ChromaDB persistent storage. If unset, an in-memory store is used and data will not persist. |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Reports service health including LLM backend status. Returns `503` if Ollama is unreachable. |
| `POST` | `/api/v1/ingest` | Parses, chunks, and embeds a document and stores it in the vector store. Re-ingesting the same `artifact_id` replaces existing chunks. |
| `POST` | `/api/v1/chat` | Retrieves relevant chunks and streams a generated answer as Server-Sent Events (SSE). |
| `POST` | `/api/v1/title` | Generates a short descriptive title from a user prompt using an LLM and respecting the given max character length.

### Chat SSE stream

The `/api/v1/chat` endpoint streams newline-delimited JSON events:

| Event type | Description |
|---|---|
| `token` | A single token fragment of the answer |
| `citation` | A source chunk used to generate the answer |
| `done` | Signals the end of the stream |
| `error` | Emitted on failure instead of the above |

## Running Tests

```bash
uv run pytest
```

With coverage:

```bash
uv run pytest --cov=src --cov-report=term-missing

```
