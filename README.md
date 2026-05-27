# sprintstart-ai

The AI and RAG pipeline service for [SprintStart](https://sprintstart.readthedocs.io/en/latest/) – an AI-assisted onboarding and knowledge-retrieval platform for software development teams.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management


## Environment

- `LLM_BACKEND` specifies the Backend to use (currently only OLLAMA)

### OLLAMA

- `OLLAMA_HOST` specifies the OLLAMA host
- `OLLAMA_MODEL` specifies the chat model to use
- `OLLAMA_EMBED_MODEL` specifies the embed model to use
