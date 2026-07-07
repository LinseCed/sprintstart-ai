"""
Manual test client for the SprintStart AI API.
Run the API first: uv run uvicorn api.app:app --reload --app-dir src
"""

import json

import httpx

BASE_URL = "http://localhost:8000/api/v1"


def check_health() -> None:
    print("--- GET /health ---")
    response = httpx.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Body:   {response.json()}\n")


def ingest_document(artifact_id: str, filename: str, content: str) -> None:
    print(f"--- POST /ingest ({filename}) ---")
    response = httpx.post(
        f"{BASE_URL}/ingest",
        json={"artifact_id": artifact_id, "filename": filename, "content": content},
    )
    print(f"Status: {response.status_code}")
    print(f"Body:   {response.json()}\n")


def chat(question: str, history: list[dict[str, str]] | None = None) -> None:
    print(f"--- POST /chat: {question!r} ---")
    with httpx.stream(
        "POST",
        f"{BASE_URL}/chat",
        json={
            "prompt": question,
            "top_k": 5,
            "min_score": 0.5,
            "context": history or [],
        },
        timeout=60,
    ) as response:
        print(f"Status: {response.status_code}")
        for line in response.iter_lines():
            if line.startswith("data: "):
                event = json.loads(line[6:])
                event_type = event.get("type")
                if event_type == "token":
                    print(event["content"], end="", flush=True)
                elif event_type == "citation":
                    print(
                        f"\n[citation] {event['filename']} "
                        f"chunk={event['chunk_id']} artifact={event['artifact_id']}"
                    )
                elif event_type == "done":
                    print("\n[done]")
                elif event_type == "error":
                    print(f"\n[error] {event['message']}")
    print()


if __name__ == "__main__":
    check_health()

    ingest_document(
        artifact_id="sprint-42-retro",
        filename="retro.md",
        content=(
            "# Sprint 42 Retro\n\n"
            "## What went well\n"
            "Good collaboration between frontend and backend teams.\n\n"
            "## Blockers\n"
            "Missing designs delayed the auth feature by 3 days. "
            "CI pipeline was flaky and caused multiple failed deploys.\n\n"
            "## Action items\n"
            "- Design handoff process to be agreed before sprint start.\n"
            "- Investigate flaky tests in CI."
        ),
    )

    chat("What were the main blockers in sprint 42?")

    chat(
        question="Can you summarize that?",
        history=[
            {"role": "user", "content": "What were the main blockers in sprint 42?"},
            {
                "role": "assistant",
                "content": (
                    "The main blockers were missing designs and a flaky CI pipeline."
                ),
            },
        ],
    )
