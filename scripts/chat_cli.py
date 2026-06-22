from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import httpx
from cli_colors import C
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

TEXT_EXTENSIONS = {".txt", ".json", ".md", ".yaml", ".yml", ".toml"}
CODE_EXTENSIONS = {".py", ".js", ".ts", ".go"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
INGESTABLE = TEXT_EXTENSIONS | CODE_EXTENSIONS | IMAGE_EXTENSIONS


class MarkdownStream:
    """Live-renders a streamed markdown answer with rich.

    On a terminal the accumulated markdown is re-rendered in a Live region as
    each token arrives. When output is redirected (a pipe, or a non-terminal),
    tokens are written through verbatim so non-interactive output stays plain.
    """

    def __init__(self, console: Console) -> None:
        self._console = console
        self._buffer = ""
        self._live: Live | None = None

    def feed(self, text: str) -> None:
        self._buffer += text
        if not self._console.is_terminal:
            sys.stdout.write(text)
            sys.stdout.flush()
            return
        if self._live is None:
            self._live = Live(
                console=self._console,
                auto_refresh=False,
                vertical_overflow="visible",
            )
            self._live.start()
        self._live.update(Markdown(self._buffer), refresh=True)

    def close(self) -> None:
        if self._live is not None:
            self._live.update(Markdown(self._buffer), refresh=True)
            self._live.stop()
            self._live = None
        elif not self._console.is_terminal:
            sys.stdout.write("\n")
            sys.stdout.flush()


def _iter_sse(response: httpx.Response) -> Iterator[dict[str, object]]:
    """Yield decoded JSON payloads from an SSE `data:` stream."""
    for line in response.iter_lines():
        if line.startswith("data: "):
            try:
                yield json.loads(line[len("data: ") :])
            except json.JSONDecodeError:
                continue


class Client:
    def __init__(self, base_url: str) -> None:
        self._base = base_url.rstrip("/")
        self._http = httpx.Client(timeout=httpx.Timeout(None, connect=5.0))
        self._console = Console()
        self.history: list[dict[str, str]] = []

    def ask(self, prompt: str) -> None:
        payload = {
            "prompt": prompt,
            "context": self.history,
        }

        answer_parts: list[str] = []
        citations: list[dict[str, object]] = []
        printed_prefix = False
        renderer = MarkdownStream(self._console)

        try:
            with self._http.stream(
                "POST", f"{self._base}/api/v1/chat", json=payload
            ) as response:
                if response.status_code != 200:
                    response.read()
                    self._print_http_error(response)
                    return

                for event in _iter_sse(response):
                    kind = event.get("type")
                    if kind == "tool_use":
                        name = event.get("name")
                        cap = event.get("kind")
                        print(C.dim(f"  · {name} [{cap}]"))
                    elif kind == "token":
                        if not printed_prefix:
                            sys.stdout.write(C.green("ai>") + "\n")
                            sys.stdout.flush()
                            printed_prefix = True
                        content = str(event.get("content", ""))
                        renderer.feed(content)
                        answer_parts.append(content)
                    elif kind == "citation":
                        citations.append(event)
                    elif kind == "error":
                        message = event.get("message", "unknown error")
                        print(C.red(f"\n[error] {message}"))
                        return
        except httpx.ConnectError:
            self.report_unreachable()
            return
        except httpx.HTTPError as exc:
            print(C.red(f"[error] request failed: {exc}"))
            return

        if printed_prefix:
            renderer.close()
        self._print_citations(citations)

        answer = "".join(answer_parts).strip()
        if answer:
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": answer})

    def _print_citations(self, citations: list[dict[str, object]]) -> None:
        if not citations:
            return
        seen: set[tuple[object, object]] = set()
        print(C.dim("sources:"))
        for c in citations:
            key = (c.get("filename"), c.get("section_path"))
            if key in seen:
                continue
            seen.add(key)
            section = c.get("section_path")
            suffix = f" › {section}" if section else ""
            print(C.dim(f"  - {c.get('filename')}{suffix}"))

    def ingest_path(self, raw_path: str, artifact_id: str | None) -> None:
        path = Path(raw_path).expanduser()
        if not path.exists():
            print(C.red(f"[error] no such path: {path}"))
            return

        if path.is_dir():
            files = sorted(
                p for p in path.rglob("*") if p.is_file() and p.suffix in INGESTABLE
            )
            if not files:
                print(C.yellow(f"no ingestable files found under {path}"))
                return
            print(C.dim(f"ingesting {len(files)} file(s) from {path}…"))
            for file in files:
                self._ingest_file(file, artifact_id=None)
        else:
            self._ingest_file(path, artifact_id=artifact_id)

    def _ingest_file(self, path: Path, artifact_id: str | None) -> None:
        suffix = path.suffix.lower()
        if suffix not in INGESTABLE:
            hint = ""
            if suffix == ".pdf":
                hint = " (PDFs aren't supported over this JSON API)"
            print(C.yellow(f"  skip {path.name}: unsupported type {suffix!r}{hint}"))
            return

        try:
            raw = path.read_bytes()
        except OSError as exc:
            print(C.red(f"  fail {path.name}: {exc}"))
            return

        if suffix in IMAGE_EXTENSIONS:
            content = base64.b64encode(raw).decode("ascii")
        else:
            content = raw.decode("utf-8", errors="replace")

        body = {
            "artifact_id": artifact_id or path.name,
            "filename": path.name,
            "content": content,
        }

        try:
            response = self._http.post(f"{self._base}/api/v1/ingest", json=body)
        except httpx.ConnectError:
            self.report_unreachable()
            return
        except httpx.HTTPError as exc:
            print(C.red(f"  fail {path.name}: {exc}"))
            return

        if response.status_code == 200:
            data = response.json()
            count = data.get("chunk_count", "?")
            note = C.yellow(" (0 chunks — nothing stored)") if count == 0 else ""
            print(
                C.green(f"  ok   {path.name}")
                + C.dim(f" → artifact '{body['artifact_id']}', {count} chunk(s)")
                + note
            )
        else:
            detail = self._detail(response)
            print(C.red(f"  fail {path.name}: {response.status_code} {detail}"))

    def health(self) -> tuple[str, str | None] | None:
        try:
            response = self._http.get(f"{self._base}/api/v1/health", timeout=20.0)
        except httpx.HTTPError:
            return None
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError):
            return ("unknown", None)
        detail = data.get("detail")
        return str(data.get("status", "unknown")), str(detail) if detail else None

    def report_unreachable(self) -> None:
        print(C.red(f"[error] cannot reach the service at {self._base}"))
        print(C.dim("        is it running?  uv run python -m src.main"))

    def _print_http_error(self, response: httpx.Response) -> None:
        print(C.red(f"[error] {response.status_code} {self._detail(response)}"))

    @staticmethod
    def _detail(response: httpx.Response) -> str:
        try:
            return str(response.json().get("detail", response.text))
        except (json.JSONDecodeError, ValueError):
            return response.text

    def close(self) -> None:
        self._http.close()


HELP = """\
commands:
  <anything>              ask a question about your ingested files
  /ingest <path> [id]     ingest a file or directory (artifact id defaults to filename)
  /history                show the conversation history used for context
  /reset                  clear the conversation history
  /help                   show this help
  /quit                   exit
"""


def _handle_command(client: Client, line: str) -> bool:
    parts = line.split()
    cmd = parts[0].lower()

    if cmd in ("/quit", "/exit", "/q"):
        return False
    if cmd in ("/help", "/h", "/?"):
        print(HELP)
    elif cmd == "/reset":
        client.history.clear()
        print(C.dim("history cleared."))
    elif cmd == "/history":
        if not client.history:
            print(C.dim("(history is empty)"))
        for entry in client.history:
            role = entry["role"]
            tag = C.cyan(role) if role == "user" else C.green(role)
            print(f"{tag}: {entry['content']}")
    elif cmd == "/ingest":
        if len(parts) < 2:
            print(C.yellow("usage: /ingest <path> [artifact_id]"))
        else:
            artifact_id = parts[2] if len(parts) > 2 else None
            client.ingest_path(parts[1], artifact_id)
    else:
        print(C.yellow(f"unknown command {cmd!r} — try /help"))
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Terminal client for SprintStart AI.")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SPRINTSTART_URL", "http://localhost:8000"),
        help="Base URL of the service (default: %(default)s).",
    )
    args = parser.parse_args()

    client = Client(args.base_url)

    print(C.bold("SprintStart AI — terminal client"))
    print(C.dim(f"connected to {args.base_url}"))
    result = client.health()
    if result is None:
        client.report_unreachable()
    else:
        status, detail = result
        colour = C.green if status == "ok" else C.yellow
        line = C.dim("health: ") + colour(status)
        if detail:
            line += C.dim(f" — {detail}")
        print(line)
    print(C.dim("type /help for commands, /quit to exit\n"))

    try:
        while True:
            try:
                line = input(C.cyan("you> ")).strip()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print(C.dim("\n(use /quit to exit)"))
                continue

            if not line:
                continue
            if line.startswith("/"):
                if not _handle_command(client, line):
                    break
            else:
                client.ask(line)
    finally:
        client.close()

    print(C.dim("bye."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
