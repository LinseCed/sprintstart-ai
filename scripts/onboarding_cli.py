from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
from cli_colors import C
from rich.console import Console
from rich.table import Table

console = Console()


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

    def generate(
        self,
        working_area: str,
        experience: str,
        skills: list[str],
        tags: list[str],
    ) -> dict[str, Any] | None:
        """Stream a path; print stage progress; return the final `path` event."""
        payload = {
            "working_area": working_area,
            "experience": experience,
            "skills": skills,
            "tags": tags,
        }

        result: dict[str, Any] | None = None
        try:
            with self._http.stream(
                "POST", f"{self._base}/api/v1/onboarding/path", json=payload
            ) as response:
                if response.status_code != 200:
                    response.read()
                    self._print_http_error(response)
                    return None

                for event in _iter_sse(response):
                    kind = event.get("type")
                    if kind == "stage":
                        print(C.dim(f"  · {event.get('name')}…"))
                    elif kind == "path":
                        result = dict(event)
                    elif kind == "error":
                        message = event.get("message", "unknown error")
                        print(C.red(f"\n[error] {message}"))
                        return None
        except httpx.ConnectError:
            self.report_unreachable()
            return None
        except httpx.HTTPError as exc:
            print(C.red(f"[error] request failed: {exc}"))
            return None

        return result

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


def _render_path(event: dict[str, Any]) -> None:
    path = event.get("path", {})
    quality = event.get("quality", {})

    area = path.get("working_area", "?")
    experience = path.get("experience", "?")
    console.rule(title=f"Onboarding path — {area} / {experience}", characters="=")

    for phase in path.get("phases", []):
        print("")
        console.rule(title=str(phase.get("title", "")), characters="-")
        table = Table(expand=True)
        table.add_column("#", justify="right", no_wrap=True)
        table.add_column("step")
        table.add_column("req", no_wrap=True)
        table.add_column("origin", no_wrap=True)
        table.add_column("sources")

        for i, step in enumerate(phase.get("steps", []), start=1):
            sources = ", ".join(
                sorted({c.get("filename", "") for c in step.get("citations", [])})
            )
            req = step.get("requirement", "")
            req_cell = C.red(req) if req == "required" else C.dim(req)
            origin = step.get("origin", "")
            origin_cell = C.cyan(origin) if origin == "llm" else C.dim(origin)
            table.add_row(
                str(i),
                str(step.get("title", "")),
                req_cell,
                origin_cell,
                C.dim(sources) if sources else C.dim("—"),
            )
        console.print(table)

    _render_quality(quality)


def _render_quality(quality: dict[str, Any]) -> None:
    print("")
    console.rule(title="Quality", characters="-")
    table = Table(expand=False)
    table.add_column("coverage", justify="center")
    table.add_column("grounded", justify="center")
    table.add_column("ordering", justify="center")
    table.add_column("score", justify="center")
    table.add_row(
        f"{quality.get('coverage', 0)}",
        f"{quality.get('grounded_ratio', 0)}",
        f"{quality.get('ordering_valid', False)}",
        f"{quality.get('score', 0)}",
    )
    console.print(table)
    for note in quality.get("notes", []):
        print(C.yellow(f"  ! {note}"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a personalized onboarding path from SprintStart AI."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SPRINTSTART_URL", "http://localhost:8000"),
        help="Base URL of the service (default: %(default)s).",
    )
    parser.add_argument(
        "-a",
        "--working-area",
        help="Working area, e.g. backend, frontend, devops. Prompted if omitted.",
    )
    parser.add_argument(
        "-e",
        "--experience",
        help="Coarse experience level, e.g. junior, mid, senior. Prompted if omitted.",
    )
    parser.add_argument(
        "--skills",
        default="",
        help="Comma-separated skill tags (optional).",
    )
    parser.add_argument(
        "--tags",
        default="",
        help="Comma-separated free-form tags (optional).",
    )
    parser.add_argument(
        "--yaml",
        action="store_true",
        help="Print the raw YAML path instead of the rendered tables.",
    )
    parser.add_argument(
        "--out",
        metavar="FILE",
        help="Write the YAML path to FILE.",
    )
    return parser.parse_args()


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    args = _parse_args()
    client = Client(args.base_url)

    print(C.bold("SprintStart AI — onboarding path client"))
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
    print("")

    working_area = args.working_area
    experience = args.experience
    try:
        if not working_area:
            working_area = input(C.cyan("working area> ")).strip()
        if not experience:
            experience = input(C.cyan("experience> ")).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        client.close()
        return 1

    if not working_area or not experience:
        print(C.red("[error] working area and experience are required"))
        client.close()
        return 1

    try:
        event = client.generate(
            working_area, experience, _csv(args.skills), _csv(args.tags)
        )
    finally:
        client.close()

    if event is None:
        return 1

    path_yaml = str(event.get("path_yaml", ""))

    if args.out:
        Path(args.out).expanduser().write_text(path_yaml, encoding="utf-8")
        print(C.green(f"wrote YAML to {args.out}"))

    if args.yaml:
        print(path_yaml)
    else:
        _render_path(event)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
