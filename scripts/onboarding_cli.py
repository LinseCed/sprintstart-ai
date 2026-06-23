from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _client import ServiceClient, add_base_url_arg
from cli_colors import C
from rich.console import Console
from rich.table import Table

console = Console()


def _generate(
    client: ServiceClient,
    working_area: str,
    experience: str,
    skills: list[str],
    tags: list[str],
) -> dict[str, Any] | None:
    """Stream a path; print stage progress; return the final `path` event."""
    payload: dict[str, object] = {
        "working_area": working_area,
        "experience": experience,
        "skills": skills,
        "tags": tags,
    }

    result: dict[str, Any] | None = None
    for event in client.events("/api/v1/onboarding/path", payload):
        kind = event.get("type")
        if kind == "stage":
            print(C.dim(f"  · {event.get('name')}…"))
        elif kind == "path":
            result = dict(event)
        elif kind == "error":
            print(C.red(f"\n[error] {event.get('message', 'unknown error')}"))
            return None
    return result


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


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def add_arguments(parser: argparse.ArgumentParser) -> None:
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
        "--skills", default="", help="Comma-separated skill tags (optional)."
    )
    parser.add_argument(
        "--tags", default="", help="Comma-separated free-form tags (optional)."
    )
    parser.add_argument(
        "--yaml",
        action="store_true",
        help="Print the raw YAML path instead of the rendered tables.",
    )
    parser.add_argument("--out", metavar="FILE", help="Write the YAML path to FILE.")


def run(args: argparse.Namespace) -> int:
    client = ServiceClient(args.base_url)
    client.print_banner("SprintStart AI — onboarding path")
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
        event = _generate(
            client, working_area, experience, _csv(args.skills), _csv(args.tags)
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a personalized onboarding path from SprintStart AI."
    )
    add_base_url_arg(parser)
    add_arguments(parser)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
