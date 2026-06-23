from __future__ import annotations

import argparse
from collections import Counter
from typing import Any

import httpx
from _client import ServiceClient, add_base_url_arg
from cli_colors import C
from rich.console import Console
from rich.table import Table

console = Console()

_SNIPPET = 80


def _snippet(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= _SNIPPET else flat[: _SNIPPET - 1] + "…"


def _status(client: ServiceClient) -> int:
    try:
        response = client.get("/api/v1/vector-db/status")
    except httpx.ConnectError:
        client.report_unreachable()
        return 1
    if response.status_code != 200:
        client.print_http_error(response)
        return 1

    data = response.json()
    backend = data.get("backend", "?")
    count = data.get("chunk_count", 0)
    print(C.dim("store:  ") + C.bold(str(backend)))
    print(C.dim("chunks: ") + C.bold(str(count)))

    if count:
        _print_artifact_summary(client)
    else:
        print(C.yellow("\nthe corpus is empty — ingest documents first:"))
        print(C.dim("  sprintstart ingest <path>"))
    return 0


def _print_artifact_summary(client: ServiceClient) -> None:
    """Group the first page(s) of chunks by artifact for a quick overview."""
    chunks = _collect_chunks(client, max_chunks=500)
    if not chunks:
        return
    by_artifact = Counter(
        str(c.get("filename", c.get("artifact_id", "?"))) for c in chunks
    )

    print("")
    table = Table(title="artifacts (sampled)", expand=False)
    table.add_column("filename")
    table.add_column("chunks", justify="right")
    for filename, n in by_artifact.most_common():
        table.add_row(filename, str(n))
    console.print(table)


def _collect_chunks(client: ServiceClient, max_chunks: int) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    offset = 0
    page = 100
    while len(collected) < max_chunks:
        try:
            response = client.get("/api/v1/vector-db/chunks", limit=page, offset=offset)
        except httpx.HTTPError:
            break
        if response.status_code != 200:
            break
        data = response.json()
        items = data.get("items", [])
        collected.extend(items)
        offset += page
        if offset >= data.get("total", 0) or not items:
            break
    return collected


def _list(client: ServiceClient, args: argparse.Namespace) -> int:
    if args.artifact:
        path = f"/api/v1/vector-db/artifacts/{args.artifact}/chunks"
    else:
        path = "/api/v1/vector-db/chunks"
    try:
        response = client.get(path, limit=args.limit, offset=args.offset)
    except httpx.ConnectError:
        client.report_unreachable()
        return 1
    if response.status_code != 200:
        client.print_http_error(response)
        return 1

    data = response.json()
    items = data.get("items", [])
    print(
        C.dim(
            f"showing {len(items)} of {data.get('total', 0)} "
            f"(offset {data.get('offset', 0)})"
        )
    )
    _render_chunk_table(items)
    return 0


def _search(client: ServiceClient, args: argparse.Namespace) -> int:
    payload: dict[str, object] = {
        "query": args.query,
        "top_k": args.top_k,
        "min_score": args.min_score,
    }
    try:
        response = client.post("/api/v1/vector-db/search", payload)
    except httpx.ConnectError:
        client.report_unreachable()
        return 1
    if response.status_code != 200:
        client.print_http_error(response)
        return 1

    items = response.json().get("items", [])
    if not items:
        print(C.yellow("no matches"))
        return 0
    _render_chunk_table(items, show_score=True)
    return 0


def _render_chunk_table(items: list[dict[str, Any]], show_score: bool = False) -> None:
    table = Table(expand=True)
    if show_score:
        table.add_column("score", justify="right", no_wrap=True)
    table.add_column("filename", no_wrap=True)
    table.add_column("pos", justify="right", no_wrap=True)
    table.add_column("text")
    for item in items:
        row = []
        if show_score:
            row.append(f"{item.get('score', 0):.3f}")
        row.extend(
            [
                str(item.get("filename", "?")),
                str(item.get("position", "")),
                C.dim(_snippet(str(item.get("text", "")))),
            ]
        )
        table.add_row(*row)
    console.print(table)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="corpus_action")

    sub.add_parser("status", help="Show backend + chunk count and artifact summary.")

    p_list = sub.add_parser("list", help="List stored chunks (paginated).")
    p_list.add_argument("--artifact", help="Limit to one artifact_id.")
    p_list.add_argument("--limit", type=int, default=20)
    p_list.add_argument("--offset", type=int, default=0)

    p_search = sub.add_parser("search", help="Embed a query and search the store.")
    p_search.add_argument("query", help="Query text.")
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--min-score", type=float, default=0.0)


def run(args: argparse.Namespace) -> int:
    client = ServiceClient(args.base_url)
    try:
        action = getattr(args, "corpus_action", None)
        if action == "list":
            return _list(client, args)
        if action == "search":
            return _search(client, args)
        return _status(client)
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect the ingested corpus (vector store) of SprintStart AI."
    )
    add_base_url_arg(parser)
    add_arguments(parser)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
