"""Review and govern AI-proposed onboarding blueprints over the HTTP service.

Drafts are produced by the generation job (the API `POST .../generate`, also
exposed here as `generate`, or the offline `blueprint_refresh.py`). They are
never activated automatically — this tool is the human-approval surface:

    sprintstart blueprints generate [--scope global ...]
    sprintstart blueprints drafts
    sprintstart blueprints diff <scope>
    sprintstart blueprints approve <scope>
    sprintstart blueprints reject <scope>
    sprintstart blueprints versions <scope>
    sprintstart blueprints rollback <scope> <version>
"""

from __future__ import annotations

import argparse

import httpx
from _client import ServiceClient, add_base_url_arg
from cli_colors import C
from rich.console import Console
from rich.table import Table

console = Console()

_BASE = "/api/v1/onboarding/blueprints"


def _handle(client: ServiceClient, response: httpx.Response) -> bool:
    if response.status_code not in (200, 204):
        client.print_http_error(response)
        return False
    return True


def _generate(client: ServiceClient, args: argparse.Namespace) -> int:
    response = client.post(f"{_BASE}/generate", {"scopes": args.scope or None})
    if not _handle(client, response):
        return 1
    outcomes = response.json().get("outcomes", [])
    table = Table(title="generation outcomes", expand=False)
    table.add_column("status", no_wrap=True)
    table.add_column("scope")
    table.add_column("version", justify="right")
    table.add_column("notes")
    for o in outcomes:
        table.add_row(
            str(o.get("status", "")),
            str(o.get("scope", "")),
            str(o.get("draft_version") or "—"),
            C.dim("; ".join(o.get("notes", [])) or "—"),
        )
    console.print(table)
    return 0


def _drafts(client: ServiceClient, _args: argparse.Namespace) -> int:
    response = client.get(f"{_BASE}/drafts")
    if not _handle(client, response):
        return 1
    items = response.json().get("items", [])
    if not items:
        print(C.dim("no drafts awaiting review"))
        return 0
    table = Table(title="pending drafts", expand=False)
    table.add_column("scope")
    table.add_column("version", justify="right")
    table.add_column("steps", justify="right")
    table.add_column("changes", justify="right")
    table.add_column("blocked", no_wrap=True)
    for item in items:
        bp = item.get("blueprint", {})
        diff = item.get("diff", {})
        blocked = diff.get("blocked", False)
        table.add_row(
            str(bp.get("scope", "")),
            str(bp.get("version", "")),
            str(len(bp.get("steps", []))),
            str(len(diff.get("changes", []))),
            C.red("yes") if blocked else C.dim("no"),
        )
    console.print(table)
    return 0


def _diff(client: ServiceClient, args: argparse.Namespace) -> int:
    response = client.get(f"{_BASE}/drafts/{args.scope}/diff")
    if not _handle(client, response):
        return 1
    diff = response.json()
    console.rule(
        title=f"{diff.get('scope')} — v{diff.get('active_version')} → "
        f"v{diff.get('draft_version')}"
    )
    table = Table(expand=False)
    table.add_column("change", no_wrap=True)
    table.add_column("step id")
    table.add_column("protected", no_wrap=True)
    for change in diff.get("changes", []):
        kind = change.get("change", "")
        protected = change.get("protected", False)
        colour = C.red if kind in ("removed", "downgraded") else C.dim
        table.add_row(
            colour(kind),
            str(change.get("id", "")),
            C.yellow("yes") if protected else C.dim("no"),
        )
    console.print(table)
    if diff.get("blocked"):
        print(C.red("\n! blocked: a protected step would be removed or downgraded"))
    return 0


def _approve(client: ServiceClient, args: argparse.Namespace) -> int:
    response = client.post(f"{_BASE}/drafts/{args.scope}/approve", {})
    if not _handle(client, response):
        return 1
    bp = response.json()
    print(C.green(f"approved {bp.get('scope')} → active v{bp.get('version')}"))
    return 0


def _reject(client: ServiceClient, args: argparse.Namespace) -> int:
    response = client.delete(f"{_BASE}/drafts/{args.scope}")
    if not _handle(client, response):
        return 1
    print(C.dim(f"discarded draft for {args.scope}"))
    return 0


def _versions(client: ServiceClient, args: argparse.Namespace) -> int:
    response = client.get(f"{_BASE}/{args.scope}/versions")
    if not _handle(client, response):
        return 1
    versions = response.json().get("versions", [])
    print(C.dim(f"{args.scope}: ") + (", ".join(versions) or C.dim("(none)")))
    return 0


def _rollback(client: ServiceClient, args: argparse.Namespace) -> int:
    response = client.post(f"{_BASE}/{args.scope}/rollback", {"version": args.version})
    if not _handle(client, response):
        return 1
    bp = response.json()
    print(C.green(f"rolled back {bp.get('scope')} to v{bp.get('version')}"))
    return 0


_ACTIONS = {
    "generate": _generate,
    "drafts": _drafts,
    "diff": _diff,
    "approve": _approve,
    "reject": _reject,
    "versions": _versions,
    "rollback": _rollback,
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="blueprint_action")

    p_gen = sub.add_parser("generate", help="Draft/update blueprints from the corpus.")
    p_gen.add_argument(
        "--scope", action="append", default=[], help="Scope to generate; repeatable."
    )

    sub.add_parser("drafts", help="List pending drafts.")

    for name, help_text in (
        ("diff", "Show a draft's diff against active."),
        ("approve", "Promote a draft to active."),
        ("reject", "Discard a draft."),
        ("versions", "List retained versions."),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("scope", help="Blueprint scope, e.g. global or area:backend.")

    p_rb = sub.add_parser("rollback", help="Restore a retained version as active.")
    p_rb.add_argument("scope", help="Blueprint scope, e.g. global or area:backend.")
    p_rb.add_argument("version", help="Version to restore.")


def run(args: argparse.Namespace) -> int:
    action = getattr(args, "blueprint_action", None)
    if action is None:
        print(C.yellow("specify a subcommand: " + ", ".join(_ACTIONS)))
        return 1
    client = ServiceClient(args.base_url)
    try:
        return _ACTIONS[action](client, args)
    except httpx.ConnectError:
        client.report_unreachable()
        return 1
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Review AI-proposed blueprints.")
    add_base_url_arg(parser)
    add_arguments(parser)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
