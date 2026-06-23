"""Offline, schedulable blueprint-generation job (issue #110).

Runs the corpus → draft generation in-process against the configured vector
store and LLM backend, then prints the per-scope outcomes. Drafts land in the
review queue (``blueprints/drafts/``); nothing is activated here — promotion is
the human-approved API/CLI step.

Intended for cron/CI, e.g. a nightly refresh:

    uv run python scripts/blueprint_refresh.py
    uv run python scripts/blueprint_refresh.py --scope global --scope area:backend

Unlike the other terminal tools this talks to the store/LLM directly rather than
the HTTP service, so it can be scheduled without the API running.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# This job imports application code directly (not over HTTP), so put ``src`` on
# the path the same way pytest does (pyproject ``pythonpath = ["src"]``).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cli_colors import C  # noqa: E402


def run(args: argparse.Namespace) -> int:
    load_dotenv()

    from api.dependencies import get_llm, get_store
    from onboarding.generation import generate_blueprints

    print(C.bold("SprintStart AI — blueprint refresh"))
    store = get_store()
    print(C.dim(f"corpus chunks: {store.count()}"))

    outcomes = generate_blueprints(get_llm(), store, scopes=args.scope or None)

    print("")
    for outcome in outcomes:
        colour = {
            "created": C.green,
            "updated": C.green,
            "escalated": C.yellow,
            "unchanged": C.dim,
            "skipped": C.dim,
        }.get(outcome.status, C.dim)
        version = f" v{outcome.draft_version}" if outcome.draft_version else ""
        print(f"  {colour(outcome.status):<12} {outcome.scope}{version}")
        for note in outcome.notes:
            print(C.dim(f"      · {note}"))

    drafted = [o for o in outcomes if o.draft_version]
    if drafted:
        print(C.dim(f"\n{len(drafted)} draft(s) written to the review queue."))
        print(C.dim("review & approve:  sprintstart blueprints drafts"))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scope",
        action="append",
        default=[],
        help="Scope to (re)generate; repeatable. Default: all known scopes.",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
