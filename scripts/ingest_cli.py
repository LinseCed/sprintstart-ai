from __future__ import annotations

import argparse

from _client import ServiceClient, add_base_url_arg


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", help="File or directory to ingest.")
    parser.add_argument(
        "artifact_id",
        nargs="?",
        default=None,
        help="Artifact id (defaults to the filename; ignored for directories).",
    )


def run(args: argparse.Namespace) -> int:
    client = ServiceClient(args.base_url)
    try:
        client.ingest_path(args.path, args.artifact_id)
    finally:
        client.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest documents into SprintStart AI."
    )
    add_base_url_arg(parser)
    add_arguments(parser)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
