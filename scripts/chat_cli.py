from __future__ import annotations

import argparse
import sys

from _client import ServiceClient, add_base_url_arg
from cli_colors import C
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown


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
                transient=True,
            )
            self._live.start()
        self._live.update(Markdown(self._buffer), refresh=True)

    def close(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._console.print(Markdown(self._buffer))
            self._live = None
        elif not self._console.is_terminal:
            sys.stdout.write("\n")
            sys.stdout.flush()


class ChatClient(ServiceClient):
    def __init__(self, base_url: str) -> None:
        super().__init__(base_url)
        self.history: list[dict[str, str]] = []

    def ask(self, prompt: str) -> None:
        payload: dict[str, object] = {"prompt": prompt, "context": self.history}

        answer_parts: list[str] = []
        citations: list[dict[str, object]] = []
        printed_prefix = False
        renderer = MarkdownStream(self.console)

        for event in self.events("/api/v1/chat", payload):
            kind = event.get("type")
            if kind == "tool_use":
                print(C.dim(f"  · {event.get('name')} [{event.get('kind')}]"))
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
                print(C.red(f"\n[error] {event.get('message', 'unknown error')}"))
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
        seen: set[object] = set()
        print(C.dim("sources:"))
        for c in citations:
            filename = c.get("filename")
            if filename in seen:
                continue
            seen.add(filename)
            print(C.dim(f"  - {filename}"))


HELP = """\
commands:
  <anything>              ask a question about your ingested files
  /ingest <path> [id]     ingest a file or directory (artifact id defaults to filename)
  /history                show the conversation history used for context
  /reset                  clear the conversation history
  /help                   show this help
  /quit                   exit
"""


def _handle_command(client: ChatClient, line: str) -> bool:
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


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """No chat-specific args; the REPL is interactive."""


def run(args: argparse.Namespace) -> int:
    client = ChatClient(args.base_url)
    client.print_banner("SprintStart AI — chat")
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat client for SprintStart AI.")
    add_base_url_arg(parser)
    add_arguments(parser)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
