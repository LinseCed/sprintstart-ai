import threading
from dataclasses import dataclass
from pathlib import Path

from tree_sitter_language_pack import (
    Node,
    Parser,
    Tree,
    detect_language_from_extension,
    get_parser,
)

from ingestion.language_utils import TOP_LEVEL_NODES_FOR_SUPPORTED_LANGUAGES


@dataclass
class CodeSymbol:
    name: str | None
    kind: str
    content: str
    start_byte: int
    end_byte: int


# tree-sitter Parser objects are unsendable (pyo3): a parser may only be used on
# the thread that created it. Cache per-thread so parsers created on a FastAPI
# threadpool worker are never reused on another thread.
_thread_local_parsers = threading.local()


def get_cached_parser(language: str) -> Parser:
    cache: dict[str, Parser] = getattr(_thread_local_parsers, "parsers", None) or {}
    parser = cache.get(language)
    if parser is None:
        parser = get_parser(language)
        cache[language] = parser
        _thread_local_parsers.parsers = cache
    return parser


def parse_with_tree_sitter(
    filename: str, content: bytes
) -> tuple[list[CodeSymbol], str]:
    """
    Parse source code using tree-sitter.

    Extracts top-level symbols and the file preamble.

    Args:
        filename:
            Source filename including extension.

        content:
            Raw file bytes.

    Returns:
        Tuple consisting of:

        - list[CodeSymbol]
        - preamble string

    Raises:
        UnsupportedLanguageError:
            No supported grammar exists.

        ParseTreeError:
            Tree-sitter parsing failed.
    """
    symbols: list[CodeSymbol] = []
    preamble_parts: list[str] = []
    first_symbol_start: int | None = None
    text: str = content.decode("utf-8", errors="replace")

    extension: str = Path(filename).suffix.lower()

    if extension not in TOP_LEVEL_NODES_FOR_SUPPORTED_LANGUAGES.keys():
        raise UnsupportedLanguageError(
            f"No tree-sitter support for extension '{extension}'."
        )
    normalized_extension: str = extension.lstrip(".")

    code_language: str | None = detect_language_from_extension(normalized_extension)

    if code_language is None:
        raise UnsupportedLanguageError(
            f"Could not detect language for '{normalized_extension}'."
        )
    try:
        parser: Parser = get_cached_parser(code_language)
        tree: Tree | None = parser.parse_bytes(content)
    except Exception as exc:
        raise ParseTreeError(f"Failed creating parse tree for '{filename}'.") from exc

    if tree is None:
        raise ParseTreeError(f"Tree-sitter returned no parse tree for '{filename}'.")
    root_node: Node = tree.root_node()

    for i in range(root_node.child_count()):
        child_node = root_node.child(i)
        if child_node is None:
            continue
        if child_node.kind() in TOP_LEVEL_NODES_FOR_SUPPORTED_LANGUAGES[extension]:
            child_node_start_byte = child_node.start_byte()
            child_node_end_byte = child_node.end_byte()
            child_node_content = text[child_node_start_byte:child_node_end_byte]
            symbols.append(
                CodeSymbol(
                    _extract_name(child_node, text),
                    child_node.kind(),
                    child_node_content,
                    child_node_start_byte,
                    child_node_end_byte,
                )
            )
            first_symbol_start = (
                child_node_start_byte
                if first_symbol_start is None
                else min(first_symbol_start, child_node_start_byte)
            )
        elif first_symbol_start is None:
            preamble_parts.append(text[child_node.start_byte() : child_node.end_byte()])

    preamble = "\n".join(preamble_parts).strip()
    return symbols, preamble


def _extract_name(
    node: Node | None,
    text: str,
) -> str | None:
    """
    Extract the first identifier found in a subtree.
    """

    if node is None:
        return None

    # TODO:
    # language-specific symbol extraction
    if node.kind() == "identifier":
        return text[node.start_byte() : node.end_byte()]

    for i in range(node.child_count()):
        child = node.child(i)

        result = _extract_name(child, text)

        if result is not None:
            return result

    return None


class TreeSitterParserError(Exception):
    """Base exception for tree-sitter parsing."""


class UnsupportedLanguageError(TreeSitterParserError):
    """Raised when no tree-sitter grammar exists."""


class ParseTreeError(TreeSitterParserError):
    """Raised when tree-sitter parsing failed."""
