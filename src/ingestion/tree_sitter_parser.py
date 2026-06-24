from dataclasses import dataclass
from tree_sitter_language_pack import Parser, Tree, Node, get_parser, detect_language_from_extension
from pathlib import Path


TOP_LEVEL_NODES_FOR_SUPPORTED_LANGUAGES = {
    ".py": {
        "function_definition",
        "class_definition",
        "decorated_definition",
    },
    ".js": {
        "function_declaration",
        "class_declaration",
        "lexical_declaration",
    },
    ".ts": {
        "function_declaration",
        "class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
    },
    ".go": {
        "function_declaration",
        "method_declaration",
        "type_declaration",
    },
}


@dataclass
class  CodeSymbol:
    name: str | None
    kind: str
    content: str
    start_byte: int
    end_byte: int


def parse_with_tree_sitter(filename: str, content: bytes) -> tuple[list[CodeSymbol], str]:
    symbols: list[CodeSymbol] = []
    preamble_parts = list[str] = []
    first_symbol_start: int| None = None
    text: str = ""

    extension: str = Path(filename).suffix

    if extension not in TOP_LEVEL_NODES_FOR_SUPPORTED_LANGUAGES.keys():
        raise Exception # Todo
    
    code_language: str | None = detect_language_from_extension(extension)

    if code_language:
        parser:Parser = get_parser(code_language) # what when not possible
        tree: Tree | None = parser.parse_bytes(content) # what when not possible
        root_node: Node = tree.root_node() # type: ignore
        #what when node or child node does not exist


        for i in range(root_node.child_count()):
            child_node = root_node.child(i)
            if child_node.kind() in TOP_LEVEL_NODES_FOR_SUPPORTED_LANGUAGES[extension]: # type: ignore
                child_node_start_byte = child_node.start_byte() # type: ignore
                child_node_end_byte = child_node.end_byte() # type: ignore
                text = content.decode("utf-8", errors="replace")
                child_node_content = text[child_node_start_byte: child_node_end_byte]
                symbols.append(
                    CodeSymbol(
                        _extract_name(child_node, text),
                        child_node.kind(), # type: ignore
                        child_node_content,
                        child_node_start_byte,
                        child_node_end_byte
                    )
                )
                first_symbol_start = (
                            child_node_start_byte
                            if first_symbol_start is None
                            else min(first_symbol_start, child_node_start_byte)
                )
            else:
                # preamble candidate (imports, comments, etc.)
                if first_symbol_start is None:
                    preamble_parts.append(text[child_node_start_byte:child_node_end_byte])


    else:
        pass
        # throw some exception

    preamble = "\n".join(preamble_parts).strip()
    return symbols, preamble


def _extract_name(node: Node | None, text: str) -> str | None:
    for i in range(node.child_count): # type: ignore
        child = node.child(i) # type: ignore
        if child.kind() == "identifier": # type: ignore
            return text[child.start_byte:child.end_byte] # type: ignore
    return None