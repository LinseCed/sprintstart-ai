from dataclasses import dataclass


@dataclass(slots = True)
class ParsedChunk:
    """
    Represents a processed text chunk produced by a file parser.

    Attributes:
        content (str): The raw or extracted text content of the chunk.
        kind (str): The type/category of the chunk (e.g. "text", "code", "markdown").
        metadata (dict[str, str]): Additional information about the chunk,
            such as source file name, position/index, or encoding details.
    """

    content: str
    kind: str
    metadata: dict[str, str]
