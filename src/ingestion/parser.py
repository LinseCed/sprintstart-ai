import logging
from pathlib import Path

from ingestion.models import ParsedChunk
from ingestion.text_parser import parse_text

logger = logging.getLogger(__name__)

# TODO: discuss what to add
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".go",
}

# TODO: discuss what to add
TEXT_EXTENSIONS = {
    ".txt",
    ".json",
    ".md",
    ".yaml",
    ".toml",
}

PDF_EXTENSION = {".pdf"}


def parse(filename: str, content: bytes) -> list[ParsedChunk]:
    """Parse a file into structured chunks based on its file extension.

    The parser acts as a dispatcher and forwards the file content
    to the appropriate parser implementation.

    Supported file types:
    - Source code files (.py, .js, .ts, .go) -> parsed into code chunks
    - Text-based files (.txt, .json, .md) -> single text chunk
    - PDF files (.pdf) -> extracted text chunk

    Unsupported file types return an empty list and log a warning
    instead of raising an exception.

    Args:
        filename (str):  Name of the uploaded file including extension.
        content (bytes): Raw file content as bytes.

    Returns:
        list[ParsedChunk]: A list of ParsedChunk objects extracted from the file.
                           Returns an empty list if the file type is unsupported.
    """

    # dispatcher organizes wich parser to use

    file_suffix = Path(filename).suffix

    if file_suffix in CODE_EXTENSIONS:
        raise NotImplementedError

    elif file_suffix in TEXT_EXTENSIONS:
        return parse_text(filename, content)

    elif file_suffix in PDF_EXTENSION:
        raise NotImplementedError

    else:
        # unsupported file type
        logger.warning(f"Unsupported file type was given: {filename}")
        return []
