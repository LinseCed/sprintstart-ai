import logging
from pathlib import Path

from ingestion.code_parser import parse_code
from ingestion.image_parser import parse_image
from ingestion.language_utils import TOP_LEVEL_NODES_FOR_SUPPORTED_LANGUAGES
from ingestion.models import ParsedChunk
from ingestion.pdf_parser import parse_pdf
from ingestion.text_parser import parse_text

logger = logging.getLogger(__name__)

CODE_EXTENSIONS = TOP_LEVEL_NODES_FOR_SUPPORTED_LANGUAGES.keys()

PDF_EXTENSION = {".pdf"}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def parse(filename: str, content: bytes) -> list[ParsedChunk]:
    """Parse a file into structured chunks based on its file extension.

    The parser acts as a dispatcher and forwards the file content
    to the appropriate parser implementation.

    Supported file types:
    - Source code files (.py, .js, .ts, .go) -> parsed into code chunks
    - Text-based files (.txt, .json, .md) -> single text chunk
    - PDF files (.pdf) -> extracted text chunk
    - Image files (.png, .jpg, .jpeg, .gif, .webp, .bmp) -> image chunk

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
        return parse_code(filename, content)

    elif file_suffix in PDF_EXTENSION:
        return parse_pdf(filename, content)

    elif file_suffix in IMAGE_EXTENSIONS:
        return parse_image(filename, content)

    else:
        return parse_text(filename, content)
