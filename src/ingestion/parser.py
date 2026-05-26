import logging
from pathlib import Path

from ingestion.code_parser import parse_code
from ingestion.models import ParsedChunk
from ingestion.pdf_parser import parse_pdf
from ingestion.text_parser import parse_text

logger = logging.getLogger(__name__)

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
}

PDF_EXTENSION = {
    ".pdf"
}

def parse(filename: str, content: bytes) -> list[ParsedChunk]:
    
    # dispatcher organizes wich parser to use

    file_suffix = Path(filename).suffix

    if file_suffix in CODE_EXTENSIONS:
        return parse_code(filename, content)

    elif file_suffix in TEXT_EXTENSIONS:
        return parse_text(filename, content)

    elif file_suffix in PDF_EXTENSION: 
        return parse_pdf(filename, content)

    else: 
        # unsupported file type
        logger.warning(f"Unsupported file type was given: {filename}")
        return []


