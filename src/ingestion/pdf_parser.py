import io
import logging

from pypdf import PdfReader

from ingestion.chunker import chunk_text
from ingestion.models import ParsedChunk

logger = logging.getLogger(__name__)


def parse_pdf(filename: str, content: bytes) -> list[ParsedChunk]:
    """
    Parse a PDF file into structured text chunks.

    Each PDF page is processed independently. Extracted text is split
    into fixed-size chunks using `chunk_text` (max 512 characters per chunk).

    Each resulting chunk is enriched with PDF-specific metadata.

    Behavior:
        - Each page is processed separately
        - Pages without extractable text are skipped with a warning
        - Each page is split into multiple chunks if needed
        - Each chunk is labeled with kind="pdf"
        - Page-level metadata is attached to each chunk

    Metadata added:
        - page_number: The 1-based index of the PDF page the chunk belongs to
        - chunk_index: Sequential index of the chunk within the current page
        - global_pdf_chunk_index: Sequential global index of the chunk across the entire PDF document

    Args:
        filename (str):
            Name of the source PDF file.

        content (bytes):
            Raw binary content of the uploaded PDF file.

    Returns:
        list[ParsedChunk]:
            A flat list of all extracted and chunked PDF content,
            enriched with page and chunk metadata.
    """
    
    try:
        reader = PdfReader(io.BytesIO(content)) # in memory processing
    except Exception as e:
        logger.error("Failed to read PDF %s: %s", filename, e)
        return [] 
    
    pdf_chunks: list[ParsedChunk] = []
    global_pdf_chunk_index: int = 0
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text()
        except Exception as e:
            logger.warning(
                "Failed to extract text from page %s in %s: %s",page_number,filename,e)
            continue 

        if not text or not text.strip():
            logger.warning("Skipping empty page %s in %s",page_number,filename)
            continue

        chunks: list[ParsedChunk] = chunk_text(filename, text)
        for chunk in chunks:
            chunk.kind = "pdf"
            chunk.metadata["global_pdf_chunk_index"] = str(global_pdf_chunk_index)
            chunk.metadata["page_number"] = str(page_number)
            pdf_chunks.append(chunk)
            global_pdf_chunk_index += 1

    return pdf_chunks