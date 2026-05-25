from src.rag.types import Chunk, Citation


def build_citations(chunks: list[Chunk]) -> list[Citation]:
    return [
        Citation(
            filename=chunk.filename,
            section_path=chunk.heading_path,
            chunk_id=chunk.id,
        )
        for chunk in chunks
    ]