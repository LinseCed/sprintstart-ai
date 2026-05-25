from rag.types import Citation, ScoredChunk


def build_citations(chunks: list[ScoredChunk]) -> list[Citation]:
    return [
        Citation(
            filename=chunk.filename,
            section_path=chunk.heading_path,
            chunk_id=chunk.id,
        )
        for chunk in chunks
    ]
