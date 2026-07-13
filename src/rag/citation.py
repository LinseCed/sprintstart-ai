from rag.types import Citation, ScoredChunk


def build_citations(chunks: list[ScoredChunk]) -> list[Citation]:
    return [
        Citation(
            chunk_id=chunk.id,
            filename=chunk.filename,
            source_url=chunk.source_url,
        )
        for chunk in chunks
    ]
