from rag.types import Citation, ScoredChunk


def build_citations(chunks: list[ScoredChunk]) -> list[Citation]:
    return [
        Citation(
            filename=chunk.filename,
            chunk_id=chunk.id,
        )
        for chunk in chunks
    ]
