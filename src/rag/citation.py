from rag.types import Citation, ScoredChunk


def build_citations(chunks: list[ScoredChunk]) -> list[Citation]:
    return [
        Citation(
            filename=chunk.filename,
            chunk_id=chunk.id,
            artifact_id=chunk.artifact_id,
            source_url=chunk.source_url,
            start_line=chunk.start_line,
            start_page=chunk.start_page,
        )
        for chunk in chunks
    ]
