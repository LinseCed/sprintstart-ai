from ingestion.models import ParsedChunk


def parse_code(filename: str, content: bytes) -> list[ParsedChunk]:
    raise NotImplementedError
