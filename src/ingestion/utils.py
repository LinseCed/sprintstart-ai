from pathlib import Path


def build_metadata(path: Path) -> dict[str, str]: 
    """  Build metadata for a parsed file chunk.

    The metadata contains basic file information that can later
    be used for retrieval, debugging, filtering, or tracing
    chunk origins inside the ingestion pipeline.

    Args:
        path (Path): Path object representing the source file.

    Returns:
        dict[str, str]: A dictionary containing file metadata such as:
                        - source: absolute file path
                        - filename: file name including extension
                        - type: file extension
    """
    
    return {
        "source": str(path.resolve()),
        "filename": path.name,
        "type": path.suffix
    }