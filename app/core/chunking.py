from app.config import settings


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """
    Splits text into overlapping word-based chunks.

    Why word-based + overlap (instead of fixed character slicing)?
    - Bangla and English have different average character-per-word density,
      so a word-based approach keeps chunk "meaning size" more consistent
      across both languages.
    - Overlap (default 100 words) ensures a sentence/idea split across the
      chunk boundary still has context in the neighboring chunk, which
      improves retrieval recall for the RAG step.

    Args:
        text: raw extracted text (from OCR)
        chunk_size: words per chunk
        overlap: number of overlapping words between consecutive chunks

    Returns:
        List of text chunks (strings)
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(chunk_size - overlap, 1)  # avoid infinite loop if overlap >= chunk_size

    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break

    print(f"[CHUNKING] Split text into {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={overlap}).")
    return chunks
