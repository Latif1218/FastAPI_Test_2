from app.config import settings


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
   
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
