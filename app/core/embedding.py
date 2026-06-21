from sentence_transformers import SentenceTransformer
from app.config import settings

_model = None  # lazy-loaded singleton so the model loads only once


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[EMBEDDING] Loading model: {settings.EMBEDDING_MODEL} ...")
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
        print("[EMBEDDING] Model loaded.")
    return _model


def embed_texts(texts: list[str], is_query: bool = False) -> list[list[float]]:
    model = get_embedding_model()
    prefix = "query: " if is_query else "passage: "
    prefixed_texts = [prefix + t for t in texts]

    embeddings = model.encode(
        prefixed_texts,
        normalize_embeddings=True,  # cosine similarity works directly on normalized vectors
        show_progress_bar=False,
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([query], is_query=True)[0]
