import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Paths ---
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UPLOAD_DIR: str = os.path.join(BASE_DIR, "storage", "uploads")
    CHROMA_DIR: str = os.path.join(BASE_DIR, "storage", "chroma_db")

    # --- OCR ---
    OCR_LANGUAGES: str = "ben+eng"   # Tesseract language codes (Bangla + English)
    TESSERACT_CMD: str | None = None  # set this if tesseract isn't on PATH (e.g. Windows)

    # --- Chunking ---
    CHUNK_SIZE: int = 500       # words per chunk
    CHUNK_OVERLAP: int = 100    # overlapping words between chunks

    # --- Embedding ---
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-base"

    # --- ChromaDB ---
    CHROMA_COLLECTION_NAME: str = "documents"

    # --- LLM (Ollama) ---
    OLLAMA_MODEL: str = "llama3"
    OLLAMA_HOST: str = "http://localhost:11434"

    class Config:
        env_file = ".env"


settings = Settings()

# Make sure required folders exist at startup
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_DIR, exist_ok=True)
