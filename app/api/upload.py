import os
import uuid
from datetime import date

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.config import settings
from app.core.ocr import extract_text, detect_language_simple
from app.core.chunking import chunk_text
from app.core.embedding import embed_texts
from app.core.vector_store import add_chunks
from app.schemas.schemas import UploadResponse

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    doc_id = str(uuid.uuid4())
    saved_filename = f"{doc_id}{ext}"
    saved_path = os.path.join(settings.UPLOAD_DIR, saved_filename)

    # 1. Save the uploaded file locally
    with open(saved_path, "wb") as f:
        content = await file.read()
        f.write(content)
    print(f"[UPLOAD] Saved file to {saved_path}")

    # 2. Local OCR extraction (no external API calls)
    try:
        raw_text = extract_text(saved_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")

    if not raw_text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from this document.")

    language = detect_language_simple(raw_text)
    doc_type = "pdf" if ext == ".pdf" else "image"
    upload_date = date.today().isoformat()

    # 3. Chunk the extracted text
    chunks = chunk_text(raw_text)
    if not chunks:
        raise HTTPException(status_code=422, detail="Text extracted but chunking produced no chunks.")

    # 4. Embed each chunk
    embeddings = embed_texts(chunks, is_query=False)

    # 5. Build metadata for each chunk + store in ChromaDB
    chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "doc_id": doc_id,
            "filename": file.filename,
            "language": language,
            "doc_type": doc_type,
            "upload_date": upload_date,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]

    add_chunks(chunk_ids, chunks, embeddings, metadatas)

    return UploadResponse(
        filename=file.filename,
        doc_id=doc_id,
        language_detected=language,
        total_chunks=len(chunks),
        message="Document processed and stored successfully (fully local pipeline).",
    )
