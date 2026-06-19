from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import upload, search

app = FastAPI(
    title="Local OCR & Dynamic RAG System",
    description="Fully local document OCR + multilingual (Bangla/English) RAG pipeline.",
    version="1.0.0",
)

# Allow local frontend / Swagger UI / testing tools to call the API freely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Local OCR & RAG system is running."}



app.include_router(upload.router, tags=["Upload"])
app.include_router(search.router, tags=["Search"])