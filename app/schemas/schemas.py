from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date


class UploadResponse(BaseModel):
    filename: str
    doc_id: str
    language_detected: str
    total_chunks: int
    message: str


class SearchFilters(BaseModel):
    """
    Manual metadata filters the user can apply alongside the
    natural-language semantic query.
    All fields are optional — only the ones provided are applied.
    """
    language: Optional[str] = Field(
        default=None, description="Filter by language code, e.g. 'bn' or 'en'"
    )
    doc_type: Optional[str] = Field(
        default=None, description="Filter by document type, e.g. 'pdf', 'image'"
    )
    date_from: Optional[date] = Field(
        default=None, description="Only include documents uploaded on/after this date"
    )
    date_to: Optional[date] = Field(
        default=None, description="Only include documents uploaded on/before this date"
    )
    filename: Optional[str] = Field(
        default=None, description="Filter by exact source filename"
    )


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: Optional[SearchFilters] = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: dict
    score: float


class SearchResponse(BaseModel):
    query: str
    answer: str
    sources: List[RetrievedChunk]
