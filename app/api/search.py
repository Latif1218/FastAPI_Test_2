from fastapi import APIRouter, HTTPException

from app.core.embedding import embed_query
from app.core.vector_store import query_chunks
from app.core.rag import generate_answer
from app.schemas.schemas import SearchRequest, SearchResponse, RetrievedChunk

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 1. Embed the user's natural language query
    query_vector = embed_query(request.query)

    # 2. Hybrid search: metadata filter (manual) + vector similarity (semantic)
    results = query_chunks(
        query_embedding=query_vector,
        top_k=request.top_k,
        filters=request.filters,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    if not documents:
        return SearchResponse(
            query=request.query,
            answer="No matching documents found for the given query and filters.",
            sources=[],
        )

    # 3. Generate the final answer using the local LLM (Ollama) with retrieved context
    answer = generate_answer(request.query, documents)

    # 4. Build response with sources (so users can trace which chunks were used)
    sources = [
        RetrievedChunk(
            chunk_id=ids[i],
            text=documents[i],
            metadata=metadatas[i],
            score=1 - distances[i],  # convert distance -> similarity score
        )
        for i in range(len(documents))
    ]

    return SearchResponse(query=request.query, answer=answer, sources=sources)
