import chromadb
from app.config import settings

_client = None
_collection = None


def get_collection():
 
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
        _collection = _client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[VECTOR_STORE] Connected to ChromaDB collection "
              f"'{settings.CHROMA_COLLECTION_NAME}' at {settings.CHROMA_DIR}")
    return _collection


def add_chunks(chunk_ids: list[str], chunk_texts: list[str],
               embeddings: list[list[float]], metadatas: list[dict]):
    
    collection = get_collection()
    collection.add(
        ids=chunk_ids,
        documents=chunk_texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"[VECTOR_STORE] Added {len(chunk_ids)} chunks to the vector store.")


def build_where_clause(filters) -> dict | None:
    
    if filters is None:
        return None

    conditions = []

    if filters.language:
        conditions.append({"language": {"$eq": filters.language}})
    if filters.doc_type:
        conditions.append({"doc_type": {"$eq": filters.doc_type}})
    if filters.filename:
        conditions.append({"filename": {"$eq": filters.filename}})
    if filters.date_from:
        conditions.append({"upload_date": {"$gte": filters.date_from.isoformat()}})
    if filters.date_to:
        conditions.append({"upload_date": {"$lte": filters.date_to.isoformat()}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def query_chunks(query_embedding: list[float], top_k: int = 5, filters=None):
    
    collection = get_collection()
    where_clause = build_where_clause(filters)

    print(f"[VECTOR_STORE] Querying top_k={top_k}, where={where_clause}")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_clause,
    )
    return results
