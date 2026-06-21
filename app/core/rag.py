import ollama
from app.config import settings


RAG_PROMPT_TEMPLATE = """You are a helpful assistant answering questions based ONLY on the
provided context. The context may contain Bangla, English, or both.
If the answer is not present in the context, say you don't know — do not make up information.

Context:
{context}

Question: {question}

Answer (respond in the same language as the question where possible):"""


def build_context(chunks: list[str]) -> str:
    return "\n\n---\n\n".join(chunks)


def generate_answer(question: str, retrieved_chunks: list[str]) -> str:
    
    context = build_context(retrieved_chunks)
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    print(f"[RAG] Sending prompt to local Ollama model: {settings.OLLAMA_MODEL}")

    client = ollama.Client(host=settings.OLLAMA_HOST)
    response = client.chat(
        model=settings.OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response["message"]["content"]
    print("[RAG] Answer generated.")
    return answer
