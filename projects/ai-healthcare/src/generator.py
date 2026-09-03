from retriever import load_chunks, build_embeddings, retrieve
from retriever_rerank import build_reranker, retrieve as retrieve_reranked
import requests

API_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

def generate_llm_only(question: str, max_tokens: int = 300) -> dict:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": question}
        ],
        "max_tokens": max_tokens,
    }

    response = requests.post(API_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()

def generate_standard_rag(
    question: str,
    model,
    embeddings,
    chunks: list[dict],
    top_k: int = 3,
    max_tokens: int = 300,
) -> dict:
    retrieved = retrieve(question, model, embeddings, chunks, top_k=top_k)

    context = "\n\n".join(
        f"[Source {item['rank']}] {item['title']}\n{item['text']}"
        for item in retrieved
    )

    prompt = (
        "Answer the healthcare question using only the evidence provided below. "
        "If the evidence is insufficient, say so. "
        "Support factual claims with source markers such as [Source 1].\n\n"
        f"Evidence:\n{context}\n\n"
        f"Question: {question}"
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }

    response = requests.post(API_URL, json=payload, timeout=120)
    response.raise_for_status()

    return {
        "response": response.json(),
        "retrieved_chunks": retrieved,
    }


def generate_improved_rag(
    question: str,
    model,
    embeddings,
    chunks: list[dict],
    reranker,
    top_k: int = 3,
    candidate_k: int = 10,
    max_tokens: int = 300,
) -> dict:
    retrieved = retrieve_reranked(
        question,
        model,
        embeddings,
        chunks,
        reranker,
        top_k=top_k,
        candidate_k=candidate_k,
    )

    context = "\n\n".join(
        f"[Source {item['rank']}] {item['title']}\n{item['text']}"
        for item in retrieved
    )

    prompt = (
        "Answer the healthcare question using only the evidence provided below. "
        "If the evidence is insufficient, say so. "
        "Support factual claims with source markers such as [Source 1].\n\n"
        f"Evidence:\n{context}\n\n"
        f"Question: {question}"
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }

    response = requests.post(API_URL, json=payload, timeout=120)
    response.raise_for_status()

    return {
        "response": response.json(),
        "retrieved_chunks": retrieved,
    }


def main():
    question = "What are the main risk factors for cardiovascular disease?"

    chunks = load_chunks()
    model, embeddings = build_embeddings(chunks)

    result = generate_standard_rag(
        question,
        model,
        embeddings,
        chunks,
        top_k=3,
        max_tokens=200,
    )

    print("\nRetrieved evidence:")
    for item in result["retrieved_chunks"]:
        print(
            f"Rank {item['rank']} | "
            f"score={item['score']:.4f} | "
            f"{item['chunk_id']}"
        )

    response = result["response"]

    print("\nGrounded answer:\n")
    print(response["choices"][0]["message"]["content"])

    print("\nUsage:")
    print(response.get("usage", {}))

    print("\nMetrics:")
    print(response.get("metrics", {}))


if __name__ == "__main__":
    main()
