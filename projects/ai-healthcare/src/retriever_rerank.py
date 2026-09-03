import json
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parents[1]
CHUNKS_FILE = PROJECT_DIR / "data" / "processed" / "healthcare_chunks.jsonl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def load_chunks() -> list[dict]:
    chunks = []

    with CHUNKS_FILE.open(encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    return chunks


def build_embeddings(chunks: list[dict]):
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    return model, embeddings

def build_reranker():
    return CrossEncoder(RERANKER_MODEL)

def retrieve(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: list[dict],
    reranker: CrossEncoder,
    top_k: int = 3,
    candidate_k: int = 10,
) -> list[dict]:
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    dense_scores = embeddings @ query_embedding
    candidate_indices = np.argsort(dense_scores)[::-1][:candidate_k]

    pairs = [(query, chunks[index]["text"]) for index in candidate_indices]
    rerank_scores = reranker.predict(pairs)

    order = np.argsort(rerank_scores)[::-1][:top_k]
    results = []

    for rank, position in enumerate(order, start=1):
        index = candidate_indices[position]
        item = dict(chunks[index])
        item["rank"] = rank
        item["dense_score"] = float(dense_scores[index])
        item["rerank_score"] = float(rerank_scores[position])
        results.append(item)

    return results

def main():
    chunks = load_chunks()
    model, embeddings = build_embeddings(chunks)
    reranker = build_reranker()

    query = "What are the main risk factors for cardiovascular disease?"
    results = retrieve(query, model, embeddings, chunks, reranker, top_k=3, candidate_k=10)

    print(f"\nQuery: {query}\n")

    for result in results:
        print(
            f"Rank {result['rank']} | "
            f"dense={result['dense_score']:.4f} | rerank={result['rerank_score']:.4f} | "
            f"{result['chunk_id']} | "
            f"{result['title']}"
        )
        print(result["text"][:500])
        print()


if __name__ == "__main__":
    main()
