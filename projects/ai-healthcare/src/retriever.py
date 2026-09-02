import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parents[1]
CHUNKS_FILE = PROJECT_DIR / "data" / "processed" / "healthcare_chunks.jsonl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

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

def retrieve(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: list[dict],
    top_k: int = 3,
) -> list[dict]:
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    scores = embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for rank, index in enumerate(top_indices, start=1):
        item = dict(chunks[index])
        item["rank"] = rank
        item["score"] = float(scores[index])
        results.append(item)

    return results

def main():
    chunks = load_chunks()
    model, embeddings = build_embeddings(chunks)

    query = "What are the main risk factors for cardiovascular disease?"
    results = retrieve(query, model, embeddings, chunks, top_k=3)

    print(f"\nQuery: {query}\n")

    for result in results:
        print(
            f"Rank {result['rank']} | "
            f"score={result['score']:.4f} | "
            f"{result['chunk_id']} | "
            f"{result['title']}"
        )
        print(result["text"][:500])
        print()


if __name__ == "__main__":
    main()
