import csv
from pathlib import Path

from retriever import load_chunks, build_embeddings, retrieve

PROJECT_DIR = Path(__file__).resolve().parents[1]
QUESTIONS_FILE = PROJECT_DIR / "data" / "evaluation" / "retrieval_questions.csv"

def load_questions() -> list[dict]:
    with QUESTIONS_FILE.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def evaluate() -> None:
    questions = load_questions()
    chunks = load_chunks()
    model, embeddings = build_embeddings(chunks)

    recall_at_1 = 0
    recall_at_3 = 0
    reciprocal_rank_sum = 0.0

    for item in questions:
        results = retrieve(
            item["question"],
            model,
            embeddings,
            chunks,
            top_k=3,
        )

        first_relevant_rank = None

        for result in results:
            if result["document_id"] == item["expected_document_id"]:
                first_relevant_rank = result["rank"]
                break

        if first_relevant_rank == 1:
            recall_at_1 += 1

        if first_relevant_rank is not None:
            recall_at_3 += 1
            reciprocal_rank_sum += 1.0 / first_relevant_rank

        print(
            f"{item['question_id']} | "
            f"expected={item['expected_document_id']} | "
            f"first_relevant_rank={first_relevant_rank}"
        )

    total = len(questions)

    print()
    print(f"Questions: {total}")
    print(f"Recall@1: {recall_at_1 / total:.4f}")
    print(f"Recall@3: {recall_at_3 / total:.4f}")
    print(f"MRR: {reciprocal_rank_sum / total:.4f}")

if __name__ == "__main__":
    evaluate()
