import csv
import json
import time
from pathlib import Path

from generator import generate_llm_only, generate_standard_rag, generate_improved_rag
from retriever import load_chunks, build_embeddings
from retriever_rerank import build_reranker

EVAL_FILE = Path("projects/ai-healthcare/data/evaluation/retrieval_hard_questions.csv")
RESULTS_FILE = Path("projects/ai-healthcare/results/generation_comparison.jsonl")

SELECTED_IDS = {
    "hard_diabetes_01",
    "hard_diabetes_02",
    "hard_obesity_01",
    "hard_obesity_03",
    "hard_diet_02",
    "hard_diet_04",
    "hard_cvd_01",
    "hard_cvd_02",
}

def load_questions() -> list[dict]:
    with EVAL_FILE.open(newline="", encoding="utf-8") as handle:
        questions = list(csv.DictReader(handle))

    return [item for item in questions if item["question_id"] in SELECTED_IDS]

def main():
    questions = load_questions()
    chunks = load_chunks()
    model, embeddings = build_embeddings(chunks)
    reranker = build_reranker()

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text("", encoding="utf-8")

    print(f"Running generation comparison on {len(questions)} questions...")
