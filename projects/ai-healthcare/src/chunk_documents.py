import json
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = normalize_text(text)

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = []
    current_len = 0
    chunk_index = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        added_len = len(sentence) + (1 if current else 0)

        if current and current_len + added_len > chunk_size:
            chunk_text_value = " ".join(current)
            start_char = text.find(chunk_text_value)
            end_char = start_char + len(chunk_text_value)

            chunks.append({
                "chunk_index": chunk_index,
                "start_char": start_char,
                "end_char": end_char,
                "text": chunk_text_value,
            })

            overlap_sentences = []
            overlap_len = 0
            for prev_sentence in reversed(current):
                overlap_sentences.insert(0, prev_sentence)
                overlap_len += len(prev_sentence) + 1
                if overlap_len >= overlap:
                    break

            current = overlap_sentences
            current_len = len(" ".join(current))
            chunk_index += 1

        current.append(sentence)
        current_len = len(" ".join(current))

    if current:
        chunk_text_value = " ".join(current)
        start_char = text.find(chunk_text_value)
        end_char = start_char + len(chunk_text_value)

        chunks.append({
            "chunk_index": chunk_index,
            "start_char": start_char,
            "end_char": end_char,
            "text": chunk_text_value,
        })

    return chunks


def chunk_document(document: dict) -> list[dict]:
    chunks = []

    for item in chunk_text(document["text"]):
        chunks.append(
            {
                "chunk_id": f"{document['document_id']}_chunk_{item['chunk_index']:03d}",
                "document_id": document["document_id"],
                "title": document["title"],
                "organization": document["organization"],
                "topic": document["topic"],
                "url": document["url"],
                "retrieved_date": document["retrieved_date"],
                "chunk_index": item["chunk_index"],
                "start_char": item["start_char"],
                "end_char": item["end_char"],
                "chunk_length": len(item["text"]),
                "text": item["text"],
            }
        )

    return chunks

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks = []

    for path in sorted(RAW_DIR.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        document_chunks = chunk_document(document)
        all_chunks.extend(document_chunks)

        print(
            f"{document['document_id']}: "
            f"{len(document_chunks)} chunks"
        )

    output_path = PROCESSED_DIR / "healthcare_chunks.jsonl"

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_chunks)} chunks to {output_path}")


if __name__ == "__main__":
    main()
