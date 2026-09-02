import csv
import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCES_FILE = PROJECT_DIR / "data" / "sources.csv"
RAW_DIR = PROJECT_DIR / "data" / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_page_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body
    if main is None:
        return ""

    return clean_text(main.get_text(" ", strip=True))


def download_document(source: dict) -> dict:
    response = requests.get(
        source["url"],
        timeout=30,
        headers={"User-Agent": "ai-lab-healthcare-research/1.0"},
    )
    response.raise_for_status()

    text = extract_page_text(response.text)

    return {
        "document_id": source["document_id"],
        "title": source["title"],
        "organization": source["organization"],
        "topic": source["topic"],
        "url": source["url"],
        "retrieved_date": source["retrieved_date"],
        "notes": source["notes"],
        "text": text,
        "character_count": len(text),
    }


def main():
    with SOURCES_FILE.open(newline="", encoding="utf-8") as f:
        sources = list(csv.DictReader(f))

    for source in sources:
        print(f"Downloading: {source['title']}")

        document = download_document(source)

        output_path = RAW_DIR / f"{source['document_id']}.json"
        output_path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(
            f"Saved {output_path.name} "
            f"({document['character_count']} characters)"
        )


if __name__ == "__main__":
    main()
