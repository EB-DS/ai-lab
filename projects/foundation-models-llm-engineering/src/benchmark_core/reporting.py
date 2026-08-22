import csv
import json
from pathlib import Path
from typing import Any


RUN_FIELDNAMES = [
    "experiment",
    "model",
    "prompt_id",
    "run",
    "input_tokens",
    "generated_tokens",
    "generation_seconds",
    "tokens_per_second",
    "peak_vram_allocated_gb",
    "peak_vram_reserved_gb",
]


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=RUN_FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(rows)


def build_run_row(
    *,
    experiment: str,
    model: str,
    prompt_id: str,
    run_number: int,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "model": model,
        "prompt_id": prompt_id,
        "run": run_number,
        "input_tokens": result["input_tokens"],
        "generated_tokens": result["generated_tokens"],
        "generation_seconds": round(
            result["generation_seconds"], 6
        ),
        "tokens_per_second": round(
            result["tokens_per_second"], 6
        ),
        "peak_vram_allocated_gb": round(
            result["peak_vram_allocated_gb"], 6
        ),
        "peak_vram_reserved_gb": round(
            result["peak_vram_reserved_gb"], 6
        ),
    }
