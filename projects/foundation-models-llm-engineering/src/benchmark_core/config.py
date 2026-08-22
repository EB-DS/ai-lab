from pathlib import Path
from typing import Any

import torch
import yaml


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The YAML configuration must contain a mapping.")

    return config


def apply_overrides(
    config: dict[str, Any],
    *,
    model: str | None = None,
    max_new_tokens: int | None = None,
    runs: int | None = None,
) -> dict[str, Any]:
    if model:
        config["model"]["name"] = model

    if max_new_tokens is not None:
        if max_new_tokens < 1:
            raise ValueError("--max-new-tokens must be at least 1.")
        config["generation"]["max_new_tokens"] = max_new_tokens

    if runs is not None:
        if runs < 1:
            raise ValueError("--runs must be at least 1.")
        config["experiment"]["measured_runs"] = runs

    return config


def validate_config(config: dict[str, Any]) -> None:
    required_sections = {
        "experiment",
        "model",
        "generation",
        "prompts",
        "output",
    }

    missing = sorted(required_sections - config.keys())

    if missing:
        raise ValueError(
            f"Missing required configuration sections: {missing}"
        )

    if not isinstance(config["prompts"], list) or not config["prompts"]:
        raise ValueError("At least one prompt must be configured.")

    dtype_name = config["model"].get("dtype", "bfloat16")

    if dtype_name not in DTYPE_MAP:
        supported = ", ".join(DTYPE_MAP)
        raise ValueError(
            f"Unsupported dtype '{dtype_name}'. Supported: {supported}"
        )


def resolve_dtype(dtype_name: str) -> torch.dtype:
    try:
        return DTYPE_MAP[dtype_name]
    except KeyError as exc:
        supported = ", ".join(DTYPE_MAP)
        raise ValueError(
            f"Unsupported dtype '{dtype_name}'. Supported: {supported}"
        ) from exc
