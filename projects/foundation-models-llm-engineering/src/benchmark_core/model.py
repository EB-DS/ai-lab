import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import resolve_dtype


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    device: torch.device
    load_seconds: float
    model_name: str
    dtype_name: str


def load_model(
    model_name: str,
    dtype_name: str = "bfloat16",
    device_map: str = "auto",
) -> LoadedModel:
    dtype = resolve_dtype(dtype_name)

    started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
    )

    model.eval()

    load_seconds = time.perf_counter() - started
    device = next(model.parameters()).device

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        device=device,
        load_seconds=load_seconds,
        model_name=model_name,
        dtype_name=dtype_name,
    )


def build_inputs(
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    encoded = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    )

    return {
        key: value.to(device)
        for key, value in encoded.items()
    }
