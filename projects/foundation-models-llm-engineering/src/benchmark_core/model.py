import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .config import resolve_dtype


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    device: torch.device
    load_seconds: float
    model_name: str
    dtype_name: str
    quantization_mode: str


def build_quantization_config(
    quantization: dict[str, Any] | None,
) -> tuple[BitsAndBytesConfig | None, str]:
    if not quantization:
        return None, "none"

    mode = quantization.get("mode", "none")

    if mode == "none":
        return None, "none"

    if mode == "int8":
        return (
            BitsAndBytesConfig(
                load_in_8bit=True,
            ),
            "int8",
        )

    if mode == "nf4":
        compute_dtype_name = quantization.get(
            "compute_dtype",
            "bfloat16",
        )
        compute_dtype = resolve_dtype(compute_dtype_name)

        return (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=quantization.get(
                    "use_double_quant",
                    True,
                ),
            ),
            "nf4",
        )

    raise ValueError(
        f"Unsupported quantization mode: {mode}. "
        "Supported modes: none, int8, nf4"
    )


def load_model(
    model_name: str,
    dtype_name: str = "bfloat16",
    device_map: str = "auto",
    quantization: dict[str, Any] | None = None,
) -> LoadedModel:
    dtype = resolve_dtype(dtype_name)

    quantization_config, quantization_mode = (
        build_quantization_config(quantization)
    )

    started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_kwargs: dict[str, Any] = {
        "device_map": device_map,
    }

    if quantization_config is None:
        model_kwargs["dtype"] = dtype
    else:
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
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
        quantization_mode=quantization_mode,
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
