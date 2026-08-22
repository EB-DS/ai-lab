import time
from typing import Any

import torch

from .hardware import (
    get_peak_memory_metrics,
    reset_gpu_metrics,
    synchronize_gpu,
)


def run_generation(
    model: Any,
    tokenizer: Any,
    inputs: dict[str, torch.Tensor],
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    input_tokens = inputs["input_ids"].shape[-1]

    reset_gpu_metrics()

    started = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=generation_config["max_new_tokens"],
            do_sample=generation_config.get("do_sample", False),
        )

    synchronize_gpu()

    elapsed = time.perf_counter() - started

    generated_ids = outputs[0, input_tokens:]
    generated_tokens = generated_ids.shape[-1]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    tokens_per_second = (
        generated_tokens / elapsed if elapsed > 0 else 0.0
    )

    memory = get_peak_memory_metrics()

    return {
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "generation_seconds": elapsed,
        "tokens_per_second": tokens_per_second,
        **memory,
        "response": response,
    }


def run_warmup(
    model: Any,
    tokenizer: Any,
    inputs: dict[str, torch.Tensor],
    generation_config: dict[str, Any],
    warmup_runs: int,
) -> None:
    warmup_config = {
        **generation_config,
        "max_new_tokens": min(
            8,
            generation_config["max_new_tokens"],
        ),
    }

    for _ in range(warmup_runs):
        run_generation(
            model=model,
            tokenizer=tokenizer,
            inputs=inputs,
            generation_config=warmup_config,
        )
