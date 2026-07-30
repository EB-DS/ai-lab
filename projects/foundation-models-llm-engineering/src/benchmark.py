import argparse
import csv
import json
import platform
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a configuration-driven LLM inference benchmark."
    )
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to the YAML benchmark configuration.",
    )
    parser.add_argument(
        "--model",
        help="Override the model name from the configuration.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Override max_new_tokens from the configuration.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        help="Override the number of measured runs.",
    )
    return parser.parse_args()


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
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.model:
        config["model"]["name"] = args.model

    if args.max_new_tokens is not None:
        if args.max_new_tokens < 1:
            raise ValueError("--max-new-tokens must be at least 1.")
        config["generation"]["max_new_tokens"] = args.max_new_tokens

    if args.runs is not None:
        if args.runs < 1:
            raise ValueError("--runs must be at least 1.")
        config["experiment"]["measured_runs"] = args.runs

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    return ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * fraction


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


def run_generation(
    model: Any,
    tokenizer: Any,
    inputs: dict[str, torch.Tensor],
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    input_tokens = inputs["input_ids"].shape[-1]

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=generation_config["max_new_tokens"],
            do_sample=generation_config.get("do_sample", False),
        )

    torch.cuda.synchronize()
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

    return {
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "generation_seconds": elapsed,
        "tokens_per_second": tokens_per_second,
        "peak_vram_allocated_gb": (
            torch.cuda.max_memory_allocated() / (1024**3)
        ),
        "peak_vram_reserved_gb": (
            torch.cuda.max_memory_reserved() / (1024**3)
        ),
        "response": response,
    }


def summarize_prompt_runs(
    prompt_id: str,
    prompt_text: str,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    latencies = [run["generation_seconds"] for run in runs]
    throughputs = [run["tokens_per_second"] for run in runs]

    return {
        "prompt_id": prompt_id,
        "prompt": prompt_text,
        "run_count": len(runs),
        "input_tokens": runs[0]["input_tokens"],
        "generated_tokens": [
            run["generated_tokens"] for run in runs
        ],
        "latency_seconds": {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "min": min(latencies),
            "max": max(latencies),
            "p95": percentile(latencies, 0.95),
        },
        "tokens_per_second": {
            "mean": statistics.mean(throughputs),
            "median": statistics.median(throughputs),
            "min": min(throughputs),
            "max": max(throughputs),
        },
        "peak_vram_allocated_gb": max(
            run["peak_vram_allocated_gb"] for run in runs
        ),
        "peak_vram_reserved_gb": max(
            run["peak_vram_reserved_gb"] for run in runs
        ),
        "response": runs[-1]["response"],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
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

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    config = apply_overrides(config, args)
    validate_config(config)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this benchmark.")

    experiment = config["experiment"]
    model_config = config["model"]
    generation_config = config["generation"]
    output_config = config["output"]

    seed = int(experiment.get("seed", 42))
    warmup_runs = int(experiment.get("warmup_runs", 1))
    measured_runs = int(experiment.get("measured_runs", 3))

    set_seed(seed)

    device_name = torch.cuda.get_device_name(0)
    dtype_name = model_config.get("dtype", "bfloat16")
    dtype = DTYPE_MAP[dtype_name]

    print(f"Experiment: {experiment['name']}")
    print(f"GPU: {device_name}")
    print(f"Loading model: {model_config['name']}")

    load_started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(model_config["name"])

    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        dtype=dtype,
        device_map=model_config.get("device_map", "auto"),
    )
    model.eval()

    model_load_seconds = time.perf_counter() - load_started
    model_device = next(model.parameters()).device

    all_run_rows: list[dict[str, Any]] = []
    prompt_summaries: list[dict[str, Any]] = []

    for prompt_number, prompt in enumerate(config["prompts"], start=1):
        prompt_id = prompt["id"]
        prompt_text = prompt["text"]

        print(
            f"\nPrompt {prompt_number}/{len(config['prompts'])}: "
            f"{prompt_id}"
        )

        inputs = build_inputs(
            tokenizer=tokenizer,
            system_prompt=config.get("system_prompt", ""),
            user_prompt=prompt_text,
            device=model_device,
        )

        for warmup_index in range(warmup_runs):
            print(f"  Warm-up {warmup_index + 1}/{warmup_runs}")
            run_generation(
                model=model,
                tokenizer=tokenizer,
                inputs=inputs,
                generation_config={
                    **generation_config,
                    "max_new_tokens": min(
                        8,
                        generation_config["max_new_tokens"],
                    ),
                },
            )

        prompt_runs: list[dict[str, Any]] = []

        for run_index in range(1, measured_runs + 1):
            result = run_generation(
                model=model,
                tokenizer=tokenizer,
                inputs=inputs,
                generation_config=generation_config,
            )
            prompt_runs.append(result)

            all_run_rows.append(
                {
                    "experiment": experiment["name"],
                    "model": model_config["name"],
                    "prompt_id": prompt_id,
                    "run": run_index,
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
            )

            print(
                f"  Run {run_index}/{measured_runs}: "
                f"{result['generation_seconds']:.4f}s, "
                f"{result['tokens_per_second']:.2f} tokens/s"
            )

        prompt_summaries.append(
            summarize_prompt_runs(
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                runs=prompt_runs,
            )
        )

    payload = {
        "experiment": {
            "name": experiment["name"],
            "seed": seed,
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "completed_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        },
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": device_name,
        },
        "model": {
            "name": model_config["name"],
            "dtype": dtype_name,
            "device_map": model_config.get("device_map", "auto"),
            "load_seconds": model_load_seconds,
        },
        "generation": generation_config,
        "prompts": prompt_summaries,
    }

    output_directory = Path(output_config["directory"])
    json_path = output_directory / output_config["json_filename"]
    csv_path = output_directory / output_config["csv_filename"]

    write_json(json_path, payload)
    write_csv(csv_path, all_run_rows)

    mean_latencies = [
        summary["latency_seconds"]["mean"]
        for summary in prompt_summaries
    ]
    mean_throughputs = [
        summary["tokens_per_second"]["mean"]
        for summary in prompt_summaries
    ]

    print("\nBenchmark summary")
    print("-" * 60)
    print(f"Model load time:       {model_load_seconds:.4f} seconds")
    print(f"Prompts evaluated:     {len(prompt_summaries)}")
    print(f"Measured runs/prompt:  {measured_runs}")
    print(
        f"Mean generation time:  "
        f"{statistics.mean(mean_latencies):.4f} seconds"
    )
    print(
        f"Mean throughput:       "
        f"{statistics.mean(mean_throughputs):.4f} tokens/second"
    )
    print(f"JSON result:           {json_path}")
    print(f"CSV result:            {csv_path}")


if __name__ == "__main__":
    main()
