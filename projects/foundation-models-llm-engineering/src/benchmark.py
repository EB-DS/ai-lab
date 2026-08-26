import argparse
from datetime import datetime, timezone
from pathlib import Path

from benchmark_core.config import (
    apply_overrides,
    load_config,
    validate_config,
)
from benchmark_core.hardware import get_environment_metadata
from benchmark_core.metrics import (
    summarize_experiment,
    summarize_prompt_runs,
)
from benchmark_core.model import build_inputs, load_model
from benchmark_core.reporting import (
    build_run_row,
    write_csv,
    write_json,
)
from benchmark_core.runner import (
    run_generation,
    run_warmup,
)


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


def main() -> None:
    args = parse_args()

    config = load_config(Path(args.config))

    config = apply_overrides(
        config,
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        runs=args.runs,
    )

    validate_config(config)

    experiment = config["experiment"]
    model_config = config["model"]
    generation_config = config["generation"]
    output_config = config["output"]

    warmup_runs = int(
        experiment.get("warmup_runs", 1)
    )

    measured_runs = int(
        experiment.get("measured_runs", 3)
    )

    print(f"Experiment: {experiment['name']}")

    environment = get_environment_metadata()

    print(f"GPU: {environment['gpu_name']}")
    print(f"Loading model: {model_config['name']}")

    loaded = load_model(
        model_name=model_config["name"],
        dtype_name=model_config.get(
            "dtype",
            "bfloat16",
        ),
        device_map=model_config.get(
            "device_map",
            "auto",
        ),
        quantization=model_config.get(
            "quantization",
        ),
    )

    all_run_rows = []
    prompt_summaries = []

    for prompt_number, prompt in enumerate(
        config["prompts"],
        start=1,
    ):
        prompt_id = prompt["id"]
        prompt_text = prompt["text"]

        print(
            f"\nPrompt {prompt_number}/"
            f"{len(config['prompts'])}: "
            f"{prompt_id}"
        )

        inputs = build_inputs(
            tokenizer=loaded.tokenizer,
            system_prompt=config.get(
                "system_prompt",
                "",
            ),
            user_prompt=prompt_text,
            device=loaded.device,
        )

        if warmup_runs > 0:
            print(
                f"  Running {warmup_runs} warm-up run(s)"
            )

            run_warmup(
                model=loaded.model,
                tokenizer=loaded.tokenizer,
                inputs=inputs,
                generation_config=generation_config,
                warmup_runs=warmup_runs,
            )

        prompt_runs = []

        for run_number in range(
            1,
            measured_runs + 1,
        ):
            result = run_generation(
                model=loaded.model,
                tokenizer=loaded.tokenizer,
                inputs=inputs,
                generation_config=generation_config,
            )

            prompt_runs.append(result)

            row = build_run_row(
                experiment=experiment["name"],
                model=model_config["name"],
                prompt_id=prompt_id,
                run_number=run_number,
                result=result,
            )

            all_run_rows.append(row)

            print(
                f"  Run {run_number}/"
                f"{measured_runs}: "
                f"{result['generation_seconds']:.4f}s, "
                f"{result['tokens_per_second']:.2f} tokens/s"
            )

        prompt_summary = summarize_prompt_runs(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            runs=prompt_runs,
        )

        prompt_summaries.append(
            prompt_summary
        )

    experiment_summary = summarize_experiment(
        prompt_summaries
    )

    payload = {
        "experiment": {
            "name": experiment["name"],
            "seed": experiment.get("seed", 42),
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "completed_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        },
        "environment": environment,
        "model": {
            "name": loaded.model_name,
            "dtype": loaded.dtype_name,
            "load_seconds": loaded.load_seconds,
        },
        "generation": generation_config,
        "summary": experiment_summary,
        "prompts": prompt_summaries,
    }

    output_directory = Path(
        output_config["directory"]
    )

    json_path = (
        output_directory
        / output_config["json_filename"]
    )

    csv_path = (
        output_directory
        / output_config["csv_filename"]
    )

    write_json(
        json_path,
        payload,
    )

    write_csv(
        csv_path,
        all_run_rows,
    )

    print("\nBenchmark summary")
    print("-" * 60)

    print(
        f"Model load time:       "
        f"{loaded.load_seconds:.4f} seconds"
    )

    print(
        f"Prompts evaluated:     "
        f"{len(prompt_summaries)}"
    )

    print(
        f"Measured runs/prompt:  "
        f"{measured_runs}"
    )

    print(
        f"Mean generation time:  "
        f"{experiment_summary['mean_generation_seconds']:.4f} "
        f"seconds"
    )

    print(
        f"Mean throughput:       "
        f"{experiment_summary['mean_tokens_per_second']:.4f} "
        f"tokens/second"
    )

    print(f"JSON result:           {json_path}")
    print(f"CSV result:            {csv_path}")


if __name__ == "__main__":
    main()
