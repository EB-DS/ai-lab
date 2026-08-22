import statistics
from typing import Any


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


def summarize_prompt_runs(
    prompt_id: str,
    prompt_text: str,
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    if not runs:
        raise ValueError("At least one benchmark run is required.")

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


def summarize_experiment(
    prompt_summaries: list[dict[str, Any]],
) -> dict[str, float]:
    if not prompt_summaries:
        raise ValueError("At least one prompt summary is required.")

    mean_latencies = [
        summary["latency_seconds"]["mean"]
        for summary in prompt_summaries
    ]

    mean_throughputs = [
        summary["tokens_per_second"]["mean"]
        for summary in prompt_summaries
    ]

    return {
        "mean_generation_seconds": statistics.mean(mean_latencies),
        "mean_tokens_per_second": statistics.mean(mean_throughputs),
    }
