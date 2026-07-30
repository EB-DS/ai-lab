import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a baseline LLM inference benchmark."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Hugging Face model name or local model path.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Explain how organizations can use open-weight language models "
            "to solve real-world business problems."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--output",
        default="results/baseline_benchmark.json",
        help="Path for the benchmark result.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this benchmark.")

    device_name = torch.cuda.get_device_name(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(f"GPU: {device_name}")
    print(f"Loading model: {args.model}")

    load_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    load_time = time.perf_counter() - load_start

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI engineering assistant. Provide practical, "
                "accurate, and concise responses."
            ),
        },
        {"role": "user", "content": args.prompt},
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    ).to(model.device)

    input_tokens = inputs["input_ids"].shape[-1]

    # Warm-up generation reduces one-time initialization effects.
    with torch.inference_mode():
        model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
        )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    generation_start = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    torch.cuda.synchronize()
    generation_time = time.perf_counter() - generation_start

    generated_ids = outputs[0][input_tokens:]
    generated_tokens = generated_ids.shape[-1]
    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    tokens_per_second = (
        generated_tokens / generation_time if generation_time > 0 else 0
    )

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

    result = {
        "model": args.model,
        "gpu": device_name,
        "precision": "bfloat16",
        "prompt": args.prompt,
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "max_new_tokens": args.max_new_tokens,
        "model_load_seconds": round(load_time, 4),
        "generation_seconds": round(generation_time, 4),
        "tokens_per_second": round(tokens_per_second, 4),
        "peak_vram_gb": round(peak_vram_gb, 4),
        "response": response,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print("\nBenchmark summary")
    print("-" * 60)
    print(f"Model load time:     {result['model_load_seconds']} seconds")
    print(f"Generation time:     {result['generation_seconds']} seconds")
    print(f"Input tokens:        {result['input_tokens']}")
    print(f"Generated tokens:    {result['generated_tokens']}")
    print(f"Tokens per second:   {result['tokens_per_second']}")
    print(f"Peak VRAM:           {result['peak_vram_gb']} GB")
    print(f"Result saved to:     {output_path}")
    print("\nModel response")
    print("-" * 60)
    print(response)


if __name__ == "__main__":
    main()
