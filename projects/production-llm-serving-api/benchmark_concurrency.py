import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

URL = "http://127.0.0.1:8000/v1/chat/completions"
LEVELS = [1, 2, 4]
REQUESTS_PER_LEVEL = 4

PAYLOAD = {
    "messages": [
        {
            "role": "user",
            "content": "Explain briefly why reproducible benchmarking matters for production LLM systems.",
        }
    ],
    "max_tokens": 80,
}


def send_request():
    start = time.perf_counter()

    with httpx.Client(timeout=120.0) as client:
        response = client.post(URL, json=PAYLOAD)

    elapsed = time.perf_counter() - start
    response.raise_for_status()
    data = response.json()

    completion_tokens = data["usage"]["completion_tokens"]
    server_latency = data["metrics"]["latency_seconds"]

    return {
        "client_latency_seconds": elapsed,
        "server_latency_seconds": server_latency,
        "completion_tokens": completion_tokens,
    }


rows = []

for concurrency in LEVELS:
    print(f"\nRunning concurrency={concurrency}...")

    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request) for _ in range(REQUESTS_PER_LEVEL)]
        results = [future.result() for future in as_completed(futures)]

    wall_time = time.perf_counter() - wall_start

    mean_latency = sum(r["client_latency_seconds"] for r in results) / len(results)
    total_tokens = sum(r["completion_tokens"] for r in results)

    requests_per_second = len(results) / wall_time
    tokens_per_second = total_tokens / wall_time

    row = {
        "concurrency": concurrency,
        "requests": len(results),
        "mean_latency_seconds": round(mean_latency, 4),
        "wall_time_seconds": round(wall_time, 4),
        "requests_per_second": round(requests_per_second, 4),
        "completion_tokens_per_second": round(tokens_per_second, 4),
    }

    rows.append(row)

    print(row)

with open("results/concurrency_benchmark.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nSaved results/concurrency_benchmark.csv")
