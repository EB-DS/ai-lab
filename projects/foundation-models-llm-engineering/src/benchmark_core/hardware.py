import platform
from typing import Any

import torch
import transformers


def ensure_cuda_available() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this benchmark.")


def get_environment_metadata() -> dict[str, Any]:
    ensure_cuda_available()

    properties = torch.cuda.get_device_properties(0)

    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_gb": round(
            properties.total_memory / (1024**3),
            4,
        ),
        "gpu_compute_capability": [
            properties.major,
            properties.minor,
        ],
        "gpu_multiprocessor_count": properties.multi_processor_count,
    }


def reset_gpu_metrics() -> None:
    ensure_cuda_available()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def synchronize_gpu() -> None:
    ensure_cuda_available()
    torch.cuda.synchronize()


def get_peak_memory_metrics() -> dict[str, float]:
    ensure_cuda_available()

    return {
        "peak_vram_allocated_gb": (
            torch.cuda.max_memory_allocated() / (1024**3)
        ),
        "peak_vram_reserved_gb": (
            torch.cuda.max_memory_reserved() / (1024**3)
        ),
    }
