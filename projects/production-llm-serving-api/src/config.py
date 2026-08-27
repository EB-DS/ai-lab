import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    backend: str
    model_name: str
    host: str
    port: int


def load_settings() -> Settings:
    return Settings(
        backend=os.getenv(
            "LLM_BACKEND",
            "mock",
        ),
        model_name=os.getenv(
            "LLM_MODEL_NAME",
            "local-llm",
        ),
        host=os.getenv(
            "LLM_HOST",
            "0.0.0.0",
        ),
        port=int(
            os.getenv(
                "LLM_PORT",
                "8000",
            )
        ),
    )
