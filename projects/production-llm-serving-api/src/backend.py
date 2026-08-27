import time
from abc import ABC, abstractmethod


class LLMBackend(ABC):
    name: str

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> dict:
        raise NotImplementedError


class MockLLMBackend(LLMBackend):
    name = "mock"

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> dict:
        started = time.perf_counter()

        user_messages = [
            message["content"]
            for message in messages
            if message["role"] == "user"
        ]

        latest_user_message = (
            user_messages[-1]
            if user_messages
            else ""
        )

        response_text = (
            "Mock response from the local LLM serving API. "
            f"You asked: {latest_user_message}"
        )

        elapsed = time.perf_counter() - started

        return {
            "text": response_text,
            "latency_seconds": elapsed,
        }


def create_backend(backend_name: str) -> LLMBackend:
    if backend_name == "mock":
        return MockLLMBackend()

    raise ValueError(
        f"Unsupported backend: {backend_name}"
    )
