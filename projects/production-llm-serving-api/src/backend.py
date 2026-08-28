import time
from abc import ABC, abstractmethod
from threading import Thread
from collections.abc import Iterator


class LLMBackend(ABC):
    name: str

    @property
    def is_ready(self) -> bool:
        return True

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> dict:
        raise NotImplementedError

    def stream_generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> Iterator[str]:
        result = self.generate(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        yield result["text"]


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
            "prompt_tokens": None,
            "completion_tokens": None,
        }


class TransformersBackend(LLMBackend):
    name = "transformers"

    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
        device_map: str = "auto",
    ):
        self.model_name = model_name
        self.dtype = dtype
        self.device_map = device_map
        self._is_ready = False

        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TextIteratorStreamer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Transformers backend requires torch and "
                "transformers to be installed."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Transformers backend requires a CUDA-capable GPU."
            )

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }

        if dtype not in dtype_map:
            raise ValueError(
                f"Unsupported dtype: {dtype}"
            )

        self.torch = torch
        self.TextIteratorStreamer = TextIteratorStreamer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype_map[dtype],
            device_map=device_map,
        )

        self.model.eval()

        self.device = next(
            self.model.parameters()
        ).device

        self._is_ready = True

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> dict:
        started = time.perf_counter()

        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        input_tokens = inputs["input_ids"].shape[-1]

        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
        }

        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        generated = output[0][input_tokens:]

        response_text = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )

        elapsed = time.perf_counter() - started

        return {
            "text": response_text,
            "latency_seconds": elapsed,
            "prompt_tokens": input_tokens,
            "completion_tokens": len(generated),
        }


    def stream_generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> Iterator[str]:
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        streamer = self.TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
        }

        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        thread = Thread(
            target=self.model.generate,
            kwargs=generation_kwargs,
            daemon=True,
        )
        thread.start()

        for text_chunk in streamer:
            if text_chunk:
                yield text_chunk

        thread.join()


def create_backend(
    backend_name: str,
    model_name: str = "local-llm",
) -> LLMBackend:
    if backend_name == "mock":
        return MockLLMBackend()

    if backend_name == "transformers":
        return TransformersBackend(
            model_name=model_name,
        )

    raise ValueError(
        f"Unsupported backend: {backend_name}"
    )
