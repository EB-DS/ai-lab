import uuid

from fastapi import FastAPI

from .backend import create_backend
from .config import load_settings
from .schemas import ChatCompletionRequest


settings = load_settings()
backend = create_backend(settings.backend)

app = FastAPI(
    title="Production LLM Serving API",
    version="0.4.0",
)


@app.get("/")
def root():
    return {
        "message": "Production LLM Serving API is running.",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "production-llm-serving-api",
        "version": "0.4.0",
        "backend": backend.name,
        "model": settings.model_name,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": settings.model_name,
                "object": "model",
                "owned_by": "ai-lab",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completion(request: ChatCompletionRequest):
    messages = [
        {
            "role": message.role,
            "content": message.content,
        }
        for message in request.messages
    ]

    result = backend.generate(
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["text"],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
        "metrics": {
            "latency_seconds": round(
                result["latency_seconds"],
                6,
            ),
            "backend": backend.name,
        },
    }
