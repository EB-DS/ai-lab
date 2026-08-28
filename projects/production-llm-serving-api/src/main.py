import json
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from .backend import create_backend
from .config import load_settings
from .schemas import ChatCompletionRequest


settings = load_settings()
backend = create_backend(
    settings.backend,
    settings.model_name,
)

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

    if request.stream:
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        def event_stream():
            role_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(role_chunk)}\n\n"

            for text_chunk in backend.stream_generate(
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            ):
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": text_chunk},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

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
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": (
                result["prompt_tokens"]
                + result["completion_tokens"]
                if result["prompt_tokens"] is not None
                and result["completion_tokens"] is not None
                else None
            ),
        },
        "metrics": {
            "latency_seconds": round(
                result["latency_seconds"],
                6,
            ),
            "backend": backend.name,
        },
    }
