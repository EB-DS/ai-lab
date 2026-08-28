from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == (
        "Production LLM Serving API is running."
    )


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "production-llm-serving-api"
    assert data["backend"] == "mock"
    assert data["model"] == "local-llm"


def test_models():
    response = client.get("/v1/models")

    assert response.status_code == 200

    data = response.json()

    assert data["object"] == "list"
    assert data["data"][0]["id"] == "local-llm"


def test_chat_completion():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "local-llm",
            "messages": [
                {
                    "role": "user",
                    "content": "Explain quantization.",
                }
            ],
            "max_tokens": 128,
            "temperature": 0,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["object"] == "chat.completion"
    assert data["model"] == "local-llm"
    assert data["choices"][0]["message"]["role"] == "assistant"

    assert (
        "Explain quantization."
        in data["choices"][0]["message"]["content"]
    )

    assert data["metrics"]["backend"] == "mock"

    assert data["usage"]["prompt_tokens"] is None
    assert data["usage"]["completion_tokens"] is None
    assert data["usage"]["total_tokens"] is None


def test_invalid_max_tokens():
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                }
            ],
            "max_tokens": 0,
        },
    )

    assert response.status_code == 422

def test_stream_chat_completion():
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "local-llm",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello streaming",
                }
            ],
            "max_tokens": 32,
            "temperature": 0,
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith(
            "text/event-stream"
        )

        body = "".join(response.iter_text())

    assert '"object": "chat.completion.chunk"' in body
    assert '"role": "assistant"' in body
    assert "Hello streaming" in body
    assert '"finish_reason": "stop"' in body
    assert "data: [DONE]" in body
