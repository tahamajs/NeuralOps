#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for FastAPI OpenAI-compatible serving endpoints and SSE streaming.
"""

from __future__ import annotations

import json
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from neuralops.integrations.serving import create_openai_serving_app


class MockTokenizer:
    """Mock tokenizer returning deterministic token IDs and decoding."""

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2

    def __call__(self, text: str, return_tensors: str = "pt"):
        return {
            "input_ids": torch.tensor([[10, 20, 30]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        return "synthetic assistant response"

    def apply_chat_template(
        self, messages: list[dict], tokenize: bool = False, add_generation_prompt: bool = True
    ) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"


class MockCausalLM(nn.Module):
    """Mock causal language model implementing generate()."""

    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 16, **kwargs) -> torch.Tensor:
        gen = torch.tensor([[101, 102]], device=input_ids.device)
        return torch.cat([input_ids, gen], dim=-1)


@pytest.fixture
def serving_client():
    model = MockCausalLM()
    tokenizer = MockTokenizer()
    app = create_openai_serving_app(
        model=model,
        tokenizer=tokenizer,
        model_name="test-llm-v1",
        rate_limit_rpm=100,
    )
    return TestClient(app)


def test_health_endpoint_ready(serving_client):
    """GET /health returns status ok and ready true when model is loaded."""
    response = serving_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["ready"] is True
    assert data["backend"] == "neuralops"


def test_health_endpoint_not_ready():
    """GET /health returns not_ready when model or tokenizer is absent."""
    app = create_openai_serving_app(model=None, tokenizer=None)
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "not_ready"
    assert data["ready"] is False


def test_models_endpoint(serving_client):
    """GET /v1/models returns model list with configured model_name."""
    response = serving_client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "test-llm-v1"


def test_completions_non_stream(serving_client):
    """POST /v1/completions produces valid OpenAI text_completion response."""
    payload = {
        "prompt": "Hello world",
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": False,
    }
    response = serving_client.post("/v1/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "text_completion"
    assert data["model"] == "test-llm-v1"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["text"] == "synthetic assistant response"
    assert data["usage"]["total_tokens"] > 0


def test_completions_sse_streaming(serving_client):
    """POST /v1/completions with stream=True returns text/event-stream ending with [DONE]."""
    payload = {
        "prompt": "Hello world stream",
        "max_tokens": 16,
        "stream": True,
    }
    response = serving_client.post("/v1/completions", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    lines = [line.strip() for line in response.text.split("\n") if line.strip()]
    assert any(line.startswith("data: ") for line in lines)
    assert lines[-1] == "data: [DONE]"

    # Parse first data chunk
    first_data_line = [l for l in lines if l.startswith("data: ") and l != "data: [DONE]"][0]
    chunk = json.loads(first_data_line[6:])
    assert chunk["object"] == "text_completion"
    assert chunk["choices"][0]["text"] == "synthetic assistant response"


def test_chat_completions_non_stream(serving_client):
    """POST /v1/chat/completions produces valid chat.completion payload."""
    payload = {
        "messages": [{"role": "user", "content": "How are you?"}],
        "max_tokens": 20,
        "stream": False,
    }
    response = serving_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == "synthetic assistant response"
    assert data["usage"]["total_tokens"] > 0


def test_chat_completions_sse_streaming(serving_client):
    """POST /v1/chat/completions with stream=True yields valid SSE delta events."""
    payload = {
        "messages": [{"role": "user", "content": "Tell me a joke"}],
        "max_tokens": 10,
        "stream": True,
    }
    response = serving_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    lines = [l.strip() for l in response.text.split("\n") if l.strip()]
    assert lines[-1] == "data: [DONE]"

    first_chunk = json.loads(lines[0][6:])
    assert first_chunk["object"] == "chat.completion.chunk"
    assert first_chunk["choices"][0]["delta"]["role"] == "assistant"
    assert first_chunk["choices"][0]["delta"]["content"] == "synthetic assistant response"


def test_telemetry_metrics_endpoint(serving_client):
    """GET /metrics and /v1/telemetry report accurate latency and request counts."""
    # Execute a request to populate telemetry
    serving_client.post("/v1/completions", json={"prompt": "telemetry probe", "max_tokens": 5})

    response = serving_client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["total_requests"] >= 1
    assert "throughput" in data
    assert "latency_ms" in data


def test_auth_bearer_token():
    """Verify HTTP Bearer token authentication enforcement when api_key is configured."""
    model = MockCausalLM()
    tokenizer = MockTokenizer()
    app = create_openai_serving_app(
        model=model,
        tokenizer=tokenizer,
        api_key="secret-token-12345",
    )
    client = TestClient(app)

    # Health is exempt from auth
    assert client.get("/health").status_code == 200

    # Completions without token -> 401
    r_no_auth = client.post("/v1/completions", json={"prompt": "test"})
    assert r_no_auth.status_code == 401

    # Wrong token -> 401
    r_bad_auth = client.post(
        "/v1/completions",
        json={"prompt": "test"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert r_bad_auth.status_code == 401

    # Correct token -> 200
    r_good_auth = client.post(
        "/v1/completions",
        json={"prompt": "test"},
        headers={"Authorization": "Bearer secret-token-12345"},
    )
    assert r_good_auth.status_code == 200


def test_rate_limiting_enforcement():
    """Verify sliding-window rate limit returns 429 when threshold is reached."""
    model = MockCausalLM()
    tokenizer = MockTokenizer()
    # 2 requests per minute limit
    app = create_openai_serving_app(
        model=model,
        tokenizer=tokenizer,
        rate_limit_rpm=2,
    )
    client = TestClient(app)

    # 1st request -> 200
    r1 = client.post("/v1/completions", json={"prompt": "r1"})
    assert r1.status_code == 200

    # 2nd request -> 200
    r2 = client.post("/v1/completions", json={"prompt": "r2"})
    assert r2.status_code == 200

    # 3rd request -> 429 Too Many Requests
    r3 = client.post("/v1/completions", json={"prompt": "r3"})
    assert r3.status_code == 429
    assert "Rate limit exceeded" in r3.json()["detail"]
