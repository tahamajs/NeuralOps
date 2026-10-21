#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Regression tests that OpenAI-shaped serving responses come from model inference."""

import pytest
import torch
import torch.nn as nn

fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("starlette.testclient").TestClient

from neuralops.integrations.serving import create_openai_serving_app


class _Tokenizer:
    def __call__(self, prompt, return_tensors):
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[3, 4]]), "attention_mask": torch.tensor([[1, 1]])}

    def decode(self, token_ids, skip_special_tokens):
        assert skip_special_tokens
        return "model-generated answer" if token_ids.numel() else ""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return "templated chat prompt"


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.last_generation_kwargs = None

    def generate(self, input_ids, attention_mask, **kwargs):
        self.last_generation_kwargs = kwargs
        completion = torch.tensor([[9, 10]], device=input_ids.device)
        return torch.cat([input_ids, completion], dim=1)


def test_generation_endpoint_requires_real_model_and_tokenizer():
    client = TestClient(create_openai_serving_app())

    response = client.post("/v1/completions", json={"prompt": "hello"})

    assert response.status_code == 503
    assert response.json()["detail"] == "A model and tokenizer must be loaded to serve requests"


def test_completion_endpoint_returns_model_output_and_actual_usage():
    model = _Model()
    client = TestClient(create_openai_serving_app(model, _Tokenizer()))

    response = client.post("/v1/completions", json={"prompt": "hello", "max_tokens": 2, "temperature": 0})

    assert response.status_code == 200
    payload = response.json()
    assert payload["choices"][0]["text"] == "model-generated answer"
    assert payload["usage"] == {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
    assert model.last_generation_kwargs == {"max_new_tokens": 2, "do_sample": False}


def test_chat_stream_emits_actual_generated_text():
    client = TestClient(create_openai_serving_app(_Model(), _Tokenizer()))
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}], "stream": True},
    )

    assert response.status_code == 200
    assert "model-generated answer" in response.text
    assert "[DONE]" in response.text
