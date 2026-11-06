#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests for serving security defaults and API key authentication."""

from __future__ import annotations

import inspect
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

import neuralops.api as api
from neuralops.cli.serve_cmd import build_serve_parser
from neuralops.integrations.serving import create_openai_serving_app


def test_default_host_is_localhost():
    """Assert api.serve and serve_cmd default host to 127.0.0.1."""
    sig = inspect.signature(api.serve)
    assert sig.parameters["host"].default == "127.0.0.1", (
        f"api.serve host default should be 127.0.0.1, got {sig.parameters['host'].default}"
    )

    parser = build_serve_parser()
    assert parser.get_default("host") == "127.0.0.1", (
        f"serve_cmd parser host default should be 127.0.0.1, got {parser.get_default('host')}"
    )


def test_non_localhost_without_api_key_fails(monkeypatch):
    """Assert serve(host="0.0.0.0") without an API key raises ValueError."""
    monkeypatch.delenv("NEURALOPS_API_KEY", raising=False)
    dummy_model = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="Refusing to bind serving to non-localhost address without an API key"):
        api.serve(dummy_model, host="0.0.0.0", api_key=None)


def test_api_key_required_for_protected_endpoints():
    """With TestClient, assert missing auth returns 401 and valid key returns 200."""
    dummy_model = nn.Linear(4, 4)
    app = create_openai_serving_app(dummy_model, tokenizer=None, api_key="secret-key-1234")
    client = TestClient(app)

    # Missing auth -> 401
    resp_unauth = client.get("/v1/models")
    assert resp_unauth.status_code == 401
    assert "detail" in resp_unauth.json()

    # Wrong auth -> 401
    resp_wrong = client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})
    assert resp_wrong.status_code == 401

    # Valid auth -> 200
    resp_auth = client.get("/v1/models", headers={"Authorization": "Bearer secret-key-1234"})
    assert resp_auth.status_code == 200


def test_health_exempt_from_auth():
    """Assert /health endpoint works without auth even when API key is set."""
    dummy_model = nn.Linear(4, 4)
    app = create_openai_serving_app(dummy_model, tokenizer=None, api_key="secret-key-1234")
    client = TestClient(app)

    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json().get("backend") == "neuralops"
