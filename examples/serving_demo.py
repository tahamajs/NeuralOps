#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Serving demo: serve model with FastAPI OpenAI-compatible API."""

from __future__ import annotations

import sys
from transformers import GPT2Config, GPT2LMHeadModel
from neuralops.serving import create_openai_serving_app


def main() -> None:
    print("NeuralOps - Production Serving Demo (FastAPI)")

    # Create lightweight model for demonstration
    config = GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=64,
        vocab_size=256,
        n_positions=128,
    )
    model = GPT2LMHeadModel(config)
    model.eval()

    # Create FastAPI serving app
    print("Creating OpenAI-compatible serving application...")
    app = create_openai_serving_app(
        model=model,
        model_name="neuralops-gpt2-demo",
        rate_limit_rpm=120,
    )

    print("FastAPI application instantiated successfully!")
    print(f"Routes registered: {[route.path for route in app.routes if hasattr(route, 'path')]}")

    # Test the /health endpoint using FastAPI TestClient if available
    try:
        from starlette.testclient import TestClient

        client = TestClient(app)
        response = client.get("/health")
        print(f"Health check status: {response.status_code}")
        print(f"Health check response: {response.json()}")
        assert response.status_code == 200
    except ImportError:
        print("TestClient not installed; skipping local HTTP roundtrip.")

    print("\nTo launch live production server, run:")
    print("  uvicorn examples.serving_demo:app --host 0.0.0.0 --port 8000")
    print("Serving demo completed successfully.")


# Module-level app for uvicorn invocation: `uvicorn examples.serving_demo:app`
app = create_openai_serving_app(model_name="neuralops-gpt2-demo")

if __name__ == "__main__":
    main()
