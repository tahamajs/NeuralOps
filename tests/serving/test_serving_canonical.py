#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for canonical serving architecture and kwarg validation (finding F-93).
"""

from __future__ import annotations

import pytest

import neuralops.integrations.serving as int_serving
import neuralops.serving as serving
from neuralops.core.exceptions import ConfigError, IntegrationNotImplementedError


def test_serving_is_integrations_app():
    """Verify neuralops.serving re-exports canonical create_openai_serving_app from integrations.serving."""
    assert serving.create_openai_serving_app is int_serving.create_openai_serving_app


def test_api_serve_kwargs_accepted():
    """Verify backend and kv_cache_budget_gb kwargs are validated or honoured, never silently ignored."""
    # 1. Valid kwargs accepted and tracked on app.state
    app = int_serving.create_openai_serving_app(backend="fastapi", kv_cache_budget_gb=8.0)
    assert app.state.backend == "fastapi"
    assert app.state.kv_cache_budget_gb == 8.0

    app2 = int_serving.create_openai_serving_app(backend="neuralops_native", kv_cache_budget_gb=2.5)
    assert app2.state.backend == "neuralops_native"
    assert app2.state.kv_cache_budget_gb == 2.5

    # 2. Unsupported external serving engines fail closed with IntegrationNotImplementedError
    with pytest.raises(IntegrationNotImplementedError):
        int_serving.create_openai_serving_app(backend="vllm")

    with pytest.raises(IntegrationNotImplementedError):
        int_serving.create_openai_serving_app(backend="sglang")

    with pytest.raises(IntegrationNotImplementedError):
        int_serving.create_openai_serving_app(backend="tgi")

    # 3. Invalid backend raises ConfigError
    with pytest.raises(ConfigError):
        int_serving.create_openai_serving_app(backend="invalid_engine_xyz")

    # 4. Invalid non-positive kv_cache_budget_gb raises ValueError
    with pytest.raises(ValueError):
        int_serving.create_openai_serving_app(kv_cache_budget_gb=-1.0)

    with pytest.raises(ValueError):
        int_serving.create_openai_serving_app(kv_cache_budget_gb=0.0)

    # 5. Unexpected kwargs raise TypeError (never silently ignored)
    with pytest.raises(TypeError):
        int_serving.create_openai_serving_app(unexpected_foo="bar")


def test_benchmark_serving_is_harness_not_server():
    """Verify neuralops.benchmarks.serving is a microbenchmark harness, not an HTTP server."""
    import neuralops.benchmarks.serving as bench_serving

    assert hasattr(bench_serving, "run_benchmark")
    assert not hasattr(bench_serving, "create_openai_serving_app")
    assert not hasattr(bench_serving, "app")
    assert not hasattr(bench_serving, "FastAPI")
