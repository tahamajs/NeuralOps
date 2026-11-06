#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests for UI CORS restriction and --share auth enforcement."""

from __future__ import annotations

import argparse
import pytest
from fastapi.middleware.cors import CORSMiddleware

from neuralops.ui.cli import cmd_ui
from neuralops.ui.server import create_app


def test_cors_not_wildcard():
    """Assert allow_origins does not contain '*' and only lists localhost origins."""
    app = create_app()
    cors_middlewares = [m for m in app.user_middleware if m.cls == CORSMiddleware]
    assert len(cors_middlewares) == 1, "CORSMiddleware must be configured"

    cors_options = cors_middlewares[0].kwargs
    allow_origins = cors_options.get("allow_origins", [])

    assert "*" not in allow_origins, "CORS allow_origins must not contain wildcard '*'"
    assert len(allow_origins) > 0, "CORS allow_origins must be defined"
    for origin in allow_origins:
        assert "127.0.0.1" in origin or "localhost" in origin, (
            f"Origin '{origin}' is not a localhost origin"
        )


def test_share_requires_auth():
    """Assert cmd_ui with --share and without --auth raises ValueError."""
    args = argparse.Namespace(
        port=8080,
        host="127.0.0.1",
        share=True,
        static=None,
        no_browser=True,
        auth=None,
    )
    with pytest.raises(ValueError, match="--auth is required when sharing"):
        cmd_ui(args)
