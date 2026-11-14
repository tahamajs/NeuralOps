#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Pytest fixtures and configuration for neuralops.ui test suite."""

from __future__ import annotations

import warnings


def pytest_configure(config) -> None:
    """Register warning filters for UI tests."""
    warnings.filterwarnings("ignore", message=".*httpx.*", category=DeprecationWarning)
    try:
        from starlette.exceptions import StarletteDeprecationWarning

        warnings.filterwarnings("ignore", category=StarletteDeprecationWarning)
    except (ImportError, AttributeError):
        pass
