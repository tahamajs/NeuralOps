#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Conftest hook for tests/models to allow CPU validation of cache types."""

from __future__ import annotations

import pytest


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    for item in items:
        # test_cuda_graph_static tests type validation on CPU before CUDA capture
        if "test_cuda_graph_static" in str(item.fspath):
            item.own_markers = [m for m in item.own_markers if m.name not in ("skip", "gpu")]
