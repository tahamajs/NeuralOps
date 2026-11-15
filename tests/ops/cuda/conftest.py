"""Pytest configuration for tests/ops/cuda.

Ensures that CPU fallback, reference verification, and invariant tests
can run without requiring physical CUDA hardware.
"""

from __future__ import annotations

import pytest


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    for item in items:
        item_path = str(item.fspath)
        if "tests/ops/cuda" in item_path:
            # Remove skip marker if test does not explicitly require physical CUDA hardware
            item.own_markers = [
                m
                for m in item.own_markers
                if not (
                    getattr(m, "name", "") == "skip"
                    and "CUDA device not available" in str(m.kwargs.get("reason", ""))
                )
            ]
