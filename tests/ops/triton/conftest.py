"""Pytest configuration for tests/ops/triton.

Ensures that AST invariant checks and CPU fallback / parity tests in
tests/ops/triton can run without requiring a physical CUDA device or
Triton GPU compiler.
"""

from __future__ import annotations

import pytest


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    for item in items:
        item_path = str(item.fspath)
        if "tests/ops/triton" in item_path:
            # Remove any skip markers added solely because CUDA/Triton hardware is absent
            item.own_markers = [
                m
                for m in item.own_markers
                if not (
                    getattr(m, "name", "") == "skip"
                    and (
                        "CUDA device not available" in str(m.kwargs.get("reason", ""))
                        or "Triton compiler not available" in str(m.kwargs.get("reason", ""))
                    )
                )
            ]
