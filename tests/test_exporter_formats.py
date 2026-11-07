#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests verifying export format validation consistency and fail-closed AWQ behavior (F-64)."""

from __future__ import annotations

import pytest
import torch.nn as nn

from neuralops.export.exporter import SUPPORTED_EXPORT_FORMATS, export_model


def test_error_message_lists_only_supported_formats(tmp_path) -> None:
    """Verify that unknown export formats raise ValueError listing only genuinely supported formats without AWQ."""
    model = nn.Linear(4, 4)
    with pytest.raises(ValueError) as exc_info:
        export_model(model, tmp_path / "out", format="non_existent_format_xyz")

    msg = str(exc_info.value)
    assert "Unsupported export format 'non_existent_format_xyz'" in msg
    assert "awq" not in msg.lower().split("choose from:")[1]

    for fmt in SUPPORTED_EXPORT_FORMATS:
        assert fmt in msg


def test_awq_raises_not_implemented(tmp_path) -> None:
    """Verify that format='awq' raises an honest NotImplementedError rather than fake-quantizing."""
    model = nn.Linear(4, 4)
    with pytest.raises(NotImplementedError) as exc_info:
        export_model(model, tmp_path / "out", format="awq")

    msg = str(exc_info.value)
    assert "AWQ export is not implemented" in msg
    assert "supported AWQ tool" in msg
