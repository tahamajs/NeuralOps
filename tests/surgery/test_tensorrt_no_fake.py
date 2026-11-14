#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests verifying that TensorRT engine export refuses to produce fake artifacts
and raises NotImplementedError.
"""

import pytest
import torch.nn as nn

from neuralops.surgery.hf_cache_bridge import export_tensorrt_engine


def test_export_tensorrt_engine_raises_not_implemented(tmp_path):
    """Verify export_tensorrt_engine raises NotImplementedError."""
    model = nn.Sequential(nn.Linear(16, 16))
    out_file = tmp_path / "model.engine"

    with pytest.raises(NotImplementedError, match="TensorRT export requires ONNX intermediate; not implemented"):
        export_tensorrt_engine(model, out_file)
    assert not out_file.exists()
