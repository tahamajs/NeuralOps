#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for quantize_int4 surgery operator.
"""

import pytest

from neuralops.core.errors import IntegrationNotImplementedError
from neuralops.surgery.registry import get_surgery
from neuralops.surgery.weights.quantize import quantize_int4


def test_quantize_int4_metadata():
    meta = get_surgery("quantize_int4")
    assert meta.name == "quantize_int4"
    assert meta.category == "weights"
    assert meta.requires_gpu is True
    assert meta.requires_calibration is True
    assert meta._experimental is True


def test_quantize_int4_fail_closed_when_library_missing():
    with pytest.warns(UserWarning, match="experimental"):
        with pytest.raises(IntegrationNotImplementedError, match="autoawq"):
            quantize_int4("dummy_model", method="awq")

    with pytest.warns(UserWarning, match="experimental"):
        with pytest.raises(IntegrationNotImplementedError, match="gptqmodel"):
            quantize_int4("dummy_model", method="gptq")


def test_quantize_int4_invalid_method():
    with pytest.warns(UserWarning, match="experimental"):
        with pytest.raises(ValueError, match="Unsupported quantization method"):
            quantize_int4("dummy_model", method="invalid_method")
