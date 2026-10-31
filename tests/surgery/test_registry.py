#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for surgery registry infrastructure.
"""

import pytest

from neuralops.surgery.registry import (
    SURGERY_REGISTRY,
    get_surgeries_for_model,
    get_surgery,
    list_surgeries,
    register_surgery,
)


def test_surgery_registration_and_retrieval():
    try:
        @register_surgery(
            name="test_dummy_surgery",
            category="custom_test_cat",
            compat=["llama", "mistral"],
            requires_gpu=False,
        )
        def dummy_fn(model, **kwargs):
            return model

        assert "test_dummy_surgery" in list_surgeries()
        assert "test_dummy_surgery" in list_surgeries(category="custom_test_cat")
        assert len(list_surgeries(category="non_existent_category")) == 0

        meta = get_surgery("test_dummy_surgery")
        assert meta.name == "test_dummy_surgery"
        assert meta.category == "custom_test_cat"
        assert "llama" in meta.compat

        llama_surgeries = get_surgeries_for_model("llama")
        assert "test_dummy_surgery" in llama_surgeries

        with pytest.raises(KeyError):
            get_surgery("non_existent_surgery")
    finally:
        SURGERY_REGISTRY.pop("test_dummy_surgery", None)


def test_experimental_surgery_warning():
    try:
        @register_surgery(
            name="test_exp_op",
            category="mlp",
            _experimental=True,
        )
        def exp_fn(model):
            return model

        with pytest.warns(UserWarning, match="experimental"):
            exp_fn("mock_model")
    finally:
        SURGERY_REGISTRY.pop("test_exp_op", None)
