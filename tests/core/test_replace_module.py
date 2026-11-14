# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for core module surgery: replace_module and get_parent_and_child_name.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from neuralops.core.model_utils import get_parent_and_child_name, replace_module


class SimpleContainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


class NestedComposite(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(16, 16),
                nn.Linear(16, 16),
            ]
        )
        self.dict_blocks = nn.ModuleDict(
            {
                "attn": nn.Linear(16, 16),
                "mlp": nn.Linear(16, 16),
            }
        )
        self.sub = SimpleContainer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.dict_blocks["attn"](x)
        x = self.dict_blocks["mlp"](x)
        return self.sub(x)


def test_replace_module_standard_attribute():
    """Verify replacing a direct child module via dotted attribute name."""
    model = SimpleContainer()
    new_layer = nn.Identity()

    replace_module(model, "fc1", new_layer)
    assert model.fc1 is new_layer

    x = torch.randn(2, 16)
    out = model(x)
    assert out.shape == (2, 8)


def test_replace_module_in_module_list():
    """Verify replacing an element inside nn.ModuleList by numeric index."""
    model = NestedComposite()
    replacement = nn.Identity()

    replace_module(model, "layers.0", replacement)
    assert model.layers[0] is replacement

    replace_module(model, "layers.1", nn.Linear(16, 16))
    assert isinstance(model.layers[1], nn.Linear)


def test_replace_module_in_module_dict():
    """Verify replacing an entry inside nn.ModuleDict by key."""
    model = NestedComposite()
    replacement = nn.Identity()

    replace_module(model, "dict_blocks.attn", replacement)
    assert model.dict_blocks["attn"] is replacement


def test_replace_module_nested_attribute():
    """Verify replacing a deeply nested child submodule."""
    model = NestedComposite()
    replacement = nn.Linear(16, 32)

    replace_module(model, "sub.fc1", replacement)
    assert model.sub.fc1 is replacement


def test_get_parent_and_child_name_standard():
    """Verify get_parent_and_child_name on flat attribute."""
    model = SimpleContainer()
    parent, child_name = get_parent_and_child_name(model, "fc1")
    assert parent is model
    assert child_name == "fc1"


def test_get_parent_and_child_name_module_list():
    """Verify get_parent_and_child_name indexing into ModuleList."""
    model = NestedComposite()
    parent, child_name = get_parent_and_child_name(model, "layers.1")
    assert parent is model.layers
    assert child_name == "1"


def test_get_parent_and_child_name_module_dict():
    """Verify get_parent_and_child_name indexing into ModuleDict."""
    model = NestedComposite()
    parent, child_name = get_parent_and_child_name(model, "dict_blocks.mlp")
    assert parent is model.dict_blocks
    assert child_name == "mlp"


def test_get_parent_and_child_name_nested():
    """Verify get_parent_and_child_name on nested module hierarchy."""
    model = NestedComposite()
    parent, child_name = get_parent_and_child_name(model, "sub.fc2")
    assert parent is model.sub
    assert child_name == "fc2"


def test_get_parent_and_child_name_empty():
    """Verify get_parent_and_child_name with empty string returns root model."""
    model = SimpleContainer()
    parent, child_name = get_parent_and_child_name(model, "")
    assert parent is model
    assert child_name == ""


def test_replace_module_forward_correctness():
    """Verify mathematical output correctness after replacing a module."""
    model = SimpleContainer()

    class CustomMultiplier(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 2.0

    multiplier = CustomMultiplier()
    replace_module(model, "fc1", multiplier)

    # Set fc2 weight to identity and zero bias
    with torch.no_grad():
        model.fc2 = nn.Linear(16, 16, bias=False)
        model.fc2.weight.copy_(torch.eye(16))

    x = torch.ones(1, 16)
    out = model(x)
    expected = torch.ones(1, 16) * 2.0
    assert torch.allclose(out, expected)
