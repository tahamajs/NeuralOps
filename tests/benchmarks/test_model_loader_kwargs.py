from unittest.mock import MagicMock
import torch.nn as nn
from neuralops.benchmarks.core.model_loader import apply_attention_module


def test_apply_attention_filters_unexpected_kwargs(monkeypatch):
    from neuralops.core.registry import ATTENTION_REGISTRY

    # Pick an existing registered key from ATTENTION_REGISTRY
    existing_key = next(iter(ATTENTION_REGISTRY.keys()))

    mock_inject = MagicMock(return_value="injected_model")
    monkeypatch.setattr("neuralops.core.attention_utils.inject_attention", mock_inject)

    dummy_model = nn.Linear(10, 10)
    res = apply_attention_module(
        dummy_model,
        module_name=existing_key,
        rank=32,
        unexpected_extra_param="will_be_dropped",
    )
    assert res == "injected_model"
    assert "unexpected_extra_param" not in mock_inject.call_args[1]
