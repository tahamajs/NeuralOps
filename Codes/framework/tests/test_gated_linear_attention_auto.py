import framework.attentions.gated_linear_attention
import pytest
import torch
import torch.nn as nn


def test_gated_linear_attention_module_present():
    assert hasattr(framework.attentions, 'gated_linear_attention')

def test_gated_linear_attention_injection_safe():
    from framework.attentions.gated_linear_attention import inject_gated_linear
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_gated_linear(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
