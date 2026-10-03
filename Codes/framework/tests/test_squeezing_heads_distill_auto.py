import framework.attentions.squeezing_heads_distill
import pytest
import torch
import torch.nn as nn


def test_squeezing_heads_distill_module_present():
    assert hasattr(framework.attentions, 'squeezing_heads_distill')

def test_squeezing_heads_distill_injection_safe():
    from framework.attentions.squeezing_heads_distill import \
        inject_squeezing_heads
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_squeezing_heads(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
