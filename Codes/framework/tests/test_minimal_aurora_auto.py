import framework.attentions.minimal_aurora
import pytest
import torch
import torch.nn as nn


def test_minimal_aurora_module_present():
    assert hasattr(framework.attentions, 'minimal_aurora')

def test_minimal_aurora_injection_safe():
    from framework.attentions.minimal_aurora import inject_minimal_aurora
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_minimal_aurora(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
