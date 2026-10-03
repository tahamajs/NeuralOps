import framework.attentions.leka_v3
import pytest
import torch
import torch.nn as nn


def test_leka_v3_module_present():
    assert hasattr(framework.attentions, 'leka_v3')

def test_leka_v3_injection_safe():
    from framework.attentions.leka_v3 import inject_leka_v3
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_leka_v3(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
