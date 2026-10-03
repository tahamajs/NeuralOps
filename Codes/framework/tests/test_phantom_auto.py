import framework.attentions.phantom
import pytest
import torch
import torch.nn as nn


def test_phantom_module_present():
    assert hasattr(framework.attentions, 'phantom')

def test_phantom_injection_safe():
    from framework.attentions.phantom import inject_phantom
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_phantom(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
