import framework.attentions.eat
import pytest
import torch
import torch.nn as nn


def test_eat_module_present():
    assert hasattr(framework.attentions, 'eat')

def test_eat_injection_safe():
    from framework.attentions.eat import inject_eat
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_eat(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
