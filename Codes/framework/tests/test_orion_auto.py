import framework.attentions.orion
import pytest
import torch
import torch.nn as nn


def test_orion_module_present():
    assert hasattr(framework.attentions, 'orion')

def test_orion_injection_safe():
    from framework.attentions.orion import inject_orion
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_orion(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
