import framework.attentions.minimal_nova
import pytest
import torch
import torch.nn as nn


def test_minimal_nova_module_present():
    assert hasattr(framework.attentions, 'minimal_nova')

def test_minimal_nova_injection_safe():
    from framework.attentions.minimal_nova import inject_minimal_nova
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_minimal_nova(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
