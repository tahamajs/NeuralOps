import framework.attentions.minimal_prism
import pytest
import torch
import torch.nn as nn


def test_minimal_prism_module_present():
    assert hasattr(framework.attentions, 'minimal_prism')

def test_minimal_prism_injection_safe():
    from framework.attentions.minimal_prism import inject_minimal_prism
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_minimal_prism(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
