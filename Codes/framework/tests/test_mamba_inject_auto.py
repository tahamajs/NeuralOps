import framework.attentions.mamba_inject
import pytest
import torch
import torch.nn as nn


def test_mamba_inject_module_present():
    assert hasattr(framework.attentions, 'mamba_inject')

def test_mamba_inject_injection_safe():
    from framework.attentions.mamba_inject import inject_mamba
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_mamba(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
