import framework.attentions.simple_ssm
import pytest
import torch
import torch.nn as nn


def test_simple_ssm_module_present():
    assert hasattr(framework.attentions, 'simple_ssm')

def test_simple_ssm_injection_safe():
    from framework.attentions.simple_ssm import inject_simple_ssm
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_simple_ssm(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
