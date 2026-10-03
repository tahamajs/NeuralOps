import framework.attentions.jamba_info
import pytest
import torch
import torch.nn as nn


def test_jamba_info_module_present():
    assert hasattr(framework.attentions, 'jamba_info')

def test_jamba_info_injection_safe():
    from framework.attentions.jamba_info import inject_jamba_info
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_jamba_info(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
