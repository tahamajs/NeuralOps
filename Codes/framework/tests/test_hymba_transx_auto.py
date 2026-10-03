import framework.attentions.hymba_transx
import pytest
import torch
import torch.nn as nn


def test_hymba_transx_module_present():
    assert hasattr(framework.attentions, 'hymba_transx')

def test_hymba_transx_injection_safe():
    from framework.attentions.hymba_transx import inject_hymba_transx
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_hymba_transx(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
