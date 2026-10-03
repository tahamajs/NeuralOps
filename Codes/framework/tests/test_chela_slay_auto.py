import framework.attentions.chela_slay
import pytest
import torch
import torch.nn as nn


def test_chela_slay_module_present():
    assert hasattr(framework.attentions, 'chela_slay')

def test_chela_slay_injection_safe():
    from framework.attentions.chela_slay import inject_chela_slay
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_chela_slay(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
