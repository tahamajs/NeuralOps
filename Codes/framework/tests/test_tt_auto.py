import framework.attentions.tt
import pytest
import torch
import torch.nn as nn


def test_tt_module_present():
    assert hasattr(framework.attentions, 'tt')

def test_tt_injection_safe():
    from framework.attentions.tt import inject_token_compressed
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_token_compressed(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
