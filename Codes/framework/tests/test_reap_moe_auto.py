import framework.attentions.reap_moe
import pytest
import torch
import torch.nn as nn


def test_reap_moe_module_present():
    assert hasattr(framework.attentions, 'reap_moe')

def test_reap_moe_injection_safe():
    from framework.attentions.reap_moe import inject_reap_moe
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_reap_moe(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
