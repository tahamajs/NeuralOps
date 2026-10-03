import framework.attentions.hgrn2_gated_deltanet
import pytest
import torch
import torch.nn as nn


def test_hgrn2_gated_deltanet_module_present():
    assert hasattr(framework.attentions, 'hgrn2_gated_deltanet')

def test_hgrn2_gated_deltanet_injection_safe():
    from framework.attentions.hgrn2_gated_deltanet import \
        inject_hgrn2_gated_deltanet
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_hgrn2_gated_deltanet(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
