import framework.attentions.hgrn2_layer
import pytest
import torch
import torch.nn as nn


def test_hgrn2_layer_module_present():
    assert hasattr(framework.attentions, 'hgrn2_layer')

def test_hgrn2_layer_injection_safe():
    from framework.attentions.hgrn2_layer import inject_hgrn2_layer
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_hgrn2_layer(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
