import framework.attentions.gated_deltanet_block
import pytest
import torch
import torch.nn as nn


def test_gated_deltanet_block_module_present():
    assert hasattr(framework.attentions, 'gated_deltanet_block')

def test_gated_deltanet_block_injection_safe():
    from framework.attentions.gated_deltanet_block import \
        inject_gated_deltanet_block
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_gated_deltanet_block(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
