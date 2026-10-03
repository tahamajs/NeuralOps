import framework.attentions.minimal_svd_v2
import pytest
import torch
import torch.nn as nn


def test_minimal_svd_v2_module_present():
    assert hasattr(framework.attentions, 'minimal_svd_v2')

def test_minimal_svd_v2_injection_safe():
    from framework.attentions.minimal_svd_v2 import inject_minimal_svd_v2
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_minimal_svd_v2(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
