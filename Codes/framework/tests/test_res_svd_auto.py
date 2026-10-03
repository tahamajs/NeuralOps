import framework.attentions.res_svd
import pytest
import torch
import torch.nn as nn


def test_res_svd_module_present():
    assert hasattr(framework.attentions, 'res_svd')

def test_res_svd_injection_safe():
    from framework.attentions.res_svd import inject_res_svd
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_res_svd(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
