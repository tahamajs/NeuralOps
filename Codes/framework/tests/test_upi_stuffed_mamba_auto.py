import framework.attentions.upi_stuffed_mamba
import pytest
import torch
import torch.nn as nn


def test_upi_stuffed_mamba_module_present():
    assert hasattr(framework.attentions, 'upi_stuffed_mamba')

def test_upi_stuffed_mamba_injection_safe():
    from framework.attentions.upi_stuffed_mamba import inject_upi_mamba
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_upi_mamba(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
