import framework.attentions.flash_attention3_fp8
import pytest
import torch
import torch.nn as nn


def test_flash_attention3_fp8_module_present():
    assert hasattr(framework.attentions, 'flash_attention3_fp8')

def test_flash_attention3_fp8_injection_safe():
    from framework.attentions.flash_attention3_fp8 import inject_flash3_fp8
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_flash3_fp8(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
