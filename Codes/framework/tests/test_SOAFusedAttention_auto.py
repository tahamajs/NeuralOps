import framework.attentions.SOAFusedAttention
import pytest
import torch
import torch.nn as nn


def test_SOAFusedAttention_module_present():
    assert hasattr(framework.attentions, 'SOAFusedAttention')

def test_SOAFusedAttention_injection_safe():
    from framework.attentions.SOAFusedAttention import inject_soa_fused_attn
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_soa_fused_attn(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
