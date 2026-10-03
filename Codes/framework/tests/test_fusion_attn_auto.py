import framework.attentions.fusion_attn
import pytest
import torch
import torch.nn as nn


def test_fusion_attn_module_present():
    assert hasattr(framework.attentions, 'fusion_attn')

def test_fusion_attn_injection_safe():
    from framework.attentions.fusion_attn import inject_fusion_attn
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_fusion_attn(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
