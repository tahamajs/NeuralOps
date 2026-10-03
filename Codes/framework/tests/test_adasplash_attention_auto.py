import framework.attentions.adasplash_attention
import pytest
import torch
import torch.nn as nn


def test_adasplash_attention_module_present():
    assert hasattr(framework.attentions, 'adasplash_attention')

def test_adasplash_attention_injection_safe():
    from framework.attentions.adasplash_attention import inject_adasplash
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_adasplash(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
