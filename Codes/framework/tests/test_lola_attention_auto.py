import framework.attentions.lola_attention
import pytest
import torch
import torch.nn as nn


def test_lola_attention_module_present():
    assert hasattr(framework.attentions, 'lola_attention')

def test_lola_attention_injection_safe():
    from framework.attentions.lola_attention import inject_lola_attention
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_lola_attention(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
