import pytest
import torch
import torch.nn as nn
import framework.attentions.adaptive_span_attention

def test_adaptive_span_attention_module_present():
    assert hasattr(framework.attentions, 'adaptive_span_attention')

def test_adaptive_span_attention_injection_safe():
    from framework.attentions.adaptive_span_attention import inject_adaptive_span
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_adaptive_span(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
