import framework.attentions.span_expanded_attention
import pytest
import torch
import torch.nn as nn


def test_span_expanded_attention_module_present():
    assert hasattr(framework.attentions, 'span_expanded_attention')

def test_span_expanded_attention_injection_safe():
    from framework.attentions.span_expanded_attention import \
        inject_span_expanded
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_span_expanded(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
