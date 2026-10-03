import framework.attentions.progressive_token_pruning
import pytest
import torch
import torch.nn as nn


def test_progressive_token_pruning_module_present():
    assert hasattr(framework.attentions, 'progressive_token_pruning')

def test_progressive_token_pruning_injection_safe():
    from framework.attentions.progressive_token_pruning import \
        inject_progressive_token_pruning
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_progressive_token_pruning(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
