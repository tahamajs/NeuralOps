import framework.attentions.cosformer
import pytest
import torch
import torch.nn as nn


def test_cosformer_module_present():
    assert hasattr(framework.attentions, 'cosformer')

def test_cosformer_injection_safe():
    from framework.attentions.cosformer import inject_cosformer
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_cosformer(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
