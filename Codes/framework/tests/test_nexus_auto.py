import framework.attentions.nexus
import pytest
import torch
import torch.nn as nn


def test_nexus_module_present():
    assert hasattr(framework.attentions, 'nexus')

def test_nexus_injection_safe():
    from framework.attentions.nexus import inject_nexus_into_distilbert
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_nexus_into_distilbert(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
