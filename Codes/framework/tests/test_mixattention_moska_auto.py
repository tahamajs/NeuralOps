import framework.attentions.mixattention_moska
import pytest
import torch
import torch.nn as nn


def test_mixattention_moska_module_present():
    assert hasattr(framework.attentions, 'mixattention_moska')

def test_mixattention_moska_injection_safe():
    from framework.attentions.mixattention_moska import \
        inject_mixattention_moska
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_mixattention_moska(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
