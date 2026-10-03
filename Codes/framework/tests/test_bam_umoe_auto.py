import framework.attentions.bam_umoe
import pytest
import torch
import torch.nn as nn


def test_bam_umoe_module_present():
    assert hasattr(framework.attentions, 'bam_umoe')

def test_bam_umoe_injection_safe():
    from framework.attentions.bam_umoe import inject_bam_umoe
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_bam_umoe(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
