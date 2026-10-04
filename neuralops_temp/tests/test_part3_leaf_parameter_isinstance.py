"""Part 3 Verification: isinstance-based Parameter Targeting (RCA-3)."""
import pytest
import torch
import torch.nn as nn
from neuralops.core.attention_utils import _copy_projection_weights


def test_parameter_copy_success():
    class DirectParam(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(32, 32)
    m = DirectParam()
    src_w = torch.randn(32, 32)
    _copy_projection_weights(m.q_proj, src_w, None)
    assert torch.allclose(m.q_proj.weight, src_w)

def test_property_target_protection():
    class FactorizedProp(nn.Module):
        def __init__(self):
            super().__init__()
            self.U = nn.Parameter(torch.randn(32, 8))
            self.V = nn.Parameter(torch.randn(8, 32))
        @property
        def weight(self):
            return self.U @ self.V

    m = FactorizedProp()
    src_w = torch.randn(32, 32)
    # Must safely skip without throwing AttributeError on read-only property
    _copy_projection_weights(m, src_w, None)
