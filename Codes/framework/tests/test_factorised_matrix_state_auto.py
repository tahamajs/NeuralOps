import framework.attentions.factorised_matrix_state
import pytest
import torch
import torch.nn as nn


def test_factorised_matrix_state_module_present():
    assert hasattr(framework.attentions, 'factorised_matrix_state')

def test_factorised_matrix_state_injection_safe():
    from framework.attentions.factorised_matrix_state import \
        inject_factorised_matrix_state
    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(64, 64)
    lyr = DummyLayer()
    try:
        res = inject_factorised_matrix_state(layer=lyr)
        assert res is None or isinstance(res, nn.Module)
    except Exception as e:
        pytest.fail(f'Injection crashed: {e}')
