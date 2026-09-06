import pytest
import torch
import torch.nn as nn
import framework.attentions.moe_deltakv_paged

def test_moe_deltakv_paged_module_present():
    assert hasattr(framework.attentions, 'moe_deltakv_paged')
