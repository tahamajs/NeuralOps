import framework.attentions.moe_deltakv_paged
import pytest
import torch
import torch.nn as nn


def test_moe_deltakv_paged_module_present():
    assert hasattr(framework.attentions, 'moe_deltakv_paged')
