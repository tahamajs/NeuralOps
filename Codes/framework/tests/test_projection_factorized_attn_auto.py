import framework.attentions.projection_factorized_attn
import pytest
import torch
import torch.nn as nn


def test_projection_factorized_attn_module_present():
    assert hasattr(framework.attentions, 'projection_factorized_attn')
