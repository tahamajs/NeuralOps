import pytest
import torch
import torch.nn as nn
import framework.attentions.SVD-Mamba

def test_SVD-Mamba_module_present():
    assert hasattr(framework.attentions, 'SVD-Mamba')
