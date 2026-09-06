import pytest
import torch
import torch.nn as nn
import framework.attentions.rala_compressor

def test_rala_compressor_module_present():
    assert hasattr(framework.attentions, 'rala_compressor')
