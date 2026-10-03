import framework.attentions.rala_compressor
import pytest
import torch
import torch.nn as nn


def test_rala_compressor_module_present():
    assert hasattr(framework.attentions, 'rala_compressor')
