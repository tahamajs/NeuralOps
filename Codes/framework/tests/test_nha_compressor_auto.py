import framework.attentions.nha_compressor
import pytest
import torch
import torch.nn as nn


def test_nha_compressor_module_present():
    assert hasattr(framework.attentions, 'nha_compressor')
